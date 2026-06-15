"""
raosim.jax.theta_b_solve — θ_B as a solved unknown inside the BVP (J3b-2).

Replaces the seed secant's frozen kernel with the differentiable march:
the unknown vector gains one trailing component θ_B, and the residual
recomputes the kernel BD *in-graph* via :func:`raosim.jax.moc_kernel.
march_kernel` each evaluation — mass-closure target, D-state pins, and
the BD-flux scale all read the live BD(θ_B).

Wiring strategy (zero changes to the J2-parity-gated assembly):
``assembly.make_residual`` reads the BD polyline through
``jnp.asarray(sp.bd_*)``, and ``StaticParams`` is a plain NamedTuple —
so a ``_replace`` with TRACED arrays flows through the existing
masked formulations untouched.  The march's padded BD (tail = exact
axis-state repeats) is safe by construction: padding segments have
ds = 0, which ``bd_flux_to_fraction`` / ``_polyline_mass_flux`` already
mask (``ds > 1e-12``), and the arc-length cumsum plateaus so
``kernel_d_fraction`` keeps parametrising the valid arc only.

Correctness gate (pinned in tests/test_jax_theta_b_solve.py): at the
seed θ_B the live-BD residual equals the static-``kernel_bd`` residual
to ~1e-8 — automatic, because the march reproduces the NumPy seed
kernel's BD at bit parity (tests/test_jax_moc_kernel.py).

θ_B window: the march's pre-clamp wall grid (raw landings + fixed
dθ-limit special steps) is θ_B-independent, so d(residual)/dθ_B is
exact and smooth while θ_B stays within its current grid interval;
crossing a 0.25° boundary changes the clamp-row index (a small
discontinuity).  Solver bounds default to ± half a special step around
the seed; per-ladder-rung re-assembly re-centres the window exactly
like ``kernel_bd`` re-seeding today.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax
import jax.numpy as jnp
import optimistix as optx

from raosim.jax import assembly
from raosim.jax.api import LM_ATOL, LM_RTOL, SEED_NUDGE, _logit
from raosim.jax.moc_kernel import KernelRow, march_kernel
from raosim.jax.primitives import mach_angle, prandtl_meyer

_CE_M_FLOOR = assembly._CE_M_FLOOR


def start_line_from_kernel(kernel) -> KernelRow:
    """TT' (the seed kernel's start line) as a wall-first KernelRow."""
    tt = kernel.rrcs[0]
    M = jnp.maximum(jnp.asarray([float(p.M) for p in tt]), 1.000001)
    return KernelRow(
        x=jnp.asarray([float(p.x) for p in tt]),
        r=jnp.asarray([float(p.r) for p in tt]),
        theta=jnp.asarray([float(p.theta) for p in tt]),
        nu=jnp.maximum(prandtl_meyer(M, kernel.gamma), 1e-8),
        mu=mach_angle(M),
        M=M,
    )


def _n_rows_bound(theta_b_max: float, dtheta_limit: float) -> int:
    """Static row bound: special steps advance dθ_limit/2 per row; raw
    rows advance faster, so the special-only count is an upper bound."""
    return int(math.ceil(theta_b_max / (0.5 * dtheta_limit))) + 8


def make_residual_theta_b(
    config,
    kernel,
    *,
    theta_b_max: float,
    dtheta_limit: float = math.radians(0.5),
    n_corr: int = 40,
    physics_weight: float | None = None,
):
    """Build ``fn(u_ext) -> r`` with ``u_ext = [u..., theta_B]``.

    ``kernel`` is the seed ``MOCKernel`` (provides TT' and the default
    march parameters); ``config.kernel_bd`` must still be populated (it
    seeds the static scale fallbacks and keeps ``params_from_config``
    semantics identical to the frozen-BD path).
    """
    sp = assembly.params_from_config(config, physics_weight=physics_weight)
    sl = start_line_from_kernel(kernel)
    n_rows = _n_rows_bound(float(theta_b_max), float(dtheta_limit))
    Rt = float(config.Rt)
    Rd = float(config.throat_downstream_radius_factor * config.Rt)
    gamma = float(config.gamma)

    def fn(u_ext):
        u_ext = jnp.asarray(u_ext, dtype=jnp.float64)
        u, theta_b = u_ext[:-1], u_ext[-1]
        m = march_kernel(sl, theta_b, Rt, Rd, gamma,
                         n_rows=n_rows, n_corr=n_corr,
                         dtheta_limit=dtheta_limit)
        bd = m.bd
        # Full-BD flux scale, rao_variational.curve_mass_flux flavour
        # (CE-side midpoint clamp) — mirrors params_from_config's
        # static bd_full_flux; ds=0 padding segments are masked inside.
        bd_full = assembly._polyline_mass_flux(
            bd.x, bd.r, bd.M, bd.theta, gamma, _CE_M_FLOOR)
        sp_live = sp._replace(
            bd_x=bd.x, bd_r=bd.r, bd_M=bd.M, bd_theta=bd.theta,
            bd_full_flux=bd_full,
        )
        return assembly.make_residual(sp_live)(u)

    return fn, n_rows


def least_squares_jax_theta_b(
    config,
    kernel,
    u0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    theta_b_seed: float | None = None,
    theta_b_halfwidth: float | None = None,
    dtheta_limit: float = math.radians(0.5),
    max_steps: int | None = None,
    physics_weight: float | None = None,
):
    """Optimistix LM over ``[u, theta_B]`` with the live-BD residual.

    Mirrors :func:`raosim.jax.api.least_squares_jax` (sigmoid box
    reparametrisation, constraint-weight ladder, unweighted reporting);
    returns the same result namespace with ``x`` EXCLUDING θ_B (so the
    NumPy shell's unpacking is unchanged) plus ``theta_b`` solved.

    Default θ_B bounds: ± half a special step (dθ_limit/4) around the
    seed — inside the smooth window of the march's clamp row.  The
    caller re-centres per ladder rung / outer iteration if θ_B walks
    to a bound (exactly like kernel_bd re-seeding today).
    """
    tb0 = float(theta_b_seed if theta_b_seed is not None
                else kernel.theta_B)
    half = float(theta_b_halfwidth if theta_b_halfwidth is not None
                 else 0.25 * dtheta_limit)
    fn, n_rows = make_residual_theta_b(
        config, kernel, theta_b_max=tb0 + 2.0 * half,
        dtheta_limit=dtheta_limit, physics_weight=physics_weight,
    )

    lo = jnp.asarray(np.concatenate([np.asarray(lower, dtype=float),
                                     [tb0 - half]]))
    hi = jnp.asarray(np.concatenate([np.asarray(upper, dtype=float),
                                     [tb0 + half]]))
    span = hi - lo
    u0_ext = jnp.asarray(np.concatenate([np.asarray(u0, dtype=float),
                                         [tb0]]))

    frac0 = jnp.clip((u0_ext - lo) / span, SEED_NUDGE, 1.0 - SEED_NUDGE)
    z0 = _logit(frac0)

    sp = assembly.params_from_config(config, physics_weight=physics_weight)
    ladder = getattr(config, "jax_constraint_weight_ladder", None) or (1.0,)
    n_res = int(np.asarray(fn(u0_ext)).size)
    cids = assembly.constraint_indices(sp)

    steps = int(max_steps if max_steps is not None
                else max(config.max_nfev, 256))
    solver = optx.LevenbergMarquardt(rtol=LM_RTOL, atol=LM_ATOL)

    z = z0
    converged = False
    n_steps = 0
    for W in ladder:
        wvec = np.ones(n_res)
        if cids:
            wvec[cids] = float(W)
        wj = jnp.asarray(wvec)

        def obj(zz, args, _wj=wj):
            return _wj * fn(lo + span * jax.nn.sigmoid(zz))

        sol = optx.least_squares(
            obj, solver, z, args=None, max_steps=steps, throw=False,
        )
        z = sol.value
        converged = bool(sol.result == optx.RESULTS.successful)
        n_steps += int(sol.stats.get("num_steps", 0))

    u_star_ext = np.asarray(lo + span * jax.nn.sigmoid(z), dtype=float)
    r_star = np.asarray(fn(jnp.asarray(u_star_ext)), dtype=float)
    return SimpleNamespace(
        x=u_star_ext[:-1],
        theta_b=float(u_star_ext[-1]),
        theta_b_bounds=(float(tb0 - half), float(tb0 + half)),
        success=converged,
        message=(
            f"optimistix LevenbergMarquardt (theta_B live) converged in "
            f"{n_steps} steps"
            if converged else
            f"optimistix LevenbergMarquardt (theta_B live) stopped without "
            f"meeting rtol/atol after {n_steps} steps (max_steps={steps})"
        ),
        cost=float(0.5 * float(np.dot(r_star, r_star))),
        nfev=n_steps,
        max_abs_residual=(float(np.max(np.abs(r_star)))
                          if r_star.size else 0.0),
        backend="jax",
        n_rows=n_rows,
    )


__all__ = [
    "start_line_from_kernel",
    "make_residual_theta_b",
    "least_squares_jax_theta_b",
]
