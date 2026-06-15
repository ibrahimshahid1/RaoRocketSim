"""
raosim.jax.api — public entry points for the JAX backend.

Two layers, one boundary (JAX_DIFFERENTIABLE_PLAN.md §2): the NumPy shell
in ``rao_variational.solve_rao_bvp`` keeps owning seeding, kernel
construction, reliability gating, diagnostics, and output assembly.  This
module owns only the *inner least-squares solve* — Optimistix
Levenberg–Marquardt with the exact autodiff Jacobian of the assembled
residual (``raosim.jax.assembly``), replacing scipy's finite-difference
``least_squares``.

Bound handling: scipy's trust-region-reflective supports box bounds
natively; Optimistix LM does not.  We reparametrise through a smooth
bijection ``u = lo + (hi - lo) * sigmoid(z)`` so every iterate is strictly
inside the box (this matters: ``kernel_d_fraction``'s 0.7 cap is the
Option-2 valid-region workaround and must hold during iteration, not just
at the end).  The seed is nudged a relative ``SEED_NUDGE`` into the
interior so no ``z`` starts at ±inf; convergence is judged on the
*residual*, which is reparametrisation-invariant.

``solve_rao_bvp_jax(config)`` is sugar for
``solve_rao_bvp(replace(config, solver_backend="jax"))`` and returns the
identical ``RaoSolution`` structure (same reliability ladder, same
diagnostics), so export/plotting/CLI consumers are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax
import jax.numpy as jnp
import optimistix as optx

from raosim.jax import assembly

#: Relative interior nudge applied to seed components sitting on a bound.
SEED_NUDGE = 1e-3

#: LM tolerances (JAX_DIFFERENTIABLE_PLAN.md §4.4).
LM_RTOL = 1e-8
LM_ATOL = 1e-10


def _logit(p):
    return jnp.log(p) - jnp.log1p(-p)


def least_squares_jax(
    config,
    u0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_steps: int | None = None,
    physics_weight: float | None = None,
):
    """
    Solve the assembled Rao BVP residual with Optimistix LM + exact Jacobian.

    Drop-in for the scipy ``least_squares`` call inside ``solve_rao_bvp``:
    returns an object with ``x`` (np.ndarray, strictly inside the bounds),
    ``success``, ``message``, ``cost``, ``nfev`` (LM steps), and
    ``max_abs_residual``.

    ``config`` must already carry ``kernel_bd`` (the solve-time config).
    """
    sp = assembly.params_from_config(config, physics_weight=physics_weight)
    residual = assembly.make_residual(sp)

    lo = jnp.asarray(np.asarray(lower, dtype=float))
    hi = jnp.asarray(np.asarray(upper, dtype=float))
    span = hi - lo

    frac0 = (jnp.asarray(np.asarray(u0, dtype=float)) - lo) / span
    frac0 = jnp.clip(frac0, SEED_NUDGE, 1.0 - SEED_NUDGE)
    z0 = _logit(frac0)

    def fn(z, args):
        u = lo + span * jax.nn.sigmoid(z)
        return residual(u)

    steps = int(max_steps if max_steps is not None else max(config.max_nfev, 256))
    solver = optx.LevenbergMarquardt(rtol=LM_RTOL, atol=LM_ATOL)
    sol = optx.least_squares(
        fn, solver, z0, args=None, max_steps=steps, throw=False,
    )

    u_star = np.asarray(lo + span * jax.nn.sigmoid(sol.value), dtype=float)
    r_star = np.asarray(residual(jnp.asarray(u_star)), dtype=float)
    converged = bool(sol.result == optx.RESULTS.successful)
    n_steps = int(sol.stats.get("num_steps", -1))
    return SimpleNamespace(
        x=u_star,
        success=converged,
        message=(
            f"optimistix LevenbergMarquardt converged in {n_steps} steps"
            if converged else
            f"optimistix LevenbergMarquardt stopped without meeting "
            f"rtol/atol after {n_steps} steps (max_steps={steps})"
        ),
        cost=float(0.5 * float(np.dot(r_star, r_star))),
        nfev=n_steps,
        max_abs_residual=float(np.max(np.abs(r_star))) if r_star.size else 0.0,
        backend="jax",
    )


def solve_rao_bvp_jax(config):
    """Differentiable-backend Rao BVP solve; returns a ``RaoSolution``.

    Identical shell to the NumPy path — only the inner least-squares step
    runs under JAX (exact ``jacfwd`` Jacobians inside Optimistix LM).
    """
    from raosim.rao_variational import solve_rao_bvp

    return solve_rao_bvp(replace(config, solver_backend="jax"))


# --------------------------------------------------------------------------- #
# J6 — gradient API                                                            #
# --------------------------------------------------------------------------- #
def rao_sensitivities(config):
    """Exact gradients of Cf/Isp/etc. w.r.t. design params and nodes. (J6)"""
    raise NotImplementedError(
        "J6: gradient API — lands once the J4/J5 gates pass; see "
        "JAX_DIFFERENTIABLE_PLAN.md §6."
    )


__all__ = ["least_squares_jax", "solve_rao_bvp_jax", "rao_sensitivities",
           "SEED_NUDGE", "LM_RTOL", "LM_ATOL"]
