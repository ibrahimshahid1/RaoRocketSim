"""J6 v1 — exact gradient API over the solved Rao BVP (plan §6).

Scope (v1, honest about what is differentiated):

* **Node-level tolerance fields** — exact reverse-mode gradients of the
  control-surface thrust coefficient with respect to every solved
  unknown: ``dCf_dM / dCf_dtheta / dCf_dr`` per CE/DE node, plus the
  scalar tail (``dCf_dkdf`` — D sliding along the frozen kernel BD —
  and the multipliers, which must come out ~0 at a converged optimum).
  The per-node ``dCf_dr`` is the plan's "manufacturing-tolerance map"
  evaluated on the control surface: how much Cf moves if the surface
  deviates at node i *without re-solving* — exactly the right object
  for a tolerance interpretation.
* **Explicit design partials at fixed u*** — ``dCf_dpa`` and
  ``dCf_dgamma`` through the functional's explicit dependence.  These
  are *partial*, not total, derivatives: they exclude the implicit
  shift of the solution u*(p).  ``dCf_dpa`` is analytically
  ``-(r_E^2 - r_D^2)/Rt^2`` (the trapezoidal pressure term telescopes),
  which the test suite pins as the J6 known-sign/value gate.
* **Jacobian diagnostics** — exact ``jacfwd`` Jacobian of the residual
  at u*, with its extreme singular values (conditioning of the
  converged system).

Deferred to v2 (documented, not silently missing):

* Total design derivatives ``d(outputs)/d(Rt, epsilon, length_pct,
  gamma)`` via the implicit function theorem through
  ``optimistix.least_squares`` — needs the assembly's solve constants
  lifted into a traced ``args`` pytree, and (for anything reaching the
  kernel: Rt, Rd, theta_B) the J3b differentiable march, since the
  kernel BD is a frozen NumPy artifact today.
* ``dCf_dwall`` on the *bell wall* nodes — needs the BDE region march
  differentiable (same J3b family).
* ``hessian_thrust`` — the §11 soft-mode study; ``jax.hessian`` of
  :func:`cf_from_u` slots in once wanted.

Cf convention: :func:`cf_de_jax` is a line-for-line ``jax.numpy`` port
of :func:`raosim.nasa_moc.surface_thrust_coefficient` (momentum +
pressure flux through the DE control surface, normalised by
``pi * Rt^2 * p0``), so the parity test can hold to ~1e-12 in x64.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

import raosim.jax  # noqa: F401  -- enables jax_enable_x64 before jax use
import jax
import jax.numpy as jnp

from raosim.jax import assembly
from raosim.jax.primitives import (
    isentropic_density_ratio,
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
)

__all__ = ["RaoSensitivities", "cf_de_jax", "cf_from_u", "rao_sensitivities"]

_M_FLOOR = 1.000001


def cf_de_jax(x, r, M, theta, gamma, Rt, pa_over_p0=0.0):
    """Surface-integrated thrust coefficient on a DE polyline (jnp).

    Port of :func:`raosim.nasa_moc.surface_thrust_coefficient`; the
    ``continue`` guards become multiplicative masks so the expression
    stays differentiable (masked segments contribute exactly 0, as in
    the NumPy loop).
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    M = jnp.asarray(M)
    theta = jnp.asarray(theta)

    dx = x[1:] - x[:-1]
    dr = r[1:] - r[:-1]
    beta = jnp.arctan2(dr, dx)
    sin_beta = jnp.sin(beta)

    Mm = jnp.maximum(0.5 * (M[:-1] + M[1:]), _M_FLOOR)
    th = 0.5 * (theta[:-1] + theta[1:])
    rm = jnp.maximum(0.5 * (r[:-1] + r[1:]), 1e-12)

    p_ratio = isentropic_pressure_ratio(Mm, gamma)
    rho_ratio = isentropic_density_ratio(Mm, gamma)
    T_ratio = isentropic_temperature_ratio(Mm, gamma)
    V_sq = gamma * Mm * Mm * T_ratio

    valid = (jnp.abs(dr) > 1e-14) & (jnp.abs(sin_beta) > 1e-14)
    safe_sin_beta = jnp.where(valid, sin_beta, 1.0)

    momentum = (rho_ratio * V_sq * jnp.cos(th)
                * jnp.sin(beta - th) / safe_sin_beta)
    pressure = p_ratio - pa_over_p0
    contrib = 2.0 * jnp.pi * rm * (momentum + pressure) * dr
    F_total = jnp.sum(jnp.where(valid, contrib, 0.0))
    return F_total / jnp.maximum(jnp.pi * Rt * Rt, 1e-12)


def cf_from_u(u, sp: "assembly.StaticParams", n: int, Rt,
              gamma, pa_over_p0):
    """Cf of the solved control surface as a function of the unknowns.

    Slices ``u`` with the legacy (couple_wall=False) layout
    ``[M(n), theta(n), r(n), lambda2, lambda3, log_C, kdf]``,
    reconstructs x by the same left-Mach integration the residual uses
    (anchored at D's kernel position via ``kdf``), and evaluates
    :func:`cf_de_jax`.  Fully differentiable, including through the
    ``kdf -> x_D`` chain (D sliding along the frozen BD).
    """
    u = jnp.asarray(u)
    M = u[:n]
    theta = u[n:2 * n]
    r = u[2 * n:3 * n]
    kdf = u[3 * n + 3]
    xD, _rD, _thD, _MD = assembly.bd_point_at_fraction(sp, kdf)
    x = assembly.integrate_x_from_left_mach(r, theta, M, xD)
    return cf_de_jax(x, r, M, theta, gamma, Rt, pa_over_p0)


@dataclass(frozen=True)
class RaoSensitivities:
    """J6 v1 result bundle.  See the module docstring for semantics."""

    u_star: np.ndarray
    cf: float
    max_scaled: float
    # -- node-level tolerance fields (exact, fixed-residual) ----------
    dCf_du: np.ndarray
    dCf_dM: np.ndarray
    dCf_dtheta: np.ndarray
    dCf_dr: np.ndarray
    dCf_dkdf: float
    dCf_dscalars: dict[str, float]
    # -- explicit design partials at fixed u* -------------------------
    dCf_dpa_explicit: float
    dCf_dgamma_explicit: float
    # -- system diagnostics -------------------------------------------
    jacobian: np.ndarray
    sigma_max: float
    sigma_min: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def condition_number(self) -> float:
        return float(self.sigma_max / max(self.sigma_min, 1e-300))


def _repack_u_from_solution(solution, n: int) -> np.ndarray:
    ce = solution.control_surface
    return np.concatenate([
        np.asarray(ce.M, dtype=float),
        np.asarray(ce.theta, dtype=float),
        np.asarray(ce.r, dtype=float),
        np.asarray([ce.lambda2, ce.lambda3, ce.log_C,
                    ce.kernel_d_fraction], dtype=float),
    ])


def rao_sensitivities(config, *, solution=None):
    """Exact gradients of Cf w.r.t. the solved unknowns (J6 v1).

    Parameters
    ----------
    config
        :class:`raosim.rao_variational.RaoSolverConfig`.  v1 supports
        the J4-gate layout only (``couple_wall=False``).
    solution
        Optional pre-computed :class:`RaoSolution` *for this config*
        (skips the solve).  Must carry ``kernel_points`` (always true
        for ``solve_rao_bvp`` outputs).

    Returns
    -------
    RaoSensitivities
    """
    import raosim.rao_variational as rv
    from raosim.moc import FlowNode

    if getattr(config, "couple_wall", False):
        raise NotImplementedError(
            "rao_sensitivities v1 supports couple_wall=False (the J4-gate "
            "layout); the coupled-wall unknown vector lands with v2."
        )

    if solution is None:
        solution = rv.solve_rao_bvp(replace(config, solver_backend="jax"))

    n = int(len(solution.control_surface.r))
    u_star = _repack_u_from_solution(solution, n)

    kernel_bd = tuple(
        FlowNode(x=float(p.x), r=float(p.r), M=float(p.M),
                 theta=float(p.theta))
        for p in solution.kernel_points
    )
    solve_config = replace(config, kernel_bd=kernel_bd, couple_wall=False)
    sp = assembly.params_from_config(solve_config)
    residual_fn = assembly.make_residual(sp)

    uj = jnp.asarray(u_star)
    res = np.asarray(residual_fn(uj), dtype=float)
    max_scaled = float(np.max(np.abs(res))) if res.size else float("nan")

    Rt = float(config.Rt)
    gamma = float(config.gamma)
    pa = float(config.pa_over_p0)

    cf_val = float(cf_from_u(uj, sp, n, Rt, gamma, pa))

    # Node-level fields: one reverse-mode sweep.
    g_u = np.asarray(jax.grad(
        lambda u: cf_from_u(u, sp, n, Rt, gamma, pa))(uj), dtype=float)
    # Explicit design partials at fixed u*.
    g_pa = float(jax.grad(
        lambda p: cf_from_u(uj, sp, n, Rt, gamma, p))(pa))
    g_gamma = float(jax.grad(
        lambda g: cf_from_u(uj, sp, n, Rt, g, pa))(gamma))

    jac = np.asarray(jax.jacfwd(residual_fn)(uj), dtype=float)
    svals = np.linalg.svd(jac, compute_uv=False)

    return RaoSensitivities(
        u_star=u_star,
        cf=cf_val,
        max_scaled=max_scaled,
        dCf_du=g_u,
        dCf_dM=g_u[:n].copy(),
        dCf_dtheta=g_u[n:2 * n].copy(),
        dCf_dr=g_u[2 * n:3 * n].copy(),
        dCf_dkdf=float(g_u[3 * n + 3]),
        dCf_dscalars={
            "lambda2": float(g_u[3 * n]),
            "lambda3": float(g_u[3 * n + 1]),
            "log_C": float(g_u[3 * n + 2]),
            "kernel_d_fraction": float(g_u[3 * n + 3]),
        },
        dCf_dpa_explicit=g_pa,
        dCf_dgamma_explicit=g_gamma,
        jacobian=jac,
        sigma_max=float(svals[0]),
        sigma_min=float(svals[-1]),
        diagnostics={
            "n_control": n,
            "n_residuals": int(res.size),
            "converged": bool(solution.control_surface.converged),
            "reliability": str(getattr(solution, "reliability", "")),
        },
    )
