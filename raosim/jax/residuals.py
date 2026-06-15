"""
raosim.jax.residuals — vectorized Rao/MOC residual leaves (J2).

Faithful JAX ports of the pure, node-wise residual functions that the BVP
differentiates through:

  - ``raosim.rao_residuals``: C+/C- axisymmetric compatibility, left-Mach
    geometry, wall tangency, C+ child position.
  - ``raosim.rao_variational``: the CE-array blocks
    ``_ce_axisymmetric_compatibility_residual_groups``,
    ``_rao_algebraic_stationarity_residuals``,
    ``_rao_left_mach_geometry_residuals``, ``_ce_smoothness_regularization``,
    and the per-node/per-segment algebraic + differential Rao stationarity.

These are vmap'd over node pairs instead of Python ``for`` loops, but compute
exactly the same numbers (J2 gate: 1e-10 parity on a fixed CE state — see
``tests/test_jax_residual_parity.py``).  Scaling (``/radians(1°)``, ``/ds``)
matches the NumPy callers so the assembled residual magnitudes — and the J4
``max_scaled < 2e-3`` gate — are directly comparable.

The full grouped/weighted assembly (``_rao_bvp_residual_groups``) plus the
mass/length/stationarity integrals land in **J3** once the CE/kernel geometry is
ported; those blocks are coupled to the marching construction, not pure leaves.

State convention: a CE/wall polyline is passed as four 1-D arrays
``(x, r, M, theta)``.  Mach is clamped to >= 1.001 to mirror
``_control_surface_flow_nodes`` (which builds ``FlowNode(M=max(M, 1.001))``).
"""

from __future__ import annotations

import math

import raosim.jax  # noqa: F401  -- enables x64
import jax.numpy as jnp

from raosim.jax.primitives import (
    prandtl_meyer, mach_angle, mstar_from_M,
    isentropic_density_ratio, isentropic_temperature_ratio,
)

_M_FLOOR = 1.001
_ONE_DEG = math.radians(1.0)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clamp_M(M):
    return jnp.maximum(jnp.asarray(M, dtype=jnp.float64), _M_FLOOR)


def _mu(M):
    return mach_angle(_clamp_M(M))


def _nu(M, gamma):
    return prandtl_meyer(_clamp_M(M), gamma)


def _source_axisym(theta, mu, r):
    """Axisymmetric source S = sinθ sinμ / r (ds-form).

    CORRECTED 2026-06-11 (mirror of rao_residuals._source_axisym): the
    old Q± carried a spurious cosμ/cos(θ±μ) factor.  Oracle-validated;
    see the NumPy twin's docstring for the evidence.
    """
    ok = r > 1e-12
    r_safe = jnp.where(ok, r, 1.0)
    return jnp.where(ok, jnp.sin(theta) * jnp.sin(mu) / r_safe, 0.0)


# --------------------------------------------------------------------------- #
# node-pair leaves (vectorized over consecutive segments)                     #
# --------------------------------------------------------------------------- #
def _c_axisym(x, r, M, theta, gamma, sign):
    """C+ (sign=+1) or C− (sign=-1) axisymmetric compatibility per segment.

    CORRECTED 2026-06-11 invariant pairing (Anderson MCF §11.4; Z&H
    Vol. 2 Ch. 17), nodes ordered downstream:
      C+ (slope θ+μ): d(θ − ν) = −S ds
      C− (slope θ−μ): d(θ + ν) = +S ds
    i.e. invariant k = θ − sign·ν, residual = dk + sign·S·ds.
    (Previously k = θ + sign·ν with the cos-factored Q — the families'
    relations were mirrored.)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    Mc = _clamp_M(M)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    nu = _nu(Mc, gamma)
    mu = _mu(Mc)

    k = theta - sign * nu                     # θ ∓ ν (C+: θ−ν; C−: θ+ν)
    lhs = k[1:] - k[:-1]
    ds = jnp.hypot(x[1:] - x[:-1], r[1:] - r[:-1])
    th_avg = 0.5 * (theta[:-1] + theta[1:])
    mu_avg = 0.5 * (mu[:-1] + mu[1:])
    r_avg = jnp.maximum(0.5 * (r[:-1] + r[1:]), 1e-12)
    S = _source_axisym(th_avg, mu_avg, r_avg)
    return lhs + sign * S * ds


def residual_Cplus_axisym(x, r, M, theta, gamma):
    """Per-segment C+ residual.  Mirrors rao_residuals.residual_Cplus_axisym."""
    return _c_axisym(x, r, M, theta, gamma, +1.0)


def residual_Cminus_axisym(x, r, M, theta, gamma):
    """Per-segment C- residual.  Mirrors rao_residuals.residual_Cminus_axisym."""
    return _c_axisym(x, r, M, theta, gamma, -1.0)


def residual_left_mach_geometry(x, r, M, theta):
    """Per-segment dr - dx·tan(θ_avg + μ_avg).  Mirrors rao_residuals version."""
    x = jnp.asarray(x, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    mu = _mu(M)
    dx = x[1:] - x[:-1]
    dr = r[1:] - r[:-1]
    th_avg = 0.5 * (theta[:-1] + theta[1:])
    mu_avg = 0.5 * (mu[:-1] + mu[1:])
    return dr - dx * jnp.tan(th_avg + mu_avg)


def residual_wall_tangency(x, r, theta):
    """Per-segment dr - dx·tan(θ_avg).  Mirrors rao_residuals.residual_wall_tangency."""
    x = jnp.asarray(x, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    dx = x[1:] - x[:-1]
    dr = r[1:] - r[:-1]
    th_avg = 0.5 * (theta[:-1] + theta[1:])
    return dr - dx * jnp.tan(th_avg)


def residual_cplus_child_position(parent, child):
    """dr - tan(θ_avg+μ_avg)·dx for a (parent, child) pair given as (x,r,M,theta)."""
    px, pr, pM, pth = parent
    cx, cr, cM, cth = child
    dx = cx - px
    dr = cr - pr
    th_avg = 0.5 * (pth + cth)
    mu_avg = 0.5 * (_mu(pM) + _mu(cM))
    return dr - jnp.tan(th_avg + mu_avg) * dx


# --------------------------------------------------------------------------- #
# per-node / per-segment Rao stationarity                                     #
# --------------------------------------------------------------------------- #
def rao_stationarity_residual(M, theta, log_C, gamma):
    """
    Algebraic Rao stationarity at each node (log form), vectorized.

    log(M*) + log|cos(θ−α)| − log(cos α) − log_C,  α = asin(1/M).
    Mirrors rao_variational.rao_stationarity_residual (normal branch).
    """
    Mc = _clamp_M(M)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    alpha = jnp.arcsin(jnp.clip(1.0 / Mc, 0.0, 1.0))
    cos_a = jnp.cos(alpha)
    cos_tma = jnp.cos(theta - alpha)
    Ms = mstar_from_M(Mc, gamma)
    return jnp.log(Ms) + jnp.log(jnp.abs(cos_tma)) - jnp.log(cos_a) - log_C


def rao_stationarity_fd_residual(M, theta, gamma):
    """Differential Rao stationarity between adjacent nodes (per segment)."""
    Mc = _clamp_M(M)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    alpha = jnp.arcsin(jnp.clip(1.0 / Mc, 0.0, 1.0))
    Ms = mstar_from_M(Mc, gamma)
    d_ln_Ms = jnp.log(Ms[1:]) - jnp.log(Ms[:-1])
    dth = theta[1:] - theta[:-1]
    da = alpha[1:] - alpha[:-1]
    th = 0.5 * (theta[:-1] + theta[1:])
    a = 0.5 * (alpha[:-1] + alpha[1:])
    return d_ln_Ms - (dth - da) * jnp.tan(th - a) + da * jnp.tan(a)


# --------------------------------------------------------------------------- #
# CE-array blocks (match the NumPy callers' scaling exactly)                   #
# --------------------------------------------------------------------------- #
def ce_axisymmetric_compatibility_groups(x, r, M, theta, gamma):
    """(cplus, cminus) each /radians(1°).  Mirrors
    _ce_axisymmetric_compatibility_residual_groups."""
    cp = residual_Cplus_axisym(x, r, M, theta, gamma) / _ONE_DEG
    cm = residual_Cminus_axisym(x, r, M, theta, gamma) / _ONE_DEG
    return cp, cm


def ce_algebraic_stationarity(x, r, M, theta, log_C, gamma):
    """Per-node algebraic stationarity. Mirrors _rao_algebraic_stationarity_residuals."""
    return rao_stationarity_residual(M, theta, log_C, gamma)


def ce_left_mach(x, r, M, theta):
    """Per-segment left-Mach residual /ds.  Mirrors _rao_left_mach_geometry_residuals."""
    x = jnp.asarray(x, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    raw = residual_left_mach_geometry(x, r, M, theta)
    ds = jnp.maximum(jnp.hypot(x[1:] - x[:-1], r[1:] - r[:-1]), 1e-12)
    return raw / ds


def ce_smoothness_regularization(M, theta, gamma):
    """concat(diff(θ+ν,2), diff(θ−ν,2)) / radians(1°).  Mirrors
    _ce_smoothness_regularization."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    nu = _nu(M, gamma)
    kp = theta + nu
    km = theta - nu
    if kp.shape[0] < 3:
        return jnp.zeros(0, dtype=jnp.float64)
    return jnp.concatenate([jnp.diff(kp, n=2), jnp.diff(km, n=2)]) / _ONE_DEG


def curve_mass_flux(x, r, M, theta, gamma):
    """
    Axisymmetric mass flux integrated over a curve (REWRITE_PLAN.md §2.D).

        dṁ = 2π R ρV sin(β − θ) ds ,   β = atan2(dr, dx) is the local segment angle.

    Non-dimensional (ρ/ρ0, V = M·sqrt(γT/T0)).  This is the surface-normal flux
    that replaces the quasi-1D throat target ``ρV*·A_t``; mass closure is then
    ``curve_mass_flux(CE) − curve_mass_flux(kernel_BD)``.  Pure and
    differentiable — the kernel-BD side and point D arrive with the J3b march.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    Mc = _clamp_M(M)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    dx = x[1:] - x[:-1]
    dr = r[1:] - r[:-1]
    ds = jnp.hypot(dx, dr)
    beta = jnp.arctan2(dr, dx)
    M_avg = 0.5 * (Mc[:-1] + Mc[1:])
    th_avg = 0.5 * (theta[:-1] + theta[1:])
    r_avg = jnp.maximum(0.5 * (r[:-1] + r[1:]), 1e-9)
    rho = isentropic_density_ratio(M_avg, gamma)
    T = isentropic_temperature_ratio(M_avg, gamma)
    V = M_avg * jnp.sqrt(gamma * T)
    dmd = 2.0 * jnp.pi * r_avg * rho * V * jnp.abs(jnp.sin(beta - th_avg)) * ds
    return jnp.sum(dmd)


__all__ = [
    "curve_mass_flux",
    "residual_Cplus_axisym",
    "residual_Cminus_axisym",
    "residual_left_mach_geometry",
    "residual_wall_tangency",
    "residual_cplus_child_position",
    "rao_stationarity_residual",
    "rao_stationarity_fd_residual",
    "ce_axisymmetric_compatibility_groups",
    "ce_algebraic_stationarity",
    "ce_left_mach",
    "ce_smoothness_regularization",
]
