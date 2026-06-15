"""
raosim.jax.primitives — gas-dynamics leaf primitives in JAX.

These mirror, term for term, the calorically-perfect closed forms in
``raosim.gas_dynamics`` (which are themselves cross-checked against
``propulsion_texts/prmeyer.pdf`` and ``rao1999.pdf``).  Every Rao/MOC residual
differentiates through this module, so the J1 gate is bit-parity with the NumPy
versions (1e-10 on closed forms; round-trip consistency on the two iterative
inverses).

Sonic-safety: ``mach_angle`` and ``prandtl_meyer`` are singular/branch-sensitive
at M -> 1.  Arguments to ``sqrt``/``asin`` are guarded so values and first
derivatives stay finite at the throat.  See JAX_DIFFERENTIABLE_PLAN.md §4.1.

x64 is enabled by ``raosim.jax``'s package __init__; importing this module pulls
that in.
"""

from __future__ import annotations

import raosim.jax  # noqa: F401  -- ensures jax_enable_x64 is set before use
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Isentropic ratios                                                           #
# --------------------------------------------------------------------------- #
def isentropic_temperature_ratio(M, gamma):
    """T/T0 = (1 + (gamma-1)/2 · M²)⁻¹."""
    M = jnp.asarray(M)
    return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * M * M)


def isentropic_pressure_ratio(M, gamma):
    """p/p0 = (T/T0)^(gamma/(gamma-1))."""
    return isentropic_temperature_ratio(M, gamma) ** (gamma / (gamma - 1.0))


def isentropic_density_ratio(M, gamma):
    """rho/rho0 = (T/T0)^(1/(gamma-1))."""
    return isentropic_temperature_ratio(M, gamma) ** (1.0 / (gamma - 1.0))


def area_mach_relation(M, gamma):
    """A/A* = (1/M)·[ (2/(gamma+1))·(1 + (gamma-1)/2·M²) ]^((gamma+1)/(2(gamma-1)))."""
    M = jnp.asarray(M)
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (1.0 / M) * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * M * M)) ** (gp1 / (2.0 * gm1))


# --------------------------------------------------------------------------- #
# Prandtl–Meyer and Mach angle (sonic-safe)                                   #
# --------------------------------------------------------------------------- #
def prandtl_meyer(M, gamma):
    """
    Prandtl-Meyer function ν(M) in radians.

        ν = sqrt((g+1)/(g-1))·atan(sqrt((g-1)/(g+1)·(M²-1))) − atan(sqrt(M²-1))

    Guarded so ν(1)=0 exactly and the derivative is finite at M=1.
    """
    M = jnp.asarray(M)
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    q = jnp.sqrt(gp1 / gm1)
    msq = jnp.maximum(M * M - 1.0, 0.0)
    s = jnp.sqrt(msq)
    return q * jnp.arctan(jnp.sqrt(gm1 / gp1 * msq)) - jnp.arctan(s)


def mach_angle(M):
    """Mach angle μ = arcsin(1/M), radians.  1/M clamped to (0, 1]."""
    M = jnp.asarray(M)
    return jnp.arcsin(jnp.clip(1.0 / M, 0.0, 1.0))


def mstar_from_M(M, gamma):
    """
    Critical Mach number M* = V/a* = sqrt[ (g+1)M² / (2 + (g-1)M²) ].

    Rao-Beck-Booth optimum-thrust stationarity (AIAA 99-2584;
    propulsion_texts/rao1999.pdf, Eq. 3).
    """
    M = jnp.asarray(M)
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return jnp.sqrt(gp1 * M * M / (2.0 + gm1 * M * M))


# --------------------------------------------------------------------------- #
# Iterative inverses (fixed-iteration Newton; jit/vmap-safe, no python branch) #
# --------------------------------------------------------------------------- #
def mach_from_prandtl_meyer(nu, gamma, n_iter: int = 60):
    """
    Invert ν(M) -> M via fixed-count Newton, derivative
    dν/dM = sqrt(M²−1) / (M·(1 + (g−1)/2·M²)).

    Mirrors ``gas_dynamics.mach_from_prandtl_meyer``.  ν=0 maps to M=1 exactly.
    """
    nu = jnp.asarray(nu, dtype=jnp.float64)
    gm1 = gamma - 1.0
    M0 = jnp.maximum(1.0 + nu, 1.01)

    def body(_, M):
        nu_c = prandtl_meyer(M, gamma)
        msq = jnp.maximum(M * M - 1.0, 1e-30)
        denom = M * (1.0 + 0.5 * gm1 * M * M)
        dnu_dM = jnp.sqrt(msq) / denom
        dM = (nu - nu_c) / dnu_dM
        return jnp.maximum(M + dM, 1.0 + 1e-12)

    M = lax.fori_loop(0, n_iter, body, M0)
    return jnp.where(nu <= 0.0, 1.0, M)


def mach_from_area_ratio(area_ratio, gamma, supersonic: bool = True,
                         n_iter: int = 80):
    """
    Invert A/A* -> M (supersonic branch by default) via fixed-count Newton.

    ``supersonic`` is a static python bool (branch selection happens outside the
    differentiated region — see JAX_DIFFERENTIABLE_PLAN.md §5).  Mirrors
    ``gas_dynamics.mach_from_area_ratio``.
    """
    ar = jnp.asarray(area_ratio, dtype=jnp.float64)
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    exp = gp1 / (2.0 * gm1)

    if supersonic:
        M0 = jnp.maximum(1.0 + 0.5 * jnp.log(jnp.maximum(ar, 1.0)), 1.01)
    else:
        M0 = jnp.asarray(0.5, dtype=jnp.float64)

    def body(_, M):
        A = area_mach_relation(M, gamma)
        bracket = 1.0 + 0.5 * gm1 * M * M
        dA = A * (-1.0 / M + exp * gm1 * M / bracket)
        dM = -(A - ar) / dA
        M = M + dM
        return jnp.maximum(M, 1e-6)

    return lax.fori_loop(0, n_iter, body, M0)


def thrust_coefficient(Me, gamma, Pe_over_Pc, Pa_over_Pc, epsilon):
    """Ideal 1-D thrust coefficient (mirror of gas_dynamics.thrust_coefficient)."""
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    momentum = jnp.sqrt(
        (2.0 * gamma * gamma / gm1)
        * (2.0 / gp1) ** (gp1 / gm1)
        * (1.0 - Pe_over_Pc ** (gm1 / gamma))
    )
    return momentum + (Pe_over_Pc - Pa_over_Pc) * epsilon


__all__ = [
    "isentropic_temperature_ratio",
    "isentropic_pressure_ratio",
    "isentropic_density_ratio",
    "area_mach_relation",
    "prandtl_meyer",
    "mach_angle",
    "mstar_from_M",
    "mach_from_prandtl_meyer",
    "mach_from_area_ratio",
    "thrust_coefficient",
]
