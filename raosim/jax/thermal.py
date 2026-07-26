"""
raosim.jax.thermal — differentiable nozzle thermal/performance physics.

JAX (jnp) ports of the design-driving physics so they enter a
gradient-based constrained design loop (raosim.jax.design_opt):

* full Bartz (1957) gas-side h_g + σ  (mirror of physics.bartz_*)
* Sieder-Tate coolant-side h_c          (mirror of physics.sieder_tate_*)
* the series gas/wall/coolant circuit -> throat wall temperature
* Schmucker separation margin            (mirror of separation.schmucker_*)
* ambient thrust coefficient             (primitives.thrust_coefficient)

The constraints evaluate at closed-form stations — the throat (M=1,
A_t/A=1) for cooling and the exit (isentropic Me from ε) for
separation — so no contour march is needed in the differentiated path.
Parity with the NumPy originals is the J-style acceptance gate
(tests/test_jax_thermal.py); every function is jit/grad-safe.
"""

from __future__ import annotations

import raosim.jax  # noqa: F401  -- enables x64 before any array is built

import jax.numpy as jnp

from raosim.jax.primitives import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)


# --------------------------------------------------------------------------- #
# Gas-side: full Bartz (1957)                                                  #
# --------------------------------------------------------------------------- #
def bartz_sigma(mach, gamma, wall_temperature, Tc, *, omega=0.6):
    """Bartz property-variation factor σ (jnp; see physics.bartz_sigma)."""
    M = jnp.asarray(mach)
    f = 1.0 + 0.5 * (gamma - 1.0) * M * M
    base = 0.5 * (wall_temperature / Tc) * f + 0.5
    return 1.0 / (base ** (0.8 - omega / 5.0) * f ** (omega / 5.0))


def bartz_hg(mach, area_throat_over_area, *, Dt, Pc, c_star, cp, Pr, mu,
             gamma, Tc, wall_temperature, throat_curvature_radius, omega=0.6):
    """Gas-side h_g [W/(m²·K)] — full Bartz correlation (jnp mirror of
    physics.bartz_heat_transfer_coefficient)."""
    sigma = bartz_sigma(mach, gamma, wall_temperature, Tc, omega=omega)
    coeff = (
        0.026 / Dt ** 0.2
        * (mu ** 0.2 * cp / Pr ** 0.6)
        * (Pc / c_star) ** 0.8
        * (Dt / throat_curvature_radius) ** 0.1
    )
    return coeff * jnp.clip(area_throat_over_area, 1e-9, 1.0) ** 0.9 * sigma


def recovery_temperature(mach, gamma, Tc, Pr):
    """Adiabatic-wall temperature with turbulent recovery r = Pr^(1/3)."""
    M = jnp.asarray(mach)
    r = Pr ** (1.0 / 3.0)
    f = 1.0 + 0.5 * (gamma - 1.0) * M * M
    return Tc * (1.0 + r * 0.5 * (gamma - 1.0) * M * M) / f


# --------------------------------------------------------------------------- #
# Coolant-side: Sieder-Tate                                                    #
# --------------------------------------------------------------------------- #
def sieder_tate_hc(mass_flux, D_h, *, k, cp, mu_bulk, mu_wall):
    """Coolant-side h_c [W/(m²·K)] — Sieder-Tate (jnp mirror of
    physics.sieder_tate_coefficient)."""
    Re = mass_flux * D_h / mu_bulk
    Pr = mu_bulk * cp / k
    Nu = (0.027 * Re ** 0.8 * Pr ** (1.0 / 3.0)
          * (mu_bulk / mu_wall) ** 0.14)
    return Nu * k / D_h


# --------------------------------------------------------------------------- #
# Series thermal circuit at the throat (the binding cooling station)          #
# --------------------------------------------------------------------------- #
def throat_wall_temperature(
    *, Rt, Pc, c_star, cp_gas, Pr_gas, mu_gas, gamma, Tc,
    throat_curvature_radius, coolant_temperature, h_c, t_wall, k_wall,
    omega=0.6, n_iter=8,
):
    """Gas-side wall temperature at the throat (M=1, A_t/A=1) from the
    series gas/wall/coolant circuit::

        q = (T_aw − T_c) / (1/h_g + t_w/k_w + 1/h_c)
        T_wg = T_aw − q/h_g

    h_g (Bartz) depends on T_wg through σ, so a few fixed-point passes
    (unrolled, differentiable) make it self-consistent.  Returns
    ``(T_wg, q)``.
    """
    Dt = 2.0 * Rt
    Taw = recovery_temperature(1.0, gamma, Tc, Pr_gas)
    T_wg = 0.6 * Taw

    def body(T_wg, _):
        h_g = bartz_hg(1.0, 1.0, Dt=Dt, Pc=Pc, c_star=c_star, cp=cp_gas,
                       Pr=Pr_gas, mu=mu_gas, gamma=gamma, Tc=Tc,
                       wall_temperature=T_wg,
                       throat_curvature_radius=throat_curvature_radius,
                       omega=omega)
        R_tot = 1.0 / h_g + t_wall / k_wall + 1.0 / h_c
        q = jnp.maximum((Taw - coolant_temperature) / R_tot, 0.0)
        return Taw - q / h_g, q

    # Unrolled fixed point (static n_iter; differentiable).
    q = 0.0
    for _ in range(n_iter):
        T_wg, q = body(T_wg, None)
    return T_wg, q


# --------------------------------------------------------------------------- #
# Separation (Schmucker) and ambient thrust coefficient                       #
# --------------------------------------------------------------------------- #
def schmucker_separation_margin(epsilon, gamma, Pc, Pa, *, supersonic=True):
    """Separation margin Pe/p_sep (>1 = attached) — Schmucker criterion
    p_sep/Pa = (1.88·Me − 1)^(−0.64), evaluated at the exit Mach number
    (Östlund 2002 Eq. 30; jnp mirror of separation.schmucker_separation_ratio).

    Corrected 2026-07-22: the previous form (Pa/Pc)^0.8 / Me was a
    cross-labeled variant ~1.75× off the literature Schmucker at Me = 3
    (docs/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md §A.2.1).  Keep in
    lock-step with the NumPy twin (parity test in
    tests/test_jax_thermal_design_opt.py).
    """
    Me = mach_from_area_ratio(epsilon, gamma, supersonic=supersonic)
    Pe_over_Pc = isentropic_pressure_ratio(Me, gamma)
    denom = jnp.maximum(1.88 * Me - 1.0, 1e-12)
    p_sep_over_Pc = denom ** (-0.64) * (Pa / Pc)
    return Pe_over_Pc / p_sep_over_Pc


def ambient_thrust_coefficient(epsilon, gamma, Pc, Pa):
    """Thrust coefficient at ambient pressure Pa (jnp; uses
    primitives.thrust_coefficient).  Peaks near optimum expansion
    Pe ≈ Pa, so it is a genuine interior objective in ε."""
    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
    Pe_over_Pc = isentropic_pressure_ratio(Me, gamma)
    return thrust_coefficient(Me, gamma, Pe_over_Pc, Pa / Pc, epsilon)
