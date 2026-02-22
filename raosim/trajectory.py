"""
trajectory.py – Optional vertical-ascent trajectory integrator.

Simple 1-D point-mass simulation with gravity, drag, and atmospheric density.
Kept as a separate module so the nozzle design tools can be used independently.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

from raosim.atmosphere import density

g0 = 9.80665
R_EARTH = 6_371_000.0  # m


@dataclass
class StageResult:
    burn_time: float       # s
    ideal_dv: float        # m/s
    actual_dv: float       # m/s
    gravity_loss: float    # m/s
    drag_loss: float       # m/s
    burnout_alt: float     # m
    burnout_vel: float     # m/s
    final_mass: float      # kg


def simulate_stage(
    m0: float,
    m_prop: float,
    thrust: float,
    Isp: float,
    Cd: float,
    A_ref: float,
    h0: float = 0.0,
    v0: float = 0.0,
    dt: float = 0.02,
    coast_to_apogee: bool = True,
) -> StageResult:
    """
    Integrate a single-stage vertical ascent.

    Parameters
    ----------
    m0     : initial total mass (propellant + dry + payload)  [kg]
    m_prop : propellant mass  [kg]
    thrust : constant thrust  [N]
    Isp    : specific impulse  [s]
    Cd     : ballistic drag coefficient
    A_ref  : reference cross-section area  [m²]
    h0, v0 : initial altitude [m] and velocity [m/s]
    dt     : time step  [s]
    coast_to_apogee : if True, continue integrating after burnout until v=0
    """
    m_dot = thrust / (g0 * Isp)
    t_burn = m_prop / m_dot

    m = m0
    h = h0
    v = v0
    loss_g = 0.0
    loss_d = 0.0
    t = 0.0

    # ── powered phase ─────────────────────────────────────────────
    while t < t_burn:
        rho = density(max(h, 0.0))
        D = 0.5 * rho * v * abs(v) * Cd * A_ref
        grav = g0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2
        a = (thrust - D) / m - grav
        loss_g += grav * dt
        loss_d += (D / m) * dt if m > 0.0 else 0.0
        v += a * dt
        h += v * dt
        m -= m_dot * dt
        t += dt

    burnout_v = v
    burnout_h = h

    # ── coast phase ───────────────────────────────────────────────
    if coast_to_apogee:
        while v > 0.0 and h > -100.0:
            rho = density(max(h, 0.0))
            D = 0.5 * rho * v * abs(v) * Cd * A_ref
            grav = g0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2
            a = -grav - D / m
            loss_g += grav * dt
            loss_d += (D / m) * dt if m > 0.0 else 0.0
            v += a * dt
            h += v * dt
            t += dt

    ideal_dv = g0 * Isp * math.log(m0 / max(m0 - m_prop, 0.1))
    actual_dv = math.sqrt(max(0.0, 2.0 * g0 * (h - h0)))

    return StageResult(
        burn_time=t_burn,
        ideal_dv=ideal_dv,
        actual_dv=actual_dv,
        gravity_loss=loss_g,
        drag_loss=loss_d,
        burnout_alt=burnout_h,
        burnout_vel=burnout_v,
        final_mass=m,
    )
