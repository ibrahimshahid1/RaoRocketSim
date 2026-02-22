"""
engine.py – Rocket engine performance computed from nozzle + propellant.

Combines gas-dynamics relations with propellant thermochemistry to produce
thrust, Isp, mass-flow rate, and related performance metrics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

from raosim.gas_dynamics import (
    thrust_coefficient,
    isentropic_pressure_ratio,
    mach_from_area_ratio,
)
from raosim.propellants import Propellant

g0 = 9.80665   # m/s²


@dataclass
class EnginePerformance:
    """Container for computed engine performance parameters."""
    # inputs
    Pc: float           # chamber pressure  [Pa]
    Pa: float           # ambient pressure  [Pa]
    Rt: float           # throat radius  [m]
    epsilon: float      # expansion ratio  Ae/At
    propellant_name: str

    # gas dynamics
    Me: float           # exit Mach number
    Pe: float           # exit pressure  [Pa]
    Pe_over_Pc: float
    Cf_ideal: float     # ideal thrust coefficient
    Cf_actual: float    # Cf with efficiency correction

    # propellant thermo
    c_star: float       # characteristic velocity  [m/s]
    gamma: float
    R_gas: float        # J/(kg·K)
    Tc: float           # K

    # performance
    At: float           # throat area  [m²]
    Ae: float           # exit area  [m²]
    thrust: float       # F  [N]
    Isp: float          # specific impulse  [s]
    Ve: float           # effective exhaust velocity  [m/s]
    m_dot: float        # mass flow rate  [kg/s]
    eta_Isp: float      # efficiency factor used


def compute_engine_performance(
    Pc: float,
    Pa: float,
    Rt: float,
    epsilon: float,
    prop: Propellant,
) -> EnginePerformance:
    """
    Compute key engine parameters.

    Parameters
    ----------
    Pc      : chamber (stagnation) pressure  [Pa]
    Pa      : ambient pressure  [Pa]
    Rt      : throat radius  [m]
    epsilon : nozzle expansion ratio  Ae/At
    prop    : Propellant instance

    Returns
    -------
    EnginePerformance dataclass
    """
    gamma = prop.gamma
    At = math.pi * Rt**2
    Ae = At * epsilon

    # Exit Mach from area ratio
    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)

    # Exit pressure (isentropic)
    Pe_over_Pc = isentropic_pressure_ratio(Me, gamma)
    Pe = Pe_over_Pc * Pc
    Pa_over_Pc = Pa / Pc

    # Thrust coefficient (ideal 1-D)
    Cf_ideal = thrust_coefficient(Me, gamma, Pe_over_Pc, Pa_over_Pc, epsilon)

    # Apply nozzle efficiency
    Cf_actual = Cf_ideal * prop.eta_Isp

    # Characteristic velocity
    c_star = prop.c_star   # already computed in Propellant.__post_init__

    # Performance outputs
    thrust = Cf_actual * Pc * At
    m_dot = Pc * At / c_star
    Isp = Cf_actual * c_star / g0
    Ve = Isp * g0

    return EnginePerformance(
        Pc=Pc, Pa=Pa, Rt=Rt, epsilon=epsilon,
        propellant_name=prop.name,
        Me=Me, Pe=Pe, Pe_over_Pc=Pe_over_Pc,
        Cf_ideal=Cf_ideal, Cf_actual=Cf_actual,
        c_star=c_star, gamma=gamma, R_gas=prop.R_gas, Tc=prop.Tc,
        At=At, Ae=Ae,
        thrust=thrust, Isp=Isp, Ve=Ve, m_dot=m_dot,
        eta_Isp=prop.eta_Isp,
    )
