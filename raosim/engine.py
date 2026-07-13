"""
engine.py – Rocket engine performance computed from nozzle + propellant.

Combines gas-dynamics relations with propellant thermochemistry to produce
thrust, Isp, mass-flow rate, and related performance metrics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raosim.gas_dynamics import (
    thrust_coefficient,
    isentropic_pressure_ratio,
    mach_from_area_ratio,
)
from raosim.propellants import Propellant

if TYPE_CHECKING:
    from raosim.frozen_flow import FrozenIdealGasTable, FrozenNozzleExpansion

g0 = 9.80665   # m/s²


@dataclass
class EnginePerformance:
    """Container for computed engine performance parameters."""

    Pc: float           # chamber pressure  [Pa]
    Pa: float           # ambient pressure  [Pa]
    Rt: float           # throat radius  [m]
    epsilon: float      # expansion ratio  Ae/At
    propellant_name: str


    Me: float           # exit Mach number
    Pe: float           # exit pressure  [Pa]
    Pe_over_Pc: float
    Cf_ideal: float     # ideal thrust coefficient
    Cf_actual: float    # Cf with efficiency correction


    c_star: float       # characteristic velocity  [m/s]
    gamma: float
    R_gas: float        # J/(kg·K)
    Tc: float           # K


    At: float           # throat area  [m²]
    Ae: float           # exit area  [m²]
    thrust: float       # F  [N]
    Isp: float          # specific impulse  [s]
    Ve: float           # effective exhaust velocity  [m/s]
    m_dot: float        # mass flow rate  [kg/s]
    eta_Isp: float      # overall Isp efficiency used  (= eta_cstar·eta_CF)
    eta_cstar: float    # combustion (c*) efficiency
    eta_CF: float       # nozzle (thrust-coefficient) efficiency
    c_star_effective: float  # delivered c*  (= c_star·eta_cstar)  [m/s]
    expansion_model: str = "constant_gamma"
    gamma_throat: float | None = None
    gamma_exit: float | None = None
    exit_temperature: float | None = None
    frozen_flow_fingerprint: str | None = None
    reference_propellant_c_star: float | None = None
    frozen_flow: "FrozenNozzleExpansion | None" = None


def compute_engine_performance(
    Pc: float,
    Pa: float,
    Rt: float,
    epsilon: float,
    prop: Propellant,
    *,
    frozen_gas: "FrozenIdealGasTable | None" = None,
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
    if frozen_gas is not None:
        return compute_frozen_engine_performance(
            Pc=Pc,
            Pa=Pa,
            Rt=Rt,
            epsilon=epsilon,
            prop=prop,
            frozen_gas=frozen_gas,
        )

    gamma = prop.gamma
    At = math.pi * Rt**2
    Ae = At * epsilon


    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)


    Pe_over_Pc = isentropic_pressure_ratio(Me, gamma)
    Pe = Pe_over_Pc * Pc
    Pa_over_Pc = Pa / Pc


    Cf_ideal = thrust_coefficient(Me, gamma, Pe_over_Pc, Pa_over_Pc, epsilon)


    # Nozzle efficiency acts on the thrust coefficient; combustion
    # efficiency acts on c* (and therefore on the mass flow needed to hold
    # Pc).  Isp = Cf_actual·c*_eff/g0 = eta_CF·eta_cstar·Isp_ideal, so the
    # net Isp multiplier is unchanged from a lumped eta_Isp while thrust and
    # mass flow are now attributed to the correct physics.
    Cf_actual = Cf_ideal * prop.eta_CF
    c_star = prop.c_star
    c_star_eff = c_star * prop.eta_cstar


    thrust = Cf_actual * Pc * At
    m_dot = Pc * At / c_star_eff
    Isp = Cf_actual * c_star_eff / g0
    Ve = Isp * g0

    return EnginePerformance(
        Pc=Pc, Pa=Pa, Rt=Rt, epsilon=epsilon,
        propellant_name=prop.name,
        Me=Me, Pe=Pe, Pe_over_Pc=Pe_over_Pc,
        Cf_ideal=Cf_ideal, Cf_actual=Cf_actual,
        c_star=c_star, gamma=gamma, R_gas=prop.R_gas, Tc=prop.Tc,
        At=At, Ae=Ae,
        thrust=thrust, Isp=Isp, Ve=Ve, m_dot=m_dot,
        eta_Isp=prop.eta_Isp, eta_cstar=prop.eta_cstar, eta_CF=prop.eta_CF,
        c_star_effective=c_star_eff,
        expansion_model="constant_gamma",
        gamma_throat=gamma,
        gamma_exit=gamma,
        exit_temperature=prop.Tc * (
            1.0 + 0.5 * (gamma - 1.0) * Me**2
        ) ** -1.0,
        reference_propellant_c_star=c_star,
    )


def compute_frozen_engine_performance(
    *,
    Pc: float,
    Pa: float,
    Rt: float,
    epsilon: float,
    prop: Propellant,
    frozen_gas: "FrozenIdealGasTable",
) -> EnginePerformance:
    """Compute performance from the separate variable-cp frozen Q1D solve.

    The gas table must represent the same molecular weight as ``prop``.  This
    prevents a property table for one composition from being combined with the
    mass-flow/efficiency identity of another propellant.  Existing constant-
    gamma callers remain on :func:`compute_engine_performance` unchanged.
    """

    from raosim.frozen_flow import (
        FrozenIdealGasTable,
        solve_frozen_nozzle_expansion,
    )

    if not isinstance(frozen_gas, FrozenIdealGasTable):
        raise TypeError("frozen_gas must be FrozenIdealGasTable")
    if Pc <= 0.0 or Rt <= 0.0 or epsilon < 1.0 or Pa < 0.0:
        raise ValueError(
            "frozen performance requires Pc>0, Rt>0, epsilon>=1, and Pa>=0"
        )
    mw_relative_error = abs(
        frozen_gas.molecular_weight_kg_mol - prop.Mw
    ) / max(abs(prop.Mw), 1.0e-30)
    if mw_relative_error > 1.0e-3:
        raise ValueError(
            "frozen gas molecular weight does not match the selected propellant "
            f"({mw_relative_error:.3%} relative difference)"
        )

    expansion = solve_frozen_nozzle_expansion(
        frozen_gas,
        chamber_pressure_pa=Pc,
        chamber_temperature_k=prop.Tc,
        expansion_ratio=epsilon,
        ambient_pressure_pa=Pa,
    )
    At = math.pi * Rt**2
    Ae = At * epsilon
    cf_ideal = expansion.thrust_coefficient
    cf_actual = cf_ideal * prop.eta_CF
    c_star = expansion.characteristic_velocity_m_s
    c_star_effective = c_star * prop.eta_cstar
    thrust = cf_actual * Pc * At
    mass_flow = Pc * At / c_star_effective
    specific_impulse = cf_actual * c_star_effective / g0

    return EnginePerformance(
        Pc=Pc,
        Pa=Pa,
        Rt=Rt,
        epsilon=epsilon,
        propellant_name=prop.name,
        Me=expansion.exit.mach,
        Pe=expansion.exit.pressure_pa,
        Pe_over_Pc=expansion.exit.pressure_ratio,
        Cf_ideal=cf_ideal,
        Cf_actual=cf_actual,
        c_star=c_star,
        gamma=frozen_gas.gamma(prop.Tc),
        R_gas=frozen_gas.gas_constant_j_kg_k,
        Tc=prop.Tc,
        At=At,
        Ae=Ae,
        thrust=thrust,
        Isp=specific_impulse,
        Ve=specific_impulse * g0,
        m_dot=mass_flow,
        eta_Isp=prop.eta_Isp,
        eta_cstar=prop.eta_cstar,
        eta_CF=prop.eta_CF,
        c_star_effective=c_star_effective,
        expansion_model="frozen_variable_cp_q1d",
        gamma_throat=expansion.throat.gamma,
        gamma_exit=expansion.exit.gamma,
        exit_temperature=expansion.exit.temperature_k,
        frozen_flow_fingerprint=expansion.input_fingerprint_sha256,
        reference_propellant_c_star=float(prop.c_star),
        frozen_flow=expansion,
    )
