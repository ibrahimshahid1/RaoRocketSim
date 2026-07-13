"""
cea.py - Optional RocketCEA/NASA CEA thermochemistry integration.

RocketCEA is intentionally optional. When it is not installed, callers can
fall back to the built-in propellant table while preserving an explicit warning
that the values are demo-grade constants rather than CEA-derived properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raosim.propellants import Propellant, get_propellant


THERMO_CONSTANT_GAMMA = "constant_gamma"
THERMO_CEA_FROZEN = "cea_frozen"
THERMO_CEA_EQUILIBRIUM = "cea_equilibrium"
CEA_THERMO_MODES = {THERMO_CEA_FROZEN, THERMO_CEA_EQUILIBRIUM}
THERMO_MODES = {THERMO_CONSTANT_GAMMA, *CEA_THERMO_MODES}


@dataclass
class ThermochemistryResult:
    """Resolved combustion-product properties and provenance."""

    propellant: Propellant
    mode: str
    source: str
    cea_available: bool
    chamber_state: dict[str, Any] = field(default_factory=dict)
    exit_state: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def rocketcea_available() -> bool:
    try:
        import rocketcea  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def cea_propellant(
    *,
    oxidizer: str,
    fuel: str,
    Pc: float,
    mixture_ratio: float,
    eta_Isp: float = 0.95,
    thermo_mode: str = THERMO_CEA_FROZEN,
    epsilon: float | None = None,
) -> Propellant:
    """
    Build a Propellant from RocketCEA chamber properties.

    Parameters use SI units except RocketCEA's mixture ratio convention.
    """
    try:
        from rocketcea.cea_obj_w_units import CEA_Obj  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RocketCEA is not installed; install rocketcea or run without --cea."
        ) from exc

    if thermo_mode not in CEA_THERMO_MODES:
        raise ValueError("thermo_mode must be 'cea_frozen' or 'cea_equilibrium'")
    if thermo_mode == THERMO_CEA_EQUILIBRIUM:
        raise NotImplementedError(
            "cea_equilibrium is not implemented by the current constant-gamma "
            "nozzle model.  RocketCEA equilibrium exit properties cannot be "
            "represented by a single chamber gamma; use cea_frozen or add a "
            "variable-composition expansion model."
        )

    cea = CEA_Obj(
        oxName=oxidizer,
        fuelName=fuel,
        pressure_units="Pa",
        temperature_units="K",
        cstar_units="m/s",
    )
    mw_gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc, MR=mixture_ratio)
    if isinstance(mw_gamma, tuple) and len(mw_gamma) >= 2:
        mw, gamma = float(mw_gamma[0]), float(mw_gamma[1])
    else:
        raise RuntimeError("RocketCEA did not return chamber molecular weight/gamma.")

    Tc = float(cea.get_Tcomb(Pc=Pc, MR=mixture_ratio))
    c_star = float(cea.get_Cstar(Pc=Pc, MR=mixture_ratio))
    prop = Propellant(
        name=f"CEA {oxidizer}/{fuel} ({thermo_mode})",
        gamma=gamma,
        Mw=mw / 1000.0,
        Tc=Tc,
        eta_Isp=eta_Isp,
        OF=mixture_ratio,
    )
    prop.c_star = c_star
    return prop


def resolve_thermochemistry(
    *,
    thermo_mode: str,
    propellant_name: str | None,
    Pc: float,
    mixture_ratio: float | None = None,
    oxidizer: str | None = None,
    fuel: str | None = None,
    eta_Isp: float = 0.95,
    epsilon: float | None = None,
    require_cea: bool = False,
) -> ThermochemistryResult:
    """Resolve thermochemistry for preliminary or validated design workflows."""
    if thermo_mode not in THERMO_MODES:
        raise ValueError(
            "thermo_mode must be one of: " + ", ".join(sorted(THERMO_MODES))
        )

    # Do not accept two public mode names that execute the same calculation.
    # The nozzle/performance stack presently carries one calorically-perfect
    # chamber gamma through the entire expansion, so an equilibrium-shifting
    # CEA expansion would be falsely labelled if allowed here.
    if thermo_mode == THERMO_CEA_EQUILIBRIUM:
        raise NotImplementedError(
            "cea_equilibrium is unsupported until the nozzle solver consumes "
            "station-dependent equilibrium composition/properties.  Select "
            "cea_frozen for the supported chamber-snapshot approximation."
        )

    warnings: list[str] = []
    if thermo_mode == THERMO_CONSTANT_GAMMA:
        if require_cea:
            raise RuntimeError("validated mode requires CEA thermochemistry")
        if not propellant_name:
            raise ValueError("propellant_name is required for constant-gamma mode")
        prop = get_propellant(propellant_name)
        warnings.append(
            "Using built-in constant-gamma propellant properties; "
            "results are preliminary only."
        )
        return ThermochemistryResult(
            propellant=prop,
            mode=thermo_mode,
            source="built_in_constant_gamma",
            cea_available=rocketcea_available(),
            chamber_state={
                "gamma": prop.gamma,
                "Mw": prop.Mw,
                "Tc": prop.Tc,
                "c_star": prop.c_star,
            },
            warnings=warnings,
        )

    ox = oxidizer
    fu = fuel
    if (ox is None or fu is None) and propellant_name and "/" in propellant_name:
        ox, fu = propellant_name.split("/", 1)
    if not ox or not fu or mixture_ratio is None:
        raise ValueError(
            "CEA thermochemistry requires oxidizer, fuel, and mixture_ratio/O-F."
        )

    try:
        prop = cea_propellant(
            oxidizer=ox,
            fuel=fu,
            Pc=Pc,
            mixture_ratio=mixture_ratio,
            eta_Isp=eta_Isp,
            thermo_mode=thermo_mode,
            epsilon=epsilon,
        )
    except Exception as exc:
        if require_cea:
            raise RuntimeError(f"CEA thermochemistry required but unavailable/failed: {exc}") from exc
        if not propellant_name:
            raise RuntimeError(
                "CEA failed and no built-in propellant fallback was provided."
            ) from exc
        fallback = get_propellant(propellant_name)
        warnings.append(f"CEA requested but unavailable/failed: {exc}")
        warnings.append(
            f"Using built-in {fallback.name} constants instead; "
            "results are preliminary only."
        )
        return ThermochemistryResult(
            propellant=fallback,
            mode=thermo_mode,
            source="built_in_fallback_after_cea_failure",
            cea_available=False,
            chamber_state={
                "gamma": fallback.gamma,
                "Mw": fallback.Mw,
                "Tc": fallback.Tc,
                "c_star": fallback.c_star,
            },
            warnings=warnings,
        )

    warnings.append(
        "RocketCEA supplies the chamber thermochemical snapshot only. The "
        "downstream nozzle still uses one calorically-perfect chamber gamma; "
        "this is a constant-gamma frozen-composition approximation, not a "
        "station-resolved CEA frozen-performance expansion."
    )
    return ThermochemistryResult(
        propellant=prop,
        mode=thermo_mode,
        source="rocketcea",
        cea_available=True,
        chamber_state={
            "gamma": prop.gamma,
            "Mw": prop.Mw,
            "Tc": prop.Tc,
            "c_star": prop.c_star,
            "mixture_ratio": mixture_ratio,
            "oxidizer": ox,
            "fuel": fu,
        },
        exit_state={
            "epsilon": epsilon,
            "thermo_mode": thermo_mode,
            "expansion_model": "constant_gamma_chamber_snapshot",
            "station_resolved_cea_exit_properties": False,
            "composition_evolution": "frozen_by_approximation_at_chamber_state",
        },
        warnings=warnings,
    )


def propellant_from_request(
    *,
    propellant_name: str | None,
    use_cea: bool,
    Pc: float,
    mixture_ratio: float | None = None,
    oxidizer: str | None = None,
    fuel: str | None = None,
    eta_Isp: float = 0.95,
) -> tuple[Propellant, list[str]]:
    """Resolve propellant data and return warnings from any fallback path."""
    mode = THERMO_CEA_FROZEN if use_cea else THERMO_CONSTANT_GAMMA
    result = resolve_thermochemistry(
        thermo_mode=mode,
        propellant_name=propellant_name,
        Pc=Pc,
        mixture_ratio=mixture_ratio,
        oxidizer=oxidizer,
        fuel=fuel,
        eta_Isp=eta_Isp,
        require_cea=False,
    )
    return result.propellant, result.warnings
