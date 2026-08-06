"""
cea.py - Optional RocketCEA/NASA CEA thermochemistry integration.

RocketCEA is intentionally optional. When it is not installed, callers can
fall back to the built-in propellant table while preserving an explicit warning
that the values are demo-grade constants rather than CEA-derived properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from raosim.gas_dynamics import characteristic_velocity
from raosim.propellants import Propellant, get_propellant


THERMO_CONSTANT_GAMMA = "constant_gamma"
THERMO_CEA_FROZEN = "cea_frozen"
THERMO_CEA_EQUILIBRIUM = "cea_equilibrium"
THERMO_PINNED_CHAMBER = "pinned_chamber_state"
CEA_THERMO_MODES = {THERMO_CEA_FROZEN, THERMO_CEA_EQUILIBRIUM}
THERMO_MODES = {
    THERMO_CONSTANT_GAMMA,
    THERMO_PINNED_CHAMBER,
    *CEA_THERMO_MODES,
}


@dataclass(frozen=True)
class PinnedChamberState:
    """One immutable calorically-perfect chamber state used for parity.

    This is the host-side counterpart of the chamber-property values retained
    in ``raosim.mdo.state.EngineState``.  It lets ``design_nozzle_v2`` consume
    exactly the state solved by the MDO instead of silently resampling live CEA
    (or falling back to a different built-in propellant record).
    """

    gamma: float
    Tc: float
    R_gas: float
    c_star_ideal: float
    source: str
    surface_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("gamma", "Tc", "R_gas", "c_star_ideal"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PinnedChamberState.{name} must be finite and positive")
        if float(self.gamma) <= 1.0:
            raise ValueError("PinnedChamberState.gamma must be greater than one")
        if not str(self.source).strip():
            raise ValueError("PinnedChamberState.source must identify its provenance")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedChamberState":
        return cls(
            gamma=float(value["gamma"]),
            Tc=float(value["Tc"]),
            R_gas=float(value["R_gas"]),
            c_star_ideal=float(value["c_star_ideal"]),
            source=str(value["source"]),
            surface_fingerprint=(
                str(value["surface_fingerprint"])
                if value.get("surface_fingerprint") is not None
                else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gamma": float(self.gamma),
            "Tc": float(self.Tc),
            "R_gas": float(self.R_gas),
            "c_star_ideal": float(self.c_star_ideal),
            "source": str(self.source),
            "surface_fingerprint": self.surface_fingerprint,
        }


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
    eta_cstar: float | None = None,
    eta_CF: float | None = None,
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
    return _with_efficiency_overrides(
        prop,
        eta_cstar=eta_cstar,
        eta_CF=eta_CF,
    )


def _with_efficiency_overrides(
    propellant: Propellant,
    *,
    eta_cstar: float | None,
    eta_CF: float | None,
) -> Propellant:
    """Clone ``propellant`` with optional delivered-efficiency overrides.

    With neither override, return the provider's existing propellant unchanged:
    built-in constant-gamma entries retain their database split, while a CEA
    propellant retains the historical ``eta_Isp``-only convention.  A partial
    override inherits the other component from that resolved baseline.  This
    makes the explicit pair suitable for MDO/traditional parity without
    changing legacy callers.
    """
    if eta_cstar is None and eta_CF is None:
        return propellant

    clone = Propellant(
        name=propellant.name,
        gamma=propellant.gamma,
        Mw=propellant.Mw,
        Tc=propellant.Tc,
        eta_cstar=(
            float(propellant.eta_cstar)
            if eta_cstar is None else float(eta_cstar)
        ),
        eta_CF=(
            float(propellant.eta_CF)
            if eta_CF is None else float(eta_CF)
        ),
        OF=propellant.OF,
        source=propellant.source,
    )
    # RocketCEA supplies c* directly; rebuilding a Propellant from its chamber
    # snapshot would otherwise replace that value with the constant-gamma
    # reconstruction.  Preserving it is harmless and exact for table entries.
    clone.c_star = float(propellant.c_star)
    return clone


def resolve_thermochemistry(
    *,
    thermo_mode: str,
    propellant_name: str | None,
    Pc: float,
    mixture_ratio: float | None = None,
    oxidizer: str | None = None,
    fuel: str | None = None,
    eta_Isp: float = 0.95,
    eta_cstar: float | None = None,
    eta_CF: float | None = None,
    epsilon: float | None = None,
    require_cea: bool = False,
    pinned_chamber_state: PinnedChamberState | Mapping[str, Any] | None = None,
) -> ThermochemistryResult:
    """Resolve thermochemistry for preliminary or validated design workflows."""
    if thermo_mode not in THERMO_MODES:
        raise ValueError(
            "thermo_mode must be one of: " + ", ".join(sorted(THERMO_MODES))
        )
    if pinned_chamber_state is not None and thermo_mode != THERMO_PINNED_CHAMBER:
        raise ValueError(
            "pinned_chamber_state requires thermo_mode='pinned_chamber_state'"
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
    if thermo_mode == THERMO_PINNED_CHAMBER:
        if require_cea:
            raise RuntimeError(
                "validated mode requires an independent CEA thermochemistry "
                "evaluation; a pinned MDO chamber state is parity evidence only"
            )
        if pinned_chamber_state is None:
            raise ValueError(
                "thermo_mode='pinned_chamber_state' requires "
                "pinned_chamber_state"
            )
        pinned = (
            pinned_chamber_state
            if isinstance(pinned_chamber_state, PinnedChamberState)
            else PinnedChamberState.from_mapping(pinned_chamber_state)
        )
        if not propellant_name:
            raise ValueError(
                "propellant_name is required with a pinned chamber state"
            )
        reconstructed_cstar = characteristic_velocity(
            float(pinned.gamma),
            float(pinned.R_gas),
            float(pinned.Tc),
        )
        if not math.isclose(
            reconstructed_cstar,
            float(pinned.c_star_ideal),
            rel_tol=1.0e-10,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "pinned chamber c_star_ideal is inconsistent with its "
                "gamma/R_gas/Tc calorically-perfect convention"
            )
        prop = Propellant(
            name=str(propellant_name),
            gamma=float(pinned.gamma),
            Mw=8.314462618 / float(pinned.R_gas),
            Tc=float(pinned.Tc),
            eta_Isp=float(eta_Isp),
            eta_cstar=eta_cstar,
            eta_CF=eta_CF,
            OF=float(mixture_ratio) if mixture_ratio is not None else 0.0,
            source=str(pinned.source),
        )
        # Preserve the exact state values.  Propellant reconstructs R from a
        # rounded historical Ru constant, so leaving its derived values in place
        # would introduce a small but needless parity error.
        prop.R_gas = float(pinned.R_gas)
        prop.c_star = float(pinned.c_star_ideal)
        warnings.append(
            "Using the chamber-property snapshot pinned in the solved MDO "
            "EngineState for common-input parity. This is not an independent "
            "RocketCEA re-evaluation and remains preliminary thermochemistry."
        )
        return ThermochemistryResult(
            propellant=prop,
            mode=thermo_mode,
            source=str(pinned.source),
            cea_available=rocketcea_available(),
            chamber_state={
                **pinned.as_dict(),
                "Mw": prop.Mw,
                "eta_cstar": prop.eta_cstar,
                "eta_CF": prop.eta_CF,
                "eta_Isp": prop.eta_Isp,
                "mixture_ratio": mixture_ratio,
            },
            warnings=warnings,
        )

    if thermo_mode == THERMO_CONSTANT_GAMMA:
        if require_cea:
            raise RuntimeError("validated mode requires CEA thermochemistry")
        if not propellant_name:
            raise ValueError("propellant_name is required for constant-gamma mode")
        prop = _with_efficiency_overrides(
            get_propellant(propellant_name),
            eta_cstar=eta_cstar,
            eta_CF=eta_CF,
        )
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
                "eta_cstar": prop.eta_cstar,
                "eta_CF": prop.eta_CF,
                "eta_Isp": prop.eta_Isp,
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
            eta_cstar=eta_cstar,
            eta_CF=eta_CF,
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
        fallback = _with_efficiency_overrides(
            get_propellant(propellant_name),
            eta_cstar=eta_cstar,
            eta_CF=eta_CF,
        )
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
                "eta_cstar": fallback.eta_cstar,
                "eta_CF": fallback.eta_CF,
                "eta_Isp": fallback.eta_Isp,
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
            "eta_cstar": prop.eta_cstar,
            "eta_CF": prop.eta_CF,
            "eta_Isp": prop.eta_Isp,
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
    eta_cstar: float | None = None,
    eta_CF: float | None = None,
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
        eta_cstar=eta_cstar,
        eta_CF=eta_CF,
        require_cea=False,
    )
    return result.propellant, result.warnings
