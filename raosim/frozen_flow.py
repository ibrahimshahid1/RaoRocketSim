"""Thermally-perfect, frozen-composition quasi-one-dimensional nozzle flow.

The existing characteristic solvers use the constant-``gamma`` Prandtl-Meyer,
critical-Mach, compatibility, mass-flux, and thrust relations.  This module is
therefore deliberately separate: it does **not** pass a station-dependent
``gamma`` into MOC or Rao invariants.

For a fixed ideal-gas composition, ``R`` is constant while ``cp(T)`` may vary.
The implementation uses an explicit piecewise-linear ``cp`` table and exact
segment integrals:

``h(T2)-h(T1) = integral(cp dT)``

``s(T2,p2)-s(T1,p1) = integral(cp/T dT) - R ln(p2/p1)``

An isentropic station follows from total enthalpy and entropy.  The throat is
the sonic root ``u^2 = gamma(T) R T``; mass conservation then gives
``A/A* = G*/G``.  Exit thrust is reconstructed directly from momentum and
pressure, with ``c* = p0/G*``.

All quantities are SI.  Extrapolation outside the declared property table is
forbidden, equilibrium composition changes are excluded, and a property table
must identify its composition freeze basis, generation state/tool/database,
and upstream source-artifact digest.  Every result reports closure residuals
and a deterministic input fingerprint.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping


R_UNIVERSAL_J_MOL_K = 8.31446261815324
MODEL_ID = "thermally_perfect_frozen_composition_q1d_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FREEZE_BASES = frozenset(
    {
        "chamber_equilibrium_snapshot",
        "externally_fixed_composition",
        "manufactured_composition",
    }
)
_MIN_NORMAL_FLOAT = sys.float_info.min
_LOG_MIN_NORMAL_FLOAT = math.log(_MIN_NORMAL_FLOAT)


class FrozenFlowError(ValueError):
    """Raised when property evidence or a frozen-flow solve is ineligible."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise FrozenFlowError(f"{name} must be a finite number, not bool")
    value = float(value)
    if not math.isfinite(value):
        raise FrozenFlowError(f"{name} must be finite")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise FrozenFlowError(f"{name} must be > 0")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise FrozenFlowError(f"{name} must be >= 0")
    return value


def _safe_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise FrozenFlowError(f"{name} must be text")
    value = value.strip()
    if not value or any(character in value for character in "\r\n\x00"):
        raise FrozenFlowError(f"{name} must be one nonblank safe line")
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenFlowError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise FrozenFlowError(f"non-finite JSON numeric constant is forbidden: {value}")


def _require_json_number(name: str, value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "number or null" if nullable else "number"
        raise FrozenFlowError(f"{name} must be a JSON {qualifier}")


def _optional_positive(name: str, value: float | None) -> float | None:
    return None if value is None else _positive(name, value)


def _sha256(name: str, value: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise FrozenFlowError(
            f"{name} must be a 64-character lowercase SHA-256 digest"
        )
    return digest


def _usable_mass_flux(name: str, value: float) -> float:
    """Reject zero, subnormal, and non-finite fluxes before division.

    A subnormal mass flux is mathematically positive but is not a usable
    conservation denominator: subsequent area-ratio reconstruction may
    overflow or silently lose essentially all significant digits.
    """

    flux = _finite(name, value)
    if flux <= 0.0:
        raise FrozenFlowError(f"{name} underflowed to zero or is non-positive")
    if flux < _MIN_NORMAL_FLOAT:
        raise FrozenFlowError(f"{name} is a subnormal underflow value")
    return flux


def _mass_flux_ratio(numerator: float, denominator: float, *, name: str) -> float:
    numerator = _usable_mass_flux(f"{name} numerator mass flux", numerator)
    denominator = _usable_mass_flux(f"{name} denominator mass flux", denominator)
    ratio = numerator / denominator
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise FrozenFlowError(f"{name} mass-flux ratio is non-finite or non-positive")
    return ratio


@dataclass(frozen=True)
class FrozenIdealGasTable:
    """Fixed-composition ideal gas with a bounded piecewise-linear cp table.

    ``composition_mass_fractions`` names the composition held fixed during
    expansion.  The table must already represent that mixture; species
    chemistry is not reconstructed or equilibrated by this class.  A
    ``chamber_equilibrium_snapshot`` is valid only at its recorded chamber
    pressure and temperature; externally fixed and manufactured mixtures are
    not bound to one operating pressure by this solver.
    """

    molecular_weight_kg_mol: float
    composition_mass_fractions: Mapping[str, float]
    temperature_nodes_k: tuple[float, ...]
    cp_nodes_j_kg_k: tuple[float, ...]
    source: str
    freeze_basis: str
    composition_state_pressure_pa: float | None
    composition_state_temperature_k: float | None
    mixture_ratio: float | None
    generator: str
    generator_version: str
    thermo_database: str
    source_artifact_sha256: str
    input_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        mw = _positive("molecular_weight_kg_mol", self.molecular_weight_kg_mol)
        if not 1.0e-3 <= mw <= 0.5:
            raise FrozenFlowError(
                "molecular_weight_kg_mol must be in the physically bounded "
                "range [0.001, 0.5] kg/mol"
            )
        object.__setattr__(self, "molecular_weight_kg_mol", mw)

        try:
            temperatures = tuple(
                _positive(f"temperature_nodes_k[{i}]", value)
                for i, value in enumerate(self.temperature_nodes_k)
            )
            cps = tuple(
                _positive(f"cp_nodes_j_kg_k[{i}]", value)
                for i, value in enumerate(self.cp_nodes_j_kg_k)
            )
        except TypeError as exc:
            raise FrozenFlowError("temperature and cp nodes must be sequences") from exc
        if len(temperatures) < 2 or len(cps) != len(temperatures):
            raise FrozenFlowError(
                "temperature_nodes_k and cp_nodes_j_kg_k must have equal length >= 2"
            )
        if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise FrozenFlowError("temperature nodes must be strictly increasing")
        object.__setattr__(self, "temperature_nodes_k", temperatures)
        object.__setattr__(self, "cp_nodes_j_kg_k", cps)

        composition: dict[str, float] = {}
        if not isinstance(self.composition_mass_fractions, Mapping):
            raise FrozenFlowError("composition_mass_fractions must be a mapping")
        for raw_species, raw_fraction in self.composition_mass_fractions.items():
            species = _safe_text("composition species", raw_species)
            if species in composition:
                raise FrozenFlowError(f"duplicate composition species: {species}")
            composition[species] = _nonnegative(
                f"composition_mass_fractions[{species}]", raw_fraction
            )
        if not composition or not math.isclose(
            sum(composition.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise FrozenFlowError("composition mass fractions must sum to one")
        object.__setattr__(
            self,
            "composition_mass_fractions",
            MappingProxyType(dict(sorted(composition.items()))),
        )
        object.__setattr__(self, "source", _safe_text("source", self.source))
        freeze_basis = _safe_text("freeze_basis", self.freeze_basis)
        if freeze_basis not in _FREEZE_BASES:
            raise FrozenFlowError(
                "freeze_basis must be one of: " + ", ".join(sorted(_FREEZE_BASES))
            )
        object.__setattr__(self, "freeze_basis", freeze_basis)

        state_pressure = _optional_positive(
            "composition_state_pressure_pa", self.composition_state_pressure_pa
        )
        state_temperature = _optional_positive(
            "composition_state_temperature_k", self.composition_state_temperature_k
        )
        mixture_ratio = _optional_positive("mixture_ratio", self.mixture_ratio)
        if freeze_basis == "chamber_equilibrium_snapshot":
            missing = [
                name
                for name, value in (
                    ("composition_state_pressure_pa", state_pressure),
                    ("composition_state_temperature_k", state_temperature),
                    ("mixture_ratio", mixture_ratio),
                )
                if value is None
            ]
            if missing:
                raise FrozenFlowError(
                    "chamber_equilibrium_snapshot requires: " + ", ".join(missing)
                )
        if state_temperature is not None and not (
            temperatures[0] <= state_temperature <= temperatures[-1]
        ):
            raise FrozenFlowError(
                "composition_state_temperature_k must lie inside the cp table bounds"
            )
        object.__setattr__(self, "composition_state_pressure_pa", state_pressure)
        object.__setattr__(self, "composition_state_temperature_k", state_temperature)
        object.__setattr__(self, "mixture_ratio", mixture_ratio)
        object.__setattr__(self, "generator", _safe_text("generator", self.generator))
        object.__setattr__(
            self,
            "generator_version",
            _safe_text("generator_version", self.generator_version),
        )
        object.__setattr__(
            self, "thermo_database", _safe_text("thermo_database", self.thermo_database)
        )
        object.__setattr__(
            self,
            "source_artifact_sha256",
            _sha256("source_artifact_sha256", self.source_artifact_sha256),
        )
        if self.input_artifact_sha256 is not None:
            object.__setattr__(
                self,
                "input_artifact_sha256",
                _sha256("input_artifact_sha256", self.input_artifact_sha256),
            )

        minimum_cv = min(cp - self.gas_constant_j_kg_k for cp in cps)
        if minimum_cv <= 0.0:
            raise FrozenFlowError(
                "every cp node must exceed the mixture gas constant so cv > 0"
            )

    @property
    def gas_constant_j_kg_k(self) -> float:
        return R_UNIVERSAL_J_MOL_K / self.molecular_weight_kg_mol

    @property
    def minimum_temperature_k(self) -> float:
        return self.temperature_nodes_k[0]

    @property
    def maximum_temperature_k(self) -> float:
        return self.temperature_nodes_k[-1]

    @property
    def fingerprint_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "raosim.frozen_ideal_gas_table.v2",
            "model_id": MODEL_ID,
            "molecular_weight_kg_mol": self.molecular_weight_kg_mol,
            "gas_constant_j_kg_k": self.gas_constant_j_kg_k,
            "composition_mass_fractions": dict(self.composition_mass_fractions),
            "temperature_nodes_k": list(self.temperature_nodes_k),
            "cp_nodes_j_kg_k": list(self.cp_nodes_j_kg_k),
            "source": self.source,
            "freeze_basis": self.freeze_basis,
            "composition_state_pressure_pa": self.composition_state_pressure_pa,
            "composition_state_temperature_k": self.composition_state_temperature_k,
            "mixture_ratio": self.mixture_ratio,
            "generator": self.generator,
            "generator_version": self.generator_version,
            "thermo_database": self.thermo_database,
            "source_artifact_sha256": self.source_artifact_sha256,
            "input_artifact_sha256": self.input_artifact_sha256,
        }

    def check_operating_state(
        self, *, chamber_pressure_pa: float, chamber_temperature_k: float
    ) -> None:
        """Require a chamber-equilibrium composition to match its freeze state."""

        if self.freeze_basis != "chamber_equilibrium_snapshot":
            return
        pressure = _positive("chamber_pressure_pa", chamber_pressure_pa)
        temperature = self._check_temperature(chamber_temperature_k)
        assert self.composition_state_pressure_pa is not None
        assert self.composition_state_temperature_k is not None
        if not math.isclose(
            pressure,
            self.composition_state_pressure_pa,
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        ):
            raise FrozenFlowError(
                "operating chamber_pressure_pa does not match the "
                "chamber-equilibrium composition snapshot pressure"
            )
        if not math.isclose(
            temperature,
            self.composition_state_temperature_k,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise FrozenFlowError(
                "operating chamber_temperature_k does not match the "
                "chamber-equilibrium composition snapshot temperature"
            )

    def _check_temperature(self, temperature_k: float) -> float:
        temperature = _positive("temperature_k", temperature_k)
        if not self.minimum_temperature_k <= temperature <= self.maximum_temperature_k:
            raise FrozenFlowError(
                f"temperature {temperature:g} K is outside cp table bounds "
                f"[{self.minimum_temperature_k:g}, {self.maximum_temperature_k:g}] K"
            )
        return temperature

    def _segment(self, temperature_k: float) -> tuple[float, float]:
        temperature = self._check_temperature(temperature_k)
        index = min(
            max(bisect_right(self.temperature_nodes_k, temperature) - 1, 0),
            len(self.temperature_nodes_k) - 2,
        )
        t0 = self.temperature_nodes_k[index]
        t1 = self.temperature_nodes_k[index + 1]
        cp0 = self.cp_nodes_j_kg_k[index]
        cp1 = self.cp_nodes_j_kg_k[index + 1]
        slope = (cp1 - cp0) / (t1 - t0)
        intercept = cp0 - slope * t0
        return slope, intercept

    def cp(self, temperature_k: float) -> float:
        temperature = self._check_temperature(temperature_k)
        slope, intercept = self._segment(temperature)
        return slope * temperature + intercept

    def cv(self, temperature_k: float) -> float:
        return self.cp(temperature_k) - self.gas_constant_j_kg_k

    def gamma(self, temperature_k: float) -> float:
        cp = self.cp(temperature_k)
        return cp / (cp - self.gas_constant_j_kg_k)

    def enthalpy_change(self, from_temperature_k: float, to_temperature_k: float) -> float:
        """Return ``h(to)-h(from) = integral(cp dT)`` exactly per segment."""

        return self._integral(from_temperature_k, to_temperature_k, entropy=False)

    def standard_entropy_change(
        self, from_temperature_k: float, to_temperature_k: float
    ) -> float:
        """Return ``integral_from^to cp(T)/T dT`` exactly per segment."""

        return self._integral(from_temperature_k, to_temperature_k, entropy=True)

    def _integral(self, start: float, end: float, *, entropy: bool) -> float:
        start = self._check_temperature(start)
        end = self._check_temperature(end)
        if start == end:
            return 0.0
        sign = 1.0
        if end < start:
            start, end = end, start
            sign = -1.0
        breakpoints = [start]
        breakpoints.extend(
            value for value in self.temperature_nodes_k if start < value < end
        )
        breakpoints.append(end)
        total = 0.0
        for lower, upper in zip(breakpoints, breakpoints[1:]):
            slope, intercept = self._segment(0.5 * (lower + upper))
            if entropy:
                total += slope * (upper - lower) + intercept * math.log(upper / lower)
            else:
                total += (
                    0.5 * slope * (upper**2 - lower**2)
                    + intercept * (upper - lower)
                )
        return sign * total


@dataclass(frozen=True)
class FrozenFlowStation:
    """One static station on a fixed-composition isentrope."""

    branch: str
    area_ratio: float
    temperature_k: float
    pressure_pa: float
    pressure_ratio: float
    density_kg_m3: float
    velocity_m_s: float
    sound_speed_m_s: float
    mach: float
    cp_j_kg_k: float
    cv_j_kg_k: float
    gamma: float
    mass_flux_kg_m2_s: float
    energy_relative_residual: float
    entropy_relative_residual: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "area_ratio": self.area_ratio,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "pressure_ratio": self.pressure_ratio,
            "density_kg_m3": self.density_kg_m3,
            "velocity_m_s": self.velocity_m_s,
            "sound_speed_m_s": self.sound_speed_m_s,
            "mach": self.mach,
            "cp_j_kg_k": self.cp_j_kg_k,
            "cv_j_kg_k": self.cv_j_kg_k,
            "gamma": self.gamma,
            "mass_flux_kg_m2_s": self.mass_flux_kg_m2_s,
            "energy_relative_residual": self.energy_relative_residual,
            "entropy_relative_residual": self.entropy_relative_residual,
        }


@dataclass(frozen=True)
class FrozenNozzleExpansion:
    """Solved frozen-composition throat/exit state and performance closure."""

    gas: FrozenIdealGasTable
    chamber_pressure_pa: float
    chamber_temperature_k: float
    expansion_ratio: float
    ambient_pressure_pa: float
    throat: FrozenFlowStation
    exit: FrozenFlowStation
    characteristic_velocity_m_s: float
    momentum_thrust_coefficient: float
    pressure_thrust_coefficient: float
    thrust_coefficient: float
    sonic_relative_residual: float
    exit_area_relative_residual: float
    exit_mass_relative_residual: float
    input_fingerprint_sha256: str

    def station(self, area_ratio: float, *, supersonic: bool) -> FrozenFlowStation:
        """Solve another station using the same stagnation and throat state."""

        return _station_for_area_ratio(
            self.gas,
            self.chamber_pressure_pa,
            self.chamber_temperature_k,
            self.throat,
            area_ratio,
            supersonic=supersonic,
        )

    @property
    def all_closures_pass(self) -> bool:
        tolerance = 5.0e-10
        return (
            self.sonic_relative_residual <= tolerance
            and self.exit_area_relative_residual <= tolerance
            and self.exit_mass_relative_residual <= tolerance
            and self.throat.energy_relative_residual <= tolerance
            and self.throat.entropy_relative_residual <= tolerance
            and self.exit.energy_relative_residual <= tolerance
            and self.exit.entropy_relative_residual <= tolerance
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "raosim.frozen_nozzle_expansion.v1",
            "model_id": MODEL_ID,
            "composition_model": "fixed_mass_fractions_no_reaction",
            "property_model": "piecewise_linear_cp_exact_integrals",
            "geometry_model": "quasi_one_dimensional_area_ratio",
            "gas_fingerprint_sha256": self.gas.fingerprint_sha256,
            "input_fingerprint_sha256": self.input_fingerprint_sha256,
            "chamber_pressure_pa": self.chamber_pressure_pa,
            "chamber_temperature_k": self.chamber_temperature_k,
            "expansion_ratio": self.expansion_ratio,
            "ambient_pressure_pa": self.ambient_pressure_pa,
            "throat": self.throat.as_dict(),
            "exit": self.exit.as_dict(),
            "characteristic_velocity_m_s": self.characteristic_velocity_m_s,
            "momentum_thrust_coefficient": self.momentum_thrust_coefficient,
            "pressure_thrust_coefficient": self.pressure_thrust_coefficient,
            "thrust_coefficient": self.thrust_coefficient,
            "closures": {
                "sonic_relative_residual": self.sonic_relative_residual,
                "exit_area_relative_residual": self.exit_area_relative_residual,
                "exit_mass_relative_residual": self.exit_mass_relative_residual,
                "all_pass": self.all_closures_pass,
            },
            "applicability": {
                "frozen_composition": True,
                "ideal_gas": True,
                "inviscid": True,
                "adiabatic": True,
                "isentropic": True,
                "quasi_one_dimensional": True,
                "equilibrium_chemistry": False,
                "moc_or_rao_characteristics": False,
                "profile_aware_wall_pressure_screen": True,
                "empirical_separation_screen": True,
                "boundary_layer_model": False,
                "separation_flow_solution": False,
                "physically_validated": False,
            },
        }


def solve_frozen_nozzle_expansion(
    gas: FrozenIdealGasTable,
    *,
    chamber_pressure_pa: float,
    chamber_temperature_k: float,
    expansion_ratio: float,
    ambient_pressure_pa: float = 0.0,
) -> FrozenNozzleExpansion:
    """Solve the choked supersonic expansion of one fixed-composition gas."""

    if not isinstance(gas, FrozenIdealGasTable):
        raise FrozenFlowError("gas must be FrozenIdealGasTable")
    pressure = _positive("chamber_pressure_pa", chamber_pressure_pa)
    temperature = gas._check_temperature(chamber_temperature_k)
    gas.check_operating_state(
        chamber_pressure_pa=pressure, chamber_temperature_k=temperature
    )
    ratio = _finite("expansion_ratio", expansion_ratio)
    if ratio < 1.0:
        raise FrozenFlowError("expansion_ratio must be >= 1")
    ambient = _nonnegative("ambient_pressure_pa", ambient_pressure_pa)

    throat_temperature = _solve_throat_temperature(gas, temperature)
    throat_unscaled = _station_from_temperature(
        gas,
        pressure,
        temperature,
        throat_temperature,
        branch="sonic",
        area_ratio=1.0,
    )
    throat = throat_unscaled
    sonic_residual = abs(throat.velocity_m_s**2 - throat.sound_speed_m_s**2) / max(
        throat.sound_speed_m_s**2, 1.0e-30
    )
    exit_station = _station_for_area_ratio(
        gas,
        pressure,
        temperature,
        throat,
        ratio,
        supersonic=True,
    )
    throat_mass_flux = _usable_mass_flux(
        "throat mass flux", throat.mass_flux_kg_m2_s
    )
    exit_mass_flux = _usable_mass_flux("exit mass flux", exit_station.mass_flux_kg_m2_s)
    cstar = pressure / throat_mass_flux
    momentum_cf = throat_mass_flux * exit_station.velocity_m_s / pressure
    pressure_cf = ratio * (exit_station.pressure_pa - ambient) / pressure
    cf = momentum_cf + pressure_cf
    reconstructed_area = _mass_flux_ratio(
        throat_mass_flux, exit_mass_flux, name="exit area reconstruction"
    )
    area_residual = abs(reconstructed_area - ratio) / ratio
    mass_residual = abs(
        exit_mass_flux * ratio - throat_mass_flux
    ) / throat_mass_flux

    fingerprint = _canonical_sha256(
        {
            "model_id": MODEL_ID,
            "gas_fingerprint_sha256": gas.fingerprint_sha256,
            "chamber_pressure_pa": pressure,
            "chamber_temperature_k": temperature,
            "expansion_ratio": ratio,
            "ambient_pressure_pa": ambient,
        }
    )
    result = FrozenNozzleExpansion(
        gas=gas,
        chamber_pressure_pa=pressure,
        chamber_temperature_k=temperature,
        expansion_ratio=ratio,
        ambient_pressure_pa=ambient,
        throat=throat,
        exit=exit_station,
        characteristic_velocity_m_s=cstar,
        momentum_thrust_coefficient=momentum_cf,
        pressure_thrust_coefficient=pressure_cf,
        thrust_coefficient=cf,
        sonic_relative_residual=sonic_residual,
        exit_area_relative_residual=area_residual,
        exit_mass_relative_residual=mass_residual,
        input_fingerprint_sha256=fingerprint,
    )
    if not result.all_closures_pass:
        raise FrozenFlowError("frozen-flow conservation closure failed")
    return result


def expansion_ratio_from_pressure_frozen(
    gas: FrozenIdealGasTable,
    *,
    chamber_pressure_pa: float,
    chamber_temperature_k: float,
    exit_pressure_pa: float,
) -> tuple[float, FrozenFlowStation]:
    """Return the supersonic matched-expansion ``A/A*`` and exit station."""

    chamber_pressure = _positive("chamber_pressure_pa", chamber_pressure_pa)
    exit_pressure = _positive("exit_pressure_pa", exit_pressure_pa)
    if exit_pressure >= chamber_pressure:
        raise FrozenFlowError("exit_pressure_pa must be below chamber pressure")
    temperature = gas._check_temperature(chamber_temperature_k)
    seed = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=chamber_pressure,
        chamber_temperature_k=temperature,
        expansion_ratio=1.0,
        ambient_pressure_pa=exit_pressure,
    )
    target_ratio = exit_pressure / chamber_pressure
    if target_ratio >= seed.throat.pressure_ratio:
        raise FrozenFlowError(
            "requested matched pressure is above the sonic throat pressure; "
            "no supersonic divergent-nozzle solution exists"
        )

    def pressure_residual(static_temperature: float) -> float:
        station = _station_from_temperature(
            gas,
            chamber_pressure,
            temperature,
            static_temperature,
            branch="supersonic",
            area_ratio=1.0,
        )
        return station.pressure_ratio - target_ratio

    low = gas.minimum_temperature_k
    high = seed.throat.temperature_k
    if pressure_residual(low) > 0.0:
        raise FrozenFlowError(
            "cp table does not extend low enough to reach the requested exit pressure"
        )
    exit_temperature = _bisect(
        pressure_residual, low, high, name="matched exit pressure"
    )
    raw_station = _station_from_temperature(
        gas,
        chamber_pressure,
        temperature,
        exit_temperature,
        branch="supersonic",
        area_ratio=1.0,
    )
    ratio = _mass_flux_ratio(
        seed.throat.mass_flux_kg_m2_s,
        raw_station.mass_flux_kg_m2_s,
        name="matched expansion area reconstruction",
    )
    station = FrozenFlowStation(**{**raw_station.__dict__, "area_ratio": ratio})
    return ratio, station


def _solve_throat_temperature(gas: FrozenIdealGasTable, stagnation_temperature: float) -> float:
    low = gas.minimum_temperature_k
    high = stagnation_temperature
    if high <= low:
        raise FrozenFlowError(
            "cp table must extend below the chamber stagnation temperature"
        )

    def residual(temperature: float) -> float:
        velocity_squared = 2.0 * gas.enthalpy_change(temperature, high)
        sound_squared = gas.gamma(temperature) * gas.gas_constant_j_kg_k * temperature
        return velocity_squared - sound_squared

    if residual(low) <= 0.0:
        raise FrozenFlowError(
            "cp table does not extend to a low enough temperature to bracket the sonic throat"
        )
    return _bisect(residual, low, high, name="sonic throat")


def _station_for_area_ratio(
    gas: FrozenIdealGasTable,
    chamber_pressure: float,
    chamber_temperature: float,
    throat: FrozenFlowStation,
    area_ratio: float,
    *,
    supersonic: bool,
) -> FrozenFlowStation:
    ratio = _finite("area_ratio", area_ratio)
    if ratio < 1.0:
        raise FrozenFlowError("area_ratio must be >= 1")
    if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-13):
        return throat

    def residual(temperature: float) -> float:
        station = _station_from_temperature(
            gas,
            chamber_pressure,
            chamber_temperature,
            temperature,
            branch="supersonic" if supersonic else "subsonic",
            area_ratio=ratio,
        )
        return _mass_flux_ratio(
            throat.mass_flux_kg_m2_s,
            station.mass_flux_kg_m2_s,
            name="station area residual",
        ) - ratio

    if supersonic:
        low = gas.minimum_temperature_k
        high = throat.temperature_k
        if residual(low) < 0.0:
            maximum = ratio + residual(low)
            raise FrozenFlowError(
                f"cp table lower bound supports A/A* only to {maximum:.6g}, "
                f"below requested {ratio:.6g}; extend the table to lower T"
            )
    else:
        low = throat.temperature_k
        # Avoid the exact zero-velocity stagnation endpoint.
        high = math.nextafter(chamber_temperature, low)
        if residual(high) < 0.0:
            raise FrozenFlowError("subsonic branch failed to bracket the area ratio")
    solved_temperature = _bisect(
        residual,
        low,
        high,
        name="supersonic area" if supersonic else "subsonic area",
    )
    station = _station_from_temperature(
        gas,
        chamber_pressure,
        chamber_temperature,
        solved_temperature,
        branch="supersonic" if supersonic else "subsonic",
        area_ratio=ratio,
    )
    actual_ratio = _mass_flux_ratio(
        throat.mass_flux_kg_m2_s,
        station.mass_flux_kg_m2_s,
        name="station area reconstruction",
    )
    return FrozenFlowStation(
        **{**station.__dict__, "area_ratio": actual_ratio}
    )


def _station_from_temperature(
    gas: FrozenIdealGasTable,
    chamber_pressure: float,
    chamber_temperature: float,
    static_temperature: float,
    *,
    branch: str,
    area_ratio: float,
) -> FrozenFlowStation:
    temperature = gas._check_temperature(static_temperature)
    available_enthalpy = gas.enthalpy_change(temperature, chamber_temperature)
    if available_enthalpy < 0.0:
        raise FrozenFlowError("static temperature cannot exceed stagnation temperature")
    velocity = math.sqrt(max(2.0 * available_enthalpy, 0.0))
    entropy_thermal = gas.standard_entropy_change(
        chamber_temperature, temperature
    )
    log_pressure_ratio = entropy_thermal / gas.gas_constant_j_kg_k
    if log_pressure_ratio < _LOG_MIN_NORMAL_FLOAT:
        raise FrozenFlowError(
            "static pressure ratio is below the minimum normal float; "
            "property range or requested expansion causes numerical underflow"
        )
    pressure_ratio = math.exp(log_pressure_ratio)
    if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
        raise FrozenFlowError("static pressure ratio is non-finite or non-positive")
    pressure = chamber_pressure * pressure_ratio
    density = pressure / (gas.gas_constant_j_kg_k * temperature)
    cp = gas.cp(temperature)
    cv = cp - gas.gas_constant_j_kg_k
    gamma = cp / cv
    sound = math.sqrt(gamma * gas.gas_constant_j_kg_k * temperature)
    mach = velocity / sound
    mass_flux = density * velocity
    _usable_mass_flux("station mass flux", mass_flux)
    energy_scale = max(gas.enthalpy_change(gas.minimum_temperature_k, chamber_temperature), 1.0)
    energy_residual = abs(available_enthalpy - 0.5 * velocity**2) / energy_scale
    entropy_residual_dimensional = abs(
        entropy_thermal - gas.gas_constant_j_kg_k * math.log(pressure_ratio)
    )
    entropy_scale = max(abs(entropy_thermal), gas.gas_constant_j_kg_k, 1.0)
    entropy_residual = entropy_residual_dimensional / entropy_scale
    return FrozenFlowStation(
        branch=branch,
        area_ratio=area_ratio,
        temperature_k=temperature,
        pressure_pa=pressure,
        pressure_ratio=pressure_ratio,
        density_kg_m3=density,
        velocity_m_s=velocity,
        sound_speed_m_s=sound,
        mach=mach,
        cp_j_kg_k=cp,
        cv_j_kg_k=cv,
        gamma=gamma,
        mass_flux_kg_m2_s=mass_flux,
        energy_relative_residual=energy_residual,
        entropy_relative_residual=entropy_residual,
    )


def _bisect(function, low: float, high: float, *, name: str) -> float:
    f_low = function(low)
    f_high = function(high)
    if not math.isfinite(f_low) or not math.isfinite(f_high):
        raise FrozenFlowError(f"{name} root has non-finite bracket residual")
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise FrozenFlowError(
            f"{name} root is not bracketed: f(low)={f_low:.6g}, f(high)={f_high:.6g}"
        )
    for _ in range(200):
        middle = 0.5 * (low + high)
        f_middle = function(middle)
        if not math.isfinite(f_middle):
            raise FrozenFlowError(f"{name} root produced a non-finite residual")
        if abs(f_middle) <= 1.0e-13 or abs(high - low) <= 1.0e-12 * max(middle, 1.0):
            return middle
        if f_low * f_middle <= 0.0:
            high = middle
            f_high = f_middle
        else:
            low = middle
            f_low = f_middle
    return 0.5 * (low + high)


def load_frozen_gas_table(path: str | Path) -> FrozenIdealGasTable:
    """Load the strict JSON cp-table schema and bind the source-file hash."""

    source_path = Path(path).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise FrozenFlowError("frozen gas table must be a normal JSON file")
    raw = source_path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenFlowError(f"invalid frozen gas JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenFlowError("frozen gas JSON root must be an object")
    required = {
        "schema_version",
        "model",
        "molecular_weight_kg_mol",
        "composition_mass_fractions",
        "temperature_nodes_k",
        "cp_nodes_j_kg_k",
        "source",
        "freeze_basis",
        "composition_state_pressure_pa",
        "composition_state_temperature_k",
        "mixture_ratio",
        "generator",
        "generator_version",
        "thermo_database",
        "source_artifact_sha256",
    }
    if set(payload) != required:
        raise FrozenFlowError(
            "frozen gas JSON keys must exactly equal: " + ", ".join(sorted(required))
        )
    if payload["schema_version"] != 2 or payload["model"] != MODEL_ID:
        raise FrozenFlowError(
            f"frozen gas JSON must use schema_version 2 and model {MODEL_ID!r}"
        )
    if not isinstance(payload["composition_mass_fractions"], dict):
        raise FrozenFlowError("composition_mass_fractions must be an object")
    if not isinstance(payload["temperature_nodes_k"], list):
        raise FrozenFlowError("temperature_nodes_k must be an array")
    if not isinstance(payload["cp_nodes_j_kg_k"], list):
        raise FrozenFlowError("cp_nodes_j_kg_k must be an array")
    _require_json_number(
        "molecular_weight_kg_mol", payload["molecular_weight_kg_mol"]
    )
    for index, value in enumerate(payload["composition_mass_fractions"].values()):
        _require_json_number(f"composition_mass_fractions value {index}", value)
    for index, value in enumerate(payload["temperature_nodes_k"]):
        _require_json_number(f"temperature_nodes_k[{index}]", value)
    for index, value in enumerate(payload["cp_nodes_j_kg_k"]):
        _require_json_number(f"cp_nodes_j_kg_k[{index}]", value)
    for name in (
        "composition_state_pressure_pa",
        "composition_state_temperature_k",
        "mixture_ratio",
    ):
        _require_json_number(name, payload[name], nullable=True)
    return FrozenIdealGasTable(
        molecular_weight_kg_mol=payload["molecular_weight_kg_mol"],
        composition_mass_fractions=payload["composition_mass_fractions"],
        temperature_nodes_k=tuple(payload["temperature_nodes_k"]),
        cp_nodes_j_kg_k=tuple(payload["cp_nodes_j_kg_k"]),
        source=payload["source"],
        freeze_basis=payload["freeze_basis"],
        composition_state_pressure_pa=payload["composition_state_pressure_pa"],
        composition_state_temperature_k=payload["composition_state_temperature_k"],
        mixture_ratio=payload["mixture_ratio"],
        generator=payload["generator"],
        generator_version=payload["generator_version"],
        thermo_database=payload["thermo_database"],
        source_artifact_sha256=payload["source_artifact_sha256"],
        input_artifact_sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "FrozenFlowError",
    "FrozenFlowStation",
    "FrozenIdealGasTable",
    "FrozenNozzleExpansion",
    "MODEL_ID",
    "load_frozen_gas_table",
    "expansion_ratio_from_pressure_frozen",
    "solve_frozen_nozzle_expansion",
]
