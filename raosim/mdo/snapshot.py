"""Versioned host-side output contract for engine analyses.

The differentiable MDO core intentionally returns JAX-native numerical
objects, while the traditional design workflow returns a rich
``ValidatedDesignResult`` containing reports and artifacts.  This module is
the non-differentiated boundary between those two worlds:

* :func:`snapshot_from_mdo` converts an ``EngineResult`` into a stable,
  JSON-ready analysis contract.
* :func:`snapshot_from_traditional` converts a ``ValidatedDesignResult`` (and
  optional electric-pump sizing) into the same contract.
* :func:`compare_snapshots` compares every common scalar and every common
  normalized axial profile.

Missing physics is data, not zero.  Every unavailable value is represented by
``SnapshotValue(value=None, availability_reason="...")``.  In particular, the
current MDO chamber and injector mass placeholders are never exported as real
zero-mass hardware.

This is deliberately a host-side module.  It may use NumPy, rebuild a station
grid for reporting, preserve Python result objects, and carry CAD/report paths;
none of it is part of a jitted or differentiated function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTRACT_NAME = "raosim.engine-analysis-snapshot"
SNAPSHOT_CONTRACT_VERSION = "2.0.0"
CONTRACT_VERSION = SNAPSHOT_CONTRACT_VERSION
FILM_INJECTOR_UNAVAILABLE_REASON = (
    "a separate film injector/orifice and branch state are not modeled; the "
    "retained main-pintle screening calculation uses total fuel and is not a "
    "physical core-fuel injector result when film flow is nonzero"
)
SNAPSHOT_FIELD_MANIFEST: dict[str, tuple[str, ...]] = {
    "performance": (
        "chamber_pressure_pa", "ambient_pressure_pa", "thrust_n",
        "specific_impulse_delivered_s", "specific_impulse_ideal_s",
        "mass_flow_total_kg_s", "mass_flow_fuel_kg_s",
        "mass_flow_oxidizer_kg_s", "mixture_ratio", "cf_ideal",
        "cf_delivered", "c_star_ideal_m_s", "c_star_delivered_m_s",
        "eta_cstar", "eta_cf", "exit_mach", "exit_pressure_pa",
    ),
    "geometry": (
        "throat_radius_m", "throat_area_m2", "expansion_ratio",
        "exit_radius_m", "length_pct", "contraction_ratio", "l_star_m",
        "throat_upstream_radius_ratio", "throat_downstream_radius_ratio",
        # Chamber closure and wetted area.  These exist so a chamber-geometry
        # convention drift between the two pipelines fails a parity comparison
        # loudly instead of hiding inside heat load and hardware mass -- which
        # is exactly what an 11.7% wetted-area gap did until 2026-07-31.
        "chamber_barrel_length_m", "chamber_volume_m3",
        "chamber_volume_target_m3", "wetted_area_m2",
        "axial_coordinate_m", "radius_profile_m", "area_ratio_profile",
        "mach_profile",
    ),
    "thermal": (
        "gas_side_wall_temperature_max_k",
        "coolant_side_wall_temperature_max_k", "heat_flux_max_w_m2",
        "gas_side_coefficient_max_w_m2_k", "thermal_stress_max_pa",
        "pressure_stress_pa", "combined_stress_max_pa",
        "thermal_stress_profile", "pressure_stress_profile",
        "combined_stress_profile",
        "gas_side_wall_temperature_profile",
        "coolant_side_wall_temperature_profile", "heat_flux_profile",
        "gas_side_coefficient_profile",
    ),
    "cooling": (
        "method", "coolant_name", "coolant_mass_flow_kg_s",
        "film_mass_flow_kg_s", "film_fraction_of_fuel",
        "fuel_flow_topology", "fuel_flow_closure_residual_kg_s",
        "coolant_inlet_temperature_k", "coolant_outlet_temperature_k",
        "coolant_pressure_drop_pa", "coolant_velocity_m_s", "coolant_mach",
        "land_width_min_m", "coolant_temperature_profile",
        "coolant_pressure_profile", "gas_pressure_profile",
        "liner_pressure_differential_profile",
    ),
    "injector": (
        "type", "architecture", "sizing", "fuel_dp_fraction",
        "oxidizer_dp_fraction", "fuel_dp_pa", "oxidizer_dp_pa", "fuel_cd",
        "oxidizer_cd", "pintle_diameter_m", "slot_count", "momentum_ratio",
        "spray_half_angle_deg", "blockage_factor", "transition_margin_m2",
        "fuel_velocity_m_s", "oxidizer_velocity_m_s", "fuel_flow_area_m2",
        "oxidizer_flow_area_m2", "slot_width_m", "tip_opening_m",
        "tip_branch_area_m2", "center_gap_area_m2", "branch_consistency",
        "fuel_chug_margin", "oxidizer_chug_margin",
    ),
    "feed_electrical": (
        "architecture", "fuel_tank_pressure_pa", "oxidizer_tank_pressure_pa",
        "fuel_density_kg_m3", "oxidizer_density_kg_m3",
        "fuel_vapor_pressure_pa", "oxidizer_vapor_pressure_pa",
        "line_pressure_loss_pa", "fuel_required_pressure_rise_pa",
        "oxidizer_required_pressure_rise_pa", "pump_speed_rpm",
        "fuel_volumetric_flow_m3_s", "oxidizer_volumetric_flow_m3_s",
        "fuel_pump_head_m", "oxidizer_pump_head_m", "fuel_specific_speed",
        "oxidizer_specific_speed", "fuel_npsh_available_pa",
        "oxidizer_npsh_available_pa", "fuel_suction_specific_speed",
        "oxidizer_suction_specific_speed", "fuel_nss_margin",
        "oxidizer_nss_margin", "fuel_tip_speed_m_s",
        "oxidizer_tip_speed_m_s", "fuel_tip_speed_margin_m_s",
        "oxidizer_tip_speed_margin_m_s", "fuel_pump_efficiency",
        "oxidizer_pump_efficiency", "fuel_hydraulic_power_w",
        "oxidizer_hydraulic_power_w", "fuel_shaft_power_w",
        "oxidizer_shaft_power_w", "electric_power_total_w",
        "motor_efficiency", "inverter_efficiency",
    ),
    "masses": (
        "pump_mass_kg", "motor_mass_kg", "inverter_mass_kg",
        "battery_energy_limited_mass_kg", "battery_power_limited_mass_kg",
        "battery_energy_installed_mass_kg",
        "battery_power_installed_mass_kg", "battery_selected_mass_kg",
        "battery_objective_mass_kg", "electric_package_mass_kg",
        "electric_package_objective_mass_kg",
        "dry_mass_partial_exact_mass_kg",
        "dry_mass_partial_objective_mass_kg",
        "thrust_chamber_liner_mass_kg", "thrust_chamber_land_mass_kg",
        "thrust_chamber_closeout_mass_kg", "thrust_chamber_mass_kg",
        "injector_mass_kg", "total_engine_package_mass_kg",
        "engine_hardware_mass_ledger",
        "raw_mass_ledger",
    ),
    "constraints_gates": (
        "all_constraints_feasible", "optimizer_constraints_feasible",
        "numerical_validity", "physics_feasible", "requirements_feasible",
        "workflow_readiness_feasible", "constraint_margins", "diagnostics",
        "outer_thrust_residual", "cooling_residual_max_abs_k",
        "authoritative_design_gates",
    ),
    "provenance": (
        "analysis_source", "contract_version", "propellant", "coolant",
        "thermochemistry", "geometry_model", "material_assumptions",
        "input_conventions", "mission", "design",
    ),
    "artifacts": ("files", "report_sections", "cad_files"),
}


def _json_ready(value: Any) -> Any:
    """Return a JSON-compatible representation without losing ``None``."""

    if isinstance(value, float) and not np.isfinite(value):
        # Raw provenance/report payloads can contain mathematical sentinels
        # such as an intentionally disabled ``max_heat_flux=inf``.  Preserve
        # that meaning explicitly; a bare JSON null is indistinguishable from
        # an unsupported or accidentally missing value.
        if np.isnan(value):
            description = "NaN"
        elif value > 0.0:
            description = "positive infinity"
        else:
            description = "negative infinity"
        return {
            "value": None,
            "availability_reason": (
                f"source metadata contained {description}; JSON has no finite "
                "numeric representation for this value"
            ),
        }
    if isinstance(value, SnapshotValue):
        return value.to_dict()
    if isinstance(value, NormalizedProfile):
        return value.to_dict()
    if isinstance(value, SnapshotSection):
        return value.to_dict()
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    # JAX scalar/array objects implement __array__, but ordinary Python
    # objects (including the preserved source result) should not be coerced.
    if (
        not isinstance(value, (str, bytes, int, float, bool, type(None)))
        and hasattr(value, "__array__")
    ):
        arr = np.asarray(value)
        return _json_ready(arr.item() if arr.ndim == 0 else arr.tolist())
    return value


def _host_value(value: Any) -> Any:
    """Convert a numerical JAX/NumPy value to a stable host representation."""

    if value is None:
        return None
    if isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _host_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_host_value(v) for v in value)
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.ndim == 0:
        item = arr.item()
        return float(item) if isinstance(item, (np.floating, float)) else item
    try:
        return np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return value


def _nested_nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    """Return paths to every non-finite real/complex number in ``value``."""

    if isinstance(value, SnapshotValue):
        return _nested_nonfinite_paths(value.value, f"{path}.value")
    if is_dataclass(value):
        return _nested_nonfinite_paths(asdict(value), path)
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_nested_nonfinite_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (tuple, list)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nested_nonfinite_paths(item, f"{path}[{index}]"))
        return paths
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            indices = np.argwhere(~np.isfinite(value))
            return [
                path + "".join(f"[{int(i)}]" for i in index)
                for index in indices
            ]
        return _nested_nonfinite_paths(value.tolist(), path)
    if isinstance(value, np.generic):
        return _nested_nonfinite_paths(value.item(), path)
    if isinstance(value, (float, complex)):
        return [] if np.isfinite(value) else [path]
    if (
        not isinstance(value, (str, bytes, int, bool, type(None), Path))
        and hasattr(value, "__array__")
    ):
        try:
            return _nested_nonfinite_paths(np.asarray(value), path)
        except Exception:
            return []
    return []


@dataclass(frozen=True)
class SnapshotValue:
    """One contract value with explicit availability metadata."""

    value: Any
    availability_reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None and not self.availability_reason:
            raise ValueError(
                "an unavailable snapshot value requires availability_reason"
            )
        if self.value is not None and self.availability_reason is not None:
            raise ValueError(
                "an available snapshot value cannot have availability_reason"
            )

    @property
    def available(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _json_ready(self.value),
            "availability_reason": self.availability_reason,
        }


def available(value: Any) -> SnapshotValue:
    """Build an available contract value."""

    if value is None:
        raise ValueError("available() cannot wrap None")
    host = _host_value(value)
    nonfinite_paths = _nested_nonfinite_paths(host)
    if nonfinite_paths:
        preview = ", ".join(nonfinite_paths[:8])
        if len(nonfinite_paths) > 8:
            preview += f", … ({len(nonfinite_paths)} total)"
        raise ValueError(
            "available() cannot wrap non-finite numerical values at "
            f"{preview}; use unavailable() with a reason"
        )
    return SnapshotValue(host)


def unavailable(reason: str) -> SnapshotValue:
    """Build an explicitly unavailable contract value."""

    return SnapshotValue(None, str(reason))


def _film_injector_sensitive(
    value: SnapshotValue, film_mass_flow: float
) -> SnapshotValue:
    """Hide total-fuel main-pintle surrogates when fuel is split to film."""

    return (
        unavailable(FILM_INJECTOR_UNAVAILABLE_REASON)
        if film_mass_flow > 0.0
        else value
    )


def maybe(value: Any, reason: str) -> SnapshotValue:
    """Build an available value, or an unavailable value with ``reason``."""

    if value is None:
        return unavailable(reason)
    try:
        return available(value)
    except ValueError as exc:
        return unavailable(f"{reason}; resolved value was invalid: {exc}")


def _tri_state_value(
    status: str,
    *,
    unknown_reason: str,
) -> SnapshotValue:
    """Represent pass/fail/unknown without coercing unknown to ``False``."""

    if status == "pass":
        return available(True)
    if status == "fail":
        return available(False)
    if status == "unknown":
        return unavailable(unknown_reason)
    raise ValueError(f"unsupported feasibility status {status!r}")


def _constraint_margin_payload(
    specs: Any,
    values: np.ndarray,
    applicable: np.ndarray,
    availability: np.ndarray,
    required: np.ndarray,
    reason_codes: np.ndarray,
    mission: Any,
) -> dict[str, Any]:
    """Build JSON-safe row records for the shared constraint manifest."""

    from raosim.mdo.constraints import reason_text

    payload: dict[str, Any] = {}
    for index, spec in enumerate(specs):
        row_value = float(values[index])
        row_reason = reason_text(int(reason_codes[index]), spec, mission)
        if bool(applicable[index]) and bool(availability[index]):
            if np.isfinite(row_value):
                value: float | None = row_value
            else:
                value = None
                row_reason = "constraint margin is non-finite"
        else:
            value = None
        payload[str(spec.engine_key)] = {
            "name": spec.name,
            "kind": spec.kind,
            "optimizer_role": spec.optimizer_role,
            "category": spec.category,
            "units": spec.units,
            "source_id": spec.source_id,
            "value": value,
            "applicable": bool(applicable[index]),
            "available": bool(availability[index]),
            "required": bool(required[index]),
            "reason": row_reason or None,
        }
    return payload


@dataclass(frozen=True)
class NormalizedProfile:
    """A profile on a throat-aware normalized axial coordinate.

    The chamber/convergent side spans ``[-1, 0]`` and the divergent side spans
    ``[0, 1]``.  This keeps the throat aligned when the MDO fixed grid and the
    authoritative contour contain different station counts and physical
    lengths.
    """

    coordinate: np.ndarray
    values: np.ndarray
    coordinate_name: str = "throat_normalized_axial"
    units: str | None = None

    def __post_init__(self) -> None:
        c = np.asarray(self.coordinate, dtype=float)
        v = np.asarray(self.values, dtype=float)
        if c.ndim != 1 or v.ndim != 1 or c.shape != v.shape:
            raise ValueError("profile coordinate and values must be same-length 1-D arrays")
        if c.size < 2:
            raise ValueError("a comparable profile requires at least two stations")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)):
            raise ValueError("profile coordinate and values must be finite")
        object.__setattr__(self, "coordinate", c)
        object.__setattr__(self, "values", v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate.tolist(),
            "values": self.values.tolist(),
            "coordinate_name": self.coordinate_name,
            "units": self.units,
        }


@dataclass(frozen=True)
class SnapshotSection:
    """Named values belonging to one output-contract section."""

    fields: dict[str, SnapshotValue]

    def __getitem__(self, key: str) -> SnapshotValue:
        return self.fields[key]

    def get(self, key: str, default: Any = None) -> SnapshotValue | Any:
        return self.fields.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {name: value.to_dict() for name, value in self.fields.items()}


@dataclass(frozen=True)
class EngineAnalysisSnapshot:
    """Host-side, versioned engine-analysis output contract."""

    source: str
    performance: SnapshotSection
    geometry: SnapshotSection
    thermal: SnapshotSection
    cooling: SnapshotSection
    injector: SnapshotSection
    feed_electrical: SnapshotSection
    masses: SnapshotSection
    constraints_gates: SnapshotSection
    provenance: SnapshotSection
    warnings: tuple[str, ...]
    artifacts: SnapshotSection
    optimizer_metadata: dict[str, Any] | None = None
    source_result: Any = field(default=None, repr=False, compare=False)
    auxiliary_results: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    contract_name: str = CONTRACT_NAME
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_name != CONTRACT_NAME or self.contract_version != CONTRACT_VERSION:
            raise ValueError(
                "snapshot identity/version does not match the current contract"
            )
        for section_name, expected in SNAPSHOT_FIELD_MANIFEST.items():
            section = getattr(self, section_name)
            actual = tuple(section.fields)
            if actual != expected:
                missing = [name for name in expected if name not in actual]
                extra = [name for name in actual if name not in expected]
                raise ValueError(
                    f"{section_name} does not match snapshot contract "
                    f"{CONTRACT_VERSION}; missing={missing}, extra={extra}, "
                    "or field order changed"
                )

    def section(self, name: str) -> SnapshotSection:
        return getattr(self, name)

    @property
    def authoritative_result(self) -> Any:
        """The preserved traditional result, when this is an authoritative snapshot."""

        return self.source_result if self.source == "traditional" else None

    @property
    def cad_artifacts(self) -> dict[str, str]:
        """CAD paths exposed through the contract (empty when unavailable)."""

        field_value = self.artifacts.get("cad_files")
        if field_value is None or not field_value.available:
            return {}
        return dict(field_value.value)

    def report_payload(self) -> dict[str, Any]:
        """Stable payload for report writers and downstream API responses."""

        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "source": self.source,
            "performance": self.performance.to_dict(),
            "geometry": self.geometry.to_dict(),
            "thermal": self.thermal.to_dict(),
            "cooling": self.cooling.to_dict(),
            "injector": self.injector.to_dict(),
            "feed_electrical": self.feed_electrical.to_dict(),
            "masses": self.masses.to_dict(),
            "constraints_gates": self.constraints_gates.to_dict(),
            "provenance": self.provenance.to_dict(),
            "warnings": list(self.warnings),
            "artifacts": self.artifacts.to_dict(),
            "optimizer_metadata": _json_ready(self.optimizer_metadata),
            "source_result_preserved": self.source_result is not None,
            "auxiliary_result_names": sorted(self.auxiliary_results),
        }


def _design_dict(design: Any) -> dict[str, Any]:
    if isinstance(design, Mapping):
        return dict(design)
    if hasattr(design, "as_dict"):
        return dict(design.as_dict())
    if is_dataclass(design):
        return asdict(design)
    raise TypeError("design must be a mapping, DesignVector, or dataclass")


def _normal_coordinate(x: Any, radius: Any) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    r_arr = np.asarray(radius, dtype=float)
    if x_arr.ndim != 1 or r_arr.shape != x_arr.shape:
        raise ValueError("x and radius must be same-length 1-D arrays")
    throat = int(np.nanargmin(r_arr))
    x_t = float(x_arr[throat])
    out = np.zeros_like(x_arr)
    upstream_span = x_t - float(np.nanmin(x_arr[: throat + 1]))
    downstream_span = float(np.nanmax(x_arr[throat:])) - x_t
    if throat:
        out[:throat] = (
            (x_arr[:throat] - x_t) / max(upstream_span, 1.0e-30)
        )
    if throat + 1 < x_arr.size:
        out[throat + 1 :] = (
            (x_arr[throat + 1 :] - x_t) / max(downstream_span, 1.0e-30)
        )
    return out


def _profile(
    x: Any,
    radius: Any,
    values: Any,
    *,
    units: str | None = None,
) -> SnapshotValue:
    try:
        return available(
            NormalizedProfile(
                _normal_coordinate(x, radius),
                np.asarray(values, dtype=float),
                units=units,
            )
        )
    except Exception as exc:
        return unavailable(f"profile could not be normalized: {type(exc).__name__}: {exc}")


def _mdo_design_value(design: Mapping[str, Any], key: str) -> Any:
    value = design.get(key)
    return _host_value(value)


def snapshot_from_mdo(
    result: Any,
    design: Any | None = None,
    mission: Any | None = None,
    *,
    optimizer_metadata: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    surfaces: Any | None = None,
) -> EngineAnalysisSnapshot:
    """Convert a pure-JAX ``EngineState`` or legacy ``EngineResult``.

    Preferred use is ``snapshot_from_mdo(state, mission=mission)`` after
    :func:`raosim.mdo.state.solve_engine_state`.  When that solve used an
    explicit ``ChamberSurfaces`` object, pass the same object as ``surfaces=``;
    its complete evaluator fingerprint is validated against the state.  The
    compatibility form
    ``snapshot_from_mdo(result, design, mission)`` remains available for code
    that still owns an ``EngineResult``.
    """

    if mission is None and design is not None and hasattr(design, "thrust"):
        # Convenient positional form: snapshot_from_mdo(state, mission).
        mission, design = design, None
    if mission is None:
        raise TypeError("snapshot_from_mdo requires mission=")
    if _looks_like_engine_state(result):
        _validate_engine_state_schema(result)
        if design is not None:
            _validate_engine_state_design(result, design)
        return _snapshot_from_engine_state(
            result,
            mission,
            optimizer_metadata=optimizer_metadata,
            artifacts=artifacts,
            surfaces=surfaces,
        )
    if design is None:
        raise TypeError(
            "legacy EngineResult conversion requires design; use "
            "snapshot_from_mdo(result, design, mission)"
        )
    if surfaces is not None:
        raise TypeError(
            "surfaces= is supported only for the versioned EngineState path; "
            "legacy EngineResult does not retain a verifiable surface identity"
        )
    return _snapshot_from_engine_result(
        result,
        design,
        mission,
        optimizer_metadata=optimizer_metadata,
        artifacts=artifacts,
    )


def _looks_like_engine_state(value: Any) -> bool:
    return (
        hasattr(value, "schema_version")
        and hasattr(value, "performance")
        and hasattr(value, "geometry")
        and hasattr(value, "residuals")
        and not hasattr(value, "cooling")
    )


def _validate_engine_state_schema(state: Any) -> None:
    """Reject states whose numerical field semantics are not version 2."""

    from raosim.mdo.state import ENGINE_STATE_SCHEMA_VERSION

    try:
        actual = int(np.asarray(state.schema_version))
    except Exception as exc:
        raise ValueError(
            "EngineState schema_version is not a scalar integer"
        ) from exc
    if actual != ENGINE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported EngineState schema version: "
            f"state={actual}, adapter={ENGINE_STATE_SCHEMA_VERSION}; "
            "v1 O/F semantics cannot be relabeled or migrated safely; "
            "re-solve with the original MissionSpec and property surfaces"
        )


def _validate_engine_state_design(state: Any, design: Any) -> None:
    """Reject a host adapter call that pairs a state with another design."""

    from raosim.mdo.schema import DesignVector

    supplied = _design_dict(design)
    names = DesignVector.names()
    missing = [name for name in names if name not in supplied]
    if missing:
        raise ValueError(
            "EngineState/design mismatch: supplied design is missing "
            + ", ".join(missing)
        )
    expected = np.asarray(
        [float(supplied[name]) for name in names], dtype=float
    )
    actual = np.asarray(state.design_vector, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(
            "EngineState/design mismatch: state design-vector shape "
            f"{actual.shape} != supplied shape {expected.shape}"
        )
    if not np.allclose(
        actual, expected, rtol=1.0e-12, atol=1.0e-12
    ):
        differences = [
            f"{name}: state={left:.12g}, supplied={right:.12g}"
            for name, left, right in zip(
                names, actual.tolist(), expected.tolist(), strict=True
            )
            if not np.isclose(left, right, rtol=1.0e-12, atol=1.0e-12)
        ]
        raise ValueError(
            "EngineState/design mismatch: " + "; ".join(differences)
        )


def _validate_engine_state_mission(
    state: Any,
    mission: Any,
    *,
    surfaces: Any | None = None,
) -> None:
    """Reject a host adapter call that pairs a state with different inputs."""

    from raosim.mdo.engine import chamber_surfaces_for
    from raosim.mdo.state import (
        mission_fingerprint_words,
        surface_signature,
    )

    conventions = getattr(state, "input_conventions", None)
    if conventions is None:
        raise ValueError(
            "EngineState is missing its numerical input-convention block; "
            "regenerate it with the current versioned state adapter"
        )
    field_map = {
        "thrust": "thrust",
        "ambient_pressure": "Pa",
        "burn_time": "burn_time",
        "mission_OF": "OF",
        "eta_cstar_nominal": "eta_cstar",
        "eta_CF": "eta_CF",
        "throat_ru_factor": "throat_ru_factor",
        "throat_rd_factor": "throat_rd_factor",
        "contraction_ratio": "contraction_ratio",
        "l_star": "l_star",
        "length_pct": "length_pct",
        "cooling_fraction": "cooling_fraction",
        "coolant_temperature": "coolant_temperature",
        "rho_coolant": "rho_cool",
        "cp_coolant": "cp_cool",
        "k_coolant": "k_cool",
        "mu_coolant": "mu_cool",
        "rho_fuel": "rho_fuel",
        "rho_oxidizer": "rho_ox",
        "vapor_pressure_fuel": "p_vapor_fuel",
        "vapor_pressure_oxidizer": "p_vapor_ox",
        "tank_pressure_fuel": "P_tank_fuel",
        "tank_pressure_oxidizer": "P_tank_ox",
        "line_dp_allowance": "line_dp_allowance",
        "injector_cd_fuel": "injector_cd_fuel",
        "injector_cd_oxidizer": "injector_cd_ox",
        "pintle_slot_count": "pintle_slot_count",
        "pump_efficiency_nominal": "eta_pump",
        "pump_head_coefficient": "pump_head_coefficient",
        "pump_tip_speed_max": "pump_tip_speed_max",
        "pump_nss_max": "pump_nss_max",
        "motor_efficiency": "eta_motor",
        "inverter_efficiency": "eta_inverter",
        "discharge_efficiency": "eta_discharge",
        "battery_energy_density": "battery_energy_density",
        "battery_power_density": "battery_power_density",
        "battery_structural_margin": "battery_structural_margin",
        "motor_power_density": "motor_power_density",
        "inverter_power_density": "inverter_power_density",
        "pump_specific_mass": "pump_specific_mass",
        "liner_conductivity": "k_wall",
        "liner_elastic_modulus": "liner_E",
        "liner_thermal_expansion": "liner_alpha",
        "liner_poisson": "liner_poisson",
        "liner_allowable_stress": "liner_sigma_allow",
        "liner_structural_fos": "liner_structural_fos",
        "liner_max_gas_side_temperature": "liner_T_wg_max",
        "channel_count": "n_channels",
        "film_capacity_margin": "film_capacity_margin",
        "film_system_capacity_fraction": "film_system_capacity_fraction",
    }
    mismatches: list[str] = []
    expected_mission_fingerprint = np.asarray(
        mission_fingerprint_words(mission), dtype=np.uint32
    )
    state_mission_fingerprint = np.asarray(
        conventions.mission_fingerprint, dtype=np.uint32
    )
    if not np.array_equal(
        state_mission_fingerprint, expected_mission_fingerprint
    ):
        mismatches.append(
            "full MissionSpec fingerprint differs (including one or more "
            "physics, geometry, film, feed, material, or solver assumptions)"
        )
    try:
        expected_surfaces = (
            surfaces if surfaces is not None else chamber_surfaces_for(mission)
        )
        expected_surface_signature = np.asarray(
            surface_signature(expected_surfaces), dtype=np.uint32
        )
    except Exception as exc:
        mismatches.append(
            "chamber-property surfaces could not be reconstructed from the "
            f"MissionSpec: {type(exc).__name__}: {exc}"
        )
    else:
        state_surface_signature = np.asarray(
            conventions.surface_signature, dtype=np.uint32
        )
        if (
            state_surface_signature.shape != expected_surface_signature.shape
            or not np.array_equal(
                state_surface_signature, expected_surface_signature
            )
        ):
            mismatches.append(
                "chamber-property surface signature differs from the "
                "surface table used to solve the EngineState"
            )
    expected_propellant_code = int.from_bytes(
        hashlib.sha256(
            str(mission.propellant_name).strip().lower().encode("utf-8")
        ).digest()[:4],
        "big",
    )
    state_propellant_code = int(
        np.asarray(conventions.propellant_name_code)
    )
    if state_propellant_code != expected_propellant_code:
        mismatches.append(
            "propellant_name: numerical identity code does not match the "
            "MissionSpec name"
        )
    for state_name, mission_name in field_map.items():
        state_value = float(np.asarray(getattr(conventions, state_name)))
        mission_value = float(getattr(mission, mission_name))
        if not np.isclose(
            state_value, mission_value, rtol=1.0e-12, atol=1.0e-12
        ):
            mismatches.append(
                f"{mission_name}: state={state_value:.12g}, "
                f"mission={mission_value:.12g}"
            )
    from raosim.mdo.schema import validate_mixture_ratio

    try:
        effective_of = validate_mixture_ratio(
            conventions.OF, name="EngineState.input_conventions.OF"
        )
    except ValueError as exc:
        mismatches.append(str(exc))
    else:
        performance_of = float(np.asarray(state.performance.OF))
        contract_of = float(np.asarray(state.design_vector, dtype=float)[-1])
        if not np.isclose(
            effective_of, performance_of, rtol=1.0e-12, atol=1.0e-12
        ):
            mismatches.append(
                "effective OF differs between input conventions and performance"
            )
        if not np.isclose(
            effective_of, contract_of, rtol=1.0e-12, atol=1.0e-12
        ):
            mismatches.append(
                "effective OF differs between input conventions and the "
                "fixed design contract"
            )
        if (
            not bool(np.asarray(conventions.of_is_variable))
            and not np.isclose(
                effective_of,
                float(mission.OF),
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            mismatches.append(
                "fixed-layout effective OF differs from MissionSpec.OF"
            )
    pump_speed = float(np.asarray(state.design_vector, dtype=float)[5])
    if not np.isclose(
        float(np.asarray(conventions.pump_speed_rpm)),
        pump_speed,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        mismatches.append(
            "pump_speed_rpm does not match the state design vector"
        )
    if mismatches:
        raise ValueError(
            "EngineState/MissionSpec convention mismatch: "
            + "; ".join(mismatches)
        )


def _snapshot_from_engine_state(
    state: Any,
    mission: Any,
    *,
    optimizer_metadata: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any] | None,
    surfaces: Any | None,
) -> EngineAnalysisSnapshot:
    """Host adapter for the fixed-shape pure-JAX ``EngineState``."""

    from raosim.mdo.engine import chamber_surfaces_for
    from raosim.mdo.schema import DesignVector
    from raosim.mdo.state import (
        MASS_FIELD_NAMES,
        MASS_UNAVAILABLE_REASONS,
    )
    from raosim.mdo.constraints import (
        ENGINE_CONSTRAINT_SPECS,
        status_from_rows,
    )

    resolved_surfaces = (
        surfaces if surfaces is not None else chamber_surfaces_for(mission)
    )
    _validate_engine_state_mission(
        state, mission, surfaces=resolved_surfaces
    )
    p = state.performance
    g = state.geometry
    t = state.thermal
    i = state.injector
    fp = state.fuel_pump
    op = state.oxidizer_pump
    e = state.electrical
    c = state.input_conventions
    design_values = np.asarray(state.design_vector, dtype=float)
    d = dict(zip(DesignVector.names(), design_values, strict=True))
    gx = np.asarray(g.x, dtype=float)
    gr = np.asarray(g.r, dtype=float)

    performance = SnapshotSection({
        "chamber_pressure_pa": available(p.Pc),
        "ambient_pressure_pa": available(p.Pa),
        "thrust_n": available(p.thrust_delivered),
        "specific_impulse_delivered_s": available(p.Isp_delivered),
        "specific_impulse_ideal_s": available(p.Isp_ideal),
        "mass_flow_total_kg_s": available(p.mdot_total),
        "mass_flow_fuel_kg_s": available(p.mdot_fuel_total),
        "mass_flow_oxidizer_kg_s": available(p.mdot_oxidizer),
        "mixture_ratio": available(p.OF),
        "cf_ideal": available(p.Cf_ideal),
        "cf_delivered": available(p.Cf_delivered),
        "c_star_ideal_m_s": available(p.cstar_ideal),
        "c_star_delivered_m_s": available(p.cstar_delivered),
        "eta_cstar": available(p.eta_cstar),
        "eta_cf": available(p.eta_CF),
        "exit_mach": available(p.Me),
        "exit_pressure_pa": available(p.Pe),
    })
    geometry = SnapshotSection({
        "throat_radius_m": available(p.Rt),
        "throat_area_m2": available(p.At),
        "expansion_ratio": available(p.epsilon),
        "exit_radius_m": available(p.Re),
        "length_pct": available(c.length_pct),
        "contraction_ratio": available(c.contraction_ratio),
        "l_star_m": available(c.l_star),
        "throat_upstream_radius_ratio": available(
            float(g.Ru_input) / max(float(p.Rt), 1.0e-30)
        ),
        "throat_downstream_radius_ratio": available(
            float(g.Rd_applied) / max(float(p.Rt), 1.0e-30)
        ),
        "chamber_barrel_length_m": available(g.chamber_length),
        "chamber_volume_m3": available(g.chamber_volume),
        "chamber_volume_target_m3": available(g.chamber_volume_target),
        "wetted_area_m2": available(g.wetted_area),
        "axial_coordinate_m": available(gx),
        "radius_profile_m": _profile(gx, gr, gr, units="m"),
        "area_ratio_profile": _profile(
            gx, gr, g.area_ratio, units="-"
        ),
        "mach_profile": _profile(gx, gr, g.mach, units="-"),
    })
    thermal = SnapshotSection({
        "gas_side_wall_temperature_max_k": available(np.max(t.T_wg)),
        "coolant_side_wall_temperature_max_k": available(np.max(t.T_wc)),
        "heat_flux_max_w_m2": available(np.max(t.q_flux)),
        "gas_side_coefficient_max_w_m2_k": available(np.max(t.h_g)),
        "thermal_stress_max_pa": available(np.max(t.sigma_thermal)),
        "pressure_stress_pa": available(t.sigma_pressure),
        "combined_stress_max_pa": available(np.max(t.sigma_combined)),
        "thermal_stress_profile": _profile(
            gx, gr, t.sigma_thermal, units="Pa"
        ),
        "pressure_stress_profile": _profile(
            gx, gr, t.sigma_pressure_profile, units="Pa"
        ),
        "combined_stress_profile": _profile(
            gx, gr, t.sigma_combined, units="Pa"
        ),
        "gas_side_wall_temperature_profile": _profile(
            gx, gr, t.T_wg, units="K"
        ),
        "coolant_side_wall_temperature_profile": _profile(
            gx, gr, t.T_wc, units="K"
        ),
        "heat_flux_profile": _profile(gx, gr, t.q_flux, units="W/m^2"),
        "gas_side_coefficient_profile": _profile(
            gx, gr, t.h_g, units="W/(m^2 K)"
        ),
    })
    cooling = SnapshotSection({
        "method": available("regenerative"),
        "coolant_name": available(_mdo_coolant_name(mission)),
        "coolant_mass_flow_kg_s": available(p.mdot_regen_jacket),
        "film_mass_flow_kg_s": available(p.mdot_film),
        "film_fraction_of_fuel": available(
            float(p.mdot_film) / max(float(p.mdot_fuel_total), 1.0e-30)
        ),
        "fuel_flow_topology": available(
            "common_upstream_pump_then_regen_and_film_split"
            if float(p.mdot_film) > 0.0
            else "direct_regen_jacket_to_injector"
        ),
        "fuel_flow_closure_residual_kg_s": available(
            p.mdot_regen_jacket
            + p.mdot_film
            - p.mdot_fuel_total
        ),
        "coolant_inlet_temperature_k": available(
            c.coolant_temperature
        ),
        "coolant_outlet_temperature_k": available(t.T_coolant_exit),
        "coolant_pressure_drop_pa": available(t.dp_total),
        "coolant_velocity_m_s": available(t.coolant_velocity),
        "coolant_mach": available(t.coolant_mach),
        "land_width_min_m": available(t.land_min),
        "coolant_temperature_profile": _profile(
            gx, gr, t.T_coolant, units="K"
        ),
        "coolant_pressure_profile": _profile(
            gx, gr, t.coolant_pressure, units="Pa"
        ),
        "gas_pressure_profile": _profile(
            gx, gr, t.gas_pressure, units="Pa"
        ),
        "liner_pressure_differential_profile": _profile(
            gx, gr, t.liner_pressure_differential, units="Pa"
        ),
    })
    injector = SnapshotSection({
        "type": available("pintle"),
        "architecture": available("fixed_discrete"),
        "sizing": available("auto"),
        "fuel_dp_fraction": available(i.dp_fuel / p.Pc),
        "oxidizer_dp_fraction": available(i.dp_oxidizer / p.Pc),
        "fuel_dp_pa": available(i.dp_fuel),
        "oxidizer_dp_pa": available(i.dp_oxidizer),
        "fuel_cd": available(c.injector_cd_fuel),
        "oxidizer_cd": available(c.injector_cd_oxidizer),
        "pintle_diameter_m": available(d["D_pintle"]),
        "slot_count": available(int(c.pintle_slot_count)),
        "momentum_ratio": _film_injector_sensitive(
            available(i.momentum_ratio), float(p.mdot_film)
        ),
        "spray_half_angle_deg": _film_injector_sensitive(
            available(i.spray_half_angle_deg), float(p.mdot_film)
        ),
        "blockage_factor": _film_injector_sensitive(
            available(i.blockage_factor), float(p.mdot_film)
        ),
        "transition_margin_m2": _film_injector_sensitive(
            available(i.transition_margin), float(p.mdot_film)
        ),
        "fuel_velocity_m_s": _film_injector_sensitive(
            available(i.velocity_fuel), float(p.mdot_film)
        ),
        "oxidizer_velocity_m_s": available(i.velocity_oxidizer),
        "fuel_flow_area_m2": _film_injector_sensitive(
            available(i.area_fuel), float(p.mdot_film)
        ),
        "oxidizer_flow_area_m2": available(i.area_oxidizer),
        "slot_width_m": _film_injector_sensitive(
            available(i.slot_width), float(p.mdot_film)
        ),
        "tip_opening_m": _film_injector_sensitive(
            available(i.tip_opening), float(p.mdot_film)
        ),
        "tip_branch_area_m2": _film_injector_sensitive(
            available(i.area_tip_branch), float(p.mdot_film)
        ),
        "center_gap_area_m2": _film_injector_sensitive(
            available(i.area_center_gap), float(p.mdot_film)
        ),
        "branch_consistency": _film_injector_sensitive(
            available(i.branch_consistency), float(p.mdot_film)
        ),
        "fuel_chug_margin": available(i.chug_margin_fuel),
        "oxidizer_chug_margin": available(i.chug_margin_oxidizer),
    })
    feed_electrical = SnapshotSection({
        "architecture": available("electric_pump_fed"),
        "fuel_tank_pressure_pa": available(c.tank_pressure_fuel),
        "oxidizer_tank_pressure_pa": available(c.tank_pressure_oxidizer),
        "fuel_density_kg_m3": available(c.rho_fuel),
        "oxidizer_density_kg_m3": available(c.rho_oxidizer),
        "fuel_vapor_pressure_pa": available(c.vapor_pressure_fuel),
        "oxidizer_vapor_pressure_pa": available(c.vapor_pressure_oxidizer),
        "line_pressure_loss_pa": available(c.line_dp_allowance),
        "fuel_required_pressure_rise_pa": available(fp.pressure_rise),
        "oxidizer_required_pressure_rise_pa": available(op.pressure_rise),
        "pump_speed_rpm": available(fp.rpm),
        "fuel_volumetric_flow_m3_s": available(fp.volumetric_flow),
        "oxidizer_volumetric_flow_m3_s": available(op.volumetric_flow),
        "fuel_pump_head_m": available(fp.head),
        "oxidizer_pump_head_m": available(op.head),
        "fuel_specific_speed": available(fp.specific_speed),
        "oxidizer_specific_speed": available(op.specific_speed),
        "fuel_npsh_available_pa": available(
            fp.npsh_available * c.rho_fuel * 9.80665
        ),
        "oxidizer_npsh_available_pa": available(
            op.npsh_available * c.rho_oxidizer * 9.80665
        ),
        "fuel_suction_specific_speed": available(fp.suction_specific_speed),
        "oxidizer_suction_specific_speed": available(op.suction_specific_speed),
        "fuel_nss_margin": available(fp.nss_margin),
        "oxidizer_nss_margin": available(op.nss_margin),
        "fuel_tip_speed_m_s": available(fp.tip_speed),
        "oxidizer_tip_speed_m_s": available(op.tip_speed),
        "fuel_tip_speed_margin_m_s": available(fp.tip_speed_margin),
        "oxidizer_tip_speed_margin_m_s": available(op.tip_speed_margin),
        "fuel_pump_efficiency": available(fp.efficiency),
        "oxidizer_pump_efficiency": available(op.efficiency),
        "fuel_hydraulic_power_w": available(fp.hydraulic_power),
        "oxidizer_hydraulic_power_w": available(op.hydraulic_power),
        "fuel_shaft_power_w": available(fp.shaft_power),
        "oxidizer_shaft_power_w": available(op.shaft_power),
        "electric_power_total_w": available(e.electric_power_total),
        "motor_efficiency": available(c.motor_efficiency),
        "inverter_efficiency": available(c.inverter_efficiency),
    })

    mass_values = np.asarray(state.masses.values, dtype=float)
    mass_available = np.asarray(state.masses.availability, dtype=bool)
    mass_by_name: dict[str, SnapshotValue] = {}
    for idx, name in enumerate(MASS_FIELD_NAMES):
        if mass_available[idx]:
            mass_by_name[name] = available(mass_values[idx])
        else:
            mass_by_name[name] = unavailable(
                MASS_UNAVAILABLE_REASONS.get(
                    name, f"EngineState v2 marks {name} unavailable"
                )
            )
    masses = SnapshotSection({
        "pump_mass_kg": mass_by_name["pump_mass"],
        "motor_mass_kg": mass_by_name["motor_mass"],
        "inverter_mass_kg": mass_by_name["inverter_mass"],
        "battery_energy_limited_mass_kg": mass_by_name[
            "battery_energy_cell_mass"
        ],
        "battery_power_limited_mass_kg": mass_by_name[
            "battery_power_cell_mass"
        ],
        "battery_energy_installed_mass_kg": mass_by_name[
            "battery_energy_installed_mass"
        ],
        "battery_power_installed_mass_kg": mass_by_name[
            "battery_power_installed_mass"
        ],
        "battery_selected_mass_kg": mass_by_name[
            "battery_governing_installed_mass"
        ],
        "battery_objective_mass_kg": mass_by_name["battery_objective_mass"],
        "electric_package_mass_kg": mass_by_name[
            "electric_feed_package_exact_mass"
        ],
        "electric_package_objective_mass_kg": mass_by_name[
            "electric_feed_package_objective_mass"
        ],
        "dry_mass_partial_exact_mass_kg": mass_by_name[
            "dry_mass_partial_exact_mass"
        ],
        "dry_mass_partial_objective_mass_kg": mass_by_name[
            "dry_mass_partial_objective_mass"
        ],
        "thrust_chamber_liner_mass_kg": mass_by_name[
            "thrust_chamber_liner_mass"
        ],
        "thrust_chamber_land_mass_kg": mass_by_name[
            "thrust_chamber_land_mass"
        ],
        "thrust_chamber_closeout_mass_kg": mass_by_name[
            "thrust_chamber_closeout_mass"
        ],
        "thrust_chamber_mass_kg": mass_by_name["thrust_chamber_mass"],
        "injector_mass_kg": mass_by_name["injector_mass"],
        "total_engine_package_mass_kg": mass_by_name["total_dry_mass"],
        "engine_hardware_mass_ledger": unavailable(
            "the differentiable MDO integrates thrust-chamber metal on the "
            "station grid but does not resolve a per-part hardware ledger; "
            "that requires the host-side machined layouts priced by "
            "raosim.mass_ledger"
        ),
        "raw_mass_ledger": available({
            name: field_value.to_dict()
            for name, field_value in mass_by_name.items()
        }),
    })
    constraint_values = np.asarray(state.constraints.values, dtype=float)
    constraint_applicable = np.asarray(
        state.constraints.applicable, dtype=bool
    )
    constraint_available = np.asarray(
        state.constraints.available, dtype=bool
    )
    constraint_required = np.asarray(
        state.constraints.required, dtype=bool
    )
    constraint_reasons = np.asarray(
        state.constraints.reason_codes, dtype=np.int32
    )
    optimizer_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.optimizer_role == "hard"
    )
    numerical_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category == "numerical"
    )
    physics_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category not in {"numerical", "requirement"}
    )
    requirement_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category == "requirement"
    )
    status_args = (
        constraint_values,
        constraint_applicable,
        constraint_available,
        constraint_required,
    )
    all_status = status_from_rows(*status_args, nonfinite="unknown")
    optimizer_status = status_from_rows(
        *status_args, indices=optimizer_indices, nonfinite="unknown"
    )
    numerical_status = status_from_rows(
        *status_args, indices=numerical_indices
    )
    physics_status = status_from_rows(
        *status_args, indices=physics_indices, nonfinite="unknown"
    )
    requirement_row_status = status_from_rows(
        *status_args, indices=requirement_indices, nonfinite="unknown"
    )
    # The engine state contains only engine-side limit rows.  It does not carry
    # the complete ResolvedRequirement (notably the Isp epsilon contract), so a
    # non-failing subset cannot be promoted to a requirements pass.
    requirements_status = (
        "fail" if requirement_row_status == "fail" else "unknown"
    )
    margin_dict = _constraint_margin_payload(
        ENGINE_CONSTRAINT_SPECS,
        constraint_values,
        constraint_applicable,
        constraint_available,
        constraint_required,
        constraint_reasons,
        mission,
    )
    constraints_gates = SnapshotSection({
        "all_constraints_feasible": _tri_state_value(
            all_status,
            unknown_reason=(
                "one or more applicable required constraint models are "
                "unavailable"
            ),
        ),
        "optimizer_constraints_feasible": _tri_state_value(
            optimizer_status,
            unknown_reason=(
                "one or more applicable optimizer constraint models are "
                "unavailable"
            ),
        ),
        "numerical_validity": _tri_state_value(
            numerical_status,
            unknown_reason=(
                "one or more mandatory numerical convergence checks are "
                "unavailable"
            ),
        ),
        "physics_feasible": _tri_state_value(
            physics_status,
            unknown_reason=(
                "one or more applicable physical mechanisms lack validated "
                "model coverage"
            ),
        ),
        "requirements_feasible": _tri_state_value(
            requirements_status,
            unknown_reason=(
                "the EngineState does not retain the complete resolved "
                "requirement contract, including the Isp epsilon row"
            ),
        ),
        "workflow_readiness_feasible": unavailable(
            "the MDO state evaluates physics/numerical constraints, not "
            "authoritative workflow, CAD, release, or readiness gates"
        ),
        "constraint_margins": available(margin_dict),
        "diagnostics": available({
            "state_schema_version": int(state.schema_version),
            "all_residuals_converged": bool(state.residuals.all_converged),
            "finite": bool(state.residuals.finite),
            "all_constraints_status": all_status,
            "optimizer_constraints_status": optimizer_status,
            "numerical_validity_status": numerical_status,
            "physics_status": physics_status,
            "requirements_status": requirements_status,
        }),
        "outer_thrust_residual": available(state.residuals.outer[0]),
        "cooling_residual_max_abs_k": available(state.residuals.cooling_max),
        "authoritative_design_gates": unavailable(
            "authoritative design gates are evaluated only by design_nozzle_v2"
        ),
    })
    provenance = SnapshotSection({
        "analysis_source": available("differentiable_mdo_engine_state"),
        "contract_version": available(CONTRACT_VERSION),
        "propellant": available(str(mission.propellant_name)),
        "coolant": available(_mdo_coolant_name(mission)),
        "thermochemistry": available({
            "mode": (
                "mdo_cea_surface"
                if mission.cea_table_path
                else "mdo_constant_property"
            ),
            "surface_provenance": str(resolved_surfaces.provenance),
            "surface_fingerprint": [
                int(word)
                for word in np.asarray(
                    c.surface_signature, dtype=np.uint32
                ).tolist()
            ],
        }),
        "geometry_model": available("mdo_fixed_topology_rao_top_grid"),
        "material_assumptions": available(_material_assumptions(mission)),
        "input_conventions": available({
            **{
                name: _host_value(value)
                for name, value in c._asdict().items()
            },
            "eta_cstar_effective": float(p.eta_cstar),
        }),
        "mission": available(_json_ready(mission)),
        "design": available({k: float(v) for k, v in d.items()}),
    })
    artifact_values = dict(artifacts or {})
    artifact_section = SnapshotSection({
        "files": available(artifact_values) if artifact_values else unavailable(
            "the MDO numerical solve did not generate report or CAD artifacts"
        ),
        "report_sections": unavailable(
            "the pure numerical MDO state has no host report sections"
        ),
        "cad_files": unavailable(
            "the pure numerical MDO state does not generate CAD"
        ),
    })
    warnings_list = [
        "MDO electric package mass excludes unavailable thrust-chamber/nozzle "
        "and injector hardware mass branches.",
    ]
    if float(p.mdot_film) > 0.0:
        warnings_list.append(
            "Film-sensitive main-pintle outputs are unavailable because "
            "the separate film injector/orifice is not modeled."
        )
    warnings = tuple(warnings_list)
    return EngineAnalysisSnapshot(
        source="mdo",
        performance=performance,
        geometry=geometry,
        thermal=thermal,
        cooling=cooling,
        injector=injector,
        feed_electrical=feed_electrical,
        masses=masses,
        constraints_gates=constraints_gates,
        provenance=provenance,
        warnings=warnings,
        artifacts=artifact_section,
        optimizer_metadata=(
            dict(optimizer_metadata) if optimizer_metadata is not None else None
        ),
        source_result=state,
    )


def _snapshot_from_engine_result(
    result: Any,
    design: Any,
    mission: Any,
    *,
    optimizer_metadata: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> EngineAnalysisSnapshot:
    """Convert an MDO ``EngineResult`` to the versioned host contract."""

    from raosim.mdo.grid import (
        build_station_grid, chamber_barrel_length, chamber_volume,
    )
    from raosim.mdo.constraints import (
        ConstraintReasonCode,
        ENGINE_CONSTRAINT_SPECS,
        constraint_metadata,
        status_from_rows,
    )
    from raosim.mdo.engine import chamber_surfaces_for
    from raosim.mdo.schema import validate_mixture_ratio

    d = _design_dict(design)
    of = validate_mixture_ratio(
        getattr(result, "OF", None), name="legacy EngineResult.OF"
    )
    # Provenance is a fixed physical contract even when the incoming fixed-mode
    # mapping came from a legacy sentinel-bearing DesignVector.
    d["OF"] = of
    Pc = float(d["Pc"])
    eps = float(d["eps"])
    Rt = float(np.asarray(result.Rt))
    mdot = float(np.asarray(result.mdot))
    eta_cstar = float(np.asarray(result.eta_cstar))
    cf_del = float(np.asarray(getattr(result, "Cf_delivered", result.Cf)))
    cf_ideal = float(np.asarray(getattr(result, "Cf_ideal", result.Cf)))
    eta_cf = float(np.asarray(getattr(result, "eta_CF", cf_del / cf_ideal)))
    cstar_del = float(result.Isp) * float(mission.g0) / max(cf_del, 1.0e-30)
    cstar_ideal = cstar_del / max(eta_cstar, 1.0e-30)
    thrust = cf_del * Pc * np.pi * Rt * Rt
    mdot_f = mdot / (1.0 + of)
    mdot_o = mdot - mdot_f

    grid = build_station_grid(result.Rt, d["eps"], mission)
    gx = np.asarray(grid.x, dtype=float)
    gr = np.asarray(grid.r, dtype=float)
    gar = np.asarray(grid.area_ratio, dtype=float)
    gmach = np.asarray(grid.mach, dtype=float)

    perf = SnapshotSection({
        "chamber_pressure_pa": available(Pc),
        "ambient_pressure_pa": available(float(mission.Pa)),
        "thrust_n": available(thrust),
        "specific_impulse_delivered_s": available(result.Isp),
        "specific_impulse_ideal_s": available(
            cf_ideal * cstar_ideal / float(mission.g0)
        ),
        "mass_flow_total_kg_s": available(mdot),
        "mass_flow_fuel_kg_s": available(mdot_f),
        "mass_flow_oxidizer_kg_s": available(mdot_o),
        "mixture_ratio": available(of),
        "cf_ideal": available(cf_ideal),
        "cf_delivered": available(cf_del),
        "c_star_ideal_m_s": available(cstar_ideal),
        "c_star_delivered_m_s": available(cstar_del),
        "eta_cstar": available(eta_cstar),
        "eta_cf": available(eta_cf),
        "exit_mach": available(result.Me),
        "exit_pressure_pa": available(result.Pe),
    })
    geometry = SnapshotSection({
        "throat_radius_m": available(Rt),
        "throat_area_m2": available(np.pi * Rt * Rt),
        "expansion_ratio": available(eps),
        "exit_radius_m": available(Rt * np.sqrt(eps)),
        "length_pct": available(float(mission.length_pct)),
        "contraction_ratio": available(float(mission.contraction_ratio)),
        "l_star_m": available(float(mission.l_star)),
        "throat_upstream_radius_ratio": available(
            float(getattr(mission, "throat_ru_factor", 1.5))
        ),
        "throat_downstream_radius_ratio": available(
            float(getattr(mission, "throat_rd_factor", 0.382))
        ),
        "chamber_barrel_length_m": available(
            float(chamber_barrel_length(Rt, mission))
        ),
        "chamber_volume_m3": available(float(chamber_volume(Rt, mission))),
        "chamber_volume_target_m3": available(
            float(mission.l_star) * float(np.pi * Rt * Rt)
        ),
        "wetted_area_m2": available(_polyline_wetted_area(gx, gr)),
        "axial_coordinate_m": available(gx),
        "radius_profile_m": _profile(gx, gr, gr, units="m"),
        "area_ratio_profile": _profile(gx, gr, gar, units="-"),
        "mach_profile": _profile(gx, gr, gmach, units="-"),
    })

    cool = result.cooling
    pressure_profile = np.asarray(
        getattr(
            cool,
            "sigma_pressure_profile",
            np.full_like(
                np.asarray(cool.sigma_thermal, dtype=float),
                float(cool.sigma_pressure),
            ),
        ),
        dtype=float,
    )
    combined_profile = (
        np.asarray(cool.sigma_thermal, dtype=float)
        + np.abs(pressure_profile)
    )
    thermal = SnapshotSection({
        "gas_side_wall_temperature_max_k": available(np.max(result.T_wg)),
        "coolant_side_wall_temperature_max_k": available(np.max(cool.T_wc)),
        "heat_flux_max_w_m2": available(np.max(cool.q_flux)),
        "gas_side_coefficient_max_w_m2_k": available(np.max(cool.h_g)),
        "thermal_stress_max_pa": available(np.max(cool.sigma_thermal)),
        "pressure_stress_pa": available(cool.sigma_pressure),
        "combined_stress_max_pa": available(np.max(combined_profile)),
        "thermal_stress_profile": _profile(
            gx, gr, cool.sigma_thermal, units="Pa"
        ),
        "pressure_stress_profile": _profile(
            gx, gr, pressure_profile, units="Pa"
        ),
        "combined_stress_profile": _profile(
            gx, gr, combined_profile, units="Pa"
        ),
        "gas_side_wall_temperature_profile": _profile(
            gx, gr, result.T_wg, units="K"
        ),
        "coolant_side_wall_temperature_profile": _profile(
            gx, gr, cool.T_wc, units="K"
        ),
        "heat_flux_profile": _profile(gx, gr, cool.q_flux, units="W/m^2"),
        "gas_side_coefficient_profile": _profile(
            gx, gr, cool.h_g, units="W/(m^2 K)"
        ),
    })
    film_frac = float(d.get("film_frac", 0.0))
    jacket_flow = (
        float(mission.cooling_fraction) * mdot_f * (1.0 - film_frac)
    )
    film_flow = mdot_f * film_frac
    cooling = SnapshotSection({
        "method": available("regenerative"),
        "coolant_name": available(_mdo_coolant_name(mission)),
        "coolant_mass_flow_kg_s": available(jacket_flow),
        "film_mass_flow_kg_s": available(film_flow),
        "film_fraction_of_fuel": available(film_frac),
        "fuel_flow_topology": available(
            "common_upstream_pump_then_regen_and_film_split"
            if film_flow > 0.0
            else "direct_regen_jacket_to_injector"
        ),
        "fuel_flow_closure_residual_kg_s": available(
            jacket_flow + film_flow - mdot_f
        ),
        "coolant_inlet_temperature_k": available(
            float(mission.coolant_temperature)
        ),
        "coolant_outlet_temperature_k": available(cool.T_coolant_exit),
        "coolant_pressure_drop_pa": available(result.dp_regen),
        "coolant_velocity_m_s": available(cool.coolant_velocity),
        "coolant_mach": available(cool.coolant_mach),
        "land_width_min_m": available(cool.land_min),
        "coolant_temperature_profile": _profile(
            gx, gr, cool.T_coolant, units="K"
        ),
        "coolant_pressure_profile": (
            _profile(gx, gr, cool.coolant_pressure, units="Pa")
            if hasattr(cool, "coolant_pressure")
            else unavailable(
                "legacy EngineResult did not retain a stationwise "
                "coolant-pressure profile"
            )
        ),
        "gas_pressure_profile": (
            _profile(gx, gr, cool.gas_pressure, units="Pa")
            if hasattr(cool, "gas_pressure")
            else unavailable(
                "legacy EngineResult did not retain a stationwise gas-pressure "
                "profile"
            )
        ),
        "liner_pressure_differential_profile": (
            _profile(
                gx,
                gr,
                cool.liner_pressure_differential,
                units="Pa",
            )
            if hasattr(cool, "liner_pressure_differential")
            else unavailable(
                "legacy EngineResult did not retain the stationwise "
                "coolant-minus-gas pressure differential"
            )
        ),
    })
    inj = result.injector
    injector = SnapshotSection({
        "type": available("pintle"),
        "architecture": available("fixed_discrete"),
        "sizing": available("auto"),
        "fuel_dp_fraction": available(float(d["dp_f_frac"])),
        "oxidizer_dp_fraction": available(float(d["dp_o_frac"])),
        "fuel_dp_pa": available(inj.dp_fuel),
        "oxidizer_dp_pa": available(inj.dp_ox),
        "fuel_cd": available(float(mission.injector_cd_fuel)),
        "oxidizer_cd": available(float(mission.injector_cd_ox)),
        "pintle_diameter_m": available(float(d["D_pintle"])),
        "slot_count": available(int(mission.pintle_slot_count)),
        "momentum_ratio": _film_injector_sensitive(
            available(inj.momentum_ratio), film_flow
        ),
        "spray_half_angle_deg": _film_injector_sensitive(
            available(inj.spray_half_angle_deg), film_flow
        ),
        "blockage_factor": _film_injector_sensitive(
            available(inj.blockage_factor), film_flow
        ),
        "transition_margin_m2": _film_injector_sensitive(
            available(inj.transition_margin), film_flow
        ),
        "fuel_velocity_m_s": _film_injector_sensitive(
            available(inj.v_fuel), film_flow
        ),
        "oxidizer_velocity_m_s": available(inj.v_ox),
        "fuel_flow_area_m2": _film_injector_sensitive(
            available(inj.area_fuel), film_flow
        ),
        "oxidizer_flow_area_m2": available(inj.area_ox),
        "slot_width_m": _film_injector_sensitive(
            available(inj.slot_width), film_flow
        ),
        "tip_opening_m": _film_injector_sensitive(
            available(inj.tip_opening), film_flow
        ),
        "tip_branch_area_m2": _film_injector_sensitive(
            available(inj.area_tip_branch), film_flow
        ),
        "center_gap_area_m2": _film_injector_sensitive(
            available(inj.area_center_gap), film_flow
        ),
        "branch_consistency": _film_injector_sensitive(
            available(inj.branch_consistency), film_flow
        ),
        "fuel_chug_margin": available(inj.chug_margin_fuel),
        "oxidizer_chug_margin": available(inj.chug_margin_ox),
    })
    feed = result.feed
    feed_electrical = SnapshotSection({
        "architecture": available("electric_pump_fed"),
        "fuel_tank_pressure_pa": available(float(mission.P_tank_fuel)),
        "oxidizer_tank_pressure_pa": available(float(mission.P_tank_ox)),
        "fuel_density_kg_m3": available(float(mission.rho_fuel)),
        "oxidizer_density_kg_m3": available(float(mission.rho_ox)),
        "fuel_vapor_pressure_pa": available(float(mission.p_vapor_fuel)),
        "oxidizer_vapor_pressure_pa": available(float(mission.p_vapor_ox)),
        "line_pressure_loss_pa": available(float(mission.line_dp_allowance)),
        "fuel_required_pressure_rise_pa": available(result.dp_rise_fuel),
        "oxidizer_required_pressure_rise_pa": available(result.dp_rise_ox),
        "pump_speed_rpm": available(float(d["N_rpm"])),
        "fuel_volumetric_flow_m3_s": available(feed.fuel.Q),
        "oxidizer_volumetric_flow_m3_s": available(feed.ox.Q),
        "fuel_pump_head_m": available(feed.fuel.head),
        "oxidizer_pump_head_m": available(feed.ox.head),
        "fuel_specific_speed": available(feed.fuel.specific_speed),
        "oxidizer_specific_speed": available(feed.ox.specific_speed),
        "fuel_npsh_available_pa": available(
            feed.fuel.npsh_available * float(mission.rho_fuel) * 9.80665
        ),
        "oxidizer_npsh_available_pa": available(
            feed.ox.npsh_available * float(mission.rho_ox) * 9.80665
        ),
        "fuel_suction_specific_speed": available(
            feed.fuel.suction_specific_speed
        ),
        "oxidizer_suction_specific_speed": available(
            feed.ox.suction_specific_speed
        ),
        "fuel_nss_margin": available(feed.fuel.nss_margin),
        "oxidizer_nss_margin": available(feed.ox.nss_margin),
        "fuel_tip_speed_m_s": available(feed.fuel.tip_speed),
        "oxidizer_tip_speed_m_s": available(feed.ox.tip_speed),
        "fuel_tip_speed_margin_m_s": available(feed.fuel.tip_speed_margin),
        "oxidizer_tip_speed_margin_m_s": available(feed.ox.tip_speed_margin),
        "fuel_pump_efficiency": available(feed.fuel.efficiency),
        "oxidizer_pump_efficiency": available(feed.ox.efficiency),
        "fuel_hydraulic_power_w": available(feed.fuel.P_hydraulic),
        "oxidizer_hydraulic_power_w": available(feed.ox.P_hydraulic),
        "fuel_shaft_power_w": available(feed.fuel.P_shaft),
        "oxidizer_shaft_power_w": available(feed.ox.P_shaft),
        "electric_power_total_w": available(feed.P_electric_total),
        "motor_efficiency": available(float(mission.eta_motor)),
        "inverter_efficiency": available(float(mission.eta_inverter)),
    })

    raw_ledger = dict(getattr(result, "mass_ledger", {}) or {})
    battery_selected = (
        max(
            float(feed.battery.energy_limited_mass),
            float(feed.battery.power_limited_mass),
        )
        * float(mission.battery_structural_margin)
    )
    electric_package_exact = (
        float(feed.pump_mass)
        + float(feed.motor_mass)
        + float(feed.inverter_mass)
        + battery_selected
    )
    electric_package_objective = getattr(
        result,
        "electric_package_objective_mass",
        getattr(result, "objective_mass", result.package_mass),
    )
    chamber_mass_value = raw_ledger.get("thrust_chamber")
    dry_mass_partial_exact = getattr(
        result,
        "dry_mass_partial_exact_mass",
        (
            electric_package_exact + float(chamber_mass_value)
            if chamber_mass_value is not None
            else None
        ),
    )
    dry_mass_partial_objective = getattr(
        result,
        "dry_mass_partial_objective_mass",
        (
            float(electric_package_objective) + float(chamber_mass_value)
            if chamber_mass_value is not None
            else None
        ),
    )
    masses = SnapshotSection({
        "pump_mass_kg": available(feed.pump_mass),
        "motor_mass_kg": available(feed.motor_mass),
        "inverter_mass_kg": available(feed.inverter_mass),
        "battery_energy_limited_mass_kg": available(
            feed.battery.energy_limited_mass
        ),
        "battery_power_limited_mass_kg": available(
            feed.battery.power_limited_mass
        ),
        "battery_energy_installed_mass_kg": available(
            float(feed.battery.energy_limited_mass)
            * float(mission.battery_structural_margin)
        ),
        "battery_power_installed_mass_kg": available(
            float(feed.battery.power_limited_mass)
            * float(mission.battery_structural_margin)
        ),
        "battery_selected_mass_kg": available(battery_selected),
        "battery_objective_mass_kg": maybe(
            raw_ledger.get(
                "battery_objective_smooth", raw_ledger.get("battery")
            ),
            "legacy EngineResult did not expose the battery objective branch",
        ),
        "electric_package_mass_kg": available(electric_package_exact),
        "electric_package_objective_mass_kg": available(
            electric_package_objective
        ),
        "dry_mass_partial_exact_mass_kg": maybe(
            dry_mass_partial_exact,
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "dry_mass_partial_objective_mass_kg": maybe(
            dry_mass_partial_objective,
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "thrust_chamber_liner_mass_kg": maybe(
            raw_ledger.get("thrust_chamber_liner"),
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "thrust_chamber_land_mass_kg": maybe(
            raw_ledger.get("thrust_chamber_lands"),
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "thrust_chamber_closeout_mass_kg": maybe(
            raw_ledger.get("thrust_chamber_closeout"),
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "thrust_chamber_mass_kg": maybe(
            raw_ledger.get("thrust_chamber"),
            _LEGACY_NO_CHAMBER_MASS,
        ),
        "injector_mass_kg": unavailable(
            "the MDO sizes injector flow areas but not injector hardware "
            "mass; the injector ledger needs the host-side machined layout "
            "priced by raosim.mass_ledger.injector_mass_ledger"
        ),
        "total_engine_package_mass_kg": unavailable(
            "a total engine mass cannot be formed until injector hardware, "
            "the bolted interface and the propellant-side plumbing are in the "
            "ledger; the thrust-chamber structure alone is not a dry mass"
        ),
        "engine_hardware_mass_ledger": unavailable(
            "the differentiable MDO integrates thrust-chamber metal on the "
            "station grid but does not resolve a per-part hardware ledger"
        ),
        "raw_mass_ledger": available({
            key: (
                unavailable(
                    f"{key} is an explicit MDO placeholder, not a physical mass"
                ).to_dict()
                if "placeholder" in key
                else available(value).to_dict()
            )
            for key, value in raw_ledger.items()
        }),
    })
    margins = {
        key: _host_value(value)
        for key, value in dict(getattr(result, "constraints", {})).items()
    }
    diagnostics = {
        key: _host_value(value)
        for key, value in dict(getattr(result, "diagnostics", {})).items()
    }
    resolved_surfaces = chamber_surfaces_for(mission)
    (
        constraint_applicable,
        constraint_available,
        constraint_required,
        constraint_reasons,
    ) = constraint_metadata(mission, resolved_surfaces)
    constraint_values = np.asarray([
        float(margins.get(str(spec.engine_key), np.nan))
        for spec in ENGINE_CONSTRAINT_SPECS
    ], dtype=float)
    missing_rows = np.asarray([
        str(spec.engine_key) not in margins for spec in ENGINE_CONSTRAINT_SPECS
    ], dtype=bool)
    # Compatibility results can predate rows in the current manifest.  A
    # missing applicable row is unknown model coverage, never an implicit pass.
    missing_applicable = missing_rows & constraint_applicable
    constraint_available = constraint_available & ~missing_applicable
    constraint_reasons = np.where(
        missing_applicable,
        int(ConstraintReasonCode.MODEL_UNAVAILABLE),
        constraint_reasons,
    ).astype(np.int32)
    optimizer_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.optimizer_role == "hard"
    )
    numerical_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category == "numerical"
    )
    physics_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category not in {"numerical", "requirement"}
    )
    requirement_indices = tuple(
        index for index, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category == "requirement"
    )
    status_args = (
        constraint_values,
        constraint_applicable,
        constraint_available,
        constraint_required,
    )
    all_status = status_from_rows(*status_args, nonfinite="unknown")
    optimizer_status = status_from_rows(
        *status_args, indices=optimizer_indices, nonfinite="unknown"
    )
    numerical_status = status_from_rows(
        *status_args, indices=numerical_indices
    )
    physics_status = status_from_rows(
        *status_args, indices=physics_indices, nonfinite="unknown"
    )
    requirement_row_status = status_from_rows(
        *status_args, indices=requirement_indices, nonfinite="unknown"
    )
    requirements_status = (
        "fail" if requirement_row_status == "fail" else "unknown"
    )
    margin_payload = _constraint_margin_payload(
        ENGINE_CONSTRAINT_SPECS,
        constraint_values,
        constraint_applicable,
        constraint_available,
        constraint_required,
        constraint_reasons,
        mission,
    )
    gates = SnapshotSection({
        "all_constraints_feasible": _tri_state_value(
            all_status,
            unknown_reason=(
                "one or more applicable required constraint models are "
                "unavailable"
            ),
        ),
        "optimizer_constraints_feasible": _tri_state_value(
            optimizer_status,
            unknown_reason=(
                "one or more applicable optimizer constraint models are "
                "unavailable"
            ),
        ),
        "numerical_validity": _tri_state_value(
            numerical_status,
            unknown_reason=(
                "one or more mandatory numerical convergence checks are "
                "unavailable"
            ),
        ),
        "physics_feasible": _tri_state_value(
            physics_status,
            unknown_reason=(
                "one or more applicable physical mechanisms lack validated "
                "model coverage"
            ),
        ),
        "requirements_feasible": _tri_state_value(
            requirements_status,
            unknown_reason=(
                "legacy EngineResult does not retain the complete resolved "
                "requirement contract, including the Isp epsilon row"
            ),
        ),
        "workflow_readiness_feasible": unavailable(
            "the MDO result evaluates physics/numerical constraints, not "
            "authoritative workflow, CAD, release, or readiness gates"
        ),
        "constraint_margins": available(margin_payload),
        "diagnostics": available({
            **diagnostics,
            "solver_status_ok": bool(
                getattr(result, "solver_status_ok", False)
            ),
            "solver_converged": bool(
                getattr(result, "solver_converged", False)
            ),
            "finite": bool(getattr(result, "finite", False)),
            "all_constraints_status": all_status,
            "optimizer_constraints_status": optimizer_status,
            "numerical_validity_status": numerical_status,
            "physics_status": physics_status,
            "requirements_status": requirements_status,
        }),
        "outer_thrust_residual": available(result.thrust_residual),
        "cooling_residual_max_abs_k": available(
            np.max(np.abs(np.asarray(cool.residual, dtype=float)))
        ),
        "authoritative_design_gates": unavailable(
            "authoritative design gates are evaluated only by design_nozzle_v2"
        ),
    })
    provenance = SnapshotSection({
        "analysis_source": available("differentiable_mdo_screen"),
        "contract_version": available(CONTRACT_VERSION),
        "propellant": available(str(mission.propellant_name)),
        "coolant": available(_mdo_coolant_name(mission)),
        "thermochemistry": available(
            "mdo_cea_surface" if mission.cea_table_path else "mdo_constant_property"
        ),
        "geometry_model": available("mdo_fixed_topology_rao_top_grid"),
        "material_assumptions": available(_material_assumptions(mission)),
        "input_conventions": available(
            _mission_input_conventions(
                mission, d, effective_eta_cstar=eta_cstar
            )
        ),
        "mission": available(_json_ready(mission)),
        "design": available({k: _host_value(v) for k, v in d.items()}),
    })
    artifact_values = dict(artifacts or {})
    artifact_section = SnapshotSection({
        "files": available(artifact_values) if artifact_values else unavailable(
            "the MDO numerical solve did not generate report or CAD artifacts"
        ),
        "report_sections": unavailable(
            "the legacy MDO EngineResult has no host report sections"
        ),
        "cad_files": unavailable(
            "the MDO numerical solve does not generate CAD"
        ),
    })
    warnings_list = [
        "MDO electric package mass excludes thrust-chamber/nozzle and injector mass.",
    ]
    if film_flow > 0.0:
        warnings_list.append(
            "Film-sensitive main-pintle outputs are unavailable because "
            "the separate film injector/orifice is not modeled."
        )
    warnings = tuple(warnings_list)
    return EngineAnalysisSnapshot(
        source="mdo",
        performance=perf,
        geometry=geometry,
        thermal=thermal,
        cooling=cooling,
        injector=injector,
        feed_electrical=feed_electrical,
        masses=masses,
        constraints_gates=gates,
        provenance=provenance,
        warnings=warnings,
        artifacts=artifact_section,
        optimizer_metadata=(
            dict(optimizer_metadata) if optimizer_metadata is not None else None
        ),
        source_result=result,
    )


def _mdo_coolant_name(mission: Any) -> str:
    try:
        from raosim.mdo.propellants import get_propellant

        return str(get_propellant(mission.propellant_name).coolant_name)
    except Exception:
        pair = str(getattr(mission, "propellant_name", ""))
        return pair.split("/", 1)[1] if "/" in pair else "unknown"


def _material_assumptions(mission: Any) -> dict[str, Any]:
    """Report the traced wall constants and whether an alloy actually backs them.

    ``liner_material_name is None`` means the class-default constants are in
    force and no catalog record was applied.  Those defaults are not any one
    alloy, so the report must not name one: it says ``unattributed`` and marks
    the selection unresolved rather than implying a material was chosen.
    """

    liner = getattr(mission, "liner_material_name", None)
    closeout = getattr(mission, "closeout_material_name", None)
    return {
        "name": str(liner) if liner else "unattributed_class_default",
        "liner_selection_resolved": liner is not None,
        "conductivity_w_m_k": float(mission.k_wall),
        "density_kg_m3": float(mission.rho_wall),
        "elastic_modulus_pa": float(mission.liner_E),
        "thermal_expansion_1_k": float(mission.liner_alpha),
        "poisson_ratio": float(mission.liner_poisson),
        "allowable_stress_pa": float(mission.liner_sigma_allow),
        "structural_fos": float(mission.liner_structural_fos),
        "yield_strength_pa": (
            float(mission.liner_sigma_allow)
            * float(mission.liner_structural_fos)
        ),
        "max_gas_side_wall_temperature_k": float(mission.liner_T_wg_max),
        "coolant_side_wall_limit_k": float(mission.rp1_coking_wall_temp_K),
        # SP-8087 sec. 2.1.3.1: the jacket is a separate, usually hardenable
        # alloy, so it carries its own selection and its own provenance row.
        "closeout": {
            "name": str(closeout) if closeout else "unattributed_class_default",
            "selection_resolved": closeout is not None,
            "density_kg_m3": float(mission.rho_closeout)
            if mission.rho_closeout is not None else float(mission.rho_wall),
            "yield_strength_pa": float(mission.closeout_sigma_yield),
            "structural_fos": float(mission.closeout_structural_fos),
            "elastic_modulus_pa": float(mission.closeout_E),
            "poisson_ratio": float(mission.closeout_poisson),
        },
    }


def _mission_input_conventions(
    mission: Any,
    design: Mapping[str, Any],
    *,
    effective_eta_cstar: float | None = None,
) -> dict[str, Any]:
    """Host copy of the numerical assumptions carried by ``EngineState``."""

    return {
        "propellant": str(mission.propellant_name),
        "ambient_pressure_pa": float(mission.Pa),
        "burn_time_s": float(mission.burn_time),
        "mixture_ratio": float(design.get("OF", mission.OF)),
        "mission_mixture_ratio": float(mission.OF),
        "eta_cstar_nominal": float(mission.eta_cstar),
        "eta_cstar_effective": (
            float(effective_eta_cstar)
            if effective_eta_cstar is not None
            else unavailable(
                "the legacy adapter was not given a solved effective eta_cstar"
            ).to_dict()
        ),
        "eta_cf": float(mission.eta_CF),
        "throat_ru_over_rt": float(mission.throat_ru_factor),
        "throat_rd_over_rt": float(mission.throat_rd_factor),
        "cooling_fraction": float(mission.cooling_fraction),
        "fuel_film_fraction": float(design.get("film_frac", 0.0)),
        "fuel_density_kg_m3": float(mission.rho_fuel),
        "oxidizer_density_kg_m3": float(mission.rho_ox),
        "fuel_vapor_pressure_pa": float(mission.p_vapor_fuel),
        "oxidizer_vapor_pressure_pa": float(mission.p_vapor_ox),
        "fuel_tank_pressure_pa": float(mission.P_tank_fuel),
        "oxidizer_tank_pressure_pa": float(mission.P_tank_ox),
        "line_pressure_loss_pa": float(mission.line_dp_allowance),
        "fuel_injector_dp_fraction": float(design["dp_f_frac"]),
        "oxidizer_injector_dp_fraction": float(design["dp_o_frac"]),
        "fuel_injector_cd": float(mission.injector_cd_fuel),
        "oxidizer_injector_cd": float(mission.injector_cd_ox),
        "pump_speed_rpm": float(design["N_rpm"]),
        "material": _material_assumptions(mission),
    }


def _traditional_input_conventions(
    result: Any, input_obj: Any, electric_pump_result: Any | None
) -> dict[str, Any]:
    """Extract the actually resolved authoritative input convention."""

    inj = _traditional_injector(result)
    feed = inj.get("feed", {}) if isinstance(inj, Mapping) else {}
    feed_system = (
        inj.get("feed_system", {}) if isinstance(inj, Mapping) else {}
    )
    split = (
        feed_system.get("fuel_flow_split")
        if isinstance(feed_system, Mapping)
        else None
    )
    jacket_mdot = (
        float(split["regen_jacket_mass_flow_kg_s"])
        if isinstance(split, Mapping)
        and split.get("regen_jacket_mass_flow_kg_s") is not None
        else (
            float(input_obj.cooling.coolant_mass_flow)
            if input_obj.cooling.coolant_mass_flow is not None
            else None
        )
    )
    film_mdot = (
        float(split["film_bypass_mass_flow_kg_s"])
        if isinstance(split, Mapping)
        and split.get("film_bypass_mass_flow_kg_s") is not None
        else float(input_obj.cooling.fuel_film_mass_flow or 0.0)
    )
    total_fuel_mdot = (
        float(split["total_fuel_mass_flow_kg_s"])
        if isinstance(split, Mapping)
        and split.get("total_fuel_mass_flow_kg_s") is not None
        else (
            jacket_mdot + film_mdot
            if jacket_mdot is not None
            else None
        )
    )
    nonfilm_fuel_mdot = (
        total_fuel_mdot - film_mdot
        if total_fuel_mdot is not None
        else None
    )
    return {
        "propellant": str(result.propellant.name),
        "ambient_pressure_pa": float(result.performance.Pa),
        "burn_time_s": _pump_assumption(electric_pump_result, "burn_time_s"),
        "mixture_ratio": float(
            input_obj.thermo.mixture_ratio
            if input_obj.thermo.mixture_ratio is not None
            else result.propellant.OF
        ),
        "eta_cstar_nominal": unavailable(
            "traditional DesignInput retains the effective eta_cstar used by "
            "the solve, but not the pre-film/pre-coupling nominal value"
        ).to_dict(),
        "eta_cstar_effective": float(result.performance.eta_cstar),
        "eta_cf": float(result.performance.eta_CF),
        "throat_ru_over_rt": float(
            input_obj.throat_geometry.upstream_radius_ratio
        ),
        "throat_rd_over_rt": float(
            input_obj.throat_geometry.downstream_radius_ratio
        ),
        "cooling_fraction": (
            jacket_mdot / max(nonfilm_fuel_mdot, 1.0e-30)
            if jacket_mdot is not None and nonfilm_fuel_mdot is not None
            else None
        ),
        "fuel_film_fraction": (
            film_mdot / max(total_fuel_mdot, 1.0e-30)
            if total_fuel_mdot is not None
            else None
        ),
        "fuel_film_mass_flow_kg_s": film_mdot,
        "fuel_feed_properties": _host_value(
            feed.get("fuel") if isinstance(feed, Mapping) else None
        ),
        "oxidizer_feed_properties": _host_value(
            feed.get("oxidizer") if isinstance(feed, Mapping) else None
        ),
        "fuel_tank_pressure_pa": (
            input_obj.injector.feed_system.fuel.tank_pressure
        ),
        "oxidizer_tank_pressure_pa": (
            input_obj.injector.feed_system.oxidizer.tank_pressure
        ),
        "fuel_injector_dp_fraction": float(
            input_obj.injector.fuel_dp_fraction
        ),
        "oxidizer_injector_dp_fraction": float(
            input_obj.injector.oxidizer_dp_fraction
        ),
        "fuel_injector_cd": float(input_obj.injector.fuel_cd),
        "oxidizer_injector_cd": float(input_obj.injector.oxidizer_cd),
        "pump_speed_rpm": _pump_assumption(
            electric_pump_result, "pump_rpm"
        ),
        "material": _json_ready(input_obj.material),
    }


def _traditional_injector(result: Any) -> Mapping[str, Any]:
    section = result.report_sections.get("injector", {})
    return section if isinstance(section, Mapping) else {}


def _field_from_mapping(
    mapping: Mapping[str, Any],
    key: str,
    reason: str,
) -> SnapshotValue:
    return maybe(mapping.get(key), reason)


def snapshot_from_traditional(
    result: Any,
    electric_pump_result: Any | None = None,
    *,
    optimizer_metadata: Mapping[str, Any] | None = None,
) -> EngineAnalysisSnapshot:
    """Convert a ``ValidatedDesignResult`` to the versioned host contract."""

    perf_obj = result.performance
    contour = result.contour
    x = np.asarray(contour.get("x"), dtype=float)
    radius = np.asarray(contour.get("y"), dtype=float)
    Rt = float(perf_obj.Rt)
    area_ratio = (radius / Rt) ** 2
    thermal_raw = result.report_sections.get("thermal", {}) or {}
    cooling_raw = result.report_sections.get("cooling", {}) or {}
    structural_raw = result.report_sections.get("structural", {}) or {}
    hardware_mass = result.report_sections.get("hardware_mass", {}) or {}
    inj_raw = _traditional_injector(result)
    input_obj = result.input
    mixture_ratio = input_obj.thermo.mixture_ratio
    if mixture_ratio is None:
        mixture_ratio = getattr(result.propellant, "OF", None)
    mdot_f = (
        perf_obj.m_dot / (1.0 + float(mixture_ratio))
        if mixture_ratio is not None and float(mixture_ratio) > 0.0
        else None
    )
    mdot_o = perf_obj.m_dot - mdot_f if mdot_f is not None else None
    film_mass_flow = float(
        getattr(input_obj.cooling, "fuel_film_mass_flow", 0.0) or 0.0
    )
    film_fraction = (
        film_mass_flow / max(float(mdot_f), 1.0e-30)
        if mdot_f is not None
        else None
    )
    film_thermal_reason = (
        "the traditional pipeline records the separate fuel-film branch but "
        "does not apply a wall-film heat-load/effectiveness model; its "
        "regenerative-only thermal result is not comparable to the MDO film "
        "solution"
    )

    def film_sensitive(value: SnapshotValue) -> SnapshotValue:
        return (
            unavailable(film_thermal_reason)
            if film_mass_flow > 0.0
            else value
        )

    pressure_stress_profile = structural_raw.get("pressure_stress_profile")
    pressure_stress_max_abs = (
        float(np.nanmax(np.abs(np.asarray(
            pressure_stress_profile, dtype=float
        ))))
        if pressure_stress_profile is not None
        and np.asarray(pressure_stress_profile).size
        else (
            abs(float(structural_raw["pressure_stress"]))
            if structural_raw.get("pressure_stress") is not None
            else None
        )
    )
    thermal_stress_profile = structural_raw.get("thermal_stress_profile")
    thermal_stress_max = (
        float(np.nanmax(np.asarray(thermal_stress_profile, dtype=float)))
        if thermal_stress_profile is not None
        and np.asarray(thermal_stress_profile).size
        else structural_raw.get("thermal_stress")
    )

    performance = SnapshotSection({
        "chamber_pressure_pa": available(perf_obj.Pc),
        "ambient_pressure_pa": available(perf_obj.Pa),
        "thrust_n": available(perf_obj.thrust),
        "specific_impulse_delivered_s": available(perf_obj.Isp),
        "specific_impulse_ideal_s": available(
            perf_obj.Cf_ideal * perf_obj.c_star / 9.80665
        ),
        "mass_flow_total_kg_s": available(perf_obj.m_dot),
        "mass_flow_fuel_kg_s": maybe(
            mdot_f, "mixture ratio is unavailable in the traditional result"
        ),
        "mass_flow_oxidizer_kg_s": maybe(
            mdot_o, "mixture ratio is unavailable in the traditional result"
        ),
        "mixture_ratio": maybe(
            mixture_ratio, "mixture ratio is unavailable in the traditional result"
        ),
        "cf_ideal": available(perf_obj.Cf_ideal),
        "cf_delivered": available(perf_obj.Cf_actual),
        "c_star_ideal_m_s": available(perf_obj.c_star),
        "c_star_delivered_m_s": available(perf_obj.c_star_effective),
        "eta_cstar": available(perf_obj.eta_cstar),
        "eta_cf": available(perf_obj.eta_CF),
        "exit_mach": available(perf_obj.Me),
        "exit_pressure_pa": available(perf_obj.Pe),
    })
    throat_geometry = contour.get("throat_geometry", {}) or {}
    geometry = SnapshotSection({
        "throat_radius_m": available(perf_obj.Rt),
        "throat_area_m2": available(perf_obj.At),
        "expansion_ratio": available(perf_obj.epsilon),
        "exit_radius_m": maybe(
            contour.get("Re"),
            "authoritative contour did not expose an exit radius",
        ),
        "length_pct": maybe(
            contour.get("length_pct"),
            "authoritative contour did not expose the Rao length percentage",
        ),
        "contraction_ratio": maybe(
            contour.get("contraction_ratio"),
            "traditional contour did not include a chamber contraction ratio",
        ),
        "l_star_m": maybe(
            contour.get("L_star"),
            "traditional contour did not include characteristic length",
        ),
        "throat_upstream_radius_ratio": maybe(
            throat_geometry.get("upstream_radius_ratio"),
            "traditional contour did not expose the upstream throat-radius ratio",
        ),
        "throat_downstream_radius_ratio": maybe(
            throat_geometry.get("downstream_radius_ratio"),
            "traditional contour did not expose the downstream throat-radius ratio",
        ),
        "chamber_barrel_length_m": maybe(
            (contour.get("chamber") or {}).get("Lc"),
            "traditional contour did not expose the solved barrel length",
        ),
        "chamber_volume_m3": maybe(
            contour.get("V_chamber"),
            "traditional contour did not expose the measured chamber volume",
        ),
        "chamber_volume_target_m3": maybe(
            contour.get("V_target"),
            "traditional contour did not expose the target L*.A_t volume",
        ),
        "wetted_area_m2": available(
            float(_polyline_wetted_area(x, radius))
        ),
        "axial_coordinate_m": available(x),
        "radius_profile_m": _profile(x, radius, radius, units="m"),
        "area_ratio_profile": _profile(x, radius, area_ratio, units="-"),
        "mach_profile": _profile(
            x,
            radius,
            thermal_raw.get("mach"),
            units="-",
        ) if thermal_raw.get("mach") is not None else unavailable(
            "traditional thermal report did not retain a Mach profile"
        ),
    })
    thermal = SnapshotSection({
        "gas_side_wall_temperature_max_k": film_sensitive(maybe(
            cooling_raw.get("peak_gas_side_wall_temperature"),
            "regenerative gas-side wall temperatures were not evaluated",
        )),
        "coolant_side_wall_temperature_max_k": film_sensitive(maybe(
            _array_max(cooling_raw.get("coolant_side_wall_temperature")),
            "regenerative coolant-side wall temperatures were not evaluated",
        )),
        "heat_flux_max_w_m2": film_sensitive(maybe(
            thermal_raw.get("q_max"),
            "the traditional thermal report did not expose peak heat flux",
        )),
        "gas_side_coefficient_max_w_m2_k": film_sensitive(maybe(
            _array_max(thermal_raw.get("h_g")),
            "the traditional thermal report did not expose gas-side coefficient",
        )),
        "thermal_stress_max_pa": film_sensitive(maybe(
            thermal_stress_max,
            "the selected traditional structural model did not expose thermal stress",
        )),
        "pressure_stress_pa": maybe(
            pressure_stress_max_abs,
            "the selected traditional structural model did not expose pressure stress",
        ),
        "combined_stress_max_pa": film_sensitive(maybe(
            structural_raw.get("combined_stress"),
            "the selected traditional structural model did not expose combined stress",
        )),
        "thermal_stress_profile": film_sensitive(_traditional_profile(
            structural_raw,
            x,
            radius,
            "thermal_stress_profile",
            "Pa",
        )),
        "pressure_stress_profile": _traditional_profile(
            structural_raw,
            x,
            radius,
            "pressure_stress_profile",
            "Pa",
        ),
        "combined_stress_profile": film_sensitive(_traditional_profile(
            structural_raw,
            x,
            radius,
            "combined_stress_profile",
            "Pa",
        )),
        "gas_side_wall_temperature_profile": film_sensitive(_traditional_profile(
            cooling_raw,
            x,
            radius,
            "gas_side_wall_temperature",
            "K",
        )),
        "coolant_side_wall_temperature_profile": film_sensitive(_traditional_profile(
            cooling_raw,
            x,
            radius,
            "coolant_side_wall_temperature",
            "K",
        )),
        "heat_flux_profile": film_sensitive(_traditional_profile(
            cooling_raw if cooling_raw.get("q") is not None else thermal_raw,
            x,
            radius,
            "q",
            "W/m^2",
        )),
        "gas_side_coefficient_profile": film_sensitive(_traditional_profile(
            thermal_raw,
            x,
            radius,
            "h_g",
            "W/(m^2 K)",
        )),
    })
    feed_for_split = inj_raw.get("feed_system") or {}
    split_ledger = (
        feed_for_split.get("fuel_flow_split")
        if isinstance(feed_for_split, Mapping)
        else None
    )
    cooling = SnapshotSection({
        "method": available(str(input_obj.cooling.method)),
        "coolant_name": maybe(
            input_obj.cooling.coolant,
            "no coolant was requested in the traditional design input",
        ),
        "coolant_mass_flow_kg_s": maybe(
            input_obj.cooling.coolant_mass_flow,
            "traditional cooling flow was not supplied or resolved",
        ),
        "film_mass_flow_kg_s": available(film_mass_flow),
        "film_fraction_of_fuel": maybe(
            film_fraction,
            "cycle fuel mass flow was unavailable, so the film fraction "
            "could not be normalized",
        ),
        "fuel_flow_topology": maybe(
            (
                split_ledger.get("topology")
                if isinstance(split_ledger, Mapping)
                else (
                    "direct_regen_jacket_to_injector"
                    if input_obj.cooling.method == "regenerative"
                    and film_mass_flow == 0.0
                    else None
                )
            ),
            "traditional run did not resolve a fuel cooling/film topology",
        ),
        "fuel_flow_closure_residual_kg_s": maybe(
            (
                split_ledger.get("closure_residual_kg_s")
                if isinstance(split_ledger, Mapping)
                else None
            ),
            "traditional run did not resolve a fuel split ledger",
        ),
        "coolant_inlet_temperature_k": maybe(
            cooling_raw.get("coolant_inlet_temperature"),
            "regenerative cooling was not evaluated",
        ),
        "coolant_outlet_temperature_k": film_sensitive(maybe(
            cooling_raw.get("coolant_outlet_temperature"),
            "regenerative cooling was not evaluated",
        )),
        "coolant_pressure_drop_pa": maybe(
            cooling_raw.get("coolant_pressure_drop"),
            "regenerative pressure drop was not evaluated",
        ),
        "coolant_velocity_m_s": maybe(
            cooling_raw.get("channel_velocity"),
            "regenerative channel velocity was not evaluated",
        ),
        "coolant_mach": unavailable(
            "the traditional cooling report does not expose liquid coolant Mach"
        ),
        "land_width_min_m": maybe(
            _array_min(cooling_raw.get("land_width")),
            "the traditional cooling report did not expose channel land width",
        ),
        "coolant_temperature_profile": film_sensitive(_traditional_profile(
            cooling_raw, x, radius, "coolant_temperature", "K"
        )),
        "coolant_pressure_profile": _traditional_profile(
            cooling_raw, x, radius, "coolant_pressure", "Pa"
        ),
        "gas_pressure_profile": _traditional_profile(
            cooling_raw, x, radius, "gas_pressure", "Pa"
        ),
        "liner_pressure_differential_profile": _traditional_profile(
            cooling_raw,
            x,
            radius,
            "liner_pressure_differential",
            "Pa",
        ),
    })

    injector = SnapshotSection({
        "type": available(str(input_obj.injector.type)),
        "architecture": available(str(input_obj.injector.architecture)),
        "sizing": available(str(input_obj.injector.sizing)),
        "fuel_dp_fraction": available(input_obj.injector.fuel_dp_fraction),
        "oxidizer_dp_fraction": available(input_obj.injector.oxidizer_dp_fraction),
        "fuel_dp_pa": _nested_stream_value(inj_raw, "fuel", "dp_pa"),
        "oxidizer_dp_pa": _nested_stream_value(inj_raw, "oxidizer", "dp_pa"),
        "fuel_cd": available(input_obj.injector.fuel_cd),
        "oxidizer_cd": available(input_obj.injector.oxidizer_cd),
        "pintle_diameter_m": _field_from_mapping(
            inj_raw,
            "pintle_diameter_m",
            "pintle injector sizing was disabled or unavailable",
        ),
        "slot_count": maybe(
            inj_raw.get("slot_count", input_obj.injector.geometry.slot_count),
            "pintle slot count was unavailable",
        ),
        "momentum_ratio": _film_injector_sensitive(_field_from_mapping(
            inj_raw,
            "total_momentum_ratio",
            "pintle injector sizing was disabled or unavailable",
        ), film_mass_flow),
        "spray_half_angle_deg": _film_injector_sensitive(_field_from_mapping(
            inj_raw,
            "spray_half_angle_deg",
            "pintle injector sizing was disabled or unavailable",
        ), film_mass_flow),
        "blockage_factor": _film_injector_sensitive(_field_from_mapping(
            inj_raw,
            "blockage_factor",
            "pintle injector sizing was disabled or unavailable",
        ), film_mass_flow),
        "transition_margin_m2": _film_injector_sensitive(_nested_value(
            inj_raw,
            ("actuation", "transition_margin_m2"),
            "the selected traditional injector architecture has no Son transition margin",
        ), film_mass_flow),
        "fuel_velocity_m_s": _film_injector_sensitive(_nested_stream_value(
            inj_raw, "fuel", "velocity_m_s"
        ), film_mass_flow),
        "oxidizer_velocity_m_s": _nested_stream_value(
            inj_raw, "oxidizer", "velocity_m_s"
        ),
        "fuel_flow_area_m2": _film_injector_sensitive(_nested_stream_value(
            inj_raw, "fuel", "area_m2"
        ), film_mass_flow),
        "oxidizer_flow_area_m2": _nested_stream_value(
            inj_raw, "oxidizer", "area_m2"
        ),
        "slot_width_m": _film_injector_sensitive(_nested_value(
            inj_raw,
            ("slots", "detail", "slot_width"),
            "traditional injector did not expose a discrete slot width",
        ), film_mass_flow),
        "tip_opening_m": _film_injector_sensitive(_nested_value(
            inj_raw,
            ("actuation", "opening_distance_m"),
            "the selected traditional injector has no movable tip opening",
        ), film_mass_flow),
        "tip_branch_area_m2": _film_injector_sensitive(_nested_value(
            inj_raw,
            ("actuation", "tip_area_m2"),
            "the selected traditional injector has no Son tip-area branch",
        ), film_mass_flow),
        "center_gap_area_m2": _film_injector_sensitive(_nested_value(
            inj_raw,
            ("actuation", "center_gap_area_m2"),
            "the selected traditional injector has no center-gap area branch",
        ), film_mass_flow),
        "branch_consistency": unavailable(
            "the traditional injector does not expose the MDO "
            "branch-consistency scalar"
        ),
        "fuel_chug_margin": unavailable(
            "the traditional stability report does not expose the MDO "
            "fuel chug margin"
        ),
        "oxidizer_chug_margin": unavailable(
            "the traditional stability report does not expose the MDO "
            "oxidizer chug margin"
        ),
    })

    feed_dict = inj_raw.get("feed_system") or {}
    pump_lines = getattr(electric_pump_result, "lines", {}) or {}
    feed_electrical = SnapshotSection({
        "architecture": (
            available("electric_pump_fed")
            if electric_pump_result is not None
            else maybe(
                feed_dict.get("architecture"),
                "injector feed-system ledger was not evaluated",
            )
        ),
        "fuel_tank_pressure_pa": maybe(
            input_obj.injector.feed_system.fuel.tank_pressure,
            "fuel tank pressure was not supplied",
        ),
        "oxidizer_tank_pressure_pa": maybe(
            input_obj.injector.feed_system.oxidizer.tank_pressure,
            "oxidizer tank pressure was not supplied",
        ),
        "fuel_density_kg_m3": _feed_line_value(
            feed_dict, "fuel", "density_kg_m3"
        ),
        "oxidizer_density_kg_m3": _feed_line_value(
            feed_dict, "oxidizer", "density_kg_m3"
        ),
        "fuel_vapor_pressure_pa": _feed_line_value(
            feed_dict, "fuel", "vapor_pressure_pa"
        ),
        "oxidizer_vapor_pressure_pa": _feed_line_value(
            feed_dict, "oxidizer", "vapor_pressure_pa"
        ),
        "line_pressure_loss_pa": _common_line_loss(input_obj),
        "fuel_required_pressure_rise_pa": _feed_line_value(
            feed_dict, "fuel", "required_pressure_rise_pa"
        ),
        "oxidizer_required_pressure_rise_pa": _feed_line_value(
            feed_dict, "oxidizer", "required_pressure_rise_pa"
        ),
        "pump_speed_rpm": _common_pump_speed(pump_lines),
        "fuel_volumetric_flow_m3_s": _pump_line_attr(
            pump_lines, "fuel", "volumetric_flow"
        ),
        "oxidizer_volumetric_flow_m3_s": _pump_line_attr(
            pump_lines, "oxidizer", "volumetric_flow"
        ),
        "fuel_pump_head_m": _pump_line_attr(pump_lines, "fuel", "head"),
        "oxidizer_pump_head_m": _pump_line_attr(
            pump_lines, "oxidizer", "head"
        ),
        "fuel_specific_speed": _pump_component_attr(
            pump_lines, "fuel", "impeller", "specific_speed"
        ),
        "oxidizer_specific_speed": _pump_component_attr(
            pump_lines, "oxidizer", "impeller", "specific_speed"
        ),
        "fuel_npsh_available_pa": _feed_line_value(
            feed_dict, "fuel", "npsh_available_pa"
        ),
        "oxidizer_npsh_available_pa": _feed_line_value(
            feed_dict, "oxidizer", "npsh_available_pa"
        ),
        "fuel_suction_specific_speed": _pump_component_attr(
            pump_lines, "fuel", "inducer", "suction_specific_speed"
        ),
        "oxidizer_suction_specific_speed": _pump_component_attr(
            pump_lines, "oxidizer", "inducer", "suction_specific_speed"
        ),
        "fuel_nss_margin": unavailable(
            "traditional pump sizing does not use the MDO "
            "suction-specific-speed cap"
        ),
        "oxidizer_nss_margin": unavailable(
            "traditional pump sizing does not use the MDO "
            "suction-specific-speed cap"
        ),
        "fuel_tip_speed_m_s": _pump_component_attr(
            pump_lines, "fuel", "impeller", "tip_speed"
        ),
        "oxidizer_tip_speed_m_s": _pump_component_attr(
            pump_lines, "oxidizer", "impeller", "tip_speed"
        ),
        "fuel_tip_speed_margin_m_s": _pump_tip_speed_margin(
            pump_lines, "fuel", electric_pump_result
        ),
        "oxidizer_tip_speed_margin_m_s": _pump_tip_speed_margin(
            pump_lines, "oxidizer", electric_pump_result
        ),
        "fuel_pump_efficiency": _pump_line_attr(
            pump_lines, "fuel", "efficiency"
        ),
        "oxidizer_pump_efficiency": _pump_line_attr(
            pump_lines, "oxidizer", "efficiency"
        ),
        "fuel_hydraulic_power_w": _pump_line_attr(
            pump_lines, "fuel", "hydraulic_power"
        ),
        "oxidizer_hydraulic_power_w": _pump_line_attr(
            pump_lines, "oxidizer", "hydraulic_power"
        ),
        "fuel_shaft_power_w": _pump_line_attr(
            pump_lines, "fuel", "shaft_power"
        ),
        "oxidizer_shaft_power_w": _pump_line_attr(
            pump_lines, "oxidizer", "shaft_power"
        ),
        "electric_power_total_w": maybe(
            getattr(getattr(electric_pump_result, "battery", None), "electric_power", None),
            "electric-pump sizing was not requested or feed duty was unavailable",
        ),
        "motor_efficiency": maybe(
            _pump_assumption(electric_pump_result, "motor_efficiency"),
            "electric-pump sizing was not requested",
        ),
        "inverter_efficiency": maybe(
            _pump_assumption(electric_pump_result, "inverter_efficiency"),
            "electric-pump sizing was not requested",
        ),
    })

    battery = getattr(electric_pump_result, "battery", None)
    pump_rollup = getattr(electric_pump_result, "mass_rollup", None)
    rollup_reason = (
        getattr(pump_rollup, "unavailable_reason", None)
        if pump_rollup is not None
        else "traditional pump result predates the two-stream mass-rollup contract"
    )
    if pump_rollup is not None:
        # Authoritative completeness comes from the domain rollup.  Do not
        # reconstruct a total from whichever stream rows happened to resolve.
        pump_mass = pump_rollup.core_pump_mass_kg
        motor_mass = pump_rollup.motor_mass_kg
        inverter_mass = pump_rollup.inverter_mass_kg
        electric_mass = pump_rollup.complete_package_mass_kg
        battery_selected_mass = pump_rollup.battery_selected_mass_kg
    else:
        # Compatibility only for pre-rollup result objects.
        motor_mass = _sum_drive_attr(pump_lines, "motor_mass")
        inverter_mass = _sum_drive_attr(pump_lines, "inverter_mass")
        pump_mass = _pump_hardware_mass(electric_pump_result)
        battery_selected_mass = getattr(battery, "mass", None)
        electric_parts = [
            pump_mass, motor_mass, inverter_mass, battery_selected_mass
        ]
        electric_mass = (
            sum(float(v) for v in electric_parts)
            if all(v is not None for v in electric_parts)
            else None
        )
    traditional_chamber_mass = _hardware_mass_subsystem(
        hardware_mass, "thrust_chamber"
    )
    dry_mass_partial_exact = (
        float(electric_mass) + float(traditional_chamber_mass)
        if electric_mass is not None and traditional_chamber_mass is not None
        else None
    )
    masses = SnapshotSection({
        "pump_mass_kg": maybe(
            pump_mass,
            rollup_reason or "two-stream core pump hardware is incomplete",
        ),
        "motor_mass_kg": maybe(
            motor_mass,
            "traditional electric-drive sizing was not requested",
        ),
        "inverter_mass_kg": maybe(
            inverter_mass,
            "traditional electric-drive sizing was not requested",
        ),
        "battery_energy_limited_mass_kg": maybe(
            getattr(battery, "mass_energy_limited", None),
            "traditional battery sizing was not requested",
        ),
        "battery_power_limited_mass_kg": maybe(
            getattr(battery, "mass_power_limited", None),
            "traditional battery sizing was not requested",
        ),
        "battery_energy_installed_mass_kg": maybe(
            (
                float(battery.mass_energy_limited)
                * float(
                    _pump_assumption(
                        electric_pump_result, "battery_structural_margin"
                    )
                )
                if battery is not None
                and _pump_assumption(
                    electric_pump_result, "battery_structural_margin"
                ) is not None
                else None
            ),
            "traditional battery sizing was not requested",
        ),
        "battery_power_installed_mass_kg": maybe(
            (
                float(battery.mass_power_limited)
                * float(
                    _pump_assumption(
                        electric_pump_result, "battery_structural_margin"
                    )
                )
                if battery is not None
                and _pump_assumption(
                    electric_pump_result, "battery_structural_margin"
                ) is not None
                else None
            ),
            "traditional battery sizing was not requested",
        ),
        "battery_selected_mass_kg": maybe(
            battery_selected_mass,
            rollup_reason or "traditional battery sizing was not requested",
        ),
        "battery_objective_mass_kg": unavailable(
            "the traditional battery model reports its exact governing branch, "
            "not the MDO smooth-max objective"
        ),
        "electric_package_mass_kg": maybe(
            electric_mass,
            rollup_reason or (
                "not every traditional electric-package mass branch was available"
            ),
        ),
        "electric_package_objective_mass_kg": unavailable(
            "the traditional pipeline has no smooth electric-package objective"
        ),
        "dry_mass_partial_exact_mass_kg": maybe(
            dry_mass_partial_exact,
            "the exact partial subtotal needs both a complete traditional "
            "electric package and resolved thrust-chamber hardware mass",
        ),
        "dry_mass_partial_objective_mass_kg": unavailable(
            "the traditional pipeline reports as-built/exact masses, not the "
            "MDO smooth partial-dry objective"
        ),
        "thrust_chamber_liner_mass_kg": maybe(
            _hardware_mass_component(hardware_mass, "hot-gas liner"),
            _hardware_mass_reason(hardware_mass, "hot-gas liner"),
        ),
        "thrust_chamber_land_mass_kg": maybe(
            _hardware_mass_component(hardware_mass, "regen channel lands"),
            _hardware_mass_reason(hardware_mass, "regen channel lands"),
        ),
        "thrust_chamber_closeout_mass_kg": maybe(
            _hardware_mass_component(
                hardware_mass, "structural closeout / jacket"
            ),
            _hardware_mass_reason(
                hardware_mass, "structural closeout / jacket"
            ),
        ),
        "thrust_chamber_mass_kg": maybe(
            traditional_chamber_mass,
            _hardware_mass_reason(hardware_mass, subsystem="thrust_chamber"),
        ),
        "injector_mass_kg": maybe(
            _hardware_mass_subsystem(hardware_mass, "injector"),
            _hardware_mass_reason(hardware_mass, subsystem="injector"),
        ),
        "total_engine_package_mass_kg": unavailable(
            "a total engine mass cannot be formed until propellant valves, "
            "lines, manifolds, gimbal and mount hardware are in the ledger; "
            "the thrust chamber, bolted interface, injector and electric feed "
            "system alone are not an engine dry mass"
        ),
        "engine_hardware_mass_ledger": maybe(
            hardware_mass or None,
            "the traditional design report has no hardware_mass section",
        ),
        "raw_mass_ledger": maybe(
            (
                [item.to_dict() for item in electric_pump_result.hardware_bom]
                if electric_pump_result is not None
                else None
            ),
            "traditional electric-pump hardware BOM was not generated",
        ),
    })
    gate_dict = result.gate_report.to_dict()
    pump_gate_dict = (
        electric_pump_result.feasibility.to_dict()
        if electric_pump_result is not None
        else None
    )
    readiness_categories = {
        "workflow", "cad", "configuration", "release", "manufacturing"
    }
    physics_checks = [
        check for check in result.gate_report.checks
        if check.category not in readiness_categories
    ]
    readiness_checks = [
        check for check in result.gate_report.checks
        if check.category in readiness_categories
    ]
    design_physics_feasible = all(check.passed for check in physics_checks)
    physics_feasible = (
        design_physics_feasible and bool(electric_pump_result.feasible)
        if electric_pump_result is not None
        else None
    )
    readiness_feasible = all(check.passed for check in readiness_checks)
    whole_engine_feasibility_reason = (
        "whole-engine feasibility is unavailable because authoritative "
        "electric-pump sizing was not supplied; the traditional design gates "
        "cover the chamber/nozzle/injector workflow only"
    )
    constraints_gates = SnapshotSection({
        "all_constraints_feasible": (
            available(
                bool(result.gate_report.passed)
                and bool(electric_pump_result.feasible)
            )
            if electric_pump_result is not None
            else unavailable(whole_engine_feasibility_reason)
        ),
        "optimizer_constraints_feasible": unavailable(
            "the traditional design workflow does not execute the MDO "
            "optimizer constraint manifest"
        ),
        "numerical_validity": unavailable(
            "traditional heterogeneous design gates do not expose the MDO "
            "residual and solver-status manifest rows"
        ),
        "physics_feasible": (
            available(physics_feasible)
            if physics_feasible is not None
            else unavailable(whole_engine_feasibility_reason)
        ),
        "requirements_feasible": unavailable(
            "the traditional result does not retain a ResolvedRequirement "
            "contract, so design-gate success is not a requirement verdict"
        ),
        "workflow_readiness_feasible": available(readiness_feasible),
        "constraint_margins": unavailable(
            "traditional gates are pass/fail checks with heterogeneous values, "
            "not one common signed-margin vector"
        ),
        "diagnostics": available({
            "design_status": result.design_status,
            "validated": bool(result.validated),
        }),
        "outer_thrust_residual": maybe(
            result.report_sections.get("thrust_closure", {}).get(
                "thrust_residual_n"
            ),
            "traditional thrust closure did not expose a scalar residual",
        ),
        "cooling_residual_max_abs_k": unavailable(
            "traditional cooling does not expose the nonlinear residual vector"
        ),
        "authoritative_design_gates": available(_json_ready({
            "design": gate_dict,
            "electric_pumps": pump_gate_dict,
        })),
    })
    provenance = SnapshotSection({
        "analysis_source": available("traditional_design_nozzle_v2"),
        "contract_version": available(CONTRACT_VERSION),
        "propellant": available(result.propellant.name),
        "coolant": maybe(
            input_obj.cooling.coolant,
            "no coolant was requested",
        ),
        "thermochemistry": available({
            "mode": result.thermochemistry.mode,
            "source": result.thermochemistry.source,
        }),
        "geometry_model": available(str(contour.get("method", input_obj.method))),
        "material_assumptions": available(_json_ready(input_obj.material)),
        "input_conventions": available(
            _traditional_input_conventions(
                result, input_obj, electric_pump_result
            )
        ),
        "mission": unavailable(
            "traditional DesignInput does not carry the MDO MissionSpec object"
        ),
        "design": available(_json_ready(input_obj)),
    })
    files = {key: str(path) for key, path in (result.files or {}).items()}
    artifacts = SnapshotSection({
        "files": available(files) if files else unavailable(
            "this traditional design run did not request file/CAD artifacts"
        ),
        "report_sections": available(_json_ready(result.report_sections)),
        "cad_files": (
            available(_cad_files(files))
            if _cad_files(files)
            else unavailable(
                "this traditional design run did not request CAD artifacts"
            )
        ),
    })
    warnings = list(result.warnings or ())
    if film_mass_flow > 0.0:
        warnings.append(
            "Fuel-film mass and hydraulic topology were retained, but "
            "film-sensitive traditional thermal outputs are unavailable "
            "because design_nozzle_v2 does not yet apply the MDO wall-film "
            "heat-load model."
        )
        warnings.append(
            "Film-sensitive main-pintle outputs are unavailable because the "
            "separate film injector/orifice and branch state are not modeled."
        )
    if electric_pump_result is not None:
        warnings.extend(getattr(electric_pump_result, "notes", ()) or ())
    return EngineAnalysisSnapshot(
        source="traditional",
        performance=performance,
        geometry=geometry,
        thermal=thermal,
        cooling=cooling,
        injector=injector,
        feed_electrical=feed_electrical,
        masses=masses,
        constraints_gates=constraints_gates,
        provenance=provenance,
        warnings=tuple(dict.fromkeys(str(w) for w in warnings)),
        artifacts=artifacts,
        optimizer_metadata=(
            dict(optimizer_metadata) if optimizer_metadata is not None else None
        ),
        source_result=result,
        auxiliary_results=(
            {"electric_pumps": electric_pump_result}
            if electric_pump_result is not None
            else {}
        ),
    )


def _array_max(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return float(np.nanmax(arr)) if arr.size else None


def _cad_files(files: Mapping[str, str]) -> dict[str, str]:
    return {
        key: path
        for key, path in files.items()
        if Path(path).suffix.lower() in {".step", ".stp", ".stl", ".ipt", ".dxf"}
    }


def _array_min(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return float(np.nanmin(arr)) if arr.size else None


def _traditional_profile(
    section: Mapping[str, Any],
    fallback_x: np.ndarray,
    fallback_radius: np.ndarray,
    key: str,
    units: str,
) -> SnapshotValue:
    values = section.get(key)
    if values is None:
        return unavailable(f"traditional report did not expose {key}")
    local_x = np.asarray(section.get("x", fallback_x), dtype=float)
    if local_x.shape != np.asarray(values).shape:
        return unavailable(
            f"traditional {key} stations do not align with an axial coordinate"
        )
    if local_x.shape == fallback_x.shape and np.allclose(
        local_x, fallback_x, rtol=0.0, atol=1.0e-12
    ):
        local_radius = fallback_radius
    else:
        local_radius = np.interp(local_x, fallback_x, fallback_radius)
    return _profile(local_x, local_radius, values, units=units)


def _nested_value(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
    reason: str,
) -> SnapshotValue:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return unavailable(reason)
        value = value[key]
    return maybe(value, reason)


def _nested_stream_value(
    injector: Mapping[str, Any],
    role: str,
    key: str,
) -> SnapshotValue:
    # Injector ``to_dict`` exposes streams by geometry name as top-level
    # ``annulus``/``slots`` entries.  Their ``role`` says which propellant.
    for name in ("annulus", "slots"):
        stream = injector.get(name)
        if isinstance(stream, Mapping) and stream.get("role") == role:
            return maybe(
                stream.get(key),
                f"traditional injector {role} stream did not expose {key}",
            )
    return unavailable(
        f"traditional injector {role} stream was not evaluated"
    )


def _feed_line_value(
    feed: Mapping[str, Any],
    role: str,
    key: str,
) -> SnapshotValue:
    lines = feed.get("lines", {}) if isinstance(feed, Mapping) else {}
    line = lines.get(role, {}) if isinstance(lines, Mapping) else {}
    return maybe(
        line.get(key) if isinstance(line, Mapping) else None,
        f"traditional {role} feed duty was unavailable",
    )


def _pump_line_attr(
    lines: Mapping[str, Any],
    role: str,
    attr: str,
) -> SnapshotValue:
    line = lines.get(role) if isinstance(lines, Mapping) else None
    return maybe(
        getattr(line, attr, None),
        f"traditional {role} electric-pump sizing was unavailable",
    )


def _pump_component_attr(
    lines: Mapping[str, Any],
    role: str,
    component: str,
    attr: str,
) -> SnapshotValue:
    line = lines.get(role) if isinstance(lines, Mapping) else None
    part = getattr(line, component, None)
    return maybe(
        getattr(part, attr, None),
        f"traditional {role} pump {component} did not expose {attr}",
    )


def _common_line_loss(input_obj: Any) -> SnapshotValue:
    fuel = getattr(input_obj.injector.feed_system.fuel, "line_loss", None)
    oxidizer = getattr(
        input_obj.injector.feed_system.oxidizer, "line_loss", None
    )
    if fuel is None or oxidizer is None:
        return unavailable("traditional feed-line loss was not supplied")
    if not np.isclose(float(fuel), float(oxidizer), rtol=0.0, atol=1.0e-12):
        return unavailable(
            "traditional fuel and oxidizer line-pressure losses differ"
        )
    return available(float(fuel))


def _pump_tip_speed_margin(
    lines: Mapping[str, Any],
    role: str,
    pump_result: Any,
) -> SnapshotValue:
    line = lines.get(role) if isinstance(lines, Mapping) else None
    impeller = getattr(line, "impeller", None)
    tip_speed = getattr(impeller, "tip_speed", None)
    limit = _pump_assumption(pump_result, "material_tip_speed_limit_m_s")
    if tip_speed is None or limit is None:
        return unavailable(
            f"traditional {role} pump tip-speed margin was unavailable"
        )
    return available(float(limit) - float(tip_speed))


def _common_pump_speed(lines: Mapping[str, Any]) -> SnapshotValue:
    speeds = []
    for line in lines.values() if isinstance(lines, Mapping) else ():
        drive = getattr(line, "drive", None)
        if drive is not None and getattr(drive, "rpm", None) is not None:
            speeds.append(float(drive.rpm))
    if not speeds:
        return unavailable("traditional electric-pump sizing was not requested")
    if not np.allclose(speeds, speeds[0], rtol=1.0e-9, atol=1.0e-6):
        return unavailable("traditional fuel and oxidizer pump speeds differ")
    return available(speeds[0])


def _pump_assumption(pump_result: Any, key: str) -> Any:
    assumptions = getattr(pump_result, "assumptions", None)
    return assumptions.get(key) if isinstance(assumptions, Mapping) else None


def _sum_drive_attr(lines: Mapping[str, Any], attr: str) -> float | None:
    values = []
    for line in lines.values() if isinstance(lines, Mapping) else ():
        drive = getattr(line, "drive", None)
        value = getattr(drive, attr, None)
        if value is None:
            return None
        values.append(float(value))
    return sum(values) if values else None


def _polyline_wetted_area(x: Any, radius: Any) -> float:
    """Gas-side wetted area of a meridional polyline.

    Mirrors :func:`raosim.mdo.grid.wetted_area` and
    :func:`raosim.regen_profile._nodal_weights_from_segments` exactly, so the
    two pipelines' wetted areas are comparable numbers rather than two
    different quadratures.  This is the quantity that scales both the total
    heat load and the wall mass, which is why it is a first-class snapshot
    field rather than a derived diagnostic.
    """

    x = np.asarray(x, dtype=float)
    radius = np.asarray(radius, dtype=float)
    seg = np.hypot(np.diff(x), np.diff(radius))
    weights = np.empty(seg.size + 1, dtype=float)
    weights[0] = 0.5 * seg[0]
    weights[-1] = 0.5 * seg[-1]
    weights[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return float(np.sum(2.0 * np.pi * radius * weights))


_LEGACY_NO_CHAMBER_MASS = (
    "this EngineResult predates the station-grid chamber mass integral "
    "(raosim.mdo.mass); its ledger has no thrust-chamber entry"
)


def _hardware_mass_items(hardware_mass: Mapping[str, Any] | None) -> list[dict]:
    if not hardware_mass:
        return []
    items = hardware_mass.get("items")
    return list(items) if isinstance(items, list) else []


def _hardware_mass_component(
    hardware_mass: Mapping[str, Any] | None, component: str
) -> float | None:
    """Mass of one named ledger row, or ``None`` if it did not resolve."""

    for item in _hardware_mass_items(hardware_mass):
        if item.get("component") == component:
            value = item.get("mass_kg")
            if value is None:
                return None
            return float(value) * int(item.get("quantity", 1) or 1)
    return None


def _hardware_mass_subsystem(
    hardware_mass: Mapping[str, Any] | None, subsystem: str
) -> float | None:
    """Subsystem rollup, withheld entirely if any of its rows is unavailable.

    Summing only the resolved rows would turn a lower bound into a claimed
    hardware mass, which is exactly the failure mode the availability contract
    exists to prevent.
    """

    rows = [
        item for item in _hardware_mass_items(hardware_mass)
        if item.get("subsystem") == subsystem
    ]
    if not rows or any(item.get("mass_kg") is None for item in rows):
        return None
    return sum(
        float(item["mass_kg"]) * int(item.get("quantity", 1) or 1)
        for item in rows
    )


def _hardware_mass_reason(
    hardware_mass: Mapping[str, Any] | None,
    component: str | None = None,
    *,
    subsystem: str | None = None,
) -> str:
    """A stable, specific unavailability reason for the mass section."""

    if not hardware_mass:
        return (
            "the traditional design report has no hardware_mass section; "
            "rebuild the design with a current raosim.design"
        )
    rows = _hardware_mass_items(hardware_mass)
    if component is not None:
        for item in rows:
            if item.get("component") == component:
                return (
                    item.get("unavailable_reason")
                    or f"'{component}' did not resolve to a mass"
                )
        return (
            f"the hardware mass ledger has no '{component}' row; "
            + str(hardware_mass.get("unavailable_reason")
                  or "; ".join(hardware_mass.get("notes") or ())
                  or "the contributing geometry was not resolved")
        )
    matching = [item for item in rows if item.get("subsystem") == subsystem]
    if not matching:
        return (
            f"the hardware mass ledger has no '{subsystem}' rows; "
            + str(hardware_mass.get("unavailable_reason")
                  or "; ".join(hardware_mass.get("notes") or ())
                  or "the contributing geometry was not resolved")
        )
    missing = [
        f"{item.get('component')}: {item.get('unavailable_reason')}"
        for item in matching if item.get("mass_kg") is None
    ]
    return (
        f"the '{subsystem}' mass ledger is incomplete -- " + "; ".join(missing)
        if missing else f"the '{subsystem}' mass rollup is available"
    )


def _pump_hardware_mass(pump_result: Any) -> float | None:
    """Return a pump-hardware mass only when its core BOM is complete.

    The core is the wetted, load-carrying pump: hydraulic rotor and diffusion
    system, mechanical drivetrain, pressure boundary, and the inlet/outlet port
    stubs that close it.  Summing only the non-``None`` rows would turn an
    incomplete lower bound into a claimed pump mass, violating the snapshot's
    availability contract, so a single missing row still withholds the total.

    Instrumentation is deliberately outside the core: sensor and harness mass
    depends on the flight avionics architecture, not on the pump sizing, and
    ``size_electric_pumps`` correctly reports it as unknown.
    """

    if pump_result is None:
        return None
    core = [
        item
        for item in getattr(pump_result, "hardware_bom", ())
        if item.subsystem in {
            "hydraulic", "mechanical", "pressure_boundary", "interface",
        }
    ]
    if not core or any(item.mass_estimate_kg is None for item in core):
        return None
    return sum(
        float(item.mass_estimate_kg) * int(item.quantity)
        for item in core
    )


@dataclass(frozen=True)
class FieldComparison:
    """Comparison of one scalar or normalized profile."""

    left: Any
    right: Any
    absolute_delta: float | None
    relative_delta: float | None
    status: str
    signed_delta: float | None = None
    not_comparable_reason: str | None = None
    mean_absolute_delta: float | None = None
    sample_count: int | None = None

    @property
    def not_comparable(self) -> bool:
        return self.status == "not_comparable"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class SnapshotComparison:
    """All common scalar and profile comparisons between two snapshots."""

    left_source: str
    right_source: str
    scalars: dict[str, FieldComparison]
    profiles: dict[str, FieldComparison]
    abs_tol: float
    rel_tol: float

    @property
    def comparable_count(self) -> int:
        return sum(
            not item.not_comparable
            for item in (*self.scalars.values(), *self.profiles.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_source": self.left_source,
            "right_source": self.right_source,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "comparable_count": self.comparable_count,
            "scalars": {k: v.to_dict() for k, v in self.scalars.items()},
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
        }


_COMPARABLE_SECTIONS = (
    "performance",
    "geometry",
    "thermal",
    "cooling",
    "injector",
    "feed_electrical",
    "masses",
    "constraints_gates",
)


def compare_snapshots(
    left: EngineAnalysisSnapshot,
    right: EngineAnalysisSnapshot,
    *,
    abs_tol: float = 1.0e-9,
    rel_tol: float = 1.0e-6,
    profile_points: int = 101,
) -> SnapshotComparison:
    """Compare every common scalar and normalized axial profile.

    Numeric scalars use the requested tolerances.  Categorical scalars such as
    topology, architecture, and method names use exact equality so convention
    drift is visible instead of silently disappearing from the comparison.
    """

    scalars: dict[str, FieldComparison] = {}
    profiles: dict[str, FieldComparison] = {}
    for section_name in _COMPARABLE_SECTIONS:
        left_fields = left.section(section_name).fields
        right_fields = right.section(section_name).fields
        for name in sorted(set(left_fields) & set(right_fields)):
            path = f"{section_name}.{name}"
            lv, rv = left_fields[name], right_fields[name]
            if not lv.available or not rv.available:
                reason = "; ".join(
                    text
                    for text in (
                        (
                            f"{left.source}: {lv.availability_reason}"
                            if not lv.available else None
                        ),
                        (
                            f"{right.source}: {rv.availability_reason}"
                            if not rv.available else None
                        ),
                    )
                    if text
                )
                target = (
                    profiles
                    if isinstance(lv.value, NormalizedProfile)
                    or isinstance(rv.value, NormalizedProfile)
                    or name.endswith("_profile")
                    else scalars
                )
                target[path] = FieldComparison(
                    left=_json_ready(lv.value),
                    right=_json_ready(rv.value),
                    absolute_delta=None,
                    relative_delta=None,
                    status="not_comparable",
                    not_comparable_reason=reason,
                )
                continue
            if isinstance(lv.value, NormalizedProfile) or isinstance(
                rv.value, NormalizedProfile
            ):
                profiles[path] = _compare_profiles(
                    lv.value,
                    rv.value,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    points=profile_points,
                )
                continue
            if _is_scalar_number(lv.value) and _is_scalar_number(rv.value):
                a, b = float(lv.value), float(rv.value)
                if not np.isfinite(a) or not np.isfinite(b):
                    scalars[path] = FieldComparison(
                        left=a,
                        right=b,
                        absolute_delta=None,
                        relative_delta=None,
                        status="not_comparable",
                        not_comparable_reason="one or both scalar values are non-finite",
                    )
                    continue
                delta = b - a
                scale = max(abs(a), abs(b), 1.0e-30)
                absolute_delta = abs(delta)
                relative_delta = absolute_delta / scale
                scalars[path] = FieldComparison(
                    left=a,
                    right=b,
                    absolute_delta=absolute_delta,
                    relative_delta=relative_delta,
                    status=(
                        "within_tolerance"
                        if absolute_delta <= abs_tol + rel_tol * scale
                        else "different"
                    ),
                    signed_delta=delta,
                )
                continue
            if _is_categorical_scalar(lv.value) and _is_categorical_scalar(
                rv.value
            ):
                left_value = _json_ready(lv.value)
                right_value = _json_ready(rv.value)
                scalars[path] = FieldComparison(
                    left=left_value,
                    right=right_value,
                    absolute_delta=None,
                    relative_delta=None,
                    status=(
                        "within_tolerance"
                        if left_value == right_value
                        else "different"
                    ),
                )
    return SnapshotComparison(
        left_source=left.source,
        right_source=right.source,
        scalars=scalars,
        profiles=profiles,
        abs_tol=float(abs_tol),
        rel_tol=float(rel_tol),
    )


def _is_scalar_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, (bool, np.bool_))
    )


def _is_categorical_scalar(value: Any) -> bool:
    """Return whether *value* is an exact-match host scalar."""

    return isinstance(value, (str, bytes, bool, np.bool_))


def _compare_profiles(
    left: Any,
    right: Any,
    *,
    abs_tol: float,
    rel_tol: float,
    points: int,
) -> FieldComparison:
    if not isinstance(left, NormalizedProfile) or not isinstance(
        right, NormalizedProfile
    ):
        return FieldComparison(
            left=_json_ready(left),
            right=_json_ready(right),
            absolute_delta=None,
            relative_delta=None,
            status="not_comparable",
            not_comparable_reason="one side is not a normalized profile",
        )
    lc, lv = _clean_profile(left)
    rc, rv = _clean_profile(right)
    if lc.size < 2 or rc.size < 2:
        return FieldComparison(
            left=left.to_dict(),
            right=right.to_dict(),
            absolute_delta=None,
            relative_delta=None,
            status="not_comparable",
            not_comparable_reason="one profile has fewer than two finite unique stations",
        )
    lo = max(float(lc[0]), float(rc[0]))
    hi = min(float(lc[-1]), float(rc[-1]))
    if not hi > lo:
        return FieldComparison(
            left=left.to_dict(),
            right=right.to_dict(),
            absolute_delta=None,
            relative_delta=None,
            status="not_comparable",
            not_comparable_reason="normalized profiles have no overlapping axial domain",
        )
    coord = np.linspace(lo, hi, max(int(points), 2))
    a = np.interp(coord, lc, lv)
    b = np.interp(coord, rc, rv)
    delta = b - a
    max_abs = float(np.max(np.abs(delta)))
    mean_abs = float(np.mean(np.abs(delta)))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0e-30)
    max_rel = max_abs / scale
    return FieldComparison(
        left={"units": left.units, "stations": int(left.values.size)},
        right={"units": right.units, "stations": int(right.values.size)},
        absolute_delta=max_abs,
        relative_delta=max_rel,
        status=(
            "within_tolerance"
            if max_abs <= abs_tol + rel_tol * scale
            else "different"
        ),
        mean_absolute_delta=mean_abs,
        sample_count=int(coord.size),
    )


def _clean_profile(profile: NormalizedProfile) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(profile.coordinate, dtype=float)
    v = np.asarray(profile.values, dtype=float)
    mask = np.isfinite(c) & np.isfinite(v)
    c, v = c[mask], v[mask]
    if not c.size:
        return c, v
    order = np.argsort(c)
    c, v = c[order], v[order]
    unique, idx = np.unique(c, return_index=True)
    return unique, v[idx]


__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "SNAPSHOT_FIELD_MANIFEST",
    "EngineAnalysisSnapshot",
    "FieldComparison",
    "NormalizedProfile",
    "SnapshotComparison",
    "SnapshotSection",
    "SnapshotValue",
    "available",
    "compare_snapshots",
    "maybe",
    "snapshot_from_mdo",
    "snapshot_from_traditional",
    "unavailable",
]
