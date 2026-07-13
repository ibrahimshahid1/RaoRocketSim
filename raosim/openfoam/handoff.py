"""Fail-closed contract for an OpenFOAM VOF-to-parcel handoff.

This module deliberately does not import OpenFOAM bindings, read a case, or run
subprocesses.  It records the evidence produced by a separate VOF extraction
workflow and determines whether that evidence is strong enough to construct the
radial-sheet source used by :mod:`raosim.spray`.

All dimensional values are SI.  In particular,
``full_sheet_thickness_mean_m`` is the *full* liquid-sheet thickness (the
movable-pintle opening convention), never a half-thickness or hydraulic
diameter.  A handoff is usable only when every derived and caller-supplied
required gate passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal

from ..spray.primary import RadialSheetGeometry


class VOFHandoffValidationError(ValueError):
    """Raised for malformed or physically ineligible VOF handoff evidence."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise VOFHandoffValidationError(f"{name} must be nonblank")
    return result


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise VOFHandoffValidationError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise VOFHandoffValidationError(f"{name} must be > 0")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise VOFHandoffValidationError(f"{name} must be >= 0")
    return result


def _fraction(name: str, value: float, *, include_one: bool = True) -> float:
    result = _finite(name, value)
    upper_ok = result <= 1.0 if include_one else result < 1.0
    if result <= 0.0 or not upper_ok:
        right = "]" if include_one else ")"
        raise VOFHandoffValidationError(f"{name} must be in (0, 1{right}")
    return result


def _sha256(name: str, value: str) -> str:
    result = _text(name, value).lower()
    if result.startswith("sha256:"):
        result = result[7:]
    if not _SHA256_RE.fullmatch(result):
        raise VOFHandoffValidationError(
            f"{name} must be a 64-character SHA-256 digest"
        )
    return result


def _vector2(name: str, value: tuple[float, float]) -> tuple[float, float]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise VOFHandoffValidationError(f"{name} must contain two values") from exc
    if len(raw) != 2:
        raise VOFHandoffValidationError(f"{name} must contain two values")
    return (_finite(f"{name}[0]", raw[0]), _finite(f"{name}[1]", raw[1]))


def _bounds(name: str, value: tuple[float, float], *, nonnegative: bool) -> tuple[float, float]:
    lower, upper = _vector2(name, value)
    if nonnegative and lower < 0.0:
        raise VOFHandoffValidationError(f"{name}[0] must be >= 0")
    if upper <= lower:
        raise VOFHandoffValidationError(f"{name} upper bound must exceed lower bound")
    return lower, upper


def _norm2(value: tuple[float, float]) -> float:
    return math.hypot(value[0], value[1])


def _relative_scalar_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / abs(reference)


def _relative_vector_error(
    actual: tuple[float, float], reference: tuple[float, float]
) -> float:
    denominator = _norm2(reference)
    if denominator <= 0.0:
        return math.inf
    return math.hypot(actual[0] - reference[0], actual[1] - reference[1]) / denominator


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VOFArtifactProvenance:
    """Immutable identity of the case, inputs, solver build, and extractor."""

    case_id: str
    case_fingerprint_sha256: str
    input_fingerprint_sha256: str
    solver_name: str
    solver_version: str
    solver_version_fingerprint_sha256: str
    extraction_code_fingerprint_sha256: str

    def __post_init__(self) -> None:
        for name in ("case_id", "solver_name", "solver_version"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        for name in (
            "case_fingerprint_sha256",
            "input_fingerprint_sha256",
            "solver_version_fingerprint_sha256",
            "extraction_code_fingerprint_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))


@dataclass(frozen=True)
class VOFAveragingWindow:
    """Physical-time interval used for time-averaged sheet extraction."""

    start_time_s: float
    end_time_s: float
    sample_count: int
    observed_flow_through_times: float
    required_flow_through_times: float

    def __post_init__(self) -> None:
        start = _nonnegative("start_time_s", self.start_time_s)
        end = _positive("end_time_s", self.end_time_s)
        if end <= start:
            raise VOFHandoffValidationError("end_time_s must exceed start_time_s")
        if isinstance(self.sample_count, bool) or int(self.sample_count) != self.sample_count:
            raise VOFHandoffValidationError("sample_count must be an integer >= 2")
        count = int(self.sample_count)
        if count < 2:
            raise VOFHandoffValidationError("sample_count must be an integer >= 2")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(
            self,
            "observed_flow_through_times",
            _positive("observed_flow_through_times", self.observed_flow_through_times),
        )
        object.__setattr__(
            self,
            "required_flow_through_times",
            _positive("required_flow_through_times", self.required_flow_through_times),
        )

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def coverage_passed(self) -> bool:
        return self.observed_flow_through_times >= self.required_flow_through_times


@dataclass(frozen=True)
class VOFSheetExtractionDefinition:
    """Pinned definition of how a liquid sheet is extracted from VOF fields."""

    alpha_liquid_threshold: float
    algorithm_id: str
    algorithm_version_fingerprint_sha256: str
    connected_component_policy: str
    interpolation_rule: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alpha_liquid_threshold",
            _fraction(
                "alpha_liquid_threshold", self.alpha_liquid_threshold, include_one=False
            ),
        )
        for name in ("algorithm_id", "connected_component_policy", "interpolation_rule"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "algorithm_version_fingerprint_sha256",
            _sha256(
                "algorithm_version_fingerprint_sha256",
                self.algorithm_version_fingerprint_sha256,
            ),
        )


@dataclass(frozen=True)
class VOFSheetStatistics:
    """Mass-weighted sheet state at the VOF-to-Lagrangian handoff surface.

    Velocity components follow the spray convention: ``x`` is axial and ``r``
    points radially outward.  Standard deviations describe temporal/spatial
    variation over the declared averaging window.
    """

    exit_radius_m: float
    axial_location_m: float
    full_sheet_thickness_mean_m: float
    full_sheet_thickness_standard_deviation_m: float
    axial_velocity_mean_m_s: float
    radial_velocity_mean_m_s: float
    axial_velocity_standard_deviation_m_s: float
    radial_velocity_standard_deviation_m_s: float
    maximum_thickness_coefficient_of_variation: float
    maximum_velocity_coefficient_of_variation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "exit_radius_m", _positive("exit_radius_m", self.exit_radius_m))
        object.__setattr__(
            self, "axial_location_m", _finite("axial_location_m", self.axial_location_m)
        )
        object.__setattr__(
            self,
            "full_sheet_thickness_mean_m",
            _positive("full_sheet_thickness_mean_m", self.full_sheet_thickness_mean_m),
        )
        if self.full_sheet_thickness_mean_m >= 2.0 * self.exit_radius_m:
            raise VOFHandoffValidationError(
                "full sheet thickness must be smaller than the sheet exit diameter"
            )
        for name in (
            "full_sheet_thickness_standard_deviation_m",
            "axial_velocity_standard_deviation_m_s",
            "radial_velocity_standard_deviation_m_s",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        axial = _nonnegative("axial_velocity_mean_m_s", self.axial_velocity_mean_m_s)
        radial = _positive("radial_velocity_mean_m_s", self.radial_velocity_mean_m_s)
        object.__setattr__(self, "axial_velocity_mean_m_s", axial)
        object.__setattr__(self, "radial_velocity_mean_m_s", radial)
        for name in (
            "maximum_thickness_coefficient_of_variation",
            "maximum_velocity_coefficient_of_variation",
        ):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))

    @property
    def velocity_mean_m_s(self) -> tuple[float, float]:
        return self.axial_velocity_mean_m_s, self.radial_velocity_mean_m_s

    @property
    def thickness_coefficient_of_variation(self) -> float:
        return (
            self.full_sheet_thickness_standard_deviation_m
            / self.full_sheet_thickness_mean_m
        )

    @property
    def velocity_coefficient_of_variation(self) -> float:
        return math.hypot(
            self.axial_velocity_standard_deviation_m_s,
            self.radial_velocity_standard_deviation_m_s,
        ) / math.hypot(self.axial_velocity_mean_m_s, self.radial_velocity_mean_m_s)

    @property
    def tip_angle_deg(self) -> float:
        return math.degrees(
            math.atan2(self.axial_velocity_mean_m_s, self.radial_velocity_mean_m_s)
        )


@dataclass(frozen=True)
class VOFLiquidFluxBalance:
    """Liquid mass and momentum flux closure across the extraction surface.

    Momentum vectors are ordered ``(axial, radial)`` and have units N
    (kg m/s^2), i.e. the area-integrated convective momentum flux.
    """

    liquid_name: str
    liquid_density_kg_m3: float
    inlet_mass_flow_rate_kg_s: float
    extracted_mass_flow_rate_kg_s: float
    inlet_momentum_flux_n: tuple[float, float]
    extracted_momentum_flux_n: tuple[float, float]
    mass_closure_relative_tolerance: float
    momentum_closure_relative_tolerance: float
    kinematic_momentum_relative_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "liquid_name", _text("liquid_name", self.liquid_name))
        object.__setattr__(
            self,
            "liquid_density_kg_m3",
            _positive("liquid_density_kg_m3", self.liquid_density_kg_m3),
        )
        for name in ("inlet_mass_flow_rate_kg_s", "extracted_mass_flow_rate_kg_s"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("inlet_momentum_flux_n", "extracted_momentum_flux_n"):
            vector = _vector2(name, getattr(self, name))
            if _norm2(vector) <= 0.0:
                raise VOFHandoffValidationError(f"{name} magnitude must be > 0")
            object.__setattr__(self, name, vector)
        for name in (
            "mass_closure_relative_tolerance",
            "momentum_closure_relative_tolerance",
            "kinematic_momentum_relative_tolerance",
        ):
            object.__setattr__(self, name, _fraction(name, getattr(self, name)))

    @property
    def mass_relative_error(self) -> float:
        return _relative_scalar_error(
            self.extracted_mass_flow_rate_kg_s, self.inlet_mass_flow_rate_kg_s
        )

    @property
    def momentum_relative_error(self) -> float:
        return _relative_vector_error(
            self.extracted_momentum_flux_n, self.inlet_momentum_flux_n
        )


@dataclass(frozen=True)
class CarrierAxisymmetricFieldEvidence:
    """Identity, domain, and reference properties of the carrier field.

    The SHA-256 field identity binds the full axisymmetric arrays stored by the
    external workflow.  The scalar values here are the resolved reference state
    and transport properties used when creating the field, not a replacement
    for those arrays.
    """

    operating_point_id: str
    field_fingerprint_sha256: str
    state_fingerprint_sha256: str
    fluid_name: str
    axial_bounds_m: tuple[float, float]
    radial_bounds_m: tuple[float, float]
    grid_shape_axial_radial: tuple[int, int]
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    temperature_k: float
    pressure_pa: float
    specific_heat_j_kg_k: float
    thermal_conductivity_w_m_k: float
    mean_axial_velocity_m_s: float
    mean_radial_velocity_m_s: float
    turbulent_kinetic_energy_m2_s2: float
    turbulent_dissipation_rate_m2_s3: float
    coordinate_system: str = field(init=False, default="axisymmetric_x_r")

    def __post_init__(self) -> None:
        for name in ("operating_point_id", "fluid_name"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        for name in ("field_fingerprint_sha256", "state_fingerprint_sha256"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(
            self, "axial_bounds_m", _bounds("axial_bounds_m", self.axial_bounds_m, nonnegative=False)
        )
        object.__setattr__(
            self, "radial_bounds_m", _bounds("radial_bounds_m", self.radial_bounds_m, nonnegative=True)
        )
        try:
            shape = tuple(self.grid_shape_axial_radial)
        except TypeError as exc:
            raise VOFHandoffValidationError(
                "grid_shape_axial_radial must contain two integers >= 2"
            ) from exc
        if len(shape) != 2 or any(
            isinstance(value, bool) or int(value) != value or int(value) < 2
            for value in shape
        ):
            raise VOFHandoffValidationError(
                "grid_shape_axial_radial must contain two integers >= 2"
            )
        object.__setattr__(self, "grid_shape_axial_radial", (int(shape[0]), int(shape[1])))
        for name in (
            "density_kg_m3",
            "dynamic_viscosity_pa_s",
            "temperature_k",
            "pressure_pa",
            "specific_heat_j_kg_k",
            "thermal_conductivity_w_m_k",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("mean_axial_velocity_m_s", "mean_radial_velocity_m_s"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in (
            "turbulent_kinetic_energy_m2_s2",
            "turbulent_dissipation_rate_m2_s3",
        ):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        if (
            self.turbulent_kinetic_energy_m2_s2 > 0.0
            and self.turbulent_dissipation_rate_m2_s3 <= 0.0
        ):
            raise VOFHandoffValidationError(
                "positive carrier turbulent kinetic energy requires positive dissipation"
            )


ConvergenceKind = Literal["mesh", "time_step", "domain", "averaging"]


@dataclass(frozen=True)
class VOFConvergenceStudy:
    """One independently fingerprinted numerical refinement comparison."""

    kind: ConvergenceKind
    baseline_run_fingerprint_sha256: str
    refined_run_fingerprint_sha256: str
    monitored_metric: str
    relative_change: float
    acceptance_tolerance: float
    refinement_ratio: float

    def __post_init__(self) -> None:
        if self.kind not in {"mesh", "time_step", "domain", "averaging"}:
            raise VOFHandoffValidationError(
                "convergence kind must be mesh, time_step, domain, or averaging"
            )
        for name in ("baseline_run_fingerprint_sha256", "refined_run_fingerprint_sha256"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if self.baseline_run_fingerprint_sha256 == self.refined_run_fingerprint_sha256:
            raise VOFHandoffValidationError(
                "baseline and refined convergence runs must have different fingerprints"
            )
        object.__setattr__(self, "monitored_metric", _text("monitored_metric", self.monitored_metric))
        object.__setattr__(self, "relative_change", _nonnegative("relative_change", self.relative_change))
        object.__setattr__(
            self, "acceptance_tolerance", _fraction("acceptance_tolerance", self.acceptance_tolerance)
        )
        ratio = _positive("refinement_ratio", self.refinement_ratio)
        if ratio <= 1.0:
            raise VOFHandoffValidationError("refinement_ratio must be > 1")
        object.__setattr__(self, "refinement_ratio", ratio)

    @property
    def passed(self) -> bool:
        return self.relative_change <= self.acceptance_tolerance


@dataclass(frozen=True)
class VOFConvergenceEvidence:
    """Required mesh, time-step, domain, and averaging refinements."""

    mesh: VOFConvergenceStudy
    time_step: VOFConvergenceStudy
    domain: VOFConvergenceStudy
    averaging: VOFConvergenceStudy

    def __post_init__(self) -> None:
        expected = {
            "mesh": self.mesh,
            "time_step": self.time_step,
            "domain": self.domain,
            "averaging": self.averaging,
        }
        for kind, study in expected.items():
            if not isinstance(study, VOFConvergenceStudy) or study.kind != kind:
                raise VOFHandoffValidationError(
                    f"{kind} convergence evidence must contain a {kind!r} study"
                )

    @property
    def studies(self) -> tuple[VOFConvergenceStudy, ...]:
        return self.mesh, self.time_step, self.domain, self.averaging

    @property
    def passed(self) -> bool:
        return all(study.passed for study in self.studies)


@dataclass(frozen=True)
class VOFHandoffGate:
    """Additional evidence gate, with an immutable evidence fingerprint."""

    name: str
    passed: bool
    detail: str
    evidence_fingerprint_sha256: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text("gate name", self.name))
        object.__setattr__(self, "detail", _text("gate detail", self.detail))
        if not isinstance(self.passed, bool):
            raise VOFHandoffValidationError("gate passed must be boolean")
        if not isinstance(self.required, bool):
            raise VOFHandoffValidationError("gate required must be boolean")
        object.__setattr__(
            self,
            "evidence_fingerprint_sha256",
            _sha256("gate evidence_fingerprint_sha256", self.evidence_fingerprint_sha256),
        )


@dataclass(frozen=True)
class VOFToLagrangianHandoff:
    """Complete, gate-derived bridge from VOF sheet data to parcel geometry."""

    provenance: VOFArtifactProvenance
    averaging_window: VOFAveragingWindow
    extraction: VOFSheetExtractionDefinition
    sheet: VOFSheetStatistics
    liquid_flux: VOFLiquidFluxBalance
    carrier_field: CarrierAxisymmetricFieldEvidence
    convergence: VOFConvergenceEvidence
    declared_gates: tuple[VOFHandoffGate, ...] = ()

    def __post_init__(self) -> None:
        required_types = (
            ("provenance", VOFArtifactProvenance),
            ("averaging_window", VOFAveragingWindow),
            ("extraction", VOFSheetExtractionDefinition),
            ("sheet", VOFSheetStatistics),
            ("liquid_flux", VOFLiquidFluxBalance),
            ("carrier_field", CarrierAxisymmetricFieldEvidence),
            ("convergence", VOFConvergenceEvidence),
        )
        for name, expected_type in required_types:
            if not isinstance(getattr(self, name), expected_type):
                raise VOFHandoffValidationError(
                    f"{name} must be {expected_type.__name__}"
                )
        gates = tuple(self.declared_gates)
        if any(not isinstance(gate, VOFHandoffGate) for gate in gates):
            raise VOFHandoffValidationError(
                "declared_gates must contain only VOFHandoffGate instances"
            )
        names = [gate.name for gate in gates]
        if len(names) != len(set(names)):
            raise VOFHandoffValidationError("declared gate names must be unique")
        reserved = set(self._derived_gate_names())
        collisions = reserved.intersection(names)
        if collisions:
            raise VOFHandoffValidationError(
                "declared gates cannot replace derived gates: " + ", ".join(sorted(collisions))
            )
        object.__setattr__(self, "declared_gates", gates)

    @staticmethod
    def _derived_gate_names() -> tuple[str, ...]:
        return (
            "averaging_window_coverage",
            "sheet_thickness_variation",
            "sheet_velocity_variation",
            "liquid_mass_flux_closure",
            "liquid_momentum_flux_closure",
            "sheet_kinematic_momentum_closure",
            "carrier_field_domain_coverage",
            "mesh_convergence",
            "time_step_convergence",
            "domain_convergence",
            "averaging_convergence",
        )

    def _derived_gate(
        self, name: str, passed: bool, detail: str, payload: Any
    ) -> VOFHandoffGate:
        return VOFHandoffGate(
            name=name,
            passed=bool(passed),
            detail=detail,
            evidence_fingerprint_sha256=_payload_sha256(payload),
            required=True,
        )

    @property
    def kinematic_momentum_relative_error(self) -> float:
        mass = self.liquid_flux.extracted_mass_flow_rate_kg_s
        velocity = self.sheet.velocity_mean_m_s
        expected = (mass * velocity[0], mass * velocity[1])
        return _relative_vector_error(
            self.liquid_flux.extracted_momentum_flux_n, expected
        )

    @property
    def carrier_domain_contains_sheet(self) -> bool:
        x0, x1 = self.carrier_field.axial_bounds_m
        r0, r1 = self.carrier_field.radial_bounds_m
        return (
            x0 <= self.sheet.axial_location_m <= x1
            and r0 <= self.sheet.exit_radius_m <= r1
        )

    @property
    def gates(self) -> tuple[VOFHandoffGate, ...]:
        sheet = self.sheet
        flux = self.liquid_flux
        window = self.averaging_window
        derived = [
            self._derived_gate(
                "averaging_window_coverage",
                window.coverage_passed,
                f"observed {window.observed_flow_through_times:.6g} flow-through times; "
                f"required {window.required_flow_through_times:.6g}",
                {
                    "start_s": window.start_time_s,
                    "end_s": window.end_time_s,
                    "samples": window.sample_count,
                    "observed": window.observed_flow_through_times,
                    "required": window.required_flow_through_times,
                },
            ),
            self._derived_gate(
                "sheet_thickness_variation",
                sheet.thickness_coefficient_of_variation
                <= sheet.maximum_thickness_coefficient_of_variation,
                f"CV={sheet.thickness_coefficient_of_variation:.6g}; limit="
                f"{sheet.maximum_thickness_coefficient_of_variation:.6g}",
                {
                    "mean_m": sheet.full_sheet_thickness_mean_m,
                    "std_m": sheet.full_sheet_thickness_standard_deviation_m,
                    "limit": sheet.maximum_thickness_coefficient_of_variation,
                },
            ),
            self._derived_gate(
                "sheet_velocity_variation",
                sheet.velocity_coefficient_of_variation
                <= sheet.maximum_velocity_coefficient_of_variation,
                f"vector CV={sheet.velocity_coefficient_of_variation:.6g}; limit="
                f"{sheet.maximum_velocity_coefficient_of_variation:.6g}",
                {
                    "mean_m_s": sheet.velocity_mean_m_s,
                    "std_m_s": (
                        sheet.axial_velocity_standard_deviation_m_s,
                        sheet.radial_velocity_standard_deviation_m_s,
                    ),
                    "limit": sheet.maximum_velocity_coefficient_of_variation,
                },
            ),
            self._derived_gate(
                "liquid_mass_flux_closure",
                flux.mass_relative_error <= flux.mass_closure_relative_tolerance,
                f"relative error={flux.mass_relative_error:.6g}; tolerance="
                f"{flux.mass_closure_relative_tolerance:.6g}",
                {
                    "inlet_kg_s": flux.inlet_mass_flow_rate_kg_s,
                    "extracted_kg_s": flux.extracted_mass_flow_rate_kg_s,
                    "tolerance": flux.mass_closure_relative_tolerance,
                },
            ),
            self._derived_gate(
                "liquid_momentum_flux_closure",
                flux.momentum_relative_error
                <= flux.momentum_closure_relative_tolerance,
                f"relative error={flux.momentum_relative_error:.6g}; tolerance="
                f"{flux.momentum_closure_relative_tolerance:.6g}",
                {
                    "inlet_n": flux.inlet_momentum_flux_n,
                    "extracted_n": flux.extracted_momentum_flux_n,
                    "tolerance": flux.momentum_closure_relative_tolerance,
                },
            ),
            self._derived_gate(
                "sheet_kinematic_momentum_closure",
                self.kinematic_momentum_relative_error
                <= flux.kinematic_momentum_relative_tolerance,
                f"relative error={self.kinematic_momentum_relative_error:.6g}; tolerance="
                f"{flux.kinematic_momentum_relative_tolerance:.6g}",
                {
                    "mass_kg_s": flux.extracted_mass_flow_rate_kg_s,
                    "velocity_m_s": sheet.velocity_mean_m_s,
                    "extracted_n": flux.extracted_momentum_flux_n,
                    "tolerance": flux.kinematic_momentum_relative_tolerance,
                },
            ),
            self._derived_gate(
                "carrier_field_domain_coverage",
                self.carrier_domain_contains_sheet,
                "carrier x-r domain contains the sheet handoff ring"
                if self.carrier_domain_contains_sheet
                else "carrier x-r domain does not contain the sheet handoff ring",
                {
                    "field": self.carrier_field.field_fingerprint_sha256,
                    "x_bounds_m": self.carrier_field.axial_bounds_m,
                    "r_bounds_m": self.carrier_field.radial_bounds_m,
                    "sheet_x_m": sheet.axial_location_m,
                    "sheet_r_m": sheet.exit_radius_m,
                },
            ),
        ]
        for study in self.convergence.studies:
            derived.append(
                self._derived_gate(
                    f"{study.kind}_convergence",
                    study.passed,
                    f"{study.monitored_metric}: relative change={study.relative_change:.6g}; "
                    f"tolerance={study.acceptance_tolerance:.6g}",
                    {
                        "kind": study.kind,
                        "baseline": study.baseline_run_fingerprint_sha256,
                        "refined": study.refined_run_fingerprint_sha256,
                        "metric": study.monitored_metric,
                        "change": study.relative_change,
                        "tolerance": study.acceptance_tolerance,
                        "ratio": study.refinement_ratio,
                    },
                )
            )
        return tuple(derived) + self.declared_gates

    @property
    def failed_required_gates(self) -> tuple[VOFHandoffGate, ...]:
        return tuple(gate for gate in self.gates if gate.required and not gate.passed)

    @property
    def ready_for_lagrangian(self) -> bool:
        return not self.failed_required_gates

    def to_radial_sheet_geometry(self) -> RadialSheetGeometry:
        """Return parcel-source geometry only after every required gate passes."""

        failed = self.failed_required_gates
        if failed:
            raise VOFHandoffValidationError(
                "VOF handoff is not eligible for parcel conversion; failed required "
                "gates: " + ", ".join(gate.name for gate in failed)
            )
        return RadialSheetGeometry(
            exit_radius=self.sheet.exit_radius_m,
            # Preserve the full physical sheet/opening thickness convention.
            sheet_thickness=self.sheet.full_sheet_thickness_mean_m,
            axial_location=self.sheet.axial_location_m,
            tip_angle_deg=self.sheet.tip_angle_deg,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, SI-labelled audit representation."""

        return {
            "schema": "raosim.vof_to_lagrangian_handoff.v1",
            "provenance": {
                "case_id": self.provenance.case_id,
                "case_fingerprint_sha256": self.provenance.case_fingerprint_sha256,
                "input_fingerprint_sha256": self.provenance.input_fingerprint_sha256,
                "solver_name": self.provenance.solver_name,
                "solver_version": self.provenance.solver_version,
                "solver_version_fingerprint_sha256": self.provenance.solver_version_fingerprint_sha256,
                "extraction_code_fingerprint_sha256": self.provenance.extraction_code_fingerprint_sha256,
            },
            "averaging_window": {
                "start_time_s": self.averaging_window.start_time_s,
                "end_time_s": self.averaging_window.end_time_s,
                "duration_s": self.averaging_window.duration_s,
                "sample_count": self.averaging_window.sample_count,
                "observed_flow_through_times": self.averaging_window.observed_flow_through_times,
                "required_flow_through_times": self.averaging_window.required_flow_through_times,
            },
            "extraction": {
                "alpha_liquid_threshold": self.extraction.alpha_liquid_threshold,
                "algorithm_id": self.extraction.algorithm_id,
                "algorithm_version_fingerprint_sha256": self.extraction.algorithm_version_fingerprint_sha256,
                "connected_component_policy": self.extraction.connected_component_policy,
                "interpolation_rule": self.extraction.interpolation_rule,
            },
            "sheet": {
                "exit_radius_m": self.sheet.exit_radius_m,
                "axial_location_m": self.sheet.axial_location_m,
                "full_sheet_thickness_mean_m": self.sheet.full_sheet_thickness_mean_m,
                "full_sheet_thickness_standard_deviation_m": self.sheet.full_sheet_thickness_standard_deviation_m,
                "axial_velocity_mean_m_s": self.sheet.axial_velocity_mean_m_s,
                "radial_velocity_mean_m_s": self.sheet.radial_velocity_mean_m_s,
                "axial_velocity_standard_deviation_m_s": self.sheet.axial_velocity_standard_deviation_m_s,
                "radial_velocity_standard_deviation_m_s": self.sheet.radial_velocity_standard_deviation_m_s,
                "tip_angle_deg_derived": self.sheet.tip_angle_deg,
            },
            "liquid_flux": {
                "liquid_name": self.liquid_flux.liquid_name,
                "liquid_density_kg_m3": self.liquid_flux.liquid_density_kg_m3,
                "inlet_mass_flow_rate_kg_s": self.liquid_flux.inlet_mass_flow_rate_kg_s,
                "extracted_mass_flow_rate_kg_s": self.liquid_flux.extracted_mass_flow_rate_kg_s,
                "inlet_momentum_flux_n": list(self.liquid_flux.inlet_momentum_flux_n),
                "extracted_momentum_flux_n": list(self.liquid_flux.extracted_momentum_flux_n),
                "mass_relative_error": self.liquid_flux.mass_relative_error,
                "momentum_relative_error": self.liquid_flux.momentum_relative_error,
                "kinematic_momentum_relative_error": self.kinematic_momentum_relative_error,
            },
            "carrier_field": {
                "operating_point_id": self.carrier_field.operating_point_id,
                "coordinate_system": self.carrier_field.coordinate_system,
                "field_fingerprint_sha256": self.carrier_field.field_fingerprint_sha256,
                "state_fingerprint_sha256": self.carrier_field.state_fingerprint_sha256,
                "fluid_name": self.carrier_field.fluid_name,
                "axial_bounds_m": list(self.carrier_field.axial_bounds_m),
                "radial_bounds_m": list(self.carrier_field.radial_bounds_m),
                "grid_shape_axial_radial": list(self.carrier_field.grid_shape_axial_radial),
                "density_kg_m3": self.carrier_field.density_kg_m3,
                "dynamic_viscosity_pa_s": self.carrier_field.dynamic_viscosity_pa_s,
                "temperature_k": self.carrier_field.temperature_k,
                "pressure_pa": self.carrier_field.pressure_pa,
                "specific_heat_j_kg_k": self.carrier_field.specific_heat_j_kg_k,
                "thermal_conductivity_w_m_k": self.carrier_field.thermal_conductivity_w_m_k,
                "mean_axial_velocity_m_s": self.carrier_field.mean_axial_velocity_m_s,
                "mean_radial_velocity_m_s": self.carrier_field.mean_radial_velocity_m_s,
                "turbulent_kinetic_energy_m2_s2": self.carrier_field.turbulent_kinetic_energy_m2_s2,
                "turbulent_dissipation_rate_m2_s3": self.carrier_field.turbulent_dissipation_rate_m2_s3,
            },
            "convergence": [
                {
                    "kind": study.kind,
                    "baseline_run_fingerprint_sha256": study.baseline_run_fingerprint_sha256,
                    "refined_run_fingerprint_sha256": study.refined_run_fingerprint_sha256,
                    "monitored_metric": study.monitored_metric,
                    "relative_change": study.relative_change,
                    "acceptance_tolerance": study.acceptance_tolerance,
                    "refinement_ratio": study.refinement_ratio,
                    "passed": study.passed,
                }
                for study in self.convergence.studies
            ],
            "gates": [
                {
                    "name": gate.name,
                    "passed": gate.passed,
                    "required": gate.required,
                    "detail": gate.detail,
                    "evidence_fingerprint_sha256": gate.evidence_fingerprint_sha256,
                }
                for gate in self.gates
            ],
            "ready_for_lagrangian": self.ready_for_lagrangian,
        }

    @property
    def contract_fingerprint_sha256(self) -> str:
        return _payload_sha256(self.to_dict())


__all__ = [
    "CarrierAxisymmetricFieldEvidence",
    "VOFArtifactProvenance",
    "VOFAveragingWindow",
    "VOFConvergenceEvidence",
    "VOFConvergenceStudy",
    "VOFHandoffGate",
    "VOFHandoffValidationError",
    "VOFLiquidFluxBalance",
    "VOFSheetExtractionDefinition",
    "VOFSheetStatistics",
    "VOFToLagrangianHandoff",
]
