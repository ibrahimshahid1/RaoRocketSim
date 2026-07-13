"""Geometry-dispatched, deterministic primary parcel source initialization.

This module distinguishes the movable-pintle *radial liquid sheet* used by the
Radhakrishnan Lagrangian/WAVE workflow from the injector geometries currently
implemented in :mod:`raosim.injector`.  Only that radial-sheet geometry is
declared eligible for the literature-calibrated primary path.  Axial annuli,
rectangular slots, and round holes can be converted into geometric source blobs
for downstream drag/secondary-breakup studies, but this module deliberately
does not invent LISA, slot-sheet, or round-jet primary-breakup correlations for
them.

Coordinates are Cartesian SI: ``x`` is the engine/chamber axis and ``y,z`` span
the radial plane.  For radial-sheet tip angle ``alpha`` the convention is

``alpha = 0 deg``: purely radial, perpendicular to the axial carrier gas;
``u_x = U sin(alpha)`` and ``u_r = U cos(alpha)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np

from .types import (
    LiquidProperties,
    ParcelCloud,
    SprayValidationError,
    _readonly_array,
)


_SECONDARY_ONLY_STATUS = "secondary_only_unvalidated_primary"
_RADIAL_WAVE_STATUS = "literature_calibrated_wave_primary"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _sheet_thickness_method(
    method: str | None,
    *,
    source: str,
) -> Literal["vof", "measured"]:
    """Normalize independently resolved sheet-thickness evidence.

    The movable-injector result predates an explicit ``method`` field.  For
    that result only, a source label which explicitly says VOF or measured is
    also an unambiguous method declaration.  Hydraulic/continuity estimates
    and primary-breakup correlations are deliberately not admitted.
    """

    source_text = str(source).strip()
    source_key = source_text.casefold().replace("-", "_")
    disallowed = (
        "continuity_equivalent",
        "continuity equivalent",
        "hydraulic_equivalent",
        "hydraulic equivalent",
        "correlation",
        "screening",
        "screen_only",
        "screen only",
        "mechanical opening as",
    )
    if any(token in source_key for token in disallowed):
        raise SprayValidationError(
            "sheet-thickness evidence must be VOF-resolved or measured; "
            "continuity-equivalent, hydraulic, correlation, and screening "
            "estimates are not a liquid-sheet handoff"
        )

    if method is not None:
        method_key = str(method).strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "vof": "vof",
            "eulerian_vof": "vof",
            "volume_of_fluid": "vof",
            "measured": "measured",
            "measurement": "measured",
            "experimental": "measured",
            "experiment": "measured",
            "optical_measurement": "measured",
        }
        normalized = aliases.get(method_key)
        if normalized is None:
            raise SprayValidationError(
                "sheet_thickness_method must declare VOF or measured evidence; "
                f"got {method!r}"
            )
        return normalized

    if "vof" in source_key or "volume of fluid" in source_key:
        return "vof"
    if any(
        token in source_key
        for token in (
            "measured",
            "measurement",
            "experimental",
            "experiment",
            "optical",
            "shadowgraph",
            "x_ray",
        )
    ):
        return "measured"
    raise SprayValidationError(
        "sheet-thickness evidence is missing an explicit method: declare VOF "
        "or measured in sheet_thickness_method or sheet_thickness_source"
    )


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SprayValidationError(f"{name} must be finite and > 0")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SprayValidationError(f"{name} must be finite")
    return value


def _count(name: str, value: int, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise SprayValidationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _angle(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0.0 <= value <= 90.0:
        raise SprayValidationError(f"{name} must be in [0, 90] deg")
    return value


@dataclass(frozen=True)
class RadialSheetGeometry:
    """Movable-pintle radial-sheet source used by Radhakrishnan et al.

    ``sheet_thickness`` is the full, resolved physical liquid-sheet thickness,
    not a half-gap, hydraulic diameter, or mechanical pintle opening.  A
    mechanical opening may be recorded separately as
    ``mechanical_opening_distance``.  The source ring lies at ``exit_radius``
    and ``axial_location``.

    The first four fields retain the original positional constructor.  The
    evidence fields are optional so independently constructed literature/VOF
    contracts remain source compatible; the injector-result adapter below is
    stricter and always populates all of them.
    """

    exit_radius: float
    sheet_thickness: float
    axial_location: float
    tip_angle_deg: float
    mechanical_opening_distance: float | None = None
    sheet_thickness_method: str | None = None
    sheet_thickness_source: str | None = None
    sheet_thickness_artifact_sha256: str | None = None

    injection_form: ClassVar[str] = "movable_pintle_radial_sheet"
    model_id: ClassVar[str] = "radhakrishnan_radial_sheet_wave_primary"
    applicability_status: ClassVar[str] = _RADIAL_WAVE_STATUS
    primary_path_eligible: ClassVar[bool] = True
    provenance: ClassVar[str] = (
        "Radhakrishnan, Lee & Koo (2021), detailed LOX/GCH4 variable-area "
        "pintle spray model; radial sheet handed to the Lagrangian WAVE path"
    )
    local_source: ClassVar[str] = (
        "propulsion_texts/pintle_injector/radhakrishnan2021.pdf"
    )
    applicability: ClassVar[str] = (
        "subcritical liquid radial sheet from a movable-pintle opening; "
        "literature calibration and downstream WAVE settings must match the case"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "exit_radius", _positive("exit_radius", self.exit_radius))
        object.__setattr__(
            self, "sheet_thickness", _positive("sheet_thickness", self.sheet_thickness)
        )
        object.__setattr__(
            self, "axial_location", _finite("axial_location", self.axial_location)
        )
        object.__setattr__(self, "tip_angle_deg", _angle("tip_angle_deg", self.tip_angle_deg))
        if self.mechanical_opening_distance is not None:
            object.__setattr__(
                self,
                "mechanical_opening_distance",
                _positive(
                    "mechanical_opening_distance",
                    self.mechanical_opening_distance,
                ),
            )

        evidence = (
            self.sheet_thickness_method,
            self.sheet_thickness_source,
            self.sheet_thickness_artifact_sha256,
        )
        if any(value is not None for value in evidence):
            if not all(value is not None for value in evidence):
                raise SprayValidationError(
                    "sheet-thickness evidence requires method, source, and "
                    "artifact SHA-256 together"
                )
            source = str(self.sheet_thickness_source).strip()
            if not source:
                raise SprayValidationError("sheet_thickness_source must be nonblank")
            method = _sheet_thickness_method(
                str(self.sheet_thickness_method), source=source
            )
            digest = str(self.sheet_thickness_artifact_sha256).strip().lower()
            if _SHA256_PATTERN.fullmatch(digest) is None:
                raise SprayValidationError(
                    "sheet_thickness_artifact_sha256 must contain 64 hexadecimal characters"
                )
            object.__setattr__(self, "sheet_thickness_method", method)
            object.__setattr__(self, "sheet_thickness_source", source)
            object.__setattr__(self, "sheet_thickness_artifact_sha256", digest)
        if self.sheet_thickness >= 2.0 * self.exit_radius:
            raise SprayValidationError(
                "radial sheet thickness must be smaller than its exit diameter"
            )

    @property
    def opening_distance(self) -> float:
        """Mechanical pintle opening, with a legacy constructor fallback.

        New injector-result handoffs always provide the explicit mechanical
        opening.  When a caller uses the original four-field constructor there
        is no independent opening datum, so this compatibility property retains
        its historical sheet-thickness value.  Physics code should use
        ``mechanical_opening_distance`` and ``sheet_thickness`` directly.
        """

        if self.mechanical_opening_distance is not None:
            return self.mechanical_opening_distance
        return self.sheet_thickness

    @property
    def opening_distance_basis(self) -> str:
        """State whether ``opening_distance`` is explicit or a legacy alias."""

        if self.mechanical_opening_distance is not None:
            return "explicit_mechanical_pintle_opening"
        return "legacy_sheet_thickness_alias_no_mechanical_opening_evidence"

    @property
    def initial_parcel_diameter(self) -> float:
        """Full resolved liquid-sheet thickness, never the mechanical opening."""

        return self.sheet_thickness


def radial_sheet_geometry_from_injector_result(
    result: Any,
    *,
    axial_location: float = 0.0,
) -> RadialSheetGeometry:
    """Build the movable radial-sheet source from a solved injector result.

    This is intentionally a fail-closed boundary.  Only the Son continuous-gap
    architecture is accepted, the injector solve must be feasible, its
    dedicated sheet-handoff gate must pass, and the resolved thickness must
    carry VOF or measurement provenance plus a configuration artifact digest.
    The continuity-equivalent thickness retained in ``slots.detail`` is a
    hydraulic screen and is never used here.
    """

    if getattr(result, "architecture", None) != "son_continuous_movable":
        raise SprayValidationError(
            "injector spray handoff requires architecture='son_continuous_movable'"
        )
    if getattr(result, "feasible", None) is not True:
        raise SprayValidationError(
            "injector spray handoff requires a feasible injector design result"
        )

    slots = getattr(result, "slots", None)
    if slots is None or getattr(slots, "geometry", None) != "continuous_radial_gap":
        raise SprayValidationError(
            "injector spray handoff requires slots.geometry='continuous_radial_gap'"
        )
    detail = getattr(slots, "detail", None)
    if not isinstance(detail, Mapping):
        raise SprayValidationError(
            "continuous radial-gap result is missing its geometry detail ledger"
        )
    if detail.get("injection_form") != "movable_pintle_radial_sheet":
        raise SprayValidationError(
            "continuous radial-gap result has no movable-pintle radial-sheet form"
        )

    actuation = getattr(result, "actuation", None)
    if actuation is None:
        raise SprayValidationError(
            "continuous radial-gap result is missing its actuation ledger"
        )

    gate_status = {
        getattr(gate, "name", None): getattr(gate, "status", None)
        for gate in tuple(getattr(result, "gates", ()) or ())
    }.get("movable_sheet_thickness_handoff")
    if gate_status != "pass":
        raise SprayValidationError(
            "injector movable_sheet_thickness_handoff gate must be present and pass"
        )

    thickness = getattr(actuation, "sheet_thickness", None)
    source = getattr(actuation, "sheet_thickness_source", None)
    digest = getattr(actuation, "sheet_thickness_artifact_sha256", None)
    if thickness is None or source is None or digest is None:
        raise SprayValidationError(
            "actuation sheet-thickness evidence requires thickness, source, and "
            "artifact SHA-256"
        )
    source = str(source).strip()
    if not source:
        raise SprayValidationError("actuation sheet_thickness_source must be nonblank")

    detail_thickness = detail.get("sheet_thickness")
    detail_source = detail.get("sheet_thickness_source")
    detail_digest = detail.get("sheet_thickness_artifact_sha256")
    evidence_pairs = (
        (
            "fluid name",
            detail.get("sheet_thickness_fluid_name"),
            getattr(actuation, "sheet_thickness_fluid_name", None),
        ),
        (
            "geometry fingerprint",
            detail.get("sheet_thickness_geometry_fingerprint_sha256"),
            getattr(
                actuation,
                "sheet_thickness_geometry_fingerprint_sha256",
                None,
            ),
        ),
        (
            "resolved geometry fingerprint",
            detail.get("resolved_geometry_fingerprint_sha256"),
            getattr(actuation, "resolved_geometry_fingerprint_sha256", None),
        ),
        (
            "opening range",
            detail.get("sheet_thickness_opening_range"),
            getattr(actuation, "sheet_thickness_opening_range", None),
        ),
        (
            "pressure-drop range",
            detail.get("sheet_thickness_pressure_drop_range"),
            getattr(actuation, "sheet_thickness_pressure_drop_range", None),
        ),
        (
            "mass-flow range",
            detail.get("sheet_thickness_mass_flow_range"),
            getattr(actuation, "sheet_thickness_mass_flow_range", None),
        ),
    )
    if (
        detail_thickness is None
        or detail_source != getattr(actuation, "sheet_thickness_source", None)
        or detail_digest != getattr(actuation, "sheet_thickness_artifact_sha256", None)
        or not math.isclose(
            float(detail_thickness),
            float(thickness),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ):
        raise SprayValidationError(
            "actuation and radial-gap detail ledgers disagree on sheet-thickness evidence"
        )
    for label, detail_value, actuation_value in evidence_pairs:
        if isinstance(detail_value, (tuple, list)) and isinstance(
            actuation_value, (tuple, list)
        ):
            disagree = tuple(detail_value) != tuple(actuation_value)
        else:
            disagree = detail_value != actuation_value
        if detail_value is None or actuation_value is None or disagree:
            raise SprayValidationError(
                "actuation and radial-gap detail ledgers disagree on "
                f"sheet-thickness {label}"
            )

    method_candidates = (
        getattr(actuation, "sheet_thickness_method", None),
        detail.get("sheet_thickness_method"),
    )
    explicit_methods = tuple(
        value for value in method_candidates if value is not None
    )
    if len(explicit_methods) == 2:
        first = _sheet_thickness_method(explicit_methods[0], source=source)
        second = _sheet_thickness_method(explicit_methods[1], source=source)
        if first != second:
            raise SprayValidationError(
                "actuation and radial-gap detail ledgers disagree on sheet-thickness method"
            )
        method: str | None = first
    else:
        method = explicit_methods[0] if explicit_methods else None
    normalized_method = _sheet_thickness_method(method, source=source)

    digest_text = str(digest).strip().lower()
    if _SHA256_PATTERN.fullmatch(digest_text) is None:
        raise SprayValidationError(
            "sheet_thickness_artifact_sha256 must contain 64 hexadecimal characters"
        )

    opening = getattr(actuation, "opening_distance", None)
    post_diameter = detail.get("post_diameter")
    tip_angle = detail.get("tip_angle_deg")
    if opening is None or post_diameter is None or tip_angle is None:
        raise SprayValidationError(
            "continuous radial-gap result is missing opening, post diameter, or tip angle"
        )

    role = getattr(slots, "role", None)
    feed_map = getattr(result, "feed", None)
    if not isinstance(feed_map, Mapping) or role not in feed_map:
        raise SprayValidationError(
            "continuous radial-gap result is missing its radial feed-state ledger"
        )
    feed_state = feed_map[role]
    if not (
        getattr(feed_state, "liquid_ok", False)
        and getattr(feed_state, "phase", None) == "liquid"
    ):
        raise SprayValidationError(
            "movable radial-sheet handoff requires a resolved liquid feed state"
        )

    def fluid_key(value: Any) -> str:
        return "".join(
            character
            for character in str(value or "").casefold()
            if character.isalnum()
        )

    evidence_fluid = getattr(actuation, "sheet_thickness_fluid_name", None)
    if not evidence_fluid or fluid_key(evidence_fluid) != fluid_key(
        getattr(feed_state, "name", None)
    ):
        raise SprayValidationError(
            "sheet-thickness evidence fluid does not match the solved radial stream"
        )
    if (
        getattr(
            actuation,
            "sheet_thickness_geometry_fingerprint_sha256",
            None,
        )
        != getattr(actuation, "resolved_geometry_fingerprint_sha256", None)
    ):
        raise SprayValidationError(
            "sheet-thickness evidence geometry fingerprint does not match the "
            "solved movable-pintle geometry"
        )
    operating_values = (
        (
            "opening",
            float(opening),
            getattr(actuation, "sheet_thickness_opening_range", None),
        ),
        (
            "pressure drop",
            float(getattr(slots, "dp", float("nan"))),
            getattr(actuation, "sheet_thickness_pressure_drop_range", None),
        ),
        (
            "mass flow",
            float(getattr(slots, "mdot", float("nan"))),
            getattr(actuation, "sheet_thickness_mass_flow_range", None),
        ),
    )
    for label, value, validity in operating_values:
        if (
            validity is None
            or len(validity) != 2
            or not math.isfinite(value)
            or not float(validity[0]) <= value <= float(validity[1])
        ):
            raise SprayValidationError(
                f"solved sheet {label} lies outside the evidence validity range"
            )

    return RadialSheetGeometry(
        exit_radius=0.5 * float(post_diameter),
        sheet_thickness=float(thickness),
        axial_location=axial_location,
        tip_angle_deg=float(tip_angle),
        mechanical_opening_distance=float(opening),
        sheet_thickness_method=normalized_method,
        sheet_thickness_source=source,
        sheet_thickness_artifact_sha256=digest_text,
    )


@dataclass(frozen=True)
class AxialAnnularSheetGeometry:
    """Current injector's downstream axial annular liquid sheet."""

    inner_radius: float
    outer_radius: float
    axial_location: float

    injection_form: ClassVar[str] = "continuous_annular_sheet"
    model_id: ClassVar[str] = "axial_annular_geometric_blob_source"
    applicability_status: ClassVar[str] = _SECONDARY_ONLY_STATUS
    primary_path_eligible: ClassVar[bool] = False
    provenance: ClassVar[str] = (
        "repository geometric source mapping from the solved axial annulus; "
        "no primary annular-sheet breakup correlation selected"
    )
    local_source: ClassVar[str] = "raosim/injector.py"
    applicability: ClassVar[str] = (
        "geometric initialization for downstream secondary-breakup diagnostics only"
    )

    def __post_init__(self) -> None:
        inner = _positive("inner_radius", self.inner_radius)
        outer = _positive("outer_radius", self.outer_radius)
        if outer <= inner:
            raise SprayValidationError("outer_radius must be greater than inner_radius")
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "outer_radius", outer)
        object.__setattr__(
            self, "axial_location", _finite("axial_location", self.axial_location)
        )

    @property
    def mean_radius(self) -> float:
        return 0.5 * (self.inner_radius + self.outer_radius)

    @property
    def sheet_thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def initial_parcel_diameter(self) -> float:
        return self.sheet_thickness


@dataclass(frozen=True)
class PlanarSlotJetGeometry:
    """Current injector's discrete rectangular radial slot sources."""

    slot_count: int
    exit_radius: float
    slot_width: float
    slot_height: float
    slot_length: float
    axial_location: float
    cant_angle_deg: float = 0.0

    injection_form: ClassVar[str] = "planar_slot_jet"
    model_id: ClassVar[str] = "planar_slot_geometric_blob_source"
    applicability_status: ClassVar[str] = _SECONDARY_ONLY_STATUS
    primary_path_eligible: ClassVar[bool] = False
    provenance: ClassVar[str] = (
        "repository geometric source mapping from solved rectangular pintle slots; "
        "no slot-jet primary-breakup correlation selected"
    )
    local_source: ClassVar[str] = "raosim/injector.py"
    applicability: ClassVar[str] = (
        "geometric hydraulic-diameter blobs for secondary-breakup diagnostics only"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_count", _count("slot_count", self.slot_count))
        for name in ("exit_radius", "slot_width", "slot_height", "slot_length"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(
            self, "axial_location", _finite("axial_location", self.axial_location)
        )
        object.__setattr__(
            self, "cant_angle_deg", _angle("cant_angle_deg", self.cant_angle_deg)
        )

    @property
    def hydraulic_diameter(self) -> float:
        return 2.0 * self.slot_width * self.slot_height / (
            self.slot_width + self.slot_height
        )

    @property
    def initial_parcel_diameter(self) -> float:
        # This is an unbroken geometric blob scale, not a primary-breakup result.
        return self.hydraulic_diameter


@dataclass(frozen=True)
class RoundHoleJetGeometry:
    """Current injector's discrete cylindrical radial-hole sources."""

    hole_count: int
    exit_radius: float
    hole_diameter: float
    hole_length: float
    axial_location: float
    cant_angle_deg: float = 0.0

    injection_form: ClassVar[str] = "round_hole_jet"
    model_id: ClassVar[str] = "round_hole_geometric_blob_source"
    applicability_status: ClassVar[str] = _SECONDARY_ONLY_STATUS
    primary_path_eligible: ClassVar[bool] = False
    provenance: ClassVar[str] = (
        "repository geometric source mapping from solved round pintle holes; "
        "no round-jet primary-breakup correlation selected"
    )
    local_source: ClassVar[str] = "raosim/injector.py"
    applicability: ClassVar[str] = (
        "orifice-diameter blobs for secondary-breakup diagnostics only"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "hole_count", _count("hole_count", self.hole_count))
        for name in ("exit_radius", "hole_diameter", "hole_length"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(
            self, "axial_location", _finite("axial_location", self.axial_location)
        )
        object.__setattr__(
            self, "cant_angle_deg", _angle("cant_angle_deg", self.cant_angle_deg)
        )

    @property
    def initial_parcel_diameter(self) -> float:
        # Blob injection at the as-machined orifice scale; no primary prediction.
        return self.hole_diameter


PrimaryGeometry: TypeAlias = (
    RadialSheetGeometry
    | AxialAnnularSheetGeometry
    | PlanarSlotJetGeometry
    | RoundHoleJetGeometry
)


@dataclass(frozen=True)
class PrimaryModelMetadata:
    model_id: str
    injection_form: str
    applicability_status: str
    provenance: str
    local_source: str
    applicability: str
    initialization_diameter_basis: str
    primary_path_eligible: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "model_id": self.model_id,
            "injection_form": self.injection_form,
            "applicability_status": self.applicability_status,
            "provenance": self.provenance,
            "local_source": self.local_source,
            "applicability": self.applicability,
            "initialization_diameter_basis": self.initialization_diameter_basis,
            "primary_path_eligible": self.primary_path_eligible,
        }


@dataclass(frozen=True)
class PrimaryGate:
    name: str
    status: Literal["pass", "warn", "fail", "info"]
    detail: str

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.detail).strip():
            raise SprayValidationError("primary gate name and detail must be nonblank")
        if self.status not in {"pass", "warn", "fail", "info"}:
            raise SprayValidationError("invalid primary gate status")

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "info"}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class PrimaryParcelInitialization:
    """One role's mass- and momentum-closed deterministic parcel source."""

    cloud: ParcelCloud
    role: str
    liquid: LiquidProperties
    mass_flow_rate: float
    injection_duration: float
    injected_mass: float
    injected_momentum: np.ndarray
    model: PrimaryModelMetadata
    gates: tuple[PrimaryGate, ...]

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if not role or any(item != role for item in self.cloud.roles):
            raise SprayValidationError(
                "primary result role must be nonblank and match every parcel role"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "mass_flow_rate", _positive("mass_flow_rate", self.mass_flow_rate)
        )
        object.__setattr__(
            self,
            "injection_duration",
            _positive("injection_duration", self.injection_duration),
        )
        object.__setattr__(self, "injected_mass", _positive("injected_mass", self.injected_mass))
        momentum = _readonly_array(
            self.injected_momentum,
            name="injected_momentum",
            dtype=float,
            ndim=1,
        )
        if momentum.shape != (3,):
            raise SprayValidationError("injected_momentum must have shape (3,)")
        object.__setattr__(self, "injected_momentum", momentum)
        object.__setattr__(self, "gates", tuple(self.gates))

    @property
    def primary_path_eligible(self) -> bool:
        """Whether this source may enter its primary model, not the engine cycle."""

        return self.model.primary_path_eligible and not any(
            gate.status == "fail" for gate in self.gates
        )

    @property
    def represented_mass(self) -> float:
        return math.fsum(
            self.cloud.represented_liquid_mass(self.liquid.density).tolist()
        )

    @property
    def relative_mass_residual(self) -> float:
        return (self.represented_mass - self.injected_mass) / self.injected_mass

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "mass_flow_rate_kg_s": self.mass_flow_rate,
            "injection_duration_s": self.injection_duration,
            "injected_mass_kg": self.injected_mass,
            "represented_mass_kg": self.represented_mass,
            "relative_mass_residual": self.relative_mass_residual,
            "injected_momentum_kg_m_s": self.injected_momentum.tolist(),
            "parcel_count": self.cloud.count,
            "primary_path_eligible": self.primary_path_eligible,
            "model": self.model.to_dict(),
            "gates": [gate.to_dict() for gate in self.gates],
        }


def _continuous_azimuthal_vectors(parcel_count: int) -> np.ndarray:
    """Evenly stratified, explicitly antipodal radial unit vectors."""

    n = _count("parcel_count", parcel_count, minimum=4)
    if n % 2:
        raise SprayValidationError(
            "continuous-sheet parcel_count must be even for exact antipodal symmetry"
        )
    half = n // 2
    theta = (np.arange(half, dtype=float) + 0.5) * math.pi / half
    base = np.column_stack((np.cos(theta), np.sin(theta)))
    vectors = np.empty((n, 2), dtype=float)
    vectors[0::2] = base
    vectors[1::2] = -base
    return vectors


def _element_vectors(element_count: int, parcels_per_element: int) -> np.ndarray:
    """Deterministically repeat uniformly spaced discrete opening directions."""

    count = _count("element_count", element_count)
    per = _count("parcels_per_element", parcels_per_element)
    if count % 2 == 0:
        half = count // 2
        theta = np.arange(half, dtype=float) * 2.0 * math.pi / count
        base = np.column_stack((np.cos(theta), np.sin(theta)))
        elements = np.empty((count, 2), dtype=float)
        elements[0::2] = base
        elements[1::2] = -base
    else:
        theta = np.arange(count, dtype=float) * 2.0 * math.pi / count
        elements = np.column_stack((np.cos(theta), np.sin(theta)))
    return np.repeat(elements, per, axis=0)


def _metadata(geometry: PrimaryGeometry) -> PrimaryModelMetadata:
    if isinstance(geometry, RadialSheetGeometry):
        basis = "full_movable_pintle_radial_sheet_thickness"
    elif isinstance(geometry, AxialAnnularSheetGeometry):
        basis = "full_axial_annular_gap_geometric_blob_not_primary_breakup"
    elif isinstance(geometry, PlanarSlotJetGeometry):
        basis = "rectangular_slot_hydraulic_diameter_geometric_blob_not_primary_breakup"
    else:
        basis = "round_hole_diameter_geometric_blob_not_primary_breakup"
    return PrimaryModelMetadata(
        model_id=geometry.model_id,
        injection_form=geometry.injection_form,
        applicability_status=geometry.applicability_status,
        provenance=geometry.provenance,
        local_source=geometry.local_source,
        applicability=geometry.applicability,
        initialization_diameter_basis=basis,
        primary_path_eligible=geometry.primary_path_eligible,
    )


def _gates(metadata: PrimaryModelMetadata) -> tuple[PrimaryGate, ...]:
    geometry_gate = PrimaryGate(
        "primary_source_geometry",
        "pass",
        f"explicit {metadata.injection_form} geometry dispatched without fallback",
    )
    if metadata.primary_path_eligible:
        return (
            geometry_gate,
            PrimaryGate(
                "primary_model_applicability",
                "pass",
                "radial sheet is the only implemented geometry admitted to the "
                "Radhakrishnan-calibrated WAVE primary path",
            ),
            PrimaryGate(
                "primary_cycle_coupling",
                "pass",
                "primary geometry is coupling-capable; downstream breakup, "
                "conservation, and benchmark gates must still pass",
            ),
        )
    return (
        geometry_gate,
        PrimaryGate(
            "primary_model_applicability",
            "warn",
            f"{metadata.injection_form} is tagged {_SECONDARY_ONLY_STATUS}; "
            "only geometric blobs are initialized",
        ),
        PrimaryGate(
            "primary_cycle_coupling",
            "fail",
            "cycle coupling is blocked because no validated geometry-specific "
            "primary-breakup model is implemented",
        ),
    )


def initialize_primary_parcels(
    geometry: PrimaryGeometry,
    *,
    role: str,
    liquid: LiquidProperties,
    mass_flow_rate: float,
    injection_velocity: float,
    injection_duration: float,
    parcel_count: int,
) -> PrimaryParcelInitialization:
    """Create a deterministic, weighted parcel source for one injector stream.

    Statistical multiplicity is solved from the physical source mass so the
    represented liquid mass closes to ``mass_flow_rate * injection_duration``.
    No random numbers are used.
    """

    if not isinstance(
        geometry,
        (
            RadialSheetGeometry,
            AxialAnnularSheetGeometry,
            PlanarSlotJetGeometry,
            RoundHoleJetGeometry,
        ),
    ):
        raise SprayValidationError(
            f"unsupported primary geometry type {type(geometry).__name__}"
        )
    role = str(role).strip()
    if not role:
        raise SprayValidationError("role must be nonblank")
    mdot = _positive("mass_flow_rate", mass_flow_rate)
    speed = _positive("injection_velocity", injection_velocity)
    duration = _positive("injection_duration", injection_duration)
    n = _count("parcel_count", parcel_count)

    if isinstance(geometry, (RadialSheetGeometry, AxialAnnularSheetGeometry)):
        radial_vectors = _continuous_azimuthal_vectors(n)
    else:
        element_count = (
            geometry.slot_count
            if isinstance(geometry, PlanarSlotJetGeometry)
            else geometry.hole_count
        )
        if n % element_count:
            raise SprayValidationError(
                f"parcel_count={n} must be divisible by {element_count} discrete openings"
            )
        radial_vectors = _element_vectors(element_count, n // element_count)

    if isinstance(geometry, AxialAnnularSheetGeometry):
        source_radius = geometry.mean_radius
        axial_fraction = 1.0
        radial_fraction = 0.0
        diameter = geometry.initial_parcel_diameter
    else:
        source_radius = geometry.exit_radius
        angle_deg = (
            geometry.tip_angle_deg
            if isinstance(geometry, RadialSheetGeometry)
            else geometry.cant_angle_deg
        )
        angle = math.radians(angle_deg)
        axial_fraction = math.sin(angle)
        radial_fraction = math.cos(angle)
        diameter = geometry.initial_parcel_diameter

    position = np.empty((n, 3), dtype=float)
    position[:, 0] = geometry.axial_location
    position[:, 1:] = source_radius * radial_vectors
    velocity = np.empty((n, 3), dtype=float)
    velocity[:, 0] = speed * axial_fraction
    velocity[:, 1:] = speed * radial_fraction * radial_vectors

    target_mass = mdot * duration
    physical_drop_mass = liquid.density * math.pi / 6.0 * diameter**3
    multiplicity = target_mass / (n * physical_drop_mass)
    cloud = ParcelCloud(
        position=position,
        velocity=velocity,
        diameter=np.full(n, diameter),
        temperature=np.full(n, liquid.temperature),
        statistical_weight=np.full(n, multiplicity),
        roles=(role,) * n,
    )
    represented = cloud.represented_liquid_mass(liquid.density)
    represented_mass = math.fsum(represented.tolist())
    tolerance = 64.0 * np.finfo(float).eps * max(
        target_mass, np.finfo(float).tiny
    )
    if abs(represented_mass - target_mass) > tolerance:
        raise RuntimeError(
            "primary parcel multiplicity failed mass closure: "
            f"represented={represented_mass:.17g}, target={target_mass:.17g}"
        )
    momentum = np.array([
        math.fsum((represented * velocity[:, component]).tolist())
        for component in range(3)
    ])
    metadata = _metadata(geometry)
    return PrimaryParcelInitialization(
        cloud=cloud,
        role=role,
        liquid=liquid,
        mass_flow_rate=mdot,
        injection_duration=duration,
        injected_mass=target_mass,
        injected_momentum=momentum,
        model=metadata,
        gates=_gates(metadata),
    )


__all__ = [
    "AxialAnnularSheetGeometry",
    "PlanarSlotJetGeometry",
    "PrimaryGate",
    "PrimaryGeometry",
    "PrimaryModelMetadata",
    "PrimaryParcelInitialization",
    "RadialSheetGeometry",
    "RoundHoleJetGeometry",
    "initialize_primary_parcels",
    "radial_sheet_geometry_from_injector_result",
]
