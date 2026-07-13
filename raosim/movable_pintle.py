"""Literature-routed geometry and static actuation for a movable pintle.

This module implements the *continuous radial gap* geometry in Son et al.,
``Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines``
(JPP 33(4), 2017, DOI 10.2514/1.B36301).  It is intentionally separate from
the repository's fixed radial slots and holes.

The controlling tip/post area is Son Eq. (1), evaluated in its numerically
stable expanded form.  It is capped by the fixed centre-gap annulus.  Once the
tip area reaches that cap, pintle travel no longer controls the minimum flow
area; the paper reports an abrupt flow/spray transition there.  Design solves
therefore stay below a declared fraction of the transition area.

No universal discharge-coefficient-versus-stroke or actuator-force law is
invented.  A user calibration may be interpolated versus opening fraction only
when its artifact hash, exact Son-geometry fingerprint, fluid, and operating
domain match; otherwise the caller's constant Cd is retained and explicitly
labelled uncalibrated.  Static actuation uses only user-declared unbalanced
pressure area, spring, seal friction, moving mass/acceleration, stem area, and
capacity. Dynamic response, seal life, thermal growth, and qualification are
outside this preliminary model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable


SON2017_MODEL_ID = "son2017_continuous_radial_gap"


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


@dataclass
class MovablePintleSpec:
    """Geometry, calibration, stops, and static actuator inputs.

    Diameters and openings are metres, areas are square metres, forces are
    newtons, acceleration is m/s^2, stress is pascals, and Reynolds number is
    dimensionless.  ``maximum_opening=None`` derives an open stop at
    ``transition_area_fraction`` of the centre-gap transition.
    """

    post_diameter: float | None = None
    post_thickness: float | None = None
    center_gap_diameter: float | None = None
    pintle_rod_diameter: float | None = None
    maximum_opening: float | None = None
    commanded_opening: float | None = None
    transition_area_fraction: float = 0.95
    minimum_uniform_sheet_opening: float = 1.0e-4

    # (opening / maximum_opening, Cd), sorted from 0 to 1.  Son Eq. (3)
    # defines how Cd is measured but does not supply a universal curve.
    cd_vs_opening_fraction: tuple[tuple[float, float], ...] = ()
    cd_calibration_source: str | None = None
    cd_calibration_artifact_sha256: str | None = None
    cd_geometry_fingerprint_sha256: str | None = None
    cd_reynolds_range: tuple[float, float] | None = None
    cd_pressure_drop_range: tuple[float, float] | None = None
    cd_temperature_range: tuple[float, float] | None = None
    cd_cavitation_number_range: tuple[float, float] | None = None
    cd_fluid_name: str | None = None

    # Position metrology and shutoff evidence.
    position_tolerance: float | None = None
    position_feedback_resolution: float | None = None
    backlash: float | None = None
    closed_leakage_area: float | None = None
    metrology_source: str | None = None
    metrology_artifact_sha256: str | None = None
    leakage_source: str | None = None
    leakage_artifact_sha256: str | None = None

    # Static force ledger.  Pressure balance is never inferred: callers must
    # supply the net projected area on which manifold-to-chamber dP acts.
    unbalanced_pressure_area: float | None = None
    spring_preload_force: float = 0.0
    seal_friction_force: float | None = None
    moving_mass: float | None = None
    maximum_acceleration: float | None = None
    actuator_force_capacity: float | None = None
    force_safety_factor: float = 1.5
    stem_diameter: float | None = None
    stem_allowable_stress: float | None = None
    actuator_source: str | None = None
    actuator_artifact_sha256: str | None = None

    # The mechanical opening is not the liquid-sheet thickness.  These fields
    # admit only separately measured/VOF evidence for a parcel handoff.
    sheet_thickness: float | None = None
    sheet_thickness_method: str | None = None
    sheet_thickness_source: str | None = None
    sheet_thickness_artifact_sha256: str | None = None
    sheet_thickness_geometry_fingerprint_sha256: str | None = None
    sheet_thickness_fluid_name: str | None = None
    sheet_thickness_opening_range: tuple[float, float] | None = None
    sheet_thickness_pressure_drop_range: tuple[float, float] | None = None
    sheet_thickness_mass_flow_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class MovablePintleActuation:
    """Resolved geometry, metering, and static force ledger."""

    model_id: str
    opening_distance: float
    minimum_opening_distance: float
    maximum_opening: float
    opening_fraction: float
    transition_opening: float
    tip_minimum_area: float
    center_gap_area: float
    effective_metering_area: float
    transition_area_fraction: float
    transition_margin: float
    discharge_coefficient: float
    discharge_coefficient_model: str
    discharge_coefficient_source: str | None
    discharge_coefficient_artifact_sha256: str | None
    discharge_coefficient_geometry_fingerprint_sha256: str | None
    resolved_geometry_fingerprint_sha256: str
    calibration_reynolds_range: tuple[float, float] | None
    calibration_pressure_drop_range: tuple[float, float] | None
    calibration_temperature_range: tuple[float, float] | None
    calibration_cavitation_number_range: tuple[float, float] | None
    calibration_fluid_name: str | None
    position_uncertainty_fraction: float | None
    metrology_source: str | None
    metrology_artifact_sha256: str | None
    pressure_force: float | None
    momentum_reaction_force: float
    spring_preload_force: float
    seal_friction_force: float | None
    inertia_force: float | None
    required_actuator_force: float | None
    actuator_force_capacity: float | None
    actuator_force_margin: float | None
    stem_axial_stress: float | None
    stem_allowable_stress: float | None
    stem_stress_margin: float | None
    leakage_source: str | None
    leakage_artifact_sha256: str | None
    actuator_source: str | None
    actuator_artifact_sha256: str | None
    sheet_thickness: float | None
    sheet_thickness_method: str | None
    sheet_thickness_source: str | None
    sheet_thickness_artifact_sha256: str | None
    sheet_thickness_geometry_fingerprint_sha256: str | None
    sheet_thickness_fluid_name: str | None
    sheet_thickness_opening_range: tuple[float, float] | None
    sheet_thickness_pressure_drop_range: tuple[float, float] | None
    sheet_thickness_mass_flow_range: tuple[float, float] | None
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "opening_distance_m": self.opening_distance,
            "minimum_opening_distance_m": self.minimum_opening_distance,
            "maximum_opening_m": self.maximum_opening,
            "opening_fraction": self.opening_fraction,
            "transition_opening_m": self.transition_opening,
            "tip_minimum_area_m2": self.tip_minimum_area,
            "center_gap_area_m2": self.center_gap_area,
            "effective_metering_area_m2": self.effective_metering_area,
            "transition_area_fraction": self.transition_area_fraction,
            "transition_margin": self.transition_margin,
            "discharge_coefficient": self.discharge_coefficient,
            "discharge_coefficient_model": self.discharge_coefficient_model,
            "discharge_coefficient_source": self.discharge_coefficient_source,
            "discharge_coefficient_artifact_sha256": (
                self.discharge_coefficient_artifact_sha256
            ),
            "discharge_coefficient_geometry_fingerprint_sha256": (
                self.discharge_coefficient_geometry_fingerprint_sha256
            ),
            "resolved_geometry_fingerprint_sha256": (
                self.resolved_geometry_fingerprint_sha256
            ),
            "calibration_reynolds_range": (
                list(self.calibration_reynolds_range)
                if self.calibration_reynolds_range is not None else None
            ),
            "calibration_pressure_drop_range_pa": (
                list(self.calibration_pressure_drop_range)
                if self.calibration_pressure_drop_range is not None else None
            ),
            "calibration_temperature_range_k": (
                list(self.calibration_temperature_range)
                if self.calibration_temperature_range is not None else None
            ),
            "calibration_cavitation_number_range": (
                list(self.calibration_cavitation_number_range)
                if self.calibration_cavitation_number_range is not None else None
            ),
            "calibration_fluid_name": self.calibration_fluid_name,
            "position_uncertainty_fraction": self.position_uncertainty_fraction,
            "metrology_source": self.metrology_source,
            "metrology_artifact_sha256": self.metrology_artifact_sha256,
            "pressure_force_n": self.pressure_force,
            "momentum_reaction_force_n": self.momentum_reaction_force,
            "spring_preload_force_n": self.spring_preload_force,
            "seal_friction_force_n": self.seal_friction_force,
            "inertia_force_n": self.inertia_force,
            "required_actuator_force_n": self.required_actuator_force,
            "actuator_force_capacity_n": self.actuator_force_capacity,
            "actuator_force_margin": self.actuator_force_margin,
            "stem_axial_stress_pa": self.stem_axial_stress,
            "stem_allowable_stress_pa": self.stem_allowable_stress,
            "stem_stress_margin": self.stem_stress_margin,
            "leakage_source": self.leakage_source,
            "leakage_artifact_sha256": self.leakage_artifact_sha256,
            "actuator_source": self.actuator_source,
            "actuator_artifact_sha256": self.actuator_artifact_sha256,
            "sheet_thickness_m": self.sheet_thickness,
            "sheet_thickness_method": self.sheet_thickness_method,
            "sheet_thickness_source": self.sheet_thickness_source,
            "sheet_thickness_artifact_sha256": (
                self.sheet_thickness_artifact_sha256
            ),
            "sheet_thickness_geometry_fingerprint_sha256": (
                self.sheet_thickness_geometry_fingerprint_sha256
            ),
            "sheet_thickness_fluid_name": self.sheet_thickness_fluid_name,
            "sheet_thickness_opening_range_m": (
                list(self.sheet_thickness_opening_range)
                if self.sheet_thickness_opening_range is not None else None
            ),
            "sheet_thickness_pressure_drop_range_pa": (
                list(self.sheet_thickness_pressure_drop_range)
                if self.sheet_thickness_pressure_drop_range is not None else None
            ),
            "sheet_thickness_mass_flow_range_kg_s": (
                list(self.sheet_thickness_mass_flow_range)
                if self.sheet_thickness_mass_flow_range is not None else None
            ),
            "hard_stops": {
                "closed_m": 0.0,
                "open_m": self.maximum_opening,
                "command_within_stops": (
                    0.0 < self.opening_distance <= self.maximum_opening
                ),
            },
            "assumptions": list(self.assumptions),
            "hardware_qualified": False,
        }


def center_gap_area(center_gap_diameter: float, pintle_rod_diameter: float) -> float:
    """Fixed centre-gap annular area, ``pi/4 (Dcg^2-Dpr^2)``."""

    outer = _positive("center_gap_diameter", center_gap_diameter)
    inner = _positive("pintle_rod_diameter", pintle_rod_diameter)
    if outer <= inner:
        raise ValueError("center_gap_diameter must exceed pintle_rod_diameter")
    return math.pi * (outer * outer - inner * inner) / 4.0


def son_minimum_tip_area(
    opening_distance: float,
    *,
    post_diameter: float,
    post_thickness: float,
    tip_angle_deg: float,
) -> float:
    """Son et al. 2017 Eq. (1), stable at zero tip angle.

    The expanded expression is

    ``Amin = pi [2 rf L cos(theta) - L^2 sin(theta) cos(theta)^2]``

    with ``rf = Rpost-tpost``.  It is algebraically equivalent to the printed
    equation and its ``theta -> 0`` limit is ``2*pi*rf*L``.
    """

    opening = _nonnegative("opening_distance", opening_distance)
    diameter = _positive("post_diameter", post_diameter)
    thickness = _positive("post_thickness", post_thickness)
    angle = float(tip_angle_deg)
    if not math.isfinite(angle) or not 0.0 <= angle < 90.0:
        raise ValueError("tip_angle_deg must be finite and in [0, 90)")
    effective_radius = 0.5 * diameter - thickness
    if effective_radius <= 0.0:
        raise ValueError("post_thickness must be smaller than post radius")
    theta = math.radians(angle)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    area = math.pi * (
        2.0 * effective_radius * opening * cosine
        - opening * opening * sine * cosine * cosine
    )
    if area < -1.0e-15 or not math.isfinite(area):
        raise ValueError("opening lies outside the monotone Son tip-area branch")
    return max(area, 0.0)


def minimum_opening_distance(opening_distance: float, tip_angle_deg: float) -> float:
    """Perpendicular post-tip/tip-surface distance ``Lmin=Lopen*cos(theta)``."""

    opening = _nonnegative("opening_distance", opening_distance)
    angle = float(tip_angle_deg)
    if not math.isfinite(angle) or not 0.0 <= angle < 90.0:
        raise ValueError("tip_angle_deg must be finite and in [0, 90)")
    return opening * math.cos(math.radians(angle))


def opening_for_tip_area(
    target_area: float,
    *,
    post_diameter: float,
    post_thickness: float,
    tip_angle_deg: float,
) -> float:
    """Invert Son Eq. (1) on its monotone small-opening branch."""

    target = _positive("target_area", target_area)
    effective_radius = 0.5 * _positive("post_diameter", post_diameter) - _positive(
        "post_thickness", post_thickness
    )
    if effective_radius <= 0.0:
        raise ValueError("post_thickness must be smaller than post radius")
    theta = math.radians(float(tip_angle_deg))
    cosine = math.cos(theta)
    sine = math.sin(theta)
    if not 0.0 <= float(tip_angle_deg) < 90.0:
        raise ValueError("tip_angle_deg must be in [0, 90)")
    if sine <= 1.0e-12:
        return target / (2.0 * math.pi * effective_radius)
    # The expanded Eq. (1) is quadratic in L.  Use the smaller root; the
    # larger root is beyond the physical monotone branch.
    discriminant = effective_radius * effective_radius - (
        target * sine / math.pi
    )
    if discriminant <= 0.0:
        raise ValueError("target area is outside the Son tip-area branch")
    return (
        effective_radius - math.sqrt(discriminant)
    ) / (sine * cosine)


def transition_opening(spec: MovablePintleSpec, *, tip_angle_deg: float) -> float:
    """Opening at which Son tip area equals the fixed centre-gap area."""

    if None in (
        spec.post_diameter,
        spec.post_thickness,
        spec.center_gap_diameter,
        spec.pintle_rod_diameter,
    ):
        raise ValueError(
            "movable pintle requires post_diameter, post_thickness, "
            "center_gap_diameter, and pintle_rod_diameter"
        )
    cap = center_gap_area(spec.center_gap_diameter, spec.pintle_rod_diameter)
    return opening_for_tip_area(
        cap,
        post_diameter=spec.post_diameter,
        post_thickness=spec.post_thickness,
        tip_angle_deg=tip_angle_deg,
    )


def resolve_maximum_opening(
    spec: MovablePintleSpec, *, tip_angle_deg: float
) -> tuple[float, float, float]:
    """Return ``(open_stop, transition, area_fraction_at_open_stop)``."""

    transition = transition_opening(spec, tip_angle_deg=tip_angle_deg)
    cap = center_gap_area(spec.center_gap_diameter, spec.pintle_rod_diameter)
    fraction = float(spec.transition_area_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("transition_area_fraction must be in (0, 1)")
    if spec.maximum_opening is None:
        maximum = opening_for_tip_area(
            fraction * cap,
            post_diameter=spec.post_diameter,
            post_thickness=spec.post_thickness,
            tip_angle_deg=tip_angle_deg,
        )
    else:
        maximum = _positive("maximum_opening", spec.maximum_opening)
    area_fraction = son_minimum_tip_area(
        maximum,
        post_diameter=spec.post_diameter,
        post_thickness=spec.post_thickness,
        tip_angle_deg=tip_angle_deg,
    ) / cap
    if maximum >= transition or area_fraction >= 1.0:
        raise ValueError(
            "maximum_opening reaches/exceeds the center-gap transition where "
            "pintle travel no longer controls minimum area"
        )
    return maximum, transition, area_fraction


def movable_geometry_fingerprint(
    spec: MovablePintleSpec,
    *,
    tip_angle_deg: float,
) -> str:
    """Deterministic fingerprint of the Son hydraulic geometry contract.

    Calibration and sheet artifacts declare this digest so data from a
    different post, rod, centre gap, tip angle, or hard-stop definition cannot
    silently pass an operating-domain check.
    """

    maximum, transition, stop_area_fraction = resolve_maximum_opening(
        spec, tip_angle_deg=tip_angle_deg
    )
    payload = {
        "schema": "raosim.son2017_movable_geometry.v1",
        "model_id": SON2017_MODEL_ID,
        "post_diameter_m": _positive("post_diameter", spec.post_diameter),
        "post_thickness_m": _positive("post_thickness", spec.post_thickness),
        "center_gap_diameter_m": _positive(
            "center_gap_diameter", spec.center_gap_diameter
        ),
        "pintle_rod_diameter_m": _positive(
            "pintle_rod_diameter", spec.pintle_rod_diameter
        ),
        "tip_angle_deg": float(tip_angle_deg),
        "maximum_opening_m": maximum,
        "transition_opening_m": transition,
        "open_stop_area_fraction": stop_area_fraction,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_cd_curve(
    spec: MovablePintleSpec,
) -> tuple[tuple[float, float], ...] | None:
    if not spec.cd_vs_opening_fraction:
        return None
    points = tuple((float(x), float(cd)) for x, cd in spec.cd_vs_opening_fraction)
    if len(points) < 2:
        raise ValueError("Cd calibration requires at least two points")
    if points[0][0] != 0.0 or points[-1][0] != 1.0:
        raise ValueError("Cd calibration opening fractions must span exactly 0 to 1")
    if any(
        not math.isfinite(x)
        or not math.isfinite(cd)
        or not 0.0 <= x <= 1.0
        or not 0.0 < cd <= 1.0
        for x, cd in points
    ):
        raise ValueError("Cd calibration points require fraction in [0,1], Cd in (0,1]")
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("Cd calibration fractions must be strictly increasing")
    if not str(spec.cd_calibration_source or "").strip():
        raise ValueError("Cd calibration points require cd_calibration_source")
    digest = str(spec.cd_calibration_artifact_sha256 or "").strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(
            "Cd calibration points require a 64-character artifact SHA-256"
        )
    geometry_digest = str(
        spec.cd_geometry_fingerprint_sha256 or ""
    ).strip().lower()
    if len(geometry_digest) != 64 or any(
        c not in "0123456789abcdef" for c in geometry_digest
    ):
        raise ValueError(
            "Cd calibration points require a 64-character geometry fingerprint"
        )
    return points


def discharge_coefficient_at_opening(
    spec: MovablePintleSpec,
    *,
    opening_distance: float,
    maximum_opening: float,
    fallback_cd: float,
) -> tuple[float, str, str | None]:
    """Interpolate calibrated ``Cd(L/Lmax)`` or label a constant fallback."""

    maximum = _positive("maximum_opening", maximum_opening)
    opening = _nonnegative("opening_distance", opening_distance)
    if opening > maximum * (1.0 + 1.0e-12):
        raise ValueError("opening_distance exceeds maximum_opening")
    curve = _validated_cd_curve(spec)
    if curve is None:
        cd = float(fallback_cd)
        if not math.isfinite(cd) or not 0.0 < cd <= 1.0:
            raise ValueError("fallback Cd must be in (0, 1]")
        return cd, "constant_uncalibrated", None
    fraction = min(max(opening / maximum, 0.0), 1.0)
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if fraction <= x1:
            weight = (fraction - x0) / (x1 - x0)
            return (
                y0 + weight * (y1 - y0),
                "linear_calibrated_cd_vs_opening_fraction",
                str(spec.cd_calibration_source),
            )
    return curve[-1][1], "linear_calibrated_cd_vs_opening_fraction", str(
        spec.cd_calibration_source
    )


def solve_opening_for_mass_flow(
    spec: MovablePintleSpec,
    *,
    tip_angle_deg: float,
    required_mass_flow: float,
    fallback_cd: float,
    mass_flux_for_cd: Callable[[float], float],
) -> tuple[float, float, float, str, str | None, float, float, float]:
    """Solve the implicit calibrated-Cd metering law for opening.

    Returns opening, effective area, Cd, Cd-model, source, maximum opening,
    transition opening, and open-stop area fraction.
    """

    required = _positive("required_mass_flow", required_mass_flow)
    maximum, transition, stop_area_fraction = resolve_maximum_opening(
        spec, tip_angle_deg=tip_angle_deg
    )

    def delivered(opening: float) -> tuple[float, float, str, str | None, float]:
        cd, model, source = discharge_coefficient_at_opening(
            spec,
            opening_distance=opening,
            maximum_opening=maximum,
            fallback_cd=fallback_cd,
        )
        area = son_minimum_tip_area(
            opening,
            post_diameter=spec.post_diameter,
            post_thickness=spec.post_thickness,
            tip_angle_deg=tip_angle_deg,
        )
        flux = _positive("mass flux from Cd", mass_flux_for_cd(cd))
        return flux * area, cd, model, source, area

    capacity, *_ = delivered(maximum)
    if capacity < required:
        raise ValueError(
            f"required movable-pintle mass flow {required:.6g} kg/s exceeds "
            f"open-stop capacity {capacity:.6g} kg/s below the center-gap transition"
        )
    low, high = 0.0, maximum
    for _ in range(100):
        middle = 0.5 * (low + high)
        flow, *_ = delivered(middle)
        if flow < required:
            low = middle
        else:
            high = middle
    opening = 0.5 * (low + high)
    flow, cd, model, source, area = delivered(opening)
    if abs(flow - required) / required > 1.0e-10:
        raise ValueError("movable-pintle opening solve did not close mass flow")
    return (
        opening,
        area,
        cd,
        model,
        source,
        maximum,
        transition,
        stop_area_fraction,
    )


def static_actuator_ledger(
    spec: MovablePintleSpec,
    *,
    pressure_drop: float,
    delivered_mass_flow: float,
    injection_velocity: float,
    axial_momentum_fraction: float = 1.0,
) -> dict[str, float | None]:
    """Resolve declared static pressure/momentum/friction/inertia loads."""

    dp = _positive("pressure_drop", pressure_drop)
    mdot = _positive("delivered_mass_flow", delivered_mass_flow)
    velocity = _positive("injection_velocity", injection_velocity)
    pressure_force = (
        None
        if spec.unbalanced_pressure_area is None
        else dp * _nonnegative(
            "unbalanced_pressure_area", spec.unbalanced_pressure_area
        )
    )
    axial_fraction = _nonnegative(
        "axial_momentum_fraction", axial_momentum_fraction
    )
    if axial_fraction > 1.0:
        raise ValueError("axial_momentum_fraction must be <= 1")
    momentum_force = mdot * velocity * axial_fraction
    preload = _nonnegative("spring_preload_force", spec.spring_preload_force)
    friction = (
        None
        if spec.seal_friction_force is None
        else _nonnegative("seal_friction_force", spec.seal_friction_force)
    )
    inertia = None
    if spec.moving_mass is not None and spec.maximum_acceleration is not None:
        inertia = _positive("moving_mass", spec.moving_mass) * _nonnegative(
            "maximum_acceleration", spec.maximum_acceleration
        )
    required = None
    if pressure_force is not None and friction is not None and inertia is not None:
        factor = _positive("force_safety_factor", spec.force_safety_factor)
        if factor < 1.0:
            raise ValueError("force_safety_factor must be >= 1")
        required = factor * (
            pressure_force + momentum_force + preload + friction + inertia
        )
    capacity = (
        None
        if spec.actuator_force_capacity is None
        else _positive("actuator_force_capacity", spec.actuator_force_capacity)
    )
    force_margin = (
        capacity / required
        if capacity is not None and required is not None and required > 0.0
        else None
    )
    stem_stress = None
    if required is not None and spec.stem_diameter is not None:
        diameter = _positive("stem_diameter", spec.stem_diameter)
        stem_stress = required / (math.pi * diameter * diameter / 4.0)
    allowable = (
        None
        if spec.stem_allowable_stress is None
        else _positive("stem_allowable_stress", spec.stem_allowable_stress)
    )
    stress_margin = (
        allowable / stem_stress
        if allowable is not None and stem_stress is not None and stem_stress > 0.0
        else None
    )
    return {
        "pressure_force": pressure_force,
        "momentum_reaction_force": momentum_force,
        "spring_preload_force": preload,
        "seal_friction_force": friction,
        "inertia_force": inertia,
        "required_actuator_force": required,
        "actuator_force_capacity": capacity,
        "actuator_force_margin": force_margin,
        "stem_axial_stress": stem_stress,
        "stem_allowable_stress": allowable,
        "stem_stress_margin": stress_margin,
    }


__all__ = [
    "MovablePintleActuation",
    "MovablePintleSpec",
    "SON2017_MODEL_ID",
    "center_gap_area",
    "discharge_coefficient_at_opening",
    "minimum_opening_distance",
    "movable_geometry_fingerprint",
    "opening_for_tip_area",
    "resolve_maximum_opening",
    "solve_opening_for_mass_flow",
    "son_minimum_tip_area",
    "static_actuator_ledger",
    "transition_opening",
]
