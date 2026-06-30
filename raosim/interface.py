"""Injector-to-chamber interface screening utilities.

This module keeps the injector body / chamber joint separate from the pintle
metering geometry.  It is a preliminary design ledger, not a bolted-joint
qualification model.  The implemented checks follow standard first-pass
mechanics:

* pressure separating force: F = Pc * A_projected
* chamber shell hoop screen: sigma = Pc * r / t (thin pressure-vessel form)
* injector faceplate bending screen: clamped circular plate under uniform
  pressure, sigma ~= 0.75 * Pc * a^2 / t^2 (Roark/classical plate theory)
* bolt clamp screen: total clamp load >= separation_factor * F, with per-bolt
  stress over an approximate tensile area (Shigley-style bolted-joint screen)
* edge distance / pitch screens: common preliminary machine-design heuristics
  (about 1.5 hole diameters to a free edge, about 3 diameters pitch)

These are deliberately conservative checks to keep the CLI honest about what it
knows.  Final hardware still needs gasket/seal design, preload scatter, thread
engagement, flange flexibility, thermal gradients, fatigue, and FEA/test data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


_FACEPLATE_CLAMPED_K = 0.75
_DEFAULT_JOINT_SEPARATION_FACTOR = 1.5
_DEFAULT_EDGE_DISTANCE_FACTOR = 1.5
_DEFAULT_PITCH_FACTOR = 3.0
_THREAD_TENSILE_AREA_FACTOR = 0.75
_DEFAULT_INTERFACE_BOLT_COUNT = 8
_DEFAULT_INTERFACE_BOLT_HOLE = 6.0e-3
_DEFAULT_INTERFACE_FLANGE_LENGTH = 6.0e-3


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _status_from_margin(margin: float | None, *, missing: str = "info") -> str:
    if margin is None:
        return missing
    # Auto-sized layouts often land exactly on a rule boundary; avoid turning
    # binary floating-point dust into a failed mechanical screen.
    return "pass" if margin >= -1.0e-12 else "fail"


def _finite_positive(value) -> float | None:
    v = _finite(value)
    return v if v is not None and v > 0.0 else None


def _round_up_even(value: float) -> int:
    count = int(math.ceil(max(value, 1.0)))
    return count if count % 2 == 0 else count + 1


def _station_array(value, n: int, name: str, *, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(n, float(default))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass
class InterfaceGate:
    name: str
    status: str
    detail: str
    value: float | None = None
    limit: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "value": self.value,
            "limit": self.limit,
        }


@dataclass
class CompositeRegenWallScreen:
    """Bonded liner plus closeout jacket hoop screen for a sized regen wall."""

    model: str
    qualification_status: str
    liner_material: str | None
    jacket_material: str | None
    governing_index: int
    governing_component: str
    chamber_pressure: float
    chamber_radius: float
    structural_fos: float
    liner_allowable_stress: float
    jacket_allowable_stress: float
    min_liner_margin: float
    min_jacket_margin: float
    min_margin: float
    liner_total_stress: float
    jacket_total_stress: float
    liner_local_sp125_stress: float
    liner_global_membrane_stress: float
    jacket_coolant_hoop_stress: float
    jacket_global_membrane_stress: float
    global_residual_pressure: float
    global_residual_membrane_load: float
    t_liner_equivalent_min: float
    t_liner_equivalent_max: float
    t_jacket_min: float
    t_jacket_max: float
    land_fraction_min: float
    land_fraction_max: float
    screened_station_count: int
    screen_selection: str
    stress_free_temperature: float

    @property
    def status(self) -> str:
        return "pass" if self.min_margin >= 1.0 else "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "qualification_status": self.qualification_status,
            "liner_material": self.liner_material,
            "jacket_material": self.jacket_material,
            "governing_index": self.governing_index,
            "governing_component": self.governing_component,
            "chamber_pressure_pa": self.chamber_pressure,
            "chamber_radius_m": self.chamber_radius,
            "structural_fos": self.structural_fos,
            "liner_allowable_stress_pa": self.liner_allowable_stress,
            "jacket_allowable_stress_pa": self.jacket_allowable_stress,
            "min_liner_margin": self.min_liner_margin,
            "min_jacket_margin": self.min_jacket_margin,
            "min_margin": self.min_margin,
            "liner_total_stress_pa": self.liner_total_stress,
            "jacket_total_stress_pa": self.jacket_total_stress,
            "liner_local_sp125_stress_pa": self.liner_local_sp125_stress,
            "liner_global_membrane_stress_pa": self.liner_global_membrane_stress,
            "jacket_coolant_hoop_stress_pa": self.jacket_coolant_hoop_stress,
            "jacket_global_membrane_stress_pa": self.jacket_global_membrane_stress,
            "global_residual_pressure_pa": self.global_residual_pressure,
            "global_residual_membrane_load_n_per_m":
                self.global_residual_membrane_load,
            "t_liner_equivalent_range_m": [
                self.t_liner_equivalent_min,
                self.t_liner_equivalent_max,
            ],
            "t_jacket_range_m": [self.t_jacket_min, self.t_jacket_max],
            "land_fraction_range": [
                self.land_fraction_min,
                self.land_fraction_max,
            ],
            "screened_station_count": self.screened_station_count,
            "screen_selection": self.screen_selection,
            "stress_free_temperature_K": self.stress_free_temperature,
        }


@dataclass
class InjectorInterfaceLedger:
    """Resolved chamber/injector mechanical-interface screen."""

    chamber_radius: float
    chamber_pressure: float
    projected_area: float
    separating_force: float
    wall_thickness: float | None
    face_outer_diameter: float | None
    face_thickness: float | None
    face_required_thickness: float | None
    bolt_count: int | None
    bolt_circle_diameter: float | None
    bolt_hole_diameter: float | None
    bolt_diameter: float | None
    required_total_clamp: float | None
    required_clamp_per_bolt: float | None
    bolt_stress: float | None
    bolt_allowable_stress: float | None
    inner_edge_distance: float | None
    outer_edge_distance: float | None
    pitch: float | None
    composite_wall: CompositeRegenWallScreen | None = None
    gates: list[InterfaceGate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not any(g.status == "fail" for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "injector_chamber_interface_screen",
            "literature_basis": [
                "pressure separating force F = Pc*A (standard pressure-vessel load)",
                "thin-wall hoop stress sigma = Pc*r/t (Barlow/pressure-vessel screen)",
                "clamped circular plate stress sigma ~= 0.75*Pc*a^2/t^2 (Roark/classical plates)",
                "bolt preload/separation screen after Shigley-style bolted-joint design",
                "edge-distance and pitch checks are preliminary machine-design heuristics",
            ],
            "chamber_radius_m": self.chamber_radius,
            "chamber_pressure_pa": self.chamber_pressure,
            "projected_area_m2": self.projected_area,
            "separating_force_n": self.separating_force,
            "wall_thickness_m": self.wall_thickness,
            "face_outer_diameter_m": self.face_outer_diameter,
            "face_thickness_m": self.face_thickness,
            "face_required_thickness_m": self.face_required_thickness,
            "bolt_count": self.bolt_count,
            "bolt_circle_diameter_m": self.bolt_circle_diameter,
            "bolt_hole_diameter_m": self.bolt_hole_diameter,
            "bolt_diameter_m": self.bolt_diameter,
            "required_total_clamp_n": self.required_total_clamp,
            "required_clamp_per_bolt_n": self.required_clamp_per_bolt,
            "bolt_stress_pa": self.bolt_stress,
            "bolt_allowable_stress_pa": self.bolt_allowable_stress,
            "inner_edge_distance_m": self.inner_edge_distance,
            "outer_edge_distance_m": self.outer_edge_distance,
            "bolt_pitch_m": self.pitch,
            "composite_wall": (
                self.composite_wall.to_dict() if self.composite_wall else None
            ),
            "feasible": self.feasible,
            "gates": [g.to_dict() for g in self.gates],
            "notes": self.notes,
        }


@dataclass
class InterfaceGeometryResolution:
    """Resolved bolt-together chamber flange / injector face dimensions."""

    chamber_radius: float
    chamber_outer_diameter: float
    flange_outer_diameter: float
    flange_length: float
    face_outer_diameter: float
    face_thickness: float
    bolt_count: int
    bolt_circle_diameter: float
    bolt_hole_diameter: float
    bolt_diameter: float | None
    inner_edge_distance: float
    outer_edge_distance: float
    bolt_pitch: float
    edge_distance_requirement: float
    pitch_requirement: float
    auto_sized_fields: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "resolved_bolted_chamber_injector_interface",
            "chamber_radius_m": self.chamber_radius,
            "chamber_outer_diameter_m": self.chamber_outer_diameter,
            "flange_outer_diameter_m": self.flange_outer_diameter,
            "flange_length_m": self.flange_length,
            "injector_face_outer_diameter_m": self.face_outer_diameter,
            "injector_face_thickness_m": self.face_thickness,
            "bolt_count": self.bolt_count,
            "bolt_circle_diameter_m": self.bolt_circle_diameter,
            "bolt_hole_diameter_m": self.bolt_hole_diameter,
            "bolt_diameter_m": self.bolt_diameter,
            "inner_edge_distance_m": self.inner_edge_distance,
            "outer_edge_distance_m": self.outer_edge_distance,
            "bolt_pitch_m": self.bolt_pitch,
            "edge_distance_requirement_m": self.edge_distance_requirement,
            "pitch_requirement_m": self.pitch_requirement,
            "auto_sized_fields": self.auto_sized_fields,
            "notes": self.notes,
        }


def resolve_bolted_interface_geometry(
    *,
    chamber_radius: float,
    chamber_pressure: float | None = None,
    wall_thickness: float | None = None,
    flange_outer_diameter: float | None = None,
    flange_length: float | None = None,
    face_outer_diameter: float | None = None,
    face_thickness: float | None = None,
    bolt_count: int | None = None,
    bolt_circle_diameter: float | None = None,
    bolt_hole_diameter: float | None = None,
    bolt_diameter: float | None = None,
    bolt_allowable_stress: float | None = None,
    material_yield_strength: float | None = None,
    structural_fos: float = 1.5,
    min_feature: float | None = None,
    min_tool_diameter: float | None = None,
    minimum_face_outer_diameter: float | None = None,
    minimum_face_thickness: float | None = None,
    minimum_bolt_circle_diameter: float | None = None,
    minimum_bolt_hole_diameter: float | None = None,
    edge_distance_factor: float = _DEFAULT_EDGE_DISTANCE_FACTOR,
    pitch_factor: float = _DEFAULT_PITCH_FACTOR,
    joint_separation_factor: float = _DEFAULT_JOINT_SEPARATION_FACTOR,
    default_bolt_count: int = _DEFAULT_INTERFACE_BOLT_COUNT,
) -> InterfaceGeometryResolution:
    """Resolve matching chamber flange and injector-face dimensions.

    The resolver is deliberately a layout screen, not a bolted-joint design.
    It fills missing dimensions, grows undersized layout values, and preserves
    explicit larger user values.  Strength is represented only through a
    first-pass bolt-count estimate when pressure and allowable data exist; the
    detailed joint check remains :func:`screen_injector_chamber_interface`.
    """

    r = float(chamber_radius)
    if r <= 0.0 or not math.isfinite(r):
        raise ValueError("chamber_radius must be positive")
    if structural_fos <= 0.0:
        raise ValueError("structural_fos must be positive")
    if edge_distance_factor <= 0.0 or pitch_factor <= 0.0:
        raise ValueError("edge and pitch factors must be positive")

    wall = _finite_positive(wall_thickness) or 0.0
    feature = max(_finite_positive(min_feature) or 3.0e-4, 1.0e-9)
    tool = max(_finite_positive(min_tool_diameter) or feature, feature)
    chamber_d = 2.0 * r
    chamber_od = chamber_d + 2.0 * wall
    auto: dict[str, str] = {}
    notes: list[str] = []

    def resolved(name: str, supplied, minimum: float, default: float | None = None) -> float:
        supplied_value = _finite_positive(supplied)
        target = max(float(minimum), float(default if default is not None else minimum))
        if supplied_value is None:
            auto[name] = "auto_sized"
            return target
        if supplied_value < minimum:
            auto[name] = f"increased_from_{supplied_value:.9g}_m"
            notes.append(
                f"{name} increased from {supplied_value:.6g} m to "
                f"{minimum:.6g} m to satisfy bolt/flange layout rules."
            )
            return float(minimum)
        return supplied_value

    bolt_hole_min = max(
        minimum_bolt_hole_diameter or 0.0,
        2.5 * tool,
        2.0 * feature,
    )
    bolt_hole_default = max(
        bolt_hole_min,
        _DEFAULT_INTERFACE_BOLT_HOLE,
        0.06 * chamber_d,
    )
    hole = resolved(
        "bolt_hole_diameter",
        bolt_hole_diameter,
        bolt_hole_min,
        bolt_hole_default,
    )

    requested_count = bolt_count if bolt_count is not None else default_bolt_count
    count = int(max(requested_count, 1))
    if bolt_count is None:
        auto["bolt_count"] = "auto_sized"

    bolt_allow = _finite_positive(bolt_allowable_stress)
    if bolt_allow is None and material_yield_strength is not None:
        bolt_allow = float(material_yield_strength) / structural_fos
    Pc = _finite_positive(chamber_pressure)
    if Pc is not None and bolt_allow is not None and bolt_allow > 0.0:
        inferred_bolt = _finite_positive(bolt_diameter) or 0.9 * hole
        tensile_area = (
            _THREAD_TENSILE_AREA_FACTOR
            * math.pi * inferred_bolt * inferred_bolt / 4.0
        )
        capacity = max(bolt_allow * tensile_area, 1.0e-12)
        required_clamp = joint_separation_factor * Pc * math.pi * r * r
        strength_count = _round_up_even(required_clamp / capacity)
        if bolt_count is None and strength_count > count:
            count = strength_count
            auto["bolt_count"] = "auto_sized_from_pressure_load"

    edge_req = edge_distance_factor * hole
    pitch_req = pitch_factor * hole
    bcd_min = max(
        chamber_od + hole + 2.0 * edge_req,
        count * pitch_req / math.pi,
        minimum_bolt_circle_diameter or 0.0,
    )
    bcd_default = max(bcd_min, chamber_od + 6.0 * hole)
    bcd = resolved(
        "bolt_circle_diameter",
        bolt_circle_diameter,
        bcd_min,
        bcd_default,
    )

    face_od_min = max(
        chamber_od + 4.0 * edge_req,
        bcd + hole + 2.0 * edge_req,
        minimum_face_outer_diameter or 0.0,
    )
    face_od_resolved = resolved(
        "injector_face_od",
        face_outer_diameter,
        face_od_min,
        face_od_min,
    )
    flange_od_resolved = resolved(
        "flange_od",
        flange_outer_diameter,
        max(face_od_min, face_od_resolved),
        max(face_od_min, face_od_resolved),
    )
    matched_od = max(face_od_resolved, flange_od_resolved)
    if face_od_resolved < matched_od:
        auto["injector_face_od"] = "matched_to_flange_od"
        face_od_resolved = matched_od
    if flange_od_resolved < matched_od:
        auto["flange_od"] = "matched_to_injector_face_od"
        flange_od_resolved = matched_od

    plate_req = 0.0
    yield_strength = _finite_positive(material_yield_strength)
    if Pc is not None and yield_strength is not None:
        allowable = yield_strength / structural_fos
        if allowable > 0.0:
            plate_req = r * math.sqrt(_FACEPLATE_CLAMPED_K * Pc / allowable)
    face_t_min = max(
        minimum_face_thickness or 0.0,
        plate_req,
        2.0 * hole,
        6.0 * tool,
        _DEFAULT_INTERFACE_FLANGE_LENGTH,
    )
    face_t = resolved(
        "injector_face_thickness",
        face_thickness,
        face_t_min,
        face_t_min,
    )
    flange_l = resolved(
        "flange_length",
        flange_length,
        max(2.0 * hole, 3.0 * wall, 6.0 * tool),
        max(2.0 * hole, 3.0 * wall, 6.0 * tool, _DEFAULT_INTERFACE_FLANGE_LENGTH),
    )

    inner_edge = 0.5 * (bcd - chamber_od) - 0.5 * hole
    outer_edge = 0.5 * flange_od_resolved - 0.5 * bcd - 0.5 * hole
    pitch = math.pi * bcd / count
    notes.append(
        "Resolved flange and injector face are preliminary matching layout "
        "dimensions; final joint design still needs gasket/seal compression, "
        "preload scatter, thread engagement, thermal distortion, FEA, and test."
    )

    return InterfaceGeometryResolution(
        chamber_radius=r,
        chamber_outer_diameter=chamber_od,
        flange_outer_diameter=flange_od_resolved,
        flange_length=flange_l,
        face_outer_diameter=face_od_resolved,
        face_thickness=face_t,
        bolt_count=count,
        bolt_circle_diameter=bcd,
        bolt_hole_diameter=hole,
        bolt_diameter=_finite_positive(bolt_diameter),
        inner_edge_distance=inner_edge,
        outer_edge_distance=outer_edge,
        bolt_pitch=pitch,
        edge_distance_requirement=edge_req,
        pitch_requirement=pitch_req,
        auto_sized_fields=auto,
        notes=notes,
    )


def screen_composite_regen_wall(
    *,
    chamber_pressure: float,
    wall_profile: Any,
    liner_material: Any,
    jacket_material: Any | None = None,
    structural_fos: float = 1.5,
    gas_side_wall_temperature=None,
    coolant_side_wall_temperature=None,
    coolant_temperature=None,
    coolant_pressure=None,
    liner_pressure_differential=None,
    heat_flux=None,
    stress_free_temperature: float = 293.15,
    screen_station_index: int | None = None,
    screen_selection: str | None = None,
) -> CompositeRegenWallScreen:
    """Screen chamber hoop sharing in a bonded liner plus jacket regen wall.

    The local copper liner channel-roof term stays separate from the global
    chamber hoop membrane.  The global membrane uses a common hoop strain for
    smeared copper ribs plus the outer jacket at each station:

        eps = (N_theta + sum(E*t*alpha*dT)) / sum(E*t)

    ``coolant_pressure`` is treated as the absolute pressure contained by the
    outer closeout jacket.  The common-strain pressure load is therefore only
    residual chamber-over-coolant pressure, avoiding double-counting the jacket
    hoop term.  This is a preliminary bonded-shell screen, not CHT/FEA
    qualification.
    """

    Pc = float(chamber_pressure)
    if Pc <= 0.0:
        raise ValueError("chamber_pressure must be positive")
    if structural_fos <= 0.0:
        raise ValueError("structural_fos must be positive")
    if wall_profile is None:
        raise ValueError("wall_profile is required")

    x = np.asarray(wall_profile.x, dtype=float)
    r_inner = np.asarray(wall_profile.r_inner, dtype=float)
    t_hot = np.asarray(wall_profile.t_hot, dtype=float)
    channel_width = np.asarray(wall_profile.channel_width, dtype=float)
    channel_height = np.asarray(wall_profile.channel_height, dtype=float)
    land_width = np.asarray(wall_profile.land_width, dtype=float)
    t_jacket = np.asarray(wall_profile.t_jacket, dtype=float)
    n = len(x)
    for name, arr in (
        ("r_inner", r_inner),
        ("t_hot", t_hot),
        ("channel_width", channel_width),
        ("channel_height", channel_height),
        ("land_width", land_width),
        ("t_jacket", t_jacket),
    ):
        if arr.shape != (n,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"wall_profile.{name} must be finite shape ({n},)")
    if np.any(r_inner <= 0.0) or np.any(t_hot <= 0.0) or np.any(t_jacket <= 0.0):
        raise ValueError("wall radii and thicknesses must be positive")

    jmat = jacket_material if jacket_material is not None else liner_material
    E_l = float(getattr(liner_material, "elastic_modulus"))
    a_l = float(getattr(liner_material, "thermal_expansion"))
    nu_l = float(getattr(liner_material, "poisson_ratio"))
    k_l = float(getattr(liner_material, "conductivity"))
    Sy_l = float(getattr(liner_material, "yield_strength"))
    E_j = float(getattr(jmat, "elastic_modulus"))
    a_j = float(getattr(jmat, "thermal_expansion"))
    Sy_j = float(getattr(jmat, "yield_strength"))

    land_fraction = land_width / np.maximum(land_width + channel_width, 1e-12)
    land_fraction = np.clip(land_fraction, 0.0, 1.0)
    t_liner_eq = t_hot + land_fraction * channel_height

    from raosim.regen_profile import normal_offset_contour

    _, r_jacket = normal_offset_contour(
        x,
        r_inner,
        t_hot + channel_height,
    )

    T_wg = _station_array(
        gas_side_wall_temperature,
        n,
        "gas_side_wall_temperature",
        default=stress_free_temperature,
    )
    T_wc = _station_array(
        coolant_side_wall_temperature,
        n,
        "coolant_side_wall_temperature",
        default=stress_free_temperature,
    )
    T_c = _station_array(
        coolant_temperature,
        n,
        "coolant_temperature",
        default=stress_free_temperature,
    )
    liner_mean_T = 0.5 * (T_wg + T_wc)
    jacket_mean_T = 0.5 * (T_wc + T_c)
    dT_l = liner_mean_T - float(stress_free_temperature)
    dT_j = jacket_mean_T - float(stress_free_temperature)

    coolant_p = _station_array(
        coolant_pressure,
        n,
        "coolant_pressure",
        default=Pc,
    )
    liner_dp = _station_array(
        liner_pressure_differential,
        n,
        "liner_pressure_differential",
        default=0.0,
    )
    q = _station_array(heat_flux, n, "heat_flux", default=0.0)

    if screen_station_index is None:
        active = np.ones(n, dtype=bool)
        selection = screen_selection or "all_stations"
    else:
        selected = int(screen_station_index)
        if selected < 0:
            selected += n
        if selected < 0 or selected >= n:
            raise ValueError(
                f"screen_station_index {screen_station_index} outside 0..{n - 1}"
            )
        active = np.zeros(n, dtype=bool)
        active[selected] = True
        selection = screen_selection or f"station_{selected}"
    active_indices = np.flatnonzero(active)

    local_radius = np.maximum(0.5 * channel_width, 1e-9)
    liner_pressure = np.abs(liner_dp) * local_radius / np.maximum(t_hot, 1e-12)
    liner_thermal = (
        E_l * a_l * q * t_hot
        / max(2.0 * (1.0 - nu_l) * max(k_l, 1e-12), 1e-12)
    )
    liner_local = liner_pressure + liner_thermal

    residual_pressure = np.maximum(Pc - coolant_p, 0.0)
    N_theta = residual_pressure * r_inner
    denom = E_l * t_liner_eq + E_j * t_jacket
    eps = (
        N_theta
        + E_l * t_liner_eq * a_l * dT_l
        + E_j * t_jacket * a_j * dT_j
    ) / np.maximum(denom, 1e-12)
    sigma_l_global = E_l * (eps - a_l * dT_l)
    sigma_j_global = E_j * (eps - a_j * dT_j)
    jacket_hoop = coolant_p * r_jacket / np.maximum(t_jacket, 1e-12)

    liner_total = liner_local + np.abs(sigma_l_global)
    jacket_total = jacket_hoop + np.abs(sigma_j_global)
    liner_allow = Sy_l / structural_fos
    jacket_allow = Sy_j / structural_fos
    liner_margin = liner_allow / np.maximum(liner_total, 1e-9)
    jacket_margin = jacket_allow / np.maximum(jacket_total, 1e-9)
    component_margin = np.minimum(liner_margin, jacket_margin)
    idx = int(active_indices[np.nanargmin(component_margin[active])])
    governing_component = (
        "liner" if liner_margin[idx] <= jacket_margin[idx] else "jacket"
    )

    return CompositeRegenWallScreen(
        model="bonded_smeared_liner_jacket_residual_common_strain_screen",
        qualification_status="screening_only_not_validated_cht_fea",
        liner_material=getattr(liner_material, "name", None),
        jacket_material=getattr(jmat, "name", None),
        governing_index=idx,
        governing_component=governing_component,
        chamber_pressure=Pc,
        chamber_radius=float(r_inner[idx]),
        structural_fos=float(structural_fos),
        liner_allowable_stress=float(liner_allow),
        jacket_allowable_stress=float(jacket_allow),
        min_liner_margin=float(np.nanmin(liner_margin[active])),
        min_jacket_margin=float(np.nanmin(jacket_margin[active])),
        min_margin=float(component_margin[idx]),
        liner_total_stress=float(liner_total[idx]),
        jacket_total_stress=float(jacket_total[idx]),
        liner_local_sp125_stress=float(liner_local[idx]),
        liner_global_membrane_stress=float(sigma_l_global[idx]),
        jacket_coolant_hoop_stress=float(jacket_hoop[idx]),
        jacket_global_membrane_stress=float(sigma_j_global[idx]),
        global_residual_pressure=float(residual_pressure[idx]),
        global_residual_membrane_load=float(N_theta[idx]),
        t_liner_equivalent_min=float(np.min(t_liner_eq[active])),
        t_liner_equivalent_max=float(np.max(t_liner_eq[active])),
        t_jacket_min=float(np.min(t_jacket[active])),
        t_jacket_max=float(np.max(t_jacket[active])),
        land_fraction_min=float(np.min(land_fraction[active])),
        land_fraction_max=float(np.max(land_fraction[active])),
        screened_station_count=int(active_indices.size),
        screen_selection=selection,
        stress_free_temperature=float(stress_free_temperature),
    )


def screen_injector_chamber_interface(
    *,
    chamber_pressure: float,
    chamber_radius: float,
    wall_thickness: float | None = None,
    face_outer_diameter: float | None = None,
    face_thickness: float | None = None,
    flange_outer_diameter: float | None = None,
    flange_length: float | None = None,
    bolt_count: int | None = None,
    bolt_circle_diameter: float | None = None,
    bolt_hole_diameter: float | None = None,
    bolt_diameter: float | None = None,
    material_yield_strength: float | None = None,
    material_elastic_modulus: float | None = None,
    material_poisson_ratio: float | None = None,
    structural_fos: float = 1.5,
    bolt_allowable_stress: float | None = None,
    composite_wall_screen: CompositeRegenWallScreen | None = None,
    joint_separation_factor: float = _DEFAULT_JOINT_SEPARATION_FACTOR,
    edge_distance_factor: float = _DEFAULT_EDGE_DISTANCE_FACTOR,
    pitch_factor: float = _DEFAULT_PITCH_FACTOR,
) -> InjectorInterfaceLedger:
    """Return a literature-labeled injector/chamber interface screen.

    All lengths are meters and pressures/stresses are Pa.  Missing quantities
    produce information gates instead of invented pass/fail results.
    """

    Pc = float(chamber_pressure)
    r = float(chamber_radius)
    if Pc <= 0.0 or r <= 0.0:
        raise ValueError("chamber_pressure and chamber_radius must be positive")
    if structural_fos <= 0.0 or joint_separation_factor <= 0.0:
        raise ValueError("safety/separation factors must be positive")

    chamber_d = 2.0 * r
    face_od = _finite(face_outer_diameter)
    if face_od is None:
        face_od = _finite(flange_outer_diameter)
    projected_area = math.pi * r * r
    separating_force = Pc * projected_area
    allowable = None
    if material_yield_strength is not None:
        allowable = float(material_yield_strength) / structural_fos

    face_t_req = None
    if allowable is not None and allowable > 0.0:
        face_t_req = r * math.sqrt(_FACEPLATE_CLAMPED_K * Pc / allowable)

    # Face deflection is reported only when E and nu are available.  It is not
    # a gate because seal compression limits are gasket-specific.
    face_deflection = None
    if (
        face_thickness is not None
        and material_elastic_modulus is not None
        and material_poisson_ratio is not None
        and face_thickness > 0.0
        and material_elastic_modulus > 0.0
    ):
        nu = float(material_poisson_ratio)
        face_deflection = (
            3.0 * (1.0 - nu * nu) * Pc * r**4
            / (16.0 * float(material_elastic_modulus) * face_thickness**3)
        )

    required_total_clamp = None
    required_per_bolt = None
    bolt_stress = None
    inferred_bolt_d = _finite(bolt_diameter)
    if inferred_bolt_d is None and bolt_hole_diameter is not None:
        # Clearance holes are larger than the bolt major diameter.  0.9 is a
        # screening estimate only; expose the inferred value in the ledger.
        inferred_bolt_d = 0.9 * float(bolt_hole_diameter)
    if bolt_count is not None and bolt_count > 0:
        required_total_clamp = joint_separation_factor * separating_force
        required_per_bolt = required_total_clamp / int(bolt_count)
        if inferred_bolt_d is not None and inferred_bolt_d > 0.0:
            tensile_area = (
                _THREAD_TENSILE_AREA_FACTOR
                * math.pi * inferred_bolt_d * inferred_bolt_d / 4.0
            )
            bolt_stress = required_per_bolt / tensile_area

    bolt_allow = _finite(bolt_allowable_stress)
    if bolt_allow is None and material_yield_strength is not None:
        # If no separate fastener material was supplied, use the selected body
        # material as an explicitly labeled screening proxy.
        bolt_allow = float(material_yield_strength) / structural_fos

    inner_edge = outer_edge = pitch = None
    if (
        bolt_circle_diameter is not None
        and bolt_hole_diameter is not None
        and bolt_count is not None
        and bolt_count > 0
    ):
        bcd = float(bolt_circle_diameter)
        hole = float(bolt_hole_diameter)
        inner_edge = 0.5 * (bcd - chamber_d) - 0.5 * hole
        if face_od is not None:
            outer_edge = 0.5 * face_od - 0.5 * bcd - 0.5 * hole
        pitch = math.pi * bcd / int(bolt_count)

    gates: list[InterfaceGate] = []

    # 1. Wall pressure-only hoop screen. The thermostructural regen profile is
    # handled elsewhere; this prevents the interface summary from pretending a
    # 1 mm reference wall was sized.
    if composite_wall_screen is not None:
        comp = composite_wall_screen
        gates.append(InterfaceGate(
            "composite_regen_wall_hoop",
            comp.status,
            (
                f"bonded liner+jacket hoop screen margin {comp.min_margin:.2f}; "
                f"{comp.screen_selection}, "
                f"liner {comp.liner_total_stress/1e6:.1f} MPa vs "
                f"{comp.liner_allowable_stress/1e6:.1f} MPa, jacket "
                f"{comp.jacket_total_stress/1e6:.1f} MPa vs "
                f"{comp.jacket_allowable_stress/1e6:.1f} MPa"
            ),
            value=comp.min_margin,
            limit=1.0,
        ))
    elif wall_thickness is None:
        gates.append(InterfaceGate(
            "chamber_wall_hoop_pressure", "info",
            "no chamber wall thickness supplied; wall STL would be a reference shell",
        ))
    elif allowable is None:
        gates.append(InterfaceGate(
            "chamber_wall_hoop_pressure", "info",
            "wall thickness supplied but no material yield/allowable stress was supplied",
            value=float(wall_thickness),
        ))
    else:
        hoop = Pc * r / max(float(wall_thickness), 1e-12)
        margin = allowable - hoop
        gates.append(InterfaceGate(
            "chamber_wall_hoop_pressure", _status_from_margin(margin),
            f"thin-wall hoop stress {hoop/1e6:.1f} MPa vs allowable {allowable/1e6:.1f} MPa",
            value=hoop, limit=allowable,
        ))

    # 2. Injector face OD must cover the bore. This is a geometry/readiness
    # check; seal land is handled by bolt edge distances when a pattern exists.
    if face_od is None:
        gates.append(InterfaceGate(
            "injector_face_covers_bore", "info",
            "injector face/flange OD not supplied",
            limit=chamber_d,
        ))
    else:
        margin = face_od - chamber_d
        gates.append(InterfaceGate(
            "injector_face_covers_bore", _status_from_margin(margin),
            f"face OD {face_od*1e3:.1f} mm vs chamber bore {chamber_d*1e3:.1f} mm",
            value=face_od, limit=chamber_d,
        ))

    # 3. Faceplate bending. If thickness is missing but material exists, report
    # the required thickness as an info gate.
    if face_t_req is None:
        gates.append(InterfaceGate(
            "injector_faceplate_bending", "info",
            "material yield not supplied; clamped circular-plate thickness not evaluated",
            value=face_thickness, limit=None,
        ))
    elif face_thickness is None:
        gates.append(InterfaceGate(
            "injector_faceplate_bending", "info",
            f"requires t_face >= {face_t_req*1e3:.2f} mm by clamped-plate screen",
            limit=face_t_req,
        ))
    else:
        margin = float(face_thickness) - face_t_req
        gates.append(InterfaceGate(
            "injector_faceplate_bending", _status_from_margin(margin),
            f"face thickness {float(face_thickness)*1e3:.2f} mm vs required {face_t_req*1e3:.2f} mm",
            value=float(face_thickness), limit=face_t_req,
        ))

    # 4. Bolt clamp load / tensile stress.
    if bolt_count is None:
        gates.append(InterfaceGate(
            "bolt_joint_separation", "info",
            "bolt pattern not supplied; pressure separating force is reported only",
            value=separating_force,
        ))
    elif bolt_count <= 0:
        gates.append(InterfaceGate(
            "bolt_joint_separation", "fail",
            "bolt_count must be positive when supplied",
            value=float(bolt_count), limit="> 0",
        ))
    elif bolt_stress is None:
        gates.append(InterfaceGate(
            "bolt_joint_separation", "info",
            f"requires total clamp >= {required_total_clamp:.0f} N; bolt diameter/hole missing",
            value=required_total_clamp,
        ))
    elif bolt_allow is None:
        gates.append(InterfaceGate(
            "bolt_joint_separation", "info",
            f"per-bolt stress {bolt_stress/1e6:.1f} MPa; no bolt allowable supplied",
            value=bolt_stress,
        ))
    else:
        margin = bolt_allow - bolt_stress
        gates.append(InterfaceGate(
            "bolt_joint_separation", _status_from_margin(margin),
            f"per-bolt stress {bolt_stress/1e6:.1f} MPa vs allowable {bolt_allow/1e6:.1f} MPa",
            value=bolt_stress, limit=bolt_allow,
        ))

    # 5. Bolt pattern geometry.
    if bolt_circle_diameter is None or bolt_hole_diameter is None or bolt_count is None:
        gates.append(InterfaceGate(
            "bolt_pattern_lands", "info",
            "bolt circle/hole/count incomplete; edge-distance and pitch not evaluated",
        ))
    else:
        hole = float(bolt_hole_diameter)
        edge_req = edge_distance_factor * hole
        pitch_req = pitch_factor * hole
        margins = [
            inner_edge - edge_req if inner_edge is not None else None,
            outer_edge - edge_req if outer_edge is not None else None,
            pitch - pitch_req if pitch is not None else None,
        ]
        finite_margins = [m for m in margins if m is not None]
        min_margin = min(finite_margins) if finite_margins else None
        gates.append(InterfaceGate(
            "bolt_pattern_lands", _status_from_margin(min_margin),
            (
                f"edge req {edge_req*1e3:.1f} mm, pitch req {pitch_req*1e3:.1f} mm; "
                f"inner {inner_edge*1e3:.1f} mm, outer "
                f"{outer_edge*1e3:.1f} mm, pitch {pitch*1e3:.1f} mm"
                if outer_edge is not None
                else "face/flange OD missing; outer bolt land not evaluated"
            ),
            value=min_margin, limit=0.0,
        ))

    notes = [
        "Injector/chamber interface screen is preliminary; it does not replace "
        "ASME/NASA joint design, gasket compression analysis, preload scatter, "
        "thread engagement checks, FEA, inspection, or proof testing.",
        f"Pressure separating force Pc*pi*r^2 = {separating_force:.0f} N.",
    ]
    if face_deflection is not None:
        notes.append(
            f"Clamped-plate center deflection estimate is {face_deflection*1e3:.3f} mm; "
            "no gasket-specific deflection gate is applied."
        )
    if inferred_bolt_d is not None and bolt_diameter is None and bolt_hole_diameter is not None:
        notes.append(
            "Bolt diameter inferred as 0.9*bolt_hole_diameter for screening; "
            "supply the actual bolt diameter/material for a meaningful joint check."
        )
    if flange_length is not None:
        notes.append(
            "flange_length is carried for CAD/interface metadata; bending of the "
            "flange ring is not solved in this screen."
        )

    return InjectorInterfaceLedger(
        chamber_radius=r,
        chamber_pressure=Pc,
        projected_area=projected_area,
        separating_force=separating_force,
        wall_thickness=_finite(wall_thickness),
        face_outer_diameter=face_od,
        face_thickness=_finite(face_thickness),
        face_required_thickness=face_t_req,
        bolt_count=int(bolt_count) if bolt_count is not None else None,
        bolt_circle_diameter=_finite(bolt_circle_diameter),
        bolt_hole_diameter=_finite(bolt_hole_diameter),
        bolt_diameter=inferred_bolt_d,
        required_total_clamp=required_total_clamp,
        required_clamp_per_bolt=required_per_bolt,
        bolt_stress=bolt_stress,
        bolt_allowable_stress=bolt_allow,
        inner_edge_distance=inner_edge,
        outer_edge_distance=outer_edge,
        pitch=pitch,
        composite_wall=composite_wall_screen,
        gates=gates,
        notes=notes,
    )
