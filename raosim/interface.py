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


_FACEPLATE_CLAMPED_K = 0.75
_DEFAULT_JOINT_SEPARATION_FACTOR = 1.5
_DEFAULT_EDGE_DISTANCE_FACTOR = 1.5
_DEFAULT_PITCH_FACTOR = 3.0
_THREAD_TENSILE_AREA_FACTOR = 0.75


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _status_from_margin(margin: float | None, *, missing: str = "info") -> str:
    if margin is None:
        return missing
    return "pass" if margin >= 0.0 else "fail"


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
            "feasible": self.feasible,
            "gates": [g.to_dict() for g in self.gates],
            "notes": self.notes,
        }


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
    if wall_thickness is None:
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
        gates=gates,
        notes=notes,
    )

