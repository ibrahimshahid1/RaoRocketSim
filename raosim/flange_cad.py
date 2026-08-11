"""flange_cad.py — chamber flange solid, bolt pattern and fastener callout.

The gap this closes
-------------------
Every other part of the bolted chamber/injector joint existed as geometry: the
injector faceplate is exported with its bolt-hole pattern, o-ring groove and
inlet bosses (``injector_cad._build_machined_faceplate``), and the chamber wall
is exported from its meridian (``export.export_step`` / ``regen_cad``).  The
**chamber-side flange was not modelled at all** — no ring, no holes.  The
assembly therefore had a faceplate with twelve bolt holes and nothing to bolt it
to, and ``raosim.mass_ledger.flange_bolt_mass_ledger`` was pricing a part that
no exporter wrote.

This module builds that ring, and emits an *orderable* fastener callout rather
than only a diameter, because "correctly sized bolts" means a thread
designation, a property class, a grip length and a torque — not a number in
metres.

Preload and torque
------------------
The recommended preload follows standard bolted-joint practice: a target preload
of ``k_preload`` times the proof load, with ``k_preload = 0.75`` for
reusable/non-permanent joints, and the short-form torque relation

    T = K * F_i * d

with a nut factor ``K`` (0.20 dry steel, 0.15 lubricated) — Shigley,
*Mechanical Engineering Design*.  These are machine-design conventions, not
propulsion-corpus values, so the callout is labelled ``screening_sized`` and
carries its assumptions explicitly.  ``K`` in particular is empirical and
scatters by 20-30 % in practice; a flight joint needs measured torque-tension
data or angle/stretch control, which this does not replace.

The flange itself is **not** a qualified joint design.  It has no gasket
seating-stress calculation, no flange rotation, no preload scatter, no thermal
gradient and no fatigue analysis.  It is the geometry that matches the sizing
screens already implemented in :mod:`raosim.interface`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "cadquery_available",
    "FastenerCallout",
    "fastener_callout",
    "build_chamber_flange",
    "export_chamber_flange_step",
]

# Shigley short-form torque coefficients.
_NUT_FACTOR = {"dry": 0.20, "lubricated": 0.15, "plated": 0.18}
_PRELOAD_FRACTION_REUSABLE = 0.75
_PRELOAD_FRACTION_PERMANENT = 0.90


def cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
    except Exception:
        return False
    return True


def _cq():
    try:
        import cadquery as cq
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "chamber flange STEP export requires CadQuery/OpenCascade "
            "(pip install cadquery)"
        ) from exc
    return cq


@dataclass(frozen=True)
class FastenerCallout:
    """An orderable fastener specification for one joint."""

    designation: str
    property_class: str
    count: int
    nominal_diameter: float
    thread_pitch: float
    stress_area: float
    grip_length: float
    minimum_length: float
    proof_stress: float
    proof_load: float
    target_preload: float
    nut_factor: float
    lubrication: str
    tightening_torque: float
    separation_load: float
    load_per_bolt: float
    utilisation: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "iso_metric_fastener_callout",
            "designation": self.designation,
            "property_class": self.property_class,
            "count": self.count,
            "nominal_diameter_m": self.nominal_diameter,
            "thread_pitch_m": self.thread_pitch,
            "stress_area_m2": self.stress_area,
            "grip_length_m": self.grip_length,
            "minimum_bolt_length_m": self.minimum_length,
            "proof_stress_pa": self.proof_stress,
            "proof_load_n": self.proof_load,
            "target_preload_n": self.target_preload,
            "nut_factor": self.nut_factor,
            "lubrication": self.lubrication,
            "tightening_torque_n_m": self.tightening_torque,
            "joint_separation_load_n": self.separation_load,
            "load_per_bolt_n": self.load_per_bolt,
            "utilisation": self.utilisation,
            "status": "screening_sized",
            "qualification_status": (
                "requires measured torque-tension data or angle/stretch "
                "control, gasket seating stress, preload scatter, thermal "
                "gradient and fatigue analysis before flight use"
            ),
            "notes": self.notes,
        }


def fastener_callout(
    sizing: Any,
    *,
    lubrication: str = "dry",
    reusable: bool = True,
    thread_engagement_factor: float = 1.0,
) -> FastenerCallout:
    """Turn a :class:`raosim.interface.BoltedInterfaceSizing` into a callout.

    ``grip`` is the clamped stack (flange length + faceplate thickness); the
    minimum bolt length adds a nut height and two thread pitches of protrusion,
    which is the usual minimum for full nut engagement.
    """

    from raosim.interface import _BOLT_CLASSES, validate_bolted_interface_geometry

    if lubrication not in _NUT_FACTOR:
        raise ValueError(
            f"unknown lubrication {lubrication!r}; "
            f"choose from {sorted(_NUT_FACTOR)}"
        )
    res = validate_bolted_interface_geometry(sizing.resolution)
    d = float(sizing.bolt_nominal_diameter)
    pitch = float(sizing.bolt_pitch_thread)
    area = float(sizing.bolt_stress_area)
    proof, _ult = _BOLT_CLASSES[sizing.bolt_class]

    grip = float(res.flange_length) + float(res.face_thickness)
    nut_height = 0.8 * d
    minimum_length = grip + nut_height + 2.0 * pitch

    proof_load = proof * area
    fraction = (
        _PRELOAD_FRACTION_REUSABLE if reusable else _PRELOAD_FRACTION_PERMANENT
    )
    preload = fraction * proof_load
    K = _NUT_FACTOR[lubrication]
    torque = K * preload * d

    notes = [
        f"target preload = {fraction:.2f} x proof load "
        f"({'reusable/non-permanent' if reusable else 'permanent'} joint)",
        f"torque from the Shigley short-form T = K F_i d with K = {K:.2f} "
        f"({lubrication}); K scatters 20-30 % in practice, so this is a "
        "starting value for torque-tension testing, not a flight value",
        f"minimum length = grip {grip * 1e3:.2f} mm + nut {nut_height * 1e3:.2f} mm "
        f"+ 2 pitches protrusion",
    ]
    if thread_engagement_factor > 1.0:
        notes.append(
            f"thread engagement factor {thread_engagement_factor:g} applied "
            "for a tapped blind hole in a softer parent material"
        )
    return FastenerCallout(
        designation=sizing.bolt_designation,
        property_class=sizing.bolt_class,
        count=int(res.bolt_count),
        nominal_diameter=d,
        thread_pitch=pitch,
        stress_area=area,
        grip_length=grip,
        minimum_length=minimum_length * thread_engagement_factor,
        proof_stress=proof,
        proof_load=proof_load,
        target_preload=preload,
        nut_factor=K,
        lubrication=lubrication,
        tightening_torque=torque,
        separation_load=float(sizing.separation_load),
        load_per_bolt=float(sizing.load_per_bolt),
        utilisation=float(sizing.bolt_utilisation),
        notes=notes,
    )


def build_chamber_flange(
    cq,
    resolution: Any,
    *,
    seal_groove_width: float | None = None,
    seal_groove_depth: float | None = None,
    fillet: float | None = None,
):
    """Build the chamber-side flange ring with its bolt pattern.

    Geometry, in the injector-face coordinate system used by
    :mod:`raosim.injector_cad` (chamber side at ``Z = 0``, flange extending to
    ``Z = +flange_length`` behind the face):

    * annulus from the chamber bore to the flange outer diameter;
    * ``bolt_count`` through-holes on the bolt circle, matching the faceplate
      pattern hole-for-hole so the two parts actually mate;
    * an optional o-ring groove on the sealing face, mirroring the groove the
      faceplate carries.

    The chamber *bore* is the inner diameter, not the chamber outer diameter:
    the flange is a collar around the chamber, so its inner surface is the
    chamber's outer surface and the two are fused in the assembly.
    """

    from raosim.interface import validate_bolted_interface_geometry

    resolution = validate_bolted_interface_geometry(resolution)
    mm = 1000.0
    r_in = 0.5 * float(resolution.chamber_outer_diameter) * mm
    r_out = 0.5 * float(resolution.flange_outer_diameter) * mm
    length = float(resolution.flange_length) * mm
    if r_out <= r_in:
        raise ValueError(
            "flange outer diameter must exceed the chamber outer diameter"
        )
    if length <= 0.0:
        raise ValueError("flange length must be positive")

    flange = (
        cq.Workplane("XY").circle(r_out).circle(r_in).extrude(length)
    )

    count = int(resolution.bolt_count)
    hole_r = 0.5 * float(resolution.bolt_hole_diameter) * mm
    if count > 0 and hole_r > 0.0:
        bcr = 0.5 * float(resolution.bolt_circle_diameter) * mm
        eps = 1.0e-3
        points = [
            (
                bcr * math.cos(2.0 * math.pi * i / count),
                bcr * math.sin(2.0 * math.pi * i / count),
            )
            for i in range(count)
        ]
        holes = (
            cq.Workplane("XY").workplane(offset=-eps)
            .pushPoints(points).circle(hole_r).extrude(length + 2.0 * eps)
        )
        flange = flange.cut(holes)

    if seal_groove_width and seal_groove_depth:
        sw = float(seal_groove_width) * mm
        sd = float(seal_groove_depth) * mm
        # Centre the groove between the bore and the bolt circle so it seals
        # inboard of the fasteners, which is where the pressure boundary is.
        bcr = 0.5 * float(resolution.bolt_circle_diameter) * mm
        sr = 0.5 * (r_in + bcr)
        groove = (
            cq.Workplane("XY").circle(sr + 0.5 * sw).circle(sr - 0.5 * sw)
            .extrude(sd)
        )
        flange = flange.cut(groove)

    if fillet:
        try:
            flange = flange.edges("|Z").fillet(float(fillet) * mm)
        except Exception:
            # A fillet that cannot be applied is a manufacturing detail, not a
            # reason to fail the export; the un-filleted ring is still valid.
            pass
    return flange


def export_chamber_flange_step(
    resolution: Any,
    path: str | Path,
    *,
    sizing: Any | None = None,
    seal_groove_width: float | None = None,
    seal_groove_depth: float | None = None,
) -> dict[str, Any]:
    """Write ``chamber_flange.step`` and return its report.

    The export is gated the same way the rest of the repository's CAD is: the
    file is re-imported and must come back as exactly one valid solid with
    positive volume, otherwise the call raises rather than leaving an invalid
    artifact on disk.
    """

    cq = _cq()
    path = Path(path)
    flange = build_chamber_flange(
        cq, resolution,
        seal_groove_width=seal_groove_width,
        seal_groove_depth=seal_groove_depth,
    )
    solids = [s for v in flange.vals() for s in v.Solids()]
    if len(solids) != 1 or not solids[0].isValid():
        raise RuntimeError(
            "chamber flange did not resolve to a single valid solid; the "
            "bolt circle or seal groove may be cutting the ring apart"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(flange, str(path))

    reimported = cq.importers.importStep(str(path))
    back = [s for v in reimported.vals() for s in v.Solids()]
    volume_mm3 = float(sum(abs(s.Volume()) for s in back))
    if len(back) != 1 or not back[0].isValid() or volume_mm3 <= 0.0:
        raise RuntimeError(
            f"chamber flange re-import gate failed for {path.name}"
        )

    report: dict[str, Any] = {
        "file": str(path),
        "neutral_file_linear_unit": "mm",
        "solid_count": len(back),
        "valid": True,
        "volume_mm3": volume_mm3,
        "volume_m3": volume_mm3 * 1.0e-9,
        "bolt_count": int(resolution.bolt_count),
        "bolt_circle_diameter_m": float(resolution.bolt_circle_diameter),
        "bolt_hole_diameter_m": float(resolution.bolt_hole_diameter),
        "flange_outer_diameter_m": float(resolution.flange_outer_diameter),
        "flange_length_m": float(resolution.flange_length),
        "status": "preliminary_layout_requires_joint_qualification",
        "notes": [
            "bolt pattern matches the injector faceplate hole-for-hole so the "
            "two parts mate",
            "no gasket seating stress, flange rotation, preload scatter, "
            "thermal gradient or fatigue analysis is included",
        ],
    }
    if sizing is not None:
        report["fastener_callout"] = fastener_callout(sizing).to_dict()
    return report
