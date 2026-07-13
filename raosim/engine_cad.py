"""engine_cad.py - engine-level STEP assembly (pump CAD plan Phase 3).

Aggregates already-exported and topology-gated STEP artifacts (regen wall,
jacket, machined pintle, pump packages, shared battery) into one named
``engine`` assembly for layout and trade review.  Nothing is re-derived
here: every child is imported from its own gated export, placed by a
documented layout transform, and the saved assembly is re-imported and
gated like the wall path.  The pump mounting flange is screened with the
existing bolted-interface resolver (:mod:`raosim.interface`), not designed.

Axes: engine axis +Z with the injector face plane at Z = 0 and the chamber
downstream at Z > 0 (the pintle STEP convention).  The wall/jacket STEPs
are revolved about +X by :mod:`raosim.export` and are rotated into +Z here.
Pump shaft axes stay parallel to the engine axis, packages placed behind
the injector face; the placements are layout placeholders, not routed
feed lines.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from raosim.pump_cad_brep import cadquery_available, inspect_pump_step

__all__ = [
    "cadquery_available",
    "audit_engine_component_interference",
    "export_engine_assembly",
    "pump_mount_flange_screen",
]


def audit_engine_component_interference(
    components: dict[str, Any], *, tolerance_mm3: float = 1.0e-6
) -> dict[str, Any]:
    """Reject positive-volume collisions between placed engine components.

    Coplanar mounting faces are permitted.  The inputs are already at the
    neutral-file boundary, so all reported volumes are mm^3.
    """
    names = list(components)
    pairs: list[dict[str, Any]] = []
    maximum = 0.0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            try:
                common = components[left].intersect(components[right])
                volume = float(sum(abs(s.Volume()) for s in common.Solids()))
            except Exception as exc:
                return {
                    "passed": False,
                    "status": "failed_to_evaluate",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pairs": pairs,
                }
            maximum = max(maximum, volume)
            pairs.append({
                "components": [left, right],
                "overlap_mm3": volume,
                "status": "pass" if volume <= tolerance_mm3 else "fail",
            })
    return {
        "passed": maximum <= tolerance_mm3,
        "status": "pass" if maximum <= tolerance_mm3 else "fail",
        "tolerance_mm3": float(tolerance_mm3),
        "maximum_overlap_mm3": maximum,
        "pairs": pairs,
    }


def _cq():
    try:
        import cadquery as cq
    except Exception as exc:
        raise RuntimeError(
            "engine assembly STEP export requires CadQuery/OpenCascade "
            "(pip install cadquery)"
        ) from exc
    return cq


def pump_mount_flange_screen(pump_result) -> dict[str, Any]:
    """First-pass pump mounting-flange bolt layout per stream.

    Reuses :func:`raosim.interface.resolve_bolted_interface_geometry` with
    the solved casing radius and required outlet pressure - a layout
    screen, not a bolted-joint design.
    """
    from raosim.interface import resolve_bolted_interface_geometry

    lines = (
        pump_result.get("lines")
        if isinstance(pump_result, dict)
        else pump_result.lines
    ) or {}
    screens: dict[str, Any] = {}
    for role, line in lines.items():
        ref = (
            line.get("reference_geometry")
            if isinstance(line, dict)
            else getattr(line, "reference_geometry", None)
        )
        if ref is None:
            continue
        scroll = (
            ref.get("volute_scroll") if isinstance(ref, dict)
            else ref.volute_scroll
        ) or {}
        casing_r = scroll.get("casing_inner_radius_m")
        if isinstance(line, dict):
            # PumpLineSizing.to_dict serializes the duty under
            # ``required_pressure_rise_pa``.  The previous lookup used a BOM
            # key that is absent from the serialized sizing object, silently
            # disabling the pressure-loaded bolt screen after JSON roundtrip.
            pressure = line.get("required_pressure_rise_pa")
            thermal_stress = line.get("thermal_stress") or {}
            loads = thermal_stress.get("loads") or {}
            pressure = loads.get("casing_pressure_pa", pressure)
        else:
            pressure = getattr(line, "pressure_rise", None)
            thermal_stress = getattr(line, "thermal_stress", None)
            if thermal_stress is not None:
                pressure = thermal_stress.loads.get(
                    "casing_pressure_pa", pressure
                )
        if not casing_r:
            continue
        resolution = resolve_bolted_interface_geometry(
            chamber_radius=float(casing_r),
            chamber_pressure=(
                float(pressure) if pressure is not None else None
            ),
            wall_thickness=scroll.get("casing_wall_thickness_m"),
        )
        entry = resolution.to_dict() if hasattr(resolution, "to_dict") else {
            "bolt_count": getattr(resolution, "bolt_count", None),
        }
        entry["source"] = (
            "raosim.interface.resolve_bolted_interface_geometry layout "
            "screen applied to the pump casing (plan Phase 3)"
        )
        screens[role] = entry
    return screens


def export_engine_assembly(
    out_path,
    artifacts: dict[str, str | Path],
    *,
    pump_result=None,
    clearance_m: float = 0.010,
) -> dict[str, Any]:
    """Assemble topology-gated STEP artifacts into one ``engine_assembly.step``.

    ``artifacts`` maps child names to STEP paths; recognized names get the
    documented layout transform (``wall``/``jacket``/``regen_wall`` rotate
    +X to +Z; ``pintle_injector`` identity; ``fuel_pump``/``oxidizer_pump``
    behind the face at +/-X; ``shared_battery_pack`` at -Y).  Unknown names
    are placed at the origin.  Missing files are skipped with a note.
    """
    cq = _cq()
    out_path = Path(out_path)
    assembly = cq.Assembly(name="engine")
    notes: list[str] = []
    children: dict[str, Any] = {}

    imported: dict[str, Any] = {}
    bboxes: dict[str, Any] = {}
    for name, path in artifacts.items():
        path = Path(path)
        if not path.exists():
            notes.append(f"{name} skipped: {path} not found")
            continue
        shape = cq.importers.importStep(str(path))
        solids = [s for v in shape.vals() for s in v.Solids()]
        if not solids or not all(s.isValid() for s in solids):
            notes.append(f"{name} skipped: {path.name} failed re-import")
            continue
        compound = cq.Compound.makeCompound(solids)
        imported[name] = compound
        bboxes[name] = compound.BoundingBox()

    def _rot_x_to_z(shape):
        return shape.rotate(
            cq.Vector(0, 0, 0), cq.Vector(0, 1, 0), -90.0
        )

    clearance = clearance_m * 1000.0
    if clearance <= 0.0:
        raise ValueError("engine assembly clearance_m must be positive")
    wall_radius = 0.0
    for name in ("wall", "jacket", "regen_wall"):
        if name in bboxes:
            bb = bboxes[name]
            wall_radius = max(wall_radius, 0.5 * bb.ylen, 0.5 * bb.zlen)
    occupied_y_min = min(
        [-wall_radius]
        + [
            float(bboxes[name].ymin)
            for name in ("fuel_pump", "oxidizer_pump")
            if name in bboxes
        ]
    )

    placed: dict[str, Any] = {}
    for name, shape in imported.items():
        if name in ("wall", "jacket", "regen_wall"):
            shape = _rot_x_to_z(shape)
            bb = shape.BoundingBox()
            # Chamber inlet plane onto the injector face plane.
            shape = shape.translate(cq.Vector(0, 0, -bb.zmin))
        elif name in ("fuel_pump", "oxidizer_pump"):
            bb = bboxes[name]
            if name == "fuel_pump":
                # Place the actual asymmetric package bounding box (including
                # outlet and inverter), not an assumed origin-centred box.
                x_off = wall_radius + clearance - bb.xmin
            else:
                x_off = -wall_radius - clearance - bb.xmax
            # Fully behind the injector face plane.
            shape = shape.translate(
                cq.Vector(x_off, 0.0, -(bb.zmax + clearance))
            )
        elif name == "shared_battery_pack":
            bb = bboxes[name]
            # Clear the full pump envelopes as well as the chamber wall.  A
            # wall-only offset can put the battery through an asymmetric
            # volute/outlet even though each individual STEP is valid.
            y_off = occupied_y_min - clearance - bb.ymax
            shape = shape.translate(
                cq.Vector(0.0, y_off, -(bb.zmax + clearance))
            )
        assembly.add(shape, name=name)
        placed[name] = shape
        children[name] = str(artifacts[name])

    if not children:
        raise ValueError(
            "engine assembly has no valid STEP children to place"
        )
    interference = audit_engine_component_interference(placed)
    if not interference["passed"]:
        raise RuntimeError(
            "engine assembly component interference gate failed: "
            f"{interference}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assembly.save(str(out_path))
    info = inspect_pump_step(out_path)
    if not (info["valid"] and info["volume_mm3"] > 0.0):
        raise RuntimeError(
            f"engine assembly re-import gate failed for {out_path.name}"
        )

    placed_boxes = {
        child.name: {
            "xmin": float(child.obj.BoundingBox().xmin),
            "xmax": float(child.obj.BoundingBox().xmax),
            "ymin": float(child.obj.BoundingBox().ymin),
            "ymax": float(child.obj.BoundingBox().ymax),
            "zmin": float(child.obj.BoundingBox().zmin),
            "zmax": float(child.obj.BoundingBox().zmax),
        }
        for child in assembly.children
        if child.obj is not None
    }
    result: dict[str, Any] = {
        "path": str(out_path),
        "children": children,
        "diagnostics": info,
        "component_bounding_boxes_mm": placed_boxes,
        "placement_clearance_mm": clearance,
        "assembly_gates": {
            "component_interference": interference,
        },
        "hardware_qualified": False,
        "external_release_blockers": [
            "routed and supported propellant/coolant lines",
            "mount, bracket, fastener, seal, and tolerance-stack drawings",
            "thermal growth, structural loads, vibration, and rotordynamics",
            "proof/leak, cold-flow, and hot-fire qualification evidence",
        ],
        "units": {
            "public_api_linear_unit": "m",
            "neutral_file_linear_unit": "mm",
            "volume_unit": "mm^3",
        },
        "notes": notes,
    }
    if pump_result is not None:
        try:
            result["pump_mount_flange_screen"] = pump_mount_flange_screen(
                pump_result
            )
        except Exception as exc:  # screening is best-effort at engine level
            notes.append(f"pump mount flange screen skipped: {exc}")
    sidecar = out_path.with_suffix(".cad.json")
    sidecar.write_text(json.dumps({
        "schema": "raosim.engine_cad.v1",
        "artifact": out_path.name,
        "children": children,
        "units": result["units"],
        "diagnostics": info,
        "component_bounding_boxes_mm": placed_boxes,
        "placement_clearance_mm": clearance,
        "assembly_gates": result["assembly_gates"],
        "hardware_qualified": False,
        "external_release_blockers": result["external_release_blockers"],
    }, indent=2) + "\n", encoding="utf-8")
    result["unit_sidecar"] = str(sidecar)
    return result
