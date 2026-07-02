"""engine_cad.py - engine-level STEP assembly (pump CAD plan Phase 3).

Aggregates already-exported and validated STEP artifacts (regen wall,
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

import math
from pathlib import Path
from typing import Any

from raosim.pump_cad_brep import cadquery_available, inspect_pump_step

__all__ = [
    "cadquery_available",
    "export_engine_assembly",
    "pump_mount_flange_screen",
]


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
        pressure = (
            line.get("required_outlet_pressure_pa")
            if isinstance(line, dict)
            else getattr(line, "pressure_rise", None)
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
    """Assemble validated STEP artifacts into one ``engine_assembly.step``.

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
    wall_radius = 0.0
    for name in ("wall", "jacket", "regen_wall"):
        if name in bboxes:
            bb = bboxes[name]
            wall_radius = max(wall_radius, 0.5 * bb.ylen, 0.5 * bb.zlen)

    for name, shape in imported.items():
        if name in ("wall", "jacket", "regen_wall"):
            shape = _rot_x_to_z(shape)
            bb = shape.BoundingBox()
            # Chamber inlet plane onto the injector face plane.
            shape = shape.translate(cq.Vector(0, 0, -bb.zmin))
        elif name in ("fuel_pump", "oxidizer_pump"):
            bb = bboxes[name]
            side = 1.0 if name == "fuel_pump" else -1.0
            x_off = side * (wall_radius + 0.5 * bb.xlen + clearance)
            # Fully behind the injector face plane.
            shape = shape.translate(
                cq.Vector(x_off, 0.0, -(bb.zmax + clearance))
            )
        elif name == "shared_battery_pack":
            bb = bboxes[name]
            y_off = -(wall_radius + 0.5 * bb.ylen + clearance)
            shape = shape.translate(
                cq.Vector(0.0, y_off, -(bb.zmax + clearance))
            )
        assembly.add(shape, name=name)
        children[name] = str(artifacts[name])

    if not children:
        raise ValueError(
            "engine assembly has no valid STEP children to place"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assembly.save(str(out_path))
    info = inspect_pump_step(out_path)
    if not (info["valid"] and info["volume_mm3"] > 0.0):
        raise RuntimeError(
            f"engine assembly re-import gate failed for {out_path.name}"
        )

    result: dict[str, Any] = {
        "path": str(out_path),
        "children": children,
        "diagnostics": info,
        "notes": notes,
    }
    if pump_result is not None:
        try:
            result["pump_mount_flange_screen"] = pump_mount_flange_screen(
                pump_result
            )
        except Exception as exc:  # screening is best-effort at engine level
            notes.append(f"pump mount flange screen skipped: {exc}")
    return result
