"""OpenCascade regenerative-wall manufacturing geometry.

This module builds one material body containing:

* the hot-gas liner;
* the ribs/lands between passages;
* the outer closeout jacket;
* lofted rectangular channel voids following axial or helical paths;
* annular inlet/outlet plenums; and
* radial inlet/outlet ports.

All public geometry inputs use SI metres.  CadQuery/OpenCascade uses
millimetres internally, so the conversion is explicit at the kernel boundary.
The result is exported as a true STEP B-rep and re-imported for validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from raosim.regen_profile import normal_offset_contour

_MM = 1000.0


def cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
    except Exception:
        return False
    return True


def _cq():
    try:
        import cadquery as cq
    except Exception as exc:
        raise RuntimeError(
            "cooling-aware B-rep export requires CadQuery/OpenCascade"
        ) from exc
    return cq


def _clean_profile(profile: Any, max_sections: int) -> dict[str, np.ndarray]:
    fields = {
        "x": np.asarray(profile.x, dtype=float),
        "r": np.asarray(profile.r_inner, dtype=float),
        "t_hot": np.asarray(profile.t_hot, dtype=float),
        "w": np.asarray(profile.channel_width, dtype=float),
        "h": np.asarray(profile.channel_height, dtype=float),
        "t_jacket": np.asarray(profile.t_jacket, dtype=float),
    }
    n = len(fields["x"])
    if any(a.shape != (n,) for a in fields.values()):
        raise ValueError("all RegenWallProfile arrays must have equal length")
    order = np.argsort(fields["x"], kind="stable")
    fields = {k: v[order] for k, v in fields.items()}
    span = max(float(np.ptp(fields["x"])), 1e-12)
    keep = np.concatenate([
        [True],
        np.diff(fields["x"]) > 1e-9 * span,
    ])
    fields = {k: v[keep] for k, v in fields.items()}
    if len(fields["x"]) < 4:
        raise ValueError("regen CAD needs at least four unique axial stations")
    if len(fields["x"]) > max_sections:
        idx = np.unique(
            np.linspace(0, len(fields["x"]) - 1, max_sections).astype(int)
        )
        fields = {k: v[idx] for k, v in fields.items()}
    if min(
        float(np.min(fields[k]))
        for k in ("r", "t_hot", "w", "h", "t_jacket")
    ) <= 0.0:
        raise ValueError("regen CAD dimensions must be positive")
    return fields


def _normal_offset_mm(x, r, distance):
    xo, ro = normal_offset_contour(x, r, distance)
    return xo * _MM, ro * _MM


def _revolve_between(cq, x, r_inner, r_outer_x, r_outer):
    points = list(zip(x * _MM, r_inner * _MM))
    points += list(zip(r_outer_x[::-1] * _MM, r_outer[::-1] * _MM))
    return (
        cq.Workplane("XY")
        .polyline([(float(a), float(b)) for a, b in points])
        .close()
        .revolve(360.0, (0, 0, 0), (1, 0, 0))
        .val()
    )


def _annular_plenum(cq, x, r, t_hot, h, x_lo, x_hi, clearance_m):
    mask = (x >= x_lo) & (x <= x_hi)
    ids = np.flatnonzero(mask)
    if len(ids) < 3:
        nearest = np.argsort(
            np.minimum(np.abs(x - x_lo), np.abs(x - x_hi))
        )[:4]
        ids = np.sort(nearest)
    xx = x[ids]
    rr = r[ids]
    floor_x, floor_r = normal_offset_contour(
        xx, rr, t_hot[ids] + clearance_m
    )
    ceil_x, ceil_r = normal_offset_contour(
        xx, rr, t_hot[ids] + h[ids] - clearance_m
    )
    points = list(zip(floor_x * _MM, floor_r * _MM))
    points += list(zip(ceil_x[::-1] * _MM, ceil_r[::-1] * _MM))
    return (
        cq.Workplane("XY")
        .polyline([(float(a), float(b)) for a, b in points])
        .close()
        .revolve(360.0, (0, 0, 0), (1, 0, 0))
        .val()
    )


def _channel_loft(
    cq,
    *,
    x,
    r,
    t_hot,
    width,
    height,
    helix_turns,
    x_lo,
    x_hi,
    clearance_m,
):
    mask = (x >= x_lo) & (x <= x_hi)
    ids = np.flatnonzero(mask)
    if len(ids) < 4:
        raise ValueError("channel path has fewer than four loft sections")
    x = x[ids]
    r = r[ids]
    t_hot = t_hot[ids]
    width = width[ids]
    height = height[ids]

    floor_x, floor_r = normal_offset_contour(
        x, r, t_hot + clearance_m
    )
    center_x, center_r = normal_offset_contour(
        x, r, t_hot + 0.5 * height
    )
    ceil_x, ceil_r = normal_offset_contour(
        x, r, t_hot + height - clearance_m
    )
    frac = (x - x[0]) / max(float(x[-1] - x[0]), 1e-12)
    theta = 2.0 * math.pi * float(helix_turns) * frac

    center = np.column_stack([
        center_x,
        center_r * np.cos(theta),
        center_r * np.sin(theta),
    ]) * _MM
    floor = np.column_stack([
        floor_x,
        floor_r * np.cos(theta),
        floor_r * np.sin(theta),
    ]) * _MM
    ceiling = np.column_stack([
        ceil_x,
        ceil_r * np.cos(theta),
        ceil_r * np.sin(theta),
    ]) * _MM

    tangent = np.gradient(center, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    wall_normal = ceiling - floor
    wall_normal /= np.maximum(
        np.linalg.norm(wall_normal, axis=1)[:, None], 1e-12
    )
    width_axis = np.cross(tangent, wall_normal)
    width_axis /= np.maximum(
        np.linalg.norm(width_axis, axis=1)[:, None], 1e-12
    )

    wires = []
    effective_height = np.maximum(height - 2.0 * clearance_m, 1e-9) * _MM
    for c, nvec, bvec, w, h in zip(
        center, wall_normal, width_axis, width * _MM, effective_height
    ):
        points = [
            c - 0.5 * h * nvec - 0.5 * w * bvec,
            c - 0.5 * h * nvec + 0.5 * w * bvec,
            c + 0.5 * h * nvec + 0.5 * w * bvec,
            c + 0.5 * h * nvec - 0.5 * w * bvec,
        ]
        wires.append(
            cq.Wire.makePolygon(
                [cq.Vector(*map(float, p)) for p in points],
                close=True,
            )
        )
    return cq.Solid.makeLoft(wires, ruled=True)


def inspect_regen_step(path: str | Path) -> dict:
    """Re-import a STEP file and report B-rep topology and dimensions."""
    cq = _cq()
    path = Path(path).expanduser().resolve()
    imported = cq.importers.importStep(str(path))
    shapes = imported.vals()
    solids = [solid for shape in shapes for solid in shape.Solids()]
    valid = bool(solids) and all(s.isValid() for s in solids)
    if solids:
        bbox = solids[0].BoundingBox()
        volume = float(sum(s.Volume() for s in solids))
        bbox_mm = {
            "x": float(bbox.xlen),
            "y": float(bbox.ylen),
            "z": float(bbox.zlen),
        }
    else:
        volume = 0.0
        bbox_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "path": str(path),
        "representation": "brep",
        "shape_count": len(shapes),
        "solid_count": len(solids),
        "valid": valid,
        "single_solid": len(solids) == 1,
        "volume_mm3": volume,
        "bounding_box_mm": bbox_mm,
    }


def export_regen_brep(
    profile: Any,
    path: str | Path,
    *,
    max_sections: int = 28,
    manifold_length_fraction: float = 0.06,
    end_seal_fraction: float = 0.015,
    port_diameter: float | None = None,
    boolean_clearance: float = 5e-6,
    fuzzy_tolerance_mm: float = 1e-3,
    stl_path: str | Path | None = None,
) -> dict:
    """Build and export one cooling-aware regenerative STEP solid.

    The outer envelope is the fused liner+ribs+jacket material.  One lofted
    passage is circular-patterned by ``channel_count`` and Boolean-cut from
    it.  Annular plenums intersect every passage near each end, and radial
    ports connect the plenums to the exterior.
    """
    cq = _cq()
    data = _clean_profile(profile, max_sections=max_sections)
    x = data["x"]
    r = data["r"]
    t_hot = data["t_hot"]
    w = data["w"]
    h = data["h"]
    t_jacket = data["t_jacket"]
    n_channels = int(profile.channel_count)
    if n_channels < 2:
        raise ValueError("regen B-rep needs at least two channels")
    if not (0.0 < manifold_length_fraction < 0.25):
        raise ValueError("manifold_length_fraction must be in (0, 0.25)")
    if not (0.0 < end_seal_fraction < manifold_length_fraction):
        raise ValueError(
            "end_seal_fraction must be positive and shorter than the manifold"
        )
    if boolean_clearance <= 0.0:
        raise ValueError("boolean_clearance must be positive")

    x_outer, r_outer = normal_offset_contour(
        x, r, t_hot + h + t_jacket
    )
    envelope = _revolve_between(cq, x, r, x_outer, r_outer)
    if not envelope.isValid() or len(envelope.Solids()) != 1:
        raise RuntimeError("failed to construct a valid one-solid wall envelope")
    envelope_volume = float(envelope.Volume())

    length = float(x[-1] - x[0])
    plenum_length = max(
        manifold_length_fraction * length,
        2.5 * float(np.max(h)),
    )
    seal = max(
        end_seal_fraction * length,
        float(np.max(t_jacket)),
    )
    inlet_lo = float(x[0] + seal)
    inlet_hi = float(min(inlet_lo + plenum_length, x[-1] - 0.4 * length))
    outlet_hi = float(x[-1] - seal)
    outlet_lo = float(max(outlet_hi - plenum_length, x[0] + 0.4 * length))
    channel_lo = 0.5 * (inlet_lo + inlet_hi)
    channel_hi = 0.5 * (outlet_lo + outlet_hi)
    if channel_hi <= channel_lo:
        raise ValueError("no axial room remains between the two manifolds")

    base_channel = _channel_loft(
        cq,
        x=x,
        r=r,
        t_hot=t_hot,
        width=w,
        height=h,
        helix_turns=float(profile.helix_turns),
        x_lo=channel_lo,
        x_hi=channel_hi,
        clearance_m=boolean_clearance,
    )
    if not base_channel.isValid():
        raise RuntimeError("base channel loft is invalid")
    channels = [
        base_channel.rotate(
            (0, 0, 0), (1, 0, 0), 360.0 * i / n_channels
        )
        for i in range(n_channels)
    ]
    channel_compound = cq.Compound.makeCompound(channels)
    body = envelope.cut(channel_compound, tol=fuzzy_tolerance_mm)
    if not body.isValid():
        raise RuntimeError("channel Boolean cut produced an invalid body")

    inlet_plenum = _annular_plenum(
        cq, x, r, t_hot, h, inlet_lo, inlet_hi, boolean_clearance
    )
    outlet_plenum = _annular_plenum(
        cq, x, r, t_hot, h, outlet_lo, outlet_hi, boolean_clearance
    )
    # Sequential cuts are substantially more robust than one compound
    # Boolean for curved annular plenums.
    body = body.cut(inlet_plenum, tol=fuzzy_tolerance_mm)
    body = body.cut(outlet_plenum, tol=fuzzy_tolerance_mm)

    def _port(x_port, angle_deg):
        local_h = float(np.interp(x_port, x, h))
        local_outer = float(np.interp(x_port, x_outer, r_outer))
        local_floor = float(np.interp(x_port, x, r + t_hot))
        diameter = (
            0.75 * local_h
            if port_diameter is None
            else min(float(port_diameter), 0.90 * local_h)
        )
        if diameter <= 0.0:
            raise ValueError("port diameter must be positive")
        radius_mm = 0.5 * diameter * _MM
        start_radius_mm = (local_outer + diameter) * _MM
        end_radius_mm = (local_floor + 0.25 * local_h) * _MM
        length_mm = start_radius_mm - end_radius_mm
        theta = math.radians(angle_deg)
        outward = np.array([0.0, math.cos(theta), math.sin(theta)])
        start = np.array([x_port * _MM, 0.0, 0.0]) + outward * start_radius_mm
        return cq.Solid.makeCylinder(
            radius_mm,
            length_mm,
            pnt=tuple(map(float, start)),
            dir=tuple(map(float, -outward)),
        ), diameter

    inlet_x = 0.5 * (inlet_lo + inlet_hi)
    outlet_x = 0.5 * (outlet_lo + outlet_hi)
    inlet_port, inlet_diameter = _port(inlet_x, 0.0)
    outlet_port, outlet_diameter = _port(outlet_x, 180.0)
    body = body.cut(inlet_port, tol=fuzzy_tolerance_mm)
    body = body.cut(outlet_port, tol=fuzzy_tolerance_mm).clean()

    solids = body.Solids()
    if not body.isValid() or len(solids) != 1:
        volumes = sorted((float(s.Volume()) for s in solids), reverse=True)
        raise RuntimeError(
            "regen Boolean model is not one valid solid "
            f"(solid_count={len(solids)}, volumes_mm3={volumes[:8]})"
        )

    # Connectivity checks on one representative passage are sufficient
    # because the circular pattern and annular plenums are exact symmetries.
    inlet_overlap = float(base_channel.intersect(inlet_plenum).Volume())
    outlet_overlap = float(base_channel.intersect(outlet_plenum).Volume())
    inlet_port_overlap = float(inlet_port.intersect(inlet_plenum).Volume())
    outlet_port_overlap = float(outlet_port.intersect(outlet_plenum).Volume())
    overlaps = {
        "channel_to_inlet_plenum_mm3": inlet_overlap,
        "channel_to_outlet_plenum_mm3": outlet_overlap,
        "inlet_port_to_plenum_mm3": inlet_port_overlap,
        "outlet_port_to_plenum_mm3": outlet_port_overlap,
    }
    if any(value <= 1e-6 for value in overlaps.values()):
        raise RuntimeError(f"coolant network is not fully connected: {overlaps}")

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(body, str(path), exportType="STEP")
    if stl_path is not None:
        stl_path = Path(stl_path).expanduser().resolve()
        cq.exporters.export(body, str(stl_path), exportType="STL")

    inspection = inspect_regen_step(path)
    if not inspection["valid"] or not inspection["single_solid"]:
        raise RuntimeError(f"STEP re-import validation failed: {inspection}")
    body_volume = float(body.Volume())
    removed_volume = envelope_volume - body_volume
    return {
        "step_path": str(path),
        "stl_path": str(stl_path) if stl_path is not None else None,
        "representation": "open_cascade_brep",
        "single_solid": True,
        "valid": True,
        "channel_count": n_channels,
        "loft_sections": int(np.count_nonzero(
            (x >= channel_lo) & (x <= channel_hi)
        )),
        "helix_turns": float(profile.helix_turns),
        "manifold_length_m": float(plenum_length),
        "end_seal_length_m": float(seal),
        "inlet_port_diameter_m": float(inlet_diameter),
        "outlet_port_diameter_m": float(outlet_diameter),
        "envelope_volume_mm3": envelope_volume,
        "solid_volume_mm3": body_volume,
        "coolant_void_volume_mm3": removed_volume,
        "network_overlaps": overlaps,
        "inspection": inspection,
        "units": "CadQuery kernel/STEP millimetres; public API metres",
        "model": "fused_liner_ribs_jacket_with_lofted_passage_voids",
    }
