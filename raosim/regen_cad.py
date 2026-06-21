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

from raosim.regen_profile import helix_stretch_factors, normal_offset_contour

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


def _radius_at_x(x_query, x_profile, r_profile):
    """Interpolate a normal-offset profile onto common axial sections."""
    order = np.argsort(np.asarray(x_profile, dtype=float), kind="stable")
    xp = np.asarray(x_profile, dtype=float)[order]
    rp = np.asarray(r_profile, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(xp) > 1e-12])
    return np.interp(np.asarray(x_query, dtype=float), xp[keep], rp[keep])


def _annular_sector_wire(cq, x_m, r_inner_m, r_outer_m, theta0, theta1):
    """Closed wall-conformal annular-sector wire in a plane normal to x."""
    x_mm = float(x_m) * _MM
    ri_mm = float(r_inner_m) * _MM
    ro_mm = float(r_outer_m) * _MM
    if not (0.0 < ri_mm < ro_mm):
        raise ValueError("annular-sector radii must satisfy 0 < inner < outer")
    if not theta1 > theta0:
        raise ValueError("annular-sector angles must be increasing")

    # CadQuery's circle in the YZ plane uses angle=-90 deg at +Y.
    a0 = math.degrees(float(theta0)) - 90.0
    a1 = math.degrees(float(theta1)) - 90.0
    floor = cq.Edge.makeCircle(
        ri_mm, (x_mm, 0.0, 0.0), (1.0, 0.0, 0.0), a0, a1
    )
    ceiling = cq.Edge.makeCircle(
        ro_mm, (x_mm, 0.0, 0.0), (1.0, 0.0, 0.0), a0, a1
    )

    def point(radius_mm, theta):
        return cq.Vector(
            x_mm,
            radius_mm * math.cos(float(theta)),
            radius_mm * math.sin(float(theta)),
        )

    side0 = cq.Edge.makeLine(point(ri_mm, theta0), point(ro_mm, theta0))
    side1 = cq.Edge.makeLine(point(ri_mm, theta1), point(ro_mm, theta1))
    wire = cq.Wire.assembleEdges([floor, side1, ceiling, side0])
    if not wire.isValid() or not wire.IsClosed():
        raise RuntimeError("failed to build a closed annular-sector wire")
    return wire


def _pattern_wrapped_shapes(base_shape, count: int):
    """Circular pattern using OCC locations/shared geometry, not deep copies."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    wrapped = [base_shape.wrapped]
    axis = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0))
    for i in range(1, int(count)):
        trsf = gp_Trsf()
        trsf.SetRotation(axis, 2.0 * math.pi * i / int(count))
        wrapped.append(
            BRepBuilderAPI_Transform(
                base_shape.wrapped, trsf, False, False
            ).Shape()
        )
    return wrapped


def _kernel_boolean(
    cq,
    argument_shapes,
    tool_shapes=(),
    *,
    operation: str,
    fuzzy_tolerance_mm: float,
    glue: str = "off",
):
    """One parallel OpenCascade multi-shape Boolean operation."""
    from OCP.BOPAlgo import BOPAlgo_GlueEnum
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    arguments = TopTools_ListOfShape()
    for shape in argument_shapes:
        arguments.Append(getattr(shape, "wrapped", shape))
    tools = TopTools_ListOfShape()
    for shape in tool_shapes:
        tools.Append(getattr(shape, "wrapped", shape))

    if operation == "fuse":
        if tools.IsEmpty():
            # BRepAlgoAPI_Fuse expects at least one argument and one tool.
            all_shapes = list(argument_shapes)
            if len(all_shapes) < 2:
                return all_shapes[0], {
                    "operation": "fuse",
                    "run_parallel": True,
                    "glue": glue,
                }
            arguments = TopTools_ListOfShape()
            arguments.Append(getattr(all_shapes[0], "wrapped", all_shapes[0]))
            tools = TopTools_ListOfShape()
            for shape in all_shapes[1:]:
                tools.Append(getattr(shape, "wrapped", shape))
        op = BRepAlgoAPI_Fuse()
    elif operation == "cut":
        if tools.IsEmpty():
            raise ValueError("cut operation needs at least one tool")
        op = BRepAlgoAPI_Cut()
    else:
        raise ValueError(f"unknown Boolean operation {operation!r}")

    glue_map = {
        "off": BOPAlgo_GlueEnum.BOPAlgo_GlueOff,
        "shift": BOPAlgo_GlueEnum.BOPAlgo_GlueShift,
        "full": BOPAlgo_GlueEnum.BOPAlgo_GlueFull,
    }
    op.SetArguments(arguments)
    op.SetTools(tools)
    op.SetRunParallel(True)
    op.SetUseOBB(True)
    op.SetNonDestructive(True)
    op.SetGlue(glue_map[glue])
    op.SetFuzzyValue(float(fuzzy_tolerance_mm))
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"OpenCascade {operation} did not complete")
    op.SimplifyResult(True, True)
    shape = cq.Shape.cast(op.Shape())
    return shape, {
        "operation": operation,
        "run_parallel": True,
        "use_obb": True,
        "non_destructive": True,
        "glue": glue,
        "fuzzy_tolerance_mm": float(fuzzy_tolerance_mm),
    }


def _dominant_solid(cq, shape, *, max_sliver_fraction: float = 1e-5):
    """Return one valid dominant solid, tolerating only kernel-scale slivers."""
    solids = shape.Solids()
    if not solids:
        raise RuntimeError("Boolean result contains no solids")
    dominant = max(solids, key=lambda solid: float(solid.Volume()))
    dominant_volume = abs(float(dominant.Volume()))
    sliver_volume = sum(
        abs(float(solid.Volume()))
        for solid in solids
        if not solid.isSame(dominant)
    )
    fraction = sliver_volume / max(dominant_volume, 1e-12)
    if not dominant.isValid() or fraction > max_sliver_fraction:
        volumes = sorted(
            (float(solid.Volume()) for solid in solids), reverse=True
        )
        raise RuntimeError(
            "Boolean result is not one dominant valid solid "
            f"(solid_count={len(solids)}, sliver_fraction={fraction:.3e}, "
            f"volumes_mm3={volumes[:8]})"
        )
    return cq.Shape.cast(dominant.wrapped), {
        "raw_solid_count": len(solids),
        "discarded_sliver_volume_mm3": float(sliver_volume),
        "discarded_sliver_fraction": float(fraction),
        "max_sliver_fraction": float(max_sliver_fraction),
    }


def _shape_fix(cq, shape, *, precision_mm: float):
    """Run OpenCascade's general shape healer before STEP serialization."""
    from OCP.ShapeFix import ShapeFix_Shape

    fixer = ShapeFix_Shape(shape.wrapped)
    fixer.SetPrecision(float(precision_mm))
    fixer.SetMinTolerance(0.1 * float(precision_mm))
    fixer.SetMaxTolerance(10.0 * float(precision_mm))
    fixer.Perform()
    fixed = cq.Shape.cast(fixer.Shape())
    return fixed, {
        "performed": True,
        "precision_mm": float(precision_mm),
        "minimum_tolerance_mm": 0.1 * float(precision_mm),
        "maximum_tolerance_mm": 10.0 * float(precision_mm),
    }


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
    n_channels,
    x_lo,
    x_hi,
    clearance_m,
):
    """One channel void as a loft of ANNULAR-SEGMENT cross-sections.

    A regen channel is bounded by the cylindrical floor (liner OD) and
    ceiling (floor + h) — both surfaces of revolution — and two near-radial
    rib faces.  Building each loft section as an annular segment whose four
    corners lie ON those floor/ceiling radii (``r_floor``/``r_ceil``), rather
    than a flat frame-oriented rectangle, makes the void faces coincide with
    the body's revolved surfaces, so the Boolean cut is clean.  This matters
    for HELICAL channels: a flat rectangle's chord-vs-arc mismatch against the
    cylinders sheds many tiny sliver solids and the single-solid model fails;
    the annular segment cuts to one valid solid with no slivers.  The arc
    half-width is clamped below the half-pitch so neighbouring channels never
    overlap when the section is circular-patterned.
    """
    mask = (x >= x_lo) & (x <= x_hi)
    ids = np.flatnonzero(mask)
    if len(ids) < 4:
        raise ValueError("channel path has fewer than four loft sections")
    x = x[ids]
    r = r[ids]
    t_hot = t_hot[ids]
    width = width[ids]
    height = height[ids]

    # Resolve the helix finely enough that the angular step between loft
    # sections stays small (≈ 8°).  A ruled loft between widely-rotated
    # sections cuts a straight chord across the spiral, under-counting the
    # swept void volume (the void fraction otherwise shrinks as turns rise).
    # The channel sweeps ``helix_turns`` full turns over this active span.
    sweep_deg = 360.0 * abs(float(helix_turns))
    n_need = max(len(x), int(math.ceil(sweep_deg / 8.0)) + 1)
    if n_need > len(x):
        xn = np.linspace(float(x[0]), float(x[-1]), n_need)
        r = np.interp(xn, x, r)
        t_hot = np.interp(xn, x, t_hot)
        width = np.interp(xn, x, width)
        height = np.interp(xn, x, height)
        x = xn

    # Wall-normal offset curves shift both x and r.  Interpolate those curves
    # back onto common axial section planes so each arc lies on the same
    # revolved floor/ceiling surface used by the material envelope.
    xf, rf = normal_offset_contour(x, r, t_hot + clearance_m)
    xc, rc = normal_offset_contour(
        x, r, t_hot + height - clearance_m
    )
    xm, rm = normal_offset_contour(x, r, t_hot + 0.5 * height)
    r_floor = _radius_at_x(x, xf, rf)
    r_ceil = _radius_at_x(x, xc, rc)
    r_mid = _radius_at_x(x, xm, rm)
    frac = (x - x[0]) / max(float(x[-1] - x[0]), 1e-12)
    theta_c = 2.0 * math.pi * float(helix_turns) * frac
    half_pitch = math.pi / max(int(n_channels), 1)
    eff_width = np.maximum(width - 2.0 * clearance_m, 1e-6)
    stretch = helix_stretch_factors(
        x,
        r,
        helix_turns=helix_turns,
        t_wall=t_hot,
        channel_height=height,
    )
    half_arc = (
        eff_width * stretch
        / (2.0 * np.maximum(r_mid, 1e-9))
    )
    if np.any(half_arc >= half_pitch):
        raise ValueError(
            "channel width reaches or exceeds the circular pitch in the "
            "cooling-aware B-rep"
        )

    wires = []
    for i in range(len(x)):
        th0, th1 = theta_c[i] - half_arc[i], theta_c[i] + half_arc[i]
        wires.append(
            _annular_sector_wire(
                cq, x[i], r_floor[i], r_ceil[i], th0, th1
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


# --------------------------------------------------------------------------- #
# Full-N channel wall — lands as patterned positive solids and channels as     #
# gaps. This removes the N separate channel-cut operations; one final material #
# fuse is still required to produce a single neutral B-rep solid.              #
# --------------------------------------------------------------------------- #
def build_channel_wall_compound(
    profile: Any,
    *,
    max_sections: int = 24,
    end_seal_fraction: float = 0.015,
    bond_overlap: float = 50e-6,
):
    """Build patterned positive material for a full-N regenerative wall.

    The channels are the gaps between one patterned rib/land per pitch.  This
    avoids an expensive Boolean cut for every channel.  The returned compound
    is an intermediate topology containing liner, jacket, ribs, and two end
    seals; :func:`export_channel_wall_step` performs one kernel-level
    multi-shape fuse so the exported STEP is a single material solid.
    """
    cq = _cq()
    data = _clean_profile(profile, max_sections=max_sections)
    x = data["x"]; r = data["r"]; t_hot = data["t_hot"]
    w = data["w"]; h = data["h"]; t_jacket = data["t_jacket"]
    N = int(profile.channel_count)
    if N < 2:
        raise ValueError("channel wall needs at least two channels")
    helix = float(profile.helix_turns)

    # Resolve the helix finely enough that the rib twist between loft sections
    # stays small (≈ 8°/section), so the swept land volume is faithful.
    sweep_deg = 360.0 * abs(helix)
    n_need = max(len(x), int(math.ceil(sweep_deg / 8.0)) + 1)
    if n_need > len(x):
        xn = np.linspace(float(x[0]), float(x[-1]), n_need)
        r = np.interp(xn, x, r); t_hot = np.interp(xn, x, t_hot)
        w = np.interp(xn, x, w); h = np.interp(xn, x, h)
        t_jacket = np.interp(xn, x, t_jacket); x = xn

    xf, rf = normal_offset_contour(x, r, t_hot)
    xc, rc = normal_offset_contour(x, r, t_hot + h)
    xo, ro = normal_offset_contour(x, r, t_hot + h + t_jacket)
    xm, rm = normal_offset_contour(x, r, t_hot + 0.5 * h)
    r_floor = _radius_at_x(x, xf, rf)
    r_ceil = _radius_at_x(x, xc, rc)
    r_outer = _radius_at_x(x, xo, ro)
    r_mid = _radius_at_x(x, xm, rm)

    liner = _revolve_between(cq, x, r, x, r_floor)
    jacket = _revolve_between(cq, x, r_ceil, x, r_outer)
    if not (liner.isValid() and jacket.isValid()):
        raise RuntimeError("liner/jacket revolve is not a valid solid")
    envelope_volume = float(
        _revolve_between(cq, x, r, x, r_outer).Volume()
    )

    stretch = helix_stretch_factors(
        x,
        r,
        helix_turns=helix,
        t_wall=t_hot,
        channel_height=h,
    )
    half_arc = (
        w * stretch
        / (2.0 * np.maximum(r_mid, 1e-9))
    )
    if np.any(half_arc >= math.pi / N):
        raise ValueError(
            "channel width leaves zero or negative rib width at one or more "
            "stations"
        )
    frac = (x - x[0]) / max(float(x[-1] - x[0]), 1e-12)
    off = 2.0 * math.pi * helix * frac

    def _land_wire(i):
        # One rib spans from the right edge of channel 0 to the left edge of
        # channel 1: [off+half, off+2π/N−half], r_floor→r_ceil.
        a0 = off[i] + half_arc[i]
        a1 = off[i] + 2.0 * math.pi / N - half_arc[i]
        return _annular_sector_wire(
            cq,
            x[i],
            max(r_floor[i] - bond_overlap, r[i] + 0.1 * t_hot[i]),
            r_ceil[i] + bond_overlap,
            a0,
            a1,
        )

    base_land = cq.Solid.makeLoft([_land_wire(i) for i in range(len(x))], ruled=True)
    if not base_land.isValid():
        raise RuntimeError("base land loft is not a valid solid")
    lands = [
        cq.Shape.cast(shape)
        for shape in _pattern_wrapped_shapes(base_land, N)
    ]

    length = float(x[-1] - x[0])
    seal_length = max(
        float(end_seal_fraction) * length,
        2.0 * float(np.max(t_jacket)),
    )
    if 2.0 * seal_length >= length:
        raise ValueError("end seals consume the available channel length")
    inlet_seal = _annular_plenum(
        cq, x, r, t_hot, h,
        float(x[0]), float(x[0] + seal_length), -bond_overlap,
    )
    outlet_seal = _annular_plenum(
        cq, x, r, t_hot, h,
        float(x[-1] - seal_length), float(x[-1]), -bond_overlap,
    )
    parts = [liner, jacket, inlet_seal, outlet_seal] + lands
    compound = cq.Compound.makeCompound(parts)

    solids = compound.Solids()
    info = {
        "channel_count": N,
        "intermediate_solid_count": len(solids),
        "all_solids_valid": bool(all(s.isValid() for s in solids)),
        "helix_turns": helix,
        "envelope_volume_mm3": envelope_volume,
        "end_seal_length_m": seal_length,
        "bond_overlap_m": float(bond_overlap),
        "model": "patterned_positive_ribs_channels_as_gaps_intermediate",
    }
    return compound, info


def export_channel_wall_step(
    profile: Any,
    path: str | Path,
    *,
    max_sections: int = 24,
    end_seal_fraction: float = 0.015,
    bond_overlap: float = 50e-6,
    fuzzy_tolerance_mm: float = 1e-3,
    include_manifolds: bool = False,
    manifold_length_fraction: float = 0.06,
    ports_per_manifold: int = 4,
    port_area_ratio: float = 1.0,
    port_diameter: float | None = None,
    stl_path: str | Path | None = None,
) -> dict:
    """Export a full-N, single-solid regenerative wall STEP B-rep.

    One rib is patterned with shared OCC geometry and liner + jacket + ribs +
    end seals are joined by one parallel multi-shape fuse.  Thus the channels
    remain geometric gaps and no per-channel subtraction is required.

    With ``include_manifolds=True``, two annular plenums and a configurable
    set of radial ports are removed in one additional multi-tool cut.
    Default port area equals the total channel flow area by continuity; this
    is a geometry/hydraulic-area screen, not a manifold-maldistribution CFD
    model.
    """
    cq = _cq()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    compound, info = build_channel_wall_compound(
        profile,
        max_sections=max_sections,
        end_seal_fraction=end_seal_fraction,
        bond_overlap=bond_overlap,
    )
    body, fuse_kernel = _kernel_boolean(
        cq,
        compound.Solids(),
        operation="fuse",
        fuzzy_tolerance_mm=fuzzy_tolerance_mm,
        glue="shift",
    )
    if not body.isValid() or len(body.Solids()) != 1:
        # GlueShift is fast when the prescribed overlap is recognized.  Retry
        # the general intersection path before declaring the model invalid.
        body, fuse_kernel = _kernel_boolean(
            cq,
            compound.Solids(),
            operation="fuse",
            fuzzy_tolerance_mm=fuzzy_tolerance_mm,
            glue="off",
        )
    body, fuse_healing = _dominant_solid(cq, body)

    data = _clean_profile(profile, max_sections=max_sections)
    x = data["x"]; r = data["r"]; t_hot = data["t_hot"]
    w = data["w"]; h = data["h"]; t_jacket = data["t_jacket"]
    x_outer, r_outer = normal_offset_contour(
        x, r, t_hot + h + t_jacket
    )
    network_overlaps: dict[str, float] = {}
    cut_kernel = None
    manifold_metrics = None
    port_diameters: list[float] = []

    if include_manifolds:
        if not (0.0 < manifold_length_fraction < 0.25):
            raise ValueError("manifold_length_fraction must be in (0, 0.25)")
        if int(ports_per_manifold) < 1:
            raise ValueError("ports_per_manifold must be at least one")
        if port_area_ratio <= 0.0:
            raise ValueError("port_area_ratio must be positive")

        length = float(x[-1] - x[0])
        seal = float(info["end_seal_length_m"])
        plenum_length = max(
            manifold_length_fraction * length,
            2.5 * float(np.max(h)),
        )
        inlet_lo = float(x[0] + seal)
        inlet_hi = float(min(inlet_lo + plenum_length, x[-1] - 0.4 * length))
        outlet_hi = float(x[-1] - seal)
        outlet_lo = float(max(outlet_hi - plenum_length, x[0] + 0.4 * length))
        if outlet_lo <= inlet_hi:
            raise ValueError("no axial room remains between the two plenums")

        inlet_plenum = _annular_plenum(
            cq, x, r, t_hot, h, inlet_lo, inlet_hi, -bond_overlap
        )
        outlet_plenum = _annular_plenum(
            cq, x, r, t_hot, h, outlet_lo, outlet_hi, -bond_overlap
        )

        def build_ports(x_port, angular_offset):
            local_w = float(np.interp(x_port, x, w))
            local_h = float(np.interp(x_port, x, h))
            local_outer = float(np.interp(x_port, x_outer, r_outer))
            local_floor = float(np.interp(x_port, x, r + t_hot))
            total_channel_area = int(profile.channel_count) * local_w * local_h
            diameter = (
                float(port_diameter)
                if port_diameter is not None
                else math.sqrt(
                    4.0 * port_area_ratio * total_channel_area
                    / (math.pi * int(ports_per_manifold))
                )
            )
            if diameter <= 0.0:
                raise ValueError("port diameter must be positive")
            if diameter > plenum_length:
                raise ValueError(
                    f"area-sized port diameter {diameter*1e3:.2f} mm exceeds "
                    f"the {plenum_length*1e3:.2f} mm plenum length; increase "
                    "ports_per_manifold or manifold_length_fraction"
                )
            solids = []
            for j in range(int(ports_per_manifold)):
                theta = (
                    angular_offset
                    + 2.0 * math.pi * j / int(ports_per_manifold)
                )
                outward = np.array(
                    [0.0, math.cos(theta), math.sin(theta)]
                )
                start_radius_mm = (local_outer + diameter) * _MM
                end_radius_mm = (
                    local_floor + 0.35 * local_h
                ) * _MM
                start = (
                    np.array([x_port * _MM, 0.0, 0.0])
                    + outward * start_radius_mm
                )
                solids.append(
                    cq.Solid.makeCylinder(
                        0.5 * diameter * _MM,
                        start_radius_mm - end_radius_mm,
                        pnt=tuple(map(float, start)),
                        dir=tuple(map(float, -outward)),
                    )
                )
            return solids, diameter, total_channel_area

        inlet_x = 0.5 * (inlet_lo + inlet_hi)
        outlet_x = 0.5 * (outlet_lo + outlet_hi)
        representative_channel = _channel_loft(
            cq,
            x=x,
            r=r,
            t_hot=t_hot,
            width=w,
            height=h,
            helix_turns=float(profile.helix_turns),
            n_channels=int(profile.channel_count),
            x_lo=inlet_lo,
            x_hi=outlet_hi,
            clearance_m=0.5 * bond_overlap,
        )
        inlet_ports, inlet_diameter, inlet_channel_area = build_ports(
            inlet_x, 0.0
        )
        outlet_ports, outlet_diameter, outlet_channel_area = build_ports(
            outlet_x, math.pi / int(ports_per_manifold)
        )
        port_diameters = [inlet_diameter, outlet_diameter]

        tools = [
            inlet_plenum,
            outlet_plenum,
            *inlet_ports,
            *outlet_ports,
        ]
        body, cut_kernel = _kernel_boolean(
            cq,
            [body],
            tools,
            operation="cut",
            fuzzy_tolerance_mm=fuzzy_tolerance_mm,
            glue="off",
        )
        body, cut_healing = _dominant_solid(
            cq, body, max_sliver_fraction=2e-4
        )

        inlet_port_overlap = min(
            float(port.intersect(inlet_plenum).Volume())
            for port in inlet_ports
        )
        outlet_port_overlap = min(
            float(port.intersect(outlet_plenum).Volume())
            for port in outlet_ports
        )
        network_overlaps = {
            "channel_to_inlet_plenum_mm3": float(
                representative_channel.intersect(inlet_plenum).Volume()
            ),
            "channel_to_outlet_plenum_mm3": float(
                representative_channel.intersect(outlet_plenum).Volume()
            ),
            "minimum_inlet_port_to_plenum_mm3": inlet_port_overlap,
            "minimum_outlet_port_to_plenum_mm3": outlet_port_overlap,
        }
        if min(network_overlaps.values()) <= 1e-6:
            raise RuntimeError(
                f"port/plenum network is not connected: {network_overlaps}"
            )

        def plenum_area_ratio(x_port, channel_area):
            local_h = float(np.interp(x_port, x, h))
            local_mid = float(np.interp(x_port, x, r + t_hot + 0.5 * h))
            return 2.0 * math.pi * local_mid * local_h / max(
                channel_area, 1e-12
            )

        manifold_metrics = {
            "manifold_length_m": float(plenum_length),
            "ports_per_manifold": int(ports_per_manifold),
            "port_area_ratio_to_total_channels": float(port_area_ratio),
            "inlet_port_diameter_m": float(inlet_diameter),
            "outlet_port_diameter_m": float(outlet_diameter),
            "inlet_plenum_area_ratio_to_channels": float(
                plenum_area_ratio(inlet_x, inlet_channel_area)
            ),
            "outlet_plenum_area_ratio_to_channels": float(
                plenum_area_ratio(outlet_x, outlet_channel_area)
            ),
            "hydraulic_status":
                "continuity_area_screen_only_no_maldistribution_solution",
        }
    else:
        cut_healing = None

    body = body.clean()
    body, shape_fix = _shape_fix(
        cq, body, precision_mm=max(float(fuzzy_tolerance_mm), 1e-4)
    )
    body = body.clean()
    body, final_healing = _dominant_solid(cq, body)
    if not body.isValid():
        raise RuntimeError("final cleaned regenerative wall is not valid")

    inspection = None
    step_precision_mode = None
    for precision_mode in (0, 1, -1):
        body.exportStep(
            str(path),
            write_pcurves=True,
            precision_mode=precision_mode,
        )
        candidate = inspect_regen_step(path)
        if candidate["valid"] and candidate["single_solid"]:
            inspection = candidate
            step_precision_mode = precision_mode
            break
    if stl_path is not None:
        stl_path = Path(stl_path).expanduser().resolve()
        cq.exporters.export(body, str(stl_path), exportType="STL")

    if inspection is None:
        inspection = inspect_regen_step(path)
        raise RuntimeError(
            "STEP round-trip validation failed for precision modes "
            f"0, 1, and -1: {inspection}"
        )
    material_volume = float(body.Volume())
    envelope_volume = float(info["envelope_volume_mm3"])
    info.update({
        "path": str(path),
        "stl_path": str(stl_path) if stl_path is not None else None,
        "representation": "open_cascade_brep",
        "single_solid": True,
        "valid": True,
        "solid_count": 1,
        "material_volume_mm3": material_volume,
        "coolant_void_volume_mm3": envelope_volume - material_volume,
        "void_fraction": (
            envelope_volume - material_volume
        ) / max(envelope_volume, 1e-9),
        "include_manifolds": bool(include_manifolds),
        "manifold_metrics": manifold_metrics,
        "network_overlaps": network_overlaps,
        "port_diameters_m": port_diameters,
        "fuse_kernel": fuse_kernel,
        "fuse_healing": fuse_healing,
        "cut_kernel": cut_kernel,
        "cut_healing": cut_healing,
        "final_healing": final_healing,
        "shape_fix": shape_fix,
        "inspection": inspection,
        "step_precision_mode": step_precision_mode,
        "model": (
            "full_n_patterned_ribs_single_solid_with_plenums_and_ports"
            if include_manifolds
            else "full_n_patterned_ribs_single_solid_channels_as_gaps"
        ),
        "units": "CadQuery/OpenCascade millimetres; public API metres",
    })
    return info


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
    include_manifolds: bool = False,
    stl_path: str | Path | None = None,
) -> dict:
    """Build and export one cooling-aware regenerative STEP solid.

    The outer envelope is the fused liner+ribs+jacket material.  One lofted
    passage is circular-patterned by ``channel_count`` and Boolean-cut from
    it, leaving the liner + ribs + jacket as a single B-rep with real channel
    voids — the robust default.

    ``include_manifolds=True`` additionally cuts annular inlet/outlet plenums
    that intersect every passage and radial ports to the exterior, forming a
    connected coolant network.  Those extra curved booleans are markedly less
    robust in OpenCascade (they can leave the body non-manifold or shatter
    it), so they are opt-in; the default ships the validated channel-void
    solid.
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
    if include_manifolds:
        inlet_lo = float(x[0] + seal)
        inlet_hi = float(min(inlet_lo + plenum_length, x[-1] - 0.4 * length))
        outlet_hi = float(x[-1] - seal)
        outlet_lo = float(max(outlet_hi - plenum_length, x[0] + 0.4 * length))
        channel_lo = 0.5 * (inlet_lo + inlet_hi)
        channel_hi = 0.5 * (outlet_lo + outlet_hi)
        if channel_hi <= channel_lo:
            raise ValueError("no axial room remains between the two manifolds")
    else:
        # No manifolds: passages span nearly the full wall, sealed at the ends.
        channel_lo = float(x[0] + seal)
        channel_hi = float(x[-1] - seal)

    base_channel = _channel_loft(
        cq,
        x=x,
        r=r,
        t_hot=t_hot,
        width=w,
        height=h,
        helix_turns=float(profile.helix_turns),
        n_channels=n_channels,
        x_lo=channel_lo,
        x_hi=channel_hi,
        clearance_m=boolean_clearance,
    )
    if not base_channel.isValid():
        raise RuntimeError("base channel loft is invalid")
    channel_tools = _pattern_wrapped_shapes(base_channel, n_channels)
    body, channel_kernel = _kernel_boolean(
        cq,
        [envelope],
        channel_tools,
        operation="cut",
        fuzzy_tolerance_mm=fuzzy_tolerance_mm,
        glue="off",
    )
    if not body.isValid():
        raise RuntimeError("channel Boolean cut produced an invalid body")

    overlaps: dict = {}
    inlet_diameter = outlet_diameter = None
    inlet_plenum = outlet_plenum = inlet_port = outlet_port = None
    if include_manifolds:
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
    if not solids:
        raise RuntimeError("regen Boolean model produced no solids")
    # OCC fuzzy booleans on curved revolved faces can shed near-zero sliver
    # fragments alongside the real body.  Keep the dominant manifold solid and
    # tolerate only negligible slivers; a genuine split (a real second solid)
    # still fails.
    manifold = max(solids, key=lambda s: float(s.Volume()))
    manifold_volume = float(manifold.Volume())
    sliver_volume = sum(abs(float(s.Volume())) for s in solids if s is not manifold)
    if not manifold.isValid() or sliver_volume > 1e-4 * max(manifold_volume, 1e-12):
        volumes = sorted((float(s.Volume()) for s in solids), reverse=True)
        raise RuntimeError(
            "regen Boolean model is not one valid solid "
            f"(solid_count={len(solids)}, volumes_mm3={volumes[:8]})"
        )
    body = manifold

    if include_manifolds:
        # Connectivity checks on one representative passage are sufficient
        # because the circular pattern and annular plenums are exact symmetries.
        overlaps = {
            "channel_to_inlet_plenum_mm3": float(base_channel.intersect(inlet_plenum).Volume()),
            "channel_to_outlet_plenum_mm3": float(base_channel.intersect(outlet_plenum).Volume()),
            "inlet_port_to_plenum_mm3": float(inlet_port.intersect(inlet_plenum).Volume()),
            "outlet_port_to_plenum_mm3": float(outlet_port.intersect(outlet_plenum).Volume()),
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
        "include_manifolds": bool(include_manifolds),
        "manifold_length_m": float(plenum_length) if include_manifolds else None,
        "end_seal_length_m": float(seal),
        "inlet_port_diameter_m": float(inlet_diameter) if inlet_diameter else None,
        "outlet_port_diameter_m": float(outlet_diameter) if outlet_diameter else None,
        "envelope_volume_mm3": envelope_volume,
        "solid_volume_mm3": body_volume,
        "coolant_void_volume_mm3": removed_volume,
        "network_overlaps": overlaps,
        "channel_boolean_kernel": channel_kernel,
        "inspection": inspection,
        "units": "CadQuery kernel/STEP millimetres; public API metres",
        "model": ("fused_liner_ribs_jacket_with_manifolds_and_ports"
                  if include_manifolds else
                  "fused_liner_ribs_jacket_with_lofted_passage_voids"),
    }
