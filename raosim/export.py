"""
export.py – CSV and STL export for nozzle contours.

STL supports two modes:
  - Inner surface only  (wall_thickness=None)  – for CAD/CAM reference
  - Solid wall mesh     (wall_thickness > 0)   – a topology/volume-validated
    triangle solid with outer wall, closed end rings, and optional flange.
    Manufacturing qualification remains outside this exporter.
"""

from __future__ import annotations
import json
import math
import struct
import numpy as np
from pathlib import Path

from raosim.regen_profile import normal_offset_contour


def export_csv(x: np.ndarray, y: np.ndarray, path: str | Path,
               n_points: int | None = None) -> Path:
    """
    Write (x, y) contour to CSV.

    Parameters
    ----------
    x, y     : contour arrays  [m]
    path     : output file path
    n_points : if given, subsample to this many equally-spaced points

    Returns
    -------
    Resolved Path of the written file.
    """
    path = Path(path).expanduser().resolve()

    if n_points is not None and n_points < len(x):
        idx = np.linspace(0, len(x) - 1, n_points).astype(int)
        x, y = x[idx], y[idx]

    with open(path, "w") as f:
        f.write("x_m,y_m\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.8e},{yi:.8e}\n")

    return path


# ─────────────────────────────────────────────────────────────────────
# STL helpers
# ─────────────────────────────────────────────────────────────────────

def _thickness_array(thickness, n: int) -> np.ndarray:
    """Validate/broadcast a scalar or station-wise wall thickness."""
    arr = np.asarray(thickness, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    elif arr.shape != (n,):
        raise ValueError(f"wall_thickness array must have shape ({n},)")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("wall_thickness must be finite and positive at every station")
    return arr


def _offset_contour(x: np.ndarray, y: np.ndarray,
                    thickness) -> tuple[np.ndarray, np.ndarray]:
    """Offset a 2-D contour outward along surface normals.

    ``thickness`` may be one scalar or one value per contour station.  A
    station-wise array is the geometry bridge from the thermal/structural
    wall profile to the revolved solid.
    """
    t = _thickness_array(thickness, len(x))
    return normal_offset_contour(x, y, t)


def _revolve_surface(x, y, theta, n_angular, inward=False):
    """Revolve a contour and return triangle list.

    *inward=True* flips the winding so normals point toward the axis
    (used for the inner surface of a solid body).
    """
    triangles = []
    n_axial = len(x)

    for i in range(n_axial - 1):
        for j in range(n_angular):
            x0, r0 = x[i],     y[i]
            x1, r1 = x[i + 1], y[i + 1]
            t0 = theta[j]
            t1 = theta[(j + 1) % n_angular]

            p00 = np.array([x0, r0 * np.cos(t0), r0 * np.sin(t0)])
            p10 = np.array([x1, r1 * np.cos(t0), r1 * np.sin(t0)])
            p01 = np.array([x0, r0 * np.cos(t1), r0 * np.sin(t1)])
            p11 = np.array([x1, r1 * np.cos(t1), r1 * np.sin(t1)])

            if inward:
                # Winding for inward-facing normals (inner surface)
                a, b, c = p00, p10, p11
                d, e, f = p00, p11, p01
            else:
                # Winding for outward-facing normals (outer surface)
                a, b, c = p00, p11, p10
                d, e, f = p00, p01, p11

            n1 = np.cross(b - a, c - a)
            nm = np.linalg.norm(n1)
            if nm > 0:
                n1 /= nm
            triangles.append((n1, a, b, c))

            n2 = np.cross(e - d, f - d)
            nm2 = np.linalg.norm(n2)
            if nm2 > 0:
                n2 /= nm2
            triangles.append((n2, d, e, f))

    return triangles


def _ring_cap(x_inner, r_inner, x_outer, r_outer,
              theta, n_angular, direction):
    """Join inner and outer circular edges with a triangulated ring.

    *direction* = -1 for inlet (normal in −x), +1 for exit (normal in +x).
    The two circles may have different axial coordinates because a true
    surface-normal wall offset generally shifts both radius *and* x.
    """
    triangles = []
    nx = np.array([float(direction), 0.0, 0.0])

    for j in range(n_angular):
        t0, t1 = theta[j], theta[(j + 1) % n_angular]

        pi0 = np.array([x_inner, r_inner * np.cos(t0), r_inner * np.sin(t0)])
        pi1 = np.array([x_inner, r_inner * np.cos(t1), r_inner * np.sin(t1)])
        po0 = np.array([x_outer, r_outer * np.cos(t0), r_outer * np.sin(t0)])
        po1 = np.array([x_outer, r_outer * np.cos(t1), r_outer * np.sin(t1)])

        if direction > 0:
            # Exit cap: outward = +x
            vertices = ((pi0, po0, po1), (pi0, po1, pi1))
        else:
            # Inlet cap: outward = −x
            vertices = ((pi0, po1, po0), (pi0, pi1, po1))
        for a, b, c in vertices:
            normal = np.cross(b - a, c - a)
            magnitude = np.linalg.norm(normal)
            if magnitude > 0.0:
                normal /= magnitude
            triangles.append((normal, a, b, c))

    return triangles


def _annular_cap(x_pos, r_inner, r_outer, theta, n_angular, direction):
    """Create a planar annular end-cap at *x_pos*."""
    return _ring_cap(
        x_pos, r_inner, x_pos, r_outer, theta, n_angular, direction
    )


def _write_stl(path: Path, triangles: list):
    """Write triangle list to binary STL."""
    with open(path, "wb") as f:
        header = b"LREKit nozzle" + b"\x00" * (80 - 13)
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(struct.pack("<H", 0))


def _mesh_diagnostics(triangles: list, weld_tolerance: float = 0.0) -> dict:
    """Measure topology, orientation, and enclosed volume of STL triangles.

    Vertices are first rounded to binary-STL ``float32`` precision so this
    checks the artifact that is actually written, not only the source arrays.
    ``weld_tolerance`` may be used when inspecting third-party meshes whose
    nominally shared vertices differ by a small numerical tolerance.
    """
    from collections import Counter

    edge_counts: Counter = Counter()
    edge_orientation: Counter = Counter()
    signed_volume = 0.0
    degenerate = 0

    def vertex_id(vertex):
        value = np.asarray(vertex, dtype=np.float32)
        if weld_tolerance > 0.0:
            return tuple(
                np.rint(value.astype(float) / weld_tolerance).astype(np.int64)
            )
        return tuple(value.tolist())

    for _normal, v1, v2, v3 in triangles:
        vertices = [
            np.asarray(vertex, dtype=np.float32).astype(float)
            for vertex in (v1, v2, v3)
        ]
        ids = [vertex_id(vertex) for vertex in vertices]
        area_vector = np.cross(vertices[1] - vertices[0],
                               vertices[2] - vertices[0])
        if len(set(ids)) < 3 or np.linalg.norm(area_vector) == 0.0:
            degenerate += 1
            continue
        signed_volume += (
            np.dot(vertices[0], np.cross(vertices[1], vertices[2])) / 6.0
        )
        for start, end in (
            (ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])
        ):
            key = tuple(sorted((start, end)))
            edge_counts[key] += 1
            edge_orientation[key] += 1 if (start, end) == key else -1

    boundary = sum(count == 1 for count in edge_counts.values())
    nonmanifold = sum(count > 2 for count in edge_counts.values())
    inconsistent = sum(
        edge_counts[key] == 2 and orientation != 0
        for key, orientation in edge_orientation.items()
    )
    return {
        "triangle_count": len(triangles),
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "inconsistent_winding_edge_count": inconsistent,
        "degenerate_triangle_count": degenerate,
        "signed_volume_m3": float(signed_volume),
        "volume_m3": float(abs(signed_volume)),
        "watertight": (
            boundary == 0
            and nonmanifold == 0
            and inconsistent == 0
            and degenerate == 0
        ),
    }


def inspect_stl(path: str | Path, weld_tolerance: float = 0.0) -> dict:
    """Inspect a binary STL with the same gates used by :func:`export_stl`."""
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * count
    if len(data) != expected_size:
        raise ValueError(
            f"{path} is not a supported binary STL "
            f"(expected {expected_size} bytes, found {len(data)})"
        )
    triangles = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        normal = np.asarray(values[0:3], dtype=float)
        v1 = np.asarray(values[3:6], dtype=float)
        v2 = np.asarray(values[6:9], dtype=float)
        v3 = np.asarray(values[9:12], dtype=float)
        triangles.append((normal, v1, v2, v3))
        offset += 50
    return _mesh_diagnostics(triangles, weld_tolerance=weld_tolerance)


def _revolved_profile_volume(profile: list[tuple[float, float]]) -> float:
    """Exact volume enclosed by a piecewise-linear x-radius profile."""
    if len(profile) < 3:
        return 0.0
    closed = [*profile, profile[0]]
    integral = 0.0
    for (x0, r0), (x1, r1) in zip(closed, closed[1:]):
        integral += (
            math.pi
            * (x1 - x0)
            * (r0 * r0 + r0 * r1 + r1 * r1)
            / 3.0
        )
    return abs(integral)


def _validate_solid_mesh(
    triangles: list,
    profile: list[tuple[float, float]],
    n_angular: int,
) -> dict:
    """Reject open, inconsistently oriented, or wrong-volume solid meshes."""
    diagnostics = _mesh_diagnostics(triangles)
    if not diagnostics["watertight"]:
        raise RuntimeError(
            "solid STL topology validation failed: "
            f"{diagnostics['boundary_edge_count']} boundary edges, "
            f"{diagnostics['nonmanifold_edge_count']} non-manifold edges, "
            f"{diagnostics['inconsistent_winding_edge_count']} "
            "inconsistently wound edges, "
            f"{diagnostics['degenerate_triangle_count']} degenerate triangles"
        )
    angle = 2.0 * math.pi / n_angular
    expected_volume = (
        _revolved_profile_volume(profile) * math.sin(angle) / angle
    )
    actual_volume = diagnostics["volume_m3"]
    relative_error = abs(actual_volume - expected_volume) / max(
        expected_volume, 1e-30
    )
    diagnostics["expected_faceted_volume_m3"] = float(expected_volume)
    diagnostics["relative_volume_error"] = float(relative_error)
    if diagnostics["signed_volume_m3"] <= 0.0 or relative_error > 1e-5:
        raise RuntimeError(
            "solid STL volume/orientation validation failed: "
            f"signed volume={diagnostics['signed_volume_m3']:.9g} m^3, "
            f"expected={expected_volume:.9g} m^3, "
            f"relative error={relative_error:.3g}"
        )
    return diagnostics


def _solid_triangles(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                     n_angular: int, wall_thickness,
                     flange_od: float | None = None,
                     flange_length: float | None = None) -> list:
    """Build a closed faceted solid from an inner contour."""
    x_outer, y_outer = _offset_contour(x, y, wall_thickness)
    triangles: list = []
    triangles.extend(_revolve_surface(x, y, theta, n_angular, inward=True))
    triangles.extend(
        _revolve_surface(x_outer, y_outer, theta, n_angular, inward=False)
    )
    triangles.extend(
        _ring_cap(
            x[-1], y[-1], x_outer[-1], y_outer[-1],
            theta, n_angular, direction=+1,
        )
    )

    if flange_od is not None and flange_length is not None:
        r_flange = flange_od / 2.0
        x_inlet = x[0]
        x_fl_end = x_inlet - flange_length
        n_fl = 10
        x_fl = np.linspace(x_fl_end, x_inlet, n_fl)
        y_fl = np.full(n_fl, r_flange)
        triangles.extend(
            _revolve_surface(x_fl, y_fl, theta, n_angular, inward=False)
        )
        # The flange moves the inlet end-cap upstream.  Continue the hot-gas
        # bore to that cap; otherwise its inner circular edge is left open.
        x_bore = np.array([x_fl_end, x_inlet], dtype=float)
        y_bore = np.full(2, y[0], dtype=float)
        triangles.extend(
            _revolve_surface(x_bore, y_bore, theta, n_angular, inward=True)
        )
        triangles.extend(
            _annular_cap(x_fl_end, y[0], r_flange, theta, n_angular,
                         direction=-1)
        )
        triangles.extend(
            _ring_cap(
                x_outer[0], y_outer[0], x_inlet, r_flange,
                theta, n_angular, direction=+1,
            )
        )
    else:
        triangles.extend(
            _ring_cap(
                x[0], y[0], x_outer[0], y_outer[0],
                theta, n_angular, direction=-1,
            )
        )

    return triangles


def _closed_profile(x: np.ndarray, y: np.ndarray, wall_thickness,
                    flange_od: float | None = None,
                    flange_length: float | None = None) -> list[tuple[float, float]]:
    """Return a closed x-radius profile for CAD revolve operations."""
    x_outer, y_outer = _offset_contour(x, y, wall_thickness)
    profile: list[tuple[float, float]] = []

    if flange_od is not None and flange_length is not None:
        x_fl_end = float(x[0] - flange_length)
        profile.append((x_fl_end, float(y[0])))

    profile.extend((float(xi), float(yi)) for xi, yi in zip(x, y))
    profile.append((float(x_outer[-1]), float(y_outer[-1])))
    profile.extend(
        (float(xi), float(yi)) for xi, yi in zip(x_outer[-2::-1], y_outer[-2::-1])
    )

    if flange_od is not None and flange_length is not None:
        r_flange = float(flange_od / 2.0)
        profile.append((float(x[0]), r_flange))
        profile.append((float(x[0] - flange_length), r_flange))

    return profile


def _export_step_with_cadquery(
    profile: list[tuple[float, float]],
    path: Path,
    *,
    flange_length: float | None = None,
    bolt_count: int | None = None,
    bolt_circle_diameter: float | None = None,
    bolt_hole_diameter: float | None = None,
    seal_center_radius: float | None = None,
    seal_groove_width: float | None = None,
    seal_groove_depth: float | None = None,
) -> bool:
    try:
        import cadquery as cq  # type: ignore
    except Exception:
        return False

    try:
        # Public geometry is SI metres; CadQuery/OpenCascade STEP geometry is
        # conventionally millimetres.  Keep the kernel boundary explicit.
        profile_mm = [(1000.0 * x, 1000.0 * r) for x, r in profile]
        solid = (
            cq.Workplane("XY")
            .polyline(profile_mm)
            .close()
            .revolve(360.0, (0, 0, 0), (1, 0, 0))
        )

        # A chamber flange and injector face are one mechanical interface,
        # not two unrelated disks.  When a resolved pattern is supplied, cut
        # the chamber flange with the same count/BCD/hole diameter used by the
        # injector CAD.  The nozzle axis is +X, so the bolt circle lies in YZ.
        if bolt_count:
            if flange_length is None:
                raise ValueError("bolt holes require flange_length")
            if bolt_circle_diameter is None or bolt_hole_diameter is None:
                raise ValueError(
                    "bolt_count requires bolt_circle_diameter and "
                    "bolt_hole_diameter"
                )
            x_face = min(x for x, _r in profile_mm)
            length_mm = 1000.0 * float(flange_length)
            hole_r = 500.0 * float(bolt_hole_diameter)
            bolt_r = 500.0 * float(bolt_circle_diameter)
            eps = max(1.0e-3, 1.0e-4 * length_mm)
            for i in range(int(bolt_count)):
                theta = 2.0 * math.pi * i / int(bolt_count)
                cutter = cq.Solid.makeCylinder(
                    hole_r,
                    length_mm + 2.0 * eps,
                    cq.Vector(
                        x_face - eps,
                        bolt_r * math.cos(theta),
                        bolt_r * math.sin(theta),
                    ),
                    cq.Vector(1.0, 0.0, 0.0),
                )
                solid = solid.cut(cutter)

        # Normally the O-ring gland is cut in the injector face and this
        # flange supplies a continuous flat mating land.  An explicit positive
        # depth is supported for metal C-seal/gasket development where the
        # chamber side owns the gland; leaving it None preserves the physically
        # correct one-sided O-ring joint.
        if seal_groove_depth is not None and seal_groove_depth > 0.0:
            if seal_center_radius is None or seal_groove_width is None:
                raise ValueError(
                    "seal_groove_depth requires seal_center_radius and "
                    "seal_groove_width"
                )
            x_face = min(x for x, _r in profile_mm)
            center = 1000.0 * float(seal_center_radius)
            half_w = 500.0 * float(seal_groove_width)
            depth = 1000.0 * float(seal_groove_depth)
            outer = cq.Solid.makeCylinder(
                center + half_w, depth,
                cq.Vector(x_face, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0),
            )
            inner = cq.Solid.makeCylinder(
                center - half_w, depth,
                cq.Vector(x_face, 0.0, 0.0), cq.Vector(1.0, 0.0, 0.0),
            )
            solid = solid.cut(outer.cut(inner))
        cq.exporters.export(solid, str(path), exportType="STEP")
        return True
    except Exception:
        return False


def _write_faceted_step(path: Path, triangles: list,
                        metadata: dict | None = None) -> None:
    """Write a simple AP214 faceted STEP fallback."""
    metadata = metadata or {}
    lines: list[str] = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('LREKit faceted nozzle solid'),'2;1');",
        "FILE_NAME('rao_nozzle.step','',('LREKit'),('LREKit'),'','','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));",
        "ENDSEC;",
        "DATA;",
    ]

    next_id = 1

    def entity(text: str) -> int:
        nonlocal next_id
        idx = next_id
        lines.append(f"#{idx}={text};")
        next_id += 1
        return idx

    face_ids: list[int] = []
    for _normal, v1, v2, v3 in triangles:
        point_ids = [
            entity(
                "CARTESIAN_POINT('',("
                f"{float(v[0]):.9E},{float(v[1]):.9E},{float(v[2]):.9E}))"
            )
            for v in (v1, v2, v3)
        ]
        vertex_ids = [entity(f"VERTEX_POINT('',#{pid})") for pid in point_ids]
        loop_id = entity(
            "POLY_LOOP('',("
            + ",".join(f"#{vid}" for vid in vertex_ids)
            + "))"
        )
        bound_id = entity(f"FACE_OUTER_BOUND('',#{loop_id},.T.)")
        face_ids.append(entity(f"FACE('',(#{bound_id}))"))

    shell_id = entity(
        "CLOSED_SHELL('',("
        + ",".join(f"#{fid}" for fid in face_ids)
        + "))"
    )
    brep_id = entity(f"FACETED_BREP('LREKit nozzle',#{shell_id})")
    entity("GEOMETRIC_REPRESENTATION_CONTEXT(3)")
    entity(f"SHAPE_REPRESENTATION('LREKit nozzle',(#{brep_id}),#%d)" % (next_id - 1))

    for key, value in sorted(metadata.items()):
        safe_key = str(key).replace("'", "")
        safe_value = str(value).replace("'", "")
        lines.append(f"/* {safe_key}: {safe_value} */")

    lines.extend(["ENDSEC;", "END-ISO-10303-21;"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def export_stl(x: np.ndarray, y: np.ndarray, path: str | Path,
               n_angular: int = 64,
               wall_thickness=None,
               flange_od: float | None = None,
               flange_length: float | None = None) -> Path:
    """
    Revolve the 2-D contour around the x-axis and write a binary STL.

    Parameters
    ----------
    x, y           : contour arrays [m]  (y = radial distance from axis)
    path           : output file path
    n_angular      : angular divisions around the axis (default 64)
    wall_thickness : nozzle wall thickness [m], scalar or one value per
                     contour station.  If provided, the STL
                     is a closed solid body with inner flow surface,
                     outer surface, and sealed end rings. STL remains a
                     faceted exchange mesh, not an editable CAD B-rep or
                     manufacturing qualification.
    flange_od      : outer diameter of an inlet mounting flange [m].
                     Only used when *wall_thickness* is set.
    flange_length  : axial extent of the flange from the inlet [m].

    Returns
    -------
    Resolved Path of the written file.
    """
    path = Path(path).expanduser().resolve()
    if n_angular < 3:
        raise ValueError("n_angular must be at least 3")
    theta = 2.0 * np.pi * np.arange(n_angular) / n_angular

    # ── Inner-surface-only mode (original behaviour) ──────────────
    if wall_thickness is None:
        triangles = _revolve_surface(x, y, theta, n_angular, inward=False)
        _write_stl(path, triangles)
        return path

    triangles = _solid_triangles(
        x, y, theta, n_angular, wall_thickness,
        flange_od=flange_od, flange_length=flange_length,
    )
    profile = _closed_profile(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        wall_thickness,
        flange_od=flange_od,
        flange_length=flange_length,
    )
    _validate_solid_mesh(triangles, profile, n_angular)
    _write_stl(path, triangles)
    return path


def _clean_meridian_for_brep(
    x,
    y,
    thickness,
    max_pts: int = 400,
    *,
    throat_location: float | None = None,
    min_downstream_pts: int = 150,
):
    """Sort, de-seam and section-aware sample a meridian for the B-rep wire.

    The raw solver contour concatenates the convergent / throat / bell
    segments, whose ``x`` can DOUBLE BACK at the seams (non-monotone, with
    duplicates).  Fed straight to CadQuery the closed revolve wire then
    self-intersects and the true-B-rep export silently fails to the faceted
    fallback. Sorting and dropping duplicates yields a clean simple wire.

    For a full thrust chamber, every injector-to-throat station is retained so
    STEP construction preserves the chamber's exact polyline-frustum volume.
    Only the divergent nozzle is downsampled. The station-wise ``thickness`` is
    carried with the retained points so a variable ``t_hot(x)`` survives.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = _thickness_array(thickness, len(x))
    order = np.argsort(x, kind="stable")
    xs, ys, ts = x[order], y[order], t[order]
    span = max(float(xs.max() - xs.min()), 1e-12)
    keep = np.concatenate([[True], np.abs(np.diff(xs)) > 1e-9 * span])
    xs, ys, ts = xs[keep], ys[keep], ts[keep]
    if len(xs) > max_pts and throat_location is not None:
        throat_idx = int(np.argmin(np.abs(xs - float(throat_location))))
        upstream = np.arange(throat_idx + 1, dtype=int)
        downstream_all = np.arange(throat_idx + 1, len(xs), dtype=int)
        downstream_count = min(
            len(downstream_all),
            max(
                int(min_downstream_pts),
                int(max_pts) - len(upstream),
            ),
        )
        if downstream_count:
            downstream = downstream_all[
                np.unique(
                    np.linspace(
                        0, len(downstream_all) - 1, downstream_count
                    ).astype(int)
                )
            ]
            idx = np.concatenate((upstream, downstream))
        else:
            idx = upstream
        xs, ys, ts = xs[idx], ys[idx], ts[idx]
    elif len(xs) > max_pts:
        idx = np.unique(np.linspace(0, len(xs) - 1, max_pts).astype(int))
        xs, ys, ts = xs[idx], ys[idx], ts[idx]
    return xs, ys, ts


def export_step(x: np.ndarray, y: np.ndarray, path: str | Path,
                n_angular: int = 64,
                wall_thickness=None,
                flange_od: float | None = None,
                flange_length: float | None = None,
                bolt_count: int | None = None,
                bolt_circle_diameter: float | None = None,
                bolt_hole_diameter: float | None = None,
                seal_center_radius: float | None = None,
                seal_groove_width: float | None = None,
                seal_groove_depth: float | None = None,
                metadata: dict | None = None,
                require_brep: bool = False,
                throat_location: float | None = None) -> Path:
    """
    Export a solid revolved wall body as STEP.

    STEP export requires a positive scalar or station-wise wall thickness.
    CadQuery is used when available to create a true revolved B-rep; otherwise
    a faceted AP214 STEP fallback is written so CAD review can still proceed.
    Set ``require_brep=True`` to reject that triangle-based fallback.
    """
    if wall_thickness is None:
        raise ValueError("STEP export requires wall_thickness > 0")

    path = Path(path).expanduser().resolve()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _thickness_array(wall_thickness, len(x))
    theta = 2.0 * np.pi * np.arange(n_angular) / n_angular
    # The B-rep revolve needs a clean simple wire; the faceted fallback uses
    # the full-resolution contour as-is.
    xb, yb, tb = _clean_meridian_for_brep(
        x, y, wall_thickness, throat_location=throat_location
    )
    profile = _closed_profile(
        xb, yb, tb,
        flange_od=flange_od, flange_length=flange_length,
    )

    has_interface_features = bool(
        bolt_count
        or seal_center_radius is not None
        or seal_groove_width is not None
        or (seal_groove_depth is not None and seal_groove_depth > 0.0)
    )
    if bolt_count:
        if flange_od is None or flange_length is None:
            raise ValueError("bolt holes require flange_od and flange_length")
        if bolt_circle_diameter is None or bolt_hole_diameter is None:
            raise ValueError(
                "bolt_count requires bolt_circle_diameter and "
                "bolt_hole_diameter"
            )
        if int(bolt_count) < 3:
            raise ValueError("bolt_count must be at least 3")
        edge = (
            0.5 * float(flange_od)
            - 0.5 * float(bolt_circle_diameter)
            - 0.5 * float(bolt_hole_diameter)
        )
        if edge <= 0.0:
            raise ValueError("bolt pattern breaks through the flange outer edge")
    if seal_center_radius is not None or seal_groove_width is not None:
        if seal_center_radius is None or seal_groove_width is None:
            raise ValueError(
                "seal_center_radius and seal_groove_width must be supplied together"
            )
        if flange_od is None:
            raise ValueError("seal mating land requires flange_od")
        seal_inner = float(seal_center_radius) - 0.5 * float(seal_groove_width)
        seal_outer = float(seal_center_radius) + 0.5 * float(seal_groove_width)
        if seal_inner <= float(y[0]) or seal_outer >= 0.5 * float(flange_od):
            raise ValueError("seal envelope does not fit on the chamber flange land")
        if bolt_count and bolt_circle_diameter and bolt_hole_diameter:
            bolt_inner = (
                0.5 * float(bolt_circle_diameter)
                - 0.5 * float(bolt_hole_diameter)
            )
            if seal_outer >= bolt_inner:
                raise ValueError("seal envelope overlaps the chamber bolt holes")

    if has_interface_features:
        exported = _export_step_with_cadquery(
            profile,
            path,
            flange_length=flange_length,
            bolt_count=bolt_count,
            bolt_circle_diameter=bolt_circle_diameter,
            bolt_hole_diameter=bolt_hole_diameter,
            seal_center_radius=seal_center_radius,
            seal_groove_width=seal_groove_width,
            seal_groove_depth=seal_groove_depth,
        )
    else:
        # Keep this two-argument call for callers/tests that monkeypatch the
        # optional CadQuery backend.
        exported = _export_step_with_cadquery(profile, path)

    if not exported:
        if has_interface_features:
            raise RuntimeError(
                "chamber bolt/seal interface features require a true "
                "CadQuery/OpenCascade B-rep; a faceted STEP cannot carry "
                "the shared injector pattern"
            )
        if require_brep:
            raise RuntimeError(
                "true B-rep STEP export requires CadQuery/OpenCascade; "
                "install the optional 'cadquery' dependency or omit require_brep"
            )
        triangles = _solid_triangles(
            x, y, theta, n_angular, wall_thickness,
            flange_od=flange_od, flange_length=flange_length,
        )
        _write_faceted_step(path, triangles, metadata=metadata)

    inspection = None
    if step_representation(path) == "brep":
        try:
            import cadquery as cq  # type: ignore

            imported = cq.importers.importStep(str(path))
            solids = [s for value in imported.vals() for s in value.Solids()]
            inspection = {
                "solid_count": len(solids),
                "all_solids_valid": bool(solids) and all(
                    solid.isValid() for solid in solids
                ),
                "volume_mm3": float(sum(abs(s.Volume()) for s in solids)),
            }
            if not (
                inspection["solid_count"] == 1
                and inspection["all_solids_valid"]
                and inspection["volume_mm3"] > 0.0
            ):
                raise RuntimeError(
                    f"wall STEP round-trip gate failed: {inspection}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "wall STEP round-trip inspection failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    sidecar = path.with_suffix(".cad.json")
    sidecar.write_text(json.dumps({
        "schema": "raosim.cad_export.v1",
        "artifact": path.name,
        "public_api_linear_unit": "m",
        "neutral_file_linear_unit": "mm",
        "volume_unit": "mm^3",
        "representation": step_representation(path),
        "round_trip_inspection": inspection,
        "interface": {
            "flange_outer_diameter_m": flange_od,
            "flange_length_m": flange_length,
            "bolt_count": int(bolt_count) if bolt_count else None,
            "bolt_circle_diameter_m": bolt_circle_diameter,
            "bolt_hole_diameter_m": bolt_hole_diameter,
            "seal_center_radius_m": seal_center_radius,
            "seal_groove_width_m": seal_groove_width,
            "seal_groove_depth_m": seal_groove_depth,
            "seal_ownership": (
                "chamber_gland" if seal_groove_depth else
                "injector_gland_chamber_flat_mating_land"
                if seal_center_radius is not None else None
            ),
        },
        "metadata": metadata or {},
    }, indent=2) + "\n", encoding="utf-8")

    return path


def step_representation(path: str | Path) -> str:
    """Classify a generated STEP artifact as ``brep`` or ``faceted_brep``."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return "faceted_brep" if "FACETED_BREP" in text else "brep"


def package_ipt_request(step_path: str | Path, path: str | Path,
                        metadata: dict | None = None) -> Path:
    """
    Write an Inventor IPT conversion manifest.

    Native IPT is proprietary and is not written directly by this project.
    The manifest records the authoritative STEP file and the metadata needed
    by a downstream Inventor automation task to create/package an IPT.
    """
    step_path = Path(step_path).expanduser().resolve()
    path = Path(path).expanduser().resolve()
    manifest = {
        "status": "inventor_conversion_required",
        "authoritative_step": str(step_path),
        "ipt_target": str(path.with_suffix(".ipt")),
        "note": (
            "LREKit does not write native IPT directly. Use Autodesk "
            "Inventor automation to import the STEP and save the IPT."
        ),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
