"""
export.py – CSV and STL export for nozzle contours.

STL supports two modes:
  - Inner surface only  (wall_thickness=None)  – for CAD/CAM reference
  - Solid nozzle body   (wall_thickness > 0)   – for direct manufacturing
    Includes outer wall, closed end-caps, and optional inlet flange.
"""

from __future__ import annotations
import json
import math
import struct
import numpy as np
from pathlib import Path


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

def _offset_contour(x: np.ndarray, y: np.ndarray,
                    thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset a 2-D contour outward by *thickness* along surface normals."""
    n = len(x)
    x_out = np.zeros(n)
    y_out = np.zeros(n)

    for i in range(n):
        # Central-difference tangent (forward/backward at ends)
        if i == 0:
            tx, ty = x[1] - x[0], y[1] - y[0]
        elif i == n - 1:
            tx, ty = x[-1] - x[-2], y[-1] - y[-2]
        else:
            tx, ty = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]

        length = math.sqrt(tx * tx + ty * ty)
        if length > 1e-15:
            # Outward normal: perpendicular to tangent, away from axis
            nx = -ty / length
            ny =  tx / length
        else:
            nx, ny = 0.0, 1.0

        x_out[i] = x[i] + thickness * nx
        y_out[i] = y[i] + thickness * ny

    return x_out, y_out


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
            t1 = theta[j + 1]

            p00 = np.array([x0, r0 * np.cos(t0), r0 * np.sin(t0)])
            p10 = np.array([x1, r1 * np.cos(t0), r1 * np.sin(t0)])
            p01 = np.array([x0, r0 * np.cos(t1), r0 * np.sin(t1)])
            p11 = np.array([x1, r1 * np.cos(t1), r1 * np.sin(t1)])

            if inward:
                # Winding for inward-facing normals (inner surface)
                a, b, c = p00, p11, p10
                d, e, f = p00, p01, p11
            else:
                # Winding for outward-facing normals (outer surface)
                a, b, c = p00, p10, p11
                d, e, f = p00, p11, p01

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


def _annular_cap(x_pos, r_inner, r_outer, theta, n_angular, direction):
    """Create an annular end-cap at *x_pos*.

    *direction* = -1 for inlet (normal in −x), +1 for exit (normal in +x).
    """
    triangles = []
    nx = np.array([float(direction), 0.0, 0.0])

    for j in range(n_angular):
        t0, t1 = theta[j], theta[j + 1]

        pi0 = np.array([x_pos, r_inner * np.cos(t0), r_inner * np.sin(t0)])
        pi1 = np.array([x_pos, r_inner * np.cos(t1), r_inner * np.sin(t1)])
        po0 = np.array([x_pos, r_outer * np.cos(t0), r_outer * np.sin(t0)])
        po1 = np.array([x_pos, r_outer * np.cos(t1), r_outer * np.sin(t1)])

        if direction > 0:
            # Exit cap: outward = +x
            triangles.append((nx, pi0, po0, po1))
            triangles.append((nx, pi0, po1, pi1))
        else:
            # Inlet cap: outward = −x
            triangles.append((nx, pi0, po1, po0))
            triangles.append((nx, pi0, pi1, po1))

    return triangles


def _write_stl(path: Path, triangles: list):
    """Write triangle list to binary STL."""
    with open(path, "wb") as f:
        header = b"RaoRocketSim nozzle" + b"\x00" * (80 - 19)
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(struct.pack("<H", 0))


def _solid_triangles(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                     n_angular: int, wall_thickness: float,
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
        _annular_cap(x[-1], y[-1], y_outer[-1], theta, n_angular, direction=+1)
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
        triangles.extend(
            _annular_cap(x_fl_end, y[0], r_flange, theta, n_angular,
                         direction=-1)
        )
        triangles.extend(
            _annular_cap(x_inlet, y_outer[0], r_flange, theta, n_angular,
                         direction=+1)
        )
    else:
        triangles.extend(
            _annular_cap(x[0], y[0], y_outer[0], theta, n_angular,
                         direction=-1)
        )

    return triangles


def _closed_profile(x: np.ndarray, y: np.ndarray, wall_thickness: float,
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


def _export_step_with_cadquery(profile: list[tuple[float, float]], path: Path) -> bool:
    try:
        import cadquery as cq  # type: ignore
    except Exception:
        return False

    try:
        solid = (
            cq.Workplane("XY")
            .polyline(profile)
            .close()
            .revolve(360.0, (0, 0, 0), (1, 0, 0))
        )
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
        "FILE_DESCRIPTION(('RaoRocketSim faceted nozzle solid'),'2;1');",
        "FILE_NAME('rao_nozzle.step','',('RaoRocketSim'),('RaoRocketSim'),'','','');",
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
    brep_id = entity(f"FACETED_BREP('RaoRocketSim nozzle',#{shell_id})")
    entity("GEOMETRIC_REPRESENTATION_CONTEXT(3)")
    entity(f"SHAPE_REPRESENTATION('RaoRocketSim nozzle',(#{brep_id}),#%d)" % (next_id - 1))

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
               wall_thickness: float | None = None,
               flange_od: float | None = None,
               flange_length: float | None = None) -> Path:
    """
    Revolve the 2-D contour around the x-axis and write a binary STL.

    Parameters
    ----------
    x, y           : contour arrays [m]  (y = radial distance from axis)
    path           : output file path
    n_angular      : angular divisions around the axis (default 64)
    wall_thickness : nozzle wall thickness [m].  If provided, the STL
                     is a closed solid body with inner flow surface,
                     outer surface, and sealed end-caps — ready for
                     3-D printing or CNC machining.
    flange_od      : outer diameter of an inlet mounting flange [m].
                     Only used when *wall_thickness* is set.
    flange_length  : axial extent of the flange from the inlet [m].

    Returns
    -------
    Resolved Path of the written file.
    """
    path = Path(path).expanduser().resolve()
    theta = np.linspace(0, 2 * np.pi, n_angular + 1)

    # ── Inner-surface-only mode (original behaviour) ──────────────
    if wall_thickness is None:
        triangles = _revolve_surface(x, y, theta, n_angular, inward=False)
        _write_stl(path, triangles)
        return path

    triangles = _solid_triangles(
        x, y, theta, n_angular, wall_thickness,
        flange_od=flange_od, flange_length=flange_length,
    )

    _write_stl(path, triangles)
    return path


def export_step(x: np.ndarray, y: np.ndarray, path: str | Path,
                n_angular: int = 64,
                wall_thickness: float | None = None,
                flange_od: float | None = None,
                flange_length: float | None = None,
                metadata: dict | None = None) -> Path:
    """
    Export a solid nozzle body as STEP.

    STEP export requires a positive wall thickness. CadQuery is used when
    available to create a true revolved B-rep; otherwise a faceted AP214 STEP
    fallback is written so CAD review can still proceed.
    """
    if wall_thickness is None or wall_thickness <= 0.0:
        raise ValueError("STEP export requires wall_thickness > 0")

    path = Path(path).expanduser().resolve()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_angular + 1)
    profile = _closed_profile(
        x, y, wall_thickness,
        flange_od=flange_od, flange_length=flange_length,
    )

    if not _export_step_with_cadquery(profile, path):
        triangles = _solid_triangles(
            x, y, theta, n_angular, wall_thickness,
            flange_od=flange_od, flange_length=flange_length,
        )
        _write_faceted_step(path, triangles, metadata=metadata)

    return path


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
            "RaoRocketSim does not write native IPT directly. Use Autodesk "
            "Inventor automation to import the STEP and save the IPT."
        ),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
