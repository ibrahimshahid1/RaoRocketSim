"""
export.py – CSV and STL export for nozzle contours.
"""

from __future__ import annotations
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


def export_stl(x: np.ndarray, y: np.ndarray, path: str | Path,
               n_angular: int = 64) -> Path:
    """
    Revolve the 2-D contour around the x-axis and write a binary STL.

    The contour (x, y) is interpreted as the generatrix in the meridional
    plane.  y is the radial distance from the centerline.

    Parameters
    ----------
    x, y       : contour arrays  [m]
    path       : output file path
    n_angular  : number of angular divisions around the axis (default 64)

    Returns
    -------
    Resolved Path of the written file.
    """
    path = Path(path).expanduser().resolve()

    theta = np.linspace(0, 2 * np.pi, n_angular + 1)  # +1 to close the loop
    n_axial = len(x)

    # Build the mesh:  for each axial segment and each angular segment
    # we have 2 triangles (a quad split diagonally).
    triangles = []
    for i in range(n_axial - 1):
        for j in range(n_angular):
            # 4 corners of the quad
            x0, r0 = x[i],     y[i]
            x1, r1 = x[i + 1], y[i + 1]
            t0 = theta[j]
            t1 = theta[j + 1]

            p00 = np.array([x0, r0 * np.cos(t0), r0 * np.sin(t0)])
            p10 = np.array([x1, r1 * np.cos(t0), r1 * np.sin(t0)])
            p01 = np.array([x0, r0 * np.cos(t1), r0 * np.sin(t1)])
            p11 = np.array([x1, r1 * np.cos(t1), r1 * np.sin(t1)])

            # triangle 1:  p00, p10, p11
            n1 = np.cross(p10 - p00, p11 - p00)
            norm1 = np.linalg.norm(n1)
            if norm1 > 0:
                n1 /= norm1
            triangles.append((n1, p00, p10, p11))

            # triangle 2:  p00, p11, p01
            n2 = np.cross(p11 - p00, p01 - p00)
            norm2 = np.linalg.norm(n2)
            if norm2 > 0:
                n2 /= norm2
            triangles.append((n2, p00, p11, p01))

    # Write binary STL
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)                           # 80-byte header
        f.write(struct.pack("<I", len(triangles)))       # triangle count
        for normal, v1, v2, v3 in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<fff", *v3))
            f.write(struct.pack("<H", 0))                # attribute byte count

    return path
