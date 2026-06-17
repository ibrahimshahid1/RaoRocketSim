"""
regen_geometry.py — 3-D nozzle geometry with regenerative cooling channels.

Builds a 3-D surface of revolution of the nozzle wall plus the N
regenerative cooling channels (the "coils") wrapping it, consistent
with the cooling-analysis geometry: each channel is an annular-segment
cross-section of width ``w`` (arc) × height ``h`` (radial), seated on
the inner wall at radius ``r_inner + t_wall``, with the channel count
``N`` and pitch ``2πr/N`` from the :class:`CoolingSpec`.  Channels run
axially (``helix_turns=0``, matching the 1-D/2-D/3-D cooling model's
straight-channel assumption) or helically (``helix_turns>0``, the
spiral "coil" look of tube-wall regen).

Outputs: a binary STL of the wall + channels for CAD/print, and a
Matplotlib 3-D render.  Pure NumPy + Matplotlib (no JAX).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Wall + channel geometry                                                      #
# --------------------------------------------------------------------------- #
def _wall_profile(contour: dict, region: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """Return the meridional wall profile ``(x, r_inner)``.

    ``region="full"`` uses the whole contour (convergent + throat +
    bell); ``"bell"`` uses just the divergent bell wall.
    """
    if region == "bell" and "x_bell" in contour:
        x = np.asarray(contour["x_bell"], dtype=float)
        r = np.asarray(contour["y_bell"], dtype=float)
    else:
        x = np.asarray(contour["x"], dtype=float)
        r = np.asarray(contour["y"], dtype=float)
    order = np.argsort(x)
    return x[order], r[order]


def nozzle_wall_surface(x: np.ndarray, r: np.ndarray, n_theta: int = 120):
    """Surface of revolution of the inner (hot-gas) wall: ``(n_x, n_theta, 3)``."""
    th = np.linspace(0.0, 2.0 * math.pi, n_theta)
    X = x[:, None] * np.ones_like(th)[None, :]
    Y = r[:, None] * np.cos(th)[None, :]
    Z = r[:, None] * np.sin(th)[None, :]
    return np.stack([X, Y, Z], axis=-1)


def regen_channel_rails(
    x: np.ndarray,
    r_inner: np.ndarray,
    *,
    n_channels: int,
    channel_width: float,
    channel_height: float,
    wall_thickness: float,
    helix_turns: float = 0.0,
    region_fraction: tuple[float, float] = (0.0, 1.0),
) -> list[np.ndarray]:
    """The four 3-D corner rails of each channel's annular-segment tube.

    Returns a list of ``n_channels`` arrays, each ``(n_x, 4, 3)``: the
    floor−, floor+, ceil+, ceil− corner rails along the channel (suited
    to building swept quad faces).  Channels span the axial fraction
    ``region_fraction`` of the wall.  The arc half-width ``Δθ/2 = w/(2
    r_c)`` is clamped below the half-pitch so channels never overlap.
    """
    n = len(x)
    x0, x1 = float(x[0]), float(x[-1])
    L = max(x1 - x0, 1e-12)
    frac = (x - x0) / L
    lo, hi = region_fraction
    in_region = (frac >= lo) & (frac <= hi)

    r_floor = r_inner + wall_thickness
    r_ceil = r_floor + channel_height
    r_mid = r_floor + 0.5 * channel_height
    pitch_angle = 2.0 * math.pi / max(n_channels, 1)
    half_arc = np.minimum(channel_width / (2.0 * np.maximum(r_mid, 1e-9)),
                          0.45 * pitch_angle)

    channels: list[np.ndarray] = []
    for j in range(n_channels):
        theta0 = 2.0 * math.pi * j / n_channels
        theta = theta0 + 2.0 * math.pi * helix_turns * frac
        rails = np.zeros((n, 4, 3))
        for c, (rr, sgn) in enumerate([(r_floor, -1.0), (r_floor, +1.0),
                                       (r_ceil, +1.0), (r_ceil, -1.0)]):
            th = theta + sgn * half_arc
            rails[:, c, 0] = x
            rails[:, c, 1] = rr * np.cos(th)
            rails[:, c, 2] = rr * np.sin(th)
        # Collapse the channel onto the wall outside its active region so
        # it neither renders nor exports there.
        rails[~in_region] = np.stack([
            x[~in_region],
            (r_floor[~in_region]) * np.cos(2.0 * math.pi * j / n_channels
                                           + 2.0 * math.pi * helix_turns * frac[~in_region]),
            (r_floor[~in_region]) * np.sin(2.0 * math.pi * j / n_channels
                                           + 2.0 * math.pi * helix_turns * frac[~in_region]),
        ], axis=-1)[:, None, :]
        channels.append(rails)
    return channels


def _channel_triangles(rails: np.ndarray) -> list[tuple]:
    """Triangulate one channel's four swept side faces (floor, +side,
    ceiling, −side) into a closed tube surface."""
    n = rails.shape[0]
    tris: list[tuple] = []
    # face loops: (0,1) floor, (1,2) +side, (2,3) ceiling, (3,0) −side
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        for i in range(n - 1):
            p0, p1 = rails[i, a], rails[i, b]
            p2, p3 = rails[i + 1, a], rails[i + 1, b]
            tris.append((p0, p1, p3))
            tris.append((p0, p3, p2))
    return tris


# --------------------------------------------------------------------------- #
# Export + render                                                              #
# --------------------------------------------------------------------------- #
def _surface_triangles(verts: np.ndarray) -> list[tuple]:
    n_i, n_j, _ = verts.shape
    tris: list[tuple] = []
    for i in range(n_i - 1):
        for j in range(n_j - 1):
            a, b = verts[i, j], verts[i, j + 1]
            c, d = verts[i + 1, j], verts[i + 1, j + 1]
            tris.append((a, b, c))
            tris.append((b, d, c))
    return tris


def write_stl(path: Path, triangle_groups: list[list[tuple]], name: str = "regen_nozzle"):
    """Binary STL from triangle groups (wall + each channel)."""
    tris = [t for group in triangle_groups for t in group]
    with open(path, "wb") as f:
        f.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
            f.write(struct.pack("<3f", *nrm))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return len(tris)


def plot_regen_3d(wall_verts, channel_rails, *, save_path=None,
                  show=False, scale=1.0e3, wall_alpha=0.18,
                  station_temperature=None, station_x=None,
                  temperature_label="wall T [K]"):
    """Render the (semi-transparent) wall + the channels in 3-D.

    If ``station_temperature`` (per-axial-station scalar, e.g. the
    gas-side wall temperature from a cooling analysis) and its
    ``station_x`` are given, the channels are coloured by that field
    (interpolated to the geometry's x), turning the model into a
    flowfield-style hot-spot map; otherwise channels are coloured by
    index.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    W = wall_verts * scale
    ax.plot_surface(W[..., 0], W[..., 1], W[..., 2], color="0.6",
                    alpha=wall_alpha, linewidth=0, shade=True, zorder=0)

    by_temp = station_temperature is not None and station_x is not None
    if by_temp:
        gx = channel_rails[0][:, 0, 0]            # geometry x (per station)
        Tg = np.interp(gx, np.asarray(station_x), np.asarray(station_temperature))
        norm = mcolors.Normalize(vmin=float(np.nanmin(Tg)), vmax=float(np.nanmax(Tg)))
        hot = matplotlib.colormaps["inferno"]
    cmap = plt.get_cmap("turbo")
    quads = []
    colors = []
    for k, rails in enumerate(channel_rails):
        rr = rails * scale
        idx_col = cmap((k % len(channel_rails)) / max(len(channel_rails) - 1, 1))
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            for i in range(rr.shape[0] - 1):
                quads.append([rr[i, a], rr[i, b], rr[i + 1, b], rr[i + 1, a]])
                colors.append(hot(norm(0.5 * (Tg[i] + Tg[i + 1]))) if by_temp
                              else idx_col)
    pc = Poly3DCollection(quads, facecolors=colors, edgecolors="none", alpha=0.95)
    ax.add_collection3d(pc)
    if by_temp:
        m = cm.ScalarMappable(norm=norm, cmap=hot); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.6, label=temperature_label)

    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    title = f"Regen-cooled nozzle — {len(channel_rails)} channels"
    ax.set_title(title + (" (coloured by wall temperature)" if by_temp else ""))
    try:
        rng = np.ptp(W.reshape(-1, 3), axis=0)
        ax.set_box_aspect(tuple(np.maximum(rng, 1e-6)))
    except Exception:
        pass
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def generate_regen_nozzle(
    contour: dict,
    cooling: Any,
    wall_thickness: float,
    *,
    helix_turns: float = 0.0,
    region: str = "full",
    region_fraction: tuple[float, float] = (0.0, 1.0),
    n_theta: int = 120,
    stl_path: str | Path | None = None,
    png_path: str | Path | None = None,
    cooling_result: dict | None = None,
) -> dict:
    """One-call 3-D regen-nozzle generator.

    Builds the wall surface + the ``N`` channels (annular-segment tubes,
    ``helix_turns`` controls axial-vs-coil), optionally writes a binary
    STL and a 3-D PNG, and returns the geometry handles + a summary.
    """
    x, r_inner = _wall_profile(contour, region=region)
    N = int(getattr(cooling, "channel_count", 0) or 0)
    w = float(getattr(cooling, "channel_width", 0.0) or 0.0)
    h = float(getattr(cooling, "channel_height", 0.0) or 0.0)
    if N <= 0 or w <= 0.0 or h <= 0.0:
        raise ValueError("regen geometry needs positive channel_count/width/height")

    # Channel-fit check at the tightest (smallest-radius) station.
    r_mid_min = float(np.min(r_inner)) + wall_thickness + 0.5 * h
    pitch_min = 2.0 * math.pi * r_mid_min / N
    fits = (N * w) <= 2.0 * math.pi * (float(np.min(r_inner)) + wall_thickness)

    wall_verts = nozzle_wall_surface(x, r_inner, n_theta=n_theta)
    rails = regen_channel_rails(
        x, r_inner, n_channels=N, channel_width=w, channel_height=h,
        wall_thickness=wall_thickness, helix_turns=helix_turns,
        region_fraction=region_fraction)

    summary = {
        "n_channels": N, "channel_width": w, "channel_height": h,
        "wall_thickness": float(wall_thickness), "helix_turns": float(helix_turns),
        "channels_fit": bool(fits), "min_pitch": float(pitch_min),
        "x_range": (float(x[0]), float(x[-1])),
        "exit_radius": float(r_inner[-1]),
    }

    out: dict = {"x": x, "r_inner": r_inner, "wall_verts": wall_verts,
                 "channel_rails": rails, "summary": summary}

    if stl_path is not None:
        groups = [_surface_triangles(wall_verts)]
        groups += [_channel_triangles(rr) for rr in rails]
        n_tri = write_stl(Path(stl_path), groups)
        out["stl_path"] = str(stl_path)
        out["n_triangles"] = n_tri
    if png_path is not None:
        st_T = st_x = None
        if cooling_result is not None:
            st_T = cooling_result.get("gas_side_wall_temperature")
            st_x = cooling_result.get("x")
        fig = plot_regen_3d(wall_verts, rails, save_path=str(png_path),
                            station_temperature=st_T, station_x=st_x)
        fig.clf()
        out["png_path"] = str(png_path)
    return out
