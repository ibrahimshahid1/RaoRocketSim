"""
moc_diagrams.py — construction diagrams for the Rao/MOC nozzle build.

Where ``flow_viz`` renders the *solved field* (Mach/p/theta/T contours,
streamlines), this module renders the *construction* itself — the geometry
and characteristic topology the solver actually builds:

* :func:`plot_kernel_expansion_fan` — the throat **kernel**: the TT'
  starting line and the right-running characteristics (RRCs) fanning
  through the expansion arc up to the wall point B (the throat expansion
  fan of the method of characteristics).
* :func:`plot_rao_topology` — the annotated **Rao B-D-E topology**: the
  throat-arc wall (0→B), the last kernel RRC BD (B→D), the control
  surface / left-running characteristic DE (D→E), and the wall
  streamline (B→E), with the mass-flux balance ``mdot_BD == mdot_DE``
  that fixes the optimum contour (Rao 1958).
* :func:`plot_bde_mesh` — the **BDE characteristic net**: the interior
  mesh NASA's ``CalcBDERegion``/``CalcRemainingMesh`` marches from the
  control surface down to the axis, drawn as its two characteristic
  families.  A near-axis zoom makes visible the singular axis
  convergence (where rows are legitimately negative-r truncated) — the
  region the topology audit deliberately does *not* treat as a folded
  cell.
* :func:`plot_bde_diagnostics` — solved BDE-node Mach, pressure, and flow
  angle contours together with explicit C+/C- compatibility residuals,
  wall-to-axis row mass conservation, and the local same-family crossing
  audit.  This is the numerical companion to the topology-only mesh plot.
* :func:`plot_bde_integrity` — a compact four-panel numerical audit of
  Mach-line geometry/compatibility, oriented cell areas, the axis state,
  field smoothness, and several interpolated axial mass-flow cuts.

These figures read the in-memory ``bde_artifacts`` the ``bde`` wall method
stashes on ``solution.construction_diagnostics`` (kernel, NASA topology,
BDE region).  Each follows the ``flow_viz`` conventions: pass
``save_path=...`` to write a PNG (Agg backend) or ``show=True`` for an
interactive window; the Matplotlib ``Figure`` is returned.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Artifact access                                                              #
# --------------------------------------------------------------------------- #
def _bde_artifacts(solution) -> dict:
    """Return the stashed BDE construction artifacts or raise ValueError."""
    diag = getattr(solution, "construction_diagnostics", None) or {}
    art = diag.get("bde_artifacts")
    if not isinstance(art, dict) or art.get("bde_region") is None:
        raise ValueError(
            "no BDE construction artifacts on the solution — the diagram needs "
            "a wall_method='bde' solve run with evaluate_moc=True"
        )
    return art


def _xy(nodes) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([float(n.x) for n in nodes], dtype=float)
    rs = np.array([float(n.r) for n in nodes], dtype=float)
    return xs, rs


def _new_ax(save_path, show, figsize):
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, save_path, show):
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# 1. Throat kernel / expansion fan                                             #
# --------------------------------------------------------------------------- #
def plot_kernel_expansion_fan(
    solution,
    *,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Draw the throat kernel: the TT' starting line and the RRCs fanning
    through the expansion arc to the wall point B, nodes coloured by Mach."""
    plt = _new_ax(save_path, show, (10, 5))
    from matplotlib import cm, colors as mcolors

    art = _bde_artifacts(solution)
    kernel = art["kernel"]
    rrcs = [rrc for rrc in getattr(kernel, "rrcs", []) if rrc]
    if len(rrcs) < 2:
        raise ValueError("kernel has too few RRCs to draw an expansion fan")

    all_M = [float(n.M) for rrc in rrcs for n in rrc]
    norm = mcolors.Normalize(min(all_M), max(all_M))
    cmap = plt.get_cmap("turbo")

    fig, ax = plt.subplots(figsize=(10, 5))
    # RRCs (right-running characteristics) — the fan lines.
    for k, rrc in enumerate(rrcs):
        xs, rs = _xy(rrc)
        ax.plot(xs, rs, color="0.6", lw=0.4, alpha=0.35, zorder=1)
    # TT' starting line and BD (final RRC) highlighted.
    x0, r0 = _xy(rrcs[0]); ax.plot(x0, r0, color="tab:blue", lw=2.0,
                                   label="TT' starting line", zorder=3)
    xb, rb = _xy(rrcs[-1]); ax.plot(xb, rb, color="tab:red", lw=2.0,
                                    label="BD (final kernel RRC)", zorder=3)
    # nodes coloured by Mach.
    xN = [float(n.x) for rrc in rrcs for n in rrc]
    rN = [float(n.r) for rrc in rrcs for n in rrc]
    mN = [float(n.M) for rrc in rrcs for n in rrc]
    ax.scatter(xN, rN, c=mN, cmap=cmap, norm=norm, s=5, zorder=2,
               edgecolors="none")
    # wall (first node of each RRC = wall side).
    wx = [float(rrc[0].x) for rrc in rrcs]
    wr = [float(rrc[0].r) for rrc in rrcs]
    ax.plot(wx, wr, color="k", lw=1.8, zorder=4, label="expansion-arc wall")
    ax.plot([float(kernel.B.x)], [float(kernel.B.r)], "k^", ms=9, zorder=5)
    ax.annotate("B", (float(kernel.B.x), float(kernel.B.r)),
                textcoords="offset points", xytext=(6, 6), fontsize=11,
                fontweight="bold")

    ax.axhline(0.0, color="0.7", lw=0.5, ls=":")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 pad=0.01, fraction=0.045, label="Mach number")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    thb = math.degrees(float(getattr(kernel, "theta_B", 0.0)))
    ax.set_title(title or
                 f"Throat kernel — MOC expansion fan  (θ_B ≈ {thb:.2f}°, "
                 f"{len(rrcs)} RRCs)", fontsize=11)
    ax.legend(loc="upper center", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# 2. Rao B-D-E topology                                                        #
# --------------------------------------------------------------------------- #
def plot_rao_topology(
    solution,
    *,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Annotated Rao optimum-thrust topology: arc wall, BD, control surface
    DE, and the wall streamline B→E, with the mass-flux balance."""
    plt = _new_ax(save_path, show, (11, 5))
    art = _bde_artifacts(solution)
    topo = art["nasa_topology"]
    kernel = art["kernel"]
    bfe = art["bde_region"]

    fig, ax = plt.subplots(figsize=(11, 5))

    # throat-arc wall (0 -> B): first node of each kernel RRC.
    rrcs = [rrc for rrc in getattr(kernel, "rrcs", []) if rrc]
    if rrcs:
        awx = [float(rrc[0].x) for rrc in rrcs]
        awr = [float(rrc[0].r) for rrc in rrcs]
        ax.plot(awx, awr, color="0.4", lw=2.0, label="throat-arc wall (0→B)")

    bdx, bdr = _xy(topo.BD)
    ax.plot(bdx, bdr, color="tab:red", lw=2.2,
            label="last kernel RRC (B→axis; BD = B→D)")
    dex, der = _xy(topo.DE)
    ax.plot(dex, der, color="tab:blue", lw=2.2,
            label="DE — control surface (LRC)")
    if bfe.wall_contour:
        wcx, wcr = _xy(bfe.wall_contour)
        ax.plot(wcx, wcr, color="k", lw=2.4, label="wall streamline (B→E)")

    for node, name, dxy in (
        (topo.B, "B", (-10, 8)),
        (topo.D, "D", (8, -14)),
        (topo.E, "E", (8, 6)),
    ):
        ax.plot([float(node.x)], [float(node.r)], "o", color="k", ms=6, zorder=5)
        ax.annotate(name, (float(node.x), float(node.r)),
                    textcoords="offset points", xytext=dxy, fontsize=12,
                    fontweight="bold", zorder=6)

    ax.axhline(0.0, color="0.7", lw=0.5, ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    mbd = float(getattr(topo, "mass_BD", float("nan")))
    mde = float(getattr(topo, "mass_DE", float("nan")))
    rel = (abs(mde - mbd) / abs(mbd)) if abs(mbd) > 0 else float("nan")
    ax.set_title(title or
                 "Rao B-D-E optimum-thrust topology", fontsize=11)
    ax.text(0.99, 0.03,
            f"mass balance  ṁ_BD={mbd:.4g},  ṁ_DE={mde:.4g}  "
            f"(rel Δ={rel:.1e})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# 3. BDE characteristic net                                                    #
# --------------------------------------------------------------------------- #
def plot_bde_mesh(
    solution,
    *,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Draw the physical B-D-E net and the valid auxiliary continuation."""
    plt = _new_ax(save_path, show, (15, 6))
    art = _bde_artifacts(solution)
    bfe = art["bde_region"]
    topo = art["nasa_topology"]
    physical_rows = list(bfe.rows or bfe.grid_rows)
    auxiliary_rows = list(getattr(bfe, "full_grid_rows", ()) or ())
    if len(physical_rows) < 2:
        raise ValueError("BDE region has too few rows to draw a mesh")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    def draw_families(ax, rows):
        # column links (LRC family): same-index nodes across adjacent rows
        for j in range(len(rows) - 1):
            prev, curr = rows[j], rows[j + 1]
            n = min(len(prev), len(curr))
            for i in range(n):
                # Axis-to-axis (or interior-to-axis after a ragged
                # truncation) is an axis boundary connector, not an LRC.
                if min(float(prev[i].r), float(curr[i].r)) <= 1e-12:
                    continue
                ax.plot([float(prev[i].x), float(curr[i].x)],
                        [float(prev[i].r), float(curr[i].r)],
                        color="tab:orange", lw=0.5, alpha=0.55, zorder=1)
        # row links (RRC family): adjacent nodes within a row
        for row in rows:
            xs, rs = _xy(row)
            ax.plot(xs, rs, color="tab:blue", lw=0.5, alpha=0.6, zorder=1)
        allx = np.asarray([float(n.x) for row in rows for n in row])
        allr = np.asarray([float(n.r) for row in rows for n in row])
        ax.plot(allx, allr, ".", color="0.15", ms=1.6, zorder=2)
        ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
        ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
        return allx, allr

    px, pr = draw_families(axes[0], physical_rows)
    if bfe.wall_contour:
        wcx, wcr = _xy(bfe.wall_contour)
        axes[0].plot(wcx, wcr, color="k", lw=2.0, zorder=4)
    dex, der = _xy(topo.DE)
    axes[0].plot(dex, der, color="tab:green", lw=1.6, ls="--", zorder=3)
    axes[0].plot([float(topo.D.x)], [float(topo.D.r)], "r*", ms=13, zorder=5)
    axes[0].plot(
        [float(topo.E.x)], [float(topo.E.r)], "*", color="tab:green",
        ms=13, zorder=5,
    )
    axes[0].set_xlim(float(np.min(px)), float(np.max(px)))
    axes[0].set_ylim(0.95 * float(np.min(pr)), 1.05 * float(np.max(pr)))
    axes[0].set_title("physical B-D-E characteristic region", fontsize=10)

    if auxiliary_rows:
        axx, axr = draw_families(axes[1], auxiliary_rows)
        downstream = axx >= float(topo.E.x)
        downstream_r = axr[downstream]
        axes[1].set_xlim(float(topo.E.x), float(np.max(axx)))
        axes[1].set_ylim(
            max(0.0, 0.92 * float(np.min(downstream_r))),
            1.05 * float(np.max(downstream_r)),
        )
    else:
        axes[1].text(
            0.5, 0.5, "auxiliary continuation unavailable",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    axes[1].set_title(
        "auxiliary post-DE continuation — stopped before caustic",
        fontsize=10,
    )

    # legend proxies
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], color="tab:blue", lw=1.5, label="RRC family (rows)"),
        Line2D([0], [0], color="tab:orange", lw=1.5, label="LRC family (columns)"),
        Line2D([0], [0], color="k", lw=2.0, label="wall streamline"),
        Line2D([0], [0], color="tab:green", lw=1.5, ls="--",
               label="DE control surface"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="r",
               markersize=12, label="D"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:green",
               markersize=12, label="E"),
    ]
    axes[0].legend(handles=proxies, loc="upper right", fontsize=8,
                   framealpha=0.9)

    n_trunc = int(getattr(bfe, "topology_truncated_rows", 0))
    fig.suptitle(
        title or
        f"BDE characteristic net — {len(physical_rows)} rows"
        f"  ({n_trunc} downstream caustic termination)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# 4. BDE flow-state / conservation diagnostics                                #
# --------------------------------------------------------------------------- #
def _downstream_pair(p0, p1):
    """Return a characteristic segment ordered in increasing axial x."""
    return (p0, p1) if float(p1.x) >= float(p0.x) else (p1, p0)


def _bde_residual_samples(rows, gamma: float) -> dict[str, np.ndarray]:
    """Compatibility residuals at the measured BDE links.

    RRC rows are C- characteristics; same-index links between consecutive
    rows are LRC/C+ characteristics.  The final segment ending at ``r=0`` is
    omitted because the regular axis unit process, not the off-axis source
    quadrature, supplies that compatibility condition.
    """
    from raosim.rao_residuals import (
        residual_Cminus_axisym,
        residual_Cplus_axisym,
    )

    samples = {"cminus": [], "cplus": []}
    axis_tol = 1e-12
    for row_idx, row in enumerate(rows):
        for link_idx, (a, b) in enumerate(zip(row[:-1], row[1:])):
            if min(float(a.r), float(b.r)) <= axis_tol:
                continue
            p0, p1 = _downstream_pair(a, b)
            samples["cminus"].append((
                0.5 * (float(a.x) + float(b.x)),
                0.5 * (float(a.r) + float(b.r)),
                math.degrees(residual_Cminus_axisym(p0, p1, gamma)),
                row_idx,
                link_idx,
            ))

    for row_idx, (prev, curr) in enumerate(zip(rows[:-1], rows[1:]), start=1):
        for link_idx in range(min(len(prev), len(curr))):
            a, b = prev[link_idx], curr[link_idx]
            if min(float(a.r), float(b.r)) <= axis_tol:
                continue
            p0, p1 = _downstream_pair(a, b)
            samples["cplus"].append((
                0.5 * (float(a.x) + float(b.x)),
                0.5 * (float(a.r) + float(b.r)),
                math.degrees(residual_Cplus_axisym(p0, p1, gamma)),
                row_idx,
                link_idx,
            ))

    return {
        name: np.asarray(values, dtype=float).reshape((-1, 5))
        for name, values in samples.items()
    }


def _bde_row_mass(rows, gamma: float) -> np.ndarray:
    """Solver-native normalized mass flux across each wall-to-axis row."""
    from raosim.nasa_moc import MOCNode, calc_massflow_along_rrc

    values = []
    for row in rows:
        moc_row = [
            MOCNode(
                float(p.x), float(p.r), float(p.M), float(p.theta), gamma
            )
            for p in row
        ]
        values.append(float(calc_massflow_along_rrc(moc_row, gamma)[0]))
    return np.asarray(values, dtype=float)


def _proper_segment_intersection(a, b, c, d):
    """Interior intersection point of two 2-D segments, or ``None``."""
    r = b - a
    s = d - c
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) <= 1e-18:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / denom)
    u = float((q[0] * r[1] - q[1] * r[0]) / denom)
    eps = 1e-9
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return a + t * r
    return None


def _bde_same_family_crossings(rows) -> dict[str, list[tuple[float, float]]]:
    """Local crossings between adjacent RRCs and neighbouring LRCs.

    Comparing adjacent members is sufficient to detect the first loss of
    family ordering and avoids the misleading all-to-all intersections caused
    by many distant characteristics converging into the same axis band.
    """
    crossings: dict[str, list[tuple[float, float]]] = {
        "rrc": [], "lrc": [],
    }

    # Adjacent row characteristics (RRC/C- family).
    for row0, row1 in zip(rows[:-1], rows[1:]):
        seg0 = [
            (np.array([a.x, a.r], dtype=float),
             np.array([b.x, b.r], dtype=float))
            for a, b in zip(row0[:-1], row0[1:])
        ]
        seg1 = [
            (np.array([a.x, a.r], dtype=float),
             np.array([b.x, b.r], dtype=float))
            for a, b in zip(row1[:-1], row1[1:])
        ]
        for a, b in seg0:
            amin, amax = np.minimum(a, b), np.maximum(a, b)
            for c, d in seg1:
                cmin, cmax = np.minimum(c, d), np.maximum(c, d)
                if np.any(amax < cmin) or np.any(cmax < amin):
                    continue
                point = _proper_segment_intersection(a, b, c, d)
                if point is not None:
                    crossings["rrc"].append((float(point[0]), float(point[1])))

    # Same-index row-to-row links are the LRC/C+ family.  A loss of ordering
    # appears as two of these links crossing within one row strip.
    for row0, row1 in zip(rows[:-1], rows[1:]):
        segments = []
        for i in range(min(len(row0), len(row1))):
            if min(float(row0[i].r), float(row1[i].r)) <= 1e-12:
                continue
            a = np.array([row0[i].x, row0[i].r], dtype=float)
            b = np.array([row1[i].x, row1[i].r], dtype=float)
            segments.append((a, b))
        for i, (a, b) in enumerate(segments):
            amin, amax = np.minimum(a, b), np.maximum(a, b)
            for c, d in segments[i + 1:]:
                cmin, cmax = np.minimum(c, d), np.maximum(c, d)
                if np.any(amax < cmin) or np.any(cmax < amin):
                    continue
                point = _proper_segment_intersection(a, b, c, d)
                if point is not None:
                    crossings["lrc"].append((float(point[0]), float(point[1])))
    return crossings


def _draw_bde_mesh_context(ax, rows, wall, *, alpha=0.16, lw=0.25):
    for row in rows:
        xs, rs = _xy(row)
        ax.plot(xs, rs, color="0.25", lw=lw, alpha=alpha, zorder=2)
    for prev, curr in zip(rows[:-1], rows[1:]):
        for i in range(min(len(prev), len(curr))):
            if min(float(prev[i].r), float(curr[i].r)) <= 1e-12:
                continue
            ax.plot(
                [float(prev[i].x), float(curr[i].x)],
                [float(prev[i].r), float(curr[i].r)],
                color="0.25", lw=lw, alpha=alpha, zorder=2,
            )
    if wall:
        wx, wr = _xy(wall)
        ax.plot(wx, wr, color="k", lw=1.3, zorder=4)


def plot_bde_diagnostics(
    solution,
    gamma: float = 1.4,
    *,
    p0: float | None = None,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Plot BDE flow fields and the numerical checks hidden by a mesh plot.

    The field panels use the physical B-D-E characteristic strip.  Pressure is shown as
    ``p/p0`` unless a stagnation pressure ``p0`` [Pa] is supplied.  Mass is
    the solver's normalized axisymmetric flux, so the conservation panel is
    reported as relative error rather than misleading dimensional kg/s.
    """
    plt = _new_ax(save_path, show, (16, 9))
    from matplotlib import colors as mcolors
    from matplotlib import tri as mtri
    from raosim.gas_dynamics import isentropic_pressure_ratio

    art = _bde_artifacts(solution)
    bfe = art["bde_region"]
    topo = art["nasa_topology"]
    rows = list(bfe.rows or bfe.grid_rows)
    auxiliary_rows = list(getattr(bfe, "full_grid_rows", ()) or ())
    if len(rows) < 2:
        raise ValueError("BDE region has too few cropped rows for diagnostics")

    nodes = [node for row in rows for node in row]
    x = np.asarray([float(node.x) for node in nodes], dtype=float)
    r = np.asarray([float(node.r) for node in nodes], dtype=float)
    mach = np.asarray([float(node.M) for node in nodes], dtype=float)
    theta_deg = np.degrees(np.asarray(
        [float(node.theta) for node in nodes], dtype=float
    ))
    pressure_ratio = np.asarray(
        isentropic_pressure_ratio(mach, gamma), dtype=float
    )
    pressure = pressure_ratio if p0 is None else pressure_ratio * p0 / 1e5
    pressure_label = "Pressure ratio p/p0" if p0 is None else "Pressure [bar]"

    triangulation = mtri.Triangulation(x, r)
    # Do not let Delaunay bridge unusually large gaps in the ragged,
    # axis-truncated rows.
    triangles = triangulation.triangles
    tri_x, tri_r = x[triangles], r[triangles]
    edge_lengths = np.maximum.reduce([
        np.hypot(tri_x[:, 1] - tri_x[:, 0], tri_r[:, 1] - tri_r[:, 0]),
        np.hypot(tri_x[:, 2] - tri_x[:, 1], tri_r[:, 2] - tri_r[:, 1]),
        np.hypot(tri_x[:, 0] - tri_x[:, 2], tri_r[:, 0] - tri_r[:, 2]),
    ])
    local_lengths = [
        math.hypot(float(b.x - a.x), float(b.r - a.r))
        for row in rows for a, b in zip(row[:-1], row[1:])
    ]
    length_scale = float(np.median(local_lengths)) if local_lengths else 0.0
    if length_scale > 0.0:
        triangulation.set_mask(edge_lengths > 8.0 * length_scale)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    field_specs = (
        (mach, "turbo", "Mach number", "Mach"),
        (pressure, "viridis_r", pressure_label, "Pressure"),
        (theta_deg, "coolwarm", "Flow angle [deg]", "Flow angle"),
    )
    for ax, (values, cmap, cbar_label, panel_title) in zip(axes[0], field_specs):
        contour = ax.tricontourf(triangulation, values, levels=24, cmap=cmap)
        _draw_bde_mesh_context(ax, rows, bfe.wall_contour)
        fig.colorbar(contour, ax=ax, pad=0.01, fraction=0.046,
                     label=cbar_label)
        ax.set_title(panel_title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")

    # ``rows`` is the uncropped physical B-D-E strip, whose indices preserve
    # the two true characteristic families.  Wall-cropped rows are ragged and
    # cannot be linked by same array index.
    residuals = _bde_residual_samples(rows, gamma)
    cminus = residuals["cminus"]
    cplus = residuals["cplus"]
    residual_values = np.concatenate([
        np.abs(cminus[:, 2]), np.abs(cplus[:, 2])
    ])
    positive = residual_values[residual_values > 0.0]
    vmin = max(float(np.percentile(positive, 2)) if positive.size else 1e-8,
               1e-8)
    vmax = max(float(np.max(positive)) if positive.size else vmin, vmin * 10.0)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    ax_res = axes[1, 0]
    _draw_bde_mesh_context(ax_res, rows, bfe.wall_contour, alpha=0.10)
    scatter = ax_res.scatter(
        cminus[:, 0], cminus[:, 1], c=np.maximum(np.abs(cminus[:, 2]), vmin),
        norm=norm, cmap="magma", s=6, marker="o", linewidths=0,
        label="RRC / C-", zorder=3,
    )
    ax_res.scatter(
        cplus[:, 0], cplus[:, 1], c=np.maximum(np.abs(cplus[:, 2]), vmin),
        norm=norm, cmap="magma", s=7, marker="^", linewidths=0,
        label="LRC / C+", zorder=3,
    )
    fig.colorbar(scatter, ax=ax_res, pad=0.01, fraction=0.046,
                 label="|compatibility residual| [deg]")
    cminus_rms = float(np.sqrt(np.mean(cminus[:, 2] ** 2)))
    cplus_rms = float(np.sqrt(np.mean(cplus[:, 2] ** 2)))
    ax_res.set_title(
        f"Characteristic compatibility  RMS C-={cminus_rms:.3g}°, "
        f"C+={cplus_rms:.3g}°"
    )
    ax_res.set_xlabel("x [m]"); ax_res.set_ylabel("r [m]")
    ax_res.legend(loc="upper right", fontsize=8)

    cut_x, cut_mass, coverage = _axial_mass_cuts(solution, gamma)
    diag = getattr(solution, "construction_diagnostics", None) or {}
    thrust_diag = diag.get("thrust_sanity", {})
    mass_ref = float(thrust_diag.get("kernel_throat_mass_flux", np.nan))
    if not math.isfinite(mass_ref) or abs(mass_ref) <= 1e-15:
        mass_ref = float(np.nanmean(cut_mass))
    mass_error_pct = 100.0 * (cut_mass / mass_ref - 1.0)
    valid_mass = np.isfinite(mass_error_pct)
    ax_mass = axes[1, 1]
    ax_mass.plot(
        cut_x[valid_mass], mass_error_pct[valid_mass], "o-",
        color="tab:blue", ms=3, lw=1.3,
    )
    ax_mass.axhline(0.0, color="0.25", lw=0.8)
    ax_mass.axhline(1.0, color="0.55", lw=0.7, ls="--")
    ax_mass.axhline(-1.0, color="0.55", lw=0.7, ls="--")
    max_mass_error = (
        float(np.max(np.abs(mass_error_pct[valid_mass])))
        if np.any(valid_mass) else float("nan")
    )
    ax_mass.set_title(f"Axial mass: max |error|={max_mass_error:.2f}%")
    ax_mass.set_xlabel("axial cut x [m]")
    ax_mass.set_ylabel("mass-flow error vs throat [%]")
    ax_mass.grid(True, alpha=0.2)

    crossings = _bde_same_family_crossings(auxiliary_rows)
    ax_axis = axes[1, 2]
    if auxiliary_rows:
        _draw_bde_mesh_context(
            ax_axis, auxiliary_rows, bfe.wall_contour, alpha=0.45, lw=0.4
        )
    full_x = np.asarray(
        [float(node.x) for row in auxiliary_rows for node in row], dtype=float
    )
    full_r = np.asarray(
        [float(node.r) for row in auxiliary_rows for node in row], dtype=float
    )
    downstream = full_x >= float(topo.E.x)
    downstream_x = full_x[downstream]
    downstream_r = full_r[downstream]
    if downstream_x.size:
        ax_axis.set_xlim(
            float(np.min(downstream_x)), float(np.max(downstream_x))
        )
        ax_axis.set_ylim(
            max(0.0, 0.92 * float(np.min(downstream_r))),
            1.05 * float(np.max(downstream_r)),
        )
    for family, marker, color in (
        ("rrc", "x", "tab:red"), ("lrc", "+", "tab:purple"),
    ):
        pts = crossings[family]
        if pts:
            xp, rp = np.asarray(pts, dtype=float).T
            ax_axis.scatter(xp, rp, marker=marker, color=color, s=42,
                            linewidths=1.4, label=f"{family.upper()} crossing")
    n_topology_truncated = int(getattr(bfe, "topology_truncated_rows", 0))
    ax_axis.set_title(
        f"Post-DE prefix: {len(crossings['rrc'])} RRC / "
        f"{len(crossings['lrc'])} LRC crossings; "
        f"{n_topology_truncated} caustic stop"
    )
    ax_axis.set_xlabel("x [m]"); ax_axis.set_ylabel("r [m]")
    if crossings["rrc"] or crossings["lrc"]:
        ax_axis.legend(loc="upper right", fontsize=8)

    n_trunc = int(getattr(bfe, "topology_truncated_rows", 0))
    fig.suptitle(
        title or (
            f"BDE flow and conservation diagnostics — {len(rows)} rows, "
            f"{n_trunc} downstream caustic stop"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, save_path, show)


# --------------------------------------------------------------------------- #
# 5. Four-panel BDE integrity audit                                            #
# --------------------------------------------------------------------------- #
def _bde_geometry_samples(rows) -> dict[str, np.ndarray]:
    """Mach-line direction error for every measured characteristic link.

    The robust error is the wrapped angle between the geometric segment and
    ``theta +/- asin(1/M)`` at its midpoint.  The literal slope difference
    ``dr/dx - tan(theta +/- mu)`` is retained beside it for reporting.
    """
    samples = {"cminus": [], "cplus": []}

    def append(name, a, b, row_idx, link_idx, sign):
        p0, p1 = _downstream_pair(a, b)
        dx = float(p1.x - p0.x)
        dr = float(p1.r - p0.r)
        theta = 0.5 * (float(p0.theta) + float(p1.theta))
        # The marching geometry uses TanAvg: the tangent of the mean endpoint
        # Mach-line angles.  Average the endpoint Mach angles themselves;
        # asin(1 / average(M)) introduces a visible but artificial O(ds^2)
        # direction residual.
        mu = 0.5 * (
            math.asin(1.0 / max(float(p0.M), 1.000001))
            + math.asin(1.0 / max(float(p1.M), 1.000001))
        )
        predicted_angle = theta + sign * mu
        geometric_angle = math.atan2(dr, dx)
        angle_error = math.atan2(
            math.sin(geometric_angle - predicted_angle),
            math.cos(geometric_angle - predicted_angle),
        )
        geometric_slope = dr / dx if abs(dx) > 1e-15 else math.copysign(
            float("inf"), dr
        )
        predicted_slope = math.tan(predicted_angle)
        slope_error = geometric_slope - predicted_slope
        samples[name].append((
            0.5 * (float(a.x) + float(b.x)),
            0.5 * (float(a.r) + float(b.r)),
            math.degrees(angle_error),
            slope_error,
            row_idx,
            link_idx,
        ))

    for row_idx, row in enumerate(rows):
        for link_idx, (a, b) in enumerate(zip(row[:-1], row[1:])):
            append("cminus", a, b, row_idx, link_idx, -1.0)
    for row_idx, (prev, curr) in enumerate(zip(rows[:-1], rows[1:]), start=1):
        for link_idx in range(min(len(prev), len(curr))):
            if min(float(prev[link_idx].r), float(curr[link_idx].r)) <= 1e-12:
                continue
            append(
                "cplus", prev[link_idx], curr[link_idx],
                row_idx, link_idx, +1.0,
            )
    return {
        name: np.asarray(values, dtype=float).reshape((-1, 6))
        for name, values in samples.items()
    }


def _bde_cell_data(rows):
    """Return consistently oriented BDE quadrilaterals and signed areas."""
    polygons = []
    raw_areas = []
    centers = []
    indices = []
    for row_idx, (row0, row1) in enumerate(zip(rows[:-1], rows[1:])):
        n_cell = max(0, min(len(row0), len(row1)) - 1)
        # When one row was negative-r truncated, its last index is a newly
        # closed axis point while the same index in the longer row is still an
        # interior node.  That final ragged polygon has no four-link MOC cell
        # topology, so do not manufacture a quadrilateral from it.
        if len(row0) != len(row1):
            n_cell = max(0, n_cell - 1)
        for i in range(n_cell):
            polygon = np.asarray([
                [row0[i].x, row0[i].r],
                [row0[i + 1].x, row0[i + 1].r],
                [row1[i + 1].x, row1[i + 1].r],
                [row1[i].x, row1[i].r],
            ], dtype=float)
            x, y = polygon[:, 0], polygon[:, 1]
            area = 0.5 * float(
                np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
            )
            polygons.append(polygon)
            raw_areas.append(area)
            centers.append(np.mean(polygon, axis=0))
            indices.append((row_idx, i))
    raw = np.asarray(raw_areas, dtype=float)
    nonzero = raw[np.abs(raw) > 0.0]
    orientation = float(np.sign(np.median(nonzero))) if nonzero.size else 1.0
    if orientation == 0.0:
        orientation = 1.0
    oriented = orientation * raw
    return (
        polygons,
        oriented,
        np.asarray(centers, dtype=float).reshape((-1, 2)),
        np.asarray(indices, dtype=int).reshape((-1, 2)),
    )


def _bde_neighbor_jumps(rows, gamma: float) -> dict[str, np.ndarray]:
    from raosim.gas_dynamics import isentropic_pressure_ratio

    jumps = {"mach": [], "theta_deg": [], "pressure_ratio": []}

    def append(a, b):
        jumps["mach"].append(abs(float(b.M) - float(a.M)))
        jumps["theta_deg"].append(abs(math.degrees(float(b.theta - a.theta))))
        pa = float(isentropic_pressure_ratio(float(a.M), gamma))
        pb = float(isentropic_pressure_ratio(float(b.M), gamma))
        jumps["pressure_ratio"].append(abs(pb - pa))

    for row in rows:
        for a, b in zip(row[:-1], row[1:]):
            append(a, b)
    for prev, curr in zip(rows[:-1], rows[1:]):
        for i in range(min(len(prev), len(curr))):
            if min(float(prev[i].r), float(curr[i].r)) <= 1e-12:
                continue
            append(prev[i], curr[i])
    return {name: np.asarray(values, dtype=float) for name, values in jumps.items()}


def _axial_mass_cuts(solution, gamma: float, *, n_cuts: int = 7):
    """Interpolate normalized axial mass flux on several vertical cuts."""
    from scipy.interpolate import LinearNDInterpolator
    from raosim.gas_dynamics import (
        isentropic_density_ratio,
        isentropic_temperature_ratio,
    )

    art = _bde_artifacts(solution)
    bfe = art["bde_region"]
    kernel = art["kernel"]
    wall = list(bfe.wall_contour)
    if len(wall) < 3:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    nodes = []
    for rrc in getattr(kernel, "rrcs", ()):
        nodes.extend(rrc)
    # The valid auxiliary prefix fills the lower-radius part of axial cuts.
    # It is caustic-free through the nozzle exit; using only the upper B-D-E
    # strip leaves an interpolation gap and creates a false mass-flow drift.
    flow_rows = getattr(bfe, "full_grid_rows", ()) or bfe.grid_rows
    for row in flow_rows:
        nodes.extend(row)
    if len(nodes) < 8:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    points = np.asarray([(float(p.x), float(p.r)) for p in nodes], dtype=float)
    M = np.asarray([float(p.M) for p in nodes], dtype=float)
    theta = np.asarray([float(p.theta) for p in nodes], dtype=float)
    T = np.asarray(isentropic_temperature_ratio(M, gamma), dtype=float)
    rho = np.asarray(isentropic_density_ratio(M, gamma), dtype=float)
    axial_flux = rho * M * np.sqrt(gamma * T) * np.cos(theta)

    # Average exact duplicate locations before constructing the interpolant.
    rounded = np.round(points, decimals=13)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    flux_sum = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    np.add.at(flux_sum, inverse, axial_flux)
    np.add.at(counts, inverse, 1.0)
    interp = LinearNDInterpolator(unique, flux_sum / counts, fill_value=np.nan)

    wall_x = np.asarray([float(p.x) for p in wall], dtype=float)
    wall_r = np.asarray([float(p.r) for p in wall], dtype=float)
    order = np.argsort(wall_x)
    wall_x, wall_r = wall_x[order], wall_r[order]
    pad = 0.04 * max(float(wall_x[-1] - wall_x[0]), 0.0)
    cuts = np.linspace(wall_x[0] + pad, wall_x[-1] - pad, n_cuts)
    masses = np.full(n_cuts, np.nan, dtype=float)
    coverage = np.zeros(n_cuts, dtype=float)
    for k, x_cut in enumerate(cuts):
        radius = float(np.interp(x_cut, wall_x, wall_r))
        r_cut = np.linspace(0.0, radius, 500)
        values = np.asarray(
            interp(np.column_stack([np.full_like(r_cut, x_cut), r_cut])),
            dtype=float,
        )
        finite = np.isfinite(values)
        coverage[k] = float(np.mean(finite))
        if coverage[k] < 0.98:
            continue
        if not np.all(finite):
            values = np.interp(r_cut, r_cut[finite], values[finite])
        masses[k] = float(np.trapezoid(2.0 * math.pi * r_cut * values, r_cut))
    return cuts, masses, coverage


def plot_bde_integrity(
    solution,
    gamma: float = 1.4,
    *,
    residual_tol: float = 2e-3,
    save_path: str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Four-panel BDE audit: links, cells, axis state, and field smoothness."""
    plt = _new_ax(save_path, show, (15, 10))
    from matplotlib import colors as mcolors
    from matplotlib import tri as mtri
    from matplotlib.collections import PolyCollection
    from raosim.gas_dynamics import (
        isentropic_density_ratio,
        isentropic_pressure_ratio,
        isentropic_temperature_ratio,
    )

    art = _bde_artifacts(solution)
    bfe = art["bde_region"]
    # The requested numerical audit belongs to the physical B-D-E strip.
    # ``full_grid_rows`` is only an auxiliary DE-to-axis continuation and can
    # legitimately terminate at a downstream caustic beyond the exit.
    rows = list(bfe.rows or bfe.grid_rows)
    if len(rows) < 2:
        raise ValueError("BDE region has too few rows for an integrity audit")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Source-inclusive compatibility, with Mach-line direction outliers.
    compatibility = _bde_residual_samples(rows, gamma)
    geometry = _bde_geometry_samples(rows)
    compat_all = np.concatenate([
        np.abs(compatibility["cminus"][:, 2]),
        np.abs(compatibility["cplus"][:, 2]),
    ])
    positive = compat_all[compat_all > 0.0]
    vmin = max(float(np.percentile(positive, 2)) if positive.size else 1e-8,
               1e-8)
    vmax = max(float(np.max(positive)) if positive.size else vmin, vmin * 10.0)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    ax = axes[0, 0]
    _draw_bde_mesh_context(ax, rows, bfe.wall_contour, alpha=0.08)
    plotted = None
    for name, marker, label in (
        ("cminus", "o", "RRC / C-"), ("cplus", "^", "LRC / C+"),
    ):
        data = compatibility[name]
        plotted = ax.scatter(
            data[:, 0], data[:, 1],
            c=np.maximum(np.abs(data[:, 2]), vmin),
            norm=norm, cmap="magma", marker=marker, s=6,
            linewidths=0, label=label, zorder=3,
        )
        geom = geometry[name]
        bad = np.abs(geom[:, 2]) > 0.05
        if np.any(bad):
            ax.scatter(
                geom[bad, 0], geom[bad, 1], marker="x", s=10,
                color="tab:cyan", linewidths=0.5, zorder=4,
            )
    fig.colorbar(plotted, ax=ax, pad=0.01, fraction=0.046,
                 label="|axisymmetric compatibility residual| [deg]")
    within = 100.0 * float(np.mean(compat_all <= residual_tol))
    geom_all = np.concatenate([
        np.abs(geometry["cminus"][:, 2]),
        np.abs(geometry["cplus"][:, 2]),
    ])
    ax.set_title(
        f"Characteristic links: {within:.1f}% compatibility <= "
        f"{residual_tol:g}°; direction p99={np.percentile(geom_all, 99):.3g}°"
    )
    ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    ax.legend(loc="upper right", fontsize=8)

    # 2. Oriented quadrilateral areas.  Positive means the prevailing grid
    # orientation; zero/negative cells are explicitly outlined.
    polygons, areas, centers, _ = _bde_cell_data(rows)
    area_scale = float(np.median(np.abs(areas))) if areas.size else 1.0
    if area_scale <= 0.0:
        area_scale = 1.0
    normalized_area = areas / area_scale
    max_abs = max(float(np.max(np.abs(normalized_area))), 1.0)
    area_norm = mcolors.SymLogNorm(
        linthresh=1e-3, linscale=0.8, vmin=-max_abs, vmax=max_abs, base=10,
    )
    ax = axes[0, 1]
    cells = PolyCollection(
        polygons, array=normalized_area, cmap="coolwarm", norm=area_norm,
        edgecolors="none", linewidths=0.0,
    )
    ax.add_collection(cells)
    area_tol = max(1e-12 * area_scale, 1e-18)
    invalid = areas <= area_tol
    if np.any(invalid):
        bad_cells = PolyCollection(
            [polygons[i] for i in np.flatnonzero(invalid)],
            facecolors="none", edgecolors="red", linewidths=0.7,
        )
        ax.add_collection(bad_cells)
    fig.colorbar(cells, ax=ax, pad=0.01, fraction=0.046,
                 label="oriented signed area / median |area|")
    ax.autoscale_view()
    ax.set_title(
        f"Cell orientation: {int(np.sum(invalid))}/{len(areas)} "
        "zero or negative-area cells"
    )
    ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")

    # 3. Axis condition and finite derived thermodynamics.  B-D-E itself does
    # not touch the axis; use the kernel RRC axis nodes that actually carry
    # the symmetry boundary condition instead of mistaking the auxiliary
    # post-DE continuation for part of the nozzle domain.
    kernel = art["kernel"]
    axis = [rrc[-1] for rrc in getattr(kernel, "rrcs", ()) if rrc]
    if not axis:
        raise ValueError("kernel contains no axis nodes for the integrity audit")
    axis_x = np.asarray([float(p.x) for p in axis], dtype=float)
    axis_M = np.asarray([float(p.M) for p in axis], dtype=float)
    axis_theta = np.degrees(np.asarray([float(p.theta) for p in axis], dtype=float))
    axis_r = np.asarray([float(p.r) for p in axis], dtype=float)
    axis_T = np.asarray(isentropic_temperature_ratio(axis_M, gamma), dtype=float)
    axis_p = np.asarray(isentropic_pressure_ratio(axis_M, gamma), dtype=float)
    axis_rho = np.asarray(isentropic_density_ratio(axis_M, gamma), dtype=float)
    finite = bool(np.all(np.isfinite(np.concatenate([
        axis_x, axis_r, axis_M, axis_theta, axis_T, axis_p, axis_rho,
    ]))))
    ax = axes[1, 0]
    line_theta, = ax.plot(
        axis_x, axis_theta, "o-", color="tab:blue", ms=3,
        label="axis theta",
    )
    theta_span = max(float(np.max(np.abs(axis_theta))), 1e-6)
    ax.set_ylim(-1.25 * theta_span, 1.25 * theta_span)
    ax.axhline(0.0, color="0.3", lw=0.7)
    ax.set_xlabel("axis-node x [m]"); ax.set_ylabel("theta_axis [deg]")
    ax_m = ax.twinx()
    line_m, = ax_m.plot(
        axis_x, axis_M, "s--", color="tab:orange", ms=3,
        label="axis Mach",
    )
    ax_m.set_ylabel("Mach number")
    ax.legend([line_theta, line_m], ["axis theta", "axis Mach"],
              loc="best", fontsize=8)
    ax.set_title(
        f"Axis condition (kernel RRCs): max |theta|={np.max(np.abs(axis_theta)):.3g}°, "
        f"max |r|={np.max(np.abs(axis_r)):.2e} m, finite={finite}"
    )
    ax.grid(True, alpha=0.18)

    # 4. Mach field, neighbour smoothness, and axial mass-cut inset.
    nodes = [node for row in rows for node in row]
    x = np.asarray([float(p.x) for p in nodes], dtype=float)
    r = np.asarray([float(p.r) for p in nodes], dtype=float)
    M = np.asarray([float(p.M) for p in nodes], dtype=float)
    triangulation = mtri.Triangulation(x, r)
    ax = axes[1, 1]
    mach_plot = ax.tricontourf(triangulation, M, levels=24, cmap="turbo")
    _draw_bde_mesh_context(ax, rows, bfe.wall_contour, alpha=0.12)
    fig.colorbar(mach_plot, ax=ax, pad=0.01, fraction=0.046,
                 label="Mach number")
    jumps = _bde_neighbor_jumps(rows, gamma)
    ax.set_title(
        f"Mach smoothness: neighbor |dM| p99={np.percentile(jumps['mach'], 99):.3g}, "
        f"max={np.max(jumps['mach']):.3g}"
    )
    ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")

    cut_x, cut_mass, coverage = _axial_mass_cuts(solution, gamma)
    valid = np.isfinite(cut_mass)
    diag = getattr(solution, "construction_diagnostics", None) or {}
    mass_ref = float(
        diag.get("thrust_sanity", {}).get("kernel_throat_mass_flux", np.nan)
    )
    if not math.isfinite(mass_ref) or abs(mass_ref) <= 1e-15:
        mass_ref = float(np.nanmean(cut_mass)) if np.any(valid) else 1.0
    cut_error = 100.0 * (cut_mass / mass_ref - 1.0)
    for value in cut_x[valid]:
        ax.axvline(value, color="white", lw=0.55, ls="--", alpha=0.8)
    inset = ax.inset_axes([0.53, 0.06, 0.43, 0.32])
    if np.any(valid):
        inset.plot(cut_x[valid], cut_error[valid], "o-", ms=2.5, lw=1.0,
                   color="tab:blue")
        inset.axhline(0.0, color="0.3", lw=0.6)
        inset.set_title(
            f"axial cuts: max |mass error|={np.nanmax(np.abs(cut_error)):.2f}%",
            fontsize=7,
        )
    else:
        inset.text(0.5, 0.5, "axial cuts unavailable", ha="center", va="center",
                   transform=inset.transAxes, fontsize=7)
        inset.set_title(
            f"coverage max={np.max(coverage) if coverage.size else 0.0:.0%}",
            fontsize=7,
        )
    inset.set_xlabel("x [m]", fontsize=7)
    inset.set_ylabel("mass error [%]", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.18)

    fig.suptitle(title or "BDE numerical integrity audit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, save_path, show)
