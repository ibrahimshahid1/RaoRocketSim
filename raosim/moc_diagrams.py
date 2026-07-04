"""
moc_diagrams.py — construction diagrams for the Rao/MOC nozzle build.

Where ``flow_viz`` renders the *solved field* (Mach/temperature contours,
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

All three read the in-memory ``bde_artifacts`` the ``bde`` wall method
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
    """Draw the BDE interior characteristic net (both families), the wall
    contour, and a near-axis zoom of the singular convergence region."""
    plt = _new_ax(save_path, show, (15, 6))
    art = _bde_artifacts(solution)
    bfe = art["bde_region"]
    topo = art["nasa_topology"]
    rows = list(getattr(bfe, "full_grid_rows", ()) or bfe.grid_rows)
    if len(rows) < 2:
        raise ValueError("BDE region has too few rows to draw a mesh")

    # extents
    allx = np.array([float(n.x) for row in rows for n in row])
    allr = np.array([float(n.r) for row in rows for n in row])
    x_lo, x_hi = float(allx.min()), float(allx.max())
    r_hi = float(allr.max())
    # zoom box around the axis convergence (bottom-right of the mesh)
    zx_lo = x_lo + 0.55 * (x_hi - x_lo)
    zr_hi = 0.16 * r_hi

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (xlim, rlim, sub) in zip(axes, [
        ((x_lo, x_hi), (0.0, 1.05 * r_hi), "full BDE net"),
        ((zx_lo, x_hi), (0.0, zr_hi), "near-axis zoom — MOC axis convergence"),
    ]):
        # column links (LRC family): same-index nodes across adjacent rows
        for j in range(len(rows) - 1):
            prev, curr = rows[j], rows[j + 1]
            n = min(len(prev), len(curr))
            for i in range(n):
                ax.plot([float(prev[i].x), float(curr[i].x)],
                        [float(prev[i].r), float(curr[i].r)],
                        color="tab:orange", lw=0.5, alpha=0.55, zorder=1)
        # row links (RRC family): adjacent nodes within a row
        for row in rows:
            xs, rs = _xy(row)
            ax.plot(xs, rs, color="tab:blue", lw=0.5, alpha=0.6, zorder=1)
        # nodes
        ax.plot(allx, allr, ".", color="0.15", ms=1.6, zorder=2)
        # wall contour + DE + D/E markers
        if bfe.wall_contour:
            wcx, wcr = _xy(bfe.wall_contour)
            ax.plot(wcx, wcr, color="k", lw=2.0, zorder=4)
        dex, der = _xy(topo.DE)
        ax.plot(dex, der, color="tab:green", lw=1.6, ls="--", zorder=3)
        ax.plot([float(topo.D.x)], [float(topo.D.r)], "r*", ms=13, zorder=5)
        ax.plot([float(topo.E.x)], [float(topo.E.r)], "*", color="tab:green",
                ms=13, zorder=5)
        ax.axhline(0.0, color="0.6", lw=0.7, ls=":")
        ax.set_xlim(*xlim); ax.set_ylim(*rlim)
        ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
        ax.set_title(sub, fontsize=10)

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

    n_trunc = int(getattr(bfe, "negative_r_truncated_rows", 0))
    fig.suptitle(
        title or
        f"BDE characteristic net — {len(rows)} rows"
        f"  ({n_trunc} axis-truncated near r→0; normal MOC convergence)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, save_path, show)
