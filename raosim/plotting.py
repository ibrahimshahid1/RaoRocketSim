"""
plotting.py – 2-D and 3-D nozzle visualisation.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection)
from raosim.gas_dynamics import isentropic_pressure_ratio
from raosim.nozzle_geometry import compute_curvature


def plot_nozzle_2d(contour: dict, *, show: bool = True,
                   save_path: str | None = None) -> plt.Figure:
    """
    Plot the 2-D cross-section of the bell nozzle with annotations.

    Parameters
    ----------
    contour : dict returned by ``bell_nozzle_contour``
    show    : call plt.show()
    save_path : if given, save to file

    Returns
    -------
    matplotlib Figure
    """
    x = contour['x']
    y = contour['y']
    Rt = contour['Rt']
    Re = contour['Re']
    Ln = contour['Ln']
    theta_n = contour['theta_n']
    theta_e = contour['theta_e']
    epsilon = contour['epsilon']
    Nx, Ny = contour['N']
    Ex, Ey = contour['E']
    P1x, P1y = contour['P1']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_aspect('equal')


    ax.plot(x, y, color='#1a73e8', lw=2.2, label='Nozzle contour')
    ax.plot(x, -y, color='#1a73e8', lw=2.2)


    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.6)


    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.6)


    ax.plot(0, Rt, 'ko', ms=5, zorder=5)
    ax.plot(Nx, Ny, 's', color='#e8710a', ms=6, zorder=5, label=f'N (θ_n={theta_n:.1f}°)')
    ax.plot(Ex, Ey, 'D', color='#d93025', ms=6, zorder=5, label=f'E (θ_e={theta_e:.1f}°)')
    ax.plot(P1x, P1y, '^', color='#0d652d', ms=6, zorder=5, label='P₁ (Bézier CP)')


    ax.plot([Nx, P1x, Ex], [Ny, P1y, Ey], '--', color='#0d652d', lw=0.8, alpha=0.6)


    ax.annotate(f'R_t = {Rt*1000:.2f} mm', xy=(0, Rt),
                xytext=(Ln * 0.15, Rt + Re * 0.12),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
                fontsize=8, color='#333')
    ax.annotate(f'R_e = {Re*1000:.2f} mm', xy=(Ex, Ey),
                xytext=(Ex - Ln * 0.25, Ey + Re * 0.08),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
                fontsize=8, color='#333')


    length_pct = contour['length_pct']
    ax.set_title(
        f"Rao {length_pct:.0f}% Bell Nozzle  —  "
        f"ε = {epsilon:.1f},  L = {Ln*1000:.1f} mm,  "
        f"θₙ = {theta_n:.1f}°,  θₑ = {theta_e:.1f}°",
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('Axial position  x [m]')
    ax.set_ylabel('Radial position  y [m]')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_nozzle_3d(contour: dict, n_angular: int = 64, *,
                   show: bool = True,
                   save_path: str | None = None) -> plt.Figure:
    """
    Plot the 3-D surface-of-revolution of the bell nozzle.
    """
    x = contour['x']
    y = contour['y']


    n_axial = min(len(x), 150)
    idx = np.linspace(0, len(x) - 1, n_axial).astype(int)
    x_sub = x[idx]
    y_sub = y[idx]

    theta = np.linspace(0, 2 * np.pi, n_angular)
    T, X_mesh = np.meshgrid(theta, x_sub)
    R_mesh = np.tile(y_sub, (n_angular, 1)).T
    Y_mesh = R_mesh * np.cos(T)
    Z_mesh = R_mesh * np.sin(T)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh,
                    color='lightsteelblue', alpha=0.85,
                    edgecolor='steelblue', linewidth=0.15)


    _set_axes_equal_3d(ax)
    ax.view_init(elev=20, azim=-130)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    epsilon = contour['epsilon']
    ax.set_title(f'3-D Rao Bell Nozzle  (ε = {epsilon:.1f})', fontsize=11,
                 fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_curvature(contour: dict, *, show: bool = True) -> plt.Figure:
    """Plot wall curvature κ along the nozzle axis."""
    x = contour['x']
    y = contour['y']
    kappa = compute_curvature(x, y)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, kappa, color='#d93025', lw=1.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Curvature κ [1/m]')
    ax.set_title('Wall curvature distribution')
    ax.grid(True, ls=':', alpha=0.4)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def _solution_wall(solution, geometry: str):
    """Resolve the wall polyline for a Rao/MOC solution by mode."""
    if geometry == "raw":
        wall = getattr(solution, "wall_raw", None)
        if wall is None:
            raise ValueError("solution has no wall_raw; pass geometry='export'")
        return np.asarray(wall, dtype=float), "wall (raw, BVP output)"
    if geometry == "export":
        wall = getattr(solution, "wall_export", None)
        if wall is None:
            raise ValueError("solution has no wall_export")
        return np.asarray(wall, dtype=float), "wall (export, post-processed)"
    raise ValueError(f"geometry must be 'raw' or 'export', got {geometry!r}")


def plot_characteristic_net(solution, *, geometry: str = "raw",
                            ax=None, show: bool = False,
                            save_path: str | None = None):
    """
    Plot the wall, control surface, kernel starting line, and full
    characteristic net.  Defaults to ``geometry='raw'`` so silent
    post-processing is visible.

    See REWRITE_PLAN.md §12 (Phase 13) for the diagnostic motivation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    wall, wall_label = _solution_wall(solution, geometry)
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
            label=wall_label, zorder=4)

    ce = getattr(solution, "control_surface", None)
    if ce is not None:
        ce_x = np.asarray(getattr(ce, "x", []), dtype=float)
        ce_r = np.asarray(getattr(ce, "r", []), dtype=float)
        if ce_x.size and ce_r.size:
            ax.plot(ce_x, ce_r, "--", color="C3", linewidth=1.4,
                    label="control surface CE", zorder=3)

    kernel = getattr(solution, "kernel_points", None)
    if kernel:
        ax.plot([p.x for p in kernel], [p.r for p in kernel],
                "o-", color="C2", markersize=3, linewidth=0.9,
                label="kernel / starting line", zorder=3)

    char_net = getattr(solution, "characteristic_net", None) or []
    for row in char_net:
        pts = row.all_points() if hasattr(row, "all_points") else list(row)
        if len(pts) < 2:
            continue
        xs = [p.x for p in pts]
        rs = [p.r for p in pts]
        ax.plot(xs, rs, color="#1a73e8", linewidth=0.5, alpha=0.55, zorder=2)

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(f"Characteristic net  —  {geometry}")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_flowfield_mach(solution, *, geometry: str = "raw",
                        ax=None, cmap: str = "viridis",
                        show: bool = False,
                        save_path: str | None = None):
    """
    Scatter every node in the characteristic net coloured by Mach number,
    overlaid on the wall.  Defaults to raw wall geometry.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    wall, wall_label = _solution_wall(solution, geometry)
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
            label=wall_label, zorder=4)

    char_net = getattr(solution, "characteristic_net", None) or []
    xs, rs, Ms = [], [], []
    for row in char_net:
        pts = row.all_points() if hasattr(row, "all_points") else list(row)
        for p in pts:
            xs.append(p.x)
            rs.append(p.r)
            Ms.append(p.M)
    if xs:
        sc = ax.scatter(xs, rs, c=Ms, s=14, cmap=cmap, zorder=3)
        plt.colorbar(sc, ax=ax, label="Mach")

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(f"Mach number field  —  {geometry}")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_topology(topology, *, ax=None, show: bool = False,
                  save_path: str | None = None):
    """
    Annotated overlay of the explicit Rao construction objects
    (REWRITE_PLAN §12.1 plot #9): TT', throat-arc wall, BF, B, BD, D,
    DE, E, and the ``streamline_BE`` bell wall.

    Accepts a :class:`raosim.moc_topology.RaoTopology` (Phase 12.6) or
    any duck-typed object carrying those fields.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    def _xy(seq):
        return ([p.x for p in seq], [p.r for p in seq])

    tt = getattr(topology, "TT_prime", ())
    if tt:
        ax.plot(*_xy(tt), color="C2", linewidth=1.0, label="TT' start line",
                zorder=2)
    arc = getattr(topology, "arc_wall", ())
    if arc:
        ax.plot(*_xy(arc), color="black", linewidth=2.0,
                label="throat-arc wall", zorder=4)
    bf = getattr(topology, "BF", ())
    if bf:
        ax.plot(*_xy(bf), color="C0", linewidth=1.0, linestyle="--",
                label="BF (final RRC)", zorder=2)
    bd = getattr(topology, "BD", ())
    if bd:
        ax.plot(*_xy(bd), color="C0", linewidth=2.0,
                label="BD (mass-flow curve)", zorder=3)
    de = getattr(topology, "DE", ())
    if de:
        ax.plot(*_xy(de), color="C3", linewidth=1.6, linestyle="--",
                label="DE (control surface)", zorder=3)
    s_be = getattr(topology, "streamline_BE", ())
    if s_be:
        ax.plot(*_xy(s_be), color="C1", linewidth=2.0,
                label="streamline BE (bell wall)", zorder=4)

    for name, marker in (("B", "o"), ("D", "s"), ("E", "^")):
        p = getattr(topology, name, None)
        if p is not None:
            ax.plot([p.x], [p.r], marker, color="k", markersize=6, zorder=5)
            ax.annotate(name, (p.x, p.r), textcoords="offset points",
                        xytext=(6, 6), fontsize=10, fontweight="bold")

    theta_b = getattr(topology, "theta_B", None)
    title = "Rao construction topology"
    if theta_b is not None and math.isfinite(theta_b):
        title += f"  (theta_B = {math.degrees(theta_b):.2f} deg)"
    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_net_diagnostics(solution, *, geometry: str = "raw",
                         worst_fraction: float = 0.05,
                         gamma: float = 1.4,
                         ax=None, show: bool = False,
                         save_path: str | None = None):
    """Spec plot #8 (REWRITE_PLAN §12.1): the characteristic net with
    problematic links highlighted.

    Per-link residuals come from the net link plumbing
    (``characteristic_net_links`` /
    ``characteristic_net_compatibility_residuals`` — the corrected C±
    invariant forms), crossings from ``check_characteristic_crossing``,
    and the aggregate context line from the solution's
    ``RaoResidualReport``.  The worst ``worst_fraction`` of links (and
    every link beyond 3× its family RMS) are drawn red; offending link
    indices are printed to stdout and attached to the returned figure
    as ``fig.net_diagnostics``.

    Solutions without a forward-audit net (``evaluate_moc=False``)
    fall back to the CE's own C+ chain as the link set, so the plot is
    always meaningful for the characteristic formulation.
    """
    from raosim.rao_residuals import residual_Cplus_axisym
    from raosim.rao_variational import (
        _control_surface_flow_nodes,
        characteristic_net_compatibility_residuals,
        characteristic_net_links,
        check_characteristic_crossing,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    wall, wall_label = _solution_wall(solution, geometry)
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
            label=wall_label, zorder=4)

    # ---- gather links: (family, index, p0, p1, |residual|) ----------
    segments: list[tuple[str, int, tuple, tuple, float]] = []
    rows = getattr(solution, "characteristic_net", None) or []
    crossings = 0
    if rows:
        links = characteristic_net_links(rows)
        compat = characteristic_net_compatibility_residuals(rows, gamma)
        for fam in ("cplus", "cminus"):
            for idx, (link, res) in enumerate(
                    zip(links[fam], compat[fam])):
                segments.append((
                    fam, idx,
                    (link.parent.x, link.parent.r),
                    (link.child.x, link.child.r),
                    abs(float(res)),
                ))
        crossings = check_characteristic_crossing(rows)

    ce = getattr(solution, "control_surface", None)
    if ce is not None and getattr(ce, "x", None) is not None:
        # The CE is a C+ characteristic by construction — its segment
        # residuals are first-class links (and the only ones available
        # when the forward audit was skipped).
        nodes = _control_surface_flow_nodes(ce)
        for i in range(len(nodes) - 1):
            res = residual_Cplus_axisym(nodes[i], nodes[i + 1], gamma)
            segments.append((
                "ce_cplus", i,
                (nodes[i].x, nodes[i].r),
                (nodes[i + 1].x, nodes[i + 1].r),
                abs(float(res)),
            ))

    flagged: list[tuple[str, int, float]] = []
    if segments:
        vals = np.asarray([s[4] for s in segments], dtype=float)
        # Worst-fraction threshold plus a 3×RMS outlier rule.
        q = np.quantile(vals, 1.0 - worst_fraction) if vals.size > 1 else np.inf
        rms = float(np.sqrt(np.mean(vals ** 2))) if vals.size else 0.0
        cut = min(q, 3.0 * rms) if rms > 0 else q
        for fam, idx, p0, p1, v in segments:
            bad = bool(v >= cut and v > 0.0)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=("red" if bad else "#1a73e8"),
                    linewidth=(1.6 if bad else 0.5),
                    alpha=(0.95 if bad else 0.45),
                    zorder=(3 if bad else 2))
            if bad:
                flagged.append((fam, idx, v))

    if flagged:
        print("plot_net_diagnostics: flagged links "
              "(family, index, |residual|):")
        for fam, idx, v in sorted(flagged, key=lambda t: -t[2]):
            print(f"  {fam}[{idx}] |res|={v:.3e}")

    report = getattr(solution, "residuals", None)
    ctx = ""
    if report is not None:
        ctx = (f"   max_scaled={getattr(report, 'max_scaled', float('nan')):.2e}"
               f"  wall_tangency_rms="
               f"{getattr(report, 'wall_tangency_rms', None)}")
    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(
        f"Net diagnostics — {len(flagged)}/{len(segments)} links flagged, "
        f"{crossings} crossings{ctx}", fontsize=9)
    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    fig.net_diagnostics = {
        "n_links": len(segments),
        "flagged": flagged,
        "crossings": int(crossings),
    }

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _net_points(solution):
    """All characteristic-net points of a solution as a flat list."""
    pts = []
    for row in getattr(solution, "characteristic_net", None) or []:
        pts.extend(row.all_points() if hasattr(row, "all_points")
                   else list(row))
    return pts


def _plot_flowfield(solution, value_fn, label, *, geometry="raw",
                    ax=None, cmap="viridis", show=False, save_path=None):
    """Shared scatter-overlay machinery for the flowfield plots."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    wall, wall_label = _solution_wall(solution, geometry)
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
            label=wall_label, zorder=4)

    pts = _net_points(solution)
    if pts:
        xs = [p.x for p in pts]
        rs = [p.r for p in pts]
        vs = [value_fn(p) for p in pts]
        sc = ax.scatter(xs, rs, c=vs, s=14, cmap=cmap, zorder=3)
        plt.colorbar(sc, ax=ax, label=label)

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(f"{label} field  —  {geometry}")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_flowfield_pressure(solution, gamma: float, *, geometry="raw",
                            ax=None, cmap="plasma", show=False,
                            save_path=None):
    """p/p0 at every net node (REWRITE_PLAN §12.1 plot #4).

    Verifies monotonic decrease downstream — pockets of recompression
    inside a clean expansion flag crossed characteristics.
    """
    return _plot_flowfield(
        solution,
        lambda p: isentropic_pressure_ratio(max(p.M, 1.000001), gamma),
        "p/p0", geometry=geometry, ax=ax, cmap=cmap, show=show,
        save_path=save_path,
    )


def plot_flowfield_theta(solution, *, geometry="raw", ax=None,
                         cmap="coolwarm", show=False, save_path=None):
    """Flow angle [deg] at every net node (§12.1 plot #5)."""
    return _plot_flowfield(
        solution, lambda p: math.degrees(p.theta), "flow angle  [deg]",
        geometry=geometry, ax=ax, cmap=cmap, show=show,
        save_path=save_path,
    )


def plot_nozzle_geometry(solution, *, geometry="raw", ax=None,
                         show=False, save_path=None):
    """Wall contour + axis + throat/exit annotations (§12.1 plot #1)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    wall, wall_label = _solution_wall(solution, geometry)
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
            label=wall_label, zorder=4)
    i_throat = int(np.argmin(wall[:, 1]))
    ax.plot([wall[i_throat, 0]], [wall[i_throat, 1]], "o", color="C3",
            zorder=5)
    ax.annotate("throat", wall[i_throat], textcoords="offset points",
                xytext=(6, -12), fontsize=9)
    ax.plot([wall[-1, 0]], [wall[-1, 1]], "^", color="C3", zorder=5)
    ax.annotate("E", wall[-1], textcoords="offset points", xytext=(6, 6),
                fontsize=10, fontweight="bold")

    ax.axhline(0.0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title(f"Nozzle geometry  —  {geometry}")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _row_wall_node(row):
    pts = row.all_points() if hasattr(row, "all_points") else list(row)
    return max(pts, key=lambda p: p.r) if pts else None


def plot_wall_distributions(solution, gamma: float, *, geometry="raw",
                            show=False, save_path=None):
    """x vs wall Mach / theta / p_p0, stacked (§12.1 plot #6).

    Wall states are taken from each net row's outermost (max-r) node —
    the marched wall points.  Requires ``solution.characteristic_net``.
    """
    rows = getattr(solution, "characteristic_net", None) or []
    nodes = [n for n in (_row_wall_node(r) for r in rows) if n is not None]
    if not nodes:
        raise ValueError(
            "plot_wall_distributions needs a populated characteristic_net "
            "(run the solve with evaluate_moc=True)"
        )
    nodes.sort(key=lambda p: p.x)
    xs = np.asarray([p.x for p in nodes])
    Ms = np.asarray([max(p.M, 1.000001) for p in nodes])
    ths = np.asarray([math.degrees(p.theta) for p in nodes])
    ps = np.asarray([isentropic_pressure_ratio(M, gamma) for M in Ms])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, ys, lbl in zip(
        axes, (Ms, ths, ps),
        ("wall Mach", "wall angle  [deg]", "wall p/p0"),
    ):
        ax.plot(xs, ys, "o-", markersize=3, linewidth=1.0)
        ax.set_ylabel(lbl)
        ax.grid(True, ls=":", alpha=0.4)
    axes[-1].set_xlabel("axial position  x [m]")
    axes[0].set_title("Wall distributions (net wall nodes)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_exit_plane(solution, gamma: float, *, x_band: float = 0.02,
                    show=False, save_path=None):
    """r vs M / theta / p_p0 near the exit station (§12.1 plot #7).

    Collects net nodes with ``x >= (1 - x_band) * x_max``.  Catches
    non-uniform exit profiles, residual turning, over/under-expansion.
    """
    pts = _net_points(solution)
    if not pts:
        raise ValueError(
            "plot_exit_plane needs a populated characteristic_net "
            "(run the solve with evaluate_moc=True)"
        )
    x_max = max(p.x for p in pts)
    sel = [p for p in pts if p.x >= (1.0 - x_band) * x_max]
    sel.sort(key=lambda p: p.r)
    rs = np.asarray([p.r for p in sel])
    Ms = np.asarray([max(p.M, 1.000001) for p in sel])
    ths = np.asarray([math.degrees(p.theta) for p in sel])
    ps = np.asarray([isentropic_pressure_ratio(M, gamma) for M in Ms])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, vals, lbl in zip(
        axes, (Ms, ths, ps),
        ("Mach", "flow angle  [deg]", "p/p0"),
    ):
        ax.plot(vals, rs, "o-", markersize=3, linewidth=1.0)
        ax.set_xlabel(lbl)
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("radial position  r [m]")
    fig.suptitle(f"Exit-plane profiles (x within {x_band:.0%} of x_max)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_nasa_overlay(solution, nasa_outputs_dir, *, Rt: float | None = None,
                      geometry="raw", ax=None, show=False, save_path=None):
    """Overlay NASA wall.out onto the solution wall (§12.1 plot #10).

    The NASA reference is in R*-normalised units; the solution wall is
    in metres.  ``Rt`` sets the normalisation (defaults to the wall's
    minimum radius, which is the throat for a full contour).
    """
    from pathlib import Path

    from raosim.legacy_io import parse_wall_out

    wall, wall_label = _solution_wall(solution, geometry)
    rt = float(Rt) if Rt is not None else float(np.min(wall[:, 1]))
    if rt <= 0:
        raise ValueError("Rt must be positive for R* normalisation")

    nasa = parse_wall_out(Path(nasa_outputs_dir) / "wall.out")
    nx = nasa.column([c for c in nasa.columns if "x" in c.lower()][0])
    nr = nasa.column([c for c in nasa.columns if "r" in c.lower()][0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    ax.plot(wall[:, 0] / rt, wall[:, 1] / rt, color="black",
            linewidth=2.0, label=f"{wall_label} (/R*)", zorder=3)
    ax.scatter(nx, nr, s=16, color="C3", marker="x",
               label="NASA wall.out", zorder=4)
    ax.set_xlabel("x / R*")
    ax.set_ylabel("r / R*")
    ax.set_title("NASA reference overlay")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_sensitivity_field(x, r, values, *, label="|dCf/dr|  [1/m]",
                           signed: bool = False, wall=None, ax=None,
                           cmap: str = "magma", show: bool = False,
                           save_path: str | None = None):
    """
    Manufacturing-tolerance map (JAX plan §6 / J6): paint a per-node
    sensitivity field onto a polyline — typically ``|dCf/dr_i|`` from
    :func:`raosim.jax.sensitivities.rao_sensitivities` on the solved
    control surface, optionally over the wall contour for context.

    Parameters
    ----------
    x, r
        Node coordinates of the polyline carrying the field.
    values
        Per-node sensitivity values (same length as ``x``).
    signed
        If True plot the signed field on a diverging colormap centred
        at 0; otherwise plot ``|values|``.
    wall
        Optional (n, 2) wall polyline drawn in black for context.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(values, dtype=float)
    if x.shape != r.shape or x.shape != v.shape:
        raise ValueError("x, r, values must have identical shapes")

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    if wall is not None:
        wall = np.asarray(wall, dtype=float)
        ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2.0,
                label="wall", zorder=2)

    if signed:
        vmax = float(np.max(np.abs(v))) or 1.0
        sc = ax.scatter(x, r, c=v, s=42, cmap="coolwarm",
                        vmin=-vmax, vmax=vmax, zorder=3,
                        edgecolors="k", linewidths=0.3)
    else:
        sc = ax.scatter(x, r, c=np.abs(v), s=42, cmap=cmap, zorder=3,
                        edgecolors="k", linewidths=0.3)
    ax.plot(x, r, color="grey", linewidth=0.7, alpha=0.6, zorder=2)
    plt.colorbar(sc, ax=ax, label=label)

    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("axial position  x [m]")
    ax.set_ylabel("radial position  r [m]")
    ax.set_title("Sensitivity field" + ("  (signed)" if signed else ""))
    ax.set_aspect("equal", "box")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _set_axes_equal_3d(ax):
    """Force equal aspect ratio on a 3-D axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d(origin[0] - radius, origin[0] + radius)
    ax.set_ylim3d(origin[1] - radius, origin[1] + radius)
    ax.set_zlim3d(origin[2] - radius, origin[2] + radius)
