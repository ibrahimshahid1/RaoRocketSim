"""
plotting.py – 2-D and 3-D nozzle visualisation.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection)
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


def _set_axes_equal_3d(ax):
    """Force equal aspect ratio on a 3-D axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d(origin[0] - radius, origin[0] + radius)
    ax.set_ylim3d(origin[1] - radius, origin[1] + radius)
    ax.set_zlim3d(origin[2] - radius, origin[2] + radius)
