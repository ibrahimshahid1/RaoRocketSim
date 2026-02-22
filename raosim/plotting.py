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

    # Upper and lower contours
    ax.plot(x, y, color='#1a73e8', lw=2.2, label='Nozzle contour')
    ax.plot(x, -y, color='#1a73e8', lw=2.2)

    # Centerline
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.6)

    # Throat plane
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.6)

    # Key points
    ax.plot(0, Rt, 'ko', ms=5, zorder=5)
    ax.plot(Nx, Ny, 's', color='#e8710a', ms=6, zorder=5, label=f'N (θ_n={theta_n:.1f}°)')
    ax.plot(Ex, Ey, 'D', color='#d93025', ms=6, zorder=5, label=f'E (θ_e={theta_e:.1f}°)')
    ax.plot(P1x, P1y, '^', color='#0d652d', ms=6, zorder=5, label='P₁ (Bézier CP)')

    # Bézier control polygon (dashed)
    ax.plot([Nx, P1x, Ex], [Ny, P1y, Ey], '--', color='#0d652d', lw=0.8, alpha=0.6)

    # Annotations
    ax.annotate(f'R_t = {Rt*1000:.2f} mm', xy=(0, Rt),
                xytext=(Ln * 0.15, Rt + Re * 0.12),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
                fontsize=8, color='#333')
    ax.annotate(f'R_e = {Re*1000:.2f} mm', xy=(Ex, Ey),
                xytext=(Ex - Ln * 0.25, Ey + Re * 0.08),
                arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'),
                fontsize=8, color='#333')

    # Title
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

    # Subsample for manageable rendering
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

    # Equal aspect
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


def _set_axes_equal_3d(ax):
    """Force equal aspect ratio on a 3-D axes."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d(origin[0] - radius, origin[0] + radius)
    ax.set_ylim3d(origin[1] - radius, origin[1] + radius)
    ax.set_zlim3d(origin[2] - radius, origin[2] + radius)
