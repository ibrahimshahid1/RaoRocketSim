"""
nozzle_comparison.py – Compare different nozzle contour methods.

Provides utilities to:
  - Compute 2D divergence loss from exit-plane flow profiles
  - Calculate nozzle efficiency η_n = Cf_net / Cf_ideal
  - Generate side-by-side comparison tables and overlay plots

References:
  - NASA report on "Perfect Bell Nozzle Parametric Optimization Curves" (1983)
  - G. V. R. Rao, "Recent Developments in Rocket Nozzle Configurations" (1961)
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt

from raosim.gas_dynamics import (
    mach_from_area_ratio,
    isentropic_pressure_ratio,
    isentropic_density_ratio,
    isentropic_temperature_ratio,
    thrust_coefficient,
)
from raosim.conical import conical_divergence_factor


def divergence_loss_2d(contour: dict, gamma: float,
                       n_samples: int = 100) -> float:
    """
    Estimate the 2D divergence loss factor by integrating axial
    momentum across the exit plane.

        η_div = (∫ ρ·ux·|u|·r·dr) / (∫ ρ·|u|²·r·dr)

    For a perfectly axial exit flow, η_div = 1.
    For a conical nozzle with half-angle α, η_div → (1+cos α)/2.

    Parameters
    ----------
    contour   : dict from bell_nozzle_contour or conical_nozzle_contour
    gamma     : ratio of specific heats
    n_samples : radial samples across the exit plane

    Returns
    -------
    η_div ∈ (0, 1]
    """
    Rt = contour['Rt']
    Re = contour['Re']
    At = math.pi * Rt ** 2

    # Estimate exit-plane angle distribution from the bell section slope
    x_bell = contour.get('x_bell', contour['x'])
    y_bell = contour.get('y_bell', contour['y'])

    if len(x_bell) < 2:
        return 1.0

    # Exit wall angle
    dx = x_bell[-1] - x_bell[-2]
    dy = y_bell[-1] - y_bell[-2]
    theta_wall = math.atan2(dy, dx) if abs(dx) > 1e-15 else 0.0

    # Sample radial stations across exit plane
    # Assume flow angle varies linearly from 0 (axis) to θ_wall (wall)
    r_samples = np.linspace(0, Re, n_samples)

    numerator = 0.0   # ∫ ρ·ux·|u|·r·dr
    denominator = 0.0  # ∫ ρ·|u|²·r·dr

    for i in range(n_samples - 1):
        r_mid = 0.5 * (r_samples[i] + r_samples[i + 1])
        dr = r_samples[i + 1] - r_samples[i]
        frac = r_mid / Re

        # Local flow angle (linear assumption from axis to wall)
        theta_local = frac * theta_wall

        # Local area ratio
        if r_mid < 1e-15:
            M_local = mach_from_area_ratio(1.001, gamma, supersonic=True)
        else:
            ar = (r_mid / Rt) ** 2
            ar = max(ar, 1.0 + 1e-6)
            try:
                M_local = mach_from_area_ratio(ar, gamma, supersonic=True)
            except ValueError:
                M_local = 1.5

        rho = isentropic_density_ratio(M_local, gamma)
        T = isentropic_temperature_ratio(M_local, gamma)
        V = M_local * math.sqrt(T)
        ux = V * math.cos(theta_local)

        numerator += rho * ux * V * r_mid * dr
        denominator += rho * V * V * r_mid * dr

    if denominator < 1e-30:
        return 1.0

    return numerator / denominator


def nozzle_efficiency(Cf_actual: float, Cf_ideal: float) -> float:
    """
    Nozzle efficiency factor.

        η_n = Cf_actual / Cf_ideal

    Per NASA "Perfect Bell Nozzle" report (1983): Cf_ideal is the
    1D isentropic thrust coefficient for the given area ratio.
    """
    if abs(Cf_ideal) < 1e-15:
        return 0.0
    return Cf_actual / Cf_ideal


def compare_contours(
    contours: dict[str, dict],
    Pc: float,
    Pa: float,
    gamma: float,
) -> list[dict]:
    """
    Generate a comparison table for multiple nozzle contours.

    Parameters
    ----------
    contours : dict mapping name → contour dict
    Pc       : chamber pressure [Pa]
    Pa       : ambient pressure [Pa]
    gamma    : ratio of specific heats

    Returns
    -------
    list of dicts, one per contour, with comparison metrics
    """
    results = []

    for name, c in contours.items():
        epsilon = c['epsilon']
        Rt = c['Rt']

        # 1D ideal performance
        Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
        Pe_Pc = isentropic_pressure_ratio(Me, gamma)
        Pa_Pc = Pa / Pc
        Cf_ideal = thrust_coefficient(Me, gamma, Pe_Pc, Pa_Pc, epsilon)

        # 2D divergence loss
        eta_div = divergence_loss_2d(c, gamma)

        # Corrected Cf
        Cf_corrected = Cf_ideal * eta_div

        # Nozzle efficiency
        eta_n = nozzle_efficiency(Cf_corrected, Cf_ideal)

        # Geometric data
        Ln = c.get('Ln', 0.0)
        theta_n = c.get('theta_n', 0.0)
        theta_e = c.get('theta_e', 0.0)
        length_pct = c.get('length_pct', 0.0)

        # Conical baseline comparison
        Re = c.get('Re', math.sqrt(epsilon) * Rt)
        L15 = (Re - Rt) / math.tan(math.radians(15.0))
        actual_pct = (Ln / L15 * 100) if L15 > 0 else 0.0

        results.append({
            'name': name,
            'epsilon': epsilon,
            'Ln_mm': Ln * 1000,
            'length_pct': actual_pct,
            'theta_n_deg': theta_n,
            'theta_e_deg': theta_e,
            'Cf_ideal': Cf_ideal,
            'eta_div': eta_div,
            'Cf_corrected': Cf_corrected,
            'eta_n': eta_n,
            'Cf_loss_pct': (1.0 - eta_div) * 100,
            'contour_type': c.get('contour_type', c.get('method', 'unknown')),
        })

    return results


def print_comparison_table(results: list[dict]) -> str:
    """Format comparison results as a text table."""
    lines = []
    lines.append("")
    lines.append("  ── Nozzle Contour Comparison ─────────────────────────────────")
    lines.append(f"  {'Method':<16s} {'ε':>6s} {'L [mm]':>8s} {'L%':>6s} "
                 f"{'θ_n[°]':>7s} {'θ_e[°]':>7s} {'Cf_1D':>7s} "
                 f"{'η_div':>7s} {'Cf_2D':>7s} {'Loss%':>6s}")
    lines.append("  " + "─" * 82)

    for r in results:
        lines.append(
            f"  {r['name']:<16s} {r['epsilon']:6.1f} {r['Ln_mm']:8.1f} "
            f"{r['length_pct']:6.1f} {r['theta_n_deg']:7.1f} "
            f"{r['theta_e_deg']:7.1f} {r['Cf_ideal']:7.4f} "
            f"{r['eta_div']:7.4f} {r['Cf_corrected']:7.4f} "
            f"{r['Cf_loss_pct']:6.2f}"
        )

    lines.append("")
    return "\n".join(lines)


def plot_contour_comparison(
    contours: dict[str, dict],
    results: list[dict] | None = None,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay plot of multiple nozzle contour profiles.

    Parameters
    ----------
    contours  : dict mapping name → contour dict
    results   : comparison results (for annotation)
    show      : display the plot
    save_path : path to save the figure
    """
    colors = ['#1a73e8', '#d93025', '#0d652d', '#e8710a', '#9334e6']
    styles = ['-', '--', '-.', ':', '-']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # Top: contour overlay
    ax = axes[0]
    for idx, (name, c) in enumerate(contours.items()):
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        x_mm = c['x'] * 1000
        y_mm = c['y'] * 1000
        ax.plot(x_mm, y_mm, color=color, ls=style, lw=2, label=name)
        ax.plot(x_mm, -y_mm, color=color, ls=style, lw=2, alpha=0.3)

    ax.axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('r [mm]')
    ax.set_title('Nozzle Contour Comparison', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, ls=':', alpha=0.3)
    ax.set_aspect('equal')

    # Bottom: bell section detail (divergent only)
    ax2 = axes[1]
    for idx, (name, c) in enumerate(contours.items()):
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        x_b = c.get('x_bell', c['x'])
        y_b = c.get('y_bell', c['y'])
        ax2.plot(x_b * 1000, y_b * 1000, color=color,
                 ls=style, lw=2, label=name)

    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('r [mm]')
    ax2.set_title('Divergent Section Detail', fontweight='bold', fontsize=10)
    ax2.grid(True, ls=':', alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig
