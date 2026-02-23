"""
wall_pressure.py – Estimate wall pressure distribution p(x) along the nozzle.

Uses the local area ratio A(x) = π·y(x)² to find M(x) via the area-Mach
relation, then applies the isentropic pressure relation.

Includes a monotonicity check per NASA SP-8120: wall pressure should
decrease continuously toward the exit.  A positive dp/dx downstream of the
throat signals separation risk.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from raosim.gas_dynamics import (
    mach_from_area_ratio,
    isentropic_pressure_ratio,
    area_mach_relation,
)


def wall_pressure_distribution(
    contour: dict,
    Pc: float,
    gamma: float,
) -> dict:
    """
    Compute the wall pressure p(x) along the nozzle contour.

    Parameters
    ----------
    contour : dict from ``bell_nozzle_contour``
    Pc      : chamber stagnation pressure  [Pa]
    gamma   : ratio of specific heats

    Returns
    -------
    dict with keys:
        'x'         : axial coordinates  [m]
        'p'         : wall pressure  [Pa]
        'p_over_Pc' : p/Pc
        'M'         : local Mach number
        'monotonic'        : True if p decreases continuously downstream of throat
        'violation_indices': array of indices where dp/dx > 0 downstream of throat
    """
    x = contour['x']
    y = contour['y']
    Rt = contour['Rt']
    At = np.pi * Rt**2

    n = len(x)
    M_arr = np.zeros(n)
    p_arr = np.zeros(n)

    # Find throat (minimum y, closest to Rt at x≈0)
    throat_idx = np.argmin(np.abs(y - Rt))

    for i in range(n):
        A_local = np.pi * y[i]**2
        ar = A_local / At

        if ar < 1.0:
            ar = 1.0  # clamp at throat

        try:
            if i <= throat_idx:
                # Upstream of throat → subsonic branch
                M_arr[i] = mach_from_area_ratio(ar, gamma, supersonic=False)
            else:
                # Downstream of throat → supersonic branch
                M_arr[i] = mach_from_area_ratio(ar, gamma, supersonic=True)
        except (ValueError, ZeroDivisionError):
            M_arr[i] = 1.0

        p_arr[i] = Pc * isentropic_pressure_ratio(M_arr[i], gamma)

    # Monotonicity check downstream of throat
    p_downstream = p_arr[throat_idx:]
    dp = np.diff(p_downstream)
    violations = np.where(dp > 0)[0] + throat_idx
    is_monotonic = len(violations) == 0

    return {
        'x': x,
        'p': p_arr,
        'p_over_Pc': p_arr / Pc,
        'M': M_arr,
        'monotonic': is_monotonic,
        'violation_indices': violations,
        'throat_idx': throat_idx,
    }


def plot_wall_pressure(wp: dict, *, show: bool = True,
                       save_path: str | None = None) -> plt.Figure:
    """Plot p(x)/Pc with throat and any violation regions highlighted."""
    x = wp['x']
    p_Pc = wp['p_over_Pc']
    throat_idx = wp['throat_idx']
    violations = wp['violation_indices']

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Pressure
    ax1.plot(x * 1000, p_Pc, color='#1a73e8', lw=2, label='p(x)/Pc')
    ax1.axvline(x[throat_idx] * 1000, color='grey', ls='--', lw=0.8,
                label='Throat')

    # Highlight violations
    if len(violations) > 0:
        for idx in violations:
            ax1.axvspan(x[idx] * 1000, x[min(idx + 1, len(x) - 1)] * 1000,
                        color='#d93025', alpha=0.15)
        ax1.plot([], [], color='#d93025', alpha=0.3, lw=8,
                 label='⚠ dp/dx > 0 (separation risk)')

    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('p / Pc')
    ax1.set_title('Wall Pressure Distribution', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, ls=':', alpha=0.4)

    # Mach on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x * 1000, wp['M'], color='#e8710a', lw=1.2, ls='--',
             alpha=0.7, label='M(x)')
    ax2.set_ylabel('Mach number', color='#e8710a')
    ax2.tick_params(axis='y', labelcolor='#e8710a')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig
