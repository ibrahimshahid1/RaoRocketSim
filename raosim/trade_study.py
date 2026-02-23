"""
trade_study.py – Parameter sweep and trade-study plotting.

Sweep one design variable while holding others constant.  Returns results
as a list of dicts (easily convertible to a DataFrame) and produces
multi-panel comparison plots.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from raosim.gas_dynamics import mach_from_area_ratio, isentropic_pressure_ratio
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.engine import compute_engine_performance
from raosim.propellants import Propellant


def sweep_epsilon(
    epsilons: np.ndarray | list[float],
    Pc: float, Pa: float, Rt: float,
    prop: Propellant,
    length_pct: float = 80.0,
) -> list[dict]:
    """
    Sweep expansion ratio ε and compute performance for each.

    Returns list of dicts with keys:
        epsilon, Me, Pe, Cf, Isp, thrust, m_dot, Ln, theta_n, theta_e
    """
    results = []
    for eps in epsilons:
        perf = compute_engine_performance(Pc, Pa, Rt, eps, prop)
        try:
            tn, te = lookup_angles(eps, length_pct)
        except Exception:
            tn, te = None, None
        try:
            contour = bell_nozzle_contour(Rt, eps, tn, te, length_pct)
            Ln = contour['Ln']
        except Exception:
            Ln = None
        results.append({
            'epsilon': eps,
            'Me': perf.Me,
            'Pe': perf.Pe,
            'Cf': perf.Cf_actual,
            'Isp': perf.Isp,
            'thrust': perf.thrust,
            'm_dot': perf.m_dot,
            'Ln': Ln,
            'theta_n': tn,
            'theta_e': te,
        })
    return results


def sweep_Pc(
    pressures_bar: np.ndarray | list[float],
    Pa: float, Rt: float, epsilon: float,
    prop: Propellant,
) -> list[dict]:
    """Sweep chamber pressure Pc [bar]."""
    results = []
    for Pc_bar in pressures_bar:
        Pc = Pc_bar * 1e5
        perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)
        results.append({
            'Pc_bar': Pc_bar,
            'Me': perf.Me,
            'Pe': perf.Pe,
            'Cf': perf.Cf_actual,
            'Isp': perf.Isp,
            'thrust': perf.thrust,
            'm_dot': perf.m_dot,
        })
    return results


def sweep_Rt(
    radii_mm: np.ndarray | list[float],
    Pc: float, Pa: float, epsilon: float,
    prop: Propellant,
) -> list[dict]:
    """Sweep throat radius Rt [mm]."""
    results = []
    for Rt_mm in radii_mm:
        Rt = Rt_mm / 1000.0
        perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)
        results.append({
            'Rt_mm': Rt_mm,
            'Me': perf.Me,
            'Pe': perf.Pe,
            'Cf': perf.Cf_actual,
            'Isp': perf.Isp,
            'thrust': perf.thrust,
            'm_dot': perf.m_dot,
        })
    return results


def plot_trade_study(results: list[dict], x_key: str,
                     title: str = "Trade Study",
                     *, show: bool = True,
                     save_path: str | None = None) -> plt.Figure:
    """
    Multi-panel trade-study plot.

    Parameters
    ----------
    results : list of dicts from a sweep function
    x_key   : the key to use as the x-axis (e.g. 'epsilon', 'Pc_bar')
    title   : plot super-title
    """
    x_vals = [r[x_key] for r in results]

    # Determine which y-keys to plot
    y_keys = []
    labels = {}
    if 'Isp' in results[0]:
        y_keys.append('Isp')
        labels['Isp'] = 'Isp [s]'
    if 'Cf' in results[0]:
        y_keys.append('Cf')
        labels['Cf'] = 'Cf (actual)'
    if 'thrust' in results[0]:
        y_keys.append('thrust')
        labels['thrust'] = 'Thrust [N]'
    if 'Ln' in results[0] and results[0]['Ln'] is not None:
        y_keys.append('Ln')
        labels['Ln'] = 'Nozzle length [m]'
    if 'm_dot' in results[0]:
        y_keys.append('m_dot')
        labels['m_dot'] = 'ṁ [kg/s]'

    n_panels = len(y_keys)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    colors = ['#1a73e8', '#d93025', '#0d652d', '#e8710a', '#9334e6']
    for i, key in enumerate(y_keys):
        y_vals = [r[key] for r in results]
        if None in y_vals:
            # Filter out Nones
            pairs = [(xv, yv) for xv, yv in zip(x_vals, y_vals) if yv is not None]
            if not pairs:
                continue
            xf, yf = zip(*pairs)
        else:
            xf, yf = x_vals, y_vals
        axes[i].plot(xf, yf, 'o-', color=colors[i % len(colors)], lw=2, ms=4)
        axes[i].set_ylabel(labels.get(key, key), fontsize=10)
        axes[i].grid(True, ls=':', alpha=0.4)

    axes[-1].set_xlabel(x_key, fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig
