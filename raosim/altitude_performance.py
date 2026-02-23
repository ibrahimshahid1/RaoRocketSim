"""
altitude_performance.py – Engine performance vs altitude map.

Sweeps altitude from sea level to vacuum, shows how thrust, Isp, and Cf
change as ambient pressure drops.  Marks the altitude at which flow
separation starts/stops.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from raosim.atmosphere import isa
from raosim.engine import compute_engine_performance, g0
from raosim.propellants import Propellant
from raosim.separation import check_separation
from raosim.nozzle_geometry import bell_nozzle_contour


def altitude_performance_map(
    Pc: float,
    Rt: float,
    epsilon: float,
    prop: Propellant,
    contour: dict | None = None,
    h_max: float = 100_000.0,
    n_points: int = 200,
    separation_method: str = 'schmucker',
) -> dict:
    """
    Compute engine performance as a function of altitude.

    Parameters
    ----------
    Pc, Rt, epsilon, prop : engine parameters
    contour               : nozzle contour dict (for separation check)
    h_max                 : max altitude  [m]  (default 100 km)
    n_points              : number of altitude samples
    separation_method     : criterion for separation check

    Returns
    -------
    dict with arrays:
        'h'            : altitudes  [m]
        'Pa'           : ambient pressures  [Pa]
        'thrust'       : thrust  [N]
        'Isp'          : specific impulse  [s]
        'Cf'           : thrust coefficient
        'Pe'           : exit pressure  [Pa]
        'separated'    : boolean array
        'h_sep_onset'  : altitude where separation ends (nozzle starts flowing
                         fully) [m], or None
    """
    altitudes = np.linspace(0, h_max, n_points)
    thrust_arr = np.zeros(n_points)
    Isp_arr = np.zeros(n_points)
    Cf_arr = np.zeros(n_points)
    Pa_arr = np.zeros(n_points)
    Pe_arr = np.zeros(n_points)
    sep_arr = np.zeros(n_points, dtype=bool)

    for i, h in enumerate(altitudes):
        _, Pa, _ = isa(h)
        Pa_arr[i] = Pa

        perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)
        thrust_arr[i] = perf.thrust
        Isp_arr[i] = perf.Isp
        Cf_arr[i] = perf.Cf_actual
        Pe_arr[i] = perf.Pe

        if contour is not None:
            sep_result = check_separation(contour, Pc, Pa, prop.gamma,
                                          method=separation_method)
            sep_arr[i] = sep_result['separated']

    # Find altitude where separation ends (transition from separated to not)
    h_sep_onset = None
    for i in range(1, n_points):
        if sep_arr[i - 1] and not sep_arr[i]:
            # Linearly interpolate
            h_sep_onset = float(altitudes[i])
            break

    return {
        'h': altitudes,
        'Pa': Pa_arr,
        'thrust': thrust_arr,
        'Isp': Isp_arr,
        'Cf': Cf_arr,
        'Pe': Pe_arr,
        'separated': sep_arr,
        'h_sep_onset': h_sep_onset,
    }


def plot_altitude_performance(apm: dict, *, show: bool = True,
                              save_path: str | None = None) -> plt.Figure:
    """Multi-panel altitude performance plot."""
    h_km = apm['h'] / 1000.0
    sep = apm['separated']
    h_sep = apm['h_sep_onset']

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Thrust
    ax = axes[0]
    ax.plot(h_km, apm['thrust'] / 1000, color='#1a73e8', lw=2)
    if np.any(sep):
        ax.fill_between(h_km, 0, apm['thrust'].max() / 1000,
                         where=sep, alpha=0.1, color='#d93025',
                         label='Separated flow')
    if h_sep is not None:
        ax.axvline(h_sep / 1000, color='#d93025', ls='--', lw=1,
                   label=f'Sep. clears @ {h_sep/1000:.1f} km')
    ax.set_ylabel('Thrust [kN]')
    ax.legend(fontsize=8)
    ax.grid(True, ls=':', alpha=0.4)

    # Panel 2: Isp
    ax = axes[1]
    ax.plot(h_km, apm['Isp'], color='#0d652d', lw=2)
    if np.any(sep):
        ax.fill_between(h_km, apm['Isp'].min(), apm['Isp'].max(),
                         where=sep, alpha=0.1, color='#d93025')
    ax.set_ylabel('Isp [s]')
    ax.grid(True, ls=':', alpha=0.4)

    # Panel 3: Cf
    ax = axes[2]
    ax.plot(h_km, apm['Cf'], color='#e8710a', lw=2)
    if np.any(sep):
        ax.fill_between(h_km, apm['Cf'].min(), apm['Cf'].max(),
                         where=sep, alpha=0.1, color='#d93025')
    ax.set_ylabel('Cf (actual)')
    ax.set_xlabel('Altitude [km]')
    ax.grid(True, ls=':', alpha=0.4)

    fig.suptitle('Engine Performance vs Altitude', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig
