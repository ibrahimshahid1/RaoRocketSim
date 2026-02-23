"""
separation.py – Flow separation prediction for overexpanded nozzles.

Implements three empirical separation criteria widely used in rocket nozzle
design, per the survey in Stark (2009) and NASA SP-8120:

  • Summerfield (simple):   p_sep ≈ 0.4 · Pa
  • Kalt-Badal:             p_sep/Pa ≈ 1 / (1 + 0.5·γ·Me²)^0.195  (approx)
  • Schmucker (turbulent):  p_sep/Pa ≈ (Pa/Pc)^0.8 / Me

References
----------
- NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
- R. Stark, "Flow Separation in Rocket Nozzles – An Overview" (2009)
"""

from __future__ import annotations
import math
import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
)


def summerfield_separation_pressure(Pa: float) -> float:
    """
    Summerfield criterion (simplest rule-of-thumb):
        p_sep ≈ 0.4 · Pa

    Returns the wall pressure at which separation is expected.
    """
    return 0.4 * Pa


def kalt_badal_separation_ratio(Me: float, gamma: float) -> float:
    """
    Kalt-Badal criterion.  Returns p_sep / Pa.

        p_sep/Pa ≈ (1 / (1.88·Me − 1))

    Valid for Me > ~1.5.
    """
    if Me <= 1.0:
        return float('inf')  # no separation for subsonic
    denom = 1.88 * Me - 1.0
    if denom <= 0:
        return float('inf')
    return 1.0 / denom


def schmucker_separation_ratio(Me: float, Pa_over_Pc: float) -> float:
    """
    Schmucker criterion (fully turbulent BL):
        p_sep/Pc ≈ (Pa/Pc)^0.8 · Me^(-1)

    Returns p_sep / Pc.
    """
    if Me <= 1.0:
        return 1.0
    return (Pa_over_Pc ** 0.8) / Me


def check_separation(
    contour: dict,
    Pc: float,
    Pa: float,
    gamma: float,
    method: str = 'schmucker',
) -> dict:
    """
    Check whether the nozzle will experience flow separation at the given
    ambient pressure.

    Parameters
    ----------
    contour : dict from ``bell_nozzle_contour``
    Pc      : chamber pressure  [Pa]
    Pa      : ambient pressure  [Pa]
    gamma   : ratio of specific heats
    method  : 'summerfield', 'kalt_badal', or 'schmucker'

    Returns
    -------
    dict with:
        'separated'     : bool
        'method'        : str
        'p_sep'         : separation pressure  [Pa]
        'x_sep'         : axial location of separation  [m]  (None if no sep)
        'y_sep'         : radial location  [m]  (None if no sep)
        'margin'        : Pe/p_sep  (>1 means no separation)
        'exit_pressure' : Pe  [Pa]
    """
    x = contour['x']
    y = contour['y']
    Rt = contour['Rt']
    At = np.pi * Rt**2
    epsilon = contour['epsilon']

    # Exit conditions
    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
    Pe = Pc * isentropic_pressure_ratio(Me, gamma)

    # Compute separation pressure
    if method == 'summerfield':
        p_sep = summerfield_separation_pressure(Pa)
    elif method == 'kalt_badal':
        ratio = kalt_badal_separation_ratio(Me, gamma)
        p_sep = ratio * Pa
    elif method == 'schmucker':
        ratio_Pc = schmucker_separation_ratio(Me, Pa / Pc)
        p_sep = ratio_Pc * Pc
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use 'summerfield', 'kalt_badal', or 'schmucker'.")

    separated = Pe < p_sep
    x_sep = None
    y_sep = None

    if separated:
        # Find where wall pressure drops below p_sep
        throat_idx = np.argmin(np.abs(y - Rt))
        for i in range(throat_idx, len(x)):
            A_local = np.pi * y[i]**2
            ar = max(A_local / At, 1.0)
            try:
                M_local = mach_from_area_ratio(ar, gamma, supersonic=True)
            except Exception:
                continue
            p_local = Pc * isentropic_pressure_ratio(M_local, gamma)
            if p_local <= p_sep:
                x_sep = float(x[i])
                y_sep = float(y[i])
                break

    margin = Pe / p_sep if p_sep > 0 else float('inf')

    return {
        'separated': separated,
        'method': method,
        'p_sep': p_sep,
        'x_sep': x_sep,
        'y_sep': y_sep,
        'margin': margin,
        'exit_pressure': Pe,
    }


def separation_summary(result: dict) -> str:
    """Format a human-readable separation check summary."""
    lines = []
    lines.append(f"  Separation check ({result['method']}):")
    lines.append(f"    Exit pressure Pe = {result['exit_pressure']:.0f} Pa")
    lines.append(f"    Separation pressure p_sep = {result['p_sep']:.0f} Pa")
    lines.append(f"    Margin Pe/p_sep = {result['margin']:.3f}")
    if result['separated']:
        lines.append(f"    ⚠  SEPARATION PREDICTED at x = {result['x_sep']*1000:.1f} mm")
    else:
        lines.append(f"    ✓  No separation expected")
    return "\n".join(lines)
