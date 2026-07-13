"""
separation.py – Flow separation prediction for overexpanded nozzles.

Implements three empirical separation criteria widely used in rocket nozzle
design, per the survey in Stark (2009) and NASA SP-8120:

  • Summerfield (simple):   p_sep ≈ 0.4 · Pa
  • Kalt-Badal:             p_sep/Pa ≈ 1 / (1.88·M − 1)
  • Schmucker (turbulent):  p_sep/Pc ≈ (Pa/Pc)^0.8 / M

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
    *,
    frozen_expansion=None,
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
    x = np.asarray(contour['x'], dtype=float)
    y = np.asarray(contour['y'], dtype=float)
    Rt = contour['Rt']
    At = np.pi * Rt**2
    epsilon = contour['epsilon']


    if frozen_expansion is not None:
        if not math.isclose(
            float(frozen_expansion.chamber_pressure_pa),
            float(Pc), rel_tol=1.0e-10, abs_tol=0.0,
        ):
            raise ValueError("frozen expansion chamber pressure does not match Pc")
        if not math.isclose(
            float(frozen_expansion.expansion_ratio),
            float(epsilon), rel_tol=1.0e-10, abs_tol=1.0e-12,
        ):
            raise ValueError("frozen expansion ratio does not match contour epsilon")
        Me = float(frozen_expansion.exit.mach)
        Pe = float(frozen_expansion.exit.pressure_pa)
    else:
        Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
        Pe = Pc * isentropic_pressure_ratio(Me, gamma)


    if method not in {'summerfield', 'kalt_badal', 'schmucker'}:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use 'summerfield', 'kalt_badal', or 'schmucker'.")

    def criterion_pressure(M_local: float) -> float:
        if method == 'summerfield':
            return summerfield_separation_pressure(Pa)
        if method == 'kalt_badal':
            return kalt_badal_separation_ratio(M_local, gamma) * Pa
        return schmucker_separation_ratio(M_local, Pa / Pc) * Pc

    # The Mach-dependent criteria must be evaluated at each candidate wall
    # station.  Using Me once and then marching against a constant threshold
    # mixes an exit condition with upstream wall states and can move the
    # predicted onset substantially.
    p_sep_exit = criterion_pressure(Me)
    separated = False
    x_sep = None
    y_sep = None
    onset_threshold = None
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    for i in range(throat_idx, len(x)):
        A_local = np.pi * y[i]**2
        ar = max(A_local / At, 1.0)
        if frozen_expansion is not None:
            station = frozen_expansion.station(ar, supersonic=True)
            M_local = station.mach
            p_local = station.pressure_pa
        else:
            try:
                M_local = mach_from_area_ratio(ar, gamma, supersonic=True)
            except Exception:
                continue
            p_local = Pc * isentropic_pressure_ratio(M_local, gamma)
        threshold = criterion_pressure(M_local)
        if p_local <= threshold:
            separated = True
            x_sep = float(x[i])
            y_sep = float(y[i])
            onset_threshold = float(threshold)
            break

    p_sep = onset_threshold if onset_threshold is not None else p_sep_exit
    margin = Pe / p_sep_exit if p_sep_exit > 0 else float('inf')

    return {
        'separated': separated,
        'method': method,
        'p_sep': p_sep,
        'exit_criterion_pressure': p_sep_exit,
        'onset_criterion_pressure': onset_threshold,
        'criterion_evaluated_locally': True,
        'x_sep': x_sep,
        'y_sep': y_sep,
        'margin': margin,
        'exit_pressure': Pe,
        'expansion_model': (
            'frozen_variable_cp_q1d'
            if frozen_expansion is not None else 'constant_gamma'
        ),
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
