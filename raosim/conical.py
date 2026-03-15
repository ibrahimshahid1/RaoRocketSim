"""
conical.py – Conical nozzle contour and divergence correction.

The conical nozzle is the simplest diverging nozzle design and serves as
the baseline reference for bell nozzle length (the "80% bell" convention
means the bell is 80% of the length of a 15° half-angle cone with the
same area ratio).

Divergence correction (Rao 1961):
    η_div = (1 + cos α) / 2

For α = 15°, η_div ≈ 0.9830 → the 1D Cf decrement is ~1.7%.

References:
  - G. V. R. Rao, "Recent Developments in Rocket Nozzle Configurations,"
    ARS Journal (1961)
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
"""

from __future__ import annotations
import math
import numpy as np


def conical_divergence_factor(alpha: float) -> float:
    """
    Conical-flow divergence correction factor.

        η_div = (1 + cos α) / 2

    Parameters
    ----------
    alpha : half-angle of the conical nozzle [radians]

    Returns
    -------
    η_div ∈ (0, 1]
    """
    return 0.5 * (1.0 + math.cos(alpha))


def conical_nozzle_length(Rt: float, epsilon: float,
                          half_angle_deg: float = 15.0) -> float:
    """
    Axial length of a conical diverging section.

        L = (Re - Rt) / tan(α)

    Parameters
    ----------
    Rt            : throat radius [m]
    epsilon       : expansion ratio Ae/At
    half_angle_deg : cone half-angle [°]

    Returns
    -------
    Length [m]
    """
    Re = math.sqrt(epsilon) * Rt
    return (Re - Rt) / math.tan(math.radians(half_angle_deg))


def conical_nozzle_contour(
    Rt: float,
    epsilon: float,
    half_angle_deg: float = 15.0,
    n_pts: int = 200,
    convergent_half_angle_deg: float = 45.0,
    Ru_factor: float = 1.5,
    Rd_factor: float = 0.382,
) -> dict:
    """
    Generate a conical nozzle contour (convergent arc + throat arc + cone).

    The convergent and throat arcs use the same Ru/Rd fillet radii as the
    bell nozzle, so the throat region is identical.  Only the divergent
    section differs: a straight cone instead of a shaped bell.

    Parameters
    ----------
    Rt                      : throat radius [m]
    epsilon                 : expansion ratio Ae/At (must be > 1)
    half_angle_deg          : divergent cone half-angle [°] (default 15)
    n_pts                   : points per section (default 200)
    convergent_half_angle_deg : upstream inlet half-angle [°] (default 45)
    Ru_factor               : upstream curvature / Rt (default 1.5)
    Rd_factor               : downstream curvature / Rt (default 0.382)

    Returns
    -------
    dict with keys compatible with bell_nozzle_contour:
        'x', 'y'         : full contour arrays [m]
        'theta_n'         : initial wall angle (= half_angle) [°]
        'theta_e'         : exit wall angle (= half_angle) [°]
        'Ln'              : nozzle length [m]
        'Re'              : exit radius [m]
        'Rt'              : throat radius [m]
        'Ru', 'Rd'        : fillet radii [m]
        'epsilon'         : expansion ratio
        'eta_div'         : conical divergence factor
        'x_conv', 'y_conv', 'x_throat', 'y_throat', 'x_div', 'y_div'
    """
    if epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")

    alpha = math.radians(half_angle_deg)
    Re = math.sqrt(epsilon) * Rt
    Ru = Ru_factor * Rt
    Rd = Rd_factor * Rt

    # --- Section 1: Convergent circular arc (same as bell) ---
    y_cu = Rt + Ru
    x_cu = 0.0
    angle_start_conv = -(math.pi / 2.0 + math.radians(convergent_half_angle_deg))
    angle_end_conv = -math.pi / 2.0
    t_conv = np.linspace(angle_start_conv, angle_end_conv, n_pts)
    x_conv = x_cu + Ru * np.cos(t_conv)
    y_conv = y_cu + Ru * np.sin(t_conv)

    # --- Section 2: Downstream throat arc ---
    # Arc from -π/2 to (α - π/2), where α is the cone half-angle
    y_cd = Rt + Rd
    x_cd = 0.0
    angle_start_throat = -math.pi / 2.0
    angle_end_throat = alpha - math.pi / 2.0
    t_thr = np.linspace(angle_start_throat, angle_end_throat, n_pts)
    x_throat = x_cd + Rd * np.cos(t_thr)
    y_throat = y_cd + Rd * np.sin(t_thr)

    # --- Section 3: Straight conical divergent section ---
    # Start where the throat arc ends
    x_start = x_throat[-1]
    y_start = y_throat[-1]

    # Length of the cone from the inflection point to exit
    Ln_full = conical_nozzle_length(Rt, epsilon, half_angle_deg)
    x_end = Ln_full
    y_end = Re

    x_div = np.linspace(x_start, x_end, n_pts)
    y_div = y_start + (x_div - x_start) * math.tan(alpha)

    # --- Assemble full contour ---
    x_full = np.concatenate([x_conv, x_throat, x_div])
    y_full = np.concatenate([y_conv, y_throat, y_div])

    eta_div = conical_divergence_factor(alpha)

    return {
        'x': x_full,
        'y': y_full,
        'theta_n': half_angle_deg,
        'theta_e': half_angle_deg,
        'Ln': Ln_full,
        'Re': Re,
        'Rt': Rt,
        'Ru': Ru,
        'Rd': Rd,
        'epsilon': epsilon,
        'length_pct': 100.0,   # conical is the 100% length reference
        'eta_div': eta_div,
        'contour_type': 'conical',
        'half_angle_deg': half_angle_deg,
        'N': (x_throat[-1], y_throat[-1]),
        'E': (x_end, y_end),
        'P1': (0.5 * (x_throat[-1] + x_end),
               0.5 * (y_throat[-1] + y_end)),
        'x_conv': x_conv,
        'y_conv': y_conv,
        'x_throat': x_throat,
        'y_throat': y_throat,
        'x_bell': x_div,    # compatibility alias
        'y_bell': y_div,
    }
