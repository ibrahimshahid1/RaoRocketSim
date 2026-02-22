"""
nozzle_geometry.py – Rao Thrust-Optimized Parabolic (TOP) bell nozzle contour.

Generates the three-section contour:
  1. Upstream circular arc  (convergent side, R_u = 1.5 Rₜ)
  2. Downstream circular arc (supersonic side, R_d = 0.382 Rₜ)
  3. Quadratic Bézier bell  (canted parabola from inflection N to exit E)

References:
  - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958/1961
  - NASA SP-8120, "Liquid Rocket Engine Nozzles"
  - Seitzman, Georgia Tech AE 6450 nozzle-geometry lecture notes
"""

from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ══════════════════════════════════════════════════════════════════════
#  Empirical angle tables  (θ_n, θ_e) vs (ε, L%)
# ══════════════════════════════════════════════════════════════════════
# Digitised from NASA SP-8120 / Rao (1961) / Seitzman lecture charts.
# Rows = expansion ratios, Columns = bell length fractions.

_EPSILON_VALS = np.array([4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                         dtype=float)

_LPCT_VALS = np.array([60, 70, 80, 90, 100], dtype=float)

#  θ_n  (initial wall angle at inflection, degrees)
#  Shape: (len(_EPSILON_VALS), len(_LPCT_VALS))
_THETA_N_TABLE = np.array([
    # L%:  60     70     80     90    100           ε
    [33.0, 28.5, 25.0, 22.5, 20.5],  #  4
    [35.0, 30.0, 26.5, 23.5, 21.5],  #  5
    [36.5, 31.5, 27.5, 24.5, 22.0],  #  6
    [38.5, 33.0, 29.0, 26.0, 23.5],  #  8
    [40.0, 34.5, 30.0, 27.0, 24.5],  # 10
    [42.0, 36.5, 32.0, 28.5, 26.0],  # 15
    [43.5, 38.0, 33.0, 29.5, 27.0],  # 20
    [44.5, 38.5, 34.0, 30.5, 27.5],  # 25
    [45.0, 39.0, 34.5, 31.0, 28.0],  # 30
    [45.5, 39.5, 35.0, 31.5, 28.5],  # 35
    [46.0, 40.0, 35.5, 32.0, 29.0],  # 40
    [46.0, 40.0, 35.5, 32.0, 29.0],  # 45
    [46.5, 40.5, 36.0, 32.5, 29.5],  # 50
], dtype=float)

#  θ_e  (exit wall angle, degrees)
_THETA_E_TABLE = np.array([
    # L%:  60     70     80     90    100           ε
    [17.0, 14.0, 11.5,  9.5,  8.0],  #  4
    [18.5, 15.0, 12.5, 10.5,  8.5],  #  5
    [19.5, 16.0, 13.5, 11.0,  9.0],  #  6
    [21.0, 17.5, 14.5, 12.0, 10.0],  #  8
    [22.0, 18.5, 15.5, 13.0, 10.5],  # 10
    [24.0, 20.5, 17.0, 14.0, 11.5],  # 15
    [25.5, 21.5, 18.0, 15.0, 12.5],  # 20
    [26.5, 22.5, 18.5, 15.5, 13.0],  # 25
    [27.0, 23.0, 19.0, 16.0, 13.5],  # 30
    [27.5, 23.5, 19.5, 16.0, 13.5],  # 35
    [28.0, 24.0, 20.0, 16.5, 14.0],  # 40
    [28.0, 24.0, 20.0, 16.5, 14.0],  # 45
    [28.5, 24.5, 20.0, 17.0, 14.0],  # 50
], dtype=float)

# Build interpolators (bilinear on the (ε, L%) grid)
_interp_theta_n = RegularGridInterpolator(
    (_EPSILON_VALS, _LPCT_VALS), _THETA_N_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)
_interp_theta_e = RegularGridInterpolator(
    (_EPSILON_VALS, _LPCT_VALS), _THETA_E_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)


def lookup_angles(epsilon: float, length_pct: float) -> tuple[float, float]:
    """
    Interpolate the standard (θ_n, θ_e) from Rao/NASA charts.

    Parameters
    ----------
    epsilon    : area expansion ratio  Ae/At
    length_pct : bell length as percent of 15° half-angle cone (60–100)

    Returns
    -------
    (theta_n, theta_e) in degrees
    """
    pt = np.array([[epsilon, length_pct]])
    theta_n = float(_interp_theta_n(pt)[0])
    theta_e = float(_interp_theta_e(pt)[0])
    return theta_n, theta_e


# ══════════════════════════════════════════════════════════════════════
#  Contour generation
# ══════════════════════════════════════════════════════════════════════

def _full_cone_length(Rt: float, epsilon: float) -> float:
    """Length of a 15° half-angle conical nozzle with the same ε."""
    Re = math.sqrt(epsilon) * Rt
    return (Re - Rt) / math.tan(math.radians(15.0))


def bell_nozzle_contour(
    Rt: float,
    epsilon: float,
    theta_n_deg: float | None = None,
    theta_e_deg: float | None = None,
    length_pct: float = 80.0,
    n_pts: int = 200,
    convergent_half_angle_deg: float = 45.0,
    Ru_factor: float = 1.5,
    Rd_factor: float = 0.382,
) -> dict:
    """
    Generate a Rao TOP bell nozzle contour.

    Parameters
    ----------
    Rt          : throat radius  [m]
    epsilon     : area expansion ratio  Ae/At  (must be > 1)
    theta_n_deg : initial wall angle at inflection [°]  (None → lookup)
    theta_e_deg : exit wall angle [°]  (None → lookup)
    length_pct  : bell length as % of 15° cone  (default 80)
    n_pts       : points per contour section  (default 200)
    convergent_half_angle_deg : upstream inlet half-angle  (default 45°)
    Ru_factor   : upstream curvature / Rt  (default 1.5)
    Rd_factor   : downstream curvature / Rt  (default 0.382)

    Returns
    -------
    dict with keys:
        'x', 'y'        : 1-D numpy arrays of the full contour  [m]
        'theta_n'        : actual θ_n used  [°]
        'theta_e'        : actual θ_e used  [°]
        'Ln'             : bell length  [m]
        'Re'             : exit radius  [m]
        'Ru'             : upstream fillet radius  [m]
        'Rd'             : downstream fillet radius  [m]
        'N'              : (x, y) of inflection point
        'E'              : (x, y) of exit point
        'P1'             : (x, y) of Bézier control point
        'x_conv', 'y_conv'   : convergent arc arrays
        'x_throat', 'y_throat' : downstream arc arrays
        'x_bell', 'y_bell'    : bell Bézier arrays
    """
    if epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")

    # ── Lookup angles if not provided ─────────────────────────────
    if theta_n_deg is None or theta_e_deg is None:
        tn_lookup, te_lookup = lookup_angles(epsilon, length_pct)
        if theta_n_deg is None:
            theta_n_deg = tn_lookup
        if theta_e_deg is None:
            theta_e_deg = te_lookup

    theta_n = math.radians(theta_n_deg)
    theta_e = math.radians(theta_e_deg)

    Re = math.sqrt(epsilon) * Rt
    Ru = Ru_factor * Rt          # upstream fillet radius
    Rd = Rd_factor * Rt          # downstream fillet radius
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)

    # ────────────────────────────────────────────────────────────────
    # Section 1: Upstream circular arc (convergent side)
    # ────────────────────────────────────────────────────────────────
    # Arc center is on the y-axis at y_cu = Rt + Ru
    # The arc sweeps from the convergent inlet angle to the throat.
    # At the throat (bottom of arc), the tangent is horizontal → angle = -π/2.
    # At the inlet side, angle = -(π/2 + convergent_half_angle).
    y_cu = Rt + Ru      # center y
    x_cu = 0.0          # center x  (throat at x=0)

    angle_start_conv = -(math.pi / 2.0 + math.radians(convergent_half_angle_deg))
    angle_end_conv = -math.pi / 2.0

    t_conv = np.linspace(angle_start_conv, angle_end_conv, n_pts)
    x_conv = x_cu + Ru * np.cos(t_conv)
    y_conv = y_cu + Ru * np.sin(t_conv)

    # ────────────────────────────────────────────────────────────────
    # Section 2: Downstream circular arc (supersonic side)
    # ────────────────────────────────────────────────────────────────
    # Arc center at (0, Rt + Rd).  The arc starts at the throat (-π/2)
    # and sweeps clockwise (angle increasing) up to θ_n relative to
    # horizontal, i.e., ends at angle (θ_n - π/2).
    y_cd = Rt + Rd
    x_cd = 0.0

    angle_start_throat = -math.pi / 2.0
    angle_end_throat = theta_n - math.pi / 2.0

    t_thr = np.linspace(angle_start_throat, angle_end_throat, n_pts)
    x_throat = x_cd + Rd * np.cos(t_thr)
    y_throat = y_cd + Rd * np.sin(t_thr)

    # Inflection point N  (end of downstream arc)
    Nx = x_throat[-1]
    Ny = y_throat[-1]

    # ────────────────────────────────────────────────────────────────
    # Section 3: Quadratic Bézier bell  (canted parabola N → E)
    # ────────────────────────────────────────────────────────────────
    Ex = Ln
    Ey = Re

    # Control point P1 = intersection of the two tangent rays
    #   Ray from N with slope tan(θ_n):   y - Ny = tan(θ_n)·(x - Nx)
    #   Ray from E with slope tan(θ_e):   y - Ey = tan(θ_e)·(x - Ex)
    m1 = math.tan(theta_n)
    m2 = math.tan(theta_e)

    if abs(m1 - m2) < 1e-12:
        raise ValueError("θ_n ≈ θ_e → tangent lines are parallel; "
                         "cannot form a Bézier bell section")

    # y = m1·x + (Ny - m1·Nx) = m2·x + (Ey - m2·Ex)
    c1 = Ny - m1 * Nx
    c2 = Ey - m2 * Ex
    P1x = (c2 - c1) / (m1 - m2)
    P1y = m1 * P1x + c1

    # Quadratic Bézier: B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2
    t = np.linspace(0.0, 1.0, n_pts)
    omt = 1.0 - t
    x_bell = omt**2 * Nx + 2.0 * omt * t * P1x + t**2 * Ex
    y_bell = omt**2 * Ny + 2.0 * omt * t * P1y + t**2 * Ey

    # ────────────────────────────────────────────────────────────────
    # Concatenate all three sections
    # ────────────────────────────────────────────────────────────────
    x_full = np.concatenate([x_conv, x_throat, x_bell])
    y_full = np.concatenate([y_conv, y_throat, y_bell])

    return {
        'x': x_full,
        'y': y_full,
        'theta_n': theta_n_deg,
        'theta_e': theta_e_deg,
        'Ln': Ln,
        'Re': Re,
        'Rt': Rt,
        'Ru': Ru,
        'Rd': Rd,
        'epsilon': epsilon,
        'length_pct': length_pct,
        'N': (Nx, Ny),
        'E': (Ex, Ey),
        'P1': (P1x, P1y),
        'x_conv': x_conv,
        'y_conv': y_conv,
        'x_throat': x_throat,
        'y_throat': y_throat,
        'x_bell': x_bell,
        'y_bell': y_bell,
    }


def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute signed curvature κ(s) along a planar parametric curve
    using finite differences.

        κ = (x'·y'' − y'·x'') / (x'² + y'²)^(3/2)

    Returns an array the same length as x (uses central differences in
    interior, one-sided at endpoints).
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature
