"""
nozzle_geometry.py – Rao Thrust-Optimized Parabolic (TOP) bell nozzle contour.

Generates the three-section contour:
  1. Upstream circular arc  (convergent side, R_u = 1.5 Rₜ)
  2. Downstream circular arc (supersonic side, R_d = 0.382 Rₜ)
  3. Quadratic Bézier bell  (canted parabola from inflection N to exit E)

References:
  - NASA SP-125, Figure 4-14 / Eq.4-7 / worked example4-2 (PDF pp100,107)
  - NASA SP-8120, Figure5 and section2.1.2.1.2 (PDF p29), conceptual cross-check
  - The originating Rao1960 chart paper is not held; no inspection claimed.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from raosim.throat_geometry import ThroatGeometrySpec, resolve_throat_geometry
from raosim.validation import add_contour_reliability_metadata



# Scientific basis: model_id=MODEL_NOZZLE_RAO_TOP_BEZIER,
# source=DOC_NTRS_19710019929, Figure 4-14 (PDF p.100), worked point p.107.
# Reviewer measurements, chart uncertainty and exact source anchor are kept
# together.  Rao 1960 is not held; no original-chart-paper inspection claimed.
from raosim.rao_top_chart import (
    EPSILON as _EPSILON_VALS,
    LENGTH_PERCENT as _LPCT_VALS,
    THETA_N_DEG as _THETA_N_TABLE,
    THETA_E_DEG as _THETA_E_TABLE,
    EVIDENCE as _RAO_CHART_EVIDENCE,
)

_interp_theta_n = RegularGridInterpolator(
    (_EPSILON_VALS, _LPCT_VALS), _THETA_N_TABLE,
    method="linear", bounds_error=True,
)
_interp_theta_e = RegularGridInterpolator(
    (_EPSILON_VALS, _LPCT_VALS), _THETA_E_TABLE,
    method="linear", bounds_error=True,
)


def lookup_angles(epsilon: float, length_pct: float) -> tuple[float, float]:
    """
    Interpolate reviewer measurements of SP-125 Figure 4-14 wall angles.

    The held chart covers epsilon 5-50 and lengths 60-100 percent.  Values are
    approximate chart readings, with a source-printed anchor at (14, 80).
    Outside-domain queries raise ValueError; no extrapolation is attributed
    to the source.  See ``raosim.rao_top_chart.EVIDENCE`` for uncertainty.

    Parameters
    ----------
    epsilon    : area expansion ratio  Ae/At
    length_pct : bell length as percent of 15° half-angle cone (60–100)

    Returns
    -------
    (theta_n, theta_e) in degrees
    """
    status = rao_chart_domain_status(epsilon, length_pct)
    if not status["in_domain"]:
        raise ValueError(
            "Rao/TOP angle lookup is outside the held SP-125 chart domain "
            f"epsilon={status['epsilon_range']}, "
            f"length_pct={status['length_pct_range']}; supply independently "
            "justified explicit angles or select a supported contour method."
        )
    pt = np.array([[epsilon, length_pct]])
    theta_n = float(_interp_theta_n(pt)[0])
    theta_e = float(_interp_theta_e(pt)[0])
    return theta_n, theta_e


def rao_chart_domain_status(epsilon: float, length_pct: float) -> dict:
    """Return source-chart membership; lookup refuses points outside it."""
    eps_min, eps_max = float(_EPSILON_VALS[0]), float(_EPSILON_VALS[-1])
    lp_min, lp_max = float(_LPCT_VALS[0]), float(_LPCT_VALS[-1])
    in_domain = (
        math.isfinite(epsilon) and math.isfinite(length_pct)
        and eps_min <= epsilon <= eps_max and lp_min <= length_pct <= lp_max
    )
    return {
        "in_domain": bool(in_domain),
        "extrapolated": not bool(in_domain),
        "epsilon_range": [eps_min, eps_max],
        "length_pct_range": [lp_min, lp_max],
    }




def _full_cone_length(
    Rt: float, epsilon: float, downstream_radius_ratio: float = 0.382,
) -> float:
    """SP-125 Eq.4-7: equivalent 15-degree cone with throat rounding.

    The reference cone has the same downstream circular radius as this nozzle,
    as required by the definition preceding Figure 4-14 (PDF p.100) and the
    worked calculation on PDF p.107.  Setting the ratio to zero recovers the
    sharp-corner convention; that is not the source's rounded-throat length.
    """
    alpha = math.radians(15.0)
    Re = math.sqrt(epsilon) * Rt
    Rd = downstream_radius_ratio * Rt
    return (Re - Rt + Rd * (1.0 / math.cos(alpha) - 1.0)) / math.tan(alpha)


def _shift_contour_x(contour: dict, offset: float) -> dict:
    """Translate public contour coordinates to an absolute throat station."""
    if offset == 0.0:
        return contour
    for key in ("x", "x_conv", "x_throat", "x_bell"):
        if key in contour:
            contour[key] = np.asarray(contour[key], dtype=float) + offset
    for key in ("N", "E", "P1"):
        if key in contour:
            point = contour[key]
            contour[key] = (float(point[0]) + offset, float(point[1]))
    return contour


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
    method: str = 'bezier',
    starting_line_method: str = 'kliegel_levine',
    gamma: float = 1.4,
    pa_over_p0: float = 0.0,
    rao_moc_n_control: int = 12,
    rao_moc_n_kernel: int = 12,
    rao_moc_max_nfev: int = 200,
    rao_moc_evaluate_moc: bool = True,
    throat_geometry: ThroatGeometrySpec | None = None,
    *,
    rao_moc_theta_n_guess_deg: float = 30.0,
    rao_moc_solver_backend: str = "jax",
    rao_moc_wall_method: str = "coupled",
    rao_moc_kernel_d_fraction_max: float | None = None,
    rao_moc_physics_weight: float | None = None,
    rao_moc_kernel_dtheta_limit_deg: float = 0.5,
    rao_moc_kernel_massflow_rel_tol: float = 0.02,
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
    method      : 'bezier' (default, Bézier approximation),
                  'moc' (MOC + Rao optimization),
                  'rao' (legacy experimental variational path), or
                  'rao_variational_moc' (auditable residual/MOC path)
    starting_line_method : MOC-only transonic data line passed to
                           raosim.moc.solve_flowfield ('kliegel_levine'
                           default, or 'nasa_visible_kliegel_levine')
    pa_over_p0 : Rao-variational-only design ambient/stagnation pressure ratio
    rao_moc_n_control, rao_moc_n_kernel, rao_moc_max_nfev,
    rao_moc_evaluate_moc, rao_moc_theta_n_guess_deg,
    rao_moc_solver_backend, rao_moc_wall_method,
    rao_moc_kernel_d_fraction_max, rao_moc_physics_weight,
    rao_moc_kernel_dtheta_limit_deg, rao_moc_kernel_massflow_rel_tol :
        Rao-variational-MOC-only residual solve controls

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
    spec = resolve_throat_geometry(
        throat_geometry,
        upstream_radius_ratio=Ru_factor,
        downstream_radius_ratio=Rd_factor,
        convergent_half_angle_deg=convergent_half_angle_deg,
    )
    convergent_half_angle_deg = spec.convergent_half_angle_deg
    Ru_factor = spec.upstream_radius_ratio
    Rd_factor = spec.downstream_radius_ratio
    throat_x = float(spec.throat_location)

    if method == 'moc':
        from raosim.rao_optimizer import moc_bell_nozzle
        contour = moc_bell_nozzle(
            Rt, epsilon, gamma=gamma, length_pct=length_pct,
            convergent_half_angle_deg=convergent_half_angle_deg,
            Ru_factor=Ru_factor, Rd_factor=Rd_factor,
            starting_line_method=starting_line_method,
        )
        _shift_contour_x(contour, throat_x)
        contour["throat_geometry"] = spec.to_dict()
        contour["throat_location"] = throat_x
        return add_contour_reliability_metadata(contour, 'moc', gamma)

    if method == 'rao':
        from raosim.rao_variational import rao_variational_contour
        contour = rao_variational_contour(
            Rt, epsilon, gamma=gamma, pa_over_p0=pa_over_p0,
            length_pct=length_pct,
            convergent_half_angle_deg=convergent_half_angle_deg,
            Ru_factor=Ru_factor, Rd_factor=Rd_factor,
        )
        _shift_contour_x(contour, throat_x)
        contour["throat_geometry"] = spec.to_dict()
        contour["throat_location"] = throat_x
        return add_contour_reliability_metadata(contour, 'rao', gamma)

    if method == 'rao_variational_moc':
        from raosim.rao_variational import rao_variational_moc_contour
        contour = rao_variational_moc_contour(
            Rt, epsilon, gamma=gamma, pa_over_p0=pa_over_p0,
            length_pct=length_pct,
            convergent_half_angle_deg=convergent_half_angle_deg,
            Ru_factor=Ru_factor,
            throat_downstream_radius_factor=Rd_factor,
            thetaN_guess_deg=rao_moc_theta_n_guess_deg,
            starting_line_method=starting_line_method,
            n_control=rao_moc_n_control,
            n_kernel=rao_moc_n_kernel,
            max_nfev=rao_moc_max_nfev,
            evaluate_moc=rao_moc_evaluate_moc,
            solver_backend=rao_moc_solver_backend,
            wall_method=rao_moc_wall_method,
            kernel_d_fraction_max=rao_moc_kernel_d_fraction_max,
            physics_weight=rao_moc_physics_weight,
            kernel_dtheta_limit_deg=rao_moc_kernel_dtheta_limit_deg,
            kernel_massflow_rel_tol=rao_moc_kernel_massflow_rel_tol,
        )
        _shift_contour_x(contour, throat_x)
        contour["throat_geometry"] = spec.to_dict()
        contour["throat_location"] = throat_x
        return add_contour_reliability_metadata(contour, 'rao_variational_moc', gamma)

    if epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")


    theta_n_from_chart = theta_n_deg is None
    theta_e_from_chart = theta_e_deg is None
    chart_status = rao_chart_domain_status(epsilon, length_pct)
    if theta_n_from_chart or theta_e_from_chart:
        tn_lookup, te_lookup = lookup_angles(epsilon, length_pct)
        if theta_n_from_chart:
            theta_n_deg = tn_lookup
        if theta_e_from_chart:
            theta_e_deg = te_lookup

    theta_n = math.radians(theta_n_deg)
    theta_e = math.radians(theta_e_deg)

    Re = math.sqrt(epsilon) * Rt
    Ru = spec.upstream_radius(Rt)
    Rd = spec.downstream_radius(Rt)
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon, Rd_factor)


    y_cu = Rt + Ru      # center y
    x_cu = throat_x

    angle_start_conv = -(math.pi / 2.0 + math.radians(convergent_half_angle_deg))
    angle_end_conv = -math.pi / 2.0

    t_conv = np.linspace(angle_start_conv, angle_end_conv, n_pts)
    x_conv = x_cu + Ru * np.cos(t_conv)
    y_conv = y_cu + Ru * np.sin(t_conv)


    y_cd = Rt + Rd
    x_cd = throat_x

    angle_start_throat = -math.pi / 2.0
    angle_end_throat = theta_n - math.pi / 2.0

    t_thr = np.linspace(angle_start_throat, angle_end_throat, n_pts)
    x_throat = x_cd + Rd * np.cos(t_thr)
    y_throat = y_cd + Rd * np.sin(t_thr)


    Nx = x_throat[-1]
    Ny = y_throat[-1]


    Ex = throat_x + Ln
    Ey = Re


    m1 = math.tan(theta_n)
    m2 = math.tan(theta_e)

    if abs(m1 - m2) < 1e-12:
        raise ValueError("θ_n ≈ θ_e → tangent lines are parallel; "
                         "cannot form a Bézier bell section")


    c1 = Ny - m1 * Nx
    c2 = Ey - m2 * Ex
    P1x = (c2 - c1) / (m1 - m2)
    P1y = m1 * P1x + c1


    t = np.linspace(0.0, 1.0, n_pts)
    omt = 1.0 - t
    x_bell = omt**2 * Nx + 2.0 * omt * t * P1x + t**2 * Ex
    y_bell = omt**2 * Ny + 2.0 * omt * t * P1y + t**2 * Ey


    x_full = np.concatenate([x_conv, x_throat[1:], x_bell[1:]])
    y_full = np.concatenate([y_conv, y_throat[1:], y_bell[1:]])

    contour = {
        'x': x_full,
        'y': y_full,
        'theta_n': theta_n_deg,
        'theta_e': theta_e_deg,
        'Ln': Ln,
        'Re': Re,
        'Rt': Rt,
        'Ru': Ru,
        'Rd': Rd,
        'throat_geometry': spec.to_dict(),
        'throat_location': throat_x,
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
        'method': 'bezier',
        'angle_source': {
            'theta_n': 'rao_top_chart' if theta_n_from_chart else 'user',
            'theta_e': 'rao_top_chart' if theta_e_from_chart else 'user',
        },
        'rao_chart_domain': chart_status,
        'rao_chart_evidence': dict(_RAO_CHART_EVIDENCE) if (
            theta_n_from_chart or theta_e_from_chart
        ) else None,
        'length_reference': 'SP-125 Eq.4-7 rounded-throat 15-degree cone',
        'rao_chart_extrapolated': bool(
            chart_status['extrapolated']
            and (theta_n_from_chart or theta_e_from_chart)
        ),
    }
    if contour['rao_chart_extrapolated']:
        contour['warnings'] = [
            "Rao/TOP angle lookup extrapolated outside the digitized chart "
            f"domain epsilon={chart_status['epsilon_range']} and "
            f"length_pct={chart_status['length_pct_range']}; this contour is "
            "diagnostic and is not benchmark-qualified for design export."
        ]
    return add_contour_reliability_metadata(contour, 'bezier', gamma)


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
