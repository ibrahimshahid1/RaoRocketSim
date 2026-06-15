"""
rao_optimizer.py – Rao constrained optimization for bell nozzle contour.

Layer 2: finds the wall contour that maximizes thrust, subject to
geometric and regularity constraints, using the coupled MOC flow solver.

Uses SplineWall + coupled march_coupled_net (no frozen kernel).
θ_n is a design variable seeded from NASA SP-8120 lookup tables.
No scipy dependency.

References:
  - Rao, G.V.R., "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA SP-8120, "Liquid Rocket Engine Nozzles," 1976

Approximation note:
  The coupled optimizer inherits the starting-line approximation selected
  in raosim.moc.solve_flowfield. The default is the historical
  quasi-1D area-ratio initialization for backward compatibility; use
  starting_line_method='sauer_modified' for the compact transonic
  correction or 'kliegel_levine' for the full third-order series.
"""

from __future__ import annotations
import math
import numpy as np
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised in fallback test
    minimize = None
    SCIPY_AVAILABLE = False

from raosim.gas_dynamics import isentropic_pressure_ratio, isentropic_density_ratio
from raosim.moc import solve_flowfield
from raosim.validation import add_contour_reliability_metadata
from raosim.wall_model import SplineWall


def _full_cone_length(Rt: float, epsilon: float) -> float:
    Re = math.sqrt(epsilon) * Rt
    return (Re - Rt) / math.tan(math.radians(15.0))


def _lookup_theta_n(epsilon: float, length_pct: float) -> float:
    """Seed θ_n from NASA SP-8120 near-optimum bell correlations."""
    if length_pct >= 90:
        if epsilon <= 5: return 24.0
        elif epsilon <= 15: return 26.0
        elif epsilon <= 30: return 28.0
        else: return 30.0
    elif length_pct >= 75:
        if epsilon <= 5: return 28.0
        elif epsilon <= 15: return 30.0
        elif epsilon <= 30: return 33.0
        else: return 35.0
    else:
        if epsilon <= 5: return 32.0
        elif epsilon <= 15: return 34.0
        elif epsilon <= 30: return 37.0
        else: return 40.0


def _nelder_mead(func, x0, max_iter=200, tol=1e-7):
    """Nelder-Mead simplex (numpy-only)."""
    n = len(x0)
    alpha, gamma_nm, rho, sigma = 1.0, 2.0, 0.5, 0.5

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        v = x0.copy()
        v[i] += 0.05 * max(abs(x0[i]), 1e-4)
        simplex[i + 1] = v

    f_vals = np.array([func(simplex[i]) for i in range(n + 1)])
    converged = False

    for _ in range(max_iter):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        if np.std(f_vals) < tol:
            converged = True
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)

        if fr < f_vals[0]:
            xe = centroid + gamma_nm * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1], f_vals[-1] = xe, fe
            else:
                simplex[-1], f_vals[-1] = xr, fr
        elif fr < f_vals[-2]:
            simplex[-1], f_vals[-1] = xr, fr
        else:
            xc = centroid + rho * ((simplex[-1] if fr >= f_vals[-1] else xr) - centroid)
            fc = func(xc)
            if fc < min(fr, f_vals[-1]):
                simplex[-1], f_vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_vals[i] = func(simplex[i])

    best = np.argmin(f_vals)
    return simplex[best], f_vals[best], converged


def optimize_wall(Rt: float, epsilon: float, gamma: float = 1.4,
                  length_pct: float = 80.0,
                  n_control: int = 5,
                  n_char: int = 30,
                  max_iter: int = 200,
                  starting_line_method: str = 'area_ratio',
                  enforce_pressure_monotonic: bool = False) -> dict:
    """
    Find thrust-optimal wall via constrained optimization.

    Decision variables: [theta_n, c1, c2, ..., c_n] where c_i are
    spline control-point radii and theta_n is the initial wall angle.
    """
    Re = math.sqrt(epsilon) * Rt
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)
    Rd = 0.382 * Rt

    theta_n_seed = math.radians(_lookup_theta_n(epsilon, length_pct))

    Ny_seed = Rt + Rd * (1.0 - math.cos(theta_n_seed))
    Nx_seed = Rd * math.sin(theta_n_seed)

    r_init = np.linspace(Ny_seed, Re, n_control + 2)[1:-1]
    x0 = np.concatenate([[theta_n_seed], r_init])

    def _unpack(params, project=False):
        theta_n = params[0]
        control_r = params[1:]
        if project:
            theta_n = np.clip(theta_n, math.radians(15), math.radians(45))
            control_r = np.clip(control_r, Rt, Re)
            control_r = np.sort(control_r)
        return theta_n, control_r

    def _build_wall(theta_n, control_r):
        Ny = Rt + Rd * (1.0 - math.cos(theta_n))
        Nx = Rd * math.sin(theta_n)
        wall = SplineWall.from_controls(control_r, Nx, Ny, Ln, Re, theta_n)
        return wall, Nx, Ny

    def objective(params):
        theta_n, control_r = _unpack(params, project=not SCIPY_AVAILABLE)

        try:
            wall, _, _ = _build_wall(theta_n, control_r)
            result = solve_flowfield(
                Rt, epsilon, gamma, wall, n_char,
                starting_line_method=starting_line_method,
            )
            metrics = result['exit_metrics']

            thrust_term = -metrics['F_dimensional']

            flow_quality_penalty = 0.0
            flow_quality_penalty += 1.0 * max(0, metrics['theta_max'] - 5.0)**2
            flow_quality_penalty += 0.5 * metrics['theta_rms']**2

            ws, wr, _ = wall.sample(50)
            dr = np.diff(wr)
            geometry_penalty = 500.0 * np.sum(np.minimum(dr, 0)**2)

            slopes = np.diff(wr) / np.diff(ws)
            curvature = np.diff(slopes)
            geometry_penalty += 5.0 * np.sum(curvature**2)

            cost = thrust_term + flow_quality_penalty + geometry_penalty

            return cost
        except Exception:
            return 1e6

    if SCIPY_AVAILABLE:
        bounds = [(math.radians(15), math.radians(45))] + [(Rt, Re)] * n_control
        ws_cons = np.linspace(0.0, 1.0, 80)

        def _radius_progression(params):
            theta_n, control_r = _unpack(params)
            wall, _, _ = _build_wall(theta_n, control_r)
            x = wall.x_start + ws_cons * (wall.x_end - wall.x_start)
            r = np.array([wall.r(xi) for xi in x])
            return np.diff(r)

        def _angle_progression(params):
            theta_n, control_r = _unpack(params)
            wall, _, _ = _build_wall(theta_n, control_r)
            x = wall.x_knots
            t = np.array([wall.theta(xi) for xi in x])
            return -np.diff(t)  # non-increasing theta downstream

        def _fixed_exit_radius(params):
            theta_n, control_r = _unpack(params)
            wall, _, _ = _build_wall(theta_n, control_r)
            return wall.r(wall.x_end) - Re

        def _fixed_length(params):
            theta_n, control_r = _unpack(params)
            wall, _, _ = _build_wall(theta_n, control_r)
            return wall.x_end - Ln

        constraints = [
            {'type': 'ineq', 'fun': lambda p: np.diff(p[1:])},
            {'type': 'ineq', 'fun': _radius_progression},
            {'type': 'ineq', 'fun': _angle_progression},
            {'type': 'eq', 'fun': _fixed_exit_radius},
            {'type': 'eq', 'fun': _fixed_length},
        ]

        if enforce_pressure_monotonic:
            def _pressure_progression(params):
                theta_n, control_r = _unpack(params)
                wall, _, _ = _build_wall(theta_n, control_r)
                result = solve_flowfield(Rt, epsilon, gamma, wall, n_char)
                wall_rows = [row.wall for row in result['rows'] if row.wall is not None]
                if len(wall_rows) < 3:
                    return np.array([0.0])
                wall_rows = sorted(wall_rows, key=lambda p: p.x)
                p = np.array([isentropic_pressure_ratio(pt.M, gamma) for pt in wall_rows])
                return p[:-1] - p[1:]

            constraints.append({'type': 'ineq', 'fun': _pressure_progression})

        def _max_constraint_violation(x):
            worst = 0.0
            for c in constraints:
                v = np.atleast_1d(c['fun'](x))
                if c['type'] == 'ineq':
                    worst = max(worst, float(np.max(np.maximum(0.0, -v))))
                else:
                    worst = max(worst, float(np.max(np.abs(v))))
            return worst

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8},
        )
        opt_x = res.x
        converged = bool(res.success)
        if not converged:
            feasible = _max_constraint_violation(opt_x) < 1e-5
            converged = feasible and np.isfinite(objective(opt_x))
        optimizer = 'scipy-SLSQP'
    else:
        opt_x, _, converged = _nelder_mead(objective, x0, max_iter=max_iter)
        optimizer = 'nelder-mead'

    theta_n_opt, control_r_opt = _unpack(opt_x, project=True)

    wall_opt, Nx_opt, Ny_opt = _build_wall(theta_n_opt, control_r_opt)
    final = solve_flowfield(
        Rt, epsilon, gamma, wall_opt, n_char,
        starting_line_method=starting_line_method,
    )

    return {
        'wall': wall_opt,
        'rows': final['rows'],
        'exit_samples': final['exit_samples'],
        'exit_metrics': final['exit_metrics'],
        'theta_n': math.degrees(theta_n_opt),
        'theta_e': math.degrees(wall_opt.theta(wall_opt.x_end)),
        'Nx': Nx_opt, 'Ny': Ny_opt,
        'Ex': Ln, 'Ey': Re,
        'converged': bool(converged),
        'control_points': control_r_opt,
        'optimizer': optimizer,
    }


def moc_bell_nozzle(Rt: float, epsilon: float, gamma: float = 1.4,
                    length_pct: float = 80.0,
                    n_control: int = 5, n_char: int = 30,
                    convergent_half_angle_deg: float = 45.0,
                    Ru_factor: float = 1.5,
                    max_iter: int = 200,
                    starting_line_method: str = 'area_ratio') -> dict:
    """
    Generate optimized bell nozzle contour via coupled MOC + optimization.

    Returns contour dict compatible with bell_nozzle_contour().
    """
    Re = math.sqrt(epsilon) * Rt
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)
    Ru = Ru_factor * Rt
    Rd = 0.382 * Rt

    opt = optimize_wall(Rt, epsilon, gamma, length_pct,
                        n_control, n_char, max_iter,
                        starting_line_method=starting_line_method)

    conv_angle = math.radians(convergent_half_angle_deg)
    n_conv = 100
    t_conv = np.linspace(-(math.pi/2 + conv_angle), -math.pi/2, n_conv)
    x_conv = Ru * np.cos(t_conv)
    y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

    theta_n_rad = math.radians(opt['theta_n'])
    t_thr = np.linspace(-math.pi/2, theta_n_rad - math.pi/2, n_conv)
    x_throat = Rd * np.cos(t_thr)
    y_throat = (Rt + Rd) + Rd * np.sin(t_thr)

    # ── Bell section: quadratic Bézier from optimised angles ─────
    # The MoC optimizer determines the thrust-optimal θ_n and θ_e.
    # Rather than sampling the sparse SplineWall (which is only C1
    # at its knots and visually kinked), we construct the divergent
    # bell as a quadratic Bézier — identical to the standard Rao
    # near-optimum approximation, but with MoC-optimised angles.
    # This is C∞ smooth, guaranteed monotonic, and G1-continuous
    # with the upstream throat arc at the inflection point N.
    Nx_opt = opt['Nx']
    Ny_opt = opt['Ny']
    theta_e_rad = math.radians(opt['theta_e'])

    # Bézier control point P1: intersection of the tangent lines
    # from N (slope = tan θ_n) and E (slope = tan θ_e).
    m1 = math.tan(theta_n_rad)
    m2 = math.tan(theta_e_rad)
    if abs(m1 - m2) > 1e-12:
        x_p1 = (Re - Ny_opt - m2 * Ln + m1 * Nx_opt) / (m1 - m2)
    else:
        x_p1 = 0.5 * (Nx_opt + Ln)
    y_p1 = Ny_opt + m1 * (x_p1 - Nx_opt)

    N_pt = np.array([Nx_opt, Ny_opt])
    E_pt = np.array([Ln, Re])
    P1_pt = np.array([x_p1, y_p1])

    t_bell = np.linspace(0.0, 1.0, n_conv)
    bell = ((1 - t_bell) ** 2 * N_pt[:, None]
            + 2 * (1 - t_bell) * t_bell * P1_pt[:, None]
            + t_bell ** 2 * E_pt[:, None])
    wall_x = bell[0]
    wall_r = bell[1]

    x_full = np.concatenate([x_conv, x_throat, wall_x])
    y_full = np.concatenate([y_conv, y_throat, wall_r])

    metrics = opt['exit_metrics']

    contour = {
        'x': x_full,
        'y': y_full,
        'theta_n': opt['theta_n'],
        'theta_e': opt['theta_e'],
        'Ln': Ln,
        'Re': Re,
        'Rt': Rt,
        'Ru': Ru,
        'Rd': Rd,
        'epsilon': epsilon,
        'length_pct': length_pct,
        'N': (Nx_opt, Ny_opt),
        'E': (Ln, Re),
        'P1': (x_p1, y_p1),
        'x_conv': x_conv,
        'y_conv': y_conv,
        'x_throat': x_throat,
        'y_throat': y_throat,
        'x_bell': wall_x,
        'y_bell': wall_r,
        'method': 'moc',
        'exit_theta_max': metrics['theta_max'],
        'exit_theta_rms': metrics['theta_rms'],
        'exit_M_uniformity': metrics['M_std'],
        'exit_M_mean': metrics['M_mean'],
        'optimization_converged': opt['converged'],
        'starting_line_method': starting_line_method,
    }
    return add_contour_reliability_metadata(contour, 'moc', gamma)
