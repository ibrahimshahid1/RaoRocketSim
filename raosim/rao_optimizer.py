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
"""

from __future__ import annotations
import math
import numpy as np

from raosim.gas_dynamics import isentropic_pressure_ratio, isentropic_density_ratio
from raosim.moc import solve_flowfield
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
                  max_iter: int = 200) -> dict:
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

    def objective(params):
        theta_n = np.clip(params[0], math.radians(15), math.radians(45))
        control_r = params[1:]
        control_r = np.clip(control_r, Rt, Re)

        Ny = Rt + Rd * (1.0 - math.cos(theta_n))
        Nx = Rd * math.sin(theta_n)

        try:
            wall = SplineWall.from_controls(
                control_r, Nx, Ny, Ln, Re, theta_n
            )
            result = solve_flowfield(Rt, epsilon, gamma, wall, n_char)
            metrics = result['exit_metrics']

            cost = -metrics['F']
            cost += 10.0 * max(0, metrics['theta_max'] - 5.0)**2
            cost += 5.0 * metrics['theta_rms']**2

            ws, wr, _ = wall.sample(50)
            dr = np.diff(wr)
            cost += 100.0 * np.sum(np.minimum(dr, 0)**2)

            return cost
        except Exception:
            return 1e6

    opt_x, opt_f, converged = _nelder_mead(objective, x0, max_iter=max_iter)

    theta_n_opt = np.clip(opt_x[0], math.radians(15), math.radians(45))
    control_r_opt = np.clip(opt_x[1:], Rt, Re)

    Ny = Rt + Rd * (1.0 - math.cos(theta_n_opt))
    Nx = Rd * math.sin(theta_n_opt)

    wall = SplineWall.from_controls(control_r_opt, Nx, Ny, Ln, Re, theta_n_opt)
    final = solve_flowfield(Rt, epsilon, gamma, wall, n_char)

    return {
        'wall': wall,
        'rows': final['rows'],
        'exit_samples': final['exit_samples'],
        'exit_metrics': final['exit_metrics'],
        'theta_n': math.degrees(theta_n_opt),
        'theta_e': math.degrees(wall.theta(wall.x_end)),
        'Nx': Nx, 'Ny': Ny,
        'Ex': Ln, 'Ey': Re,
        'converged': converged,
        'control_points': control_r_opt,
    }


def moc_bell_nozzle(Rt: float, epsilon: float, gamma: float = 1.4,
                    length_pct: float = 80.0,
                    n_control: int = 5, n_char: int = 30,
                    convergent_half_angle_deg: float = 45.0,
                    Ru_factor: float = 1.5,
                    max_iter: int = 200) -> dict:
    """
    Generate optimized bell nozzle contour via coupled MOC + optimization.

    Returns contour dict compatible with bell_nozzle_contour().
    """
    Re = math.sqrt(epsilon) * Rt
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)
    Ru = Ru_factor * Rt
    Rd = 0.382 * Rt

    opt = optimize_wall(Rt, epsilon, gamma, length_pct,
                        n_control, n_char, max_iter)

    conv_angle = math.radians(convergent_half_angle_deg)
    n_conv = 100
    t_conv = np.linspace(-(math.pi/2 + conv_angle), -math.pi/2, n_conv)
    x_conv = Ru * np.cos(t_conv)
    y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

    theta_n_rad = math.radians(opt['theta_n'])
    t_thr = np.linspace(-math.pi/2, theta_n_rad - math.pi/2, n_conv)
    x_throat = Rd * np.cos(t_thr)
    y_throat = (Rt + Rd) + Rd * np.sin(t_thr)

    wall_x, wall_r, _ = opt['wall'].sample(100)

    x_full = np.concatenate([x_conv, x_throat, wall_x])
    y_full = np.concatenate([y_conv, y_throat, wall_r])

    metrics = opt['exit_metrics']

    return {
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
        'N': (opt['Nx'], opt['Ny']),
        'E': (opt['Ex'], opt['Ey']),
        'P1': (0.5*(opt['Nx']+opt['Ex']), 0.5*(opt['Ny']+opt['Ey'])),
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
    }
