"""
moc.py – Axisymmetric Method of Characteristics flow solver.

Layer 1 of the Rao nozzle design: given a starting line and wall
geometry, march the characteristic net with coupled wall feedback.

Architecture:
  Each row contains: axis point + interior points + wall point.
  The wall point at row k becomes a parent for row k+1.
  This is the correct MOC nozzle construction: wall turning and
  characteristic interactions are built simultaneously.

Three primitive solvers:
  1. solve_interior_point: two adjacent parents → intersection
  2. solve_axis_point:     symmetry BC (θ=0, r=0)
  3. solve_wall_point:     wall tangency BC (θ = wall angle)

Axisymmetric compatibility equations (Anderson Ch. 11):
  Along C⁺: dθ + dν = Q⁺·ds
  Along C⁻: dθ − dν = Q⁻·ds
  Q⁺ =  sin(θ)·sin(μ)·cos(μ) / (r·cos(θ + μ))
  Q⁻ = −sin(θ)·sin(μ)·cos(μ) / (r·cos(θ − μ))
  (δ=1 axisymmetric, δ=0 planar)

References:
  - Anderson, Modern Compressible Flow, 3rd ed., Ch. 11
  - Zucrow & Hoffman, Gas Dynamics, Vol. 2, Ch. 16
  - NASA SP-8120, Liquid Rocket Engine Nozzles
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np

from raosim.gas_dynamics import (
    mach_from_area_ratio,
    mach_angle,
    mach_from_prandtl_meyer,
    prandtl_meyer,
)


@dataclass
class CharPoint:
    """A point in the characteristic net.

    compat_plus / compat_minus store local values of θ+ν and θ−ν.
    In axisymmetric flow these are NOT global invariants.
    """
    x: float
    r: float
    theta: float
    M: float
    nu: float
    mu: float
    compat_plus: float
    compat_minus: float


@dataclass
class CharRow:
    """One row of the characteristic net.

    axis:     point on the symmetry axis (θ=0, r=0), or None for row 0
    interior: interior points from adjacent-pair intersections
    wall:     point on the wall boundary (θ = wall angle), or None for row 0
    """
    axis: CharPoint | None
    interior: list[CharPoint] = field(default_factory=list)
    wall: CharPoint | None = None

    def all_points(self) -> list[CharPoint]:
        pts = []
        if self.axis is not None:
            pts.append(self.axis)
        pts.extend(self.interior)
        if self.wall is not None:
            pts.append(self.wall)
        return pts


def _make_point(x: float, r: float, theta: float, M: float,
                gamma: float) -> CharPoint:
    nu = prandtl_meyer(M, gamma)
    mu = mach_angle(M)
    return CharPoint(
        x=x, r=r, theta=theta, M=M,
        nu=nu, mu=mu,
        compat_plus=theta + nu,
        compat_minus=theta - nu,
    )


def solve_interior_point(p_minus: CharPoint, p_plus: CharPoint,
                         gamma: float, axisymmetric: bool = True,
                         tol: float = 1e-8, max_iter: int = 10) -> CharPoint:
    """
    Interior unit process: C⁻ from p_minus (above) ∩ C⁺ from p_plus (below).

    Predictor-corrector with axisymmetric source terms.
    """
    theta3 = 0.5 * (p_minus.compat_minus + p_plus.compat_plus)
    nu3 = 0.5 * (p_plus.compat_plus - p_minus.compat_minus)
    if nu3 < 1e-8:
        nu3 = 1e-8
    M3 = mach_from_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)

    x3 = 0.5 * (p_minus.x + p_plus.x)
    r3 = 0.5 * (p_minus.r + p_plus.r)

    for _ in range(max_iter):
        theta3_old = theta3

        slope_m = math.tan(0.5*(p_minus.theta + theta3) - 0.5*(p_minus.mu + mu3))
        slope_p = math.tan(0.5*(p_plus.theta + theta3) + 0.5*(p_plus.mu + mu3))

        denom = slope_p - slope_m
        if abs(denom) > 1e-15:
            x3 = ((p_plus.r - p_minus.r) - slope_p*p_plus.x + slope_m*p_minus.x) / denom
            r3 = p_minus.r + slope_m * (x3 - p_minus.x)
        if r3 < 0:
            r3 = 0.0

        cm = p_minus.compat_minus
        cp = p_plus.compat_plus

        if axisymmetric and r3 > 1e-10:
            ds_m = math.sqrt((x3-p_minus.x)**2 + (r3-p_minus.r)**2)
            ds_p = math.sqrt((x3-p_plus.x)**2 + (r3-p_plus.r)**2)

            th_m = 0.5*(p_minus.theta + theta3)
            mu_m = 0.5*(p_minus.mu + mu3)
            r_m = 0.5*(p_minus.r + r3)

            th_p = 0.5*(p_plus.theta + theta3)
            mu_p = 0.5*(p_plus.mu + mu3)
            r_p = 0.5*(p_plus.r + r3)

            cos_tm = math.cos(th_m - mu_m)
            cos_tp = math.cos(th_p + mu_p)

            Qm = 0.0
            if abs(cos_tm) > 1e-15 and r_m > 1e-10:
                Qm = -math.sin(th_m) * math.sin(mu_m)*math.cos(mu_m) / (r_m*cos_tm)

            Qp = 0.0
            if abs(cos_tp) > 1e-15 and r_p > 1e-10:
                Qp = math.sin(th_p) * math.sin(mu_p)*math.cos(mu_p) / (r_p*cos_tp)

            cm = p_minus.compat_minus + Qm * ds_m
            cp = p_plus.compat_plus + Qp * ds_p

        theta3 = 0.5 * (cm + cp)
        nu3 = 0.5 * (cp - cm)
        if nu3 < 1e-8:
            nu3 = 1e-8
        M3 = mach_from_prandtl_meyer(nu3, gamma)
        mu3 = mach_angle(M3)

        if abs(theta3 - theta3_old) < tol:
            break

    return CharPoint(
        x=x3, r=r3, theta=theta3, M=M3,
        nu=nu3, mu=mu3,
        compat_plus=theta3 + nu3,
        compat_minus=theta3 - nu3,
    )


def solve_axis_point(p_above: CharPoint, gamma: float,
                     axisymmetric: bool = True,
                     tol: float = 1e-8, max_iter: int = 10) -> CharPoint:
    """
    Axis unit process: C⁺ from p_above reaches centerline.
    Symmetry BC: θ=0, r=0. Handles sin(θ)/r singularity.
    """
    theta3 = 0.0
    nu3 = p_above.compat_plus
    if nu3 < 1e-8:
        nu3 = 1e-8
    M3 = mach_from_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    x3 = p_above.x

    for _ in range(max_iter):
        slope_p = math.tan(0.5*p_above.theta + 0.5*(p_above.mu + mu3))
        if abs(slope_p) > 1e-15:
            x3 = p_above.x - p_above.r / slope_p
        else:
            x3 = p_above.x + 2.0 * p_above.r

        cp = p_above.compat_plus
        if axisymmetric and p_above.r > 1e-10:
            ds = math.sqrt((x3-p_above.x)**2 + p_above.r**2)
            th_avg = 0.5 * p_above.theta
            mu_avg = 0.5 * (p_above.mu + mu3)
            r_avg = 0.5 * p_above.r
            cos_tp = math.cos(th_avg + mu_avg)
            if abs(cos_tp) > 1e-15 and r_avg > 1e-10:
                sin_th = th_avg if abs(th_avg) < 1e-10 else math.sin(th_avg)
                Qp = sin_th * math.sin(mu_avg)*math.cos(mu_avg) / (r_avg*cos_tp)
                cp = p_above.compat_plus + Qp * ds

        nu3 = cp
        if nu3 < 1e-8:
            nu3 = 1e-8
        M3 = mach_from_prandtl_meyer(nu3, gamma)
        mu3 = mach_angle(M3)

    return CharPoint(
        x=x3, r=0.0, theta=0.0, M=M3,
        nu=nu3, mu=mu3,
        compat_plus=nu3,
        compat_minus=-nu3,
    )


def solve_wall_point(p_inside: CharPoint, wall, gamma: float,
                     axisymmetric: bool = True,
                     tol: float = 1e-8, max_iter: int = 10) -> CharPoint:
    """
    Wall unit process: C⁺ from p_inside reaches the wall.
    BC: θ_flow = wall.theta(x_hit).

    Uses wall.intersect_char() to find the geometric intersection,
    then reads wall.theta(x) for the boundary condition.
    """
    mu_avg = p_inside.mu
    theta_avg = p_inside.theta
    char_slope = math.tan(theta_avg + mu_avg)

    x_hit, r_hit = wall.intersect_char(p_inside.x, p_inside.r, char_slope)

    theta_w = wall.theta(x_hit)

    cp = p_inside.compat_plus

    if axisymmetric and r_hit > 1e-10 and p_inside.r > 1e-10:
        ds = math.sqrt((x_hit-p_inside.x)**2 + (r_hit-p_inside.r)**2)
        th_avg = 0.5 * (p_inside.theta + theta_w)
        mu_est = p_inside.mu
        r_avg = 0.5 * (p_inside.r + r_hit)
        cos_tp = math.cos(th_avg + mu_est)
        if abs(cos_tp) > 1e-15 and r_avg > 1e-10:
            Qp = math.sin(th_avg)*math.sin(mu_est)*math.cos(mu_est)/(r_avg*cos_tp)
            cp = p_inside.compat_plus + Qp * ds

    nu_w = cp - theta_w
    if nu_w < 1e-8:
        nu_w = 1e-8
    M_w = mach_from_prandtl_meyer(nu_w, gamma)
    mu_w = mach_angle(M_w)

    for iteration in range(max_iter):
        mu_avg_new = 0.5 * (p_inside.mu + mu_w)
        theta_avg_new = 0.5 * (p_inside.theta + theta_w)
        char_slope = math.tan(theta_avg_new + mu_avg_new)

        x_hit, r_hit = wall.intersect_char(p_inside.x, p_inside.r, char_slope)
        theta_w = wall.theta(x_hit)

        cp = p_inside.compat_plus
        if axisymmetric and r_hit > 1e-10 and p_inside.r > 1e-10:
            ds = math.sqrt((x_hit-p_inside.x)**2 + (r_hit-p_inside.r)**2)
            th_avg = 0.5 * (p_inside.theta + theta_w)
            mu_avg = 0.5 * (p_inside.mu + mu_w)
            r_avg = 0.5 * (p_inside.r + r_hit)
            cos_tp = math.cos(th_avg + mu_avg)
            if abs(cos_tp) > 1e-15 and r_avg > 1e-10:
                Qp = math.sin(th_avg)*math.sin(mu_avg)*math.cos(mu_avg)/(r_avg*cos_tp)
                cp = p_inside.compat_plus + Qp * ds

        nu_w_new = cp - theta_w
        if nu_w_new < 1e-8:
            nu_w_new = 1e-8
        M_w_new = mach_from_prandtl_meyer(nu_w_new, gamma)
        mu_w_new = mach_angle(M_w_new)

        if abs(nu_w_new - nu_w) < tol:
            nu_w = nu_w_new
            M_w = M_w_new
            mu_w = mu_w_new
            break
        nu_w = nu_w_new
        M_w = M_w_new
        mu_w = mu_w_new

    return CharPoint(
        x=x_hit, r=r_hit, theta=theta_w, M=M_w,
        nu=nu_w, mu=mu_w,
        compat_plus=theta_w + nu_w,
        compat_minus=theta_w - nu_w,
    )


def approximate_starting_line(Rt: float, Rd: float, theta_n_max: float,
                              gamma: float, n_points: int = 40) -> list[CharPoint]:
    """
    Approximate transonic starting line on the downstream throat arc.

    Engineering approximation (not a proper transonic solution).
    For conventional bell-nozzle design ranges, published sources
    indicate modest sensitivity (NASA SP-8120).

    Points ordered from axis-side (small θ) to wall-side (θ ≈ θ_n).
    """
    At = math.pi * Rt * Rt
    y_center = Rt + Rd

    angles = np.linspace(1e-4, theta_n_max, n_points)
    points = []

    for ang in angles:
        arc_angle = ang - math.pi / 2.0
        x = Rd * math.cos(arc_angle)
        r = y_center + Rd * math.sin(arc_angle)

        A_local = math.pi * r * r
        ar = A_local / At
        if ar < 1.0:
            ar = 1.0 + 1e-6

        M = mach_from_area_ratio(ar, gamma, supersonic=True)
        pt = _make_point(x, r, ang, M, gamma)
        points.append(pt)

    return points


def march_coupled_net(starting_line: list[CharPoint], wall,
                      gamma: float, axisymmetric: bool = True,
                      max_rows: int = 500) -> list[CharRow]:
    """
    March the characteristic net with coupled wall feedback.

    Each row contains:
      - axis point (symmetry BC: θ=0, r=0)
      - interior points (adjacent-pair intersections)
      - wall point (tangency BC: θ = wall angle)

    The wall point at row k is a parent for row k+1.
    Row k+1 has one fewer interior point than row k (net shrinks).

    Terminates when the row has only axis + wall (no interior),
    or when the wall exit is reached.
    """
    if len(starting_line) < 3:
        raise ValueError("Need at least 3 starting-line points")

    row0 = CharRow(
        axis=None,
        interior=list(starting_line),
        wall=None,
    )
    rows = [row0]
    prev_pts = list(starting_line)

    for row_idx in range(1, max_rows):
        if len(prev_pts) < 3:
            break

        new_pts: list[CharPoint] = []

        if prev_pts[0].r < 1e-10:
            axis_pt = solve_axis_point(prev_pts[1], gamma, axisymmetric)
        else:
            axis_pt = solve_axis_point(prev_pts[0], gamma, axisymmetric)
        new_pts.append(axis_pt)

        interior = []
        for j in range(len(prev_pts) - 2):
            pt = solve_interior_point(prev_pts[j], prev_pts[j + 1],
                                      gamma, axisymmetric)
            interior.append(pt)
            new_pts.append(pt)

        last_parent = prev_pts[-2] if len(prev_pts) >= 2 else prev_pts[-1]
        wall_pt = solve_wall_point(last_parent, wall, gamma, axisymmetric)
        new_pts.append(wall_pt)

        row = CharRow(axis=axis_pt, interior=interior, wall=wall_pt)
        rows.append(row)

        if wall_pt.x >= wall.x_end - 1e-10:
            break

        prev_pts = new_pts

    return rows


def sample_exit_plane(rows: list[CharRow], x_exit: float,
                      gamma: float, n_samples: int = 30) -> list[dict]:
    """
    Build exit-plane flow profile from the coupled characteristic net.

    Strategy: the exit-plane profile is bounded by:
      - Axis: θ=0, M from the last axis point
      - Wall: θ=wall angle, M from the last wall point(s) near exit

    Intermediate radial stations are populated by interpolating
    between wall points sorted by proximity to x_exit.

    This avoids interior-point contamination.
    """
    wall_pts = [row.wall for row in rows if row.wall is not None and row.wall.M < 20]
    axis_pts = [row.axis for row in rows if row.axis is not None and row.axis.M < 20]

    if not wall_pts:
        return [{'r': 0.0, 'theta': 0.0, 'M': 1.0, 'nu': 0.0}]

    near_wall = sorted(wall_pts, key=lambda p: abs(p.x - x_exit))
    best_wall = near_wall[0]

    best_axis = axis_pts[-1] if axis_pts else CharPoint(
        x=x_exit, r=0.0, theta=0.0, M=best_wall.M * 0.8,
        nu=0.0, mu=math.pi/2, compat_plus=0.0, compat_minus=0.0
    )

    R_exit = best_wall.r
    samples = []

    for i in range(n_samples):
        frac = i / max(n_samples - 1, 1)
        r = frac * R_exit
        theta = frac * best_wall.theta
        M = best_axis.M + frac * (best_wall.M - best_axis.M)
        if M < 1.0:
            M = 1.0
        nu = best_axis.nu + frac * (best_wall.nu - best_axis.nu)
        samples.append({'r': r, 'theta': theta, 'M': M, 'nu': nu})

    return samples


def compute_exit_thrust(samples: list[dict], gamma: float,
                        p_ambient: float = 0.0,
                        Pc: float = 1.0, Tc: float = 1.0,
                        R_gas: float = 1.0) -> dict:
    """
    Compute thrust from exit-plane samples using the momentum-pressure
    integral:

        F = 2π ∫₀ᴿᵉ (ρ·ux² + (p - pa)) · r · dr

    Uses isentropic relations: p/Pc = (1 + (γ-1)/2·M²)^(-γ/(γ-1))

    Returns dict with F_normalized, theta_max, theta_rms, M_std.
    """
    if len(samples) < 2:
        return {'F': 0.0, 'theta_max': 0.0, 'theta_rms': 0.0, 'M_std': 0.0, 'M_mean': 1.0}

    gm1 = gamma - 1.0
    F = 0.0

    thetas = []
    machs = []

    for i in range(len(samples) - 1):
        s1, s2 = samples[i], samples[i + 1]
        dr = s2['r'] - s1['r']
        if dr <= 0:
            continue

        r_mid = 0.5 * (s1['r'] + s2['r'])
        M_mid = 0.5 * (s1['M'] + s2['M'])
        th_mid = 0.5 * (s1['theta'] + s2['theta'])

        if M_mid < 1.0:
            M_mid = 1.0

        p_ratio = (1.0 + 0.5 * gm1 * M_mid**2) ** (-gamma / gm1)
        rho_ratio = (1.0 + 0.5 * gm1 * M_mid**2) ** (-1.0 / gm1)
        T_ratio = 1.0 / (1.0 + 0.5 * gm1 * M_mid**2)

        V = M_mid * math.sqrt(gamma * T_ratio)
        ux = V * math.cos(th_mid)

        integrand = rho_ratio * ux**2 + (p_ratio - p_ambient)
        F += integrand * 2.0 * math.pi * r_mid * dr

        thetas.append(th_mid)
        machs.append(M_mid)

    thetas_arr = np.array(thetas) if thetas else np.array([0.0])
    machs_arr = np.array(machs) if machs else np.array([1.0])

    return {
        'F': F,
        'theta_max': float(np.max(np.abs(np.degrees(thetas_arr)))),
        'theta_rms': float(np.degrees(np.sqrt(np.mean(thetas_arr**2)))),
        'M_std': float(np.std(machs_arr)),
        'M_mean': float(np.mean(machs_arr)),
    }


def solve_flowfield(Rt: float, epsilon: float, gamma: float,
                    wall, n_char: int = 40) -> dict:
    """
    Full MOC forward solve with coupled wall marching.

    Layer 1 interface for the Layer 2 optimizer.

    Parameters
    ----------
    Rt      : throat radius [m]
    epsilon : expansion ratio Ae/At
    gamma   : ratio of specific heats
    wall    : SplineWall instance with r(x), theta(x), intersect_char()
    n_char  : points on initial characteristic line

    Returns
    -------
    dict with rows, exit_samples, exit_metrics, starting_line
    """
    Rd = 0.382 * Rt
    Re = math.sqrt(epsilon) * Rt
    theta_n = wall.theta(wall.x_start)

    starting_line = approximate_starting_line(Rt, Rd, theta_n, gamma, n_char)
    rows = march_coupled_net(starting_line, wall, gamma, axisymmetric=True)

    exit_samples = sample_exit_plane(rows, wall.x_end, gamma)
    exit_metrics = compute_exit_thrust(exit_samples, gamma)

    return {
        'rows': rows,
        'exit_samples': exit_samples,
        'exit_metrics': exit_metrics,
        'starting_line': starting_line,
    }
