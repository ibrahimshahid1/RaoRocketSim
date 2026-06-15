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

Starting-line approximation limits:
  - method='area_ratio' uses a quasi-1D area-Mach estimate along the
    throat arc and tends to under-represent curvature-driven transonic
    effects near M≈1.
  - method='hall' uses a compact Hall-inspired polynomial correction for
    curved-throat transonic flow; this implementation is intentionally
    simplified and should be treated as an engineering approximation
    rather than a full Hall/Kliegel-Levine solution.
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
class FlowNode:
    """Lightweight flow-state adapter for residual checks."""

    x: float
    r: float
    M: float
    theta: float

    @property
    def mu(self) -> float:
        return mach_angle(self.M)


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

    def to_flow_node(self) -> FlowNode:
        """Return the minimal flow-state view used by Rao residual helpers."""
        return FlowNode(x=self.x, r=self.r, M=self.M, theta=self.theta)


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

        denom = slope_m - slope_p
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

    # Ensure downstream progression (guard against degenerate configurations)
    x_min_parent = min(p_minus.x, p_plus.x)
    if x3 < x_min_parent:
        x3 = x_min_parent + 1e-8 * max(abs(x_min_parent), 1e-6)
        r3 = 0.5 * (p_minus.r + p_plus.r)

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
    Axis unit process: C⁻ from p_above reaches centerline.
    Symmetry BC: θ=0, r=0. Handles sin(θ)/r singularity.

    Along the C⁻ characteristic from p_above to the axis:
        (θ − ν)₃ = (θ − ν)_above + Q⁻·ds
    At the axis θ₃ = 0, so ν₃ = −(θ − ν)_above = ν_above − θ_above.
    """
    theta3 = 0.0
    # C⁻ invariant: at axis θ=0 → ν₃ = −compat_minus_above
    nu3 = -p_above.compat_minus
    if nu3 < 1e-8:
        nu3 = 1e-8
    M3 = mach_from_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    x3 = p_above.x

    for _ in range(max_iter):
        # C⁻ slope: tan(θ_avg − μ_avg) < 0 → goes forward and downward
        slope_m = math.tan(0.5*p_above.theta - 0.5*(p_above.mu + mu3))
        if abs(slope_m) > 1e-15:
            x3 = p_above.x - p_above.r / slope_m
        else:
            x3 = p_above.x + 2.0 * p_above.r

        cm = p_above.compat_minus
        if axisymmetric and p_above.r > 1e-10:
            ds = math.sqrt((x3-p_above.x)**2 + p_above.r**2)
            th_avg = 0.5 * p_above.theta
            mu_avg = 0.5 * (p_above.mu + mu3)
            r_avg = 0.5 * p_above.r
            cos_tm = math.cos(th_avg - mu_avg)
            if abs(cos_tm) > 1e-15 and r_avg > 1e-10:
                sin_th = th_avg if abs(th_avg) < 1e-10 else math.sin(th_avg)
                Qm = -sin_th * math.sin(mu_avg)*math.cos(mu_avg) / (r_avg*cos_tm)
                cm = p_above.compat_minus + Qm * ds

        # At axis θ=0 → ν₃ = −cm
        nu3 = -cm
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


_VALID_STARTING_LINE_METHODS = (
    'kliegel_levine',
    'sauer_modified',
    'area_ratio',
    'hall',  # deprecated alias for 'sauer_modified'
)


def approximate_starting_line(Rt: float, Rd: float, theta_n_max: float,
                               gamma: float, n_points: int = 40,
                               method: str = 'kliegel_levine') -> list[CharPoint]:
    """
    Approximate transonic starting line on the downstream throat arc.

    Parameters
    ----------
    Rt          : throat radius [m]
    Rd          : downstream arc radius [m]
    theta_n_max : maximum angle [rad]
    gamma       : ratio of specific heats
    n_points    : number of starting-line points
    method      : one of ``'kliegel_levine'`` (default), ``'sauer_modified'``,
                  ``'area_ratio'``, or ``'hall'`` (deprecated alias for
                  ``'sauer_modified'``).

    Methods
    -------
    ``'kliegel_levine'`` *(default)*
        Third-order Kliegel-Levine 1969 toroidal-coordinate expansion via
        :func:`raosim.transonic_kernel.kliegel_levine`.  Valid down to
        Rc/Rt ~ 0.5 (well below this codebase's Rd/Rt = 0.382).
    ``'sauer_modified'``
        The leading-order Hall/Sauer formula ``M = 1 + a1·ξ + a2·ξ²``
        with the original two-term coefficient set.  Quick and dirty;
        only accurate for large throat curvature (Rc/Rt > 2).
    ``'hall'``
        Deprecated alias for ``'sauer_modified'`` — the previous name was
        misleading (this is leading-order Sauer, not the Hall series).
        Emits a :class:`DeprecationWarning`.
    ``'area_ratio'``
        Pure quasi-1D area-Mach inversion at each arc point.  Ignores 2D
        throat curvature entirely.

    Points are ordered from axis-side (small θ) to wall-side (θ ≈ θ_n).

    References
    ----------
    * Hall, I. M., *Transonic Flow in Two-Dimensional and Axially-
      Symmetric Nozzles*, QJMAM 15(4), 1962.
    * Kliegel, J. R. and Levine, J. N., *Transonic Flow in Small Throat
      Radius of Curvature Nozzles*, AIAA J. 7(7), 1969.
    """
    if method not in _VALID_STARTING_LINE_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_STARTING_LINE_METHODS!r}, got {method!r}"
        )
    if method == 'hall':
        import warnings as _warnings
        _warnings.warn(
            "approximate_starting_line(method='hall') is a deprecated alias "
            "for 'sauer_modified' (the previous 'hall' implementation was the "
            "leading-order Sauer reduction, not the Hall series).  Use "
            "'kliegel_levine' for full third-order accuracy or "
            "'sauer_modified' for the previous numerical behaviour.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = 'sauer_modified'

    At = math.pi * Rt * Rt
    y_center = Rt + Rd

    angles = np.linspace(1e-4, theta_n_max, n_points)
    points = []

    if method == 'sauer_modified':
        rho_c = Rd / Rt
        gp1 = gamma + 1.0
        a1 = math.sqrt(2.0 / (gp1 * rho_c))
        a2 = gp1 / (12.0 * rho_c)
    elif method == 'kliegel_levine':
        from raosim.transonic_kernel import kliegel_levine, GEOM_AXI

    for ang in angles:
        arc_angle = ang - math.pi / 2.0
        x = Rd * math.cos(arc_angle)
        r = y_center + Rd * math.sin(arc_angle)

        if method == 'kliegel_levine':
            state = kliegel_levine(
                r_over_Rt=r / Rt,
                x_over_Rt=x / Rt,
                gamma=gamma,
                Rc_over_Rt=Rd / Rt,
                geom=GEOM_AXI,
            )
            M = max(state.M, 1.0 + 1e-6)
            theta_local = state.theta if abs(state.theta) > 1e-9 else ang
        elif method == 'sauer_modified':
            xi = (r - Rt) / Rt
            M = 1.0 + a1 * xi + a2 * xi * xi
            if M < 1.0 + 1e-6:
                M = 1.0 + 1e-6
            theta_local = ang
        else:  # area_ratio
            A_local = math.pi * r * r
            ar = A_local / At
            if ar < 1.0:
                ar = 1.0 + 1e-6
            M = mach_from_area_ratio(ar, gamma, supersonic=True)
            theta_local = ang

        pt = _make_point(x, r, theta_local, M, gamma)
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
            # p_minus = upper (j+1), p_plus = lower (j): correct C⁻/C⁺ pairing
            pt = solve_interior_point(prev_pts[j + 1], prev_pts[j],
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
    Build an exit-plane profile by intersecting late-row characteristic
    segments with x = x_exit.

    For each adjacent pair in the last few converged rows, points are
    ordered with the same convention as solve_interior_point:
      - p_minus: upper (larger radius)
      - p_plus:  lower (smaller radius)
    The geometric segment between each pair is intersected with the
    exit station, and (theta, nu, M) are linearly interpolated.
    """
    _ = gamma

    valid_rows = [row for row in rows if row.axis is not None and row.wall is not None]
    if not valid_rows:
        return [{'r': 0.0, 'theta': 0.0, 'M': 1.0, 'nu': 0.0}]

    tail_rows = valid_rows[-min(len(valid_rows), 8):]
    wall_pts = [row.wall for row in valid_rows if row.wall is not None and math.isfinite(row.wall.M)]
    axis_pts = [row.axis for row in valid_rows if row.axis is not None and math.isfinite(row.axis.M)]

    if not wall_pts:
        return [{'r': 0.0, 'theta': 0.0, 'M': 1.0, 'nu': 0.0}]

    near_wall = min(wall_pts, key=lambda p: abs(p.x - x_exit))
    Re = max(1e-10, max(p.r for p in wall_pts))

    def _interp_at_x(points: list[CharPoint], fallback: CharPoint) -> CharPoint:
        if not points:
            return fallback
        pts = sorted(points, key=lambda p: p.x)
        if len(pts) == 1:
            return pts[0]
        for a, b in zip(pts[:-1], pts[1:]):
            if (a.x <= x_exit <= b.x) or (b.x <= x_exit <= a.x):
                dx = b.x - a.x
                if abs(dx) < 1e-12:
                    return a
                t = (x_exit - a.x) / dx
                return CharPoint(
                    x=x_exit,
                    r=a.r + t * (b.r - a.r),
                    theta=a.theta + t * (b.theta - a.theta),
                    M=max(1.0, a.M + t * (b.M - a.M)),
                    nu=a.nu + t * (b.nu - a.nu),
                    mu=a.mu + t * (b.mu - a.mu),
                    compat_plus=0.0,
                    compat_minus=0.0,
                )
        return min(pts, key=lambda p: abs(p.x - x_exit))

    axis_ref = _interp_at_x(
        axis_pts,
        CharPoint(
            x=x_exit, r=0.0, theta=0.0, M=max(1.0, near_wall.M * 0.8),
            nu=0.0, mu=math.pi / 2.0, compat_plus=0.0, compat_minus=0.0,
        ),
    )
    wall_ref = _interp_at_x(wall_pts, near_wall)
    wall_theta_ref = max(wall_pts, key=lambda p: p.r)

    intersections: list[dict] = []
    tol = 1e-10

    for row in tail_rows:
        pts = row.all_points()
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            p_plus = pts[i]
            p_minus = pts[i + 1]
            x0, x1 = p_plus.x, p_minus.x
            if not (min(x0, x1) - tol <= x_exit <= max(x0, x1) + tol):
                continue
            dx = x1 - x0
            if abs(dx) < 1e-14:
                continue
            t = (x_exit - x0) / dx
            if t < -tol or t > 1.0 + tol:
                continue
            r = p_plus.r + t * (p_minus.r - p_plus.r)
            if r < -tol or r > Re + tol:
                continue
            intersections.append({
                'r': min(max(r, 0.0), Re),
                'theta': p_plus.theta + t * (p_minus.theta - p_plus.theta),
                'M': max(1.0, p_plus.M + t * (p_minus.M - p_plus.M)),
                'nu': p_plus.nu + t * (p_minus.nu - p_plus.nu),
            })

    intersections.append({
        'r': 0.0,
        'theta': 0.0,
        'M': max(1.0, axis_ref.M),
        'nu': axis_ref.nu,
    })
    intersections.append({
        'r': Re,
        'theta': wall_theta_ref.theta,
        'M': max(1.0, wall_ref.M),
        'nu': wall_ref.nu,
    })

    intersections.sort(key=lambda s: s['r'])
    merged: list[dict] = []
    for s in intersections:
        if merged and abs(s['r'] - merged[-1]['r']) < 1e-8:
            merged[-1] = s
        else:
            merged.append(s)

    if len(merged) < 2:
        return merged

    if n_samples <= 0:
        return merged

    if len(merged) == 2 and n_samples >= 2:
        r_new = np.linspace(0.0, Re, n_samples)
        r_pair = [merged[0]['r'], merged[1]['r']]
        theta_pair = [merged[0]['theta'], merged[1]['theta']]
        M_pair = [merged[0]['M'], merged[1]['M']]
        nu_pair = [merged[0]['nu'], merged[1]['nu']]
        return [
            {
                'r': float(r),
                'theta': float(np.interp(r, r_pair, theta_pair)),
                'M': float(max(1.0, np.interp(r, r_pair, M_pair))),
                'nu': float(np.interp(r, r_pair, nu_pair)),
            }
            for r in r_new
        ]

    r_src = np.array([p['r'] for p in merged])
    theta_src = np.array([p['theta'] for p in merged])
    M_src = np.array([p['M'] for p in merged])
    nu_src = np.array([p['nu'] for p in merged])

    r_new = np.linspace(0.0, Re, n_samples)
    return [
        {
            'r': float(r),
            'theta': float(np.interp(r, r_src, theta_src)),
            'M': float(max(1.0, np.interp(r, r_src, M_src))),
            'nu': float(np.interp(r, r_src, nu_src)),
        }
        for r in r_new
    ]


def compute_exit_thrust(samples: list[dict], gamma: float,
                        p_ambient: float = 0.0,
                        Pc: float = 1.0, Tc: float = 1.0,
                        R_gas: float = 1.0) -> dict:
    """
    Compute thrust from exit-plane samples using the momentum-pressure
    integral:

        F = 2π ∫₀ᴿᵉ (ρ·ux² + (p - pa)) · r · dr

    Uses chamber/stagnation assumptions with local Mach and flow angle:
      T = Tc / (1 + (γ-1)/2 M²)
      p = Pc * (T/Tc)^(γ/(γ-1))
      ρ = p / (R_gas * T)
      Vx = M * sqrt(γ R_gas T) * cos(theta)

    Returns dimensional and normalized thrust metrics.
    """
    if len(samples) < 2:
        return {
            'F': 0.0,
            'F_dimensional': 0.0,
            'F_normalized': 0.0,
            'Cf': 0.0,
            'theta_max': 0.0,
            'theta_rms': 0.0,
            'M_std': 0.0,
            'M_mean': 1.0,
        }

    ordered = sorted(samples, key=lambda s: s['r'])
    r_vals = np.array([max(0.0, float(s['r'])) for s in ordered], dtype=float)
    m_vals = np.array([max(1.0, float(s['M'])) for s in ordered], dtype=float)
    th_vals = np.array([float(s['theta']) for s in ordered], dtype=float)

    # Collapse duplicate radial stations (possible from trace/interpolation artifacts).
    r_unique, inv = np.unique(r_vals, return_inverse=True)
    if r_unique.size < 2:
        return {
            'F': 0.0,
            'F_dimensional': 0.0,
            'F_normalized': 0.0,
            'Cf': 0.0,
            'theta_max': float(np.max(np.abs(np.degrees(th_vals)))) if th_vals.size else 0.0,
            'theta_rms': float(np.degrees(np.sqrt(np.mean(th_vals**2)))) if th_vals.size else 0.0,
            'M_std': float(np.std(m_vals)) if m_vals.size else 0.0,
            'M_mean': float(np.mean(m_vals)) if m_vals.size else 1.0,
        }

    m_accum = np.zeros_like(r_unique)
    th_accum = np.zeros_like(r_unique)
    counts = np.zeros_like(r_unique)
    for idx, u_idx in enumerate(inv):
        m_accum[u_idx] += m_vals[idx]
        th_accum[u_idx] += th_vals[idx]
        counts[u_idx] += 1.0
    m_vals = m_accum / np.maximum(counts, 1.0)
    th_vals = th_accum / np.maximum(counts, 1.0)
    r_vals = r_unique

    gm1 = gamma - 1.0
    t_ratio = 1.0 / (1.0 + 0.5 * gm1 * m_vals**2)
    T = Tc * t_ratio
    p = Pc * np.power(t_ratio, gamma / gm1)
    rho = p / np.maximum(R_gas * T, 1e-30)
    V = m_vals * np.sqrt(np.maximum(gamma * R_gas * T, 0.0))
    vx = V * np.cos(th_vals)

    integrand = rho * vx**2 + (p - p_ambient)
    F_dim = float(2.0 * math.pi * np.trapezoid(integrand * r_vals, r_vals))

    r_exit = float(np.max(r_vals))
    At = math.pi * r_exit * r_exit if r_exit > 0.0 else 0.0
    F_norm = F_dim / max(Pc * At, 1e-30) if At > 0.0 else 0.0

    return {
        'F': F_dim,
        'F_dimensional': F_dim,
        'F_normalized': F_norm,
        'Cf': F_norm,
        'theta_max': float(np.max(np.abs(np.degrees(th_vals)))),
        'theta_rms': float(np.degrees(np.sqrt(np.mean(th_vals**2)))),
        'M_std': float(np.std(m_vals)),
        'M_mean': float(np.mean(m_vals)),
    }


def solve_flowfield(Rt: float, epsilon: float, gamma: float,
                    wall, n_char: int = 40,
                    starting_line_method: str = 'kliegel_levine') -> dict:
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
    starting_line_method : method passed to approximate_starting_line
                           ('area_ratio' default, or 'hall')

    Returns
    -------
    dict with rows, exit_samples, exit_metrics, starting_line
    """
    Rd = 0.382 * Rt
    Re = math.sqrt(epsilon) * Rt
    theta_n = wall.theta(wall.x_start)

    starting_line = approximate_starting_line(
        Rt, Rd, theta_n, gamma, n_char, method=starting_line_method
    )
    rows = march_coupled_net(starting_line, wall, gamma, axisymmetric=True)

    exit_samples = sample_exit_plane(rows, wall.x_end, gamma)
    exit_metrics = compute_exit_thrust(exit_samples, gamma)

    return {
        'rows': rows,
        'exit_samples': exit_samples,
        'exit_metrics': exit_metrics,
        'starting_line': starting_line,
    }
