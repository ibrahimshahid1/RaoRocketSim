"""
moc.py – Axisymmetric Method of Characteristics unit processes and the
forward flow-field analysis entry point.

Layer 1 of the Rao nozzle design.  This module owns the three Anderson-form
unit processes (interior / axis / wall point) and the transonic start-line
helpers; the bounded characteristic net itself is marched by
:func:`raosim.moc_bounded.march_bounded_net`, which
:func:`solve_flowfield` drives.

History (see ``docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md``):
the original :func:`march_coupled_net` marched zig-zag "fronts" in which
the computed axis and wall points were never parents of any later point,
so the net was only the shrinking Cauchy triangle of its starting line —
it never reached the exit and its exit plane was a two-point
interpolation.  It is retained, deprecated, for the audit trail only.

Three primitive solvers:
  1. solve_interior_point: two adjacent parents → intersection
  2. solve_axis_point:     symmetry BC (θ=0, r=0)
  3. solve_wall_point:     wall tangency BC (θ = wall angle)

Axisymmetric compatibility equations, with nodes ordered downstream:
  Along C+ (slope θ+μ): d(θ − ν) = −S ds
  Along C- (slope θ−μ): d(θ + ν) = +S ds
  S = sin(θ) sin(μ) / r
  (S=0 for planar flow)
  (Anderson MCF 3rd ed. eqs 11.52/11.53; oracle-validated 2026-06-11
  against the NASA M3.5Perf net, see tests/test_characteristic_pairing.py)

References:
  - Anderson, Modern Compressible Flow, 3rd ed., Ch. 11
  - Rice, T., NASA/JHU 3-D nozzle design code manual (20030067852)
  - NASA SP-8120, Liquid Rocket Engine Nozzles

Starting lines:
  - :func:`solve_flowfield` starts from the axis-to-wall Kliegel–Levine
    data line of :func:`raosim.nasa_moc._make_throat_initial_line`
    (default ``'kliegel_levine'``; ``'nasa_visible_kliegel_levine'`` keeps
    the NASA C++ expression for oracle parity).
  - :func:`approximate_starting_line` samples the DOWNSTREAM THROAT ARC
    (the wall T→N), not a data line.  It is kept for the transonic-kernel
    tests and the deprecated march; do not use it as MOC initial data.
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
    # Optional explicit parent map, one entry per ``all_points()`` position:
    # ``{"cplus": (row, index) | None, "cminus": (row, index) | None}`` in
    # ``all_points()`` indexing of the parent row.  Populated by
    # :meth:`raosim.moc_bounded.BoundedNet.to_char_rows`; ``None`` on rows
    # produced by the deprecated front march (whose topology is implicit).
    parents: list[dict | None] | None = None

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


class UpstreamIntersection(RuntimeError):
    """The C⁻/C⁺ intersection landed upstream of both parents (a crossing)."""


def solve_interior_point(p_minus: CharPoint, p_plus: CharPoint,
                         gamma: float, axisymmetric: bool = True,
                         tol: float = 1e-8, max_iter: int = 10,
                         on_upstream: str = "keep") -> CharPoint:
    """
    Interior unit process: C⁻ from p_minus (above) ∩ C⁺ from p_plus (below).

    Predictor-corrector with axisymmetric source terms.

    ``on_upstream`` governs a converged intersection that lies upstream of
    both parents (like-characteristic crossing): ``"keep"`` (default)
    returns the point exactly as solved so the caller can audit it (the
    NASA remaining-mesh port checks the signed cell area); ``"raise"``
    raises :class:`UpstreamIntersection`; ``"clamp"`` reproduces the
    pre-2026-09 behaviour of moving ``x3`` downstream and resetting ``r3``
    without re-solving the compatibility equations — that point satisfies
    neither the intersection nor the compatibility relations and is kept
    only for regression comparison.
    """
    if on_upstream not in {"keep", "raise", "clamp"}:
        raise ValueError("on_upstream must be 'keep', 'raise' or 'clamp'")
    # CORRECTED 2026-06-11 invariant pairing (Anderson MCF §11.4; Z&H
    # Vol. 2 Ch. 17; oracle-validated — see rao_residuals._source_axisym):
    #   along C− (slope θ−μ, from p_minus above):  d(θ+ν) = +S ds
    #   along C+ (slope θ+μ, from p_plus below):   d(θ−ν) = −S ds
    # with S = sinθ sinμ / r.  Planar predictor:
    #   K− = (θ+ν)_pminus,  K+ = (θ−ν)_pplus,
    #   θ3 = (K− + K+)/2,   ν3 = (K− − K+)/2.
    # (The pre-correction code carried θ−ν along C− and θ+ν along C+.)
    theta3 = 0.5 * (p_minus.compat_plus + p_plus.compat_minus)
    nu3 = 0.5 * (p_minus.compat_plus - p_plus.compat_minus)
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

        k_minus = p_minus.compat_plus    # θ+ν carried along C−
        k_plus = p_plus.compat_minus     # θ−ν carried along C+

        if axisymmetric and r3 > 1e-10:
            ds_m = math.sqrt((x3-p_minus.x)**2 + (r3-p_minus.r)**2)
            ds_p = math.sqrt((x3-p_plus.x)**2 + (r3-p_plus.r)**2)

            th_m = 0.5*(p_minus.theta + theta3)
            mu_m = 0.5*(p_minus.mu + mu3)
            r_m = max(0.5*(p_minus.r + r3), 1e-12)

            th_p = 0.5*(p_plus.theta + theta3)
            mu_p = 0.5*(p_plus.mu + mu3)
            r_p = max(0.5*(p_plus.r + r3), 1e-12)

            S_m = math.sin(th_m) * math.sin(mu_m) / r_m
            S_p = math.sin(th_p) * math.sin(mu_p) / r_p

            k_minus = p_minus.compat_plus + S_m * ds_m
            k_plus = p_plus.compat_minus - S_p * ds_p

        theta3 = 0.5 * (k_minus + k_plus)
        nu3 = 0.5 * (k_minus - k_plus)
        if nu3 < 1e-8:
            nu3 = 1e-8
        M3 = mach_from_prandtl_meyer(nu3, gamma)
        mu3 = mach_angle(M3)

        if abs(theta3 - theta3_old) < tol:
            break

    x_min_parent = min(p_minus.x, p_plus.x)
    if x3 < x_min_parent:
        if on_upstream == "raise":
            raise UpstreamIntersection(
                f"intersection x={x3:.6g} lies upstream of both parents "
                f"(min x={x_min_parent:.6g})")
        if on_upstream == "clamp":
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

    CORRECTED 2026-06-11 (invariant pairing): along the C⁻ from
    p_above to the axis, (θ + ν)₃ = (θ + ν)_above + S·ds with
    S = sinθ sinμ / r.  At the axis θ₃ = 0, so
    ν₃ = (θ + ν)_above + S·ds.
    """
    theta3 = 0.0
    # C⁻ invariant: at axis θ=0 → ν₃ = K− = compat_plus_above (planar)
    nu3 = p_above.compat_plus
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

        k_minus = p_above.compat_plus
        if axisymmetric and p_above.r > 1e-10:
            ds = math.sqrt((x3-p_above.x)**2 + p_above.r**2)
            th_avg = 0.5 * p_above.theta
            mu_avg = 0.5 * (p_above.mu + mu3)
            r_avg = max(0.5 * p_above.r, 1e-12)
            sin_th = th_avg if abs(th_avg) < 1e-10 else math.sin(th_avg)
            S = sin_th * math.sin(mu_avg) / r_avg
            k_minus = p_above.compat_plus + S * ds

        # At axis θ=0 → ν₃ = K−
        nu3 = k_minus
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

    CORRECTED 2026-06-11 (invariant pairing): along the C⁺ from
    p_inside to the wall, K⁺ = (θ − ν)_inside − S·ds with
    S = sinθ sinμ / r; at the wall θ = θ_w, so ν_w = θ_w − K⁺.

    Uses wall.intersect_char() to find the geometric intersection,
    then reads wall.theta(x) for the boundary condition.
    """
    mu_avg = p_inside.mu
    theta_avg = p_inside.theta
    char_slope = math.tan(theta_avg + mu_avg)

    x_hit, r_hit = wall.intersect_char(p_inside.x, p_inside.r, char_slope)

    theta_w = wall.theta(x_hit)

    k_plus = p_inside.compat_minus

    if axisymmetric and r_hit > 1e-10 and p_inside.r > 1e-10:
        ds = math.sqrt((x_hit-p_inside.x)**2 + (r_hit-p_inside.r)**2)
        th_avg = 0.5 * (p_inside.theta + theta_w)
        mu_est = p_inside.mu
        r_avg = max(0.5 * (p_inside.r + r_hit), 1e-12)
        S = math.sin(th_avg) * math.sin(mu_est) / r_avg
        k_plus = p_inside.compat_minus - S * ds

    nu_w = theta_w - k_plus
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

        k_plus = p_inside.compat_minus
        if axisymmetric and r_hit > 1e-10 and p_inside.r > 1e-10:
            ds = math.sqrt((x_hit-p_inside.x)**2 + (r_hit-p_inside.r)**2)
            th_avg = 0.5 * (p_inside.theta + theta_w)
            mu_avg = 0.5 * (p_inside.mu + mu_w)
            r_avg = max(0.5 * (p_inside.r + r_hit), 1e-12)
            S = math.sin(th_avg) * math.sin(mu_avg) / r_avg
            k_plus = p_inside.compat_minus - S * ds

        nu_w_new = theta_w - k_plus
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
                               method: str = 'kliegel_levine', *,
                               transonic_curvature_radius: float | None = None,
                               ) -> list[CharPoint]:
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
    transonic_curvature_radius : convergent-side radius used in the
                  Hall/Kliegel-Levine coefficients.  Defaults to ``Rd`` for
                  backward-compatible low-level calls; integrated nozzle
                  workflows pass the shared upstream ``Ru`` explicitly.

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
    Rc = Rd if transonic_curvature_radius is None else float(
        transonic_curvature_radius
    )
    if Rc <= 0.0:
        raise ValueError("transonic_curvature_radius must be positive")

    angles = np.linspace(1e-4, theta_n_max, n_points)
    points = []

    if method == 'sauer_modified':
        rho_c = Rc / Rt
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
                Rc_over_Rt=Rc / Rt,
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
    DEPRECATED front march — NOT a bounded characteristic net.

    Each "row" is a zig-zag front built only from the previous front's
    interior points: the axis point (from ``prev[1]``) and the wall point
    (from ``prev[-2]``) are appended but are NEVER parents of any later
    point, so boundary reflections never re-enter the interior, the front
    shrinks by one point per row and the march dies after ``len(start)``
    rows — typically at ~30 % of the nozzle length.  With the throat-arc
    "starting line" of :func:`approximate_starting_line` every interior
    point also lies outside the wall and every wall hit clamps to the arc
    end.  Confirmed 2026-09-21; see
    ``docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md``.

    Use :func:`raosim.moc_bounded.march_bounded_net` (via
    :func:`solve_flowfield`).  Kept only so historical diagnostics can be
    reproduced; emits :class:`DeprecationWarning`.
    """
    import warnings as _warnings
    _warnings.warn(
        "march_coupled_net is a shrinking front march whose axis/wall "
        "points are never parents (not a bounded MOC net); use "
        "raosim.moc_bounded.march_bounded_net / moc.solve_flowfield",
        DeprecationWarning, stacklevel=2,
    )
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
        # After the first march, prev_pts[0] is an axis boundary point.
        # It cannot also seed an interior unit process: doing so reuses
        # prev_pts[1]'s C- for both the new axis and an interior child,
        # prevents the triangular net from shrinking, and can introduce
        # a characteristic crossing.
        parent_start = 1 if prev_pts[0].r < 1e-10 else 0
        for j in range(parent_start, len(prev_pts) - 2):
            # p_minus = upper (j+1), p_plus = lower (j): correct C⁻/C⁺ pairing
            pt = solve_interior_point(prev_pts[j + 1], prev_pts[j],
                                      gamma, axisymmetric,
                                      on_upstream="clamp")
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
    Exit-plane profile by cropping every characteristic row at ``x = x_exit``.

    Each row's ``all_points()`` polyline (one right-running characteristic
    when the rows come from :func:`raosim.moc_bounded.march_bounded_net`)
    is linearly interpolated to the exit station exactly as NASA's
    ``CropNozzleToLength`` does (MOC_GridCalc_BDE.cpp 1341-1429); the
    samples are the crossings sorted by radius, with an ``r = 0`` sample
    extrapolated when the innermost crossing is off-axis.  ``n_samples``
    > 0 resamples the crossings onto a uniform radial grid (legacy
    signature); ``n_samples <= 0`` returns the raw crossings.  Rows that do
    not reach ``x_exit`` contribute nothing — no anchor points are
    fabricated.
    """
    from raosim.moc_bounded import crop_polyline_to_x

    samples: list[dict] = []
    for row in rows:
        pts = row.all_points()
        if len(pts) < 2:
            continue
        samples.extend(crop_polyline_to_x(pts, x_exit, gamma))
    samples = [s for s in samples if math.isfinite(s['M']) and s['r'] >= -1e-12]
    samples.sort(key=lambda s: s['r'])
    merged: list[dict] = []
    for smp in samples:
        if merged and abs(smp['r'] - merged[-1]['r']) < 1e-12:
            merged[-1] = smp
        else:
            merged.append(smp)
    if not merged:
        return []
    if merged[0]['r'] > 1e-12:
        if len(merged) >= 2:
            a0, b0 = merged[0], merged[1]
            dr = b0['r'] - a0['r']
            M0 = a0['M'] - (b0['M'] - a0['M']) * a0['r'] / dr if dr > 0 else a0['M']
        else:
            M0 = merged[0]['M']
        M0 = max(1.0, M0)
        merged.insert(0, {'r': 0.0, 'theta': 0.0, 'M': M0,
                          'nu': prandtl_meyer(M0, gamma)})
    if n_samples <= 0 or len(merged) < 2:
        return merged

    r_src = np.array([p['r'] for p in merged])
    theta_src = np.array([p['theta'] for p in merged])
    M_src = np.array([p['M'] for p in merged])
    nu_src = np.array([p['nu'] for p in merged])
    r_new = np.linspace(0.0, float(r_src[-1]), n_samples)
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
                        R_gas: float = 1.0,
                        throat_radius: float | None = None) -> dict:
    """
    Compute thrust from exit-plane samples using the momentum-pressure
    integral:

        F = 2π ∫₀ᴿᵉ (ρ·ux² + (p - pa)) · r · dr

    Uses chamber/stagnation assumptions with local Mach and flow angle:
      T = Tc / (1 + (γ-1)/2 M²)
      p = Pc * (T/Tc)^(γ/(γ-1))
      ρ = p / (R_gas * T)
      Vx = M * sqrt(γ R_gas T) * cos(theta)

    Returns dimensional and normalized thrust metrics.  A physical nozzle
    thrust coefficient requires ``throat_radius`` because ``Cf = F/(Pc*At)``.
    Low-level legacy calls that omit it retain an explicitly labelled exit-area
    normalization and must not present that value as a nozzle Cf.
    """
    if len(samples) < 2:
        reference_area = (
            math.pi * throat_radius**2
            if throat_radius is not None and throat_radius > 0.0 else 0.0
        )
        return {
            'F': 0.0,
            'F_dimensional': 0.0,
            'F_normalized': 0.0,
            'Cf': 0.0,
            'normalization_basis': (
                'throat_area' if throat_radius is not None
                else 'exit_area_legacy_not_nozzle_cf'
            ),
            'reference_area': reference_area,
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
        reference_area = (
            math.pi * throat_radius**2
            if throat_radius is not None and throat_radius > 0.0 else 0.0
        )
        return {
            'F': 0.0,
            'F_dimensional': 0.0,
            'F_normalized': 0.0,
            'Cf': 0.0,
            'normalization_basis': (
                'throat_area' if throat_radius is not None
                else 'exit_area_legacy_not_nozzle_cf'
            ),
            'reference_area': reference_area,
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
    exit_area = math.pi * r_exit * r_exit if r_exit > 0.0 else 0.0
    if throat_radius is not None:
        if not math.isfinite(throat_radius) or throat_radius <= 0.0:
            raise ValueError("throat_radius must be finite and positive")
        reference_area = math.pi * throat_radius * throat_radius
        normalization_basis = "throat_area"
        cf = F_dim / max(Pc * reference_area, 1e-30)
    else:
        reference_area = exit_area
        normalization_basis = "exit_area_legacy_not_nozzle_cf"
        cf = None
    F_norm = (
        F_dim / max(Pc * reference_area, 1e-30)
        if reference_area > 0.0 else 0.0
    )

    return {
        'F': F_dim,
        'F_dimensional': F_dim,
        'F_normalized': F_norm,
        'Cf': cf,
        'normalization_basis': normalization_basis,
        'reference_area': reference_area,
        'theta_max': float(np.max(np.abs(np.degrees(th_vals)))),
        'theta_rms': float(np.degrees(np.sqrt(np.mean(th_vals**2)))),
        'M_std': float(np.std(m_vals)),
        'M_mean': float(np.mean(m_vals)),
    }


_FLOWFIELD_STARTING_LINE_METHODS = (
    'kliegel_levine',
    'nasa_visible_kliegel_levine',
    'source_visible_kliegel_levine',
)


def solve_flowfield(Rt: float, epsilon: float, gamma: float,
                    wall, n_char: int = 40,
                    starting_line_method: str = 'kliegel_levine',
                    throat_upstream_radius_factor: float = 1.5,
                    throat_downstream_radius_factor: float = 0.382,
                    *,
                    dtheta_limit_deg: float = 0.5,
                    mass_tol: float = 0.02,
                    on_invalid: str = 'refine') -> dict:
    """
    Full forward MOC analysis of a given bell wall (bounded net).

    Layer 1 interface for the Layer 2 optimizer.  The wall ``wall`` is
    the bell N→E (e.g. :class:`raosim.wall_model.SplineWall`) starting at
    the end of the downstream throat arc; the arc T→N of radius
    ``Rd = throat_downstream_radius_factor·Rt`` is reconstructed here and
    marched with the NASA arc unit processes, so the kernel rows equal
    :func:`raosim.nasa_moc.build_kernel`.

    Parameters
    ----------
    Rt      : throat radius [m]
    epsilon : expansion ratio Ae/At (exit radius is taken from the wall)
    gamma   : ratio of specific heats
    wall    : bell wall with r(x), dr_dx(x), theta(x), x_start, x_end;
              ``wall.x_start`` must equal ``Rd·sin(theta_n)`` with
              ``theta_n = wall.theta(wall.x_start)``
    n_char  : points on the transonic data line (axis → throat-plane wall)
    starting_line_method : ``'kliegel_levine'`` (default, corrected KL
              series), ``'nasa_visible_kliegel_levine'`` /
              ``'source_visible_kliegel_levine'`` (NASA C++ expression,
              oracle parity).  The old ``'area_ratio'`` / ``'hall'`` /
              ``'sauer_modified'`` options are refused: none of them is an
              axis-to-wall transonic data line.
    throat_upstream_radius_factor : Ru/Rt driving the Kliegel–Levine
              expansion (the series is parameterised by the convergent-side
              curvature; SP-8120 recommends Ru/Rt ≥ 0.6, ≈ 1.0)
    throat_downstream_radius_factor : Rd/Rt of the arc T→N
    dtheta_limit_deg : NASA ``DTHETAB`` wall-turning limit per RRC
    mass_tol : per-RRC mass-flow abort (NASA 2 %)
    on_invalid : ``'refine'`` | ``'reject'`` | ``'stop'`` (see
              :func:`raosim.moc_bounded.march_bounded_net`)

    Returns
    -------
    dict with ``rows`` (legacy :class:`CharRow` view carrying the explicit
    parent map), ``net`` (:class:`raosim.moc_bounded.BoundedNet`),
    ``exit_samples``, ``exit_metrics`` (``Cf`` normalised by ``Pc·At``,
    plus ``mdot_exit_over_start``), ``starting_line``, ``complete`` and
    ``stop_reason``.  ``complete`` is False when the march stopped before
    the wall reached the exit — the exit metrics are then partial and must
    not be reported as nozzle performance.
    """
    from raosim.moc_bounded import (
        CompositeWall, crop_to_exit_plane, exit_plane_mass_flow,
        march_bounded_net,
    )
    from raosim.nasa_moc import ArcWall, _make_throat_initial_line

    if throat_upstream_radius_factor <= 0.0:
        raise ValueError("throat_upstream_radius_factor must be positive")
    if throat_downstream_radius_factor <= 0.0:
        raise ValueError("throat_downstream_radius_factor must be positive")
    if starting_line_method not in _FLOWFIELD_STARTING_LINE_METHODS:
        raise ValueError(
            "starting_line_method must be one of "
            f"{_FLOWFIELD_STARTING_LINE_METHODS!r}, got "
            f"{starting_line_method!r} — the forward net needs an "
            "axis-to-wall transonic data line"
        )
    if n_char < 3:
        raise ValueError("n_char must be at least 3")
    # Hall/Kliegel-Levine transonic theory is parameterised by the
    # convergent-side curvature (C++ CalcInitialThroatLine(rUp, ...)).
    Ru = throat_upstream_radius_factor * Rt
    Rd = throat_downstream_radius_factor * Rt
    theta_n = float(wall.theta(wall.x_start))
    if theta_n <= 0.0:
        raise ValueError("wall must start with a positive expansion angle")

    arc = ArcWall(Rt, Rd, theta_n)
    composite = CompositeWall(arc, wall)

    starting_line = _make_throat_initial_line(
        Rt, Ru, theta_n, gamma, n_char, starting_line_method,
    )
    net = march_bounded_net(
        starting_line, composite, gamma, arc=arc,
        dtheta_limit=math.radians(dtheta_limit_deg),
        mass_tol=mass_tol, on_invalid=on_invalid,
        x_exit=float(wall.x_end),
    )
    rows = net.to_char_rows()

    exit_samples = crop_to_exit_plane(net, wall.x_end)
    exit_metrics = compute_exit_thrust(
        exit_samples, gamma, throat_radius=Rt
    )
    mdot_exit = exit_plane_mass_flow(exit_samples, gamma)
    exit_metrics['mdot_exit_over_start'] = (
        mdot_exit / net.massflow[0] if net.massflow and net.massflow[0] > 0
        else float('nan')
    )
    exit_metrics['complete'] = bool(net.complete)
    _ = epsilon  # exit radius comes from the wall; kept for API stability

    return {
        'rows': rows,
        'net': net,
        'exit_samples': exit_samples,
        'exit_metrics': exit_metrics,
        'starting_line': starting_line,
        'complete': bool(net.complete),
        'stop_reason': net.stop_reason,
    }
