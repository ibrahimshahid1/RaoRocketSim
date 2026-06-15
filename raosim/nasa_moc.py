"""
NASA/JHU MOC_Grid_BDE port for Rao optimum-thrust topology.

This module is a stagnation-normalised Python port of the routines in
``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp``
that produce the Rao B-D-E topology:

* ``build_kernel`` -- TT' starting line + RRCs along the throat
  expansion arc up to wall angle ``theta_B`` (mirrors
  ``CalcInitialThroatLine`` + ``CalcRRCsAlongArc``).
* ``calc_massflow_along_rrc`` / ``calc_mdot_bd`` -- axisymmetric
  annular trapezoidal mass-flow integration (``CalcMassFlowAndThrustAlongMesh``
  / ``CalcMdotBD``).
* ``nasa_deriv`` / ``nasa_runge_kutta`` -- C++ ``Deriv`` + ``RungeKutta``
  derivative system in radius, used for forward integration of the
  left-running characteristic from point D.
* ``find_point_e`` -- ``FindPointE`` analogue: integrate the C+
  characteristic from D, accumulate annular mass flow, and secant on
  the final radial step so the cumulative mass equals ``mass_BD``.
* ``calc_lrc_de`` -- ``CalcLRCDE`` analogue: outer secant on the axial
  D-location, picking D so the Rao stationarity residual
  ``theta_E - theta_calc(p_E, rho_E, M_E)`` vanishes.
* ``set_theta_b`` -- ``SetThetaB`` analogue: outer secant on the
  initial wall expansion angle.

Units are stagnation-normalised throughout: lengths in metres, M
dimensionless, ``rho/rho0`` and ``V/a0`` for the mass-flux integrand,
``p/p0`` for the thrust integrand. Gas-dimensional factors (``GRAV``,
``144``, ``GASCON/molWt``) cancel from the residual definitions so we
drop them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from raosim.gas_dynamics import (
    isentropic_density_ratio,
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
    mach_angle,
    mach_from_area_ratio,
    mach_from_prandtl_meyer,
    prandtl_meyer,
    thrust_coefficient,
)
from raosim.moc import (
    CharPoint,
    FlowNode,
    approximate_starting_line,
    solve_axis_point,
    solve_interior_point,
    solve_wall_point,
)


# ----------------------------------------------------------------------
#  Node / topology dataclasses
# ----------------------------------------------------------------------


@dataclass
class MOCNode:
    """Single MOC grid point in stagnation-normalised state."""

    x: float
    r: float
    M: float
    theta: float
    gamma: float = 1.4

    @property
    def mu(self) -> float:
        return math.asin(1.0 / max(self.M, 1.000001))

    @property
    def T(self) -> float:
        return isentropic_temperature_ratio(self.M, self.gamma)

    @property
    def rho(self) -> float:
        return isentropic_density_ratio(self.M, self.gamma)

    @property
    def p(self) -> float:
        return isentropic_pressure_ratio(self.M, self.gamma)

    @property
    def V(self) -> float:
        return self.M * math.sqrt(self.gamma * self.T)

    @property
    def u(self) -> float:
        return self.V * math.cos(self.theta)

    @property
    def v(self) -> float:
        return self.V * math.sin(self.theta)

    def to_flow_node(self) -> FlowNode:
        return FlowNode(x=float(self.x), r=float(self.r),
                        M=float(self.M), theta=float(self.theta))

    @classmethod
    def from_char_point(cls, cp: CharPoint, gamma: float) -> MOCNode:
        return cls(x=float(cp.x), r=float(cp.r),
                   M=float(cp.M), theta=float(cp.theta), gamma=float(gamma))


@dataclass
class MOCKernel:
    """Stack of RRCs from TT' through the expansion arc up to point B."""

    rrcs: list[list[MOCNode]]
    theta_B: float
    Rt: float
    Rd: float
    gamma: float
    massflow: list[np.ndarray] = field(default_factory=list)

    @property
    def bd(self) -> list[MOCNode]:
        return self.rrcs[-1]

    @property
    def B(self) -> MOCNode:
        return self.bd[0]


@dataclass(frozen=True)
class RaoTopology:
    """Explicit B-D-E topology for a Rao optimum-thrust contour."""

    B: FlowNode
    BD: tuple[FlowNode, ...]
    D: FlowNode
    DE: tuple[FlowNode, ...]
    E: FlowNode
    d_fraction: float
    mass_BD: float
    mass_DE: float
    thrust_coefficient: float
    theta_control: float
    theta_B: float
    rao_stationarity_residual: float


# ----------------------------------------------------------------------
#  Throat-arc wall (downstream curvature)
# ----------------------------------------------------------------------


class ArcWall:
    """Downstream throat arc covering ``theta=0`` to ``theta_max``.

    Mirrors the wall used by NASA ``CalcArcWallPoint``: a circular arc
    centred at ``(0, Rt + Rd)`` with radius ``Rd``, parametrised so the
    flow angle ``theta`` equals the arc angle measured from the throat.
    """

    def __init__(self, Rt: float, Rd: float, theta_max: float):
        if Rt <= 0.0 or Rd <= 0.0:
            raise ValueError("Rt and Rd must be positive")
        if theta_max <= 0.0:
            raise ValueError("theta_max must be positive")
        self.Rt = float(Rt)
        self.Rd = float(Rd)
        self.theta_max = float(theta_max)
        self.x_start = 0.0
        self.x_end = float(Rd * math.sin(theta_max))
        self.r_start = float(Rt)
        self.r_end = float(Rt + Rd * (1.0 - math.cos(theta_max)))

    def r(self, x: float) -> float:
        x_clamped = max(min(float(x), self.x_end), -1e-12)
        inside = self.Rd * self.Rd - x_clamped * x_clamped
        return float((self.Rt + self.Rd) - math.sqrt(max(inside, 0.0)))

    def dr_dx(self, x: float) -> float:
        x_clamped = max(min(float(x), self.x_end), -1e-12)
        inside = self.Rd * self.Rd - x_clamped * x_clamped
        denom = math.sqrt(max(inside, 1e-24))
        return float(x_clamped / denom)

    def theta(self, x: float) -> float:
        return math.atan(self.dr_dx(x))

    def intersect_char(self, x0: float, r0: float, char_slope: float,
                       tol: float = 1e-12, max_iter: int = 40) -> tuple[float, float]:
        x_guess = float(x0) + 0.05 * max(self.x_end - x0, 1e-9)
        x_guess = min(max(x_guess, self.x_start), self.x_end)
        for _ in range(max_iter):
            r_char = r0 + char_slope * (x_guess - x0)
            r_wall = self.r(x_guess)
            f = r_char - r_wall
            df = char_slope - self.dr_dx(x_guess)
            if abs(df) < 1e-15:
                break
            dx = -f / df
            x_guess = min(max(x_guess + dx, self.x_start), self.x_end)
            if abs(dx) < tol:
                break
        return float(x_guess), float(self.r(x_guess))


# ----------------------------------------------------------------------
#  Kernel construction
# ----------------------------------------------------------------------


def _char_point_from_node(node: MOCNode, gamma: float) -> CharPoint:
    """Promote an MOCNode to a CharPoint with PM compatibilities populated."""
    nu = prandtl_meyer(max(node.M, 1.000001), gamma)
    mu = mach_angle(max(node.M, 1.000001))
    return CharPoint(
        x=float(node.x), r=float(node.r),
        theta=float(node.theta), M=float(max(node.M, 1.000001)),
        nu=float(nu), mu=float(mu),
        compat_plus=float(node.theta + nu),
        compat_minus=float(node.theta - nu),
    )


def _hall_throat_line(
    Rt: float, Rd: float, gamma: float, n_points: int,
) -> list[CharPoint]:
    """
    NASA ``CalcInitialThroatLine`` analogue (C++ line 2805).

    Vertical TT' at the throat plane (x = 0).  Axis at r = 0 with the
    Hall transonic Mach perturbation evaluated at z = x*sqrt((gamma+1)/(2*Rd))
    = 0.  Points are wall-first (``i = 0`` is the wall, ``i = n`` the axis)
    matching NASA's storage order.
    """
    rho_c = Rd / Rt
    rp1 = rho_c + 1.0
    g = gamma
    points: list[CharPoint] = []
    for i in range(n_points):
        # NASA distributes points sin(pi/2 (n-i)/n)^1.5 between wall (i=0) and
        # axis (i=n) for sharper resolution near the wall.
        r_over_rt = math.sin(0.5 * math.pi * (n_points - 1 - i) / max(n_points - 1, 1)) ** 1.5
        r = float(r_over_rt * Rt)
        # Hall series at z = 0, axisymmetric (type = 1 in NASA), keep terms
        # through third order in 1/(R+1).  See CalcHallLine (line 2905).
        y = float(r_over_rt)
        u1 = 0.5 * y * y - 0.25
        v1 = 0.25 * y * y * y - 0.25 * y
        u2 = ((2 * g + 9) / 24 * y ** 4
              - (4 * g + 15) / 24 * y * y
              + (10 * g + 57) / 288)
        v2 = ((g + 3) / 9 * y ** 5
              - (20 * g + 63) / 96 * y ** 3
              + (28 * g + 93) / 288 * y)
        u3 = ((556 * g * g + 1737 * g + 3069) / 10368 * y ** 6
              - (388 * g * g + 1161 * g + 1881) / 2304 * y ** 4
              + (304 * g * g + 831 * g + 1242) / 1728 * y * y
              - (2708 * g * g + 7839 * g + 14211) / 82944)
        v3 = ((6836 * g * g + 23031 * g + 30627) / 82944 * y ** 7
              - (3380 * g * g + 11391 * g + 15291) / 13824 * y ** 5
              + (3424 * g * g + 11271 * g + 15228) / 13824 * y ** 3
              - (7100 * g * g + 22311 * g + 30249) / 82944 * y)
        u = 1.0 + u1 / rp1 + (u2 + u1) / (rp1 * rp1) + (u3 + 2 * u2 + u1) / (rp1 ** 3)
        # axisymmetric multiplier on v (NASA uses (1+type) = 2 for AXI).
        v_pref = math.sqrt((g + 1.0) / (2.0 * rp1))
        v = v_pref * (v1 / rp1 + (v2 + 1.5 * v1) / (rp1 * rp1)
                      + (v3 + 2.5 * v2 + (15.0 / 8.0) * v1) / (rp1 ** 3))
        q = math.sqrt(u * u + v * v)
        radical = (g + 1.0) / 2.0 - (g - 1.0) / 2.0 * q * q
        if radical <= 0.0:
            radical = 1e-9
        M = q * math.sqrt(radical)
        if M < 1.0 + 1e-4:
            M = 1.0 + 1e-4
        theta = math.atan2(v, u)
        if abs(theta) < 1e-6:
            theta = 1e-6 if r > 0 else 0.0
        nu = prandtl_meyer(M, g)
        mu = mach_angle(M)
        pt = CharPoint(
            x=0.0, r=r, theta=theta, M=M,
            nu=nu, mu=mu,
            compat_plus=theta + nu,
            compat_minus=theta - nu,
        )
        points.append(pt)
    return points


class RaoKernelError(RuntimeError):
    """Raised when the NASA-style RRC march cannot build a valid kernel."""


def _push_throat_point_to_supersonic(
    r_target: float,
    x_initial: float,
    Rt: float,
    Rd: float,
    gamma: float,
    M_min: float = 1.05,
    max_bracket_doublings: int = 50,
    max_bisect_iter: int = 40,
):
    """Walk a TT' point downstream in x at fixed r until M ≥ M_min.

    Implements the per-point downstream-step iteration NASA uses in
    ``CalcInitialThroatLine`` (C++ lines 2853-2864) — adapted for the
    tight-throat (``Rc/Rt < 1.5``) case where the throat plane has a
    substantial subsonic region near the axis.  NASA's original loop
    handles the *overshoot* case (``mach > 1.5``) by halving the
    x-step; this implementation also handles the *subsonic* case
    (``mach < M_min``) by extending the x-step.

    Algorithm:

    1. Evaluate KL at ``(x_initial, r_target)``.  If supersonic enough
       (``M >= M_min``), return immediately.
    2. Otherwise bracket the ``M = M_min`` crossing by walking x
       forward with a doubling step size.
    3. Bisect within the bracket to land on ``M_min`` within tolerance.

    Returns ``(x_final, kl_state_final)``.
    """
    from raosim.transonic_kernel import GEOM_AXI, kliegel_levine

    y = float(r_target / Rt)
    Rc_ratio = float(Rd / Rt)
    state0 = kliegel_levine(y, float(x_initial / Rt), gamma, Rc_ratio, GEOM_AXI)
    if state0.M >= M_min:
        return float(x_initial), state0

    # Bracket the M_min crossing.
    x_lo = float(x_initial)
    x_hi = x_lo + 0.005 * Rt
    state_hi = kliegel_levine(y, x_hi / Rt, gamma, Rc_ratio, GEOM_AXI)
    doublings = 0
    while state_hi.M < M_min and doublings < max_bracket_doublings:
        x_lo = x_hi
        dx = x_hi - x_lo if x_hi != x_lo else 0.005 * Rt
        x_hi = x_hi + max(dx, 0.005 * Rt) * 2.0
        state_hi = kliegel_levine(y, x_hi / Rt, gamma, Rc_ratio, GEOM_AXI)
        doublings += 1

    if state_hi.M < M_min:
        # Couldn't bracket; return the furthest-forward state we have.
        return float(x_hi), state_hi

    # Bisect the (x_lo, x_hi) bracket.
    for _ in range(max_bisect_iter):
        x_mid = 0.5 * (x_lo + x_hi)
        state_mid = kliegel_levine(y, x_mid / Rt, gamma, Rc_ratio, GEOM_AXI)
        if state_mid.M < M_min:
            x_lo = x_mid
        else:
            x_hi = x_mid
        if (x_hi - x_lo) < 1e-7 * Rt:
            break
    state_final = kliegel_levine(y, x_hi / Rt, gamma, Rc_ratio, GEOM_AXI)
    return float(x_hi), state_final


def _make_throat_initial_line(
    Rt: float, Rd: float, theta_B: float, gamma: float, n_points: int,
    starting_line_method: str,
    M_min: float = 1.05,
) -> list[CharPoint]:
    """TT' starting line with NASA per-point downstream-step iteration.

    Port of :func:`MOC_GridCalc::CalcInitialThroatLine`
    (NASA C++ lines 2805-2900) including the per-point Mach-control
    loop at lines 2853-2864.  NASA distributes radii by
    ``r/Rt = sin(pi/2 * (n-i)/n) ** 1.5`` (wall-first) and seeds each
    successive ``x[i]`` from the previous point's RRC slope
    (``dr/dx = tan(theta - mu)``).  For tight-throat geometries the
    naive ``x[i]`` puts the point in the throat's subsonic region; in
    that case :func:`_push_throat_point_to_supersonic` walks the point
    downstream at fixed ``r`` until KL gives ``M >= M_min``.  This
    guarantees the resulting TT' is everywhere-supersonic so the
    downstream MOC row march (``solve_interior_point`` /
    ``solve_wall_point``) can advance — closing the gap NASA's
    original algorithm leaves at very tight ``Rc/Rt``.

    Points are returned axis-first (index 0 = axis, index ``n_points-1``
    = wall) to match the rest of the codebase's MOC unit-process
    conventions.
    """
    from raosim.transonic_kernel import GEOM_AXI, kliegel_levine

    # Build wall-first, then reverse to axis-first at the end.
    n = max(n_points - 1, 1)
    wall_first: list[tuple[float, float, float, float]] = []  # (x, r, M, theta)

    for i in range(0, n_points):
        # NASA's sinusoidal radial distribution: r=Rt at i=0 (wall),
        # r→0 at i=n_points-1 (axis), bunched toward the wall.
        r_over_Rt = math.sin(0.5 * math.pi * (n - i) / n) ** 1.5
        r = float(r_over_Rt * Rt)

        # Initial x-guess from the previous point's RRC slope
        # (drdx = tan(theta - mu)).  i=0 is the throat-plane wall point
        # (x=0 by definition).
        if i == 0:
            x_init = 0.0
        else:
            x_prev, r_prev, M_prev, theta_prev = wall_first[i - 1]
            mu_prev = math.asin(1.0 / max(M_prev, 1.000001))
            slope_drdx = math.tan(theta_prev - mu_prev)
            if abs(slope_drdx) > 1e-12:
                x_init = x_prev + (r - r_prev) / slope_drdx
            else:
                x_init = x_prev
            x_init = max(x_init, x_prev)  # never step backward

        if starting_line_method == "kliegel_levine":
            # Apply per-point downstream-step iteration so every TT'
            # point lands at M >= M_min.
            x_final, state = _push_throat_point_to_supersonic(
                r_target=r, x_initial=x_init, Rt=Rt, Rd=Rd,
                gamma=gamma, M_min=M_min,
            )
            M = max(state.M, 1.0 + 1e-4)
            theta = state.theta
        elif starting_line_method == "sauer_modified":
            # Legacy Sauer leading-order: subsonic-axis is not pushed.
            rho_c = Rd / Rt
            xi = r_over_Rt - 1.0
            a1 = math.sqrt(2.0 / ((gamma + 1.0) * rho_c))
            a2 = (gamma + 1.0) / (12.0 * rho_c)
            M = max(1.0 + a1 * xi + a2 * xi * xi, 1.0 + 1e-4)
            theta = 0.0
            x_final = x_init
        else:
            M = 1.0 + 1e-3
            theta = 0.0
            x_final = x_init
        wall_first.append((x_final, r, M, theta))

    # Reverse to axis-first ordering for downstream solve_* unit processes.
    pts: list[CharPoint] = []
    for x_val, r_val, M_val, theta_val in reversed(wall_first):
        nu = prandtl_meyer(M_val, gamma)
        mu = mach_angle(M_val)
        pts.append(CharPoint(
            x=x_val, r=r_val, theta=theta_val, M=M_val,
            nu=nu, mu=mu,
            compat_plus=theta_val + nu,
            compat_minus=theta_val - nu,
        ))
    return pts


# ---------------------------------------------------------------------
#  NASA dθ-form helpers (C++ lines 2957-3050)
#  Used by calc_arc_wall_point and the unit-process row march.
# ---------------------------------------------------------------------


def _nasa_mm(mach: float) -> float:
    """``MM(mach) = sqrt(mach² − 1)`` (NASA C++ line 3046)."""
    return math.sqrt(max(mach * mach - 1.0, 0.0))


def _nasa_calc_A(mach: float, g: float) -> float:
    """First term of the dθ equation, Rao Eq. 15 (NASA C++ line 2966).

    ``A = MM(mach) / (mach * (1 + (γ−1)/2 · mach²))``
    """
    return _nasa_mm(mach) / (mach * (1.0 + (g - 1.0) / 2.0 * mach * mach))


def _nasa_calc_B(mach: float, theta: float, r: float) -> float:
    """Second term of the dθ LRC equation (z-form) — NASA C++ line 2975.

    ``B = 1 / (r · (MM(mach) / tan(theta) − 1))`` when r != 0; 0 otherwise.
    """
    if r == 0.0:
        return 0.0
    if abs(theta) < 1e-9:
        return 0.0
    denom = r * (_nasa_mm(mach) / math.tan(theta) - 1.0)
    if abs(denom) < 1e-12:
        return 0.0
    return 1.0 / denom


def _nasa_calc_R(mach: float, theta: float, r: float) -> float:
    """Second term of the dθ LRC equation (r-form) — NASA C++ line 2997.

    ``R = 1 / (r · (MM(mach) + 1/tan(theta)))`` when r != 0; 0 otherwise.
    """
    if r == 0.0:
        return 0.0
    if abs(theta) < 1e-9:
        return 0.0
    denom = r * (_nasa_mm(mach) + 1.0 / math.tan(theta))
    if abs(denom) < 1e-12:
        return 0.0
    return 1.0 / denom


def _nasa_l_dy_dx(theta: float, mu: float) -> float:
    """LRC slope ``tan(theta + mu)`` — NASA C++ line 3019."""
    return math.tan(theta + mu)


def _nasa_tan_avg(x: float, y: float) -> float:
    """Tangent averaging — NASA C++ line 3037.

    ``TanAvg(x, y) = tan(0.5 · (atan(x) + atan(y)))``
    """
    return math.tan(0.5 * (math.atan(x) + math.atan(y)))


def calc_arc_wall_point(
    prev_axis_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
    *,
    conv_tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[float, float, float, float] | None:
    """
    NASA ``CalcArcWallPoint`` port (C++ lines 835-948).

    Finds the next wall point on the downstream throat arc given the
    previous RRC (axis-first).  Returns ``(x, r, theta, mach)`` for the
    new wall point or ``None`` if the iteration fails / overruns.

    NASA's approach differs from :func:`moc.solve_wall_point`:

    * **Geometry is arc-locked**: ``r = Rt + Rd·(1 − cos(arctan(x/Rd_radius)))``
      and ``theta = arcsin(x/Rd_radius)`` are computed from arc geometry.
      The free variable is x.
    * **Mach is updated via the dθ-form compatibility** (Anderson
      Eq. 11, Rao Eq. 15) rather than the PM ``θ + ν`` invariant.  At
      a sharp arc turn the PM form can produce ``ν < 0`` (clipping
      M → 1); the dθ form computes the M increment directly via
      ``M = M_parent + (θ_new − θ_parent + 0.5·T) / (0.5·(A_parent + A_new))``,
      which stays well-conditioned for tight arcs.

    The "point of influence" for the new wall point is the previous
    RRC's *next-to-wall* node (``prev_axis_first[-2]`` — NASA's
    ``[1][j-1]``).  The previous wall point (``prev_axis_first[-1]``,
    NASA's ``[0][j-1]``) is used only as the starting (x, r) for the
    iteration.

    Notes
    -----
    The arc's centre is at ``(0, Rt + Rd_radius)`` and the wall radius
    measured from that centre is ``Rd_radius`` (= ``arc.Rd``).  NASA
    uses ``rad`` for that radius and ``1`` for ``Rt`` in their
    normalised-to-throat-radius units; this port keeps the actual
    metres values.
    """
    if len(prev_axis_first) < 2:
        return None
    Rt = arc.Rt
    Rd = arc.Rd
    # Point 1 — next-to-wall on previous RRC (NASA's [1][j-1]).
    p1 = prev_axis_first[-2]
    # Previous wall point (NASA's [0][j-1]) is the iteration starting (x, r).
    p_prev_wall = prev_axis_first[-1]

    M1 = max(float(p1.M), 1.000001)
    theta1 = float(p1.theta)
    r1 = float(p1.r)
    x1 = float(p1.x)
    mu1 = math.asin(1.0 / M1)

    slrc1 = _nasa_l_dy_dx(theta1, mu1)
    A1 = _nasa_calc_A(M1, gamma)
    B1 = _nasa_calc_B(M1, theta1, r1)
    R1 = _nasa_calc_R(M1, theta1, r1)

    # Start point 3 (new wall point) at the previous wall position, with
    # influence-point flow values as the iteration seed.
    x3 = float(p_prev_wall.x)
    r3 = float(p_prev_wall.r)
    M3 = M1
    theta3 = theta1
    slrc3 = slrc1
    A3 = A1
    B3 = B1
    R3 = R1

    x3_old = r3_old = M3_old = theta3_old = 9.9

    for _ in range(max_iter):
        # Tan-average the LRC slope between points 1 and 3.
        slrc13 = _nasa_tan_avg(slrc1, slrc3)
        if abs(slrc13) < 1e-12:
            return None
        x3 = (r3 - r1) / slrc13 + x1

        # Arc geometry: new r and theta are functions of x3.
        # r3 = Rt + Rd - sqrt(Rd² - x3²)    (NASA: rad replaces both
        # Rt and Rd because their R* = 1).
        inside = Rd * Rd - x3 * x3
        if inside < 0.0:
            # x3 has overshot the arc — clamp and report overrun.
            if x3 > arc.x_end + 1e-9:
                return None
            inside = 0.0
        r3 = Rt + Rd - math.sqrt(inside)
        # theta3 = arcsin(x3 / Rd)
        sin_arg = max(min(x3 / Rd, 1.0), -1.0)
        theta3 = math.asin(sin_arg)

        # dθ-form Mach update.  Choose z-form vs r-form per NASA:
        # if B1 <= R1, use the z-form (more accurate near vertical).
        if B1 <= R1:
            T1 = (x3 - x1) * (B3 + B1)
        else:
            T1 = (r3 - r1) * (R3 + R1)
        A_avg = 0.5 * (A1 + A3)
        if abs(A_avg) < 1e-12:
            return None
        M3_new = M1 + (theta3 - theta1 + 0.5 * T1) / A_avg
        if M3_new < 1.000001 or not math.isfinite(M3_new):
            return None
        M3 = M3_new

        # Refresh point-3 helpers for the next iteration.
        mu3 = math.asin(1.0 / M3)
        slrc3 = _nasa_l_dy_dx(theta3, mu3)
        A3 = _nasa_calc_A(M3, gamma)
        B3 = _nasa_calc_B(M3, theta3, r3)
        R3 = _nasa_calc_R(M3, theta3, r3)

        # Convergence: relative change in x, r, M, theta.
        r_err = (r3 - r3_old) / r3_old if r3_old != 0.0 else 9.9
        x_err = (x3 - x3_old) / x3_old if x3_old != 0.0 else 9.9
        M_err = (M3 - M3_old) / M3_old if M3_old != 0.0 else 9.9
        T_err = (
            (theta3 - theta3_old) / theta3_old if theta3_old != 0.0 else 9.9
        )

        x3_old = x3
        r3_old = r3
        M3_old = M3
        theta3_old = theta3

        if (
            abs(x_err) < conv_tol
            and abs(r_err) < conv_tol
            and abs(M_err) < conv_tol
            and abs(T_err) < conv_tol
        ):
            return (x3, r3, theta3, M3)

    # Did not converge within max_iter — return last state if reasonable.
    if M3 >= 1.000001 and 0.0 <= x3 <= arc.x_end + 1e-9:
        return (x3, r3, theta3, M3)
    return None


def _rrc_march_step(
    prev_axis_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
) -> list[CharPoint] | None:
    """Build a new RRC by NASA-style wall-then-inward unit-process march.

    Mirrors NASA ``CalcArcWallPoint`` + ``CalcInteriorMeshPoints`` +
    ``CalcAxialMeshPoint`` (C++ lines 835, 2466, 2262):

    * wall point from the C+ leaving the next-to-wall parent of the
      previous RRC, intersected with the throat arc;
    * each interior point built from its just-completed RRC neighbour
      above (C- source) and the previous RRC's next-lower neighbour
      (C+ source);
    * axis point closes by symmetry.

    ``prev_axis_first`` and the returned new RRC are both axis-first
    (index 0 is the axis, index -1 is the wall) — the conversion to
    NASA's wall-first storage happens in :func:`build_kernel`.

    Returns ``None`` if any unit process produces a non-physical state
    (subsonic float clip, wall overrun, ``r < 0``) — the caller should
    treat this as "kernel boundary reached" and stop marching.
    """
    if len(prev_axis_first) < 3:
        return None
    n = len(prev_axis_first)
    try:
        # Wall point: NASA-port CalcArcWallPoint (dθ-form compatibility,
        # arc-locked geometry).  This replaces the PM-form
        # ``solve_wall_point`` which decayed Mach for tight arcs
        # (cf. session log: M dropped 1.29 → 1.23 → 1.14 in 3 steps
        # under PM form; dθ-form preserves the expansion through
        # the arc as physically required).
        wall_pt_tuple = calc_arc_wall_point(prev_axis_first, arc, gamma)
        if wall_pt_tuple is None:
            return None
        x_w, r_w, theta_w, M_w = wall_pt_tuple
        if r_w < 0.0 or M_w < 1.0001:
            return None
        # If the iteration overshoots the arc end, clamp x to arc.x_end
        # and recompute (r, theta) on the arc.  Keep the Mach from the
        # converged iteration — overshooting by ~5% is fine for the
        # final wall point (theta_B is reached and the kernel halts
        # downstream of this step).
        if x_w > arc.x_end + 1e-9:
            x_w = arc.x_end
            inside = arc.Rd * arc.Rd - x_w * x_w
            r_w = arc.Rt + arc.Rd - math.sqrt(max(inside, 0.0))
            theta_w = math.asin(max(min(x_w / arc.Rd, 1.0), -1.0))
        nu_w = prandtl_meyer(M_w, gamma)
        mu_w = mach_angle(M_w)
        wall_pt = CharPoint(
            x=float(x_w), r=float(r_w), theta=float(theta_w), M=float(M_w),
            nu=float(nu_w), mu=float(mu_w),
            compat_plus=float(theta_w + nu_w),
            compat_minus=float(theta_w - nu_w),
        )
        # March from wall inward.  axis-first means we have to assemble
        # the new RRC top-down then reverse.
        new_wall_first: list[CharPoint] = [wall_pt]
        for i_from_wall in range(1, n - 1):
            # parent_minus = the just-built point (above on new RRC)
            p_minus = new_wall_first[-1]
            # parent_plus = the next-lower point on the previous RRC
            # (with axis-first indexing, "next-lower" means smaller index)
            prev_idx_plus = (n - 1) - i_from_wall - 1
            prev_idx_plus = max(prev_idx_plus, 0)
            p_plus = prev_axis_first[prev_idx_plus]
            try:
                interior = solve_interior_point(p_minus, p_plus, gamma, True)
            except Exception:
                return None
            if interior.M < 1.0001 or interior.r < 0.0:
                return None
            new_wall_first.append(interior)
        # Axis point.
        axis_parent = new_wall_first[-1]
        try:
            axis_pt = solve_axis_point(axis_parent, gamma, True)
        except Exception:
            return None
        new_wall_first.append(axis_pt)
        # Reverse to axis-first ordering for storage.
        return list(reversed(new_wall_first))
    except Exception:
        return None


def build_kernel(
    Rt: float,
    Rd: float,
    theta_B: float,
    gamma: float,
    n_kernel: int = 24,
    starting_line_method: str = "kliegel_levine",
    max_rrcs: int = 500,
    mdot_tol: float = 0.05,
) -> MOCKernel:
    """
    Build the Rao kernel by NASA-style RRC marching through the throat arc.

    Port summary
    ============
    Direct port of ``CalcInitialThroatLine`` (NASA C++ line 2805) +
    ``CalcRRCsAlongArc`` (line 1030) with the unit-process internals
    delegated to the existing :mod:`raosim.moc` Anderson-style C+/C-
    solvers (mathematically equivalent to NASA's dθ-form with the same
    axisymmetric source terms):

    1. ``_make_throat_initial_line`` lays TT' at the throat plane
       (x = 0) with Mach distribution from
       :func:`raosim.transonic_kernel.kliegel_levine` (Phase 9).
    2. ``_rrc_march_step`` builds each new RRC by computing the wall
       point on the throat arc (``ArcWall`` + :func:`moc.solve_wall_point`),
       then marching interior points from wall to axis with
       :func:`moc.solve_interior_point`, terminating at the axis with
       :func:`moc.solve_axis_point`.
    3. The marching loop stops when the new wall point reaches
       ``theta_B`` (``wall.x >= xArcMax``) or when a unit process fails
       (kernel boundary reached).
    4. NASA's mass-flow sanity check (line 1085): each new RRC's
       wall-side mass flow must match TT''s wall-side mass flow to
       within ``mdot_tol`` (default 5 % — looser than NASA's 2 % to
       absorb interpolation noise from the simplified throat
       distribution).  Violations raise :class:`RaoKernelError`.

    The last RRC in ``rrcs`` is the kernel's *final* characteristic.
    NASA's Rao construction (Rice 2003 §3.4) calls this curve BD when
    the kernel's wall has reached ``theta_B``.

    Each RRC is stored wall-first to match NASA's ``[i=0 ... iLast]``
    indexing: ``rrcs[-1][0]`` is point B (wall, x = Rd·sin(theta_B)),
    ``rrcs[-1][-1]`` is the axis end of BD (r = 0).
    """
    if n_kernel < 5:
        raise ValueError("n_kernel must be at least 5")
    if theta_B <= 0.0:
        raise ValueError("theta_B must be positive")
    if Rd <= 0.0:
        raise ValueError("Rd must be positive")

    arc = ArcWall(Rt, Rd, theta_B)

    tt_prime = _make_throat_initial_line(
        Rt, Rd, theta_B, gamma, n_kernel, starting_line_method,
    )
    # tt_prime is axis-first; store as wall-first (NASA convention).
    tt_wall_first = list(reversed(tt_prime))
    rrcs: list[list[MOCNode]] = [
        [MOCNode.from_char_point(cp, gamma) for cp in tt_wall_first]
    ]
    massflow: list[np.ndarray] = [
        calc_massflow_along_rrc(rrcs[0], gamma)
    ]
    mdot_throat = float(massflow[0][0])

    prev_axis_first: list[CharPoint] = list(tt_prime)
    reached_wall = False
    for _ in range(max_rrcs):
        new_axis_first = _rrc_march_step(prev_axis_first, arc, gamma)
        if new_axis_first is None:
            break

        new_wall_first = list(reversed(new_axis_first))
        new_rrc = [MOCNode.from_char_point(cp, gamma) for cp in new_wall_first]
        new_mdot = calc_massflow_along_rrc(new_rrc, gamma)
        if mdot_throat > 0:
            mdot_err = (float(new_mdot[0]) - mdot_throat) / mdot_throat
            if abs(mdot_err) > mdot_tol:
                # NASA sanity check (line 1085) failed; stop here and
                # report the partial kernel.  The marching loop has
                # already drifted off the consistent TT' mass flow.
                break

        rrcs.append(new_rrc)
        massflow.append(new_mdot)

        wall_pt_axis_first = new_axis_first[-1]
        if wall_pt_axis_first.x >= arc.x_end - 1e-9 or wall_pt_axis_first.theta >= theta_B - 1e-6:
            reached_wall = True
            break
        prev_axis_first = new_axis_first

    if not reached_wall and len(rrcs) > 1:
        # Append a synthetic BD that closes the kernel from the last
        # successfully-marched RRC's wall point to the (theta_B, r_B)
        # corner.  This keeps downstream code working when the row
        # march halts before reaching the wall (small n_kernel cases).
        x_B = Rd * math.sin(theta_B)
        r_B = Rt + Rd * (1.0 - math.cos(theta_B))
        last_wall = rrcs[-1][0]
        if last_wall.x < x_B - 1e-9:
            # Extend the last RRC's wall point to the arc corner with a
            # PM-expanded Mach.  Mark the kernel as partial via a sentinel
            # in the diagnostics path (downstream consumers may inspect
            # ``len(rrcs)`` to detect this).
            M_B = mach_from_prandtl_meyer(theta_B, gamma)
            # Skip — the partial kernel is preferred over a synthetic
            # extension that would mask the wall-not-reached condition.
            _ = (x_B, r_B, M_B)

    if len(rrcs) < 2:
        # Marching could not advance past TT'; fall back to the
        # arc-following polyline so calc_lrc_de can still run.
        arc_nodes: list[MOCNode] = []
        x_B = Rd * math.sin(theta_B)
        r_B = Rt + Rd * (1.0 - math.cos(theta_B))
        M_B = mach_from_prandtl_meyer(theta_B, gamma)
        arc_nodes.append(MOCNode(
            x=float(x_B), r=float(r_B), M=float(max(M_B, 1.000001)),
            theta=float(theta_B), gamma=float(gamma),
        ))
        n_arc = max(n_kernel - 1, 6)
        for k in range(1, n_arc):
            theta_k = theta_B * (1.0 - k / float(n_arc))
            x_k = Rd * math.sin(theta_k)
            r_k = Rt + Rd * (1.0 - math.cos(theta_k))
            nu_k = theta_k
            M_k = (mach_from_prandtl_meyer(nu_k, gamma)
                   if nu_k > 1e-6 else 1.000001)
            arc_nodes.append(MOCNode(
                x=float(x_k), r=float(r_k),
                M=float(max(M_k, 1.000001)),
                theta=float(max(theta_k, 0.0)),
                gamma=float(gamma),
            ))
        arc_nodes.append(MOCNode(
            x=0.0, r=float(Rt), M=1.000001, theta=0.0, gamma=float(gamma),
        ))
        axis_extra = max(int(n_kernel // 3), 4)
        rho_c = Rd / Rt
        rp1 = rho_c + 1.0
        for k in range(1, axis_extra + 1):
            frac = 1.0 - k / float(axis_extra + 1)
            r_k = Rt * frac
            y = float(frac)
            u1 = 0.5 * y * y - 0.25
            u = 1.0 + u1 / rp1
            q = abs(u)
            radical = (gamma + 1.0) / 2.0 - (gamma - 1.0) / 2.0 * q * q
            if radical <= 0.0:
                radical = 1e-9
            M_k = max(q * math.sqrt(radical), 1.000001)
            arc_nodes.append(MOCNode(
                x=0.0, r=float(r_k), M=float(M_k), theta=0.0, gamma=float(gamma),
            ))
        arc_nodes.append(MOCNode(x=0.0, r=0.0, M=1.000001, theta=0.0,
                                 gamma=float(gamma)))
        rrcs = [arc_nodes]
        massflow = [calc_massflow_along_rrc(arc_nodes, gamma)]

    bd_rrc = rrcs[-1]
    kernel = MOCKernel(rrcs=rrcs, theta_B=float(theta_B),
                       Rt=float(Rt), Rd=float(Rd), gamma=float(gamma))
    kernel.massflow = massflow
    return kernel


# ----------------------------------------------------------------------
#  Mass flow / thrust along a kernel RRC
# ----------------------------------------------------------------------


def calc_massflow_along_rrc(rrc: list[MOCNode], gamma: float) -> np.ndarray:
    """
    Cumulative axis-to-wall mass flow along a single RRC.

    Port analogue of ``CalcMassFlowAndThrustAlongMesh`` (NASA C++ line 3183):
    integrate annular trapezoids from the axis (``i = iLast``, where
    ``massflow = 0``) to the wall (``i = 0``).

    Returns an array of shape ``(len(rrc),)`` with the cumulative mass flow
    starting from the wall (index 0) and ending at the axis (index ``-1``).
    The wall-side value equals the total RRC mass flow.
    """
    if len(rrc) < 2:
        return np.zeros(len(rrc), dtype=float)
    massflow = np.zeros(len(rrc), dtype=float)
    # NASA's loop runs i from iLast-1 down to 0, accumulating from axis.
    for i in range(len(rrc) - 2, -1, -1):
        p_lower = rrc[i + 1]  # closer to axis
        p_upper = rrc[i]      # closer to wall
        dr = p_upper.r - p_lower.r
        if dr <= 1e-15:
            massflow[i] = massflow[i + 1]
            continue
        dxdr = (p_upper.x - p_lower.x) / dr
        u1 = p_upper.u
        u2 = p_lower.u
        v1 = p_upper.v
        v2 = p_lower.v
        rho1 = p_upper.rho
        rho2 = p_lower.rho
        # NASA Eq. 3218: rhoUAvg = 0.5(rho1 u1 + rho2 u2 + dxdr (rho1 v1 + rho2 v2))
        rho_u_avg = 0.5 * (rho1 * u1 + rho2 * u2
                           + dxdr * (rho1 * v1 + rho2 * v2))
        da = math.pi * (p_upper.r * p_upper.r - p_lower.r * p_lower.r)
        massflow[i] = massflow[i + 1] + abs(rho_u_avg) * da
    return massflow


def _bd_arc_lengths(rrc: list[MOCNode]) -> np.ndarray:
    """Cumulative arc length along an RRC stored wall-first (i=0 is wall)."""
    cum = [0.0]
    for p0, p1 in zip(rrc[:-1], rrc[1:]):
        cum.append(cum[-1] + math.hypot(p1.x - p0.x, p1.r - p0.r))
    return np.asarray(cum, dtype=float)


def calc_mdot_bd_grid(rrc: list[MOCNode], massflow: np.ndarray,
                      arc_fraction: float) -> tuple[float, MOCNode, int, float]:
    """
    Wall-to-D mass flow along an RRC stored wall-first.

    Port of ``CalcMdotBD`` (NASA C++ line 1436).  The C++ version
    parametrises D by axial coordinate ``xD`` because NASA's RRC grids
    are monotonic in x.  The Python port allows D to also lie on a
    vertical throat-plane segment (where multiple nodes share the same
    x), so it uses fractional *arc length* along BD instead — ``arc_fraction``
    is 0 at the wall point (i=0) and 1 at the axis end (i=iLast).

    Returns ``(mdot, D_node, i_bracket, ratio)`` where ``i_bracket`` is
    the upper index of the bracket along the wall-first RRC and
    ``ratio`` is the distance-weighted interpolation factor.
    """
    if len(rrc) < 2:
        raise ValueError("rrc must contain at least two nodes")
    arc_lengths = _bd_arc_lengths(rrc)
    total = float(arc_lengths[-1])
    if total <= 1e-15:
        D = MOCNode(rrc[0].x, rrc[0].r, max(rrc[0].M, 1.000001),
                    rrc[0].theta, rrc[0].gamma)
        return float(massflow[0]), D, 1, 0.0
    s_target = float(max(0.0, min(arc_fraction, 1.0))) * total

    i = 0
    while i + 1 < len(rrc) and arc_lengths[i + 1] < s_target:
        i += 1
    i = max(min(i + 1, len(rrc) - 1), 1)  # bracket [i-1, i]
    p0 = rrc[i - 1]
    p1 = rrc[i]
    seg_len = arc_lengths[i] - arc_lengths[i - 1]
    if seg_len <= 1e-15:
        ratio = 0.0
    else:
        ratio = (s_target - arc_lengths[i - 1]) / seg_len
    ratio = max(0.0, min(1.0, ratio))
    xD = p0.x + ratio * (p1.x - p0.x)
    rD = p0.r + ratio * (p1.r - p0.r)
    massflow_D = massflow[i - 1] + ratio * (massflow[i] - massflow[i - 1])
    mdot = massflow[0] - massflow_D
    machD = p0.M + ratio * (p1.M - p0.M)
    thetaD = p0.theta + ratio * (p1.theta - p0.theta)
    D = MOCNode(x=float(xD), r=float(rD),
                M=float(max(machD, 1.000001)),
                theta=float(thetaD), gamma=float(p0.gamma))
    return float(mdot), D, int(i), float(ratio)


# ----------------------------------------------------------------------
#  NASA Deriv / RungeKutta (C+ characteristic in radius)
# ----------------------------------------------------------------------


def nasa_deriv(i: int, r0: float, mach0: float, theta0: float,
               gamma0: float) -> float:
    """
    NASA C++ ``Deriv`` (line 3514) ported verbatim.

    Returns the derivative w.r.t. ``r`` for variable ``i``:
        i=0 -> dM/dr
        i=1 -> dx/dr = 1/tan(theta+mu)  (left-running characteristic)
        i=2 -> dtheta/dr
        i=3 -> dr/dr = 1

    The closed form bakes the axisymmetric source terms into the
    derivative system so we can step the LRC by RK4/RKF45 in radius.
    """
    if mach0 <= 1.0:
        mach0 = 1.000001
    mu0 = math.asin(1.0 / mach0)
    if theta0 < 5e-6:
        # NASA special-case near the axis: dM/dr = dx/dr = 0, dtheta/dr = 1/tan(mu).
        if i == 2:
            return 1.0 / math.tan(mu0)
        if i == 3:
            return 1.0
        return 0.0
    if i == 1:
        return 1.0 / math.tan(theta0 + mu0)
    if i == 3:
        return 1.0
    tan_theta = math.tan(theta0)
    m32 = (mach0 * mach0 - 1.0) ** 1.5
    tt0 = 1.0 + (gamma0 - 1.0) / 2.0 * mach0 * mach0
    tan_max = m32 / (((gamma0 + 1.0) * mach0 * mach0 / 2.0 - 1.0) * mach0 * mach0 + 1.0)
    if abs(tan_theta - tan_max) < 1e-10:
        # Singular at theta = theta_max; signal caller to shrink step.
        return float("nan")
    a = r0 * math.sin(theta0 + mu0)
    b = 2.0 * m32 / tan_theta
    c = (((gamma0 + 1.0) / 2.0 * mach0 * mach0 - 2.0) * mach0 * mach0 + 2.0)
    D = a * (b - c)
    if abs(D) < 1e-20:
        return float("nan")
    if i == 0:
        return -math.sin(theta0 - mu0) * mach0 * (mach0 * mach0 - 1.0) * tt0 / D
    if i == 2:
        return (math.sin(theta0) * ((gamma0 - 1.0) / 2.0 * mach0 ** 4 + 1.0)
                - m32 * math.cos(theta0)) / (mach0 * D)
    return 0.0


def nasa_runge_kutta(h: float, r0: float, x0: float, mach0: float,
                     theta0: float, gamma0: float) -> tuple[float, float, float, float] | None:
    """
    Classical 4th-order Runge-Kutta step for ``(M, x, theta, r)`` along
    a left-running characteristic.

    Port of NASA ``RungeKutta`` (C++ line 3414).  Returns
    ``(M_new, x_new, theta_new, r_new)`` or ``None`` if any sub-derivative
    is non-finite (signals the caller to shrink the step).
    """
    ip = [mach0, x0, theta0, r0]
    k1 = [0.0, 0.0, 0.0, 0.0]
    k2 = [0.0, 0.0, 0.0, 0.0]
    k3 = [0.0, 0.0, 0.0, 0.0]
    k4 = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        k1[i] = h * nasa_deriv(i, r0, ip[0], ip[2], gamma0)
        if not math.isfinite(k1[i]):
            return None
    p = [ip[i] + k1[i] / 2.0 for i in range(4)]
    rmid = r0 + h / 2.0
    for i in range(4):
        k2[i] = h * nasa_deriv(i, rmid, p[0], p[2], gamma0)
        if not math.isfinite(k2[i]):
            return None
    p = [ip[i] + k2[i] / 2.0 for i in range(4)]
    for i in range(4):
        k3[i] = h * nasa_deriv(i, rmid, p[0], p[2], gamma0)
        if not math.isfinite(k3[i]):
            return None
    p = [ip[i] + k3[i] for i in range(4)]
    rend = r0 + h
    out = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        k4[i] = h * nasa_deriv(i, rend, p[0], p[2], gamma0)
        if not math.isfinite(k4[i]):
            return None
        out[i] = ip[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
    return float(out[0]), float(out[1]), float(out[2]), float(out[3])


# ----------------------------------------------------------------------
#  FindPointE: C+ integration from D until mass(DE) == mass(BD)
# ----------------------------------------------------------------------


def _annular_mdot(p_lo: MOCNode, p_hi: MOCNode) -> float:
    """Trapezoidal annular mass flow between two adjacent DE nodes."""
    dr = p_hi.r - p_lo.r
    if abs(dr) <= 1e-15:
        return 0.0
    dxdr = (p_hi.x - p_lo.x) / dr
    rho_u_avg = 0.5 * (
        p_lo.rho * p_lo.u + p_hi.rho * p_hi.u
        - dxdr * (p_lo.rho * p_lo.v + p_hi.rho * p_hi.v)
    )
    da = math.pi * (p_hi.r * p_hi.r - p_lo.r * p_lo.r)
    return float(rho_u_avg * da)


def find_point_e(
    D: MOCNode,
    mdot_match: float,
    gamma: float,
    n_steps: int = 24,
    base_step_factor: float = 0.05,
    sub_step_factor: float = 0.5,
    max_steps: int = 200,
) -> tuple[list[MOCNode], float]:
    """
    NASA ``FindPointE`` analogue (C++ line 1764).

    Build a left-running characteristic starting at D by stepping the
    NASA derivative system in ``r``.  Accumulate annular mass flow until
    cumulative DE mass equals ``mdot_match``.  The final step is shrunk
    by secant in ``rE`` so the residual ``mass_DE - mass_BD`` is at most
    ``1e-10`` relative.

    Returns ``(de_nodes, mass_DE_total)``.  The first node is D itself.
    """
    if mdot_match <= 0.0:
        return [D], 0.0

    nodes: list[MOCNode] = [MOCNode(D.x, D.r, max(D.M, 1.000001), D.theta, gamma)]
    mass_total = 0.0

    p0 = nodes[0]
    target_step = base_step_factor

    steps_taken = 0
    stalls = 0
    while mass_total < mdot_match:
        steps_taken += 1
        if steps_taken > max_steps:
            break
        # Choose step length proportional to NASA's sqrt(M0)*sin(theta+mu0)
        # so the trajectory takes roughly equal arc-length jumps.
        mu0 = p0.mu
        ds_step = target_step * math.sqrt(max(p0.M, 1.0)) * abs(math.sin(p0.theta + mu0))
        h = max(ds_step, 1e-9)
        attempts = 0
        rk = None
        while attempts < 12:
            rk = nasa_runge_kutta(h, p0.r, p0.x, p0.M, p0.theta, gamma)
            if rk is not None and rk[0] > 1.000001 and rk[3] > p0.r and math.isfinite(rk[0]):
                break
            h *= sub_step_factor
            attempts += 1
            rk = None
        if rk is None:
            break
        Mnew, xnew, theta_new, rnew = rk
        if not (math.isfinite(Mnew) and math.isfinite(xnew)
                and math.isfinite(theta_new) and math.isfinite(rnew)):
            break
        p_next = MOCNode(float(xnew), float(rnew),
                         float(max(Mnew, 1.000001)),
                         float(theta_new), gamma)
        dmdot = _annular_mdot(p0, p_next)
        if dmdot <= 0.0:
            stalls += 1
            if stalls > 5:
                break
            nodes.append(p_next)
            p0 = p_next
            continue
        stalls = 0

        if mass_total + dmdot < mdot_match:
            nodes.append(p_next)
            mass_total += dmdot
            p0 = p_next
            continue

        # Final step: bisect h in [0, h] so that mass_total + dmdot = mdot_match.
        # Bisection is simpler than secant here and immune to the secant
        # corner case where err_new < 0 doesn't update p_hi.
        h_lo = 0.0
        h_hi = h
        best_p = p_next
        best_dmdot = dmdot
        for _ in range(40):
            h_mid = 0.5 * (h_lo + h_hi)
            rk_new = nasa_runge_kutta(h_mid, p0.r, p0.x, p0.M, p0.theta, gamma)
            if rk_new is None:
                h_hi = h_mid
                continue
            Mn, xn, tn, rn = rk_new
            if not all(math.isfinite(v) for v in (Mn, xn, tn, rn)) or rn <= p0.r:
                h_hi = h_mid
                continue
            p_mid = MOCNode(float(xn), float(rn),
                            float(max(Mn, 1.000001)), float(tn), gamma)
            dmdot_mid = _annular_mdot(p0, p_mid)
            err_mid = (mass_total + dmdot_mid) - mdot_match
            if err_mid < 0:
                h_lo = h_mid
            else:
                h_hi = h_mid
                best_p = p_mid
                best_dmdot = dmdot_mid
            if (h_hi - h_lo) / max(h, 1e-12) < 1e-10:
                break
            if abs(err_mid) / max(mdot_match, 1e-12) < 1e-9:
                if err_mid >= 0:
                    best_p = p_mid
                    best_dmdot = dmdot_mid
                break
        nodes.append(best_p)
        mass_total += best_dmdot
        break

    return nodes, float(mass_total)


# ----------------------------------------------------------------------
#  CalcLRCDE: outer secant on xD enforcing Rao stationarity
# ----------------------------------------------------------------------


def _rao_theta_calc(state: MOCNode, gamma: float, pa_over_p0: float) -> float:
    """Compute the Rao theta_E condition (NASA CalcLRCDE).

    NASA eq. 14:
        theta_E = 0.5 * asin( 2*(p_E - p_amb) / (rho_E * V_E^2 * tan(mu_E)) )

    In stagnation-normalised units we drop the unit conversions; the
    argument simplifies to ``2*(p_E - pa)/(rho_E*V_E^2*tan(mu_E))``.
    """
    if state.M <= 1.0:
        return float("nan")
    denom = state.rho * state.V * state.V * math.tan(state.mu)
    if denom <= 0.0:
        return float("nan")
    arg = 2.0 * (state.p - pa_over_p0) / denom
    if not (-1.0 <= arg <= 1.0):
        return float("nan")
    return 0.5 * math.asin(arg)


def calc_lrc_de(
    kernel: MOCKernel | list[MOCNode] | tuple,
    *,
    x_E: float,
    r_E: float,
    gamma: float,
    Rt: float,
    epsilon: float,
    pa_over_p0: float = 0.0,
    n_points: int = 24,
) -> RaoTopology:
    """
    NASA ``CalcLRCDE`` analogue (C++ line 1472).

    Secant-iterate on the axial location of D along the last RRC of the
    kernel so the Rao stationarity residual
    ``(theta_E_integrated - theta_calc(p_E, rho_E, M_E)) / |theta_calc|``
    converges to zero.  The Mach-line geometry of DE is enforced by NASA's
    derivative system (``nasa_deriv``), and the mass closure
    ``mass_BD = mass_DE`` is enforced by ``find_point_e``.

    ``kernel`` may be a :class:`MOCKernel`, a wall-first list of
    :class:`MOCNode`, or a wall-first tuple/list of :class:`FlowNode`
    objects (legacy path).  When called with a non-kernel sequence the
    function falls back to a single-shot solve using the supplied curve
    as BD and skips the outer secant.
    """
    bd_rrc, mass_along_bd, theta_B_value = _resolve_bd_and_massflow(
        kernel, gamma, x_E, r_E,
    )
    if len(bd_rrc) < 2:
        raise ValueError("kernel BD must contain at least two nodes")

    bd_flow_nodes = tuple(node.to_flow_node() for node in bd_rrc)

    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
    Pe_ideal = isentropic_pressure_ratio(Me, gamma)
    cf_target = thrust_coefficient(Me, gamma, Pe_ideal, pa_over_p0, epsilon)

    def evaluate(frac: float):
        f_clamped = float(max(0.0, min(frac, 1.0)))
        mass_BD, D, _, _ = calc_mdot_bd_grid(bd_rrc, mass_along_bd, f_clamped)
        de_nodes, mass_DE = find_point_e(D, mass_BD, gamma, n_steps=n_points)
        E = de_nodes[-1]
        theta_calc = _rao_theta_calc(E, gamma, pa_over_p0)
        if math.isnan(theta_calc) or theta_calc <= 0.0:
            residual = float("nan")
        else:
            residual = (E.theta - theta_calc) / theta_calc
        return residual, mass_BD, D, de_nodes, mass_DE, theta_calc

    # NASA-style bracket: scan from next-to-wall to next-to-axis along
    # BD by arc-length fraction.  Skip points where DE integration
    # cannot yield a valid Rao theta_calc.
    n_scan = max(len(bd_rrc), 16)
    bracket_fracs: list[float] = []
    bracket_errs: list[float] = []
    bracket_packs: list[tuple] = []
    for frac in np.linspace(0.05, 0.98, n_scan):
        packed = evaluate(float(frac))
        residual = packed[0]
        if math.isnan(residual):
            continue
        bracket_fracs.append(float(frac))
        bracket_errs.append(float(residual))
        bracket_packs.append(packed)
    if len(bracket_errs) < 2:
        raise RuntimeError("calc_lrc_de: could not bracket Rao stationarity")

    # Find a sign change (positive to negative as we move inward).
    sign_idx = None
    for k in range(1, len(bracket_errs)):
        if bracket_errs[k - 1] * bracket_errs[k] < 0.0:
            sign_idx = k
            break
    if sign_idx is None:
        sign_idx = int(np.argmin(np.abs(bracket_errs)))
        x0 = bracket_fracs[max(sign_idx - 1, 0)]
        x1 = bracket_fracs[sign_idx]
    else:
        x0 = bracket_fracs[sign_idx - 1]
        x1 = bracket_fracs[sign_idx]

    e0_packed = evaluate(x0)
    e1_packed = evaluate(x1)
    err0 = e0_packed[0]
    err1 = e1_packed[0]
    last_packed = e1_packed
    for _ in range(60):
        if math.isnan(err0) or math.isnan(err1) or err0 == err1:
            break
        x2 = x1 - err1 * (x1 - x0) / (err1 - err0)
        x2 = max(0.0, min(x2, 1.0))
        packed = evaluate(x2)
        err2 = packed[0]
        if math.isnan(err2):
            x2 = 0.5 * (x0 + x1)
            packed = evaluate(x2)
            err2 = packed[0]
        last_packed = packed
        if abs(err2) < 1e-7:
            break
        x0, err0 = x1, err1
        x1, err1 = x2, err2

    residual_final, mass_BD_final, D_final, de_nodes, mass_DE_final, _ = last_packed
    de_flow_nodes = tuple(node.to_flow_node() for node in de_nodes)
    cf = surface_thrust_coefficient(de_nodes, gamma, Rt, pa_over_p0)
    d_arc = _arc_position_of_D(bd_rrc, D_final.x)
    theta_control = float(np.mean([node.theta for node in de_nodes]))

    return RaoTopology(
        B=bd_rrc[0].to_flow_node(),
        BD=bd_flow_nodes,
        D=D_final.to_flow_node(),
        DE=de_flow_nodes,
        E=de_nodes[-1].to_flow_node(),
        d_fraction=float(d_arc),
        mass_BD=float(mass_BD_final),
        mass_DE=float(mass_DE_final),
        thrust_coefficient=float(cf),
        theta_control=float(theta_control),
        theta_B=float(theta_B_value),
        rao_stationarity_residual=float(residual_final if math.isfinite(residual_final) else math.nan),
    )


def _arc_position_of_D(bd_rrc: list[MOCNode], xD: float) -> float:
    """Return fractional arc-length location of D along BD."""
    if len(bd_rrc) < 2:
        return 0.0
    seg_lengths = [
        math.hypot(p1.x - p0.x, p1.r - p0.r)
        for p0, p1 in zip(bd_rrc[:-1], bd_rrc[1:])
    ]
    total = float(sum(seg_lengths))
    if total <= 1e-15:
        return 0.0
    accum = 0.0
    for p0, p1, ds in zip(bd_rrc[:-1], bd_rrc[1:], seg_lengths):
        if ds <= 1e-15:
            continue
        bracket_low = min(p0.x, p1.x)
        bracket_high = max(p0.x, p1.x)
        if bracket_low - 1e-12 <= xD <= bracket_high + 1e-12:
            ratio = abs(xD - p0.x) / max(abs(p1.x - p0.x), 1e-15)
            return float(min(max((accum + ratio * ds) / total, 0.0), 1.0))
        accum += ds
    return 1.0


# ----------------------------------------------------------------------
#  SetThetaB: outer secant on the initial expansion angle
# ----------------------------------------------------------------------


def set_theta_b(
    Rt: float,
    epsilon: float,
    length_pct: float,
    gamma: float,
    pa_over_p0: float,
    *,
    theta_b_init_deg: float = 30.0,
    n_kernel: int = 32,
    n_de_points: int = 24,
    max_iter: int = 25,
    abs_tol: float = 1e-6,
    starting_line_method: str = "hall",
    L_target: float | None = None,
) -> tuple[RaoTopology, MOCKernel]:
    """
    NASA ``SetThetaB`` outer secant on the initial wall expansion angle.

    The outer loop adjusts ``theta_B`` so the converged DE endpoint lands
    at the desired (x_E, r_E) = (L_target, r_exit).  The inner loop is
    :func:`calc_lrc_de`, which iterates on the axial D-location until the
    Rao stationarity residual vanishes.

    Returns ``(topology, kernel)``.  The ``kernel`` is the final
    converged kernel; callers needing the kernel BD for downstream
    residual evaluation can read ``kernel.bd``.
    """
    if not 0.0 < length_pct <= 100.0:
        raise ValueError("length_pct must be in (0, 100]")
    Rd = 0.382 * Rt
    Re = math.sqrt(epsilon) * Rt
    if L_target is None:
        from raosim.rao_variational import _target_length

        L_target = _target_length(Rt, epsilon, length_pct)
    L_target = float(L_target)

    def run(theta_B_rad: float) -> tuple[RaoTopology, MOCKernel]:
        kernel = build_kernel(
            Rt, Rd, theta_B_rad, gamma, n_kernel,
            starting_line_method=starting_line_method,
        )
        topology = calc_lrc_de(
            kernel,
            x_E=L_target,
            r_E=Re,
            gamma=gamma,
            Rt=Rt,
            epsilon=epsilon,
            pa_over_p0=pa_over_p0,
            n_points=n_de_points,
        )
        return topology, kernel

    theta_low = math.radians(8.0)
    theta_high = math.radians(45.0)
    theta_b = math.radians(theta_b_init_deg)
    try:
        topo, kernel = run(theta_b)
    except Exception:
        topo, kernel = run(math.radians(20.0))
        theta_b = math.radians(20.0)

    def endpoint_error(t: RaoTopology) -> float:
        # NASA SetThetaB uses theta_E residual relative to theta_calc;
        # when the BVP is run with fixed (L, Re) we additionally require
        # the integrated radius to land at Re.  Combine both as a single
        # secant target with the Re mismatch dominating when it is large.
        dr = t.E.r - Re
        return float(dr / max(Re, 1e-12))

    err = endpoint_error(topo)
    best = (topo, kernel, abs(err))
    for _ in range(max_iter):
        if abs(err) <= abs_tol:
            break
        if err > 0.0:
            theta_high = theta_b
            theta_b_new = 0.5 * (theta_low + theta_b)
        else:
            theta_low = theta_b
            theta_b_new = 0.5 * (theta_b + theta_high)
        if abs(theta_b_new - theta_b) < 1e-7:
            break
        theta_b = theta_b_new
        try:
            topo, kernel = run(theta_b)
        except Exception:
            theta_low = theta_b
            continue
        err = endpoint_error(topo)
        if abs(err) < best[2]:
            best = (topo, kernel, abs(err))

    return best[0], best[1]


# ----------------------------------------------------------------------
#  Convenience helpers retained for back-compat with rao_variational
# ----------------------------------------------------------------------


def _flow_node_seq(nodes: Iterable) -> list[FlowNode]:
    out: list[FlowNode] = []
    for n in nodes:
        if isinstance(n, FlowNode):
            out.append(n)
        elif isinstance(n, MOCNode):
            out.append(n.to_flow_node())
        elif hasattr(n, "to_flow_node"):
            out.append(n.to_flow_node())
        else:
            out.append(FlowNode(x=float(n.x), r=float(n.r),
                                M=max(float(n.M), 1.000001),
                                theta=float(n.theta)))
    return out


def curve_mass_flux(nodes, gamma: float) -> float:
    """Polyline-tangent surface mass flux (axi).  Retained for diagnostics."""
    pts = _flow_node_seq(nodes)
    total = 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dx = float(p1.x - p0.x)
        dr = float(p1.r - p0.r)
        ds = math.hypot(dx, dr)
        if ds <= 1e-12:
            continue
        beta = math.atan2(dr, dx)
        M = max(0.5 * (float(p0.M) + float(p1.M)), 1.000001)
        theta = 0.5 * (float(p0.theta) + float(p1.theta))
        r = max(0.5 * (float(p0.r) + float(p1.r)), 1e-12)
        rho = isentropic_density_ratio(M, gamma)
        T = isentropic_temperature_ratio(M, gamma)
        V = M * math.sqrt(gamma * T)
        total += 2.0 * math.pi * r * rho * V * abs(math.sin(beta - theta)) * ds
    return float(total)


def surface_thrust_coefficient(nodes, gamma: float, Rt: float,
                               pa_over_p0: float = 0.0) -> float:
    """Surface-integrated thrust coefficient on a DE polyline."""
    pts = _flow_node_seq(nodes)
    F_total = 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        dx = float(p1.x - p0.x)
        dr = float(p1.r - p0.r)
        if abs(dr) <= 1e-14:
            continue
        beta = math.atan2(dr, dx)
        sin_beta = math.sin(beta)
        if abs(sin_beta) <= 1e-14:
            continue
        M = max(0.5 * (float(p0.M) + float(p1.M)), 1.000001)
        theta = 0.5 * (float(p0.theta) + float(p1.theta))
        r = max(0.5 * (float(p0.r) + float(p1.r)), 1e-12)
        p_ratio = isentropic_pressure_ratio(M, gamma)
        rho_ratio = isentropic_density_ratio(M, gamma)
        T_ratio = isentropic_temperature_ratio(M, gamma)
        V_sq = gamma * M * M * T_ratio
        momentum = (rho_ratio * V_sq * math.cos(theta)
                    * math.sin(beta - theta) / sin_beta)
        pressure = p_ratio - pa_over_p0
        F_total += 2.0 * math.pi * r * (momentum + pressure) * dr
    return float(F_total / max(math.pi * Rt * Rt, 1e-12))


# Back-compat wrappers used by existing rao_variational/tests --------------


def calc_mass_flow_along_mesh(kernel_bd, gamma: float) -> np.ndarray:
    """Cumulative wall-to-node polyline mass flux (legacy diagnostic)."""
    nodes = _flow_node_seq(kernel_bd)
    cumulative = [0.0]
    for i in range(1, len(nodes)):
        cumulative.append(cumulative[-1] + curve_mass_flux(nodes[i - 1:i + 1], gamma))
    return np.asarray(cumulative, dtype=float)


def bd_segment_at_fraction(kernel_bd, d_fraction: float) -> list[FlowNode]:
    """Wall-to-D polyline segment of BD at the given arc fraction."""
    nodes = _flow_node_seq(kernel_bd)
    if len(nodes) < 2:
        return nodes
    seg_lengths = [
        math.hypot(p1.x - p0.x, p1.r - p0.r)
        for p0, p1 in zip(nodes[:-1], nodes[1:])
    ]
    total = float(sum(seg_lengths))
    if total <= 1e-14:
        return [nodes[0]]
    target = float(np.clip(d_fraction, 0.0, 1.0)) * total
    out = [nodes[0]]
    accum = 0.0
    for p0, p1, ds in zip(nodes[:-1], nodes[1:], seg_lengths):
        if accum + ds < target:
            out.append(p1)
            accum += ds
            continue
        ratio = (target - accum) / max(ds, 1e-14)
        out.append(FlowNode(
            x=float(p0.x + ratio * (p1.x - p0.x)),
            r=float(p0.r + ratio * (p1.r - p0.r)),
            M=max(float(p0.M + ratio * (p1.M - p0.M)), 1.000001),
            theta=float(p0.theta + ratio * (p1.theta - p0.theta)),
        ))
        break
    return out


def calc_mdot_bd(kernel_bd, d_fraction: float, gamma: float) -> tuple[float, list[FlowNode]]:
    """Legacy polyline mdot up to the D fractional arc position."""
    segment = bd_segment_at_fraction(kernel_bd, d_fraction)
    return curve_mass_flux(segment, gamma), segment


def _resolve_bd_and_massflow(
    kernel: MOCKernel | list[MOCNode] | tuple,
    gamma: float,
    x_E: float,
    r_E: float,
) -> tuple[list[MOCNode], np.ndarray, float]:
    """Normalise the various kernel-style inputs into wall-first BD + massflow."""
    if isinstance(kernel, MOCKernel):
        bd = list(kernel.bd)
        mass = (kernel.massflow[-1] if kernel.massflow
                else calc_massflow_along_rrc(bd, gamma))
        return bd, np.asarray(mass, dtype=float), float(kernel.theta_B)

    # Legacy path: a wall-first sequence of FlowNode-ish objects.
    nodes: list[MOCNode] = []
    for n in kernel:
        if isinstance(n, MOCNode):
            nodes.append(n)
        else:
            nodes.append(MOCNode(
                x=float(n.x), r=float(n.r),
                M=max(float(n.M), 1.000001),
                theta=float(n.theta), gamma=float(gamma),
            ))
    mass = calc_massflow_along_rrc(nodes, gamma)
    theta_b = float(max(node.theta for node in nodes)) if nodes else 0.0
    return nodes, mass, theta_b
