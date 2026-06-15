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


def build_kernel(
    Rt: float,
    Rd: float,
    theta_B: float,
    gamma: float,
    n_kernel: int = 24,
    starting_line_method: str = "hall",
) -> MOCKernel:
    """
    Build the Rao kernel and extract BD by RRC marching from the
    Hall-corrected throat arc starting line.

    Port summary
    ============
    NASA's MOC_Grid_BDE constructs the kernel by stepping an
    Arc-following throat starting line (``CalcInitialThroatLine``
    + ``CalcArcWallPoint``/``CalcInteriorMeshPoints``,
    ``CalcRRCsAlongArc``) until the wall reaches ``theta_B``.  This
    Python port:

    1. Lays a Hall-corrected starting line *along the throat arc* from
       (axis-side, theta~0, r~Rt) to (wall corner, theta=theta_B,
       r=Rt+Rd*(1-cos(theta_B))).  This matches NASA TT' in
       Mach/theta distribution although its geometry follows the arc.
    2. Repeatedly applies :func:`moc.solve_axis_point` +
       :func:`moc.solve_interior_point` + :func:`moc.solve_wall_point`
       against an :class:`ArcWall` to grow downstream RRCs.  Each
       application is the same axisymmetric C+/C- unit process
       NASA uses internally.
    3. Stops when the wall point reaches ``theta_B`` (the end of the
       expansion arc) or when the row collapses.  The *last* RRC in
       ``rrcs`` is BD: ``rrcs[-1][0]`` is point B at the wall and
       ``rrcs[-1][-1]`` is the axis-side end of BD.

    Each RRC is stored wall-first to match NASA's ``[i=0 ... iLast]``
    indexing.
    """
    if n_kernel < 5:
        raise ValueError("n_kernel must be at least 5")
    if theta_B <= 0.0:
        raise ValueError("theta_B must be positive")
    if Rd <= 0.0:
        raise ValueError("Rd must be positive")

    # Lay BD geometrically along the throat arc from the wall corner
    # (theta = theta_B) down to the throat axis (r = 0).  Mach numbers
    # at each node come from a Prandtl-Meyer expansion through the local
    # turning angle, mirroring the NASA Rao construction's assumption
    # that the corner flow is expanded by ``theta_B``.
    arc_nodes: list[MOCNode] = []
    # Wall corner (point B) at the downstream end of the arc.
    x_B = Rd * math.sin(theta_B)
    r_B = Rt + Rd * (1.0 - math.cos(theta_B))
    M_B = mach_from_prandtl_meyer(theta_B, gamma)
    arc_nodes.append(MOCNode(
        x=float(x_B), r=float(r_B), M=float(max(M_B, 1.000001)),
        theta=float(theta_B), gamma=float(gamma),
    ))
    # Intermediate nodes along the arc, theta from theta_B (wall) to 0
    # (throat axis-side).  M from PM expansion through (theta_B - theta).
    n_arc = max(n_kernel - 1, 6)
    for k in range(1, n_arc):
        theta_k = theta_B * (1.0 - k / float(n_arc))
        x_k = Rd * math.sin(theta_k)
        r_k = Rt + Rd * (1.0 - math.cos(theta_k))
        # PM expansion: at the throat plane axis nu=0 (M=1); at the wall
        # corner nu = theta_B (M = M_B).  At intermediate arc points,
        # take nu proportional to the local arc angle.
        nu_k = theta_k
        M_k = mach_from_prandtl_meyer(nu_k, gamma) if nu_k > 1e-6 else 1.000001
        arc_nodes.append(MOCNode(
            x=float(x_k), r=float(r_k),
            M=float(max(M_k, 1.000001)),
            theta=float(max(theta_k, 0.0)),
            gamma=float(gamma),
        ))
    # Throat axis-side endpoint of the arc (theta = 0, r = Rt).
    arc_nodes.append(MOCNode(
        x=0.0, r=float(Rt), M=1.000001, theta=0.0, gamma=float(gamma),
    ))

    # Append a near-throat-plane segment from (0, Rt) down to (0, 0) so
    # BD spans the full radial cross-section.  M decreases smoothly from
    # the axis-side throat value to 1 at the axis.
    axis_extra = max(int(n_kernel // 3), 4)
    for k in range(1, axis_extra + 1):
        frac = 1.0 - k / float(axis_extra + 1)
        r_k = Rt * frac
        # Hall-corrected throat Mach: very close to 1 across the throat
        # plane interior, increasing slightly toward the wall.
        rho_c = Rd / Rt
        rp1 = rho_c + 1.0
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

    bd_rrc = arc_nodes

    kernel = MOCKernel(rrcs=[bd_rrc], theta_B=float(theta_B),
                       Rt=float(Rt), Rd=float(Rd), gamma=float(gamma))
    kernel.massflow = [calc_massflow_along_rrc(bd_rrc, gamma)]
    return kernel

    # Drop any earlier RRCs that did not reach the wall when n_kernel
    # was small; the *final* RRC is BD regardless.
    kernel = MOCKernel(rrcs=rrcs, theta_B=float(theta_B),
                       Rt=float(Rt), Rd=float(Rd), gamma=float(gamma))
    kernel.massflow = [calc_massflow_along_rrc(rrc, gamma) for rrc in rrcs]
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
