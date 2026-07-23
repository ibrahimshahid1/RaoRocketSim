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
    fallback_used: bool = False
    reached_wall: bool = False
    # How ``theta_B`` was chosen, for honest downstream reporting
    # (J5 de-circularization: ``RaoSolution.theta_N`` echoes this angle
    # under the characteristic formulation).  ``"fixed_end_secant"`` =
    # set_theta_b converged it against the fixed-(L, eps) closure;
    # ``"bvp_solved"`` = theta_B was a live unknown of the JAX BVP
    # (RaoSolverConfig.solve_theta_b, J3b-2) and the kernel was
    # re-frozen at the LM-solved angle; ``"frozen_override"`` =
    # RaoSolverConfig.theta_b_freeze_deg; ``"seed_guess"`` = the secant
    # failed/was skipped and the kernel was built at the raw guess
    # angle (chart-flavoured — treat the reported theta_N as
    # low-quality); ``"kernel_build_angle"`` = constructed directly via
    # build_kernel outside the seeding path.
    theta_b_provenance: str = "kernel_build_angle"

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


@dataclass(frozen=True)
class BDERegion:
    """Rows produced by the post-kernel BFE port slice.

    ``rows`` keeps the raw B-to-DE ``CalcBDERegion`` seed strip.
    ``grid_rows`` is the physical B-D-E strip after wall cropping and ends
    on DE; it deliberately excludes the auxiliary DE-to-axis continuation.
    ``full_grid_rows`` retains only the valid prefix of that auxiliary
    continuation for diagnostics.  If it approaches a downstream caustic,
    the prefix terminates before the first zero/reversed cell.
    """

    rows: tuple[tuple[FlowNode, ...], ...]
    iD: int
    grid_rows: tuple[tuple[FlowNode, ...], ...] = ()
    full_grid_rows: tuple[tuple[FlowNode, ...], ...] = ()
    wall_contour: tuple[FlowNode, ...] = ()
    # ``complete_remaining_mesh`` is True when every remaining-mesh row was
    # built and terminated on the axis (r == 0) — i.e. the march ran to
    # completion.  It is independent of negative-r truncation, which is a
    # normal near-axis event (see ``negative_r_truncated_rows``); a completed
    # mesh may still contain axis-truncated rows.
    complete_remaining_mesh: bool = False
    wall_contour_complete: bool = False
    # Number of CalcRemainingMesh rows that terminated early because the
    # interior unit process produced a negative-radius point.  NASA's source
    # closes such rows at the axis and continues, so this is a diagnostic
    # count, not a failure: it lets audits distinguish "ran to completion"
    # from "ran to completion but some rows were axis-truncated".
    negative_r_truncated_rows: int = 0
    # Rows whose downstream continuation approached a characteristic
    # caustic: the next quadrilateral would have zero or reversed
    # orientation.  The invalid point is discarded and the still-valid row
    # is closed with the regular axis unit process.  This protects the
    # construction mesh without changing the upstream B-D-E seed strip used
    # to locate the wall.
    topology_truncated_rows: int = 0


@dataclass(frozen=True)
class RaoSourceContour:
    """Visible-source NASA/JHU contour construction artifact.

    This is the source-port path through the stages currently available in
    this module: kernel march, D/DE construction, BDE remaining mesh, and wall
    contour extraction.  It intentionally reports length/exit closure as a
    diagnostic because ``SetThetaB``/``CropNozzleToLength`` are not yet
    canonical in Python.
    """

    kernel: MOCKernel
    topology: RaoTopology
    bfe: BDERegion
    wall: tuple[FlowNode, ...]
    wall_export: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FullControlSurfaceResult:
    """Thrust audit for Rao's complete ``C-D-E`` control surface.

    ``DE`` alone carries only the mass selected between B and D.  Rao's
    momentum balance (1958, eqs. 1--2 and fig. 1) is written on the complete
    surface from the symmetry-axis point C, through D, to the wall point E.
    ``CD`` is recovered from the actual kernel connectivity; no mass-fraction
    scaling of an ideal thrust coefficient is used.
    """

    CD: tuple[FlowNode, ...]
    CDE: tuple[FlowNode, ...]
    cf_cd: float
    cf_de: float
    cf_cde: float
    mass_flux_cd: float
    mass_flux_de: float
    mass_flux_cde: float
    kernel_throat_mass_flux: float
    mass_residual_rel: float
    d_projection_distance: float
    d_state_mach_jump: float
    d_state_theta_jump: float
    d_projection_tol_over_rt: float
    d_mach_jump_tol: float
    d_theta_jump_tol: float
    mass_residual_rel_tol: float
    complete: bool


FULL_CONTROL_D_PROJECTION_TOL_OVER_RT = 5.0e-3
FULL_CONTROL_D_MACH_JUMP_TOL = 2.0e-2
FULL_CONTROL_D_THETA_JUMP_TOL = math.radians(0.5)
FULL_CONTROL_MASS_RESIDUAL_REL_TOL = 2.0e-2


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


class ThetaBTooLow(RaoKernelError):
    """NASA ``SEC_FAIL_LOW``: the theta_B kernel cannot satisfy the inner
    D/E condition from below — the outer SetThetaB loop should raise
    theta_B (C++ MOC_GridCalc_BDE.cpp lines 359-420)."""


class ThetaBTooHigh(RaoKernelError):
    """NASA ``SEC_FAIL_HIGH``: the inner D/E condition overshoots even at
    the wall end of BD — the outer SetThetaB loop should lower theta_B."""


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


def _visible_source_kl_throat(
    r_over_Rt: float,
    x_over_Rt: float,
    gamma: float,
    Rc_over_Rt: float,
) -> tuple[float, float]:
    """Literal visible-source ``KLThroat`` AXI branch.

    This intentionally preserves the checked-in C++ coefficients and typos in
    ``MOC_GridCalc_BDE.cpp`` instead of using
    :mod:`raosim.transonic_kernel`'s mathematically corrected KL evaluator.
    It is used only by the explicit ``nasa_visible_kliegel_levine`` starting
    line mode so source-faithful row-march work can be separated from the
    corrected KL kernel.
    """
    y = float(r_over_Rt)
    x = float(x_over_Rt)
    G = float(gamma)
    RS = float(Rc_over_Rt)
    z = x * math.sqrt(2.0 * RS / (G + 1.0))
    RSP = RS + 1.0
    u1 = y * y / 2.0 - 0.25 + z
    v1 = y * y * y / 4.0 - y / 4.0 + y * z
    u2 = (
        (2 * G + 9) * y ** 4 / 24.0
        - (4 * G + 15) * y * y / 24.0
        + (10 * G + 57) / 288.0
        + z * (y * y - 0.0)  # C++ ``z*(y*y - 5/8)``: INTEGER division, 5/8 == 0.
        #                      The explicit compatibility mode reproduces the
        #                      checked-in outputs_M3.5Perf with the term dropped;
        #                      transcribing it as 0.625 made this source-visible
        #                      port diverge from the overlay (TT' axis M 1.293
        #                      vs the fixture's 1.500)
        #                      and broke the RRC march that consumes the line.
        #                      Hall/Kliegel-Levine theory *does* carry y^2-5/8
        #                      (Hall 1962 u2; Zucrow & Hoffman V2 Ch.16) — the
        #                      mathematically corrected term lives in
        #                      raosim.transonic_kernel.kliegel_levine.  This
        #                      function's contract is binary fidelity, not
        #                      theory fidelity.
        - (2 * G - 3) * z * z / 6.0
    )
    v2 = (
        (G + 3) * y ** 5 / 9.0
        - (20 * G + 63) * y ** 3 / 96.0
        + (28 * G + 93) * y / 288.0
        + z * ((2 * G + 9) * y ** 3 / 6.0 - (4 * G + 15) * y / 12.0)
        + y * z * z
    )
    u3 = (
        (556 * G * G + 1737 * G + 3069) * y ** 6 / 10368.0
        - (388 * G * G + 1161 * G + 1881) * y ** 4 / 2304.0
        + (304 * G * G + 831 * G + 1242) * y * y / 1728.0
        - (2708 * G * G + 7839 * G + 14211) / 82944.0
        + z * (
            (52 * G * G + 51 * G + 327) * y ** 4 / 34.0
            - (52 * G * G + 75 * G + 279) * y * y / 192.0
            + (92 * G * G + 180 * G + 639) / 1152.0
        )
        + z * z * (-(7 * G - 3) * y * y / 8.0 + (13 * G - 27) / 48.0)
        + (4 * G * G - 57 * G + 27) * z ** 3 / 144.0
    )
    v3 = (
        (6836 * G * G + 23031 * G + 30627) * y ** 7 / 82944.0
        - (3380 * G * G + 11391 * G + 15291) * y ** 5 / 13824.0
        + (3424 * G * G + 11271 * G + 15228) * y ** 3 / 13824.0
        - (7100 * G * G + 22311 * G + 30249) * y / 82944.0
        + z
        * (
            (556 * G * G + 1737 * G + 3069) * y ** 5 / 1728.0
            * (388 * G * G + 1161 * G + 1181) * y * y / 576.0
            + (304 * G * G + 831 * G + 1242) * y / 864.0
        )
        + z * z * (
            (52 * G * G + 51 * G + 327) * y ** 3 / 192.0
            - (52 * G * G + 75 * G + 279) * y / 192.0
        )
        - z ** 3 * (7 * G - 3) * y / 12.0
    )
    U = 1.0 + u1 / RSP + (u1 + u2) / (RSP * RSP) + (u1 + 2.0 * u2 + u3) / (RSP ** 3)
    V = math.sqrt((G + 1.0) / (2.0 * RSP)) * (
        v1 / RSP
        + (1.5 * v1 + v2) / (RSP * RSP)
        + (15.0 / 8.0 * v1 + 2.5 * v2 + v3) / (RSP ** 3)
    )
    if abs(V) < 1e-5:
        V = 0.0
    theta = math.atan2(V, U)
    if abs(theta) < 1e-5:
        theta = 0.0
    return math.hypot(U, V), theta


def _make_throat_initial_line(
    Rt: float, Rd: float, theta_B: float, gamma: float, n_points: int,
    starting_line_method: str,
    M_min: float = 1.05,
) -> list[CharPoint]:
    """TT' starting line with NASA per-point downstream-step iteration.

    .. important:: ``Rd`` here is the radius of curvature the transonic
       (Kliegel-Levine / Sauer) series is evaluated with.  Physically this
       is the **upstream** wall radius of curvature: the C++ original
       passes ``rUp`` ("rUp: upstream throat radius",
       ``CalcInitialThroatLine`` line 2805) into ``KLThroat``, and the
       Hall/Kliegel-Levine expansions are derived for the convergent-side
       curvature (Hall 1962; Kliegel & Levine 1969; Zucrow & Hoffman V2
       Ch.16).  Callers that march a *downstream* arc afterwards (e.g.
       :func:`build_kernel`) must pass the upstream radius here and keep
       the downstream radius for the arc — see ``build_kernel(Ru=...)``.

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

        if starting_line_method in {
            "nasa_visible_kliegel_levine",
            "source_visible_kliegel_levine",
        }:
            mach_trial = 2.0
            x_final = x_init
            if i == 0:
                M, theta = _visible_source_kl_throat(
                    r_over_Rt, 0.0, gamma, Rd / Rt,
                )
            else:
                # Literal NASA loop: enter with mach=2 and double drdx until
                # the KL point is no longer above the arbitrary M=1.5 cap.
                x_prev, r_prev, M_prev, theta_prev = wall_first[i - 1]
                mu_prev = math.asin(1.0 / max(M_prev, 1.000001))
                drdx = math.tan(theta_prev - mu_prev)
                if abs(drdx) < 1e-14:
                    drdx = -1e-14
                M = mach_trial
                theta = 0.0
                while M > 1.5:
                    x_final = x_prev + (r - r_prev) / drdx
                    M, theta = _visible_source_kl_throat(
                        r_over_Rt, x_final / Rt, gamma, Rd / Rt,
                    )
                    drdx *= 2.0
        elif starting_line_method == "kliegel_levine":
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


def _nasa_calc_b(mach: float, theta: float, r: float) -> float:
    """Second term of the dθ RRC equation (z-form) — NASA C++ line 2986."""
    if r == 0.0:
        return 0.0
    if abs(theta) < 1e-9:
        return 0.0
    denom = r * (_nasa_mm(mach) / math.tan(theta) + 1.0)
    if abs(denom) < 1e-12:
        return 0.0
    return 1.0 / denom


def _nasa_calc_R_star(mach: float, theta: float, r: float) -> float:
    """Second term of the dθ RRC equation (r-form) — NASA C++ line 3008."""
    if r == 0.0:
        return 0.0
    if abs(theta) < 1e-9:
        return 0.0
    denom = r * (_nasa_mm(mach) - 1.0 / math.tan(theta))
    if abs(denom) < 1e-12:
        return 0.0
    return 1.0 / denom


def _nasa_l_dy_dx(theta: float, mu: float) -> float:
    """LRC slope ``tan(theta + mu)`` — NASA C++ line 3019."""
    return math.tan(theta + mu)


def _nasa_r_dy_dx(theta: float, mu: float) -> float:
    """RRC slope ``tan(theta - mu)`` — NASA C++ line 3028."""
    return math.tan(theta - mu)


def _nasa_tan_avg(x: float, y: float) -> float:
    """Tangent averaging — NASA C++ line 3037.

    ``TanAvg(x, y) = tan(0.5 · (atan(x) + atan(y)))``
    """
    return math.tan(0.5 * (math.atan(x) + math.atan(y)))


def _char_point_from_values(
    x: float,
    r: float,
    theta: float,
    mach: float,
    gamma: float,
) -> CharPoint:
    mach = float(max(mach, 1.000001))
    theta = float(theta)
    nu = prandtl_meyer(mach, gamma)
    mu = mach_angle(mach)
    return CharPoint(
        x=float(x),
        r=float(r),
        theta=theta,
        M=mach,
        nu=float(nu),
        mu=float(mu),
        compat_plus=float(theta + nu),
        compat_minus=float(theta - nu),
    )


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


def _safe_rel_err(new: float, old: float) -> float:
    if old == 0.0:
        return 9.9
    return (new - old) / old


def _calc_arc_wall_point_raw(
    prev_wall_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 50,
) -> CharPoint | None:
    """Visible-source port of NASA ``CalcArcWallPoint`` before special clamp."""
    if len(prev_wall_first) < 2:
        return None
    p1 = prev_wall_first[1]
    p_prev_wall = prev_wall_first[0]

    M1 = max(float(p1.M), 1.000001)
    theta1 = float(p1.theta)
    r1 = float(p1.r)
    x1 = float(p1.x)
    mu1 = math.asin(1.0 / M1)

    slrc1 = _nasa_l_dy_dx(theta1, mu1)
    A1 = _nasa_calc_A(M1, gamma)
    B1 = _nasa_calc_B(M1, theta1, r1)
    R1 = _nasa_calc_R(M1, theta1, r1)

    x3 = float(x1)
    r3 = float(p_prev_wall.r)
    M3 = M1
    theta3 = theta1
    slrc3 = slrc1
    A3 = A1
    B3 = B1
    R3 = R1

    x3_old = r3_old = M3_old = theta3_old = 9.9
    for _ in range(max_iter):
        slrc13 = _nasa_tan_avg(slrc1, slrc3)
        if abs(slrc13) < 1e-14:
            return None
        x3 = (r3 - r1) / slrc13 + x1
        inside = arc.Rd * arc.Rd - x3 * x3
        if inside < 0.0:
            return None
        r3 = arc.Rt + arc.Rd - math.sqrt(inside)
        theta3 = math.asin(max(min(x3 / arc.Rd, 1.0), -1.0))

        if B1 <= R1:
            T1 = (x3 - x1) * (B3 + B1)
        else:
            T1 = (r3 - r1) * (R3 + R1)
        denom = 0.5 * (A1 + A3)
        if abs(denom) < 1e-14:
            return None
        M3 = M1 + (theta3 - theta1 + 0.5 * T1) / denom
        if not math.isfinite(M3) or M3 < 1.0:
            return None

        slrc3 = _nasa_l_dy_dx(theta3, math.asin(1.0 / max(M3, 1.000001)))
        A3 = _nasa_calc_A(M3, gamma)
        B3 = _nasa_calc_B(M3, theta3, r3)
        R3 = _nasa_calc_R(M3, theta3, r3)

        x_err = _safe_rel_err(x3, x3_old)
        r_err = _safe_rel_err(r3, r3_old)
        M_err = _safe_rel_err(M3, M3_old)
        T_err = _safe_rel_err(theta3, theta3_old)
        x3_old = x3
        r3_old = r3
        M3_old = M3
        theta3_old = theta3
        if (
            abs(x_err) <= conv_tol
            and abs(r_err) <= conv_tol
            and abs(M_err) <= conv_tol
            and abs(T_err) <= conv_tol
        ):
            return _char_point_from_values(x3, r3, theta3, M3, gamma)
    return _char_point_from_values(x3, r3, theta3, M3, gamma)


def _calc_special_wall_point(
    prev_wall_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
    alpha: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 50,
) -> CharPoint | None:
    """Port NASA ``CalcSpecialWallPoint`` for small arc-angle increments."""
    if len(prev_wall_first) < 2:
        return None
    p1 = prev_wall_first[1]
    p2 = prev_wall_first[0]
    theta3 = float(alpha)
    x3 = arc.Rd * math.sin(theta3)
    r3 = arc.Rt + arc.Rd * (1.0 - math.cos(theta3))

    M1 = max(float(p1.M), 1.000001)
    M2 = max(float(p2.M), 1.000001)
    theta1 = float(p1.theta)
    theta2 = float(p2.theta)
    mu1 = math.asin(1.0 / M1)
    mu2 = math.asin(1.0 / M2)

    slrc1 = _nasa_l_dy_dx(theta1, mu1)
    slrc2 = _nasa_l_dy_dx(theta2, mu2)
    srrc1 = _nasa_r_dy_dx(theta1, mu1)
    srrc2 = _nasa_r_dy_dx(theta2, mu2)
    A1 = _nasa_calc_A(M1, gamma)
    A2 = _nasa_calc_A(M2, gamma)
    B1 = _nasa_calc_B(M1, theta1, p1.r)
    B2 = _nasa_calc_B(M2, theta2, p2.r)
    R1 = _nasa_calc_R(M1, theta1, p1.r)
    R2 = _nasa_calc_R(M2, theta2, p2.r)

    slrc3 = slrc1
    slrc4 = slrc2
    A3 = A1
    B3 = B1
    R3 = R1
    s4rrc = _nasa_tan_avg(srrc1, srrc2)
    A_err = B_err = R_err = K_err = 9.9
    M3 = M1

    for _ in range(max_iter):
        slope34 = _nasa_tan_avg(slrc3, slrc4)
        if abs(s4rrc - slope34) < 1e-14:
            x4 = x3
        elif abs(slope34) < 10000.0:
            x4 = (
                r3 - p2.r + s4rrc * p2.x - slope34 * x3
            ) / (s4rrc - slope34)
        else:
            x4 = x3
        denom = p2.x - p1.x
        if abs(denom) < 1e-14:
            ratio = 0.0
        else:
            ratio = (x4 - p1.x) / denom

        A4 = A1 + ratio * (A2 - A1)
        theta4 = theta1 + ratio * (theta2 - theta1)
        slrc4 = slrc1 + ratio * (slrc2 - slrc1)
        M4 = M1 + ratio * (M2 - M1)

        if abs(B2) <= abs(R2):
            B4 = B1 + ratio * (B2 - B1)
            T4 = (x3 - x4) * (B3 + B4)
        else:
            R4 = R1 + ratio * (R2 - R1)
            r4 = p1.r + ratio * (p2.r - p1.r)
            T4 = (r3 - r4) * (R3 + R4)

        denom_m = 0.5 * (A4 + A3)
        if abs(denom_m) < 1e-14:
            return None
        M3 = M4 + (theta3 - theta4 + 0.5 * T4) / denom_m
        if not math.isfinite(M3) or M3 < 1.0:
            return None

        K3_new = _nasa_l_dy_dx(theta3, math.asin(1.0 / max(M3, 1.000001)))
        A3_new = _nasa_calc_A(M3, gamma)
        B3_new = _nasa_calc_B(M3, theta3, r3)
        R3_new = _nasa_calc_R(M3, theta3, r3)

        K_err = _safe_rel_err(K3_new, slrc3)
        A_err = _safe_rel_err(A3_new, A3)
        B_err = _safe_rel_err(B3_new, B3)
        R_err = _safe_rel_err(R3_new, R3)
        slrc3 = K3_new
        A3 = A3_new
        B3 = B3_new
        R3 = R3_new

        if (
            abs(A_err) <= conv_tol
            and abs(B_err) <= conv_tol
            and abs(R_err) <= conv_tol
            and abs(K_err) <= conv_tol
        ):
            return _char_point_from_values(x3, r3, theta3, M3, gamma)
    return _char_point_from_values(x3, r3, theta3, M3, gamma)


def _calc_arc_wall_point_with_special(
    prev_wall_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
    dtheta_limit: float,
) -> tuple[CharPoint, bool] | None:
    raw = _calc_arc_wall_point_raw(prev_wall_first, arc, gamma)
    if raw is None:
        return None
    prev_wall = prev_wall_first[0]
    if (
        raw.theta - prev_wall.theta > dtheta_limit
        or raw.theta > arc.theta_max
    ):
        alpha = min(arc.theta_max, prev_wall.theta + 0.5 * dtheta_limit)
        special = _calc_special_wall_point(prev_wall_first, arc, gamma, alpha)
        if special is None:
            return None
        return special, True
    return raw, False


def _calc_interior_mesh_point(
    prev_wall_first: list[CharPoint],
    new_wall_first: list[CharPoint],
    i: int,
    special_flag: bool,
    gamma: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[CharPoint | None, bool]:
    """Port one NASA ``CalcInteriorMeshPoints`` point.

    Returns ``(point, negative_r)``.  ``negative_r`` mirrors NASA's
    ``r[i][j] < 0`` branch: the caller should discard the point and close
    the row with an axial mesh point.
    """
    i_last_prev = len(prev_wall_first) - 1
    ii = i if special_flag else i + 1
    if ii > i_last_prev:
        ii = i_last_prev
    p1 = prev_wall_first[ii]
    p2 = new_wall_first[i - 1]

    M1 = max(float(p1.M), 1.000001)
    M2 = max(float(p2.M), 1.000001)
    TH1 = float(p1.theta)
    TH2 = float(p2.theta)
    s1 = _nasa_l_dy_dx(TH1, math.asin(1.0 / M1))
    s2 = _nasa_r_dy_dx(TH2, math.asin(1.0 / M2))
    A1 = _nasa_calc_A(M1, gamma)
    A2 = _nasa_calc_A(M2, gamma)

    if p1.r != 0.0:
        B1 = _nasa_calc_B(M1, TH1, p1.r)
        R1 = _nasa_calc_R(M1, TH1, p1.r)
    else:
        if ii > 0:
            p1_off_axis = prev_wall_first[ii - 1]
            M1o = max(float(p1_off_axis.M), 1.000001)
            TH1o = float(p1_off_axis.theta)
            B1 = _nasa_calc_B(M1o, TH1o, p1_off_axis.r)
            R1 = _nasa_calc_R(M1o, TH1o, p1_off_axis.r)
        else:
            B1 = 0.0
            R1 = 0.0

    B2 = _nasa_calc_B(M2, TH2, p2.r)
    b2 = _nasa_calc_b(M2, TH2, p2.r)
    R2 = _nasa_calc_R(M2, TH2, p2.r)
    RS2 = _nasa_calc_R_star(M2, TH2, p2.r)

    s3lrc = s1
    s3rrc = s2
    b3 = b2
    B3 = B1
    R3 = R1
    RS3 = RS2
    A3 = 0.5 * (A1 + A2)
    M3 = 9.9
    TH3 = TH1
    x3 = r3 = 9.9
    x3_old = r3_old = m3_old = 9.9
    x_err = r_err = M_err = 9.9

    min_m_err = float("inf")
    min_state: tuple[float, float, float, float] | None = None
    for _ in range(max_iter):
        if not (abs(x_err) > conv_tol or abs(r_err) > conv_tol or abs(M_err) > conv_tol):
            break
        if M3 < 1.0:
            break

        slope13 = _nasa_tan_avg(s1, s3lrc)
        slope23 = _nasa_tan_avg(s2, s3rrc)
        if slope13 > 10000.0:
            x3 = p2.x
        elif slope23 > 10000.0:
            x3 = p1.x
        else:
            denom = slope23 - slope13
            if abs(denom) < 1e-14:
                return None, False
            x3 = (
                p1.r - p2.r - slope13 * p1.x + slope23 * p2.x
            ) / denom

        if abs(s2) <= abs(s1):
            r3 = p2.r + slope23 * (x3 - p2.x)
        else:
            r3 = p1.r + slope13 * (x3 - p1.x)

        if abs(b2) <= abs(RS2):
            T2 = (x3 - p2.x) * (b2 + b3)
        else:
            T2 = (r3 - p2.r) * (RS3 + RS2)
        if abs(B1) <= abs(R1):
            T1 = (x3 - p1.x) * (B1 + B3)
        else:
            T1 = (r3 - p1.r) * (R3 + R1)

        denom_m = A1 + A2 + 2.0 * A3
        if abs(denom_m) < 1e-14:
            return None, False
        M3 = (
            2.0 * (TH2 - TH1)
            + M2 * (A2 + A3)
            + M1 * (A1 + A3)
            + T1
            + T2
        ) / denom_m
        if not math.isfinite(M3):
            return None, False
        if M3 <= 1.0:
            break

        A3 = _nasa_calc_A(M3, gamma)
        TH3 = 0.5 * (TH1 + TH2) + 0.25 * (
            M2 * (A3 + A2)
            - M1 * (A1 + A3)
            - M3 * (A2 - A1)
            + T2
            - T1
        )
        if TH3 < 0.0:
            TH3 = 0.0

        mu3 = math.asin(1.0 / max(M3, 1.000001))
        s3lrc = _nasa_l_dy_dx(TH3, mu3)
        s3rrc = _nasa_r_dy_dx(TH3, mu3)
        B3 = _nasa_calc_B(M3, TH3, r3)
        b3 = _nasa_calc_b(M3, TH3, r3)
        R3 = _nasa_calc_R(M3, TH3, r3)
        RS3 = _nasa_calc_R_star(M3, TH3, r3)

        x_err = _safe_rel_err(x3, x3_old)
        r_dif = r3 - r3_old
        if abs(r_dif) < 1e-5 and abs(r3) < 1e-4:
            r_err = 1e-3 * r_dif
        else:
            r_err = _safe_rel_err(r3, r3_old)
        M_err = _safe_rel_err(M3, m3_old)
        if abs(M_err) < min_m_err:
            min_m_err = abs(M_err)
            min_state = (x3, r3, M3, TH3)
        x3_old = x3
        r3_old = r3
        m3_old = M3
    else:
        if min_m_err <= 5e-4 and min_state is not None:
            x3, r3, M3, TH3 = min_state
        else:
            return None, False

    if M3 < 1.0:
        if min_m_err <= 5e-4 and min_state is not None:
            x3, r3, M3, TH3 = min_state
        else:
            return None, False
    if r3 < 0.0:
        return None, True
    if p2.theta != 0.0 and TH3 < 0.0:
        TH3 = 0.0
    return _char_point_from_values(x3, r3, TH3, M3, gamma), False


def _calc_axial_mesh_point(
    new_wall_first: list[CharPoint],
    gamma: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 500,
) -> CharPoint | None:
    """Port NASA ``CalcAxialMeshPoint`` for the row-closing axis point."""
    if not new_wall_first:
        return None
    p2 = new_wall_first[-1]
    M2 = max(float(p2.M), 1.000001)
    TH2 = float(p2.theta)
    s2 = _nasa_r_dy_dx(TH2, math.asin(1.0 / M2))
    A2 = _nasa_calc_A(M2, gamma)
    b2 = _nasa_calc_b(M2, TH2, p2.r)
    s3 = s2
    A3 = A2
    x3_old = m3_old = 9.9
    M_err = x_err = 9.9
    x3 = p2.x
    M3 = M2
    for _ in range(max_iter):
        if not (abs(M_err) > conv_tol or abs(x_err) > conv_tol):
            break
        slope23 = _nasa_tan_avg(s2, s3)
        if abs(slope23) < 1e-14:
            return None
        x3 = p2.x - p2.r / slope23
        denom = A2 + A3
        if abs(denom) < 1e-14:
            return None
        M3 = M2 + 2.0 * (TH2 + b2 * (x3 - p2.x)) / denom
        if not math.isfinite(M3) or M3 < 1.0:
            return None
        s3 = _nasa_r_dy_dx(0.0, math.asin(1.0 / max(M3, 1.000001)))
        A3 = _nasa_calc_A(M3, gamma)
        M_err = _safe_rel_err(M3, m3_old)
        x_err = _safe_rel_err(x3, x3_old)
        x3_old = x3
        m3_old = M3
    return _char_point_from_values(x3, 0.0, 0.0, M3, gamma)


def _rrc_march_step(
    prev_axis_first: list[CharPoint],
    arc: ArcWall,
    gamma: float,
    dtheta_limit: float,
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
    prev_wall_first = list(reversed(prev_axis_first))
    try:
        wall_result = _calc_arc_wall_point_with_special(
            prev_wall_first, arc, gamma, dtheta_limit,
        )
        if wall_result is None:
            return None
        wall_pt, special_flag = wall_result
        if wall_pt.r < 0.0 or wall_pt.M < 1.0001:
            return None

        # NASA wall-first row assembly.  When a special wall point is
        # inserted, iLast[j] = iLast[j-1] + 1 and point-1 indexing shifts
        # from i+1 to i.
        new_wall_first: list[CharPoint] = [wall_pt]
        i_end = len(prev_wall_first) - 1 + (1 if special_flag else 0)
        for i in range(1, i_end):
            interior, negative_r = _calc_interior_mesh_point(
                prev_wall_first, new_wall_first, i, special_flag, gamma,
            )
            if negative_r:
                break
            if interior is None:
                return None
            if interior.M < 1.0001:
                return None
            new_wall_first.append(interior)

        axis_pt = _calc_axial_mesh_point(new_wall_first, gamma)
        if axis_pt is None:
            return None
        new_wall_first.append(axis_pt)
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
    dtheta_limit: float = 0.5 * math.pi / 180.0,
    Ru: float | None = None,
) -> MOCKernel:
    """
    Build the Rao kernel by NASA-style RRC marching through the throat arc.

    Port summary
    ============
    Direct port of ``CalcInitialThroatLine`` (NASA C++ line 2805) +
    ``CalcRRCsAlongArc`` (line 1030).  The row march uses the visible
    source-shaped dθ unit processes: special arc-wall insertion
    (``CalcArcWallPoint`` / ``CalcSpecialWallPoint``), wall-to-axis
    interior points (``CalcInteriorMeshPoints``), and the symmetry-axis
    closure (``CalcAxialMeshPoint``):

    1. ``_make_throat_initial_line`` lays TT' at the throat plane
       (x = 0).  ``kliegel_levine`` uses the corrected KL evaluator;
       ``nasa_visible_kliegel_levine`` / ``source_visible_kliegel_levine``
       preserve the visible C++ expression, and ``sauer_modified`` uses
       the source-compatible Sauer starting line.
    2. ``_rrc_march_step`` builds each new RRC by computing the wall
       point on the throat arc, then marching interior points from wall
       to axis with NASA's dθ source terms, terminating at the axis by
       symmetry.
    3. The marching loop stops when the new wall point reaches
       ``theta_B`` (``wall.x >= xArcMax``) or when a unit process fails
       (kernel boundary reached).
    4. NASA's mass-flow sanity check (line 1085): each new RRC's
       wall-side mass flow must match TT''s wall-side mass flow to
       within ``mdot_tol`` (default 5 %).  Violations stop the march and
       leave the kernel visibly partial instead of manufacturing a full
       NASA-reference claim.

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

    # The transonic start line is governed by the *upstream* wall radius
    # of curvature (C++ ``CalcInitialThroatLine(rUp, ...)`` -> ``KLThroat``);
    # the downstream radius only shapes the arc the kernel marches along.
    # ``Ru=None`` preserves the legacy behaviour (Ru == Rd), which is exact
    # for the NASA M3.5Perf reference (Upstream/R* = Downstream/R* = 1).
    R_start_line = float(Ru) if Ru is not None else Rd
    if R_start_line <= 0.0:
        raise ValueError("Ru must be positive")

    tt_prime = _make_throat_initial_line(
        Rt, R_start_line, theta_B, gamma, n_kernel, starting_line_method,
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
        new_axis_first = _rrc_march_step(
            prev_axis_first, arc, gamma, dtheta_limit,
        )
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

    fallback_used = False
    if len(rrcs) < 2:
        # Marching could not advance past TT'; fall back to the
        # arc-following polyline so calc_lrc_de can still run.
        fallback_used = True
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
                       Rt=float(Rt), Rd=float(Rd), gamma=float(gamma),
                       fallback_used=bool(fallback_used),
                       reached_wall=bool(reached_wall))
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

    Folded RRCs (Phase 12.4)
    ------------------------
    At sharp downstream radii (e.g. Rao's Rd = 0.382·Rt) the RRC slope
    ``tan(theta − mu)`` changes sign mid-row once the wall angle passes
    ~24°: ``theta`` crosses ``mu`` along the characteristic and the curve
    climbs in r before descending to the axis (verified non-crossing with
    its neighbours — a benign fold, not a limit line).  The C++ integrates
    straight through such segments — ``fabs(mdot_a) * fabs(da)`` with **no**
    monotonicity guard (MOC_GridCalc_BDE.cpp:3217-3228) — so the port must
    not zero them.  An earlier revision of this function skipped segments
    with ``dr <= 1e-15``, silently dropping ~13% of the row mass on the
    first folded RRC and tripping :func:`build_kernel`'s mass sanity check;
    that was the true cause of the apparent ~24.2° "march cap".  The folded
    branch below uses the algebraically identical product form
    ``0.5·pi·(r1+r2)·|rho_u_sum·dr + rho_v_sum·dx|`` (= ``fabs(mdot_a)·
    fabs(da)`` for any ``dr != 0``), which stays finite at ``dr == 0``
    where the raw C++ expression would produce ``inf · 0``.
    """
    if len(rrc) < 2:
        return np.zeros(len(rrc), dtype=float)
    massflow = np.zeros(len(rrc), dtype=float)
    # NASA's loop runs i from iLast-1 down to 0, accumulating from axis.
    for i in range(len(rrc) - 2, -1, -1):
        p_lower = rrc[i + 1]  # closer to axis
        p_upper = rrc[i]      # closer to wall
        dr = p_upper.r - p_lower.r
        u1 = p_upper.u
        u2 = p_lower.u
        v1 = p_upper.v
        v2 = p_lower.v
        rho1 = p_upper.rho
        rho2 = p_lower.rho
        if dr > 1e-15:
            # NASA C++ line 3213:
            # dxdr = (x[i+1][j] - x[i][j]) / (r[i][j] - r[i+1][j])
            # with i wall-side and i+1 axis-side.  Kept verbatim so
            # monotone rows remain bit-identical to the oracle-validated
            # behaviour.
            dxdr = (p_lower.x - p_upper.x) / dr
            # NASA Eq. 3218: rhoUAvg = 0.5(rho1 u1 + rho2 u2 + dxdr (rho1 v1 + rho2 v2))
            rho_u_avg = 0.5 * (rho1 * u1 + rho2 * u2
                               + dxdr * (rho1 * v1 + rho2 * v2))
            da = math.pi * (p_upper.r * p_upper.r - p_lower.r * p_lower.r)
            massflow[i] = massflow[i + 1] + abs(rho_u_avg) * da
        else:
            # Folded or level segment: C++ fabs(mdot_a)*fabs(da) in the
            # regularised product form (see docstring).
            ru_sum = rho1 * u1 + rho2 * u2
            rv_sum = rho1 * v1 + rho2 * v2
            dx = p_lower.x - p_upper.x
            massflow[i] = massflow[i + 1] + (
                0.5 * math.pi * (p_upper.r + p_lower.r)
                * abs(ru_sum * dr + rv_sum * dx)
            )
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

    # NASA's (commented) intent in FindPointE: accumulate DE in roughly
    # ``nRRCPlus`` equal mass increments (``dMdot = mdotMatch/nRRCPlus``,
    # C++ line ~1900).  Capping each accepted step's annular mass at
    # mdot_match/n_steps guarantees DE carries >= n_steps nodes, which the
    # downstream consumers (CE seeding, length estimate) need; the
    # uncapped path could jump D->E in one RK step on short nozzles.
    dmdot_cap = mdot_match / max(int(n_steps), 1)

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
                # Mass-increment cap: shrink the step until this segment
                # carries at most dmdot_cap.
                p_try = MOCNode(float(rk[1]), float(rk[3]),
                                float(max(rk[0], 1.000001)),
                                float(rk[2]), gamma)
                if _annular_mdot(p0, p_try) > dmdot_cap and attempts < 11:
                    h *= sub_step_factor
                    attempts += 1
                    rk = None
                    continue
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


def _calc_lrc_de_fixed_end(
    evaluate,
    bd_rrc: list[MOCNode],
    bd_flow_nodes: tuple,
    mass_along_bd: np.ndarray,
    theta_B_value: float,
    r_E: float,
    Rt: float,
    gamma: float,
    pa_over_p0: float,
) -> RaoTopology:
    """NASA ``CalcLRCDE`` FIXEDEND branch (C++ lines 1560-1610).

    Walk D inward along the actual BD grid nodes until the inner residual
    ``(r_E_found - r_E)/r_E`` changes sign from negative to positive, then
    secant between the bracketing nodes (C++ tolerance 1e-7, 50 iters).

    Fail semantics: if even D = B (zero BD mass, E = B) overshoots the
    target radius the kernel is over-expanded -> :class:`ThetaBTooHigh`
    (C++ ``SEC_FAIL_HIGH``).  If the walk exhausts BD without reaching the
    target, the kernel carries too little expansion ->
    :class:`ThetaBTooLow`.  (The C++ reuses ``SEC_FAIL_HIGH`` for the
    exhausted case via a code path shared with the RAO branch; for the
    FIXEDEND walk the physical monotonicity is the other way — r_E grows
    with D depth, so exhaustion means theta_B must *rise*.  The outer
    ``set_theta_b`` relies on these directions to bracket.)
    """
    arcs = _bd_arc_lengths(bd_rrc)
    total = float(arcs[-1])
    if total <= 1e-15:
        raise RaoKernelError("BD has zero arc length")
    node_fracs = [float(a / total) for a in arcs]

    # Pre-check at D = B (C++: param_err[0] = r[0][j] - rMatch).
    err_b = (bd_rrc[0].r - r_E) / max(abs(r_E), 1e-12)
    if err_b > 0.0:
        raise ThetaBTooHigh(
            f"wall point B (r={bd_rrc[0].r:.6g}) already exceeds the target "
            f"exit radius {r_E:.6g}; lower theta_B"
        )

    # Walk D inward along grid nodes and retain every finite sign-changing
    # interval.  ``find_point_e`` is adaptive; on coarse/folded BD rows an
    # isolated trial can take a different integration branch and produce a
    # very large, non-physical radius jump.  Accepting the first sign change
    # then brackets that discontinuity instead of the nearby smooth
    # fixed-end root.  Select the interval whose endpoints are closest to
    # zero, the discrete analogue of NASA's local secant walk.
    prev_frac = node_fracs[0]
    prev_err = err_b
    brackets: list[tuple[float, float, float, float]] = []
    for i in range(1, len(bd_rrc)):
        frac_i = node_fracs[i]
        err_i, *_ = evaluate(frac_i)
        if not math.isfinite(err_i):
            continue
        if math.isfinite(prev_err) and prev_err * err_i <= 0.0:
            brackets.append((prev_frac, prev_err, frac_i, err_i))
        prev_frac, prev_err = frac_i, err_i
    if not brackets:
        raise ThetaBTooLow(
            f"no D along BD reaches the target exit radius {r_E:.6g} "
            f"(deepest r_E error {prev_err:.3g}); raise theta_B"
        )
    bracket = min(
        brackets,
        key=lambda values: (
            max(abs(values[1]), abs(values[3])),
            abs(values[1]) + abs(values[3]),
            values[2] - values[0],
        ),
    )

    # Keep the sign-changing bracket.  The former unbracketed secant update
    # replaced the negative endpoint with the positive endpoint after the
    # first iteration; for the M3.5 reference it then returned an E radius
    # about 0.3 % short while reporting the result as the fixed-end solution.
    # A safeguarded regula-falsi/bisection is inexpensive relative to the DE
    # integration and preserves NASA's stated fixed-end contract.
    lo, err_lo, hi, err_hi = bracket
    packed_lo = evaluate(lo)
    packed_hi = evaluate(hi)
    err_lo = float(packed_lo[0])
    err_hi = float(packed_hi[0])
    last_frac = lo if abs(err_lo) <= abs(err_hi) else hi
    last_packed = packed_lo if abs(err_lo) <= abs(err_hi) else packed_hi
    for _ in range(60):
        if abs(float(last_packed[0])) <= 1e-7:
            break
        if not (math.isfinite(err_lo) and math.isfinite(err_hi)):
            break
        denom = err_hi - err_lo
        frac = (
            hi - err_hi * (hi - lo) / denom
            if abs(denom) > 1e-15 else 0.5 * (lo + hi)
        )
        # Regula falsi can pin one endpoint on a strongly curved residual;
        # force a bisection whenever it hugs either edge of the bracket.
        width = hi - lo
        if (
            not math.isfinite(frac)
            or frac <= lo + 0.05 * width
            or frac >= hi - 0.05 * width
        ):
            frac = 0.5 * (lo + hi)
        packed = evaluate(frac)
        err = float(packed[0])
        if not math.isfinite(err):
            frac = 0.5 * (lo + hi)
            packed = evaluate(frac)
            err = float(packed[0])
        if not math.isfinite(err):
            break
        if abs(err) < abs(float(last_packed[0])):
            last_frac = frac
            last_packed = packed
        if err_lo * err <= 0.0:
            hi, err_hi, packed_hi = frac, err, packed
        else:
            lo, err_lo, packed_lo = frac, err, packed
        if hi - lo <= 1e-10:
            break

    residual_final, mass_BD_final, D_final, de_nodes, mass_DE_final, _ = last_packed
    de_flow_nodes = tuple(node.to_flow_node() for node in de_nodes)
    cf = surface_thrust_coefficient(de_nodes, gamma, Rt, pa_over_p0)
    d_arc = float(last_frac)
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
        rao_stationarity_residual=float(
            residual_final if math.isfinite(residual_final) else math.nan
        ),
    )


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
    end_condition: str = "rao_free",
) -> RaoTopology:
    """
    NASA ``CalcLRCDE`` analogue (C++ line 1472).

    Secant-iterate on the location of D along the last RRC of the kernel.
    The inner residual depends on ``end_condition`` (NASA's ``nType``):

    ``"rao_free"`` (C++ ``nType == RAO``)
        Rao's free-exit stationarity: ``(theta_E - theta_calc)/theta_calc``
        with ``theta_calc`` from Rao eq. 14 evaluated at E.  Nozzle length
        is an *output*; ``x_E``/``r_E`` do not constrain the topology.
    ``"fixed_end"`` (C++ ``nType == FIXEDEND``)
        The DE endpoint must land on the prescribed exit radius:
        ``param_err = r_E_found - r_E``.  This is the branch a fixed
        (L, epsilon) design point needs; the *length* mismatch is then the
        outer ``set_theta_b`` parameter, mirroring NASA's SetThetaB /
        CalcLRCDE division of labour (C++ lines 294-470 and 1560-1610).
        Raises :class:`ThetaBTooHigh` when even D=B overshoots ``r_E``
        (C++ ``SEC_FAIL_HIGH``) and :class:`ThetaBTooLow` when no D along
        BD reaches it (the kernel carries too little expansion).

    The Mach-line geometry of DE is enforced by NASA's derivative system
    (``nasa_deriv``), and the mass closure ``mass_BD = mass_DE`` is
    enforced by ``find_point_e``.

    ``kernel`` may be a :class:`MOCKernel`, a wall-first list of
    :class:`MOCNode`, or a wall-first tuple/list of :class:`FlowNode`
    objects (legacy path).  When called with a non-kernel sequence the
    function falls back to a single-shot solve using the supplied curve
    as BD and skips the outer secant.
    """
    if end_condition not in ("rao_free", "fixed_end"):
        raise ValueError(
            f"end_condition must be 'rao_free' or 'fixed_end', "
            f"got {end_condition!r}"
        )
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
        if end_condition == "fixed_end":
            # NASA FIXEDEND: absolute exit-radius mismatch, normalised by
            # r_E so the 1e-7 tolerance below is scale-free.
            residual = (E.r - r_E) / max(abs(r_E), 1e-12)
            theta_calc = float("nan")
        else:
            theta_calc = _rao_theta_calc(E, gamma, pa_over_p0)
            if math.isnan(theta_calc) or theta_calc <= 0.0:
                residual = float("nan")
            else:
                residual = (E.theta - theta_calc) / theta_calc
        return residual, mass_BD, D, de_nodes, mass_DE, theta_calc

    if end_condition == "fixed_end":
        return _calc_lrc_de_fixed_end(
            evaluate, bd_rrc, bd_flow_nodes, mass_along_bd,
            theta_B_value, r_E, Rt, gamma, pa_over_p0,
        )

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
#  CalcBDERegion: first post-kernel BFE slice
# ----------------------------------------------------------------------


def _insert_node_on_bd(
    bd_rrc: list[MOCNode],
    D: FlowNode,
) -> tuple[list[MOCNode], int]:
    """Return the full BD row containing D exactly, plus D's index."""
    if len(bd_rrc) < 2:
        raise ValueError("BD row must contain at least two nodes")
    best_i = 1
    best_dist = float("inf")
    d_xy = np.array([D.x, D.r], dtype=float)
    for i, (p0, p1) in enumerate(zip(bd_rrc[:-1], bd_rrc[1:]), start=1):
        a = np.array([p0.x, p0.r], dtype=float)
        b = np.array([p1.x, p1.r], dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-24:
            t = 0.0
        else:
            t = float(np.clip(np.dot(d_xy - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(d_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_i = i
    tol = 1e-10
    if math.hypot(D.x - bd_rrc[best_i].x, D.r - bd_rrc[best_i].r) <= tol:
        return list(bd_rrc), best_i
    if math.hypot(D.x - bd_rrc[best_i - 1].x, D.r - bd_rrc[best_i - 1].r) <= tol:
        return list(bd_rrc), best_i - 1
    d_node = MOCNode(
        x=float(D.x), r=float(D.r), M=float(max(D.M, 1.000001)),
        theta=float(D.theta), gamma=float(bd_rrc[0].gamma),
    )
    row = list(bd_rrc[:best_i]) + [d_node] + list(bd_rrc[best_i:])
    return row, best_i


def _moc_to_char_point(node: MOCNode, gamma: float) -> CharPoint:
    """Convert a NASA-port node to the characteristic helper representation."""
    return _char_point_from_values(
        float(node.x), float(node.r), float(node.theta),
        float(max(node.M, 1.000001)), float(gamma),
    )


def _flow_to_moc_node(node: FlowNode, gamma: float) -> MOCNode:
    return MOCNode(
        float(node.x), float(node.r), float(max(node.M, 1.000001)),
        float(node.theta), float(gamma),
    )


def _calc_bde_back_point(
    previous_row: list[MOCNode],
    current_row: list[MOCNode | None],
    i: int,
    gamma: float,
    *,
    conv_tol: float = 1e-10,
    max_iter: int = 50,
) -> MOCNode | None:
    """Port one point from NASA ``CalcBDERegion`` (C++ line 3258)."""
    p1 = previous_row[i]
    p2 = current_row[i + 1]
    if p2 is None:
        return None

    M1 = max(float(p1.M), 1.000001)
    M2 = max(float(p2.M), 1.000001)
    TH1 = float(p1.theta)
    TH2 = float(p2.theta)
    s1 = _nasa_l_dy_dx(TH1, math.asin(1.0 / M1))
    s2 = _nasa_r_dy_dx(TH2, math.asin(1.0 / M2))
    A1 = _nasa_calc_A(M1, gamma)
    A2 = _nasa_calc_A(M2, gamma)
    B1 = _nasa_calc_B(M1, TH1, p1.r)
    R1 = _nasa_calc_R(M1, TH1, p1.r)
    B2 = _nasa_calc_B(M2, TH2, p2.r)
    b2 = _nasa_calc_b(M2, TH2, p2.r)
    R2 = _nasa_calc_R(M2, TH2, p2.r)
    RS2 = _nasa_calc_R_star(M2, TH2, p2.r)

    s3lrc = s1
    s3rrc = s2
    b3 = b2
    B3 = B1
    R3 = R1
    RS3 = _nasa_calc_R_star(M1, TH1, p1.r)
    A3 = 0.5 * (A1 + A2)
    x3_old = r3_old = m3_old = theta3_old = 9.9
    x_err = r_err = M_err = T_err = 9.9
    x3 = r3 = M3 = TH3 = 9.9

    for _ in range(max_iter):
        if not (
            abs(x_err) > conv_tol
            or abs(r_err) > conv_tol
            or abs(M_err) > conv_tol
            or abs(T_err) > conv_tol
        ):
            break
        slope13 = _nasa_tan_avg(s1, s3lrc)
        slope23 = _nasa_tan_avg(s2, s3rrc)
        denom = slope23 - slope13
        if abs(denom) < 1e-14:
            return None
        x3 = (
            p1.r - p2.r - slope13 * p1.x + slope23 * p2.x
        ) / denom
        if abs(s2) <= abs(s1):
            r3 = p2.r + slope23 * (x3 - p2.x)
        else:
            r3 = p1.r + slope13 * (x3 - p1.x)

        if abs(b2) <= abs(RS2):
            T2 = (x3 - p2.x) * (b2 + b3)
        else:
            T2 = (r3 - p2.r) * (RS3 + RS2)
        if abs(B1) <= abs(R1):
            T1 = (x3 - p1.x) * (B1 + B3)
        else:
            T1 = (r3 - p1.r) * (R3 + R1)

        denom_m = A1 + A2 + 2.0 * A3
        if abs(denom_m) < 1e-14:
            return None
        M3 = (
            2.0 * (TH2 - TH1)
            + M2 * (A2 + A3)
            + M1 * (A1 + A3)
            + T1
            + T2
        ) / denom_m
        if not math.isfinite(M3) or M3 < 1.0:
            return None
        A3 = _nasa_calc_A(M3, gamma)
        TH3 = 0.5 * (TH1 + TH2) + 0.25 * (
            M2 * (A3 + A2)
            - M1 * (A1 + A3)
            - M3 * (A2 - A1)
            + T2
            - T1
        )
        if TH3 != 0.0:
            TH3 = max(TH3, 0.0)

        mu3 = math.asin(1.0 / max(M3, 1.000001))
        s3lrc = _nasa_l_dy_dx(TH3, mu3)
        s3rrc = _nasa_r_dy_dx(TH3, mu3)
        B3 = _nasa_calc_B(M3, TH3, r3)
        b3 = _nasa_calc_b(M3, TH3, r3)
        R3 = _nasa_calc_R(M3, TH3, r3)
        RS3 = _nasa_calc_R_star(M3, TH3, r3)

        x_err = _safe_rel_err(x3, x3_old)
        r_err = _safe_rel_err(r3, r3_old)
        M_err = _safe_rel_err(M3, m3_old)
        T_err = _safe_rel_err(TH3, theta3_old)
        x3_old = x3
        r3_old = r3
        m3_old = M3
        theta3_old = TH3

    return MOCNode(float(x3), float(r3), float(M3), float(TH3), float(gamma))


def _calc_remaining_mesh_row(
    previous_full_row: list[MOCNode],
    current_seed_row: list[MOCNode],
    iD: int,
    gamma: float,
) -> tuple[list[MOCNode] | None, bool, bool]:
    """Port one ``CalcRemainingMesh`` row below the known DE point.

    Returns ``(row, negative_r_truncated, topology_truncated)``.
    ``negative_r_truncated`` is
    True when the interior march stopped early because the unit process
    produced a negative-radius point; NASA's source then closes the row at
    the axis and continues, so the row is still usable but shorter than the
    nominal ``iLast[j-1] + 1`` length.  Callers must not treat such rows as
    fully marched.
    """
    if len(current_seed_row) != iD + 1:
        raise ValueError("current_seed_row must contain wall through DE")
    if len(previous_full_row) <= iD:
        return None, False, False

    current = list(current_seed_row)
    prev_chars = [_moc_to_char_point(node, gamma) for node in previous_full_row]
    previous_reached_axis = abs(float(previous_full_row[-1].r)) <= 1e-12

    # NASA sets iLast[j] = iLast[j-1] + 1 and computes interior points for
    # iD+1 <= i < iLast[j].  The final index is closed by CalcAxialMeshPoint.
    truncated = False
    topology_truncated = False

    def signed_cell_area(candidate: MOCNode, index: int) -> float:
        polygon = (
            previous_full_row[index - 1],
            previous_full_row[index],
            candidate,
            current[index - 1],
        )
        return 0.5 * sum(
            float(a.x) * float(b.r) - float(a.r) * float(b.x)
            for a, b in zip(polygon, polygon[1:] + polygon[:1])
        )

    seed_areas = [
        signed_cell_area(current[index], index)
        for index in range(1, min(len(current), len(previous_full_row)))
    ]
    nonzero_seed_areas = [area for area in seed_areas if abs(area) > 1e-24]
    orientation = (
        math.copysign(1.0, float(np.median(nonzero_seed_areas)))
        if nonzero_seed_areas else 1.0
    )

    while len(current) < len(previous_full_row):
        curr_chars = [_moc_to_char_point(node, gamma) for node in current]
        point = solve_interior_point(
            curr_chars[-1], prev_chars[len(current)], gamma,
            axisymmetric=True, tol=1e-10, max_iter=50,
        )
        # ``solve_interior_point`` regularizes a negative-radius predictor to
        # r=0.  Treat that as the same near-axis event as the NASA unit
        # process's explicit negative-r return, then close from the last
        # genuinely off-axis node.
        negative_r = float(point.r) <= 1e-12
        if negative_r:
            truncated = True
            break
        index = len(current)
        area = orientation * signed_cell_area(
            MOCNode.from_char_point(point, gamma), index
        )
        prev_edge = math.hypot(
            float(previous_full_row[index].x - previous_full_row[index - 1].x),
            float(previous_full_row[index].r - previous_full_row[index - 1].r),
        )
        cross_edge = math.hypot(
            float(current[index - 1].x - previous_full_row[index - 1].x),
            float(current[index - 1].r - previous_full_row[index - 1].r),
        )
        area_tol = 1e-12 * max(prev_edge * cross_edge, 1e-24)
        if area <= area_tol:
            topology_truncated = True
            break
        current.append(MOCNode.from_char_point(point, gamma))

    # A zero/reversed cell is a characteristic caustic, not an axis point.
    # Do not draw a synthetic C- segment through it.  Once a predecessor was
    # caustic-truncated, later rows inherit that finite valid frontier rather
    # than pretending the auxiliary continuation still reaches the axis.
    if topology_truncated or not previous_reached_axis:
        return current, truncated, topology_truncated

    curr_chars = [_moc_to_char_point(node, gamma) for node in current]
    axis = solve_axis_point(
        curr_chars[-1], gamma,
        axisymmetric=True, tol=1e-10, max_iter=50,
    )
    current.append(MOCNode.from_char_point(axis, gamma))
    return current, truncated, topology_truncated


def _de_cumulative_mass(de_nodes: list[MOCNode]) -> list[float]:
    """Mass accumulated along DE, matching ``FindPointE`` conventions."""
    masses = [0.0]
    for p0, p1 in zip(de_nodes[:-1], de_nodes[1:]):
        dmdot = _annular_mdot(p0, p1)
        masses.append(masses[-1] + max(0.0, float(dmdot)))
    return masses


def _wall_contour_segment_mdot(p0: MOCNode, p1: MOCNode) -> float:
    """Signed segment mass used by NASA ``CalcWallContour``.

    ``p0`` is the lower/axis-side point and ``p1`` is the upper/wall-side
    point.  This intentionally differs from ``calc_massflow_along_rrc``:
    the C++ wall crop uses a minus sign on the transverse flux term and
    does not wrap the segment integral in ``fabs``.
    """
    dr = p1.r - p0.r
    if abs(dr) <= 1e-15:
        return 0.0
    dxdr = (p1.x - p0.x) / dr
    rho_u_avg = 0.5 * (
        p0.rho * p0.u + p1.rho * p1.u
        - dxdr * (p0.rho * p0.v + p1.rho * p1.v)
    )
    da = math.pi * (p1.r * p1.r - p0.r * p0.r)
    return float(rho_u_avg * da)


def _interp_moc_node(p0: MOCNode, p1: MOCNode, ratio: float) -> MOCNode:
    """Linear interpolation from ``p0`` to ``p1`` in physical/state space."""
    t = float(max(0.0, min(1.0, ratio)))
    gamma = p0.gamma + t * (p1.gamma - p0.gamma)
    return MOCNode(
        x=float(p0.x + t * (p1.x - p0.x)),
        r=float(p0.r + t * (p1.r - p0.r)),
        M=float(max(p0.M + t * (p1.M - p0.M), 1.000001)),
        theta=float(p0.theta + t * (p1.theta - p0.theta)),
        gamma=float(gamma),
    )


def _wall_point_on_segment(
    lower: MOCNode,
    upper: MOCNode,
    mdot0: float,
    mdot_match: float,
) -> MOCNode | None:
    """Secant/bisection wall point between lower and upper row nodes."""
    scale = max(abs(float(mdot_match)), 1e-12)

    def err_for_ratio(ratio: float) -> float:
        candidate = _interp_moc_node(lower, upper, ratio)
        return (
            mdot0 + _wall_contour_segment_mdot(lower, candidate)
            - mdot_match
        ) / scale

    err_lo = err_for_ratio(0.0)
    err_hi = err_for_ratio(1.0)
    if not (math.isfinite(err_lo) and math.isfinite(err_hi)):
        return None
    if abs(err_lo) <= 1e-10:
        return lower
    if abs(err_hi) <= 1e-10:
        return upper
    if err_lo * err_hi > 0.0:
        seg = _wall_contour_segment_mdot(lower, upper)
        if abs(seg) <= 1e-15:
            return None
        return _interp_moc_node(lower, upper, (mdot_match - mdot0) / seg)

    lo = 0.0
    hi = 1.0
    r0 = lo
    r1 = hi
    e0 = err_lo
    e1 = err_hi
    best = 0.5
    for _ in range(50):
        if e0 != e1:
            r2 = r1 - e1 * (r1 - r0) / (e1 - e0)
        else:
            r2 = 0.5 * (lo + hi)
        if not math.isfinite(r2) or r2 <= lo or r2 >= hi:
            r2 = 0.5 * (lo + hi)
        e2 = err_for_ratio(r2)
        if not math.isfinite(e2):
            return None
        best = r2
        if abs(e2) <= 1e-10 or hi - lo <= 1e-10:
            break
        if err_lo * e2 <= 0.0:
            hi = r2
            err_hi = e2
        else:
            lo = r2
            err_lo = e2
        r0, e0 = r1, e1
        r1, e1 = r2, e2
    return _interp_moc_node(lower, upper, best)


def _calc_wall_contour_rows(
    bd_full_row: list[MOCNode],
    bfe_full_rows: list[list[MOCNode]],
    de_masses: list[float],
    iD: int,
    gamma: float,
) -> tuple[list[list[MOCNode]], list[MOCNode], bool]:
    """Port NASA ``CalcWallContour`` and retain only the physical B-D-E strip.

    The mass match uses indices ``0..iD``.  Nodes beyond ``iD`` are the
    auxiliary DE-to-axis continuation and are not part of the nozzle's
    B-D-E region, so they must not leak into ``grid_rows`` or field audits.
    """
    if not bfe_full_rows:
        return [], [], False
    if len(de_masses) < len(bfe_full_rows) + 1:
        return bfe_full_rows, [row[0] for row in bfe_full_rows], False

    bd_massflow = calc_massflow_along_rrc(bd_full_row, gamma)
    if iD >= len(bd_massflow):
        return bfe_full_rows, [row[0] for row in bfe_full_rows], False
    mass_bd_grid = float(bd_massflow[0] - bd_massflow[iD])

    cropped_rows: list[list[MOCNode]] = []
    wall_nodes: list[MOCNode] = []
    complete = True
    last_post = len(bfe_full_rows) - 1

    for j, row in enumerate(bfe_full_rows):
        if len(row) <= iD:
            complete = False
            cropped_rows.append(row)
            wall_nodes.append(row[0])
            continue
        if j == last_post:
            cropped = list(row[iD:iD + 1])
            cropped_rows.append(cropped)
            wall_nodes.append(cropped[0])
            continue

        mdot_match = mass_bd_grid - float(de_masses[j + 1])
        if mdot_match <= 1e-12:
            cropped = list(row[iD:iD + 1])
            cropped_rows.append(cropped)
            wall_nodes.append(cropped[0])
            continue

        mdot = 0.0
        mdot0 = 0.0
        lower_idx = iD
        lower = row[lower_idx]
        found: tuple[int, MOCNode] | None = None
        for upper_idx in range(iD - 1, -1, -1):
            upper = row[upper_idx]
            segment_mdot = _wall_contour_segment_mdot(lower, upper)
            mdot += segment_mdot
            if mdot < mdot_match:
                lower = upper
                lower_idx = upper_idx
                mdot0 = mdot
                continue
            wall = _wall_point_on_segment(
                lower, upper, mdot0, mdot_match,
            )
            if wall is None:
                complete = False
                break
            found = (upper_idx, wall)
            break

        if found is None:
            complete = False
            cropped_rows.append(row)
            wall_nodes.append(row[0])
            continue
        upper_idx, wall = found
        cropped = [wall] + list(row[upper_idx + 1:iD + 1])
        cropped_rows.append(cropped)
        wall_nodes.append(wall)

    return cropped_rows, wall_nodes, complete


def calc_bde_region(kernel: MOCKernel, topology: RaoTopology) -> BDERegion:
    """Port NASA ``CalcBDERegion`` for the wall-to-DE post-kernel rows.

    This back-calculates the region bounded by wall-B, BD, and DE, then
    runs the first source-shaped ports of ``CalcRemainingMesh`` and
    ``CalcWallContour`` so callers can compare a BFE-family artifact
    without claiming NASA-reference parity.
    """
    bd_full_row, iD = _insert_node_on_bd(kernel.bd, topology.D)
    bd_seed_row = bd_full_row[:iD + 1]
    de_nodes = [
        MOCNode(float(node.x), float(node.r), float(max(node.M, 1.000001)),
                float(node.theta), float(kernel.gamma))
        for node in topology.DE
    ]
    if not de_nodes:
        return BDERegion(rows=(), iD=iD, complete_remaining_mesh=False)
    if math.hypot(de_nodes[0].x - topology.D.x, de_nodes[0].r - topology.D.r) > 1e-9:
        de_nodes.insert(0, MOCNode(
            float(topology.D.x), float(topology.D.r),
            float(max(topology.D.M, 1.000001)), float(topology.D.theta),
            float(kernel.gamma),
        ))

    rows: list[tuple[FlowNode, ...]] = []
    bfe_full_rows: list[list[MOCNode]] = []
    previous_seed = bd_seed_row
    previous_full = bd_full_row
    negative_r_truncations = 0
    topology_truncations = 0
    for de_node in de_nodes[1:]:
        current: list[MOCNode | None] = [None] * (iD + 1)
        current[iD] = de_node
        for i in range(iD - 1, -1, -1):
            point = _calc_bde_back_point(previous_seed, current, i, kernel.gamma)
            if point is None:
                return BDERegion(
                    rows=tuple(rows), iD=iD, complete_remaining_mesh=False,
                    negative_r_truncated_rows=negative_r_truncations,
                    topology_truncated_rows=topology_truncations,
                )
            current[i] = point
        completed = [node for node in current if node is not None]
        rows.append(tuple(node.to_flow_node() for node in completed))
        remaining, row_truncated, row_topology_truncated = _calc_remaining_mesh_row(
            previous_full, completed, iD, kernel.gamma,
        )
        if row_truncated:
            negative_r_truncations += 1
        if row_topology_truncated:
            topology_truncations += 1
        if remaining is None:
            return BDERegion(
                rows=tuple(rows), iD=iD,
                grid_rows=tuple(
                    tuple(node.to_flow_node() for node in row)
                    for row in bfe_full_rows
                ),
                full_grid_rows=tuple(
                    tuple(node.to_flow_node() for node in row)
                    for row in bfe_full_rows
                ),
                complete_remaining_mesh=False,
                negative_r_truncated_rows=negative_r_truncations,
                topology_truncated_rows=topology_truncations,
            )
        bfe_full_rows.append(remaining)
        previous_seed = completed
        previous_full = remaining

    de_masses = _de_cumulative_mass(de_nodes)
    cropped_rows, wall_nodes, wall_complete = _calc_wall_contour_rows(
        bd_full_row, bfe_full_rows, de_masses, iD, kernel.gamma,
    )

    # ``complete_remaining_mesh`` certifies the mesh ran to completion — every
    # remaining-mesh row was built and terminated on the axis (r == 0).
    # Negative-r truncation is a NORMAL feature of axisymmetric MOC converging
    # onto the singular axis (NASA CalcRemainingMesh drops such points and
    # continues); it is reported separately via ``negative_r_truncated_rows``
    # and does not by itself make the mesh incomplete.
    axis_tol = 1e-6 * max(kernel.Rt, 1e-12)
    reached_axis = bool(bfe_full_rows) and all(
        row and abs(row[-1].r) <= axis_tol for row in bfe_full_rows
    )
    return BDERegion(
        rows=tuple(rows),
        iD=iD,
        grid_rows=tuple(
            tuple(node.to_flow_node() for node in row)
            for row in cropped_rows
        ),
        full_grid_rows=tuple(
            tuple(node.to_flow_node() for node in row)
            for row in bfe_full_rows
        ),
        wall_contour=tuple(node.to_flow_node() for node in wall_nodes),
        complete_remaining_mesh=reached_axis,
        wall_contour_complete=bool(wall_complete),
        negative_r_truncated_rows=negative_r_truncations,
        topology_truncated_rows=topology_truncations,
    )


def build_source_contour_from_kernel(
    kernel: MOCKernel,
    *,
    x_E: float,
    r_E: float,
    epsilon: float,
    pa_over_p0: float = 0.0,
    n_de_points: int = 24,
    exit_rel_tol: float = 1e-3,
) -> RaoSourceContour:
    """Build the current visible-source NASA contour artifact from a kernel.

    This is the explicit source-port orchestration for the stages currently
    available in Python:

    ``build_kernel`` -> ``calc_lrc_de`` -> ``calc_bde_region`` (including
    remaining mesh and wall contour extraction).

    The returned diagnostics deliberately distinguish source-stage completion
    from final nozzle length closure.  ``CropNozzleToLength`` and the canonical
    outer ``CalcContouredNozzle``/``SetThetaB`` loop are not claimed here.
    """
    target_x = float(x_E)
    target_r = float(r_E)
    topology = calc_lrc_de(
        kernel,
        x_E=target_x,
        r_E=target_r,
        gamma=kernel.gamma,
        Rt=kernel.Rt,
        epsilon=float(epsilon),
        pa_over_p0=float(pa_over_p0),
        n_points=int(n_de_points),
        end_condition="fixed_end",
    )
    bfe = calc_bde_region(kernel, topology)
    kernel_wall = tuple(
        rrc[0].to_flow_node() for rrc in kernel.rrcs if rrc
    )
    wall = kernel_wall + tuple(bfe.wall_contour)
    wall_export = np.asarray([[node.x, node.r] for node in wall], dtype=float)

    exit_dx = float(topology.E.x - target_x)
    exit_dr = float(topology.E.r - target_r)
    x_scale = max(abs(target_x), kernel.Rt, 1e-12)
    r_scale = max(abs(target_r), kernel.Rt, 1e-12)
    exit_rel_error = max(abs(exit_dx) / x_scale, abs(exit_dr) / r_scale)
    length_closed = bool(exit_rel_error <= float(exit_rel_tol))
    physical_mesh_complete = bool(
        len(bfe.rows) == max(len(topology.DE) - 1, 0)
        and bfe.wall_contour_complete
    )
    # The source contour is only "complete" once length is also closed via
    # the (not-yet-ported) CropNozzleToLength stage.  The auxiliary post-DE
    # continuation is not part of this physical completion criterion.
    source_contour_complete = bool(
        not kernel.fallback_used
        and kernel.reached_wall
        and physical_mesh_complete
        and bfe.wall_contour_complete
        and length_closed
        and len(wall) > 0
    )
    diagnostics = {
        "canonical_reference_track": "visible_source_port",
        "stage": "kernel_fixed_end_lrc_de_bde_remaining_wall",
        "source_contour_complete": source_contour_complete,
        "kernel_fallback_used": bool(kernel.fallback_used),
        "kernel_reached_wall": bool(kernel.reached_wall),
        "kernel_rrcs": len(kernel.rrcs),
        "bfe_complete_remaining_mesh": bool(bfe.complete_remaining_mesh),
        "bfe_physical_mesh_complete": physical_mesh_complete,
        "bfe_wall_contour_complete": bool(bfe.wall_contour_complete),
        "bfe_negative_r_truncated_rows": int(bfe.negative_r_truncated_rows),
        "bfe_topology_truncated_rows": int(bfe.topology_truncated_rows),
        "bfe_grid_rows": len(bfe.grid_rows),
        "bfe_full_grid_rows": len(bfe.full_grid_rows),
        "wall_points": len(wall),
        "target_exit": {"x": target_x, "r": target_r},
        "source_exit": {"x": float(topology.E.x), "r": float(topology.E.r)},
        "exit_dx": exit_dx,
        "exit_dr": exit_dr,
        "exit_rel_error": float(exit_rel_error),
        "length_closed": length_closed,
        "exit_rel_tol": float(exit_rel_tol),
        "crop_nozzle_to_length": "not_ported",
        "outer_theta_b_driver": "fixed_kernel_input",
        "nasa_reference_matched_eligible": False,
    }
    return RaoSourceContour(
        kernel=kernel,
        topology=topology,
        bfe=bfe,
        wall=wall,
        wall_export=wall_export,
        diagnostics=diagnostics,
    )


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
    Ru: float | None = None,
    end_condition: str = "fixed_end",
) -> tuple[RaoTopology, MOCKernel]:
    """
    NASA ``SetThetaB`` outer loop on the initial wall expansion angle.

    Division of labour mirrors the C++ (MOC_GridCalc_BDE.cpp lines
    294-470): the *inner* loop (:func:`calc_lrc_de`) places D along BD so
    the DE endpoint satisfies the end condition; the *outer* loop here
    adjusts ``theta_B`` so the remaining exit parameter matches.

    With ``end_condition="fixed_end"`` (default — the fixed (L, epsilon)
    design point): inner residual is ``r_E - Re``; outer residual is
    ``(x_E - L_target)/L_target``.  :class:`ThetaBTooLow` /
    :class:`ThetaBTooHigh` from the inner loop move the bisection bracket
    directly (the C++ SEC_FAIL_LOW/HIGH handling at lines 359-430).

    With ``end_condition="rao_free"`` the legacy behaviour is kept:
    inner = Rao free-exit stationarity, outer = exit-radius mismatch.

    Returns ``(topology, kernel)``.  ``kernel.bd`` is the converged BD.
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
            Ru=Ru,
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
            end_condition=end_condition,
        )
        return topology, kernel

    def endpoint_error(t: RaoTopology) -> float:
        if end_condition == "fixed_end":
            # Outer parameter: nozzle length (inner already pinned r_E).
            return float((t.E.x - L_target) / max(L_target, 1e-12))
        return float((t.E.r - Re) / max(Re, 1e-12))

    theta_low = math.radians(5.0)
    theta_high = math.radians(44.0)
    theta_b = math.radians(theta_b_init_deg)

    best: tuple[RaoTopology, MOCKernel, float] | None = None
    err = None
    topo = kernel = None
    for _ in range(max_iter):
        try:
            topo, kernel = run(theta_b)
            err = endpoint_error(topo)
        except ThetaBTooLow:
            theta_low = max(theta_low, theta_b)
            theta_b = 0.5 * (theta_b + theta_high)
            continue
        except ThetaBTooHigh:
            theta_high = min(theta_high, theta_b)
            theta_b = 0.5 * (theta_low + theta_b)
            continue
        except RaoKernelError:
            # Kernel build/march failure — treat like "too low" (weak
            # kernels fail earliest) but keep the bracket shrinking.
            theta_low = max(theta_low, theta_b)
            theta_b = 0.5 * (theta_b + theta_high)
            continue

        if best is None or abs(err) < best[2]:
            best = (topo, kernel, abs(err))
        if abs(err) <= abs_tol:
            break
        # err > 0: DE ends beyond the target length.  A gentler initial
        # expansion (lower theta_B) makes the nozzle *longer* for the same
        # exit radius, so overlength means theta_B must RISE; underlength
        # means it must fall.  (Same monotonicity NASA's SetThetaB trace
        # shows in outputs_M3.5Perf/ThetaB.out: paramErr falls as ThetaB
        # falls toward the converged value from above.)
        if err > 0.0:
            theta_low = max(theta_low, theta_b)
            theta_b = 0.5 * (theta_b + theta_high)
        else:
            theta_high = min(theta_high, theta_b)
            theta_b = 0.5 * (theta_low + theta_b)
        if theta_high - theta_low < 1e-7:
            break

    if best is None:
        raise RaoKernelError(
            "set_theta_b: no theta_B in "
            f"[{math.degrees(theta_low):.1f}, {math.degrees(theta_high):.1f}] deg "
            "produced a valid topology"
        )
    best[1].theta_b_provenance = "fixed_end_secant"
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


def _interpolate_kernel_row_index(
    row: list[MOCNode], fractional_index: float,
) -> MOCNode:
    """Interpolate one wall-first kernel row by fractional grid index."""
    if not row:
        raise ValueError("kernel row is empty")
    q = float(np.clip(fractional_index, 0.0, len(row) - 1.0))
    if q >= len(row) - 1:
        return row[-1]
    i = int(math.floor(q))
    t = q - i
    return _interp_moc_node(row[i], row[i + 1], t)


def _project_node_to_kernel_row(
    row: list[MOCNode], node: FlowNode,
) -> tuple[float, MOCNode, float]:
    """Project ``node`` onto a kernel row.

    Returns fractional wall-first index, interpolated kernel state, and the
    Euclidean x-r projection distance.
    """
    if len(row) < 2:
        raise ValueError("kernel row must contain at least two nodes")
    target = np.asarray([node.x, node.r], dtype=float)
    best_distance = float("inf")
    best_q = 0.0
    for i, (p0, p1) in enumerate(zip(row[:-1], row[1:])):
        a = np.asarray([p0.x, p0.r], dtype=float)
        b = np.asarray([p1.x, p1.r], dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-24 else float(
            np.clip(np.dot(target - a, ab) / denom, 0.0, 1.0)
        )
        distance = float(np.linalg.norm(target - (a + t * ab)))
        if distance < best_distance:
            best_distance = distance
            best_q = float(i + t)
    return (
        best_q,
        _interpolate_kernel_row_index(row, best_q),
        best_distance,
    )


def trace_kernel_cd(
    kernel: MOCKernel,
    D: FlowNode,
) -> tuple[tuple[FlowNode, ...], dict]:
    """Recover Rao's kernel-side C-D characteristic from row connectivity.

    During ``CalcRRCsAlongArc`` an interior point ``i`` receives its C+
    (left-running in the present naming convention) parent from index ``i``
    of the previous row when a special wall point enlarged the row, otherwise
    from index ``i+1``.  Reversing that connectivity from D therefore reaches
    the symmetry-axis point C without inventing a streamline or scaling the
    DE result by mass fraction.
    """
    if not kernel.rrcs or len(kernel.rrcs[-1]) < 2:
        raise ValueError("kernel must contain a populated final BD row")
    q, projected_D, projection_distance = _project_node_to_kernel_row(
        kernel.rrcs[-1], D,
    )
    backwards: list[FlowNode] = [FlowNode(
        x=float(D.x), r=float(D.r), M=float(D.M), theta=float(D.theta),
    )]
    reached_axis = abs(float(D.r)) <= 1e-10 * max(kernel.Rt, 1.0)
    for j in range(len(kernel.rrcs) - 1, 0, -1):
        current = kernel.rrcs[j]
        previous = kernel.rrcs[j - 1]
        # A special wall insertion increases the row length by one and leaves
        # the C+ parent index unchanged.  Otherwise the parent is i+1.
        q = q if len(current) > len(previous) else q + 1.0
        q = min(q, len(previous) - 1.0)
        parent = _interpolate_kernel_row_index(previous, q)
        backwards.append(parent.to_flow_node())
        if q >= len(previous) - 1.0 - 1e-12:
            reached_axis = abs(parent.r) <= 1e-8 * max(kernel.Rt, 1.0)
            break

    cd = tuple(reversed(backwards))
    radii = np.asarray([node.r for node in cd], dtype=float)
    monotone_radius = bool(
        radii.size >= 2
        and np.all(np.diff(radii) >= -1e-9 * max(kernel.Rt, 1.0))
    )
    diagnostics = {
        "method": "nasa_kernel_row_connectivity",
        "points": len(cd),
        "reached_axis": bool(reached_axis),
        "monotone_axis_to_d_radius": monotone_radius,
        "d_projection_distance": float(projection_distance),
        "d_projection_distance_over_rt": float(
            projection_distance / max(kernel.Rt, 1e-12)
        ),
        "d_state_mach_jump": float(D.M - projected_D.M),
        "d_state_theta_jump": float(D.theta - projected_D.theta),
    }
    return cd, diagnostics


def full_control_surface_thrust(
    kernel: MOCKernel,
    de_nodes: Iterable,
    *,
    gamma: float,
    Rt: float,
    pa_over_p0: float = 0.0,
) -> FullControlSurfaceResult:
    """Integrate thrust and mass on the complete Rao C-D-E surface."""
    de = tuple(_flow_node_seq(de_nodes))
    if len(de) < 2:
        raise ValueError("DE must contain at least two nodes")
    cd, diag = trace_kernel_cd(kernel, de[0])
    cde = tuple(cd) + tuple(de[1:])
    cf_cd = surface_thrust_coefficient(cd, gamma, Rt, pa_over_p0)
    cf_de = surface_thrust_coefficient(de, gamma, Rt, pa_over_p0)
    cf_cde = surface_thrust_coefficient(cde, gamma, Rt, pa_over_p0)
    mass_cd = curve_mass_flux(cd, gamma)
    mass_de = curve_mass_flux(de, gamma)
    mass_cde = curve_mass_flux(cde, gamma)
    throat_mass = (
        float(kernel.massflow[0][0])
        if kernel.massflow and len(kernel.massflow[0]) else float("nan")
    )
    mass_residual_rel = (
        (mass_cde - throat_mass) / throat_mass
        if math.isfinite(throat_mass) and abs(throat_mass) > 1e-14
        else float("nan")
    )
    projection_rel = float(
        diag["d_projection_distance"] / max(abs(kernel.Rt), 1e-12)
    )
    complete = bool(
        diag["reached_axis"]
        and diag["monotone_axis_to_d_radius"]
        and math.isfinite(cf_cde)
        and cf_cde > 0.0
        and projection_rel <= FULL_CONTROL_D_PROJECTION_TOL_OVER_RT
        and abs(float(diag["d_state_mach_jump"])) <= FULL_CONTROL_D_MACH_JUMP_TOL
        and abs(float(diag["d_state_theta_jump"])) <= FULL_CONTROL_D_THETA_JUMP_TOL
        and math.isfinite(mass_residual_rel)
        and abs(mass_residual_rel) <= FULL_CONTROL_MASS_RESIDUAL_REL_TOL
    )
    return FullControlSurfaceResult(
        CD=tuple(cd), CDE=cde,
        cf_cd=float(cf_cd), cf_de=float(cf_de), cf_cde=float(cf_cde),
        mass_flux_cd=float(mass_cd), mass_flux_de=float(mass_de),
        mass_flux_cde=float(mass_cde),
        kernel_throat_mass_flux=float(throat_mass),
        mass_residual_rel=float(mass_residual_rel),
        d_projection_distance=float(diag["d_projection_distance"]),
        d_state_mach_jump=float(diag["d_state_mach_jump"]),
        d_state_theta_jump=float(diag["d_state_theta_jump"]),
        d_projection_tol_over_rt=FULL_CONTROL_D_PROJECTION_TOL_OVER_RT,
        d_mach_jump_tol=FULL_CONTROL_D_MACH_JUMP_TOL,
        d_theta_jump_tol=FULL_CONTROL_D_THETA_JUMP_TOL,
        mass_residual_rel_tol=FULL_CONTROL_MASS_RESIDUAL_REL_TOL,
        complete=complete,
    )


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
