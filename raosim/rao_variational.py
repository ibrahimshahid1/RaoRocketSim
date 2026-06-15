"""
rao_variational.py – experimental Rao-style variational nozzle prototype.

This module is an exploratory implementation of the Rao control-surface idea.
It follows the Rao/NASA functional form for thrust, mass flow, and length on
the supersonic control surface, including the design ambient-pressure term.
The public API remains tagged as experimental until published benchmarks pass.

The implementation currently:
  1. Defines a discretized control surface CE in the supersonic region.
  2. Evaluates the Rao thrust, mass-flow, and length integrands.
  3. Uses a constrained direct-method optimizer to maximize thrust subject to
     fixed mass flow and fixed nozzle length.
  4. Attempts a CE-driven MOC wall construction.

Known limitations: we want to overcome these, i want you to
  - The optimizer is a finite-dimensional direct method, not yet a benchmarked
    reproduction of Rao's original hand/tabular solution workflow.
  - The MOC wall may require post-processing to fit the contour API, so exact
    characteristic compatibility is not guaranteed.
  - Literature benchmarks still xfail; use the Bezier Rao/TOP path for trusted
    preliminary geometry.

References:
  - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA TM (1990), Rao method re-derivation with explicit functionals
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from enum import Enum
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    least_squares = None

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    isentropic_density_ratio,
    isentropic_temperature_ratio,
    mach_from_area_ratio,
    mach_angle,
    mstar_from_M,
    prandtl_meyer,
    mach_from_prandtl_meyer,
    area_mach_relation,
    thrust_coefficient as ideal_thrust_coefficient,
)
from raosim.moc import (
    _make_point,
    CharPoint,
    CharRow,
    FlowNode,
    solve_interior_point,
    solve_axis_point,
    solve_wall_point,
    approximate_starting_line,
    march_coupled_net,
    sample_exit_plane,
    compute_exit_thrust,
)
from raosim.rao_residuals import (
    residual_Cminus_axisym,
    residual_Cplus_axisym,
    residual_cplus_child_position,
    residual_intersection,
    residual_left_mach_geometry,
    residual_wall_tangency,
)
from raosim.nasa_moc import (
    calc_lrc_de,
    calc_mdot_bd,
    surface_thrust_coefficient,
)
from raosim.validation import add_contour_reliability_metadata
from raosim.wall_model import SplineWall


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ControlSurface:
    """
    Discretized control surface CE.

    Each station i defines the flow state at radius r_i:
    M[i]     : Mach number
    theta[i] : flow angle from axis [rad]
    phi[i]   : control-surface inclination from axis [rad]
    r[i]     : radial position [m]
    x[i]     : axial position [m] for BVP-backed CE geometry
    """
    r: np.ndarray
    M: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    x: np.ndarray | None = None
    lambda2: float = 0.0    # Lagrange multiplier for mass-flow constraint
    lambda3: float = 0.0    # Lagrange multiplier for length constraint
    log_C: float = 0.0      # log of the Rao algebraic stationarity constant
    kernel_d_fraction: float = 1.0  # solved point D location along kernel BD
    converged: bool = False
    residual_norm: float | None = None
    mdot_target: float | None = None
    mdot_residual: float | None = None
    length_target: float | None = None
    length_residual: float | None = None
    transversality: float | None = None
    thrust: float | None = None
    objective: float | None = None
    solver_message: str | None = None
    optimizer_success: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class WallSurface:
    """
    Wall polyline carried as a first-class BVP unknown (Phase 6).

    Parallel to :class:`ControlSurface`: ``x[0]`` is wall point N (end of
    the throat arc), ``x[-1]`` is wall point E (exit).  All state arrays
    are sized ``n_wall``.  ``theta[i]`` is the local flow angle at the
    wall node; at convergence this equals the wall slope via the
    streamline / wall-tangency residual block.
    """

    x: np.ndarray
    r: np.ndarray
    M: np.ndarray
    theta: np.ndarray


class ContourReliability(str, Enum):
    """Explicit maturity levels for generated contour data."""

    GEOMETRIC_APPROXIMATION = "geometric_approximation"
    MOC_COMPATIBLE = "moc_compatible"
    RAO_VARIATIONAL_RESIDUAL_SOLVED = "rao_variational_residual_solved"
    BENCHMARK_VALIDATED = "benchmark_validated"
    CFD_CHECKED = "cfd_checked"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"


class RaoInvalidRegionError(RuntimeError):
    """Raised when requested inputs are outside the smooth-flow Rao region."""


class RaoEndpointMismatchError(RuntimeError):
    """Raised when raw Rao/MOC wall endpoints do not close to target geometry."""


RAO_MOC_ASSUMPTIONS = (
    "axisymmetric",
    "inviscid",
    "isentropic",
    "constant_gamma",
)


def summarize_group(name: str, arr: np.ndarray) -> dict:
    """Return compact max/RMS diagnostics for one residual block."""
    values = np.asarray(arr, dtype=float)
    return {
        "name": name,
        "count": int(values.size),
        "max": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(values**2))) if values.size else 0.0,
    }


@dataclass
class RaoSolverConfig:
    """Configuration for the finite-dimensional Rao variational/MOC solve."""

    Rt: float
    epsilon: float
    gamma: float = 1.4
    pa_over_p0: float = 0.0
    length_pct: float = 80.0
    throat_downstream_radius_factor: float = 0.382
    thetaN_guess_deg: float = 30.0
    n_control: int = 12
    n_kernel: int = 12
    max_nfev: int = 200
    residual_tol: float = 2e-3
    starting_line_method: str = "kliegel_levine"
    evaluate_moc: bool = True
    residual_blocks: tuple[str, ...] | None = None
    wall_method: str = "coupled"
    kernel_bd: tuple[FlowNode, ...] | None = field(default=None, repr=False)
    # Phase 6: when ``couple_wall`` is True the wall (x, r, M, theta) is
    # added to the BVP unknown vector and solved jointly with the CE.
    # New residual blocks ``wall_endpoint``, ``wall_tangency``,
    # ``cplus_ce_to_wall``, ``wall_intersection`` become active.  Default
    # is False so legacy callers stay on the postprocessing wall path.
    couple_wall: bool = False
    n_wall: int = 12
    # Phase 7 prereq (REWRITE_PLAN §13.1): controls whether θ_N / θ_E
    # are seeded from the chart only or actively pinned in the residual
    # stack.  Required for an uncontaminated chart benchmark.
    #
    #   "free"        — chart values seed the initial guess but never
    #                   appear in the residual stack.  Use for Phase 7
    #                   chart benchmarking.  (default)
    #   "chart_soft"  — adds a small (weight=1e-3) chart-anchor
    #                   residual to nudge convergence for debugging.
    #   "chart_hard"  — pins θ_N / θ_E to chart values (weight=1.0);
    #                   emits a ``DeprecationWarning``.  The legacy
    #                   behaviour before Phase 4.
    angle_boundary_mode: str = "free"


DEFAULT_RAO_RESIDUAL_BLOCKS = (
    "mass",
    "length",
    "moc_cplus",
    "moc_cminus",
    "ce_geometry",
    "regularization",
    "penalties",
    "algebraic_stationarity",
    # NOTE: ``left_mach`` is intentionally omitted from the default
    # block set as of the "left-Mach-by-construction" refactor.  CE
    # ``x_ce`` is reconstructed from ``dr/dx = tan(theta+mu)`` inside
    # ``_unpack_bvp`` (see :func:`_integrate_ce_x_from_left_mach`),
    # making the left-Mach geometry residual identically zero by
    # construction.  The block is kept as an ALL_ option so
    # diagnostics can still report ``left_mach_rms`` (which should be
    # ~1e-12 on a converged solve — this is the exactness gate
    # enforced by ``test_left_mach_geometry_is_exact_after_refactor``).
    # Phase 6 wall blocks: included by default but no-ops unless the
    # solver is configured with couple_wall=True (and therefore actually
    # populates the wall portion of the unknown vector).
    "wall_endpoint",
    "wall_tangency",
    "cplus_ce_to_wall",
    "wall_intersection",
)

# Unified weight applied to the three derivative-form Rao physics
# residual blocks (algebraic_stationarity, moc_cplus, moc_cminus).
# left_mach uses unit weight because it is a geometric residual already
# normalised by segment chord length.  Empirical ramp results
# (tests/test_phase6_coupled_wall.py::test_physics_weight_ramp_...):
#   * weight = 0.02 → baseline; mass residual ~6e-3
#   * weight = 0.05 → default; mass residual ~2e-2, physics ~3e-2 raw
#   * weight = 0.25 → mass residual ~4e-1 (loose), physics tighter
#   * weight = 0.50 → next target; expected feasible per ramp trend
#   * weight = 1.00 → currently xfailed
#     (test_solve_rao_bvp_reaches_rao_residual_solved_at_weight_1)
# Closing the 1.0 xfail unlocks RAO_VARIATIONAL_RESIDUAL_SOLVED
# reliability.  Likely culprits, in order of likelihood: linear
# CE↔wall pairing in the Phase 6 coupled-wall builder (try free
# pair_fraction unknowns), under-resolved wall (bump n_wall to 20),
# or the Bezier wall seed not yet wired in for couple_wall=True.
PHYSICS_WEIGHT = 0.05

ALL_RAO_RESIDUAL_BLOCKS = DEFAULT_RAO_RESIDUAL_BLOCKS + (
    "stationarity",        # numerical Euler-Lagrange (reference; not in default)
    "transversality",      # natural at free endpoint only; off for fixed L, eps
    "left_mach",           # diagnostic: should be ~1e-12 after the refactor
)


RAO_RESIDUAL_ABLATIONS = {
    "mass_length": ("mass", "length"),
    "mass_length_stationarity": ("mass", "length", "stationarity"),
    "mass_length_moc": ("mass", "length", "moc_cplus", "moc_cminus"),
    "mass_length_moc_stationarity": (
        "mass", "length", "moc_cplus", "moc_cminus", "stationarity"
    ),
    "mass_length_geometry": ("mass", "length", "ce_geometry"),
    "mass_length_geometry_cminus": (
        "mass", "length", "ce_geometry", "moc_cminus"
    ),
    "default_plus_stationarity": DEFAULT_RAO_RESIDUAL_BLOCKS + ("stationarity",),
    "default_plus_transversality": DEFAULT_RAO_RESIDUAL_BLOCKS + ("transversality",),
    "all": ALL_RAO_RESIDUAL_BLOCKS,
}


@dataclass
class RaoResidualReport:
    """Scaled and dimensional residual summary for an attempted Rao solve."""

    max_scaled: float
    rms_scaled: float
    mass_residual_rel: float
    length_residual_rel: float
    stationarity_rms: float
    algebraic_stationarity_rms: float
    left_mach_rms: float
    regularization_rms: float
    transversality_scaled: float
    wall_tangency_rms: float | None = None
    characteristic_crossings: int = 0
    group_summaries: list[dict] = field(default_factory=list)


@dataclass
class MOCNetCompatibilityReport:
    """Detailed forward-MOC audit against a candidate wall."""

    cplus_rms: float
    cminus_rms: float
    cplus_max: float
    cminus_max: float
    intersection_rms: float
    wall_boundary_rms: float
    wall_tangency_rms: float
    crossings: int
    bad_rows: list[int]
    passes: bool
    wall_boundary_dx_rms: float = 0.0
    wall_boundary_dr_rms: float = 0.0
    wall_boundary_dx_max: float = 0.0
    wall_boundary_dr_max: float = 0.0
    intersection_max: float = 0.0
    wall_tangency_max: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cplus_rms": self.cplus_rms,
            "cminus_rms": self.cminus_rms,
            "cplus_max": self.cplus_max,
            "cminus_max": self.cminus_max,
            "intersection_rms": self.intersection_rms,
            "intersection_max": self.intersection_max,
            "wall_boundary_rms": self.wall_boundary_rms,
            "wall_boundary_dx_rms": self.wall_boundary_dx_rms,
            "wall_boundary_dr_rms": self.wall_boundary_dr_rms,
            "wall_boundary_dx_max": self.wall_boundary_dx_max,
            "wall_boundary_dr_max": self.wall_boundary_dr_max,
            "wall_tangency_rms": self.wall_tangency_rms,
            "wall_tangency_max": self.wall_tangency_max,
            "crossings": self.crossings,
            "bad_rows": list(self.bad_rows),
            "passes": self.passes,
        }


@dataclass(frozen=True)
class MOCNetLink:
    """One explicit parent-child edge in a marched MOC net."""

    row: int
    family: str
    role: str
    parent: CharPoint
    child: CharPoint
    parent_index: int
    child_index: int


@dataclass
class RaoResidualGroups:
    """Named residual blocks for ablation and debugging."""

    mass: np.ndarray
    length: np.ndarray
    transversality: np.ndarray
    stationarity: np.ndarray
    algebraic_stationarity: np.ndarray
    left_mach: np.ndarray
    moc_cplus: np.ndarray
    moc_cminus: np.ndarray
    ce_geometry: np.ndarray
    regularization: np.ndarray
    penalties: np.ndarray
    # Phase 6 coupled-wall blocks (empty arrays when couple_wall=False).
    wall_endpoint: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_tangency: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cplus_ce_to_wall: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_intersection: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.mass,
            self.length,
            self.transversality,
            self.stationarity,
            self.algebraic_stationarity,
            self.left_mach,
            self.moc_cplus,
            self.moc_cminus,
            self.ce_geometry,
            self.regularization,
            self.penalties,
            self.wall_endpoint,
            self.wall_tangency,
            self.cplus_ce_to_wall,
            self.wall_intersection,
        ])

    def summaries(self) -> list[dict]:
        return [
            summarize_group("mass", self.mass),
            summarize_group("length", self.length),
            summarize_group("transversality", self.transversality),
            summarize_group("stationarity", self.stationarity),
            summarize_group("algebraic_stationarity", self.algebraic_stationarity),
            summarize_group("left_mach", self.left_mach),
            summarize_group("moc_cplus", self.moc_cplus),
            summarize_group("moc_cminus", self.moc_cminus),
            summarize_group("ce_geometry", self.ce_geometry),
            summarize_group("regularization", self.regularization),
            summarize_group("penalties", self.penalties),
            summarize_group("wall_endpoint", self.wall_endpoint),
            summarize_group("wall_tangency", self.wall_tangency),
            summarize_group("cplus_ce_to_wall", self.cplus_ce_to_wall),
            summarize_group("wall_intersection", self.wall_intersection),
        ]


@dataclass
class RaoRawSolution:
    """Unresampled solver data used for diagnostics and validation."""

    wall_points: np.ndarray
    control_surface: ControlSurface
    characteristic_net: list[CharRow] = field(default_factory=list)
    kernel_points: list[CharPoint] = field(default_factory=list)
    construction_diagnostics: dict = field(default_factory=dict)


@dataclass
class RaoSolution:
    """Auditable result object for the Rao variational/MOC path."""

    wall_raw: np.ndarray
    wall_export: np.ndarray
    control_surface: ControlSurface
    characteristic_net: list[CharRow]
    kernel_points: list[CharPoint]
    theta_N: float
    theta_E: float
    thrust_coefficient: float
    residuals: RaoResidualReport
    reliability: ContourReliability
    converged: bool
    shock_free: bool
    hardware_qualified: bool
    assumptions: tuple[str, ...]
    construction_diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    topology: object | None = None  # nasa_moc.RaoTopology when available

    def to_contour_dict(self, *, Rt: float, epsilon: float, length_pct: float,
                        pa_over_p0: float, Ru_factor: float = 1.5,
                        convergent_half_angle_deg: float = 45.0) -> dict:
        """Return compatibility dict shaped like ``bell_nozzle_contour``."""
        Re = math.sqrt(epsilon) * Rt
        Ru = Ru_factor * Rt
        Rd = 0.382 * Rt
        wall_x = self.wall_export[:, 0]
        wall_r = self.wall_export[:, 1]

        conv_angle = math.radians(convergent_half_angle_deg)
        n_conv = 100
        t_conv = np.linspace(-(math.pi / 2 + conv_angle), -math.pi / 2, n_conv)
        x_conv = Ru * np.cos(t_conv)
        y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

        theta_n = self.theta_N
        t_thr = np.linspace(-math.pi / 2, theta_n - math.pi / 2, n_conv)
        x_throat = Rd * np.cos(t_thr)
        y_throat = (Rt + Rd) + Rd * np.sin(t_thr)

        x_full = np.concatenate([x_conv, x_throat, wall_x])
        y_full = np.concatenate([y_conv, y_throat, wall_r])
        Nx = float(x_throat[-1])
        Ny = float(y_throat[-1])

        contour = {
            "x": x_full,
            "y": y_full,
            "theta_n": math.degrees(theta_n),
            "theta_e": math.degrees(self.theta_E),
            "Ln": float(wall_x[-1]),
            "Re": Re,
            "Rt": Rt,
            "Ru": Ru,
            "Rd": Rd,
            "epsilon": epsilon,
            "pa_over_p0": pa_over_p0,
            "length_pct": length_pct,
            "N": (Nx, Ny),
            "E": (float(wall_x[-1]), float(wall_r[-1])),
            "P1": (0.5 * (Nx + float(wall_x[-1])), 0.5 * (Ny + float(wall_r[-1]))),
            "x_conv": x_conv,
            "y_conv": y_conv,
            "x_throat": x_throat,
            "y_throat": y_throat,
            "x_bell": wall_x,
            "y_bell": wall_r,
            "method": "rao_variational_moc",
            "contour_type": "rao_variational_moc",
            "variational_status": self.reliability.value,
            "reliability": self.reliability.value,
            "assumptions": list(self.assumptions),
            "hardware_qualified": self.hardware_qualified,
            "rao_full_optimum_claimed": self.reliability in {
                ContourReliability.BENCHMARK_VALIDATED,
                ContourReliability.CFD_CHECKED,
                ContourReliability.EXPERIMENTALLY_VALIDATED,
            },
            "optimization_converged": self.converged,
            "shock_free": self.shock_free,
            "residuals": self.residuals.__dict__,
            "control_surface": self.control_surface,
            "characteristic_net": self.characteristic_net,
            "kernel_points": self.kernel_points,
            "raw_wall_points": self.wall_raw,
            "construction_diagnostics": self.construction_diagnostics,
            "wall_export_resampled": True,
            "thrust_coefficient": self.thrust_coefficient,
            "warnings": list(self.warnings),
        }
        return contour


def _design_angles_rad(epsilon: float, length_pct: float,
                       fallback_theta_n_deg: float = 30.0) -> tuple[float, float]:
    """Return wall design throat/exit angles independently of CE state."""
    try:
        from raosim.nozzle_geometry import lookup_angles

        theta_n_deg, theta_e_deg = lookup_angles(epsilon, length_pct)
    except Exception:
        theta_n_deg = fallback_theta_n_deg
        theta_e_deg = max(0.0, 0.5 * fallback_theta_n_deg)
    return math.radians(theta_n_deg), math.radians(theta_e_deg)


# ─────────────────────────────────────────────────────────────────────
#  Rao functional integrands  (NASA TM 1990 formulation)
# ─────────────────────────────────────────────────────────────────────

def _isentropic_V(M: float, gamma: float) -> float:
    """
    Non-dimensional velocity V/V_max as a function of Mach number.
    V/V_max = M · sqrt(T/T₀) / M_max_equivalent

    For the Rao functional we use the normalized velocity:
        V̄ = M · a / a₀ = M · sqrt(T/T₀)
    where T/T₀ = (1 + (γ-1)/2·M²)^(-1)
    """
    T_ratio = isentropic_temperature_ratio(M, gamma)
    return M * math.sqrt(T_ratio)


def _isentropic_rhoV(M: float, gamma: float) -> float:
    """ρ·V (non-dimensional) = ρ/ρ₀ · V̄."""
    rho_ratio = isentropic_density_ratio(M, gamma)
    V = _isentropic_V(M, gamma)
    return rho_ratio * V


def thrust_integrand(M: float, theta: float, phi: float,
                     r: float, gamma: float,
                     pa_over_p0: float = 0.0) -> float:
    """
    Thrust contribution per unit radial extent on the control surface.

    In the Rao/NASA control-surface formulation:

        dT/dr = 2πr · [
            (p - pa) + ρV² sin(φ - θ) cos(θ) / sin(φ)
        ]

    This implementation uses stagnation-normalized isentropic properties, so
    the pressure term is ``p/p0 - pa/p0`` and the momentum term uses
    ``ρ/ρ0 · γM²T/T0``.  The ambient subtraction is deliberately unprojected:
    Rao's pressure thrust is the local axial pressure difference.
    """
    if abs(math.sin(phi)) < 1e-15:
        return 0.0

    p_ratio = isentropic_pressure_ratio(M, gamma)
    rho_ratio = isentropic_density_ratio(M, gamma)
    T_ratio = isentropic_temperature_ratio(M, gamma)

    # Non-dimensional velocity squared: V̄² = γ·M²·(T/T₀)
    V_sq = gamma * M * M * T_ratio

    sin_diff = math.sin(phi - theta)
    sin_phi = math.sin(phi)

    # Momentum flux through the oblique control surface plus pressure thrust.
    momentum = rho_ratio * V_sq * math.cos(theta) * sin_diff / sin_phi
    pressure = p_ratio - pa_over_p0

    return 2.0 * math.pi * r * (momentum + pressure)


def massflow_integrand(M: float, theta: float, phi: float,
                       r: float, gamma: float,
                       pa_over_p0: float = 0.0) -> float:
    """
    Mass-flow contribution per unit radial extent on CE.

        f₂ = 2πr · ρV · sin(φ-θ) / sin(φ)

    In non-dimensional form:
        f₂ = 2πr · (ρ/ρ₀) · V̄ · sin(φ-θ) / sin(φ)
    """
    _ = pa_over_p0
    if abs(math.sin(phi)) < 1e-15:
        return 0.0
    rhoV = _isentropic_rhoV(M, gamma)
    sin_diff = math.sin(phi - theta)
    sin_phi = math.sin(phi)
    return 2.0 * math.pi * r * rhoV * sin_diff / sin_phi


def length_integrand(phi: float) -> float:
    """
    Length constraint integrand.

        f₃ = cot(φ) = cos(φ) / sin(φ)

    The nozzle length is: L = z_C + ∫ cot(φ) dr  over CE
    """
    if abs(math.sin(phi)) < 1e-15:
        return 0.0
    return math.cos(phi) / math.sin(phi)


# ─────────────────────────────────────────────────────────────────────
#  Partial derivatives for the Euler-Lagrange stationarity conditions
# ─────────────────────────────────────────────────────────────────────

def _numerical_partials(f, M, theta, phi, r, gamma,
                        pa_over_p0=0.0, dh=1e-6):
    """
    Compute ∂f/∂M, ∂f/∂θ, ∂f/∂φ by central differences.

    Used for the stationarity conditions:
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂M = 0
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂θ = 0
        ∂(f₁ + λ₂f₂ + λ₃f₃)/∂φ = 0
    """
    f0 = f(M, theta, phi, r, gamma, pa_over_p0)

    # ∂f/∂M
    M_lo = max(M - dh, 1.0 + 1e-8)
    M_hi = M + dh
    df_dM = (f(M_hi, theta, phi, r, gamma, pa_over_p0) -
             f(M_lo, theta, phi, r, gamma, pa_over_p0)) / (M_hi - M_lo)

    # ∂f/∂θ
    df_dtheta = (f(M, theta + dh, phi, r, gamma, pa_over_p0) -
                 f(M, theta - dh, phi, r, gamma, pa_over_p0)) / (2.0 * dh)

    # ∂f/∂φ
    phi_lo = max(phi - dh, 1e-8)
    phi_hi = min(phi + dh, math.pi - 1e-8)
    df_dphi = (f(M, theta, phi_hi, r, gamma, pa_over_p0) -
               f(M, theta, phi_lo, r, gamma, pa_over_p0)) / (phi_hi - phi_lo)

    return df_dM, df_dtheta, df_dphi


def stationarity_residuals(M: float, theta: float, phi: float,
                           r: float, gamma: float,
                           lambda2: float, lambda3: float,
                           pa_over_p0: float = 0.0) -> np.ndarray:
    """
    Numerical Euler-Lagrange residuals at a single CE station.

    .. note::
       Reference implementation -- not in the default residual stack.
       Phase 3 of the Rao rewrite replaced this finite-difference Euler-Lagrange
       form with the closed-form Rao algebraic stationarity
       (``rao_stationarity_residual``).  Kept here for ablation studies and
       as a cross-check.

    Computes:

        R1 = df1/dM  + lambda2 df2/dM
        R2 = df1/dt  + lambda2 df2/dt
        R3 = df1/dp  + lambda2 df2/dp  + lambda3 df3/dp

    Note: f3 = cot(phi) depends only on phi, so df3/dM = df3/dtheta = 0.
    """
    df1 = _numerical_partials(
        thrust_integrand, M, theta, phi, r, gamma, pa_over_p0
    )
    df2 = _numerical_partials(
        massflow_integrand, M, theta, phi, r, gamma, pa_over_p0
    )

    # ∂f₃/∂φ = -1/sin²(φ)
    if abs(math.sin(phi)) > 1e-15:
        df3_dphi = -1.0 / (math.sin(phi) ** 2)
    else:
        df3_dphi = -1e15

    R = np.array([
        df1[0] + lambda2 * df2[0],              # ∂/∂M
        df1[1] + lambda2 * df2[1],              # ∂/∂θ
        df1[2] + lambda2 * df2[2] + lambda3 * df3_dphi,  # ∂/∂φ
    ])
    return R


def transversality_residual(M: float, theta: float, phi: float,
                            r: float, gamma: float,
                            lambda2: float, lambda3: float,
                            pa_over_p0: float = 0.0) -> float:
    """
    Endpoint (transversality) condition for free exit radius:

        (f₁ + λ₂·f₂ + λ₃·f₃)|_E = 0
    """
    f1 = thrust_integrand(M, theta, phi, r, gamma, pa_over_p0)
    f2 = massflow_integrand(M, theta, phi, r, gamma, pa_over_p0)
    f3 = length_integrand(phi)
    return f1 + lambda2 * f2 + lambda3 * f3


# ─────────────────────────────────────────────────────────────────────
#  Control-surface solver
# ─────────────────────────────────────────────────────────────────────

def _initial_ce_guess(Rt: float, Re: float, Ln: float,
                      gamma: float, n_pts: int) -> ControlSurface:
    """
    Initial guess for the control surface distribution.

    Uses quasi-1D estimates:
      - θ(r) linearly decreasing from ~θ_n to ~0 (parallel exit)
      - M(r) from area ratio at each radial station
      - φ(r) ≈ angle of a line connecting throat to exit
    """
    At = math.pi * Rt * Rt

    # The CE used by this prototype spans the wall-side supersonic control
    # surface from just downstream of the throat toward the exit wall.
    r = np.linspace(1.05 * Rt, Re, n_pts)
    M = np.zeros(n_pts)
    theta = np.zeros(n_pts)
    phi = np.zeros(n_pts)

    for i in range(n_pts):
        # Use a smooth supersonic initial distribution from just above sonic
        # at the axis toward the design exit Mach at the wall.  This is still
        # a heuristic seed; the returned contour remains experimental.
        frac = i / max(n_pts - 1, 1)
        ar = 1.0 + (math.pi * Re ** 2 / At - 1.0) * max(frac, 1e-6)
        try:
            M[i] = mach_from_area_ratio(ar, gamma, supersonic=True)
        except ValueError:
            M[i] = 1.5
        # θ linearly decreasing from ~20° to ~0°
        frac = i / max(n_pts - 1, 1)
        theta[i] = math.radians(20.0 * (1.0 - frac))
        # φ initial guess: ~60° (typical CE inclination)
        phi[i] = math.radians(60.0 + 20.0 * frac)

    return ControlSurface(r=r, M=M, theta=theta, phi=phi)


def _integrate_ce(
    ce: ControlSurface,
    gamma: float,
    pa_over_p0: float = 0.0,
) -> tuple[float, float, float]:
    """Integrate thrust (F), mass flow (ṁ), and length (L) over the CE."""
    if ce.x is not None:
        return _integrate_ce_curve(ce, gamma, pa_over_p0)

    n = len(ce.r)
    F_total = 0.0
    mdot_total = 0.0
    L_total = 0.0

    for i in range(n - 1):
        dr = ce.r[i + 1] - ce.r[i]
        if dr <= 0:
            continue

        # Midpoint values
        r_mid = 0.5 * (ce.r[i] + ce.r[i + 1])
        M_mid = 0.5 * (ce.M[i] + ce.M[i + 1])
        theta_mid = 0.5 * (ce.theta[i] + ce.theta[i + 1])
        phi_mid = 0.5 * (ce.phi[i] + ce.phi[i + 1])

        if M_mid < 1.001:
            M_mid = 1.001

        F_total += thrust_integrand(M_mid, theta_mid, phi_mid,
                                    r_mid, gamma, pa_over_p0) * dr
        mdot_total += massflow_integrand(M_mid, theta_mid, phi_mid,
                                         r_mid, gamma) * dr
        L_total += length_integrand(phi_mid) * dr

    return F_total, mdot_total, L_total


def _integrate_ce_curve(
    ce: ControlSurface,
    gamma: float,
    pa_over_p0: float = 0.0,
) -> tuple[float, float, float]:
    """Integrate CE quantities using solved physical (x, r) geometry."""
    if ce.x is None:
        return _integrate_ce(ce, gamma, pa_over_p0)

    F_total = 0.0
    mdot_total = 0.0
    L_total = 0.0
    for i in range(len(ce.r) - 1):
        dx = float(ce.x[i + 1] - ce.x[i])
        dr = float(ce.r[i + 1] - ce.r[i])
        ds = math.hypot(dx, dr)
        if ds < 1e-12:
            continue
        beta = math.atan2(dr, dx)
        r_mid = 0.5 * (float(ce.r[i]) + float(ce.r[i + 1]))
        M_mid = max(0.5 * (float(ce.M[i]) + float(ce.M[i + 1])), 1.001)
        theta_mid = 0.5 * (float(ce.theta[i]) + float(ce.theta[i + 1]))

        # The legacy integrands are written per dr. Multiplying by dr with
        # phi=beta is equivalent to integrating over ds for monotone CE
        # segments because dr/sin(beta) = ds.
        F_total += thrust_integrand(
            M_mid, theta_mid, beta, r_mid, gamma, pa_over_p0
        ) * dr
        mdot_total += massflow_integrand(
            M_mid, theta_mid, beta, r_mid, gamma
        ) * dr
        L_total += dx

    return F_total, mdot_total, L_total


def solve_optimal_control_surface(
    Rt: float,
    epsilon: float,
    gamma: float,
    length_pct: float = 80.0,
    n_ce_pts: int = 25,
    max_outer_iter: int = 80,
    max_inner_iter: int = 15,
    tol: float = 1e-6,
    pa_over_p0: float = 0.0,
) -> ControlSurface:
    """
    Find the optimal control-surface distribution (M, θ, φ) that
    maximizes thrust subject to mass-flow and length constraints.

    Strategy:
      1. Start with a quasi-1D initial guess for CE
      2. Iterate:
         a. Fix λ₂, λ₃ → sweep CE stations enforcing stationarity
         b. Update λ₂, λ₃ from constraint residuals (mass flow, length)
      3. Enforce transversality at the exit

    Parameters
    ----------
    Rt          : throat radius [m]
    epsilon     : expansion ratio
    gamma       : ratio of specific heats
    pa_over_p0   : design ambient/stagnation pressure ratio Pa/P0
    length_pct  : bell length as % of 15° cone
    n_ce_pts    : number of CE stations
    max_outer_iter : max Lagrange-multiplier iterations
    max_inner_iter : max station-level Newton iterations
    tol         : convergence tolerance

    Returns
    -------
    ControlSurface with optimized (M, θ, φ) distribution
    """
    Re = math.sqrt(epsilon) * Rt
    At = math.pi * Rt * Rt
    if pa_over_p0 < 0.0:
        raise ValueError("pa_over_p0 must be non-negative")

    # Target length (same convention as bell nozzle)
    L_15 = (Re - Rt) / math.tan(math.radians(15.0))
    Ln_target = (length_pct / 100.0) * L_15

    # Target mass flow (from throat conditions, quasi-1D)
    # ṁ = ρ*·a*·A* → normalized: ṁ_norm = (ρ/ρ₀·V̄)_throat · A*
    # At M=1: ρV = ρ₀·(2/(γ+1))^((γ+1)/(2(γ-1))) · a₀ · 1
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    rhoV_star = (2.0 / gp1) ** (gp1 / (2.0 * gm1))
    mdot_target = rhoV_star * At  # normalized by ρ₀·a₀, dimensional area retained

    # Initial guess
    ce = _initial_ce_guess(Rt, Re, Ln_target, gamma, n_ce_pts)

    # Initial Lagrange multipliers (heuristic seeds)
    lambda2 = -0.5
    lambda3 = 0.01

    relaxation = 0.3   # under-relaxation for stability
    lambda_lr = 0.05   # learning rate for λ updates

    for outer in range(max_outer_iter):
        # ─── Inner loop: enforce stationarity at each CE station ───
        for i in range(1, n_ce_pts - 1):
            M_i = ce.M[i]
            theta_i = ce.theta[i]
            phi_i = ce.phi[i]
            r_i = ce.r[i]

            for inner in range(max_inner_iter):
                R = stationarity_residuals(M_i, theta_i, phi_i,
                                           r_i, gamma, lambda2, lambda3,
                                           pa_over_p0)
                if np.linalg.norm(R) < tol:
                    break

                # Numerical Jacobian of R w.r.t. (M, θ, φ)
                dh = 1e-5
                J = np.zeros((3, 3))
                state = np.array([M_i, theta_i, phi_i])
                for j in range(3):
                    sp = state.copy()
                    sm = state.copy()
                    sp[j] += dh
                    sm[j] -= dh
                    # Clamp Mach and phi
                    sp[0] = max(sp[0], 1.001)
                    sm[0] = max(sm[0], 1.001)
                    sp[2] = np.clip(sp[2], 1e-4, math.pi - 1e-4)
                    sm[2] = np.clip(sm[2], 1e-4, math.pi - 1e-4)

                    Rp = stationarity_residuals(sp[0], sp[1], sp[2],
                                                r_i, gamma, lambda2, lambda3,
                                                pa_over_p0)
                    Rm = stationarity_residuals(sm[0], sm[1], sm[2],
                                                r_i, gamma, lambda2, lambda3,
                                                pa_over_p0)
                    J[:, j] = (Rp - Rm) / (sp[j] - sm[j])

                try:
                    delta = np.linalg.solve(J, -R)
                except np.linalg.LinAlgError:
                    break

                # Damped update
                M_i += relaxation * delta[0]
                theta_i += relaxation * delta[1]
                phi_i += relaxation * delta[2]

                # Clamp to physical ranges
                M_i = max(M_i, 1.001)
                theta_i = np.clip(theta_i, -0.1, math.radians(50))
                phi_i = np.clip(phi_i, math.radians(5), math.radians(175))

            ce.M[i] = M_i
            ce.theta[i] = theta_i
            ce.phi[i] = phi_i

        # Enforce boundary: near-parallel exit (θ → 0 at wall/exit)
        ce.theta[-1] = max(0.0, ce.theta[-2] * 0.3)

        # ─── Constraint residuals and λ update ───
        F_val, mdot_val, L_val = _integrate_ce(ce, gamma, pa_over_p0)

        # Mass-flow constraint residual
        mdot_err = mdot_val - mdot_target
        # Length constraint residual
        L_err = L_val - Ln_target

        # Update multipliers by gradient of constraint violation
        lambda2 -= lambda_lr * mdot_err
        lambda3 -= lambda_lr * L_err

        # ─── Transversality at exit ───
        T_res = transversality_residual(
            ce.M[-1], ce.theta[-1], ce.phi[-1],
            ce.r[-1], gamma, lambda2, lambda3, pa_over_p0
        )
        # Adjust exit φ to satisfy transversality
        if abs(T_res) > tol:
            dh = 1e-5
            T_p = transversality_residual(
                ce.M[-1], ce.theta[-1], ce.phi[-1] + dh,
                ce.r[-1], gamma, lambda2, lambda3, pa_over_p0
            )
            dT_dphi = (T_p - T_res) / dh
            if abs(dT_dphi) > 1e-15:
                ce.phi[-1] -= relaxation * T_res / dT_dphi
                ce.phi[-1] = np.clip(ce.phi[-1], math.radians(5),
                                     math.radians(175))

        # Convergence check
        total_residual = abs(mdot_err) + abs(L_err) + abs(T_res)
        ce.residual_norm = float(total_residual)
        ce.mdot_target = float(mdot_target)
        ce.mdot_residual = float(mdot_err)
        ce.length_residual = float(L_err)
        ce.transversality = float(T_res)
        if total_residual < tol * 10:
            ce.converged = True
            break

    ce.lambda2 = lambda2
    ce.lambda3 = lambda3
    if not ce.converged:
        ce.warnings.append(
            "Variational stationarity solve did not meet convergence tolerance."
        )
    return ce


# ─────────────────────────────────────────────────────────────────────
#  MOC wall construction from optimal CE
# ─────────────────────────────────────────────────────────────────────

def _construct_wall_from_ce(
    Rt: float, epsilon: float, gamma: float,
    ce: ControlSurface, Ln: float,
    n_char: int = 30,
    resample_for_export: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Construct the nozzle wall from the optimal control surface using
    the true Rao Method of Characteristics (backward construction).

    This implements the core of Rao's 1958 method:
      1. The CE provides target flow conditions (M, θ) at radial stations.
      2. C⁻ characteristics from the CE carry compatibility data
         (θ − ν) backward toward the wall.
      3. C⁺ characteristics from interior points carry (θ + ν) forward.
      4. Each wall point is determined by intersecting C⁺ from the
         nearest interior point with C⁻ from the CE, subject to the
         streamline (wall tangency) condition: θ_flow = θ_wall.

    The result is an experimental wall trace from a CE-driven characteristic
    construction.  If post-processing is needed to fit the public contour API,
    diagnostics mark characteristic compatibility as not preserved.

    References
    ----------
    - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
    - NASA TM (1990), Rao method re-derivation with explicit functionals
    - Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 11
    """
    Re = math.sqrt(epsilon) * Rt
    Rd = 0.382 * Rt
    theta_n = max(float(ce.theta[0]), math.radians(1.0))
    diagnostics = {
        "fallback_used": False,
        "postprocessed": False,
        "moc_compatibility_preserved": True,
        "clamp_hits": 0,
        "nonmonotonic_x_drops": 0,
        "warnings": [],
    }

    Nx = Rd * math.sin(theta_n)
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))

    # ── Step 1: Compute CE physical (x, r) positions ──────────────
    # Along the CE surface with inclination φ(r):  dx/dr = cot(φ).
    # BVP-backed solutions carry x explicitly; legacy CE objects still use
    # the historical reconstruction path.
    n_ce = len(ce.r)
    if ce.x is not None:
        x_ce = np.asarray(ce.x, dtype=float).copy()
    else:
        x_ce = np.zeros(n_ce)
        x_ce[0] = Nx + 0.05 * (Ln - Nx)   # start slightly past inflection
        for i in range(1, n_ce):
            phi_avg = 0.5 * (ce.phi[i - 1] + ce.phi[i])
            dr = ce.r[i] - ce.r[i - 1]
            sin_phi = math.sin(phi_avg)
            if abs(sin_phi) > 1e-10:
                x_ce[i] = x_ce[i - 1] + math.cos(phi_avg) / sin_phi * dr
            else:
                x_ce[i] = x_ce[i - 1] + 0.5 * dr

        # Normalise legacy CE guesses so they span from just past inflection to
        # the exit.  Solved BVP x_ce values are never rescaled here.
        if x_ce[-1] > x_ce[0]:
            scale = (Ln - x_ce[0]) / (x_ce[-1] - x_ce[0])
            x_ce = x_ce[0] + (x_ce - x_ce[0]) * scale
        else:
            x_ce = np.linspace(x_ce[0], Ln, n_ce)

    # ── Step 2: Build CE interpolation arrays ─────────────────────
    ce_r = np.asarray(ce.r, dtype=float)
    ce_M = np.maximum(np.asarray(ce.M, dtype=float), 1.001)
    ce_theta = np.asarray(ce.theta, dtype=float)

    ce_nu = np.array([prandtl_meyer(M, gamma) for M in ce_M])
    ce_mu = np.array([mach_angle(M) for M in ce_M])
    ce_cm = ce_theta - ce_nu          # C⁻ compatibility values on CE

    def _interp(r_q, data):
        """Linearly interpolate a CE quantity at radius *r_q*."""
        r_clamped = float(np.clip(r_q, ce_r[0], ce_r[-1]))
        return float(np.interp(r_clamped, ce_r, data))

    # ── Step 3: Transonic starting line ───────────────────────────
    starting_line = approximate_starting_line(
        Rt, Rd, theta_n, gamma, n_char, method='area_ratio'
    )

    # ── Step 4: March with CE-driven wall points ──────────────────
    prev_pts = list(starting_line)
    wall_x_list: list[float] = [float(Nx)]
    wall_r_list: list[float] = [float(Ny)]

    for _row in range(1, 500):
        if len(prev_pts) < 2:
            break

        new_pts: list[CharPoint] = []

        # Axis point (symmetry BC: θ=0, r=0)
        if prev_pts[0].r < 1e-10:
            axis_pt = solve_axis_point(prev_pts[1], gamma, True)
        else:
            axis_pt = solve_axis_point(prev_pts[0], gamma, True)
        new_pts.append(axis_pt)

        # Interior points (standard adjacent-pair solve)
        interior: list[CharPoint] = []
        for j in range(len(prev_pts) - 2):
            # p_minus = upper, p_plus = lower
            pt = solve_interior_point(
                prev_pts[j + 1], prev_pts[j], gamma, True
            )
            interior.append(pt)
            new_pts.append(pt)

        # ── CE-driven wall point ──────────────────────────────────
        p_in = interior[-1] if interior else new_pts[-1]

        # Initial estimate: project C⁺ from p_in upward
        r_w = p_in.r + 0.05 * (Re - p_in.r)
        x_w = p_in.x + 0.01 * max(Ln - p_in.x, 1e-6)
        theta_w = p_in.theta
        M_w = max(p_in.M, 1.001)
        nu_w = prandtl_meyer(M_w, gamma)
        mu_w = mach_angle(M_w)

        for _it in range(20):
            # C⁻ from CE interpolated at current wall radius
            cm_ce = _interp(r_w, ce_cm)
            M_ce  = max(_interp(r_w, ce_M), 1.001)
            theta_ce = _interp(r_w, ce_theta)
            mu_ce = mach_angle(M_ce)
            x_ce_loc = _interp(r_w, x_ce)

            # C⁺ from interior with axisymmetric source correction
            cp = p_in.compat_plus
            if r_w > 1e-10 and p_in.r > 1e-10:
                ds = math.sqrt(
                    (x_w - p_in.x) ** 2 + (r_w - p_in.r) ** 2
                )
                if ds > 1e-12:
                    th_a = 0.5 * (p_in.theta + theta_w)
                    mu_a = 0.5 * (p_in.mu + mu_w)
                    r_a  = 0.5 * (p_in.r + r_w)
                    cos_tp = math.cos(th_a + mu_a)
                    if abs(cos_tp) > 1e-15 and r_a > 1e-10:
                        Qp = (math.sin(th_a) * math.sin(mu_a)
                              * math.cos(mu_a) / (r_a * cos_tp))
                        cp = p_in.compat_plus + Qp * ds

            # Solve compatibility:  θ_w + ν_w = cp,  θ_w − ν_w = cm_ce
            theta_w_new = 0.5 * (cp + cm_ce)
            nu_w_new    = 0.5 * (cp - cm_ce)
            if nu_w_new < 1e-8:
                nu_w_new = 1e-8
            M_w_new  = mach_from_prandtl_meyer(nu_w_new, gamma)
            mu_w_new = mach_angle(M_w_new)

            # New position: intersection of C⁺ from p_in & C⁻ from CE
            sl_plus = math.tan(
                0.5 * (p_in.theta + theta_w_new)
                + 0.5 * (p_in.mu + mu_w_new)
            )
            sl_minus = math.tan(
                0.5 * (theta_ce + theta_w_new)
                - 0.5 * (mu_ce + mu_w_new)
            )
            denom = sl_plus - sl_minus
            if abs(denom) > 1e-15:
                x_new = (
                    _interp(r_w, ce_r) - p_in.r
                    + sl_plus * p_in.x - sl_minus * x_ce_loc
                ) / denom
                r_new = p_in.r + sl_plus * (x_new - p_in.x)
            else:
                x_new, r_new = x_w, r_w

            # Clamp to physical region and count every time the construction
            # leaves the admissible local marching domain.
            x_before_clamp = x_new
            r_before_clamp = r_new
            r_new = max(r_new, p_in.r + 1e-8)
            r_new = min(r_new, Re * 1.1)
            x_new = max(x_new, p_in.x + 1e-8)
            x_new = min(x_new, Ln * 1.2)
            if x_new != x_before_clamp or r_new != r_before_clamp:
                diagnostics["clamp_hits"] += 1

            converged = (
                abs(x_new - x_w) < 1e-10
                and abs(r_new - r_w) < 1e-10
                and abs(theta_w_new - theta_w) < 1e-10
            )
            x_w, r_w = x_new, r_new
            theta_w = theta_w_new
            nu_w, M_w, mu_w = nu_w_new, M_w_new, mu_w_new
            if converged:
                break

        wall_pt = _make_point(x_w, r_w, theta_w, M_w, gamma)
        new_pts.append(wall_pt)
        wall_x_list.append(x_w)
        wall_r_list.append(r_w)

        if x_w >= Ln * 0.98:
            break
        prev_pts = new_pts

    # ── Step 5: Post-process wall contour ─────────────────────────
    wall_x = np.array(wall_x_list)
    wall_r = np.array(wall_r_list)

    if len(wall_x) < 3:
        raise RuntimeError(
            "Rao variational MOC construction produced too few wall points."
        )

    # Drop any non-monotonic-x entries
    valid = np.ones(len(wall_x), dtype=bool)
    for i in range(1, len(wall_x)):
        if wall_x[i] <= wall_x[i - 1]:
            valid[i] = False
            diagnostics["nonmonotonic_x_drops"] += 1
            diagnostics["postprocessed"] = True
            diagnostics["moc_compatibility_preserved"] = False
    wall_x, wall_r = wall_x[valid], wall_r[valid]

    if len(wall_x) < 3:
        raise RuntimeError(
            "Rao variational MOC construction lost too many non-monotonic wall points."
        )

    if not resample_for_export:
        return wall_x, wall_r, diagnostics

    # Resampling/end-point enforcement is a geometric export step, not an exact
    # characteristic solution.  Keep that fact visible in diagnostics.
    x_uniform = np.linspace(float(wall_x[0]), Ln, 100)
    r_uniform = np.interp(x_uniform, wall_x, wall_r)
    if abs(r_uniform[0] - Ny) > 1e-10 or abs(r_uniform[-1] - Re) > 1e-10:
        diagnostics["postprocessed"] = True
        diagnostics["moc_compatibility_preserved"] = False
    r_uniform[0] = Ny
    r_uniform[-1] = Re

    min_dr = float(np.min(np.diff(r_uniform))) if len(r_uniform) > 1 else 0.0
    if min_dr < -1e-9:
        diagnostics["warnings"].append(
            "Constructed wall radius required monotonic geometric cleanup; "
            "contour remains experimental."
        )
        diagnostics["postprocessed"] = True
        diagnostics["moc_compatibility_preserved"] = False
        r_uniform = np.maximum.accumulate(r_uniform)
        r_uniform[-1] = Re

    if not diagnostics["moc_compatibility_preserved"]:
        diagnostics["warnings"].append(
            "MOC wall was geometrically post-processed; exact characteristic "
            "compatibility is not guaranteed."
        )

    return x_uniform, r_uniform, diagnostics


# ─────────────────────────────────────────────────────────────────────
#  Rao variational/MOC boundary-value scaffold
# ─────────────────────────────────────────────────────────────────────

def _target_length(Rt: float, epsilon: float, length_pct: float) -> float:
    Re = math.sqrt(epsilon) * Rt
    return (length_pct / 100.0) * ((Re - Rt) / math.tan(math.radians(15.0)))


def _target_mdot(Rt: float, gamma: float) -> float:
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    rhoV_star = (2.0 / gp1) ** (gp1 / (2.0 * gm1))
    return rhoV_star * math.pi * Rt * Rt


def _flow_nodes_from_curve(curve_or_nodes) -> list[FlowNode]:
    """Return a curve as minimal flow-state nodes."""
    if isinstance(curve_or_nodes, ControlSurface):
        return _control_surface_flow_nodes(curve_or_nodes)

    nodes: list[FlowNode] = []
    for item in curve_or_nodes:
        if isinstance(item, FlowNode):
            nodes.append(item)
        elif hasattr(item, "to_flow_node"):
            nodes.append(item.to_flow_node())
        else:
            nodes.append(FlowNode(
                x=float(item.x),
                r=float(item.r),
                M=max(float(item.M), 1.001),
                theta=float(item.theta),
            ))
    return nodes


def curve_mass_flux(curve_or_nodes, gamma: float) -> float:
    """
    Axisymmetric mass flux through a discrete flow-state curve.

    Rao's mass closure is between the mass flux crossing the optimized
    control surface DE and the kernel cross-section BD.  For a polyline
    segment with tangent angle beta, the nondimensional surface flux is

        dmdot = 2*pi*r*rho*V*|sin(beta - theta)|*ds

    where rho and V are stagnation-normalized isentropic quantities.  The
    same integral is used for both DE and BD so the residual compares like
    with like rather than comparing against the quasi-1D throat value.
    """
    nodes = _flow_nodes_from_curve(curve_or_nodes)
    total = 0.0
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        dx = float(p1.x - p0.x)
        dr = float(p1.r - p0.r)
        ds = math.hypot(dx, dr)
        if ds < 1e-12:
            continue
        beta = math.atan2(dr, dx)
        M = max(0.5 * (float(p0.M) + float(p1.M)), 1.001)
        theta = 0.5 * (float(p0.theta) + float(p1.theta))
        r = max(0.5 * (float(p0.r) + float(p1.r)), 1e-12)
        rho = isentropic_density_ratio(M, gamma)
        T = isentropic_temperature_ratio(M, gamma)
        V = M * math.sqrt(gamma * T)
        total += (
            2.0 * math.pi * r * rho * V
            * abs(math.sin(beta - theta)) * ds
        )
    return float(total)


def _interp_flow_node(p0: FlowNode, p1: FlowNode, t: float) -> FlowNode:
    """Linear state interpolation on one characteristic segment."""
    q = float(np.clip(t, 0.0, 1.0))
    return FlowNode(
        x=float(p0.x + q * (p1.x - p0.x)),
        r=float(p0.r + q * (p1.r - p0.r)),
        M=max(float(p0.M + q * (p1.M - p0.M)), 1.001),
        theta=float(p0.theta + q * (p1.theta - p0.theta)),
    )


def kernel_bd_segment(kernel_bd, d_fraction: float) -> list[FlowNode]:
    """
    Return the B-to-D subset of the kernel BD left-running Mach line.

    ``d_fraction`` is the solved point-D location by arc length, with 0 at B
    (the wall-side end of BD) and 1 at the deepest available kernel point.
    """
    nodes = _flow_nodes_from_curve(kernel_bd)
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
    if target <= 0.0:
        return [nodes[0]]

    out = [nodes[0]]
    accum = 0.0
    for p0, p1, ds in zip(nodes[:-1], nodes[1:], seg_lengths):
        if ds <= 1e-14:
            continue
        if accum + ds < target:
            out.append(p1)
            accum += ds
            continue
        out.append(_interp_flow_node(p0, p1, (target - accum) / ds))
        break
    return out


def _kernel_left_mach_line_from_starting_line(
    starting_line: list[CharPoint],
    gamma: float,
) -> list[CharPoint]:
    """
    Build the wall-originating left-running Mach line BD inside the kernel.

    The simplified Python kernel starts from TT' and repeatedly applies the
    interior/axis MOC unit processes without a wall boundary.  The wall-side
    point of each shrinking row lies on the last C- family line, the BD
    cross-section used by Rao's mass closure.
    """
    if not starting_line:
        return []
    prev_pts = list(starting_line)
    bd = [prev_pts[-1]]
    while len(prev_pts) >= 2:
        new_pts: list[CharPoint] = []
        if prev_pts[0].r < 1e-10 and len(prev_pts) > 1:
            axis_pt = solve_axis_point(prev_pts[1], gamma, True)
        else:
            axis_pt = solve_axis_point(prev_pts[0], gamma, True)
        new_pts.append(axis_pt)

        for j in range(len(prev_pts) - 2):
            new_pts.append(solve_interior_point(
                prev_pts[j + 1], prev_pts[j], gamma, True
            ))

        bd.append(new_pts[-1])
        if new_pts[-1].r <= 1e-10 or len(new_pts) < 2:
            break
        prev_pts = new_pts
    return bd


def _seed_kernel_d_fraction(
    kernel_bd,
    target_flux: float,
    gamma: float,
) -> float:
    """Seed D by matching the initial CE flux on the kernel BD curve."""
    full_flux = curve_mass_flux(kernel_bd, gamma)
    if full_flux <= 1e-14 or target_flux <= 0.0:
        return 1.0
    if target_flux >= full_flux:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mid_flux = curve_mass_flux(kernel_bd_segment(kernel_bd, mid), gamma)
        if mid_flux < target_flux:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mass_closure_fluxes(
    ce: ControlSurface,
    config: RaoSolverConfig,
) -> tuple[float, float, float, list[FlowNode]]:
    """
    Return CE mass, kernel BD mass target, reference scale, and BD segment.

    If the caller did not supply a kernel BD curve, fall back to the legacy
    throat reference so older helper-level calls remain evaluable.  The public
    ``solve_rao_bvp`` path always supplies ``config.kernel_bd``.
    """
    ce_flux = curve_mass_flux(ce, config.gamma)
    if config.kernel_bd:
        bd_flux, bd_segment = calc_mdot_bd(
            config.kernel_bd, ce.kernel_d_fraction, config.gamma
        )
        bd_full_flux = curve_mass_flux(config.kernel_bd, config.gamma)
        mdot_ref = max(
            abs(bd_full_flux),
            abs(bd_flux),
            _target_mdot(config.Rt, config.gamma),
            1e-12,
        )
        return ce_flux, bd_flux, mdot_ref, bd_segment

    throat_target = _target_mdot(config.Rt, config.gamma)
    return ce_flux, throat_target, max(abs(throat_target), 1e-12), []


def _initial_ce_from_kernel(config: RaoSolverConfig):
    """
    Build a kernel-seeded control-surface initial guess and kernel BD.

    The kernel is built by the NASA-style routines in :mod:`raosim.nasa_moc`:
    :func:`nasa_moc.build_kernel` lays BD along the throat arc with PM-
    expanded Mach values, and :func:`nasa_moc.calc_lrc_de` runs the
    Rao secant on the D location with NASA's C+ derivative system for DE.
    The resulting topology is stored on the returned ControlSurface and
    on the kernel_bd flow-node tuple for downstream residual evaluation.

    Returns ``(ce, kernel_bd_flow_nodes, topology)`` where ``topology`` is
    a :class:`nasa_moc.RaoTopology` capturing point B, BD, point D, DE,
    point E, mass closures, and the Rao stationarity constant.
    """
    from raosim.nasa_moc import build_kernel as _build_kernel
    from raosim.nasa_moc import RaoTopology as _RaoTopology
    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    Ln = _target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    theta_b_seed = math.radians(config.thetaN_guess_deg)

    kernel = _build_kernel(
        Rt, Rd, theta_b_seed, config.gamma, config.n_kernel,
        starting_line_method=config.starting_line_method,
    )
    kernel_bd_flow_nodes = [node.to_flow_node() for node in kernel.bd]

    topology: _RaoTopology | None = None
    try:
        topology = calc_lrc_de(
            kernel,
            x_E=Ln, r_E=Re,
            gamma=config.gamma, Rt=Rt, epsilon=config.epsilon,
            pa_over_p0=config.pa_over_p0,
            n_points=config.n_control,
        )
    except Exception:
        topology = None

    ce = _initial_ce_guess(Rt, Re, Ln, config.gamma, config.n_control)
    try:
        Me_target = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me_target = max(float(ce.M[-1]), 2.0)

    # Use the old axis-to-exit linear seed which least_squares is
    # calibrated for.  The NASA topology drives the kernel_d_fraction
    # unknown and supplies the kernel BD curve for the mass-closure
    # residual, but the CE r-grid spans axis to exit (the BVP's native
    # parametrisation).
    frac = np.linspace(0.0, 1.0, config.n_control)
    ce.M = np.maximum(ce.M, 1.001 + (Me_target - 1.001) * frac ** 0.85)
    ce.theta = np.clip(
        theta_b_seed * (1.0 - 0.55 * frac),
        math.radians(-5.0), math.radians(55.0),
    )
    ce.x = np.linspace(0.0, Ln, config.n_control)
    # Seed kernel_d_fraction so the wall-to-D mass flow along BD equals
    # the quasi-1D throat target (best initial guess for the BVP mass
    # closure when the kernel BD curve already spans the full radial
    # cross-section).  The least_squares solve then refines it.
    if kernel_bd_flow_nodes:
        throat_target = _target_mdot(config.Rt, config.gamma)
        ce.kernel_d_fraction = _seed_kernel_d_fraction(
            kernel_bd_flow_nodes, throat_target, config.gamma,
        )
    elif topology is not None:
        ce.kernel_d_fraction = float(topology.d_fraction)
    else:
        ce.kernel_d_fraction = 0.5

    ce.phi = _phi_from_curve(ce.x, ce.r)
    ce.phi = np.clip(ce.phi, math.radians(5.0), math.radians(88.0))
    return ce, kernel_bd_flow_nodes, topology


def _phi_from_curve(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return node-centered CE inclinations from a physical polyline."""
    x_arr = np.asarray(x, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    n = len(r_arr)
    if n < 2:
        return np.full(n, math.radians(45.0))

    seg_beta = np.arctan2(np.diff(r_arr), np.diff(x_arr))
    phi = np.empty(n, dtype=float)
    phi[0] = seg_beta[0]
    phi[-1] = seg_beta[-1]
    if n > 2:
        phi[1:-1] = 0.5 * (seg_beta[:-1] + seg_beta[1:])
    return phi


def _integrate_ce_x_from_left_mach(
    r: np.ndarray,
    theta: np.ndarray,
    M: np.ndarray,
    x_start: float,
) -> np.ndarray:
    """
    Reconstruct CE axial positions from the left-Mach-line ODE.

    DE is a left-running C+ characteristic, so
    ``dr/dx = tan(theta + mu)`` and equivalently
    ``dx/dr = cot(theta + mu) = 1/tan(theta + mu)``.  Trapezoidal
    integration in ``r`` from ``x_start`` (= ``D.x`` on the kernel BD)
    gives ``x[i]`` exactly consistent with the left-Mach-line geometry,
    making ``left_mach`` residuals identically zero by construction.

    Parameters
    ----------
    r : (n,) array
        CE radii, axis-first.  Must be monotone non-decreasing.
    theta, M : (n,) arrays
        CE flow angles (rad) and Mach numbers at the corresponding radii.
    x_start : float
        Axial coordinate of CE node 0 (point D on the kernel BD).

    Returns
    -------
    x : (n,) array
        Axial coordinates of every CE node, satisfying
        ``dx/dr = 1 / tan(theta_avg + mu_avg)`` segment by segment.
    """
    r_arr = np.asarray(r, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    M_arr = np.asarray(M, dtype=float)
    n = len(r_arr)
    x = np.empty(n, dtype=float)
    x[0] = float(x_start)
    # Use midpoint averaging — this is exactly the form
    # ``residual_left_mach_geometry(p0, p1)`` checks:
    #   dr - dx * tan(theta_avg + mu_avg) = 0
    # so the integrated x makes that residual identically zero
    # (bit-comparable to machine precision).
    for i in range(1, n):
        m_lo = max(float(M_arr[i - 1]), 1.000001)
        m_hi = max(float(M_arr[i]), 1.000001)
        mu_avg = 0.5 * (math.asin(1.0 / m_lo) + math.asin(1.0 / m_hi))
        theta_avg = 0.5 * (float(theta_arr[i - 1]) + float(theta_arr[i]))
        denom = math.tan(theta_avg + mu_avg)
        dr_step = float(r_arr[i]) - float(r_arr[i - 1])
        if abs(denom) < 1e-12:
            # Near-vertical C+ slope (e.g. M → 1 at a corner).  Fall back
            # to a small forward step so the integration doesn't NaN.
            x[i] = x[i - 1] + 1e-9
        else:
            x[i] = x[i - 1] + dr_step / denom
    return x


def _pack_bvp(
    ce: ControlSurface,
    lambda2: float,
    lambda3: float,
    log_C: float | None = None,
    wall: WallSurface | None = None,
) -> np.ndarray:
    """Pack CE (+ optional wall) state into the BVP unknown vector.

    Layout after the left-Mach-by-construction refactor.  ``x_ce`` is
    NOT carried — it is reconstructed at unpack time via
    :func:`_integrate_ce_x_from_left_mach` from
    ``(r_ce, theta_ce, M_ce)`` and the kernel BD point D
    (selected by ``kernel_d_fraction``).  Eliminating ``x_ce`` from the
    unknowns removes a degenerate basin where the optimiser could
    satisfy left-Mach geometry by moving ``x`` at the expense of mass:

        [M_ce, theta_ce, r_ce,
         (M_w, theta_w, x_w, r_w),   # Phase 6 only, full 4-tuple kept
         lambda2, lambda3, log_C, kernel_d_fraction]

    Size: ``3*n_ce + 4*n_wall + 4``  (was ``4*n_ce + 4*n_wall + 4``).
    """
    log_C_val = float(log_C) if log_C is not None else float(ce.log_C)
    parts: list[np.ndarray] = [
        np.asarray(ce.M, dtype=float),
        np.asarray(ce.theta, dtype=float),
        np.asarray(ce.r, dtype=float),
    ]
    if wall is not None:
        parts.extend([
            np.asarray(wall.M, dtype=float),
            np.asarray(wall.theta, dtype=float),
            np.asarray(wall.x, dtype=float),
            np.asarray(wall.r, dtype=float),
        ])
    parts.append(np.asarray(
        [lambda2, lambda3, log_C_val, float(ce.kernel_d_fraction)],
        dtype=float,
    ))
    return np.concatenate(parts)


def _unpack_bvp(
    u: np.ndarray, r: np.ndarray, *,
    n_wall: int = 0,
    kernel_bd: tuple | None = None,
    gamma: float = 1.4,
) -> tuple[ControlSurface, WallSurface | None]:
    """Unpack the BVP unknown vector into ``(CE, optional wall)``.

    ``ce.x`` is reconstructed from the left-Mach-line ODE: at the kernel
    BD point D (selected by ``kernel_d_fraction``) we have
    ``x_start = D.x``; downstream CE nodes integrate
    ``dx/dr = 1/tan(theta + mu)`` along the C+ characteristic.  This
    makes the left-Mach-line geometry exact by construction and
    removes the old ``left_mach`` residual block as an independent
    constraint.

    Parameters
    ----------
    u : ndarray
        Packed BVP unknown vector (layout per :func:`_pack_bvp`).
    r : ndarray
        Template CE radii (used only to recover ``n_ce``; the actual
        radii come from ``u``).
    n_wall : int
        Number of wall unknowns to expect.  Pass 0 for legacy
        no-coupled-wall callers.
    kernel_bd : tuple of FlowNode | None
        Kernel BD curve.  When provided, the CE start ``x[0]`` is the
        ``x``-coordinate of point D interpolated at
        ``kernel_d_fraction``.  When ``None`` (legacy / unit tests),
        ``x[0] = 0`` is used as a stub.
    gamma : float
        Used only by ``calc_mdot_bd`` to locate D on the kernel BD;
        any positive value works for the geometric lookup.
    """
    n = len(r)
    M_ce = np.asarray(u[:n], dtype=float).copy()
    theta_ce = np.asarray(u[n:2 * n], dtype=float).copy()
    r_ce = np.asarray(u[2 * n:3 * n], dtype=float).copy()

    base_after_ce = 3 * n
    wall: WallSurface | None = None
    if n_wall > 0 and u.size >= base_after_ce + 4 * n_wall + 4:
        w_M = np.asarray(u[base_after_ce: base_after_ce + n_wall],
                          dtype=float).copy()
        w_theta = np.asarray(
            u[base_after_ce + n_wall: base_after_ce + 2 * n_wall], dtype=float
        ).copy()
        w_x = np.asarray(
            u[base_after_ce + 2 * n_wall: base_after_ce + 3 * n_wall], dtype=float
        ).copy()
        w_r = np.asarray(
            u[base_after_ce + 3 * n_wall: base_after_ce + 4 * n_wall], dtype=float
        ).copy()
        wall = WallSurface(x=w_x, r=w_r, M=w_M, theta=w_theta)
        scalar_start = base_after_ce + 4 * n_wall
    else:
        scalar_start = base_after_ce

    log_C_val = float(u[scalar_start + 2]) if u.size >= scalar_start + 3 else 0.0
    kernel_d_fraction = (
        float(u[scalar_start + 3]) if u.size >= scalar_start + 4 else 1.0
    )

    # Reconstruct ``x_ce`` from the left-Mach-line ODE.  ``x_start`` is
    # the x-coordinate of point D on the kernel BD curve.
    if kernel_bd:
        try:
            _, bd_segment = calc_mdot_bd(kernel_bd, kernel_d_fraction, gamma)
            x_start = float(bd_segment[-1].x) if bd_segment else 0.0
        except Exception:
            x_start = 0.0
    else:
        x_start = 0.0
    x_ce = _integrate_ce_x_from_left_mach(r_ce, theta_ce, M_ce, x_start)
    phi = _phi_from_curve(x_ce, r_ce)

    ce = ControlSurface(
        r=r_ce,
        M=M_ce,
        theta=theta_ce,
        phi=phi,
        x=x_ce,
        lambda2=float(u[scalar_start]),
        lambda3=float(u[scalar_start + 1]),
        log_C=log_C_val,
        kernel_d_fraction=kernel_d_fraction,
    )
    return ce, wall


def _stationarity_matrix(ce: ControlSurface, gamma: float,
                         pa_over_p0: float) -> np.ndarray:
    rows = []
    for i in range(1, len(ce.r) - 1):
        rows.append(stationarity_residuals(
            float(ce.M[i]), float(ce.theta[i]), float(ce.phi[i]),
            float(ce.r[i]), gamma, ce.lambda2, ce.lambda3, pa_over_p0,
        ))
    if not rows:
        return np.zeros((0, 3))
    return np.vstack(rows)


def _ce_smoothness_regularization(ce: ControlSurface, gamma: float) -> np.ndarray:
    """
    Small CE smoothness stabilizer.

    This is deliberately not a MOC compatibility residual.  The second
    differences of theta +/- nu only regularize the discretized control
    surface; axisymmetric C+/C- compatibility must include source terms.
    """
    nu = np.array([prandtl_meyer(max(float(M), 1.001), gamma) for M in ce.M])
    kp = ce.theta + nu
    km = ce.theta - nu
    if len(kp) < 3:
        return np.zeros(0)
    scale = math.radians(1.0)
    return np.concatenate([np.diff(kp, n=2), np.diff(km, n=2)]) / scale


def _control_surface_flow_nodes(ce: ControlSurface) -> list[FlowNode]:
    """Reconstruct an unscaled CE polyline and expose it as flow nodes."""
    n = len(ce.r)
    if n == 0:
        return []
    if ce.x is not None:
        x = np.asarray(ce.x, dtype=float)
    else:
        x = np.zeros(n, dtype=float)
        for i in range(1, n):
            phi_avg = 0.5 * (float(ce.phi[i - 1]) + float(ce.phi[i]))
            dr = float(ce.r[i] - ce.r[i - 1])
            sin_phi = math.sin(phi_avg)
            if abs(sin_phi) > 1e-10:
                x[i] = x[i - 1] + math.cos(phi_avg) / sin_phi * dr
            else:
                x[i] = x[i - 1] + 0.5 * dr
    return [
        FlowNode(
            x=float(xi),
            r=float(ri),
            M=max(float(Mi), 1.001),
            theta=float(thetai),
        )
        for xi, ri, Mi, thetai in zip(x, ce.r, ce.M, ce.theta)
    ]


def _ce_axisymmetric_compatibility_residual_groups(
    ce: ControlSurface,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical C+/C- compatibility residuals on adjacent CE segments."""
    nodes = _control_surface_flow_nodes(ce)
    if len(nodes) < 2:
        return np.zeros(0), np.zeros(0)
    cplus: list[float] = []
    cminus: list[float] = []
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        cplus.append(residual_Cplus_axisym(p0, p1, gamma))
        cminus.append(residual_Cminus_axisym(p0, p1, gamma))
    scale = math.radians(1.0)
    return (
        np.asarray(cplus, dtype=float) / scale,
        np.asarray(cminus, dtype=float) / scale,
    )


def rao_stationarity_residual(
    node: FlowNode,
    log_C: float,
    gamma: float,
) -> float:
    """
    Algebraic Rao optimum-thrust stationarity at a single CE node.

    Rao-Beck-Booth (AIAA 99-2584, 1999) integrate Rao's 1958 stationarity
    along the supersonic control surface DE in closed form:

        M* · cos(θ − α) / cos(α) = C        (constant along DE)

    where M* = V/a* is the critical Mach (``mstar_from_M``), α = arcsin(1/M)
    is the Mach angle, and θ is the local flow angle.  We work with
    log(M* cos(θ−α)/cos(α)) − log_C so that log_C can be carried as the
    solver unknown -- this is well-conditioned for M* and cos(α) of order
    unity, and maps directly onto Rao's "constant of integration" C.

    Reference
    ---------
    propulsion_texts/rao1999.pdf -- AIAA 99-2584 Eq. 3
    propulsion_texts/RaoRecentDevinRockNozConfig.pdf -- Rao 1961
    """
    M = max(float(node.M), 1.0 + 1e-9)
    alpha = math.asin(1.0 / M)
    cos_a = math.cos(alpha)
    cos_tma = math.cos(float(node.theta) - alpha)
    Mstar = mstar_from_M(M, gamma)
    if cos_a <= 1e-12 or abs(cos_tma) <= 1e-12 or Mstar <= 1e-12:
        # Singular branch: fall back to the unscaled algebraic form.
        C = math.exp(log_C)
        return Mstar * cos_tma / max(cos_a, 1e-12) - C
    return math.log(Mstar) + math.log(abs(cos_tma)) - math.log(cos_a) - float(log_C)


def rao_stationarity_fd_residual(
    p0: FlowNode,
    p1: FlowNode,
    gamma: float,
) -> float:
    """
    Differential Rao stationarity between adjacent CE nodes (secondary check).

    Differentiating the algebraic form gives:

        d(ln M*) − (dθ − dα) tan(θ − α) + dα tan(α) = 0

    Used as a secondary residual during development; once the algebraic form
    converges robustly the differential check should hold automatically.

    Reference: rao1999.pdf, Eq. 4 (segment-to-segment consistency).
    """
    M0 = max(float(p0.M), 1.0 + 1e-9)
    M1 = max(float(p1.M), 1.0 + 1e-9)
    a0 = math.asin(1.0 / M0)
    a1 = math.asin(1.0 / M1)
    Ms0 = mstar_from_M(M0, gamma)
    Ms1 = mstar_from_M(M1, gamma)
    d_ln_Ms = math.log(Ms1) - math.log(Ms0)
    dth = float(p1.theta) - float(p0.theta)
    da = a1 - a0
    th = 0.5 * (float(p0.theta) + float(p1.theta))
    a = 0.5 * (a0 + a1)
    return d_ln_Ms - (dth - da) * math.tan(th - a) + da * math.tan(a)


def _rao_algebraic_stationarity_residuals(
    ce: ControlSurface,
    gamma: float,
) -> np.ndarray:
    """Per-node algebraic Rao stationarity residuals (vectorised over CE)."""
    nodes = _control_surface_flow_nodes(ce)
    if not nodes:
        return np.zeros(0, dtype=float)
    return np.array(
        [rao_stationarity_residual(p, ce.log_C, gamma) for p in nodes],
        dtype=float,
    )


def _rao_left_mach_geometry_residuals(ce: ControlSurface) -> np.ndarray:
    """
    Per-segment residuals enforcing dr/dx = tan(θ + α) along the CE.

    Rao DE is defined as a left-running Mach line; the segment tangent must
    therefore equal tan(θ + μ) at every interior CE step (§2.C).  Scaled by
    the local CE chord length so the residual is dimensionless.
    """
    nodes = _control_surface_flow_nodes(ce)
    if len(nodes) < 2:
        return np.zeros(0, dtype=float)
    out: list[float] = []
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        ds = math.hypot(p1.x - p0.x, p1.r - p0.r)
        scale = max(ds, 1e-12)
        out.append(residual_left_mach_geometry(p0, p1) / scale)
    return np.asarray(out, dtype=float)


def _seed_log_C_from_ce(ce: ControlSurface, gamma: float) -> float:
    """Seed log_C from the median CE node (initial-guess best estimate)."""
    nodes = _control_surface_flow_nodes(ce)
    if not nodes:
        return 0.0
    p = nodes[len(nodes) // 2]
    M = max(float(p.M), 1.0 + 1e-9)
    alpha = math.asin(1.0 / M)
    cos_a = math.cos(alpha)
    cos_tma = math.cos(float(p.theta) - alpha)
    Mstar = mstar_from_M(M, gamma)
    if cos_a <= 1e-12 or abs(cos_tma) <= 1e-12 or Mstar <= 1e-12:
        return 0.0
    return math.log(Mstar) + math.log(abs(cos_tma)) - math.log(cos_a)


# ---------------------------------------------------------------------
#  Phase 6 — coupled wall residuals
# ---------------------------------------------------------------------


def _wall_to_flow_nodes(wall: WallSurface) -> list[FlowNode]:
    out: list[FlowNode] = []
    for i in range(len(wall.x)):
        out.append(FlowNode(
            x=float(wall.x[i]), r=float(wall.r[i]),
            M=max(float(wall.M[i]), 1.000001),
            theta=float(wall.theta[i]),
        ))
    return out


def _coupled_wall_residuals(
    ce: ControlSurface,
    wall: WallSurface,
    config: RaoSolverConfig,
) -> dict[str, np.ndarray]:
    """
    Phase 6 coupled-wall residual blocks.

    Returns four arrays under keys ``"wall_endpoint"``, ``"wall_tangency"``,
    ``"cplus_ce_to_wall"``, ``"wall_intersection"``.  All four are sized
    according to ``len(wall)`` and the CE; pair CE node i ↔ wall node i
    (linear pairing per the Phase 6 MVP from REWRITE_PLAN.md).
    """
    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    L = _target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    theta_N, _ = _design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg,
    )
    Nx = Rd * math.sin(theta_N)
    Ny = Rt + Rd * (1.0 - math.cos(theta_N))

    wall_nodes = _wall_to_flow_nodes(wall)
    ce_nodes = _control_surface_flow_nodes(ce)

    if not wall_nodes:
        return {
            "wall_endpoint": np.zeros(0),
            "wall_tangency": np.zeros(0),
            "cplus_ce_to_wall": np.zeros(0),
            "wall_intersection": np.zeros(0),
        }

    # 1. Endpoint closure: wall[0] == (Nx, Ny), wall[-1] == (L, Re).
    L_scale = max(L, 1e-12)
    Re_scale = max(Re, 1e-12)
    endpoint = np.array([
        (wall_nodes[0].x - Nx) / L_scale,
        (wall_nodes[0].r - Ny) / Re_scale,
        (wall_nodes[-1].x - L) / L_scale,
        (wall_nodes[-1].r - Re) / Re_scale,
    ], dtype=float)

    # 2. Wall tangency on each segment.
    tangency = np.array([
        residual_wall_tangency(wall_nodes[i], wall_nodes[i + 1]) / Re_scale
        for i in range(len(wall_nodes) - 1)
    ], dtype=float)

    # 3. C+ axisymmetric compatibility from CE[i] to wall[i] using the
    # linear pairing of CE nodes ↔ wall nodes.  When n_ce != n_wall,
    # interpolate the CE state at fractional index i*(n_ce-1)/(n_wall-1).
    n_w = len(wall_nodes)
    n_ce = len(ce_nodes)
    cplus = []
    intersection = []
    if n_ce >= 2 and n_w >= 1:
        theta_scale = math.radians(1.0)
        for i in range(n_w):
            frac = i * (n_ce - 1) / max(n_w - 1, 1)
            j0 = int(min(max(math.floor(frac), 0), n_ce - 1))
            j1 = int(min(j0 + 1, n_ce - 1))
            t = frac - j0
            ce0 = ce_nodes[j0]
            ce1 = ce_nodes[j1]
            ce_paired = FlowNode(
                x=ce0.x + t * (ce1.x - ce0.x),
                r=ce0.r + t * (ce1.r - ce0.r),
                M=max(ce0.M + t * (ce1.M - ce0.M), 1.000001),
                theta=ce0.theta + t * (ce1.theta - ce0.theta),
            )
            wall_pt = wall_nodes[i]
            cplus.append(
                residual_Cplus_axisym(ce_paired, wall_pt, config.gamma)
                / theta_scale
            )
            intersection.append(
                residual_cplus_child_position(ce_paired, wall_pt) / Re_scale
            )
    cplus = np.asarray(cplus, dtype=float)
    intersection = np.asarray(intersection, dtype=float)

    return {
        "wall_endpoint": endpoint,
        "wall_tangency": tangency,
        "cplus_ce_to_wall": cplus,
        "wall_intersection": intersection,
    }


def _wall_from_bezier_contour(
    bezier: dict,
    n_wall: int,
    gamma: float,
    Rt: float,
) -> WallSurface:
    """
    Build a WallSurface seed from a Bezier Rao TOP contour.

    The Bezier path (``bell_nozzle_contour(method='bezier')``) is the
    chart-calibrated Rao TOP approximation: geometrically close to the
    Rao optimum at any (ε, length_pct), so it places the wall unknown
    inside the right basin of attraction for the BVP.

    Mach numbers at each wall node are computed from the local area
    ratio ``A(x)/At = (r(x)/Rt)^2`` via :func:`mach_from_area_ratio`
    (1-D supersonic branch — an acceptable approximation along the
    bounding streamline of an axisymmetric nozzle).  Flow angle θ is
    the local wall slope ``arctan(dr/dx)``.
    """
    x_bell = np.asarray(bezier["x_bell"], dtype=float)
    y_bell = np.asarray(bezier["y_bell"], dtype=float)
    if x_bell.size < 2:
        raise ValueError("bezier x_bell must have at least two points")
    # Re-sample by uniform arc length over the bell.
    seg = np.hypot(np.diff(x_bell), np.diff(y_bell))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 1e-12:
        raise ValueError("bezier bell has zero arc length")
    targets = np.linspace(0.0, total, n_wall)
    x = np.interp(targets, arc, x_bell)
    r = np.interp(targets, arc, y_bell)
    # Local wall slope via central differences.
    dx = np.gradient(x)
    dr = np.gradient(r)
    theta = np.arctan2(dr, np.maximum(dx, 1e-12))
    theta = np.clip(theta, 0.0, math.radians(60.0))
    # Mach via 1-D area-Mach at each radius (axisymmetric).
    At = math.pi * Rt * Rt
    M = np.empty(n_wall, dtype=float)
    for i in range(n_wall):
        area = math.pi * r[i] * r[i]
        ar = max(area / At, 1.0 + 1e-9)
        try:
            M[i] = mach_from_area_ratio(ar, gamma, supersonic=True)
        except ValueError:
            M[i] = 2.0
    M = np.maximum(M, 1.001)
    return WallSurface(x=x, r=r, M=M, theta=theta)


def _initial_wall_guess(
    config: RaoSolverConfig,
    ce: ControlSurface,
    topology,
) -> WallSurface:
    """
    Seed the coupled-wall unknown.

    Default path: use the Bezier Rao-TOP contour
    (:func:`raosim.nozzle_geometry.bell_nozzle_contour` with
    ``method='bezier'``).  It is the chart-calibrated TOP approximation
    of the Rao optimum and lands the wall unknown well inside the
    physical basin of attraction — far better than a linear N→E seed
    which would force the optimizer to discover the bell shape from
    scratch.

    Falls back to a quadratic N→E linear-in-r seed if the Bezier path
    fails for any reason.
    """
    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    L = _target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    n_w = max(int(config.n_wall), 4)

    # Bezier seed (preferred).
    try:
        from raosim.nozzle_geometry import bell_nozzle_contour

        bezier = bell_nozzle_contour(
            Rt=Rt, epsilon=config.epsilon,
            length_pct=config.length_pct,
            gamma=config.gamma, pa_over_p0=config.pa_over_p0,
            convergent_half_angle_deg=45.0,
            Ru_factor=1.5, Rd_factor=config.throat_downstream_radius_factor,
            method="bezier",
        )
        return _wall_from_bezier_contour(bezier, n_w, config.gamma, Rt)
    except Exception:
        pass

    # Fallback: linear-quadratic seed from N to E.
    theta_N_chart, theta_E_chart = _design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg,
    )
    Nx = Rd * math.sin(theta_N_chart)
    Ny = Rt + Rd * (1.0 - math.cos(theta_N_chart))
    x = np.linspace(Nx, L, n_w)
    s = (x - Nx) / max(L - Nx, 1e-12)
    r = Ny + (Re - Ny) * (1.0 - (1.0 - s) ** 2)

    try:
        Me = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me = max(float(ce.M[-1]), 2.0)
    M_start = float(topology.B.M) if topology is not None else float(ce.M[0])
    M = M_start + (Me - M_start) * s
    M = np.maximum(M, 1.001)
    theta = theta_N_chart + (theta_E_chart - theta_N_chart) * s
    return WallSurface(x=x, r=r, M=M, theta=theta)


def rao_valid_region(
    ce_or_nodes,
    *,
    tol: float = 0.0,
) -> tuple[float, list[float]]:
    """
    Evaluate the Rao smooth-flow validity inequality along the control surface.

    Rao's optimum-thrust nozzle assumes a continuous, monotonic supersonic
    expansion across the control surface DE.  The classical condition for the
    optimum to exist as a smooth (shock-free) flow is

        b(s) = 1 - (dα/dθ) · [tan(θ - α) + tan(α)] / [tan(θ - α) - tan(α)]  ≥ 0

    where α = μ = arcsin(1/M) is the Mach angle and θ is the local flow
    angle along DE.  When ``b`` becomes negative the smooth-flow Rao
    construction is inapplicable and the optimum contour degenerates into
    a discontinuous (over-expanded) exit.

    References
    ----------
    - Rao, "Exhaust Nozzle Contour for Optimum Thrust", Jet Propulsion 1958
    - Rao, Beck, Booth, "Rao Variational Optimum Bell Nozzle: A Design
      Compendium", AIAA 99-2584 (1999)  -- propulsion_texts/rao1999.pdf
    - Östlund, *Flow processes in rocket engine nozzles ...*, KTH 2002
      (propulsion_texts/fulltext01.pdf, §3)

    Parameters
    ----------
    ce_or_nodes : ControlSurface | Sequence[FlowNode]
        Discrete control-surface nodes from D toward E.
    tol : float
        Slack on the inequality.  Pass ``residual_tol`` to allow numerical
        noise; values of ``b`` below ``-tol`` flag the input as outside the
        valid region.

    Returns
    -------
    (min_b, list_b) : tuple[float, list[float]]
        The minimum ``b`` value and the per-segment list.  If the input has
        fewer than two nodes ``min_b`` is ``+inf`` (vacuously valid) and
        ``list_b`` is empty.
    """
    if hasattr(ce_or_nodes, "M") and hasattr(ce_or_nodes, "theta"):
        nodes = _control_surface_flow_nodes(ce_or_nodes)
    else:
        nodes = list(ce_or_nodes)
    if len(nodes) < 2:
        return float("inf"), []

    bvalues: list[float] = []
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        dth = float(p1.theta - p0.theta)
        if abs(dth) < 1e-10:
            continue
        M0 = max(float(p0.M), 1.0 + 1e-9)
        M1 = max(float(p1.M), 1.0 + 1e-9)
        a0 = math.asin(1.0 / M0)
        a1 = math.asin(1.0 / M1)
        a = 0.5 * (a0 + a1)
        th = 0.5 * (float(p0.theta) + float(p1.theta))
        num = math.tan(th - a) + math.tan(a)
        den = math.tan(th - a) - math.tan(a)
        if abs(den) < 1e-12:
            bvalues.append(-math.inf)
        else:
            bvalues.append(1.0 - (a1 - a0) / dth * (num / den))

    if not bvalues:
        return float("inf"), []
    min_b = min(bvalues)
    _ = tol  # informational; caller compares min_b against -tol
    return min_b, bvalues


def _ce_geometry_residuals(
    ce: ControlSurface,
    r_template: np.ndarray,
    config: RaoSolverConfig,
) -> np.ndarray:
    """Endpoint, monotonicity, and boundary-state residuals for CE geometry."""
    if ce.x is None or len(ce.x) < 2:
        return np.zeros(0, dtype=float)

    L = _target_length(config.Rt, config.epsilon, config.length_pct)
    Re = math.sqrt(config.epsilon) * config.Rt
    x_scale = max(L, 1e-12)
    r_scale = max(Re, 1e-12)
    _ = r_template

    dx = np.diff(ce.x)
    dr = np.diff(ce.r)
    theta_scale = math.radians(1.0)
    if config.kernel_bd:
        _, bd_segment = calc_mdot_bd(
            config.kernel_bd, ce.kernel_d_fraction, config.gamma
        )
        d_node = bd_segment[-1]
        # ``ce.x[0]`` is structurally pinned to ``d_node.x`` by the
        # left-Mach-line integrator in ``_unpack_bvp`` (``x_start = D.x``),
        # so the ``(ce.x[0] - d_node.x)`` residual is identically zero and
        # is omitted to keep the residual stack non-degenerate.
        start = np.array([
            (float(ce.r[0]) - d_node.r) / r_scale,
            (float(ce.theta[0]) - d_node.theta) / theta_scale,
        ], dtype=float)
    else:
        start = np.zeros(0, dtype=float)

    endpoint = np.concatenate([
        start,
        np.array([
            (float(ce.x[-1]) - L) / x_scale,
            (float(ce.r[-1]) - Re) / r_scale,
        ], dtype=float),
    ])
    theta_scale = math.radians(1.0)

    # Phase 7: angle_boundary_mode controls how θ_N / θ_E enter the
    # residual stack.  "free" leaves them alone (chart values are only
    # seeds); "chart_soft" adds a small (1e-3 weight) nudge for
    # debugging; "chart_hard" pins to chart at unit weight + deprecation
    # warning (legacy pre-Phase-4 behaviour).
    mode = getattr(config, "angle_boundary_mode", "free")
    if mode in ("chart_soft", "chart_hard"):
        try:
            from raosim.nozzle_geometry import lookup_angles
            theta_n_deg, theta_e_deg = lookup_angles(config.epsilon, config.length_pct)
        except Exception:
            theta_n_deg = config.thetaN_guess_deg
            theta_e_deg = max(0.0, 0.5 * config.thetaN_guess_deg)
        if mode == "chart_hard":
            import warnings as _warnings
            _warnings.warn(
                "angle_boundary_mode='chart_hard' pins θ_N/θ_E to chart "
                "values, contaminating the chart benchmark.  Use 'free' "
                "for benchmarking.",
                DeprecationWarning, stacklevel=4,
            )
            anchor_weight = 1.0
        else:  # "chart_soft"
            anchor_weight = 1e-3
        flow_boundary = anchor_weight * np.array([
            (float(ce.theta[0]) - math.radians(theta_n_deg)) / theta_scale,
            (float(ce.theta[-1]) - math.radians(theta_e_deg)) / theta_scale,
        ], dtype=float)
    else:  # "free" (default)
        flow_boundary = np.zeros(0, dtype=float)

    monotonic = np.concatenate([
        np.maximum(-dx, 0.0) / x_scale,
        np.maximum(-dr, 0.0) / r_scale,
    ])
    return np.concatenate([endpoint, flow_boundary, monotonic])


def _enabled_residual_blocks(config: RaoSolverConfig) -> set[str]:
    blocks = DEFAULT_RAO_RESIDUAL_BLOCKS if config.residual_blocks is None else config.residual_blocks
    unknown = set(blocks).difference(ALL_RAO_RESIDUAL_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown Rao residual block(s): {sorted(unknown)}")
    return set(blocks)


def _filter_group(name: str, values: np.ndarray, active: set[str]) -> np.ndarray:
    if name in active:
        return np.asarray(values, dtype=float)
    return np.zeros(0, dtype=float)


def _rao_bvp_residual_groups(
    u: np.ndarray,
    r: np.ndarray,
    config: RaoSolverConfig,
) -> RaoResidualGroups:
    n_wall = config.n_wall if config.couple_wall else 0
    ce, wall = _unpack_bvp(
        u, r, n_wall=n_wall,
        kernel_bd=config.kernel_bd, gamma=config.gamma,
    )
    _, _, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_val, mdot_target, mdot_ref, _ = _mass_closure_fluxes(ce, config)
    L_target = _target_length(config.Rt, config.epsilon, config.length_pct)

    stat = _stationarity_matrix(ce, config.gamma, config.pa_over_p0)
    stat_scale = max(config.Rt, 1e-9)
    stat_res = stat.ravel() / stat_scale

    trans = transversality_residual(
        float(ce.M[-1]), float(ce.theta[-1]), float(ce.phi[-1]),
        float(ce.r[-1]), config.gamma, ce.lambda2, ce.lambda3,
        config.pa_over_p0,
    )
    trans_scale = max(abs(thrust_integrand(
        float(ce.M[-1]), float(ce.theta[-1]), float(ce.phi[-1]),
        float(ce.r[-1]), config.gamma, config.pa_over_p0,
    )), 1e-9)

    if ce.x is None:
        phi_theta_gap = (ce.theta + math.radians(0.25)) - ce.phi
        incidence_penalty = np.maximum(phi_theta_gap, 0.0) / math.radians(0.25)
        phi_smooth = np.diff(ce.phi, n=2) / math.radians(2.0) if len(ce.phi) > 2 else np.zeros(0)
    else:
        incidence_penalty = np.zeros(0, dtype=float)
        phi_smooth = np.zeros(0, dtype=float)
    mach_monotonic_penalty = np.maximum(-np.diff(ce.M), 0.0) / 0.05
    regularization = _ce_smoothness_regularization(ce, config.gamma)
    moc_cplus, moc_cminus = _ce_axisymmetric_compatibility_residual_groups(ce, config.gamma)
    ce_geometry = _ce_geometry_residuals(ce, r, config)
    algebraic_stat = _rao_algebraic_stationarity_residuals(ce, config.gamma)
    left_mach = _rao_left_mach_geometry_residuals(ce)
    penalties = np.concatenate([
        incidence_penalty,
        mach_monotonic_penalty,
        0.1 * phi_smooth,
    ])
    active = _enabled_residual_blocks(config)

    # Phase 6 coupled-wall residuals (no-op when wall is None).
    if wall is not None:
        wall_blocks = _coupled_wall_residuals(ce, wall, config)
    else:
        wall_blocks = {
            "wall_endpoint": np.zeros(0),
            "wall_tangency": np.zeros(0),
            "cplus_ce_to_wall": np.zeros(0),
            "wall_intersection": np.zeros(0),
        }

    return RaoResidualGroups(
        mass=_filter_group(
            "mass",
            np.array([(mdot_val - mdot_target) / max(mdot_ref, 1e-12)]),
            active,
        ),
        length=_filter_group(
            "length",
            np.array([(L_val - L_target) / max(L_target, 1e-12)]),
            active,
        ),
        transversality=_filter_group("transversality", np.array([trans / trans_scale]), active),
        stationarity=_filter_group("stationarity", stat_res, active),
        # Phase 4+: the Rao physics constraints (axisymmetric C+/C-
        # compatibility, algebraic stationarity, left-Mach geometry) are
        # all unit-scaled.  ``PHYSICS_WEIGHT`` is the unified knob for
        # the three derivative-form residuals (algebraic + C+/C-);
        # left_mach stays at unit weight because it is geometric and the
        # residual is already normalised by segment chord length.  This
        # keeps physics above the smoothness regularization (0.02) and
        # rampable as a group when the kernel quality improves.
        algebraic_stationarity=_filter_group(
            "algebraic_stationarity", PHYSICS_WEIGHT * algebraic_stat, active,
        ),
        left_mach=_filter_group("left_mach", left_mach, active),
        moc_cplus=_filter_group("moc_cplus", PHYSICS_WEIGHT * moc_cplus, active),
        moc_cminus=_filter_group("moc_cminus", PHYSICS_WEIGHT * moc_cminus, active),
        ce_geometry=_filter_group("ce_geometry", ce_geometry, active),
        regularization=_filter_group("regularization", 0.02 * regularization, active),
        penalties=_filter_group("penalties", penalties, active),
        # Phase 6 wall blocks at unit weight: the coupled-wall residuals
        # are geometric and tangency-based, so they live above smoothness
        # but in the same scale band as the physics blocks above.
        wall_endpoint=_filter_group("wall_endpoint", wall_blocks["wall_endpoint"], active),
        wall_tangency=_filter_group("wall_tangency", wall_blocks["wall_tangency"], active),
        cplus_ce_to_wall=_filter_group(
            "cplus_ce_to_wall",
            PHYSICS_WEIGHT * wall_blocks["cplus_ce_to_wall"], active,
        ),
        wall_intersection=_filter_group(
            "wall_intersection", wall_blocks["wall_intersection"], active,
        ),
    )


def _scaled_rao_bvp_residual(
    u: np.ndarray,
    r: np.ndarray,
    config: RaoSolverConfig,
    *,
    return_groups: bool = False,
) -> np.ndarray | RaoResidualGroups:
    groups = _rao_bvp_residual_groups(u, r, config)
    if return_groups:
        return groups
    return groups.flat()


def _build_residual_report(
    residual_vector: np.ndarray,
    ce: ControlSurface,
    config: RaoSolverConfig,
    r_template: np.ndarray,
    *,
    wall_tangency_rms: float | None = None,
    crossings: int = 0,
    wall: WallSurface | None = None,
) -> RaoResidualReport:
    _, _, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_val, mdot_target, mdot_ref, _ = _mass_closure_fluxes(ce, config)
    L_target = _target_length(config.Rt, config.epsilon, config.length_pct)
    stat = _stationarity_matrix(ce, config.gamma, config.pa_over_p0)
    regularization = _ce_smoothness_regularization(ce, config.gamma)
    algebraic_stat = _rao_algebraic_stationarity_residuals(ce, config.gamma)
    left_mach = _rao_left_mach_geometry_residuals(ce)
    groups = _rao_bvp_residual_groups(
        _pack_bvp(ce, ce.lambda2, ce.lambda3, ce.log_C, wall=wall),
        r_template,
        config,
    )
    trans = transversality_residual(
        float(ce.M[-1]), float(ce.theta[-1]), float(ce.phi[-1]),
        float(ce.r[-1]), config.gamma, ce.lambda2, ce.lambda3,
        config.pa_over_p0,
    )
    trans_scale = max(abs(thrust_integrand(
        float(ce.M[-1]), float(ce.theta[-1]), float(ce.phi[-1]),
        float(ce.r[-1]), config.gamma, config.pa_over_p0,
    )), 1e-9)
    return RaoResidualReport(
        max_scaled=float(np.max(np.abs(residual_vector))) if residual_vector.size else 0.0,
        rms_scaled=float(np.sqrt(np.mean(residual_vector**2))) if residual_vector.size else 0.0,
        mass_residual_rel=float((mdot_val - mdot_target) / max(mdot_ref, 1e-12)),
        length_residual_rel=float((L_val - L_target) / max(L_target, 1e-12)),
        stationarity_rms=float(np.sqrt(np.mean(stat**2))) if stat.size else 0.0,
        algebraic_stationarity_rms=(
            float(np.sqrt(np.mean(algebraic_stat**2)))
            if algebraic_stat.size else 0.0
        ),
        left_mach_rms=(
            float(np.sqrt(np.mean(left_mach**2))) if left_mach.size else 0.0
        ),
        regularization_rms=(
            float(np.sqrt(np.mean(regularization**2)))
            if regularization.size else 0.0
        ),
        transversality_scaled=float(trans / trans_scale),
        wall_tangency_rms=wall_tangency_rms,
        characteristic_crossings=int(crossings),
        group_summaries=groups.summaries(),
    )


def construct_wall_from_ce_raw(
    Rt: float,
    epsilon: float,
    gamma: float,
    ce: ControlSurface,
    Ln: float,
    n_char: int = 30,
) -> tuple[np.ndarray, dict]:
    """
    Construct raw CE-driven wall points without export resampling.

    The returned points are the only geometry used for compatibility
    diagnostics. Any later interpolation is display/export-only.
    """
    x_raw, r_raw, diagnostics = _construct_wall_from_ce(
        Rt, epsilon, gamma, ce, Ln, n_char, resample_for_export=False
    )
    return np.column_stack([x_raw, r_raw]), diagnostics


def _ce_interp_node(ce: ControlSurface, tau: float) -> FlowNode:
    """Interpolate a geometry-backed CE at normalized arc parameter tau."""
    if ce.x is None:
        nodes = _control_surface_flow_nodes(ce)
        x = np.asarray([p.x for p in nodes], dtype=float)
    else:
        x = np.asarray(ce.x, dtype=float)
    r = np.asarray(ce.r, dtype=float)
    M = np.asarray(ce.M, dtype=float)
    theta = np.asarray(ce.theta, dtype=float)
    if len(r) < 2:
        return FlowNode(float(x[0]), float(r[0]), max(float(M[0]), 1.001), float(theta[0]))

    ds = np.hypot(np.diff(x), np.diff(r))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 1e-14:
        q = np.linspace(0.0, 1.0, len(r))
    else:
        q = s / s[-1]
    t = float(np.clip(tau, 0.0, 1.0))
    return FlowNode(
        x=float(np.interp(t, q, x)),
        r=float(np.interp(t, q, r)),
        M=max(float(np.interp(t, q, M)), 1.001),
        theta=float(np.interp(t, q, theta)),
    )


def _hermite_wall_seed(
    x_wall: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    theta_start: float,
    theta_end: float,
) -> np.ndarray:
    """Cubic Hermite wall radius seed with specified endpoint slopes."""
    x0, r0 = start
    x1, r1 = end
    length = max(x1 - x0, 1e-12)
    t = np.clip((x_wall - x0) / length, 0.0, 1.0)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * r0
        + h10 * length * math.tan(theta_start)
        + h01 * r1
        + h11 * length * math.tan(theta_end)
    )


def solve_wall_from_ce_coupled(
    ce: ControlSurface,
    config: RaoSolverConfig,
    n_wall: int = 24,
) -> tuple[np.ndarray, dict]:
    """
    Minimal coupled wall strip solve from a closed CE.

    This is intentionally smaller than the full Phase-6 characteristic-net BVP:
    wall x locations and endpoints are boundary data, while interior radii,
    wall flow state, and the CE source parameter for each wall node are solved
    together.  The old sequential wall constructor remains available as a
    legacy diagnostic via ``wall_method='legacy'``.
    """
    if least_squares is None:
        raise RuntimeError("Coupled wall solve requires scipy.optimize.least_squares")
    if ce.x is None:
        raise ValueError("Coupled wall solve requires geometry-backed CE x coordinates")
    if n_wall < 4:
        raise ValueError("n_wall must be at least 4")

    Rt = config.Rt
    gamma = config.gamma
    Re = math.sqrt(config.epsilon) * Rt
    Rd = config.throat_downstream_radius_factor * Rt
    L = _target_length(Rt, config.epsilon, config.length_pct)
    theta_n, theta_e = _design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg
    )
    Nx = Rd * math.sin(theta_n)
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))

    x_wall = np.linspace(Nx, L, n_wall)
    r_seed = _hermite_wall_seed(
        x_wall, (Nx, Ny), (L, Re), theta_n, theta_e
    )
    r_seed = np.maximum.accumulate(np.clip(r_seed, min(Ny, Re), max(Ny, Re)))
    r_seed[0] = Ny
    r_seed[-1] = Re
    theta_seed = np.gradient(r_seed, x_wall, edge_order=1)
    theta_seed = np.arctan(theta_seed)
    tau_seed = np.linspace(0.0, 1.0, n_wall)
    M_seed = np.array([_ce_interp_node(ce, t).M for t in tau_seed], dtype=float)

    def radii_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
        weights = np.exp(np.clip(log_weights, -40.0, 40.0))
        total = max(float(np.sum(weights)), 1e-30)
        frac = np.cumsum(weights) / total
        r = np.empty(n_wall, dtype=float)
        r[0] = Ny
        r[1:] = Ny + (Re - Ny) * frac
        r[-1] = Re
        return r

    def unpack(u):
        m = n_wall - 1
        log_weights = u[:m]
        theta = u[m:m + n_wall]
        M = u[m + n_wall:m + 2 * n_wall]
        tau = u[m + 2 * n_wall:m + 3 * n_wall]
        r = radii_from_log_weights(log_weights)
        return r, theta, M, tau

    def pack(log_weights, theta, M, tau):
        return np.concatenate([log_weights, theta, M, tau])

    x_scale = max(L - Nx, 1e-12)
    r_scale = max(Re, 1e-12)
    theta_scale = math.radians(1.0)

    def residual(u):
        r_wall, theta_wall, M_wall, tau = unpack(u)
        cminus = []
        slope = []
        for xw, rw, Mw, thw, tw in zip(x_wall, r_wall, M_wall, theta_wall, tau):
            p_ce = _ce_interp_node(ce, float(tw))
            p_w = FlowNode(float(xw), float(rw), max(float(Mw), 1.001), float(thw))
            cminus.append(residual_Cminus_axisym(p_ce, p_w, gamma) / theta_scale)
            theta_avg = 0.5 * (p_ce.theta + p_w.theta)
            mu_avg = 0.5 * (p_ce.mu + p_w.mu)
            line = (p_w.r - p_ce.r) - math.tan(theta_avg - mu_avg) * (p_w.x - p_ce.x)
            slope.append(line / r_scale)

        dx = np.diff(x_wall)
        dr = np.diff(r_wall)
        theta_mid = 0.5 * (theta_wall[:-1] + theta_wall[1:])
        tangency = (dr - dx * np.tan(theta_mid)) / r_scale
        boundary_theta = np.array([
            (theta_wall[0] - theta_n) / theta_scale,
            (theta_wall[-1] - theta_e) / theta_scale,
        ])
        monotonic = np.concatenate([
            np.maximum(-dr, 0.0) / r_scale,
            np.maximum(-np.diff(tau), 0.0),
        ])
        tau_shape = 0.02 * (tau - tau_seed)
        return np.concatenate([
            np.asarray(cminus, dtype=float),
            np.asarray(slope, dtype=float),
            tangency,
            boundary_theta,
            monotonic,
            tau_shape,
        ])

    dr_seed = np.maximum(np.diff(r_seed), 1e-9 * max(Re - Ny, 1e-12))
    log_weight_seed = np.log(dr_seed / max(float(np.mean(dr_seed)), 1e-30))
    u0 = pack(log_weight_seed, theta_seed, M_seed, tau_seed)
    lower = pack(
        np.full(n_wall - 1, -20.0),
        np.full(n_wall, math.radians(-20.0)),
        np.full(n_wall, 1.001),
        np.zeros(n_wall),
    )
    upper = pack(
        np.full(n_wall - 1, 20.0),
        np.full(n_wall, math.radians(70.0)),
        np.full(n_wall, max(12.0, float(np.max(ce.M)) * 1.5)),
        np.ones(n_wall),
    )

    result = least_squares(
        residual,
        u0,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=max(200, config.max_nfev),
    )
    r_wall, theta_wall, M_wall, tau = unpack(result.x)
    wall = np.column_stack([x_wall, r_wall])
    final_res = residual(result.x)

    cminus_vals = []
    slope_vals = []
    for xw, rw, Mw, thw, tw in zip(x_wall, r_wall, M_wall, theta_wall, tau):
        p_ce = _ce_interp_node(ce, float(tw))
        p_w = FlowNode(float(xw), float(rw), max(float(Mw), 1.001), float(thw))
        cminus_vals.append(residual_Cminus_axisym(p_ce, p_w, gamma) / theta_scale)
        theta_avg = 0.5 * (p_ce.theta + p_w.theta)
        mu_avg = 0.5 * (p_ce.mu + p_w.mu)
        line = (p_w.r - p_ce.r) - math.tan(theta_avg - mu_avg) * (p_w.x - p_ce.x)
        slope_vals.append(line / r_scale)

    dx = np.diff(x_wall)
    dr = np.diff(r_wall)
    theta_mid = 0.5 * (theta_wall[:-1] + theta_wall[1:])
    tangency_vals = (dr - dx * np.tan(theta_mid)) / r_scale
    wall_angles = np.arctan2(dr, dx)
    wall_tangency_rms = float(np.sqrt(np.mean((wall_angles - theta_mid) ** 2)))
    endpoint_dx = float(wall[-1, 0] - L)
    endpoint_dr = float(wall[-1, 1] - Re)
    strip_success = (
        diagnostics_success := (
            float(np.max(np.abs(final_res))) if final_res.size else 0.0
        )
    ) <= 1e-2
    strip_success = (
        strip_success
        and abs(endpoint_dx) / max(L, 1e-12) <= 1e-3
        and abs(endpoint_dr) / max(Re, 1e-12) <= 1e-3
        and wall_tangency_rms < math.radians(0.25)
        and int(np.sum(np.diff(x_wall) <= 0.0)) == 0
        and int(np.sum(np.diff(r_wall) < -1e-10)) == 0
    )
    diagnostics = {
        "method": "coupled_wall_strip",
        "fallback_used": False,
        "postprocessed": False,
        "moc_compatibility_preserved": False,
        "success": bool(strip_success),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "max_residual": diagnostics_success,
        "endpoint_dx": endpoint_dx,
        "endpoint_dr": endpoint_dr,
        "wall_tangency_rms": wall_tangency_rms,
        "wall_tangency_residual_rms": float(np.sqrt(np.mean(tangency_vals**2))),
        "cminus_rms": float(np.sqrt(np.mean(np.asarray(cminus_vals) ** 2))),
        "slope_rms": float(np.sqrt(np.mean(np.asarray(slope_vals) ** 2))),
        "monotonic_x_violations": int(np.sum(np.diff(x_wall) <= 0.0)),
        "monotonic_r_violations": int(np.sum(np.diff(r_wall) < -1e-10)),
        "tau_monotonic_violations": int(np.sum(np.diff(tau) < -1e-10)),
        "clamp_hits": 0,
        "nonmonotonic_x_drops": 0,
        "ce_source_tau": tau.tolist(),
        "wall_mach": M_wall.tolist(),
        "wall_theta": theta_wall.tolist(),
        "warnings": [],
    }
    diagnostics["wall_strip_success"] = bool(strip_success)
    if not diagnostics["wall_strip_success"]:
        diagnostics["warnings"].append(
            "Coupled wall strip solve did not meet compatibility/closure gates."
        )
    return wall, diagnostics


def resample_wall_for_export(
    raw_wall: np.ndarray,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    n: int = 100,
    residual_tol: float = 2e-3,
) -> tuple[np.ndarray, dict]:
    """Create a plotting/CAD polyline from raw wall points."""
    diagnostics = {
        "resampled_for_export": True,
        "endpoint_enforced_for_export": False,
        "start_radius_delta": 0.0,
        "end_radius_delta": 0.0,
        "monotonic_cleanup_for_export": False,
    }
    if raw_wall.shape[0] < 2:
        raise ValueError("raw_wall needs at least two points")
    if residual_tol < 0.0:
        raise ValueError("residual_tol must be non-negative")
    order = np.argsort(raw_wall[:, 0])
    wall = raw_wall[order]
    x_unique, unique_idx = np.unique(wall[:, 0], return_index=True)
    r_unique = wall[unique_idx, 1]
    x_export = np.linspace(start[0], end[0], n)
    r_export = np.interp(x_export, x_unique, r_unique)
    start_delta = abs(float(r_export[0]) - start[1])
    end_delta = abs(float(r_export[-1]) - end[1])
    diagnostics["start_radius_delta"] = start_delta
    diagnostics["end_radius_delta"] = end_delta

    radius_scale = max(abs(end[1]), abs(start[1]), 1e-12)
    endpoint_limit = residual_tol * radius_scale
    if start_delta > endpoint_limit or end_delta > endpoint_limit:
        diagnostics["endpoint_enforced_for_export"] = True
        raise RaoEndpointMismatchError(
            "Raw Rao/MOC wall endpoints do not close to target geometry: "
            f"start radius delta={start_delta:.6g} m, "
            f"end radius delta={end_delta:.6g} m, "
            f"limit={endpoint_limit:.6g} m."
        )
    if np.any(np.diff(r_export) < -1e-9):
        diagnostics["monotonic_cleanup_for_export"] = True
        r_export = np.maximum.accumulate(r_export)
        cleaned_start_delta = abs(float(r_export[0]) - start[1])
        cleaned_end_delta = abs(float(r_export[-1]) - end[1])
        diagnostics["start_radius_delta"] = cleaned_start_delta
        diagnostics["end_radius_delta"] = cleaned_end_delta
        if cleaned_start_delta > endpoint_limit or cleaned_end_delta > endpoint_limit:
            raise RaoEndpointMismatchError(
                "Export monotonic cleanup moved Rao/MOC wall endpoints outside "
                "the closure tolerance: "
                f"start radius delta={cleaned_start_delta:.6g} m, "
                f"end radius delta={cleaned_end_delta:.6g} m, "
                f"limit={endpoint_limit:.6g} m."
            )
    return np.column_stack([x_export, r_export]), diagnostics


def _wall_tangency_rms(raw_wall: np.ndarray, ce: ControlSurface) -> float | None:
    if raw_wall.shape[0] < 3:
        return None
    dx = np.gradient(raw_wall[:, 0])
    dr = np.gradient(raw_wall[:, 1])
    wall_theta = np.arctan2(dr, dx)
    ce_theta = np.interp(
        raw_wall[:, 1],
        ce.r,
        ce.theta,
        left=float(ce.theta[0]),
        right=float(ce.theta[-1]),
    )
    return float(np.sqrt(np.mean((wall_theta - ce_theta) ** 2)))


def characteristic_net_links(rows: list[CharRow]) -> dict[str, list[MOCNetLink]]:
    """
    Return explicit characteristic-family links from a marched MOC net.

    ``CharRow.all_points()`` is only a row ordering convenience.  The true
    topology is the parent-child construction used by ``march_coupled_net``:
    interior points are C+ from the lower parent and C- from the upper parent,
    axis points are C- symmetry hits, and wall points are C+ wall hits.
    """
    links: dict[str, list[MOCNetLink]] = {
        "cplus": [],
        "cminus": [],
        "axis": [],
        "wall": [],
    }
    if len(rows) < 2:
        return links

    prev_pts = rows[0].all_points()
    for row_idx, row in enumerate(rows[1:], start=1):
        curr_pts = row.all_points()
        if not prev_pts or not curr_pts:
            prev_pts = curr_pts
            continue

        child_offset = 1 if row.axis is not None else 0

        if row.axis is not None:
            parent_idx = 1 if len(prev_pts) > 1 and prev_pts[0].r < 1e-10 else 0
            if parent_idx < len(prev_pts):
                link = MOCNetLink(
                    row=row_idx,
                    family="cminus",
                    role="axis",
                    parent=prev_pts[parent_idx],
                    child=row.axis,
                    parent_index=parent_idx,
                    child_index=0,
                )
                links["cminus"].append(link)
                links["axis"].append(link)

        for j, child in enumerate(row.interior):
            if j + 1 >= len(prev_pts):
                break
            links["cplus"].append(MOCNetLink(
                row=row_idx,
                family="cplus",
                role="interior",
                parent=prev_pts[j],
                child=child,
                parent_index=j,
                child_index=child_offset + j,
            ))
            links["cminus"].append(MOCNetLink(
                row=row_idx,
                family="cminus",
                role="interior",
                parent=prev_pts[j + 1],
                child=child,
                parent_index=j + 1,
                child_index=child_offset + j,
            ))

        if row.wall is not None and len(prev_pts) >= 2:
            link = MOCNetLink(
                row=row_idx,
                family="cplus",
                role="wall",
                parent=prev_pts[-2],
                child=row.wall,
                parent_index=len(prev_pts) - 2,
                child_index=len(curr_pts) - 1,
            )
            links["cplus"].append(link)
            links["wall"].append(link)

        prev_pts = curr_pts

    return links


def characteristic_net_compatibility_residuals(
    rows: list[CharRow],
    gamma: float,
) -> dict[str, np.ndarray]:
    """
    Reconstruct C+/C- parent-child residuals from a marched MOC net.

    The row topology matches ``march_coupled_net``:
      - axis child is reached by C- from the lower/axis-side parent,
      - interior children are C+ from lower parent and C- from upper parent,
      - wall child is reached by C+ from the near-wall parent.
    """
    links = characteristic_net_links(rows)
    scale = math.radians(1.0)
    return {
        "cplus": np.asarray([
            residual_Cplus_axisym(
                link.parent.to_flow_node(), link.child.to_flow_node(), gamma
            )
            for link in links["cplus"]
        ], dtype=float) / scale,
        "cminus": np.asarray([
            residual_Cminus_axisym(
                link.parent.to_flow_node(), link.child.to_flow_node(), gamma
            )
            for link in links["cminus"]
        ], dtype=float) / scale,
    }


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0


def _maxabs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def summarize_characteristic_net_compatibility(
    rows: list[CharRow],
    gamma: float,
) -> list[dict]:
    """Return max/RMS summaries for C+/C- residuals in a marched MOC net."""
    residuals = characteristic_net_compatibility_residuals(rows, gamma)
    return [
        summarize_group("net_moc_cplus", residuals["cplus"]),
        summarize_group("net_moc_cminus", residuals["cminus"]),
    ]


def moc_net_compatibility_report(
    rows: list[CharRow],
    solved_wall: np.ndarray,
    wall: SplineWall,
    gamma: float,
    *,
    x_scale: float,
    r_scale: float,
    tol: float,
) -> MOCNetCompatibilityReport:
    """Build a detailed compatibility report for a forward MOC net."""
    links = characteristic_net_links(rows)
    compat = characteristic_net_compatibility_residuals(rows, gamma)
    cplus = compat["cplus"]
    cminus = compat["cminus"]

    intersection: list[float] = []
    wall_tangency: list[float] = []
    wall_dx: list[float] = []
    wall_dr: list[float] = []
    bad_rows: set[int] = set()

    for link, value in zip(links["cplus"], cplus):
        if abs(float(value)) > tol:
            bad_rows.add(link.row)
    for link, value in zip(links["cminus"], cminus):
        if abs(float(value)) > tol:
            bad_rows.add(link.row)

    prev_pts = rows[0].all_points() if rows else []
    for row_idx, row in enumerate(rows[1:], start=1):
        curr_pts = row.all_points()
        row_residuals: list[float] = []

        for j, child in enumerate(row.interior):
            if j + 1 >= len(prev_pts):
                continue
            geom = residual_intersection(
                prev_pts[j].to_flow_node(),
                prev_pts[j + 1].to_flow_node(),
                child.to_flow_node(),
                x_scale,
                r_scale,
            )
            intersection.extend(geom.tolist())
            row_residuals.extend(np.abs(geom).tolist())

        if row.wall is not None:
            boundary_dr = (row.wall.r - wall.r(row.wall.x)) / max(r_scale, 1e-12)
            wall_dr.append(boundary_dr)
            wall_t = (row.wall.theta - wall.theta(row.wall.x)) / math.radians(1.0)
            wall_tangency.append(wall_t)
            row_residuals.extend([abs(boundary_dr), abs(wall_t)])

        if row_residuals and max(row_residuals) > tol:
            bad_rows.add(row_idx)
        prev_pts = curr_pts

    valid_rows = [
        row for row in rows[1:]
        if row.wall is not None and math.isfinite(row.wall.x) and math.isfinite(row.wall.r)
    ]
    if valid_rows and solved_wall.size:
        solved_x = np.asarray(solved_wall[:, 0], dtype=float)
        solved_r = np.asarray(solved_wall[:, 1], dtype=float)
        solved_order = np.argsort(solved_x)
        solved_x = solved_x[solved_order]
        solved_r = solved_r[solved_order]
        for row in valid_rows:
            wx = float(row.wall.x)
            if wx < solved_x[0] or wx > solved_x[-1]:
                boundary_dx = min(abs(wx - solved_x[0]), abs(wx - solved_x[-1])) / max(x_scale, 1e-12)
                boundary_dr = abs(float(row.wall.r) - float(np.interp(
                    np.clip(wx, solved_x[0], solved_x[-1]), solved_x, solved_r
                ))) / max(r_scale, 1e-12)
            else:
                boundary_dx = 0.0
                boundary_dr = (float(row.wall.r) - float(np.interp(wx, solved_x, solved_r))) / max(r_scale, 1e-12)
            wall_dx.append(boundary_dx)
            wall_dr.append(boundary_dr)

    crossing_count = check_characteristic_crossing(rows)
    cplus_max = _maxabs(cplus)
    cminus_max = _maxabs(cminus)
    intersection_arr = np.asarray(intersection, dtype=float)
    wall_dx_arr = np.asarray(wall_dx, dtype=float)
    wall_dr_arr = np.asarray(wall_dr, dtype=float)
    wall_boundary = np.concatenate([wall_dx_arr, wall_dr_arr]) if (wall_dx_arr.size or wall_dr_arr.size) else np.zeros(0)
    wall_t_arr = np.asarray(wall_tangency, dtype=float)

    passes = (
        cplus_max <= tol
        and cminus_max <= tol
        and _maxabs(intersection_arr) <= tol
        and _maxabs(wall_boundary) <= tol
        and _maxabs(wall_t_arr) <= tol
        and crossing_count == 0
        and not bad_rows
    )
    return MOCNetCompatibilityReport(
        cplus_rms=_rms(cplus),
        cminus_rms=_rms(cminus),
        cplus_max=cplus_max,
        cminus_max=cminus_max,
        intersection_rms=_rms(intersection_arr),
        intersection_max=_maxabs(intersection_arr),
        wall_boundary_rms=_rms(wall_boundary),
        wall_boundary_dx_rms=_rms(wall_dx_arr),
        wall_boundary_dr_rms=_rms(wall_dr_arr),
        wall_boundary_dx_max=_maxabs(wall_dx_arr),
        wall_boundary_dr_max=_maxabs(wall_dr_arr),
        wall_tangency_rms=_rms(wall_t_arr),
        wall_tangency_max=_maxabs(wall_t_arr),
        crossings=crossing_count,
        bad_rows=sorted(bad_rows),
        passes=bool(passes),
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray,
                        c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def characteristic_net_segments(
    rows: list[CharRow],
    *,
    families: tuple[str, ...] = ("cplus", "cminus"),
) -> list[MOCNetLink]:
    """Return true characteristic segments, grouped by explicit topology."""
    links = characteristic_net_links(rows)
    segments: list[MOCNetLink] = []
    for family in families:
        segments.extend(links.get(family, []))
    return segments


def _same_net_point(a: CharPoint, b: CharPoint, tol: float = 1e-12) -> bool:
    return a is b or math.hypot(a.x - b.x, a.r - b.r) <= tol


def _links_share_endpoint(a: MOCNetLink, b: MOCNetLink) -> bool:
    return (
        _same_net_point(a.parent, b.parent)
        or _same_net_point(a.parent, b.child)
        or _same_net_point(a.child, b.parent)
        or _same_net_point(a.child, b.child)
    )


def check_characteristic_crossing(rows: list[CharRow]) -> int:
    """Count crossings between true C+/C- characteristic segments."""
    segments = characteristic_net_segments(rows)
    crossings = 0
    points: list[tuple[np.ndarray, np.ndarray]] = []
    kept_segments: list[MOCNetLink] = []
    for segment in segments:
        a = np.array([segment.parent.x, segment.parent.r], dtype=float)
        b = np.array([segment.child.x, segment.child.r], dtype=float)
        if np.linalg.norm(b - a) <= 1e-14:
            continue
        points.append((a, b))
        kept_segments.append(segment)

    for i, (a, b) in enumerate(points):
        for j in range(i + 1, len(points)):
            if _links_share_endpoint(kept_segments[i], kept_segments[j]):
                continue
            c, d = points[j]
            if _segments_intersect(a, b, c, d):
                crossings += 1
    return crossings


def solve_rao_bvp(config: RaoSolverConfig) -> RaoSolution:
    """
    Solve the finite-dimensional Rao variational/MOC residual system.

    This is the new auditable path: least-squares solves the global
    kernel-BD mass closure, length, stationarity, left-Mach geometry, and
    CE compatibility residuals together.  MOC wall construction is evaluated
    on the raw solution and can downgrade reliability if it does not close
    cleanly.
    """
    if config.Rt <= 0.0:
        raise ValueError("Rt must be positive")
    if config.epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")
    if config.pa_over_p0 < 0.0:
        raise ValueError("pa_over_p0 must be non-negative")
    if config.n_control < 8:
        raise ValueError("n_control must be at least 8")

    ce0, kernel_bd_seed, topology_seed = _initial_ce_from_kernel(config)
    solve_config = replace(
        config,
        kernel_bd=tuple(kernel_bd_seed) if kernel_bd_seed else None,
    )
    # kernel_points is the legacy CharPoint list used downstream by the
    # construct_wall_from_ce_raw / wall MOC march; convert from the new
    # FlowNode kernel BD via the existing _make_point helper.
    kernel_points = [
        _make_point(float(p.x), float(p.r), float(p.theta), max(float(p.M), 1.000001), config.gamma)
        for p in kernel_bd_seed
    ]
    log_C0 = _seed_log_C_from_ce(ce0, config.gamma)
    ce0.log_C = log_C0
    n = len(ce0.r)
    try:
        Me = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me = 8.0
    L_target_value = _target_length(config.Rt, config.epsilon, config.length_pct)
    Re_value = math.sqrt(config.epsilon) * config.Rt

    # Phase 6: optional wall unknowns appended after the CE arrays.
    wall_seed: WallSurface | None = None
    if config.couple_wall:
        wall_seed = _initial_wall_guess(config, ce0, topology_seed)
        n_w = len(wall_seed.x)
        u0 = _pack_bvp(ce0, -0.5, 0.01, log_C0, wall=wall_seed)
    else:
        n_w = 0
        u0 = _pack_bvp(ce0, -0.5, 0.01, log_C0)

    # CE unknowns (after the left-Mach-by-construction refactor): only
    # ``M, theta, r`` per node — ``x`` is reconstructed at unpack time
    # by integrating ``dx/dr = 1/tan(theta+mu)``.
    lower_parts = [
        np.full(n, 1.001),                  # M
        np.full(n, math.radians(-10.0)),    # theta
        np.full(n, 0.0),                    # r
    ]
    upper_parts = [
        np.full(n, max(12.0, 1.5 * Me)),
        np.full(n, math.radians(65.0)),
        np.full(n, 1.05 * Re_value),
    ]
    if n_w > 0:
        lower_parts.extend([
            np.full(n_w, 1.001),
            np.full(n_w, 0.0),  # theta >= 0 (wall does not turn inward)
            np.full(n_w, 0.0),
            np.full(n_w, config.Rt),
        ])
        upper_parts.extend([
            np.full(n_w, max(12.0, 1.5 * Me)),
            np.full(n_w, math.radians(45.0)),  # theta_N_max
            np.full(n_w, max(1.2 * L_target_value, 1e-9)),
            np.full(n_w, 1.05 * Re_value),
        ])
    lower_parts.append(np.array([-1e3, -1e3, -10.0, 0.0]))
    upper_parts.append(np.array([1e3, 1e3, 10.0, 1.0]))
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)

    if config.max_nfev <= 0:
        residual0 = _scaled_rao_bvp_residual(u0, ce0.r, solve_config)

        @dataclass
        class _InitialResidualOnly:
            x: np.ndarray
            success: bool
            message: str
            cost: float

        result = _InitialResidualOnly(
            x=u0,
            success=False,
            message="initial residual evaluation only; least_squares was skipped",
            cost=float(0.5 * np.dot(residual0, residual0)),
        )
    else:
        if least_squares is None:
            raise RuntimeError(
                "solve_rao_bvp requires scipy.optimize.least_squares when "
                "max_nfev > 0. Install scipy or set max_nfev=0 for an "
                "initial residual-only diagnostic."
            )
        result = least_squares(
            _scaled_rao_bvp_residual,
            u0,
            bounds=(lower, upper),
            args=(ce0.r, solve_config),
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=config.max_nfev,
        )
    n_wall_unknown = config.n_wall if config.couple_wall else 0
    ce, solved_wall = _unpack_bvp(
        result.x, ce0.r, n_wall=n_wall_unknown,
        kernel_bd=solve_config.kernel_bd, gamma=config.gamma,
    )
    residual_vector = _scaled_rao_bvp_residual(result.x, ce0.r, solve_config)
    F_val, _, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_val, mdot_target, mdot_ref, bd_segment = _mass_closure_fluxes(
        ce, solve_config
    )
    L_target = _target_length(config.Rt, config.epsilon, config.length_pct)
    ce.thrust = float(F_val)
    ce.objective = float(result.cost)
    ce.optimizer_success = bool(result.success)
    ce.solver_message = str(result.message)
    ce.mdot_target = float(mdot_target)
    ce.mdot_residual = float((mdot_val - mdot_target) / max(mdot_ref, 1e-12))
    ce.length_target = float(L_target)
    ce.length_residual = float(L_val - L_target)
    ce.transversality = float(transversality_residual(
        float(ce.M[-1]), float(ce.theta[-1]), float(ce.phi[-1]),
        float(ce.r[-1]), config.gamma, ce.lambda2, ce.lambda3,
        config.pa_over_p0,
    ))
    ce.residual_norm = float(np.max(np.abs(residual_vector)))

    raw_wall = np.empty((0, 2))
    char_net: list[CharRow] = []
    construction_diagnostics: dict = {
        "warnings": [],
        "postprocessed": False,
        "moc_compatibility_preserved": False,
    }
    wall_tangency_rms: float | None = None
    crossings = 0
    warnings: list[str] = []

    Re = math.sqrt(config.epsilon) * config.Rt
    Rd = config.throat_downstream_radius_factor * config.Rt
    theta_n, _theta_e_design = _design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg
    )
    Nx = Rd * math.sin(theta_n)
    Ny = config.Rt + Rd * (1.0 - math.cos(theta_n))
    if config.evaluate_moc:
        try:
            if config.wall_method == "coupled":
                raw_wall, construction_diagnostics = solve_wall_from_ce_coupled(
                    ce, config, n_wall=max(8, 2 * config.n_kernel)
                )
            elif config.wall_method == "legacy":
                raw_wall, construction_diagnostics = construct_wall_from_ce_raw(
                    config.Rt, config.epsilon, config.gamma, ce, L_target,
                    config.n_kernel,
                )
            else:
                raise ValueError("wall_method must be 'coupled' or 'legacy'")
            if raw_wall.shape[0] >= 3:
                slope_start = math.tan(theta_n)
                slope_end = math.tan(_theta_e_design)
                wall = SplineWall(
                    raw_wall[:, 0],
                    raw_wall[:, 1],
                    slope_start=slope_start,
                    slope_end=slope_end,
                )
                starting = approximate_starting_line(
                    config.Rt,
                    config.throat_downstream_radius_factor * config.Rt,
                    max(theta_n, 1e-4),
                    config.gamma,
                    config.n_kernel,
                    method=config.starting_line_method,
                )
                char_net = march_coupled_net(starting, wall, config.gamma)
                crossings = check_characteristic_crossing(char_net)
                net_compatibility = summarize_characteristic_net_compatibility(
                    char_net, config.gamma
                )
                construction_diagnostics["net_compatibility"] = net_compatibility
                net_report = moc_net_compatibility_report(
                    char_net,
                    raw_wall,
                    wall,
                    config.gamma,
                    x_scale=max(L_target - Nx, 1e-12),
                    r_scale=max(Re, 1e-12),
                    tol=config.residual_tol,
                )
                construction_diagnostics["net_report"] = net_report.to_dict()
                if not net_report.passes:
                    construction_diagnostics["moc_compatibility_preserved"] = False
                    construction_diagnostics.setdefault("warnings", []).append(
                        "Forward MOC net compatibility residuals exceeded tolerance."
                    )
            wall_tangency_rms = construction_diagnostics.get("wall_tangency_rms")
            if wall_tangency_rms is None:
                wall_tangency_rms = _wall_tangency_rms(raw_wall, ce)
        except Exception as exc:
            warnings.append(f"Raw MOC wall construction failed: {exc}")
            construction_diagnostics = {
                "warnings": [str(exc)],
                "postprocessed": False,
                "moc_compatibility_preserved": False,
            }
            raw_wall = np.array([[Nx, Ny], [L_target, Re]], dtype=float)
    else:
        warnings.append("MOC wall evaluation skipped by solver configuration.")
        construction_diagnostics = {
            "warnings": ["MOC wall evaluation skipped."],
            "postprocessed": False,
            "moc_compatibility_preserved": False,
        }
        raw_wall = np.array([[Nx, Ny], [L_target, Re]], dtype=float)

    export_wall, export_diag = resample_wall_for_export(
        raw_wall,
        start=(Nx, Ny),
        end=(L_target, Re),
        residual_tol=config.residual_tol,
    )
    warnings.extend(construction_diagnostics.get("warnings", []))
    if export_diag.get("endpoint_enforced_for_export"):
        warnings.append("Wall endpoints were enforced only on export geometry.")
    if export_diag.get("monotonic_cleanup_for_export"):
        construction_diagnostics["postprocessed"] = True
        construction_diagnostics["moc_compatibility_preserved"] = False
        warnings.append("Export geometry required monotonic cleanup; raw solution is unchanged.")

    residuals = _build_residual_report(
        residual_vector, ce, solve_config, ce0.r,
        wall_tangency_rms=wall_tangency_rms,
        crossings=crossings,
        wall=solved_wall,
    )
    At = math.pi * config.Rt * config.Rt
    cf = F_val / max(At, 1e-12)
    try:
        Me_ideal = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
        Pe_over_p0 = isentropic_pressure_ratio(Me_ideal, config.gamma)
        cf_ideal = ideal_thrust_coefficient(
            Me_ideal, config.gamma, Pe_over_p0, config.pa_over_p0,
            config.epsilon,
        )
    except Exception:
        cf_ideal = float("nan")
    cf_rel_error = (
        (cf - cf_ideal) / cf_ideal
        if math.isfinite(cf_ideal) and abs(cf_ideal) > 1e-12
        else float("nan")
    )
    thrust_sanity_ok = (
        cf > 0.0
        and math.isfinite(cf_rel_error)
        and abs(cf_rel_error) <= 5e-3
    )
    bvp_ok = (
        bool(result.success)
        and residuals.max_scaled <= config.residual_tol
        and abs(residuals.mass_residual_rel) <= config.residual_tol
        and abs(residuals.length_residual_rel) <= config.residual_tol
    )
    no_postprocessing = (
        not construction_diagnostics.get("postprocessed", False)
        and not export_diag.get("endpoint_enforced_for_export", False)
        and not export_diag.get("monotonic_cleanup_for_export", False)
    )
    moc_ok = (
        no_postprocessing
        and construction_diagnostics.get("moc_compatibility_preserved", False)
        and wall_tangency_rms is not None
        and wall_tangency_rms < math.radians(0.25)
        and crossings == 0
    )

    boundary_min, _b_values = rao_valid_region(ce)
    valid_region_ok = boundary_min >= -config.residual_tol
    construction_diagnostics["boundary_min"] = float(boundary_min)
    construction_diagnostics["rao_region"] = (
        "valid_shock_free_region" if valid_region_ok
        else "invalid_short_nozzle_region"
    )
    construction_diagnostics["requires_discontinuous_exit_flow_model"] = (
        not valid_region_ok
    )
    construction_diagnostics["mass_closure"] = {
        "method": "kernel_bd_curve_flux",
        "ce_mass_flux": float(mdot_val),
        "kernel_bd_mass_flux": float(mdot_target),
        "kernel_bd_full_mass_flux": float(
            curve_mass_flux(solve_config.kernel_bd or (), config.gamma)
        ),
        "kernel_d_fraction": float(ce.kernel_d_fraction),
        "kernel_D": (
            {
                "x": float(bd_segment[-1].x),
                "r": float(bd_segment[-1].r),
                "M": float(bd_segment[-1].M),
                "theta": float(bd_segment[-1].theta),
            }
            if bd_segment else None
        ),
        "kernel_bd_nodes": len(solve_config.kernel_bd or ()),
        "kernel_bd_segment_nodes": len(bd_segment),
        "residual_scaled": float(
            (mdot_val - mdot_target) / max(mdot_ref, 1e-12)
        ),
    }
    construction_diagnostics["thrust_sanity"] = {
        "cf_surface": float(cf),
        "cf_ideal": float(cf_ideal),
        "cf_rel_error": float(cf_rel_error),
        "passes": bool(thrust_sanity_ok),
    }

    ce.converged = bool(bvp_ok)
    shock_free = crossings == 0
    if bvp_ok and moc_ok and valid_region_ok and thrust_sanity_ok:
        reliability = ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED
    elif moc_ok and valid_region_ok and thrust_sanity_ok:
        reliability = ContourReliability.MOC_COMPATIBLE
    else:
        reliability = ContourReliability.GEOMETRIC_APPROXIMATION

    if not bvp_ok:
        warnings.append(
            "Rao BVP residuals did not meet tolerance; solution is not "
            "variational-residual-solved."
        )
    if not moc_ok:
        warnings.append(
            "MOC closure/tangency/crossing diagnostics did not pass; do not "
            "treat this as a benchmarked Rao contour."
        )
    if not valid_region_ok:
        warnings.append(
            "Requested (epsilon, length_pct) lies outside the smooth-flow "
            f"Rao region (min boundary value {boundary_min:.3g}); the "
            "optimum-thrust contour is discontinuous and the variational "
            "construction is not applicable."
        )
    if not thrust_sanity_ok:
        warnings.append(
            "CE surface thrust coefficient failed the Phase 4 sanity gate; "
            "solution is not variational-residual-solved."
        )
    warnings.append(
        "Not hardware-qualified; requires published benchmark comparison, CFD, "
        "thermal/structural review, manufacturing review, inspection, and hot-fire data."
    )
    ce.warnings.extend(warnings)

    theta_e = math.atan2(export_wall[-1, 1] - export_wall[-2, 1],
                         export_wall[-1, 0] - export_wall[-2, 0])
    return RaoSolution(
        wall_raw=raw_wall,
        wall_export=export_wall,
        control_surface=ce,
        characteristic_net=char_net,
        kernel_points=kernel_points,
        theta_N=theta_n,
        theta_E=theta_e,
        thrust_coefficient=float(cf),
        residuals=residuals,
        reliability=reliability,
        converged=bool(bvp_ok and moc_ok),
        shock_free=shock_free,
        hardware_qualified=False,
        assumptions=RAO_MOC_ASSUMPTIONS,
        construction_diagnostics={
            **construction_diagnostics,
            "export": export_diag,
            "rao_topology": (
                {
                    "B": {"x": topology_seed.B.x, "r": topology_seed.B.r,
                          "M": topology_seed.B.M, "theta": topology_seed.B.theta},
                    "D": {"x": topology_seed.D.x, "r": topology_seed.D.r,
                          "M": topology_seed.D.M, "theta": topology_seed.D.theta},
                    "E": {"x": topology_seed.E.x, "r": topology_seed.E.r,
                          "M": topology_seed.E.M, "theta": topology_seed.E.theta},
                    "d_fraction": topology_seed.d_fraction,
                    "mass_BD": topology_seed.mass_BD,
                    "mass_DE": topology_seed.mass_DE,
                    "theta_B": topology_seed.theta_B,
                    "rao_stationarity_residual": topology_seed.rao_stationarity_residual,
                    "n_DE": len(topology_seed.DE),
                }
                if topology_seed is not None else None
            ),
        },
        warnings=_dedupe_strings(warnings),
        topology=topology_seed,
    )


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def rao_residual_ablation_matrix(
    config: RaoSolverConfig,
    *,
    cases: dict[str, tuple[str, ...]] | None = None,
    evaluate_moc: bool = False,
) -> list[dict]:
    """
    Run the same BVP setup with named residual-block subsets.

    This is a diagnostic tool for identifying which residual family first
    makes the current CE parameterization infeasible.  By default MOC wall
    construction is skipped so the matrix isolates the CE solve.
    """
    selected = RAO_RESIDUAL_ABLATIONS if cases is None else cases
    rows: list[dict] = []
    for name, blocks in selected.items():
        cfg = replace(config, residual_blocks=blocks, evaluate_moc=evaluate_moc)
        try:
            solution = solve_rao_bvp(cfg)
            rows.append({
                "case": name,
                "blocks": list(blocks),
                "raised": None,
                "reliability": solution.reliability.value,
                "max_scaled": solution.residuals.max_scaled,
                "rms_scaled": solution.residuals.rms_scaled,
                "group_summaries": solution.residuals.group_summaries,
                "warnings": list(solution.warnings),
            })
        except Exception as exc:
            rows.append({
                "case": name,
                "blocks": list(blocks),
                "raised": type(exc).__name__,
                "message": str(exc),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def _full_cone_length(Rt: float, epsilon: float) -> float:
    Re = math.sqrt(epsilon) * Rt
    return (Re - Rt) / math.tan(math.radians(15.0))


def rao_variational_moc_contour(
    Rt: float,
    epsilon: float,
    gamma: float = 1.4,
    pa_over_p0: float = 0.0,
    length_pct: float = 80.0,
    n_control: int = 12,
    n_kernel: int = 12,
    thetaN_guess_deg: float = 30.0,
    throat_downstream_radius_factor: float = 0.382,
    starting_line_method: str = "kliegel_levine",
    max_nfev: int = 200,
    residual_tol: float = 2e-3,
    evaluate_moc: bool = True,
    convergent_half_angle_deg: float = 45.0,
    Ru_factor: float = 1.5,
    return_solution: bool = False,
) -> dict | RaoSolution:
    """
    Generate an auditable Rao variational/MOC contour attempt.

    This path solves a global finite-dimensional residual system and separates
    raw MOC wall points from export geometry.  It remains non-hardware-qualified
    until literature benchmarks pass.
    """
    config = RaoSolverConfig(
        Rt=Rt,
        epsilon=epsilon,
        gamma=gamma,
        pa_over_p0=pa_over_p0,
        length_pct=length_pct,
        throat_downstream_radius_factor=throat_downstream_radius_factor,
        thetaN_guess_deg=thetaN_guess_deg,
        n_control=n_control,
        n_kernel=n_kernel,
        max_nfev=max_nfev,
        residual_tol=residual_tol,
        starting_line_method=starting_line_method,
        evaluate_moc=evaluate_moc,
    )
    solution = solve_rao_bvp(config)
    if return_solution:
        return solution
    contour = solution.to_contour_dict(
        Rt=Rt,
        epsilon=epsilon,
        length_pct=length_pct,
        pa_over_p0=pa_over_p0,
        Ru_factor=Ru_factor,
        convergent_half_angle_deg=convergent_half_angle_deg,
    )
    return add_contour_reliability_metadata(contour, "rao_variational_moc", gamma)


def rao_variational_contour(
    Rt: float,
    epsilon: float,
    gamma: float = 1.4,
    pa_over_p0: float = 0.0,
    length_pct: float = 80.0,
    n_ce_pts: int = 25,
    n_char: int = 30,
    convergent_half_angle_deg: float = 45.0,
    Ru_factor: float = 1.5,
    max_iter: int = 80,
) -> dict:
    """
    Generate an experimental Rao-style variational bell contour.

    Steps:
      1. Solve for optimal control-surface distribution (calculus of variations)
      2. Construct wall via MOC to realize that flowfield
      3. Prepend convergent + throat arcs

    Parameters
    ----------
    Rt          : throat radius [m]
    epsilon     : expansion ratio Ae/At
    gamma       : ratio of specific heats
    pa_over_p0  : design ambient/stagnation pressure ratio Pa/P0
    length_pct  : bell length as % of 15° cone
    n_ce_pts    : control-surface discretization points
    n_char      : MOC characteristic line points
    convergent_half_angle_deg : upstream inlet half-angle [°]
    Ru_factor   : upstream fillet ratio
    max_iter    : max variational iterations

    Returns
    -------
    dict compatible with bell_nozzle_contour() output
    """
    Re = math.sqrt(epsilon) * Rt
    Ln = (length_pct / 100.0) * _full_cone_length(Rt, epsilon)
    Ru = Ru_factor * Rt
    Rd = 0.382 * Rt

    # Step 1: Solve the variational problem on the control surface
    ce = solve_optimal_control_surface(
        Rt, epsilon, gamma, length_pct, n_ce_pts,
        max_outer_iter=max_iter, pa_over_p0=pa_over_p0
    )

    # Step 2: Construct the wall from CE via MOC
    wall_x, wall_r, construction_diagnostics = _construct_wall_from_ce(
        Rt, epsilon, gamma, ce, Ln, n_char
    )
    if len(wall_x) > 0:
        wall_x[-1] = Ln
        wall_r[-1] = Re

    # Step 3: Build convergent + throat arcs
    conv_angle = math.radians(convergent_half_angle_deg)
    n_conv = 100
    t_conv = np.linspace(-(math.pi / 2 + conv_angle), -math.pi / 2, n_conv)
    x_conv = Ru * np.cos(t_conv)
    y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

    theta_n = max(ce.theta[0], math.radians(15))
    t_thr = np.linspace(-math.pi / 2, theta_n - math.pi / 2, n_conv)
    x_throat = Rd * np.cos(t_thr)
    y_throat = (Rt + Rd) + Rd * np.sin(t_thr)

    # Full contour
    x_full = np.concatenate([x_conv, x_throat, wall_x])
    y_full = np.concatenate([y_conv, y_throat, wall_r])

    # Compute exit metrics
    theta_n_deg = math.degrees(theta_n)
    theta_e_deg = math.degrees(math.atan2(
        wall_r[-1] - wall_r[-2], wall_x[-1] - wall_x[-2]
    )) if len(wall_x) > 1 else 0.0

    Nx = x_throat[-1]
    Ny = y_throat[-1]

    warnings = [
        "Experimental Rao-style variational prototype; not a validated full "
        "Rao optimum-thrust boundary-value solution."
    ]
    warnings.extend(ce.warnings)
    warnings.extend(construction_diagnostics.get("warnings", []))

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
        'epsilon': epsilon,
        'pa_over_p0': pa_over_p0,
        'length_pct': length_pct,
        'N': (Nx, Ny),
        'E': (wall_x[-1], wall_r[-1]),
        'P1': (0.5 * (Nx + wall_x[-1]), 0.5 * (Ny + wall_r[-1])),
        'x_conv': x_conv,
        'y_conv': y_conv,
        'x_throat': x_throat,
        'y_throat': y_throat,
        'x_bell': wall_x,
        'y_bell': wall_r,
        'method': 'rao',
        'contour_type': 'experimental_rao_variational',
        'variational_status': 'experimental_not_full_rao_bvp',
        'rao_full_optimum_claimed': False,
        'optimization_converged': bool(
            ce.converged and construction_diagnostics.get("moc_compatibility_preserved")
        ),
        'moc_compatibility_preserved': bool(
            construction_diagnostics.get("moc_compatibility_preserved")
        ),
        'construction_diagnostics': construction_diagnostics,
        'control_surface': ce,
        'lambda2': ce.lambda2,
        'lambda3': ce.lambda3,
        'warnings': warnings,
    }
    return add_contour_reliability_metadata(contour, 'rao', gamma)
