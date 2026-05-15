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
    prandtl_meyer,
    mach_from_prandtl_meyer,
    area_mach_relation,
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
from raosim.rao_residuals import residual_Cminus_axisym, residual_Cplus_axisym
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
    max_nfev: int = 25
    residual_tol: float = 2e-3
    starting_line_method: str = "area_ratio"
    evaluate_moc: bool = True
    residual_blocks: tuple[str, ...] | None = None


DEFAULT_RAO_RESIDUAL_BLOCKS = (
    "mass",
    "length",
    "moc_cminus",
    "ce_geometry",
    "regularization",
    "penalties",
)

ALL_RAO_RESIDUAL_BLOCKS = DEFAULT_RAO_RESIDUAL_BLOCKS + (
    "moc_cplus",
    "stationarity",
    "transversality",
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
    "all": DEFAULT_RAO_RESIDUAL_BLOCKS,
}


@dataclass
class RaoResidualReport:
    """Scaled and dimensional residual summary for an attempted Rao solve."""

    max_scaled: float
    rms_scaled: float
    mass_residual_rel: float
    length_residual_rel: float
    stationarity_rms: float
    regularization_rms: float
    transversality_scaled: float
    wall_tangency_rms: float | None = None
    characteristic_crossings: int = 0
    group_summaries: list[dict] = field(default_factory=list)


@dataclass
class RaoResidualGroups:
    """Named residual blocks for ablation and debugging."""

    mass: np.ndarray
    length: np.ndarray
    transversality: np.ndarray
    stationarity: np.ndarray
    moc_cplus: np.ndarray
    moc_cminus: np.ndarray
    ce_geometry: np.ndarray
    regularization: np.ndarray
    penalties: np.ndarray

    def flat(self) -> np.ndarray:
        return np.concatenate([
            self.mass,
            self.length,
            self.transversality,
            self.stationarity,
            self.moc_cplus,
            self.moc_cminus,
            self.ce_geometry,
            self.regularization,
            self.penalties,
        ])

    def summaries(self) -> list[dict]:
        return [
            summarize_group("mass", self.mass),
            summarize_group("length", self.length),
            summarize_group("transversality", self.transversality),
            summarize_group("stationarity", self.stationarity),
            summarize_group("moc_cplus", self.moc_cplus),
            summarize_group("moc_cminus", self.moc_cminus),
            summarize_group("ce_geometry", self.ce_geometry),
            summarize_group("regularization", self.regularization),
            summarize_group("penalties", self.penalties),
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
    Compute the 3 Euler-Lagrange residuals at a single CE station:

        R₁ = ∂f₁/∂M + λ₂·∂f₂/∂M = 0
        R₂ = ∂f₁/∂θ + λ₂·∂f₂/∂θ = 0
        R₃ = ∂f₁/∂φ + λ₂·∂f₂/∂φ + λ₃·∂f₃/∂φ = 0

    Note: f₃ = cot(φ) depends only on φ, so ∂f₃/∂M = ∂f₃/∂θ = 0.
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
    theta_n = max(float(ce.theta[0]), math.radians(15))
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


def _initial_ce_from_kernel(config: RaoSolverConfig) -> tuple[ControlSurface, list[CharPoint]]:
    """Build a kernel-seeded control-surface initial guess."""
    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    Ln = _target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    theta_n = math.radians(config.thetaN_guess_deg)
    kernel_points = approximate_starting_line(
        Rt, Rd, theta_n, config.gamma, config.n_kernel,
        method=config.starting_line_method,
    )

    ce = _initial_ce_guess(Rt, Re, Ln, config.gamma, config.n_control)
    if kernel_points:
        k_r = np.asarray([p.r for p in kernel_points], dtype=float)
        k_M = np.asarray([p.M for p in kernel_points], dtype=float)
        k_theta = np.asarray([p.theta for p in kernel_points], dtype=float)
        order = np.argsort(k_r)
        k_r = k_r[order]
        k_M = k_M[order]
        k_theta = k_theta[order]
        overlap = ce.r <= float(k_r[-1])
        if np.any(overlap):
            ce.M[overlap] = np.interp(ce.r[overlap], k_r, k_M)
            ce.theta[overlap] = np.interp(ce.r[overlap], k_r, k_theta)

    try:
        Me = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me = max(float(ce.M[-1]), 2.0)
    frac = np.linspace(0.0, 1.0, config.n_control)
    ce.M = np.maximum(ce.M, 1.001 + (Me - 1.001) * frac**0.85)
    ce.theta = np.clip(
        math.radians(config.thetaN_guess_deg) * (1.0 - 0.55 * frac),
        math.radians(-5.0),
        math.radians(55.0),
    )
    phi_len = math.atan2(max(Re - ce.r[0], 1e-9), max(Ln, 1e-9))
    phi_base = np.full(config.n_control, max(phi_len, math.radians(8.0)))
    ce.phi = np.maximum(phi_base, ce.theta + math.radians(2.0))
    ce.phi = np.clip(ce.phi, math.radians(5.0), math.radians(88.0))
    ce.x = np.linspace(0.0, Ln, config.n_control)
    ce.phi = _phi_from_curve(ce.x, ce.r)
    return ce, kernel_points


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


def _pack_bvp(ce: ControlSurface, lambda2: float, lambda3: float) -> np.ndarray:
    if ce.x is None:
        x = np.zeros_like(ce.r, dtype=float)
        for i in range(1, len(ce.r)):
            phi_avg = 0.5 * (ce.phi[i - 1] + ce.phi[i])
            dr = ce.r[i] - ce.r[i - 1]
            sin_phi = math.sin(phi_avg)
            x[i] = x[i - 1] + (math.cos(phi_avg) / sin_phi * dr if abs(sin_phi) > 1e-10 else 0.5 * dr)
    else:
        x = np.asarray(ce.x, dtype=float)
    return np.concatenate([ce.M, ce.theta, x, ce.r, [lambda2, lambda3]])


def _unpack_bvp(u: np.ndarray, r: np.ndarray) -> ControlSurface:
    n = len(r)
    x = np.asarray(u[2 * n:3 * n], dtype=float).copy()
    r_ce = np.asarray(u[3 * n:4 * n], dtype=float).copy()
    phi = _phi_from_curve(x, r_ce)
    return ControlSurface(
        r=r_ce,
        M=np.asarray(u[:n], dtype=float).copy(),
        theta=np.asarray(u[n:2 * n], dtype=float).copy(),
        phi=phi,
        x=x,
        lambda2=float(u[4 * n]),
        lambda3=float(u[4 * n + 1]),
    )


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
    endpoint = np.array([
        (float(ce.x[0]) - 0.0) / x_scale,
        (float(ce.x[-1]) - L) / x_scale,
        (float(ce.r[-1]) - Re) / r_scale,
    ], dtype=float)
    try:
        from raosim.nozzle_geometry import lookup_angles

        theta_n_deg, theta_e_deg = lookup_angles(config.epsilon, config.length_pct)
    except Exception:
        theta_n_deg = config.thetaN_guess_deg
        theta_e_deg = max(0.0, 0.5 * config.thetaN_guess_deg)
    theta_scale = math.radians(1.0)
    flow_boundary = np.array([
        (float(ce.theta[0]) - math.radians(theta_n_deg)) / theta_scale,
        (float(ce.theta[-1]) - math.radians(theta_e_deg)) / theta_scale,
    ], dtype=float)
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
    ce = _unpack_bvp(u, r)
    _, mdot_val, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_target = _target_mdot(config.Rt, config.gamma)
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
    penalties = np.concatenate([
        incidence_penalty,
        mach_monotonic_penalty,
        0.1 * phi_smooth,
    ])
    active = _enabled_residual_blocks(config)

    return RaoResidualGroups(
        mass=_filter_group(
            "mass",
            np.array([(mdot_val - mdot_target) / max(mdot_target, 1e-12)]),
            active,
        ),
        length=_filter_group(
            "length",
            np.array([(L_val - L_target) / max(L_target, 1e-12)]),
            active,
        ),
        transversality=_filter_group("transversality", np.array([trans / trans_scale]), active),
        stationarity=_filter_group("stationarity", stat_res, active),
        moc_cplus=_filter_group("moc_cplus", moc_cplus, active),
        moc_cminus=_filter_group("moc_cminus", moc_cminus, active),
        ce_geometry=_filter_group("ce_geometry", ce_geometry, active),
        regularization=_filter_group("regularization", 0.02 * regularization, active),
        penalties=_filter_group("penalties", penalties, active),
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
) -> RaoResidualReport:
    _, mdot_val, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_target = _target_mdot(config.Rt, config.gamma)
    L_target = _target_length(config.Rt, config.epsilon, config.length_pct)
    stat = _stationarity_matrix(ce, config.gamma, config.pa_over_p0)
    regularization = _ce_smoothness_regularization(ce, config.gamma)
    groups = _rao_bvp_residual_groups(_pack_bvp(ce, ce.lambda2, ce.lambda3), r_template, config)
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
        mass_residual_rel=float((mdot_val - mdot_target) / max(mdot_target, 1e-12)),
        length_residual_rel=float((L_val - L_target) / max(L_target, 1e-12)),
        stationarity_rms=float(np.sqrt(np.mean(stat**2))) if stat.size else 0.0,
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


def _segments_intersect(a: np.ndarray, b: np.ndarray,
                        c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def check_characteristic_crossing(rows: list[CharRow]) -> int:
    """Count geometric crossings between characteristic-net segments."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        pts = row.all_points()
        for a, b in zip(pts[:-1], pts[1:]):
            segments.append((np.array([a.x, a.r]), np.array([b.x, b.r])))
    crossings = 0
    for i, (a, b) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if abs(i - j) <= 1:
                continue
            c, d = segments[j]
            if _segments_intersect(a, b, c, d):
                crossings += 1
    return crossings


def solve_rao_bvp(config: RaoSolverConfig) -> RaoSolution:
    """
    Solve the finite-dimensional Rao variational/MOC residual system.

    This is the new auditable path: least-squares solves the global
    mass-flow, length, stationarity, transversality, and CE compatibility
    residuals together.  MOC wall construction is evaluated on the raw
    solution and can downgrade reliability if it does not close cleanly.
    """
    if config.Rt <= 0.0:
        raise ValueError("Rt must be positive")
    if config.epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")
    if config.pa_over_p0 < 0.0:
        raise ValueError("pa_over_p0 must be non-negative")
    if config.n_control < 8:
        raise ValueError("n_control must be at least 8")

    ce0, kernel_points = _initial_ce_from_kernel(config)
    u0 = _pack_bvp(ce0, -0.5, 0.01)
    n = len(ce0.r)
    try:
        Me = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me = 8.0
    lower = np.concatenate([
        np.full(n, 1.001),
        np.full(n, math.radians(-10.0)),
        np.full(n, 0.0),
        np.full(n, 1e-9),
        [-1e3, -1e3],
    ])
    upper = np.concatenate([
        np.full(n, max(12.0, 1.5 * Me)),
        np.full(n, math.radians(65.0)),
        np.full(n, max(1.2 * _target_length(config.Rt, config.epsilon, config.length_pct), 1e-9)),
        np.full(n, 1.05 * math.sqrt(config.epsilon) * config.Rt),
        [1e3, 1e3],
    ])

    if config.max_nfev <= 0:
        residual0 = _scaled_rao_bvp_residual(u0, ce0.r, config)

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
            args=(ce0.r, config),
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=config.max_nfev,
        )
    ce = _unpack_bvp(result.x, ce0.r)
    residual_vector = _scaled_rao_bvp_residual(result.x, ce0.r, config)
    F_val, mdot_val, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    mdot_target = _target_mdot(config.Rt, config.gamma)
    L_target = _target_length(config.Rt, config.epsilon, config.length_pct)
    ce.thrust = float(F_val)
    ce.objective = float(result.cost)
    ce.optimizer_success = bool(result.success)
    ce.solver_message = str(result.message)
    ce.mdot_target = float(mdot_target)
    ce.mdot_residual = float(mdot_val - mdot_target)
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
    theta_n = max(float(ce.theta[0]), math.radians(15.0))
    Nx = Rd * math.sin(theta_n)
    Ny = config.Rt + Rd * (1.0 - math.cos(theta_n))
    if config.evaluate_moc:
        try:
            raw_wall, construction_diagnostics = construct_wall_from_ce_raw(
                config.Rt, config.epsilon, config.gamma, ce, L_target,
                config.n_kernel,
            )
            if raw_wall.shape[0] >= 3:
                slope_start = math.tan(max(float(ce.theta[0]), math.radians(15.0)))
                slope_end = math.tan(float(ce.theta[-1]))
                wall = SplineWall(
                    raw_wall[:, 0],
                    raw_wall[:, 1],
                    slope_start=slope_start,
                    slope_end=slope_end,
                )
                starting = approximate_starting_line(
                    config.Rt,
                    config.throat_downstream_radius_factor * config.Rt,
                    max(float(ce.theta[0]), math.radians(15.0)),
                    config.gamma,
                    config.n_kernel,
                    method=config.starting_line_method,
                )
                char_net = march_coupled_net(starting, wall, config.gamma)
                crossings = check_characteristic_crossing(char_net)
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
        residual_vector, ce, config, ce0.r,
        wall_tangency_rms=wall_tangency_rms,
        crossings=crossings,
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
    ce.converged = bool(bvp_ok)
    shock_free = crossings == 0
    if bvp_ok and moc_ok:
        reliability = ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED
    elif moc_ok:
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
    warnings.append(
        "Not hardware-qualified; requires published benchmark comparison, CFD, "
        "thermal/structural review, manufacturing review, inspection, and hot-fire data."
    )
    ce.warnings.extend(warnings)

    At = math.pi * config.Rt * config.Rt
    cf = F_val / max(At, 1e-12)
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
        },
        warnings=_dedupe_strings(warnings),
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
    starting_line_method: str = "area_ratio",
    max_nfev: int = 25,
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
