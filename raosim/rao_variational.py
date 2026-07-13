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

Known limitations:
  - The nonlinear residual solve is a finite-dimensional direct method; the
    strict Rao-1958 Nozzle-B benchmark validates its published shape/integral
    envelope, not every internal characteristic state.
  - A result is promoted only when its per-run BVP and MOC topology gates pass;
    otherwise the returned wall remains a labelled geometric approximation.

References:
  - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA TM (1990), Rao method re-derivation with explicit functionals
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
"""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
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
    full_control_surface_thrust,
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
    # Phase 6 free CE↔wall pairing: pair_fractions[i] ∈ [0, 1] is the
    # wall arc-length position that CE node i pairs with.  None when
    # the coupled-wall path is not in use (legacy behaviour).
    pair_fractions: np.ndarray | None = None
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
    """Explicit maturity levels for generated contour data.

    ``NASA_REFERENCE_MATCHED`` is reserved for a run executed in a certified
    reference configuration.  The pinned M3.5 perfect-nozzle workflow has
    source-visible TT', kernel-row, and BDE-wall regression metrics (including
    wall r(x) RMS below 1e-3), but the generic Rao BVP is not that reference
    case and is therefore never promoted merely because those package tests
    pass.  Historical fixture-generator provenance is tracked separately and
    is not promotion authority.
    """

    GEOMETRIC_APPROXIMATION = "geometric_approximation"
    MOC_COMPATIBLE = "moc_compatible"
    RAO_VARIATIONAL_RESIDUAL_SOLVED = "rao_variational_residual_solved"
    NASA_REFERENCE_MATCHED = "nasa_reference_matched"
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

NASA_REFERENCE_FIXTURE_GENERATOR_PROVENANCE = "unresolved"
NASA_REFERENCE_PROVENANCE_DOC = "docs/nasa_tt_prime_provenance.md"
NASA_REFERENCE_CANONICAL_TRACK = "visible_source_port"
NASA_REFERENCE_HISTORICAL_OVERLAY_TRACK = "historical_fixture_overlay"
NASA_REFERENCE_SOURCE_SHA256 = (
    "cc4c0bc60e53a5f46d1c37d46f68724c721ebf8a6fc6b7f4a559976f17ec20b4"
)


def _nasa_reference_validation_diagnostics() -> dict:
    """Return the current policy gate for NASA_REFERENCE_MATCHED.

    The canonical NASA reference track is the visible ``MOC_GridCalc_BDE.cpp``
    source-port workflow.  The checked-in M3.5Perf outputs remain useful
    historical overlays, but their TT' generator provenance is unresolved, so
    fixture overlay agreement is not promotion authority for this enum.
    """
    repo_root = Path(__file__).resolve().parent.parent
    source_path = (
        repo_root / "Three-Dimensional-Nozzle-Design-Code-master"
        / "MOC_Grid_BDE" / "MOC_GridCalc_BDE.cpp"
    )
    source_hash = (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_path.is_file() else None
    )
    fixture_dir = source_path.parent / "outputs_M3.5Perf"
    required_overlay_files = (
        "TT'.out", "TT'BF_Kernel.out", "LastKernel.out", "wall.out",
    )
    missing_overlay_files = [
        name for name in required_overlay_files
        if not (fixture_dir / name).is_file()
    ]
    source_identity_verified = source_hash == NASA_REFERENCE_SOURCE_SHA256
    fixture_overlay_available = not missing_overlay_files
    reference_case_verified = bool(
        source_identity_verified and fixture_overlay_available
    )
    return {
        "canonical_reference_track": NASA_REFERENCE_CANONICAL_TRACK,
        "historical_overlay_track": NASA_REFERENCE_HISTORICAL_OVERLAY_TRACK,
        "source_port_matched": reference_case_verified,
        "source_port_match_status": (
            "software_verified_reference_case"
            if reference_case_verified else "reference_assets_unavailable_or_changed"
        ),
        "source_reference_workflow_complete": reference_case_verified,
        "source_reference_workflow_scope": (
            "M3.5 perfect-nozzle TT-prime, kernel march, BDE wall extraction"
        ),
        "general_contoured_workflow_complete": False,
        "general_contoured_workflow_blockers": [
            "CropNozzleToLength is not ported for exported interior grids",
            "the current-solve BVP is not automatically the M3.5 reference configuration",
        ],
        "source_reference_metrics_available": reference_case_verified,
        "source_reference_regression_metrics": [
            {
                "name": "TT_prime_pointwise",
                "test": "tests/test_nasa_kernel_march_parity.py::test_throat_initial_line_matches_fixture",
                "tolerance": "x,r 5e-6; Mach 5e-5; theta 5e-3 deg",
            },
            {
                "name": "kernel_row_1_unit_process",
                "test": "tests/test_nasa_kernel_march_parity.py::test_march_step_reproduces_nasa_row_1_from_row_0",
                "tolerance": "x,r 2e-5; Mach,theta 2e-4",
            },
            {
                "name": "last_kernel_end_to_end",
                "test": "tests/test_nasa_kernel_march_parity.py::test_build_kernel_marches_and_matches_last_kernel",
                "tolerance": "x,r 5e-4; Mach,theta 2e-3",
            },
            {
                "name": "perfect_nozzle_wall_rms",
                "test": "tests/test_nasa_port.py::test_nasa_wall_match[M3.5Perf]",
                "tolerance": "R/Rt RMS < 1e-3",
            },
        ],
        "source_path": str(source_path.relative_to(repo_root)),
        "source_sha256": source_hash,
        "source_sha256_expected": NASA_REFERENCE_SOURCE_SHA256,
        "source_identity_verified": source_identity_verified,
        "fixture_overlay_available": fixture_overlay_available,
        "fixture_overlay_missing_files": missing_overlay_files,
        "fixture_overlay_is_promotion_authority": False,
        "fixture_generator_provenance": NASA_REFERENCE_FIXTURE_GENERATOR_PROVENANCE,
        "fixture_generator_provenance_doc": NASA_REFERENCE_PROVENANCE_DOC,
        "eligible": False,
        "blockers": [
            "the reference certificate covers only the pinned M3.5 perfect-nozzle case",
            "historical fixture-generator provenance remains unresolved",
        ],
        "historical_overlay_notes": [
            "M3.5Perf fixture deltas are diagnostics only",
            "historical M3.5Perf TT' fixture generator provenance is unresolved",
        ],
    }


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
    # Upstream (convergent-side) throat wall radius / Rt.  Governs the
    # Kliegel-Levine transonic start line TT' (the KL/Hall series are
    # derived for the upstream curvature; NASA's CalcInitialThroatLine
    # passes rUp into KLThroat).  Rao's published TOP convention is
    # R_u = 1.5 Rt with R_d = 0.382 Rt (Rao 1958; NASA SP-8120; same
    # convention as nozzle_geometry.bell_nozzle_contour's Ru_factor).
    throat_upstream_radius_factor: float = 1.5
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
    # Upper bound on ``kernel_d_fraction`` (point D's arc-length
    # position along the kernel BD curve).  See the
    # :data:`KERNEL_D_FRACTION_MAX` module-level docstring for the
    # physical justification.  Default ``None`` uses
    # :data:`KERNEL_D_FRACTION_MAX` (currently 1.0 — no cap, preserves
    # Phase 4 mass closure on the default-weight reference case).
    # Studies of the high-weight regime should pass ``0.7`` to keep D
    # away from the kernel axis and avoid the Phase 5 valid-region
    # check firing on the kernel-to-CE bridge.
    kernel_d_fraction_max: float | None = None
    # Lower bound on ``kernel_d_fraction``.  Classical Rao optima sit at
    # D fractions ~0.3-0.6 (see KERNEL_D_FRACTION_MAX docstring); with
    # the characteristic formulation on a *frozen* kernel BD the
    # unconstrained solve can drift D toward B (kdf -> ~0.02), where the
    # mass closure becomes trivially satisfiable by a near-streamline DE
    # and the BDE-region wall march degenerates (zero-mass BD segment).
    # A floor (e.g. 0.15-0.3) keeps the topology physical until theta_B/
    # BD join the iteration.  Default 0.0 preserves legacy behaviour.
    kernel_d_fraction_min: float = 0.0
    # JAX differentiable backend (JAX_DIFFERENTIABLE_PLAN.md).
    #
    #   "jax"   — (DEFAULT since 2026-06-11, DIRECTION item 2: the J4
    #             gate re-confirmed at max_scaled = 7.50e-4 on the
    #             post-12.4 seed.)  Optimistix Levenberg–Marquardt on
    #             the raosim.jax.assembly residual with *exact* autodiff
    #             Jacobians.  Requires jax/optimistix installed (pinned
    #             in requirements.txt).  Residual parity with the NumPy
    #             path is guaranteed by the J2 gate
    #             (tests/test_jax_assembly_parity.py).
    #   "numpy" — scipy.optimize.least_squares with finite-difference
    #             Jacobians (the legacy regression oracle; kept for
    #             comparisons until the raosim/jax absorption,
    #             DIRECTION item 1).
    #
    # Only the inner least-squares step changes; seeding, kernel build,
    # reliability gating, and diagnostics are identical for both backends.
    solver_backend: str = "jax"
    # CE seed strategy for the BVP unknown vector:
    #
    #   "auto"     — (default) seed the CE from the NASA fixed-end Rao
    #                topology (set_theta_b/calc_lrc_de with
    #                end_condition="fixed_end": D placed on the marched
    #                kernel BD so DE's endpoint pins r_E, theta_B secanted
    #                toward the target length).  Falls back to "linear"
    #                if the topology solve fails.
    #   "topology" — require the topology seed; raise on failure.
    #   "linear"   — the legacy axis-to-exit linear ramp seed (the only
    #                option before the kernel march fixes; kept for
    #                regression comparisons).
    ce_seed: str = "auto"
    # Residual-stack formulation (when ``residual_blocks`` is None):
    #
    #   "characteristic" — (DEFAULT since 2026-06-11, the J4-gate
    #                      configuration.)  CHARACTERISTIC_RAO_RESIDUAL_
    #                      BLOCKS: the converged-topology set (no C−
    #                      along the C+ CE; no CE→wall pairing blocks).
    #                      See the constant's docstring for the physics
    #                      + the empirical evidence.
    #   "legacy"         — DEFAULT_RAO_RESIDUAL_BLOCKS (pre-refactor
    #                      scaffold blocks included).  Opt-in for
    #                      regression comparisons until the raosim/jax
    #                      absorption.
    formulation: str = "characteristic"
    # Constraint-weight continuation for the JAX backend: the integral
    # constraints (mass, length) and endpoint pins are single residual
    # elements drowned by ~O(n) physics elements in plain least squares
    # (observed: LM sacrifices length at ~0.5 while physics closes).
    # A ladder like (1.0, 10.0, 30.0) re-solves with those elements
    # progressively up-weighted, reusing each solution as the next seed;
    # the *reported* residual is always the unweighted one.  None = off.
    # Default (1, 10, 30, 100) is the J4-gate ladder.
    jax_constraint_weight_ladder: tuple[float, ...] | None = (
        1.0, 10.0, 30.0, 100.0,
    )
    # Pin M_ce[0] to D's kernel Mach (full flow-state continuity at D).
    # Default True under the corrected characteristic formulation: the
    # smooth fixed-end existence root satisfies the complete D state.
    # Set False only for the explicit position-only diagnostic branch.
    pin_d_mach: bool = True
    # Pin theta_ce[0] to D's interpolated kernel angle.  Default True for
    # the same full-continuity reason as pin_d_mach.
    pin_d_theta: bool = True
    # Freeze the kernel arc-end angle theta_B at this value [deg],
    # bypassing the seed's inner set_theta_b secant (which otherwise
    # re-converges theta_B to the *fixed-end* closure — ~25.5 deg at the
    # eps=10/L80 reference — regardless of thetaN_guess_deg).  Purpose:
    # an OUTER theta_B iteration (Picard/secant) around the BVP needs to
    # actually control the frozen kernel; without this knob the inner
    # secant overrides it and the full-continuity stationarity floor is
    # theta_B-insensitive by construction.  The Rao-1961-grounded
    # expectation is that the fixed-(L, eps) optimum sits near the chart
    # theta_N (~28-30 deg downstream wall angle; Rao, ARS J. 31(11),
    # 1961, pp. 1490-1491), NOT at the fixed-end closure value.  D and
    # the DE seed are still placed by calc_lrc_de(end_condition=
    # "fixed_end") on the frozen kernel (r_E pinned; length left to the
    # solve).  None = legacy behaviour (inner secant owns theta_B).
    theta_b_freeze_deg: float | None = None
    # J3b-2 (2026-06-12): solve theta_B as a BVP unknown.  JAX backend
    # only.  The unknown vector gains one trailing component and the
    # residual recomputes the kernel BD *in-graph* per evaluation via
    # the differentiable march (raosim.jax.moc_kernel.march_kernel —
    # bit-parity with the NumPy kernel at the seed angle); the
    # mass-closure target and D-state pins read the live BD(theta_B).
    # After the solve the kernel is re-frozen at the solved angle
    # (provenance "bvp_solved") so every downstream consumer — mass
    # diagnostics, BDE wall, theta_N reporting — sees the solved
    # kernel.  theta_B bounds: ± a quarter dtheta-limit around the
    # seed (the march's smooth window); re-centre by re-solving if it
    # walks to a bound.  Default False: the seed secant owns theta_B
    # exactly as before.
    solve_theta_b: bool = False


# Converged-topology ("characteristic") block set.  After the
# left-Mach-by-construction refactor the CE segments are C+ characteristics
# *by construction*, so two scaffold blocks from the generic-curve era are
# structurally unsatisfiable at the Rao topology and are dropped here:
#
#   * ``moc_cminus`` — applied the C− compatibility relation along the
#     CE's C+ segments.  Rao's DE closure uses C+ relations +
#     stationarity + mass/length only (Rao 1958; Rao-Beck-Booth AIAA
#     99-2584 Eqs. 12-14; NASA FindPointE integrates only the LRC/C+
#     derivative system along DE).
#   * ``cplus_ce_to_wall`` / ``wall_intersection`` — paired CE nodes to
#     wall points along C+ slopes, but the C+ line through a DE point is
#     DE itself and meets the wall only at E.  The wall belongs to the
#     BDE-region march (nasa_moc.calc_bde_region), not to CE→wall chords.
#
# Empirical: on the ε=10/L80/w=1.0 reference the legacy stack stalls at
# max_scaled ≈ 0.3-0.5 with resolution-independent floors in exactly
# these blocks; the characteristic set reaches ~6e-2 with C+ compat at
# ~1e-3 and mass at ~3e-3 (see tests/test_jax_convergence.py).
# Opt-in via RaoSolverConfig.formulation="characteristic" until the
# Phase 6/7 gates re-baseline on it.
CHARACTERISTIC_RAO_RESIDUAL_BLOCKS = (
    "mass",
    "length",
    "moc_cplus",
    "ce_geometry",
    "regularization",
    "penalties",
    "algebraic_stationarity",
    "wall_endpoint",
    "wall_tangency",
)

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
#   * weight = 1.00 → exercised as a non-xfailed BVP-closure regression;
#     the optional wall audit can still keep a particular run below the
#     promoted reliability tier.
# Remaining promotion work is tracked by measured per-run gates, not an
# expected-failure marker.  Likely sensitivities include linear
# CE↔wall pairing in the Phase 6 coupled-wall builder (try free
# pair_fraction unknowns), under-resolved wall (bump n_wall to 20),
# or the Bezier wall seed not yet wired in for couple_wall=True.
PHYSICS_WEIGHT = 0.05

# =====================================================================
#  Phase 7 -- BENCHMARK_VALIDATED reliability promotion
# =====================================================================
#
# The release policy gates ``BENCHMARK_VALIDATED`` on:
#
#   (1) exact-variational acceptance criteria have been approved for this
#       release, AND
#   (2) the per-run residuals are < ``BENCHMARK_VALIDATED_RESIDUAL_TOL``,
#       AND
#   (3) the input (epsilon, length_pct) sits inside the validated
#       sub-grid (or is interpolable within it).
#
# ``BENCHMARK_VALIDATED_AT_RELEASE`` is the persistent flag controlling
# (1).  It remains ``False`` because the historical 1.5/3-degree target
# compares the exact variational angles to Rao's different 1960 parabola-fit
# model and is therefore explicitly inapplicable.  A published, reviewed
# exact-variational grid criterion must replace it before this flag changes.
# Until then no ``solve_rao_bvp`` call can be promoted to
# ``BENCHMARK_VALIDATED`` even if (2) and (3) hold.
BENCHMARK_VALIDATED_AT_RELEASE: bool = False
BENCHMARK_VALIDATED_RESIDUAL_TOL: float = 1e-4
BENCHMARK_VALIDATED_EPSILON_RANGE: tuple[float, float] = (6.0, 50.0)
BENCHMARK_VALIDATED_LENGTH_PCT_RANGE: tuple[float, float] = (70.0, 90.0)

# Rao 1958 Table 3 reports Cf to four decimals, while the accompanying
# Table-2 contour coordinates are rounded to two decimals.  A 5 % full-CDE
# sanity band allows the finite-angle loss visible in Rao's published
# Nozzle-B result (96.93 % of one-dimensional thrust).  It is deliberately a
# low-order integral check, not a validation
# promotion; the strict Nozzle-B literature benchmark below uses a tighter
# absolute published-data tolerance.
FULL_CONTROL_SURFACE_CF_REL_TOL: float = 5.0e-2


def is_within_benchmarked_chart_grid(epsilon: float, length_pct: float) -> bool:
    """True iff (epsilon, length_pct) lies inside the benchmarked sub-grid.

    The sub-grid bounds (:data:`BENCHMARK_VALIDATED_EPSILON_RANGE`,
    :data:`BENCHMARK_VALIDATED_LENGTH_PCT_RANGE`) are also documented
    on :func:`raosim.benchmarks.rao_variational_chart_benchmark`.
    """
    eps_lo, eps_hi = BENCHMARK_VALIDATED_EPSILON_RANGE
    lpct_lo, lpct_hi = BENCHMARK_VALIDATED_LENGTH_PCT_RANGE
    return (
        eps_lo <= float(epsilon) <= eps_hi
        and lpct_lo <= float(length_pct) <= lpct_hi
    )


# Default upper bound on ``kernel_d_fraction`` (D's arc-length position
# along the kernel BD curve, with 0 at the wall-side end B and 1 at the
# deepest kernel point near the axis).  Classical Rao has D well inside
# the kernel — at fraction ~0.3-0.6 in the published cases — never at
# the axis.  When the kernel BD only just carries the quasi-1D throat
# mass flow (the typical case at the moment), the BVP must push
# ``kernel_d_fraction`` very close to 1.0 to close mass; tightening the
# cap below ~0.9 then breaks mass closure on the default-weight path
# (see ``tests/test_rao_variational_moc.py::
# test_phase4_mass_closure_uses_kernel_bd_segment``).
#
# At ``PHYSICS_WEIGHT=1.0`` the BVP also drifts toward 1.0 but the
# resulting CE starts from the kernel axis (D at M≈1.001), the Phase 5
# valid-region check then fires on the kernel-to-CE bridge, and the
# (ε=10, length_pct=80) weight=1.0 case lands in
# ``GEOMETRIC_APPROXIMATION``.  Setting a tighter cap (e.g. 0.7) keeps
# D away from the axis and restores ``valid_shock_free_region`` for
# that case — at the cost of mass closure on the n=8 default-weight
# reference case.  The two needs are in tension until Phase 12.4's
# ``CalcRRCsAlongArc`` lands and the kernel can be extended along the
# throat arc on demand (the NASA path), so the cap is exposed as a
# per-solve knob: callers studying the high-weight regime pass
# ``kernel_d_fraction_max=0.7``; the default of ``1.0`` preserves the
# pre-cap behaviour for everything else.
KERNEL_D_FRACTION_MAX = 1.0

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
    crossing_samples: list[dict] = field(default_factory=list)

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
            "crossing_samples": list(self.crossing_samples),
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
    # Reported design angles [rad].  Characteristic formulation (J5
    # de-circularization): theta_N is the kernel arc-end angle theta_B
    # the BVP closed on (a solver output), theta_E the solved CE exit
    # flow angle.  Legacy formulation: theta_N is the Rao-1960
    # parabola-fit chart lookup and theta_E the export-wall end slope.
    # construction_diagnostics["design_angles"] carries both flavours
    # plus provenance strings.
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
    # §12.7: the full-form SOLVED topology (moc_topology.RaoTopology —
    # CharPoint curves, full_wall(), closure_report()), built by the
    # BDE wall path from the solved CE/kdf.  None for other wall
    # methods.  ``topology`` above stays the nasa_moc SEED topology.
    topology_solved: object | None = None

    def to_contour_dict(self, *, Rt: float, epsilon: float, length_pct: float,
                        pa_over_p0: float, Ru_factor: float = 1.5,
                        Rd_factor: float = 0.382,
                        convergent_half_angle_deg: float = 45.0) -> dict:
        """Return compatibility dict shaped like ``bell_nozzle_contour``."""
        Re = math.sqrt(epsilon) * Rt
        Ru = Ru_factor * Rt
        Rd = Rd_factor * Rt
        wall_x = self.wall_export[:, 0]
        wall_r = self.wall_export[:, 1]
        if len(wall_x) > 1:
            distance = np.hypot(np.diff(wall_x), np.diff(wall_r))
            keep = np.concatenate(([True], distance > 1e-14 * max(Rt, 1.0)))
            wall_x = wall_x[keep]
            wall_r = wall_r[keep]

        conv_angle = math.radians(convergent_half_angle_deg)
        n_conv = 100
        t_conv = np.linspace(-(math.pi / 2 + conv_angle), -math.pi / 2, n_conv)
        x_conv = Ru * np.cos(t_conv)
        y_conv = (Rt + Ru) + Ru * np.sin(t_conv)

        theta_n = self.theta_N
        # Throat downstream region: bridge the throat to the exported
        # characteristic wall's first station N so the junction is C0 + C1
        # ("slope matched at N", Seitzman AE6450; SP-8120 §2.1.1 throat
        # tangency).  The MOC/BDE wall is solved independently, so its first
        # station does NOT lie on the R_d = 0.382 R_t arc at the wall's own
        # initial slope: the wall sits near the chart-N station yet departs at
        # the kernel theta_B (~3-4 deg shallower).  The previous code drew a
        # plain R_d arc to the wall's axial station only, leaving a radial gap
        # AND a 3-4 deg slope kink that tripped position_continuity and
        # slope_continuity.  Fix: keep the solved wall (hence exit radius /
        # epsilon) UNCHANGED and fill throat -> N with a Hermite cubic that is
        # tangent to the throat plane at the throat (horizontal, throat radius
        # R_t) and tangent to the wall at N.  Closing the residual at the
        # source — so the wall leaves at the same angle the arc reaches it — is
        # the open kernel-theta_B (J5/J6) work.
        x0 = float(wall_x[0])
        r0 = float(wall_r[0])
        if len(wall_x) >= 2:
            theta0 = math.atan2(
                float(wall_r[1] - wall_r[0]), float(wall_x[1] - wall_x[0])
            )
        else:
            theta0 = float(theta_n)
        throat_bridge = "cubic_tangent"
        if x0 <= 1e-12 or abs(r0 - Rt) <= 1e-12:
            # Export wall already begins at the throat: no bridge station.
            x_throat = np.array([0.0])
            y_throat = np.array([Rt])
            throat_bridge = "collapsed"
        else:
            # r(x) = a x^3 + b x^2 + R_t with r(0)=R_t, r'(0)=0,
            #        r(x0)=r0, r'(x0)=tan(theta0).
            s1 = math.tan(min(max(theta0, 0.0), math.radians(89.0)))
            dr = r0 - Rt
            a = (s1 - 2.0 * dr / x0) / (x0 * x0)
            b = dr / (x0 * x0) - a * x0
            xb = np.linspace(0.0, x0, n_conv)
            x_throat = xb
            y_throat = a * xb**3 + b * xb**2 + Rt
            dydx = 3.0 * a * xb**2 + 2.0 * b * xb
            if np.any(dydx < -1e-9) or np.any(y_throat < Rt - 1e-12):
                # Non-monotonic bridge (pathological wall start): fall back to a
                # plain R_d throat arc to the wall's axial station.
                arc_end = math.asin(min(max(x0 / Rd, 0.0), 1.0)) if Rd > 0 else 0.0
                t_thr = np.linspace(-math.pi / 2, arc_end - math.pi / 2, n_conv)
                x_throat = Rd * np.cos(t_thr)
                y_throat = (Rt + Rd) + Rd * np.sin(t_thr)
                throat_bridge = "rd_arc_fallback"
        Nx = float(x_throat[-1])
        Ny = float(y_throat[-1])

        x_full = np.concatenate([x_conv, x_throat[1:], wall_x[1:]])
        y_full = np.concatenate([y_conv, y_throat[1:], wall_r[1:]])

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
                ContourReliability.NASA_REFERENCE_MATCHED,
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
        contour["throat_bell_reconciliation"] = {
            "throat_bridge": throat_bridge,
            "wall_theta0_deg": math.degrees(theta0),
            "N": (Nx, Ny),
            "exit_radius_design": float(Re),
            "exit_radius_built": float(wall_r[-1]),
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
         (θ + ν) backward toward the wall.
      3. C⁺ characteristics from interior points carry (θ − ν) forward.
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
    ce_kminus = ce_theta + ce_nu      # C- compatibility values on CE

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
            # C- from CE interpolated at current wall radius
            k_minus = _interp(r_w, ce_kminus)
            M_ce  = max(_interp(r_w, ce_M), 1.001)
            theta_ce = _interp(r_w, ce_theta)
            mu_ce = mach_angle(M_ce)
            x_ce_loc = _interp(r_w, x_ce)
            r_ce_loc = _interp(r_w, ce_r)

            # C+ from interior with axisymmetric source correction:
            # d(theta - nu) = -S ds, S = sin(theta) sin(mu) / r.
            k_plus = p_in.compat_minus
            if r_w > 1e-10 and p_in.r > 1e-10:
                ds = math.sqrt(
                    (x_w - p_in.x) ** 2 + (r_w - p_in.r) ** 2
                )
                if ds > 1e-12:
                    th_a = 0.5 * (p_in.theta + theta_w)
                    mu_a = 0.5 * (p_in.mu + mu_w)
                    r_a = max(0.5 * (p_in.r + r_w), 1e-12)
                    S = math.sin(th_a) * math.sin(mu_a) / r_a
                    k_plus = p_in.compat_minus - S * ds

            # C- from CE: d(theta + nu) = +S ds.
            if r_w > 1e-10 and r_ce_loc > 1e-10:
                ds_ce = math.sqrt(
                    (x_w - x_ce_loc) ** 2 + (r_w - r_ce_loc) ** 2
                )
                if ds_ce > 1e-12:
                    th_a = 0.5 * (theta_ce + theta_w)
                    mu_a = 0.5 * (mu_ce + mu_w)
                    r_a = max(0.5 * (r_ce_loc + r_w), 1e-12)
                    S = math.sin(th_a) * math.sin(mu_a) / r_a
                    k_minus = k_minus + S * ds_ce

            # Solve compatibility:
            #   theta_w + nu_w = K- (C-)
            #   theta_w - nu_w = K+ (C+)
            theta_w_new = 0.5 * (k_minus + k_plus)
            nu_w_new    = 0.5 * (k_minus - k_plus)
            if nu_w_new < 1e-8:
                nu_w_new = 1e-8
            M_w_new  = mach_from_prandtl_meyer(nu_w_new, gamma)
            mu_w_new = mach_angle(M_w_new)

            # New position: intersection of C+ from p_in & C- from CE
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
    *,
    fraction_max: float | None = None,
) -> float:
    """Seed D by matching the initial CE flux on the kernel BD curve.

    ``fraction_max`` clips the seed to match the BVP bounds box.  When
    ``None`` the module-level :data:`KERNEL_D_FRACTION_MAX` is used.
    """
    cap = float(fraction_max if fraction_max is not None else KERNEL_D_FRACTION_MAX)
    cap = float(np.clip(cap, 0.0, 1.0))
    full_flux = curve_mass_flux(kernel_bd, gamma)
    if full_flux <= 1e-14 or target_flux <= 0.0:
        return cap
    if target_flux >= full_flux:
        return cap
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mid_flux = curve_mass_flux(kernel_bd_segment(kernel_bd, mid), gamma)
        if mid_flux < target_flux:
            lo = mid
        else:
            hi = mid
    return min(0.5 * (lo + hi), cap)


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

    Returns ``(ce, kernel_bd_flow_nodes, topology, kernel)`` where
    ``topology`` is a :class:`nasa_moc.RaoTopology` capturing point B, BD,
    point D, DE, point E, mass closures, and the Rao stationarity
    constant, and ``kernel`` is the :class:`nasa_moc.MOCKernel` the BD
    came from (needed by the ``wall_method="bde"`` region march).
    """
    from raosim.nasa_moc import build_kernel as _build_kernel
    from raosim.nasa_moc import RaoTopology as _RaoTopology
    from raosim.nasa_moc import set_theta_b as _set_theta_b
    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    Ln = _target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    theta_b_freeze = getattr(config, "theta_b_freeze_deg", None)
    theta_b_seed = math.radians(
        theta_b_freeze if theta_b_freeze is not None
        else config.thetaN_guess_deg
    )

    seed_mode = getattr(config, "ce_seed", "auto")
    if seed_mode not in ("auto", "topology", "linear"):
        raise ValueError(
            f"ce_seed must be 'auto', 'topology', or 'linear', got {seed_mode!r}"
        )

    # ── Preferred: NASA fixed-end topology seed ────────────────────────
    # set_theta_b/calc_lrc_de(end_condition="fixed_end") place D on the
    # *marched* kernel BD so that DE (a true left-running characteristic
    # carrying exactly the B→D mass) ends at the target exit radius, and
    # secant theta_B toward the target length.  DE *is* the Rao control
    # surface, so seeding the CE from it starts the BVP inside the right
    # basin — unlike the legacy linear ramp, which predates the working
    # kernel march.
    topology: _RaoTopology | None = None
    kernel = None
    # theta_b_freeze_deg bypasses the inner secant entirely: the kernel
    # is built at exactly the frozen angle and D/DE come from the
    # fixed-end walk on that kernel (the `if kernel is None` branch
    # below).  This is what lets an outer theta_B loop actually move
    # the frozen kernel — see the config field's comment.
    if seed_mode in ("auto", "topology") and theta_b_freeze is None:
        try:
            topology, kernel = _set_theta_b(
                Rt, config.epsilon, config.length_pct,
                config.gamma, config.pa_over_p0,
                theta_b_init_deg=config.thetaN_guess_deg,
                n_kernel=config.n_kernel,
                n_de_points=max(config.n_control, 12),
                starting_line_method=config.starting_line_method,
                L_target=Ln,
                Ru=config.throat_upstream_radius_factor * Rt,
                end_condition="fixed_end",
                # Seed-quality bracket only: each outer iteration costs a
                # kernel march; ~8 bisections localise theta_B to ~0.2 deg,
                # which is plenty for a BVP seed (the solve owns the
                # length residual).
                max_iter=8,
            )
        except Exception:
            topology, kernel = None, None
            if seed_mode == "topology":
                raise

    if kernel is None:
        kernel = _build_kernel(
            Rt, Rd, theta_b_seed, config.gamma, config.n_kernel,
            starting_line_method=config.starting_line_method,
            Ru=config.throat_upstream_radius_factor * Rt,
        )
        # Honest theta_B provenance for downstream reporting: a frozen
        # override is a *commanded* angle; a guess-angle kernel (secant
        # skipped or failed) is chart-flavoured and must not masquerade
        # as a solved theta_N in the J5 benchmark.
        kernel.theta_b_provenance = (
            "frozen_override" if theta_b_freeze is not None else "seed_guess"
        )
        if topology is None:
            try:
                topology = calc_lrc_de(
                    kernel,
                    x_E=Ln, r_E=Re,
                    gamma=config.gamma, Rt=Rt, epsilon=config.epsilon,
                    pa_over_p0=config.pa_over_p0,
                    n_points=config.n_control,
                    end_condition="fixed_end",
                )
            except Exception:
                topology = None
    kernel_bd_flow_nodes = [node.to_flow_node() for node in kernel.bd]

    ce = _initial_ce_guess(Rt, Re, Ln, config.gamma, config.n_control)
    try:
        Me_target = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    except ValueError:
        Me_target = max(float(ce.M[-1]), 2.0)

    use_topology_ce = (
        seed_mode in ("auto", "topology")
        and topology is not None
        and len(topology.DE) >= 3
        and topology.mass_BD > 1e-9
    )
    if use_topology_ce:
        # Resample DE (D -> E along the C+ characteristic) onto n_control
        # nodes by arc length; this seeds (x, r, M, theta) consistently
        # with the left-Mach-line reconstruction in _unpack_bvp.
        de_x = np.asarray([p.x for p in topology.DE], dtype=float)
        de_r = np.asarray([p.r for p in topology.DE], dtype=float)
        de_M = np.asarray([p.M for p in topology.DE], dtype=float)
        de_th = np.asarray([p.theta for p in topology.DE], dtype=float)
        seg = np.hypot(np.diff(de_x), np.diff(de_r))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = max(float(arc[-1]), 1e-12)
        s = np.linspace(0.0, total, config.n_control)
        ce.x = np.interp(s, arc, de_x)
        ce.r = np.interp(s, arc, de_r)
        ce.M = np.maximum(np.interp(s, arc, de_M), 1.001)
        ce.theta = np.interp(s, arc, de_th)
    else:
        # Legacy axis-to-exit linear ramp (pre-topology behaviour).
        frac = np.linspace(0.0, 1.0, config.n_control)
        ce.M = np.maximum(ce.M, 1.001 + (Me_target - 1.001) * frac ** 0.85)
        ce.theta = np.clip(
            theta_b_seed * (1.0 - 0.55 * frac),
            math.radians(-5.0), math.radians(55.0),
        )
        ce.x = np.linspace(0.0, Ln, config.n_control)
    # Seed kernel_d_fraction.  With the topology CE seed the consistent
    # choice is the topology's own D (ce.r[0] == D.r by construction, so
    # the ce_geometry start residual begins at ~0).  Otherwise fall back
    # to matching the wall-to-D mass flow along BD against the quasi-1D
    # throat target.  The least_squares solve then refines it.
    cap_eff = (
        float(config.kernel_d_fraction_max)
        if config.kernel_d_fraction_max is not None
        else KERNEL_D_FRACTION_MAX
    )
    if use_topology_ce:
        ce.kernel_d_fraction = min(float(topology.d_fraction), cap_eff)
    elif kernel_bd_flow_nodes:
        throat_target = _target_mdot(config.Rt, config.gamma)
        ce.kernel_d_fraction = _seed_kernel_d_fraction(
            kernel_bd_flow_nodes, throat_target, config.gamma,
            fraction_max=cap_eff,
        )
    elif topology is not None:
        ce.kernel_d_fraction = min(float(topology.d_fraction), cap_eff)
    else:
        ce.kernel_d_fraction = min(0.5, cap_eff)

    ce.phi = _phi_from_curve(ce.x, ce.r)
    ce.phi = np.clip(ce.phi, math.radians(5.0), math.radians(88.0))
    return ce, kernel_bd_flow_nodes, topology, kernel


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
    (selected by ``kernel_d_fraction``).

    Phase 6 coupled-wall path additionally appends
    ``n_ce`` free CE↔wall pair fractions (one per CE node, each
    in ``[0, 1]`` indicating the wall arc-length position the CE node
    pairs with for C+ compatibility):

        [M_ce, theta_ce, r_ce,
         (M_w, theta_w, x_w, r_w),   # Phase 6 only
         lambda2, lambda3, log_C, kernel_d_fraction,
         (pair_fraction[0..n_ce-1])  # Phase 6 only
        ]

    Size (uncoupled): ``3*n_ce + 4``.
    Size (coupled):   ``3*n_ce + 4*n_wall + 4 + n_ce`` = ``4*n_ce + 4*n_wall + 4``.
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
    if wall is not None:
        # Append pair_fractions (Phase 6 free pairing); fall back to a
        # linear schedule if the CE doesn't already carry them.
        n_ce = len(ce.r)
        if ce.pair_fractions is not None and len(ce.pair_fractions) == n_ce:
            parts.append(np.asarray(ce.pair_fractions, dtype=float))
        else:
            parts.append(np.linspace(0.0, 1.0, n_ce))
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

    # Phase 6 free pairing: when the coupled-wall path is active, the
    # last ``n_ce`` scalars in ``u`` are the per-CE-node wall
    # pair_fractions.  Layout: ``[..., scalars(4), pair_fractions(n_ce)]``.
    pair_fractions: np.ndarray | None = None
    if wall is not None and u.size >= scalar_start + 4 + n:
        pair_fractions = np.asarray(
            u[scalar_start + 4: scalar_start + 4 + n], dtype=float
        ).copy()

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
        pair_fractions=pair_fractions,
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


def _wall_arc_lengths_normalized(wall: WallSurface) -> np.ndarray:
    """Cumulative arc length along the wall, normalised to ``[0, 1]``."""
    seg = np.hypot(np.diff(wall.x), np.diff(wall.r))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(wall.x))
    return arc / total


def _interp_wall_at_fraction(
    wall: WallSurface,
    arc_norm: np.ndarray,
    frac: float,
) -> FlowNode:
    """Interpolate the wall's ``(x, r, M, theta)`` at arc fraction ``frac``.

    ``arc_norm`` is the normalised arc-length array (precompute once per
    residual evaluation).  ``frac`` is clamped to ``[0, 1]``.
    """
    f = float(max(0.0, min(frac, 1.0)))
    x = float(np.interp(f, arc_norm, wall.x))
    r = float(np.interp(f, arc_norm, wall.r))
    M = float(np.interp(f, arc_norm, wall.M))
    theta = float(np.interp(f, arc_norm, wall.theta))
    return FlowNode(x=x, r=r, M=max(M, 1.000001), theta=theta)


def _coupled_wall_residuals(
    ce: ControlSurface,
    wall: WallSurface,
    config: RaoSolverConfig,
) -> dict[str, np.ndarray]:
    """
    Phase 6 coupled-wall residual blocks.

    Returns four arrays under keys ``"wall_endpoint"``, ``"wall_tangency"``,
    ``"cplus_ce_to_wall"``, ``"wall_intersection"``.  Pairing of CE
    nodes to wall positions is determined by ``ce.pair_fractions[i]``
    (a free BVP unknown in ``[0, 1]`` interpreted as a fractional
    arc-length on the wall).  When ``ce.pair_fractions`` is None the
    pairing falls back to the linear ``i / (n_ce - 1)`` schedule
    (legacy behaviour).
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

    # 3. Free CE↔wall pairing.  For each CE node i, look up the wall
    # position at arc-length fraction ``ce.pair_fractions[i]``.  This is
    # the C+ characteristic from CE[i] landing on the wall — the wall
    # point doesn't have to be one of the discrete wall nodes.
    n_ce = len(ce_nodes)
    cplus: list[float] = []
    intersection: list[float] = []
    if n_ce >= 1 and len(wall_nodes) >= 2:
        theta_scale = math.radians(1.0)
        arc_norm = _wall_arc_lengths_normalized(wall)
        pair_fracs = getattr(ce, "pair_fractions", None)
        if pair_fracs is None or len(pair_fracs) != n_ce:
            # Legacy linear-schedule fallback.
            pair_fracs = np.linspace(0.0, 1.0, n_ce)
        else:
            pair_fracs = np.asarray(pair_fracs, dtype=float)
        for i in range(n_ce):
            ce_pt = ce_nodes[i]
            wall_pt = _interp_wall_at_fraction(wall, arc_norm, float(pair_fracs[i]))
            cplus.append(
                residual_Cplus_axisym(ce_pt, wall_pt, config.gamma)
                / theta_scale
            )
            intersection.append(
                residual_cplus_child_position(ce_pt, wall_pt) / Re_scale
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
    wall: WallSurface | None = None,
) -> np.ndarray:
    """Endpoint, monotonicity, and boundary-state residuals for CE geometry.

    Endpoint at E
    -------------
    With the left-Mach-by-construction refactor, ``ce.x[-1]`` is the
    output of the integrator — pinning *both* CE and wall to ``(L, Re)``
    over-constrains the system (the CE has 3 DOFs per node and the wall
    has 4; the length endpoint cannot be carried by both).  When the
    coupled wall is active we therefore replace the absolute
    ``(ce.x[-1] - L) / L`` pin with a *coincidence residual*
    ``(ce.x[-1] - wall.x[-1]) / L``, asserting that the two surfaces
    meet at E without pinning the CE to L directly.  The wall side
    keeps its own absolute ``(wall.x[-1] - L) / L`` via the
    ``wall_endpoint`` block.  NASA's ``CalcLRCDE`` uses the same
    structural separation (the wall is the length-spanning curve; CE
    is the optimal-thrust supersonic control surface).

    Legacy callers (``couple_wall=False``) keep the direct ``(ce.x[-1] - L)``
    pin to preserve backward compatibility.
    """
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
        start_vals = [
            (float(ce.r[0]) - d_node.r) / r_scale,
        ]
        if getattr(config, "pin_d_theta", True):
            # Full D-state continuity: the CE starts from the kernel
            # point D in position and flow angle.  The corrected
            # characteristic pairing makes this the default smooth Rao
            # attachment; disabling it is a diagnostic for the old
            # position-only branch.
            start_vals.append(
                (float(ce.theta[0]) - d_node.theta) / theta_scale
            )
        if getattr(config, "pin_d_mach", True):
            # Full flow-state continuity at D (Rao 1958: the control
            # surface emanates from a point of the kernel characteristic,
            # so r, theta, and M at D are all kernel values).
            start_vals.append(float(ce.M[0]) - d_node.M)
        start = np.array(start_vals, dtype=float)
    else:
        start = np.zeros(0, dtype=float)

    if wall is not None and len(wall.x) >= 1:
        # Coincidence at E: CE end meets wall end (wall carries L pin).
        endpoint = np.concatenate([
            start,
            np.array([
                (float(ce.x[-1]) - float(wall.x[-1])) / x_scale,
                (float(ce.r[-1]) - float(wall.r[-1])) / r_scale,
            ], dtype=float),
        ])
    else:
        # Legacy: CE absolute pin to (L, Re).
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
    if config.residual_blocks is not None:
        blocks = config.residual_blocks
    else:
        formulation = getattr(config, "formulation", "legacy")
        if formulation == "characteristic":
            blocks = CHARACTERISTIC_RAO_RESIDUAL_BLOCKS
        elif formulation == "legacy":
            blocks = DEFAULT_RAO_RESIDUAL_BLOCKS
        else:
            raise ValueError(
                f"formulation must be 'legacy' or 'characteristic', "
                f"got {formulation!r}"
            )
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
    if (getattr(config, "formulation", "legacy") == "characteristic"
            and ce.x is not None and len(ce.x) >= 2):
        # Rao's length constraint is the *exit station*: L = z_C + ∫cot(φ)dr
        # (see length_integrand's docstring; Rao 1958 functional).  The
        # legacy value Σdx = x_E − x_D omits the start station z_C = x_D,
        # which double-counts x_D against the ce_geometry x_E pin — the two
        # become contradictory unless x_D → 0, and the solver resolves the
        # contradiction by driving kernel_d_fraction to the throat plane
        # (the observed kdf → 0.02 drift; length residual ≈ −x_D/L).
        # Invisible pre-kernel-fix because the degenerate BD held D at
        # x ≈ 0.  Gated to the characteristic formulation to avoid
        # re-baselining legacy tests.
        L_val = float(ce.x[-1])
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
    # CE Mach monotonicity penalty — LEGACY ONLY (2026-06-11): the CE
    # crosses streamlines, so M need not be monotone along it; the
    # exact smooth stationary DE decelerates slightly near D before
    # accelerating, and at 1.0/0.05-Mach weight this penalty is strong
    # enough to displace the correct branch (identified by ibrahim).
    # Zeroed (size preserved for JAX parity) under the characteristic
    # formulation.
    if getattr(config, "formulation", "legacy") == "characteristic":
        mach_monotonic_penalty = np.zeros(max(len(ce.M) - 1, 0), dtype=float)
    else:
        mach_monotonic_penalty = np.maximum(-np.diff(ce.M), 0.0) / 0.05
    regularization = _ce_smoothness_regularization(ce, config.gamma)
    moc_cplus, moc_cminus = _ce_axisymmetric_compatibility_residual_groups(ce, config.gamma)
    ce_geometry = _ce_geometry_residuals(ce, r, config, wall=wall)
    algebraic_stat = _rao_algebraic_stationarity_residuals(ce, config.gamma)
    left_mach = _rao_left_mach_geometry_residuals(ce)
    # Phase 6 free-pairing monotonicity: pair_fractions should be
    # weakly monotone non-decreasing.  Adjacent CE nodes pair with
    # adjacent wall arc-length positions; a non-monotonic pairing
    # would imply CE node i+1's C+ characteristic lands upstream of
    # CE node i's, which is unphysical.
    if ce.pair_fractions is not None and len(ce.pair_fractions) >= 2:
        pair_monotonic_penalty = (
            np.maximum(-np.diff(ce.pair_fractions), 0.0) / 0.01
        )
    else:
        pair_monotonic_penalty = np.zeros(0, dtype=float)
    penalties = np.concatenate([
        incidence_penalty,
        mach_monotonic_penalty,
        0.1 * phi_smooth,
        pair_monotonic_penalty,
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
    if (getattr(config, "formulation", "legacy") == "characteristic"
            and ce.x is not None and len(ce.x) >= 2):
        # Same exit-station length value the residual stack uses (Rao's
        # L = z_C + ∫cot(φ)dr); keeps the bvp_ok gate consistent with
        # the solved residual instead of re-introducing the legacy
        # Σdx = x_E − x_D bookkeeping that double-counts x_D.
        L_val = float(ce.x[-1])
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


def _wall_from_bde_region(
    ce: ControlSurface,
    kernel,
    config: RaoSolverConfig,
) -> tuple[np.ndarray, dict]:
    """Build the wall from the *solved* CE via NASA's BDE-region march.

    This is the physically consistent wall for the characteristic
    formulation: the C+ line through a DE point is DE itself, so the wall
    cannot be coupled to the CE by direct C+ chords (the deleted Phase-6
    pairing blocks).  Instead, NASA's construction (Rice 2003 §3.4;
    ``CalcBDERegion``/``CalcWallContour``) marches the region bounded by
    BD, DE, and the wall: D comes from the solved ``kernel_d_fraction``,
    DE is the solved CE polyline, the wall upstream of the BFE rows is
    the kernel's own throat-arc wall, and each BFE row's wall point is
    located by mass conservation.

    Returns ``(raw_wall, diagnostics, topology_full)`` — the first two
    in the same shape the ``coupled``/``legacy`` wall builders use
    (``moc_compatibility_preserved`` is set by the *caller's*
    forward-MOC audit, not assumed here); ``topology_full`` is the
    §11.7 full-form :class:`raosim.moc_topology.RaoTopology` of the
    solved state (None if the lift failed — wall still returned).
    """
    from raosim.nasa_moc import RaoTopology as _RaoTopology
    from raosim.nasa_moc import calc_bde_region as _calc_bde_region

    diagnostics: dict = {
        "wall_method": "bde",
        "warnings": [],
        "postprocessed": False,
        "moc_compatibility_preserved": False,
    }
    if kernel is None or not config.kernel_bd:
        raise RuntimeError("wall_method='bde' requires a marched kernel")

    # Solved topology: D at the solved arc fraction; DE = solved CE.
    mass_bd, bd_segment = calc_mdot_bd(
        config.kernel_bd, ce.kernel_d_fraction, config.gamma
    )
    d_node = bd_segment[-1]
    de_nodes = _control_surface_flow_nodes(ce)
    if len(de_nodes) < 3:
        raise RuntimeError("solved CE has fewer than 3 nodes")
    mass_de = curve_mass_flux(de_nodes, config.gamma)
    topology = _RaoTopology(
        B=config.kernel_bd[0],
        BD=tuple(config.kernel_bd),
        D=d_node,
        DE=tuple(de_nodes),
        E=de_nodes[-1],
        d_fraction=float(ce.kernel_d_fraction),
        mass_BD=float(mass_bd),
        mass_DE=float(mass_de),
        thrust_coefficient=float("nan"),
        theta_control=float(np.mean(ce.theta)),
        theta_B=float(getattr(kernel, "theta_B", 0.0)),
        rao_stationarity_residual=float("nan"),
    )
    bfe = _calc_bde_region(kernel, topology)
    kernel_wall = [rrc[0] for rrc in kernel.rrcs if rrc]
    wall_pts = [(float(p.x), float(p.r)) for p in kernel_wall]
    wall_pts += [(float(p.x), float(p.r)) for p in bfe.wall_contour]
    raw_wall = np.asarray(wall_pts, dtype=float)

    # §11.7→12.7 wiring: lift the construction into the full-form
    # RaoTopology (Phase 12.6 object — CharPoint curves, full_wall(),
    # closure_report()) so the export path carries the explicit
    # topology of the SOLVED state, not just the wall polyline.
    topo_full = None
    topology_report = {
        "audit_basis": "bde_topology_measured_mesh",
        "passes": False,
        "crossings": 0,
        "bad_rows": [],
        "cplus_rms": 0.0,
        "cminus_rms": 0.0,
        "intersection_rms": 0.0,
        "intersection_max": 0.0,
        "wall_boundary_rms": 0.0,
        "wall_boundary_dx_rms": 0.0,
        "wall_boundary_dr_rms": 0.0,
        "wall_boundary_dx_max": 0.0,
        "wall_boundary_dr_max": 0.0,
        "wall_tangency_rms": 0.0,
        "wall_tangency_max": 0.0,
        "crossing_samples": [],
    }
    measured_mesh = _bde_measured_topology_report(
        bfe,
        config.gamma,
        wall_tangency_tol=math.radians(0.25),
    )
    topology_report.update(measured_mesh)
    try:
        from raosim.moc_topology import build_topology as _build_topology

        topo_full = _build_topology(kernel, topology, bfe)
        closure = {
            k: float(v) for k, v in topo_full.closure_report().items()
        }
        diagnostics["topology_closure"] = closure
        wall = topo_full.full_wall()
        dx = np.diff(wall[:, 0]) if wall.shape[0] >= 2 else np.zeros(0)
        ds = (
            np.hypot(np.diff(wall[:, 0]), np.diff(wall[:, 1]))
            if wall.shape[0] >= 2 else np.zeros(0)
        )
        closure_distance_tol = max(1e-5 * config.Rt, 1e-10)
        mass_rel_tol = max(10.0 * config.residual_tol, 1e-6)
        distance_keys = [
            "BD_starts_at_B", "BD_ends_at_D", "DE_starts_at_D",
            "DE_ends_at_E", "wall_starts_at_B", "wall_ends_at_E",
        ]
        topology_report.update({
            "topology_closure": closure,
            "closure_distance_tol_m": float(closure_distance_tol),
            "mass_rel_tol": float(mass_rel_tol),
            "wall_points": int(wall.shape[0]),
            "wall_min_dx_m": float(np.min(dx)) if dx.size else 0.0,
            "wall_min_segment_m": float(np.min(ds)) if ds.size else 0.0,
            "wall_monotone_x": bool(
                dx.size == 0 or np.min(dx) >= -max(1e-12 * config.Rt, 1e-14)
            ),
            "wall_has_degenerate_segments": bool(
                ds.size > 0 and np.min(ds) <= max(1e-12 * config.Rt, 1e-14)
            ),
        })
        mass_ok = (
            closure.get("mass_rel_mismatch", float("inf")) <= mass_rel_tol
        )
        seams_ok = all(
            closure.get(key, float("inf")) <= closure_distance_tol
            for key in distance_keys
        )
        wall_ok = (
            topology_report["wall_monotone_x"]
            and not topology_report["wall_has_degenerate_segments"]
        )
        # Truncation is normal near the axis; only flag it when a row was
        # cut off in the bulk of the field (band fraction above tol).
        axis_band_fraction = _bde_axis_band_fraction(bfe)
        # A bulk truncation (a row cut off in the body of the field) sits at
        # tens of percent of the radius; a coarse-but-healthy axis closure is
        # a few percent.  0.10 flags the former without tripping on coarse
        # kernels.
        axis_band_tol = 0.10
        truncation_confined = axis_band_fraction <= axis_band_tol
        topology_report.update({
            "mass_closure_passes": bool(mass_ok),
            "seam_closure_passes": bool(seams_ok),
            "wall_geometry_passes": bool(wall_ok),
            "bde_complete_remaining_mesh": bool(bfe.complete_remaining_mesh),
            "bde_wall_contour_complete": bool(bfe.wall_contour_complete),
            "bde_negative_r_truncated_rows": int(
                getattr(bfe, "negative_r_truncated_rows", 0)
            ),
            "bde_truncation_axis_band_fraction": float(axis_band_fraction),
            "bde_truncation_axis_band_tol": float(axis_band_tol),
            "bde_truncation_confined_to_axis": bool(truncation_confined),
        })
        topology_report["passes"] = bool(
            bfe.complete_remaining_mesh
            and bfe.wall_contour_complete
            and truncation_confined
            and mass_ok
            and seams_ok
            and wall_ok
            and topology_report["measured_crossing_passes"]
            and topology_report["measured_wall_tangency_passes"]
        )
    except Exception as exc:  # topology lift is reporting, not the wall
        diagnostics["warnings"].append(
            f"Full-form topology lift failed: {exc}"
        )

    diagnostics.update({
        "bfe_iD": int(bfe.iD),
        "bfe_rows": len(bfe.grid_rows),
        "bfe_full_rows": len(getattr(bfe, "full_grid_rows", ()) or ()),
        "bfe_complete_remaining_mesh": bool(bfe.complete_remaining_mesh),
        "bfe_wall_contour_complete": bool(bfe.wall_contour_complete),
        "bfe_negative_r_truncated_rows": int(
            getattr(bfe, "negative_r_truncated_rows", 0)
        ),
        "kernel_wall_points": len(kernel_wall),
        "bfe_wall_points": len(bfe.wall_contour),
        "solved_mass_BD": float(mass_bd),
        "solved_mass_DE": float(mass_de),
        "net_report": topology_report,
        "wall_tangency_rms": (
            float(topology_report["wall_tangency_rms"])
            if math.isfinite(float(topology_report["wall_tangency_rms"]))
            else None
        ),
        "moc_compatibility_preserved": bool(topology_report["passes"]),
    })
    # In-memory artifacts for diagram rendering (not serialized to summary.json,
    # which is built key-by-key).  Lets the CLI draw the actual solved BDE
    # characteristic net, Rao B-D-E topology, and throat kernel.
    diagnostics["bde_artifacts"] = {
        "kernel": kernel,
        "nasa_topology": topology,
        "bde_region": bfe,
        "topology_full": topo_full,
    }
    if not (bfe.complete_remaining_mesh and bfe.wall_contour_complete):
        diagnostics["warnings"].append(
            "BDE region march incomplete; wall is partial."
        )
    if not topology_report["passes"]:
        diagnostics["warnings"].append(
            "BDE topology audit did not pass; wall construction is not "
            "MOC-promotable."
        )
    return raw_wall, diagnostics, topo_full


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

        parent_start = 1 if prev_pts[0].r < 1e-10 else 0
        for j, child in enumerate(row.interior):
            parent_j = parent_start + j
            if parent_j + 1 >= len(prev_pts):
                break
            links["cplus"].append(MOCNetLink(
                row=row_idx,
                family="cplus",
                role="interior",
                parent=prev_pts[parent_j],
                child=child,
                parent_index=parent_j,
                child_index=child_offset + j,
            ))
            links["cminus"].append(MOCNetLink(
                row=row_idx,
                family="cminus",
                role="interior",
                parent=prev_pts[parent_j + 1],
                child=child,
                parent_index=parent_j + 1,
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

        parent_start = 1 if prev_pts and prev_pts[0].r < 1e-10 else 0
        for j, child in enumerate(row.interior):
            parent_j = parent_start + j
            if parent_j + 1 >= len(prev_pts):
                continue
            geom = residual_intersection(
                prev_pts[parent_j].to_flow_node(),
                prev_pts[parent_j + 1].to_flow_node(),
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

    crossing_count, crossing_samples = _characteristic_crossing_report(
        rows, sample_limit=8
    )
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
        crossing_samples=crossing_samples,
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


def _segment_intersection_point(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> dict | None:
    """Return the xy intersection of two nonparallel line segments."""
    r = b - a
    s = d - c
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) <= 1e-18:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / denom)
    p = a + t * r
    return {"x": float(p[0]), "r": float(p[1])}


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


def _char_point_sample(point: CharPoint) -> dict:
    return {
        "x": float(point.x),
        "r": float(point.r),
        "M": float(point.M),
        "theta_deg": float(math.degrees(point.theta)),
    }


def _link_sample(link: MOCNetLink) -> dict:
    return {
        "row": int(link.row),
        "family": link.family,
        "role": link.role,
        "parent_index": int(link.parent_index),
        "child_index": int(link.child_index),
        "parent": _char_point_sample(link.parent),
        "child": _char_point_sample(link.child),
    }


def _link_crossing_report(
    segments: list[MOCNetLink],
    *,
    sample_limit: int = 0,
    cross_family_only: bool = False,
    endpoint_tol: float = 1e-12,
    max_row_gap: int | None = None,
) -> tuple[int, list[dict]]:
    """Count geometric crossings among explicit characteristic links.

    ``max_row_gap`` restricts the pairwise check to links whose stored
    ``row`` indices differ by at most that many.  A folded MOC cell is a
    *local* event between neighbouring characteristics, so ``max_row_gap=1``
    isolates true adjacent-cell folds; distant-row "crossings" are the
    harmless near-axis convergence of the field (many rows collapsing onto
    the singular axis in the same (x, r) box) and should not be flagged.
    ``None`` (the default) compares every pair — used by the raw
    characteristic-net audit and the fold unit tests.
    """
    crossings = 0
    samples: list[dict] = []
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
            if (
                cross_family_only
                and kept_segments[i].family == kept_segments[j].family
            ):
                continue
            if (
                max_row_gap is not None
                and abs(int(kept_segments[i].row) - int(kept_segments[j].row))
                > max_row_gap
            ):
                continue
            c, d = points[j]
            if _segments_intersect(a, b, c, d):
                intersection = _segment_intersection_point(a, b, c, d)
                if intersection is not None and endpoint_tol > 0.0:
                    p = np.array(
                        [intersection["x"], intersection["r"]], dtype=float
                    )
                    endpoint_distance = min(
                        float(np.linalg.norm(p - q))
                        for q in (a, b, c, d)
                    )
                    if endpoint_distance <= endpoint_tol:
                        continue
                crossings += 1
                if len(samples) < sample_limit:
                    samples.append({
                        "crossing_index": int(crossings),
                        "segment_1": _link_sample(kept_segments[i]),
                        "segment_2": _link_sample(kept_segments[j]),
                        "intersection": intersection,
                    })
    return crossings, samples


def _characteristic_crossing_report(
    rows: list[CharRow],
    *,
    sample_limit: int = 0,
) -> tuple[int, list[dict]]:
    """Count true characteristic crossings and retain a bounded sample."""
    return _link_crossing_report(
        characteristic_net_segments(rows),
        sample_limit=sample_limit,
    )


def _flow_node_char_point(node: FlowNode, gamma: float) -> CharPoint:
    return _make_point(
        float(node.x),
        float(node.r),
        float(node.theta),
        max(float(node.M), 1.000001),
        gamma,
    )


def bde_mesh_links(
    grid_rows: tuple[tuple[FlowNode, ...], ...] | list[tuple[FlowNode, ...]],
    gamma: float,
) -> list[MOCNetLink]:
    """Return measured wall-first BDE mesh links for folding audits.

    BDE rows are stored wall-first after ``CalcWallContour`` cropping.  The
    row-adjacent links are one characteristic family; the same-index links
    between neighbouring rows are the other visible mesh family.  This uses
    only nodes that survived the BDE construction, so it audits the geometry
    actually exported by the BDE wall path instead of a reconstructed forward
    net with different indexing assumptions.
    """
    char_rows = [
        [_flow_node_char_point(node, gamma) for node in row]
        for row in grid_rows
    ]
    links: list[MOCNetLink] = []
    for row_idx, row in enumerate(char_rows):
        for j, (parent, child) in enumerate(zip(row[:-1], row[1:])):
            links.append(MOCNetLink(
                row=row_idx,
                family="bde_row",
                role="wall_first_row",
                parent=parent,
                child=child,
                parent_index=j,
                child_index=j + 1,
            ))
    for row_idx, (prev, curr) in enumerate(
        zip(char_rows[:-1], char_rows[1:]), start=1
    ):
        n_link = min(len(prev), len(curr))
        for j in range(n_link):
            links.append(MOCNetLink(
                row=row_idx,
                family="bde_column",
                role="row_to_row",
                parent=prev[j],
                child=curr[j],
                parent_index=j,
                child_index=j,
            ))
    return links


def _angle_delta(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def _bde_wall_tangency_errors(
    wall_contour: tuple[FlowNode, ...] | list[FlowNode],
) -> np.ndarray:
    """Return wall-segment angle minus averaged BDE wall-node flow angle."""
    errors: list[float] = []
    for p0, p1 in zip(wall_contour[:-1], wall_contour[1:]):
        dx = float(p1.x - p0.x)
        dr = float(p1.r - p0.r)
        if math.hypot(dx, dr) <= 1e-14:
            continue
        wall_theta = math.atan2(dr, dx)
        mesh_theta = 0.5 * (float(p0.theta) + float(p1.theta))
        errors.append(_angle_delta(wall_theta, mesh_theta))
    return np.asarray(errors, dtype=float)


def _bde_axis_band_fraction(bfe) -> float:
    """Largest radius (as a fraction of the mesh's outer radius) at which any
    remaining-mesh row stops marching before its axis point.

    Negative-r truncation is expected, but only near the singular axis.  This
    measures how far off the axis the *worst* row's last interior node sits:
    a small fraction means every row marched down to the near-axis band before
    being closed; a large fraction means a row was truncated in the bulk of
    the field, which would corrupt the mass-flux integration and wall crop.
    """
    rows = getattr(bfe, "full_grid_rows", ()) or bfe.grid_rows
    if not rows:
        return 0.0
    mesh_max_r = max(
        (float(node.r) for row in rows for node in row), default=0.0
    )
    if mesh_max_r <= 0.0:
        return 0.0
    # row[-1] is the axis point (r == 0); row[-2] is the last interior node.
    worst = max(
        (float(row[-2].r) for row in rows if len(row) >= 2), default=0.0
    )
    return float(worst / mesh_max_r)


def _bde_measured_topology_report(
    bfe,
    gamma: float,
    *,
    wall_tangency_tol: float,
) -> dict:
    """Measure BDE mesh crossings and wall tangency from solved nodes."""
    mesh_rows = getattr(bfe, "full_grid_rows", ()) or bfe.grid_rows
    mesh_source = "full_grid_rows" if getattr(bfe, "full_grid_rows", ()) else "grid_rows"
    links = bde_mesh_links(mesh_rows, gamma)
    link_lengths = [
        math.hypot(
            float(link.child.x - link.parent.x),
            float(link.child.r - link.parent.r),
        )
        for link in links
    ]
    endpoint_tol = (
        max(1e-10, 1e-3 * float(np.median(link_lengths)))
        if link_lengths else 1e-10
    )
    # Only adjacent-row cross-family pairs represent a genuine folded cell.
    # Non-adjacent "crossings" are the near-axis convergence of the mesh (all
    # rows collapsing onto the singular axis, which NASA computes and then
    # discards) and are not physical folds — see max_row_gap in
    # ``_link_crossing_report``.
    crossings, samples = _link_crossing_report(
        links,
        sample_limit=8,
        cross_family_only=True,
        endpoint_tol=endpoint_tol,
        max_row_gap=1,
    )
    tangency = _bde_wall_tangency_errors(bfe.wall_contour)
    tangency_rms = _rms(tangency) if tangency.size else float("inf")
    tangency_max = _maxabs(tangency) if tangency.size else float("inf")
    crossing_passes = crossings == 0
    tangency_passes = (
        tangency.size > 0
        and math.isfinite(tangency_max)
        and tangency_max <= wall_tangency_tol
    )
    return {
        "measured_mesh_link_count": int(len(links)),
        "measured_mesh_source": mesh_source,
        "measured_crossing_basis": (
            "adjacent_row_cross_family_folds_on_completed_bde_mesh"
        ),
        "measured_crossing_endpoint_tol_m": float(endpoint_tol),
        "measured_wall_tangency_count": int(tangency.size),
        "crossings": int(crossings),
        "crossing_samples": samples,
        "wall_tangency_rms": float(tangency_rms),
        "wall_tangency_max": float(tangency_max),
        "wall_tangency_rms_deg": float(math.degrees(tangency_rms))
        if math.isfinite(tangency_rms) else float("inf"),
        "wall_tangency_max_deg": float(math.degrees(tangency_max))
        if math.isfinite(tangency_max) else float("inf"),
        "wall_tangency_tol_deg": float(math.degrees(wall_tangency_tol)),
        "measured_crossing_passes": bool(crossing_passes),
        "measured_wall_tangency_passes": bool(tangency_passes),
    }


def characteristic_crossing_samples(
    rows: list[CharRow],
    *,
    limit: int = 8,
) -> list[dict]:
    """Return a small sample of true C+/C- characteristic crossings."""
    return _characteristic_crossing_report(rows, sample_limit=max(0, limit))[1]


def check_characteristic_crossing(rows: list[CharRow]) -> int:
    """Count crossings between true C+/C- characteristic segments."""
    return _characteristic_crossing_report(rows, sample_limit=0)[0]


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

    ce0, kernel_bd_seed, topology_seed, kernel_obj = _initial_ce_from_kernel(config)
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
        # Seed free CE↔wall pair_fractions as a uniform linear schedule
        # (matches the legacy linear pairing as the initial guess; the
        # solver is free to drift them away as physics demands).
        ce0.pair_fractions = np.linspace(0.0, 1.0, n)
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
    # scalars: [lambda2, lambda3, log_C, kernel_d_fraction]
    # The kernel_d_fraction upper bound caps D's arc-length position
    # along BD.  See ``KERNEL_D_FRACTION_MAX`` and
    # ``RaoSolverConfig.kernel_d_fraction_max`` for the physical
    # justification and per-call override.
    kdf_cap = (
        float(config.kernel_d_fraction_max)
        if config.kernel_d_fraction_max is not None
        else KERNEL_D_FRACTION_MAX
    )
    kdf_cap = float(np.clip(kdf_cap, 1e-3, 1.0))
    kdf_floor = float(np.clip(
        getattr(config, "kernel_d_fraction_min", 0.0), 0.0, kdf_cap - 1e-6,
    ))
    lower_parts.append(np.array([-1e3, -1e3, -10.0, kdf_floor]))
    upper_parts.append(np.array([1e3, 1e3, 10.0, kdf_cap]))
    if n_w > 0:
        # pair_fractions ∈ [0, 1] per CE node.
        lower_parts.append(np.zeros(n))
        upper_parts.append(np.ones(n))
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
    elif config.solver_backend == "jax":
        # Differentiable backend: identical residual (J2 parity gate),
        # exact autodiff Jacobian inside Optimistix LM.  Import is local
        # so the NumPy path never requires jax to be installed.
        if getattr(config, "solve_theta_b", False):
            # J3b-2: theta_B joins the unknown vector; the kernel BD is
            # recomputed in-graph per residual evaluation.  result.x
            # excludes theta_B so the unpacking below is unchanged.
            from raosim.jax.theta_b_solve import least_squares_jax_theta_b

            result = least_squares_jax_theta_b(
                solve_config, kernel_obj, u0, lower, upper,
            )
            theta_b_star = float(result.theta_b)
            if kernel_obj is not None:
                # The angle was solved even if it landed on the seed.
                kernel_obj.theta_b_provenance = "bvp_solved"
            if (kernel_obj is not None
                    and abs(theta_b_star - float(kernel_obj.theta_B))
                    > 1e-12):
                # Re-freeze the kernel at the LM-solved angle: the
                # differentiable march is bit-parity with build_kernel,
                # so the rebuilt BD equals the live BD the solver saw.
                from raosim.nasa_moc import build_kernel as _build_kernel_j3b

                kernel_obj = _build_kernel_j3b(
                    config.Rt,
                    config.throat_downstream_radius_factor * config.Rt,
                    theta_b_star, config.gamma, config.n_kernel,
                    starting_line_method=config.starting_line_method,
                    Ru=config.throat_upstream_radius_factor * config.Rt,
                )
                kernel_obj.theta_b_provenance = "bvp_solved"
                kernel_bd_seed = [
                    node.to_flow_node() for node in kernel_obj.bd
                ]
                solve_config = replace(
                    solve_config, kernel_bd=tuple(kernel_bd_seed),
                )
                kernel_points = [
                    _make_point(float(p.x), float(p.r), float(p.theta),
                                max(float(p.M), 1.000001), config.gamma)
                    for p in kernel_bd_seed
                ]
        else:
            from raosim.jax.api import least_squares_jax

            result = least_squares_jax(solve_config, u0, lower, upper)
    elif config.solver_backend == "numpy":
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
    else:
        raise ValueError(
            f"Unknown solver_backend {config.solver_backend!r}; "
            "expected 'numpy' or 'jax'."
        )
    n_wall_unknown = config.n_wall if config.couple_wall else 0
    ce, solved_wall = _unpack_bvp(
        result.x, ce0.r, n_wall=n_wall_unknown,
        kernel_bd=solve_config.kernel_bd, gamma=config.gamma,
    )
    residual_vector = _scaled_rao_bvp_residual(result.x, ce0.r, solve_config)
    F_val, _, L_val = _integrate_ce(ce, config.gamma, config.pa_over_p0)
    if (getattr(config, "formulation", "legacy") == "characteristic"
            and ce.x is not None and len(ce.x) >= 2):
        # Exit-station length (matches the characteristic residual stack).
        L_val = float(ce.x[-1])
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
    # §12.7: the full-form solved topology (moc_topology.RaoTopology),
    # populated by the BDE wall path; None on the other wall methods.
    topology_solved = None
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
    # Rao-1960 parabola-fit chart angles.  Since the J5
    # de-circularization these are comparison data and geometry seeds
    # only (chart-N point for export endpoints, spline end slopes,
    # forward-audit starting line); the *reported* solution angles
    # under the characteristic formulation are solver outputs — see
    # the design-angle reporting block before the RaoSolution return.
    theta_n_chart, theta_e_chart = _design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg
    )
    Nx = Rd * math.sin(theta_n_chart)
    Ny = config.Rt + Rd * (1.0 - math.cos(theta_n_chart))
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
            elif config.wall_method == "bde":
                raw_wall, construction_diagnostics, topology_solved = (
                    _wall_from_bde_region(ce, kernel_obj, solve_config)
                )
            else:
                raise ValueError(
                    "wall_method must be 'coupled', 'legacy', or 'bde'"
                )
            if config.wall_method == "bde":
                net_report_dict = construction_diagnostics.get("net_report")
                if isinstance(net_report_dict, dict):
                    crossings = int(net_report_dict.get("crossings", 0) or 0)
                    if not net_report_dict.get("passes", False):
                        construction_diagnostics["moc_compatibility_preserved"] = False
                        construction_diagnostics.setdefault("warnings", []).append(
                            "BDE topology compatibility audit exceeded tolerance."
                        )
                else:
                    construction_diagnostics["moc_compatibility_preserved"] = False
                    construction_diagnostics.setdefault("warnings", []).append(
                        "BDE topology compatibility audit was unavailable."
                    )
            elif raw_wall.shape[0] >= 3:
                slope_start = math.tan(theta_n_chart)
                slope_end = math.tan(theta_e_chart)
                wall = SplineWall(
                    raw_wall[:, 0],
                    raw_wall[:, 1],
                    slope_start=slope_start,
                    slope_end=slope_end,
                )
                starting = approximate_starting_line(
                    config.Rt,
                    config.throat_downstream_radius_factor * config.Rt,
                    max(theta_n_chart, 1e-4),
                    config.gamma,
                    config.n_kernel,
                    method=config.starting_line_method,
                    transonic_curvature_radius=(
                        config.throat_upstream_radius_factor * config.Rt
                    ),
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
                crossings = int(net_report.crossings)
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

    try:
        export_wall, export_diag = resample_wall_for_export(
            raw_wall,
            start=(Nx, Ny),
            end=(L_target, Re),
            residual_tol=config.residual_tol,
        )
    except RaoEndpointMismatchError as exc:
        # Raw wall misses the target endpoints beyond the export limit.
        # For the no-postprocessing wall paths (bde) this is a *reported*
        # construction shortfall, not a crash: keep the raw wall visible,
        # export the unmodified polyline, and let the reliability ladder
        # downgrade via moc_ok=False.  (The legacy/coupled paths rarely
        # hit this because their builders enforce endpoints upstream.)
        warnings.append(f"Wall export endpoint mismatch: {exc}")
        construction_diagnostics["moc_compatibility_preserved"] = False
        construction_diagnostics["endpoint_mismatch"] = str(exc)
        export_wall = raw_wall.copy()
        export_diag = {
            "endpoint_enforced_for_export": False,
            "monotonic_cleanup_for_export": False,
            "endpoint_mismatch_beyond_limit": True,
        }
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
    # ``_integrate_ce`` integrates the optimized D-E portion only.  Rao's
    # momentum balance is defined on the complete C-D-E surface (Rao 1958,
    # fig. 1 and eq. 2).  Recover C-D from the actual marched-kernel
    # connectivity and integrate both pieces directly.  The old
    # ``Cf_ideal * BD mass fraction`` surrogate is retained below only as a
    # historical diagnostic; it is no longer accepted as a thrust gate.
    cf_de = F_val / max(At, 1e-12)
    full_thrust = None
    if kernel_obj is not None and not bool(getattr(kernel_obj, "fallback_used", False)):
        try:
            full_thrust = full_control_surface_thrust(
                kernel_obj,
                _control_surface_flow_nodes(ce),
                gamma=config.gamma,
                Rt=config.Rt,
                pa_over_p0=config.pa_over_p0,
            )
        except Exception as exc:
            warnings.append(
                f"Full C-D-E control-surface reconstruction failed: {exc}"
            )
    cf = (
        float(full_thrust.cf_cde)
        if full_thrust is not None and full_thrust.complete
        else float(cf_de)
    )
    ce.thrust = float(cf * At)
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
    kernel_bd_full_flux = curve_mass_flux(
        solve_config.kernel_bd or (), config.gamma
    )
    kernel_bd_mass_fraction = (
        mdot_target / kernel_bd_full_flux
        if abs(kernel_bd_full_flux) > 1e-14 else float("nan")
    )
    ce_mass_fraction = (
        mdot_val / kernel_bd_full_flux
        if abs(kernel_bd_full_flux) > 1e-14 else float("nan")
    )
    cf_fraction_of_ideal = (
        cf / cf_ideal
        if math.isfinite(cf_ideal) and abs(cf_ideal) > 1e-12
        else float("nan")
    )
    mass_scaled_cf = (
        cf_ideal * kernel_bd_mass_fraction
        if (
            math.isfinite(cf_ideal)
            and math.isfinite(kernel_bd_mass_fraction)
        )
        else float("nan")
    )
    mass_scaled_cf_rel_error = (
        (cf_de - mass_scaled_cf) / mass_scaled_cf
        if math.isfinite(mass_scaled_cf) and abs(mass_scaled_cf) > 1e-12
        else float("nan")
    )
    partial_control_surface = not (
        full_thrust is not None and full_thrust.complete
    )
    thrust_surface_scope = (
        "full_control_surface_cde"
        if not partial_control_surface
        else "partial_control_surface_de"
    )
    thrust_sanity_applicable = thrust_surface_scope == "full_control_surface_cde"
    full_surface_cf_ok = (
        thrust_sanity_applicable
        and cf > 0.0
        and math.isfinite(cf_rel_error)
        and abs(cf_rel_error) <= FULL_CONTROL_SURFACE_CF_REL_TOL
    )
    segment_mass_scaled_cf_applicable = partial_control_surface
    segment_mass_scaled_cf_ok = (
        partial_control_surface
        and cf > 0.0
        and math.isfinite(mass_scaled_cf_rel_error)
        and abs(mass_scaled_cf_rel_error) <= 2.0e-2
    )
    thrust_sanity_ok = full_surface_cf_ok
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
        "kernel_bd_full_mass_flux": float(kernel_bd_full_flux),
        "kernel_bd_mass_fraction": float(kernel_bd_mass_fraction),
        "ce_mass_fraction_of_full_kernel": float(ce_mass_fraction),
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
        "cf_de_partial": float(cf_de),
        "cf_cd": (
            None if full_thrust is None else float(full_thrust.cf_cd)
        ),
        "cf_ideal": float(cf_ideal),
        "cf_rel_error": float(cf_rel_error),
        "surface_scope": thrust_surface_scope,
        "applicable": bool(thrust_sanity_applicable),
        "gate_basis": (
            "direct_full_cde_surface_integral"
            if thrust_sanity_applicable
            else "unavailable_full_cde_surface"
        ),
        "not_applicable_reason": (
            "kernel C-D connectivity could not be reconstructed; D-E-only "
            "thrust is reported but is not accepted as a physical thrust gate"
            if partial_control_surface
            else None
        ),
        "full_cde_relative_tolerance": FULL_CONTROL_SURFACE_CF_REL_TOL,
        "cde_mass_flux": (
            None if full_thrust is None else float(full_thrust.mass_flux_cde)
        ),
        "kernel_throat_mass_flux": (
            None if full_thrust is None
            else float(full_thrust.kernel_throat_mass_flux)
        ),
        "cde_mass_residual_rel": (
            None if full_thrust is None
            else float(full_thrust.mass_residual_rel)
        ),
        "cde_mass_residual_rel_tol": (
            None if full_thrust is None
            else float(full_thrust.mass_residual_rel_tol)
        ),
        "cd_mass_flux": (
            None if full_thrust is None else float(full_thrust.mass_flux_cd)
        ),
        "de_mass_flux": (
            None if full_thrust is None else float(full_thrust.mass_flux_de)
        ),
        "d_projection_distance": (
            None if full_thrust is None
            else float(full_thrust.d_projection_distance)
        ),
        "d_projection_distance_over_rt": (
            None if full_thrust is None
            else float(full_thrust.d_projection_distance / max(config.Rt, 1e-12))
        ),
        "d_projection_tol_over_rt": (
            None if full_thrust is None
            else float(full_thrust.d_projection_tol_over_rt)
        ),
        "d_state_mach_jump": (
            None if full_thrust is None else float(full_thrust.d_state_mach_jump)
        ),
        "d_state_theta_jump": (
            None if full_thrust is None else float(full_thrust.d_state_theta_jump)
        ),
        "d_mach_jump_tol": (
            None if full_thrust is None else float(full_thrust.d_mach_jump_tol)
        ),
        "d_theta_jump_tol": (
            None if full_thrust is None else float(full_thrust.d_theta_jump_tol)
        ),
        "cde_reconstruction_complete": bool(
            full_thrust is not None and full_thrust.complete
        ),
        "kernel_bd_mass_fraction": float(kernel_bd_mass_fraction),
        "ce_mass_fraction_of_full_kernel": float(ce_mass_fraction),
        "cf_fraction_of_ideal": float(cf_fraction_of_ideal),
        "mass_fraction_scaled_cf": float(mass_scaled_cf),
        "mass_fraction_scaled_cf_rel_error": float(mass_scaled_cf_rel_error),
        "full_control_surface_cf_passes": bool(full_surface_cf_ok),
        "mass_fraction_scaled_cf_applicable": bool(
            segment_mass_scaled_cf_applicable
        ),
        "mass_fraction_scaled_cf_passes": (
            bool(segment_mass_scaled_cf_ok)
            if segment_mass_scaled_cf_applicable else None
        ),
        "mass_fraction_correlation": (
            bool(segment_mass_scaled_cf_ok)
            if segment_mass_scaled_cf_applicable else None
        ),
        "mass_fraction_scaling_is_gate": False,
        "passes": bool(thrust_sanity_ok),
    }

    ce.converged = bool(bvp_ok)
    shock_free = crossings == 0

    # Phase 7 BENCHMARK_VALIDATED promotion: only fires once reviewed
    # exact-variational release criteria have flipped
    # BENCHMARK_VALIDATED_AT_RELEASE to True, the input sits inside the
    # benchmarked sub-grid, and the per-run residuals are tighter than
    # BENCHMARK_VALIDATED_RESIDUAL_TOL.  See the docstrings on those
    # module-level names.
    benchmark_eligible_input = is_within_benchmarked_chart_grid(
        config.epsilon, config.length_pct
    )
    benchmark_eligible_residuals = (
        residuals.max_scaled <= BENCHMARK_VALIDATED_RESIDUAL_TOL
        and abs(residuals.mass_residual_rel) <= BENCHMARK_VALIDATED_RESIDUAL_TOL
        and abs(residuals.length_residual_rel) <= BENCHMARK_VALIDATED_RESIDUAL_TOL
    )
    benchmark_validated_ok = (
        BENCHMARK_VALIDATED_AT_RELEASE
        and benchmark_eligible_input
        and benchmark_eligible_residuals
        and bvp_ok and moc_ok and valid_region_ok and thrust_sanity_ok
    )
    construction_diagnostics["benchmark_validation"] = {
        "at_release": bool(BENCHMARK_VALIDATED_AT_RELEASE),
        "input_within_grid": bool(benchmark_eligible_input),
        "residuals_within_tol": bool(benchmark_eligible_residuals),
        "residual_tol": BENCHMARK_VALIDATED_RESIDUAL_TOL,
        "eligible": bool(benchmark_validated_ok),
    }
    construction_diagnostics["nasa_reference_validation"] = (
        _nasa_reference_validation_diagnostics()
    )

    if benchmark_validated_ok:
        reliability = ContourReliability.BENCHMARK_VALIDATED
    elif bvp_ok and moc_ok and valid_region_ok and thrust_sanity_ok:
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
    if not thrust_sanity_ok and not thrust_sanity_applicable:
        warnings.append(
            "D-E surface thrust coefficient is a partial-control-surface "
            "diagnostic; a complete kernel-connected C-D-E surface was not "
            "available, "
            "so reliability cannot promote on thrust consistency."
        )
    elif not thrust_sanity_ok:
        warnings.append(
            "Full C-D-E surface thrust coefficient failed the sanity gate; "
            "solution is not variational-residual-solved."
        )
    warnings.append(
        "Not hardware-qualified; requires published benchmark comparison, CFD, "
        "thermal/structural review, manufacturing review, inspection, and hot-fire data."
    )
    ce.warnings.extend(warnings)

    theta_e_wall_export = math.atan2(
        export_wall[-1, 1] - export_wall[-2, 1],
        export_wall[-1, 0] - export_wall[-2, 0],
    )
    # J5 de-circularization (2026-06-12): under the characteristic
    # formulation the reported (theta_N, theta_E) are SOLVER outputs:
    #
    #   theta_N := the kernel arc-end angle theta_B the converged BVP
    #              actually used (the seed secant's fixed-end closure,
    #              or the theta_b_freeze_deg override).  B *is* Rao's
    #              wall corner N — the throat arc ends and the bell
    #              begins at the kernel's last RRC.
    #   theta_E := the solved CE exit flow angle theta(E).  E sits on
    #              the wall lip with wall-tangent flow, so theta(E) is
    #              the wall exit angle by construction.  The export-wall
    #              chord is NOT a solver output: with evaluate_moc=False
    #              the raw wall degenerates to the straight chart-N →
    #              exit segment, and the pre-J5 benchmark's "solved
    #              theta_E" column was exactly that chord (pure geometry
    #              reproduces the recorded grid signature to ~0.1 deg:
    #              ~21.1° @L70 / ~18.6° @L80 / ~16.6° @L90, nearly
    #              epsilon-independent).
    #
    # The chart pair stays available as comparison data below;
    # exact-variational vs parabola-chart deltas are expected,
    # documented findings (plan STATUS 2026-06-11h), not solver errors.
    if (getattr(config, "formulation", "legacy") == "characteristic"
            and kernel_obj is not None):
        theta_n_report = float(kernel_obj.theta_B)
        theta_n_source = "kernel_theta_B:" + str(
            getattr(kernel_obj, "theta_b_provenance", "unknown")
        )
        theta_e_report = float(ce.theta[-1])
        theta_e_source = "ce_exit_flow_angle"
    else:
        theta_n_report = float(theta_n_chart)
        theta_n_source = "chart_lookup"
        theta_e_report = float(theta_e_wall_export)
        theta_e_source = "wall_export_slope"
    construction_diagnostics["design_angles"] = {
        "theta_N_reported_deg": math.degrees(theta_n_report),
        "theta_E_reported_deg": math.degrees(theta_e_report),
        "theta_N_source": theta_n_source,
        "theta_E_source": theta_e_source,
        "theta_N_chart_deg": math.degrees(theta_n_chart),
        "theta_E_chart_deg": math.degrees(theta_e_chart),
        "theta_E_wall_export_deg": math.degrees(theta_e_wall_export),
        "chart_provenance": (
            "Rao 1960 ARS J. parabola-fit charts (computed at gamma=1.23; "
            "contours gamma-insensitive per Rao 1961 ARS J. 31(11) p.1490)"
        ),
    }
    return RaoSolution(
        wall_raw=raw_wall,
        wall_export=export_wall,
        control_surface=ce,
        characteristic_net=char_net,
        kernel_points=kernel_points,
        theta_N=theta_n_report,
        theta_E=theta_e_report,
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
        topology_solved=topology_solved,
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

    The matrix always runs on the **NumPy/scipy backend**: ablations sweep
    arbitrary block subsets, including the finite-difference reference
    blocks that are deliberately NumPy-only — the JAX assembly rejects
    them with ``NotImplementedError`` (``raosim/jax/assembly.py``,
    ``SUPPORTED_BLOCKS``).  Before the 2026-06-11 default flip this was
    implicit (the default backend *was* numpy); now it is pinned
    explicitly so the diagnostic keeps working under the JAX defaults.
    """
    selected = RAO_RESIDUAL_ABLATIONS if cases is None else cases
    rows: list[dict] = []
    for name, blocks in selected.items():
        cfg = replace(config, residual_blocks=blocks,
                      evaluate_moc=evaluate_moc, solver_backend="numpy")
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
        Rd_factor=throat_downstream_radius_factor,
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
    Rd_factor: float = 0.382,
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
    Rd = Rd_factor * Rt

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
    x_full = np.concatenate([x_conv, x_throat[1:], wall_x[1:]])
    y_full = np.concatenate([y_conv, y_throat[1:], wall_r[1:]])

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
