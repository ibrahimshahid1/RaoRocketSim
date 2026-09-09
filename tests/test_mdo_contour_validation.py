from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from raosim.mdo.contour_validation import (
    DIRECT_FULL_CDE_CF_BASIS,
    VARIATIONAL_CONTOUR_MODEL_ID,
    VARIATIONAL_FORMULATION,
    VARIATIONAL_GAMMA_MODEL_ID,
    VARIATIONAL_SOLVER_SOURCE_ID,
    VARIATIONAL_VALIDATOR_MODEL_ID,
    VARIATIONAL_VALIDATOR_SOURCE_ID,
    ContourDiscrepancyTolerances,
    VariationalPostSolveConfig,
    VariationalValidationReason,
    VariationalValidationStatus,
    compare_top_and_variational_contours,
    validate_optimized_top_contour,
)
from raosim.rao_existence_scan import (
    DOMAIN_SUPPORTED,
    RESOLUTION_PASSED,
    SAMPLE_CONVERGED_INVALID,
    SAMPLE_CONVERGED_VALID,
    RaoConvergenceContext,
    RaoConvergenceDomain,
    RaoConvergenceSample,
    RaoKernelResolutionEvidence,
)
from raosim.rao_variational import (
    ContourReliability,
    ControlSurface,
    RaoFieldConvergenceEvidence,
    RaoFieldConvergenceTolerances,
    RaoRadialIncrementValidity,
    RaoResidualReport,
    RaoSolution,
)


def _tolerances(value: float = 1.0e-6) -> ContourDiscrepancyTolerances:
    return ContourDiscrepancyTolerances(
        max_radius_error_over_rt=value,
        rms_radius_error_over_rt=value,
        relative_divergent_length_error=value,
        relative_thrust_coefficient_error=value,
    )


def _state(*, cf: float = 1.5):
    return SimpleNamespace(
        performance=SimpleNamespace(
            Rt=1.0,
            epsilon=4.0,
            Pc=2.0e6,
            Pa=1.0e5,
            gamma=1.25,
            OF=2.4,
            Cf_ideal=cf,
        ),
        geometry=SimpleNamespace(
            x=np.array([-1.0, 0.0, 0.5, 1.0]),
            r=np.array([2.2, 1.0, 1.5, 2.0]),
            throat_index=np.asarray(1),
            divergent_model=np.asarray(1),
            station_valid=np.ones(4, dtype=bool),
        ),
    )


def _mission():
    return SimpleNamespace(
        length_pct=80.0,
        throat_ru_factor=1.5,
        throat_rd_factor=0.382,
        converging_half_angle_deg=30.0,
    )


def _resolution() -> RaoKernelResolutionEvidence:
    return RaoKernelResolutionEvidence(
        n_kernel=(8, 12),
        root_success=(True, True),
        theta_b_deg=(28.0, 28.01),
        kdf=(0.5, 0.501),
        residual_norm_geometry=(1.0e-4, 8.0e-5),
        status=RESOLUTION_PASSED,
        reason="test fixture is mesh resolved",
        theta_delta_tolerance_deg=0.1,
        kdf_delta_tolerance=0.01,
    )


def _domain(*, classification: str = SAMPLE_CONVERGED_VALID):
    valid = classification == SAMPLE_CONVERGED_VALID
    rows = []
    for epsilon in (3.0, 5.0):
        row = []
        for length_pct in (70.0, 90.0):
            row.append(RaoConvergenceSample(
                epsilon=epsilon,
                length_pct=length_pct,
                classification=classification,
                reason="test fixture classification",
                theta_b_deg=28.0,
                kdf=0.5,
                residual_norm_geometry=1.0e-4,
                rao_eq5_evaluable=True,
                rao_eq5_valid=valid,
                rao_eq5_min_dr_over_r=0.01 if valid else -0.01,
                resolution=_resolution(),
                source_scan_sha256="0" * 64,
            ))
        rows.append(tuple(row))
    return RaoConvergenceDomain(
        context=RaoConvergenceContext(
            Rt_reference=1.0,
            gamma=1.25,
            pa_over_p0=0.05,
            throat_downstream_radius_factor=0.382,
            throat_upstream_radius_factor=1.5,
            starting_line_method="kliegel_levine",
            kernel_dtheta_limit_deg=0.25,
            kernel_massflow_rel_tol=0.02,
            n_kernel_finest=12,
            n_kernel_ladder=(8, 12),
            root_maxfev=80,
            theta_b_seed_axis_deg=(27.0, 29.0),
            theta_b_seed_policy="fixed_axis",
            theta_b_chart_span_deg=0.0,
            theta_b_chart_count=0,
            theta_b_chart_source_id="",
            kdf_seed_axis=(0.4, 0.6),
            n_de_points=24,
            max_de_steps=240,
            stop_at="mass",
            scan_mode="geometry",
            third_residual="auto",
            residual_weights=(1.0, 1.0, 1.0),
            max_mach=12.0,
            closure_model="smooth",
        ),
        epsilon_axis=(3.0, 5.0),
        length_pct_axis=(70.0, 90.0),
        samples=tuple(rows),
        residual_norm_limit=1.0e-3,
        theta_delta_tolerance_deg=0.1,
        kdf_delta_tolerance=0.01,
        max_epsilon_spacing=3.0,
        max_length_pct_spacing=25.0,
    )


def _validity(*, available: bool = True, valid: bool = True):
    return RaoRadialIncrementValidity(
        segment_indices=(0,),
        segment_midpoint_radius_m=(1.5,),
        equation_5_lhs=(0.1,),
        equation_5_radial_coefficient=(1.0,),
        predicted_dR_over_R=((0.1 if valid else -0.1),),
        predicted_dR_m=((0.15 if valid else -0.15),),
        boundary_function_eq6=(0.1,),
        tolerance_dR_over_R=1.0e-9,
        unevaluable_segment_indices=(() if available else (0,)),
        invalid_segment_indices=(() if valid else (0,)),
        minimum_predicted_dR_over_R=(0.1 if valid else -0.1),
        minimum_predicted_dR_m=(0.15 if valid else -0.15),
        available=available,
        valid=available and valid,
        strictly_positive=available and valid,
        on_boundary=False,
        unevaluable_reasons=(() if available else ("test unavailable",)),
    )


def _solution(
    *,
    converged: bool = True,
    bvp_converged: bool | None = None,
    validity=None,
    cf: float = 1.5,
    full_cde: bool = True,
    exit_radius: float = 2.0,
):
    control = ControlSurface(
        r=np.array([1.0, 2.0]),
        M=np.array([1.5, 3.0]),
        theta=np.array([0.4, 0.1]),
        phi=np.array([0.8, 0.5]),
        x=np.array([0.0, 1.0]),
        kernel_d_fraction=0.5,
        converged=(converged if bvp_converged is None else bvp_converged),
    )
    residuals = RaoResidualReport(
        max_scaled=1.0e-4,
        rms_scaled=5.0e-5,
        mass_residual_rel=1.0e-4,
        length_residual_rel=1.0e-4,
        stationarity_rms=1.0e-4,
        algebraic_stationarity_rms=1.0e-4,
        left_mach_rms=1.0e-4,
        regularization_rms=1.0e-4,
        transversality_scaled=1.0e-4,
        wall_tangency_rms=1.0e-4,
    )
    thrust = {
        "gate_basis": (
            DIRECT_FULL_CDE_CF_BASIS if full_cde else "unavailable_full_cde_surface"
        ),
        "surface_scope": (
            "full_control_surface_cde" if full_cde else "partial_control_surface_de"
        ),
        "cde_reconstruction_complete": full_cde,
        "cf_surface": cf,
        # This internal fixed tolerance is deliberately diagnostic only in B3.
        "passes": False,
    }
    wall = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, exit_radius]])
    return RaoSolution(
        wall_raw=wall.copy(),
        wall_export=wall.copy(),
        control_surface=control,
        characteristic_net=[],
        kernel_points=[],
        theta_N=0.4,
        theta_E=0.1,
        thrust_coefficient=cf,
        residuals=residuals,
        reliability=ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED,
        converged=converged,
        shock_free=bool(converged),
        hardware_qualified=False,
        assumptions=(),
        construction_diagnostics={"thrust_sanity": thrust},
        rao_validity=_validity() if validity is None else validity,
    )


def _field_resolution_evidence(
    *,
    passes: bool = True,
    applicable: bool = True,
    tolerances: RaoFieldConvergenceTolerances | None = None,
) -> RaoFieldConvergenceEvidence:
    policy = tolerances or RaoFieldConvergenceTolerances()
    return RaoFieldConvergenceEvidence(
        applicable=applicable,
        passes=passes and applicable,
        reason=(
            "synthetic common-coordinate fields satisfy the policy"
            if passes and applicable
            else "synthetic common-coordinate field evidence is unavailable"
            if not applicable
            else "synthetic common-coordinate fields exceed the policy"
        ),
        coarse_context={"n_control": 24, "n_kernel": 24},
        fine_context={"n_control": 32, "n_kernel": 32},
        metrics={"mach_max_abs": 0.001 if passes else 0.02},
        tolerances=policy,
    )


def _passing_resolution_comparator(*args, tolerances, **kwargs):
    return _field_resolution_evidence(tolerances=tolerances)


def test_tolerance_policy_is_explicit_and_validated():
    with pytest.raises(ValueError, match="finite and non-negative"):
        ContourDiscrepancyTolerances(-1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="not bool"):
        ContourDiscrepancyTolerances(True, 0.0, 0.0, 0.0)

    policy = _tolerances(0.25)
    assert policy.to_dict() == {
        "max_radius_error_over_rt": 0.25,
        "rms_radius_error_over_rt": 0.25,
        "relative_divergent_length_error": 0.25,
        "relative_thrust_coefficient_error": 0.25,
    }
    float_like_policy = ContourDiscrepancyTolerances(
        "0.1", 0, np.float64(0.2), 0.3
    )
    assert float_like_policy.to_dict() == {
        "max_radius_error_over_rt": 0.1,
        "rms_radius_error_over_rt": 0.0,
        "relative_divergent_length_error": 0.2,
        "relative_thrust_coefficient_error": 0.3,
    }
    with pytest.raises(ValueError, match="characteristic"):
        VariationalPostSolveConfig(
            tolerances=policy,
            formulation="legacy",
        )
    with pytest.raises(ValueError, match="kernel_dtheta_limit_deg"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_dtheta_limit_deg=0.0,
        )
    with pytest.raises(ValueError, match="kernel_dtheta_limit_deg"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_dtheta_limit_deg=float("nan"),
        )
    with pytest.raises(ValueError, match="kernel_massflow_rel_tol"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_massflow_rel_tol=0.0,
        )
    with pytest.raises(ValueError, match="kernel_massflow_rel_tol"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_massflow_rel_tol=float("inf"),
        )
    with pytest.raises(ValueError, match="not bool"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_dtheta_limit_deg=True,
        )
    with pytest.raises(ValueError, match="not bool"):
        VariationalPostSolveConfig(
            tolerances=policy,
            kernel_massflow_rel_tol=True,
        )
    numeric_policy = VariationalPostSolveConfig(
        tolerances=policy,
        kernel_dtheta_limit_deg="0.25",
        kernel_massflow_rel_tol="0.02",
    )
    assert numeric_policy.kernel_dtheta_limit_deg == 0.25
    assert numeric_policy.kernel_massflow_rel_tol == 0.02
    assert numeric_policy.n_control == 24
    assert numeric_policy.n_kernel == 24
    assert numeric_policy.wall_method == "bde"
    assert numeric_policy.resolution_refinement_n_control == 32
    assert numeric_policy.resolution_refinement_n_kernel == 32
    assert (
        numeric_policy.resolved_refinement_kernel_dtheta_limit_deg == 0.25
    )
    with pytest.raises(ValueError, match="must exceed the base mesh count"):
        VariationalPostSolveConfig(
            tolerances=policy,
            resolution_refinement_n_kernel=24,
        )
    with pytest.raises(ValueError, match="both be specified or both be None"):
        VariationalPostSolveConfig(
            tolerances=policy,
            resolution_refinement_n_control=None,
        )


def test_direct_comparison_records_dimensional_and_normalized_deltas():
    result = compare_top_and_variational_contours(
        analytic_x=np.array([-1.0, 0.0, 0.5, 1.0]),
        analytic_r=np.array([2.0, 1.0, 1.5, 2.0]),
        analytic_throat_index=1,
        variational_x=np.array([0.0, 0.5, 1.0]),
        variational_r=np.array([1.0, 1.6, 2.1]),
        Rt=2.0,
        analytic_thrust_coefficient=1.5,
        variational_thrust_coefficient=1.515,
        tolerances=ContourDiscrepancyTolerances(0.1, 0.1, 0.1, 0.02),
        comparison_points=3,
    )

    assert result.max_radius_error_over_rt == pytest.approx(0.05)
    assert result.max_radius_error_m == pytest.approx(0.1)
    assert result.rms_radius_error_m == pytest.approx(
        np.sqrt((0.0**2 + 0.1**2 + 0.1**2) / 3.0)
    )
    assert result.relative_divergent_length_error == pytest.approx(0.0)
    assert result.relative_thrust_coefficient_error == pytest.approx(0.01)
    assert result.exit_radius_error_m == pytest.approx(0.1)
    assert result.exit_radius_error_over_rt == pytest.approx(0.05)
    assert result.to_dict()["max_radius_error_over_rt_limit"] == 0.1
    assert result.passes


def test_direct_comparison_exceedance_is_visible_without_repair():
    result = compare_top_and_variational_contours(
        analytic_x=np.array([0.0, 1.0]),
        analytic_r=np.array([1.0, 2.0]),
        analytic_throat_index=0,
        variational_x=np.array([0.0, 1.2]),
        variational_r=np.array([1.0, 2.2]),
        Rt=1.0,
        analytic_thrust_coefficient=1.5,
        variational_thrust_coefficient=1.4,
        tolerances=_tolerances(0.01),
    )

    assert not result.geometry_passes
    assert not result.performance_passes
    assert not result.passes


def test_missing_domain_is_typed_unknown_and_skips_variational_solve():
    called = False

    def solver(_):
        nonlocal called
        called = True
        raise AssertionError("solver must not run without B5 evidence")

    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(tolerances=_tolerances()),
        solver=solver,
    )

    assert not called
    assert not outcome.available
    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is VariationalValidationReason.CONVERGENCE_DOMAIN_NOT_SUPPLIED
    assert outcome.solved_state["Pc_pa"] == 2.0e6
    assert outcome.solved_state["pa_over_p0"] == pytest.approx(0.05)
    assert outcome.solved_state["gamma"] == 1.25
    assert outcome.solved_state["gamma_model"] == VARIATIONAL_GAMMA_MODEL_ID
    assert outcome.solved_state["mixture_ratio"] == 2.4
    assert outcome.solved_state["variational_formulation"] == (
        VARIATIONAL_FORMULATION
    )
    assert outcome.solved_state["variational_physical_model_id"] == (
        VARIATIONAL_CONTOUR_MODEL_ID
    )
    assert outcome.solved_state["postsolve_validator_selection_id"] == (
        VARIATIONAL_VALIDATOR_MODEL_ID
    )
    assert outcome.solved_state["canonical_variational_solver_source_id"] == (
        VARIATIONAL_SOLVER_SOURCE_ID
    )
    assert outcome.solved_state["variational_solver_source_id"].endswith(
        ".solver"
    )
    assert outcome.solved_state["postsolve_validator_source_id"] == (
        VARIATIONAL_VALIDATOR_SOURCE_ID
    )
    assert outcome.solved_state["variational_kernel_dtheta_limit_deg"] == 0.25
    assert outcome.solved_state["variational_kernel_massflow_rel_tol"] == 0.02


def test_invalid_solved_state_is_unknown_not_a_top_fallback():
    state = _state()
    state.performance.Pc = float("nan")
    outcome = validate_optimized_top_contour(
        state,
        _mission(),
        VariationalPostSolveConfig(tolerances=_tolerances()),
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is VariationalValidationReason.INVALID_SOLVED_STATE
    assert outcome.discrepancy is None


def test_b5_invalid_region_is_a_preflight_failure_without_running_solver():
    called = False

    def solver(_):
        nonlocal called
        called = True
        raise AssertionError("B5-invalid candidates must not run the BVP")

    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(
                classification=SAMPLE_CONVERGED_INVALID
            ),
        ),
        solver=solver,
    )

    assert not called
    assert outcome.available
    assert outcome.status is VariationalValidationStatus.FAIL
    assert outcome.reason is VariationalValidationReason.INVALID_RAO_REGION
    assert outcome.domain.lookup_status == "refused_invalid_rao_region"
    assert outcome.domain.artifact_sha256 == _domain(
        classification=SAMPLE_CONVERGED_INVALID
    ).artifact_sha256


def test_preflight_records_candidate_rt_under_proved_scale_invariance():
    state = _state()
    state.performance.Rt = 2.0
    state.geometry.x = 2.0 * state.geometry.x
    state.geometry.r = 2.0 * state.geometry.r
    outcome = validate_optimized_top_contour(
        state,
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            n_kernel=24,
            convergence_domain=_domain(
                classification=SAMPLE_CONVERGED_INVALID
            ),
        ),
        solver=lambda _: pytest.fail("preflight invalidity must skip the BVP"),
    )

    assert outcome.status is VariationalValidationStatus.FAIL
    assert outcome.domain.context["artifact"]["Rt_reference"] == 1.0
    assert outcome.domain.context["expected"]["Rt_reference"] == 2.0
    assert outcome.domain.context["expected"]["n_kernel_finest"] == 12
    assert outcome.solved_state["variational_n_kernel"] == 24
    assert outcome.domain.artifact_context_sha256 == (
        outcome.domain.expected_context_sha256
    )


def test_shared_context_mismatch_is_unknown_and_skips_solver():
    state = _state()
    state.performance.gamma = 1.3
    called = False

    def solver(_):
        nonlocal called
        called = True
        raise AssertionError("context mismatch must skip the BVP")

    outcome = validate_optimized_top_contour(
        state,
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=solver,
    )

    assert not called
    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.CONVERGENCE_DOMAIN_CONTEXT_MISMATCH
    )
    assert outcome.domain.lookup_status == "context_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kernel_dtheta_limit_deg", 0.5),
        ("kernel_massflow_rel_tol", 0.05),
    ),
)
def test_kernel_march_policy_mismatch_is_unknown_and_skips_solver(
    field, value
):
    called = False

    def solver(_):
        nonlocal called
        called = True
        raise AssertionError("context mismatch must skip the BVP")

    config = VariationalPostSolveConfig(
        tolerances=_tolerances(),
        convergence_domain=_domain(),
        **{field: value},
    )
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        config,
        solver=solver,
    )

    assert not called
    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.CONVERGENCE_DOMAIN_CONTEXT_MISMATCH
    )
    assert outcome.domain.lookup_status == "context_mismatch"


def test_unconverged_attempt_with_negative_eq5_remains_unknown():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: _solution(
            converged=False,
            validity=_validity(valid=False),
        ),
    )

    assert outcome.domain.lookup_status == DOMAIN_SUPPORTED
    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is VariationalValidationReason.VARIATIONAL_NOT_CONVERGED
    assert not outcome.available


def test_full_cde_is_required_before_postsolve_eq5_adjudication():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: _solution(
            full_cde=False,
            validity=_validity(valid=False),
        ),
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is VariationalValidationReason.FULL_CDE_THRUST_UNAVAILABLE
    assert outcome.numerical_converged is True
    assert outcome.physical_moc_closed is True


def test_raw_neighbor_increment_diagnostic_is_not_a_moc_hard_gate():
    solution = _solution(converged=True, bvp_converged=True)
    solution.construction_diagnostics["moc_compatibility_preserved"] = True
    solution.construction_diagnostics["net_report"] = {
        "measured_neighbor_smoothness_passes": False,
        "measured_neighbor_smoothness_is_physical_gate": False,
    }
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: solution,
        resolution_comparator=_passing_resolution_comparator,
    )

    assert outcome.status is VariationalValidationStatus.PASS
    assert outcome.reason is VariationalValidationReason.WITHIN_TOLERANCES
    assert outcome.numerical_converged is True
    assert outcome.physical_moc_closed is True
    assert outcome.discrepancy is not None
    assert outcome.discrepancy.adjudication_eligible
    assert outcome.discrepancy.passes
    assert outcome.resolution_convergence is not None
    assert outcome.resolution_convergence.passes


def test_b3_fails_closed_when_resolution_study_is_not_configured():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
            resolution_refinement_n_control=None,
            resolution_refinement_n_kernel=None,
        ),
        solver=lambda _: _solution(),
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.RESOLUTION_CONVERGENCE_NOT_CONFIGURED
    )
    assert outcome.discrepancy is None


def test_b3_fails_closed_on_under_resolved_common_coordinate_fields():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: _solution(),
        resolution_comparator=lambda *args, tolerances, **kwargs: (
            _field_resolution_evidence(
                passes=False,
                tolerances=tolerances,
            )
        ),
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is VariationalValidationReason.RESOLUTION_UNDER_RESOLVED
    assert outcome.resolution_convergence is not None
    assert not outcome.resolution_convergence.passes
    assert outcome.to_dict()["resolution_convergence"]["passes"] is False


def test_coupled_wall_crossing_audit_is_separate_from_bde_resolution_gate():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
            wall_method="coupled",
        ),
        solver=lambda _: _solution(),
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.RESOLUTION_CONVERGENCE_UNAVAILABLE
    )
    assert "coupled" in outcome.detail
    assert "BDE" in outcome.detail


def test_non_smoothness_moc_failure_refuses_discrepancy_comparison():
    solution = _solution(converged=False, bvp_converged=True)
    solution.construction_diagnostics["moc_compatibility_preserved"] = False
    solution.construction_diagnostics["net_report"] = {
        "measured_crossing_passes": False,
        "measured_neighbor_smoothness_passes": True,
    }
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: solution,
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.MOC_OR_PHYSICAL_CLOSURE_FAILED
    )
    assert outcome.discrepancy is None
    assert "forward net closes" in outcome.detail


def test_converged_negative_eq5_is_a_demonstrated_failure():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: _solution(validity=_validity(valid=False)),
    )

    assert outcome.status is VariationalValidationStatus.FAIL
    assert outcome.reason is VariationalValidationReason.INVALID_RAO_REGION
    assert outcome.rao_validity is not None
    assert not outcome.rao_validity.valid


def test_malformed_variational_export_is_typed_unknown():
    solution = _solution()
    solution.wall_export = np.empty((0, 2))
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(),
            convergence_domain=_domain(),
        ),
        solver=lambda _: solution,
    )

    assert outcome.status is VariationalValidationStatus.UNKNOWN
    assert outcome.reason is (
        VariationalValidationReason.NONFINITE_OR_MALFORMED_GEOMETRY
    )
    assert outcome.discrepancy is None


def test_pass_uses_outer_policy_even_when_internal_cf_sanity_flag_is_false():
    captured = None

    def solver(solver_config):
        nonlocal captured
        captured = solver_config
        return _solution()

    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(1.0e-6),
            convergence_domain=_domain(),
            wall_method="bde",
            physics_weight=1.0,
            kernel_dtheta_limit_deg=0.25,
            kernel_massflow_rel_tol=0.02,
        ),
        solver=solver,
        resolution_comparator=_passing_resolution_comparator,
    )

    assert outcome.status is VariationalValidationStatus.PASS
    assert outcome.available and outcome.passed
    assert outcome.discrepancy.passes
    assert captured.wall_method == "bde"
    assert captured.formulation == VARIATIONAL_FORMULATION
    assert captured.physics_weight == 1.0
    assert captured.kernel_dtheta_limit_deg == pytest.approx(0.25)
    assert captured.kernel_massflow_rel_tol == pytest.approx(0.02)
    assert captured.jax_constraint_weight_ladder == (1.0, 10.0, 30.0, 100.0)
    payload = outcome.to_dict()
    assert payload["domain"]["schema_version"] == 2
    assert payload["resolution_convergence"]["passes"] is True
    assert len(payload["domain"]["artifact_sha256"]) == 64
    assert payload["discrepancy"]["variational_performance_basis"] == (
        DIRECT_FULL_CDE_CF_BASIS
    )


def test_exceeded_outer_discrepancy_flags_candidate():
    outcome = validate_optimized_top_contour(
        _state(),
        _mission(),
        VariationalPostSolveConfig(
            tolerances=_tolerances(0.01),
            convergence_domain=_domain(),
        ),
        solver=lambda _: _solution(cf=1.4, exit_radius=2.2),
        resolution_comparator=_passing_resolution_comparator,
    )

    assert outcome.status is VariationalValidationStatus.FAIL
    assert outcome.reason is VariationalValidationReason.DISCREPANCY_EXCEEDED
    assert outcome.discrepancy is not None
    assert not outcome.discrepancy.passes
