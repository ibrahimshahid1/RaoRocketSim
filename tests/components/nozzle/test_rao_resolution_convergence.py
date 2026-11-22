from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lrekit.components.nozzle.solvers.moc import FlowNode
from lrekit.components.nozzle.solvers.variational import (
    ContourReliability,
    ControlSurface,
    RaoFieldConvergenceTolerances,
    RaoMOCNumericalPolicy,
    RaoResidualReport,
    RaoSolverConfig,
    RaoSolution,
    compare_rao_solutions_common_coordinates,
    CHARACTERISTIC_RAO_RESIDUAL_BLOCKS,
)


def _solution(
    count: int,
    *,
    mach_offset: float = 0.0,
    epsilon: float = 4.0,
    include_artifacts: bool = True,
) -> RaoSolution:
    axis = np.linspace(0.0, 1.0, count)
    rows = []
    for x in axis:
        row = []
        for r in axis:
            row.append(
                FlowNode(
                    float(x),
                    float(r),
                    float(2.0 + 0.2 * x + 0.1 * r + mach_offset),
                    float(0.1 + 0.02 * x - 0.01 * r),
                )
            )
        rows.append(row)
    control = ControlSurface(
        r=np.asarray([0.0, 1.0]),
        M=np.asarray([2.0, 2.3]),
        theta=np.asarray([0.1, 0.11]),
        phi=np.asarray([0.0, 0.0]),
        x=np.asarray([0.0, 1.0]),
        kernel_d_fraction=0.5,
        converged=True,
    )
    residuals = RaoResidualReport(
        max_scaled=1.0e-5,
        rms_scaled=1.0e-5,
        mass_residual_rel=1.0e-5,
        length_residual_rel=1.0e-5,
        stationarity_rms=1.0e-5,
        algebraic_stationarity_rms=1.0e-5,
        left_mach_rms=1.0e-5,
        regularization_rms=1.0e-5,
        transversality_scaled=1.0e-5,
        wall_tangency_rms=1.0e-5,
    )
    diagnostics = {
        "solver_mesh_context": {
            "Rt_m": 1.0,
            "epsilon": epsilon,
            "gamma": 1.4,
            "pa_over_p0": 0.01,
            "length_pct": 80.0,
            "throat_upstream_radius_factor": 1.5,
            "throat_downstream_radius_factor": 0.382,
            "starting_line_method": "kliegel_levine",
            "exact_d_attachment": False,
            "exact_exit_attachment": False,
            "pin_d_mach": True,
            "pin_d_theta": True,
            "formulation": "characteristic",
            "residual_blocks": sorted(CHARACTERISTIC_RAO_RESIDUAL_BLOCKS),
            "physics_weight": 0.05,
            "n_control": count,
            "n_kernel": count,
            "wall_method": "bde",
            "kernel_dtheta_limit_deg": 0.25,
            "kernel_massflow_rel_tol": 0.02,
            "kernel_max_rrcs": 500,
            "kernel_reached_wall": True,
        }
    }
    if include_artifacts:
        diagnostics["bde_artifacts"] = {
            "kernel": SimpleNamespace(rrcs=(rows[0],)),
            "bde_region": SimpleNamespace(rows=tuple(rows)),
        }
    wall = np.column_stack((axis, 1.0 + 0.5 * axis))
    return RaoSolution(
        wall_raw=wall,
        wall_export=wall.copy(),
        control_surface=control,
        characteristic_net=[],
        kernel_points=[],
        theta_N=0.4,
        theta_E=0.1,
        thrust_coefficient=1.5,
        residuals=residuals,
        reliability=ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED,
        converged=True,
        shock_free=True,
        hardware_qualified=False,
        assumptions=(),
        construction_diagnostics=diagnostics,
    )


def test_nonnested_linear_fields_pass_at_common_physical_coordinates():
    evidence = compare_rao_solutions_common_coordinates(
        _solution(3),
        _solution(5),
        Rt=1.0,
        gamma=1.4,
    )

    assert evidence.applicable
    assert evidence.passes
    assert evidence.metrics["coarse_node_count"] == 9
    assert evidence.metrics["fine_node_count"] == 25
    assert evidence.metrics["minimum_field_coverage"] == pytest.approx(1.0)
    assert evidence.metrics["mach_max_abs"] < 1.0e-12
    assert evidence.metrics["theta_max_abs_deg"] < 1.0e-12
    assert evidence.metrics["wall_radius_max_over_rt"] < 1.0e-12
    assert evidence.to_dict()["tolerances"]["classification"] == (
        "repository_numerical_convergence_policy"
    )


def test_common_coordinate_mach_exceedance_is_under_resolved():
    evidence = compare_rao_solutions_common_coordinates(
        _solution(3),
        _solution(5, mach_offset=0.02),
        Rt=1.0,
        gamma=1.4,
    )

    assert evidence.applicable
    assert not evidence.passes
    assert evidence.metrics["mach_max_abs"] == pytest.approx(0.02)
    assert "exceed" in evidence.reason


def test_common_coordinate_comparison_fails_closed_without_matching_context():
    evidence = compare_rao_solutions_common_coordinates(
        _solution(3),
        _solution(5, epsilon=5.0),
        Rt=1.0,
        gamma=1.4,
    )

    assert not evidence.applicable
    assert not evidence.passes
    assert "epsilon" in evidence.reason
    assert evidence.metrics == {}


@pytest.mark.parametrize(("Rt", "gamma"), [(2.0, 1.4), (1.0, 1.3)])
def test_common_coordinate_arguments_must_match_solution_provenance(Rt, gamma):
    evidence = compare_rao_solutions_common_coordinates(
        _solution(3),
        _solution(5),
        Rt=Rt,
        gamma=gamma,
    )

    assert not evidence.applicable
    assert not evidence.passes
    assert "arguments" in evidence.reason
    assert evidence.metrics == {}


def test_common_coordinate_comparison_requires_physical_bde_artifacts():
    evidence = compare_rao_solutions_common_coordinates(
        _solution(3),
        _solution(5, include_artifacts=False),
        Rt=1.0,
        gamma=1.4,
    )

    assert not evidence.applicable
    assert not evidence.passes
    assert "BDE" in evidence.reason


@pytest.mark.parametrize(("name", "different"), [
    ("throat_upstream_radius_factor", 1.0),
    ("throat_downstream_radius_factor", 0.45),
    ("starting_line_method", "nasa_visible_kliegel_levine"),
    ("exact_d_attachment", True),
    ("exact_exit_attachment", True),
    ("pin_d_mach", False),
    ("pin_d_theta", False),
    ("formulation", "legacy"),
    ("residual_blocks", ["mass", "length"]),
    ("physics_weight", 1.0),
])
def test_common_coordinate_comparison_refuses_different_models(name, different):
    coarse, fine = _solution(3), _solution(5)
    fine.construction_diagnostics["solver_mesh_context"][name] = different
    evidence = compare_rao_solutions_common_coordinates(coarse, fine, Rt=1.0, gamma=1.4)
    assert not evidence.applicable
    assert not evidence.passes
    assert name in evidence.reason
    assert evidence.metrics == {}


@pytest.mark.parametrize("missing_value", [None, "absent"])
def test_common_coordinate_comparison_does_not_assume_missing_model_defaults(missing_value):
    coarse, fine = _solution(3), _solution(5)
    context = coarse.construction_diagnostics["solver_mesh_context"]
    if missing_value == "absent":
        del context["exact_d_attachment"]
    else:
        context["exact_d_attachment"] = missing_value
    evidence = compare_rao_solutions_common_coordinates(coarse, fine, Rt=1.0, gamma=1.4)
    assert not evidence.applicable
    assert not evidence.passes
    assert "coarse.exact_d_attachment" in evidence.reason


def test_refinement_allows_larger_kernel_budget_without_changing_the_model():
    coarse, fine = _solution(3), _solution(5)
    fine.construction_diagnostics["solver_mesh_context"]["kernel_max_rrcs"] = 1000
    evidence = compare_rao_solutions_common_coordinates(coarse, fine, Rt=1.0, gamma=1.4)
    assert evidence.applicable
    assert evidence.passes


def test_refinement_refuses_incomplete_throat_kernel():
    coarse, fine = _solution(3), _solution(5)
    fine.construction_diagnostics["solver_mesh_context"]["kernel_reached_wall"] = False
    evidence = compare_rao_solutions_common_coordinates(coarse, fine, Rt=1.0, gamma=1.4)
    assert not evidence.applicable
    assert not evidence.passes
    assert "complete throat kernels" in evidence.reason


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mach_max_abs": True}, "not bool"),
        ({"minimum_field_coverage": 1.01}, "must be <= 1"),
        ({"wall_sample_count": 3.5}, "must be an integer"),
    ],
)
def test_common_coordinate_policy_is_strictly_typed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RaoFieldConvergenceTolerances(**kwargs)


def test_rao_moc_heuristic_bundle_is_typed_and_self_classifying():
    policy = RaoMOCNumericalPolicy()
    payload = policy.to_dict()

    assert payload["classification"] == (
        "repository_numerical_policy_not_physical_criterion"
    )
    assert payload["neighbor_mach_diagnostic_limit"] == 0.10
    assert payload["axial_mass_cut_count"] == 7
    assert payload["axial_mass_radial_sample_count"] == 500
    assert payload["full_cde_d_projection_over_rt_tol"] == 0.005
    assert payload["full_cde_d_mach_jump_tol"] == 0.02
    assert payload["full_cde_d_theta_jump_tol_deg"] == 0.5
    assert payload["full_cde_mass_mismatch_rel_tol"] == 0.02
    with pytest.raises(ValueError, match="not bool"):
        RaoMOCNumericalPolicy(axial_mass_cut_count=True)
    with pytest.raises(ValueError, match="must be an integer"):
        RaoMOCNumericalPolicy(axial_mass_radial_sample_count=10.5)


def test_solver_mesh_controls_reject_ambiguous_or_unbounded_values():
    base = {"Rt": 0.02, "epsilon": 10.0}
    with pytest.raises(ValueError, match="not bool"):
        RaoSolverConfig(**base, kernel_dtheta_limit_deg=True)
    with pytest.raises(ValueError, match="no greater than one"):
        RaoSolverConfig(**base, kernel_massflow_rel_tol=1.01)


def test_profile_worst_field_location_matches_existing_comparison_maximum():
    from tests.experiments.nozzle.audit_moc_reference_profile import worst_field_differences

    coarse, fine = _solution(3), _solution(5, mach_offset=0.001)
    evidence = compare_rao_solutions_common_coordinates(coarse, fine, Rt=1.0, gamma=1.4)
    report = worst_field_differences(coarse, fine, Rt=1.0, gamma=1.4)
    worst = report["fields"]["mach"]
    assert report["available"]
    assert abs(worst["signed_interpolated_minus_target"]) == pytest.approx(evidence.metrics["mach_max_abs"])
    assert len(worst["coordinate_x_over_rt_r_over_rt"]) == 2
