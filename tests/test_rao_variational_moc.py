import math
from types import SimpleNamespace

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.rao_variational import (
    ContourReliability,
    RAO_RESIDUAL_ABLATIONS,
    RaoEndpointMismatchError,
    RaoResidualReport,
    RaoSolverConfig,
    characteristic_net_compatibility_residuals,
    check_characteristic_crossing,
    rao_residual_ablation_matrix,
    rao_variational_moc_contour,
    resample_wall_for_export,
    solve_wall_from_ce_coupled,
    solve_rao_bvp,
    summarize_characteristic_net_compatibility,
)
from raosim.moc import approximate_starting_line, march_coupled_net
from raosim.wall_model import SplineWall


@pytest.mark.smoke
def test_rao_bvp_solution_is_auditable_and_not_hardware_qualified():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=0,
        evaluate_moc=False,
    )
    solution = solve_rao_bvp(cfg)

    assert solution.hardware_qualified is False
    assert "axisymmetric" in solution.assumptions
    assert solution.reliability in set(ContourReliability)
    assert solution.wall_raw.shape[1] == 2
    assert solution.wall_export.shape[1] == 2
    assert solution.residuals.mass_residual_rel == solution.residuals.mass_residual_rel


@pytest.mark.smoke
def test_rao_variational_moc_contour_exposes_raw_and_export_layers():
    contour = rao_variational_moc_contour(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=0,
        evaluate_moc=False,
    )

    assert contour["method"] == "rao_variational_moc"
    assert contour["contour_type"] == "rao_variational_moc"
    assert contour["design_status"] == "experimental_rao_variational_moc_bvp"
    assert contour["hardware_qualified"] is False
    assert contour["rao_full_optimum_claimed"] is False
    assert contour["wall_export_resampled"] is True
    assert "raw_wall_points" in contour
    assert contour["y"][-1] == math.sqrt(10.0) * 0.020


@pytest.mark.smoke
def test_nozzle_geometry_dispatches_rao_variational_moc():
    contour = bell_nozzle_contour(
        Rt=0.020,
        epsilon=10.0,
        method="rao_variational_moc",
        length_pct=80.0,
        rao_moc_n_control=8,
        rao_moc_n_kernel=8,
        rao_moc_max_nfev=0,
        rao_moc_evaluate_moc=False,
    )
    assert contour["method"] == "rao_variational_moc"
    assert contour["design_status"] == "experimental_rao_variational_moc_bvp"


def test_characteristic_crossing_checker_detects_empty_net_as_clean():
    assert check_characteristic_crossing([]) == 0


def test_endpoint_mismatch_raises():
    raw_wall = np.array([
        [0.0, 1.2],
        [1.0, 2.0],
    ])

    with pytest.raises(RaoEndpointMismatchError):
        resample_wall_for_export(
            raw_wall,
            start=(0.0, 1.0),
            end=(1.0, 2.0),
            residual_tol=1e-3,
        )


def test_postprocessed_does_not_claim_residual_solved(monkeypatch):
    import raosim.rao_variational as rv

    def fake_least_squares(_fun, x0, **_kwargs):
        return SimpleNamespace(x=x0, success=True, message="synthetic pass", cost=0.0)

    def fake_report(_residual_vector, _ce, _config, _r_template, *, wall_tangency_rms=None, crossings=0):
        return RaoResidualReport(
            max_scaled=0.0,
            rms_scaled=0.0,
            mass_residual_rel=0.0,
            length_residual_rel=0.0,
            stationarity_rms=0.0,
            regularization_rms=0.0,
            transversality_scaled=0.0,
            wall_tangency_rms=wall_tangency_rms,
            characteristic_crossings=crossings,
        )

    def fake_raw_wall(Rt, epsilon, _gamma, ce, Ln, _n_char):
        theta_n = max(float(ce.theta[0]), math.radians(15.0))
        Rd = 0.382 * Rt
        nx = Rd * math.sin(theta_n)
        ny = Rt + Rd * (1.0 - math.cos(theta_n))
        re = math.sqrt(epsilon) * Rt
        return (
            np.array([[nx, ny], [Ln, re]], dtype=float),
            {
                "warnings": [],
                "postprocessed": True,
                "moc_compatibility_preserved": True,
            },
        )

    monkeypatch.setattr(rv, "least_squares", fake_least_squares)
    monkeypatch.setattr(rv, "_scaled_rao_bvp_residual", lambda *_args: np.zeros(1))
    monkeypatch.setattr(rv, "_build_residual_report", fake_report)
    monkeypatch.setattr(rv, "construct_wall_from_ce_raw", fake_raw_wall)
    monkeypatch.setattr(rv, "_wall_tangency_rms", lambda _raw, _ce: 0.0)

    solution = solve_rao_bvp(RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=1,
        evaluate_moc=True,
        wall_method="legacy",
    ))

    assert solution.reliability == ContourReliability.GEOMETRIC_APPROXIMATION


def test_residual_groups_expose_dominant_blocks():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=0,
        evaluate_moc=False,
    )

    solution = solve_rao_bvp(cfg)
    names = {summary["name"] for summary in solution.residuals.group_summaries}

    assert {
        "mass",
        "length",
        "moc_cminus",
        "ce_geometry",
        "regularization",
        "penalties",
    }.issubset(names)


def test_moc_disabled_ce_residual_gate_converges():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=400,
        residual_tol=5e-3,
        evaluate_moc=False,
    )

    solution = solve_rao_bvp(cfg)

    assert solution.control_surface.converged is True
    assert solution.residuals.max_scaled <= cfg.residual_tol
    assert abs(solution.residuals.mass_residual_rel) <= cfg.residual_tol
    assert abs(solution.residuals.length_residual_rel) <= cfg.residual_tol


def test_ablation_matrix_identifies_both_families_as_invalid_ce_topology():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=200,
        residual_tol=5e-3,
        evaluate_moc=False,
    )

    rows = rao_residual_ablation_matrix(
        cfg,
        cases={
            "default": RAO_RESIDUAL_ABLATIONS["all"],
            "with_both_families": (
                "mass",
                "length",
                "ce_geometry",
                "moc_cplus",
                "moc_cminus",
            ),
        },
    )
    by_case = {row["case"]: row for row in rows}

    assert by_case["default"]["max_scaled"] <= cfg.residual_tol
    assert by_case["with_both_families"]["max_scaled"] > 1.0


def test_coupled_wall_strip_closes_without_endpoint_cheating():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=400,
        residual_tol=5e-3,
        evaluate_moc=False,
    )
    solution = solve_rao_bvp(cfg)

    wall, diagnostics = solve_wall_from_ce_coupled(
        solution.control_surface,
        cfg,
        n_wall=16,
    )

    assert diagnostics["wall_strip_success"] is True
    assert diagnostics["clamp_hits"] == 0
    assert diagnostics["nonmonotonic_x_drops"] == 0
    assert diagnostics["monotonic_x_violations"] == 0
    assert diagnostics["monotonic_r_violations"] == 0
    assert abs(diagnostics["endpoint_dx"]) / solution.wall_export[-1, 0] < 1e-3
    assert abs(diagnostics["endpoint_dr"]) / (math.sqrt(10.0) * 0.020) < 1e-3
    assert diagnostics["wall_tangency_rms"] < math.radians(0.25)
    assert wall[-1, 1] == pytest.approx(math.sqrt(10.0) * 0.020)


def test_characteristic_net_compatibility_diagnostics_are_finite():
    Rt, Rd = 0.02, 0.382 * 0.02
    theta_n = math.radians(30)
    Re = math.sqrt(10.0) * Rt
    Ln = (Re - Rt) / math.tan(math.radians(15)) * 0.8
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    Nx = Rd * math.sin(theta_n)
    wall = SplineWall.from_controls(
        np.linspace(Ny, Re, 5)[1:-1],
        Nx,
        Ny,
        Ln,
        Re,
        theta_n,
    )
    rows = march_coupled_net(
        approximate_starting_line(Rt, Rd, theta_n, 1.4, 10),
        wall,
        1.4,
        max_rows=5,
    )

    residuals = characteristic_net_compatibility_residuals(rows, 1.4)
    summaries = summarize_characteristic_net_compatibility(rows, 1.4)

    assert residuals["cplus"].size > 0
    assert residuals["cminus"].size > 0
    assert all(np.isfinite(residuals[key]).all() for key in residuals)
    assert {item["name"] for item in summaries} == {"net_moc_cplus", "net_moc_cminus"}
