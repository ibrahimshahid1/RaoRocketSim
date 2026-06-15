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
    _ce_interp_node,
    _control_surface_flow_nodes,
    characteristic_net_compatibility_residuals,
    characteristic_net_links,
    characteristic_net_segments,
    check_characteristic_crossing,
    curve_mass_flux,
    kernel_bd_segment,
    moc_net_compatibility_report,
    rao_residual_ablation_matrix,
    rao_variational_moc_contour,
    resample_wall_for_export,
    solve_wall_from_ce_coupled,
    solve_rao_bvp,
    summarize_characteristic_net_compatibility,
)
from raosim.moc import _make_point, approximate_starting_line, march_coupled_net, FlowNode
from raosim.rao_residuals import residual_Cminus_axisym, residual_Cplus_axisym
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
            algebraic_stationarity_rms=0.0,
            left_mach_rms=0.0,
            regularization_rms=0.0,
            transversality_scaled=0.0,
            wall_tangency_rms=wall_tangency_rms,
            characteristic_crossings=crossings,
        )

    def fake_raw_wall(Rt, epsilon, _gamma, ce, Ln, _n_char):
        from raosim.rao_variational import _design_angles_rad

        theta_n, _theta_e = _design_angles_rad(epsilon, 80.0)
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


def test_curve_mass_flux_is_positive_for_oblique_surface():
    nodes = [
        FlowNode(x=0.0, r=0.020, M=2.0, theta=math.radians(5.0)),
        FlowNode(x=0.020, r=0.030, M=2.2, theta=math.radians(7.0)),
        FlowNode(x=0.050, r=0.040, M=2.6, theta=math.radians(9.0)),
    ]

    assert curve_mass_flux(nodes, gamma=1.4) > 0.0


def test_phase4_mass_closure_uses_kernel_bd_segment():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=800,
        residual_tol=5e-3,
        evaluate_moc=False,
    )

    solution = solve_rao_bvp(cfg)
    mass_diag = solution.construction_diagnostics["mass_closure"]
    ce_flux = curve_mass_flux(
        _control_surface_flow_nodes(solution.control_surface),
        cfg.gamma,
    )
    bd_segment = kernel_bd_segment(
        solution.kernel_points,
        solution.control_surface.kernel_d_fraction,
    )
    bd_flux = curve_mass_flux(bd_segment, cfg.gamma)

    assert mass_diag["method"] == "kernel_bd_curve_flux"
    assert mass_diag["kernel_bd_nodes"] == len(solution.kernel_points)
    assert 0.0 <= mass_diag["kernel_d_fraction"] <= 1.0
    assert ce_flux == pytest.approx(mass_diag["ce_mass_flux"], rel=1e-12)
    assert bd_flux == pytest.approx(mass_diag["kernel_bd_mass_flux"], rel=1e-12)
    assert abs(solution.residuals.mass_residual_rel) < 1e-4


def test_moc_disabled_ce_residual_gate_keeps_constraints_tight():
    """
    Phase 3 added algebraic Rao stationarity + left-Mach-line geometry as
    soft (0.1-weighted) residuals.  The integral constraints (mass, length)
    still converge tightly with the same iteration budget, but ``max_scaled``
    now reflects how far the seed CE is from a fully-Rao-stationary
    solution.  Phase 6/7 (coupled wall + chart benchmark) should reduce
    these new residuals further.
    """
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

    # Integral constraints stay tight even with the new Rao physics blocks.
    assert abs(solution.residuals.mass_residual_rel) <= 5e-3
    assert abs(solution.residuals.length_residual_rel) <= 1e-2
    # New Rao physics converges to a finite residual; tightening is Phase 6/7.
    assert solution.residuals.algebraic_stationarity_rms < 0.5
    assert solution.residuals.left_mach_rms < 0.5


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

    # Phase 3: the default block set now includes algebraic_stationarity and
    # left_mach as soft constraints, so max_scaled no longer drops below
    # residual_tol with this seed.  The contrast with the heavily-constrained
    # case ("with_both_families" forces forward+backward MOC compatibility on
    # an under-resolved CE) still has to be at least an order of magnitude.
    default_max = by_case["default"]["max_scaled"]
    assert default_max < 0.2
    assert by_case["with_both_families"]["max_scaled"] > 5.0 * default_max


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

    wall_x = np.linspace(wall.x_start, wall.x_end, 32)
    report = moc_net_compatibility_report(
        rows,
        np.column_stack([
            wall_x,
            np.array([wall.r(float(x)) for x in wall_x]),
        ]),
        wall,
        1.4,
        x_scale=Ln,
        r_scale=Re,
        tol=1e-2,
    )
    assert isinstance(report.bad_rows, list)
    assert math.isfinite(report.cplus_rms)
    assert math.isfinite(report.cminus_rms)
    assert math.isfinite(report.wall_boundary_dr_rms)


def test_moc_net_report_passes_on_self_generated_net():
    gamma = 1.4
    r0 = 0.010
    r1 = 0.040
    n_start = 6
    starting = [
        _make_point(
            x=0.002 * t,
            r=r0 + (r1 - r0) * t,
            theta=math.radians(10.0 * t),
            M=3.0 + 0.5 * t,
            gamma=gamma,
        )
        for t in np.linspace(0.0, 1.0, n_start)
    ]
    wall_x0 = 0.005
    wall_x1 = 0.080
    wall_r0 = r1 + 0.005
    wall_r1 = wall_r0 + math.tan(math.radians(5.0)) * (wall_x1 - wall_x0)
    wall = SplineWall(
        np.array([wall_x0, wall_x1]),
        np.array([wall_r0, wall_r1]),
        slope_start=math.tan(math.radians(5.0)),
        slope_end=math.tan(math.radians(5.0)),
    )

    rows = march_coupled_net(starting, wall, gamma, max_rows=3)
    wall_x = np.linspace(wall.x_start, wall.x_end, 32)
    solved_wall = np.column_stack([
        wall_x,
        np.array([wall.r(float(x)) for x in wall_x]),
    ])

    links = characteristic_net_links(rows)
    report = moc_net_compatibility_report(
        rows,
        solved_wall,
        wall,
        gamma,
        x_scale=wall_x1 - wall_x0,
        r_scale=wall_r1,
        tol=2e-2,
    )

    assert len(links["cplus"]) > 0
    assert len(links["cminus"]) > 0
    assert len(characteristic_net_segments(rows)) == len(links["cplus"]) + len(links["cminus"])
    assert report.passes is True
    assert report.cplus_rms < 1e-5
    assert report.cminus_rms < 2e-2
    assert report.intersection_rms < 1e-5
    assert report.crossings == 0
    assert report.bad_rows == []


def test_coupled_wall_strip_uses_cminus_family():
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

    idx = len(wall) // 2
    tau = float(diagnostics["ce_source_tau"][idx])
    p_ce = _ce_interp_node(solution.control_surface, tau)
    p_w = FlowNode(
        x=float(wall[idx, 0]),
        r=float(wall[idx, 1]),
        M=max(float(diagnostics["wall_mach"][idx]), 1.001),
        theta=float(diagnostics["wall_theta"][idx]),
    )
    r_plus = abs(residual_Cplus_axisym(p_ce, p_w, cfg.gamma))
    r_minus = abs(residual_Cminus_axisym(p_ce, p_w, cfg.gamma))

    assert diagnostics["wall_strip_success"] is True
    assert r_minus / math.radians(1.0) < 1e-2
    assert r_minus < r_plus


def test_phase_27_reference_reports_forward_net_failure_precisely():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=1000,
        residual_tol=5e-3,
        evaluate_moc=True,
        wall_method="coupled",
    )

    solution = solve_rao_bvp(cfg)
    diagnostics = solution.construction_diagnostics
    report = diagnostics["net_report"]

    # Phase 3 added algebraic Rao stationarity + left-Mach geometry as soft
    # residuals; the integral constraints still converge tightly while
    # max_scaled now reflects the new physics' convergence floor.
    assert abs(solution.residuals.mass_residual_rel) <= 5e-3
    assert abs(solution.residuals.length_residual_rel) <= 1e-2
    assert diagnostics["wall_strip_success"] is True
    assert diagnostics["endpoint_dx"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["endpoint_dr"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["wall_tangency_rms"] < math.radians(0.25)
    assert diagnostics["clamp_hits"] == 0
    assert diagnostics["nonmonotonic_x_drops"] == 0
    assert diagnostics["moc_compatibility_preserved"] is False
    assert report["passes"] is False
    assert report["cplus_rms"] >= 0.0
    assert report["cminus_rms"] >= 0.0
    assert report["intersection_rms"] >= 0.0
    assert report["wall_boundary_dr_rms"] >= 0.0
    assert isinstance(report["bad_rows"], list)
