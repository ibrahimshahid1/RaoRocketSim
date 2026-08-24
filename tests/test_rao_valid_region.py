"""Tests for Rao 1999's radial-increment validity gate and diagnostic plots.

Rao-Beck-Booth 1999 (AIAA 99-2584), PDF page 4, obtains Eq. (5) by
eliminating the velocity increment between Eqs. (2) and (3):

    dR/R = {dθ[tan(θ-α)-tan(α)] - dα[tan(θ-α)+tan(α)]}
            / {sin(α)tan(α)sin(θ)/sin(θ+α)}.

The boundary is dR=0 and DE construction requires positive dR.  Eq. (6)'s
``b`` function remains a legacy diagnostic; its sign alone is not the gate.
"""

from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from raosim.moc import FlowNode
from raosim.plotting import plot_characteristic_net, plot_flowfield_mach
from raosim.rao_variational import (
    ControlSurface,
    ContourReliability,
    RaoInvalidRegionError,
    RaoSolverConfig,
    rao_radial_increment_validity,
    rao_valid_region,
    solve_rao_bvp,
)


def _make_smooth_ce_nodes() -> list[FlowNode]:
    """A monotone, gentle expansion -- well inside the valid region."""
    Ms = np.linspace(2.0, 3.5, 8)
    thetas = np.linspace(math.radians(28.0), math.radians(8.0), 8)
    rs = np.linspace(0.025, 0.063, 8)
    xs = np.linspace(0.0, 0.10, 8)
    return [FlowNode(x=float(x), r=float(r), M=float(M), theta=float(t))
            for x, r, M, t in zip(xs, rs, Ms, thetas)]


def _make_short_nozzle_ce_nodes() -> list[FlowNode]:
    """Synthetic Eq. (5) sign crossing with positive geometric ``diff(r)``.

    Segment 0 has positive predicted dR but negative legacy Eq. (6) ``b``;
    segment 1 has negative predicted dR but positive ``b``.  It therefore
    pins the distinction between Rao's complete Eq. (5) and either the
    b-function or a simple geometric radius difference.
    """
    nodes = [
        FlowNode(x=0.000, r=0.0250, M=2.0, theta=math.radians(60.0)),
        FlowNode(x=0.005, r=0.0270, M=4.0, theta=math.radians(55.0)),
        FlowNode(x=0.010, r=0.0300, M=5.0, theta=math.radians(40.0)),
    ]
    return nodes


def _make_boundary_ce_nodes() -> list[FlowNode]:
    """Two nodes whose midpoint-discrete Eq. (5) left side is exactly zero."""
    theta = math.radians(45.0)
    alpha = math.radians(20.0)
    dtheta = math.radians(-2.0)
    difference = math.tan(theta - alpha) - math.tan(alpha)
    summation = math.tan(theta - alpha) + math.tan(alpha)
    dalpha = dtheta * difference / summation
    theta0, theta1 = theta - 0.5 * dtheta, theta + 0.5 * dtheta
    alpha0, alpha1 = alpha - 0.5 * dalpha, alpha + 0.5 * dalpha
    return [
        FlowNode(x=0.000, r=0.030, M=1.0 / math.sin(alpha0), theta=theta0),
        FlowNode(x=0.010, r=0.031, M=1.0 / math.sin(alpha1), theta=theta1),
    ]


def test_radial_increment_is_not_vacuously_valid_for_too_few_nodes():
    result = rao_radial_increment_validity([])
    assert result.available is False
    assert result.valid is False
    assert result.status == "unevaluable"
    with pytest.raises(RaoInvalidRegionError, match="could not be evaluated"):
        result.require_valid()

    # Historical callers retain the old tuple shape and empty-input sentinel.
    min_b, vals = rao_valid_region([])
    assert math.isinf(min_b)
    assert vals == []


@pytest.mark.parametrize(
    ("mach", "reason_pattern"),
    [
        ([1.0, 1.5], "strictly supersonic.*M > 1"),
        ([math.nan, 1.5], "non-finite input.*M0=nan"),
        ([2.0, math.inf], "non-finite input.*M1=inf"),
    ],
)
def test_eq5_control_surface_refuses_unusable_mach_without_substitution(
    mach, reason_pattern
):
    surface = ControlSurface(
        r=np.array([0.025, 0.030]),
        M=np.asarray(mach, dtype=float),
        theta=np.radians([25.0, 20.0]),
        phi=np.radians([60.0, 55.0]),
        x=np.array([0.0, 0.01]),
    )

    result = rao_radial_increment_validity(surface)

    assert result.available is False
    assert result.valid is False
    assert result.status == "unevaluable"
    assert result.unevaluable_segment_indices == (0,)
    assert result.unevaluable_reasons
    assert math.isnan(result.predicted_dR_over_R[0])
    assert result.to_dict()["predicted_dR_over_R"] == [None]
    assert result.to_dict()["unevaluable_reasons"] == list(
        result.unevaluable_reasons
    )
    with pytest.raises(RaoInvalidRegionError, match=reason_pattern):
        result.require_valid()


def test_nasa_reference_matched_reliability_level_exists():
    assert (
        ContourReliability.NASA_REFERENCE_MATCHED.value
        == "nasa_reference_matched"
    )


def test_nasa_reference_gate_separates_verified_port_from_fixture_authority():
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
    diag = solution.construction_diagnostics["nasa_reference_validation"]

    assert diag["canonical_reference_track"] == "visible_source_port"
    assert diag["historical_overlay_track"] == "historical_fixture_overlay"
    assert diag["source_port_matched"] is True
    assert diag["source_port_match_status"] == "software_verified_reference_case"
    assert diag["source_reference_workflow_complete"] is True
    assert diag["source_reference_metrics_available"] is True
    assert diag["source_identity_verified"] is True
    assert len(diag["source_reference_regression_metrics"]) == 4
    assert diag["fixture_overlay_available"] is True
    assert diag["fixture_overlay_is_promotion_authority"] is False
    assert diag["fixture_generator_provenance"] == "unresolved"
    assert diag["eligible"] is False
    assert any("reference certificate covers only" in item for item in diag["blockers"])
    assert any("provenance remains unresolved" in item for item in diag["blockers"])
    assert solution.reliability != ContourReliability.NASA_REFERENCE_MATCHED


def test_eq5_prediction_matches_direct_equations_2_and_3_elimination():
    nodes = _make_smooth_ce_nodes()
    result = rao_radial_increment_validity(nodes)
    p0, p1 = nodes[:2]
    alpha0 = math.asin(1.0 / p0.M)
    alpha1 = math.asin(1.0 / p1.M)
    alpha = 0.5 * (alpha0 + alpha1)
    theta = 0.5 * (p0.theta + p1.theta)
    dtheta = p1.theta - p0.theta
    dalpha = alpha1 - alpha0
    lhs = (
        dtheta * (math.tan(theta - alpha) - math.tan(alpha))
        - dalpha * (math.tan(theta - alpha) + math.tan(alpha))
    )
    coefficient = (
        math.sin(alpha)
        * math.tan(alpha)
        * math.sin(theta)
        / math.sin(theta + alpha)
    )
    predicted = lhs / coefficient
    r_mid = 0.5 * (p0.r + p1.r)
    # Rao p.4 Eq. (5) is the equality left after eliminating dV*/V* between
    # Eq. (2) and the left-Mach compatibility equation, Eq. (3).
    dlog_velocity_eq2 = (
        dtheta * math.tan(theta - alpha)
        - dalpha * (math.tan(theta - alpha) + math.tan(alpha))
    )
    dlog_velocity_eq3 = dtheta * math.tan(alpha) + predicted * coefficient

    assert result.equation_5_lhs[0] == pytest.approx(lhs)
    assert result.equation_5_radial_coefficient[0] == pytest.approx(coefficient)
    assert result.predicted_dR_over_R[0] == pytest.approx(predicted)
    assert result.predicted_dR_m[0] == pytest.approx(r_mid * predicted)
    assert dlog_velocity_eq2 == pytest.approx(dlog_velocity_eq3)
    assert result.predicted_dR_m[0] != pytest.approx(p1.r - p0.r)
    assert result.to_dict()["units"]["predicted_dR"] == "m"
    assert result.to_dict()["units"]["predicted_dR_over_R"] == "1"


def test_radial_increment_positive_on_smooth_expansion():
    nodes = _make_smooth_ce_nodes()
    result = rao_radial_increment_validity(nodes)

    assert result.available is True
    assert result.valid is True
    assert result.strictly_positive is True
    assert result.status == "positive_radial_increment"
    assert min(result.predicted_dR_over_R) > 0.0

    # Eq. (6) remains available for historical diagnostics.
    min_b, vals = rao_valid_region(nodes)
    assert vals, "expected per-segment b values"
    assert min_b > 0.0


def test_eq5_sign_crossing_overrides_b_and_geometric_radius_diagnostics():
    nodes = _make_short_nozzle_ce_nodes()
    result = rao_radial_increment_validity(nodes)

    assert result.predicted_dR_over_R[0] > 0.0
    assert result.predicted_dR_over_R[1] < 0.0
    assert result.boundary_function_eq6[0] < 0.0
    assert result.boundary_function_eq6[1] > 0.0
    assert np.all(np.diff([node.r for node in nodes]) > 0.0)
    assert result.invalid_segment_indices == (1,)
    assert result.valid is False
    assert result.status == "below_boundary"
    with pytest.raises(
        RaoInvalidRegionError,
        match="No discontinuous-exit-flow.*fallback",
    ):
        result.require_valid(context="synthetic provider")


def test_exact_zero_increment_is_the_tolerance_boundary():
    result = rao_radial_increment_validity(_make_boundary_ce_nodes(), tol=1e-10)

    assert result.predicted_dR_over_R[0] == pytest.approx(0.0, abs=1e-14)
    assert result.predicted_dR_m[0] == pytest.approx(0.0, abs=1e-14)
    assert result.available is True
    assert result.valid is True
    assert result.strictly_positive is False
    assert result.on_boundary is True
    assert result.status == "boundary_within_tolerance"


def test_solve_rao_bvp_populates_rao_region_diagnostics():
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
    diag = solution.construction_diagnostics
    validity = solution.rao_validity

    assert validity is not None
    assert validity.available is True
    assert validity.valid is True
    assert "rao_region" in diag
    assert "boundary_min" in diag
    assert diag["boundary_min_kind"] == (
        "legacy_eq6_boundary_function_diagnostic_only"
    )
    assert diag["boundary_min_dR_over_R"] == pytest.approx(
        validity.minimum_predicted_dR_over_R
    )
    radial_diag = diag["rao_radial_increment_validity"]
    assert radial_diag["valid"] is True
    assert radial_diag["boundary"] == {"dR_m": 0.0, "dR_over_R": 0.0}
    assert radial_diag["legacy_boundary_function_is_verdict"] is False
    assert radial_diag["geometric_delta_r_is_verdict"] is False
    assert radial_diag["def_fallback_implemented"] is False
    assert "requires_discontinuous_exit_flow_model" in diag
    assert diag["rao_region"] in {
        "valid_shock_free_region",
        "invalid_short_nozzle_region",
    }
    assert solution.shock_free == (
        solution.residuals.characteristic_crossings == 0 and validity.valid
    )

    contour = solution.to_contour_dict(
        Rt=cfg.Rt,
        epsilon=cfg.epsilon,
        length_pct=cfg.length_pct,
        pa_over_p0=cfg.pa_over_p0,
    )
    assert contour["rao_radial_increment_validity"]["status"] == (
        "positive_radial_increment"
    )


def test_provider_refuses_invalid_or_unevaluable_result_with_typed_error():
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
    refused_states = (
        rao_radial_increment_validity(_make_short_nozzle_ce_nodes()),
        rao_radial_increment_validity([]),
    )
    for validity in refused_states:
        flagged_solution = replace(
            solution,
            rao_validity=validity,
            shock_free=False,
            reliability=ContourReliability.GEOMETRIC_APPROXIMATION,
        )
        with pytest.raises(
            RaoInvalidRegionError,
            match="Rao contour export refused.*No discontinuous-exit-flow",
        ):
            flagged_solution.to_contour_dict(
                Rt=cfg.Rt,
                epsilon=cfg.epsilon,
                length_pct=cfg.length_pct,
                pa_over_p0=cfg.pa_over_p0,
            )

    missing_validity_solution = replace(solution, rao_validity=None)
    with pytest.raises(
        RaoInvalidRegionError,
        match="Rao contour export refused.*validity was not evaluated",
    ):
        missing_validity_solution.to_contour_dict(
            Rt=cfg.Rt,
            epsilon=cfg.epsilon,
            length_pct=cfg.length_pct,
            pa_over_p0=cfg.pa_over_p0,
        )


def test_legacy_eq6_wrapper_remains_diagnostic_only():
    nodes = _make_short_nozzle_ce_nodes()
    min_b, b_values = rao_valid_region(nodes)

    assert min_b < 0.0
    assert any(b < 0.0 for b in b_values)
    assert any(b > 0.0 for b in b_values)


# ---------------------------------------------------------------------
# Plot smoke tests (raw-vs-export trap, §12.3 of REWRITE_PLAN.md)
# ---------------------------------------------------------------------

def _fake_solution_for_plot():
    """Tiny solution-like namespace with two-row characteristic net."""
    class _FakeRow:
        def __init__(self, pts):
            self._pts = pts

        def all_points(self):
            return self._pts

    pts0 = [
        SimpleNamespace(x=0.0, r=0.020, M=1.0, theta=math.radians(0.0)),
        SimpleNamespace(x=0.005, r=0.022, M=1.2, theta=math.radians(8.0)),
    ]
    pts1 = [
        SimpleNamespace(x=0.01, r=0.0, M=1.5, theta=0.0),
        SimpleNamespace(x=0.02, r=0.030, M=1.8, theta=math.radians(15.0)),
    ]
    raw_wall = np.array([[0.005, 0.022], [0.020, 0.030], [0.040, 0.060]],
                        dtype=float)
    export_wall = np.array([[0.005, 0.020], [0.020, 0.040], [0.040, 0.063]],
                           dtype=float)
    ce = SimpleNamespace(
        x=np.array([0.005, 0.020, 0.040]),
        r=np.array([0.022, 0.030, 0.060]),
    )
    kernel = [
        SimpleNamespace(x=0.0, r=0.020 + 0.001 * i,
                        M=1.0 + 0.05 * i, theta=math.radians(2.0 * i))
        for i in range(4)
    ]
    return SimpleNamespace(
        wall_raw=raw_wall,
        wall_export=export_wall,
        control_surface=ce,
        kernel_points=kernel,
        characteristic_net=[_FakeRow(pts0), _FakeRow(pts1)],
    )


def test_plot_characteristic_net_smoke():
    fig = plot_characteristic_net(_fake_solution_for_plot(), show=False)
    assert fig is not None


def test_plot_characteristic_net_geometry_switch_changes_data():
    sol = _fake_solution_for_plot()
    fig_raw = plot_characteristic_net(sol, geometry="raw", show=False)
    fig_exp = plot_characteristic_net(sol, geometry="export", show=False)
    raw_lines = fig_raw.axes[0].lines[0].get_ydata()
    exp_lines = fig_exp.axes[0].lines[0].get_ydata()
    # Raw and export wall arrays differ by construction in the fake fixture.
    assert not np.allclose(raw_lines, exp_lines)


def test_plot_characteristic_net_rejects_bad_geometry():
    with pytest.raises(ValueError):
        plot_characteristic_net(_fake_solution_for_plot(), geometry="oops",
                                show=False)


def test_plot_flowfield_mach_smoke():
    fig = plot_flowfield_mach(_fake_solution_for_plot(), show=False)
    assert fig is not None
