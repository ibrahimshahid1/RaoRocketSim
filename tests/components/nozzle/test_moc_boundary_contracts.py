from dataclasses import replace
import math

import pytest

from lrekit.components.nozzle.solvers.nasa import set_theta_b
import numpy as np

from lrekit.components.nozzle.solvers.variational import (
    RaoSolverConfig,
    _exact_d_free_indices,
    _expand_exact_d_unknowns,
    _initial_ce_from_kernel,
    _pack_bvp,
    _scaled_rao_bvp_residual,
    _BDE_PHYSICAL_TOPOLOGY_GATE_KEYS,
    _build_solver_gate_report,
    _bvp_failure_warnings,
    ContourReliability,
    RaoResidualReport,
    solve_rao_bvp,
)


@pytest.mark.parametrize("factor", [0.382, 0.45])
def test_preferred_topology_seed_preserves_downstream_throat_radius(factor):
    config = RaoSolverConfig(
        Rt=1.0, epsilon=4.973, gamma=1.23, pa_over_p0=0.0,
        length_pct=64.0452248702277, n_control=10, n_kernel=10,
        ce_seed="topology", throat_downstream_radius_factor=factor,
    )
    _, _, topology, kernel = _initial_ce_from_kernel(config)
    assert topology is not None
    assert kernel.theta_b_provenance == "fixed_end_secant"
    assert kernel.Rd == pytest.approx(factor * config.Rt)
    assert kernel.B.x == pytest.approx(kernel.Rd * math.sin(kernel.theta_B))
    assert kernel.B.r == pytest.approx(
        config.Rt + kernel.Rd * (1.0 - math.cos(kernel.theta_B))
    )


@pytest.mark.parametrize("radius", [False, True, 0.0, -1.0, float("nan"), float("inf")])
def test_set_theta_b_refuses_invalid_downstream_radius(radius):
    with pytest.raises(ValueError, match="Rd"):
        set_theta_b(1.0, 10.0, 80.0, 1.4, 0.0, Rd=radius)


@pytest.fixture(scope="module")
def exact_d_seed():
    config = RaoSolverConfig(
        Rt=1.0, epsilon=10.0, gamma=1.4, n_control=8, n_kernel=8,
        exact_d_attachment=True, evaluate_moc=False,
    )
    ce, kernel_bd, _, _ = _initial_ce_from_kernel(config)
    config = replace(config, kernel_bd=tuple(kernel_bd))
    full = _pack_bvp(ce, -0.5, 0.01)
    full[3 * config.n_control + 3] = 0.371234
    return config, ce.r, full


@pytest.mark.parametrize(("pin_mach", "pin_theta"), [(True, True), (False, True), (True, False), (False, False)])
def test_exact_d_reduction_preserves_diagnostic_pin_flags(exact_d_seed, pin_mach, pin_theta):
    from lrekit.components.nozzle.solvers.nasa import calc_mdot_bd

    config, _, template = exact_d_seed
    config = replace(config, pin_d_mach=pin_mach, pin_d_theta=pin_theta)
    n = config.n_control
    indices = _exact_d_free_indices(config, template.size)
    expanded = _expand_exact_d_unknowns(template[indices], template, config)
    _, segment = calc_mdot_bd(config.kernel_bd, expanded[3 * n + 3], config.gamma)
    point = segment[-1]
    assert template.size - indices.size == 3 + pin_mach + pin_theta
    assert expanded[2 * n] == point.r
    assert expanded[0] == (point.M if pin_mach else template[0])
    assert expanded[n] == (point.theta if pin_theta else template[n])
    np.testing.assert_array_equal(expanded[indices], template[indices])


def test_default_boundary_layout_is_unchanged(exact_d_seed):
    config, _, template = exact_d_seed
    config = replace(config, exact_d_attachment=False)
    np.testing.assert_array_equal(_exact_d_free_indices(config, template.size), np.arange(template.size))
    np.testing.assert_array_equal(_expand_exact_d_unknowns(template, template, config), template)


@pytest.mark.parametrize("overrides", [{"formulation": "legacy"}, {"solve_theta_b": True}])
def test_exact_d_rejects_unsupported_formulations(overrides):
    with pytest.raises((ValueError, NotImplementedError), match="exact_d_attachment"):
        RaoSolverConfig(Rt=1.0, epsilon=10.0, exact_d_attachment=True, **overrides)


def test_exact_d_reduced_jax_residual_and_derivative_match_host(exact_d_seed):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lrekit.numerics.jax import assembly

    config, radii, template = exact_d_seed
    indices = _exact_d_free_indices(config, template.size)
    reduced = template[indices]
    sp = assembly.params_from_config(config)
    full_residual = assembly.make_residual(sp)

    def host(vector):
        return _scaled_rao_bvp_residual(
            _expand_exact_d_unknowns(vector, template, config), radii, config,
        )

    def differentiable(vector):
        return full_residual(assembly.expand_exact_d_unknowns(
            vector, template, sp, jnp.asarray(indices),
        ))

    np.testing.assert_allclose(differentiable(reduced), host(reduced), rtol=1e-8, atol=1e-8)
    jacobian = np.asarray(jax.jacfwd(differentiable)(jnp.asarray(reduced)))
    step = 1e-6
    finite_difference = np.column_stack([
        (host(reduced + step * axis) - host(reduced - step * axis)) / (2 * step)
        for axis in np.eye(len(reduced))
    ])
    assert jacobian.shape[1] == template.size - 5
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.linalg.norm(jacobian, axis=0) > 1e-10)
    assert np.linalg.matrix_rank(jacobian) == jacobian.shape[1]
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-4, atol=2e-5)
    fraction_column = int(np.flatnonzero(indices == 3 * config.n_control + 3)[0])
    assert np.linalg.norm(jacobian[:, fraction_column]) > 1e-6


def test_exact_d_attaches_real_nozzle_b_solution_without_relaxing_gates():
    solution = solve_rao_bvp(RaoSolverConfig(
        Rt=1.0, epsilon=4.973, gamma=1.23, pa_over_p0=0.0,
        length_pct=64.0452248702277, n_control=10, n_kernel=10,
        max_nfev=100, residual_tol=0.005, evaluate_moc=True,
        wall_method="bde", solver_backend="numpy", exact_d_attachment=True,
    ))
    artifacts = solution.construction_diagnostics["bde_artifacts"]
    topology = artifacts["nasa_topology"]
    point = solution.control_surface
    assert point.r[0] == topology.D.r
    assert point.M[0] == topology.D.M
    assert point.theta[0] == topology.D.theta
    report = solution.construction_diagnostics["net_report"]
    assert report["topology_closure"]["DE_starts_at_D"] < 1e-12
    assert report["seam_closure_passes"] is True
    assert report["compatibility_tol_deg"] == pytest.approx(0.005)


def test_jax_optimizer_receives_reduced_unknowns_and_returns_full_state(exact_d_seed, monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from lrekit.numerics.jax import api

    config, _, template = exact_d_seed
    config = replace(config, jax_constraint_weight_ladder=(1.0,))
    observed = {}

    def evaluate_once(function, solver, initial, **kwargs):
        observed["unknown_count"] = initial.size
        assert np.all(np.isfinite(np.asarray(function(initial, None))))
        return SimpleNamespace(
            value=initial, result=api.optx.RESULTS.successful,
            stats={"num_steps": 1},
        )

    monkeypatch.setattr(api.optx, "least_squares", evaluate_once)
    lower = np.full(template.size, -100.0)
    upper = np.full(template.size, 100.0)
    lower[3 * config.n_control + 3] = 0.0
    upper[3 * config.n_control + 3] = 1.0
    result = api.least_squares_jax(config, template, lower, upper, max_steps=1)
    assert observed["unknown_count"] == template.size - 5
    assert result.x.size == template.size
    indices = _exact_d_free_indices(config, template.size)
    expected = _expand_exact_d_unknowns(template[indices], template, config)
    np.testing.assert_allclose(result.x, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("active_multiplier_block", ["stationarity", "transversality"])
def test_exact_d_keeps_multiplier_unknowns_when_their_equations_are_active(exact_d_seed, active_multiplier_block):
    config, _, template = exact_d_seed
    config = replace(config, residual_blocks=("mass", "length", active_multiplier_block))
    indices = _exact_d_free_indices(config, template.size)
    scalar_start = 3 * config.n_control
    assert scalar_start in indices
    assert scalar_start + 1 in indices
    assert template.size - indices.size == 3


def test_exact_d_preserves_inactive_multiplier_values(exact_d_seed):
    config, _, template = exact_d_seed
    scalar_start = 3 * config.n_control
    template = template.copy()
    template[scalar_start:scalar_start + 2] = [12.3, -4.5]
    indices = _exact_d_free_indices(config, template.size)
    assert scalar_start not in indices
    assert scalar_start + 1 not in indices
    expanded = _expand_exact_d_unknowns(template[indices], template, config)
    np.testing.assert_array_equal(expanded[scalar_start:scalar_start + 2], [12.3, -4.5])


def _gate_fixture(*, optimizer_success=False, max_scaled=0.003):
    from types import SimpleNamespace

    config = RaoSolverConfig(Rt=1.0, epsilon=10.0, solver_backend="numpy", residual_tol=0.005)
    residuals = RaoResidualReport(
        max_scaled=max_scaled, rms_scaled=0.001, mass_residual_rel=0.001,
        length_residual_rel=0.001, stationarity_rms=0.0,
        algebraic_stationarity_rms=0.0, left_mach_rms=0.0,
        regularization_rms=0.0, transversality_scaled=0.0,
    )
    result = SimpleNamespace(success=optimizer_success, message="evaluation budget exhausted", nfev=100)
    report = {key: True for key in _BDE_PHYSICAL_TOPOLOGY_GATE_KEYS}
    report.update(
        audit_basis="bde_topology_measured_mesh", passes=False,
        bde_physical_mesh_complete=False, cplus_max=0.012, cminus_max=0.002,
        compatibility_tol_deg=0.005, closure_distance_tol_m=1e-5,
        topology_closure={
            "BD_starts_at_B": 0.0, "BD_ends_at_D": 0.0,
            "DE_starts_at_D": 0.001, "DE_ends_at_E": 0.0,
            "wall_starts_at_B": 0.0, "wall_ends_at_E": 0.0,
            "mass_rel_mismatch": 0.5,
        },
        neighbor_state_increment_diagnostic_passes=False,
    )
    return config, residuals, result, report


@pytest.mark.parametrize(("success", "maximum"), [(False, 0.003), (True, 0.006)])
def test_gate_report_distinguishes_optimizer_stopping_from_residual_failure(success, maximum):
    config, residuals, result, net_report = _gate_fixture(
        optimizer_success=success, max_scaled=maximum,
    )
    report = _build_solver_gate_report(
        config, residuals, result, {"net_report": net_report},
        bvp_ok=False, moc_ok=False, valid_region_ok=True,
        validity_available=True, thrust_ok=True,
        reliability=ContourReliability.GEOMETRIC_APPROXIMATION,
    )
    warnings = _bvp_failure_warnings(report)
    assert report["optimizer"]["counter_kind"] == "function_evaluations"
    assert report["residuals"]["passes"] == (maximum <= config.residual_tol)
    assert any("optimizer" in message for message in warnings) == (not success)
    assert any("weighted assembled residual" in message for message in warnings) == (maximum > config.residual_tol)
    assert report["topology"]["failed_gates"] == ["bde_physical_mesh_complete"]
    assert "neighbor_state_increment_diagnostic_passes" not in report["topology"]["gates"]
    assert report["topology"]["seams"]["max_distance"] == 0.001
    assert report["promotion"]["reliability"] == "geometric_approximation"


def test_gate_report_preserves_missing_and_nonfinite_values_as_unavailable():
    import json

    config, residuals, result, report = _gate_fixture()
    del report["axis_condition_passes"]
    report["cplus_max"] = float("inf")
    report["topology_closure"]["DE_starts_at_D"] = float("nan")
    gates = _build_solver_gate_report(
        replace(config, solver_backend="jax"), residuals, result,
        {"net_report": report}, bvp_ok=False, moc_ok=False,
        valid_region_ok=False, validity_available=False, thrust_ok=False,
        reliability=ContourReliability.GEOMETRIC_APPROXIMATION,
    )
    assert gates["topology"]["applicable"] is True
    assert gates["topology"]["missing_gates"] == ["axis_condition_passes"]
    assert gates["topology"]["compatibility"]["cplus_max"] is None
    assert gates["topology"]["seams"]["max_distance"] is None
    assert gates["optimizer"]["counter_kind"] == "lm_steps"
    assert gates["optimizer"]["requested_max_nfev"] == config.max_nfev
    json.dumps(gates, allow_nan=False)


def test_exact_d_reference_profile_passes_all_current_gates():
    from tests.experiments.nozzle.audit_moc_reference_profile import profile_config

    config = profile_config()
    solution = solve_rao_bvp(config)
    diagnostics = solution.construction_diagnostics
    gates = diagnostics["gate_report"]
    assert solution.converged
    assert solution.reliability is ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED
    assert all(gates["promotion"][key] for key in (
        "bvp_passes", "moc_passes", "valid_region_passes", "thrust_passes",
    ))
    assert all(diagnostics["net_report"][key] for key in _BDE_PHYSICAL_TOPOLOGY_GATE_KEYS)
    assert diagnostics["topology_closure"]["DE_starts_at_D"] == 0.0
    assert gates["residuals"]["limit"] == pytest.approx(0.005)
    assert gates["topology"]["compatibility"]["limit"] == pytest.approx(0.005)
    assert gates["topology"]["seams"]["limit"] == pytest.approx(1e-5)
    assert diagnostics["export"]["endpoint_enforced_for_export"] is False
    assert diagnostics["export"]["monotonic_cleanup_for_export"] is False
