from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

import lrekit.components.nozzle.solvers.nasa as nasa
from lrekit.components.nozzle.solvers.variational import RaoSolverConfig, _initial_ce_from_kernel


@pytest.mark.parametrize("budget", [True, False, 0, -1, 2.5, float("nan"), float("inf"), None])
def test_kernel_row_budget_requires_positive_integral_value(budget):
    with pytest.raises(ValueError, match="kernel_max_rrcs"):
        RaoSolverConfig(Rt=1.0, epsilon=10.0, kernel_max_rrcs=budget)
    with pytest.raises(ValueError, match="kernel_max_rrcs"):
        nasa.build_kernel(1.0, 1.0, math.radians(15.2196), 1.4, max_rrcs=budget)


@pytest.fixture(scope="module")
def partial_kernel():
    return nasa.build_kernel(
        1.0, 1.0, math.radians(15.2196), 1.4, n_kernel=101,
        starting_line_method="nasa_visible_kliegel_levine", max_rrcs=1,
    )


def test_build_kernel_retains_explicit_diagnostic_partial_state(partial_kernel):
    assert partial_kernel.reached_wall is False
    assert partial_kernel.fallback_used is False
    assert partial_kernel.max_rrcs == 1
    assert len(partial_kernel.rrcs) == 2
    assert partial_kernel.theta_b_provenance == "kernel_build_angle"


def test_set_theta_b_does_not_construct_de_or_adjust_angle_after_budget_exhaustion(partial_kernel, monkeypatch):
    angles = []

    def build(*args, **kwargs):
        angles.append(args[2])
        assert kwargs["max_rrcs"] == 1
        return partial_kernel

    def unexpected_de(*args, **kwargs):
        pytest.fail("incomplete kernel must be rejected before DE construction")

    monkeypatch.setattr(nasa, "build_kernel", build)
    monkeypatch.setattr(nasa, "calc_lrc_de", unexpected_de)
    with pytest.raises(nasa.IncompleteKernelError) as caught:
        nasa.set_theta_b(1.0, 10.0, 80.0, 1.4, 0.01, max_rrcs=1)
    assert caught.value.reason == "row_budget_exhausted"
    assert len(angles) == 1
    assert partial_kernel.theta_b_provenance == "kernel_build_angle"


def test_preferred_bvp_seed_never_accepts_partial_kernel_or_masks_it_with_fallback(partial_kernel, monkeypatch):
    monkeypatch.setattr(nasa, "set_theta_b", lambda *args, **kwargs: (SimpleNamespace(), partial_kernel))

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("an incomplete preferred kernel must not trigger silent fallback")

    monkeypatch.setattr(nasa, "build_kernel", unexpected_fallback)
    with pytest.raises(nasa.IncompleteKernelError, match="preferred seed"):
        _initial_ce_from_kernel(RaoSolverConfig(Rt=1.0, epsilon=10.0, kernel_max_rrcs=1))


def test_fallback_bvp_seed_rejects_partial_kernel_before_de(partial_kernel, monkeypatch):
    def build(*args, **kwargs):
        assert kwargs["max_rrcs"] == 1
        return partial_kernel

    def unexpected_de(*args, **kwargs):
        pytest.fail("a partial fallback kernel cannot define DE")

    monkeypatch.setattr(nasa, "build_kernel", build)
    monkeypatch.setattr(nasa, "calc_lrc_de", unexpected_de)
    with pytest.raises(nasa.IncompleteKernelError, match="fallback seed"):
        _initial_ce_from_kernel(RaoSolverConfig(
            Rt=1.0, epsilon=10.0, ce_seed="linear", kernel_max_rrcs=1,
        ))


@pytest.fixture(scope="module")
def complete_seed():
    config = RaoSolverConfig(
        Rt=1.0, epsilon=4.973, gamma=1.23, length_pct=64.0452248702277,
        n_control=10, n_kernel=10, ce_seed="topology", kernel_max_rrcs=600,
    )
    _, bd, _, kernel = _initial_ce_from_kernel(config)
    return replace(config, kernel_bd=tuple(bd)), kernel


def test_custom_row_budget_reaches_real_preferred_seed(complete_seed):
    config, kernel = complete_seed
    assert kernel.reached_wall
    assert not kernel.fallback_used
    assert kernel.max_rrcs == config.kernel_max_rrcs == 600


def test_jax_live_angle_refuses_capacity_larger_than_declared_budget(complete_seed):
    pytest.importorskip("jax")
    from lrekit.numerics.jax.theta_b_solve import make_residual_theta_b

    config, kernel = complete_seed
    with pytest.raises(ValueError, match="static row bound exceeds kernel_max_rrcs"):
        make_residual_theta_b(replace(config, kernel_max_rrcs=1), kernel, theta_b_max=kernel.theta_B + 0.01)
    _, capacity = make_residual_theta_b(config, kernel, theta_b_max=kernel.theta_B + 0.01)
    assert capacity <= config.kernel_max_rrcs


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, float("nan"), float("inf"), None])
def test_seed_iteration_budget_requires_positive_integer(value):
    with pytest.raises(ValueError, match="kernel_seed_max_iter"):
        RaoSolverConfig(Rt=1.0, epsilon=10.0, kernel_seed_max_iter=value)
    with pytest.raises(ValueError, match="kernel_seed_max_iter"):
        nasa.set_theta_b(1.0, 10.0, 80.0, 1.4, 0.0, max_iter=value)


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf")])
def test_seed_tolerance_requires_finite_positive_numeric(value):
    with pytest.raises(ValueError, match="kernel_seed_abs_tol"):
        RaoSolverConfig(Rt=1.0, epsilon=10.0, kernel_seed_abs_tol=value)
    with pytest.raises(ValueError, match="kernel_seed_abs_tol"):
        nasa.set_theta_b(1.0, 10.0, 80.0, 1.4, 0.0, abs_tol=value)


@pytest.mark.parametrize(("error", "budget", "reason", "criterion"), [
    (0.0, 25, "parameter_tolerance_met", True),
    (0.01, 1, "iteration_budget_exhausted", False),
    (0.01, 40, "angle_bracket_floor", False),
])
def test_seed_report_distinguishes_tolerance_budget_and_angle_floor(monkeypatch, error, budget, reason, criterion):
    kernel = SimpleNamespace(reached_wall=True, fallback_used=False)
    monkeypatch.setattr(nasa, "build_kernel", lambda *args, **kwargs: kernel)
    monkeypatch.setattr(nasa, "calc_lrc_de", lambda *args, **kwargs: SimpleNamespace(
        E=SimpleNamespace(x=kwargs["x_E"] * (1.0 + error), r=kwargs["r_E"]),
    ))
    _, selected = nasa.set_theta_b(1.0, 10.0, 80.0, 1.4, 0.0, max_iter=budget, abs_tol=1e-8)
    report = selected.theta_b_seed_report
    assert report["termination_reason"] == reason
    assert report["parameter_criterion_met"] is criterion
    assert report["best_abs_relative_error"] == pytest.approx(error)
    assert 0 < report["iterations"] <= budget
    assert report["requested_abs_relative_tolerance"] == 1e-8


def test_preferred_seed_forwards_explicit_precision_controls(monkeypatch):
    actual = nasa.set_theta_b
    observed = {}

    def capture(*args, **kwargs):
        observed.update(kwargs)
        return actual(*args, **kwargs)

    monkeypatch.setattr(nasa, "set_theta_b", capture)
    config = RaoSolverConfig(
        Rt=1.0, epsilon=4.973, gamma=1.23, length_pct=64.0452248702277,
        n_control=8, n_kernel=8, ce_seed="topology",
        kernel_seed_max_iter=4, kernel_seed_abs_tol=2e-7,
    )
    _, _, _, kernel = _initial_ce_from_kernel(config)
    assert observed["max_iter"] == 4
    assert observed["abs_tol"] == 2e-7
    assert kernel.theta_b_seed_report["max_iter"] == 4
    assert kernel.theta_b_seed_report["requested_abs_relative_tolerance"] == 2e-7
