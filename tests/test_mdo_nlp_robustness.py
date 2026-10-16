"""Fail-closed NLP evaluation, independent KKT check and multi-start.

These tests replace the coupled engine with closed-form analytic engines so the
optimizer contract is tested in isolation and in seconds (the pattern of
``tests/test_mdo_variable_policies.py``).  A failing region is produced with
``equinox.error_if``: under ``jax.jit`` it raises the same
``jax.errors.JaxRuntimeError`` that Lineax raises when a linear solve inside
the engine sees non-finite input, and un-jitted it raises
``equinox.EquinoxRuntimeError`` — the two runtime failure classes the NLP must
contain (MDO issues audit 2026-10-16, §1.1 items 1-4).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import raosim.jax  # noqa: F401
import equinox as eqx
import jax
import jax.numpy as jnp

from raosim.mdo import nlp
from raosim.mdo.coolant_htd import ModelCoverageError
from raosim.mdo.schema import MissionSpec, default_design_space


def _references(mission):
    return {spec.name: float(spec.ref()) for spec in default_design_space(mission)}


def _eps_only(mission=None):
    """Pin everything except eps (one free unit coordinate)."""
    mission = mission or MissionSpec()
    fixed = _references(mission)
    fixed.pop("eps")
    return fixed


@jax.custom_vjp
def _nan_gradient_above_six(eps):
    return eps


def _poison_fwd(eps):
    return eps, eps


def _poison_bwd(eps, cotangent):
    # The primal is untouched; only the reverse-mode derivative is lost.
    return (jnp.where(eps > 6.0, jnp.nan, cotangent),)


_nan_gradient_above_six.defvjp(_poison_fwd, _poison_bwd)


def _engine(objective, *, isp=lambda eps: 10.0 * eps + 180.0, fails=None,
            poison_gradient=False):
    """Analytic stand-in for ``solve_engine``.

    ``objective(eps)`` is the mass objective; ``isp(eps)`` closes the
    epsilon row; ``fails(eps)`` marks a region where evaluation raises.
    """

    def solve(design, mission, **kwargs):
        eps = design.eps
        if poison_gradient:
            eps = _nan_gradient_above_six(eps)
        if fails is not None:
            eps = eqx.error_if(eps, fails(eps), "synthetic linear-solve failure")
        value = objective(eps) + design.Pc / 1.0e6
        constraints = {
            spec.engine_key: jnp.asarray(spec.scale)
            for spec in nlp.ENGINE_CONSTRAINT_SPECS
        }
        return SimpleNamespace(
            Isp=isp(eps),
            OF=design.layout.resolve_of(design.OF, mission.OF),
            electric_package_objective_mass=value,
            dry_mass_partial_objective_mass=value + 1.0,
            electric_package_exact_mass=value,
            dry_mass_partial_exact_mass=value + 1.0,
            chamber_mass=SimpleNamespace(total=jnp.asarray(1.0)),
            constraints=constraints,
        )

    return solve


@pytest.fixture
def patch_engine(monkeypatch):
    def apply(solve):
        monkeypatch.setattr(nlp, "solve_engine", solve)
        monkeypatch.setattr(nlp, "chamber_surfaces_for", lambda mission: object())
        return solve
    return apply


# --------------------------------------------------------------------------- #
# R1: a bad iterate is rejected, not fatal                                    #
# --------------------------------------------------------------------------- #
def test_trial_point_failure_is_backtracked_not_raised(patch_engine):
    # min (eps-5)^2 on eps in [4, 7] from eps = 7: SLSQP's first full step
    # lands near eps = 6.3, inside the failing band; the penalty makes the line
    # search backtrack and the BFGS-updated step then jumps the band.
    patch_engine(_engine(lambda eps: (eps - 5.0) ** 2,
                         fails=lambda eps: (eps > 6.1) & (eps < 6.5)))
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([1.0]),
    )
    assert result.evaluation_failures >= 1
    assert result.success
    assert result.numerical_status == "pass"
    assert result.design["eps"] == pytest.approx(5.0, abs=1e-5)
    assert "failed numerically" in result.message
    assert result.kkt_status == "pass"


def test_nonfinite_derivative_falls_back_to_central_differences(patch_engine):
    # Exact reverse-mode gradient is NaN above eps = 6; the primal is finite,
    # so the fallback recovers the derivative and the solve still converges.
    patch_engine(_engine(lambda eps: (eps - 5.0) ** 2, poison_gradient=True))
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([1.0]),
    )
    assert result.derivative_fallbacks >= 1
    assert result.evaluation_failures == 0
    assert result.success
    assert result.design["eps"] == pytest.approx(5.0, abs=1e-5)


def test_guarded_problem_fd_fallback_matches_exact_derivative():
    def f(u):
        return jnp.sum((u - 0.3) ** 2)

    def c(u):
        return jnp.stack([u[0] - 0.1, 1.0 - u[0] * u[1]])

    broken = lambda u: jnp.full_like(u, jnp.nan)  # noqa: E731
    broken_jac = lambda u: jnp.full((2, u.size), jnp.nan)  # noqa: E731
    guard = nlp._GuardedProblem(jax.jit(f), broken, jax.jit(c), broken_jac, 2)
    u = np.array([0.6, 0.45])
    grad, ok = guard.gradient(u)
    jac, jok = guard.jacobian(u)
    assert ok and jok and guard.derivative_fallbacks == 2
    np.testing.assert_allclose(grad, np.asarray(jax.grad(f)(u)), atol=1e-8)
    np.testing.assert_allclose(jac, np.asarray(jax.jacrev(c)(u)), atol=1e-8)
    # One-sided at a box face, never stepping outside [0, 1].
    edge = np.array([1.0, 0.0])
    grad_edge, ok_edge = guard.gradient(edge)
    assert ok_edge
    np.testing.assert_allclose(grad_edge, np.asarray(jax.grad(f)(edge)), atol=1e-5)


def test_penalty_exceeds_every_finite_objective_and_violates_every_row():
    def obj(u):
        return eqx.error_if(jnp.sum(u), u[0] > 0.5, "synthetic")

    guard = nlp._GuardedProblem(
        jax.jit(obj), None, jax.jit(lambda u: jnp.stack([u[0], u[0]])), None, 2,
    )
    good = guard.fun(np.array([0.4]))
    bad = guard.fun(np.array([0.9]))
    assert np.isfinite(bad) and bad > good
    assert np.all(guard.cons(np.array([0.9])) < 0.0)
    assert guard.evaluation_failures == 1
    # Cached: the same failed point is not re-counted when SLSQP asks again.
    guard.cons(np.array([0.9]))
    assert guard.evaluation_failures == 1


def test_failure_at_returned_point_is_typed_not_raised(patch_engine):
    # Every point fails: the all-fixed evaluation cannot be finite.
    patch_engine(_engine(lambda eps: (eps - 5.0) ** 2,
                         fails=lambda eps: eps > 0.0))
    result = nlp.solve_min_mass(
        MissionSpec(), 200.0, fixed_variables=_references(MissionSpec()),
    )
    assert not result.success
    assert result.numerical_status == "fail"
    assert result.workflow_status == "fail"
    assert result.physics_status == "unknown"
    assert not result.feasible
    assert np.isnan(result.objective_mass)
    assert "numerical evaluation failed at the returned point" in result.message
    assert set(result.constraints) == set(nlp._margin_names())


def test_configuration_errors_still_raise(patch_engine, monkeypatch):
    patch_engine(_engine(lambda eps: (eps - 5.0) ** 2))
    with pytest.raises(ValueError):
        nlp.solve_min_mass(MissionSpec(), 200.0, enforced=("not_a_row",))
    with pytest.raises(ValueError):
        nlp.solve_min_mass(MissionSpec(), 200.0, n_starts=0)

    def no_coverage(*args, **kwargs):
        raise ModelCoverageError("synthetic missing coolant model")

    monkeypatch.setattr(nlp, "require_htd_coverage", no_coverage)
    with pytest.raises(ModelCoverageError):
        nlp.solve_min_mass(MissionSpec(), 200.0)


# --------------------------------------------------------------------------- #
# R2: a frontier sweep isolates each floor                                    #
# --------------------------------------------------------------------------- #
def test_pareto_frontier_records_failed_floor_and_warm_starts_from_good_points(
    patch_engine, monkeypatch,
):
    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2))
    real = nlp.solve_min_mass
    seen_u0 = []

    def flaky(mission, isp_min, **kwargs):
        seen_u0.append(None if kwargs.get("u0") is None
                       else np.asarray(kwargs["u0"]).copy())
        if isp_min == 240.0:
            raise jax.errors.JaxRuntimeError("synthetic Lineax failure")
        return real(mission, isp_min, **kwargs)

    monkeypatch.setattr(nlp, "solve_min_mass", flaky)
    results = nlp.pareto_frontier(
        MissionSpec(), [230.0, 240.0, 250.0], fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
    )
    assert len(results) == 3
    assert results[0].success and results[2].success
    assert not results[1].success
    assert results[1].numerical_status == "fail"
    assert "synthetic Lineax failure" in results[1].message
    assert results[2].design["eps"] == pytest.approx(7.0, abs=1e-6)
    # The 250 s floor warm-started from the 230 s point (eps = 5), not from
    # the failed 240 s floor.
    np.testing.assert_allclose(seen_u0[2], [(5.0 - 4.0) / 3.0], atol=1e-6)


def test_pareto_frontier_survives_engine_failure_at_one_floor(patch_engine):
    # The engine itself raises around eps = 6, the 240 s optimum.
    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2,
                         fails=lambda eps: jnp.abs(eps - 6.0) < 0.2))
    results = nlp.pareto_frontier(
        MissionSpec(), [230.0, 240.0, 250.0], fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
    )
    assert len(results) == 3
    assert results[0].feasible
    assert results[0].design["eps"] == pytest.approx(5.0, abs=1e-6)
    assert results[2].feasible
    assert results[2].design["eps"] == pytest.approx(7.0, abs=1e-6)
    middle = results[1]
    assert middle.evaluation_failures >= 1
    # The 240 s optimum (eps = 6) lies inside the failing band, so no solved
    # optimum exists for this floor.  SLSQP may stall on the band edge, where
    # the Isp row is inactive and nothing balances the objective gradient;
    # the independent KKT check must refuse that point, so the floor is never
    # reported as a success, whatever path SLSQP took.
    assert not middle.success
    assert not middle.feasible
    assert middle.workflow_status == "fail"
    assert middle.kkt_status != "pass"


@jax.custom_vjp
def _nan_gradient_everywhere(eps):
    return eps


def _nan_fwd(eps):
    return eps, eps


def _nan_bwd(eps, cotangent):
    return (jnp.full_like(cotangent, jnp.nan),)


_nan_gradient_everywhere.defvjp(_nan_fwd, _nan_bwd)


def test_stall_at_failing_band_edge_is_not_reported_as_success(patch_engine):
    # The true optimum (eps = 5.5) is inside the failing band (5, 6); SLSQP
    # from eps = 7 can only stop on the band edge.  Before the KKT gate the
    # run reported a complete pass there.
    patch_engine(_engine(lambda eps: (eps - 5.5) ** 2,
                         fails=lambda eps: (eps > 5.0) & (eps < 6.0)))
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([1.0]),
    )
    assert result.evaluation_failures >= 1
    assert not result.success
    assert not result.feasible
    assert result.workflow_status == "fail"
    assert result.kkt_status == "fail"
    assert "not a confirmed KKT point" in result.message


def test_unrecoverable_derivative_ends_the_run_instead_of_zero_filling(
    patch_engine,
):
    # Exact gradients are NaN everywhere, the start sits on the lower face
    # (eps = 4) and the only finite-difference probe (eps slightly above 4)
    # is inside a failing band, so no derivative exists at the start.  A
    # zero-filled gradient made SLSQP declare convergence at eps = 4.
    def objective(eps):
        return (_nan_gradient_everywhere(eps) - 5.0) ** 2

    patch_engine(_engine(objective,
                         fails=lambda eps: (eps > 4.0) & (eps < 4.5)))
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([0.0]),
    )
    assert result.derivative_failures >= 1
    assert not result.success
    assert result.workflow_status == "fail"
    assert "unavailable" in result.message


def test_near_face_coordinate_counts_as_bound_active():
    # The Son-valve optimum leaves t_post at u = 1.5e-9; with the old 1e-9
    # tolerance its bound multiplier was dropped and stationarity failed.
    report = nlp.kkt_check(
        np.array([2.0]), np.zeros((0, 1)), np.zeros(0), np.array([1.5e-9]),
        constraint_names=(), variable_names=("t_post",),
    )
    assert report.status == "pass"
    assert report.active_constraints == ("lower:t_post",)
    assert report.multipliers["lower:t_post"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# R3: independent KKT check                                                   #
# --------------------------------------------------------------------------- #
def test_kkt_check_recovers_known_multipliers():
    # min (u0-1)^2 + (u1-1)^2  s.t.  g = 1 - u0 - u1 >= 0  (box inactive).
    # Optimum u = (0.5, 0.5), grad f = (-1, -1) = lambda * grad g with
    # grad g = (-1, -1)  ->  lambda = 1.
    u = np.array([0.5, 0.5])
    grad_f = 2.0 * (u - 1.0)
    jac_g = np.array([[-1.0, -1.0]])
    g = np.array([1.0 - u.sum()])
    report = nlp.kkt_check(grad_f, jac_g, g, u,
                           constraint_names=("sum",),
                           variable_names=("a", "b"))
    assert report.status == "pass"
    assert report.stationarity < 1e-12
    assert report.multipliers["sum"] == pytest.approx(1.0, abs=1e-12)
    assert report.active_constraints == ("sum",)
    assert report.dual_feasibility >= 0.0


def test_kkt_check_uses_box_bound_multipliers():
    # min u^2 + 1 on [0, 1] with no rows: optimum at the lower face, grad 0.
    # min -u on [0, 1]: optimum at the upper face with mu_hi = 1.
    report = nlp.kkt_check(np.array([-1.0]), np.zeros((0, 1)), np.zeros(0),
                           np.array([1.0]), constraint_names=(),
                           variable_names=("x",))
    assert report.status == "pass"
    assert report.multipliers == {"upper:x": pytest.approx(1.0)}
    assert report.active_constraints == ("upper:x",)


def test_kkt_check_fails_a_non_stationary_point():
    # Same problem, but at u = (0.2, 0.2): the row is inactive and the
    # gradient (-1.6, -1.6) has nothing to balance it.
    u = np.array([0.2, 0.2])
    report = nlp.kkt_check(2.0 * (u - 1.0), np.array([[-1.0, -1.0]]),
                           np.array([1.0 - u.sum()]), u,
                           constraint_names=("sum",),
                           variable_names=("a", "b"))
    assert report.status == "fail"
    assert report.stationarity == pytest.approx(1.6 / 1.6, rel=1e-12)
    assert report.multipliers["sum"] == 0.0


def test_kkt_check_rejects_wrong_sign_multiplier():
    # The row is active but the gradient points the other way: only a
    # negative multiplier would balance it, which NNLS refuses.
    u = np.array([0.5, 0.5])
    report = nlp.kkt_check(np.array([1.0, 1.0]), np.array([[-1.0, -1.0]]),
                           np.array([0.0]), u,
                           constraint_names=("sum",), variable_names=("a", "b"))
    assert report.status == "fail"
    assert report.multipliers["sum"] == 0.0
    assert report.stationarity == pytest.approx(1.0)


def test_kkt_check_is_unknown_without_finite_derivatives():
    report = nlp.kkt_check(np.array([np.nan]), np.zeros((0, 1)), np.zeros(0),
                           np.array([0.5]), constraint_names=(),
                           variable_names=("x",))
    assert report.status == "unknown"


def test_solve_reports_kkt_with_active_isp_row(patch_engine):
    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2))
    result = nlp.solve_min_mass(
        MissionSpec(), 240.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
    )
    assert result.design["eps"] == pytest.approx(6.0, abs=1e-7)
    assert result.kkt_status == "pass"
    assert result.kkt_stationarity <= 1e-4
    assert result.active_constraints == ("isp_epsilon",)
    # Unit box: eps = 4 + 3u.  d obj/du = 2 (eps-2) * 3 / 50 = 0.48 and
    # d g/du = 10 * 3 / 50 = 0.6 (scale 50 s), so lambda = 0.8.
    assert result.multipliers["isp_epsilon"] == pytest.approx(0.8, rel=1e-6)
    assert result.optimizer_status == "pass"


def test_all_fixed_evaluation_has_unknown_kkt(patch_engine):
    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2))
    result = nlp.solve_min_mass(
        MissionSpec(), 200.0, fixed_variables=_references(MissionSpec()),
    )
    assert result.kkt_status == "unknown"
    assert "no free design variable" in result.kkt_message


# --------------------------------------------------------------------------- #
# R4: multi-start                                                             #
# --------------------------------------------------------------------------- #
def _double_well(eps):
    # Two local minima, near eps = 4.5 (lower, by the tilt) and eps = 6.5.
    return (eps - 4.5) ** 2 * (eps - 6.5) ** 2 + 0.1 * eps


def test_single_start_from_upper_well_stays_local(patch_engine):
    patch_engine(_engine(_double_well))
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([0.95]),
    )
    assert result.design["eps"] > 6.0
    assert result.n_starts == 1
    assert result.n_basins == 1
    assert len(result.start_summaries) == 1


def test_multistart_finds_global_basin_and_counts_basins(patch_engine, monkeypatch):
    patch_engine(_engine(_double_well))
    compiled = []
    real_make = nlp._make_callables

    def counting(*args, **kwargs):
        compiled.append(1)
        return real_make(*args, **kwargs)

    monkeypatch.setattr(nlp, "_make_callables", counting)
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([0.95]), n_starts=8, start_seed=3,
    )
    assert len(compiled) == 1            # compiled once, reused for every start
    assert result.n_starts == 8
    assert len(result.start_summaries) == 8
    assert result.design["eps"] < 5.0    # the lower well wins
    assert result.n_basins == 2
    assert result.selected_start != 0
    summary = result.start_summaries[result.selected_start]
    assert summary["objective_mass"] == pytest.approx(result.objective_mass)
    assert summary["basin"] is not None
    starts = np.array([s["u_start"] for s in result.start_summaries])
    assert np.all((starts >= 0.0) & (starts <= 1.0))
    # Reproducible from the seed.
    again = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        u0=np.array([0.95]), n_starts=8, start_seed=3,
    )
    np.testing.assert_array_equal(
        starts, np.array([s["u_start"] for s in again.start_summaries])
    )


def test_multistart_with_no_feasible_start_returns_least_violating(patch_engine):
    # Isp floor above anything the box can reach: every start is infeasible.
    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2))
    result = nlp.solve_min_mass(
        MissionSpec(), 1000.0, fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        n_starts=4,
    )
    assert result.optimizer_status == "fail"
    assert result.n_basins == 0
    violations = [s["max_violation"] for s in result.start_summaries]
    assert result.max_violation == pytest.approx(min(violations))


def test_pareto_frontier_passes_multistart_through(patch_engine):
    patch_engine(_engine(_double_well))
    results = nlp.pareto_frontier(
        MissionSpec(), [0.0], fixed_variables=_eps_only(),
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
        n_starts=4, start_seed=1,
    )
    assert results[0].n_starts == 4


def test_cli_sweep_prints_failure_row_and_exports_only_good_floors(
    patch_engine, monkeypatch, capsys, tmp_path,
):
    """R2 at the CLI: a floor whose solve raises is a printed failure row; the
    sweep finishes, and --mdo-export re-evaluates only the good floors."""
    import raosim.mdo.postprocess as postprocess
    from raosim import run_nozzle

    patch_engine(_engine(lambda eps: (eps - 2.0) ** 2))
    real = nlp.solve_min_mass

    def flaky(mission, isp_min, **kwargs):
        if isp_min == 240.0:
            raise jax.errors.JaxRuntimeError("synthetic Lineax failure")
        return real(mission, isp_min, **kwargs)

    exported = []

    def reevaluate(design, mission, **kwargs):
        exported.append(kwargs["optimizer_metadata"]["isp_min_s"])
        assert kwargs["optimizer_metadata"]["kkt_status"] in {"pass", "fail", "unknown"}
        return SimpleNamespace(metadata={"authoritative_snapshot_report": "stub"})

    monkeypatch.setattr(nlp, "solve_min_mass", flaky)
    monkeypatch.setattr(postprocess, "reevaluate", reevaluate)
    monkeypatch.setattr(postprocess, "summarise", lambda rv: "    (stub summary)")
    code = run_nozzle.main([
        "--no-banner", "--engine-mdo-optimize", "--isp-sweep", "230,250,3",
        "--mdo-export", "--cad", "none", "--out", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert "Isp>= 240.0 | FAILED" in out
    assert "synthetic Lineax failure" in out
    assert out.count("checks: KKT=") == 2
    assert exported == [230.0, 250.0]
    assert code == 4   # one floor failed: the sweep completed, the run did not


def test_cli_rejects_zero_starts(capsys):
    from raosim import run_nozzle

    code = run_nozzle.main([
        "--no-banner", "--engine-mdo-optimize", "--isp-min", "190",
        "--mdo-multistart", "0",
    ])
    assert code == 2
    assert "--mdo-multistart must be >= 1" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [
    ["--engine-mdo", "--mdo-station-study"],
    ["--engine-mdo", "--mdo-multistart", "3"],
])
def test_cli_rejects_optimizer_only_flags_outside_the_optimizer(flags):
    from raosim import run_nozzle

    with pytest.raises(SystemExit) as exc:
        run_nozzle.main(["--no-banner", *flags])
    assert exc.value.code == 2
