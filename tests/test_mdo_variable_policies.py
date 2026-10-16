"""Pins and requested bounds must govern the evaluated optimization problem."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import raosim.jax  # noqa: F401
import jax
import jax.numpy as jnp

from raosim.mdo import nlp
from raosim.mdo.schema import DesignVector, MissionSpec, VariableSpec, default_design_space
from raosim.mdo.scaling import ScaledSpace


def _references(mission):
    return {spec.name: float(spec.ref()) for spec in default_design_space(mission)}


@pytest.fixture
def analytic_engine(monkeypatch):
    """Known constrained optimum: min (eps-2)^2 with Isp = 10 eps + 180.

    This isolates the NLP contract from model availability and root convergence.
    The real-engine test below independently exercises the assembled physics.
    """
    def solve(design, mission, **kwargs):
        objective = (design.eps - 2.0) ** 2 + design.Pc / 1.0e6
        constraints = {
            spec.engine_key: jnp.asarray(spec.scale)
            for spec in nlp.ENGINE_CONSTRAINT_SPECS
        }
        return SimpleNamespace(
            Isp=10.0 * design.eps + 180.0,
            OF=design.layout.resolve_of(design.OF, mission.OF),
            electric_package_objective_mass=objective,
            dry_mass_partial_objective_mass=objective + 1.0,
            electric_package_exact_mass=objective,
            dry_mass_partial_exact_mass=objective + 1.0,
            chamber_mass=SimpleNamespace(total=jnp.asarray(1.0)),
            constraints=constraints,
        )
    monkeypatch.setattr(nlp, "solve_engine", solve)
    monkeypatch.setattr(nlp, "chamber_surfaces_for", lambda mission: object())
    return solve


@pytest.mark.parametrize("lower,expected", [(4.0, 6.0), (6.5, 6.5)])
def test_slsqp_respects_pins_user_bounds_and_hard_constraints(analytic_engine, lower, expected):
    mission = MissionSpec()
    fixed = _references(mission)
    fixed.pop("eps")
    fixed.update(Pc=2.7e6, N_rpm=33700.0, film_frac=0.07)
    result = nlp.solve_min_mass(
        mission, 240.0, fixed_variables=fixed, bounds={"eps": (lower, 7.0)},
        enforced=("isp_epsilon",),
    )
    assert result.success
    assert result.feasible
    assert result.design["eps"] == pytest.approx(expected, abs=1e-7)
    assert result.Isp >= 240.0 - 1e-7
    assert result.x.shape == (12,)
    assert tuple(result.design) == DesignVector.names()
    for name, value in fixed.items():
        assert result.design[name] == value


@pytest.mark.parametrize("isp_floor,optimizer_status", [(200.0, "pass"), (1000.0, "fail")])
def test_all_fixed_evaluates_once_and_reports_infeasibility(
    analytic_engine, monkeypatch, isp_floor, optimizer_status,
):
    mission = MissionSpec()
    fixed = _references(mission)
    calls = []

    def tracked(design, mission, **kwargs):
        calls.append(design.as_dict())
        return analytic_engine(design, mission, **kwargs)

    monkeypatch.setattr(nlp, "solve_engine", tracked)
    monkeypatch.setattr(
        nlp, "minimize", lambda *args, **kwargs: pytest.fail("all-fixed NLP invoked SciPy")
    )
    result = nlp.solve_min_mass(mission, isp_floor, fixed_variables=fixed, bounds={})
    assert len(calls) == 1
    assert result.n_iter == 0
    assert result.workflow_status == "pass"
    assert result.optimizer_status == optimizer_status
    assert result.feasible is (optimizer_status == "pass")
    for name, value in fixed.items():
        assert result.design[name] == value


def test_empty_enforced_set_remains_defined(analytic_engine):
    result = nlp.solve_min_mass(
        MissionSpec(), 0.0, fixed_variables={"Pc": 3.0e6}, enforced=(),
    )
    assert result.success
    assert result.max_violation == 0.0
    assert result.optimizer_status == "pass"


def test_nonfinite_hard_margin_cannot_be_optimizer_pass(analytic_engine, monkeypatch):
    def invalid(design, mission, **kwargs):
        result = analytic_engine(design, mission, **kwargs)
        result.Isp = jnp.asarray(np.nan)
        return result

    monkeypatch.setattr(nlp, "solve_engine", invalid)
    result = nlp.solve_min_mass(
        MissionSpec(), 200.0, fixed_variables=_references(MissionSpec()),
        enforced=("isp_epsilon",),
    )
    assert result.optimizer_status == "fail"
    assert result.requirements_status == "unknown"
    assert np.isinf(result.max_violation)
    assert not result.feasible


def test_reduced_jacobian_has_no_columns_for_fixed_coordinates():
    mission = MissionSpec()
    fixed = _references(mission)
    fixed.pop("eps")
    fixed.pop("N_rpm")
    policy = nlp._OptimizationSpace.for_mission(
        mission, fixed, {"eps": (5.0, 9.0), "N_rpm": (30000.0, 40000.0)},
    )
    scaling = ScaledSpace.from_specs(policy.free_specs)
    reconstruct = jax.jit(lambda u: policy.reconstruct(scaling.to_physical(u)))
    jacobian = np.asarray(jax.jacrev(reconstruct)(jnp.asarray([0.5, 0.5])))
    expected = np.zeros((12, 2))
    expected[1, 0] = 4.0
    expected[5, 1] = 10000.0
    np.testing.assert_array_equal(jacobian, expected)


def test_fixed_of_in_active_model_survives_final_reconstruction(analytic_engine, monkeypatch):
    mission = replace(MissionSpec(), cea_table_path="test-of-dependent-surface")
    specs = default_design_space(MissionSpec()) + (VariableSpec("OF", 1.0, 4.0, 2.27),)
    monkeypatch.setattr(nlp, "default_design_space", lambda mission: specs)
    fixed = {spec.name: spec.ref() for spec in specs if spec.name != "eps"}
    fixed["OF"] = 2.6
    result = nlp.solve_min_mass(
        mission, 240.0, fixed_variables=fixed, bounds={"eps": (4.0, 7.0)},
        enforced=("isp_epsilon",),
    )
    assert result.success
    assert result.x.shape == (13,)
    assert result.x[-1] == 2.6
    assert result.design["OF"] == 2.6


def test_pareto_warm_start_projects_full_result_back_to_free_coordinates(analytic_engine):
    fixed = _references(MissionSpec())
    fixed.pop("eps")
    results = nlp.pareto_frontier(
        MissionSpec(), [230.0, 240.0], fixed_variables=fixed,
        bounds={"eps": (4.0, 7.0)}, enforced=("isp_epsilon",),
    )
    assert [result.design["eps"] for result in results] == pytest.approx([5.0, 6.0])
    assert all(result.success for result in results)


@pytest.mark.parametrize(
    "fixed,bounds,match",
    [
        ({"unknown": 1.0}, {}, "unknown"),
        ({"Pc": np.nan}, {}, "finite"),
        ({"Pc": 0.0}, {}, "physical range"),
        ({"film_frac": 1.0}, {}, "physical range"),
        ({"Pc": 3e6}, {"Pc": (2e6, 4e6)}, "both fixed and bounded"),
        ({}, {"eps": (6.0, 6.0)}, "strictly increasing"),
        ({}, {"eps": (np.nan, 9.0)}, "finite"),
        ({}, {"eps": (1e4, 2e4)}, "do not intersect"),
        ({}, {"eps": (1.0, 2.0, 3.0)}, "pair"),
        ({"OF": 9.0}, {}, "agree"),
        ({}, {"OF": (1.0, 4.0)}, "active O/F"),
    ],
)
def test_invalid_policy_fails_before_physics(fixed, bounds, match, monkeypatch):
    monkeypatch.setattr(
        nlp, "chamber_surfaces_for", lambda *args: pytest.fail("invalid policy reached physics")
    )
    with pytest.raises(ValueError, match=match):
        nlp.solve_min_mass(MissionSpec(), 200.0, fixed_variables=fixed, bounds=bounds)


def test_explicit_matching_inactive_of_pin_is_accepted():
    mission = MissionSpec(OF=2.6)
    policy = nlp._OptimizationSpace.for_mission(mission, {"OF": 2.6}, {})
    # Twelve hardware variables, less D_cg pinned for the slotted pintle.
    assert len(policy.free_specs) == 11


def test_reduced_start_rejects_full_vector_or_nonfinite_values(analytic_engine):
    for start in [np.zeros(10), np.asarray([np.nan])]:
        fixed = _references(MissionSpec())
        fixed.pop("eps")
        with pytest.raises(ValueError, match="1 finite free-variable"):
            nlp.solve_min_mass(MissionSpec(), 200.0, fixed_variables=fixed, u0=start)


def test_real_engine_solve_preserves_all_pins_and_bounded_free_speed():
    mission = MissionSpec()
    fixed = _references(mission)
    fixed.pop("N_rpm")
    result = nlp.solve_min_mass(
        mission, 0.0, fixed_variables=fixed,
        bounds={"N_rpm": (29000.0, 31000.0)},
        enforced=("isp_epsilon",), maxiter=20,
    )
    assert result.success
    assert result.numerical_status == "pass"
    assert 29000.0 <= result.design["N_rpm"] <= 31000.0
    assert np.isfinite(result.objective_mass)
    assert np.isfinite(result.Isp)
    for name, value in fixed.items():
        assert result.design[name] == value
    # Only selected rows were optimized; the complete physical verdict still
    # comes from all final gates and can honestly be a failure at these pins.
    assert tuple(result.design) == DesignVector.names()
