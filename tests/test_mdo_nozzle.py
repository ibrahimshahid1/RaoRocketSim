"""Gates for the typed MDO contour-provider adapter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import raosim.jax  # noqa: F401 -- x64 before JAX arrays
import jax
import jax.numpy as jnp

from raosim.mdo.constraints import (
    CONSTRAINT_MANIFEST,
    ConstraintReasonCode,
)
from raosim.mdo.engine import solve_engine
from raosim.mdo.grid import GridTopology, build_station_grid
from raosim.mdo.job import JobValidationError, ModelSelection
from raosim.mdo.nlp import DEFAULT_ENFORCED
from raosim.mdo.nozzle import (
    ANALYTIC_TOP_MODEL_CODE,
    ANALYTIC_TOP_PROVIDER_ID,
    ContourOutcome,
    ContourOutcomeReason,
    evaluate_contour,
)
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.snapshot import snapshot_from_mdo
from raosim.mdo.state import (
    ENGINE_CONSTRAINT_NAMES,
    engine_state_from_result,
)


def _design() -> DesignVector:
    return DesignVector(
        Pc=jnp.asarray(3.0e6),
        eps=jnp.asarray(8.0),
        dp_f_frac=jnp.asarray(0.2),
        dp_o_frac=jnp.asarray(0.2),
        D_pintle=jnp.asarray(0.020),
        N_rpm=jnp.asarray(30000.0),
        channel_width=jnp.asarray(5.0e-4),
        channel_height=jnp.asarray(1.5e-3),
        film_frac=jnp.asarray(0.10),
        t_wall=jnp.asarray(8.0e-4),
    )


class _ForcedFailureProvider:
    """Provider test double with a distinguishable, finite shape carrier."""

    provider_id = "forced_failure_for_test"
    model_code = 91

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, *, Rt, eps, mission, topo, gamma):
        self.calls += 1
        original = build_station_grid(Rt, eps, mission, topo, gamma=gamma)
        marker = jnp.asarray([0.123, 0.124, 0.125, 0.126])
        carrier = replace(original, chart_domain_violation=marker)
        return ContourOutcome(
            grid=carrier,
            available=jnp.asarray(False),
            infeasible=jnp.asarray(True),
            reason_code=jnp.asarray(
                int(ContourOutcomeReason.PROVIDER_REPORTED_INFEASIBLE),
                dtype=jnp.int32,
            ),
            provider_model_code=jnp.asarray(self.model_code, dtype=jnp.int32),
        )


def test_default_adapter_is_bit_identical_to_existing_analytic_grid():
    mission = MissionSpec()
    topo = GridTopology()
    Rt = jnp.asarray(0.0314)
    eps = jnp.asarray(8.0)
    gamma = jnp.asarray(mission.gamma)

    prior = build_station_grid(Rt, eps, mission, topo, gamma=gamma)
    outcome = evaluate_contour(
        Rt, eps, mission, topo, gamma=gamma,
        provider=ANALYTIC_TOP_PROVIDER_ID,
    )

    for field in (
        "x", "r", "area_ratio", "mach", "dseg",
        "chart_domain_violation", "wall_monotonic_margin",
    ):
        assert np.array_equal(
            np.asarray(getattr(outcome.grid, field)),
            np.asarray(getattr(prior, field)),
        ), field
    assert outcome.grid.throat_index == prior.throat_index
    assert bool(outcome.available)
    assert not bool(outcome.infeasible)
    assert int(outcome.reason_code) == int(ContourOutcomeReason.AVAILABLE)
    assert int(outcome.provider_model_code) == ANALYTIC_TOP_MODEL_CODE

    jitted = jax.jit(
        lambda rt, expansion: evaluate_contour(
            rt, expansion, mission, topo, gamma=gamma
        )
    )(Rt, eps)
    np.testing.assert_allclose(
        np.asarray(jitted.grid.x), np.asarray(prior.x), rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert bool(jitted.available)


def test_forced_provider_failure_reaches_state_and_snapshot_without_fallback():
    mission = MissionSpec()
    design = _design()
    provider = _ForcedFailureProvider()

    result = solve_engine(design, mission, contour_provider=provider)
    assert provider.calls == 1
    assert not bool(result.contour_outcome.available)
    assert bool(result.contour_outcome.infeasible)
    assert np.isfinite(float(result.constraints["contour_provider_margin"]))
    assert float(result.constraints["contour_provider_margin"]) == -1.0
    # The marker proves the adapter retained the failed provider's carrier and
    # did not rebuild/substitute the registered analytic TOP contour.
    assert np.array_equal(
        np.asarray(result.contour_outcome.grid.chart_domain_violation),
        np.asarray([0.123, 0.124, 0.125, 0.126]),
    )

    state = engine_state_from_result(result, design, mission)
    index = ENGINE_CONSTRAINT_NAMES.index("contour_provider_margin")
    assert not np.any(np.asarray(state.geometry.station_valid))
    assert int(state.geometry.divergent_model) == provider.model_code
    assert float(state.constraints.values[index]) == -1.0
    assert bool(state.constraints.available[index])
    assert bool(state.constraints.required[index])
    assert int(state.constraints.reason_codes[index]) == int(
        ConstraintReasonCode.CONTOUR_PROVIDER_FAILED
    )
    assert not bool(state.constraints.all_inequalities_nonnegative)

    snapshot = snapshot_from_mdo(state, mission=mission)
    assert snapshot.constraints_gates["physics_feasible"].value is False
    row = snapshot.constraints_gates["constraint_margins"].value[
        "contour_provider_margin"
    ]
    assert row["value"] == -1.0
    assert "shape grid is only a numerical carrier" in row["reason"]

    spec = next(
        row for row in CONSTRAINT_MANIFEST if row.name == "contour_provider"
    )
    assert spec.kind == "availability_gate"
    assert spec.optimizer_role == "final_gate"
    assert spec.name not in DEFAULT_ENFORCED


def test_forced_provider_failure_marks_nlp_candidate_physics_failed(
    monkeypatch,
):
    import raosim.mdo.nlp as nlp_module

    # Bypass only the external optimizer iterations.  solve_min_mass still
    # performs its real post-solve engine evaluation and verdict reduction at
    # the reference design, which is the NLP result contract under test.
    def accept_initial_point(_fun, u0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray(u0), success=True, nit=0,
            message="forced host optimizer completion",
        )

    monkeypatch.setattr(nlp_module, "minimize", accept_initial_point)
    provider = _ForcedFailureProvider()
    result = nlp_module.solve_min_mass(
        MissionSpec(),
        0.0,
        contour_provider=provider,
    )

    # The un-jitted post-solve evaluation, plus the traces of the compiled
    # objective/constraint programs the independent KKT check evaluates at the
    # returned point (raosim.mdo.nlp.kkt_check); the optimizer itself made no
    # calls.
    assert provider.calls >= 1
    assert result.n_iter == 0
    assert result.constraints["contour_provider"] == -1.0
    assert result.physics_status == "fail"
    assert not result.feasible


def test_unknown_provider_identity_is_rejected_instead_of_falling_back():
    with pytest.raises(JobValidationError, match="unknown MDO contour provider"):
        ModelSelection(contour_provider="variational_rao_not_registered")
