"""Station-refinement study (plan §11 grid criterion, review §15.5).

The study re-solves one fixed design at doubled station counts; it must run on
every block with a non-default topology and return finite, self-consistent
numbers.  Whether the < 0.5 % criterion holds is a measured result reported by
the study, not something this test asserts.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401
import jax.numpy as jnp

from raosim.mdo.constraints import OPTIMIZER_CONSTRAINT_SPECS
from raosim.mdo.convergence import (
    PLAN_REFINEMENT_CRITERION,
    refined_topology,
    station_refinement_study,
)
from raosim.mdo.engine import solve_engine
from raosim.mdo.grid import GridTopology
from raosim.mdo.schema import DesignVector, MissionSpec, default_design_space
from raosim.mdo.state import solve_engine_state


def _reference_design(mission):
    return {spec.name: float(spec.ref()) for spec in default_design_space(mission)}


def test_refined_topology_multiplies_every_segment():
    fine = refined_topology(GridTopology(), 2)
    assert (fine.n_chamber, fine.n_conv, fine.n_div) == (8, 12, 26)
    assert fine.n == 47
    with pytest.raises(ValueError):
        refined_topology(GridTopology(), 1)
    with pytest.raises(ValueError):
        refined_topology(GridTopology(), True)


def test_every_block_accepts_a_non_default_topology():
    mission = MissionSpec()
    layout = mission.design_layout()
    reference = _reference_design(mission)
    vector = DesignVector.from_active_array(
        jnp.asarray([reference[name] for name in layout.active_names]),
        layout, fixed_of=mission.OF,
    )
    topo = GridTopology(n_chamber=5, n_conv=7, n_div=17)
    result = solve_engine(vector, mission, topo=topo)
    assert result.contour_outcome.grid.x.shape == (topo.n,)
    assert result.T_wg.shape == (topo.n,)
    assert float(result.constraints["finite_margin"]) > 0.0
    state = solve_engine_state(vector, mission, topo=topo)
    assert state.geometry.x.shape == (topo.n,)


def test_station_refinement_study_runs_and_is_finite():
    mission = MissionSpec()
    study = station_refinement_study(
        _reference_design(mission), mission, isp_min=190.0,
    )
    assert study.base_stations == 24 and study.refined_stations == 47
    assert study.numerically_valid
    assert np.isfinite(study.objective_base) and np.isfinite(study.objective_refined)
    assert np.isfinite(study.isp_base) and np.isfinite(study.isp_refined)
    hard_rows = {spec.name for spec in OPTIMIZER_CONSTRAINT_SPECS}
    assert set(study.margins_base) == hard_rows
    assert all(np.isfinite(v) for v in study.margin_normalized_changes.values())
    assert study.criterion == PLAN_REFINEMENT_CRITERION
    # The verdict is consistent with the numbers it reports.
    graded = [abs(study.objective_relative_change)] + [
        study.margin_normalized_changes[name] for name in study.active_rows
    ]
    assert study.criterion_met == all(v < study.criterion for v in graded)
    assert study.worst_change == pytest.approx(max(graded))
    assert study.summary_lines()[0].startswith("station refinement x2")


def test_reoptimized_refinement_solves_both_grids(monkeypatch):
    """Plumbing: the refined NLP really runs on the refined topology."""
    from types import SimpleNamespace

    from raosim.mdo import nlp
    from raosim.mdo.convergence import reoptimized_refinement_study

    seen = []

    def solve(design, mission, **kwargs):
        seen.append(kwargs.get("topo"))
        value = (design.eps - 5.0) ** 2 + design.Pc / 1.0e6
        return SimpleNamespace(
            Isp=10.0 * design.eps + 180.0,
            OF=design.layout.resolve_of(design.OF, mission.OF),
            electric_package_objective_mass=value,
            dry_mass_partial_objective_mass=value + 1.0,
            electric_package_exact_mass=value,
            dry_mass_partial_exact_mass=value + 1.0,
            chamber_mass=SimpleNamespace(total=jnp.asarray(1.0)),
            constraints={
                spec.engine_key: jnp.asarray(spec.scale)
                for spec in nlp.ENGINE_CONSTRAINT_SPECS
            },
        )

    monkeypatch.setattr(nlp, "solve_engine", solve)
    monkeypatch.setattr(nlp, "chamber_surfaces_for", lambda mission: object())
    mission = MissionSpec()
    fixed = {s.name: float(s.ref()) for s in default_design_space(mission)}
    fixed.pop("eps")
    study = reoptimized_refinement_study(
        mission, 0.0, fixed_variables=fixed, bounds={"eps": (4.0, 7.0)},
        enforced=("isp_epsilon",),
    )
    stations = {topo.n for topo in seen if topo is not None}
    assert stations == {GridTopology().n, study.refined_stations}
    assert study.base_success and study.refined_success
    assert study.objective_relative_change == pytest.approx(0.0, abs=1e-12)
    assert study.active_set_unchanged
    assert study.criterion_met
