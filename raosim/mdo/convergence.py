"""
raosim.mdo.convergence — station-refinement (grid-convergence) evidence.

The coupled solve runs on a fixed-topology station grid
(:class:`raosim.mdo.grid.GridTopology`, default 4 + 6 + 13 + 1 = 24 stations).
The plan's numerical definition of done (``docs/DIFFERENTIABLE_ENGINE_MDO_PLAN
.md`` §11) asks that "doubling cooling/nozzle stations changes objective and
active margins by < 0.5 %"; the review (§15.5) found no test of it.  This
module performs that study at a given design point:

1. solve the engine at the base topology and at the topology with every
   segment count multiplied by ``factor`` (default 2: 8 + 12 + 26 + 1 = 47);
2. report the objective, Isp and every hard-row margin at both;
3. normalize the changes — relative for the objective and Isp, and for the
   margins the absolute change divided by the manifest row ``scale`` (the
   same conditioning reference the optimizer uses), because an active margin
   sits at zero and has no meaningful relative change;
4. state whether the < 0.5 % criterion holds for the objective and for every
   active row (scaled base margin ≤ 1e-5, the optimizer feasibility
   tolerance), with both evaluations numerically valid.

:func:`station_refinement_study` holds the design point fixed.  That
measures the discretization error of one design, but the plan's criterion
concerns the solved problem, so :func:`reoptimized_refinement_study`
re-solves the whole NLP on each grid and compares the optima: their
objectives, Isp and active sets.  The two readings differ (at the 13 kN
optimum the fixed design's coking margin moves ~5 % of scale, while the
re-optimized objective moves ~0.26 %), so both are reported.  Nothing is
tuned or clipped; the studies are evidence, not gates.  The < 0.5 %
threshold and the doubling procedure are the plan's repository definition of
done; no held source prescribes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.constraints import ENGINE_CONSTRAINT_SPECS, OPTIMIZER_CONSTRAINT_SPECS
from raosim.mdo.engine import chamber_surfaces_for, solve_engine
from raosim.mdo.grid import GridTopology
from raosim.mdo.nozzle import ContourProviderSelection
from raosim.mdo.objectives import (
    DEFAULT_MASS_OBJECTIVE,
    MassObjective,
    coerce_mass_objective,
    mass_objective_value,
)
from raosim.mdo.schema import DesignVector, MissionSpec

#: Plan §11: "< 0.5 %".
PLAN_REFINEMENT_CRITERION = 5.0e-3
#: Same scaled tolerance as the optimizer feasibility / KKT active set.
ACTIVE_ROW_TOLERANCE = 1.0e-5
#: The manifest's numerical final gates (finite, solver status, both inner
#: residuals): an evaluation failing any of them is not grid evidence.
_NUMERICAL_GATES = tuple(
    spec.engine_key for spec in ENGINE_CONSTRAINT_SPECS
    if spec.category == "numerical" and spec.optimizer_role == "final_gate"
)


@dataclass(frozen=True)
class StationRefinementStudy:
    """Objective, Isp and hard-row margins at two station resolutions."""

    factor: int
    base_topology: tuple[int, int, int]       # (n_chamber, n_conv, n_div)
    refined_topology: tuple[int, int, int]
    base_stations: int
    refined_stations: int
    objective_name: str
    objective_base: float
    objective_refined: float
    objective_relative_change: float
    isp_base: float
    isp_refined: float
    isp_relative_change: float
    margins_base: dict[str, float]
    margins_refined: dict[str, float]
    #: |margin_refined − margin_base| / manifest row scale.
    margin_normalized_changes: dict[str, float]
    active_rows: tuple[str, ...]
    numerically_valid: bool
    criterion: float
    criterion_met: bool
    #: Largest normalized change among the objective and the active rows.
    worst_quantity: str
    worst_change: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        verdict = "MET" if self.criterion_met else "NOT MET"
        lines = [
            f"station refinement x{self.factor}: {self.base_stations} -> "
            f"{self.refined_stations} stations; plan §11 < "
            f"{100 * self.criterion:.1f}% criterion {verdict}"
            + ("" if self.numerically_valid else " (an evaluation is not "
               "numerically valid)"),
            f"  {self.objective_name}: {self.objective_base:.6g} -> "
            f"{self.objective_refined:.6g} kg "
            f"({100 * self.objective_relative_change:+.4f}%)",
            f"  Isp: {self.isp_base:.6g} -> {self.isp_refined:.6g} s "
            f"({100 * self.isp_relative_change:+.4f}%)",
        ]
        for name in self.active_rows:
            lines.append(
                f"  active {name}: {self.margins_base[name]:+.4g} -> "
                f"{self.margins_refined[name]:+.4g} "
                f"(|change|/scale = {100 * self.margin_normalized_changes[name]:.3f}%)"
            )
        return lines


def _design_vector(design: DesignVector | Mapping[str, float],
                   mission: MissionSpec) -> DesignVector:
    if isinstance(design, DesignVector):
        return design
    layout = mission.design_layout()
    return DesignVector.from_active_array(
        jnp.asarray([float(design[name]) for name in layout.active_names]),
        layout,
        fixed_of=None if layout.of_is_variable else float(mission.OF),
    )


def _hard_row_margins(result: Any, isp_min: float | None) -> dict[str, float]:
    margins: dict[str, float] = {}
    for spec in OPTIMIZER_CONSTRAINT_SPECS:
        if spec.engine_key is None:
            if isp_min is not None:
                margins[spec.name] = float(result.Isp) - float(isp_min)
            continue
        margins[spec.name] = float(result.constraints[spec.engine_key])
    return margins


def _numerically_valid(result: Any) -> bool:
    return all(
        np.isfinite(float(result.constraints[key]))
        and float(result.constraints[key]) >= 0.0
        for key in _NUMERICAL_GATES
    )


def refined_topology(base: GridTopology, factor: int) -> GridTopology:
    """Multiply every segment's station count by ``factor``."""

    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError("factor must be an integer")
    if int(factor) < 2:
        raise ValueError("factor must be at least 2")
    return GridTopology(
        n_chamber=base.n_chamber * int(factor),
        n_conv=base.n_conv * int(factor),
        n_div=base.n_div * int(factor),
    )


def station_refinement_study(
    design: DesignVector | Mapping[str, float],
    mission: MissionSpec,
    *,
    factor: int = 2,
    base_topology: GridTopology = GridTopology(),
    objective: MassObjective | str = DEFAULT_MASS_OBJECTIVE,
    isp_min: float | None = None,
    active_rows: Iterable[str] | None = None,
    couple_eta_cstar: bool = False,
    surfaces: Any = None,
    contour_provider: ContourProviderSelection = None,
    criterion: float = PLAN_REFINEMENT_CRITERION,
) -> StationRefinementStudy:
    """Re-solve one design at refined stations and grade the change.

    ``active_rows`` overrides the active set (for example an
    ``NLPResult.active_constraints`` list; bound entries are ignored).  By
    default a hard row is active when its scaled base margin is at most
    :data:`ACTIVE_ROW_TOLERANCE`.  ``isp_min`` adds the ``isp_epsilon`` row.
    """

    selected = coerce_mass_objective(objective)
    refined = refined_topology(base_topology, factor)
    surfaces = surfaces if surfaces is not None else chamber_surfaces_for(mission)
    vector = _design_vector(design, mission)
    results = [
        solve_engine(
            vector,
            mission,
            couple_eta_cstar=couple_eta_cstar,
            surfaces=surfaces,
            topo=topo,
            contour_provider=contour_provider,
        )
        for topo in (base_topology, refined)
    ]
    base, fine = results
    margins_base = _hard_row_margins(base, isp_min)
    margins_fine = _hard_row_margins(fine, isp_min)
    scales = {spec.name: float(spec.scale) for spec in OPTIMIZER_CONSTRAINT_SPECS}
    changes = {
        name: abs(margins_fine[name] - margins_base[name]) / scales[name]
        for name in margins_base
    }
    if active_rows is None:
        active = tuple(
            name for name, value in margins_base.items()
            if value / scales[name] <= ACTIVE_ROW_TOLERANCE
        )
    else:
        active = tuple(name for name in active_rows if name in margins_base)

    objective_base = float(mass_objective_value(base, selected))
    objective_fine = float(mass_objective_value(fine, selected))
    isp_base, isp_fine = float(base.Isp), float(fine.Isp)
    objective_change = abs(objective_fine - objective_base) / max(
        abs(objective_base), 1e-300
    )
    isp_change = (isp_fine - isp_base) / max(abs(isp_base), 1e-300)
    valid = _numerically_valid(base) and _numerically_valid(fine)

    graded = {"objective": objective_change}
    graded.update({f"margin:{name}": changes[name] for name in active})
    worst_quantity, worst_change = max(graded.items(), key=lambda kv: kv[1])
    met = (
        valid
        and all(np.isfinite(v) for v in graded.values())
        and all(v < criterion for v in graded.values())
    )
    return StationRefinementStudy(
        factor=int(factor),
        base_topology=(base_topology.n_chamber, base_topology.n_conv,
                       base_topology.n_div),
        refined_topology=(refined.n_chamber, refined.n_conv, refined.n_div),
        base_stations=base_topology.n,
        refined_stations=refined.n,
        objective_name=selected.value,
        objective_base=objective_base,
        objective_refined=objective_fine,
        objective_relative_change=(objective_fine - objective_base) / max(
            abs(objective_base), 1e-300
        ),
        isp_base=isp_base,
        isp_refined=isp_fine,
        isp_relative_change=isp_change,
        margins_base=margins_base,
        margins_refined=margins_fine,
        margin_normalized_changes=changes,
        active_rows=active,
        numerically_valid=valid,
        criterion=float(criterion),
        criterion_met=bool(met),
        worst_quantity=worst_quantity,
        worst_change=float(worst_change),
    )


@dataclass(frozen=True)
class ReoptimizedRefinementStudy:
    """Optima of the same NLP solved on a base and a refined station grid."""

    factor: int
    base_stations: int
    refined_stations: int
    isp_min: float
    objective_name: str
    base_success: bool
    refined_success: bool
    base_kkt_status: str
    refined_kkt_status: str
    objective_base: float
    objective_refined: float
    objective_relative_change: float
    isp_base: float
    isp_refined: float
    isp_relative_change: float
    design_base: dict[str, float]
    design_refined: dict[str, float]
    #: Active entries (rows and bound faces) of each optimum's KKT check.
    active_base: tuple[str, ...]
    active_refined: tuple[str, ...]
    active_set_unchanged: bool
    criterion: float
    #: Both solves succeed, the objective moves < criterion and the active set
    #: is unchanged (so every active margin is still zero on the finer grid).
    criterion_met: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        verdict = "MET" if self.criterion_met else "NOT MET"
        lines = [
            f"re-optimized refinement x{self.factor}: {self.base_stations} -> "
            f"{self.refined_stations} stations; plan §11 < "
            f"{100 * self.criterion:.1f}% criterion {verdict}",
            f"  {self.objective_name}: {self.objective_base:.6g} -> "
            f"{self.objective_refined:.6g} kg "
            f"({100 * self.objective_relative_change:+.4f}%)",
            f"  Isp: {self.isp_base:.6g} -> {self.isp_refined:.6g} s "
            f"({100 * self.isp_relative_change:+.4f}%)",
            f"  KKT: {self.base_kkt_status} -> {self.refined_kkt_status}; "
            "active set "
            + ("unchanged" if self.active_set_unchanged else
               f"changed {sorted(self.active_base)} -> "
               f"{sorted(self.active_refined)}"),
        ]
        return lines


def reoptimized_refinement_study(
    mission: MissionSpec,
    isp_min: float,
    *,
    factor: int = 2,
    base_topology: GridTopology = GridTopology(),
    criterion: float = PLAN_REFINEMENT_CRITERION,
    **solve_kwargs: Any,
) -> ReoptimizedRefinementStudy:
    """Solve the ε-constraint NLP on both grids and compare the optima.

    ``solve_kwargs`` are forwarded to :func:`raosim.mdo.nlp.solve_min_mass`
    for both solves (objective, enforced rows, variable policies...).  The
    refined solve starts from the base optimum, so it measures the drift of
    the optimum rather than a different basin.
    """

    from raosim.mdo.nlp import _OptimizationSpace, solve_min_mass
    from raosim.mdo.scaling import ScaledSpace

    refined = refined_topology(base_topology, factor)
    base = solve_min_mass(mission, isp_min, topology=base_topology,
                          **solve_kwargs)
    warm = dict(solve_kwargs)
    if base.success and "u0" not in warm:
        space = _OptimizationSpace.for_mission(
            mission, warm.get("fixed_variables"), warm.get("bounds")
        )
        warm["u0"] = np.asarray(ScaledSpace.from_specs(
            space.free_specs
        ).to_unit(space.select_free(base.x)))
    fine = solve_min_mass(mission, isp_min, topology=refined, **warm)

    objective_change = (fine.objective_mass - base.objective_mass) / max(
        abs(base.objective_mass), 1e-300
    )
    isp_change = (fine.Isp - base.Isp) / max(abs(base.Isp), 1e-300)
    unchanged = set(base.active_constraints) == set(fine.active_constraints)
    met = bool(
        base.success and fine.success
        and np.isfinite(objective_change)
        and abs(objective_change) < criterion
        and unchanged
    )
    return ReoptimizedRefinementStudy(
        factor=int(factor),
        base_stations=base_topology.n,
        refined_stations=refined.n,
        isp_min=float(isp_min),
        objective_name=base.objective_name,
        base_success=bool(base.success),
        refined_success=bool(fine.success),
        base_kkt_status=base.kkt_status,
        refined_kkt_status=fine.kkt_status,
        objective_base=float(base.objective_mass),
        objective_refined=float(fine.objective_mass),
        objective_relative_change=float(objective_change),
        isp_base=float(base.Isp),
        isp_refined=float(fine.Isp),
        isp_relative_change=float(isp_change),
        design_base=dict(base.design),
        design_refined=dict(fine.design),
        active_base=tuple(base.active_constraints),
        active_refined=tuple(fine.active_constraints),
        active_set_unchanged=bool(unchanged),
        criterion=float(criterion),
        criterion_met=met,
    )


__all__ = [
    "ACTIVE_ROW_TOLERANCE",
    "PLAN_REFINEMENT_CRITERION",
    "ReoptimizedRefinementStudy",
    "StationRefinementStudy",
    "refined_topology",
    "reoptimized_refinement_study",
    "station_refinement_study",
]
