"""One entry point that runs an engine job end to end.

The point of this module is that a user should not have to know whether an
answer came from the differentiable core or the traditional pipeline.  A job
goes in; a verdict, a snapshot, and the resolved inputs that produced them come
out.  Internally the flow is always the same:

    resolve job -> solve (evaluate or optimize) -> EngineState v2
                -> resolved inputs -> authoritative traditional parity pass
                -> snapshot -> targets -> verdict

A verdict is deliberately four-valued.  "The optimizer did not converge" is not
evidence that a requirement is impossible, and "a governing model is
unavailable" is not evidence that a design is sound; collapsing either into a
boolean is how an unknown becomes a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from raosim.input_validation import WorkflowExitCode
from raosim.mdo.job import (
    EngineJobSpec,
    JobValidationError,
    ResolvedEngineJob,
    TargetRelation,
    Workflow,
    resolve_engine_job,
)
from raosim.mdo.metrics import METRIC_REGISTRY

__all__ = [
    "EngineJobResult",
    "EngineJobSetResult",
    "TargetOutcome",
    "execute_engine_job",
    "execute_engine_job_set",
    "expand_architectures",
]


# Public metric identities and optimizer implementations are deliberately
# separate contracts.  Keep the bridge explicit, and refuse to import if a
# future registry edit advertises an objective that this execution path cannot
# actually run (or hides one that it can).
_OBJECTIVE_IMPLEMENTATIONS: Mapping[str, str] = {
    "masses.dry_mass_partial": "min_dry_mass_partial",
    "masses.electric_package": "min_electric_package_mass",
}
_ADVERTISED_OBJECTIVES = frozenset(
    metric.id
    for metric in METRIC_REGISTRY.values()
    if metric.available and metric.objective_capable
)
if frozenset(_OBJECTIVE_IMPLEMENTATIONS) != _ADVERTISED_OBJECTIVES:
    raise RuntimeError(
        "objective registry/execution mismatch: advertised="
        f"{sorted(_ADVERTISED_OBJECTIVES)!r}, executable="
        f"{sorted(_OBJECTIVE_IMPLEMENTATIONS)!r}"
    )


@dataclass(frozen=True)
class TargetOutcome:
    """Whether one requested output condition was met."""

    metric: str
    relation: str
    requested: float
    requested_upper: float | None
    achieved: float | None
    status: str                       # "pass" | "fail" | "unknown"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "relation": self.relation,
            "requested": self.requested,
            "requested_upper": self.requested_upper,
            "achieved": self.achieved,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EngineJobResult:
    """Everything one job produced, plus a verdict that can say 'unknown'."""

    job: ResolvedEngineJob
    verdict: str
    exit_code: int
    design: Mapping[str, float]
    state: Any = None
    inputs: Any = None
    snapshot: Any = None
    reevaluation: Any = None
    contour_validation: Any = None
    targets: tuple[TargetOutcome, ...] = ()
    statuses: Mapping[str, str] = field(default_factory=dict)
    #: Constraint rows with a demonstrated negative margin, worst first.  A
    #: verdict that only says "violates" makes the user go hunting; naming the
    #: rows and their margins is the difference between a report and a result.
    violated_constraints: tuple[tuple[str, float], ...] = ()
    messages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "job_digest": self.job.job_digest,
            "resolved_inputs_digest": (
                self.inputs.digest() if self.inputs is not None else None
            ),
            "design": dict(self.design),
            "statuses": dict(self.statuses),
            "violated_constraints": [
                {"name": name, "margin": margin}
                for name, margin in self.violated_constraints
            ],
            "targets": [t.to_dict() for t in self.targets],
            "contour_validation": (
                self.contour_validation.to_dict()
                if self.contour_validation is not None
                else None
            ),
            "notes": list(self.job.notes),
            "messages": list(self.messages),
        }


def _snapshot_value(snapshot: Any, metric_id: str) -> tuple[float | None, str | None]:
    """Read one registered metric out of a snapshot, honouring availability."""

    metric = METRIC_REGISTRY[metric_id]
    if metric.snapshot_section is None or metric.snapshot_field is None:
        return None, f"{metric_id} has no snapshot location"
    section = getattr(snapshot, metric.snapshot_section, None)
    if section is None:
        return None, f"snapshot has no section {metric.snapshot_section!r}"
    entry = section.get(metric.snapshot_field)
    if entry is None:
        return None, f"{metric_id} is absent from this snapshot"
    value = getattr(entry, "value", entry)
    if value is None:
        reason = getattr(entry, "availability_reason", None)
        return None, reason or f"{metric_id} is unavailable in this analysis"
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, f"{metric_id} is not a scalar"


def _violated_constraints(snapshot: Any) -> tuple[tuple[str, float], ...]:
    """Name every row with a demonstrated negative margin, worst first.

    Only finite negative margins count. A nonfinite margin is a numerical
    problem owned by the numerical gate, and an unavailable row is an unknown
    -- neither is a demonstrated violation.
    """

    import math

    if snapshot is None:
        return ()
    entry = snapshot.constraints_gates.get("constraint_margins")
    margins = getattr(entry, "value", None) if entry is not None else None
    if not isinstance(margins, Mapping):
        return ()
    found: list[tuple[str, float]] = []
    for key, row in margins.items():
        raw = row.get("value") if isinstance(row, Mapping) else row
        name = row.get("name", key) if isinstance(row, Mapping) else key
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value < 0.0:
            found.append((str(name), value))
    return tuple(sorted(found, key=lambda item: item[1]))


def _evaluate_targets(spec: EngineJobSpec, snapshot: Any) -> tuple[TargetOutcome, ...]:
    outcomes: list[TargetOutcome] = []
    for target in spec.targets:
        achieved, reason = _snapshot_value(snapshot, target.metric)
        if achieved is None:
            outcomes.append(TargetOutcome(
                target.metric, target.relation.value, float(target.value),
                target.upper_value, None, "unknown", reason,
            ))
            continue
        relation = target.relation
        want = float(target.value)
        if relation is TargetRelation.AT_LEAST:
            ok = achieved >= want
        elif relation is TargetRelation.AT_MOST:
            ok = achieved <= want
        elif relation is TargetRelation.EQUAL_TO:
            ok = abs(achieved - want) <= 1.0e-9 * max(1.0, abs(want))
        else:
            ok = want <= achieved <= float(target.upper_value)
        outcomes.append(TargetOutcome(
            target.metric, relation.value, want, target.upper_value,
            achieved, "pass" if ok else "fail",
        ))
    return tuple(outcomes)


def _verdict(statuses: Mapping[str, str],
             targets: tuple[TargetOutcome, ...]) -> tuple[str, int]:
    """Reduce statuses to one four-valued verdict.

    Order matters.  A solver failure is reported as a solver failure, not as a
    violated requirement, because a failed nonconvex solve is not proof that a
    requirement is impossible.
    """

    if statuses.get("workflow") == "fail" or statuses.get("numerical") == "fail":
        return "SOLVER_FAILED", int(WorkflowExitCode.SOLVER_FAILED)
    demonstrated_failure = (
        statuses.get("optimizer") == "fail"
        or statuses.get("physics") == "fail"
        or statuses.get("requirements") == "fail"
        or statuses.get("contour_validation") == "fail"
        or any(t.status == "fail" for t in targets)
    )
    if demonstrated_failure:
        return "CANDIDATE_VIOLATES", int(WorkflowExitCode.CANDIDATE_VIOLATES)
    if (
        any(v == "unknown" for v in statuses.values())
        or any(t.status == "unknown" for t in targets)
    ):
        return "INDETERMINATE", int(WorkflowExitCode.INDETERMINATE)
    return "MET", int(WorkflowExitCode.MET)


def _isp_floor(spec: EngineJobSpec) -> float:
    """The Isp requirement the optimizer enforces, if one was stated."""

    for target in spec.targets:
        if (
            target.metric == "performance.isp_delivered"
            and target.relation is TargetRelation.AT_LEAST
        ):
            return float(target.value)
    return 0.0


def execute_engine_job(
    spec: EngineJobSpec,
    *,
    size_pumps: bool = False,
) -> EngineJobResult:
    """Resolve, solve, parity-check and judge one engine job."""

    import jax.numpy as jnp

    from raosim.mdo.engine import chamber_surfaces_for
    from raosim.mdo.nozzle import resolve_contour_provider
    from raosim.mdo.postprocess import reevaluate
    from raosim.mdo.resolved_inputs import resolve_engine_inputs
    from raosim.mdo.schema import DesignVector
    from raosim.mdo.state import solve_engine_state

    job = resolve_engine_job(spec)
    messages: list[str] = []
    statuses: dict[str, str] = {}

    if spec.workflow is Workflow.PARETO:
        raise JobValidationError(
            "workflow 'pareto' is not wired through execute_engine_job yet; "
            "run a sequence of epsilon-constrained 'optimize' jobs instead of "
            "receiving a single point labelled as a frontier"
        )

    mission = job.mission
    surfaces = chamber_surfaces_for(mission)
    contour_provider = resolve_contour_provider(spec.models.contour_provider)
    variational_validation_config = None
    variational_domain = None
    contour_comparison_policy = None
    if spec.models.postsolve_contour_validator is not None:
        from raosim.mdo.contour_validation import (
            ContourDiscrepancyTolerances,
            VariationalPostSolveConfig,
        )
        from raosim.rao_existence_scan import RaoConvergenceDomain

        variational_domain = RaoConvergenceDomain.read_json(
            str(spec.models.rao_convergence_domain)
        )
        if (
            job.rao_convergence_domain_sha256
            != variational_domain.artifact_sha256
        ):
            raise JobValidationError(
                "post-solve Rao convergence-domain artifact changed after "
                "job resolution; refusing a time-of-check/time-of-use identity "
                "mismatch"
            )
        tolerances = ContourDiscrepancyTolerances(
            max_radius_error_over_rt=float(
                spec.models.contour_max_radius_error_over_rt
            ),
            rms_radius_error_over_rt=float(
                spec.models.contour_rms_radius_error_over_rt
            ),
            relative_divergent_length_error=float(
                spec.models.contour_relative_length_error
            ),
            relative_thrust_coefficient_error=float(
                spec.models.contour_relative_cf_error
            ),
        )
        variational_validation_config = VariationalPostSolveConfig(
            tolerances=tolerances,
            convergence_domain=variational_domain,
        )
        contour_comparison_policy = (
            variational_validation_config.comparison_policy_dict()
        )

    if spec.workflow is Workflow.ANALYZE:
        design_vector = job.design_vector()
        statuses["workflow"] = "pass"
    else:
        from raosim.mdo.nlp import solve_min_mass

        if spec.objective is not None and spec.objective.direction != "min":
            raise JobValidationError(
                f"objective direction {spec.objective.direction!r} is not "
                "implemented; available objectives are minimization-only"
            )
        objective = _OBJECTIVE_IMPLEMENTATIONS.get(
            spec.objective.metric if spec.objective else ""
        )
        if objective is None:
            raise JobValidationError(
                f"objective {spec.objective.metric!r} has no differentiable "
                "optimizer implementation yet; the registry lists which "
                "objectives are available"
            )
        nlp = solve_min_mass(
            mission,
            _isp_floor(spec),
            objective=objective,
            allow_incomplete_physics=spec.models.allow_incomplete_physics,
            contour_provider=contour_provider,
        )
        statuses["workflow"] = "pass" if nlp.success else "fail"
        statuses["optimizer"] = nlp.optimizer_status
        if not nlp.success:
            messages.append(f"optimizer did not converge: {nlp.message}")
            return EngineJobResult(
                job=job,
                verdict="SOLVER_FAILED",
                exit_code=int(WorkflowExitCode.SOLVER_FAILED),
                design=dict(nlp.design),
                statuses=statuses,
                messages=tuple(messages),
            )
        layout = mission.design_layout()
        design_vector = DesignVector.from_active_array(
            jnp.asarray([float(nlp.design[n]) for n in layout.active_names]),
            layout,
            fixed_of=None if layout.of_is_variable else float(mission.OF),
        )

    state = solve_engine_state(
        design_vector,
        mission,
        surfaces=surfaces,
        contour_provider=contour_provider,
    )
    design = {k: float(v) for k, v in design_vector.as_dict().items()}

    inputs = resolve_engine_inputs(
        design_vector.as_contract_dict(effective_of=state.performance.OF),
        mission,
        effective_of=float(state.performance.OF),
        of_source=(
            "optimized" if mission.design_layout().of_is_variable
            else "mission_nominal"
        ),
        total_mass_flow=float(state.performance.mdot_total),
        surfaces=surfaces,
        contour_provider_id=contour_provider.provider_id,
        postsolve_contour_validator_id=(
            spec.models.postsolve_contour_validator
        ),
        rao_convergence_domain_sha256=(
            variational_domain.artifact_sha256
            if variational_domain is not None
            else None
        ),
        contour_comparison_policy=contour_comparison_policy,
    )
    job = job.with_inputs(inputs)

    reev = reevaluate(
        design, mission, mdo_result=state, mdo_surfaces=surfaces,
        mdo_summary={"Isp": float(state.performance.Isp_delivered)},
        size_pumps=size_pumps,
        variational_validation_config=variational_validation_config,
        contour_provider_id=contour_provider.provider_id,
        postsolve_contour_validator_id=(
            spec.models.postsolve_contour_validator
        ),
        rao_convergence_domain_sha256=(
            variational_domain.artifact_sha256
            if variational_domain is not None
            else None
        ),
        contour_comparison_policy=contour_comparison_policy,
    )
    snapshot = reev.authoritative_snapshot
    contour_validation = reev.variational_contour_validation

    # The two snapshots answer different questions and neither alone is
    # complete: the constraint manifest, physics coverage and inner-root
    # validity are evaluated by the MDO, while artifact/workflow readiness
    # belongs to the authoritative traditional pass.  Reading a gate from the
    # pipeline that never evaluated it yields ``None``, which would collapse
    # into a blanket "unknown" and hide a real violation.
    def _gate(key: str) -> str:
        for source in (reev.mdo_snapshot, snapshot):
            if source is None:
                continue
            entry = source.constraints_gates.get(key)
            value = getattr(entry, "value", None) if entry is not None else None
            if isinstance(value, str):
                return value
            if value is not None:
                return "pass" if value else "fail"
        return "unknown"

    for key, name in (
        ("optimizer_constraints_feasible", "optimizer"),
        ("numerical_validity", "numerical"),
        ("physics_feasible", "physics"),
        ("requirements_feasible", "requirements"),
        ("workflow_readiness_feasible", "workflow_readiness"),
    ):
        statuses[name] = _gate(key)

    if contour_validation is not None:
        statuses["contour_validation"] = contour_validation.status.value
        messages.append(
            "post-solve contour validation "
            f"{contour_validation.status.value}: {contour_validation.detail}"
        )

    violated = _violated_constraints(reev.mdo_snapshot)
    targets = _evaluate_targets(spec, snapshot)
    verdict, exit_code = _verdict(statuses, targets)
    messages.extend(str(w) for w in snapshot.warnings)

    return EngineJobResult(
        job=job,
        verdict=verdict,
        exit_code=exit_code,
        design=design,
        state=state,
        inputs=inputs,
        snapshot=snapshot,
        reevaluation=reev,
        contour_validation=contour_validation,
        targets=targets,
        statuses=statuses,
        violated_constraints=violated,
        messages=tuple(messages),
    )


# --------------------------------------------------------------------------- #
# Discrete architecture enumeration                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineJobSetResult:
    """Ranked outcomes for one enumerated set of discrete architectures.

    Ranking is only ever over candidates that actually passed.  A candidate
    whose verdict is ``INDETERMINATE`` is not "slightly worse" than one that
    met its requirements -- its standing is unknown, and ordering it against
    known-good designs by objective value would present unknown coverage as a
    near miss.
    """

    results: tuple[EngineJobResult, ...]
    ranked: tuple[EngineJobResult, ...]
    unranked: tuple[EngineJobResult, ...]
    objective_metric: str | None

    @property
    def best(self) -> EngineJobResult | None:
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        def label(result: EngineJobResult) -> dict[str, Any]:
            architecture = result.job.spec.architecture
            return {
                "propellant": architecture.propellant,
                "liner_material": architecture.liner_material,
                "closeout_material": architecture.closeout_material,
                "verdict": result.verdict,
                "exit_code": result.exit_code,
                "violated": [n for n, _ in result.violated_constraints],
            }

        return {
            "objective_metric": self.objective_metric,
            "ranked": [label(r) for r in self.ranked],
            "unranked": [label(r) for r in self.unranked],
            "best": label(self.best) if self.best is not None else None,
        }


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return (None,)
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def expand_architectures(spec: EngineJobSpec) -> tuple[EngineJobSpec, ...]:
    """Expand list-valued discrete selections into one job per combination.

    Discrete choices stay outside the gradient vector by construction: each
    combination becomes its own continuous problem rather than a relaxed
    variable.
    """

    import itertools
    from dataclasses import replace as _replace

    architecture = spec.architecture
    axes = {
        "propellant": _as_tuple(architecture.propellant),
        "liner_material": _as_tuple(architecture.liner_material),
        "closeout_material": _as_tuple(architecture.closeout_material),
    }
    combinations = [
        dict(zip(axes, values))
        for values in itertools.product(*axes.values())
    ]
    return tuple(
        _replace(spec, architecture=_replace(architecture, **choice))
        for choice in combinations
    )


def execute_engine_job_set(
    spec: EngineJobSpec,
    *,
    size_pumps: bool = False,
) -> EngineJobSetResult:
    """Solve every permitted discrete architecture and rank the feasible ones."""

    objective_metric = spec.objective.metric if spec.objective else None
    results: list[EngineJobResult] = []
    for candidate in expand_architectures(spec):
        try:
            results.append(execute_engine_job(candidate, size_pumps=size_pumps))
        except JobValidationError:
            raise
        except Exception as exc:
            job = resolve_engine_job(candidate)
            results.append(EngineJobResult(
                job=job,
                verdict="SOLVER_FAILED",
                exit_code=int(WorkflowExitCode.SOLVER_FAILED),
                design={},
                statuses={"workflow": "fail"},
                messages=(f"{type(exc).__name__}: {exc}",),
            ))

    passing = [r for r in results if r.verdict == "MET"]
    others = [r for r in results if r.verdict != "MET"]

    def objective_value(result: EngineJobResult) -> float:
        if objective_metric is None or result.snapshot is None:
            return float("inf")
        value, _ = _snapshot_value(result.snapshot, objective_metric)
        return float("inf") if value is None else value

    ranked = tuple(sorted(passing, key=objective_value))
    return EngineJobSetResult(
        results=tuple(results),
        ranked=ranked,
        unranked=tuple(others),
        objective_metric=objective_metric,
    )
