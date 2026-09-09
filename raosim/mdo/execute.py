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

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping

from raosim.input_validation import WorkflowExitCode
from raosim.mdo.job import (
    EngineJobSpec,
    JobValidationError,
    ResolvedEngineJob,
    TargetRelation,
    VariableMode,
    VariableRequest,
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
_REQUIRED_STATUS_KEYS = frozenset({
    "workflow", "optimizer", "numerical", "physics", "requirements",
    "workflow_readiness", "input_integrity", "requirement_coverage", "model_consistency",
})
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
    requirement_coverage: tuple[Mapping[str, Any], ...] = ()
    model_consistency: Mapping[str, Any] = field(default_factory=dict)
    output_directory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "job_digest": self.job.job_digest,
            "resolved_job": self.job.to_dict(),
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
            "requirement_coverage": [dict(row) for row in self.requirement_coverage],
            "model_consistency": dict(self.model_consistency),
            "comparison": (
                self.reevaluation.comparison.to_dict()
                if self.reevaluation is not None
                and self.reevaluation.comparison is not None else None
            ),
            "output_directory": self.output_directory,
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
        if isinstance(value, bool):
            raise ValueError("boolean is not a measured scalar")
        scalar = float(value)
        if not math.isfinite(scalar):
            return None, f"{metric_id} is nonfinite"
        return scalar, None
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

    if any(statuses.get(name) == "fail" for name in
           ("workflow", "numerical", "input_integrity", "report_publication")):
        return "SOLVER_FAILED", int(WorkflowExitCode.SOLVER_FAILED)
    demonstrated_failure = (
        any(value == "fail" for value in statuses.values())
        or any(t.status == "fail" for t in targets)
    )
    if demonstrated_failure:
        return "CANDIDATE_VIOLATES", int(WorkflowExitCode.CANDIDATE_VIOLATES)
    if (
        not _REQUIRED_STATUS_KEYS.issubset(statuses)
        or any(v not in {"pass", "not_applicable"} for v in statuses.values())
        or any(t.status == "unknown" for t in targets)
    ):
        return "INDETERMINATE", int(WorkflowExitCode.INDETERMINATE)
    return "MET", int(WorkflowExitCode.MET)


def _isp_floor(spec: EngineJobSpec) -> float:
    """The Isp requirement the optimizer enforces, if one was stated."""

    return max((float(target.value) for target in spec.targets
                if target.metric == "performance.isp_delivered"
                and target.relation in {TargetRelation.AT_LEAST,
                                        TargetRelation.RANGE,
                                        TargetRelation.EQUAL_TO}), default=0.0)


def _requirement_coverage(job: ResolvedEngineJob,
                          targets: tuple[TargetOutcome, ...]) -> tuple[Mapping[str, Any], ...]:
    if not targets and job.effective_targets:
        targets = _evaluate_targets(replace(job.spec, targets=job.effective_targets), None)
    rows = [dict(row) for row in job.requirement_coverage]
    for row in rows:
        row["status"] = "pass" if row["status"] == "supported" else "unknown"
    for index, target in enumerate(targets):
        rows.append({
            "input": f"effective_targets.{index}", "metric": target.metric,
            "consumer": "authoritative snapshot target evaluation",
            "enforcement": (
                "optimizer lower bound and postsolve check"
                if job.spec.workflow is Workflow.OPTIMIZE
                and target.metric == "performance.isp_delivered"
                and target.relation in {"at_least", "range", "equal_to"}
                else "postsolve check"
            ),
            **target.to_dict(),
        })
    return tuple(rows)


def _model_consistency(comparison: Any) -> dict[str, Any]:
    """Expose diagnostic parity without inventing physical acceptance limits."""
    if comparison is None:
        return {"status": "unknown", "reason": "model comparison unavailable"}
    rows = {**comparison.scalars,
            **{f"profiles.{key}": value for key, value in comparison.profiles.items()}}
    different = sorted(key for key, row in rows.items() if row.status == "different")
    unavailable = sorted(key for key, row in rows.items() if row.status == "not_comparable")
    invalid = sorted(key for key, row in rows.items()
                     if row.status not in {"different", "not_comparable", "within_tolerance"})
    return {
        "status": "unknown" if different or unavailable or invalid or not rows else "pass",
        "different_fields": different,
        "not_comparable_fields": unavailable,
        "unrecognized_status_fields": invalid,
        "reason": (
            "Snapshot tolerances are numerical diagnostics, not validated "
            "per-quantity physical acceptance limits. Differences and missing "
            "comparisons require adjudication; agreement is not test evidence."
        ),
        "acceptance_policy": "diagnostic_only",
        "abs_tol": comparison.abs_tol, "rel_tol": comparison.rel_tol,
    }


def execute_engine_job(
    spec: EngineJobSpec,
    *,
    size_pumps: bool = False,
) -> EngineJobResult:
    """Resolve, solve, parity-check and publish one honest run record."""
    job = resolve_engine_job(spec)
    if job.enumerations:
        raise JobValidationError(
            "enumerated variables require execute_engine_job_set; "
            "a single execution cannot omit unselected alternatives"
        )
    if spec.workflow is Workflow.PARETO:
        raise JobValidationError("workflow 'pareto' is not wired through this service")
    from raosim.mdo.run_record import RunRecord

    record = RunRecord.start(job)
    progress: dict[str, Any] = {}
    try:
        result = _execute_resolved_engine_job(job, size_pumps=size_pumps, progress=progress)
    except JobValidationError:
        if record is not None:
            record.mark_failed("job validation failed during execution")
        raise
    except Exception as exc:
        result = EngineJobResult(
            job=progress.get("job", job), verdict="SOLVER_FAILED",
            exit_code=int(WorkflowExitCode.SOLVER_FAILED), design=progress.get("design", {}),
            inputs=progress.get("inputs"), state=progress.get("state"),
            statuses={"workflow": "fail", "requirement_coverage": "unknown"},
            requirement_coverage=_requirement_coverage(job, ()),
            messages=(f"{type(exc).__name__}: {exc}",),
        )
    if record is not None:
        result = replace(result, output_directory=str(record.directory))
        try:
            record.finish(result)
        except Exception as exc:
            message = f"report publication failed: {type(exc).__name__}: {exc}"
            result = replace(
                result, verdict="SOLVER_FAILED",
                exit_code=int(WorkflowExitCode.SOLVER_FAILED),
                statuses={**result.statuses, "report_publication": "fail"},
                messages=(*result.messages, message),
            )
            record.mark_failed(message, result=result)
    return result


def _execute_resolved_engine_job(
    job: ResolvedEngineJob, *, size_pumps: bool = False,
    progress: dict[str, Any] | None = None,
) -> EngineJobResult:
    spec = job.spec
    if progress is None:
        progress = {}

    import jax.numpy as jnp

    from raosim.mdo.engine import chamber_surfaces_for
    from raosim.mdo.nozzle import resolve_contour_provider
    from raosim.mdo.postprocess import reevaluate
    from raosim.mdo.resolved_inputs import resolve_engine_inputs
    from raosim.mdo.schema import DesignVector
    from raosim.mdo.state import solve_engine_state

    target_spec = replace(spec, targets=job.effective_targets)
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
            _isp_floor(target_spec),
            objective=objective,
            allow_incomplete_physics=spec.models.allow_incomplete_physics,
            contour_provider=contour_provider,
            fixed_variables=job.fixed_variables,
            bounds=job.bounds,
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
                requirement_coverage=_requirement_coverage(job, ()),
            )
        layout = mission.design_layout()
        design_vector = DesignVector.from_active_array(
            jnp.asarray([float(nlp.design[n]) for n in layout.active_names]),
            layout,
            fixed_of=None if layout.of_is_variable else float(mission.OF),
        )

    progress["design"] = {k: float(v) for k, v in design_vector.as_dict().items()}
    state = solve_engine_state(
        design_vector,
        mission,
        surfaces=surfaces,
        contour_provider=contour_provider,
    )
    design = {k: float(v) for k, v in design_vector.as_dict().items()}
    progress["state"] = state

    inputs = resolve_engine_inputs(
        design_vector.as_contract_dict(effective_of=state.performance.OF),
        mission,
        effective_of=float(state.performance.OF),
        of_source=(
            "optimized" if mission.design_layout().of_is_variable
            else "pinned" if spec.models.mixture_ratio_mode == "pinned"
            else "mission_nominal"
        ),
        total_mass_flow=float(state.performance.mdot_total),
        eta_cstar_effective=float(state.performance.eta_cstar),
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
    progress.update(inputs=inputs, job=job)

    reev = reevaluate(
        design, mission, mdo_result=state, mdo_surfaces=surfaces,
        resolved_input_contract=inputs,
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
        observed = []
        for source in (reev.mdo_snapshot, snapshot):
            if source is None:
                continue
            entry = source.constraints_gates.get(key)
            value = getattr(entry, "value", None) if entry is not None else None
            if isinstance(value, str) and value in {"pass", "fail", "unknown", "not_applicable"}:
                observed.append(value)
            elif isinstance(value, bool):
                observed.append("pass" if value else "fail")
            elif value is not None:
                observed.append("unknown")
        if "fail" in observed:
            return "fail"
        if not observed or "unknown" in observed:
            return "unknown"
        return "pass" if "pass" in observed else "not_applicable"

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
    targets = _evaluate_targets(target_spec, snapshot)
    coverage = _requirement_coverage(job, targets)
    statuses["requirement_coverage"] = (
        "unknown" if any(row["status"] == "unknown" for row in coverage) else "pass"
    )
    integrity = reev.metadata.get("input_integrity", {})
    statuses["input_integrity"] = integrity.get("status", "unknown")
    if statuses["input_integrity"] != "pass":
        messages.append(f"resolved input integrity: {integrity or 'crosscheck unavailable'}")
    consistency = _model_consistency(reev.comparison)
    statuses["model_consistency"] = consistency["status"]
    if consistency["status"] != "pass":
        messages.append(consistency["reason"])
    if spec.objective is not None:
        objective_value, reason = _snapshot_value(snapshot, spec.objective.metric)
        statuses["objective"] = "pass" if objective_value is not None else "unknown"
        if reason is not None:
            messages.append(f"objective unavailable: {reason}")
    verdict, exit_code = _verdict(statuses, targets)
    messages.extend(str(w) for w in reev.warnings)

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
        requirement_coverage=coverage,
        model_consistency=consistency,
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

    @property
    def verdict(self) -> str:
        verdicts = {result.verdict for result in self.results}
        if "MET" in verdicts:
            return "MET"
        if "SOLVER_FAILED" in verdicts:
            return "SOLVER_FAILED"
        if verdicts == {"CANDIDATE_VIOLATES"}:
            return "CANDIDATE_VIOLATES"
        return "INDETERMINATE"

    @property
    def exit_code(self) -> int:
        return int(WorkflowExitCode[self.verdict])

    def to_dict(self) -> dict[str, Any]:
        def label(result: EngineJobResult) -> dict[str, Any]:
            architecture = result.job.spec.architecture
            return {
                "job_digest": result.job.job_digest,
                "fixed_variables": dict(result.job.fixed_variables),
                "propellant": architecture.propellant,
                "liner_material": architecture.liner_material,
                "closeout_material": architecture.closeout_material,
                "verdict": result.verdict,
                "exit_code": result.exit_code,
                "violated": [n for n, _ in result.violated_constraints],
            }

        return {
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "results": [result.to_dict() for result in self.results],
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
    from raosim.mdo.job import VARIABLE_DIMENSIONS
    from raosim.units import canonical_unit

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
    enumerated = [variable for variable in spec.variables
                  if variable.mode is VariableMode.ENUMERATE]
    expanded = []
    for choice in combinations:
        for values in itertools.product(*(variable.values for variable in enumerated)):
            selected = dict(zip((variable.name for variable in enumerated), values))
            variables = tuple(
                VariableRequest(variable.name, "fixed",
                                f"{selected[variable.name]:.17g} "
                                f"{canonical_unit(VARIABLE_DIMENSIONS[variable.name])}")
                if variable.name in selected else variable
                for variable in spec.variables
            )
            expanded.append(_replace(
                spec, architecture=_replace(architecture, **choice), variables=variables,
            ))
    return tuple(expanded)


def execute_engine_job_set(
    spec: EngineJobSpec,
    *,
    size_pumps: bool = False,
) -> EngineJobSetResult:
    """Solve every permitted discrete architecture and rank the feasible ones."""

    objective_metric = spec.objective.metric if spec.objective else None
    results: list[EngineJobResult] = []
    candidates = expand_architectures(spec)
    for candidate in candidates:
        resolve_engine_job(candidate)
    for candidate in candidates:
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
