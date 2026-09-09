"""Job-oriented verbs for the LREKit CLI.

These make the unified workflow reachable outside the Python API.  That
matters on its own terms: O/F optimization was implemented long before it was
reachable from a command line, which meant in practice it was not usable.

Verbs:

    lrekit guided                interview, preview coverage, save and/or run
    lrekit run JOB.yaml          resolve, solve, judge; exit code is the verdict
    lrekit validate JOB.yaml     resolve and report without solving
    lrekit catalog WHAT          list variables, targets, objectives,
                                 constraints, propellants or materials

Anything not recognized here falls through to the existing CLI unchanged, so
this is additive rather than a replacement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["JOB_VERBS", "main"]

#: Verbs this module owns.  Everything else is the legacy CLI's.
JOB_VERBS: frozenset[str] = frozenset({"guided", "run", "validate", "catalog"})

_CATALOG_TOPICS = (
    "variables", "targets", "objectives", "constraints", "propellants",
    "materials", "unavailable", "capabilities",
)


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate job key {key!r}; inputs cannot overwrite one another")
        result[key] = value
    return result


def _load_job_document(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "reading a YAML job needs PyYAML; install it or supply the "
                "job as JSON"
            ) from exc
        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node):
            loader.flatten_mapping(node)
            return _unique_mapping((loader.construct_object(key), loader.construct_object(value))
                                   for key, value in node.value)

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            loaded = yaml.load(text, Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML: {exc}") from exc
    else:
        loaded = json.loads(text, object_pairs_hook=_unique_mapping)
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"{path} must contain a mapping at the top level, got "
            f"{type(loaded).__name__}"
        )
    return loaded


def _build_spec(document: Mapping[str, Any]):
    """Compile a job document into an :class:`EngineJobSpec`."""

    from raosim.mdo.job import (
        ArchitectureSelection,
        EngineJobSpec,
        MissionRequest,
        ModelSelection,
        ObjectiveRequest,
        OperatingPointSpec,
        OutputPolicy,
        TargetRequest,
        VariableRequest,
    )

    unknown = set(document) - {
        "mission", "workflow", "architecture", "models", "variables",
        "targets", "objective", "operating_points", "output_policy",
        "schema_version", "custom_propellants",
    }
    if unknown:
        raise ValueError(
            "unknown job keys: " + ", ".join(sorted(unknown))
            + ". A misspelled key must not be silently ignored."
        )

    if "mission" not in document:
        raise ValueError("a job document requires a 'mission' section")

    from raosim.mdo.capabilities import require_capability
    from raosim.mdo.job import ENGINE_JOB_SCHEMA_VERSION
    require_capability("schema_version", document.get("schema_version", ENGINE_JOB_SCHEMA_VERSION))
    for key in ("mission", "architecture", "models", "variables", "objective", "output_policy"):
        value = document.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a mapping")
    for key in ("targets", "operating_points", "custom_propellants"):
        value = document.get(key)
        if value is not None and not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a sequence")

    variables = tuple(
        VariableRequest(name=name, **spec)
        for name, spec in (document.get("variables") or {}).items()
    )
    targets = tuple(
        TargetRequest(**entry) for entry in (document.get("targets") or ())
    )
    objective = document.get("objective")
    points = tuple(
        OperatingPointSpec(**entry)
        for entry in (document.get("operating_points") or ())
    )
    return EngineJobSpec(
        mission=MissionRequest(**document["mission"]),
        workflow=document.get("workflow", "analyze"),
        architecture=ArchitectureSelection(**(document.get("architecture") or {})),
        models=ModelSelection(**(document.get("models") or {})),
        variables=variables,
        targets=targets,
        objective=ObjectiveRequest(**objective) if objective else None,
        operating_points=points,
        output_policy=OutputPolicy(**(document.get("output_policy") or {})),
        schema_version=document.get("schema_version", ENGINE_JOB_SCHEMA_VERSION),
        custom_propellants=tuple(document.get("custom_propellants") or ()),
    )


def _path_arguments(argv: Sequence[str], flag: str) -> tuple[Path, Path | None]:
    if len(argv) == 1 and not argv[0].startswith("--"):
        return Path(argv[0]).expanduser(), None
    if (len(argv) == 3 and not argv[0].startswith("--")
            and argv[1] == flag and not argv[2].startswith("--")):
        return Path(argv[0]).expanduser(), Path(argv[2]).expanduser()
    raise ValueError(f"expected JOB.yaml [{flag} FILE]; unknown or incomplete arguments")


def _print_resolved(job: Any) -> None:
    print(f"  job digest        {job.job_digest}")
    print(f"  workflow          {job.spec.workflow.value}")
    print(f"  propellant        {job.mission.propellant_name}")
    print(f"  liner / closeout  {job.mission.liner_material_name} / "
          f"{job.mission.closeout_material_name}")
    print(f"  thrust            {job.mission.thrust:g} N")
    print(f"  constraint policy {job.constraint_policy.value}")
    if job.fixed_variables:
        print("  fixed variables")
        for name in sorted(job.fixed_variables):
            print(f"      {name:16s} {job.fixed_variables[name]:g}")
    if job.active_variables:
        print("  active variables")
        for name in sorted(job.active_variables):
            lo, hi = job.bounds[name]
            print(f"      {name:16s} [{lo:g}, {hi:g}]")
    if job.enumerations:
        print("  enumerations")
        for name, values in job.enumerations.items():
            print(f"      {name:16s} {list(values)}")
    for note in job.notes:
        print(f"  note: {note}")


def _cmd_validate(argv: Sequence[str]) -> int:
    from raosim.input_validation import WorkflowExitCode
    from raosim.mdo.job import resolve_engine_job

    if not argv:
        print("usage: lrekit validate JOB.yaml [--resolved-out FILE]")
        return int(WorkflowExitCode.INVALID_INPUT)
    path, out = _path_arguments(argv, "--resolved-out")

    try:
        spec = _build_spec(_load_job_document(path))
        from raosim.mdo.execute import expand_architectures

        jobs = tuple(resolve_engine_job(candidate) for candidate in expand_architectures(spec))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"job rejected: {exc}")
        return int(WorkflowExitCode.INVALID_INPUT)

    print(f"job {path} is valid")
    for job in jobs:
        _print_resolved(job)
    if out is not None:
        payload = jobs[0].to_dict() if len(jobs) == 1 else {"candidates": [job.to_dict() for job in jobs]}
        out.write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(f"  resolved job written to {out}")
    return int(WorkflowExitCode.MET)


def _cmd_run(argv: Sequence[str]) -> int:
    from raosim.input_validation import WorkflowExitCode
    from raosim.mdo.execute import execute_engine_job, execute_engine_job_set, expand_architectures

    if not argv:
        print("usage: lrekit run JOB.yaml [--out FILE]")
        return int(WorkflowExitCode.INVALID_INPUT)
    path, out = _path_arguments(argv, "--out")

    try:
        spec = _build_spec(_load_job_document(path))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"job rejected: {exc}")
        return int(WorkflowExitCode.INVALID_INPUT)

    try:
        candidates = expand_architectures(spec)
        if len(candidates) > 1:
            result_set = execute_engine_job_set(spec)
            print(f"verdict: {result_set.verdict}")
            for result in result_set.results:
                print(f"  {result.job.job_digest}: {result.verdict}")
                if result.output_directory:
                    print(f"    report directory: {result.output_directory}")
            if out is not None:
                out.write_text(json.dumps(result_set.to_dict(), indent=2, allow_nan=False))
            return result_set.exit_code
        result = execute_engine_job(candidates[0])
    except Exception as exc:  # solver/config failures are reported, not raised
        from raosim.mdo.job import JobValidationError

        if isinstance(exc, JobValidationError):
            print(f"job rejected: {exc}")
            return int(WorkflowExitCode.INVALID_INPUT)
        print(f"solve failed: {type(exc).__name__}: {exc}")
        return int(WorkflowExitCode.SOLVER_FAILED)

    print(f"verdict: {result.verdict}")
    print(f"  job digest        {result.job.job_digest}")
    if result.inputs is not None:
        print(f"  resolved inputs   {result.inputs.digest()}")
    for name, status in sorted(result.statuses.items()):
        print(f"  {name:18s} {status}")
    if result.violated_constraints:
        print("  violated constraints (worst first)")
        for name, margin in result.violated_constraints:
            print(f"      {name:24s} {margin:+.6g}")
    for target in result.targets:
        achieved = "n/a" if target.achieved is None else f"{target.achieved:.6g}"
        print(f"  target {target.metric} {target.relation} "
              f"{target.requested:g} -> {achieved} [{target.status}]")
        if target.reason:
            print(f"      {target.reason}")
    for note in result.job.notes:
        print(f"  note: {note}")
    for message in result.messages:
        print(f"  {message}")
    if result.output_directory is not None:
        print(f"  report directory  {result.output_directory}")
    if out is not None:
        out.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False))
        print(f"  result written to {out}")
    return int(result.exit_code)


def _cmd_guided(argv: Sequence[str]) -> int:
    from raosim.input_validation import WorkflowExitCode
    from lrekit.guided import GuidedAbort, run_guided

    save: Path | None = None
    should_run = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--run" and not should_run:
            should_run = True
        elif argument == "--save" and save is None and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            index += 1
            save = Path(argv[index]).expanduser()
        else:
            raise ValueError(f"unknown, duplicate or incomplete guided argument: {argument}")
        index += 1
    if should_run and save is None:
        raise ValueError("--run needs --save so the executed job is on disk and replayable")

    try:
        document = run_guided(ask=input, out=print)
    except (GuidedAbort, EOFError, KeyboardInterrupt) as exc:
        print(f"guided session ended: {exc or 'input closed'}")
        return int(WorkflowExitCode.INVALID_INPUT)

    if save is not None:
        try:
            import yaml

            save.write_text(yaml.safe_dump(document, sort_keys=False))
        except ImportError:
            save = save.with_suffix(".json")
            save.write_text(json.dumps(document, indent=2))
        print(f"job written to {save}")
    elif not should_run:
        print(json.dumps(document, indent=2))

    if not should_run:
        return int(WorkflowExitCode.MET)
    if save is None:
        print("--run needs --save so the executed job is on disk and replayable")
        return int(WorkflowExitCode.INVALID_INPUT)
    return _cmd_run([str(save)])


def _cmd_catalog(argv: Sequence[str]) -> int:
    from raosim.input_validation import WorkflowExitCode

    topic = argv[0] if argv else ""
    if topic not in _CATALOG_TOPICS or len(argv) != 1:
        print("usage: lrekit catalog {" + "|".join(_CATALOG_TOPICS) + "}")
        return int(WorkflowExitCode.INVALID_INPUT)

    if topic == "variables":
        from raosim.mdo.job import (
            DERIVABLE_VARIABLES,
            VARIABLE_DIMENSIONS,
            VariableMode,
        )

        print("design variables (mode: "
              + "|".join(m.value for m in VariableMode) + ")")
        for name, dim in sorted(VARIABLE_DIMENSIONS.items()):
            mark = "  [derived resolver unavailable]" if name in DERIVABLE_VARIABLES else ""
            print(f"  {name:16s} {dim}{mark}")
        return int(WorkflowExitCode.MET)

    if topic == "capabilities":
        from raosim.mdo.capabilities import JOB_CAPABILITIES

        for row in JOB_CAPABILITIES:
            print(f"  {row.kind:16s} {row.id}: {row.scope}")
        return int(WorkflowExitCode.MET)

    if topic in {"targets", "objectives", "unavailable"}:
        from raosim.mdo.metrics import (
            objective_capable_metrics,
            target_capable_metrics,
            unavailable_metrics,
        )

        chosen = {
            "targets": target_capable_metrics,
            "objectives": objective_capable_metrics,
            "unavailable": unavailable_metrics,
        }[topic]()
        print(f"{topic} ({len(chosen)})")
        for metric in sorted(chosen, key=lambda m: m.id):
            print(f"  {metric.id:38s} {metric.dimension}")
            if metric.unavailable_reason:
                print(f"      unavailable: {metric.unavailable_reason}")
        return int(WorkflowExitCode.MET)

    if topic == "constraints":
        from raosim.mdo.constraints import CONSTRAINT_MANIFEST

        print(f"constraint manifest ({len(CONSTRAINT_MANIFEST)})")
        for spec in CONSTRAINT_MANIFEST:
            print(f"  {spec.name:20s} {spec.optimizer_role:12s} "
                  f"{spec.category:14s} {spec.source_id or '-'}")
        return int(WorkflowExitCode.MET)

    if topic == "propellants":
        from raosim.mdo.propellants import PROPELLANTS

        print(f"propellants ({len(PROPELLANTS)})")
        for spec in PROPELLANTS.values():
            flag = " [ESTIMATED constants]" if getattr(
                spec, "estimated", False
            ) else ""
            print(f"  {spec.name:14s} O/F {spec.OF_default:<5g} "
                  f"coolant {spec.coolant_name}{flag}")
        return int(WorkflowExitCode.MET)

    from raosim.materials import material_names

    names = material_names()
    print(f"materials ({len(names)})")
    for name in names:
        print(f"  {name}")
    return int(WorkflowExitCode.MET)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a job verb.  Unknown verbs are the caller's to handle."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in JOB_VERBS:
        raise LookupError("not a job verb")
    verb, rest = args[0], args[1:]
    try:
        return {
            "guided": _cmd_guided,
            "run": _cmd_run,
            "validate": _cmd_validate,
            "catalog": _cmd_catalog,
        }[verb](rest)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        from raosim.input_validation import WorkflowExitCode

        print(f"job rejected: {exc}")
        return int(WorkflowExitCode.INVALID_INPUT)
