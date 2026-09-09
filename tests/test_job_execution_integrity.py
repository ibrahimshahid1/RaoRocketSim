"""Regression checks at the public job, CLI, verdict and report boundaries."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from lrekit.job_cli import _build_spec, _load_job_document, main as cli
from raosim.input_validation import WorkflowExitCode
from raosim.mdo.execute import (
    _isp_floor, _snapshot_value, _verdict, execute_engine_job,
    execute_engine_job_set, expand_architectures,
)
from raosim.mdo.job import (
    ArchitectureSelection, EngineJobSpec, JobValidationError, MissionRequest,
    ModelSelection, ObjectiveRequest, OperatingPointSpec, OutputPolicy,
    TargetRequest, VariableRequest, resolve_engine_job,
)
from raosim.mdo.snapshot import FieldComparison, SnapshotComparison
from raosim.units import UnitError, parse_quantity


def spec(**overrides):
    return EngineJobSpec(**{
        "mission": MissionRequest(thrust="13 kN"),
        "architecture": ArchitectureSelection(liner_material="GRCop-84"),
        **overrides,
    })


@pytest.fixture
def completed_pipeline(monkeypatch):
    """Replace numerical work with known outputs; retain real job plumbing."""
    import raosim.mdo.engine as engine
    import raosim.mdo.state as state
    import raosim.mdo.postprocess as postprocess

    gates = {name: SimpleNamespace(value=True) for name in (
        "optimizer_constraints_feasible", "numerical_validity", "physics_feasible",
        "requirements_feasible", "workflow_readiness_feasible",
    )}
    snapshot = SimpleNamespace(
        performance={"specific_impulse_delivered_s": SimpleNamespace(value=260.0)},
        constraints_gates=gates, warnings=(),
        to_dict=lambda: {"performance": {"specific_impulse_delivered_s": {"value": 260.0}}},
    )
    comparison = SnapshotComparison("mdo", "traditional", {
        "performance.isp_delivered": FieldComparison(260, 260, 0, 0, "within_tolerance", 0)
    }, {}, 1e-9, 1e-6)
    reevaluation = SimpleNamespace(
        authoritative_snapshot=snapshot, mdo_snapshot=snapshot,
        variational_contour_validation=None, warnings=(), comparison=comparison,
        metadata={"input_integrity": {"status": "pass", "differences": [], "error": None}},
    )
    observed = {}

    def solve(design, mission, **kwargs):
        observed["mission"] = mission
        return SimpleNamespace(performance=SimpleNamespace(
            OF=mission.OF, mdot_total=3.0, Isp_delivered=260.0, eta_cstar=0.91,
        ))

    def reeval(design, mission, **kwargs):
        observed["handoff"] = kwargs
        return reevaluation

    monkeypatch.setattr(engine, "chamber_surfaces_for", lambda mission: None)
    monkeypatch.setattr(state, "solve_engine_state", solve)
    monkeypatch.setattr(postprocess, "reevaluate", reeval)
    return reevaluation, observed


@pytest.mark.parametrize("text,dimension", [
    ("1 Mbar", "pressure"), ("1 mpa", "pressure"), ("1 MM", "length"),
    ("1e308 GPa", "pressure"),
])
def test_unit_prefix_case_and_conversion_overflow_are_not_guessed(text, dimension):
    with pytest.raises(UnitError):
        parse_quantity(text, dimension=dimension)


@pytest.mark.parametrize("mission,expected", [
    (MissionRequest("13 kN", thrust_condition="vacuum"), 0.0),
    (MissionRequest("13 kN", thrust_condition="altitude", altitude="10 km"), 26436.3),
])
def test_ambient_condition_reaches_both_execution_paths(completed_pipeline, mission, expected):
    result = execute_engine_job(spec(mission=mission))
    _, observed = completed_pipeline
    assert result.verdict == "MET"
    assert observed["mission"].Pa == pytest.approx(expected, rel=2e-4)
    assert result.inputs.ambient_pressure == pytest.approx(expected, rel=2e-4)
    assert observed["handoff"]["resolved_input_contract"] is result.inputs
    assert result.inputs.performance.eta_cstar_effective == 0.91


def test_operating_point_supplies_omitted_condition_and_duration():
    resolved = resolve_engine_job(spec(operating_points=(
        OperatingPointSpec("test", ambient_pressure="0 Pa", duration="30 s"),
    )))
    assert resolved.mission.Pa == 0.0
    assert resolved.mission.burn_time == 30.0


@pytest.mark.parametrize("kwargs,match", [
    ({"mission": MissionRequest("13 kN", thrust_condition="sea_level"),
      "operating_points": (OperatingPointSpec("test", ambient_pressure="0 Pa"),)}, "conflicts"),
    ({"mission": MissionRequest("13 kN", burn_duration="120 s"),
      "operating_points": (OperatingPointSpec("test", ambient_pressure="101325 Pa", duration="30 s"),)}, "conflicts"),
    ({"mission": MissionRequest("13 kN", altitude="10 km")}, "only consumed"),
    ({"mission": MissionRequest("13 kN", dry_mass_max="10 kg")}, "unavailable"),
    ({"mission": MissionRequest("13 kN", envelope_diameter_max="1 m")}, "unavailable"),
    ({"targets": (TargetRequest("performance.isp_delivered", "at_least", "200 s", condition="other"),)}, "condition"),
])
def test_conflicting_or_unsupported_requirements_are_refused(kwargs, match):
    with pytest.raises(JobValidationError, match=match):
        resolve_engine_job(spec(**kwargs))


@pytest.mark.parametrize("kwargs", [
    {"feed": "pressure_fed"}, {"injector": "impinging"}, {"cooling": "ablative"},
    {"closeout_material": "Inconel 718"},
])
def test_architecture_selection_cannot_fall_back(kwargs):
    with pytest.raises(JobValidationError):
        ArchitectureSelection(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"parts": "all"}, {"cad_formats": ("step",)}, {"cad_fidelity": "L2_reference_cad"},
    {"reports": ("summary",)}, {"reports": ("imaginary",), "output_dir": "unused"},
])
def test_unimplemented_output_policy_is_refused(kwargs):
    with pytest.raises(JobValidationError):
        OutputPolicy(**kwargs)


def test_mission_isp_and_qualification_are_evaluated_separately(completed_pipeline):
    result = execute_engine_job(spec(mission=MissionRequest(
        "13 kN", burn_duration="30 s", isp_min="270 s", qualification_duration="60 s",
    )))
    assert len(result.targets) == 1
    assert result.targets[0].status == "fail"
    assert result.verdict == "CANDIDATE_VIOLATES"
    assert result.statuses["requirement_coverage"] == "unknown"
    row = next(row for row in result.requirement_coverage if row["input"] == "mission.qualification_duration")
    assert row["status"] == "unknown"
    assert "test evidence" in row["reason"]
    result = execute_engine_job(spec(mission=MissionRequest("13 kN", qualification_duration="150 s")))
    assert result.verdict == "INDETERMINATE"


def test_optimizer_receives_resolved_pins_bounds_and_strongest_floor(monkeypatch):
    import raosim.mdo.nlp as nlp
    import raosim.mdo.engine as engine

    observed = {}

    def solve(mission, floor, **kwargs):
        observed.update(floor=floor, **kwargs)
        return SimpleNamespace(success=False, optimizer_status="unknown", message="synthetic stop", design={})

    monkeypatch.setattr(nlp, "solve_min_mass", solve)
    monkeypatch.setattr(engine, "chamber_surfaces_for", lambda mission: None)
    result = execute_engine_job(spec(
        mission=MissionRequest("13 kN", isp_min="240 s"), workflow="optimize",
        objective=ObjectiveRequest("masses.dry_mass_partial"),
        targets=(TargetRequest("performance.isp_delivered", "at_least", "200 s"),),
        variables=(VariableRequest("Pc", "fixed", "3 MPa"),
                   VariableRequest("eps", "bounded", lower=6, upper=9)),
    ))
    assert observed["floor"] == 240.0
    assert observed["fixed_variables"] == {"Pc": 3e6}
    assert observed["bounds"]["eps"] == (6, 9)
    assert "Pc" not in observed["bounds"]
    assert result.verdict == "SOLVER_FAILED"
    assert len([row for row in result.requirement_coverage if row.get("metric")]) == 2


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "NaN"])
def test_invalid_metric_is_unknown_not_a_numeric_result(value):
    snapshot = SimpleNamespace(performance={"specific_impulse_delivered_s": SimpleNamespace(value=value)})
    achieved, reason = _snapshot_value(snapshot, "performance.isp_delivered")
    assert achieved is None
    assert "not a scalar" in reason or "nonfinite" in reason


def test_handoff_drift_and_disagreement_are_visible_in_verdict(completed_pipeline):
    reeval, _ = completed_pipeline
    reeval.metadata["input_integrity"] = {"status": "fail", "differences": ["ambient pressure"]}
    result = execute_engine_job(spec())
    assert result.verdict == "SOLVER_FAILED"
    assert any("ambient pressure" in message for message in result.messages)
    reeval.metadata["input_integrity"]["status"] = "pass"
    reeval.comparison.scalars["performance.isp_delivered"] = FieldComparison(
        260, 250, 10, 10/260, "different", -10,
    )
    result = execute_engine_job(spec())
    assert result.verdict == "INDETERMINATE"
    assert result.model_consistency["different_fields"] == ["performance.isp_delivered"]
    assert result.to_dict()["comparison"]["scalars"]["performance.isp_delivered"]["signed_delta"] == -10


def test_workflow_readiness_failure_and_invalid_status_cannot_pass():
    assert _verdict({"workflow": "pass", "workflow_readiness": "fail"}, ())[0] == "CANDIDATE_VIOLATES"
    assert _verdict({"workflow": "pass", "physics": "banana"}, ())[0] == "INDETERMINATE"
    assert _verdict({"workflow": "pass"}, ())[0] == "INDETERMINATE"


def test_enumeration_preserves_units_and_executes_each_candidate(completed_pipeline):
    request = spec(variables=(VariableRequest("Pc", "enumerate", values=("2 MPa", "3 MPa")),))
    expanded = expand_architectures(request)
    assert [resolve_engine_job(child).fixed_variables["Pc"] for child in expanded] == [2e6, 3e6]
    with pytest.raises(JobValidationError, match="execute_engine_job_set"):
        execute_engine_job(request)
    results = execute_engine_job_set(request)
    assert len(results.results) == 2
    assert [result.design["Pc"] for result in results.results] == [2e6, 3e6]


def test_report_manifest_hashes_and_job_replay(completed_pipeline, tmp_path):
    request = spec(
        variables=(VariableRequest("Pc", "fixed", "3.123456789 MPa"),),
        targets=(TargetRequest("performance.isp_delivered", "range", "230 s", "270 s"),),
        output_policy=OutputPolicy(reports=("resolved_inputs", "snapshot", "comparison", "coverage"),
                                   output_dir=str(tmp_path)),
    )
    result = execute_engine_job(request)
    from pathlib import Path
    directory = Path(result.output_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["package_status"] == "complete"
    assert not manifest["hardware_release"]
    replay = _build_spec(_load_job_document(directory / "job.json"))
    assert replay.digest() == request.digest()
    for record in manifest["reports"].values():
        assert hashlib.sha256((directory / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    second = execute_engine_job(request)
    assert second.output_directory != result.output_directory
    assert (directory / "summary.json").exists()


def test_runtime_failure_retains_summary_and_stamps_missing_reports(monkeypatch, tmp_path):
    import raosim.mdo.engine as engine

    def unavailable(*args, **kwargs):
        raise ValueError("synthetic stage failure")

    monkeypatch.setattr(engine, "chamber_surfaces_for", unavailable)
    result = execute_engine_job(spec(output_policy=OutputPolicy(
        reports=("snapshot", "coverage"), output_dir=str(tmp_path), allow_partial_package=True,
    )))
    from pathlib import Path
    directory = Path(result.output_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert result.verdict == "SOLVER_FAILED"
    assert manifest["package_status"] == "partial"
    assert manifest["run_status"] == "failed"
    assert manifest["reports"]["snapshot"]["status"] == "unavailable"
    assert "synthetic stage failure" in (directory / "summary.json").read_text()


def test_later_failure_retains_solved_inputs_and_failed_publication_is_consistent(completed_pipeline, monkeypatch, tmp_path):
    import raosim.mdo.postprocess as postprocess
    from pathlib import Path

    def unavailable(*args, **kwargs):
        raise RuntimeError("traditional stage unavailable")

    monkeypatch.setattr(postprocess, "reevaluate", unavailable)
    result = execute_engine_job(spec(output_policy=OutputPolicy(
        reports=("snapshot", "resolved_inputs"), output_dir=str(tmp_path),
    )))
    assert result.inputs is not None
    assert result.design
    assert result.statuses["report_publication"] == "fail"
    directory = Path(result.output_directory)
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert summary["statuses"]["report_publication"] == "fail"
    assert manifest["verdict"] == summary["verdict"] == result.verdict
    assert manifest["package_status"] == "partial"
    assert manifest["reports"]["resolved_inputs"]["status"] == "available"


def test_cli_enumeration_emits_each_candidate(completed_pipeline, tmp_path):
    request = spec(variables=(VariableRequest("Pc", "enumerate", values=("2 MPa", "3 MPa")),))
    job_path, output = tmp_path / "job.json", tmp_path / "results.json"
    job_path.write_text(json.dumps(request.to_document()))
    assert cli(["run", str(job_path), "--out", str(output)]) == WorkflowExitCode.MET
    payload = json.loads(output.read_text())
    assert len(payload["results"]) == 2
    assert {result["design"]["Pc"] for result in payload["results"]} == {2e6, 3e6}


@pytest.mark.parametrize("kwargs", [
    {"name": "Pc", "mode": "free", "value": "3 MPa"},
    {"name": "eps", "mode": "fixed", "value": 1},
    {"name": "film_frac", "mode": "fixed", "value": True},
    {"name": "Pc", "mode": "enumerate", "values": ("3 MPa", "30 bar")},
    {"name": "eps", "mode": "enumerate", "values": "23"},
])
def test_ambiguous_variable_payloads_are_refused(kwargs):
    with pytest.raises(JobValidationError):
        VariableRequest(**kwargs)


@pytest.mark.parametrize("suffix,content", [
    ("json", '{"mission":{"thrust":"13 kN","thrust":"1 N"}}'),
    ("yaml", "mission:\n  thrust: 13 kN\n  thrust: 1 N\n"),
])
def test_duplicate_document_keys_are_refused(tmp_path, suffix, content):
    path = tmp_path / f"duplicate.{suffix}"
    path.write_text(content)
    with pytest.raises(ValueError, match="duplicate"):
        _load_job_document(path)


@pytest.mark.parametrize("extra", [
    {"schema_version": "0.1"}, {"schema_version": None}, {"models": []},
    {"architecture": False}, {"targets": {}}, {"operating_points": "ignored"},
])
def test_cli_document_does_not_drop_malformed_fields(extra):
    with pytest.raises(ValueError):
        _build_spec({"mission": {"thrust": "13 kN"}, **extra})


@pytest.mark.parametrize("arguments", [["--out"], ["--bogus", "value"], ["--out", "--oops"]])
def test_cli_missing_or_unknown_option_arguments_are_invalid(arguments):
    assert cli(["run", "job.yaml", *arguments]) == WorkflowExitCode.INVALID_INPUT
