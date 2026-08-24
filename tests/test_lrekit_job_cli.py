"""The job verbs, and the compatibility guarantee around them.

O/F optimization was implemented long before it was reachable from a command
line, which meant in practice it was not usable. These gates keep the unified
workflow reachable, and keep the verdict visible in the process exit code.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from lrekit.job_cli import JOB_VERBS, main as job_main
from raosim.input_validation import WorkflowExitCode

_JOB = textwrap.dedent(
    """
    mission:
      thrust: 13 kN
      burn_duration: 120 s
    workflow: analyze
    architecture:
      liner_material: GRCop-84
    variables:
      Pc:             {mode: fixed, value: 3 MPa}
      eps:            {mode: fixed, value: 8.0}
      dp_f_frac:      {mode: fixed, value: 0.2}
      dp_o_frac:      {mode: fixed, value: 0.2}
      D_pintle:       {mode: fixed, value: 20 mm}
      N_rpm:          {mode: fixed, value: 30 krpm}
      channel_width:  {mode: fixed, value: 0.5 mm}
      channel_height: {mode: fixed, value: 1.5 mm}
      film_frac:      {mode: fixed, value: 0.1}
      t_wall:         {mode: fixed, value: 0.8 mm}
    targets:
      - {metric: performance.isp_delivered, relation: at_least, value: 230 s}
    """
)


@pytest.fixture
def job_file(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(_JOB)
    return path


def test_validate_accepts_a_well_formed_job(job_file, capsys):
    assert job_main(["validate", str(job_file)]) == WorkflowExitCode.MET
    out = capsys.readouterr().out
    assert "is valid" in out
    assert "GRCop-84" in out
    assert "job digest" in out


def test_validate_can_emit_the_resolved_job_for_replay(job_file, tmp_path,
                                                       capsys):
    out_path = tmp_path / "resolved.json"
    code = job_main(
        ["validate", str(job_file), "--resolved-out", str(out_path)]
    )
    capsys.readouterr()
    assert code == WorkflowExitCode.MET
    payload = json.loads(out_path.read_text())
    assert payload["workflow"] == "analyze"
    assert len(payload["job_digest"]) == 64
    assert payload["fixed_variables"]["Pc"] == pytest.approx(3.0e6)


def test_a_missing_unit_reaches_the_cli_as_invalid_input(tmp_path, capsys):
    """The unit rule has to survive all the way to an exit code."""
    path = tmp_path / "bad.yaml"
    path.write_text("mission:\n  thrust: 13\n")
    assert job_main(["validate", str(path)]) == WorkflowExitCode.INVALID_INPUT
    assert "ambiguous" in capsys.readouterr().out


def test_a_misspelled_job_key_is_rejected_not_ignored(tmp_path, capsys):
    path = tmp_path / "typo.yaml"
    path.write_text("mission:\n  thrust: 13 kN\narchitecure: {}\n")
    assert job_main(["validate", str(path)]) == WorkflowExitCode.INVALID_INPUT
    assert "unknown job keys" in capsys.readouterr().out


def test_run_reports_the_verdict_through_the_exit_code(job_file, tmp_path,
                                                       capsys):
    out_path = tmp_path / "result.json"
    code = job_main(["run", str(job_file), "--out", str(out_path)])
    out = capsys.readouterr().out

    assert code in {
        int(WorkflowExitCode.MET),
        int(WorkflowExitCode.CANDIDATE_VIOLATES),
        int(WorkflowExitCode.INDETERMINATE),
    }
    assert "verdict:" in out
    payload = json.loads(out_path.read_text())
    assert payload["exit_code"] == code
    assert len(payload["resolved_inputs_digest"]) == 64
    if payload["verdict"] == "CANDIDATE_VIOLATES":
        assert payload["violated_constraints"], "a violation must name itself"
        assert "violated constraints" in out


def test_catalog_lists_every_registry_topic(capsys):
    for topic in ("variables", "targets", "objectives", "constraints",
                  "propellants", "materials", "unavailable"):
        assert job_main(["catalog", topic]) == WorkflowExitCode.MET
        assert capsys.readouterr().out.strip(), topic


def test_catalog_explains_why_a_quantity_is_unavailable(capsys):
    job_main(["catalog", "unavailable"])
    out = capsys.readouterr().out
    assert "masses.engine_dry_mass" in out
    assert "valves, lines" in out
    assert "life.cycles_to_failure" in out
    assert "not a cycle count" in out


def test_an_unknown_catalog_topic_is_invalid_input(capsys):
    assert job_main(["catalog", "nonsense"]) == WorkflowExitCode.INVALID_INPUT
    assert "usage:" in capsys.readouterr().out


def test_non_job_verbs_are_left_to_the_legacy_cli():
    """Additive by construction: no existing invocation changes meaning."""
    assert JOB_VERBS == {"guided", "run", "validate", "catalog"}
    with pytest.raises(LookupError):
        job_main(["--thrust", "13000"])
    with pytest.raises(LookupError):
        job_main([])
