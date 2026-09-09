"""The guided interview: describe the engine, not the solver."""

from __future__ import annotations

import pytest

from lrekit.guided import GuidedAbort, GuidedSession, run_guided


def _scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


_BASE = ("1", "13 kN", "sea_level", "120 s", "LOX/RP-1", "GRCop-84",
         "Inconel 718", "230 s", "1")


def test_the_wizard_never_asks_for_solver_owned_design_variables():
    """A guided flow must not become a form for the optimizer's state."""
    prompts: list[str] = []

    def ask(prompt):
        prompts.append(prompt)
        return next(answers)

    answers = iter((*_BASE, "y"))
    run_guided(ask=ask, out=lambda _: None)

    asked = " ".join(prompts).lower()
    for owned in ("chamber pressure", "expansion ratio", "channel width",
                  "channel height", "pintle diameter", "pump speed", "rpm"):
        assert owned not in asked, f"wizard asked for {owned!r}"


def test_the_interview_produces_a_job_that_validates():
    from lrekit.job_cli import _build_spec
    from raosim.mdo.job import resolve_engine_job

    document = run_guided(ask=_scripted(*_BASE, "y"), out=lambda _: None)
    job = resolve_engine_job(_build_spec(document))
    assert job.spec.workflow.value == "optimize"
    assert job.mission.liner_material_name == "GRCop-84"
    assert job.mission.closeout_material_name == "Inconel 718"
    assert document["targets"][0]["value"] == "230 s"


def test_a_complete_design_point_selects_the_analyze_workflow():
    document = run_guided(
        ask=_scripted("3", "13 kN", "sea_level", "120 s", "LOX/RP-1",
                      "GRCop-84", "Inconel 718", "", "3 MPa", "8", "0.2", "0.2",
                      "20 mm", "30 krpm", "0.5 mm", "1.5 mm", "0.1", "0.8 mm", "y"),
        out=lambda _: None,
    )
    assert document["workflow"] == "analyze"
    assert "objective" not in document
    from lrekit.job_cli import _build_spec
    from raosim.mdo.job import resolve_engine_job

    resolved = resolve_engine_job(_build_spec(document))
    assert len(resolved.fixed_variables) == 10
    assert resolved.fixed_variables["Pc"] == 3e6
    assert not any("were not stated" in note for note in resolved.notes)


def test_a_bad_unit_is_corrected_in_the_interview_not_discovered_later():
    """Re-prompting is the point: a unit slip is a 1000x design error."""
    messages: list[str] = []
    document = run_guided(
        ask=_scripted("1", "13", "13 kN", "sea_level", "120 s", "LOX/RP-1",
                      "GRCop-84", "Inconel 718", "", "1", "y"),
        out=messages.append,
    )
    assert any("ambiguous" in m for m in messages)
    assert document["mission"]["thrust"] == "13 kN"


def test_an_invalid_choice_is_rejected_and_reprompted():
    messages: list[str] = []
    document = run_guided(
        ask=_scripted("1", "13 kN", "orbit", "vacuum", "120 s", "LOX/RP-1",
                      "GRCop-84", "Inconel 718", "", "1", "y"),
        out=messages.append,
    )
    assert any("choose one of" in m for m in messages)
    assert document["mission"]["thrust_condition"] == "vacuum"


def test_coverage_is_previewed_before_anything_is_solved():
    """A methane user must learn HTD is unavailable before waiting on a solve."""
    session = GuidedSession(ask=_scripted(), out=lambda _: None)
    preview = session.preview({
        "architecture": {"propellant": "LOX/LCH4"}, "workflow": "optimize",
    })
    joined = " ".join(preview)
    assert "HTD" in joined or "htd" in joined
    assert "UNKNOWN" in joined
    assert "ESTIMATE" in joined  # methane constants are estimates


def test_rp1_preview_does_not_raise_a_false_htd_warning():
    session = GuidedSession(ask=_scripted(), out=lambda _: None)
    joined = " ".join(session.preview({
        "architecture": {"propellant": "LOX/RP-1"}, "workflow": "analyze",
    }))
    assert "UNKNOWN" not in joined
    assert "not modelled" in joined  # the standing unavailable-metrics note


def test_declining_the_confirmation_aborts():
    with pytest.raises(GuidedAbort):
        run_guided(ask=_scripted(*_BASE, "n"), out=lambda _: None)
