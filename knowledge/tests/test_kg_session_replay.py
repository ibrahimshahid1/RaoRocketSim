"""Session replay distinguishes unchanged stores from fully resolved operations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import enrichment, harvest_extraction, packet_queue, store, verification_freshness  # noqa: E402
from tools import session_replay_check as replay_tool  # noqa: E402


@pytest.fixture
def replay_env(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(store, "REPORT_DIR", root / "reports")
    monkeypatch.setattr(store, "CHUNK_DB", root / "chunks.sqlite")
    monkeypatch.setattr(packet_queue, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(replay_tool, "ROOT", tmp_path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic PDF identity; no scientific assertion")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    store.save("manifest", [{"id": "DOC_TEST", "sha256": source_sha, "source_pdf": source.name}])
    return {"root": tmp_path, "source": source, "source_sha": source_sha}


def reviewed_packet(env, payload, *, suffix=".json"):
    reviewed = env["root"] / f"reviewed{suffix}"
    reviewed.write_text(json.dumps(payload) + "\n")
    file_record = {"path": reviewed.name, "sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest()}
    packet_queue._save({"packets": {"P": {
        "id": "P", "state": "FINAL_AUDIT_ELIGIBLE", "document_id": "DOC_TEST",
        "source_pdf": env["source"].name, "source_sha256": env["source_sha"],
        "history": [{"at": "2026-10-17T12:00:00+00:00"}],
        "evidence": {"review": {"files": [file_record]}},
    }}})
    return reviewed, file_record


def seed_confirmation(env):
    equation = {"id": "EQ_TEST", "type": "Equation", "document_id": "DOC_TEST", "page": 1,
                "form": "x=3", "quote": "Synthetic source assertion", "source_sha256": env["source_sha"],
                "source_state": "CURRENT", "active": True, "provenance_kind": "SOURCE_STATES",
                "transcribed_by": {"role": "A", "model": "synthetic-transcriber"}}
    store.save("equations", [equation])
    verdict = {"object_id": equation["id"], "source_sha256": env["source_sha"], "refuted": False,
               "confidence": "high", "reason": "Original synthetic review",
               "reviewer": {"role": "F", "model": "synthetic-independent-reviewer"},
               "content_sha256": verification_freshness.content_sha256(equation)}
    assert harvest_extraction.apply_verification([verdict])["confirmed"] == 1
    return verdict


@pytest.mark.parametrize("change", ["missing", "bytes", "manifest"])
def test_missing_or_changed_source_is_unresolved_even_when_stores_are_unchanged(replay_env, change):
    reviewed_packet(replay_env, [])
    if change == "missing":
        replay_env["source"].unlink()
    elif change == "bytes":
        replay_env["source"].write_bytes(b"different PDF identity")
    else:
        store.save("manifest", [{"id": "DOC_TEST", "sha256": "different-source"}])
    result = replay_tool.replay("2026-10-16")
    assert result["no_op"] is True
    assert result["all_steps_resolved"] is False and result["success"] is False
    assert result["replay_failures"][0]["kind"] == (
        "SOURCE_FILE_UNAVAILABLE" if change == "missing" else "SOURCE_IDENTITY_MISMATCH")


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_missing_or_changed_reviewed_file_is_reported(replay_env, change):
    reviewed, _ = reviewed_packet(replay_env, [])
    if change == "missing":
        reviewed.unlink()
    else:
        reviewed.write_text("changed after packet review")
    result = replay_tool.replay("2026-10-16")
    assert result["no_op"] is False and result["success"] is False
    assert result["missing_reviewed_files"] == ([reviewed.name] if change == "missing" else [])
    assert result["changed_reviewed_files"] == ([reviewed.name] if change == "changed" else [])


@pytest.mark.parametrize("exit_code,bank_report,expected_success", [
    (2, {"already_banked": True}, False),
    (0, None, False),
    (0, b"invalid JSON", False),
    (0, {"unresolved_candidate_links": [{"candidate_id": "CAND_UNKNOWN"}]}, False),
    (0, {"already_banked": True}, True),
])
def test_bank_failures_and_missing_reports_cannot_pass_on_unchanged_stores(
        replay_env, monkeypatch, exit_code, bank_report, expected_success):
    reviewed_packet(replay_env, {"type": "result", "value": {}}, suffix=".jsonl")

    def bank(command, **kwargs):
        report_path = Path(command[command.index("--report-out") + 1])
        if isinstance(bank_report, bytes):
            report_path.write_bytes(bank_report)
        elif bank_report is not None:
            report_path.write_text(json.dumps(bank_report))
        return subprocess.CompletedProcess(command, exit_code, stdout="All results already banked", stderr="synthetic")

    monkeypatch.setattr(replay_tool.subprocess, "run", bank)
    result = replay_tool.replay("2026-10-16")
    assert result["no_op"] is True
    assert result["all_steps_resolved"] is expected_success
    assert result["success"] is expected_success
    if not expected_success:
        assert result["replay_failures"][0]["kind"] == "BANKING_FAILED"
    else:
        assert result["bank"] == {"already_banked_no_change": 1}


def test_refused_content_cannot_pass_using_an_existing_confirmation(replay_env):
    incoming = {**seed_confirmation(replay_env), "content_sha256": "0" * 64, "reason": "A new unaccepted verdict"}
    reviewed_packet(replay_env, [incoming])
    before = deepcopy(store.load("equations"))
    result = replay_tool.replay("2026-10-16")
    assert result["no_op"] is True and result["success"] is False
    assert result["verdicts"]["content_mismatch"] == 1
    assert result["replay_failures"][0]["kind"] == "VERIFICATION_REFUSED"
    assert store.load("equations") == before
    assert store.load("equations")[0]["verification"] == "CONFIRMED"


def test_current_exact_verdict_is_a_successful_noop(replay_env):
    reviewed_packet(replay_env, [seed_confirmation(replay_env)])
    result = replay_tool.replay("2026-10-16")
    assert result["no_op"] is True and result["all_steps_resolved"] is True and result["success"] is True
    assert result["verdicts"]["unchanged"] == 1


def test_retired_assertion_verdict_replay_remains_resolved_history(replay_env):
    verdict = seed_confirmation(replay_env)
    equation, = store.load("equations")
    equation.update(active=False, source_state="SUPERSEDED_EXTRACTION", verification="REFUTED")
    store.save("equations", [equation])
    reviewed_packet(replay_env, [verdict])
    before = deepcopy(store.load("equations"))
    result = replay_tool.replay("2026-10-16")
    assert result["success"] is True and result["verdicts"]["retired_or_noncurrent"] == 1
    assert store.load("equations") == before


@pytest.mark.parametrize("content_changed", [False, True])
def test_exact_superseded_verdict_remains_history_without_replacing_latest(replay_env, content_changed):
    old_verdict = seed_confirmation(replay_env)
    if content_changed:
        old_record, = store.load("equations")
        changed_record = {**old_record, "description": "A later inspected source scope"}
        verification_freshness.reconcile_record(old_record, changed_record, now="2026-10-17")
        store.save("equations", [changed_record])
    current, = store.load("equations")
    new_verdict = {**old_verdict, "reason": "A later independent review",
                   "content_sha256": verification_freshness.content_sha256(current)}
    assert harvest_extraction.apply_verification([new_verdict])["confirmed"] == 1
    reviewed_packet(replay_env, [old_verdict])
    before = deepcopy(store.load("equations"))
    result = replay_tool.replay("2026-10-16")
    assert result["success"] is True
    assert result["historical_verdicts_skipped"] == 1
    assert not result["verdicts"]  # historical recognition applies no new verdict
    assert store.load("equations") == before
    assert "does not verify current content" in result["historical_scope"]


def test_exact_historical_parser_review_does_not_replace_newer_rejection(replay_env):
    candidate = {"id": "CAND_TEST", "candidate_kind": "formula", "document_id": "DOC_TEST",
                 "source_sha256": replay_env["source_sha"], "source_state": "CURRENT", "pages": [1], "active": True}
    store.save("enrichments", [candidate])
    item = {"candidate_id": candidate["id"], "state": "SOURCE_PAGE_COMPARED", "reason": "Original QA",
            "evidence": {"pdf_page": 1}}
    payload = {"reviewer": {"role": "F", "model": "synthetic-reviewer"}, "parser_reviews": [item]}
    _, reviewed = reviewed_packet(replay_env, payload)
    evidence = dict(item["evidence"], reviewer=payload["reviewer"], review_file=reviewed["path"],
                    review_file_sha256=reviewed["sha256"])
    enrichment.record_review(candidate["id"], state=item["state"], reason=item["reason"], evidence=evidence)
    enrichment.record_review(candidate["id"], state="REJECTED", reason="Later QA", evidence={"pdf_page": 1, "version": 2})
    before = deepcopy(store.load("enrichments"))
    result = replay_tool.replay("2026-10-16")
    assert result["success"] is True and result["historical_parser_reviews_skipped"] == 1
    assert store.load("enrichments") == before
    assert store.load("enrichments")[0]["review_state"] == "REJECTED"


def test_resolved_operation_that_changes_stores_is_not_successful_noop(replay_env):
    seed_confirmation(replay_env)
    equation, = store.load("equations")
    fresh_verdict = {"object_id": equation["id"], "source_sha256": replay_env["source_sha"], "refuted": False,
                     "confidence": "high", "reason": "A genuinely new independent review",
                     "reviewer": {"role": "F", "model": "synthetic-reviewer"},
                     "content_sha256": verification_freshness.content_sha256(equation)}
    reviewed_packet(replay_env, [fresh_verdict])
    result = replay_tool.replay("2026-10-16")
    assert result["all_steps_resolved"] is True
    assert result["no_op"] is False and result["success"] is False
    assert "equations" in result["stores_changed_ignoring_timestamps"]


@pytest.mark.parametrize("no_op,resolved,expected_exit", [(True, True, 0), (True, False, 1), (False, True, 1)])
def test_command_exit_requires_both_noop_and_resolved_steps(monkeypatch, no_op, resolved, expected_exit):
    monkeypatch.setattr(replay_tool, "replay", lambda since: {
        "no_op": no_op, "all_steps_resolved": resolved, "success": no_op and resolved,
    })
    assert replay_tool.main(["--since", "2026-10-16"]) == expected_exit
