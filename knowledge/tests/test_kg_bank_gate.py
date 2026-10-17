"""S3 must retain bank-report failures and block the packet's integration."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import enrichment, graph, harvest_extraction, packet_queue as pq, report, store, verification_freshness, work_journal, work_queue  # noqa: E402
from tools import agentic_packet as ap  # noqa: E402


@pytest.mark.parametrize("key", ap.BANK_MISMATCH_KEYS)
def test_bank_mismatches_preserves_each_report_failure(key):
    failure = [{"object_id": "EQ_T", "reason": "reviewed link did not land"}]
    assert ap.bank_mismatches({key: failure, "n_equations": 1}) == {key: failure}


def test_bank_mismatches_accepts_empty_or_replayed_reports():
    assert ap.bank_mismatches({}) == {}
    assert ap.bank_mismatches({"already_banked": True}) == {}
    assert ap.bank_mismatches({key: [] for key in ap.BANK_MISMATCH_KEYS}) == {}


@pytest.fixture
def verified_packet(tmp_path, monkeypatch):
    """A stored VERIFIED packet; real queue transitions remain enabled."""
    root = tmp_path / "knowledge"
    for module, name, value in (
        (store, "KNOWLEDGE_ROOT", root), (store, "REPORT_DIR", root / "reports"),
        (store, "GRAPH_DIR", root / "graph"), (pq, "REPO_ROOT", tmp_path),
        (work_journal, "REPO_ROOT", tmp_path), (ap, "ROOT", tmp_path),
        (ap, "SESSION_DIR", root / "workflows/results/session"),
    ):
        monkeypatch.setattr(module, name, value)
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text(json.dumps({"type": "result", "value": {
        "target": "pages1-1", "doc_id": "DOC_T", "reviewer": {"role": "F", "model": "reviewer"},
        "equations": [{"label": "Eq. (1)"}],
    }}) + "\n")
    source_sha = "a" * 64
    store.save("manifest", [{"id": "DOC_T", "sha256": source_sha}])
    pq._save({"packets": {"P": {
        "id": "P", "state": "VERIFIED", "document_id": "DOC_T", "source_sha256": source_sha,
        "history": [], "evidence": {"review": {"files": [{
            "path": reviewed.name, "sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest(),
        }]}},
    }}})
    monkeypatch.setattr(graph, "assemble", lambda: {"meta": {}})
    monkeypatch.setattr(work_queue, "write", lambda: {"summary": {}})
    monkeypatch.setattr(report, "write", lambda: None)
    monkeypatch.setattr(ap, "_graph_integrity", lambda: {
        "nodes": 0, "edges": 0, "duplicate_node_ids": [], "duplicate_edges": [],
        "dangling_edges": 0, "edges_without_provenance": 0,
    })
    return tmp_path


@pytest.mark.parametrize("key", ap.BANK_MISMATCH_KEYS)
def test_s3_bank_mismatch_blocks_integration_and_keeps_evidence(verified_packet, monkeypatch, key):
    failure = [{"object_id": "EQ_T", "reason": "reviewed link did not land"}]

    def bank_with_mismatch(command, **kwargs):
        report_path = Path(command[command.index("--report-out") + 1])
        report_path.write_text(json.dumps({key: failure, "n_equations": 1}))
        return subprocess.CompletedProcess(command, 0, stdout="banked", stderr="")

    monkeypatch.setattr(ap.subprocess, "run", bank_with_mismatch)
    result = ap.s3_integrate("P", "integrator")
    expected = [{"file": "reviewed.jsonl", key: failure}]
    assert result["state"] == "BLOCKED"
    assert result["bank_mismatches"] == expected
    packet = pq.load()["packets"]["P"]
    assert packet["state"] == "BLOCKED"
    assert [step["to"] for step in packet["history"]] == ["BLOCKED"]
    assert "integration" not in packet["evidence"]
    log = json.loads((verified_packet / result["log"]).read_text())
    assert log["bank_mismatches"] == expected
    assert log["steps"][0]["mismatches"] == {key: failure}
    assert packet["history"][0]["evidence"] == {
        result["log"]: hashlib.sha256((verified_packet / result["log"]).read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("bank_report", [{"n_equations": 1}, {"already_banked": True}])
def test_s3_clean_bank_report_allows_integration(verified_packet, monkeypatch, bank_report):
    def bank_cleanly(command, **kwargs):
        Path(command[command.index("--report-out") + 1]).write_text(json.dumps(bank_report))
        return subprocess.CompletedProcess(command, 0, stdout="banked", stderr="")

    monkeypatch.setattr(ap.subprocess, "run", bank_cleanly)
    result = ap.s3_integrate("P", "integrator")
    assert result["state"] == "GRAPH_INTEGRATED"
    assert pq.load()["packets"]["P"]["state"] == "GRAPH_INTEGRATED"


@pytest.mark.parametrize("report_bytes", [None, b"{invalid JSON", b"[]", b"\xff"])
def test_s3_missing_or_unreadable_bank_report_blocks_integration(verified_packet, monkeypatch, report_bytes):
    def bank_without_usable_report(command, **kwargs):
        if report_bytes is not None:
            Path(command[command.index("--report-out") + 1]).write_bytes(report_bytes)
        return subprocess.CompletedProcess(command, 0, stdout="banked", stderr="")

    monkeypatch.setattr(ap.subprocess, "run", bank_without_usable_report)
    result = ap.s3_integrate("P", "integrator")
    assert result["state"] == "BLOCKED"
    failures = result["bank_mismatches"]
    assert len(failures) == 1 and failures[0]["file"] == "reviewed.jsonl"
    error = failures[0]["bank_report_errors"][0]
    assert error["report"].endswith("bank_reports/P.reviewed.json")
    assert error["reason"]
    packet = pq.load()["packets"]["P"]
    assert packet["state"] == "BLOCKED"
    assert [step["to"] for step in packet["history"]] == ["BLOCKED"]
    assert "integration" not in packet["evidence"]
    log = json.loads((verified_packet / result["log"]).read_text())
    assert log["bank_mismatches"] == failures
    report_path = verified_packet / error["report"]
    assert (log["steps"][0]["report"] is None) == (report_bytes is None)
    if report_bytes is None:
        assert not report_path.exists()
    else:
        assert report_path.read_bytes() == report_bytes


def test_missing_bank_report_does_not_reuse_or_overwrite_previous_attempt(verified_packet, monkeypatch):
    previous_report = ap.evidence_path("bank_reports", "P.reviewed")
    previous_report.write_text(json.dumps({"already_banked": True}))
    previous_log = ap.evidence_path("integration", "P")
    previous_log.write_text(json.dumps({"packet": "P", "previous_attempt": True}))
    original = {path: path.read_bytes() for path in (previous_report, previous_log)}
    monkeypatch.setattr(ap.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="banked", stderr=""))

    result = ap.s3_integrate("P", "integrator")
    assert result["state"] == "BLOCKED"
    assert result["log"].endswith("integration/P.attempt-2.json")
    assert result["bank_mismatches"][0]["bank_report_errors"][0]["report"].endswith(
        "bank_reports/P.reviewed.attempt-2.json")
    assert all(path.read_bytes() == content for path, content in original.items())


def verdict_packet(tmp_path, verdicts):
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(verdicts))
    queue = pq.load()
    queue["packets"]["P"]["evidence"]["review"]["files"] = [{
        "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }]
    pq._save(queue)


def confirmed_equation():
    equation = {"id": "EQ_T", "type": "Equation", "document_id": "DOC_T", "page": 3,
                "form": "x=3", "quote": "Synthetic source", "source_sha256": "a" * 64,
                "source_state": "CURRENT", "provenance_kind": "SOURCE_STATES",
                "transcribed_by": {"role": "A", "model": "transcriber"}}
    store.save("equations", [equation])
    verdict = {"object_id": "EQ_T", "source_sha256": "a" * 64, "refuted": False,
               "confidence": "high", "reason": "Original current review",
               "reviewer": {"role": "F", "model": "independent-reviewer"},
               "content_sha256": verification_freshness.content_sha256(equation)}
    assert harvest_extraction.apply_verification([verdict])["confirmed"] == 1
    return verdict


@pytest.mark.parametrize("refusal", [
    "content_mismatch", "self_verification_refused", "unsigned_refused", "worker_role_refused",
    "stale", "content_hash_required", "unknown_ids",
])
def test_s3_refused_verdict_cannot_pass_using_existing_confirmation(verified_packet, refusal):
    incoming = {**confirmed_equation(), "reason": "New reviewed bytes"}
    if refusal == "content_mismatch":
        incoming["content_sha256"] = "0" * 64
    elif refusal == "self_verification_refused":
        incoming["reviewer"] = {"role": "A", "model": "transcriber"}
    elif refusal == "unsigned_refused":
        incoming.pop("reviewer")
    elif refusal == "worker_role_refused":
        incoming["reviewer"] = {"role": "S3", "model": "worker"}
    elif refusal == "stale":
        incoming["source_sha256"] = "old-source"
    elif refusal == "content_hash_required":
        incoming.pop("content_sha256")
        equation, = store.load("equations")
        equation["description"] = "Content changed after the original verdict"
        store.save("equations", [equation])
    else:
        incoming["object_id"] = "EQ_MISSING"
    verdict_packet(verified_packet, [incoming])
    result = ap.s3_integrate("P", "integrator")
    assert result["state"] == "BLOCKED"
    assert result["bank_mismatches"] == [{"file": "reviewed.json", "verification_refusals": {refusal: 1}}]
    assert pq.load()["packets"]["P"]["state"] == "BLOCKED"
    assert store.load("equations")[0]["verification"] == "CONFIRMED"
    log = json.loads((verified_packet / result["log"]).read_text())
    if refusal != "unknown_ids":
        assert log["verdict_state_checks"][0]["match"] is True  # enum-only gating would have passed


@pytest.mark.parametrize("retired", [False, True])
def test_s3_accepts_noop_and_retired_history_verdict_replays(verified_packet, retired):
    incoming = confirmed_equation()
    if retired:
        equation, = store.load("equations")
        equation.update(active=False, source_state="SUPERSEDED_EXTRACTION", verification="REFUTED")
        store.save("equations", [equation])
    verdict_packet(verified_packet, [incoming])
    before = store.load("equations")
    result = ap.s3_integrate("P", "integrator")
    assert result["state"] == "GRAPH_INTEGRATED"
    assert store.load("equations") == before
    log = json.loads((verified_packet / result["log"]).read_text())
    assert not log["bank_mismatches"] and not log["verdict_state_mismatches"]
    assert log["verdict_state_checks"][0]["retired_history_no_change"] is retired


@pytest.mark.parametrize("refusal", ["content_mismatch", "self_verification_refused"])
def test_s3_test_replay_blocks_refused_verdicts(verified_packet, monkeypatch, refusal):
    incoming = {**confirmed_equation(), "reason": "Changed reviewed bytes"}
    if refusal == "content_mismatch":
        incoming["content_sha256"] = "0" * 64
    else:
        incoming["reviewer"] = {"role": "A", "model": "transcriber"}
    verdict_packet(verified_packet, [incoming])
    queue = pq.load()
    queue["packets"]["P"]["state"] = "GRAPH_INTEGRATED"
    pq._save(queue)
    monkeypatch.setattr(ap.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="1 passed\n", stderr=""))
    result = ap.s3_test("P", "integrator")
    assert result["state"] == "BLOCKED"
    assert result["replay"][0]["stats"][refusal] == 1
    assert result["replay"][0]["idempotent"] is False
    assert pq.load()["packets"]["P"]["state"] == "BLOCKED"


@pytest.mark.parametrize("historical_known", [False, True])
def test_s3_parser_replay_checks_complete_current_or_historical_assertion(verified_packet, monkeypatch, historical_known):
    candidate = {"id": "CAND_T", "candidate_kind": "formula", "document_id": "DOC_T", "pages": [3],
                 "source_sha256": "a" * 64, "source_state": "CURRENT", "active": True}
    store.save("enrichments", [candidate])
    item = {"candidate_id": "CAND_T", "state": "SOURCE_PAGE_COMPARED", "reason": "original QA",
            "evidence": {"pdf_page": 3}}
    payload = {"reviewer": {"role": "F", "model": "reviewer"}, "parser_reviews": [item]}
    verdict_packet(verified_packet, payload)
    queue = pq.load()
    reviewed_file = queue["packets"]["P"]["evidence"]["review"]["files"][0]
    original_evidence = dict(item["evidence"], reviewer=payload["reviewer"], review_file=reviewed_file["path"],
                             review_file_sha256=reviewed_file["sha256"])
    if historical_known:
        enrichment.record_review("CAND_T", state=item["state"], reason=item["reason"], evidence=original_evidence)
    enrichment.record_review("CAND_T", state="REJECTED" if historical_known else "SOURCE_PAGE_COMPARED",
                             reason="a different latest QA assertion", evidence={"pdf_page": 3, "version": 2})
    queue["packets"]["P"]["state"] = "GRAPH_INTEGRATED"
    pq._save(queue)
    before = store.load("enrichments")
    monkeypatch.setattr(ap.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, stdout="1 passed\n", stderr=""))
    result = ap.s3_test("P", "integrator")
    assert result["state"] == ("TESTED" if historical_known else "BLOCKED")
    assert store.load("enrichments") == before
