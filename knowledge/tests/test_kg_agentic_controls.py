"""Concurrency, packet-state and role controls for multi-agent knowledge work.

Corpus-free: every test relocates the knowledge root into a temporary tree.
"""

import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import REPO_ROOT, locks, packet_queue as pq, store, work_journal  # noqa: E402
from tools import docling_batch as batch  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(store, "REPORT_DIR", root / "reports")
    monkeypatch.setattr(pq, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(work_journal, "REPO_ROOT", tmp_path)
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"synthetic source bytes")
    import hashlib
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    store.save("manifest", [{"id": "DOC_T", "source_pdf": "source.pdf", "sha256": sha, "page_count": 5}])
    store.save("enrichments", [{"id": "ENR_ok", "type": "GeneratedDocument", "document_id": "DOC_T",
                                "source_sha256": sha, "conversion_status": "success", "observed_pages": [2, 3]},
                               {"id": "ENR_other", "type": "GeneratedDocument", "document_id": "DOC_X",
                                "source_sha256": sha, "conversion_status": "success", "observed_pages": [2]}])
    return tmp_path, sha


def _hold(lock_root, name, ready, release):
    from kg import locks as child_locks, store as child_store
    child_store.KNOWLEDGE_ROOT = Path(lock_root)
    with child_locks.exclusive(name, timeout=5):
        ready.set()
        release.wait(10)


def test_writer_lock_is_reentrant_and_refuses_a_second_process(isolated):
    tmp_path, _ = isolated
    with locks.writer():
        with locks.writer():  # nested helpers in one process must not deadlock
            store.save("claims", [{"id": "A"}])
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold, args=(str(store.KNOWLEDGE_ROOT), "docling-heavy", ready, release))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(locks.LockBusyError):
            with locks.heavy_job():
                pass
    finally:
        release.set()
        child.join(10)
    with locks.heavy_job():  # released with the holder process; no stale lock remains
        pass


def test_save_uses_private_temporary_files(isolated, monkeypatch):
    names = []
    original = store.tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        handle = original(*args, **kwargs)
        names.append(handle.name)
        return handle
    monkeypatch.setattr(store.tempfile, "NamedTemporaryFile", recording)
    store.save("claims", [{"id": "A"}])
    store.save("claims", [{"id": "B"}])
    assert len(set(names)) == 2 and all(".claims.json." in Path(n).name for n in names)
    assert not list(store.KNOWLEDGE_ROOT.glob("*.tmp")) and not list(store.KNOWLEDGE_ROOT.glob(".*.tmp"))
    assert [r["id"] for r in store.load("claims")] == ["B"]


def _verdict(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(json.dumps(rows))
    return str(path)


def test_packet_state_machine_enforces_order_roles_and_reviewed_bytes(isolated):
    tmp_path, _ = isolated
    with pytest.raises(PermissionError):
        pq.add("P1", kind="verification", document_id="DOC_T", pages=(2, 3), targets=["EQ_1"],
               priority=1, reason="x", role="S1", model="m")
    pq.add("P1", kind="verification", document_id="DOC_T", pages=(2, 3), targets=["EQ_1"],
           priority=1, reason="pending objects", role="F", model="opus")
    with pytest.raises(pq.TransitionError):  # raw/candidate states never jump to integration
        pq.transition("P1", "GRAPH_INTEGRATED", role="S3", model="sonnet")
    with pytest.raises(PermissionError):
        pq.transition("P1", "DOCLING_RUNNING", role="S2", model="sonnet")
    pq.transition("P1", "DOCLING_RUNNING", role="S1", model="sonnet")
    with pytest.raises(pq.TransitionError):  # a run of a different document is not this packet's evidence
        pq.transition("P1", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_other"])
    pq.transition("P1", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_ok"])
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    pq.transition("P1", "CANDIDATES_PREPARED", role="S2", model="sonnet", evidence_paths=[str(bundle)])
    with pytest.raises(PermissionError):  # mechanical roles cannot open or decide a source review
        pq.transition("P1", "OPUS_SOURCE_REVIEW", role="S2", model="sonnet")
    pq.transition("P1", "OPUS_SOURCE_REVIEW", role="F", model="opus")
    unsigned = _verdict(tmp_path, "unsigned.json", [{"object_id": "EQ_1", "refuted": False, "confidence": "high"}])
    with pytest.raises(pq.TransitionError):
        pq.transition("P1", "VERIFIED", role="F", model="opus", verdict_paths=[unsigned])
    refuted = _verdict(tmp_path, "refuted.json", [{"object_id": "EQ_1", "refuted": True, "confidence": "high",
                                                   "reviewer": {"role": "F", "model": "opus"}}])
    with pytest.raises(pq.TransitionError):  # zero confirmed objects cannot be labeled verified
        pq.transition("P1", "VERIFIED", role="F", model="opus", verdict_paths=[refuted])
    signed = _verdict(tmp_path, "signed.json", [{"object_id": "EQ_1", "refuted": False, "confidence": "high",
                                                 "reviewer": {"role": "F", "model": "opus"}}])
    pq.transition("P1", "VERIFIED", role="F", model="opus", verdict_paths=[signed])
    Path(signed).write_text(Path(signed).read_text() + " ")  # reviewed bytes changed afterwards
    log = tmp_path / "integration.log"
    log.write_text("banked")
    with pytest.raises(pq.TransitionError):
        pq.transition("P1", "GRAPH_INTEGRATED", role="S3", model="sonnet", evidence_paths=[str(log)])
    Path(signed).write_text(Path(signed).read_text()[:-1])
    pq.transition("P1", "GRAPH_INTEGRATED", role="S3", model="sonnet", evidence_paths=[str(log)])
    with pytest.raises(pq.TransitionError):
        pq.transition("P1", "TESTED", role="S3", model="sonnet", evidence_paths=[str(log)], test_result="fail")
    pq.transition("P1", "TESTED", role="S3", model="sonnet", evidence_paths=[str(log)], test_result="pass")
    pq.transition("P1", "FINAL_AUDIT_ELIGIBLE", role="F", model="opus")
    packet = pq.load()["packets"]["P1"]
    assert [h["to"] for h in packet["history"]] == ["QUEUED", "DOCLING_RUNNING", "RAW_CAPTURED",
        "CANDIDATES_PREPARED", "OPUS_SOURCE_REVIEW", "VERIFIED", "GRAPH_INTEGRATED", "TESTED",
        "FINAL_AUDIT_ELIGIBLE"]
    assert "unobserved_pages" not in packet["history"][2]
    assert pq.summary()["reviewed_object_outcomes"]["confirmed"] == 1
    journal = work_journal.read()
    assert {row["role"] for row in journal} == {"F", "S1", "S2", "S3"}
    assert all(row["model"] for row in journal)


def test_interrupted_docling_job_returns_to_queue_with_attempt_history(isolated):
    pq.add("P2", kind="extraction", document_id="DOC_T", pages=(1, 1), targets=[], priority=0,
           reason="new source window", role="F", model="opus")
    pq.transition("P2", "DOCLING_RUNNING", role="S1", model="sonnet")
    with pytest.raises(pq.TransitionError):
        pq.transition("P2", "QUEUED", role="S1", model="sonnet")  # an interruption needs a reason
    pq.transition("P2", "QUEUED", role="S1", model="sonnet", note="WORKER_TIMEOUT after 900 s")
    pq.transition("P2", "DOCLING_RUNNING", role="S1", model="sonnet")
    packet = pq.load()["packets"]["P2"]
    assert packet["attempts"] == 2 and packet["history"][2]["note"].startswith("WORKER_TIMEOUT")


def test_changed_source_blocks_further_progress(isolated):
    tmp_path, _ = isolated
    pq.add("P3", kind="verification", document_id="DOC_T", pages=(1, 2), targets=[], priority=0,
           reason="x", role="F", model="opus")
    (tmp_path / "source.pdf").write_bytes(b"replaced")
    manifest = store.load("manifest")
    manifest[0]["sha256"] = "0" * 64
    store.save("manifest", manifest)
    with pytest.raises(pq.TransitionError):
        pq.transition("P3", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P3", "BLOCKED", role="F", model="opus", note="source fingerprint changed")


def test_worker_roles_cannot_journal_scientific_acceptance(isolated):
    with pytest.raises(PermissionError):
        work_journal.append(role="S2", model="sonnet", task="structure", state="ACCEPTED_SCIENCE")
    entry = work_journal.append(role="F", model="opus", task="review", state="ACCEPTED_SCIENCE")
    assert entry["seq"] == 1 and entry["role"] == "F"


def test_busy_docling_refusal_is_not_attempted_page_history():
    plan = {"documents": [{"document_id": "D", "source_sha256": "s", "page_count": 3,
                           "remaining_pages": [1, 2, 3], "quality_review_pages": [], "processed_pages": []}],
            "summary": {}}
    busy = [{"document_id": "D", "source_sha256": "s", "pages": [1, 2], "state": "DOCLING_BUSY", "exit_code": 75}]
    out = batch.processing_plan(plan, busy)
    assert out["documents"][0]["unattempted_pages"] == [1, 2, 3]
    timeout = [{"document_id": "D", "source_sha256": "s", "pages": [1, 2], "state": "WORKER_TIMEOUT", "exit_code": 124}]
    out = batch.processing_plan(plan, timeout)
    assert out["documents"][0]["attempted_unfinished_pages"] == [1, 2]


def test_only_the_coordinator_reprioritizes_and_history_is_kept(isolated):
    pq.add("P4", kind="extraction", document_id="DOC_T", pages=(1, 1), targets=[], priority=1,
           reason="x", role="F", model="opus")
    with pytest.raises(PermissionError):
        pq.reprioritize("P4", 9, role="S1", model="sonnet", note="jump the queue")
    packet = pq.reprioritize("P4", 9, role="F", model="opus", note="user priority P1")
    assert packet["priority"] == 9 and packet["state"] == "QUEUED"
    assert packet["history"][-1]["priority_from"] == 1
