"""Immutable Docling attempt and bounded scheduling tests; no model inference."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import enrichment, store
from tools import docling_batch as batch, docling_enrich as enrich


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(enrichment, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(batch, "refresh", lambda: {})
    python = tmp_path / ".venv-docling/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic source identity, never parsed by Docling")
    sha = enrichment.sha256(source)
    store.save("manifest", [{"id": "DOC_TEST", "source_pdf": "source.pdf", "sha256": sha, "page_count": 3}])
    return tmp_path, {
        "document_id": "DOC_TEST", "source_pdf": "source.pdf", "source_sha256": sha,
        "page_range": [1, 1], "options": {"mode": "visually_located_formula_regions", "regions": [{"page": 1}]},
        "versions": {"docling": "synthetic"}, "model_revisions": {"model/refs/main": "revision-a"},
        "observed_pages": [1], "conversion_status": "success", "elapsed_seconds": 0.01,
    }


def publish(root, meta, *, raw=None):
    payload = raw or json.dumps({"pages": {"1": {}}, "texts": [
        {"label": "formula", "text": "x = y", "prov": [{"page_no": 1}]}]}).encode()
    return enrich.publish_attempt(payload, b"synthetic candidate x = y\n", meta,
                                  dest=root / "artifacts", root=root)


def test_content_address_includes_source_options_packages_revisions_and_outputs(isolated):
    root, meta = isolated
    meta.update(attempt_id="fixed-attempt", attempt_created_at="synthetic-time")
    first_raw, first_meta = publish(root, meta)
    original = {p: p.read_bytes() for p in (first_raw, first_raw.with_suffix(".md"), first_meta)}
    changed = [dict(meta, source_sha256="different-source"),
               dict(meta, options={**meta["options"], "scale": 4}),
               dict(meta, versions={"docling": "different-package"}),
               dict(meta, model_revisions={"model/refs/main": "revision-b"})]
    paths = {first_raw}
    for metadata in changed:
        paths.add(publish(root, metadata)[0])
    paths.add(publish(root, meta, raw=b'{"pages": {"1": {}}, "texts": []}')[0])
    assert len(paths) == 6
    assert {p: p.read_bytes() for p in original} == original
    assert publish(root, meta) == (first_raw, first_meta)
    published = json.loads(first_meta.read_text())
    assert published["model_revisions"] == meta["model_revisions"]
    assert published["attempt_id"] == "fixed-attempt"
    assert published["artifact_identity_sha256"] in first_raw.name


def test_explicit_new_attempt_does_not_overwrite_or_inherit_rejected_review(isolated):
    root, meta = isolated
    raw, metadata = publish(root, meta)
    first = enrichment.integrate(raw, metadata)
    enrichment.record_review(first["run_id"], state="REJECTED", reason="Synthetic poor transcription",
                             evidence={"source_page": 1})
    old_bytes = raw.read_bytes(), metadata.read_bytes()
    cache = enrich.cached_attempt(root / "artifacts", meta, root=root)
    assert cache["rejected"]
    second_raw, second_meta = publish(root, meta)  # distinct generated attempt_id
    second = enrichment.integrate(second_raw, second_meta)
    assert second_raw != raw and second["run_id"] != first["run_id"]
    assert old_bytes == (raw.read_bytes(), metadata.read_bytes())
    runs = {r["id"]: r for r in store.load("enrichments") if r["type"] == "GeneratedDocument"}
    assert runs[first["run_id"]]["review_state"] == "REJECTED"
    assert runs[second["run_id"]]["review_state"] == "UNREVIEWED"
    assert runs[second["run_id"]]["attempt_id"] != runs[first["run_id"]]["attempt_id"]


def test_changed_revision_and_damaged_artifacts_cannot_be_reused_or_overwritten(isolated):
    root, meta = isolated
    meta.update(attempt_id="fixed", attempt_created_at="synthetic-time")
    raw, metadata = publish(root, meta)
    assert enrich.cached_attempt(root / "artifacts", meta, root=root)
    different = {**meta, "model_revisions": {"model/refs/main": "revision-b"}}
    assert enrich.cached_attempt(root / "artifacts", different, root=root) is None
    raw.write_bytes(b"damaged previous artifact")
    assert enrich.cached_attempt(root / "artifacts", meta, root=root) is None
    before = metadata.read_bytes()
    with pytest.raises(FileExistsError, match="Immutable"):
        publish(root, meta)
    assert raw.read_bytes() == b"damaged previous artifact" and metadata.read_bytes() == before


@pytest.mark.parametrize("change", ["options", "model_revisions", "elapsed_seconds", "legacy_downgrade", "format_only"])
def test_v2_metadata_changes_refuse_cache_integration_and_invalidate_reviewed_rows(isolated, change):
    root, meta = isolated
    raw, metadata = publish(root, meta)
    integrated = enrichment.integrate(raw, metadata)
    enrichment.record_review(integrated["run_id"], state="SOURCE_PAGE_COMPARED", reason="Synthetic page comparison",
                             evidence={"source_page": 1})
    changed = json.loads(metadata.read_text())
    if change == "options":
        changed["options"]["scale"] = 9
    elif change == "model_revisions":
        changed["model_revisions"]["model/refs/main"] = "edited-revision"
    elif change == "elapsed_seconds":
        changed["elapsed_seconds"] += 1
    elif change == "legacy_downgrade":
        for key in ("artifact_schema_version", "artifact_identity_sha256", "input_identity_sha256"):
            changed.pop(key)
    # Formatting alone changes immutable file bytes while retaining logical binding.
    metadata.write_text(json.dumps(changed))
    descriptor = {key: changed[key] for key in enrich.input_identity(meta)}
    assert enrich.cached_attempt(root / "artifacts", descriptor, root=root) is None
    with pytest.raises(ValueError, match="v2"):
        enrichment.integrate(raw, metadata)
    assert enrichment.refresh()["inactive"] == 2
    rows = store.load("enrichments")
    assert all(r["source_state"] == "STALE_SOURCE" and r["metadata_integrity_state"] == "FAILED" for r in rows)
    assert all(r["review_state"] == "SOURCE_PAGE_COMPARED" for r in rows)  # history retained
    assert enrichment.graph_records()[1] == []


@pytest.mark.parametrize("region_mode", [False, True])
def test_deleted_v2_metadata_invalidates_run_and_candidates(isolated, region_mode):
    root, meta = isolated
    if not region_mode:
        meta["options"] = {"do_formula_enrichment": True}
    raw, metadata = publish(root, meta)
    enrichment.integrate(raw, metadata)
    if not region_mode:
        assert enrichment.queue()["documents"][0]["processed_pages"] == [1]
    metadata.unlink()
    assert enrich.cached_attempt(root / "artifacts", meta, root=root) is None
    assert enrichment.refresh()["inactive"] == 2
    assert all(r["metadata_integrity_state"] == "FAILED" for r in store.load("enrichments"))
    assert enrichment.queue()["documents"][0]["processed_pages"] == []


def test_valid_v2_artifact_from_older_worker_gets_binding_without_new_ids(isolated):
    _, meta = isolated
    raw, metadata = publish(isolated[0], meta)
    enrichment.integrate(raw, metadata)
    rows = store.load("enrichments")
    original_ids = {r["id"] for r in rows}
    # Workers already running before the integrity fix did not store these
    # observation fields, but their publisher emitted the same valid v2 bundle.
    for row in rows:
        for key in ("metadata_schema_version", "metadata_sha256", "metadata_integrity_state", "metadata_integrity_scope"):
            row.pop(key, None)
        if row["type"] != "GeneratedDocument":
            row.pop("metadata_artifact", None)
    store.save("enrichments", rows)
    assert enrichment.refresh()["inactive"] == 0
    after = store.load("enrichments")
    assert {r["id"] for r in after} == original_ids
    assert all(r["metadata_schema_version"] == 2 and r["metadata_integrity_state"] == "VERIFIED" for r in after)
    assert all(r["metadata_sha256"] == enrichment.sha256(metadata) for r in after)
    before = store.store_path("enrichments").read_bytes()
    enrichment.refresh()
    assert store.store_path("enrichments").read_bytes() == before


def plan(*, quality=(), processed=()):
    done = set(processed)
    return {"documents": [{"document_id": "DOC_TEST", "source_sha256": "sha", "page_count": 5,
                            "processed_pages": sorted(done), "remaining_pages": sorted(set(range(1, 6)) - done),
                            "quality_review_pages": list(quality)}],
            "summary": {"processed_pages": len(done)}}


def test_unattempted_work_excludes_rejected_requests_and_failed_attempt_backlogs():
    jobs = [{"document_id": "DOC_TEST", "source_sha256": "sha", "pages": [2, 2], "exit_code": 124},
            {"document_id": "DOC_TEST", "source_sha256": "old-source", "pages": [3, 5]}]
    runs = [{"type": "GeneratedDocument", "document_id": "DOC_TEST", "source_sha256": "sha",
             "page_range": [1, 1], "observed_pages": [], "review_state": "REJECTED"},
            {"type": "GeneratedDocument", "document_id": "DOC_TEST", "source_sha256": "sha",
             "page_range": [4, 4], "options": {"mode": "visually_located_formula_regions"}}]
    result = batch.processing_plan(plan(), jobs, runs)
    row, = result["documents"]
    assert row["quality_review_pages"] == [1]
    assert row["attempted_unfinished_pages"] == [2]
    assert row["unattempted_pages"] == [3, 4, 5]
    assert result["summary"]["processed_pages"] == 0


def test_successfully_reprocessed_page_is_not_outstanding_rejected_backlog():
    old = {"type": "GeneratedDocument", "document_id": "DOC_TEST", "source_sha256": "sha",
           "page_range": [1, 1], "review_state": "REJECTED"}
    result = batch.processing_plan(plan(processed=[1]), [], [old])
    row, = result["documents"]
    assert row["rejected_attempt_pages"] == [1]
    assert row["quality_review_pages"] == []
    assert row["unattempted_pages"] == [2, 3, 4, 5]
    assert result["summary"]["quality_review_pages"] == 0


def test_batch_skips_quality_backlog_stops_no_progress_and_resumes_next_page(isolated, monkeypatch):
    root, _ = isolated
    original = plan(quality=[1])
    after = copy.deepcopy(original)
    # Unrelated document progress must not disguise lack of progress in this job.
    other = {"document_id": "DOC_OTHER", "source_sha256": "other", "page_count": 1,
             "processed_pages": [1], "remaining_pages": [], "quality_review_pages": []}
    after["documents"].append(other)
    after["summary"]["processed_pages"] += 1
    calls = []
    queues = iter([original, after, after, after])
    monkeypatch.setattr(batch, "queue", lambda: copy.deepcopy(next(queues)))
    monkeypatch.setattr(batch.subprocess, "run", lambda command, **kw:
                        calls.append(command) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert batch.main(["--run", "--max-jobs", "5", "--pages-per-job", "1"]) == 2
    assert len(calls) == 1 and calls[0][calls[0].index("--pages") + 1] == "2:2"
    assert batch.main(["--run", "--max-jobs", "5", "--pages-per-job", "1"]) == 2
    assert len(calls) == 2 and calls[1][calls[1].index("--pages") + 1] == "3:3"
    saved = json.loads((root / "knowledge/reports/docling_queue.json").read_text())
    selected = next(d for d in saved["documents"] if d["document_id"] == "DOC_TEST")
    assert selected["quality_review_pages"] == [1]
    assert selected["attempted_unfinished_pages"] == [2, 3]


def test_explicit_reprocessing_banks_new_attempt_without_repeating_the_same_job(isolated, monkeypatch):
    _, _ = isolated
    static = plan(processed=[1])
    monkeypatch.setattr(batch, "queue", lambda: copy.deepcopy(static))
    calls = []
    old = {"id": "OLD", "type": "GeneratedDocument", "document_id": "DOC_TEST",
           "source_sha256": "sha", "page_range": [1, 1], "attempt_id": "old", "options": {}}
    store.save("enrichments", [old])
    def worker(command, **kwargs):
        calls.append(command)
        store.upsert("enrichments", [{**old, "id": "NEW", "attempt_id": "new"}])
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(batch.subprocess, "run", worker)
    assert batch.main(["--run", "--max-jobs", "5", "--reprocess", "--document-id", "DOC_TEST", "--pages", "1:1"]) == 0
    assert len(calls) == 1 and "--reprocess" in calls[0]
    logged = json.loads((batch.ROOT / "knowledge/reports/docling_jobs.jsonl").read_text())
    assert logged["state"] == "DISTINCT_ATTEMPT_BANKED_REVIEW_REQUIRED"
    assert logged["new_attempt_ids"] == ["NEW"] and logged["processed_pages_gained"] == []
