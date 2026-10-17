"""Parser candidates must not acquire scientific authority or stale links."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from kg import enrichment, store


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(enrichment, "REPO_ROOT", tmp_path)
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"fixture PDF identity")
    source_hash = enrichment.sha256(pdf)
    store.save("manifest", [{"id": "DOC_A", "sha256": source_hash}])
    raw = tmp_path / "raw.json"
    payload = {"texts": [{"label": "formula", "self_ref": "#/texts/0",
        "text": "P_{out}/(\\delta\\eta)", "orig": "damaged", "prov": [
            {"page_no": 7, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}, "charspan": [0, 3]}]},
        {"label": "text", "text": "prose"},
        {"label": "formula", "text": "", "prov": []}],
        "tables": [{"label": "table", "data": {"table_cells": []}, "prov": [{"page_no": 7}]}]}
    raw.write_text(json.dumps(payload))
    meta = {"document_id": "DOC_A", "source_sha256": source_hash, "source_pdf": "source.pdf",
        "page_range": [7, 7], "options": {"do_formula_enrichment": True},
        "versions": {"docling": "fixture"}, "raw_sha256": enrichment.sha256(raw),
        "raw_artifact": "raw.json", "conversion_status": "success"}
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps(meta))
    return pdf, raw, metadata


def test_candidates_preserve_formula_boxes_and_failures_without_support(corpus):
    _, raw, meta = corpus
    assert enrichment.integrate(raw, meta)["candidates"] == 3
    nodes, edges = enrichment.graph_records()
    assert len(nodes) == 4
    assert {e["type"] for e in edges} == {"DERIVED_FROM"}
    assert store.load("equations") == []
    formula = next(n for n in nodes if n.get("content") == "P_{out}/(\\delta\\eta)")
    assert formula["pages"] == [7]
    assert formula["page_provenance"][0]["bbox"]["l"] == 1
    assert formula["provenance_kind"] == "SYSTEM_INFERS"
    empty = next(n for n in nodes if n.get("transcription_state") == "EMPTY_TRANSCRIPTION")
    assert empty["location_state"] == "UNRESOLVED_PAGE"


def test_replay_does_not_duplicate(corpus):
    _, raw, meta = corpus
    a = enrichment.integrate(raw, meta)
    b = enrichment.integrate(raw, meta)
    assert a == b
    assert len(store.load("enrichments")) == 4


@pytest.mark.parametrize("change", ["pdf", "raw", "manifest"])
def test_changed_identity_refuses_integration(corpus, change):
    pdf, raw, meta = corpus
    if change == "manifest":
        store.save("manifest", [{"id": "DOC_A", "sha256": "other"}])
    else:
        (pdf if change == "pdf" else raw).write_bytes(b"changed")
    with pytest.raises(ValueError):
        enrichment.integrate(raw, meta)
    assert store.load("enrichments") == []


@pytest.mark.parametrize("change", ["pdf", "raw"])
def test_freshness_preserves_history_but_removes_active_links(corpus, change):
    pdf, raw, meta = corpus
    enrichment.integrate(raw, meta)
    (pdf if change == "pdf" else raw).unlink()
    assert enrichment.refresh()["inactive"] == 4
    nodes, edges = enrichment.graph_records()
    assert len(nodes) == 4
    assert edges == []
    assert enrichment.find()["total"] == 0


def test_does_not_guess_page_offsets(corpus):
    _, raw, meta = corpus
    payload = json.loads(raw.read_text())
    metadata = json.loads(meta.read_text())
    payload["texts"][0]["prov"][0]["page_no"] = 1
    rows = enrichment.candidates(payload, metadata)
    assert rows[1]["pages"] == [1]
    assert rows[1]["location_state"] == "UNRESOLVED_PAGE"


def test_rejected_parser_run_is_history_not_active_evidence(corpus):
    _, raw, meta = corpus
    run = enrichment.integrate(raw, meta)["run_id"]
    enrichment.record_review(run, state="REJECTED", reason="Crop includes unrelated text", evidence={"page": 7})
    nodes, edges = enrichment.graph_records()
    assert len(nodes) == 4
    assert not edges
    assert enrichment.find()["total"] == 0
    enrichment.integrate(raw, meta)
    assert enrichment.find()["total"] == 0
    assert all(r["review_state"] == "REJECTED" for r in store.load("enrichments"))


@pytest.mark.parametrize("declared_pages", [[1], [7]])
def test_processing_ledger_never_credits_unrequested_or_metadata_only_pages(corpus, declared_pages):
    _, raw, meta = corpus
    metadata = json.loads(meta.read_text())
    store.save("manifest", [{"id": "DOC_A", "sha256": metadata["source_sha256"],
                             "page_count": 7, "source_pdf": "source.pdf"}])
    payload = json.loads(raw.read_text())
    payload["pages"] = {"1": {}}
    payload["texts"][0]["prov"][0]["page_no"] = 1
    raw.write_text(json.dumps(payload))
    metadata.update(observed_pages=declared_pages, raw_sha256=enrichment.sha256(raw))
    meta.write_text(json.dumps(metadata))
    enrichment.integrate(raw, meta)
    run = next(r for r in store.load("enrichments") if r["type"] == "GeneratedDocument")
    assert run["observed_pages"] == [1]
    assert run["out_of_range_pages"] == [1]
    assert enrichment.queue()["documents"][0]["processed_pages"] == []


def test_legacy_artifacts_keep_explicitly_unavailable_metadata_integrity(corpus):
    _, raw, metadata = corpus
    enrichment.integrate(raw, metadata)
    assert all(r["metadata_integrity_state"] == "UNAVAILABLE" for r in store.load("enrichments"))
    assert all("Legacy" in r["metadata_integrity_scope"] for r in store.load("enrichments"))
    # Missing legacy metadata cannot retroactively acquire a v2 binding. Keep
    # the raw/source history visible with that limitation explicitly recorded.
    metadata.unlink()
    assert enrichment.refresh()["inactive"] == 0
    assert all(r["metadata_integrity_state"] == "UNAVAILABLE" for r in store.load("enrichments"))


@pytest.mark.parametrize("region_mode", [False, True])
def test_rejected_region_crop_does_not_block_unattempted_full_page_processing(corpus, region_mode):
    from tools.docling_batch import processing_plan

    _, raw, metadata_path = corpus
    meta = json.loads(metadata_path.read_text())
    store.save("manifest", [{"id": "DOC_A", "sha256": meta["source_sha256"],
                             "page_count": 7, "source_pdf": "source.pdf"}])
    if region_mode:
        meta["options"] = {"mode": "visually_located_formula_regions"}
        metadata_path.write_text(json.dumps(meta))
    run = enrichment.integrate(raw, metadata_path)["run_id"]
    enrichment.record_review(run, state="REJECTED", reason="Synthetic transcription failure",
                             evidence={"source_page": 7})
    result = processing_plan(enrichment.queue(), [], store.load("enrichments"))
    document, = result["documents"]
    assert document["processed_pages"] == []
    assert document["remaining_pages"] == list(range(1, 8))
    assert document["attempted_unfinished_pages"] == []
    assert document["quality_review_pages"] == ([] if region_mode else [7])
    assert document["region_quality_review_pages"] == ([7] if region_mode else [])
    assert document["unattempted_pages"] == list(range(1, 8 if region_mode else 7))


@pytest.mark.parametrize("region_mode", [False, True])
def test_rejected_backlogs_exclude_stale_or_other_source_fingerprints(corpus, region_mode):
    _, _, metadata_path = corpus
    meta = json.loads(metadata_path.read_text())
    store.save("manifest", [{"id": "DOC_A", "sha256": meta["source_sha256"],
                             "page_count": 7, "source_pdf": "source.pdf"}])
    record = {"type": "GeneratedDocument", "document_id": "DOC_A", "source_sha256": meta["source_sha256"],
              "source_state": "CURRENT", "review_state": "REJECTED", "active": False,
              "page_range": [7, 7], "observed_pages": [7],
              "options": {"mode": "visually_located_formula_regions"} if region_mode else {}}
    store.save("enrichments", [{**record, "id": "ENR_STALE", "source_state": "STALE_SOURCE"},
                               {**record, "id": "ENR_OTHER_SOURCE", "source_sha256": "old-source"}])
    document, = enrichment.queue()["documents"]
    assert document["processed_pages"] == []
    assert document["quality_review_pages"] == document["region_quality_review_pages"] == []
