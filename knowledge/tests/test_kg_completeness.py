"""Corpus-free regressions for honest extraction coverage and ingestion states."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import coverage, graph, ingest, store, validate
from tools import sync_corpus


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    agents = tmp_path / "propulsion_texts" / "propulsion_texts_for_agents"
    agents.mkdir(parents=True)
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", knowledge)
    monkeypatch.setattr(store, "CHUNK_DB", knowledge / "index" / "chunks.sqlite")
    monkeypatch.setattr(ingest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate, "AGENT_CORPUS_ROOT", agents)
    monkeypatch.setattr(validate, "CORPUS_ROOT", agents.parent)
    monkeypatch.setattr(validate, "VECTOR_DIR", knowledge / "index")
    monkeypatch.setattr(validate, "REPORT_DIR", knowledge / "reports")
    monkeypatch.setattr(graph, "GRAPH_DIR", knowledge / "graph")
    return tmp_path


def document(root, *, pages=1, text=None):
    (root / "paper.md").write_text(text if text is not None else
        "<!-- PDF page 1 -->\n\n# Synthetic source\n\n"
        "This synthetic test passage is source content rather than conversion metadata.\n")
    return {"id": "DOC_TEST", "sha256": "source", "source_pdf": "paper.pdf", "filename": "paper.pdf",
            "page_count": pages, "markdown_exists": True, "generated_markdown": "paper.md",
            "conversion": {"status": "good"}}


def test_zero_page_source_cannot_be_ingested_even_with_nonempty_derivative(isolated):
    doc = document(isolated, pages=0)
    store.save("manifest", [doc])
    store.save("equations", [{"id": "EQ_HISTORY", "document_id": doc["id"]}])
    for _ in range(2):
        result = ingest.ingest_chunks(verbose=False)
        assert result["ingested"] == result["unchanged"] == result["removed"] == 0
        state, = result["states"]
        assert state["state"] == "SOURCE_UNREADABLE"
        assert state["reason"] == "EMPTY_SOURCE"
        assert state["n_chunks"] == 0
        assert "Restore" in state["remediation"]
    conn = store.connect()
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert conn.execute("SELECT n_chunks FROM doc_state WHERE doc_id = 'DOC_TEST'").fetchone()[0] == 0
    conn.close()
    assert store.load("manifest") == [doc]
    assert store.load("equations")[0]["id"] == "EQ_HISTORY"


def test_empty_extraction_remains_explicit_on_incremental_ingest(isolated):
    store.save("manifest", [document(isolated, text="<!-- PDF page 1 -->\n\n")])
    for _ in range(2):
        result = ingest.ingest_chunks(verbose=False)
        assert result["ingested"] == result["unchanged"] == 0
        state, = result["states"]
        assert state["state"] == "SOURCE_NOT_INGESTED"
        assert state["reason"] == "EMPTY_EXTRACTED_TEXT"
        assert state["n_chunks"] == 0


def test_zero_page_replacement_retires_old_chunks_but_preserves_source_identity(isolated):
    doc = document(isolated)
    store.save("manifest", [doc])
    assert ingest.ingest_chunks(verbose=False)["n_chunks_total"] > 0
    doc.update(page_count=0, sha256="empty-replacement")
    store.save("manifest", [doc])
    result = ingest.ingest_chunks(verbose=False)
    assert result["states"][0]["state"] == "SOURCE_UNREADABLE"
    conn = store.connect()
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    state, = conn.execute("SELECT * FROM doc_state").fetchall()
    assert state["doc_id"] == "DOC_TEST" and state["source_sha256"] == "empty-replacement"
    conn.close()


def test_status_counts_actual_chunks_separately_from_doc_state(isolated, monkeypatch):
    doc = document(isolated)
    store.save("manifest", [doc])
    ingest.ingest_chunks(verbose=False)
    conn = store.connect()
    store.replace_doc_chunks(conn, "DOC_EMPTY", [], {
        "doc_id": "DOC_EMPTY", "source_sha256": "empty", "markdown_sha256": "metadata",
        "extraction_version": "test", "n_chunks": 0, "ingested_at": "test"})
    conn.close()
    agents = validate.AGENT_CORPUS_ROOT
    (agents / "_conversion_manifest.json").write_text(json.dumps([
        {"source_kind": "file", "sha256": "source"},
        {"source_kind": "zip_member", "sha256": "archive"}]))
    monkeypatch.setattr(sync_corpus, "scan", lambda: {
        "pdf_count": 9, "manifest_rows": 2, "missing_markdown": [],
        "stale_markdown": [], "orphan_markdown": []})
    store.save("models", [{"id": "MODEL_LEGACY", "source_document_id": doc["id"],
                           "source_support_state": "SUPPORTED"}])
    equation = {"id": "EQ_CURRENT", "document_id": doc["id"], "source_sha256": "source",
                "provenance_kind": "SOURCE_STATES", "verification": "NOT_CHECKED"}
    store.save("equations", [equation, {**equation, "id": "EQ_RETIRED", "active": False,
                                        "source_state": "SUPERSEDED_EXTRACTION"}])
    result = validate.status()
    assert result["documents"]["doc_state_entries"] == 2
    assert result["documents"]["chunked"] == 1
    assert result["documents"]["zero_chunk_documents"] == ["DOC_EMPTY"]
    assert result["corpus"]["pdfs_on_disk"] == 9
    assert result["corpus"]["tracked_file_rows"] == 1
    assert result["corpus"]["tracked_archive_members"] == 1
    assert result["extraction"]["equation_completeness"] == "NOT_ESTABLISHED"
    assert result["extraction"]["model_support"] == {"NOT_CHECKED": 1}
    assert result["extraction"]["declared_model_support"] == {"SUPPORTED": 1}
    assert result["extraction"]["models_with_current_source_review"] == 0
    assert result["extraction"]["unverified_equations_and_claims"] == 1
    assert result["extraction"]["unverified_equations_and_claims_including_history"] == 2


def test_coverage_separates_current_source_records_from_registry_and_verification():
    doc = {"id": "DOC_TEST", "sha256": "current", "conversion": {"status": "good"}}
    eq = {"id": "EQ_TEST", "document_id": doc["id"], "provenance_kind": "SOURCE_STATES",
          "source_sha256": "current", "verification": "CONFIRMED", "verifier_source_sha256": "current"}
    result = coverage.summary({
        "manifest": [doc, {"id": "DOC_EMPTY", "sha256": "empty"},
                     {"id": "DOC_ALT", "alternate_copy_of": "DOC_TEST"}],
        "equations": [eq, {**eq, "id": "EQ_STALE", "source_sha256": "old"},
                      {**eq, "id": "EQ_REFUTED", "verification": "REFUTED"}],
        "claims": [{**eq, "id": "CLM_NO_FINGERPRINT", "verifier_source_sha256": ""}],
        "validity": [{"id": "VAL_REGISTRY", "source_document_id": doc["id"], "provenance_kind": "PROJECT_DERIVES"}],
        "models": [{"id": "MODEL_LEGACY", "source_document_id": doc["id"], "source_support_state": "SUPPORTED"}],
        "software_links": [{"id": "CONST_UNKNOWN", "type": "EmpiricalConstant", "basis_class": "UNKNOWN_BASIS"}],
    }, include_documents=True)
    assert result["documents"]["canonical"] == 2
    assert result["documents"]["without_source_objects"] == 1
    assert result["source_objects"]["equations"]["current"] == 1
    assert result["source_objects"]["equations"]["independently_confirmed"] == 1
    assert result["source_objects"]["claims"]["independently_confirmed"] == 0
    assert result["source_objects"]["validity"]["current"] == 0
    assert result["models"]["current_support"] == {"NOT_CHECKED": 1}
    assert result["software_constants"]["unknown_basis"] == 1
    assert result["gaps"]["equations_and_claims_pending_independent_verification"] == 1
    assert result["equation_completeness"] == "NOT_ESTABLISHED"
    assert all(r["equation_completeness"] == "NOT_ESTABLISHED" for r in result["by_document"])


def test_even_every_document_having_confirmed_math_does_not_prove_completeness():
    result = coverage.summary({
        "manifest": [{"id": "DOC_TEST", "sha256": "current", "conversion": {"status": "good"}}],
        "equations": [{"id": "EQ_TEST", "document_id": "DOC_TEST", "provenance_kind": "SOURCE_STATES",
                       "source_sha256": "current", "verification": "CONFIRMED", "verifier_source_sha256": "current"}],
    })
    assert result["documents"]["without_equations"] == 0
    assert result["equation_completeness"] == "NOT_ESTABLISHED"
    assert coverage.summary({})["equation_completeness"] == "NOT_ESTABLISHED"


def test_incomplete_formula_ids_follow_graph_guard_and_keep_recovered_forms():
    source = {"document_id": "DOC_TEST", "provenance_kind": "SOURCE_STATES", "source_sha256": "current"}
    result = coverage.summary({
        "manifest": [{"id": "DOC_TEST", "sha256": "current"}],
        "equations": [{**source, "id": "EQ_MISSING", "form": "NOT RENDERED IN MARKDOWN"},
                      {**source, "id": "EQ_GARBLED", "form": "unreadable [OCR-garbled]"},
                      {**source, "id": "EQ_RECOVERED", "form": "p = rho R T",
                       "extraction_note": "Not rendered in old Markdown; transcribed from original PDF."}],
    })
    incomplete = result["incomplete_formula_transcriptions"]
    assert incomplete["count"] == incomplete["current_count"] == 2
    assert incomplete["ids"] == ["EQ_GARBLED", "EQ_MISSING"]
    assert result["gaps"]["incomplete_formula_transcriptions"] == 2


def test_superseded_damage_is_history_not_current_transcription_gap():
    result = coverage.summary({
        "manifest": [{"id": "DOC_TEST", "sha256": "current"}],
        "equations": [{"id": "EQ_OLD", "form": "NOT RENDERED", "active": False,
                       "source_state": "SUPERSEDED_EXTRACTION", "document_id": "DOC_TEST",
                       "provenance_kind": "SOURCE_STATES", "source_sha256": "current"}],
    })
    damaged = result["incomplete_formula_transcriptions"]
    assert damaged["historical_count"] == 1
    assert damaged["current_count"] == 0
    assert damaged["current_state"] == "NONE_FLAGGED_CURRENT"
    assert result["gaps"]["incomplete_formula_transcriptions"] == 0
    assert result["equation_completeness"] == "NOT_ESTABLISHED"


def test_validation_exposes_empty_source_and_extraction_gaps(isolated):
    doc = document(isolated, pages=0)
    store.save("manifest", [doc, {**doc, "id": "DOC_READABLE", "page_count": 1}])
    store.save("equations", [{"id": "EQ_MISSING", "document_id": "DOC_READABLE", "source_sha256": "source",
                              "provenance_kind": "SOURCE_STATES", "grounded": "GROUNDED",
                              "form": "NOT RENDERED IN MARKDOWN"}])
    ingest.ingest_chunks(verbose=False)
    result = validate.run()
    issue = next(i for i in result["issues"] if i["check"] == "empty_source_pdf")
    assert issue["state"] == "SOURCE_UNREADABLE"
    assert issue["reason"] == "EMPTY_SOURCE"
    assert result["summary"]["stats"]["coverage"]["equation_completeness"] == "NOT_ESTABLISHED"
    assert any(i["check"] == "canonical_document_without_chunks" for i in result["issues"])
    assert any(i["check"] == "source_extraction_incomplete" and i["count"] == 1 for i in result["issues"])
    assert any(i["check"] == "incomplete_formula_transcriptions" and i["ids"] == ["EQ_MISSING"] for i in result["issues"])
