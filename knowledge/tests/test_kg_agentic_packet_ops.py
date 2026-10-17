"""Role operations on one packet: candidate preparation and reviewed-file integration.

Corpus-free. Rendering and graph assembly are stubbed; the tested contract is
that S2 bundles stay provisional and S3 integrates only the exact reviewed bytes.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import graph, harvest_extraction, packet_queue as pq, store, work_journal  # noqa: E402
from tools import agentic_packet as ap  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    for module, name, value in ((store, "KNOWLEDGE_ROOT", root), (store, "REPORT_DIR", root / "reports"),
                                (store, "GRAPH_DIR", root / "graph"), (store, "CHUNK_DB", root / "chunks.sqlite"),
                                (pq, "REPO_ROOT", tmp_path), (work_journal, "REPO_ROOT", tmp_path),
                                (ap, "ROOT", tmp_path), (ap, "SESSION_DIR", root / "workflows/results/s"),
                                (ap, "RENDER_DIR", root / ".work/renders")):
        monkeypatch.setattr(module, name, value)
    pdf = tmp_path / "src.pdf"
    pdf.write_bytes(b"%PDF synthetic")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    store.save("manifest", [{"id": "DOC_T", "source_pdf": "src.pdf", "sha256": sha, "page_count": 4}])
    store.save("enrichments", [
        {"id": "ENR_1", "type": "GeneratedDocument", "document_id": "DOC_T", "source_sha256": sha,
         "conversion_status": "success", "observed_pages": [2], "page_range": [2, 2], "active": True,
         "source_state": "CURRENT", "options": {}},
        {"id": "CAND_1", "run_id": "ENR_1", "document_id": "DOC_T", "source_sha256": sha,
         "source_state": "CURRENT", "candidate_kind": "formula", "content": "x = (a + b",
         "pages": [2], "page_provenance": [{"page_no": 2, "bbox": {"l": 1}}], "review_state": "UNREVIEWED"}])
    store.save("equations", [{"id": "EQ_T", "document_id": "DOC_T", "page": 2, "form": "x = a + b",
                              "quote": "the sum is defined as follows", "variables": [{"symbol": "y"}],
                              "provenance_kind": "SOURCE_STATES"}])
    conn = store.connect()
    conn.execute("INSERT INTO chunks (chunk_id, doc_id, seq, page_start, page_end, text, source_sha256) VALUES (?,?,?,?,?,?,?)",
                 ("DOC_T#1", "DOC_T", 0, 2, 2, "Here the sum is defined as follows: x = a + b.", sha))
    conn.commit()
    conn.close()
    monkeypatch.setattr(ap, "_render", lambda pdf, doc, page: {"page": page, "path": f"r{page}.png", "sha256": "0" * 64})
    pq.add("P", kind="verification", document_id="DOC_T", pages=(2, 2), targets=["EQ_T"], priority=1,
           reason="test", role="F", model="opus")
    return tmp_path


def test_s1_reuses_an_existing_successful_run_without_new_inference(isolated, monkeypatch):
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    monkeypatch.setattr(ap.subprocess, "run", lambda *a, **k: pytest.fail("no Docling run is needed"))
    out = ap.s1_run("P", "sonnet", 60)
    assert out["state"] == "RAW_CAPTURED" and out["run_ids"] == ["ENR_1"]


def test_s2_bundle_is_provisional_and_flags_mechanical_problems(isolated):
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    out = ap.s2_prepare("P", "sonnet")
    bundle = json.loads((isolated / out["bundle"]).read_text())
    assert bundle["authority"].startswith("PROVISIONAL CANDIDATES")
    cand = bundle["candidates"][0]
    assert "UNBALANCED_()" in cand["mechanical_flags"]
    obj = bundle["stored_objects"][0]
    checks = {c["check"]: c for c in obj["mechanical_checks"]}
    assert checks["quote_anchor"]["state"] == "GROUNDED"
    assert checks["variable_symbol_absent_from_form"]["symbols"] == ["y"]
    assert obj["same_page_formula_candidates"] == ["CAND_1"]
    assert "verification" not in store.load("equations")[0]  # S2 never changes scientific state
    assert pq.load()["packets"]["P"]["state"] == "CANDIDATES_PREPARED"


def test_s3_integrates_only_signed_reviewed_bytes_and_checks_stored_state(isolated, monkeypatch):
    tmp_path = isolated
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    ap.s2_prepare("P", "sonnet")
    pq.transition("P", "OPUS_SOURCE_REVIEW", role="F", model="opus")
    verdicts = tmp_path / "v.json"
    sha = store.load("manifest")[0]["sha256"]
    verdicts.write_text(json.dumps([{"object_id": "EQ_T", "refuted": False, "confidence": "high", "reason": "ok",
                                     "source_sha256": sha, "reviewer": {"role": "F", "model": "opus"}}]))
    pq.transition("P", "VERIFIED", role="F", model="opus", verdict_paths=[str(verdicts)])
    eq = store.load("equations")
    eq[0].update(source_sha256=sha, source_state="CURRENT", active=True)
    store.save("equations", eq)
    monkeypatch.setattr(graph, "assemble", lambda: {"meta": {"nodes": 1}})
    store.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (store.GRAPH_DIR / "graph.json").write_text(json.dumps({"nodes": [{"id": "EQ_T"}], "edges": []}))
    from kg import report, work_queue
    monkeypatch.setattr(work_queue, "write", lambda: {"summary": {}})
    monkeypatch.setattr(report, "write", lambda: None)
    out = ap.s3_integrate("P", "sonnet")
    assert out["state"] == "GRAPH_INTEGRATED"
    stored = store.load("equations")[0]
    assert stored["verification"] == "CONFIRMED" and stored["verifier_role"] == "F"
    log = json.loads((tmp_path / out["log"]).read_text())
    assert log["verdict_state_mismatches"] == [] and log["integrity"]["dangling_edges"] == 0


def test_s3_applies_signed_parser_reviews_to_individual_candidates(isolated, monkeypatch):
    tmp_path = isolated
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    ap.s2_prepare("P", "sonnet")
    pq.transition("P", "OPUS_SOURCE_REVIEW", role="F", model="opus")
    sha = store.load("manifest")[0]["sha256"]
    eq = store.load("equations")
    eq[0].update(source_sha256=sha, source_state="CURRENT", active=True)
    store.save("equations", eq)
    verdicts = tmp_path / "v.json"
    verdicts.write_text(json.dumps([{"object_id": "EQ_T", "refuted": False, "confidence": "high", "reason": "ok",
                                     "source_sha256": sha, "reviewer": {"role": "F", "model": "opus"}}]))
    parser = tmp_path / "parser.json"
    parser.write_text(json.dumps({"reviewer": {"role": "F", "model": "opus"}, "parser_reviews": [
        {"candidate_id": "CAND_1", "state": "REJECTED", "reason": "unbalanced; original shows x = a + b",
         "evidence": {"pdf_page": 2}}]}))
    unsigned = tmp_path / "parser_unsigned.json"
    unsigned.write_text(json.dumps({"parser_reviews": []}))
    with pytest.raises(pq.TransitionError):
        pq.transition("P", "VERIFIED", role="F", model="opus", verdict_paths=[str(verdicts), str(unsigned)])
    pq.transition("P", "VERIFIED", role="F", model="opus", verdict_paths=[str(verdicts), str(parser)])
    monkeypatch.setattr(graph, "assemble", lambda: {"meta": {}})
    store.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (store.GRAPH_DIR / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
    from kg import report, work_queue
    monkeypatch.setattr(work_queue, "write", lambda: {"summary": {}})
    monkeypatch.setattr(report, "write", lambda: None)
    assert ap.s3_integrate("P", "sonnet")["state"] == "GRAPH_INTEGRATED"
    cand = store.index_by("enrichments")["CAND_1"]
    run = store.index_by("enrichments")["ENR_1"]
    assert cand["review_state"] == "REJECTED" and cand["active"] is False
    assert cand["review_evidence"]["reviewer"]["role"] == "F"
    assert run.get("review_state") is None  # one rejected formula does not reject the whole page run


def test_s2_dereferences_figure_captions_and_keeps_model_targets_out_of_missing(isolated):
    tmp_path = isolated
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"texts": [{"text": "Figure 4-14. theta_n and theta_e"}]}))
    import hashlib
    rows = store.load("enrichments")
    rows.append({"id": "CAND_FIG", "run_id": "ENR_1", "candidate_kind": "figure", "pages": [2],
                 "content": {"captions": [{"$ref": "#/texts/0"}], "annotations": []},
                 "raw_artifact": "raw.json", "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                 "page_provenance": []})
    store.save("enrichments", rows)
    store.save("models", [{"id": "MODEL_X"}])
    data = pq.load()
    pq.add("PM", kind="model_review", document_id="DOC_T", pages=(2, 2), targets=["MODEL_X", "EQ_T", "EQ_GONE"],
           priority=1, reason="test", role="F", model="opus")
    pq.transition("PM", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("PM", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    out = ap.s2_prepare("PM", "sonnet")
    bundle = json.loads((tmp_path / out["bundle"]).read_text())
    fig = next(c for c in bundle["candidates"] if c["id"] == "CAND_FIG")
    assert fig["caption_text"] == ["Figure 4-14. theta_n and theta_e"]
    assert bundle["model_targets"] == ["MODEL_X"] and bundle["missing_targets"] == ["EQ_GONE"]


def test_s2_table_candidates_keep_complete_grid_and_item_captions(isolated):
    tmp_path = isolated
    import hashlib
    raw = tmp_path / "raw_t.json"
    raw.write_text(json.dumps({"texts": [{"text": "Table 1. Electric properties."}],
                               "tables": [{"captions": [{"$ref": "#/texts/0"}], "data": {}}]}))
    long_cell = "x" * 5000  # a dump of this content would exceed the old 4000-character cut
    rows = store.load("enrichments")
    rows.append({"id": "CAND_TAB", "run_id": "ENR_1", "candidate_kind": "table", "pages": [2], "json_pointer": "/tables/0",
                 "content": {"num_rows": 1, "num_cols": 2, "grid": [[{"text": "A"}, {"text": long_cell}]]},
                 "raw_artifact": "raw_t.json", "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                 "page_provenance": []})
    store.save("enrichments", rows)
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    bundle = json.loads((tmp_path / ap.s2_prepare("P", "sonnet")["bundle"]).read_text())
    tab = next(c for c in bundle["candidates"] if c["id"] == "CAND_TAB")
    assert tab["table_grid"] == [["A", long_cell]]
    assert tab["caption_text"] == ["Table 1. Electric properties."]


def test_s2_flags_a_damaged_text_layer_behind_a_clean_transcription(isolated):
    """Regression (S2 reports 2026-10-16): U+FFFD and glyph-code runs in original_text went unflagged."""
    rows = store.load("enrichments")
    rows.append({"id": "CAND_DMG", "run_id": "ENR_1", "candidate_kind": "formula",
                 "content": "L = ( R - \\sqrt { R ^ { 2 } - A } )",
                 "original_text": "L open ; max .0136 R post - " + ".0129" * 12 + " A cg � sin",
                 "pages": [2], "page_provenance": [], "review_state": "UNREVIEWED"})
    store.save("enrichments", rows)
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    bundle = json.loads((isolated / ap.s2_prepare("P", "sonnet")["bundle"]).read_text())
    flags = next(c for c in bundle["candidates"] if c["id"] == "CAND_DMG")["mechanical_flags"]
    assert {"REPLACEMENT_CHARACTER_IN_ORIGINAL_TEXT", "REPEATED_GLYPH_RUN_IN_ORIGINAL_TEXT"} <= set(flags)
    assert "DAMAGE_OR_REPETITION_MARKER" not in flags  # the transcription itself is clean
    clean = next(c for c in bundle["candidates"] if c["id"] == "CAND_1")["mechanical_flags"]
    assert not {"REPLACEMENT_CHARACTER_IN_ORIGINAL_TEXT", "REPEATED_GLYPH_RUN_IN_ORIGINAL_TEXT"} & set(clean)
    # A glyph run in the transcription itself is damage, even without spaces between repeats.
    assert "DAMAGE_OR_REPETITION_MARKER" in ap._formula_flags("x = " + ".0129" * 10)


def test_s3_checks_verdicts_on_constants_assumptions_and_validity(isolated, monkeypatch):
    """Regression (S3 report 2026-10-16): entity/validity verdicts read as stored=None and blocked integration."""
    tmp_path = isolated
    sha = store.load("manifest")[0]["sha256"]
    base = {"document_id": "DOC_T", "page": 2, "quote": "the sum is defined as follows", "source_sha256": sha,
            "source_state": "CURRENT", "active": True, "provenance_kind": "SOURCE_STATES"}
    store.save("entities", [{**base, "id": "ASM_T", "type": "Assumption", "label": "one motor"},
                            {**base, "id": "CONST_T", "type": "EmpiricalConstant", "label": "density"}])
    store.save("validity", [{**base, "id": "VAL_T", "type": "ValidityEnvelope", "prose": "design point only"}])
    eq = store.load("equations")
    eq[0].update(source_sha256=sha, source_state="CURRENT", active=True)
    store.save("equations", eq)
    pq.transition("P", "DOCLING_RUNNING", role="S1", model="sonnet")
    pq.transition("P", "RAW_CAPTURED", role="S1", model="sonnet", run_ids=["ENR_1"])
    ap.s2_prepare("P", "sonnet")
    pq.transition("P", "OPUS_SOURCE_REVIEW", role="F", model="opus")
    signed = {"confidence": "high", "reason": "page", "source_sha256": sha, "reviewer": {"role": "F", "model": "opus"}}
    verdicts = tmp_path / "v.json"
    verdicts.write_text(json.dumps([{**signed, "object_id": "EQ_T", "refuted": False},
                                    {**signed, "object_id": "ASM_T", "refuted": True},
                                    {**signed, "object_id": "CONST_T", "refuted": False},
                                    {**signed, "object_id": "VAL_T", "refuted": False}]))
    pq.transition("P", "VERIFIED", role="F", model="opus", verdict_paths=[str(verdicts)])
    monkeypatch.setattr(graph, "assemble", lambda: {"meta": {}})
    store.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (store.GRAPH_DIR / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
    from kg import report, work_queue
    monkeypatch.setattr(work_queue, "write", lambda: {"summary": {}})
    monkeypatch.setattr(report, "write", lambda: None)
    out = ap.s3_integrate("P", "sonnet")
    log = json.loads((tmp_path / out["log"]).read_text())
    checks = {c["object_id"]: c for c in log["verdict_state_checks"]}
    assert out["state"] == "GRAPH_INTEGRATED" and log["verdict_state_mismatches"] == []
    assert checks["ASM_T"]["stored"] == "REFUTED" and checks["CONST_T"]["stored"] == "CONFIRMED"
    assert checks["VAL_T"]["stored"] == "CONFIRMED"
