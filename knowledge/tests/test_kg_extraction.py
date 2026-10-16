"""Isolated extraction/recovery regression tests; no literature or agents needed."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_extraction, store
from tools import bank_extraction


def _indexed_source(source_sha256):
    """Keep synthetic index rows and their ingestion identity explicit."""
    conn = store.connect()
    conn.execute("INSERT OR REPLACE INTO chunks (chunk_id, doc_id, seq, page_start, page_end, text, source_sha256) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("DOC_TEST#c1", "DOC_TEST", 0, 1, 1, "Synthetic fixture anchor", source_sha256))
    conn.execute("INSERT OR REPLACE INTO doc_state (doc_id, source_sha256, markdown_sha256, n_chunks) "
                 "VALUES (?, ?, ?, ?)", ("DOC_TEST", source_sha256, "synthetic-markdown", 1))
    conn.commit()
    conn.close()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(store, "CHUNK_DB", tmp_path / "knowledge/chunks.sqlite")
    monkeypatch.setattr(bank_extraction, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bank_extraction.graph, "assemble", lambda: {"meta": {}})
    monkeypatch.setattr(bank_extraction.report, "write", lambda: None)
    monkeypatch.setattr(bank_extraction.validate, "run", lambda: {})
    from kg import work_queue
    monkeypatch.setattr(work_queue, "write", lambda: {})

    class Grounder:
        def __init__(self, doc_id):
            conn = store.connect()
            self.chunks = store.doc_chunks(conn, doc_id)
            conn.close()

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "DOC_TEST#c1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "DocGrounder", Grounder)
    _indexed_source("original-source")
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "original-source"}])
    store.save("models", [{"id": "MODEL_TEST", "subsystem": "nozzle"},
                           {"id": "MODEL_OTHER", "subsystem": "nozzle"}])
    return tmp_path


def result(target="MODEL_TEST"):
    anchor = {"page": 1, "quote": "The equation requires a steady flow assumption."}
    return {
        "doc_id": "DOC_TEST", "target": target,
        "supports_model": {"state": "SUPPORTED", "equation_labels": ["Continuity"]},
        "equations": [{**anchor, "label": "Continuity", "form": "mass = density * volume", "assumptions": ["steady"]}],
        "claims": [{**anchor, "statement": "Steady flow is assumed", "subject": "flow"}],
        "validity_statements": [{**anchor, "statement": "Only steady flow", "bounds": []}],
        "empirical_constants": [{**anchor, "name": "Example coefficient", "value": 2}],
        "assumptions": [{**anchor, "statement": "Steady flow"}],
    }


def test_rebank_preserves_verdicts_added_assumptions_and_model_links(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    claim = store.load("claims")[0]
    harvest_extraction.apply_verification([
        {"object_id": eq["id"], "refuted": False, "reason": "Reviewed", "confidence": "high",
         "missing_assumptions": [{"statement": "constant density", "page": 1, "quote": "Constant density is assumed here."}]},
        {"object_id": claim["id"], "refuted": True, "reason": "Overstated", "confidence": "high"},
    ])
    harvest_extraction.harvest([result()], [result("pages1-2")])
    eq = store.load("equations")[0]
    assert eq["verification"] == "CONFIRMED"
    assert eq["assumptions"] == ["steady", "constant density"]
    assert eq["model_ids"] == ["MODEL_TEST"]
    assert eq["subsystem"] == "nozzle"
    assert eq["model_link_basis"] == "registry_extraction:SUPPORTED"
    claim = store.load("claims")[0]
    assert claim["verification"] == "REFUTED"
    assert claim["model_ids"] == []
    assert store.load("validity")[0]["model_id"] == "MODEL_TEST"
    constant = next(r for r in store.load("entities") if r["type"] == "EmpiricalConstant")
    assert constant["model_ids"] == ["MODEL_TEST"]
    assert eq["source_sha256"] == "original-source"
    assert all(p["source_state"] == "CURRENT" for p in store.load("provenance"))


@pytest.mark.parametrize("history", [{"active": False}, {"source_state": "SUPERSEDED_EXTRACTION"}])
def test_historical_verdict_cannot_override_current_model_support(isolated, history):
    harvest_extraction.harvest([result()], [])
    eq = {**store.load("equations")[0], **history}
    store.save("equations", [eq])
    before_models = store.load("models")
    verdict = {"object_id": eq["id"], "refuted": False, "confidence": "high",
               "reason": "Historical extraction review", "model_support_override": "NOT_FOUND_IN_SOURCE"}
    stats = harvest_extraction.apply_verification([verdict])
    assert stats["retired_or_noncurrent"] == 1
    assert stats["confirmed"] == 0
    assert store.load("models") == before_models
    assert store.load("equations") == [eq]


def test_only_explicit_grounded_assumption_reference_creates_link(isolated):
    r = result()
    r["equations"][0]["assumption_refs"] = ["Steady flow"]
    harvest_extraction.harvest([r], [])
    equation = store.load("equations")[0]
    assumption = next(a for a in store.load("entities") if a["type"] == "Assumption")
    assert equation["assumption_ids"] == [assumption["id"]]
    assert equation["quote"] == r["equations"][0]["quote"]


def test_unresolved_assumption_reference_never_fabricates_entity(isolated):
    r = result()
    r["equations"][0]["assumption_refs"] = ["Unknown extra condition"]
    report = harvest_extraction.harvest([r], [])
    assert not store.load("equations")[0].get("assumption_ids")
    assert len(report["unresolved_assumption_links"]) == 1


def test_explicit_correction_retires_damaged_form_without_inheriting_model_support(isolated):
    old_result = result()
    old_result["equations"][0].update(form="NOT RENDERED IN MARKDOWN", equation_number="(5)")
    harvest_extraction.harvest([old_result], [])
    old = store.load("equations")[0]
    new_result = result("pages1-1")
    new_result["equations"][0].update(equation_number="5", supersedes_ids=[old["id"]])
    stats = harvest_extraction.harvest([], [new_result])
    records = store.index_by("equations")
    new = next(e for e in records.values() if e["id"] != old["id"])
    assert stats["superseded_incomplete_transcriptions"] == 1
    assert records[old["id"]]["active"] is False
    assert records[old["id"]]["superseded_by"] == new["id"]
    assert new["model_ids"] == []
    assert new["corrects_ids"] == [old["id"]]


def test_changed_source_does_not_reuse_old_verification(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    harvest_extraction.apply_verification([
        {"object_id": eq["id"], "refuted": True, "reason": "Old source", "confidence": "high"}])
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "revised-source"}])
    _indexed_source("revised-source")
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    assert "verification" not in eq
    assert eq["model_ids"] == ["MODEL_TEST"]
    assert eq["source_sha256"] == "revised-source"


@pytest.mark.parametrize("mismatch", ["manifest", "missing_manifest_hash", "doc_state", "missing_doc_state", "chunk", "missing_chunk_hash", "mixed_chunks"])
def test_stale_or_unidentified_index_cannot_be_stamped_current(isolated, mismatch):
    conn = store.connect()
    if mismatch == "manifest":
        store.save("manifest", [{"id": "DOC_TEST", "sha256": "replacement-source"}])
    elif mismatch == "missing_manifest_hash":
        store.save("manifest", [{"id": "DOC_TEST"}])
    elif mismatch == "missing_doc_state":
        conn.execute("DELETE FROM doc_state WHERE doc_id = 'DOC_TEST'")
    elif mismatch == "doc_state":
        conn.execute("UPDATE doc_state SET source_sha256 = 'stale-source'")
    elif mismatch in ("chunk", "missing_chunk_hash"):
        conn.execute("UPDATE chunks SET source_sha256 = ?", ("stale-source" if mismatch == "chunk" else None,))
    else:
        conn.execute("INSERT INTO chunks (chunk_id, doc_id, seq, page_start, page_end, text, source_sha256) "
                     "VALUES ('DOC_TEST#old', 'DOC_TEST', 1, 1, 1, 'Stale leftover chunk', 'stale-source')")
    conn.commit()
    conn.close()
    before = {p.name: p.read_bytes() for p in (isolated / "knowledge").glob("*.json")}
    with pytest.raises(ValueError, match="SOURCE_INDEX_IDENTITY_MISMATCH"):
        harvest_extraction.harvest([result()], [])
    after = {p.name: p.read_bytes() for p in (isolated / "knowledge").glob("*.json")}
    assert after == before  # neither science objects nor model support changed


def test_verifier_additions_require_current_index_identity(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    conn = store.connect()
    conn.execute("UPDATE chunks SET source_sha256 = 'stale-source'")
    conn.commit()
    conn.close()
    before = {p.name: p.read_bytes() for p in (isolated / "knowledge").glob("*.json")}
    verdict = {"object_id": eq["id"], "refuted": False, "confidence": "high",
               "missing_assumptions": [{"statement": "extra condition", "page": 1,
                                         "quote": "A condition present only in the stale derivative."}]}
    with pytest.raises(ValueError, match="SOURCE_INDEX_IDENTITY_MISMATCH"):
        harvest_extraction.apply_verification([verdict])
    assert before == {p.name: p.read_bytes() for p in (isolated / "knowledge").glob("*.json")}


def test_shared_constant_and_validity_keep_all_model_links(isolated):
    harvest_extraction.harvest([result(), result("MODEL_OTHER")], [result("pages1-2")])
    assert store.load("validity")[0]["model_ids"] == ["MODEL_OTHER", "MODEL_TEST"]
    constant = next(r for r in store.load("entities") if r["type"] == "EmpiricalConstant")
    assert constant["model_ids"] == ["MODEL_OTHER", "MODEL_TEST"]


def test_banking_same_journal_twice_is_a_noop(isolated):
    journal = isolated / "results.jsonl"
    journal.write_text(json.dumps({"type": "result", "key": "one", "value": result()}) + "\n")
    assert bank_extraction.main([str(journal)]) == 0
    before = {p.name: p.read_bytes() for p in (isolated / "knowledge").rglob("*.json")}
    assert bank_extraction.main([str(journal)]) == 0
    after = {p.name: p.read_bytes() for p in (isolated / "knowledge").rglob("*.json")}
    assert before == after
    report = json.loads((isolated / "knowledge/reports/latest_extraction.json").read_text())
    assert report["grounded"] == 5
    assert report["n_equations"] == report["n_claims"] == report["n_validity"] == 1


def test_bank_appended_results_counts_unique_objects(isolated):
    journal = isolated / "results.jsonl"
    one = json.dumps({"type": "result", "key": "one", "value": result()}) + "\n"
    journal.write_text(one)
    bank_extraction.main([str(journal)])
    journal.write_text(one + json.dumps({"type": "result", "key": "two", "value": result("MODEL_OTHER")}) + "\n")
    bank_extraction.main([str(journal)])
    report = json.loads((isolated / "knowledge/reports/latest_extraction.json").read_text())
    assert report["grounded"] == 5
    assert len(report["banked_result_hashes"]) == 2
    assert len(report["model_support"]) == 2


def test_corrupt_journal_line_is_visible_but_partial_tail_is_resumable(tmp_path):
    journal = tmp_path / "results.jsonl"
    valid = json.dumps({"type": "result", "value": result()}) + "\n"
    journal.write_text(valid + '{"type":')
    assert len(bank_extraction.load_results(journal)) == 1
    journal.write_text(valid + 'not-json\n' + valid)
    with pytest.raises(ValueError, match="Invalid journal JSON"):
        bank_extraction.load_results(journal)


def test_unknown_result_target_fails_without_touching_stores(isolated):
    journal = isolated / "results.jsonl"
    journal.write_text(json.dumps({"type": "result", "value": result("UNKNOWN")}) + "\n")
    with pytest.raises(SystemExit):
        bank_extraction.main([str(journal)])
    assert store.load("equations") == []


def test_refutation_updates_effective_support_and_repeat_keeps_original_links(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load('equations')[0]
    verdict = {'object_id': eq['id'], 'refuted': True, 'reason': 'Unsupported attribution',
               'confidence': 'high', 'model_support_override': 'NOT_FOUND_IN_SOURCE'}
    harvest_extraction.apply_verification([verdict])
    assert store.index_by('models')['MODEL_TEST']['source_support_state'] == 'NOT_FOUND_IN_SOURCE'
    assert harvest_extraction.apply_verification([verdict])['unchanged'] == 1
    assert store.load('equations')[0]['model_ids_before_refutation'] == ['MODEL_TEST']
    harvest_extraction.harvest([result()], [])
    assert store.index_by('models')['MODEL_TEST']['source_support_state'] == 'NOT_FOUND_IN_SOURCE'


def test_uncertain_verdict_does_not_confirm_scientific_object(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load('equations')[0]
    stats = harvest_extraction.apply_verification([
        {'object_id': eq['id'], 'refuted': False, 'reason': 'Unreadable notation', 'confidence': 'low'}])
    assert stats['uncertain'] == 1
    assert store.load('equations')[0]['verification'] == 'UNCONFIRMED'


def test_bank_refuses_result_for_replaced_pdf(isolated):
    journal = isolated / 'stale.jsonl'
    value = {**result(), 'source_sha256': 'obsolete-source'}
    journal.write_text(json.dumps({'type': 'result', 'value': value}) + '\n')
    with pytest.raises(SystemExit):
        bank_extraction.main([str(journal)])
    assert store.load('equations') == []


def test_verification_checks_source_hash_and_keeps_inspection_evidence(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load('equations')[0]
    verdict = {'object_id': eq['id'], 'refuted': False, 'confidence': 'high',
               'reason': 'Visually compared with source PDF', 'source_sha256': 'obsolete-source',
               'evidence': {'pdf_page': 1, 'method': 'visual review'},
               'scope_note': 'Source transcription only, not hardware validation'}
    assert harvest_extraction.apply_verification([verdict])['stale'] == 1
    assert 'verification' not in store.load('equations')[0]
    verdict['source_sha256'] = 'original-source'
    assert harvest_extraction.apply_verification([verdict])['confirmed'] == 1
    rec = store.load('equations')[0]
    assert rec['verifier_evidence'] == verdict['evidence']
    assert rec['verifier_scope_note'] == verdict['scope_note']
    store.save('manifest', [{'id': 'DOC_TEST', 'sha256': 'newer-source'}])
    assert harvest_extraction.apply_verification([verdict])['stale'] == 1
    verdict.pop('source_sha256')
    assert harvest_extraction.apply_verification([verdict])['stale'] == 1


def test_replacement_verdict_does_not_inherit_previous_review_evidence(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load('equations')[0]
    first = {'object_id': eq['id'], 'refuted': False, 'confidence': 'high',
             'reason': 'Visual review', 'evidence': {'pdf_page': 1}, 'scope_note': 'Transcription only'}
    harvest_extraction.apply_verification([first])
    replacement = {'object_id': eq['id'], 'refuted': False, 'confidence': 'low',
                   'reason': 'Need to inspect missing notation'}
    harvest_extraction.apply_verification([replacement])
    rec = store.load('equations')[0]
    assert rec['verification'] == 'UNCONFIRMED'
    assert 'verifier_evidence' not in rec and 'verifier_scope_note' not in rec


def test_composite_damaged_equation_collects_all_children_across_replay(isolated):
    damaged = result()
    damaged["equations"][0].update(form="NOT RENDERED IN MARKDOWN", equation_number="9.26-9.28")
    harvest_extraction.harvest([damaged], [])
    old, = store.load("equations")
    corrected = result()
    anchor = corrected["equations"][0]
    corrected["equations"] = [{**anchor, "label": "Recovered " + number,
                               "equation_number": number, "form": "recovered form " + number,
                               "supersedes_ids": [old["id"]]}
                              for number in ("9.26", "9.27", "9.28a", "9.28b")]
    harvest_extraction.harvest([corrected], [])
    records = {r["id"]: r for r in store.load("equations")}
    children = {r["id"] for r in records.values() if r["id"] != old["id"]}
    superseded = records[old["id"]]
    first = superseded["superseded_by"]
    assert isinstance(first, str) and first in children
    assert set(superseded["superseded_by_ids"]) == children
    assert superseded["active"] is False and superseded["source_state"] == "SUPERSEDED_EXTRACTION"
    assert all(records[child]["corrects_ids"] == [old["id"]] for child in children)
    # An older packet and then reversed child order cannot erase or replace
    # the first scalar compatibility link or any of the four child IDs.
    harvest_extraction.harvest([damaged], [])
    corrected["equations"].reverse()
    harvest_extraction.harvest([corrected], [])
    again = next(r for r in store.load("equations") if r["id"] == old["id"])
    assert again["superseded_by"] == first
    assert set(again["superseded_by_ids"]) == children
    assert len(again["superseded_by_ids"]) == 4


@pytest.mark.parametrize("old_number, new_number", [
    ("9.26-9.28", "9.25"),
    ("9.26-9.28", "9.29"),
    ("9.26-9.28", "10.27"),
    ("9.26-10.28", "9.27"),
    ("9.28-9.26", "9.27"),
    ("9.26,9.28", "9.27"),
    ("9.26-9.28", "9.28bc"),
    ("9.28", "9.28a"),
])
def test_composite_supersession_does_not_guess_disjoint_or_other_numberings(isolated, old_number, new_number):
    damaged = result()
    damaged["equations"][0].update(form="OCR-garbled relation", equation_number=old_number)
    harvest_extraction.harvest([damaged], [])
    old, = store.load("equations")
    corrected = result()
    corrected["equations"][0].update(form="recovered equation", equation_number=new_number, supersedes_ids=[old["id"]])
    report = harvest_extraction.harvest([corrected], [])
    historical = next(r for r in store.load("equations") if r["id"] == old["id"])
    assert "superseded_by" not in historical
    assert report["unresolved_supersessions"][0]["old_id"] == old["id"]


@pytest.mark.parametrize("change", [
    {"document_id": "DOC_OTHER"},
    {"source_sha256": "different-source"},
    {"form": "complete original mathematical relation"},
])
def test_composite_supersession_requires_damaged_same_document_current_source(isolated, change):
    damaged = result()
    damaged["equations"][0].update(form="NOT RENDERED IN MARKDOWN", equation_number="9.26-9.28")
    harvest_extraction.harvest([damaged], [])
    old, = store.load("equations")
    store.save("equations", [{**old, **change}])
    corrected = result()
    corrected["equations"][0].update(form="recovered formula", equation_number="9.27", supersedes_ids=[old["id"]])
    report = harvest_extraction.harvest([corrected], [])
    historical = next(r for r in store.load("equations") if r["id"] == old["id"])
    assert "superseded_by_ids" not in historical
    assert report["unresolved_supersessions"][0]["old_id"] == old["id"]
