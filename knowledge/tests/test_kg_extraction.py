"""Isolated extraction/recovery regression tests; no literature or agents needed."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_extraction, store
from tools import bank_extraction


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(bank_extraction, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bank_extraction.graph, "assemble", lambda: {"meta": {}})
    monkeypatch.setattr(bank_extraction.report, "write", lambda: None)
    monkeypatch.setattr(bank_extraction.validate, "run", lambda: {})
    from kg import work_queue
    monkeypatch.setattr(work_queue, "write", lambda: {})

    class Grounder:
        chunks = [{"chunk_id": "DOC_TEST#c1"}]

        def __init__(self, _doc_id):
            pass

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "DOC_TEST#c1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "DocGrounder", Grounder)
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


def test_changed_source_does_not_reuse_old_verification(isolated):
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    harvest_extraction.apply_verification([
        {"object_id": eq["id"], "refuted": True, "reason": "Old source", "confidence": "high"}])
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "revised-source"}])
    harvest_extraction.harvest([result()], [])
    eq = store.load("equations")[0]
    assert "verification" not in eq
    assert eq["model_ids"] == ["MODEL_TEST"]
    assert eq["source_sha256"] == "revised-source"


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
