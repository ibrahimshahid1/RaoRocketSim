"""K12: independent source verdicts are scoped to the assertion inspected."""
from copy import deepcopy

import pytest

from test_kg_extraction import isolated, result, _indexed_source
from kg import api, completion, coverage, harvest_extraction, store, verification_freshness as vf


def _confirm(equation, **extra):
    verdict = {"object_id": equation["id"], "refuted": False, "confidence": "high",
               "reason": "Synthetic review only", "source_sha256": equation["source_sha256"],
               "content_sha256": vf.content_sha256(equation), **extra}
    assert harvest_extraction.apply_verification([verdict])["confirmed"] == 1
    return verdict


@pytest.mark.parametrize("field,value", [
    ("assumptions", ["steady", "constant density"]),
    ("validity", ["single-phase flow only"]),
    ("description", "A newly scoped continuity equation"),
])
def test_rebank_content_change_invalidates_independent_verdict(isolated, field, value):
    extraction = result()
    harvest_extraction.harvest([extraction], [])
    old, = store.load("equations")
    verdict = _confirm(old)
    extraction["equations"][0][field] = value
    report = harvest_extraction.harvest([extraction], [])
    changed, = store.load("equations")
    assert changed["id"] == old["id"]
    assert changed["source_sha256"] == old["source_sha256"]
    assert changed["verification_state"] == "CONTENT_CHANGED_REVIEW_REQUIRED"
    assert "verification" not in changed and "verifier_content_sha256" not in changed
    assert report["content_verifications_invalidated"] == [old["id"]]
    assert changed["verification_history"][0]["verdict"]["verification"] == "CONFIRMED"
    assert changed["verification_history"][0]["content"] == vf.content_payload(old)
    assert harvest_extraction.apply_verification([verdict])["content_mismatch"] == 1
    _confirm(changed)


def test_noop_rebank_links_and_verifier_additions_keep_content_verification(isolated):
    harvest_extraction.harvest([result()], [])
    old, = store.load("equations")
    verdict = _confirm(old, missing_assumptions=[
        {"statement": "constant density", "page": 1, "quote": "Constant density is assumed here."}])
    verified, = store.load("equations")
    assert vf.state(verified)["state"] == "CONFIRMED"
    assert verified["verifier_content_sha256"] == vf.content_sha256(verified)
    assert harvest_extraction.apply_verification([verdict])["unchanged"] == 1
    report = harvest_extraction.harvest([result(), result("MODEL_OTHER")], [])
    replayed, = store.load("equations")
    assert report["content_verifications_invalidated"] == []
    assert replayed["model_ids"] == ["MODEL_OTHER", "MODEL_TEST"]
    assert vf.state(replayed)["state"] == "CONFIRMED"
    assert replayed["verified_at"] == verified["verified_at"]
    assert not replayed.get("verification_history")


@pytest.mark.parametrize("collection,store_name,updates", [
    ("claims", "claims", {"qualifiers": ["single-row injector only"]}),
    ("validity_statements", "validity", {"bounds": [{"variable": "Re", "min": 10000}]}),
    ("empirical_constants", "entities", {"units": "mm"}),
    ("assumptions", "entities", {"page": 2}),
])
def test_non_equation_source_assertion_changes_are_content_scoped(isolated, collection, store_name, updates):
    extraction = result()
    harvest_extraction.harvest([extraction], [])
    desired_type = {"claims": "Claim", "validity_statements": "ValidityEnvelope",
                    "empirical_constants": "EmpiricalConstant", "assumptions": "Assumption"}[collection]
    old = next(r for r in store.load(store_name) if r["type"] == desired_type)
    _confirm(old)
    extraction[collection][0].update(updates)
    harvest_extraction.harvest([extraction], [])
    changed = store.index_by(store_name)[old["id"]]
    assert changed["verification_state"] == "CONTENT_CHANGED_REVIEW_REQUIRED"
    assert changed["verification_history"][0]["content"] == vf.content_payload(old)


def test_invalidated_legacy_verdict_cannot_reconfirm_changed_content(isolated):
    harvest_extraction.harvest([result()], [])
    old, = store.load("equations")
    legacy = {"object_id": old["id"], "refuted": False, "confidence": "high", "reason": "Old verdict"}
    assert harvest_extraction.apply_verification([legacy])["confirmed"] == 1
    extraction = result()
    extraction["equations"][0]["validity"] = ["A newly added limit"]
    harvest_extraction.harvest([extraction], [])
    before = store.load("equations")
    outcome = harvest_extraction.apply_verification([legacy])
    assert outcome["stale_legacy_replay"] == 1
    assert outcome["stores_saved"] is False
    assert store.load("equations") == before
    fresh_legacy = {**legacy, "reason": "A different reason still does not bind inspected content"}
    assert harvest_extraction.apply_verification([fresh_legacy])["content_hash_required"] == 1
    assert store.load("equations") == before


def test_direct_content_change_cannot_bypass_rebank_freshness_with_legacy_replay(isolated):
    harvest_extraction.harvest([result()], [])
    equation, = store.load("equations")
    legacy = {"object_id": equation["id"], "refuted": False, "confidence": "high", "reason": "Original review"}
    assert harvest_extraction.apply_verification([legacy])["confirmed"] == 1
    changed, = store.load("equations")
    changed["description"] = "Directly changed after verification"
    store.save("equations", [changed])
    assert harvest_extraction.apply_verification([{
        **legacy, "reason": "Different wording cannot bypass the missing content hash"}])["content_hash_required"] == 1
    assert harvest_extraction.apply_verification([legacy])["stale_legacy_replay"] == 1
    invalidated, = store.load("equations")
    assert invalidated["verification_state"] == "CONTENT_CHANGED_REVIEW_REQUIRED"
    assert "verification" not in invalidated
    assert invalidated["verification_history"][0]["content_snapshot_basis"] == (
        "STORED_ASSERTION_AT_INVALIDATION_NOT_PROOF_OF_INSPECTED_LEGACY_CONTENT")
    assert harvest_extraction.apply_verification([legacy])["stale_legacy_replay"] == 1
    _confirm(invalidated)


def test_unbound_legacy_confirmation_is_historical_not_current_content_evidence():
    rec = {"id": "EQ_TOY", "verification": "CONFIRMED", "source_sha256": "source",
           "verifier_source_sha256": "source", "form": "x=y"}
    assert vf.state(rec, {"sha256": "source"}) == {
        "state": "LEGACY_CONTENT_NOT_BOUND", "historical_verification": "CONFIRMED"}
    bound = {**rec, "verifier_content_sha256": vf.content_sha256(rec)}
    assert vf.state(bound, {"sha256": "source"})["state"] == "CONFIRMED"
    bound["form"] = "x=2*y"
    assert vf.state(bound, {"sha256": "source"})["state"] == "CONTENT_CHANGED_REVIEW_REQUIRED"
    assert vf.state(bound, {"sha256": "replacement"})["state"] == "STALE_SOURCE"


def test_retrieval_exposes_legacy_content_state_without_fabricating_a_review(monkeypatch):
    rec = {"id": "EQ_TOY", "document_id": "DOC_TOY", "verification": "CONFIRMED",
           "source_sha256": "source", "verifier_source_sha256": "source", "form": "x=y"}
    monkeypatch.setattr(api, "_docs", lambda: {"DOC_TOY": {"sha256": "source"}})
    summary = api.content_verification_summary(rec)
    assert summary["state"] == "LEGACY_CONTENT_NOT_BOUND"
    assert summary["historical_verification"] == "CONFIRMED"
    assert summary["content_sha256"] == vf.content_sha256(rec)
    assert "verifier_content_sha256" not in rec


def test_explicit_missing_document_never_confirms_current_source_content(monkeypatch):
    record = {"id": "EQ_ORPHAN", "document_id": "DOC_MISSING", "source_sha256": "source",
              "verifier_source_sha256": "source", "verification": "CONFIRMED", "form": "x=y",
              "provenance_kind": "SOURCE_STATES"}
    record["verifier_content_sha256"] = vf.content_sha256(record)
    # Internal hashing/no-op checks may intentionally omit source lookup.
    assert vf.state(record)["state"] == "CONFIRMED"
    assert vf.state(record, None) == {"state": "SOURCE_NOT_INGESTED", "historical_verification": "CONFIRMED"}
    monkeypatch.setattr(api, "_docs", lambda: {})
    assert api.content_verification_summary(record)["state"] == "SOURCE_NOT_INGESTED"
    result = coverage.summary({"equations": [record]})
    stats = result["source_objects"]["equations"]
    assert stats["content_bound_confirmed"] == 0
    assert stats["all_content_verification_by_state"] == {"SOURCE_NOT_INGESTED": 1}
    assert completion._content_verification(record, {})["state"] == "SOURCE_NOT_INGESTED"


def test_source_identity_change_preserves_verdict_history_separately_from_content(isolated):
    harvest_extraction.harvest([result()], [])
    original, = store.load("equations")
    _confirm(original)
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "replaced-source"}])
    _indexed_source("replaced-source")
    harvest_extraction.harvest([result()], [])
    current, = store.load("equations")
    assert vf.content_sha256(original) == vf.content_sha256(current)
    assert current["verification_state"] == "STALE_SOURCE"
    assert current["verification_history"][0]["verdict"]["verifier_source_sha256"] == "original-source"
    assert "verification" not in current


def test_content_hash_excludes_source_identity_and_derived_bookkeeping():
    rec = {"id": "EQ_TOY", "form": "x=y", "document_id": "DOC_TOY", "page": 1,
           "quote": "Fixture anchor", "source_sha256": "old", "variables": [
               {"symbol": "x", "meaning": "distance", "units": "m", "resolution_state": "UNRESOLVED"}]}
    changed = deepcopy(rec)
    changed.update(source_sha256="new", model_ids=["MODEL_TOY"], recorded_at="later",
                   enrichment_candidate_ids=["CAND_TOY"])
    changed["variables"][0].update(resolution_state="RESOLVED", quantity_candidates=["Q_TOY"])
    assert vf.content_sha256(rec) == vf.content_sha256(changed)
    changed["variables"][0]["units"] = "mm"
    assert vf.content_sha256(rec) != vf.content_sha256(changed)


def test_verdict_replay_is_byte_noop_with_content_fingerprint(isolated):
    harvest_extraction.harvest([result()], [])
    equation, = store.load("equations")
    verdict = _confirm(equation)
    paths = list((isolated / "knowledge").glob("*.json"))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    assert harvest_extraction.apply_verification([verdict])["unchanged"] == 1
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}


def _mixed_verification_records():
    base = {"document_id": "DOC_TEST", "provenance_kind": "SOURCE_STATES", "source_sha256": "source",
            "verifier_source_sha256": "source", "verification": "CONFIRMED", "form": "x=y", "page": 1}
    legacy = {**base, "id": "EQ_LEGACY"}
    bound = {**base, "id": "EQ_BOUND"}
    bound["verifier_content_sha256"] = vf.content_sha256(bound)
    changed = {**base, "id": "EQ_CHANGED"}
    changed["verifier_content_sha256"] = vf.content_sha256(changed)
    changed["description"] = "Content added after the review"
    stale = {**base, "id": "EQ_STALE", "source_sha256": "old", "verifier_source_sha256": "old"}
    return [legacy, bound, changed, stale]


def test_coverage_preserves_historical_counts_and_adds_content_bound_states():
    rows = _mixed_verification_records()
    data = coverage.summary({"manifest": [{"id": "DOC_TEST", "sha256": "source"}], "equations": rows},
                            include_documents=True)
    equations = data["source_objects"]["equations"]
    assert equations["independently_confirmed"] == 3  # legacy stored/source count is unchanged
    assert equations["stored_verdict_totals_all_records"] == {"CONFIRMED": 4}
    assert equations["content_bound_confirmed"] == 1
    assert equations["current_content_verification_by_state"] == {
        "LEGACY_CONTENT_NOT_BOUND": 1, "CONFIRMED": 1, "CONTENT_CHANGED_REVIEW_REQUIRED": 1}
    assert equations["all_content_verification_by_state"]["STALE_SOURCE"] == 1
    assert equations["current_without_content_bound_confirmation"] == 2
    assert data["documents"]["with_content_bound_confirmed_objects"] == 1
    assert data["by_document"][0]["content_bound_confirmed_equations"] == 1
    assert data["gaps"]["equations_and_claims_pending_independent_verification"] == 0
    assert "historical stored-verdict" in data["scope_note"]


def test_completion_reports_keep_legacy_totals_separate_from_content_bound_reviews(isolated, monkeypatch):
    from kg import enrichment
    monkeypatch.setattr(enrichment, "queue", lambda: {"documents": []})
    monkeypatch.setattr(completion, "_packets", lambda: {})
    monkeypatch.setattr(completion, "_docling_jobs", lambda: [])
    monkeypatch.setattr(completion, "_indexed_pages", lambda: ({}, {}))
    monkeypatch.setattr(completion, "_inputs", lambda: {})
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "source", "page_count": 1}])
    store.save("equations", _mixed_verification_records())
    store.save("models", [])
    corpus = completion.corpus_completeness()
    row = corpus["documents"][0]
    assert row["independent_verification"] == {"CONFIRMED": 4}
    assert "Historical stored verdicts" in row["independent_verification_basis"]
    assert row["content_verification"]["source_assertions_by_state"] == {
        "LEGACY_CONTENT_NOT_BOUND": 1, "CONFIRMED": 1,
        "CONTENT_CHANGED_REVIEW_REQUIRED": 1, "STALE_SOURCE": 1}
    assert corpus["denominators"]["equations_claims_confirmed"] == 4
    assert corpus["denominators"]["equations_claims_content_bound_confirmed"] == 1
    fidelity = completion.equation_fidelity_audit()
    assert fidelity["summary"]["current_by_verification"] == {"CONFIRMED": 4}
    assert fidelity["summary"]["current_content_bound_confirmed"] == 1
    assert fidelity["summary"]["current_content_verification_by_state"]["LEGACY_CONTENT_NOT_BOUND"] == 1
    assert next(r for r in fidelity["equations"] if r["id"] == "EQ_CHANGED")[
        "content_verification"]["state"] == "CONTENT_CHANGED_REVIEW_REQUIRED"
