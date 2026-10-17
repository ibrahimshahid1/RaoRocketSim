"""Parser-QA assertions retain history and replay without changing the latest verdict."""

from copy import deepcopy

import pytest

from test_kg_enrichment import corpus
from kg import enrichment, store


def candidates(corpus):
    _, raw, metadata = corpus
    run = enrichment.integrate(raw, metadata)["run_id"]
    candidate = next(row for row in store.load("enrichments") if row.get("candidate_kind") == "formula")
    return run, candidate["id"]


def review(candidate, state, label, *, historical_only=False):
    return enrichment.record_review(candidate, state=state, reason=f"Source inspection {label}",
                                    evidence={"reviewer": {"role": "F", "model": "fixture"},
                                              "review_file": f"{label}.json", "review_file_sha256": label * 64,
                                              "pdf_page": 7}, historical_only=historical_only)


def test_new_review_archives_prior_assertion_and_all_exact_replays_are_byte_noops(corpus):
    _, candidate = candidates(corpus)
    assert review(candidate, "SOURCE_PAGE_COMPARED", "a")["changed"]
    first = deepcopy(store.index_by("enrichments")[candidate])
    assert review(candidate, "REJECTED", "b")["changed"]
    latest = store.index_by("enrichments")[candidate]
    assert latest["review_state"] == "REJECTED" and latest["active"] is False
    assert latest["review_history"][0]["state"] == "SOURCE_PAGE_COMPARED"
    assert latest["review_history"][0]["evidence"] == first["review_evidence"]
    assert latest["review_history"][0]["review_sha256"] == first["review_sha256"]
    store_path = store.KNOWLEDGE_ROOT / "enrichments.json"
    before, mtime = store_path.read_bytes(), store_path.stat().st_mtime_ns
    for state, label in [("SOURCE_PAGE_COMPARED", "a"), ("REJECTED", "b"), ("SOURCE_PAGE_COMPARED", "a")]:
        result = review(candidate, state, label)
        assert result["changed"] is False
        assert result["unchanged"] + result["historical_replays"] == 1
    assert store_path.read_bytes() == before and store_path.stat().st_mtime_ns == mtime
    assert store.index_by("enrichments")[candidate]["active"] is False


def test_genuinely_new_verdict_supersedes_current_without_reactivating_raw_candidate(corpus):
    _, candidate = candidates(corpus)
    review(candidate, "SOURCE_PAGE_COMPARED", "a")
    review(candidate, "REJECTED", "b")
    assert review(candidate, "SOURCE_PAGE_COMPARED", "c")["changed"]
    latest = store.index_by("enrichments")[candidate]
    assert latest["review_state"] == "SOURCE_PAGE_COMPARED"
    assert [entry["state"] for entry in latest["review_history"]] == ["SOURCE_PAGE_COMPARED", "REJECTED"]
    assert latest["active"] is False  # explicit re-integration owns derivative activation
    before = store.load("enrichments")
    assert review(candidate, "REJECTED", "b")["historical_replays"] == 1
    assert store.load("enrichments") == before


def test_historical_recovery_retains_prior_review_without_changing_latest(corpus):
    _, candidate = candidates(corpus)
    review(candidate, "REJECTED", "b")
    latest = deepcopy(store.index_by("enrichments")[candidate])
    result = review(candidate, "SOURCE_PAGE_COMPARED", "a", historical_only=True)
    assert result["historical_added"] == 1
    recovered = store.index_by("enrichments")[candidate]
    assert {k: v for k, v in recovered.items() if k != "review_history"} == {
        k: v for k, v in latest.items() if k != "review_history"}
    assert recovered["review_history"][0]["recovered_as_history_only"] is True
    before = store.load("enrichments")
    assert review(candidate, "SOURCE_PAGE_COMPARED", "a")["historical_replays"] == 1
    assert review(candidate, "SOURCE_PAGE_COMPARED", "a", historical_only=True)["changed"] is False
    assert store.load("enrichments") == before


def test_legacy_current_review_replays_without_rewriting_and_is_archived_on_new_verdict(corpus):
    _, candidate = candidates(corpus)
    review(candidate, "SOURCE_PAGE_COMPARED", "a")
    rows = store.load("enrichments")
    for row in rows:
        for key in ("review_sha256", "review_source_identity", "review_recorded_at", "review_history"):
            row.pop(key, None)
    store.save("enrichments", rows)
    before = store.load("enrichments")
    assert review(candidate, "SOURCE_PAGE_COMPARED", "a")["changed"] is False
    assert store.load("enrichments") == before
    review(candidate, "REJECTED", "b")
    history = store.index_by("enrichments")[candidate]["review_history"]
    assert history[0]["state"] == "SOURCE_PAGE_COMPARED"
    assert history[0]["source_identity"]["source_sha256"] == before[0]["source_sha256"]


def test_canonical_evidence_key_order_does_not_create_another_assertion(corpus):
    _, candidate = candidates(corpus)
    evidence = {"pdf_page": 7, "reviewer": {"role": "F", "model": "fixture"}}
    enrichment.record_review(candidate, state="SOURCE_PAGE_COMPARED", reason="same assertion", evidence=evidence)
    before = store.load("enrichments")
    result = enrichment.record_review(candidate, state="SOURCE_PAGE_COMPARED", reason="same assertion",
                                       evidence={"reviewer": {"model": "fixture", "role": "F"}, "pdf_page": 7})
    assert result["changed"] is False
    assert store.load("enrichments") == before


@pytest.mark.parametrize("mismatch", ["manifest", "record_source", "stale_record", "evidence_source", "evidence_raw"])
def test_review_refuses_stale_or_mismatched_source_identity_without_writes(corpus, mismatch):
    _, candidate = candidates(corpus)
    evidence = {"pdf_page": 7}
    if mismatch == "manifest":
        store.save("manifest", [{"id": "DOC_A", "sha256": "other"}])
    elif mismatch in {"record_source", "stale_record"}:
        rows = store.load("enrichments")
        row = next(row for row in rows if row["id"] == candidate)
        row.update({"source_sha256": "other"} if mismatch == "record_source" else {"source_state": "STALE_SOURCE"})
        store.save("enrichments", rows)
    else:
        evidence["source_sha256" if mismatch == "evidence_source" else "raw_sha256"] = "other"
    before = store.load("enrichments")
    with pytest.raises(ValueError, match="identity is stale or mismatched"):
        enrichment.record_review(candidate, state="REJECTED", reason="stale review", evidence=evidence)
    assert store.load("enrichments") == before


def test_reintegration_retains_complete_review_history(corpus):
    _, raw, metadata = corpus
    _, candidate = candidates(corpus)
    review(candidate, "SOURCE_PAGE_COMPARED", "a")
    review(candidate, "REJECTED", "b")
    before = deepcopy(store.index_by("enrichments")[candidate])
    enrichment.integrate(raw, metadata)
    assert store.index_by("enrichments")[candidate] == before


def test_run_level_replays_preserve_each_childs_latest_review(corpus):
    run, candidate = candidates(corpus)
    review(run, "SOURCE_PAGE_COMPARED", "a")
    review(candidate, "REJECTED", "b")
    before = store.load("enrichments")
    result = review(run, "SOURCE_PAGE_COMPARED", "a")
    assert result["changed"] is False and result["historical_replays"] == 1
    assert result["unchanged"] == len(before) - 1
    assert store.load("enrichments") == before
