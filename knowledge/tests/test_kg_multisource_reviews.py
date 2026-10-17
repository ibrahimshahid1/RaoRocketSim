"""Multi-source review retention, link fidelity and declared-link bank integrity.

Synthetic sources and code isolate graph/retrieval behavior from the local corpus.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import api, graph, harvest_extraction, implementation_freshness, index, store  # noqa: E402


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(store, "CHUNK_DB", root / "vector_index" / "chunks.sqlite")
    monkeypatch.setattr(index, "VECTOR_DIR", root / "vector_index")
    monkeypatch.setattr(graph, "GRAPH_DIR", root / "graph")
    monkeypatch.setattr(implementation_freshness, "REPO_ROOT", tmp_path)

    class Grounder:
        chunks = [{}]

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "fixture#1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "_current_grounder", lambda doc_id, sha: Grounder())
    store.save("manifest", [{"id": f"DOC_{key}", "sha256": key, "filename": key + ".pdf",
                             "source_pdf": key + ".pdf", "conversion": {"status": "good"}}
                            for key in ("A", "B")])
    store.save("models", [{"id": "MODEL_TEST", "source_document_id": "DOC_A", "registry_id": "fixture.nozzle",
                           "subsystem": "nozzle", "quantity": "coefficient", "relation": "h=x",
                           "basis_class": "LITERATURE_DERIVED", "promotion_status": "screening",
                           "validation_level": "not_physically_validated", "registry_status": "available"}])
    code = tmp_path / "pkg" / "physics.py"
    code.parent.mkdir()
    code.write_text("def h(x):\n    return x\n\ndef q(x):\n    return 2*x\n")
    api.reset_caches()
    yield code
    api.reset_caches()


def result(source="A", *, fidelity="EQUIVALENT", state="SUPPORTED", claim=False, symbol="h"):
    return {"doc_id": "DOC_" + source, "target": "MODEL_TEST", "source_sha256": source,
            "reviewer": {"role": "F", "model": "fixture-reviewer"},
            "supports_model": {"state": state, "found": state == "SUPPORTED",
                               "equation_labels": ["Eq. (1)"] if fidelity == "EQUIVALENT" and state == "SUPPORTED" else [],
                               "comparison_equation_labels": ["Eq. (1)"]},
            "model_review": {"fidelity": fidelity, "implementation": [{"file": "pkg/physics.py", "symbol": symbol}]},
            "equations": [{"label": "Eq. (1)", "equation_number": "1", "page": 2, "quote": "The relation is h = x.",
                           "form": "h = x", "confidence": "high"}],
            "claims": [{"statement": "The source reports a constant coefficient.", "claim_type": "empirical_result",
                        "subject": "coefficient", "predicate": "is", "object": "constant", "page": 2,
                        "quote": "The source reports a constant coefficient.", "confidence": "high"}] if claim else []}


def model():
    return store.index_by("models")["MODEL_TEST"]


def edges(assembled, source, target):
    return {edge["type"] for edge in assembled["edges"] if edge["from"] == source and edge["to"] == target}


def test_secondary_source_keeps_primary_fingerprints_and_both_comparisons(sources):
    harvest_extraction.harvest([result()], [])
    first = copy.deepcopy(model()["source_review"])
    harvest_extraction.harvest([result("B", fidelity="DIFFERENT_FORMULA")], [])
    recorded = model()
    assert recorded["source_review"] == first
    assert recorded["source_support_sha256"] == "A"
    assert {review["reviewed_document_id"] for review in recorded["source_reviews"]} == {"DOC_A", "DOC_B"}
    eqs = {eq["document_id"]: eq for eq in store.load("equations")}
    assert set(recorded["source_comparison_equation_ids"]) == {eq["id"] for eq in eqs.values()}
    assembled = graph.assemble()
    assert edges(assembled, "MODEL_TEST", eqs["DOC_A"]["id"]) == {"IMPLEMENTS", "CITES"}
    assert edges(assembled, "MODEL_TEST", eqs["DOC_B"]["id"]) == {"CITES"}
    assert edges(assembled, "MODEL_TEST", "DOC_B") == {"CITES", "SUPPORTED_BY"}
    api.reset_caches()
    summary = api.model_review_summary(recorded)
    assert summary["state"] == "CURRENT" and summary["fidelity"] == "MIXED"
    assert summary["fidelities"] == ["DIFFERENT_FORMULA", "EQUIVALENT"]
    assert len(summary["source_reviews"]) == 2
    implementations = api.find_implementation(eqs["DOC_A"]["id"])["reviewed_implementations"]
    assert {item["reviewed_document_id"] for item in implementations} == {"DOC_A"}
    assert {item["fidelity"] for item in implementations} == {"EQUIVALENT"}


def test_negative_secondary_assessment_cannot_promote_or_erase_primary(sources):
    harvest_extraction.harvest([result(), result("B", fidelity="INSUFFICIENT_EVIDENCE", state="NOT_FOUND_IN_SOURCE")], [])
    recorded = model()
    assert recorded["source_support_state"] == "PARTIALLY_SUPPORTED"
    assert recorded["source_review"]["support_state"] == "SUPPORTED"
    assert {a["state"] for a in recorded["source_assessments"]} == {"SUPPORTED", "NOT_FOUND_IN_SOURCE"}
    assert graph.model_source_support(recorded, store.load("manifest")[0])[0] == "PARTIALLY_SUPPORTED"


def test_stale_secondary_and_changed_primary_remain_visible_independently(sources):
    harvest_extraction.harvest([result(), result("B", symbol="q")], [])
    docs = store.load("manifest")
    docs[1]["sha256"] = "changed"
    store.save("manifest", docs)
    api.reset_caches()
    summary = api.model_review_summary(model())
    assert summary["state"] == "STALE_SOURCE"
    assert {r["reviewed_document_id"]: r["state"] for r in summary["source_reviews"]} == {
        "DOC_A": "CURRENT", "DOC_B": "STALE_SOURCE"}
    assert graph.model_source_support(model(), docs[0])[0] == "PARTIALLY_SUPPORTED"
    assembled = graph.assemble()
    eqs = {eq["document_id"]: eq["id"] for eq in store.load("equations")}
    assert edges(assembled, "MODEL_TEST", eqs["DOC_A"]) == {"IMPLEMENTS", "CITES"}
    assert not edges(assembled, "MODEL_TEST", eqs["DOC_B"])
    sources.write_text("def h(x):\n    return x+1\n\ndef q(x):\n    return 2*x\n")
    api.reset_caches()
    assert api.model_review_summary(model())["state"] == "IMPLEMENTATION_CHANGED_REVIEW_REQUIRED"
    assert "IMPLEMENTS" not in edges(graph.assemble(), "MODEL_TEST", eqs["DOC_A"])


def test_replacing_one_source_review_retains_old_review_and_other_source(sources):
    harvest_extraction.harvest([result(), result("B")], [])
    old = copy.deepcopy(model()["source_review"])
    retained = copy.deepcopy(model()["source_reviews"][1])
    harvest_extraction.harvest([result(fidelity="DIFFERENT_FORMULA")], [])
    recorded = model()
    assert old in recorded["source_review_history"]
    assert retained in recorded["source_reviews"]
    assert recorded["source_review"]["fidelity"] == "DIFFERENT_FORMULA"
    eqs = {eq["document_id"]: eq["id"] for eq in store.load("equations")}
    assembled = graph.assemble()
    assert edges(assembled, "MODEL_TEST", eqs["DOC_A"]) == {"CITES"}
    assert edges(assembled, "MODEL_TEST", eqs["DOC_B"]) == {"IMPLEMENTS", "CITES"}
    before = json.dumps(recorded, sort_keys=True)
    harvest_extraction.harvest([result(fidelity="DIFFERENT_FORMULA")], [])
    assert json.dumps(model(), sort_keys=True) == before


@pytest.mark.parametrize("fidelity", ["DIFFERENT_FORMULA", "BAD_ATTRIBUTION", "INSUFFICIENT_EVIDENCE", "UNRESOLVED"])
def test_non_equivalent_claims_are_citations_including_previous_support(sources, fidelity):
    harvest_extraction.harvest([result(claim=True)], [])
    cid = store.load("claims")[0]["id"]
    assert edges(graph.assemble(), cid, "MODEL_TEST") == {"SUPPORTS"}
    # A review-only replacement must also demote previously banked claims.
    revised = result(fidelity=fidelity)
    revised["equations"] = []
    harvest_extraction.harvest([revised], [])
    claim = store.load("claims")[0]
    assert claim["model_ids"] == []
    assert claim["comparison_model_ids"] == ["MODEL_TEST"]
    assembled = graph.assemble()
    assert not edges(assembled, cid, "MODEL_TEST")
    assert edges(assembled, "MODEL_TEST", cid) == {"CITES"}
    # Banking a new claim under that review follows the same fidelity gate.
    harvest_extraction.harvest([result(fidelity=fidelity, claim=True)], [])
    assert store.load("claims")[0]["model_ids"] == []


@pytest.mark.parametrize("declared_relation", ["CITES_ONLY", "IMPLEMENTS"])
def test_declared_equation_link_disagreement_reports_bank_mismatch(sources, declared_relation):
    harvest_extraction.harvest([result()], [])
    eid = store.load("equations")[0]["id"]
    revised = result(fidelity="DIFFERENT_FORMULA" if declared_relation == "IMPLEMENTS" else "EQUIVALENT")
    revised["model_review"]["equation_links"] = [{"equation_id": eid, "relation": declared_relation}]
    report = harvest_extraction.harvest([revised], [])
    mismatch, = report["model_review_equation_link_mismatches"]
    assert mismatch["model_id"] == "MODEL_TEST" and mismatch["doc_id"] == "DOC_A"
    assert mismatch["declared_implements"] != mismatch["actual_implements"]


def test_declared_citation_set_must_match_current_source_only(sources):
    harvest_extraction.harvest([result(), result("B")], [])
    eqs = {eq["document_id"]: eq["id"] for eq in store.load("equations")}
    revised = result()
    revised["model_review"]["equation_links"] = [{"equation_id": eqs["DOC_A"], "relation": "IMPLEMENTS"}]
    report = harvest_extraction.harvest([revised], [])
    assert not report.get("model_review_equation_link_mismatches")
    revised["model_review"]["equation_links"].append({"equation_id": eqs["DOC_B"], "relation": "CITES_ONLY"})
    report = harvest_extraction.harvest([revised], [])
    mismatch, = report["model_review_equation_link_mismatches"]
    assert eqs["DOC_B"] in mismatch["declared_cites"] and eqs["DOC_B"] not in mismatch["actual_cites"]


def test_implementation_declaration_does_not_require_redundant_comparison_label(sources):
    harvest_extraction.harvest([result()], [])
    eid = store.load("equations")[0]["id"]
    revised = result()
    revised["supports_model"]["comparison_equation_labels"] = []
    revised["model_review"]["equation_links"] = [{"equation_id": eid, "relation": "IMPLEMENTS"}]
    report = harvest_extraction.harvest([revised], [])
    assert not report.get("model_review_equation_link_mismatches")


def test_unreviewed_replay_cannot_erase_comparisons_or_revive_claim_support(sources):
    reviewed = result(fidelity="DIFFERENT_FORMULA", claim=True)
    harvest_extraction.harvest([reviewed], [])
    before = copy.deepcopy(model()["source_review"])
    replay = result(claim=True)
    replay.pop("model_review")
    replay["supports_model"].pop("comparison_equation_labels")
    harvest_extraction.harvest([replay], [])
    assert model()["source_review"] == before
    eq = store.load("equations")[0]
    claim = store.load("claims")[0]
    assert model()["source_assessments"][0]["comparison_equation_ids"] == [eq["id"]]
    assert eq["model_ids"] == [] and claim["model_ids"] == []
    assembled = graph.assemble()
    assert edges(assembled, "MODEL_TEST", eq["id"]) == {"CITES"}
    assert edges(assembled, "MODEL_TEST", claim["id"]) == {"CITES"}
    assert not edges(assembled, claim["id"], "MODEL_TEST")


def test_legacy_different_review_blocks_existing_claim_support_before_rebank(sources):
    harvest_extraction.harvest([result(claim=True)], [])
    recorded = model()
    recorded.pop("source_reviews")
    recorded["source_review"]["fidelity"] = "DIFFERENT_FORMULA"
    store.save("models", [recorded])
    eq = store.load("equations")[0]
    claim = store.load("claims")[0]
    assert claim["model_ids"] == ["MODEL_TEST"]  # historical store is preserved
    assembled = graph.assemble()
    assert "IMPLEMENTS" not in edges(assembled, "MODEL_TEST", eq["id"])
    assert not edges(assembled, claim["id"], "MODEL_TEST")
    assert edges(assembled, "MODEL_TEST", claim["id"]) == {"CITES"}


def test_registry_edit_invalidates_each_source_until_individually_rereviewed(sources, monkeypatch):
    from dataclasses import replace
    from kg import harvest_registry
    from kg.schema import node_id
    from raosim import model_registry

    original = model_registry.MODEL_REGISTRY["injector.incompressible_orifice"]
    mid = node_id("MODEL", original.model_id)
    initial = model()
    initial.update(id=mid, registry_id=original.model_id)
    store.save("models", [initial])
    for source in ("A", "B"):
        review = result(source)
        review["target"] = mid
        harvest_extraction.harvest([review], [])
    changed = replace(original, relation="updated registry relation", local_source="A.pdf")
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {original.model_id: changed})
    harvest_registry.harvest()
    recorded = store.load("models")[0]
    assert set(recorded["source_registry_review_required"]) == {"DOC_A", "DOC_B"}
    assert graph.model_source_support(recorded, store.load("manifest")[0])[0] == "NOT_CHECKED"
    assert harvest_extraction.model_review_state(recorded)["state"] == "NOT_CHECKED"
    renewed = result()
    renewed["target"] = mid
    harvest_extraction.harvest([renewed], [])
    recorded = store.load("models")[0]
    assert set(recorded["source_registry_review_required"]) == {"DOC_B"}
    assert graph.model_source_support(recorded, store.load("manifest")[0])[0] == "PARTIALLY_SUPPORTED"
    states = {r["reviewed_document_id"]: r["state"] for r in harvest_extraction.model_review_state(recorded)["source_reviews"]}
    assert states == {"DOC_A": "CURRENT", "DOC_B": "NOT_CHECKED"}


def test_secondary_verification_override_does_not_poison_primary_source(sources):
    from kg import verification_freshness

    harvest_extraction.harvest([result(), result("B")], [])
    eqs = {row["document_id"]: row for row in store.load("equations")}
    secondary = eqs["DOC_B"]
    verdict = {"object_id": secondary["id"], "refuted": True, "confidence": "high",
               "source_sha256": "B", "content_sha256": verification_freshness.content_sha256(secondary),
               "model_support_override": "NOT_FOUND_IN_SOURCE", "reason": "Secondary attribution refuted",
               "reviewer": {"role": "A", "model": "independent-fixture"}}
    assert harvest_extraction.apply_verification([verdict])["refuted"] == 1
    recorded = model()
    states = {row["document_id"]: row["state"] for row in harvest_extraction.model_source_assessments(recorded)}
    assert states == {"DOC_A": "SUPPORTED", "DOC_B": "NOT_FOUND_IN_SOURCE"}
    assert recorded["source_support_state"] == "PARTIALLY_SUPPORTED"
    assert "source_support_state_verified" not in recorded
    assert recorded["refuted_equation_ids"] == [secondary["id"]]
    assert store.index_by("equations")[eqs["DOC_A"]["id"]]["model_ids"] == ["MODEL_TEST"]
    assembled = graph.assemble()
    assert edges(assembled, "MODEL_TEST", "DOC_A") == {"SUPPORTED_BY", "CITES"}
    assert edges(assembled, "MODEL_TEST", "DOC_B") == {"CITES"}
    replay = result("B")
    replay.pop("model_review")
    harvest_extraction.harvest([replay], [])
    assert model()["source_support_state"] == "PARTIALLY_SUPPORTED"
    assert {row["document_id"]: row["state"] for row in harvest_extraction.model_source_assessments(model())} == states


def test_primary_override_keeps_legacy_fields_and_secondary_assessment(sources):
    from kg import verification_freshness

    harvest_extraction.harvest([result(), result("B")], [])
    primary = next(row for row in store.load("equations") if row["document_id"] == "DOC_A")
    verdict = {"object_id": primary["id"], "refuted": True, "confidence": "high",
               "source_sha256": "A", "content_sha256": verification_freshness.content_sha256(primary),
               "model_support_override": "NOT_FOUND_IN_SOURCE", "reviewer": {"role": "A", "model": "auditor"}}
    harvest_extraction.apply_verification([verdict])
    recorded = model()
    assert recorded["source_support_state_verified"] == "NOT_FOUND_IN_SOURCE"
    assert recorded["source_support_verifier_document_id"] == "DOC_A"
    assert recorded["source_support_verifier_source_sha256"] == "A"
    assert {row["document_id"]: row["state"] for row in harvest_extraction.model_source_assessments(recorded)} == {
        "DOC_A": "NOT_FOUND_IN_SOURCE", "DOC_B": "SUPPORTED"}
    assert recorded["source_support_state"] == "PARTIALLY_SUPPORTED"


def test_verification_override_requires_matching_document_and_source_fingerprint(sources):
    harvest_extraction.harvest([result(), result("B")], [])
    recorded = model()
    recorded.update(source_support_state_verified="NOT_FOUND_IN_SOURCE",
                    source_support_verifier_document_id="DOC_B", source_support_verifier_source_sha256="B")
    recorded["source_support_verification_overrides"] = [
        {"document_id": "DOC_B", "source_sha256": "old-B", "state": "NOT_FOUND_IN_SOURCE"}]
    assert {row["document_id"]: row["state"] for row in harvest_extraction.model_source_assessments(recorded)} == {
        "DOC_A": "SUPPORTED", "DOC_B": "SUPPORTED"}
    recorded["source_support_verifier_document_id"] = "DOC_A"
    recorded["source_support_verifier_source_sha256"] = "old-A"
    assert harvest_extraction.model_source_assessments(recorded)[0]["state"] == "SUPPORTED"


def test_model_completion_distinguishes_bound_verdicts_and_excludes_refuted_stale_support(sources, monkeypatch):
    from kg import completion, verification_freshness

    harvest_extraction.harvest([result()], [])
    recorded = model()
    recorded.pop("source_review")
    recorded.pop("source_reviews")
    store.save("models", [recorded])
    base = store.load("equations")[0]
    rows = []
    for tag in ("BOUND", "LEGACY", "CONTENT_CHANGED", "REFUTED", "STALE", "MISSING"):
        equation = {**base, "id": "EQ_" + tag, "verification": "CONFIRMED", "verifier_source_sha256": "A"}
        if tag != "LEGACY":
            equation["verifier_content_sha256"] = verification_freshness.content_sha256(equation)
        if tag == "CONTENT_CHANGED":
            equation["description"] = "Changed after its verdict"
        elif tag == "REFUTED":
            equation["verification"] = "REFUTED"
        elif tag == "STALE":
            equation["source_sha256"] = "old-A"
        elif tag == "MISSING":
            equation["document_id"] = "DOC_MISSING"
        rows.append(equation)
    store.save("equations", rows)
    monkeypatch.setattr(completion, "_inputs", lambda: {})
    monkeypatch.setattr(completion, "_journals_by_target", lambda: {})
    audit = completion.model_source_audit()["models"][0]
    assert audit["supporting_equation_links"] == ["EQ_BOUND", "EQ_CONTENT_CHANGED", "EQ_LEGACY"]
    assert audit["supporting_equation_links_confirmed"] == audit["supporting_equation_links"]
    assert "Historical stored CONFIRMED" in audit["supporting_equation_links_confirmed_basis"]
    assert audit["supporting_equation_links_content_bound_confirmed"] == ["EQ_BOUND"]
