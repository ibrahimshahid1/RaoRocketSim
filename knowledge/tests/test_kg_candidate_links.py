"""K8: source objects preserve raw parser identity without promoting parser authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import graph, harvest_extraction, store  # noqa: E402

REVIEWER = {"role": "F", "model": "synthetic-reviewer"}
COLLECTIONS = {
    "equations": ("equations", "Equation", {"label": "Sample line", "form": "x=3"}),
    "claims": ("claims", "Claim", {"statement": "The sample is three", "subject": "sample"}),
    "validity_statements": ("validity", "ValidityEnvelope", {"statement": "Sample only", "applies_to": "sample"}),
    "empirical_constants": ("entities", "EmpiricalConstant", {"name": "Sample value", "value": "3", "units": "m"}),
    "assumptions": ("entities", "Assumption", {"statement": "The sample uses three"}),
}


@pytest.fixture
def candidate_env(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(graph, "GRAPH_DIR", root / "graph")

    class Grounder:
        chunks = [{"chunk_id": "DOC_TEST#c1"}]

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "DOC_TEST#c1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "_current_grounder", lambda doc, sha: Grounder())
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "current-source", "source_pdf": "source.pdf",
                             "filename": "source.pdf", "conversion": {"status": "good"}}])
    candidate = {"id": "CAND_RAW", "type": "ExtractionCandidate", "candidate_kind": "formula",
                 "document_id": "DOC_TEST", "source_sha256": "current-source", "source_state": "CURRENT",
                 "pages": [3], "location_state": "LOCATED", "content": "x = [damaged notation]",
                 "active": False, "review_state": "REJECTED", "review_reason": "Synthetic parser error"}
    store.save("enrichments", [candidate])
    return candidate


def extraction(collection="empirical_constants", *, field="corrects_candidate_ids", reviewer=REVIEWER):
    return {"doc_id": "DOC_TEST", "target": "pages3-3", "source_sha256": "current-source",
            **({"reviewer": reviewer} if reviewer else {}),
            collection: [{**COLLECTIONS[collection][2], "page": 3, "quote": "The source sample value is three.",
                          field: ["CAND_RAW"]}]}


def stored_object(collection):
    store_name, kind, _ = COLLECTIONS[collection]
    return next(row for row in store.load(store_name) if row["type"] == kind)


@pytest.mark.parametrize("collection", COLLECTIONS)
def test_rejected_raw_candidate_is_preserved_as_correction_history(candidate_env, collection):
    original = deepcopy(candidate_env)
    report = harvest_extraction.harvest([], [extraction(collection)])
    assert not report.get("unresolved_candidate_links")
    record = stored_object(collection)
    assert record["corrects_candidate_ids"] == ["CAND_RAW"]
    assert store.load("enrichments") == [original]
    edges = [edge for edge in graph.assemble()["edges"]
             if edge["from"] == record["id"] and edge["to"] == "CAND_RAW"]
    assert [edge["type"] for edge in edges] == ["CORRECTS"]
    assert edges[0]["provenance"] == record["provenance_id"]
    assert not record.get("verification")


@pytest.mark.parametrize("collection", COLLECTIONS)
def test_signed_correction_can_precede_same_packet_parser_review(candidate_env, collection):
    candidate = {**candidate_env, "active": True, "review_state": "UNREVIEWED"}
    store.save("enrichments", [candidate])
    report = harvest_extraction.harvest([], [extraction(collection)])
    assert not report.get("unresolved_candidate_links")
    assert stored_object(collection)["corrects_candidate_ids"] == ["CAND_RAW"]
    assert store.load("enrichments") == [candidate]  # no inferred parser verdict


@pytest.mark.parametrize("change,reason", [
    ({"document_id": "DOC_OTHER"}, "CANDIDATE_DOCUMENT_MISMATCH"),
    ({"source_sha256": "old-source"}, "CANDIDATE_SOURCE_MISMATCH"),
    ({"source_state": "STALE_SOURCE"}, "CANDIDATE_SOURCE_NONCURRENT"),
    ({"pages": [4]}, "CANDIDATE_PAGE_MISMATCH"),
    ({"location_state": "UNRESOLVED_PAGE"}, "CANDIDATE_PAGE_MISMATCH"),
    ({"candidate_kind": ""}, "UNKNOWN_PARSER_CANDIDATE"),
])
def test_candidate_identity_mismatch_is_refused_and_reported(candidate_env, change, reason):
    store.save("enrichments", [{**candidate_env, **change}])
    report = harvest_extraction.harvest([], [extraction()])
    record = stored_object("empirical_constants")
    assert record["corrects_candidate_ids"] == []
    assert report["unresolved_candidate_links"][0]["reason"] == reason
    assert report["unresolved_candidate_links"][0]["candidate_id"] == "CAND_RAW"
    assert not [edge for edge in graph.assemble()["edges"] if edge["from"] == record["id"] and edge["to"] == "CAND_RAW"]


@pytest.mark.parametrize("reviewer", [None, {"role": "S3", "model": "worker"}])
def test_candidate_link_requires_named_original_page_reviewer(candidate_env, reviewer):
    report = harvest_extraction.harvest([], [extraction(reviewer=reviewer)])
    assert report["unresolved_candidate_links"][0]["reason"] == "CANDIDATE_LINK_REQUIRES_ORIGINAL_PAGE_REVIEWER"
    assert stored_object("empirical_constants")["corrects_candidate_ids"] == []


def test_parser_qa_cannot_authorize_unsigned_corrected_source_value(candidate_env):
    store.save("enrichments", [{**candidate_env, "review_evidence": {"reviewer": REVIEWER, "pdf_page": 3}}])
    report = harvest_extraction.harvest([], [extraction(reviewer=None)])
    assert report["unresolved_candidate_links"][0]["reason"] == "CANDIDATE_LINK_REQUIRES_ORIGINAL_PAGE_REVIEWER"
    assert stored_object("empirical_constants")["corrects_candidate_ids"] == []


@pytest.mark.parametrize("source_sha", [None, "older-reviewed-source"])
def test_link_bearing_extraction_must_identify_current_reviewed_source(candidate_env, source_sha):
    result = extraction()
    if source_sha is None:
        result.pop("source_sha256")
    else:
        result["source_sha256"] = source_sha
    report = harvest_extraction.harvest([], [result])
    assert report["unresolved_candidate_links"][0]["reason"] == "EXTRACTION_SOURCE_MISMATCH"
    assert stored_object("empirical_constants")["corrects_candidate_ids"] == []


def test_rejected_candidate_cannot_be_derived_support(candidate_env):
    report = harvest_extraction.harvest([], [extraction(field="enrichment_candidate_ids")])
    assert report["unresolved_candidate_links"][0]["reason"] == "REJECTED_CANDIDATE_IS_CORRECTION_HISTORY_ONLY"
    assert stored_object("empirical_constants")["enrichment_candidate_ids"] == []


def test_active_reviewed_candidate_can_be_used_with_same_provenance(candidate_env):
    store.save("enrichments", [{**candidate_env, "active": True, "review_state": "SOURCE_PAGE_COMPARED"}])
    report = harvest_extraction.harvest([], [extraction(field="enrichment_candidate_ids")])
    assert not report.get("unresolved_candidate_links")
    constant = stored_object("empirical_constants")
    edge, = [edge for edge in graph.assemble()["edges"] if edge["from"] == constant["id"] and edge["to"] == "CAND_RAW"]
    assert edge["type"] == "DERIVED_FROM" and edge["provenance"] == constant["provenance_id"]


@pytest.mark.parametrize("change", [{"pages": [4]}, {"source_sha256": "old-source"}, {"source_state": "STALE_SOURCE"}])
def test_graph_rechecks_candidate_identity_without_erasing_stored_history(candidate_env, change):
    harvest_extraction.harvest([], [extraction()])
    constant = stored_object("empirical_constants")
    store.save("enrichments", [{**candidate_env, **change}])
    assert store.index_by("entities")[constant["id"]]["corrects_candidate_ids"] == ["CAND_RAW"]
    assembled = graph.assemble()
    assert not [edge for edge in assembled["edges"] if edge["from"] == constant["id"] and edge["to"] == "CAND_RAW"]
    assert assembled["meta"]["n_candidate_link_issues"] == 1
    assert assembled["candidate_link_issues"][0]["object_id"] == constant["id"]


def test_candidate_link_schema_is_available_for_every_source_object():
    schema_path = Path(__file__).resolve().parents[1] / "workflows/extraction.schema.json"
    schema = json.loads(schema_path.read_text())
    for collection in COLLECTIONS:
        properties = schema["properties"][collection]["items"]["properties"]
        assert properties["enrichment_candidate_ids"]["type"] == "array"
        assert properties["corrects_candidate_ids"]["items"]["type"] == "string"
