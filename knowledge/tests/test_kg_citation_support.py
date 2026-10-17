"""A citation is a route to inspect, never automatic scientific support."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import api, graph, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(graph, "GRAPH_DIR", tmp_path / "knowledge" / "graph")
    api.reset_caches()
    yield
    api.reset_caches()


def build(*, support=None, reviewed_sha=None, source_sha="current", source_state=None, document=True):
    docs = [{"id": "DOC_TEST", "filename": "test.pdf", "source_pdf": "propulsion_texts/test.pdf",
             "sha256": source_sha, "conversion": {"status": "good"}}] if document else []
    model = {"id": "MODEL_TEST", "registry_id": "fixture.model", "subsystem": "nozzle",
             "quantity": "flow", "relation": "mass conservation", "basis_class": "LITERATURE_DERIVED",
             "promotion_status": "screening", "validation_level": "not_physically_validated",
             "registry_status": "available", "source_document_id": "DOC_TEST",
             "source_support_notes": "The reviewed source covers only the stated scope."}
    if support is not None:
        model["source_support_state"] = support
    if reviewed_sha is not None:
        model["source_support_sha256"] = reviewed_sha
    if source_state is not None:
        model["source_state"] = source_state
    store.save("manifest", docs)
    store.save("models", [model])
    store.save("software_links", [{"id": "SW_TEST", "type": "SoftwareComponent",
                                    "qualname": "fixture:flow", "file": "fixture.py", "line": 1,
                                    "model_ids": ["MODEL_TEST"], "provenance_id": "PROV_CODE"}])
    result = graph.assemble()
    api.reset_caches()
    return result


@pytest.mark.parametrize("support", [None, "NOT_CHECKED", "NOT_FOUND_IN_SOURCE", "SOURCE_UNREADABLE"])
def test_unreviewed_or_unsupported_citation_is_preserved_without_support(isolated, support):
    assembled = build(support=support)
    edges = [e for e in assembled["edges"] if e["from"] == "MODEL_TEST" and e["to"] == "DOC_TEST"]
    assert {e["type"] for e in edges} == {"CITES"}
    result = api.find_scientific_basis("fixture.py")
    component, = result["components"]
    assert component["state"] == (support or "NOT_CHECKED")
    assert result["basis_state"] == component["state"]
    assert component["source_documents"][0]["id"] == "DOC_TEST"
    assert component["supported_source_documents"] == []
    assert component["model_support"][0]["notes"]


@pytest.mark.parametrize("support, expected", [("SUPPORTED", "OK"), ("PARTIALLY_SUPPORTED", "PARTIALLY_SUPPORTED")])
def test_current_review_keeps_its_scope_and_partial_status(isolated, support, expected):
    assembled = build(support=support, reviewed_sha="current")
    edge, = [e for e in assembled["edges"] if e["type"] == "SUPPORTED_BY"]
    assert edge["support_state"] == support
    assert edge["source_sha256"] == "current"
    assert edge["support_notes"]
    assert "not implementation or physical validation" in edge["scope"]
    component, = api.find_scientific_basis("fixture.py")["components"]
    assert component["state"] == expected
    assert component["supported_source_documents"][0]["id"] == "DOC_TEST"
    assert api.get_document("DOC_TEST")["cited_by_models"] == ["MODEL_TEST"]
    assert api.get_document("DOC_TEST")["cited_by_software"] == []


@pytest.mark.parametrize("kwargs, expected", [
    ({}, "NOT_CHECKED"),
    ({"reviewed_sha": "previous"}, "STALE_SOURCE"),
    ({"reviewed_sha": "current", "source_state": "STALE_SOURCE"}, "STALE_SOURCE"),
    ({"reviewed_sha": "current", "document": False}, "SOURCE_NOT_INGESTED"),
])
def test_support_requires_current_source_identity(isolated, kwargs, expected):
    assembled = build(support="SUPPORTED", **kwargs)
    assert not [e for e in assembled["edges"] if e["type"] == "SUPPORTED_BY"]
    result = api.find_scientific_basis("fixture.py")
    assert result["basis_state"] == expected
    component, = result["components"]
    assert component["model_support"][0]["declared_state"] == "SUPPORTED"
    assert component["source_documents"][0]["id"] == "DOC_TEST"


def test_direct_model_lookup_and_pre_rebuild_graph_do_not_hide_stale_review(isolated):
    build(support="PARTIALLY_SUPPORTED", reviewed_sha="current")
    for ref in ("MODEL_TEST", "fixture.model"):
        component, = api.find_scientific_basis(ref)["components"]
        assert component["models"][0]["id"] == "MODEL_TEST"
        assert component["state"] == "PARTIALLY_SUPPORTED"
    doc, = store.load("manifest")
    store.save("manifest", [{**doc, "sha256": "replaced"}])
    api.reset_caches()
    component, = api.find_scientific_basis("fixture.py")["components"]
    assert component["state"] == "STALE_SOURCE"
    assert component["supported_source_documents"] == []
    assert component["source_documents"][0]["id"] == "DOC_TEST"


def test_source_comparison_links_are_citations_without_implementation_claims(isolated):
    build(support="PARTIALLY_SUPPORTED", reviewed_sha="current")
    model, = store.load("models")
    store.save("models", [{**model, "source_comparison_equation_ids": ["EQ_REFERENCE", "EQ_OTHER_SOURCE"]}])
    store.save("equations", [{"id": ident, "label": ident, "form": "reference relation",
                              "document_id": "DOC_TEST", "source_sha256": sha}
                             for ident, sha in [("EQ_REFERENCE", "current"), ("EQ_OTHER_SOURCE", "older")]])
    edges = [e for e in graph.assemble()["edges"] if e["from"] == "MODEL_TEST" and e["to"].startswith("EQ_")]
    assert [(e["to"], e["type"]) for e in edges] == [("EQ_REFERENCE", "CITES")]
    assert "no IMPLEMENTS" in edges[0]["scope"]


@pytest.mark.parametrize("support", ["SUPPORTED", "PARTIALLY_SUPPORTED"])
@pytest.mark.parametrize("assessment_state, expected", [
    ({"active": False}, "NOT_CHECKED"),
    ({"source_state": "SUPERSEDED_EXTRACTION"}, "NOT_CHECKED"),
    ({"source_state": "REMOVED"}, "NOT_CHECKED"),
    ({"source_state": None}, "NOT_CHECKED"),
    ({"active": False, "source_state": "STALE_SOURCE"}, "STALE_SOURCE"),
])
def test_retired_model_review_cannot_supply_positive_support_even_with_matching_fingerprint(
        isolated, support, assessment_state, expected):
    build(support=support, reviewed_sha="current")
    model, = store.load("models")
    model.update(assessment_state)
    model["source_support_history"] = [{"source_support_state": support, "source_support_sha256": "current"}]
    store.save("models", [model])
    state, reason = graph.model_source_support(model, store.load("manifest")[0])
    assert state == expected
    assert "inactive or noncurrent" in reason if expected == "NOT_CHECKED" else "stale" in reason
    assembled = graph.assemble()
    model_node = next(n for n in assembled["nodes"] if n["id"] == "MODEL_TEST")
    assert model_node["source_support_state"] == support  # declared history remains inspectable
    assert model_node["effective_source_support_state"] == expected
    if "active" in assessment_state:
        assert model_node["active"] is assessment_state["active"]
    doc_edges = [e for e in assembled["edges"] if e["from"] == "MODEL_TEST" and e["to"] == "DOC_TEST"]
    assert {e["type"] for e in doc_edges} == {"CITES"}
    assert doc_edges[0]["support_state"] == expected
    assert store.load("models") == [model]


@pytest.mark.parametrize("support", ["SUPPORTED", "PARTIALLY_SUPPORTED"])
def test_explicit_current_active_model_review_remains_positive(isolated, support):
    build(support=support, reviewed_sha="current", source_state="CURRENT")
    model, = store.load("models")
    model["active"] = True
    store.save("models", [model])
    assert graph.model_source_support(model, store.load("manifest")[0])[0] == support
    doc_edges = [e for e in graph.assemble()["edges"] if e["from"] == "MODEL_TEST" and e["to"] == "DOC_TEST"]
    assert {e["type"] for e in doc_edges} == {"CITES", "SUPPORTED_BY"}


def test_ambiguous_glyph_does_not_assert_two_physical_dependencies(isolated):
    build(support="NOT_CHECKED")
    store.save("quantities", [{"qid": ident, "canonical": name, "dimension": dimension,
                               "si_unit": unit, "symbols": ["g"], "subsystems": []}
                              for ident, name, dimension, unit in [
                                  ("QTY_GAMMA", "specific-heat ratio", "1", "-"),
                                  ("QTY_GRAVITY", "gravitational acceleration", "L T^-2", "m/s^2")]])
    variable = {"symbol": "g", "role": "input", "resolution_state": "AMBIGUOUS_SYMBOL",
                "quantity_candidates": ["QTY_GAMMA", "QTY_GRAVITY"]}
    store.save("equations", [{"id": "EQ_AMBIGUOUS", "form": "source relation containing g",
                              "document_id": "DOC_TEST", "variables": [variable]}])
    assembled = graph.assemble()
    assert not [e for e in assembled["edges"] if e["from"] == "EQ_AMBIGUOUS"
                and e["type"] in {"INPUT_VARIABLE", "OUTPUT_VARIABLE"}]
    node = next(n for n in assembled["nodes"] if n["id"] == "EQ_AMBIGUOUS")
    assert node["unresolved_variables"] == [variable]


def test_incomplete_formula_stays_inspectable_without_implementation_support(isolated, monkeypatch):
    build(support="NOT_CHECKED")
    store.save("equations", [{"id": "EQ_DAMAGED", "label": "damaged formula", "form": "OCR-garbled relation NOT RENDERED",
                              "document_id": "DOC_TEST", "page": 2, "grounded": "GROUNDED",
                              "model_ids": ["MODEL_TEST"], "provenance_id": "PROV_EQ"},
                             {"id": "EQ_RECOVERED", "label": "recovered formula", "form": "a = b",
                              "extraction_note": "Previously NOT RENDERED; recovered from original PDF.",
                              "document_id": "DOC_TEST", "page": 3, "provenance_id": "PROV_EQ2"}])
    software = store.load("software_links")
    software[0]["equation_ids"] = ["EQ_DAMAGED"]
    software += [{"id": "LEGACY", "type": "SoftwareLink", "from": "EQ_DAMAGED", "to": "SW_TEST",
                  "edge": "IMPLEMENTED_IN", "provenance": "legacy"},
                 {"id": "LEGACY_SUPPORT", "type": "SoftwareLink", "from": "MODEL_TEST", "to": "DOC_TEST",
                  "edge": "SUPPORTED_BY", "provenance": "PROV_REGISTRY_LEGACY"}]
    store.save("software_links", software)
    assembled = graph.assemble()
    assert not [e for e in assembled["edges"] if e["type"] in {"IMPLEMENTS", "IMPLEMENTED_IN"}
                and "EQ_DAMAGED" in (e["from"], e["to"])]
    assert any(e["from"] == "EQ_DAMAGED" and e["type"] == "FOUND_IN" for e in assembled["edges"])
    assert not [e for e in assembled["edges"] if e["type"] == "SUPPORTED_BY"]
    api.reset_caches()
    found = api.find_equations("damaged")
    assert found["state"] == "INCOMPLETE_TRANSCRIPTION"
    assert api.get_equation("EQ_DAMAGED")["transcription_state"] == "INCOMPLETE_TRANSCRIPTION"
    assert api.get_equation("EQ_RECOVERED")["transcription_state"] == "NOT_FLAGGED_INCOMPLETE"
    monkeypatch.setattr(api, "semantic_search", lambda *a, **kw: {
        "hits": [], "retrieval": {"semantic": {"state": "OK"}}})
    context = api.retrieve_for_task("damaged formula")
    assert context["equations"][0]["transcription_state"] == "INCOMPLETE_TRANSCRIPTION"
    assert any("INCOMPLETE_TRANSCRIPTION" in w for w in context["implementation_warnings"])


def test_inactive_parameter_history_cannot_reenter_current_graph_or_context(isolated):
    build(support="NOT_CHECKED")
    software = store.load("software_links")
    software[0]["constant_ids"] = ["CONST_CURRENT", "CONST_OLD", "CONST_STALE"]
    software += [{"id": ident, "type": "EmpiricalConstant", "name": "motor density", "value": 1,
                  "component_id": "SW_TEST", "active": active, "source_state": state}
                 for ident, active, state in [("CONST_CURRENT", True, "CURRENT"),
                                               ("CONST_OLD", False, "CURRENT"),
                                               ("CONST_STALE", True, "STALE_SOURCE")]]
    software += [{"id": "LINK_OLD", "type": "SoftwareLink", "from": "SW_TEST", "to": "DOC_TEST",
                  "edge": "CITES", "provenance": "PROV_OLD", "active": False}]
    store.save("software_links", software)
    assembled = graph.assemble()
    assert {e["to"] for e in assembled["edges"] if e["type"] == "USES"} == {"CONST_CURRENT"}
    assert any(n["id"] == "CONST_OLD" and n["active"] is False for n in assembled["nodes"])
    assert not [e for e in assembled["edges"] if e["provenance"] == "PROV_OLD"]
    api.reset_caches()
    assert set(api._constants()) == {"CONST_CURRENT"}
