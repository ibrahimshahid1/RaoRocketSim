"""Only explicit relations between uniquely current source records are banked."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_extraction, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    # These tests exercise relation resolution only, not source extraction.
    monkeypatch.setattr(harvest_extraction, "harvest", lambda models, windows: {})
    store.save("manifest", [{"id": "DOC_A", "sha256": "source-a"},
                             {"id": "DOC_B", "sha256": "source-b"}])


def equation(ident, label, *, document="DOC_A", **extra):
    return {"id": ident, "type": "Equation", "label": label,
            "document_id": document, "source_sha256": "source-a" if document == "DOC_A" else "source-b",
            "source_state": "CURRENT", "active": True, "grounded": "GROUNDED",
            "provenance_id": "PROV_" + ident, "provenance_kind": "SOURCE_STATES", **extra}


def explicit(*, relation="DERIVED_FROM", child_doc="DOC_A"):
    return [{"group": "explicit source derivation",
             "cross_document_relations": [{"a_doc_id": child_doc, "a_label": "Reduced relation",
                                           "b_doc_id": "DOC_A", "b_label": "Parent relation",
                                           "relation": relation,
                                           "note": "Reviewer explicitly identified this derivation; assumptions remain on the source records."}]}]


def test_explicit_derivation_is_directional_and_keeps_source_provenance(isolated):
    original = [equation("EQ_PARENT", "Parent relation"), equation("EQ_CHILD", "Reduced relation")]
    store.save("equations", original)
    for _ in range(2):
        result = harvest_extraction.harvest_groups(explicit())
        assert result["cross_document_relations"] == {"attached": 1, "unresolved": []}
    stored = {r["id"]: r for r in store.load("equations")}
    assert stored["EQ_CHILD"]["derived_from"] == ["EQ_PARENT"]
    assert "derived_from" not in stored["EQ_PARENT"]
    assert stored["EQ_CHILD"]["provenance_id"] == "PROV_EQ_CHILD"
    assert stored["EQ_PARENT"]["provenance_kind"] == "SOURCE_STATES"
    relation, = store.load("relationships")
    assert (relation["from"], relation["edge"], relation["to"]) == ("EQ_CHILD", "DERIVED_FROM", "EQ_PARENT")
    assert relation["source_relation"] == "DERIVED_FROM"
    assert relation["provenance_kind"] == "SYSTEM_INFERS"
    assert relation["note"].startswith("Reviewer explicitly")


@pytest.mark.parametrize("history", [
    {"active": False},
    {"source_state": "STALE_SOURCE"},
    {"source_state": "SUPERSEDED"},
    {"verification": "REFUTED"},
    {"source_sha256": "replaced-source"},
    {"source_sha256": None},
    {"source_state": None},
    {"grounded": "UNGROUNDED"},
])
def test_history_cannot_shadow_current_equation_label(isolated, history):
    store.save("equations", [equation("EQ_PARENT", "Parent relation"),
                             equation("EQ_CHILD_CURRENT", "Reduced relation"),
                             equation("EQ_Z_HISTORY", "Reduced relation", **history)])
    result = harvest_extraction.harvest_groups(explicit())
    assert result["cross_document_relations"]["attached"] == 1
    relation, = store.load("relationships")
    assert relation["from"] == "EQ_CHILD_CURRENT"
    history_record = next(r for r in store.load("equations") if r["id"] == "EQ_Z_HISTORY")
    assert "derived_from" not in history_record


def test_ambiguous_current_labels_are_reported_not_arbitrarily_selected(isolated):
    store.save("equations", [equation("EQ_PARENT", "Parent relation"),
                             equation("EQ_CHILD_ONE", "Reduced relation"),
                             equation("EQ_CHILD_TWO", "Reduced relation")])
    result = harvest_extraction.harvest_groups(explicit())
    assert result["cross_document_relations"]["attached"] == 0
    assert "uniquely resolve" in result["cross_document_relations"]["unresolved"][0]["reason"]
    assert store.load("relationships") == []


def test_missing_manifest_source_cannot_establish_derivation(isolated):
    store.save("equations", [equation("EQ_PARENT", "Parent relation"),
                             equation("EQ_CHILD", "Reduced relation", document="DOC_MISSING")])
    result = harvest_extraction.harvest_groups(explicit(child_doc="DOC_MISSING"))
    assert result["cross_document_relations"]["attached"] == 0
    assert store.load("relationships") == []


def test_no_derivation_is_inferred_without_explicit_relation(isolated):
    store.save("equations", [equation("EQ_PARENT", "Parent relation"), equation("EQ_CHILD", "Reduced relation")])
    result = harvest_extraction.harvest_groups([{"group": "source equations only"}])
    assert result["cross_document_relations"] == {"attached": 0, "unresolved": []}
    assert store.load("relationships") == []
    assert all("derived_from" not in r for r in store.load("equations"))


def test_existing_symmetric_comparison_retains_its_contract(isolated):
    store.save("equations", [equation("EQ_PARENT", "Parent relation"),
                             equation("EQ_CHILD", "Reduced relation", document="DOC_B")])
    result = harvest_extraction.harvest_groups(explicit(relation="EQUIVALENT_TO", child_doc="DOC_B"))
    assert result["cross_document_relations"]["attached"] == 1
    records = {r["id"]: r for r in store.load("equations")}
    assert records["EQ_CHILD"]["equivalent_to"] == ["EQ_PARENT"]
    assert records["EQ_PARENT"]["equivalent_to"] == ["EQ_CHILD"]
