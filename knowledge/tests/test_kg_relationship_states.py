"""Retired relation assertions remain history even when endpoints stay current."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import graph, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(graph, "GRAPH_DIR", tmp_path / "graph")
    store.save("manifest", [{"id": "DOC_TEST", "filename": "source.pdf", "source_pdf": "source.pdf",
                             "sha256": "source", "conversion": {"status": "good"}}])
    base = {"type": "Equation", "document_id": "DOC_TEST", "source_sha256": "source",
            "source_state": "CURRENT", "active": True, "verification": "CONFIRMED",
            "grounded": "GROUNDED", "form": "x = y", "provenance_id": "PROV_TEST"}
    # Legacy cached lists deliberately retain the retired fact. They are not
    # authoritative substitutes for the explicit relationship's current state.
    store.save("equations", [
        {**base, "id": "EQ_CHILD", "label": "Child", "derived_from": ["EQ_PARENT"],
         "equivalent_to": ["EQ_PARENT"], "approximates": ["EQ_PARENT"]},
        {**base, "id": "EQ_PARENT", "label": "Parent", "equivalent_to": ["EQ_CHILD"]},
    ])


def relationship(edge, **state):
    return {"id": "REL_TEST", "type": "Relationship", "from": "EQ_CHILD", "to": "EQ_PARENT",
            "edge": edge, "source_relation": edge, "provenance_kind": "SYSTEM_INFERS",
            "note": "Explicit source comparison", "legacy_ids": ["REL_LEGACY"],
            "history": [{"id": "REL_LEGACY", "note": "Retained earlier assertion"}], **state}


@pytest.mark.parametrize("edge", ["DERIVED_FROM", "EQUIVALENT_TO", "APPROXIMATES"])
@pytest.mark.parametrize("state", [
    {"active": False},
    {"source_state": "STALE_SOURCE"},
    {"source_state": "SUPERSEDED_EXTRACTION"},
    {"verification": "REFUTED"},
])
def test_retired_relation_is_suppressed_without_erasing_history_or_cached_lists(isolated, edge, state):
    original = relationship(edge, **state)
    store.save("relationships", [original])
    equations_before = store.load("equations")
    result = graph.assemble()
    assert {n["id"] for n in result["nodes"]} >= {"EQ_CHILD", "EQ_PARENT"}
    assert not [e for e in result["edges"] if e["type"] == edge
                and {e["from"], e["to"]} == {"EQ_CHILD", "EQ_PARENT"}]
    assert store.load("relationships") == [original]
    assert store.load("equations") == equations_before


@pytest.mark.parametrize("edge", ["DERIVED_FROM", "EQUIVALENT_TO", "APPROXIMATES"])
@pytest.mark.parametrize("state", [{}, {"active": True, "source_state": "CURRENT", "verification": "CONFIRMED"}])
def test_current_and_legacy_default_relation_states_keep_exact_directed_edge(isolated, edge, state):
    original = relationship(edge, **state)
    store.save("relationships", [original])
    result = graph.assemble()
    matching = [e for e in result["edges"] if e["type"] == edge
                and {e["from"], e["to"]} == {"EQ_CHILD", "EQ_PARENT"}]
    assert len(matching) == 1
    assert (matching[0]["from"], matching[0]["to"]) == ("EQ_CHILD", "EQ_PARENT")
    assert matching[0]["provenance"] == "SYSTEM_INFERS:REL_TEST"
    assert store.load("relationships") == [original]
