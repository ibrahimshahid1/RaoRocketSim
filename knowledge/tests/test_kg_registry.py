"""Registry refresh must not erase independently harvested source evidence."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg import harvest_registry, store
from kg.schema import node_id
from raosim import model_registry


def test_registry_refresh_preserves_extraction_and_updates_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    original = model_registry.MODEL_REGISTRY["injector.incompressible_orifice"]
    model = replace(original, relation="updated relation", validity="updated validity")
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {model.model_id: model})
    mid = node_id("MODEL", model.model_id)
    vid = node_id("VAL", model.model_id)
    store.save("manifest", [{"id": "DOC_SOURCE", "source_pdf": model.local_source}])
    evidence = {
        "source_support_state": "PARTIALLY_SUPPORTED", "source_support_notes": "Source differs",
        "source_support_state_verified": "NOT_FOUND_IN_SOURCE",
        "source_support_equation_labels": ["Eq 1"], "refuted_equation_ids": ["EQ_1"],
    }
    store.save("models", [{"id": mid, "registry_id": model.model_id,
                           "relation": "updated relation", **evidence}])
    source_validity = {"id": "VAL_SOURCE", "model_id": mid, "provenance_kind": "SOURCE_STATES",
                       "prose": "Independently sourced bound", "provenance_id": "PROV_SOURCE"}
    store.save("validity", [source_validity, {"id": vid, "provenance_kind": "PROJECT_DERIVES",
                                            "prose": "old validity"}])

    for _ in range(2):
        harvest_registry.harvest()
        refreshed = store.load("models")[0]
        assert refreshed["relation"] == "updated relation"
        assert refreshed["source_document_id"] == "DOC_SOURCE"
        assert {key: refreshed[key] for key in evidence} == evidence
        validity = {row["id"]: row for row in store.load("validity")}
        assert len(validity) == 2
        assert validity["VAL_SOURCE"] == source_validity
        assert validity[vid]["prose"] == "updated validity"


def test_registry_refresh_removes_retired_registry_envelope_only(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {})
    store.save("models", [{"id": "MODEL_OLD", "registry_id": "old"}])
    store.save("validity", [
        {"id": node_id("VAL", "old"), "provenance_kind": "PROJECT_DERIVES"},
        {"id": "VAL_SOURCE", "provenance_kind": "SOURCE_STATES"},
    ])
    harvest_registry.harvest()
    assert store.load("models") == []
    assert store.load("validity") == [{"id": "VAL_SOURCE", "provenance_kind": "SOURCE_STATES"}]


def test_registry_changed_source_or_relation_requires_review(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'KNOWLEDGE_ROOT', tmp_path)
    model = model_registry.MODEL_REGISTRY['injector.incompressible_orifice']
    monkeypatch.setattr(model_registry, 'MODEL_REGISTRY', {model.model_id: model})
    mid = node_id('MODEL', model.model_id)
    store.save('manifest', [])
    store.save('models', [{'id': mid, 'registry_id': model.model_id, 'relation': 'different relation',
                           'source_support_state': 'SUPPORTED', 'source_support_state_verified': 'SUPPORTED'}])
    harvest_registry.harvest()
    record = store.load('models')[0]
    assert record['source_support_state'] == 'NOT_CHECKED'
    assert 'source_support_state_verified' not in record
    assert record['source_support_history'][0]['state'] == 'SUPPORTED'
