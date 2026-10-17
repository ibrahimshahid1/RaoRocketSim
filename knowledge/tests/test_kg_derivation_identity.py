"""Long endpoint IDs cannot collapse distinct derivation facts or their history."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_extraction, store
from kg.schema import node_id


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    monkeypatch.setattr(harvest_extraction, "harvest", lambda models, windows: {})
    store.save("manifest", [{"id": "DOC_WHITE", "sha256": "source"}])
    prefix = "EQ_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI_"
    identities = [prefix + suffix for suffix in ("CHILD_HASH", "PARENT_A_HASH", "PARENT_B_HASH")]
    store.save("equations", [{"id": ident, "label": label, "document_id": "DOC_WHITE",
                              "source_sha256": "source", "source_state": "CURRENT",
                              "grounded": "GROUNDED", "active": True}
                             for ident, label in zip(identities, ("Child", "Parent A", "Parent B"))])
    return identities


def groups(*parents, note="Explicit bounded source comparison"):
    return [{"group": "White derivation", "cross_document_relations": [
        {"a_doc_id": "DOC_WHITE", "a_label": "Child", "b_doc_id": "DOC_WHITE", "b_label": parent,
         "relation": "DERIVED_FROM", "note": note} for parent in parents]}]


def test_full_tuple_hash_retains_both_long_parent_relations_and_is_stable_on_replay(isolated):
    child, parent_a, parent_b = isolated
    assert node_id("REL", f"{child} DERIVED_FROM {parent_a}") == node_id("REL", f"{child} DERIVED_FROM {parent_b}")
    packet = groups("Parent A", "Parent B")
    assert harvest_extraction.harvest_groups(packet)["cross_document_relations"]["attached"] == 2
    before = store.load("relationships")
    assert len(before) == 2
    assert len({r["id"] for r in before}) == 2
    assert all(len(r["id"]) == len("REL_") + 64 for r in before)
    assert {r["to"] for r in before} == {parent_a, parent_b}
    harvest_extraction.harvest_groups(packet)
    assert store.load("relationships") == before
    child_record = next(r for r in store.load("equations") if r["id"] == child)
    assert set(child_record["derived_from"]) == {parent_a, parent_b}


def test_legacy_row_migrates_once_and_preserves_original_assertion_history(isolated):
    child, parent, _ = isolated
    legacy_id = node_id("REL", f"{child} DERIVED_FROM {parent}")
    original = {"id": legacy_id, "type": "Relationship", "from": child, "edge": "DERIVED_FROM", "to": parent,
                "note": "Original reviewed assertion", "recorded_at": "original-time",
                "provenance_kind": "SYSTEM_INFERS", "source_relation": "DERIVED_FROM"}
    store.save("relationships", [original])
    harvest_extraction.harvest_groups([])
    migrated, = store.load("relationships")
    assert migrated["id"] == harvest_extraction._relationship_id(child, "DERIVED_FROM", parent)
    assert migrated["legacy_ids"] == [legacy_id]
    assert migrated["history"] == [original]
    assert migrated["note"] == original["note"]
    harvest_extraction.harvest_groups([])
    assert store.load("relationships") == [migrated]
    harvest_extraction.harvest_groups(groups("Parent A"))
    updated, = store.load("relationships")
    assert updated["legacy_ids"] == [legacy_id]
    assert original in updated["history"]
    assert any(h["note"] == original["note"] and h["id"] == migrated["id"] for h in updated["history"])
    assert updated["note"] == "Explicit bounded source comparison"
    harvest_extraction.harvest_groups(groups("Parent A"))
    assert store.load("relationships") == [updated]


def test_duplicate_legacy_and_canonical_tuple_merges_history_without_duplicate_fact(isolated):
    child, parent, _ = isolated
    canonical_id = harvest_extraction._relationship_id(child, "DERIVED_FROM", parent)
    base = {"type": "Relationship", "from": child, "edge": "DERIVED_FROM", "to": parent}
    store.save("relationships", [{**base, "id": "REL_LEGACY", "note": "Legacy note"},
                                 {**base, "id": canonical_id, "note": "Other historical note"}])
    harvest_extraction.harvest_groups([])
    row, = store.load("relationships")
    assert row["id"] == canonical_id
    assert "REL_LEGACY" in row["legacy_ids"]
    assert {h["note"] for h in row["history"]} >= {"Legacy note", "Other historical note"}


@pytest.mark.parametrize("existing", [False, True])
def test_hash_collision_between_distinct_tuples_raises_before_relation_store_write(isolated, monkeypatch, existing):
    child, parent_a, parent_b = isolated
    if existing:
        store.save("relationships", [{"id": "REL_OLD_A", "from": child, "edge": "DERIVED_FROM", "to": parent_a},
                                     {"id": "REL_OLD_B", "from": child, "edge": "DERIVED_FROM", "to": parent_b}])
    before = store.load("relationships")
    monkeypatch.setattr(harvest_extraction, "_relationship_id", lambda *args: "REL_FORCED_COLLISION")
    with pytest.raises(ValueError, match="collision.*distinct tuples"):
        harvest_extraction.harvest_groups(groups("Parent A", "Parent B"))
    assert store.load("relationships") == before
