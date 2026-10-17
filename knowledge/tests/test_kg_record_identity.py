"""Duplicate survey facets stay inspectable; conflicts cannot overwrite stores."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_survey, record_identity as identity, store
from kg.schema import node_id


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(harvest_survey, "REPO_ROOT", tmp_path)
    return tmp_path


def component(**extra):
    return {"id": "SW_TEST", "type": "SoftwareComponent", "file": "pkg/example.py",
            "qualname": "pkg.example:solve", "line": 1, **extra}


def test_same_identity_preserves_all_facets_and_historical_association_union():
    first = component(role="first facet", model_ids=["MODEL_A"], test_ids=["TEST_A"], constant_ids=["CONST_A"])
    second = component(role="second facet", model_ids=["MODEL_B"], test_ids=["TEST_B"], constant_ids=["CONST_B"])
    rows, report = identity.consolidate([first, second])
    row, = rows
    assert report["duplicate_groups"] == 1
    assert row["role"] == "second facet"
    assert row["model_ids"] == ["MODEL_B"]  # historical facets do not create active edges
    assert row["historical_survey_associations"]["model_ids"] == ["MODEL_A", "MODEL_B"]
    assert row["historical_survey_associations"]["test_ids"] == ["TEST_A", "TEST_B"]
    assert {v["record"]["role"] for v in row["survey_variants"]} == {"first facet", "second facet"}
    assert "role" in row["survey_conflicts"]
    assert all(v["state"] == "HISTORICAL_SURVEY_ASSERTION" for v in row["survey_variants"])


def test_preferred_current_values_basis_and_locator_fingerprints_win_over_history():
    old = {"id": "CONST_TEST", "type": "EmpiricalConstant", "file": "pkg/example.py",
           "component_id": "SW_TEST", "name": "gain", "value": "9", "basis_class": "LITERATURE_DERIVED", "line": 3}
    current = {**old, "value": "2", "basis_class": "UNKNOWN_BASIS", "line": 8,
               "source_sha256": "current-code", "implementation_locator_history": [{"line": 3}]}
    rows, _ = identity.consolidate([old, {**old, "units": "ratio"}], preferred={old["id"]: current})
    row, = rows
    assert row["value"] == "2" and row["basis_class"] == "UNKNOWN_BASIS"
    assert row["line"] == 8 and row["source_sha256"] == "current-code"
    assert row["implementation_locator_history"] == current["implementation_locator_history"]
    assert all(v["record"]["value"] == "9" for v in row["survey_variants"])


def test_different_exact_identity_raises_with_original_records():
    first, second = component(), component(qualname="pkg.example:Other.solve")
    with pytest.raises(identity.IdentityCollisionError) as caught:
        identity.consolidate([first, second])
    assert caught.value.collisions[0]["records"] == [first, second]
    assert len(caught.value.collisions[0]["identities"]) == 2


def test_store_rejects_conflicting_duplicates_before_write_and_deduplicates_exact(isolated):
    record = component(role="retained")
    store.save("software_links", [record, dict(record)])
    assert store.load("software_links") == [record]
    before = store.store_path("software_links").read_bytes()
    with pytest.raises(store.DuplicateRecordIdError):
        store.save("software_links", [record, {**record, "role": "different facet"}])
    assert store.store_path("software_links").read_bytes() == before
    with pytest.raises(store.DuplicateRecordIdError):
        store.upsert("software_links", [record, {**record, "role": "different facet"}])
    assert store.store_path("software_links").read_bytes() == before
    # An upsert must not silently dictionary-collapse an already malformed store.
    store.store_path("software_links").write_text(json.dumps({"records": [record, {**record, "line": 2}]}))
    malformed = store.store_path("software_links").read_bytes()
    with pytest.raises(store.DuplicateRecordIdError):
        store.upsert("software_links", [])
    assert store.store_path("software_links").read_bytes() == malformed


def test_recovery_is_idempotent_preserves_parameters_and_quarantines_identity_conflicts(isolated):
    current = component(role="current", line=8, source_sha256="current-code", model_ids=[])
    collision = component(id="SW_COLLISION")
    parameter = {"id": "PARAM_CURRENT", "type": "EmpiricalConstant", "harvest_owner": "kg.harvest_parameters@1",
                 "value": "current", "basis_class": "UNKNOWN_BASIS"}
    store.save("software_links", [current, collision, parameter])
    store.save("models", [{"id": "MODEL_A", "source_support_state": "NOT_CHECKED"}])
    model_bytes = store.store_path("models").read_bytes()
    archive = {"authority": "HISTORICAL_RECOVERY_ONLY", "source": {"commit": "fixture"}, "groups": [
        {"id": "SW_TEST", "records": [component(role="old-a", model_ids=["MODEL_A"]), component(role="old-b")]},
        {"id": "SW_COLLISION", "records": [collision, {**collision, "qualname": "pkg.example:Other.solve"}]},
    ]}
    before = store.store_path("software_links").read_bytes()
    preview = identity.recover_legacy_duplicates(archive, write=False)
    assert preview["recovered_groups"] == 1 and len(preview["quarantined_groups"]) == 1
    assert store.store_path("software_links").read_bytes() == before
    identity.recover_legacy_duplicates(archive)
    rows = store.index_by("software_links")
    assert rows["SW_TEST"]["role"] == "current" and rows["SW_TEST"]["line"] == 8
    assert rows["SW_TEST"]["model_ids"] == []
    assert rows["SW_TEST"]["historical_survey_associations"]["model_ids"] == ["MODEL_A"]
    assert rows["SW_COLLISION"] == collision and rows["PARAM_CURRENT"] == parameter
    assert store.store_path("models").read_bytes() == model_bytes
    assert identity.recover_legacy_duplicates(archive)["records_updated"] == 0


def test_survey_harvest_consolidates_facets_and_retains_current_locator_baseline(isolated):
    (isolated / "pkg").mkdir()
    (isolated / "pkg/example.py").write_text("def solve(value):\n    return value + 1\n")
    old = component(id=node_id("SW", "pkg/example.py:solve"),
                    source_sha256="old-file", definition_sha256="old-body", end_line=2,
                    implementation_scientific_comparison_state="NOT_RECHECKED")
    store.save("software_links", [old])
    surveys = [{"subsystem": "verification", "components": [
        {"file": old["file"], "qualname": old["qualname"], "line": 1, "role": role}]} for role in ("facet A", "facet B")]
    result = harvest_survey.harvest(surveys, set())
    row, = store.load("software_links")
    assert result["record_identity"]["duplicate_groups"] == 1
    assert row["source_sha256"] == "old-file" and row["definition_sha256"] == "old-body"
    assert {v["record"].get("role") for v in row["survey_variants"]} >= {"facet A", "facet B"}
    assert row["role"] == "facet B"
