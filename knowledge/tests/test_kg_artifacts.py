"""Artifact retrieval may observe simulated output without creating physical evidence."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_artifacts, store


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _injector(root, *, declared="body.step"):
    source = root / "builds" / "reference" / "injector_manufacturing_report.json"
    _write(source, {
        "status": "preliminary_machined_layout_requires_cold_flow_validation",
        "mechanical_spec": {"tolerance": None, "min_tool_diameter": 0.0003},
        "resolved": {"annulus_gap_m": 0.001}, "roles": {"fuel": {"feeds": "slots"}},
        "manufacturing_gates": [{"name": "minimum_tool_diameter", "status": "pass"}],
        "cad_export": {"status": "step_written", "files": {"body": declared},
                       "inspection": {"body": {"all_solids_valid": True, "volumes_m3": [123456]}}},
    })
    return source


def _nodes(result, type_):
    return [r for r in result["records"] if r["type"] == type_]


def _sandbox_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    # Keep these tests independent of store/graph integration performed elsewhere.
    monkeypatch.setitem(store.STORE_FILES, "artifacts", "artifacts.json")


def test_relative_cad_is_only_current_bytes_and_reference_plan(tmp_path):
    source = _injector(tmp_path)
    step = source.parent / "body.step"
    step.write_bytes(b"ISO-10303-21; reference data")
    result = harvest_artifacts.collect([source], repo_root=tmp_path)
    assert result == harvest_artifacts.collect([source, source], repo_root=tmp_path)
    cad, = _nodes(result, "CADArtifact")
    config, = _nodes(result, "EngineConfiguration")
    process, = _nodes(result, "ManufacturingProcess")
    assert cad["path"] == "builds/reference/body.step"
    assert cad["observed_sha256"] == hashlib.sha256(step.read_bytes()).hexdigest()
    assert cad["generation_time_sha256"] is None
    assert cad["generation_time_integrity_state"] == cad["binding_state"] == "UNKNOWN"
    assert cad["declared_inspection"]["all_solids_valid"] is True
    assert cad["inspection_verification_state"] == cad["units_verification_state"] == "UNKNOWN"
    assert config["scope"] == "injector_layout_only"
    assert config["payload"]["mechanical_spec"]["tolerance"] is None
    assert config["producer_digest"] is None and config["producer_binding_state"] == "UNKNOWN"
    assert process["execution_state"] == "UNAVAILABLE" and process["execution_evidence"] is None
    assert process["declared_manufacturing_gates"][0]["status"] == "pass"
    assert process["state"] == "REPORT_ONLY"
    assert not {"Measurement", "Experiment", "Calibration", "ComparisonRecord"} & {
        r["type"] for r in result["records"]}
    assert all(not r.get("hardware_qualified", False) for r in result["records"])
    by_id = {p["id"]: p for p in result["provenance"]}
    # The recorded locator/selection reproduces the entire hashed snapshot,
    # including mechanical_spec and roles, not just the resolved geometry.
    raw_source = json.loads(source.read_text())
    selection = by_id[config['provenance_id']]['payload_selection']
    assert config['json_pointer'] == ''
    reconstructed = {k: raw_source[k] for k in selection['fields']}
    assert reconstructed == config['payload']
    assert harvest_artifacts._canonical_digest(reconstructed) == config['snapshot_sha256']
    assert all(r["provenance_id"] in by_id for r in result["records"])
    assert all(p["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
               for p in by_id.values())
    bindings = [r for r in result["records"] if r.get("edge") == "BOUND_TO_CONFIGURATION"]
    assert len(bindings) == 2 and all(r["to"] == config["id"] for r in bindings)
    assert all(r["binding_state"] == "UNKNOWN" for r in bindings)


def test_design_metadata_does_not_invent_a_cad_export(tmp_path):
    path = _write(tmp_path / "builds/run/design_report_v2.json", {
        "input": {"Pc": 7e6, "manufacturing": {"cad": "none", "tolerance": None}},
        "metadata": {"authoritative_cad": "STEP", "hardware_qualified": False},
    })
    result = harvest_artifacts.collect([path], repo_root=tmp_path)
    config, = _nodes(result, "EngineConfiguration")
    assert config["scope"] == "serialized_design_input"
    assert config["payload"]["Pc"] == 7e6
    assert not _nodes(result, "CADArtifact")
    assert not _nodes(result, "ManufacturingProcess")


def test_root_relative_missing_and_ambiguous_paths_stay_explicit(tmp_path):
    source = _injector(tmp_path, declared="builds/reference/body.step")
    step = source.parent / "body.step"
    step.write_bytes(b"reference")
    result = harvest_artifacts.collect([source], repo_root=tmp_path)
    assert _nodes(result, "CADArtifact")[0]["path"] == "builds/reference/body.step"
    step.unlink()
    missing = _nodes(harvest_artifacts.collect([source], repo_root=tmp_path), "CADArtifact")[0]
    assert missing["state"] == "UNAVAILABLE" and missing["observed_sha256"] is None
    assert missing["path"] == "builds/reference/body.step"

    _injector(tmp_path)
    step.write_bytes(b"local")
    (tmp_path / "body.step").write_bytes(b"root")
    ambiguous = _nodes(harvest_artifacts.collect([source], repo_root=tmp_path), "CADArtifact")[0]
    assert ambiguous["state"] == "UNAVAILABLE" and ambiguous["path"] is None
    assert ambiguous["path_resolution"] == "AMBIGUOUS_RELATIVE_PATH"


def test_external_paths_and_symlinks_are_not_read(tmp_path):
    root = tmp_path / "repo"
    external = tmp_path / "outside.step"
    external.write_bytes(b"outside content")
    source = _injector(root, declared=str(external))
    result = harvest_artifacts.collect([source], repo_root=root)
    cad, = _nodes(result, "CADArtifact")
    assert cad["state"] == "UNAVAILABLE" and cad["observed_sha256"] is None
    assert cad["path_resolution"] == "OUTSIDE_REPOSITORY"
    _injector(root)
    (source.parent / "body.step").symlink_to(external)
    result = harvest_artifacts.collect([source], repo_root=root)
    assert _nodes(result, "CADArtifact")[0]["observed_sha256"] is None
    assert _nodes(result, "CADArtifact")[0]["state"] == "UNAVAILABLE"
    external_report = tmp_path / "design_report_v2.json"
    external_report.write_bytes(b"should not be read")
    (root / "builds" / "design_report_v2.json").symlink_to(external_report)
    discovered = harvest_artifacts.collect(repo_root=root)
    assert discovered["report"]["discovered_reports"] == 2
    assert discovered["report"]["accepted_reports"] == 1
    assert any(i["state"] == "OUTSIDE_REPOSITORY" for i in discovered["report"]["issues"])


def test_invalid_source_has_no_guessed_configuration(tmp_path):
    source = tmp_path / "builds/run/design_report_v2.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"input": {"Pc": NaN}, "metadata": {}}')
    result = harvest_artifacts.collect([source], repo_root=tmp_path)
    assert result["records"] == [] and result["report"]["sources"][0]["state"] == "NONFINITE_CONFIGURATION"
    source.write_text("[]")
    assert harvest_artifacts.collect([source], repo_root=tmp_path)["records"] == []


def test_changed_missing_payload_and_report_preserve_inactive_history(tmp_path, monkeypatch):
    _sandbox_store(tmp_path, monkeypatch)
    source = _injector(tmp_path)
    step = source.parent / "body.step"
    step.write_bytes(b"version one")
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    first = _nodes({"records": store.load("artifacts")}, "CADArtifact")[0]
    original_bytes = store.store_path("artifacts").read_bytes()
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    assert store.store_path("artifacts").read_bytes() == original_bytes
    step.write_bytes(b"version two")
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    records = {r["id"]: r for r in store.load("artifacts")}
    assert records[first["id"]]["active"] is False
    assert records[first["id"]]["source_state"] == "STALE_ARTIFACT_BYTES"
    assert len([r for r in records.values() if r["type"] == "CADArtifact" and r["active"]]) == 1
    assert all(not r["active"] for r in records.values()
               if r.get("from") == first["id"])
    step.unlink()
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    active_cad = [r for r in store.load("artifacts") if r["type"] == "CADArtifact" and r["active"]]
    assert len(active_cad) == 1 and active_cad[0]["state"] == "UNAVAILABLE"
    old_provenance = {p["id"] for p in store.load("provenance")}
    source.unlink()
    harvest_artifacts.harvest(repo_root=tmp_path)
    assert all(not r["active"] and r["source_state"] == "SOURCE_UNAVAILABLE"
               for r in store.load("artifacts"))
    assert old_provenance <= {p["id"] for p in store.load("provenance")}


def test_bounded_refresh_preserves_unrelated_evidence_and_source_change(tmp_path, monkeypatch):
    _sandbox_store(tmp_path, monkeypatch)
    source = _injector(tmp_path)
    (source.parent / "body.step").write_bytes(b"one")
    unrelated = {"id": "OTHER", "type": "Experiment", "curated": True}
    store.save("artifacts", [unrelated])
    store.save("provenance", [{"id": "PROV_OTHER", "kind": "SOURCE_STATES"}])
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    before = {r["id"]: r for r in store.load("artifacts")}
    harvest_artifacts.harvest(["builds/absent/design_report_v2.json"], repo_root=tmp_path)
    assert before == {r["id"]: r for r in store.load("artifacts")}
    payload = json.loads(source.read_text())
    payload["mechanical_spec"]["tolerance"] = 1e-5
    _write(source, payload)
    harvest_artifacts.harvest([source], repo_root=tmp_path)
    after = {r["id"]: r for r in store.load("artifacts")}
    assert after["OTHER"] == unrelated
    assert all(after[rid]["source_state"] == "STALE_SOURCE" and not after[rid]["active"]
               for rid in before if rid != "OTHER")
    assert any(p["id"] == "PROV_OTHER" for p in store.load("provenance"))


def test_bounded_report_replaced_by_external_symlink_loses_active_support(tmp_path, monkeypatch):
    _sandbox_store(tmp_path, monkeypatch)
    root = tmp_path / "repo"
    source = _injector(root)
    (source.parent / "body.step").write_bytes(b"original")
    harvest_artifacts.harvest([source], repo_root=root)
    external = tmp_path / "external.json"
    external.write_text("{}")
    source.unlink()
    source.symlink_to(external)
    harvest_artifacts.harvest([source], repo_root=root)
    assert all(not r["active"] and r["source_state"] == "OUTSIDE_REPOSITORY"
               for r in store.load("artifacts"))


def test_real_report_examples_dry_collect():
    """Exercise producer payloads when this local, ignored evidence is present."""
    root = Path(__file__).resolve().parents[2]
    paths = [root / "builds/v309_20260620_224603/design_report_v2.json",
             root / "builds/lox_rp1_13kn_40bar_grcop84_inconel_pintle/pintle/injector_manufacturing_report.json"]
    if not all(path.exists() for path in paths):
        import pytest
        pytest.skip("local ignored producer examples unavailable")
    result = harvest_artifacts.collect(paths, repo_root=root)
    assert result["report"]["accepted_reports"] == 2
    assert len(_nodes(result, "EngineConfiguration")) == 2
    assert len(_nodes(result, "ManufacturingProcess")) == 1
    assert len(_nodes(result, "CADArtifact")) == 4
    assert all(c["state"] == "REPORT_ONLY" for c in _nodes(result, "CADArtifact"))
