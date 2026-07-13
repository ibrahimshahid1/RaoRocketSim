from __future__ import annotations

from datetime import date
import hashlib
import json

import pytest

from raosim.release_readiness import (
    EvidenceRecord,
    evidence_manifest_template,
    evaluate_release_readiness,
    load_evidence_manifest,
    release_requirements,
)


def _passing(requirement_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        requirement_id=requirement_id,
        passed=True,
        artifact=f"archive://qualification/{requirement_id}.pdf",
        artifact_sha256="a" * 64,
        configuration_id="ENGINE-CFG-001-REV-A",
        reviewed_by="independent engineering authority",
        review_date=date.today().isoformat(),
    )


def test_default_engine_release_is_explicitly_blocked():
    report = evaluate_release_readiness(target="engine")
    assert report.blocked is True
    assert report.evidence_complete is False
    assert report.hardware_qualified is False
    assert any("hot_fire" in blocker for blocker in report.blockers)
    with pytest.raises(RuntimeError, match="physical release is blocked"):
        report.require_complete()


@pytest.mark.parametrize("target", ["injector", "regen", "pump"])
def test_component_requirements_do_not_include_other_components(target):
    assert release_requirements(target)
    assert {item.component for item in release_requirements(target)} == {target}


def test_complete_traceable_evidence_passes_software_gate_only():
    evidence = [_passing(item.id) for item in release_requirements("injector")]
    report = evaluate_release_readiness(evidence, target="injector")
    assert report.evidence_complete is True
    assert report.blocked is False
    assert report.hardware_qualified is False
    report.require_complete()


def test_failed_or_untraceable_evidence_cannot_pass():
    requirement_id = release_requirements("pump")[0].id
    failed = EvidenceRecord(
        requirement_id=requirement_id,
        passed=False,
        artifact="",
        artifact_sha256="",
        configuration_id="",
        reviewed_by="",
        review_date="not-a-date",
        notes="flow volume is disconnected",
    )
    report = evaluate_release_readiness([failed], target="pump")
    result = next(r for r in report.requirements if r.evidence is not None)
    assert result.status == "invalid"
    assert report.blocked is True


def test_duplicate_records_are_rejected():
    requirement_id = release_requirements("regen")[0].id
    report = evaluate_release_readiness(
        [_passing(requirement_id), _passing(requirement_id)],
        target="regen",
    )
    result = next(r for r in report.requirements if r.requirement.id == requirement_id)
    assert result.status == "invalid"


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="unknown release target"):
        release_requirements("thruster-of-theseus")


def test_manifest_template_is_versioned_and_blocked(tmp_path):
    payload = evidence_manifest_template("pump")
    assert payload["schema"] == "lrekit.release_evidence.v1"
    path = tmp_path / "pump_release_evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = load_evidence_manifest(path, expected_target="pump")
    assert report.blocked is True
    assert len(report.requirements) == len(release_requirements("pump"))


def test_manifest_rejects_string_boolean(tmp_path):
    payload = evidence_manifest_template("injector")
    payload["evidence"][0]["passed"] = "yes"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_evidence_manifest(path)


def test_manifest_configuration_must_match_records_and_expected_design(tmp_path):
    payload = evidence_manifest_template("injector")
    payload["configuration_id"] = "INJECTOR-CFG-7"
    for item in payload["evidence"]:
        item.update(_passing(item["requirement_id"]).to_dict())
        item.pop("validation_errors", None)
        item["configuration_id"] = "INJECTOR-CFG-7"
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = load_evidence_manifest(
        path,
        expected_target="injector",
        expected_configuration_id="INJECTOR-CFG-7",
    )
    assert report.evidence_complete is True
    with pytest.raises(ValueError, match="does not match expected"):
        load_evidence_manifest(
            path,
            expected_configuration_id="A-DIFFERENT-CONFIGURATION",
        )

    payload["evidence"][0]["configuration_id"] = "WRONG"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match manifest"):
        load_evidence_manifest(path)


def test_local_evidence_artifact_sha_is_verified(tmp_path):
    artifact = tmp_path / "cold_flow_report.pdf"
    artifact.write_bytes(b"configuration-controlled cold-flow evidence")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    requirement = release_requirements("injector")[0]
    payload = {
        "schema": "lrekit.release_evidence.v1",
        "target": "injector",
        "configuration_id": "INJECTOR-CFG-8",
        "evidence": [{
            "requirement_id": requirement.id,
            "passed": True,
            "artifact": artifact.name,
            "artifact_sha256": digest,
            "configuration_id": "INJECTOR-CFG-8",
            "reviewed_by": "independent reviewer",
            "review_date": date.today().isoformat(),
        }],
    }
    path = tmp_path / "local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = load_evidence_manifest(path)
    first = next(r for r in report.requirements if r.requirement.id == requirement.id)
    assert first.status == "passed"

    payload["evidence"][0]["artifact_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = load_evidence_manifest(path)
    first = next(r for r in report.requirements if r.requirement.id == requirement.id)
    assert first.status == "failed"
    assert "SHA-256 mismatch" in first.detail
