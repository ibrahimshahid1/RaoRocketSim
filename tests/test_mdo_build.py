"""The expected-part manifest and the achieved-fidelity contract.

A partial assembly must never be labelled a complete engine, and a part that
was never modelled must be present-and-unavailable rather than absent.
"""

from __future__ import annotations

import json

import pytest

from raosim.mdo.build import (
    ENGINE_PART_MANIFEST,
    BuildFidelity,
    PartRecord,
    build_engine_parts,
    write_engine_build,
)


class _FakeResult:
    verdict = "MET"
    inputs = None
    job = None


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "liner.step"
    path.write_text("ISO-10303-21;")
    return path


def test_every_expected_part_appears_even_when_nothing_is_built():
    manifest = build_engine_parts(_FakeResult())
    assert len(manifest.parts) == len(ENGINE_PART_MANIFEST)
    assert not manifest.complete
    for record in manifest.parts:
        assert record.status == "unavailable"
        assert record.unavailable_reason


def test_unmodelled_subsystems_are_enumerated_not_omitted():
    """Omission is how a partial assembly starts looking complete."""
    ids = {spec.part_id for spec in ENGINE_PART_MANIFEST}
    for expected in (
        "valves.main_fuel", "lines.fuel", "rotating.bearings",
        "structure.gimbal", "structure.mounts", "igniter.assembly",
        "avionics.harness", "injector.film_ring",
    ):
        assert expected in ids


def test_an_unavailable_part_cannot_carry_a_mass_or_volume():
    """A missing body must not become a zero or an upper bound."""
    with pytest.raises(ValueError, match="must not carry a mass"):
        PartRecord(
            part_id="valves.main_fuel", subsystem="valves", quantity=1,
            status="unavailable",
            achieved_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
            supported_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
            mass_kg=0.0, unavailable_reason="not modelled",
        )
    with pytest.raises(ValueError, match="stable reason"):
        PartRecord(
            part_id="valves.main_fuel", subsystem="valves", quantity=1,
            status="unavailable",
            achieved_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
            supported_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
        )


def test_a_built_part_is_hashed_and_records_its_artifact(artifact):
    manifest = build_engine_parts(
        _FakeResult(), artifacts={"thrust_chamber.liner": str(artifact)}
    )
    built = [p for p in manifest.parts if p.status == "built"]
    assert len(built) == 1
    record = built[0]
    assert record.part_id == "thrust_chamber.liner"
    assert record.artifacts == (str(artifact),)
    assert len(record.artifact_sha256[str(artifact)]) == 64
    assert not manifest.complete


def test_a_declared_artifact_that_does_not_exist_is_unavailable(tmp_path):
    manifest = build_engine_parts(
        _FakeResult(),
        artifacts={"thrust_chamber.liner": str(tmp_path / "absent.step")},
    )
    record = next(
        p for p in manifest.parts if p.part_id == "thrust_chamber.liner"
    )
    assert record.status == "unavailable"
    assert "does not exist" in record.unavailable_reason


def test_an_artifact_outside_the_manifest_is_rejected(artifact):
    """Shipping an unenumerated part would defeat the contract."""
    with pytest.raises(ValueError, match="outside the manifest"):
        build_engine_parts(
            _FakeResult(), artifacts={"mystery.widget": str(artifact)}
        )


def test_fidelity_is_achieved_not_requested(artifact):
    manifest = build_engine_parts(
        _FakeResult(),
        requested_fidelity=BuildFidelity.L4_MANUFACTURING_REFERENCE,
        artifacts={"electrical.motor": str(artifact)},
    )
    record = next(p for p in manifest.parts if p.part_id == "electrical.motor")
    # The motor is an envelope only; it cannot reach L4 however it is asked for.
    assert record.achieved_fidelity == BuildFidelity.L1_RESOLVED_ANALYTICAL_GEOMETRY
    assert manifest.achieved_fidelity < manifest.requested_fidelity
    assert any("ceiling is set by" in n for n in manifest.notes)


def test_l5_is_never_awarded_by_a_solver(artifact):
    manifest = build_engine_parts(
        _FakeResult(),
        requested_fidelity=BuildFidelity.L5_QUALIFIED_HARDWARE,
        artifacts={"thrust_chamber.liner": str(artifact)},
    )
    assert manifest.achieved_fidelity < BuildFidelity.L5_QUALIFIED_HARDWARE
    assert any("never awarded" in n for n in manifest.notes)


def test_an_infeasible_analysis_does_not_silently_produce_release_cad(artifact):
    class _Violating(_FakeResult):
        verdict = "CANDIDATE_VIOLATES"

    manifest = build_engine_parts(
        _Violating(),
        requested_fidelity=BuildFidelity.L3_REFERENCE_ASSEMBLY,
        artifacts={"thrust_chamber.liner": str(artifact)},
    )
    assert any("diagnostic" in n for n in manifest.notes)


def test_partial_mass_is_never_presented_as_engine_dry_mass(artifact):
    manifest = build_engine_parts(
        _FakeResult(), artifacts={"thrust_chamber.liner": str(artifact)}
    )
    payload = manifest.to_dict()
    assert "resolved_partial_mass_kg" in payload
    assert "NOT engine dry mass" in payload["resolved_partial_mass_disclaimer"]
    assert payload["missing_subsystems"]
    assert payload["complete"] is False


def test_manifest_round_trips_to_disk(tmp_path, artifact):
    manifest = build_engine_parts(
        _FakeResult(), artifacts={"thrust_chamber.liner": str(artifact)}
    )
    path = write_engine_build(manifest, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["contract_id"] == "engine.complete_hardware@1"
    assert len(payload["parts"]) == len(ENGINE_PART_MANIFEST)
