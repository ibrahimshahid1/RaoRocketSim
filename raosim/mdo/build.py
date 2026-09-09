"""The expected-part manifest and one build/artifact orchestrator.

Two competing build paths is how the repository ended up able to report a
"complete" ledger for an engine missing whole subsystems.  This module owns the
opposite contract: an engine has a **known, enumerated** part list, and every
entry is either built and measured, or unavailable with a stable reason.  A
part that was never modelled is present in the manifest as unavailable, never
absent and never zero.

The fidelity ladder is deliberately *achieved*, not requested.  A caller states
a target; the build reports what it actually reached and why it stopped there.
``L5`` is never awarded automatically -- physical qualification is not a
property any solver can confer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ENGINE_PART_MANIFEST",
    "BuildFidelity",
    "EngineBuildManifest",
    "PartRecord",
    "PartSpec",
    "build_engine_parts",
    "write_engine_build",
]


class BuildFidelity(IntEnum):
    """What a produced artifact actually demonstrates."""

    L0_NUMERICAL_ONLY = 0
    L1_RESOLVED_ANALYTICAL_GEOMETRY = 1
    L2_VALID_REFERENCE_CAD = 2
    L3_REFERENCE_ASSEMBLY = 3
    L4_MANUFACTURING_REFERENCE = 4
    L5_QUALIFIED_HARDWARE = 5

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class PartSpec:
    """One part the engine is expected to have."""

    part_id: str
    subsystem: str
    quantity: int
    #: Highest fidelity the repository can currently reach for this part.
    supported_fidelity: BuildFidelity
    #: Why that ceiling, when it is below L2.
    limitation: str | None = None


#: Every part a complete electric-pump-fed engine needs.  Entries capped at
#: ``L0`` have no geometry model at all; they exist here so a manifest can
#: never imply a complete engine by omission.
ENGINE_PART_MANIFEST: tuple[PartSpec, ...] = (
    PartSpec("thrust_chamber.liner", "thrust_chamber", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("thrust_chamber.channel_lands", "thrust_chamber", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("thrust_chamber.closeout", "thrust_chamber", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("chamber_interface.flange", "chamber_interface", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("injector.body", "injector", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("injector.faceplate", "injector", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("injector.pintle_post", "injector", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("injector.pintle_tip", "injector", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("injector.orifice_plate", "injector", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("feed.fuel_pump", "feed", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("feed.oxidizer_pump", "feed", 1,
             BuildFidelity.L2_VALID_REFERENCE_CAD),
    PartSpec("electrical.motor", "electrical", 2,
             BuildFidelity.L1_RESOLVED_ANALYTICAL_GEOMETRY,
             "simplified envelope only; no vendor rotor/stator internals"),
    PartSpec("electrical.inverter", "electrical", 2,
             BuildFidelity.L1_RESOLVED_ANALYTICAL_GEOMETRY,
             "simplified envelope only; no vendor internals"),
    PartSpec("electrical.battery", "electrical", 1,
             BuildFidelity.L1_RESOLVED_ANALYTICAL_GEOMETRY,
             "simplified envelope only; no cell/module layout"),
    # --- enumerated but not modelled ------------------------------------- #
    PartSpec("valves.main_fuel", "valves", 1, BuildFidelity.L0_NUMERICAL_ONLY,
             "no valve model; docs/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md "
             "specifies the sizing and needs no new physics"),
    PartSpec("valves.main_oxidizer", "valves", 1,
             BuildFidelity.L0_NUMERICAL_ONLY, "see valves.main_fuel"),
    PartSpec("valves.igniter", "valves", 2, BuildFidelity.L0_NUMERICAL_ONLY,
             "see valves.main_fuel"),
    PartSpec("lines.fuel", "lines", 1, BuildFidelity.L0_NUMERICAL_ONLY,
             "no routed line/bellows/fitting geometry"),
    PartSpec("lines.oxidizer", "lines", 1, BuildFidelity.L0_NUMERICAL_ONLY,
             "no routed line/bellows/fitting geometry"),
    PartSpec("rotating.bearings", "rotating", 4,
             BuildFidelity.L0_NUMERICAL_ONLY,
             "no bearing, seal, wear-ring or coupling model"),
    PartSpec("rotating.seals", "rotating", 4, BuildFidelity.L0_NUMERICAL_ONLY,
             "see rotating.bearings"),
    PartSpec("structure.gimbal", "structure", 1,
             BuildFidelity.L0_NUMERICAL_ONLY,
             "no gimbal block, bearing or actuator model"),
    PartSpec("structure.mounts", "structure", 1,
             BuildFidelity.L0_NUMERICAL_ONLY,
             "no thrust structure or mount model"),
    PartSpec("igniter.assembly", "igniter", 1, BuildFidelity.L0_NUMERICAL_ONLY,
             "no complete igniter model"),
    PartSpec("injector.film_ring", "injector", 1,
             BuildFidelity.L0_NUMERICAL_ONLY,
             "film-injector hardware is explicitly unsupported"),
    PartSpec("fasteners.interface_bolts", "chamber_interface", 0,
             BuildFidelity.L1_RESOLVED_ANALYTICAL_GEOMETRY,
             "bolt count/size are sized, but no fastener solid bodies are "
             "built and no fastener standard is selected"),
    PartSpec("avionics.harness", "avionics", 1,
             BuildFidelity.L0_NUMERICAL_ONLY,
             "no wiring, connector, sensor or controller model"),
)


@dataclass(frozen=True)
class PartRecord:
    """What the build actually produced for one expected part."""

    part_id: str
    subsystem: str
    quantity: int
    status: str                       # "built" | "unavailable"
    achieved_fidelity: BuildFidelity
    supported_fidelity: BuildFidelity
    material: str | None = None
    geometry_id: str | None = None
    volume_m3: float | None = None
    mass_kg: float | None = None
    artifacts: tuple[str, ...] = ()
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status == "built" and self.unavailable_reason is not None:
            raise ValueError(
                f"{self.part_id}: a built part cannot carry an unavailable "
                "reason"
            )
        if self.status == "unavailable":
            if self.mass_kg is not None or self.volume_m3 is not None:
                raise ValueError(
                    f"{self.part_id}: an unavailable part must not carry a "
                    "mass or volume -- that is how a missing body becomes a "
                    "zero or an upper bound masquerading as a measurement"
                )
            if not self.unavailable_reason:
                raise ValueError(
                    f"{self.part_id}: unavailable requires a stable reason"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["achieved_fidelity"] = self.achieved_fidelity.label
        payload["supported_fidelity"] = self.supported_fidelity.label
        payload["artifact_sha256"] = dict(self.artifact_sha256)
        return payload


@dataclass(frozen=True)
class EngineBuildManifest:
    """The immutable record of one build."""

    contract_id: str
    parts: tuple[PartRecord, ...]
    achieved_fidelity: BuildFidelity
    requested_fidelity: BuildFidelity
    resolved_inputs_digest: str | None
    job_digest: str | None
    notes: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True only when every enumerated part was built.

        This is complete relative to the named contract, which for this
        manifest means a whole engine.  It is expected to be False today, and
        saying so is the point.
        """
        return all(p.status == "built" for p in self.parts)

    @property
    def resolved_mass_kg(self) -> float:
        """Subtotal of the parts that were actually measured.

        Explicitly a *partial* subtotal, never presented as engine dry mass.
        """
        return sum(p.mass_kg for p in self.parts if p.mass_kg is not None)

    @property
    def missing_subsystems(self) -> tuple[str, ...]:
        return tuple(sorted({
            p.subsystem for p in self.parts if p.status != "built"
        }))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "complete": self.complete,
            "achieved_fidelity": self.achieved_fidelity.label,
            "requested_fidelity": self.requested_fidelity.label,
            "resolved_partial_mass_kg": self.resolved_mass_kg,
            "resolved_partial_mass_disclaimer": (
                "subtotal of measured parts only; this is NOT engine dry mass "
                f"-- missing subsystems: {', '.join(self.missing_subsystems)}"
            ),
            "missing_subsystems": list(self.missing_subsystems),
            "job_digest": self.job_digest,
            "resolved_inputs_digest": self.resolved_inputs_digest,
            "parts": [p.to_dict() for p in self.parts],
            "notes": list(self.notes),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_engine_parts(
    result: Any,
    *,
    requested_fidelity: BuildFidelity = BuildFidelity.L0_NUMERICAL_ONLY,
    artifacts: Mapping[str, str] | None = None,
    contract_id: str = "engine.complete_hardware@1",
) -> EngineBuildManifest:
    """Assemble the expected-part manifest for one solved job.

    ``artifacts`` maps ``part_id`` to a file the caller declares. A path and
    content hash establish presence only: they do not verify CAD validity,
    units, part identity or configuration identity. This legacy path-only
    interface records such files as unverified and awards no geometry fidelity.
    A future producer validation handoff must establish those claims explicitly.
    """

    supplied = dict(artifacts or {})
    records: list[PartRecord] = []
    notes: list[str] = []

    feasible = getattr(result, "verdict", None) == "MET"
    if not feasible and requested_fidelity >= BuildFidelity.L2_VALID_REFERENCE_CAD:
        notes.append(
            "the analysis verdict is "
            f"{getattr(result, 'verdict', 'unknown')!r}, so release-looking CAD "
            "is not produced by default; artifacts here are diagnostic"
        )

    for spec in ENGINE_PART_MANIFEST:
        path_text = supplied.get(spec.part_id)
        if path_text is None:
            records.append(PartRecord(
                part_id=spec.part_id,
                subsystem=spec.subsystem,
                quantity=spec.quantity,
                status="unavailable",
                achieved_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
                supported_fidelity=spec.supported_fidelity,
                unavailable_reason=(
                    spec.limitation
                    or "no artifact was produced for this part in this build"
                ),
            ))
            continue
        path = Path(path_text)
        if not path.is_file():
            records.append(PartRecord(
                part_id=spec.part_id,
                subsystem=spec.subsystem,
                quantity=spec.quantity,
                status="unavailable",
                achieved_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
                supported_fidelity=spec.supported_fidelity,
                unavailable_reason=f"declared artifact {path} does not exist",
            ))
            continue
        records.append(PartRecord(
            part_id=spec.part_id,
            subsystem=spec.subsystem,
            quantity=spec.quantity,
            status="unavailable",
            achieved_fidelity=BuildFidelity.L0_NUMERICAL_ONLY,
            supported_fidelity=spec.supported_fidelity,
            artifacts=(str(path),),
            artifact_sha256={str(path): _sha256(path)},
            unavailable_reason=(
                "unverified artifact: path and hash do not establish CAD "
                "validity, units or binding to this part and configuration; "
                "a producer validation record is unavailable"
            ),
        ))

    unknown = set(supplied) - {s.part_id for s in ENGINE_PART_MANIFEST}
    if unknown:
        raise ValueError(
            "artifacts supplied for parts outside the manifest: "
            + ", ".join(sorted(unknown))
            + ". Add them to ENGINE_PART_MANIFEST rather than shipping a part "
            "the contract does not enumerate."
        )

    achieved_overall = (
        min(r.achieved_fidelity for r in records) if records
        else BuildFidelity.L0_NUMERICAL_ONLY
    )
    if requested_fidelity > achieved_overall:
        notes.append(
            f"requested {requested_fidelity.label} but achieved "
            f"{BuildFidelity(achieved_overall).label}; the ceiling is set by "
            "the least-complete required part"
        )
    if requested_fidelity is BuildFidelity.L5_QUALIFIED_HARDWARE:
        notes.append(
            "L5 is physical qualification and is never awarded by a solver"
        )

    inputs = getattr(result, "inputs", None)
    job = getattr(result, "job", None)
    return EngineBuildManifest(
        contract_id=contract_id,
        parts=tuple(records),
        achieved_fidelity=BuildFidelity(achieved_overall),
        requested_fidelity=requested_fidelity,
        resolved_inputs_digest=inputs.digest() if inputs is not None else None,
        job_digest=getattr(job, "job_digest", None),
        notes=tuple(notes),
    )


def write_engine_build(
    manifest: EngineBuildManifest,
    output_dir: str | Path,
    *,
    filename: str = "engine_build_manifest.json",
) -> Path:
    """Write the manifest next to the artifacts it describes."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    return path
