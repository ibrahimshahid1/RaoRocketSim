"""Evidence-backed release-readiness gates for physical engine hardware.

Numerical checks and valid CAD topology are necessary, but neither one proves
that an engine component is safe to manufacture or test.  This module keeps
those claims separate by requiring traceable external evidence for every
physical release gate.  It deliberately never sets ``hardware_qualified``;
formal qualification remains the responsibility of the applicable engineering
authority and test organisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class EvidenceRecord:
    """Traceable result supplied by an external analysis or test activity."""

    requirement_id: str
    passed: bool
    artifact: str
    artifact_sha256: str
    configuration_id: str
    reviewed_by: str
    review_date: str
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvidenceRecord":
        required = {
            "requirement_id", "passed", "artifact", "artifact_sha256",
            "configuration_id", "reviewed_by", "review_date",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "evidence record is missing required fields: " + ", ".join(missing)
            )
        if not isinstance(payload["passed"], bool):
            raise ValueError("evidence field 'passed' must be a JSON boolean")
        return cls(
            requirement_id=str(payload["requirement_id"]),
            passed=payload["passed"],
            artifact=str(payload["artifact"]),
            artifact_sha256=str(payload["artifact_sha256"]),
            configuration_id=str(payload["configuration_id"]),
            reviewed_by=str(payload["reviewed_by"]),
            review_date=str(payload["review_date"]),
            notes=str(payload.get("notes", "")),
        )

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.requirement_id.strip():
            errors.append("requirement_id is required")
        if not self.artifact.strip():
            errors.append("artifact reference is required")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.artifact_sha256.strip()):
            errors.append("artifact_sha256 must contain exactly 64 hexadecimal characters")
        if not self.configuration_id.strip():
            errors.append("configuration_id is required")
        if not self.reviewed_by.strip():
            errors.append("reviewed_by is required")
        try:
            parsed = date.fromisoformat(self.review_date)
            if parsed > date.today():
                errors.append("review_date cannot be in the future")
        except (TypeError, ValueError):
            errors.append("review_date must be ISO-8601 YYYY-MM-DD")
        return errors

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "passed": self.passed,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "configuration_id": self.configuration_id,
            "reviewed_by": self.reviewed_by,
            "review_date": self.review_date,
            "notes": self.notes,
            "validation_errors": self.validation_errors(),
        }


@dataclass(frozen=True)
class ReleaseRequirement:
    id: str
    component: str
    description: str
    evidence_type: str


@dataclass(frozen=True)
class RequirementResult:
    requirement: ReleaseRequirement
    status: str
    evidence: EvidenceRecord | None = None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict:
        return {
            "id": self.requirement.id,
            "component": self.requirement.component,
            "description": self.requirement.description,
            "evidence_type": self.requirement.evidence_type,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass
class ReleaseReadinessReport:
    target: str
    requirements: list[RequirementResult] = field(default_factory=list)
    hardware_qualified: bool = False

    @property
    def evidence_complete(self) -> bool:
        return bool(self.requirements) and all(r.passed for r in self.requirements)

    @property
    def blocked(self) -> bool:
        return not self.evidence_complete

    @property
    def blockers(self) -> list[str]:
        return [
            f"{result.requirement.id}: {result.detail}"
            for result in self.requirements
            if not result.passed
        ]

    def require_complete(self) -> None:
        if self.blocked:
            raise RuntimeError(
                f"{self.target} physical release is blocked: "
                + "; ".join(self.blockers)
            )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "evidence_complete": self.evidence_complete,
            "blocked": self.blocked,
            "hardware_qualified": False,
            "qualification_authority": "external_engineering_authority_required",
            "blockers": self.blockers,
            "requirements": [r.to_dict() for r in self.requirements],
        }


_REQUIREMENTS: tuple[ReleaseRequirement, ...] = (
    ReleaseRequirement(
        "engine.cfd_flow",
        "engine",
        "Independent chamber/nozzle/injector flow analysis covers the intended envelope.",
        "CFD review report and archived case manifest",
    ),
    ReleaseRequirement(
        "engine.cht",
        "engine",
        "Conjugate heat-transfer analysis closes gas, wall, and coolant temperatures.",
        "CHT report with mesh/time-step independence",
    ),
    ReleaseRequirement(
        "engine.structural_fea",
        "engine",
        "Pressure, thermal, joint, fatigue, and buckling loads are independently assessed.",
        "structural/thermal FEA report",
    ),
    ReleaseRequirement(
        "engine.material_allowables",
        "engine",
        "Lot/process/temperature-dependent material allowables and compatibility are approved.",
        "material and process specification",
    ),
    ReleaseRequirement(
        "engine.drawings_gdt",
        "engine",
        "Released drawings, GD&T, fits, surface finish, BOM, and process routes exist.",
        "configuration-controlled drawing package",
    ),
    ReleaseRequirement(
        "engine.manufacturing_review",
        "engine",
        "Manufacturing, cleaning, inspection, NDE, and assembly plans are approved.",
        "manufacturing readiness review record",
    ),
    ReleaseRequirement(
        "engine.proof_test",
        "engine",
        "Pressure proof and leak tests pass at approved factors and temperatures.",
        "signed proof/leak test record",
    ),
    ReleaseRequirement(
        "engine.cold_flow",
        "engine",
        "Integrated cold-flow data validate flow split, pressure loss, and distribution.",
        "cold-flow test report and calibrated data",
    ),
    ReleaseRequirement(
        "engine.combustion_stability",
        "engine",
        "Ignition, stability, acoustic response, and shutdown transients are assessed.",
        "combustion-stability review/test report",
    ),
    ReleaseRequirement(
        "engine.hot_fire",
        "engine",
        "Hot-fire data validate pressure, c-star efficiency, heat flux, and durability.",
        "hot-fire report and reduced data",
    ),
    ReleaseRequirement(
        "injector.void_connectivity",
        "injector",
        "Extracted fluid volumes prove intended circuits and isolation/leak paths.",
        "CAD fluid-volume and leak-path report",
    ),
    ReleaseRequirement(
        "injector.cold_flow_distribution",
        "injector",
        "Cold-flow testing validates Cd, spray angle, distribution, and droplet statistics.",
        "injector cold-flow/optical test report",
    ),
    ReleaseRequirement(
        "regen.flow_distribution",
        "regen",
        "Channel and manifold maldistribution is assessed over the operating envelope.",
        "regen CFD or calibrated network report",
    ),
    ReleaseRequirement(
        "regen.hydroproof",
        "regen",
        "The completed liner/jacket/manifold assembly passes hydroproof and leak tests.",
        "regen proof/leak test report",
    ),
    ReleaseRequirement(
        "pump.fluid_volume",
        "pump",
        "Connected inlet, impeller, diffuser, volute, and outlet fluid volumes are verified.",
        "pump fluid-volume CAD/CFD report",
    ),
    ReleaseRequirement(
        "pump.rotordynamics",
        "pump",
        "Critical speeds, imbalance response, shaft deflection, bearings, and seals are approved.",
        "rotordynamics and bearing/seal report",
    ),
    ReleaseRequirement(
        "pump.performance_map",
        "pump",
        "Measured head, efficiency, power, NPSH, and cavitation maps cover the duty envelope.",
        "pump cold-flow performance map",
    ),
)


_TARGET_COMPONENTS: Mapping[str, frozenset[str]] = {
    "engine": frozenset({"engine", "injector", "regen", "pump"}),
    "injector": frozenset({"injector"}),
    "regen": frozenset({"regen"}),
    "pump": frozenset({"pump"}),
}


def release_requirements(target: str = "engine") -> tuple[ReleaseRequirement, ...]:
    """Return immutable physical-release requirements for ``target``."""

    try:
        components = _TARGET_COMPONENTS[target]
    except KeyError as exc:
        raise ValueError(
            f"unknown release target {target!r}; expected one of "
            f"{sorted(_TARGET_COMPONENTS)}"
        ) from exc
    return tuple(r for r in _REQUIREMENTS if r.component in components)


def evaluate_release_readiness(
    evidence: Iterable[EvidenceRecord] = (),
    *,
    target: str = "engine",
) -> ReleaseReadinessReport:
    """Evaluate traceable evidence without inferring qualification from CAD/code."""

    records: dict[str, EvidenceRecord] = {}
    duplicate_ids: set[str] = set()
    for record in evidence:
        if record.requirement_id in records:
            duplicate_ids.add(record.requirement_id)
        records[record.requirement_id] = record

    results: list[RequirementResult] = []
    for requirement in release_requirements(target):
        record = records.get(requirement.id)
        if requirement.id in duplicate_ids:
            results.append(RequirementResult(
                requirement,
                "invalid",
                record,
                "duplicate evidence records supplied",
            ))
        elif record is None:
            results.append(RequirementResult(
                requirement,
                "missing",
                None,
                f"missing {requirement.evidence_type}",
            ))
        elif record.validation_errors():
            results.append(RequirementResult(
                requirement,
                "invalid",
                record,
                "; ".join(record.validation_errors()),
            ))
        elif not record.passed:
            results.append(RequirementResult(
                requirement,
                "failed",
                record,
                record.notes or "supplied evidence records a failed result",
            ))
        else:
            results.append(RequirementResult(
                requirement,
                "passed",
                record,
                "traceable evidence supplied; external qualification still required",
            ))

    return ReleaseReadinessReport(target=target, requirements=results)


def evidence_manifest_template(target: str = "engine") -> dict:
    """Return a deterministic, unpassed manifest for external completion."""

    return {
        "schema": "lrekit.release_evidence.v1",
        "target": target,
        "configuration_id": "",
        "evidence": [
            {
                "requirement_id": requirement.id,
                "passed": False,
                "artifact": "",
                "artifact_sha256": "",
                "configuration_id": "",
                "reviewed_by": "",
                "review_date": "",
                "notes": requirement.description,
            }
            for requirement in release_requirements(target)
        ],
    }


def load_evidence_manifest(
    path: str | Path,
    *,
    expected_target: str | None = None,
    expected_configuration_id: str | None = None,
) -> ReleaseReadinessReport:
    """Load and evaluate a versioned JSON evidence manifest."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "lrekit.release_evidence.v1":
        raise ValueError("unsupported or missing release-evidence schema")
    target = str(payload.get("target", ""))
    if expected_target is not None and target != expected_target:
        raise ValueError(
            f"evidence target {target!r} does not match expected {expected_target!r}"
        )
    configuration_id = str(payload.get("configuration_id", "")).strip()
    if expected_configuration_id is not None:
        expected_configuration_id = str(expected_configuration_id).strip()
        if not expected_configuration_id:
            raise ValueError("expected_configuration_id cannot be blank")
        if configuration_id != expected_configuration_id:
            raise ValueError(
                f"evidence configuration {configuration_id!r} does not match "
                f"expected {expected_configuration_id!r}"
            )
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("evidence manifest field 'evidence' must be a list")
    records = [EvidenceRecord.from_dict(item) for item in raw_evidence]
    if configuration_id:
        mismatched = sorted({
            record.requirement_id
            for record in records
            if record.configuration_id != configuration_id
        })
        if mismatched:
            raise ValueError(
                "evidence records do not match manifest configuration_id: "
                + ", ".join(mismatched)
            )

    # Verify local evidence bytes when the artifact is a local path or file URI.
    # Archive/HTTPS references remain externally reviewed references; their
    # presence cannot be asserted by an offline design tool.
    verified_records: list[EvidenceRecord] = []
    for record in records:
        local_path = _local_artifact_path(record.artifact, source.parent)
        if local_path is None or not record.passed:
            verified_records.append(record)
            continue
        integrity_error = _local_artifact_integrity_error(local_path, record)
        if integrity_error:
            verified_records.append(replace(
                record,
                passed=False,
                notes=integrity_error,
            ))
        else:
            verified_records.append(record)
    return evaluate_release_readiness(verified_records, target=target)


def _local_artifact_path(artifact: str, manifest_dir: Path) -> Path | None:
    """Resolve local artifact references; return ``None`` for external URIs."""

    parsed = urlparse(artifact)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme:
        return None
    candidate = Path(artifact).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    return candidate.resolve()


def _local_artifact_integrity_error(
    artifact_path: Path,
    record: EvidenceRecord,
) -> str | None:
    if not artifact_path.is_file():
        return f"local evidence artifact does not exist: {artifact_path}"
    digest = hashlib.sha256()
    with artifact_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != record.artifact_sha256.lower():
        return (
            "local evidence artifact SHA-256 mismatch: "
            f"expected {record.artifact_sha256.lower()}, got {actual}"
        )
    return None


__all__ = [
    "EvidenceRecord",
    "ReleaseRequirement",
    "RequirementResult",
    "ReleaseReadinessReport",
    "release_requirements",
    "evaluate_release_readiness",
    "evidence_manifest_template",
    "load_evidence_manifest",
]
