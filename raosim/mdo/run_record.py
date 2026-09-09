"""Isolated, atomic report publication with an explicit completeness manifest.

Report completeness and design feasibility are separate: a complete report can
describe a failed solve. No file in this package is a hardware release claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RunRecord:
    directory: Path
    requested: tuple[str, ...]
    allow_partial: bool
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, job: Any) -> RunRecord | None:
        policy = job.spec.output_policy
        if policy.output_dir is None:
            return None
        run_id = uuid.uuid4().hex
        directory = Path(policy.output_dir).expanduser().resolve() / run_id
        directory.mkdir(parents=True, exist_ok=False)
        requested = tuple(dict.fromkeys(("summary", *policy.reports)))
        record = cls(directory, requested, policy.allow_partial_package, {
            "schema_version": "1.0.0", "run_id": run_id,
            "job_digest": job.job_digest, "package_kind": "analysis_reports",
            "package_status": "in_progress", "run_status": "in_progress",
            "requested_reports": list(requested), "reports": {},
            "hardware_release": False,
        })
        record._save()
        digest = _write_json(directory / "job.json", job.spec.to_document())
        record.manifest["submitted_job"] = {"path": "job.json", "sha256": digest}
        record._save()
        return record

    def _save(self) -> None:
        _write_json(self.directory / "manifest.json", self.manifest)

    def mark_failed(self, reason: str, *, result: Any = None) -> None:
        self.manifest.update(package_status="partial", run_status="failed", error=reason)
        # If the destination itself has become unwritable, the original
        # in-progress manifest remains an explicit non-complete marker.
        try:
            if result is not None:
                self.manifest.update(verdict=result.verdict, exit_code=result.exit_code)
                digest = _write_json(self.directory / "summary.json", result.to_dict())
                self.manifest["reports"]["summary"] = {
                    "path": "summary.json", "sha256": digest, "status": "available",
                }
            self._save()
        except (OSError, TypeError, ValueError):
            pass

    def finish(self, result: Any) -> None:
        comparison = (result.reevaluation.comparison
                      if result.reevaluation is not None else None)
        payloads = {
            "summary": result.to_dict(),
            "resolved_inputs": result.inputs.to_dict() if result.inputs is not None else None,
            "snapshot": result.snapshot.to_dict() if result.snapshot is not None else None,
            "comparison": comparison.to_dict() if comparison is not None else None,
            "coverage": {
                "statuses": dict(result.statuses),
                "requirements": [dict(row) for row in result.requirement_coverage],
            },
        }
        unavailable = []
        for name in self.requested:
            payload = payloads[name]
            status = "available"
            if payload is None:
                unavailable.append(name)
                status = "unavailable"
                payload = {"status": status, "reason": "execution did not produce this report",
                           "verdict": result.verdict, "messages": list(result.messages)}
            filename = f"{name}.json"
            digest = _write_json(self.directory / filename, payload)
            self.manifest["reports"][name] = {
                "path": filename, "sha256": digest, "status": status,
            }
            self._save()
        self.manifest.update(
            package_status="partial" if unavailable else "complete",
            run_status="failed" if result.verdict == "SOLVER_FAILED" else "completed",
            verdict=result.verdict, exit_code=result.exit_code,
            unavailable_reports=unavailable,
        )
        self._save()
        if unavailable and not self.allow_partial:
            raise RuntimeError("requested reports unavailable: " + ", ".join(unavailable))
