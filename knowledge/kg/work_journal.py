"""Append-only work journal: which role and model performed each task.

Every line records the acting role, the model identity as known to that role,
input/output fingerprints and the acceptance state. The journal is evidence of
who did what; it never upgrades a scientific object by itself. A Sonnet worker
role cannot record an authoritative scientific verdict (``ACCEPTED_SCIENCE``).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT, store
from .locks import exclusive

ROLES = {
    "S1": "Sonnet S1 - Docling orchestration and corpus processing (mechanical)",
    "S2": "Sonnet S2 - candidate structuring and source metadata (mechanical)",
    "S3": "Sonnet S3 - verified graph integration and deterministic QA (mechanical)",
    "F": "Opus F - scientific reviewer, fixing agent and coordinator",
    "A": "Opus A - final independent auditor",
}
WORKER_ROLES = {"S1", "S2", "S3"}
STATES = {"STARTED", "COMPLETED", "FAILED", "BLOCKED", "PAUSED", "RAW_EXTRACTION_READY",
          "CANDIDATES_PREPARED", "ACCEPTED_SCIENCE", "REJECTED_SCIENCE", "DEFERRED",
          "INTEGRATED", "TESTED_PASS", "TESTED_FAIL", "FIXED_VERIFIED", "REVIEW_REQUIRED",
          "AUDIT_FINDINGS", "AUDIT_PASS"}
AUTHORITATIVE_STATES = {"ACCEPTED_SCIENCE", "REJECTED_SCIENCE", "FIXED_VERIFIED"}
SESSION = "2026-10-16-agentic-continuation"


def journal_path() -> Path:
    return store.REPORT_DIR / "agentic_work_journal.jsonl"


def fingerprint(path: str | Path) -> dict:
    p = Path(path)
    absolute = p if p.is_absolute() else REPO_ROOT / p
    try:
        relative = str(absolute.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        relative = str(absolute)
    if absolute.is_file():
        digest = hashlib.sha256()
        with absolute.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return {"path": relative, "sha256": digest.hexdigest(), "bytes": absolute.stat().st_size}
    return {"path": relative, "sha256": None, "state": "MISSING" if not absolute.exists() else "NOT_A_FILE"}


def append(*, role: str, model: str, task: str, state: str, packet_id: str | None = None,
           inputs: list[str] | None = None, outputs: list[str] | None = None, note: str = "",
           details: dict | None = None) -> dict:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(ROLES)}")
    if state not in STATES:
        raise ValueError(f"Unknown journal state {state!r}")
    if role in WORKER_ROLES and state in AUTHORITATIVE_STATES:
        raise PermissionError(f"Role {role} is mechanical and cannot record {state}")
    if not model or not task:
        raise ValueError("Journal entries require the acting model identity and a task description")
    path = journal_path()
    with exclusive("work-journal", timeout=60, purpose="append journal"):
        seq = 0
        if path.exists():
            with path.open("rb") as stream:
                seq = sum(1 for line in stream if line.strip())
        entry = {"seq": seq + 1, "at": datetime.now(timezone.utc).isoformat(), "session": SESSION,
                 "role": role, "role_description": ROLES[role], "model": model, "task": task,
                 "packet_id": packet_id, "state": state,
                 "inputs": [fingerprint(p) for p in inputs or []],
                 "outputs": [fingerprint(p) for p in outputs or []], "note": note}
        if details:
            entry["details"] = details
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return entry


def read() -> list[dict]:
    path = journal_path()
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Corrupt work-journal line {number}") from exc
    return rows
