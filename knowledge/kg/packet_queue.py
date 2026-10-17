"""Persistent, atomic state machine for bounded source packets.

Every packet follows::

    QUEUED -> DOCLING_RUNNING -> RAW_CAPTURED -> CANDIDATES_PREPARED
      -> OPUS_SOURCE_REVIEW -> VERIFIED | REJECTED | BLOCKED
      -> GRAPH_INTEGRATED -> TESTED -> FINAL_AUDIT_ELIGIBLE

Transitions are role-gated: mechanical workers (S1-S3) cannot record a source
review, and graph integration requires the exact reviewed verdict bytes. Raw or
candidate states can never jump to GRAPH_INTEGRATED. Rejected and blocked
packets keep their history and never count as verified. Object-level outcomes
come from the reviewed verdict files, not from the packet label.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT, store, work_journal
from .locks import exclusive

STATES = ("QUEUED", "DOCLING_RUNNING", "RAW_CAPTURED", "CANDIDATES_PREPARED", "OPUS_SOURCE_REVIEW",
          "VERIFIED", "REJECTED", "BLOCKED", "GRAPH_INTEGRATED", "TESTED", "FINAL_AUDIT_ELIGIBLE")
TRANSITIONS: dict[str, dict[str, set[str]]] = {
    "QUEUED": {"DOCLING_RUNNING": {"S1"}, "BLOCKED": {"S1", "F"}},
    "DOCLING_RUNNING": {"RAW_CAPTURED": {"S1"}, "QUEUED": {"S1", "F"}, "BLOCKED": {"S1", "F"}},
    "RAW_CAPTURED": {"CANDIDATES_PREPARED": {"S2"}, "BLOCKED": {"S2", "F"}},
    "CANDIDATES_PREPARED": {"OPUS_SOURCE_REVIEW": {"F"}, "BLOCKED": {"S2", "F"}},
    "OPUS_SOURCE_REVIEW": {"VERIFIED": {"F"}, "REJECTED": {"F"}, "BLOCKED": {"F"}},
    "VERIFIED": {"GRAPH_INTEGRATED": {"S3"}, "BLOCKED": {"S3", "F"}},
    "REJECTED": {"GRAPH_INTEGRATED": {"S3"}, "BLOCKED": {"S3", "F"}},
    "BLOCKED": {"QUEUED": {"F"}},
    "GRAPH_INTEGRATED": {"TESTED": {"S3"}, "BLOCKED": {"S3", "F"}},
    "TESTED": {"FINAL_AUDIT_ELIGIBLE": {"F"}, "BLOCKED": {"F"}},
    "FINAL_AUDIT_ELIGIBLE": {},
}
KINDS = {"verification", "extraction", "model_review"}
REVIEWER_ROLES = {"F"}  # source interpretation authority in this session


class TransitionError(ValueError):
    pass


def queue_path() -> Path:
    return store.REPORT_DIR / "packet_queue.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load() -> dict:
    path = queue_path()
    if not path.exists():
        return {"meta": {"authority": "packet processing ledger; object outcomes live in reviewed verdict files",
                         "states": list(STATES)}, "packets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    store._atomic_write_json(queue_path(), data)


def _locked():
    return exclusive("packet-queue", timeout=120, purpose="packet queue transition")


def add(packet_id: str, *, kind: str, document_id: str, pages: tuple[int, int], targets: list[str],
        priority: int, reason: str, role: str, model: str, dependencies: list[str] | None = None) -> dict:
    if role != "F":
        raise PermissionError("Only the coordinating reviewer (F) schedules source packets")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    document = store.index_by("manifest").get(document_id)
    if not document:
        raise ValueError(f"Unknown document {document_id}")
    first, last = pages
    count = document.get("page_count") or 0
    if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last <= count):
        raise ValueError(f"Pages {pages} are outside {document_id} (1..{count})")
    with _locked():
        data = load()
        if packet_id in data["packets"]:
            raise ValueError(f"Packet {packet_id} already exists; packets are never silently replaced")
        unknown = [d for d in dependencies or [] if d not in data["packets"]]
        if unknown:
            raise ValueError(f"Unknown dependency packets: {unknown}")
        packet = {"id": packet_id, "kind": kind, "document_id": document_id,
                  "source_pdf": document["source_pdf"], "source_sha256": document["sha256"],
                  "pages": [first, last], "targets": sorted(set(targets)), "priority": int(priority),
                  "reason": reason, "dependencies": list(dependencies or []), "state": "QUEUED",
                  "attempts": 0, "created_at": _now(), "evidence": {},
                  "history": [{"from": None, "to": "QUEUED", "role": role, "model": model, "at": _now(),
                               "note": reason}]}
        data["packets"][packet_id] = packet
        _save(data)
    work_journal.append(role=role, model=model, task=f"schedule {kind} packet", state="COMPLETED",
                        packet_id=packet_id, note=reason,
                        details={"document_id": document_id, "pages": [first, last], "targets": packet["targets"]})
    return packet


def _verdict_summary(paths: list[Path]) -> dict:
    """Count object outcomes in reviewed files; every verdict must name reviewer F."""
    summary = {"confirmed": 0, "refuted": 0, "uncertain": 0, "stale_or_unknown": 0,
               "extracted_objects": 0, "model_reviews": 0, "files": []}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        summary["files"].append({"path": _relative(path), "sha256": _sha256(path)})
        rows = []
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("type") == "result":
                        rows.append(row.get("value") or row.get("result"))
            for value in rows:
                reviewer = (value or {}).get("reviewer") or {}
                if reviewer.get("role") not in REVIEWER_ROLES:
                    raise TransitionError(f"{path}: result {value.get('target')} lacks reviewer role F")
                summary["extracted_objects"] += sum(len(value.get(k) or []) for k in (
                    "equations", "claims", "validity_statements", "empirical_constants", "assumptions"))
                summary["extracted_objects"] += len(value.get("cross_document_relations") or [])
                summary["model_reviews"] += int(str(value.get("target", "")).startswith("MODEL_")
                                                and bool(value.get("supports_model")))
        else:
            verdicts = json.loads(text)
            if isinstance(verdicts, dict) and "parser_reviews" in verdicts:
                # Parser QA: whether a Docling candidate matched the original page.
                if (verdicts.get("reviewer") or {}).get("role") not in REVIEWER_ROLES:
                    raise TransitionError(f"{path}: parser review lacks reviewer role F")
                summary["parser_reviews"] = summary.get("parser_reviews", 0) + len(verdicts["parser_reviews"])
                continue
            if isinstance(verdicts, dict) and "duplicate_retirements" in verdicts:
                # Identity repair: an exact duplicate retired in favour of a canonical record.
                if (verdicts.get("reviewer") or {}).get("role") not in REVIEWER_ROLES:
                    raise TransitionError(f"{path}: duplicate retirement lacks reviewer role F")
                summary["duplicate_retirements"] = (summary.get("duplicate_retirements", 0)
                                                    + len(verdicts["duplicate_retirements"]))
                continue
            if not isinstance(verdicts, list):
                raise TransitionError(f"{path}: verdict file must be a JSON list")
            for v in verdicts:
                reviewer = v.get("reviewer") or {}
                if reviewer.get("role") not in REVIEWER_ROLES:
                    raise TransitionError(f"{path}: verdict {v.get('object_id')} lacks reviewer role F")
                if v.get("refuted") is True:
                    summary["refuted"] += 1
                elif v.get("confidence") == "low":
                    summary["uncertain"] += 1
                elif v.get("refuted") is False:
                    summary["confirmed"] += 1
                else:
                    summary["stale_or_unknown"] += 1
    return summary


def transition(packet_id: str, to_state: str, *, role: str, model: str, note: str = "",
               evidence_paths: list[str] | None = None, run_ids: list[str] | None = None,
               verdict_paths: list[str] | None = None, test_result: str | None = None) -> dict:
    if to_state not in STATES:
        raise ValueError(f"Unknown state {to_state}")
    with _locked():
        data = load()
        packet = data["packets"].get(packet_id)
        if packet is None:
            raise ValueError(f"Unknown packet {packet_id}")
        current = packet["state"]
        allowed = TRANSITIONS[current]
        if to_state not in allowed:
            raise TransitionError(f"{packet_id}: {current} -> {to_state} is not a permitted transition")
        if role not in allowed[to_state]:
            raise PermissionError(f"{packet_id}: role {role} may not perform {current} -> {to_state}")
        document = store.index_by("manifest").get(packet["document_id"]) or {}
        if to_state != "BLOCKED" and document.get("sha256") != packet["source_sha256"]:
            raise TransitionError(f"{packet_id}: source PDF fingerprint changed; block and reschedule")
        if to_state in ("BLOCKED", "QUEUED") and not note:
            raise TransitionError(f"{packet_id}: {to_state} requires a recorded reason")
        record_evidence = {}
        for raw in evidence_paths or []:
            path = _resolve(raw)
            if not path.is_file():
                raise TransitionError(f"{packet_id}: evidence file missing: {raw}")
            record_evidence[_relative(path)] = _sha256(path)
        step = {"from": current, "to": to_state, "role": role, "model": model, "at": _now(), "note": note,
                "evidence": record_evidence}
        if current == "QUEUED" and to_state == "DOCLING_RUNNING":
            packet["attempts"] += 1
            step["attempt"] = packet["attempts"]
        if to_state == "RAW_CAPTURED":
            runs = {r["id"]: r for r in store.load("enrichments") if r.get("type") == "GeneratedDocument"}
            if not run_ids:
                raise TransitionError(f"{packet_id}: RAW_CAPTURED requires the Docling run ID(s)")
            first, last = packet["pages"]
            covered = set()
            for rid in run_ids:
                run = runs.get(rid)
                if (not run or run.get("document_id") != packet["document_id"]
                        or run.get("source_sha256") != packet["source_sha256"]
                        or run.get("conversion_status") not in ("success", "partial_success")):
                    raise TransitionError(f"{packet_id}: {rid} is not a successful current run of this source")
                covered.update(p for p in run.get("observed_pages", []) if first <= p <= last)
                if run.get("review_state") == "REJECTED":
                    step.setdefault("rejected_parser_runs", []).append(rid)
            step["run_ids"] = sorted(run_ids)
            step["observed_pages"] = sorted(covered)
            missing = sorted(set(range(first, last + 1)) - covered)
            if missing:
                step["unobserved_pages"] = missing
            packet["evidence"]["docling_run_ids"] = sorted(set(packet["evidence"].get("docling_run_ids", [])) | set(run_ids))
        if to_state == "CANDIDATES_PREPARED":
            if not record_evidence:
                raise TransitionError(f"{packet_id}: CANDIDATES_PREPARED requires the structured candidate bundle")
            packet["evidence"]["candidate_bundles"] = record_evidence
        if to_state in ("VERIFIED", "REJECTED"):
            paths = [_resolve(p) for p in verdict_paths or []]
            if not paths or not all(p.is_file() for p in paths):
                raise TransitionError(f"{packet_id}: a review outcome requires reviewed verdict/result file(s)")
            outcome = _verdict_summary(paths)
            positive = (outcome["confirmed"] + outcome["extracted_objects"] + outcome["model_reviews"]
                        + outcome.get("duplicate_retirements", 0))
            if to_state == "VERIFIED" and positive == 0:
                raise TransitionError(f"{packet_id}: VERIFIED requires at least one confirmed or source-reviewed object")
            if to_state == "REJECTED" and positive:
                raise TransitionError(f"{packet_id}: REJECTED cannot carry confirmed objects; use VERIFIED with object outcomes")
            packet["evidence"]["review"] = outcome
            step["outcome"] = {k: v for k, v in outcome.items() if k != "files"}
        if to_state == "GRAPH_INTEGRATED":
            review = packet["evidence"].get("review")
            if not review:
                raise TransitionError(f"{packet_id}: no reviewed verdict to integrate")
            for f in review["files"]:
                path = _resolve(f["path"])
                if not path.is_file() or _sha256(path) != f["sha256"]:
                    raise TransitionError(f"{packet_id}: reviewed file changed after review: {f['path']}")
            if not record_evidence:
                raise TransitionError(f"{packet_id}: GRAPH_INTEGRATED requires the integration log")
            packet["evidence"]["integration"] = record_evidence
        if to_state == "TESTED":
            if test_result != "pass" or not record_evidence:
                raise TransitionError(f"{packet_id}: TESTED requires a passing test log; block failures for repair")
            packet["evidence"]["tests"] = record_evidence
        packet["state"] = to_state
        packet["history"].append(step)
        packet["updated_at"] = step["at"]
        _save(data)
    journal_state = {"VERIFIED": "ACCEPTED_SCIENCE", "REJECTED": "REJECTED_SCIENCE", "BLOCKED": "BLOCKED",
                     "RAW_CAPTURED": "RAW_EXTRACTION_READY", "CANDIDATES_PREPARED": "CANDIDATES_PREPARED",
                     "GRAPH_INTEGRATED": "INTEGRATED", "TESTED": "TESTED_PASS"}.get(to_state, "COMPLETED")
    work_journal.append(role=role, model=model, task=f"packet {current} -> {to_state}", state=journal_state,
                        packet_id=packet_id, inputs=list(evidence_paths or []) + list(verdict_paths or []),
                        note=note)
    return packet


def reprioritize(packet_id: str, priority: int, *, role: str, model: str, note: str) -> dict:
    """Coordinator-only priority change; recorded in history, never a state change."""
    if role != "F":
        raise PermissionError("Only the coordinating reviewer (F) changes packet priority")
    if not note:
        raise TransitionError("A priority change requires a recorded reason")
    with _locked():
        data = load()
        packet = data["packets"].get(packet_id)
        if packet is None:
            raise ValueError(f"Unknown packet {packet_id}")
        step = {"from": packet["state"], "to": packet["state"], "role": role, "model": model, "at": _now(),
                "note": note, "priority_from": packet["priority"], "priority_to": int(priority)}
        packet["priority"] = int(priority)
        packet["history"].append(step)
        packet["updated_at"] = step["at"]
        _save(data)
    work_journal.append(role=role, model=model, task=f"reprioritize packet to {priority}", state="COMPLETED",
                        packet_id=packet_id, note=note)
    return packet


def claim_next(state: str, to_state: str, *, role: str, model: str, kind: str | None = None,
               min_priority: int | None = None) -> dict | None:
    """Atomically select the highest-priority eligible packet and advance it."""
    with _locked():
        data = load()
        done = {"TESTED", "FINAL_AUDIT_ELIGIBLE"}
        eligible = [p for p in data["packets"].values() if p["state"] == state
                    and (kind is None or p["kind"] == kind)
                    and (min_priority is None or p["priority"] >= min_priority)
                    and all(data["packets"][d]["state"] in done for d in p["dependencies"])]
        if not eligible:
            return None
        chosen = sorted(eligible, key=lambda p: (-p["priority"], p["created_at"], p["id"]))[0]
        return transition(chosen["id"], to_state, role=role, model=model, note=f"claimed from {state}")


def summary() -> dict:
    data = load()
    packets = list(data["packets"].values())
    by_state = {s: sum(p["state"] == s for p in packets) for s in STATES}
    objects = {"confirmed": 0, "refuted": 0, "uncertain": 0, "extracted_objects": 0, "model_reviews": 0}
    for p in packets:
        review = p.get("evidence", {}).get("review") or {}
        if p["state"] in ("GRAPH_INTEGRATED", "TESTED", "FINAL_AUDIT_ELIGIBLE", "VERIFIED", "REJECTED"):
            for key in objects:
                objects[key] += review.get(key, 0)
    return {"packets": len(packets), "by_state": by_state, "reviewed_object_outcomes": objects,
            "note": "Object outcomes come from reviewed verdict files; a packet label alone is not verification."}
