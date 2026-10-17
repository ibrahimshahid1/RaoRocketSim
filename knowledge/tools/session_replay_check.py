"""Replay every reviewed file of a session's finished packets and prove the stores do not change.

    .venv-jax/bin/python knowledge/tools/session_replay_check.py --since 2026-10-16 \
        --output knowledge/reports/session_replay_2026-10-16.json

For each FINAL_AUDIT_ELIGIBLE packet with history on or after ``--since``, every
reviewed file is checked against the SHA-256 bound at VERIFIED and re-applied
through the same idempotent paths S3 uses:
- extraction results are re-banked;
- parser reviews and duplicate retirements are re-recorded;
- verdicts are re-applied.
The scientific stores are compared before and after with timestamps ignored.
A non-empty ``stores_changed_ignoring_timestamps`` or ``changed_reviewed_files``
means the replay is not a no-op. ``success`` also requires every step to resolve;
missing source/reviewed files, failed banking and refused new verdicts remain
failures even when they leave the stores unchanged. Exact historical assertions
are retained history, never a new current verification. The run exits 1 unless
both requirements hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from kg import packet_queue as pq, store  # noqa: E402

TIMESTAMPS = {"recorded_at", "generated_at", "harvested_at", "checked_at", "updated_at", "at",
              "registered_at", "reviewed_at", "refreshed_at"}
STORES = ("equations", "claims", "entities", "validity", "models", "provenance", "enrichments", "relationships")


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in TIMESTAMPS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _snapshot() -> dict:
    return {name: hashlib.sha256(json.dumps(_strip(store.load(name)), sort_keys=True).encode()).hexdigest()
            for name in STORES}


def _historical_verdict(verdict: dict, record: dict, documents: dict) -> bool:
    """A previously accepted exact verdict may replay as history without replacing its successor."""
    source_sha = record.get("source_sha256")
    if (not source_sha or documents.get(record.get("document_id"), {}).get("sha256") != source_sha
            or (verdict.get("source_sha256") and verdict["source_sha256"] != source_sha)):
        return False
    result_hash = hashlib.sha256(json.dumps({"verdict": verdict, "source": source_sha}, sort_keys=True).encode()).hexdigest()
    return any(history.get("verdict", {}).get("verifier_result_hash") == result_hash
               and history["verdict"].get("verifier_source_sha256") == source_sha
               for history in record.get("verification_history", []))


def replay(since: str) -> dict:
    from kg.enrichment import record_review
    from kg.harvest_extraction import apply_verification, retire_duplicate_equations
    from tools.agentic_packet import bank_mismatches, verification_failures
    before = _snapshot()
    packets = [p for p in pq.load()["packets"].values() if p["state"] == "FINAL_AUDIT_ELIGIBLE"
               and any(h.get("at", "") >= since for h in p["history"])]
    out = {"since": since, "packets": len(packets), "files": 0, "bank": {}, "verdicts": {},
           "parser_reviews": 0, "duplicate_retirements": {"retired": 0, "unchanged": 0, "refused": 0},
           "changed_reviewed_files": [], "missing_reviewed_files": [], "replay_failures": [],
           "historical_verdicts_skipped": 0, "historical_parser_reviews_skipped": 0,
           "historical_scope": "Exact prior assertions retained as history; skipping them does not verify current content."}
    documents = store.index_by("manifest")
    for packet in sorted(packets, key=lambda p: p["id"]):
        source_path = ROOT / packet.get("source_pdf", "")
        source_sha = packet.get("source_sha256")
        try:
            observed_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            out["replay_failures"].append({"packet": packet["id"], "kind": "SOURCE_FILE_UNAVAILABLE",
                                            "path": packet.get("source_pdf"), "reason": str(exc)})
            continue
        if (not source_sha or observed_sha != source_sha
                or documents.get(packet.get("document_id"), {}).get("sha256") != source_sha):
            out["replay_failures"].append({"packet": packet["id"], "kind": "SOURCE_IDENTITY_MISMATCH",
                                            "path": packet.get("source_pdf"), "expected_sha256": source_sha,
                                            "observed_sha256": observed_sha})
            continue
        for f in packet["evidence"]["review"]["files"]:
            path = ROOT / f["path"]
            out["files"] += 1
            try:
                content = path.read_bytes()
            except OSError as exc:
                out["missing_reviewed_files"].append(f["path"])
                out["replay_failures"].append({"file": f["path"], "kind": "REVIEWED_FILE_UNAVAILABLE", "reason": str(exc)})
                continue
            if hashlib.sha256(content).hexdigest() != f["sha256"]:
                out["changed_reviewed_files"].append(f["path"])
                out["replay_failures"].append({"file": f["path"], "kind": "REVIEWED_FILE_CHANGED"})
                continue
            if path.suffix == ".jsonl":
                with tempfile.TemporaryDirectory(prefix="kg-replay-") as temporary:
                    report_path = Path(temporary) / "bank.json"
                    run = subprocess.run([sys.executable, str(HERE / "bank_extraction.py"), str(path),
                                          "--report-out", str(report_path)], cwd=ROOT, capture_output=True, text=True)
                    try:
                        bank_report = json.loads(report_path.read_text())
                        if not isinstance(bank_report, dict):
                            raise ValueError("bank report must be a JSON object")
                    except (OSError, ValueError) as exc:
                        bank_report = {"bank_report_errors": [{"reason": str(exc)}]}
                failures = bank_mismatches(bank_report)
                if run.returncode or failures:
                    out["replay_failures"].append({"file": f["path"], "kind": "BANKING_FAILED",
                                                    "exit_code": run.returncode, "mismatches": failures,
                                                    "stderr_tail": run.stderr[-2000:]})
                key = "already_banked_no_change" if not run.returncode and bank_report.get("already_banked") else f"exit_{run.returncode}_banked"
                out["bank"][key] = out["bank"].get(key, 0) + 1
                continue
            try:
                payload = json.loads(content)
            except (ValueError, UnicodeError) as exc:
                out["replay_failures"].append({"file": f["path"], "kind": "INVALID_REVIEWED_JSON", "reason": str(exc)})
                continue
            if isinstance(payload, dict) and "parser_reviews" in payload:
                for item in payload["parser_reviews"]:
                    try:
                        stats = record_review(item["candidate_id"], state=item["state"], reason=item["reason"],
                                              evidence=dict(item.get("evidence") or {}, reviewer=payload["reviewer"],
                                                            review_file=f["path"], review_file_sha256=f["sha256"]))
                        out["historical_parser_reviews_skipped"] += stats.get("historical_replays", 0)
                    except (ValueError, KeyError) as exc:
                        out["replay_failures"].append({"file": f["path"], "kind": "PARSER_REVIEW_REFUSED",
                                                        "candidate_id": item.get("candidate_id"), "reason": str(exc)})
                    out["parser_reviews"] += 1
            elif isinstance(payload, dict) and "duplicate_retirements" in payload:
                stats = retire_duplicate_equations(payload)
                out["duplicate_retirements"]["retired"] += stats["retired"]
                out["duplicate_retirements"]["unchanged"] += stats["unchanged"]
                out["duplicate_retirements"]["refused"] += len(stats["refused"])
                if stats["refused"]:
                    out["replay_failures"].append({"file": f["path"], "kind": "DUPLICATE_RETIREMENT_REFUSED",
                                                    "refused": stats["refused"]})
            else:
                if not isinstance(payload, list) or any(not isinstance(v, dict) for v in payload):
                    out["replay_failures"].append({"file": f["path"], "kind": "UNSUPPORTED_REVIEWED_PAYLOAD"})
                    continue
                for verdict in payload:
                    by_id = {r["id"]: r for name in ("equations", "claims", "entities", "validity") for r in store.load(name)}
                    if _historical_verdict(verdict, by_id.get(verdict.get("object_id"), {}), documents):
                        out["historical_verdicts_skipped"] += 1
                        continue
                    try:
                        stats = apply_verification([verdict])
                    except (ValueError, KeyError) as exc:
                        out["replay_failures"].append({"file": f["path"], "kind": "VERDICT_REJECTED",
                                                        "object_id": verdict.get("object_id"), "reason": str(exc)})
                        continue
                    for k, v in stats.items():
                        if isinstance(v, int) and not isinstance(v, bool):
                            out["verdicts"][k] = out["verdicts"].get(k, 0) + v
                    failures = verification_failures(stats)
                    if failures:
                        out["replay_failures"].append({"file": f["path"], "kind": "VERIFICATION_REFUSED",
                                                        "object_id": verdict.get("object_id"), "stats": failures})
    after = _snapshot()
    out["stores_changed_ignoring_timestamps"] = [n for n in STORES if before[n] != after[n]]
    out["no_op"] = (not out["stores_changed_ignoring_timestamps"] and not out["changed_reviewed_files"]
                    and not out["missing_reviewed_files"])
    out["all_steps_resolved"] = not out["replay_failures"]
    out["success"] = out["no_op"] and out["all_steps_resolved"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="ISO date/time; packets with history on or after it")
    ap.add_argument("--output", type=Path)
    ns = ap.parse_args(argv)
    # No outer store lock: bank_extraction runs as a child process and takes the
    # writer lock itself; each in-process operation also locks for itself.
    result = replay(ns.since)
    text = json.dumps(result, indent=1)
    if ns.output:
        ns.output.write_text(text + "\n")
    print(text)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
