"""Bind independent source verification to the scientific content inspected.

Source identity and assertion content are separate fingerprints. Graph links,
parser bookkeeping, timestamps and symbol-resolution indexes are not source
transcription content. Legacy verdicts remain inspectable, but without a content
fingerprint they cannot establish that every current field was inspected.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json


CONTENT_VERSION = "source-assertion-v1"
_DOCUMENT_NOT_CHECKED = object()
CONTENT_FIELDS = (
    "id", "type", "label", "equation_number", "form", "description", "variables",
    "assumptions", "validity", "assumption_ids", "claim_type", "subject", "predicate",
    "object", "statement", "qualifiers", "prose", "applies_to", "machine_readable",
    "machine_readable_state", "name", "value", "units", "context", "document_id",
    "page", "quote",
)


def content_payload(record: dict) -> dict:
    """Canonical source assertion; do not infer missing legacy content."""
    payload = {key: deepcopy(record[key]) for key in CONTENT_FIELDS
               if record.get(key) not in (None, "", [], {})}
    if "variables" in payload:
        payload["variables"] = [
            {k: v[k] for k in ("symbol", "meaning", "units", "role")
             if v.get(k) not in (None, "")}
            for v in payload["variables"]]
    return payload


def content_sha256(record: dict) -> str:
    encoded = json.dumps({"version": CONTENT_VERSION, "assertion": content_payload(record)},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def state(record: dict, document=_DOCUMENT_NOT_CHECKED) -> dict:
    """Effective content-verification state, retaining the historical verdict.

    Omitting ``document`` requests a pure assertion-content check. Passing an
    explicit ``None`` means a manifest lookup failed, so current source evidence
    is unavailable even if an old content fingerprint still matches.
    """
    declared = record.get("verification")
    if document is None:
        return {"state": "SOURCE_NOT_INGESTED", **(
            {"historical_verification": declared} if declared else {})}
    if not declared or declared in ("NOT_CHECKED", "NOT_INDEPENDENTLY_VERIFIED"):
        return {"state": record.get("verification_state", "NOT_CHECKED")}
    if not record.get("active", True) or record.get("source_state", "CURRENT") != "CURRENT":
        return {"state": (record.get("source_state") if record.get("source_state", "CURRENT") != "CURRENT"
                          else "INACTIVE_ASSERTION"),
                "historical_verification": declared}
    if document is not _DOCUMENT_NOT_CHECKED and (not document.get("sha256")
            or record.get("source_sha256") != document.get("sha256")
            or record.get("verifier_source_sha256") != document.get("sha256")):
        return {"state": "STALE_SOURCE", "historical_verification": declared}
    bound = record.get("verifier_content_sha256")
    if not bound:
        return {"state": "LEGACY_CONTENT_NOT_BOUND", "historical_verification": declared}
    if bound != content_sha256(record):
        return {"state": "CONTENT_CHANGED_REVIEW_REQUIRED", "historical_verification": declared}
    return {"state": declared}


def archive(record: dict, *, reason: str, now: str, old_record: dict | None = None) -> None:
    """Keep the complete prior verdict and inspected content before invalidation."""
    old = old_record or record
    verdict = {k: deepcopy(v) for k, v in old.items()
               if k.startswith("verifier_") or k in ("verification", "verified_at")}
    if verdict:
        historical = {"verdict": verdict, "content": content_payload(old),
                      "content_sha256": content_sha256(old),
                      "content_snapshot_basis": (
                          "MATCHES_RECORDED_CONTENT_FINGERPRINT"
                          if old.get("verifier_content_sha256") == content_sha256(old)
                          else "STORED_ASSERTION_AT_INVALIDATION_NOT_PROOF_OF_INSPECTED_LEGACY_CONTENT"),
                      "invalidated_because": reason, "invalidated_at": now}
        history = record.setdefault("verification_history", [])
        if historical not in history:
            history.append(historical)
    for key in list(record):
        if key.startswith("verifier_") or key in ("verification", "verified_at"):
            record.pop(key, None)
    record["verification_state"] = reason


def reconcile_record(old: dict, new: dict, *, now: str) -> bool:
    """Invalidate a verdict when a banking operation changes its source assertion."""
    if not old or not old.get("verification"):
        return False
    source_changed = old.get("source_sha256") != new.get("source_sha256")
    content_changed = content_sha256(old) != content_sha256(new)
    if source_changed or content_changed:
        archive(new, old_record=old, now=now,
                reason="STALE_SOURCE" if source_changed else "CONTENT_CHANGED_REVIEW_REQUIRED")
        return True
    return False


def historical_verdict_hashes(record: dict) -> set[str]:
    return {row["verdict"]["verifier_result_hash"] for row in record.get("verification_history", [])
            if row.get("verdict", {}).get("verifier_result_hash")}
