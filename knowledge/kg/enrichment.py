"""Docling derivatives are inspectable candidates, never automatic source support.

Keep the reviewed Markdown/chunks and scientific stores intact. A formula model
can hallucinate notation; byte hashes and bounding boxes prove traceability, not
correctness. Reviewed equations still enter through bank_extraction.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT
from .locks import writes_stores
from .store import index_by, load, save


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


LEGACY_METADATA_SCOPE = "Legacy metadata has no v2 content-address binding; raw/PDF identity checks do not verify conversion settings or metadata integrity."
V2_METADATA_SCOPE = "v2 metadata, source/options/package/model identity and artifact filenames are content-bound; this is not transcription correctness or model-runtime attestation."


def _v2_filename(path: Path) -> bool:
    return bool(re.search(r"-[0-9a-f]{64}(?:\.meta)?\.(?:json|md)$", path.name))


def validate_metadata_binding(meta: dict, metadata_path: Path, raw_path: Path, *, require_v2=False) -> dict:
    """Validate the publisher's complete v2 address without changing any file.

    Legacy artifacts retain unavailable metadata integrity. A v2-looking file or
    partial v2 metadata cannot silently downgrade to that legacy exception.
    """
    if not isinstance(meta, dict):
        raise ValueError("Docling metadata must be an object")
    modern = (require_v2 or meta.get("artifact_schema_version") == 2 or _v2_filename(raw_path)
              or any(key in meta for key in ("artifact_identity_sha256", "input_identity_sha256")))
    if not modern:
        return {"state": "UNAVAILABLE", "scope": LEGACY_METADATA_SCOPE, "schema_version": None}
    if type(meta.get("artifact_schema_version")) is not int or meta["artifact_schema_version"] != 2:
        raise ValueError("v2 Docling metadata schema missing/changed; legacy downgrade refused")
    try:
        inputs = {key: meta[key] for key in ("document_id", "source_pdf", "source_sha256", "page_range",
                                             "options", "versions", "model_revisions")}
        if not meta["attempt_id"] or not meta["attempt_created_at"]:
            raise ValueError("v2 attempt identity is empty")
        if _digest(inputs) != meta["input_identity_sha256"]:
            raise ValueError("v2 Docling input identity checksum mismatch")
        core = {key: value for key, value in meta.items() if key not in (
            "raw_artifact", "markdown_artifact", "metadata_artifact", "artifact_identity_sha256")}
        address = _digest(core)
        if address != meta["artifact_identity_sha256"]:
            raise ValueError("v2 Docling metadata content-address mismatch")
        first, last = meta["page_range"]
        mode = "regions" if meta["options"].get("mode") == "visually_located_formula_regions" else f"p{first}-{last}"
        stem = f"{meta['source_sha256'][:12]}-{mode}-{address}"
        if raw_path.name != stem + ".json" or metadata_path.name != stem + ".meta.json":
            raise ValueError("v2 Docling artifact filename does not match content address")
        markdown_path = raw_path.with_suffix(".md")
        if (meta["raw_artifact"] != _relative(raw_path)
                or meta["metadata_artifact"] != _relative(metadata_path)
                or meta["markdown_artifact"] != _relative(markdown_path)
                or metadata_path.parent.resolve() != raw_path.parent.resolve()):
            raise ValueError("v2 Docling artifact path binding mismatch")
        if not meta["raw_sha256"] or not meta["markdown_sha256"]:
            raise ValueError("v2 output fingerprint is empty")
    except (KeyError, TypeError) as exc:
        raise ValueError("v2 Docling metadata is incomplete") from exc
    return {"state": "VERIFIED", "scope": V2_METADATA_SCOPE, "schema_version": 2}


def _observed_pages(payload: dict) -> list[int]:
    """Read original page identities from the artifact, never metadata guesses."""
    if "pages" in payload:
        return sorted({int(p) for p in payload["pages"] if str(p).isdigit() and int(p) > 0})
    return sorted({p["page_no"] for name in ("texts", "tables", "pictures")
                   for item in payload.get(name, []) for p in item.get("prov", [])
                   if type(p.get("page_no")) is int and p["page_no"] > 0})


def _requested_observations(record: dict) -> set[int]:
    first, last = record.get("page_range", [0, -1])
    return {p for p in record.get("observed_pages", [])
            if type(p) is int and first <= p <= last}


def candidates(payload: dict, meta: dict) -> list[dict]:
    """Retain all formula/table detections including empty/failed transcriptions."""
    first, last = meta["page_range"]
    if type(first) is not int or type(last) is not int or not 1 <= first <= last:
        raise ValueError("Expected a positive original-PDF page range")
    observed = _observed_pages(payload)
    outside = [p for p in observed if not first <= p <= last]
    identity = {k: meta[k] for k in (
        "document_id", "source_sha256", "page_range", "options", "versions", "raw_sha256")}
    # Preserve IDs of legacy runs; new immutable attempts carry these fields.
    for key in ("model_revisions", "attempt_id"):
        if key in meta:
            identity[key] = meta[key]
    run_id = "ENR_" + _digest(identity)[:24]
    base = {"document_id": meta["document_id"], "source_sha256": meta["source_sha256"],
            "source_pdf": meta["source_pdf"], "source_state": "CURRENT",
            "provenance_kind": "SYSTEM_INFERS", "parser": "docling",
            "raw_artifact": meta["raw_artifact"], "raw_sha256": meta["raw_sha256"],
            "review_state": "UNREVIEWED", "hardware_qualified": False,
            **{key: meta.get(key) for key in ("metadata_artifact", "metadata_sha256", "metadata_schema_version",
                                             "artifact_identity_sha256", "markdown_artifact", "markdown_sha256")},
            "metadata_integrity_state": meta.get("metadata_integrity_state", "UNAVAILABLE"),
            "metadata_integrity_scope": meta.get("metadata_integrity_scope", LEGACY_METADATA_SCOPE)}
    records = [{**base, "id": run_id, "type": "GeneratedDocument",
                "label": "Docling enrichment " + meta["document_id"],
                "page_range": meta["page_range"], "options": meta["options"],
                "versions": meta["versions"], "model_revisions": meta.get("model_revisions", {}),
                "attempt_id": meta.get("attempt_id"),
                "conversion_status": meta["conversion_status"], "elapsed_seconds": meta.get("elapsed_seconds"),
                "observed_pages": observed, "out_of_range_pages": outside,
                "page_observation_state": "OUT_OF_REQUESTED_RANGE" if outside else "WITHIN_REQUESTED_RANGE",
                "metadata_artifact": meta.get("metadata_artifact"), "active": True}]
    for collection in ("texts", "tables", "pictures"):
        for i, item in enumerate(payload.get(collection, [])):
            label = item.get("label", "")
            if label not in ("formula", "table", "picture"):
                continue
            prov = item.get("prov", [])
            pages = sorted({p["page_no"] for p in prov if isinstance(p.get("page_no"), int)})
            # Never remap page numbers by guessing an offset.
            located = bool(pages) and all(first <= p <= last for p in pages)
            content = (item.get("text", "") if label == "formula" else
                       item.get("data", {}) if label == "table" else
                       {"captions": item.get("captions", []), "annotations": item.get("annotations", [])})
            candidate_kind = "figure" if label == "picture" else label
            records.append({**base, "id": "CAND_" + _digest([run_id, collection, i])[:24],
                "type": {"table": "Table", "picture": "Figure"}.get(label, "ExtractionCandidate"),
                "label": f"Docling {candidate_kind} p{pages}",
                "candidate_kind": candidate_kind, "content": content, "original_text": item.get("orig", ""),
                "run_id": run_id, "json_pointer": f"/{collection}/{i}",
                "item_ref": item.get("self_ref"), "pages": pages, "page_provenance": prov,
                "location_state": "LOCATED" if located else "UNRESOLVED_PAGE",
                "transcription_state": "CANDIDATE_UNVERIFIED" if content else "EMPTY_TRANSCRIPTION",
                "active": True})
    return records


@writes_stores("integrate Docling candidates")
def integrate(raw_path: Path, meta_path: Path) -> dict:
    metadata_bytes = meta_path.read_bytes()
    meta = json.loads(metadata_bytes)
    binding = validate_metadata_binding(meta, meta_path, raw_path)
    document = index_by("manifest").get(meta["document_id"])
    if not document or document.get("sha256") != meta["source_sha256"]:
        raise ValueError("Source manifest changed; repeat enrichment of the current PDF")
    pdf = REPO_ROOT / meta["source_pdf"]
    if not pdf.exists() or sha256(pdf) != meta["source_sha256"]:
        raise ValueError("PDF bytes missing/changed; cannot integrate candidates")
    if sha256(raw_path) != meta["raw_sha256"]:
        raise ValueError("Raw Docling artifact checksum mismatch")
    if meta["conversion_status"] not in ("success", "partial_success"):
        raise ValueError("Failed conversions cannot be integrated")
    if binding["state"] == "VERIFIED":
        markdown = REPO_ROOT / meta["markdown_artifact"]
        if not markdown.is_file() or sha256(markdown) != meta["markdown_sha256"]:
            raise ValueError("v2 Docling Markdown artifact missing/changed")
    previous = {r["id"]: r for r in load("enrichments")}
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    if binding["state"] == "VERIFIED" and any(
            r.get("metadata_artifact") == _relative(meta_path) and r.get("metadata_sha256")
            and r["metadata_sha256"] != metadata_sha for r in previous.values()):
        raise ValueError("v2 Docling metadata bytes differ from the previously integrated immutable artifact")
    meta["raw_artifact"] = _relative(raw_path)
    meta["metadata_artifact"] = _relative(meta_path)
    meta.update(metadata_sha256=metadata_sha,
                metadata_integrity_state=binding["state"], metadata_integrity_scope=binding["scope"],
                metadata_schema_version=binding["schema_version"])
    payload = json.loads(raw_path.read_text())
    new = candidates(payload, meta)
    for row in new:
        old = previous.get(row["id"], {})
        if old.get("source_sha256") == row["source_sha256"] and old.get("raw_sha256") == row["raw_sha256"]:
            for key in ("review_state", "review_reason", "review_evidence", "review_sha256",
                        "review_source_identity", "review_recorded_at", "review_history"):
                if key in old:
                    row[key] = old[key]
            if old.get("review_state") == "REJECTED":
                row["active"] = False
    previous.update({r["id"]: r for r in new})
    save("enrichments", previous.values(), meta={"authority": "unreviewed parser candidates"})
    return {"run_id": new[0]["id"], "candidates": len(new) - 1,
            "formulas": sum(r.get("candidate_kind") == "formula" for r in new),
            "tables": sum(r.get("candidate_kind") == "table" for r in new),
            "figures": sum(r.get("candidate_kind") == "figure" for r in new),
            "state": "CANDIDATES_INTEGRATED_REVIEW_REQUIRED"}


@writes_stores("refresh Docling derivatives")
def refresh() -> dict:
    """Invalidate missing/changed sources or raw artifacts without erasing history."""
    docs = index_by("manifest")
    records = load("enrichments")
    hashes = {}
    runs = {r["id"]: r for r in records if r.get("type") == "GeneratedDocument"}
    metadata = {}
    for r in records:
        d = docs.get(r["document_id"], {})
        pdf = REPO_ROOT / r["source_pdf"]
        raw = REPO_ROOT / r["raw_artifact"]
        reason = ""
        for path, expected, description in (
            (pdf, r["source_sha256"], "source PDF"), (raw, r["raw_sha256"], "raw Docling artifact")):
            if path not in hashes:
                hashes[path] = sha256(path) if path.exists() else None
            if hashes[path] != expected:
                reason = description + " missing or changed"
                break
        if not reason and d.get("sha256") != r["source_sha256"]:
            reason = "source manifest changed"
        parent = runs.get(r.get("run_id"), {})
        metadata_ref = r.get("metadata_artifact") or parent.get("metadata_artifact")
        require_v2 = (r.get("metadata_schema_version") == 2 or parent.get("metadata_schema_version") == 2
                      or _v2_filename(raw))
        meta_path = REPO_ROOT / metadata_ref if metadata_ref else raw.with_suffix(".meta.json") if require_v2 else None
        key = (meta_path, raw, require_v2)
        if key not in metadata:
            modern_expected = require_v2
            try:
                if meta_path is None or not meta_path.is_file():
                    if require_v2:
                        raise ValueError("v2 Docling metadata artifact missing")
                    metadata[key] = (None, None, {"state": "UNAVAILABLE", "scope": LEGACY_METADATA_SCOPE})
                else:
                    blob = meta_path.read_bytes()
                    meta = json.loads(blob)
                    modern_expected = modern_expected or (isinstance(meta, dict) and (
                        meta.get("artifact_schema_version") == 2 or any(k in meta for k in (
                            "artifact_identity_sha256", "input_identity_sha256"))))
                    binding = validate_metadata_binding(meta, meta_path, raw, require_v2=require_v2)
                    metadata[key] = (meta, hashlib.sha256(blob).hexdigest(), binding)
            except (OSError, ValueError, TypeError) as exc:
                metadata[key] = (None, None, {"state": "FAILED" if modern_expected else "UNAVAILABLE",
                                             "scope": str(exc) if modern_expected else LEGACY_METADATA_SCOPE})
        meta, observed_sha, binding = metadata[key]
        expected_sha = r.get("metadata_sha256") or parent.get("metadata_sha256")
        metadata_reason = ""
        if binding["state"] == "FAILED":
            metadata_reason = binding["scope"]
        elif binding["state"] == "VERIFIED":
            if expected_sha and expected_sha != observed_sha:
                metadata_reason = "v2 Docling metadata bytes changed"
            elif (meta["document_id"] != r["document_id"] or meta["source_sha256"] != r["source_sha256"]
                  or meta["raw_sha256"] != r["raw_sha256"]):
                metadata_reason = "v2 Docling metadata no longer matches stored run identity"
            else:
                markdown = REPO_ROOT / meta["markdown_artifact"]
                if markdown not in hashes:
                    hashes[markdown] = sha256(markdown) if markdown.is_file() else None
                if hashes[markdown] != meta["markdown_sha256"]:
                    metadata_reason = "v2 Docling Markdown artifact missing or changed"
            if not metadata_reason:
                r.update(metadata_artifact=_relative(meta_path), metadata_sha256=observed_sha,
                         metadata_schema_version=2, artifact_identity_sha256=meta["artifact_identity_sha256"],
                         markdown_artifact=meta["markdown_artifact"], markdown_sha256=meta["markdown_sha256"])
        r.update(metadata_integrity_state="FAILED" if metadata_reason else binding["state"],
                 metadata_integrity_scope=metadata_reason or binding["scope"])
        if metadata_reason:
            reason = reason or metadata_reason
            if observed_sha:
                r["metadata_observed_sha256"] = observed_sha
        # A previously invalidated derivative requires explicit re-integration.
        if reason:
            r.update(source_state="STALE_SOURCE", active=False, source_state_reason=reason)
    if records:
        save("enrichments", records, meta={"authority": "unreviewed parser candidates"})
    return {"records": len(records), "inactive": sum(not r.get("active", True) for r in records)}


def graph_records() -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for r in load("enrichments"):
        nodes.append(dict(r))
        if not r.get("active", True) or r.get("source_state") == "STALE_SOURCE":
            continue
        edges.append({"from": r["id"], "to": r.get("run_id") or r["document_id"],
                      "type": "DERIVED_FROM", "provenance": "SYSTEM_INFERS:docling",
                      "review_state": r.get("review_state", "UNREVIEWED")})
    return nodes, edges


def _review_identity(record: dict) -> dict:
    return {key: record.get(key) for key in ("document_id", "source_sha256", "raw_sha256")}


def _review_assertion(state: str, reason: str, evidence: dict, source_identity: dict) -> dict:
    assertion = {"state": state, "reason": reason, "evidence": deepcopy(evidence),
                 "source_identity": deepcopy(source_identity)}
    return {**assertion, "review_sha256": _digest(assertion)}


def _current_review(record: dict) -> dict | None:
    if (record.get("review_state") not in {"REJECTED", "SOURCE_PAGE_COMPARED"}
            or not record.get("review_reason") or not record.get("review_evidence")):
        return None
    review = _review_assertion(record["review_state"], record["review_reason"], record["review_evidence"],
                               record.get("review_source_identity") or _review_identity(record))
    if record.get("review_recorded_at"):
        review["recorded_at"] = record["review_recorded_at"]
    return review


def review_replay_state(record: dict, *, state: str, reason: str, evidence: dict) -> str | None:
    """Recognize an exact current-source assertion without changing review state."""
    incoming = _review_assertion(state, reason, evidence, _review_identity(record))
    current = _current_review(record)
    if current and current["review_sha256"] == incoming["review_sha256"]:
        return "CURRENT"
    for old in record.get("review_history", []):
        known = _review_assertion(old["state"], old["reason"], old["evidence"], old["source_identity"])
        if known["review_sha256"] == incoming["review_sha256"]:
            return "HISTORICAL"
    return None


@writes_stores("record Docling parser review")
def record_review(run_id: str, *, state: str, reason: str, evidence: dict, historical_only: bool = False) -> dict:
    """Record parser QA separately from scientific-object verification.

    The read-modify-write holds the store-writer lock, and the store is rewritten
    only when a genuinely new assertion arrives. Current and superseded exact
    assertions replay as no-ops. ``historical_only`` recovers a fingerprinted
    prior review without replacing the latest state or reactivating candidates.
    """
    if (state not in {"REJECTED", "SOURCE_PAGE_COMPARED"} or not isinstance(reason, str) or not reason
            or not isinstance(evidence, dict) or not evidence):
        raise ValueError("Parser review requires explicit state, reason and inspection evidence")
    from .locks import writer
    with writer(purpose="record parser review"):
        records = load("enrichments")
        selected = [r for r in records if r["id"] == run_id or r.get("run_id") == run_id]
        if not selected:
            raise ValueError("Unknown enrichment run")
        documents = index_by("manifest")
        for r in selected:
            current_sha = documents.get(r.get("document_id"), {}).get("sha256")
            if (not current_sha or r.get("source_sha256") != current_sha or r.get("source_state") != "CURRENT"
                    or (evidence.get("source_sha256") and evidence["source_sha256"] != current_sha)
                    or (evidence.get("raw_sha256") and evidence["raw_sha256"] != r.get("raw_sha256"))):
                raise ValueError(f"Parser review source/raw identity is stale or mismatched: {r['id']}")
        changed = False
        unchanged = historical_replays = historical_added = 0
        for r in selected:
            incoming = _review_assertion(state, reason, evidence, _review_identity(r))
            current = _current_review(r)
            replay_state = review_replay_state(r, state=state, reason=reason, evidence=evidence)
            if replay_state == "CURRENT":
                unchanged += 1
                continue
            history = r.get("review_history", [])
            if replay_state == "HISTORICAL":
                historical_replays += 1
                continue
            history = list(history)
            if historical_only:
                history.append({**incoming, "recovered_as_history_only": True})
                r["review_history"] = history
                historical_added += 1
            else:
                if current:
                    history.append(current)
                r.update(review_state=state, review_reason=reason, review_evidence=deepcopy(evidence),
                         review_sha256=incoming["review_sha256"], review_source_identity=incoming["source_identity"],
                         review_recorded_at=datetime.now(timezone.utc).isoformat(), review_history=history)
                if state == "REJECTED":
                    r["active"] = False
            changed = True
        if changed:
            save("enrichments", records, meta={"authority": "parser derivatives; QA is separate from science verification"})
    return {"reviewed_records": len(selected), "state": state, "changed": changed,
            "unchanged": unchanged, "historical_replays": historical_replays, "historical_added": historical_added}


def find(query: str = "", document_id: str | None = None, limit: int = 20) -> dict:
    terms = query.lower().split()
    rows = [r for r in load("enrichments") if r.get("candidate_kind")
            and r.get("active", True) and r.get("source_state") != "STALE_SOURCE"
            and (not document_id or r["document_id"] == document_id)
            and all(t in json.dumps(r, ensure_ascii=False).lower() for t in terms)]
    return {"state": "CANDIDATES_REQUIRE_SOURCE_REVIEW" if rows else "NO_ENRICHMENT_CANDIDATES",
            "total": len(rows), "candidates": rows[:limit]}


def queue() -> dict:
    """Full corpus processing ledger, separate from mathematical review coverage."""
    processed = {}
    rejected = {}
    rejected_regions = {}
    records = load("enrichments")
    for r in records:
        if (r.get("type") == "GeneratedDocument" and r.get("source_state") == "CURRENT"
                and r.get("review_state") == "REJECTED"):
            backlog = (rejected_regions if r.get("options", {}).get("mode") == "visually_located_formula_regions"
                       else rejected)
            backlog.setdefault((r["document_id"], r["source_sha256"]), set()).update(_requested_observations(r))
        if r.get("type") == "GeneratedDocument" and r.get("active", True) and r.get("source_state") == "CURRENT" and r.get("conversion_status") == "success" and r.get("options", {}).get("mode") != "visually_located_formula_regions":
            processed.setdefault((r["document_id"], r["source_sha256"]), set()).update(_requested_observations(r))
    docs = []
    for d in load("manifest"):
        if d.get("alternate_copy_of"):
            continue
        count = d.get("page_count") or 0
        done = processed.get((d["id"], d["sha256"]), set()) & set(range(1, count + 1))
        remaining = sorted(set(range(1, count + 1)) - done)
        qa_pages = sorted(rejected.get((d["id"], d["sha256"]), set()) & set(remaining))
        region_qa_pages = sorted(rejected_regions.get((d["id"], d["sha256"]), set()) & set(range(1, count + 1)))
        docs.append({"document_id": d["id"], "source_pdf": d["source_pdf"], "source_sha256": d["sha256"],
                     "page_count": count, "processed_pages": sorted(done), "remaining_pages": remaining,
                     "quality_review_pages": qa_pages,
                     "region_quality_review_pages": region_qa_pages,
                     "processing_state": "EMPTY_SOURCE" if not count else "PROCESSED" if not remaining else "PENDING",
                     "scientific_completeness": "NOT_ESTABLISHED"})
    return {"authority": "processing ledger, not equation/concept completeness certification",
            "documents": docs, "summary": {"canonical_documents": len(docs),
                "source_pages": sum(d["page_count"] for d in docs),
                "processed_pages": sum(len(d["processed_pages"]) for d in docs),
                "remaining_pages": sum(len(d["remaining_pages"]) for d in docs),
                "empty_sources": sum(d["processing_state"] == "EMPTY_SOURCE" for d in docs)}}
