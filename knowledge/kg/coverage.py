"""Lightweight evidence coverage, without pretending to measure completeness.

Counts describe stored, current source assertions. Neither a populated index nor
an independently confirmed equation establishes that all source math was found.
No PDF parsing, source writes, or index rebuilds occur here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from . import store, verification_freshness
from .graph import equation_formulation_state, model_source_support


def summary(records: Mapping[str, list[dict]] | None = None, *,
            include_documents: bool = False) -> dict:
    """Return inspectable source/model coverage; optionally include every document.

    ``records`` permits a caller to reuse an existing snapshot or supply synthetic
    fixtures. Missing stores in that mapping are empty, never loaded from disk.
    Legacy confirmed counters retain stored verdicts whose source fingerprint
    matches the manifest; additive content-bound counters also require the
    inspected assertion fingerprint. Registry validity prose is not extraction.
    """
    names = ("manifest", "equations", "claims", "entities", "validity", "models", "software_links")
    data = {name: store.load(name) if records is None else records.get(name, []) for name in names}
    docs = {d["id"]: d for d in data["manifest"]}
    canonical = {did for did, d in docs.items() if not d.get("alternate_copy_of")}
    counts: dict[str, Counter] = {did: Counter() for did in docs}
    source_stats = {}
    pending = 0
    current_incomplete = []
    for name in ("equations", "claims", "entities", "validity"):
        rows = [r for r in data[name]
                if r.get("document_id") and r.get("provenance_kind") == "SOURCE_STATES"]
        current = []
        verified = []
        content_verified = []
        content_states = Counter()
        for r in rows:
            doc = docs.get(r["document_id"])
            if (not doc or doc.get("page_count") == 0 or not r.get("active", True)
                    or r.get("source_state", "CURRENT") != "CURRENT"
                    or r.get("verification") == "REFUTED"
                    or (r.get("source_sha256") and r["source_sha256"] != doc.get("sha256"))):
                continue
            current.append(r)
            content_state = verification_freshness.state(r, doc)["state"]
            content_states[content_state] += 1
            bucket = counts[doc["id"]]
            bucket["source_objects"] += 1
            bucket[name] += 1
            if name == "equations" and equation_formulation_state(r) == "INCOMPLETE_TRANSCRIPTION":
                current_incomplete.append(r["id"])
                bucket["incomplete_formula_transcriptions"] += 1
            if (r.get("verification") == "CONFIRMED" and doc.get("sha256")
                    and r.get("verifier_source_sha256") == doc["sha256"]):
                verified.append(r)
                bucket["confirmed_source_objects"] += 1
                bucket["confirmed_" + name] += 1
            elif name in ("equations", "claims"):
                pending += 1
            if content_state == "CONFIRMED":
                content_verified.append(r)
                bucket["content_bound_confirmed_source_objects"] += 1
                bucket["content_bound_confirmed_" + name] += 1
        source_stats[name] = {"records": len(rows), "current": len(current),
                              "independently_confirmed": len(verified),
                              "independently_confirmed_basis": "Historical stored CONFIRMED verdict with matching source fingerprint; not necessarily bound to current assertion content.",
                              "stored_verdict_totals_all_records": dict(Counter(r.get("verification", "NOT_CHECKED") for r in rows)),
                              "stored_verdict_totals_current_source_objects": dict(Counter(r.get("verification", "NOT_CHECKED") for r in current)),
                              "content_bound_confirmed": len(content_verified),
                              "current_content_verification_by_state": dict(content_states),
                              "all_content_verification_by_state": dict(Counter(
                                  verification_freshness.state(r, docs.get(r.get("document_id")))["state"] for r in rows)),
                              "current_without_content_bound_confirmation": len(current) - len(content_verified),
                              "inactive_stale_refuted_or_unresolved": len(rows) - len(current)}

    declared_support = Counter()
    effective_support = Counter()
    reviewed_models = 0
    for model in data["models"]:
        did = model.get("source_document_id")
        doc = docs.get(did)
        declared_support[model.get("source_support_state") or "NOT_CHECKED"] += 1
        state, _ = model_source_support(model, doc, documents=docs)
        effective_support[state] += 1
        if did in counts:
            counts[did]["cited_models"] += 1
            if state in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
                counts[did]["supported_models"] += 1
        if (doc and doc.get("sha256") and model.get("source_support_sha256") == doc["sha256"]
                and model.get("source_state", "CURRENT") == "CURRENT"
                and state in {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_FOUND_IN_SOURCE", "SOURCE_UNREADABLE"}):
            reviewed_models += 1

    constants = [r for r in data["software_links"] if r.get("type") == "EmpiricalConstant"]
    constant_basis = Counter(r.get("basis_class") or "UNKNOWN_BASIS" for r in constants)
    current_constants = [r for r in constants
                         if r.get("active", True) and r.get("source_state", "CURRENT") == "CURRENT"]
    current_constant_basis = Counter(r.get("basis_class") or "UNKNOWN_BASIS" for r in current_constants)
    historical_constant_basis = constant_basis - current_constant_basis
    incomplete_ids = sorted(r["id"] for r in data["equations"]
                            if equation_formulation_state(r) == "INCOMPLETE_TRANSCRIPTION")
    result = {
        "equation_completeness": "NOT_ESTABLISHED",
        "concept_completeness": "NOT_ESTABLISHED",
        "scope_note": "Coverage of stored source objects against manifest identities; live PDF/Markdown bytes are not rehashed here. Existing independently_confirmed counters preserve historical stored-verdict totals; content_bound_confirmed additionally checks the assertion fingerprint. No all-equation denominator or physical validation.",
        "documents": {
            "records": len(docs), "canonical": len(canonical), "alternate_copies": len(docs) - len(canonical),
            "with_source_objects": sum(counts[did]["source_objects"] > 0 for did in canonical),
            "without_source_objects": sum(counts[did]["source_objects"] == 0 for did in canonical),
            "with_equations": sum(counts[did]["equations"] > 0 for did in canonical),
            "without_equations": sum(counts[did]["equations"] == 0 for did in canonical),
            "with_independently_confirmed_objects": sum(counts[did]["confirmed_source_objects"] > 0 for did in canonical),
            "with_content_bound_confirmed_objects": sum(counts[did]["content_bound_confirmed_source_objects"] > 0 for did in canonical),
            "conversion_requires_review": sum(docs[did].get("conversion", {}).get("status") != "good" for did in canonical),
            "conversion_low_confidence": sum(bool(docs[did].get("conversion", {}).get("low_confidence")) for did in canonical),
            "zero_page_sources": sum(docs[did].get("page_count") == 0 for did in canonical),
            "zero_page_source_ids": sorted(did for did in canonical if docs[did].get("page_count") == 0),
        },
        "source_objects": source_stats,
        "incomplete_formula_transcriptions": {"count": len(incomplete_ids), "ids": incomplete_ids,
                                              "current_count": len(current_incomplete),
                                              "current_ids": sorted(current_incomplete),
                                              "current_state": "INCOMPLETE_TRANSCRIPTION" if current_incomplete else "NONE_FLAGGED_CURRENT",
                                              "historical_count": len(incomplete_ids) - len(current_incomplete),
                                              "state": "INCOMPLETE_TRANSCRIPTION" if incomplete_ids else "NONE_FLAGGED",
                                              "note": "Explicit NOT RENDERED/OCR-garbled markers in current forms; absence of a marker does not verify a transcription."},
        "models": {"records": len(data["models"]), "declared_support": dict(declared_support),
                   "current_support": dict(effective_support), "source_reviewed_current_fingerprint": reviewed_models,
                   "without_current_source_review": len(data["models"]) - reviewed_models,
                   "without_manifest_source": sum(m.get("source_document_id") not in docs for m in data["models"])},
        "software_constants": {"records": len(constants), "by_basis": dict(constant_basis),
                               "unknown_basis": constant_basis["UNKNOWN_BASIS"],
                               "current": len(current_constants),
                               "inactive_or_noncurrent": len(constants) - len(current_constants),
                               "by_basis_current": dict(current_constant_basis),
                               "by_basis_inactive_or_noncurrent": dict(historical_constant_basis),
                               "unknown_basis_current": current_constant_basis["UNKNOWN_BASIS"],
                               "unknown_basis_inactive_or_noncurrent": historical_constant_basis["UNKNOWN_BASIS"],
                               "scope_note": "Stored totals preserve history. Current counts include active records with CURRENT source state; absent legacy state fields retain their existing active/CURRENT defaults."},
        "gaps": {"canonical_documents_without_source_objects": sum(counts[did]["source_objects"] == 0 for did in canonical),
                 "canonical_documents_without_equations": sum(counts[did]["equations"] == 0 for did in canonical),
                 "equations_and_claims_pending_independent_verification": pending,
                 "current_source_objects_without_content_bound_confirmation": sum(
                     row["current_without_content_bound_confirmation"] for row in source_stats.values()),
                 "equations_without_model_links": sum(not r.get("model_ids") for r in data["equations"]),
                 "incomplete_formula_transcriptions": len(current_incomplete),
                 "zero_page_sources": sum(docs[did].get("page_count") == 0 for did in canonical),
                 "unknown_basis_software_constants": current_constant_basis["UNKNOWN_BASIS"]},
    }
    if include_documents:
        result["by_document"] = [{"document_id": did, "canonical": did in canonical,
                                  "equation_completeness": "NOT_ESTABLISHED", **dict(counts[did]),
                                  "extraction_gap": not bool(counts[did]["source_objects"])}
                                 for did in sorted(docs)]
    return result
