"""Deterministic completeness, equation-fidelity and model-audit data products.

Each product is computed only from stores, journals and the packet ledger. The
denominators are kept separate on purpose: indexed text, parser candidates,
stored source assertions, independent verification and mathematical
completeness are different evidence states. Nothing here declares a document
mathematically complete; that state stays ``NOT_ESTABLISHED``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import KNOWLEDGE_ROOT, REPO_ROOT, store, verification_freshness
from .graph import model_source_support

SCIENCE_STORES = ("equations", "claims", "entities", "validity")


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> dict:
    files = [store.store_path(n) for n in ("manifest", "equations", "claims", "entities", "validity", "models",
                                           "enrichments", "relationships", "software_links")]
    files += [store.REPORT_DIR / "packet_queue.json", store.REPORT_DIR / "docling_jobs.jsonl"]
    out = {}
    for f in files:
        try:
            key = str(f.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            key = str(f)
        out[key] = _sha(f)
    return out


def _finish(payload: dict) -> dict:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "content_sha256": hashlib.sha256(body.encode()).hexdigest(), **payload}


def _current(rec: dict) -> bool:
    return rec.get("active", True) and rec.get("source_state", "CURRENT") == "CURRENT"


def _verification(rec: dict) -> str:
    """Historical stored verdict; use content-verification separately for freshness."""
    return rec.get("verification") or "NOT_INDEPENDENTLY_VERIFIED"


def _content_verification(rec: dict, docs: dict) -> dict:
    return verification_freshness.state(rec, docs.get(rec.get("document_id"))) | {
        "current_content_sha256": verification_freshness.content_sha256(rec),
        "recorded_content_sha256": rec.get("verifier_content_sha256"),
        "binding_basis": rec.get("verifier_content_binding"),
    }


def _packets() -> dict:
    path = store.REPORT_DIR / "packet_queue.json"
    return json.loads(path.read_text())["packets"] if path.exists() else {}


def _docling_jobs() -> list[dict]:
    path = store.REPORT_DIR / "docling_jobs.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _indexed_pages() -> tuple[dict[str, set[int]], dict[str, int]]:
    pages: dict[str, set[int]] = defaultdict(set)
    chunks: Counter = Counter()
    if not store.CHUNK_DB.exists():
        return pages, chunks
    conn = store.connect()
    try:
        for row in conn.execute("SELECT doc_id, page_start, page_end, length(text) AS n FROM chunks"):
            chunks[row["doc_id"]] += 1
            if row["n"] and row["page_start"]:
                pages[row["doc_id"]].update(range(row["page_start"], (row["page_end"] or row["page_start"]) + 1))
    finally:
        conn.close()
    return pages, chunks


def corpus_completeness() -> dict:
    from .enrichment import queue as docling_ledger
    docs = store.load("manifest")
    docs_by_id = {d["id"]: d for d in docs}
    canonical = [d for d in docs if not d.get("alternate_copy_of")]
    ledger = {d["document_id"]: d for d in docling_ledger()["documents"]}
    jobs = _docling_jobs()
    indexed, chunk_counts = _indexed_pages()
    candidates = defaultdict(Counter)
    candidate_states = defaultdict(Counter)
    region_pages = defaultdict(set)
    rejected_pages = defaultdict(set)
    for r in store.load("enrichments"):
        if r.get("type") == "GeneratedDocument":
            first, last = (r.get("page_range") or [0, -1])[:2]
            observed = {p for p in r.get("observed_pages", []) if first <= p <= last}
            if r.get("options", {}).get("mode") == "visually_located_formula_regions":
                region_pages[r["document_id"]].update(observed)
            elif r.get("review_state") == "REJECTED":
                rejected_pages[r["document_id"]].update(observed)
        elif r.get("candidate_kind"):
            candidates[r["document_id"]][r["candidate_kind"]] += 1
            candidate_states[r["document_id"]][r.get("review_state", "UNREVIEWED")] += 1
    objects = defaultdict(lambda: defaultdict(Counter))
    verification = defaultdict(Counter)
    content_verification = defaultdict(Counter)
    content_verification_eq_claims = defaultdict(Counter)
    for name in SCIENCE_STORES:
        for r in store.load(name):
            if not r.get("document_id") or r.get("provenance_kind") != "SOURCE_STATES":
                continue
            objects[r["document_id"]][name]["current" if _current(r) else "historical"] += 1
            if _current(r):
                content_state = verification_freshness.state(r, docs_by_id.get(r["document_id"]))["state"]
                content_verification[r["document_id"]][content_state] += 1
                if name in ("equations", "claims"):
                    content_verification_eq_claims[r["document_id"]][content_state] += 1
            if name in ("equations", "claims") and _current(r):
                verification[r["document_id"]][_verification(r)] += 1
    packets_by_doc = defaultdict(list)
    for p in _packets().values():
        packets_by_doc[p["document_id"]].append({"id": p["id"], "kind": p["kind"], "pages": p["pages"], "state": p["state"]})
    rows = []
    totals = Counter()
    for d in sorted(canonical, key=lambda d: d["id"]):
        did, count = d["id"], d.get("page_count") or 0
        led = ledger.get(did, {})
        attempted_unfinished = set()
        for job in jobs:
            if job.get("document_id") == did and job.get("source_sha256") == d.get("sha256") \
                    and job.get("state") in ("WORKER_TIMEOUT",) and len(job.get("pages", [])) == 2:
                attempted_unfinished.update(range(job["pages"][0], job["pages"][1] + 1))
        processed = set(led.get("processed_pages", []))
        attempted_unfinished -= processed
        cur = {name: dict(objects[did][name]) for name in SCIENCE_STORES if objects[did][name]}
        ver = dict(verification[did])
        n_indexed = len({p for p in indexed.get(did, set()) if 1 <= p <= count})
        row = {"document_id": did, "title": d.get("title"), "source_pdf": d.get("source_pdf"), "sha256": d.get("sha256"),
               "page_count": count, "conversion_status": d.get("conversion", {}).get("status"),
               "low_confidence_conversion": bool(d.get("conversion", {}).get("low_confidence")),
               "scanned_source": bool(d.get("conversion", {}).get("scanned")),
               "alternate_copies": d.get("alternate_copies", []),
               "raw_extraction": {"pages_with_indexed_text": n_indexed, "chunks": chunk_counts.get(did, 0),
                                  "docling_processed_pages": sorted(processed),
                                  "docling_rejected_pages": sorted(rejected_pages[did] - processed),
                                  "docling_region_crop_pages": sorted(region_pages[did]),
                                  "docling_attempted_unfinished_pages": sorted(attempted_unfinished),
                                  "docling_unattempted_page_count": max(0, count - len(processed | rejected_pages[did] | attempted_unfinished))},
               "structured_candidacy": {"by_kind": dict(candidates[did]), "by_review_state": dict(candidate_states[did])},
               "source_objects": cur,
               "independent_verification": ver,
               "independent_verification_basis": "Historical stored verdicts on active/current equations and claims; these totals do not assert current content binding.",
               "content_verification": {
                   "source_assertions_by_state": dict(content_verification[did]),
                   "equations_claims_by_state": dict(content_verification_eq_claims[did]),
                   "content_bound_confirmed_source_assertions": content_verification[did]["CONFIRMED"],
                   "scope": "Active/current source assertions; source and assertion content fingerprints checked separately. Legacy confirmations remain LEGACY_CONTENT_NOT_BOUND."},
               "session_packets": sorted(packets_by_doc.get(did, []), key=lambda p: p["id"]),
               "mathematical_completeness": "EMPTY_SOURCE" if not count else "NOT_ESTABLISHED"}
        rows.append(row)
        totals["source_pages"] += count
        totals["pages_with_indexed_text"] += n_indexed
        totals["docling_processed_pages"] += len(processed)
        totals["docling_rejected_pages"] += len(row["raw_extraction"]["docling_rejected_pages"])
        totals["docling_region_crop_pages"] += len(region_pages[did])
        totals["docling_attempted_unfinished_pages"] += len(attempted_unfinished)
        totals["docling_unattempted_pages"] += row["raw_extraction"]["docling_unattempted_page_count"]
        totals["documents_with_chunks"] += bool(chunk_counts.get(did))
        totals["documents_with_docling_candidates"] += bool(candidates[did])
        totals["docling_candidates"] += sum(candidates[did].values())
        totals["documents_with_current_source_objects"] += any(v.get("current") for v in cur.values())
        totals["documents_with_confirmed_objects"] += bool(ver.get("CONFIRMED"))
        totals["documents_with_content_bound_confirmed_objects"] += bool(content_verification[did]["CONFIRMED"])
        totals["source_assertions_content_bound_confirmed"] += content_verification[did]["CONFIRMED"]
        totals["equations_claims_content_bound_confirmed"] += content_verification_eq_claims[did]["CONFIRMED"]
        for name, c in cur.items():
            totals[f"{name}_current"] += c.get("current", 0)
            totals[f"{name}_historical"] += c.get("historical", 0)
        for state, n in ver.items():
            totals[f"equations_claims_{state.lower()}"] += n
        for state, n in content_verification[did].items():
            totals[f"source_assertions_content_state_{state.lower()}"] += n
    return _finish({
        "authority": "Per-document evidence denominators. Indexed text, parser candidates, stored source assertions, "
                     "independent verification and mathematical completeness are separate states; none implies the next. "
                     "Existing independent_verification and confirmed counters preserve historical stored-verdict totals; "
                     "content_verification and content_bound_confirmed counters additionally check current assertion content.",
        "inputs": _inputs(),
        "completeness_claims": {"equation_completeness": "NOT_ESTABLISHED", "concept_completeness": "NOT_ESTABLISHED",
                                "documents_declared_mathematically_complete": 0},
        "denominators": {"canonical_documents": len(canonical), "alternate_copies": len(docs) - len(canonical),
                         **dict(sorted(totals.items()))},
        "documents": rows})


def _relationships_by_equation() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in store.load("relationships"):
        out[r["from"]].append({"edge": r["edge"], "to": r["to"], "source_relation": r.get("source_relation"),
                               "provenance_kind": r.get("provenance_kind")})
        out[r["to"]].append({"edge": r["edge"], "from": r["from"], "source_relation": r.get("source_relation"),
                             "provenance_kind": r.get("provenance_kind")})
    return out


def equation_fidelity_audit() -> dict:
    eqs = store.load("equations")
    claims = store.load("claims")
    models = store.load("models")
    docs = store.index_by("manifest")
    from .harvest_extraction import model_review_state, IMPLEMENTS_FIDELITY
    docs = store.index_by("manifest")
    reviews_by_pair = {(m["id"], r["reviewed_document_id"]): r for m in models
                       for r in model_review_state(m, documents=docs).get("source_reviews", [])}

    def active_model_link(mid, equation):
        review = reviews_by_pair.get((mid, equation.get("document_id")))
        return not review or (review["state"] == "CURRENT" and review.get("fidelity") in IMPLEMENTS_FIDELITY
                              and ("resolved_implementation_equation_ids" not in review
                                   or equation["id"] in review["resolved_implementation_equation_ids"]))

    rels = _relationships_by_equation()
    implementations = defaultdict(list)
    components_by_model = defaultdict(list)
    for s in store.load("software_links"):
        if s.get("type") != "SoftwareComponent" or not s.get("active", True):
            continue
        row = {"component": s["id"], "file": s.get("file"), "symbol": s.get("ast_qualname") or s.get("qualname"),
               "locator_state": s.get("implementation_locator_state"),
               "comparison_state": s.get("implementation_scientific_comparison_state")}
        for eid in s.get("equation_ids", []) or []:
            implementations[eid].append({**row, "path": "component IMPLEMENTS equation"})
        for mid in s.get("model_ids", []) or []:
            components_by_model[mid].append(row)
    for e in eqs:
        if not _current(e):
            continue  # superseded/stale records keep history but no implementation association
        for mid in e.get("model_ids", []) or []:
            if not active_model_link(mid, e):
                continue
            for row in components_by_model.get(mid, []):
                implementations[e["id"]].append({**row, "path": f"component IMPLEMENTS {mid}; {mid} IMPLEMENTS equation"})
    compared = defaultdict(list)
    supported = defaultdict(list)
    for m in models:
        for eid in m.get("source_comparison_equation_ids", []) or []:
            compared[eid].append({"model": m["id"], "relation": "COMPARED_NOT_IMPLEMENTED"})
        for (mid, did), review in reviews_by_pair.items():
            if mid != m["id"]:
                continue
            for eid in review.get("equation_ids", []) or review.get("resolved_comparison_equation_ids", []):
                compared[eid].append({"model": mid, "relation": "STRUCTURED_REVIEW", "fidelity": review.get("fidelity"),
                                      "review_state": review["state"], "document_id": did})
    for e in eqs:
        for mid in e.get("model_ids", []) or []:
            if active_model_link(mid, e):
                supported[e["id"]].append(mid)
    rows = []
    for e in sorted(eqs, key=lambda e: e["id"]):
        variables = Counter(v.get("resolution_state") for v in e.get("variables", []))
        rows.append({
            "id": e["id"], "document_id": e.get("document_id"), "page": e.get("page"),
            "equation_number": e.get("equation_number"), "label": e.get("label"), "form": e.get("form"),
            "record_state": "CURRENT" if _current(e) else e.get("source_state", "INACTIVE"),
            "verification": _verification(e),
            "content_verification": _content_verification(e, docs),
            "verifier": ({"role": e.get("verifier_role"), "model": e.get("verifier_model")} if e.get("verifier_role")
                         else ("LEGACY_UNRECORDED" if e.get("verification") else None)),
            "transcribed_by": e.get("transcribed_by") or "LEGACY_UNRECORDED",
            "confidence": e.get("confidence"),
            "superseded_by": e.get("superseded_by_ids") or e.get("superseded_by"), "corrects_ids": e.get("corrects_ids", []),
            "parser_candidates": {"inspected": e.get("enrichment_candidate_ids", []), "corrected": e.get("corrects_candidate_ids", [])},
            "variables": {"total": sum(variables.values()), **{k.lower(): v for k, v in variables.items() if k}},
            "relationships": rels.get(e["id"], []),
            "model_support_links": supported.get(e["id"], []),
            "implementation_associations": implementations.get(e["id"], []),
            "model_comparisons": compared.get(e["id"], []),
        })
    current = [r for r in rows if r["record_state"] == "CURRENT"]
    claim_rows = [{"id": c["id"], "document_id": c.get("document_id"), "page": c.get("page"),
                   "record_state": "CURRENT" if _current(c) else c.get("source_state", "INACTIVE"),
                   "verification": _verification(c),
                   "content_verification": _content_verification(c, docs),
                   "verifier": ({"role": c.get("verifier_role"), "model": c.get("verifier_model")} if c.get("verifier_role")
                                else ("LEGACY_UNRECORDED" if c.get("verification") else None))}
                  for c in sorted(claims, key=lambda c: c["id"])]
    return _finish({
        "authority": "Stored source-mathematics records with review states. NOT_INDEPENDENTLY_VERIFIED is explicit; "
                     "verification/current_by_verification preserve historical stored-verdict totals; content_verification "
                     "checks current assertion fingerprints. A confirmed transcription is not an implementation or physical-validity verdict.",
        "inputs": _inputs(),
        "summary": {"equations": len(rows), "current": len(current), "historical_or_inactive": len(rows) - len(current),
                    "current_by_verification": dict(Counter(r["verification"] for r in current)),
                    "current_content_verification_by_state": dict(Counter(r["content_verification"]["state"] for r in current)),
                    "current_content_bound_confirmed": sum(r["content_verification"]["state"] == "CONFIRMED" for r in current),
                    "current_with_recorded_verifier": sum(isinstance(r["verifier"], dict) for r in current),
                    "current_with_implementation_associations": sum(bool(r["implementation_associations"]) for r in current),
                    "current_with_model_comparisons": sum(bool(r["model_comparisons"]) for r in current),
                    "current_with_relationships": sum(bool(r["relationships"]) for r in current),
                    "current_variables": dict(sum((Counter({k: v for k, v in r["variables"].items() if k != "total"})
                                                   for r in current), Counter())),
                    "claims_current_by_verification": dict(Counter(c["verification"] for c in claim_rows
                                                                   if c["record_state"] == "CURRENT")),
                    "claims_current_content_verification_by_state": dict(Counter(
                        c["content_verification"]["state"] for c in claim_rows if c["record_state"] == "CURRENT")),
                    "claims_current_content_bound_confirmed": sum(c["content_verification"]["state"] == "CONFIRMED"
                                                                   for c in claim_rows if c["record_state"] == "CURRENT")},
        "equations": rows, "claims": claim_rows})


def _journals_by_target() -> dict[str, list[str]]:
    out = defaultdict(set)
    results = KNOWLEDGE_ROOT / "workflows"
    for path in sorted(list(results.glob("results/**/*.jsonl")) + [results / "recovered_results.jsonl"]):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            value = row.get("value") or row.get("result") or {}
            items = value.get("results", [value]) if isinstance(value, dict) else []
            for item in items:
                target = (item or {}).get("target", "")
                if str(target).startswith("MODEL_"):
                    out[target].add(str(path.relative_to(REPO_ROOT)))
    return {k: sorted(v) for k, v in out.items()}


def model_source_audit() -> dict:
    from .harvest_extraction import model_review_state
    from .work_queue import build as queue_build
    docs = store.index_by("manifest")
    queue = queue_build()
    routes = {r["target"]: r["state"] for r in queue["model_reviews"] + queue["source_routing"]}
    journals = _journals_by_target()
    equations = store.load("equations")
    confirmed = {e["id"] for e in equations if _current(e) and e.get("verification") == "CONFIRMED"}
    content_bound_confirmed = {e["id"] for e in equations
                               if verification_freshness.state(e, docs.get(e.get("document_id")))["state"] == "CONFIRMED"}
    rows = []
    for m in sorted(store.load("models"), key=lambda m: m["id"]):
        doc = docs.get(m.get("source_document_id"))
        current, note = model_source_support(m, doc, documents=docs)
        review = model_review_state(m, documents=docs)
        source_reviews = review.get("source_reviews", [])
        by_doc = {r["reviewed_document_id"]: r for r in source_reviews}
        from .harvest_extraction import IMPLEMENTS_FIDELITY
        links = []
        for e in equations:
            per_source = by_doc.get(e.get("document_id"))
            source = docs.get(e.get("document_id"))
            if (m["id"] not in (e.get("model_ids") or []) or not _current(e)
                    or e.get("verification") == "REFUTED" or not source or not source.get("sha256")
                    or e.get("source_sha256") != source["sha256"]
                    or (per_source and (per_source["state"] != "CURRENT"
                                        or per_source.get("fidelity") not in IMPLEMENTS_FIDELITY
                                        or ("resolved_implementation_equation_ids" in per_source
                                            and e["id"] not in per_source["resolved_implementation_equation_ids"])))):
                continue
            links.append(e["id"])
        links.sort()
        source_review = m.get("source_review") or {}
        rows.append({
            "id": m["id"], "registry_id": m.get("registry_id"), "subsystem": m.get("subsystem"),
            "promotion_status": m.get("promotion_status"), "promotion_status_basis": m.get("promotion_status_basis"),
            "execution_reach": {k: (m.get("execution_reach") or {}).get(k) for k in
                                ("state", "optimizer_reach_state", "runtime_trace_state", "optimizer_leverage_state")},
            "source_document_id": m.get("source_document_id"), "local_source": m.get("local_source"),
            "source_route_state": routes.get(m["id"], "REVIEWED_CURRENT"),
            "declared_support": m.get("source_support_state", "NOT_CHECKED"),
            "current_support": current, "current_support_note": note,
            "source_support_fingerprint_current": bool(doc and m.get("source_support_sha256") == doc.get("sha256")),
            "structured_review": ({"state": review["state"], "fidelity": review.get("fidelity"),
                                   "fidelities": review.get("fidelities", []), "source_reviews": source_reviews,
                                   "source_pages": source_review.get("source_pages"),
                                   "reviewer": source_review.get("reviewer"),
                                   "implementation_fingerprints": source_review.get("implementation_fingerprints"),
                                   "physics_change_required": review.get("physics_change_required", False),
                                   "remaining_uncertainties": source_review.get("remaining_uncertainties", [])}
                                  if source_review else {"state": "NOT_CHECKED"}),
            "supporting_equation_links": links,
            "supporting_equation_links_confirmed": sorted(set(links) & confirmed),
            "supporting_equation_links_confirmed_basis": "Historical stored CONFIRMED verdicts on current source equations; may lack a current assertion-content binding.",
            "supporting_equation_links_content_bound_confirmed": sorted(set(links) & content_bound_confirmed),
            "comparison_equation_ids": m.get("source_comparison_equation_ids", []),
            "software_verification_state": m.get("software_verification_state", "UNKNOWN"),
            "physical_validation_state": m.get("physical_validation_state", "UNKNOWN"),
            "hardware_qualified": False,
            "evidence_journals": journals.get(m["id"], []),
            "audit_status": ("STRUCTURED_REVIEW_CURRENT" if review["state"] == "CURRENT"
                             else review["state"] if source_review
                             else "SOURCE_REVIEW_CURRENT_UNSTRUCTURED" if current in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_FOUND_IN_SOURCE")
                             else "NOT_CHECKED"),
        })
    return _finish({
        "authority": "All registered models. Citation, source support, structured implementation comparison, software "
                     "verification and physical validation are separate fields; NOT_CHECKED is explicit.",
        "inputs": _inputs(),
        "summary": {"models": len(rows), "by_audit_status": dict(Counter(r["audit_status"] for r in rows)),
                    "by_current_support": dict(Counter(r["current_support"] for r in rows)),
                    "by_fidelity": dict(Counter(r["structured_review"].get("fidelity") for r in rows
                                                if r["structured_review"].get("fidelity"))),
                    "by_route": dict(Counter(r["source_route_state"] for r in rows)),
                    "physical_validation_established": 0},
        "models": rows})


def write_all(report_dir: Path | None = None) -> dict:
    out_dir = report_dir or store.REPORT_DIR
    products = {"CORPUS_COMPLETENESS.json": corpus_completeness(),
                "EQUATION_FIDELITY_AUDIT.json": equation_fidelity_audit(),
                "MODEL_SOURCE_AUDIT.json": model_source_audit()}
    for name, payload in products.items():
        store._atomic_write_json(out_dir / name, payload)
    return {name: {"content_sha256": p["content_sha256"],
                   **({"denominators": p["denominators"]} if "denominators" in p else {"summary": p["summary"]})}
            for name, p in products.items()}
