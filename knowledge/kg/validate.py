"""Quality control for the knowledge stores.

Every check is a function returning ``(issues, stats)``; ``run()`` writes
``knowledge/reports/validation_<date>.json`` and ``latest_validation.json``.
Checks are honest about what they can and cannot verify.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from . import AGENT_CORPUS_ROOT, CORPUS_ROOT, REPO_ROOT
from .store import REPORT_DIR, VECTOR_DIR, connect, load


def status() -> dict:
    from tools.sync_corpus import scan
    from . import store
    sync = scan()
    docs = load("manifest")
    conversion_manifest = AGENT_CORPUS_ROOT / "_conversion_manifest.json"
    conversion_rows = json.loads(conversion_manifest.read_text()) if conversion_manifest.exists() else []
    if store.CHUNK_DB.exists():
        conn = connect()
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_doc_states = conn.execute("SELECT COUNT(*) FROM doc_state").fetchone()[0]
        n_docs_chunked = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
        zero_chunk_docs = [r[0] for r in conn.execute(
            "SELECT doc_id FROM doc_state WHERE doc_id NOT IN (SELECT DISTINCT doc_id FROM chunks) ORDER BY doc_id")]
        conn.close()
    else:
        n_chunks = n_docs_chunked = n_doc_states = 0
        zero_chunk_docs = []
    idx_path = VECTOR_DIR / "index_meta.json"
    idx_meta = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    latest = REPORT_DIR / "latest_validation.json"
    from .coverage import summary as coverage_summary
    coverage = coverage_summary()
    return {
        "corpus": {"pdfs_on_disk": sync["pdf_count"], "conversion_rows": sync["manifest_rows"],
                   "tracked_file_rows": sum(r.get("source_kind", "file") == "file" for r in conversion_rows),
                   "tracked_archive_members": sum(r.get("source_kind") == "zip_member" for r in conversion_rows),
                   "tracked_unique_payloads": len({r["sha256"] for r in conversion_rows if r.get("sha256")}),
                   "available": CORPUS_ROOT.exists() and sync["manifest_rows"] > 0,
                   "pdfs_without_markdown": [r["pdf"] for r in sync["missing_markdown"]],
                   "stale_markdown": sync["stale_markdown"],
                   "orphan_markdown": sync["orphan_markdown"],
                   "synchronized": bool(sync["manifest_rows"]) and not (
                       sync["missing_markdown"] or sync["stale_markdown"])},
        "documents": {"unique": len(docs), "chunked": n_docs_chunked,
                      "doc_state_entries": n_doc_states, "zero_chunk_documents": zero_chunk_docs,
                      "zero_chunk_document_count": len(zero_chunk_docs),
                      "conversion_status": dict(Counter(d["conversion"]["status"] for d in docs)),
                      "alternate_copies": sum(1 for d in docs if d.get("alternate_copy_of"))},
        "chunks": n_chunks, "vector_index": idx_meta,
        "stores": {k: len(load(k)) for k in ("equations", "claims", "entities", "models", "validity", "software_links", "tests", "capabilities", "requirements", "repo_documents", "provenance", "artifacts")},
        "engineering_evidence": {"active_by_type": dict(Counter(
            r["type"] for r in load("artifacts") if r.get("type") != "ArtifactLink" and r.get("active", True))),
            "note": "Configurations and CAD are observed report snapshots; manufacturing records are plans, not execution evidence."},
        "extraction": {
            "equation_completeness": "NOT_ESTABLISHED",
            "documents_with_objects": coverage["documents"]["with_source_objects"],
            "model_support": coverage["models"]["current_support"],
            "declared_model_support": coverage["models"]["declared_support"],
            "models_with_current_source_review": coverage["models"]["source_reviewed_current_fingerprint"],
            "unverified_equations_and_claims": coverage["gaps"]["equations_and_claims_pending_independent_verification"],
            "unverified_equations_and_claims_including_history": sum(
                r.get("verification") not in ("CONFIRMED", "REFUTED")
                for name in ("equations", "claims") for r in load(name)),
            "scope_note": "Current evidence uses source/active state and manifest fingerprints; historical declarations are retained separately. No mathematical completeness claim.",
        },
        "last_validation": json.loads(latest.read_text())["summary"] if latest.exists() else None,
    }


def equation_variable_stats(eqs: list[dict]) -> dict:
    """Variable-mapping counts for current equations, separately from history.

    Superseded or stale transcriptions keep their variable lists for audit, but
    counting them together with current equations misstates the open mapping work.
    """
    def counts(rows):
        states = Counter(v.get("resolution_state") for e in rows for v in e.get("variables", []))
        return {"ambiguous": states.get("AMBIGUOUS_SYMBOL", 0), "unresolved": states.get("UNRESOLVED_ENTITY", 0),
                "resolved": states.get("RESOLVED", 0), "total": sum(states.values()), "equations": len(rows)}
    current = [e for e in eqs if e.get("active", True) and e.get("source_state", "CURRENT") == "CURRENT"]
    historical = [e for e in eqs if e not in current]
    return {**counts(current), "scope": "current active equations with CURRENT source state",
            "historical_or_inactive": counts(historical), "all_records": counts(eqs)}


def run() -> dict:
    issues: list[dict] = []
    stats: dict = {}
    docs = {d["id"]: d for d in load("manifest")}
    conn = connect()
    chunk_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}
    doc_pages = {r[0]: r[1] for r in conn.execute("SELECT doc_id, MAX(page_end) FROM chunks GROUP BY doc_id")}

    # 1. PDFs / markdown consistency
    md_root = AGENT_CORPUS_ROOT / "markdown"
    for d in docs.values():
        if not d["markdown_exists"] and not d.get("alternate_copy_of"):
            issues.append({"check": "pdf_without_markdown", "id": d["id"], "path": d["source_pdf"]})
        if d.get("page_count") == 0:
            issues.append({"check": "empty_source_pdf", "id": d["id"], "path": d["source_pdf"],
                           "state": "SOURCE_UNREADABLE", "reason": "EMPTY_SOURCE",
                           "remediation": "Restore the original source PDF; the tracked payload has zero pages."})
        if not d.get("alternate_copy_of") and d["id"] not in doc_pages:
            issues.append({"check": "canonical_document_without_chunks", "id": d["id"],
                           "state": "SOURCE_NOT_INGESTED", "severity": "warning",
                           "remediation": "Inspect source readability and extraction; a doc_state entry does not establish ingestion."})
    conversion_manifest = AGENT_CORPUS_ROOT / "_conversion_manifest.json"
    conversion_rows = json.loads(conversion_manifest.read_text()) if conversion_manifest.exists() else []
    if not conversion_manifest.exists():
        issues.append({"check": "conversion_manifest_missing", "severity": "warning"})
    conv_outputs = {str((AGENT_CORPUS_ROOT / r["output_markdown"]).resolve()) for r in conversion_rows}
    for md in md_root.rglob("*.md"):
        if str(md.resolve()) not in conv_outputs:
            note_source = CORPUS_ROOT / md.relative_to(md_root)
            if md.name == "PINTLE_LITERATURE_CATALOG.md" and note_source.is_file() and note_source.read_bytes() == md.read_bytes():
                continue  # classified bibliography mirror, not an invented PDF
            issues.append({"check": "markdown_without_manifest_row", "path": str(md.relative_to(REPO_ROOT))})

    # 2. object-level provenance / orphan checks
    from .ground import DocGrounder
    grounders = {}
    prov = {p["id"]: p for p in load("provenance")}
    models = {m["id"]: m for m in load("models")}
    for name in ("equations", "claims", "entities", "validity"):
        for rec in load(name):
            if rec.get("document_id") and rec["document_id"] not in docs:
                issues.append({"check": "object_points_to_unknown_document", "store": name, "id": rec["id"], "document_id": rec["document_id"]})
            if rec.get("chunk_id") and rec["chunk_id"] not in chunk_ids:
                issues.append({"check": "object_points_to_missing_chunk", "store": name, "id": rec["id"], "chunk_id": rec["chunk_id"]})
            if rec.get("page") and rec.get("document_id") in doc_pages and rec["page"] > doc_pages[rec["document_id"]]:
                issues.append({"check": "nonexistent_page_reference", "store": name, "id": rec["id"], "page": rec["page"]})
            if rec.get("provenance_id") and rec["provenance_id"] not in prov:
                issues.append({"check": "missing_provenance_record", "store": name, "id": rec["id"]})
            if name in ("equations", "claims") and rec.get("grounded") not in ("GROUNDED", "GROUNDED_PAGE_SHIFTED"):
                issues.append({"check": "ungrounded_object_in_store", "store": name, "id": rec["id"]})
            if rec.get("document_id") and rec.get("quote"):
                did = rec["document_id"]
                if did not in grounders:
                    grounders[did] = DocGrounder(did)
                anchor = grounders[did].locate(rec["quote"], rec.get("page"))
                if anchor["state"] == "UNGROUNDED":
                    issues.append({"check": "source_anchor_not_reproduced", "store": name, "id": rec["id"]})
                elif anchor["page"] != rec.get("page"):
                    issues.append({"check": "source_anchor_page_changed", "store": name, "id": rec["id"],
                                   "recorded_page": rec.get("page"), "located_page": anchor["page"]})
            for mid in rec.get("model_ids", []) or []:
                if mid not in models:
                    issues.append({"check": "object_links_unknown_model", "store": name, "id": rec["id"], "model_id": mid})
    # equations: unresolved / ambiguous variables are reported, not fixed
    eqs = load("equations")
    stats["equation_variables"] = equation_variable_stats(eqs)
    # duplicates by form within a document, among current records only; spacing
    # around operators does not make two transcriptions of one formula distinct
    from .ground import form_key
    seen = Counter((e["document_id"], form_key(e.get("form"))) for e in eqs
                   if e.get("active", True) and e.get("source_state", "CURRENT") == "CURRENT")
    for key, n in seen.items():
        if n > 1 and key[1]:
            issues.append({"check": "duplicate_equation_form_in_document", "document_id": key[0], "form": key[1][:80], "count": n})
    seen_c = Counter((c["document_id"], (c.get("statement") or "").strip().lower()) for c in load("claims"))
    for key, n in seen_c.items():
        if n > 1:
            issues.append({"check": "duplicate_claim_in_document", "document_id": key[0], "statement": key[1][:80], "count": n})

    # 3. models: implementation-state honesty
    for m in models.values():
        if m.get("hardware_qualified"):
            issues.append({"check": "model_claims_hardware_qualified", "id": m["id"]})
        if m.get("experimentally_validated") and "not" in m.get("validation_level", ""):
            issues.append({"check": "contradictory_validation_flags", "id": m["id"]})
        if m.get("source_support_state") == "NOT_FOUND_IN_SOURCE":
            issues.append({"check": "registry_citation_not_confirmed_in_local_source", "id": m["id"], "local_source": m.get("local_source"), "severity": "warning"})
        if not m.get("source_document_id") and m.get("basis_class") == "LITERATURE_DERIVED":
            issues.append({"check": "literature_model_without_local_source", "id": m["id"]})

    # 4. software links
    sw = [s for s in load("software_links") if s.get("type") == "SoftwareComponent"]
    for s in sw:
        if not (REPO_ROOT / s["file"]).exists():
            issues.append({"check": "software_component_file_missing", "id": s["id"], "file": s["file"]})
    consts = [s for s in load("software_links") if s.get("type") == "EmpiricalConstant"]
    stats["constants_by_basis"] = dict(Counter(k.get("basis_class") for k in consts))
    stats["components_without_scientific_link"] = sum(1 for s in sw if not s.get("model_ids") and not s.get("cited_document_ids") and not s.get("equation_ids"))

    # Engineering record existence/links never promote simulations to experiments.
    artifact_records = load("artifacts")
    for rec in artifact_records:
        if not rec.get("active", True):
            continue
        if not rec.get("provenance_id") or rec["provenance_id"] not in prov:
            issues.append({"check": "artifact_missing_provenance", "id": rec["id"]})
        if rec.get("hardware_qualified"):
            issues.append({"check": "artifact_claims_hardware_qualification", "id": rec["id"]})
        if rec.get("type") == "CADArtifact" and not rec.get("observed_sha256"):
            issues.append({"check": "declared_cad_bytes_unavailable", "id": rec["id"],
                           "declared_path": rec.get("declared_path"), "severity": "warning"})
    stats["engineering_artifacts"] = {
        "active_by_type": dict(Counter(r["type"] for r in artifact_records
                                       if r.get("active", True) and r.get("type") != "ArtifactLink")),
        "inactive_snapshots": sum(not r.get("active", True) for r in artifact_records),
    }

    # 5. graph sanity
    from .graph import Graph
    try:
        g = Graph()
        deg = Counter()
        for e in g.edges:
            deg[e["from"]] += 1
            deg[e["to"]] += 1
        for nid, d in deg.most_common(5):
            n = g.nodes.get(nid, {})
            if d > 400 and n.get("type") not in ("Document", "Subsystem"):
                issues.append({"check": "implausibly_high_degree", "id": nid, "degree": d, "type": n.get("type")})
        if g.meta.get("n_dangling_dropped"):
            issues.append({"check": "graph_dangling_edges", "count": g.meta["n_dangling_dropped"], "severity": "warning"})
        no_prov = sum(1 for e in g.edges if not e.get("provenance"))
        stats["graph"] = {"nodes": len(g.nodes), "edges": len(g.edges), "edges_without_provenance": no_prov,
                          "top_degree": [(nid, d) for nid, d in deg.most_common(5)]}
        if no_prov:
            issues.append({"check": "edges_without_provenance", "count": no_prov, "severity": "warning"})
    except FileNotFoundError:
        issues.append({"check": "graph_not_built"})

    # 6. vector index / chunk duplication
    dup_chunks = conn.execute("SELECT COUNT(*) FROM (SELECT text FROM chunks GROUP BY text HAVING COUNT(*) > 1)").fetchone()[0]
    stats["duplicate_chunk_texts"] = dup_chunks
    conn.close()

    from .coverage import summary as coverage_summary
    stats["coverage"] = coverage_summary()
    gaps = stats["coverage"]["gaps"]
    for name, count in (("source_extraction_incomplete", gaps["canonical_documents_without_source_objects"]),
                        ("source_verification_incomplete", gaps["equations_and_claims_pending_independent_verification"]),
                        ("unknown_basis_software_constants", gaps["unknown_basis_software_constants"])):
        if count:
            issues.append({"check": name, "count": count, "severity": "warning",
                           "equation_completeness": "NOT_ESTABLISHED"})
    incomplete = stats["coverage"]["incomplete_formula_transcriptions"]
    if incomplete["current_count"]:
        issues.append({"check": "incomplete_formula_transcriptions", "severity": "warning",
                       "count": incomplete["current_count"], "ids": incomplete["current_ids"],
                       "state": "INCOMPLETE_TRANSCRIPTION",
                       "remediation": "Inspect original PDF pages and bank corrected transcriptions with independent verification; retain unresolved records without treating their formulas as usable."})

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "n_issues": len(issues),
               "issues_by_check": dict(Counter(i["check"] for i in issues)), "stats": stats}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = REPORT_DIR / f"validation_{stamp}.json"
    payload = {"summary": summary, "issues": issues}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (REPORT_DIR / "latest_validation.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return {"summary": summary, "issues": issues, "report_path": str(path.relative_to(REPO_ROOT))}
