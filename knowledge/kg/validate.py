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
    if store.CHUNK_DB.exists():
        conn = connect()
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_docs_chunked = conn.execute("SELECT COUNT(*) FROM doc_state").fetchone()[0]
        conn.close()
    else:
        n_chunks = n_docs_chunked = 0
    idx_path = VECTOR_DIR / "index_meta.json"
    idx_meta = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    latest = REPORT_DIR / "latest_validation.json"
    models = load("models")
    extracted = [r for name in ("equations", "claims", "entities", "validity")
                 for r in load(name) if r.get("document_id")]
    return {
        "corpus": {"pdfs_on_disk": sync["pdf_count"], "conversion_rows": sync["manifest_rows"],
                   "available": CORPUS_ROOT.exists() and sync["manifest_rows"] > 0,
                   "pdfs_without_markdown": [r["pdf"] for r in sync["missing_markdown"]],
                   "stale_markdown": sync["stale_markdown"],
                   "orphan_markdown": sync["orphan_markdown"],
                   "synchronized": bool(sync["manifest_rows"]) and not (
                       sync["missing_markdown"] or sync["stale_markdown"])},
        "documents": {"unique": len(docs), "chunked": n_docs_chunked,
                      "conversion_status": dict(Counter(d["conversion"]["status"] for d in docs)),
                      "alternate_copies": sum(1 for d in docs if d.get("alternate_copy_of"))},
        "chunks": n_chunks, "vector_index": idx_meta,
        "stores": {k: len(load(k)) for k in ("equations", "claims", "entities", "models", "validity", "software_links", "tests", "capabilities", "requirements", "repo_documents", "provenance")},
        "extraction": {
            "documents_with_objects": len({r["document_id"] for r in extracted}),
            "model_support": dict(Counter(m.get("source_support_state", "NOT_CHECKED") for m in models)),
            "unverified_equations_and_claims": sum(r.get("verification") not in ("CONFIRMED", "REFUTED")
                                                    for name in ("equations", "claims") for r in load(name)),
        },
        "last_validation": json.loads(latest.read_text())["summary"] if latest.exists() else None,
    }


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
    conversion_manifest = AGENT_CORPUS_ROOT / "_conversion_manifest.json"
    conversion_rows = json.loads(conversion_manifest.read_text()) if conversion_manifest.exists() else []
    if not conversion_manifest.exists():
        issues.append({"check": "conversion_manifest_missing", "severity": "warning"})
    conv_outputs = {str((AGENT_CORPUS_ROOT / r["output_markdown"]).resolve()) for r in conversion_rows}
    for md in md_root.rglob("*.md"):
        if str(md.resolve()) not in conv_outputs:
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
    amb = sum(1 for e in eqs for v in e.get("variables", []) if v.get("resolution_state") == "AMBIGUOUS_SYMBOL")
    unres = sum(1 for e in eqs for v in e.get("variables", []) if v.get("resolution_state") == "UNRESOLVED_ENTITY")
    stats["equation_variables"] = {"ambiguous": amb, "unresolved": unres,
                                   "total": sum(len(e.get("variables", [])) for e in eqs)}
    # duplicates by normalized form within a document
    seen = Counter((e["document_id"], (e.get("form") or "").strip().lower()) for e in eqs)
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

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "n_issues": len(issues),
               "issues_by_check": dict(Counter(i["check"] for i in issues)), "stats": stats}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = REPORT_DIR / f"validation_{stamp}.json"
    payload = {"summary": summary, "issues": issues}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (REPORT_DIR / "latest_validation.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return {"summary": summary, "issues": issues, "report_path": str(path.relative_to(REPO_ROOT))}
