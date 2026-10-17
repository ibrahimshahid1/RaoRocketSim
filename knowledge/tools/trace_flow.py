"""Trace one source object end to end through the knowledge system.

    .venv-jax/bin/python knowledge/tools/trace_flow.py EQ_... --query "son movable pintle minimum area" [--out trace.json]

Reads only: manifest, enrichments (Docling runs and candidates), the object's
store record and provenance, the assembled graph, model reviews that cite the
object, and ``kg.api.retrieve_for_task``. Each hop is reported with its state so
a missing link is visible (``MISSING``) rather than silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from kg import api, store  # noqa: E402


def trace(object_id: str, query: str) -> dict:
    record = next((r for name in ("equations", "claims", "entities", "validity") for r in store.load(name)
                   if r["id"] == object_id), None)
    if record is None:
        return {"object_id": object_id, "state": "UNRESOLVED_ENTITY"}
    doc = store.index_by("manifest").get(record.get("document_id"), {})
    prov = store.index_by("provenance").get(record.get("provenance_id"), {})
    enr = store.index_by("enrichments")
    cand_ids = list(record.get("enrichment_candidate_ids", [])) + list(record.get("corrects_candidate_ids", []))
    candidates = []
    for cid in cand_ids:
        c = enr.get(cid, {})
        run = enr.get(c.get("run_id"), {})
        candidates.append({"candidate_id": cid, "relation": "CORRECTS" if cid in record.get("corrects_candidate_ids", []) else "DERIVED_FROM",
                           "review_state": c.get("review_state"), "reviewer": (c.get("review_evidence") or {}).get("reviewer"),
                           "pages": c.get("pages"), "run_id": c.get("run_id"), "run_source_sha256": run.get("source_sha256"),
                           "raw_sha256": c.get("raw_sha256")})
    graph = json.loads((store.GRAPH_DIR / "graph.json").read_text())
    edges = [e for e in graph["edges"] if object_id in (e.get("from"), e.get("to"))]
    # Models associated with the object through the graph (IMPLEMENTS or CITES),
    # each with its structured review state recomputed from current fingerprints.
    relations = {}
    for e in edges:
        if e.get("to") == object_id and e.get("type") in ("IMPLEMENTS", "CITES") and str(e.get("from", "")).startswith("MODEL_"):
            relations.setdefault(e["from"], set()).add(e["type"])
    models_by_id = store.index_by("models")
    models = []
    for mid, kinds in sorted(relations.items()):
        summary = api.model_review_summary(models_by_id.get(mid, {}))
        models.append({"model_id": mid, "relations": sorted(kinds), "review_state": summary["state"],
                       "fidelity": summary.get("fidelity"), "reviewed_document_id": summary.get("reviewed_document_id"),
                       "reviewed_source_sha256": summary.get("reviewed_source_sha256"),
                       "implementation_fingerprints": summary.get("implementation", []),
                       "reviewer": summary.get("reviewer"), "physical_validation": summary.get("physical_validation")})
    implementation = api.find_implementation(object_id)
    package = api.retrieve_for_task(query)
    in_package = any(e.get("id") == object_id for e in package.get("equations", []) + package.get("claims", []))
    packaged_reviews = {m.get("id"): (m.get("source_review") or {}).get("state") for m in package.get("models", [])}
    reviewed_in_package = [m["model_id"] for m in models if packaged_reviews.get(m["model_id"]) == "CURRENT"]
    hops = {
        "1_original_pdf_page": {"document_id": doc.get("id"), "source_pdf": doc.get("source_pdf"), "sha256": doc.get("sha256"),
                                "page": record.get("page"), "state": "OK" if doc else "MISSING"},
        "2_raw_parser_candidates": {"candidates": candidates, "state": "OK" if candidates else "MISSING"},
        "3_reviewed_object": {"provenance_kind": record.get("provenance_kind"), "quote": record.get("quote"),
                              "grounded": record.get("grounded"), "transcribed_by": record.get("transcribed_by"),
                              "provenance_asserted_by": prov.get("asserted_by"), "state": "OK" if record.get("grounded") else "MISSING"},
        "4_independent_verification": {"verification": record.get("verification"), "verifier_role": record.get("verifier_role"),
                                       "verifier_model": record.get("verifier_model"),
                                       "independent": bool(record.get("verifier_role")) and record.get("verifier_role") != (record.get("transcribed_by") or {}).get("role"),
                                       "state": "OK" if record.get("verification") == "CONFIRMED" else "PENDING_OR_REFUTED"},
        "5_graph": {"edges": [{k: e.get(k) for k in ("from", "to", "type")} for e in edges],
                    "state": "OK" if edges else "MISSING"},
        "6_implementation_association_and_verdict": {
            "models": models, "reviewed_implementations": implementation.get("reviewed_implementations", []),
            "find_implementation_state": implementation.get("state"),
            "state": ("OK" if any(m["review_state"] == "CURRENT" for m in models)
                      else "STALE_OR_CHANGED" if models else "MISSING")},
        "7_evidence_aware_query": {"query": query, "object_returned": in_package, "state_of_package": package.get("state"),
                                   "models_with_current_review_in_package": reviewed_in_package,
                                   "implementation_warnings": package.get("implementation_warnings", [])[:8],
                                   "state": "OK" if in_package and reviewed_in_package else "MISSING"},
    }
    return {"object_id": object_id, "form": record.get("form"), "label": record.get("label"), "hops": hops,
            "complete": all(h["state"] == "OK" for h in hops.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("object_id")
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", type=Path)
    ns = ap.parse_args(argv)
    out = trace(ns.object_id, ns.query)
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if ns.out:
        ns.out.write_text(text + "\n")
    print(text)
    return 0 if out.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
