"""End-to-end (re)build of the deterministic layers of the knowledge system.

    .venv/bin/python knowledge/tools/kg_pipeline.py [--force-chunks] [--skip-sync]

Steps (all idempotent):
  1. corpus sync check          (PDF <-> Markdown; conversion needs .venv-kg, see sync_corpus.py)
  2. manifest                   knowledge/manifest.json
  3. chunks                     knowledge/vector_index/chunks.sqlite (only changed docs)
  4. vector index               knowledge/vector_index/*.npy
  5. quantities                 knowledge/quantities.json
  6. model registry harvest     knowledge/models.json, validity.json
  7. repository harvest         repo_documents.json, tests.json, capabilities.json, CITES links
  8. graph                      knowledge/graph/graph.json (+ .graphml)
  9. validation                 knowledge/reports/latest_validation.json
 10. ingestion report           knowledge/reports/latest_ingestion.json

LLM-extracted stores (equations/claims/entities from literature, survey-derived
software components) are NOT regenerated here; they are produced by the
extraction workflows and merged by kg.harvest_extraction / kg.harvest_survey.
This pipeline preserves them and re-links them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))

from kg import graph, harvest_artifacts, harvest_constraints, harvest_registry, harvest_repo, index, ingest, quantities, store, validate, report as knowledge_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-chunks", action="store_true")
    ap.add_argument("--skip-sync", action="store_true")
    ns = ap.parse_args(argv)
    t0 = datetime.now(timezone.utc)
    report: dict = {"started_at": t0.isoformat(), "steps": {}}

    if not ns.skip_sync:
        from tools.sync_corpus import scan
        sync = scan()
        store.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (store.REPORT_DIR / "latest_sync.json").write_text(json.dumps(sync, indent=1))
        report["steps"]["sync"] = {k: (len(v) if isinstance(v, (list, dict)) else v)
                                   for k, v in sync.items() if k != "generated_at"}
        if not sync["manifest_rows"] or sync["missing_markdown"] or sync["stale_markdown"]:
            print("[pipeline] Corpus is absent or not synchronized. Run knowledge/kg.sh sync --convert first.", file=sys.stderr)
            return 1

    from kg import freshness
    before = freshness.snapshot()
    docs = ingest.build_manifest(verbose=False)
    report["steps"]["manifest"] = {"documents": len(docs)}
    s = ingest.ingest_chunks(force=ns.force_chunks, verbose=False)
    report["steps"]["chunks"] = {k: v for k, v in s.items() if k != "states"}
    report["document_states"] = s["states"]
    artifacts = ("vocab.json", "idf.npy", "components.npy", "chunk_vectors.npy", "chunk_ids.json", "index_meta.json")
    meta_path = store.VECTOR_DIR / "index_meta.json"
    previous_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    conn = store.connect()
    fingerprint = index.corpus_fingerprint(conn)
    conn.close()
    if (s["ingested"] or s.get("removed") or ns.force_chunks
            or previous_meta.get("corpus_fingerprint") != fingerprint
            or not all((store.VECTOR_DIR / f).exists() for f in artifacts)):
        report["steps"]["index"] = index.build(verbose=False)
    else:
        report["steps"]["index"] = {"skipped": "no chunk changes"}
    store.save("quantities", quantities.to_records(), key="qid", meta={"n": len(quantities.QUANTITIES)})
    from . import variable_links
    report["steps"]["variable_links"] = variable_links.reconcile()
    r = harvest_registry.harvest()
    report["steps"]["registry"] = {"models": r["models"], "unresolved": r["unresolved"]}
    report["steps"]["freshness"] = freshness.reconcile(before)
    from . import enrichment
    report["steps"]["enrichments"] = enrichment.refresh()
    rr = harvest_repo.harvest()
    report["steps"]["repo"] = {k: v for k, v in rr.items() if k not in ("benchmark_candidates", "citations_unresolved")}
    report["steps"]["repo"]["citations_unresolved"] = rr["citations_unresolved"]
    try:
        report["steps"]["constraints"] = harvest_constraints.harvest()
    except Exception as exc:  # report partial failure and return nonzero below
        report["steps"]["constraints"] = {"error": repr(exc)}
    report["steps"]["artifacts"] = harvest_artifacts.harvest()
    from . import harvest_parameters
    parameters = harvest_parameters.harvest(repo_root=ROOT)
    report["steps"]["parameters"] = {k: parameters[k] for k in ("records", "default_sets", "issues")}
    from . import implementation_freshness
    report["steps"]["implementation_locators"] = implementation_freshness.reconcile(repo_root=ROOT)
    from . import constant_basis
    report["steps"]["constant_basis"] = constant_basis.reconcile(repo_root=ROOT)
    g = graph.assemble()
    report["steps"]["graph"] = g["meta"]
    v = validate.run()
    report["steps"]["validation"] = v["summary"]
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    store.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    from kg.work_queue import write as write_queue
    report["steps"]["review_queue"] = write_queue()["summary"]
    (store.REPORT_DIR / "latest_ingestion.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    knowledge_report.write()
    from kg.api import reset_caches
    reset_caches()
    display = dict(report["steps"])
    display["artifacts"] = {k: v for k, v in display["artifacts"].items() if k != "sources"}
    print(json.dumps(display, indent=1, default=str))
    return 1 if any("error" in step for step in report["steps"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
