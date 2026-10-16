"""``kg`` - command-line access to LREKit's scientific memory.

Run from the repository root:

    python knowledge/kg/cli.py <command> [args]      (any repo venv; needs numpy)
    knowledge/kg                                     (wrapper script, same thing)

Commands
    search  "query"                 hybrid lexical+semantic chunk search
    docs    "query"                 documents ranked for a query
    doc     DOC_ID                  document record + linked objects
    entity  "name" | ID             find graph nodes / resolve a symbol
    equation "concept" | EQ_ID      find or show equations
    claims  "subject" [--pred X]    find claims
    model   "concept" | MODEL_ID    find or show registry models
    validity MODEL_ID|"concept"     validity conditions for a model
    sources  ID                     provenance + source context + PDF escalation for any object
    implementation ID               software that implements a scientific object
    basis   raosim/x.py[:Symbol]    scientific basis of a software component
    path    "a" "b"                 shortest graph path between two nodes
    compare "query"                 per-document passages + recorded conflicts
    context --task "..."            retrieve_for_task() context package (JSON)
    ask     "question"              alias of context, with a compact rendering
    status                          corpus/ingestion/validation state
    sync [--convert]                check PDF<->Markdown sync (convert with --convert; needs .venv-kg)
    rebuild                         re-run manifest -> chunks -> index -> graph (no LLM extraction)
    validate                        run quality-control checks and write a report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import api  # noqa: E402


def _emit(obj, as_json: bool):
    if as_json:
        print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    else:
        _render(obj)


def _render(obj, indent=0):
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                print(f"{pad}{k}:")
                _render(v, indent + 1)
            else:
                s = str(v)
                if len(s) > 400:
                    s = s[:400] + " …"
                print(f"{pad}{k}: {s}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                print(f"{pad}- [{i}]")
                _render(v, indent + 1)
            else:
                print(f"{pad}- {v}")
    else:
        print(f"{pad}{obj}")


def _compact_context(ctx: dict) -> None:
    print(f"TASK: {ctx['task']}\nSTATE: {ctx['state']}   subsystems={ctx['subsystems']}")
    print(f"quantities: {[q['canonical'] for q in ctx['quantities']]}")
    print("\n== source documents ==")
    for d in ctx["source_documents"]:
        print(f"  {d['id']}  {str(d.get('title'))[:70]}  [{d.get('conversion_status')}]")
    print("\n== models (registry) ==")
    for m in ctx["models"]:
        print(f"  {m['id']}  {m.get('relation','')[:70]}  basis={m.get('basis_class')} promo={m.get('promotion_status')} val={m.get('validation_level')}")
    print("\n== equations ==")
    for e in ctx["equations"]:
        print(f"  {e['id']}  {str(e.get('label'))[:60]}  p{e.get('page')}  {str(e.get('form'))[:80]}")
    print("\n== claims ==")
    for c in ctx["claims"][:12]:
        print(f"  {c['id']}  [{c.get('claim_type')}] {str(c.get('statement'))[:100]}")
    print("\n== validity ==")
    for v in ctx["validity_envelopes"][:10]:
        print(f"  {v['id']}  {str(v.get('prose') or v.get('statement'))[:110]}")
    print("\n== implementations ==")
    for s in ctx["existing_implementations"]:
        print(f"  {s.get('file')}:{s.get('line')}  {s.get('qualname')}")
    print("\n== tests / benchmarks ==")
    for t in ctx["tests"][:10]:
        print(f"  {t.get('path')}  ({t.get('verification_kind')})")
    print("\n== unavailable capabilities ==")
    for c in ctx["unavailable_capabilities"]:
        print(f"  {c.get('name')}  {c.get('state')}")
    print("\n== warnings ==")
    for w in ctx["implementation_warnings"][:15]:
        print(f"  ! {w}")
    print("\n== source context (top chunks) ==")
    for h in ctx["source_context"][:6]:
        print(f"  {h['chunk_id']} p{h['page_start']}  {h.get('section','')[:50]!r}\n     {h['text'][:220].replace(chr(10),' ')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("search", "docs", "doc", "entity", "equation", "claims", "model", "validity", "sources",
                 "implementation", "basis", "compare", "ask", "artifacts", "artifact"):
        p = sub.add_parser(name)
        p.add_argument("arg")
        p.add_argument("--limit", type=int, default=10)
        if name == "claims":
            p.add_argument("--pred", default=None)
            p.add_argument("--obj", default=None)
    p = sub.add_parser("path"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("context"); p.add_argument("--task", required=True); p.add_argument("--limit", type=int, default=12)
    sub.add_parser("status")
    p = sub.add_parser("coverage", help="all-document source/equation coverage; completeness is not presumed")
    p.add_argument("--documents", action="store_true")
    p = sub.add_parser("enrichments", help="unreviewed Docling formula/table candidates")
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--document-id")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("pending", help="show a resumable review queue; never starts agents")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--subsystem", default=None)
    p = sub.add_parser("packet", help="bounded source passages for one model or extracted object")
    p.add_argument("target")
    p.add_argument("--max-chars", type=int, default=16000)
    p = sub.add_parser("sync"); p.add_argument("--convert", action="store_true")
    p = sub.add_parser("rebuild")
    p.add_argument("--force-chunks", action="store_true")
    p.add_argument("--skip-sync", action="store_true")
    sub.add_parser("validate")
    ns = ap.parse_args(argv)

    if ns.cmd == "coverage":
        from kg.coverage import summary
        _emit(summary(include_documents=ns.documents), ns.json)
    elif ns.cmd == "enrichments":
        from kg.enrichment import find
        _emit(find(ns.query, ns.document_id, ns.limit), ns.json)
    elif ns.cmd == "search":
        r = api.semantic_search(ns.arg, limit=ns.limit)
        if ns.json:
            _emit(r, True)
        else:
            print(f"state: {r['state']}")
            for h in r["hits"]:
                print(f"\n[{h['chunk_id']}] p{h['page_start']}-{h['page_end']} score={h['score']} {'(LOW CONFIDENCE) ' if h.get('low_confidence') else ''}{str(h.get('document'))[:70]}\n  section: {h['section'][:80]}\n  {h['text'][:300].replace(chr(10),' ')}")
    elif ns.cmd == "docs":
        _emit(api.search_documents(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "doc":
        _emit(api.get_document(ns.arg), ns.json)
    elif ns.cmd == "entity":
        _emit(api.get_entity(ns.arg) if ns.arg.isupper() and "_" in ns.arg and " " not in ns.arg else api.find_entity(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "artifacts":
        _emit(api.find_artifacts(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "artifact":
        _emit(api.get_artifact(ns.arg), ns.json)
    elif ns.cmd == "equation":
        _emit(api.get_equation(ns.arg) if ns.arg.startswith("EQ_") else api.find_equations(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "claims":
        _emit(api.find_claims(ns.arg, ns.pred, ns.obj, limit=ns.limit), ns.json)
    elif ns.cmd == "model":
        _emit(api.get_model(ns.arg) if ns.arg.startswith("MODEL_") else api.find_models(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "validity":
        _emit(api.find_validity_conditions(ns.arg), ns.json)
    elif ns.cmd == "sources":
        _emit(_sources(ns.arg), ns.json)
    elif ns.cmd == "implementation":
        _emit(api.find_implementation(ns.arg), ns.json)
    elif ns.cmd == "basis":
        _emit(api.find_scientific_basis(ns.arg), ns.json)
    elif ns.cmd == "path":
        _emit(api.find_path(ns.a, ns.b), ns.json)
    elif ns.cmd == "compare":
        _emit(api.compare_sources(ns.arg, limit=ns.limit), ns.json)
    elif ns.cmd == "context":
        ctx = api.retrieve_for_task(ns.task, max_chunks=ns.limit)
        _emit(ctx, True) if ns.json else _compact_context(ctx)
    elif ns.cmd == "ask":
        ctx = api.retrieve_for_task(ns.arg)
        _emit(ctx, True) if ns.json else _compact_context(ctx)
    elif ns.cmd == "pending":
        from kg.work_queue import write
        r = write()
        r["model_reviews"] = [t for t in r["model_reviews"] if not ns.subsystem or t["subsystem"] == ns.subsystem][:max(0, ns.limit)]
        r["verification"] = r["verification"][:max(0, ns.limit)]
        r["source_routing"] = r["source_routing"][:max(0, ns.limit)]
        _emit(r, ns.json)
    elif ns.cmd == "packet":
        from kg.work_queue import packet
        _emit(packet(ns.target, ns.max_chars), ns.json)
    elif ns.cmd == "status":
        from kg import validate
        _emit(validate.status(), ns.json)
    elif ns.cmd == "sync":
        import subprocess
        from kg import REPO_ROOT
        py = REPO_ROOT / ".venv-kg" / "bin" / "python"
        cmd = [str(py if py.exists() else sys.executable), str(REPO_ROOT / "knowledge/tools/sync_corpus.py")]
        if ns.convert:
            cmd.append("--convert")
        return subprocess.call(cmd)
    elif ns.cmd == "rebuild":
        from kg.pipeline import main as rebuild
        return rebuild((["--force-chunks"] if ns.force_chunks else []) +
                       (["--skip-sync"] if ns.skip_sync else []))
    elif ns.cmd == "validate":
        from kg import validate
        rep = validate.run()
        _emit(rep["summary"], ns.json)
        if not ns.json:
            print(f"\nreport written: {rep['report_path']}")
        from kg.report import write
        write()
    return 0


def _sources(obj_id: str) -> dict:
    """Provenance-first view of any object with PDF escalation."""
    from kg.store import index_by
    for store_name in ("equations", "claims", "entities", "validity"):
        rec = index_by(store_name).get(obj_id)
        if rec:
            ctx = api.get_source_context(rec["chunk_id"]) if rec.get("chunk_id") else None
            return {"object": rec, "provenance": index_by("provenance").get(rec.get("provenance_id", "")),
                    "source_context": ctx}
    m = index_by("models").get(obj_id)
    if m:
        return api.get_model(obj_id)
    if obj_id in index_by("artifacts"):
        return api.get_artifact(obj_id)
    ctx = api.get_source_context(obj_id)
    if ctx.get("state") != "INSUFFICIENT_SOURCE_CONTEXT":
        return ctx
    return {"id": obj_id, "state": "UNRESOLVED_ENTITY"}


if __name__ == "__main__":
    sys.exit(main())
