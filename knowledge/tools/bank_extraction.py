"""Bank workflow extraction results from a journal into the knowledge stores.

    .venv/bin/python knowledge/tools/bank_extraction.py <journal.jsonl> [--grouped]

Reads every ``result`` line of a Workflow journal, grounds each object against
the indexed source text (kg.ground) and upserts equations / claims / validity /
entities / provenance.  Safe to run repeatedly: ids are content-derived, so
re-banking the same journal is a no-op.  Writes
``knowledge/reports/latest_extraction.json`` and re-assembles the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from kg import REPO_ROOT, graph, harvest_extraction, report, validate  # noqa: E402
from kg.store import load  # noqa: E402


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_results(journal: Path) -> list[dict]:
    out = {}
    lines = journal.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            # A live journal can end mid-write; corruption in completed lines
            # must remain visible instead of silently losing successful work.
            if index == len(lines) - 1 and not journal.read_text().endswith("\n"):
                continue
            raise ValueError(f"Invalid journal JSON at {journal}:{index + 1}") from exc
        if not isinstance(r, dict):
            raise ValueError(f"Expected a journal object at {journal}:{index + 1}")
        if r.get("type") != "result":
            continue
        v = r.get("value") if "value" in r else r.get("result")
        if isinstance(v, dict):
            out[r.get("key") or fingerprint(v)] = v
        else:
            raise ValueError(f"Invalid result object at {journal}:{index + 1}")
    return list(out.values())


def _unique(records: list[dict]) -> list[dict]:
    return list({fingerprint(record): record for record in records}.values())


def merge_report(previous: dict, latest: dict, fingerprints: set[str]) -> dict:
    """Count stored objects, not harvest attempts, and retain audit failures."""
    merged = dict(previous)
    merged["ungrounded"] = _unique((previous.get("ungrounded") or []) + latest.get("ungrounded", []))
    merged["n_ungrounded"] = len(merged["ungrounded"])
    merged["model_support"] = {m["id"]: m["source_support_state"] for m in load("models")
                               if m.get("source_support_state")}
    records = []
    for name, kind, key in (("equations", "Equation", "n_equations"), ("claims", "Claim", "n_claims"),
                            ("validity", "ValidityEnvelope", "n_validity"),
                            ("entities", "EmpiricalConstant", "n_constants"),
                            ("entities", "Assumption", "n_assumptions")):
        selected = [r for r in load(name) if r.get("type") == kind and r.get("provenance_kind") == "SOURCE_STATES"]
        merged[key] = len(selected)
        records.extend(selected)
    merged["grounded"] = sum(r.get("grounded") == "GROUNDED" for r in records)
    merged["page_shifted"] = sum(r.get("grounded") == "GROUNDED_PAGE_SHIFTED" for r in records)
    merged["per_doc"] = {}
    for row in records:
        counts = merged["per_doc"].setdefault(row["document_id"], {"grounded": 0, "page_shifted": 0, "ungrounded": 0})
        if row.get("grounded") in ("GROUNDED", "GROUNDED_PAGE_SHIFTED"):
            counts["grounded" if row["grounded"] == "GROUNDED" else "page_shifted"] += 1
    for row in merged["ungrounded"]:
        merged["per_doc"].setdefault(row["doc_id"], {"grounded": 0, "page_shifted": 0, "ungrounded": 0})["ungrounded"] += 1
    merged["banked_result_hashes"] = sorted(set(previous.get("banked_result_hashes", [])) | fingerprints)
    merged["count_basis"] = "unique stored source objects; rejected anchors deduplicated across attempts"
    if "cross_document_relations" in latest:
        old_rel = previous.get("cross_document_relations", {})
        merged["cross_document_relations"] = {
            "attached": len(load("relationships")),
            "unresolved": _unique(old_rel.get("unresolved", []) + latest["cross_document_relations"].get("unresolved", [])),
        }
    if "skipped" in latest:
        merged["skipped"] = _unique((previous.get("skipped") or []) + latest["skipped"])
    merged["method"] = {"extraction": "Structured result journal -> kg.ground verbatim-anchor check -> stores",
                        "rules": "verbatim quote + page marker; source notation; no invented validity; honest supports_model"}
    return merged


def _write_report(path: Path | None, report: dict) -> None:
    """The complete structured result of this run, for callers that must check it."""
    if path is not None:
        from kg.store import _atomic_write_json
        _atomic_write_json(path, json.loads(json.dumps(report, default=str)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("journal")
    ap.add_argument("--grouped", action="store_true", help="results are group objects with a 'results' list")
    ap.add_argument("--report-out", type=Path, help="write this run's complete harvest report (JSON) to this file")
    ns = ap.parse_args(argv)
    from kg import locks
    with locks.writer(purpose=f"bank extraction journal {ns.journal}", timeout=None):
        return _bank(ap, ns)


def _bank(ap, ns) -> int:
    try:
        results = load_results(Path(ns.journal))
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if not results:
        ap.error("Journal contains no completed extraction results")
    from kg.store import index_by
    documents = index_by("manifest")
    for entry in results:
        for item in entry.get("results", []) if ns.grouped else [entry]:
            if item.get("source_sha256") and item["source_sha256"] != documents.get(item.get("doc_id"), {}).get("sha256"):
                ap.error(f"Source changed since extraction for {item.get('doc_id')}; review the current PDF before banking")
    prev_path = REPO_ROOT / "knowledge/reports/latest_extraction.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    seen = set(prev.get("banked_result_hashes", []))
    results = _unique(results)
    results = [r for r in results if fingerprint(r) not in seen]
    if not results:
        print("[bank] All results already banked; no changes")
        _write_report(ns.report_out, {"already_banked": True})
        return 0
    if ns.grouped:
        groups = [v for v in results if "results" in v]
        if len(groups) != len(results) or any(not isinstance(g["results"], list) for g in groups):
            ap.error("Every grouped result must contain a 'results' list")
        print(f"[bank] {len(groups)} group results; items: {sum(len(g['results']) for g in groups)}")
        rep = harvest_extraction.harvest_groups(groups)
    else:
        model_results = [v for v in results if str(v.get("target", "")).startswith("MODEL_")]
        window_results = [v for v in results if str(v.get("target", "")).startswith("pages")]
        if len(model_results) + len(window_results) != len(results):
            ap.error("Every extraction result must target MODEL_... or pages...")
        print(f"[bank] {len(model_results)} model results, {len(window_results)} window results")
        rep = harvest_extraction.harvest(model_results, window_results)
    merged = merge_report(prev, rep, {fingerprint(r) for r in results})
    print(json.dumps({k: v for k, v in rep.items() if k not in ("ungrounded", "per_doc", "model_support")}, indent=1, default=str)[:1500])
    _write_report(ns.report_out, {**rep, "already_banked": False})
    print("[bank] graph:", graph.assemble()["meta"])
    # Mark success only after the derived graph has been rebuilt. A failed run
    # can safely repeat its upserts and repair the graph on the next attempt.
    from kg.store import _atomic_write_json
    _atomic_write_json(prev_path, merged)
    validate.run()
    from kg.work_queue import write as write_queue
    write_queue()
    from kg.api import reset_caches
    reset_caches()
    report.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
