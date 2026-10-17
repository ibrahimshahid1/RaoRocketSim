"""Checkpointed local Docling continuation. Each bounded job banks before the next.

The default only writes the full-corpus plan. --run executes a limited number of
jobs in isolated workers, with wall-clock timeouts; it never starts paid agents.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from kg.enrichment import queue, refresh
from kg.store import load


def _jobs(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def processing_plan(plan, jobs, runs=()):
    """Attempted/rejected pages are backlogs, not automatically unattempted work."""
    plan = copy.deepcopy(plan)
    for d in plan["documents"]:
        valid = set(range(1, d["page_count"] + 1))
        attempted = set()
        rejected = set(d.get("quality_review_pages", [])) & valid
        for run in runs:
            if (run.get("type") != "GeneratedDocument" or run.get("document_id") != d["document_id"]
                    or run.get("source_sha256") != d["source_sha256"]
                    or run.get("options", {}).get("mode") == "visually_located_formula_regions"):
                continue
            pages = run.get("page_range", [])
            if len(pages) == 2 and all(isinstance(p, int) for p in pages):
                requested = set(range(max(1, pages[0]), min(d["page_count"], pages[1]) + 1))
                attempted.update(requested)
                if run.get("review_state") == "REJECTED":
                    rejected.update(requested)  # backlog of the request, never processing credit
        for job in jobs:
            if job.get("document_id") != d["document_id"] or job.get("source_sha256") != d["source_sha256"]:
                continue
            if job.get("state") == "DOCLING_BUSY":
                continue  # refused before inference started; nothing was attempted
            pages = job.get("pages", [])
            if len(pages) == 2 and all(isinstance(p, int) for p in pages):
                first, last = pages
                attempted.update(range(max(1, first), min(d["page_count"], last) + 1))
        remaining = set(d["remaining_pages"]) & valid
        d["rejected_attempt_pages"] = sorted(rejected)  # includes superseded parser attempts
        d["quality_review_pages"] = sorted(rejected & remaining)
        d["attempted_unfinished_pages"] = sorted((remaining & attempted) - rejected)
        d["unattempted_pages"] = sorted(remaining - attempted - rejected)
    for field in ("quality_review_pages", "attempted_unfinished_pages", "unattempted_pages"):
        plan["summary"][field] = sum(len(d[field]) for d in plan["documents"])
    plan["scheduling_scope"] = "Next unattempted processing excludes rejected quality-review and attempted-unfinished backlogs; explicit --reprocess is separate."
    return plan


def _attempt_ids(document_id, source_sha256, pages):
    return {r["id"] for r in load("enrichments") if r.get("type") == "GeneratedDocument"
            and r.get("document_id") == document_id and r.get("source_sha256") == source_sha256
            and r.get("page_range") == list(pages)
            and r.get("options", {}).get("mode") != "visually_located_formula_regions"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=1)
    ap.add_argument("--pages-per-job", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--document-id")
    ap.add_argument("--reprocess", action="store_true", help="Explicitly bank new attempts for --document-id and --pages")
    ap.add_argument("--pages", help="Inclusive explicit range, e.g. 7:8; requires --document-id. Without "
                    "--reprocess, a byte-verified cached attempt is reused instead of rerun")
    ns = ap.parse_args(argv)
    if min(ns.max_jobs, ns.pages_per_job, ns.timeout) < 1:
        ap.error("job/page/time limits must be positive")
    if (ns.reprocess and (not ns.document_id or not ns.pages)) or (ns.pages and not ns.document_id):
        ap.error("explicit pages require --document-id DOC --pages FIRST:LAST (add --reprocess for a distinct attempt)")
    refresh()
    plan_path = ROOT / "knowledge/reports/docling_queue.json"
    log_path = ROOT / "knowledge/reports/docling_jobs.jsonl"
    jobs = _jobs(log_path)
    plan = processing_plan(queue(), jobs, load("enrichments"))
    explicit = None
    if ns.pages:
        try:
            first, last = map(int, ns.pages.split(":"))
        except ValueError:
            ap.error("pages must be FIRST:LAST")
        target = next((d for d in plan["documents"] if d["document_id"] == ns.document_id), None)
        if target is None or not 1 <= first <= last <= target["page_count"]:
            ap.error("requested reprocessing range is outside the selected source PDF")
        explicit = set(range(first, last + 1))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=1))
    if not ns.run:
        print(json.dumps(plan["summary"]))
        return 0
    python = ROOT / ".venv-docling/bin/python"
    if not python.exists():
        ap.error("install knowledge/requirements-docling.lock in .venv-docling first")
    attempted_this_run = set()
    for _ in range(ns.max_jobs):
        eligible = []
        for d in plan["documents"]:
            if ns.document_id and d["document_id"] != ns.document_id:
                continue
            pages = sorted(explicit - attempted_this_run) if explicit is not None else d["unattempted_pages"]
            if pages:
                eligible.append((d, pages))
        if not eligible:
            break
        d, pending = eligible[0]
        first = pending[0]
        last = first
        while last + 1 in pending and last < first + ns.pages_per_job - 1:
            last += 1
        command = [str(python), str(ROOT / "knowledge/tools/docling_enrich.py"), d["document_id"],
                   "--pages", f"{first}:{last}", "--timeout", str(ns.timeout)]
        if ns.reprocess:
            command.append("--reprocess")
        record = {"document_id": d["document_id"], "source_sha256": d["source_sha256"],
                  "pages": [first, last], "reprocess": ns.reprocess}
        requested = set(range(first, last + 1))
        previous_pages = set(d["processed_pages"]) & requested
        previous_attempts = _attempt_ids(d["document_id"], d["source_sha256"], (first, last))
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=ns.timeout)
            record.update(exit_code=result.returncode, stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
        except subprocess.TimeoutExpired:
            record.update(exit_code=124, state="WORKER_TIMEOUT")
        if record.get("exit_code") == 75:
            # Another Docling process holds the machine-wide lock. Record the
            # refusal without converting it into attempted-page history.
            record["state"] = "DOCLING_BUSY"
            with log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            return 75
        fresh = queue()
        current = next((r for r in fresh["documents"] if r["document_id"] == d["document_id"]
                        and r["source_sha256"] == d["source_sha256"]), {})
        gained = (set(current.get("processed_pages", [])) & requested) - previous_pages
        new_attempts = _attempt_ids(d["document_id"], d["source_sha256"], (first, last)) - previous_attempts
        record.update(processed_pages_gained=sorted(gained), new_attempt_ids=sorted(new_attempts))
        if (record["exit_code"] == 0 and explicit is not None and not ns.reprocess
                and requested <= previous_pages):
            record["state"] = "ALREADY_PROCESSED_CACHE_REUSED"  # explicit scheduling of a completed page
        elif record["exit_code"] == 0 and not (new_attempts if ns.reprocess else gained):
            record.update(exit_code=2, state="NO_PROGRESS_SOURCE_REVIEW_REQUIRED")
        elif record["exit_code"] == 0 and ns.reprocess:
            record["state"] = "DISTINCT_ATTEMPT_BANKED_REVIEW_REQUIRED"
        elif record["exit_code"] == 0:
            record["state"] = "PROCESSED_REVIEW_REQUIRED"
        jobs.append(record)
        attempted_this_run.update(requested)
        plan = processing_plan(fresh, jobs, load("enrichments"))
        with log_path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        plan_path.write_text(json.dumps(plan, indent=1))
        print(json.dumps(record), flush=True)
        if record["exit_code"]:
            return record["exit_code"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
