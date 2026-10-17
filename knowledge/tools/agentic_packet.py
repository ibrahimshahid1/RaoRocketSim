"""Deterministic role operations for one source packet.

    s1-run       PACKET  Docling the packet pages (one process, cache reuse)  DOCLING_RUNNING -> RAW_CAPTURED
    s2-prepare   PACKET  renders + candidates + stored objects + checks       RAW_CAPTURED -> CANDIDATES_PREPARED
    s3-integrate PACKET  bank the exact reviewed files, rebuild graph view    VERIFIED|REJECTED -> GRAPH_INTEGRATED
    s3-test      PACKET  knowledge tests, replay and integrity checks         GRAPH_INTEGRATED -> TESTED

These commands contain no scientific judgment. Candidate bundles list parser
output and stored records side by side without deciding equivalence; banking
uses only files the reviewer (role F) signed and the queue fingerprinted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT))

from kg import packet_queue as pq, store, work_journal  # noqa: E402
from kg.ground import DocGrounder, form_key, normalize  # noqa: E402

SESSION_DIR = ROOT / "knowledge/workflows/results/agentic-2026-10-16"


def evidence_path(kind: str, pid: str) -> Path:
    """A new evidence file for this attempt; fingerprinted earlier evidence is never overwritten.

    The first attempt writes ``<kind>/<pid>.json``; later attempts (a requeue, a
    re-integration after a block) write ``<kind>/<pid>.attempt-N.json``.
    """
    folder = SESSION_DIR / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{pid}.json"
    n = 2
    while path.exists():
        path = folder / f"{pid}.attempt-{n}.json"
        n += 1
    return path


def integrity_failures(integrity: dict) -> list[str]:
    """Graph invariants that block S3 integration and testing (dangling edges included)."""
    return [name for name in ("duplicate_node_ids", "duplicate_edges", "dangling_edges", "edges_without_provenance")
            if integrity.get(name)]


# Bank-report keys that mean a reviewed result did not land as reviewed.
BANK_MISMATCH_KEYS = ("unresolved_support_links", "support_labels_refused", "unresolved_comparison_links",
                      "unresolved_supersessions", "supersession_carry_forward_undecided", "ungrounded",
                      "model_review_equation_link_mismatches", "bank_report_errors", "unresolved_candidate_links")
RENDER_DIR = ROOT / "knowledge/.work/agentic-2026-10-16/renders"
PYTHON = ROOT / ".venv-jax/bin/python"


def bank_mismatches(report: dict) -> dict:
    """Bank-report failures that block integration of a reviewed result."""
    return {name: report[name] for name in BANK_MISMATCH_KEYS if report.get(name)}


def verification_failures(stats: dict) -> dict:
    """A pre-existing verdict enum cannot hide a refused or partial new verdict."""
    return {name: stats[name] for name in (
        "unknown_ids", "ungrounded_additions", "stale", "self_verification_refused", "unsigned_refused",
        "worker_role_refused", "content_mismatch", "stale_legacy_replay", "content_hash_required",
    ) if stats.get(name)}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _packet(pid: str, *states: str) -> dict:
    packet = pq.load()["packets"].get(pid)
    if not packet:
        raise SystemExit(json.dumps({"state": "UNKNOWN_PACKET", "packet": pid}))
    if states and packet["state"] not in states:
        raise SystemExit(json.dumps({"state": "WRONG_PACKET_STATE", "packet": pid, "current": packet["state"],
                                     "expected": list(states)}))
    return packet


def covering_runs(packet: dict, *, rejected: bool = False) -> dict[int, list[str]]:
    """Successful, current whole-page runs per packet page.

    ``rejected=True`` instead lists runs that an earlier source review rejected.
    Those remain raw parser evidence; they are never reused as transcriptions.
    """
    first, last = packet["pages"]
    pages = {p: [] for p in range(first, last + 1)}
    for r in store.load("enrichments"):
        is_rejected = r.get("review_state") == "REJECTED"
        if (r.get("type") != "GeneratedDocument" or r.get("document_id") != packet["document_id"]
                or r.get("source_sha256") != packet["source_sha256"]
                or r.get("source_state") != "CURRENT" or is_rejected != rejected
                or (not rejected and not r.get("active", True))
                or r.get("conversion_status") not in ("success", "partial_success")
                or r.get("options", {}).get("mode") == "visually_located_formula_regions"):
            continue
        a, b = r.get("page_range", [0, -1])
        for p in r.get("observed_pages", []):
            if p in pages and a <= p <= b:
                pages[p].append(r["id"])
    return pages


def s1_run(pid: str, model: str, timeout: int) -> dict:
    """Process at most ONE missing page per call so a call fits a bounded tool budget."""
    packet = _packet(pid, "DOCLING_RUNNING")
    first, last = packet["pages"]
    before = covering_runs(packet)
    prior_rejected = covering_runs(packet, rejected=True)
    # A deterministic parser rerun would repeat a rejected transcription. Such a
    # page counts as raw-captured with its rejected output visible for review.
    todo = [p for p, runs in before.items() if not runs and not prior_rejected[p]]
    job = None
    exit_code = 0
    if todo:
        page = todo[0]
        command = [str(PYTHON), str(HERE / "docling_batch.py"), "--run", "--document-id", packet["document_id"],
                   "--pages", f"{page}:{page}", "--pages-per-job", "1", "--max-jobs", "1",
                   "--timeout", str(timeout)]
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout + 90)
            exit_code = result.returncode
            stdout = result.stdout
        except subprocess.TimeoutExpired:
            exit_code, stdout = 124, ""
        for line in stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "pages" in row:
                job = {k: row.get(k) for k in ("pages", "exit_code", "state", "processed_pages_gained",
                                               "new_attempt_ids")}
    after = covering_runs(packet)
    rejected_only = sorted(p for p, runs in after.items() if not runs and prior_rejected[p])
    covered = sorted(p for p, runs in after.items() if runs)
    run_ids = sorted({rid for runs in after.values() for rid in runs}
                     | {rid for p in rejected_only for rid in prior_rejected[p]})
    remaining = [p for p, runs in after.items() if not runs and p not in rejected_only]
    out = {"packet": pid, "document_id": packet["document_id"], "pages": [first, last], "job": job,
           "batch_exit_code": exit_code, "covered_pages": covered, "remaining_pages": remaining,
           "pages_with_only_rejected_parser_output": rejected_only, "run_ids": run_ids}
    if not remaining:
        note = f"Docling whole-page runs cover pages {first}-{last}"
        if rejected_only:
            note += f"; pages {rejected_only} have only previously REJECTED parser output (raw evidence, not transcription)"
        pq.transition(pid, "RAW_CAPTURED", role="S1", model=model, run_ids=run_ids, note=note)
        out["state"] = "RAW_CAPTURED"
    elif exit_code == 75:
        pq.transition(pid, "QUEUED", role="S1", model=model, note="DOCLING_BUSY: another Docling process holds the machine lock")
        out["state"] = "QUEUED_BUSY"
    elif exit_code == 0 and todo and todo[0] not in remaining:
        out["state"] = "DOCLING_RUNNING_PARTIAL"  # one page banked; call again for the next page
    else:
        reason = (f"Docling page {todo[0] if todo else '?'} not captured: batch exit {exit_code}; "
                  f"job state {(job or {}).get('state')}; covered {covered} of {first}-{last}")
        attempts = pq.load()["packets"][pid]["attempts"]
        target = "BLOCKED" if attempts >= 2 else "QUEUED"
        pq.transition(pid, target, role="S1", model=model, note=reason)
        out["state"] = target
        out["reason"] = reason
    return out


def _render(pdf: Path, document_id: str, page: int) -> dict:
    target = RENDER_DIR / document_id / f"p{page:04d}.png"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r150",
                        f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={target}", str(pdf)],
                       check=True, capture_output=True, timeout=180)
    return {"page": page, "path": _rel(target), "sha256": _sha(target), "dpi": 150,
            "renderer": "ghostscript png16m"}


_DAMAGE = re.compile(r"not\s+rendered|ocr[-\s]+garbled|�|(\b\w+\b)(?:\s+\1\b){5,}", re.I)
# A short unit repeated back to back (e.g. a radical-overline glyph code
# ".0129" emitted 44 times by a PDF text layer) is not caught by the
# whitespace-separated word rule above.
_REPEATED_UNIT = re.compile(r"(.{2,12}?)\1{7,}", re.S)


def _formula_flags(text: str, original: str = "") -> list[str]:
    """Mechanical flags only; a flag directs the reviewer to the page, it never decides."""
    flags = []
    if not text.strip():
        flags.append("EMPTY_TRANSCRIPTION")
    pairs = {"(": ")", "[": "]", "{": "}"}
    for a, b in pairs.items():
        if text.count(a) != text.count(b):
            flags.append(f"UNBALANCED_{a}{b}")
    if _DAMAGE.search(text or "") or _REPEATED_UNIT.search(text or ""):
        flags.append("DAMAGE_OR_REPETITION_MARKER")
    # The parser's text-layer rendition is evidence too: a replacement
    # character or a repeated glyph code there means the transcription was
    # produced over damaged source text and needs page comparison.
    if "�" in (original or ""):
        flags.append("REPLACEMENT_CHARACTER_IN_ORIGINAL_TEXT")
    if _REPEATED_UNIT.search(original or ""):
        flags.append("REPEATED_GLYPH_RUN_IN_ORIGINAL_TEXT")
    if len(text) > 1500:
        flags.append("VERY_LONG_TRANSCRIPTION")
    return flags


def s2_prepare(pid: str, model: str) -> dict:
    packet = _packet(pid, "RAW_CAPTURED")
    first, last = packet["pages"]
    doc = store.index_by("manifest")[packet["document_id"]]
    pdf = ROOT / doc["source_pdf"]
    if _sha(pdf) != packet["source_sha256"]:
        raise SystemExit(json.dumps({"state": "SOURCE_CHANGED", "packet": pid}))
    run_ids = set(packet["evidence"].get("docling_run_ids", []))
    renders = [_render(pdf, packet["document_id"], p) for p in range(first, last + 1)]
    candidates = []
    raw_cache: dict[str, dict] = {}

    def raw_document(r: dict) -> dict:
        if not r.get("raw_artifact"):
            return {}
        if r["raw_artifact"] not in raw_cache:
            raw_path = ROOT / r["raw_artifact"]
            ok = raw_path.is_file() and _sha(raw_path) == r.get("raw_sha256")
            raw_cache[r["raw_artifact"]] = json.loads(raw_path.read_text()) if ok else {}
        return raw_cache[r["raw_artifact"]]

    def raw_item(r: dict) -> dict:
        parts = (r.get("json_pointer") or "").strip("/").split("/")
        raw = raw_document(r)
        if len(parts) == 2 and parts[1].isdigit() and parts[0] in raw and int(parts[1]) < len(raw[parts[0]]):
            return raw[parts[0]][int(parts[1])]
        return {}

    def caption_texts(r: dict) -> list[str]:
        """Dereference caption pointers of the byte-verified raw item (figures AND tables)."""
        captions = (r.get("content") or {}).get("captions") or raw_item(r).get("captions") or []
        refs = [c.get("$ref") for c in captions if isinstance(c, dict)]
        if not refs:
            return []
        raw = raw_document(r)
        out = []
        for ref in refs:
            parts = (ref or "").lstrip("#/").split("/")
            if len(parts) == 2 and parts[1].isdigit() and parts[0] in raw and int(parts[1]) < len(raw[parts[0]]):
                out.append(raw[parts[0]][int(parts[1])].get("text", ""))
            else:
                out.append(f"UNRESOLVED_REF:{ref}")
        return out

    for r in store.load("enrichments"):
        if r.get("run_id") not in run_ids or not r.get("candidate_kind"):
            continue
        content = r.get("content")
        grid = None
        if r["candidate_kind"] == "table" and isinstance(content, dict) and isinstance(content.get("grid"), list):
            # Complete cell text matrix; a truncated JSON dump would be invalid and incomplete.
            grid = [[(cell or {}).get("text", "") if isinstance(cell, dict) else str(cell) for cell in row]
                    for row in content["grid"]]
            text = f"TABLE {content.get('num_rows')}x{content.get('num_cols')} (see table_grid; full cells in raw item {r.get('json_pointer')})"
        else:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        candidates.append({"id": r["id"], "kind": r["candidate_kind"], "pages": r.get("pages"), "table_grid": grid,
                           "json_pointer": r.get("json_pointer"),
                           "caption_text": caption_texts(r) if r["candidate_kind"] in ("figure", "table") else [],
                           "bbox": [p.get("bbox") for p in r.get("page_provenance", [])],
                           "transcription": text, "original_text": r.get("original_text", ""),
                           "transcription_state": r.get("transcription_state"), "review_state": r.get("review_state"),
                           "run_id": r.get("run_id"), "raw_artifact": r.get("raw_artifact"),
                           "raw_sha256": r.get("raw_sha256"),
                           "mechanical_flags": (_formula_flags(text, r.get("original_text") or "")
                                                if r["candidate_kind"] == "formula" else [])})
    candidates.sort(key=lambda c: (min(c["pages"] or [0]), c["kind"], c["id"]))
    grounder = DocGrounder(packet["document_id"])
    targets = set(packet["targets"])
    objects = []
    forms = Counter()
    for name in ("equations", "claims", "entities", "validity"):
        for r in store.load(name):
            if r.get("document_id") != packet["document_id"]:
                continue
            if r["id"] not in targets and not (r.get("page") and first <= r["page"] <= last):
                continue
            row = {"id": r["id"], "store": name, "type": r.get("type"), "page": r.get("page"),
                   "is_target": r["id"] in targets, "active": r.get("active", True),
                   "source_state": r.get("source_state", "CURRENT"), "verification": r.get("verification"),
                   "label": r.get("label"), "equation_number": r.get("equation_number"),
                   "form": r.get("form"), "statement": r.get("statement") or r.get("prose"),
                   "quote": r.get("quote"), "confidence": r.get("confidence"),
                   "model_ids": r.get("model_ids", []), "grounded": r.get("grounded"),
                   "variables": r.get("variables", []), "assumptions": r.get("assumptions", []),
                   "validity": r.get("validity", []), "extraction_note": r.get("extraction_note", "")}
            checks = []
            if r.get("quote"):
                anchor = grounder.locate(r["quote"], r.get("page"))
                checks.append({"check": "quote_anchor", "state": anchor["state"], "page": anchor.get("page")})
            if name == "equations":
                checks.append({"check": "formula_mechanics", "flags": _formula_flags(r.get("form") or "")})
                form_text = r.get("form") or ""
                missing = [v.get("symbol") for v in r.get("variables", [])
                           if v.get("symbol") and v["symbol"] not in form_text]
                if missing:
                    checks.append({"check": "variable_symbol_absent_from_form", "symbols": missing})
                unresolved = [v.get("symbol") for v in r.get("variables", [])
                              if v.get("resolution_state") != "RESOLVED"]
                if unresolved:
                    checks.append({"check": "unresolved_quantity_mapping", "symbols": unresolved})
                if r.get("active", True) and r.get("source_state", "CURRENT") == "CURRENT":
                    forms[form_key(form_text)] += 1
                row["same_page_formula_candidates"] = [c["id"] for c in candidates
                                                       if c["kind"] == "formula" and r.get("page") in (c["pages"] or [])]
            row["mechanical_checks"] = checks
            objects.append(row)
    for row in objects:
        if (row["store"] == "equations" and row["active"] and row["source_state"] == "CURRENT"
                and forms[form_key(row.get("form") or "")] > 1):
            row["mechanical_checks"].append({"check": "duplicate_form_in_packet"})
    models = store.index_by("models")
    model_targets = sorted(t for t in targets if t.startswith("MODEL_"))
    # A model is a review subject, not a stored source object on these pages.
    missing_targets = sorted(t for t in targets - {o["id"] for o in objects} if not t.startswith("MODEL_"))
    unknown_model_targets = [t for t in model_targets if t not in models]
    conn = store.connect()
    chunks = [dict(c) for c in conn.execute(
        "SELECT chunk_id, page_start, page_end, section, text FROM chunks WHERE doc_id=? AND page_end>=? AND page_start<=? ORDER BY seq",
        (packet["document_id"], first, last))]
    conn.close()
    bundle = {"packet": pid, "kind": packet["kind"], "document_id": packet["document_id"],
              "source_pdf": doc["source_pdf"], "source_sha256": packet["source_sha256"], "pages": [first, last],
              "prepared_by": {"role": "S2", "model": model},
              "authority": "PROVISIONAL CANDIDATES - parser output and stored records side by side; no equivalence, "
                           "correctness or support judgment is made here. The original PDF page decides.",
              "renders": renders, "docling_run_ids": sorted(run_ids), "candidates": candidates,
              "stored_objects": objects, "missing_targets": missing_targets,
              "model_targets": model_targets, "unknown_model_targets": unknown_model_targets,
              "mechanical_flag_scope": "mechanical_flags are computed for formula candidates only; figure/table candidates carry dereferenced caption_text instead",
              "markdown_chunks": chunks,
              "summary": {"candidates_by_kind": dict(Counter(c["kind"] for c in candidates)),
                          "flagged_candidates": sum(bool(c["mechanical_flags"]) for c in candidates),
                          "target_objects": sum(o["is_target"] for o in objects),
                          "objects_with_anchor_problems": sum(any(ch.get("state") == "UNGROUNDED" for ch in o["mechanical_checks"]) for o in objects)}}
    path = evidence_path("candidates", pid)
    path.write_text(json.dumps(bundle, indent=1, ensure_ascii=False))
    pq.transition(pid, "CANDIDATES_PREPARED", role="S2", model=model, evidence_paths=[str(path)],
                  note=json.dumps(bundle["summary"]))
    return {"packet": pid, "state": "CANDIDATES_PREPARED", "bundle": _rel(path), **bundle["summary"],
            "missing_targets": missing_targets}


def _graph_integrity() -> dict:
    graph = json.loads((store.GRAPH_DIR / "graph.json").read_text())
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    ids = Counter(n["id"] for n in nodes)
    keys = Counter((e["from"], e["to"], e["type"]) for e in edges)
    node_ids = set(ids)
    return {"nodes": len(nodes), "edges": len(edges),
            "duplicate_node_ids": sorted(k for k, v in ids.items() if v > 1)[:20],
            "duplicate_edges": [list(k) for k, v in keys.items() if v > 1][:20],
            "dangling_edges": sum(1 for e in edges if e["from"] not in node_ids or e["to"] not in node_ids),
            "edges_without_provenance": sum(1 for e in edges if not e.get("provenance"))}


def s3_integrate(pid: str, model: str) -> dict:
    packet = _packet(pid, "VERIFIED", "REJECTED")
    review = packet["evidence"]["review"]
    log = {"packet": pid, "at": datetime.now(timezone.utc).isoformat(), "integrated_by": {"role": "S3", "model": model},
           "steps": []}
    from kg import graph
    from kg.harvest_extraction import apply_verification
    for f in review["files"]:
        path = ROOT / f["path"]
        if _sha(path) != f["sha256"]:
            raise SystemExit(json.dumps({"state": "REVIEWED_FILE_CHANGED", "file": f["path"]}))
        if path.suffix == ".jsonl":
            report_path = evidence_path("bank_reports", f"{pid}.{path.stem}")
            result = subprocess.run([str(PYTHON), str(HERE / "bank_extraction.py"), str(path),
                                     "--report-out", str(report_path)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=1800)
            try:
                bank_report = json.loads(report_path.read_text())
                if not isinstance(bank_report, dict):
                    raise ValueError("bank report must be a JSON object")
            except (OSError, ValueError) as exc:
                bank_report = {"bank_report_errors": [{"report": _rel(report_path),
                                                       "reason": f"{type(exc).__name__}: {exc}"}]}
            log["steps"].append({"file": f["path"], "action": "bank_extraction", "exit_code": result.returncode,
                                 "report": _rel(report_path) if report_path.exists() else None,
                                 "counts": {k: bank_report.get(k) for k in ("n_equations", "n_claims", "n_constants",
                                                                            "n_assumptions", "n_validity", "already_banked")},
                                 "mismatches": bank_mismatches(bank_report),
                                 "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-2000:]})
            if result.returncode:
                raise SystemExit(json.dumps({"state": "BANKING_FAILED", "log": log}))
        else:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and "duplicate_retirements" in payload:
                from kg.harvest_extraction import retire_duplicate_equations
                stats = retire_duplicate_equations(payload)
                log["steps"].append({"file": f["path"], "action": "retire_duplicate_equations", "stats": stats,
                                     "mismatches": {"refused_duplicate_retirements": stats["refused"]} if stats["refused"] else {}})
                continue
            if isinstance(payload, dict) and "parser_reviews" in payload:
                from kg.enrichment import record_review
                reviewer = payload["reviewer"]
                results = []
                for item in payload["parser_reviews"]:
                    evidence = dict(item.get("evidence") or {}, reviewer=reviewer, review_file=f["path"],
                                    review_file_sha256=f["sha256"])
                    results.append(record_review(item["candidate_id"], state=item["state"],
                                                 reason=item["reason"], evidence=evidence))
                log["steps"].append({"file": f["path"], "action": "record_parser_reviews", "results": results})
                continue
            stats = apply_verification(payload)
            failures = verification_failures(stats)
            log["steps"].append({"file": f["path"], "action": "apply_verification", "stats": stats,
                                 "mismatches": {"verification_refusals": failures} if failures else {}})
    meta = graph.assemble()["meta"]
    from kg.work_queue import write as write_queue
    queue_summary = write_queue()["summary"]
    from kg import report
    report.write()
    from kg.api import reset_caches
    reset_caches()
    log["graph_meta"] = meta
    log["review_queue"] = queue_summary
    log["integrity"] = _graph_integrity()
    # Verdicts may target any verifiable store (constants and assumptions live
    # in entities; source-stated validity in validity), not only equations
    # and claims; an unresolved id must read as a mismatch, never as a pass.
    stored_by_id = {r["id"]: r for name in ("equations", "claims", "entities", "validity")
                    for r in store.load(name)}
    expected = []
    for f in review["files"]:
        path = ROOT / f["path"]
        payload = None if path.suffix == ".jsonl" else json.loads(path.read_text())
        if isinstance(payload, list):
            for v in payload:
                rec = stored_by_id.get(v["object_id"]) or {}
                want = "REFUTED" if v.get("refuted") else "UNCONFIRMED" if v.get("confidence") == "low" else "CONFIRMED"
                retired = bool(rec and rec.get("source_state") != "STALE_SOURCE" and (
                    not rec.get("active", True) or rec.get("source_state", "CURRENT") != "CURRENT"))
                expected.append({"object_id": v["object_id"], "expected": want, "stored": rec.get("verification"),
                                 "retired_history_no_change": retired,
                                 "match": retired or rec.get("verification") == want})
    log["verdict_state_checks"] = expected
    log["verdict_state_mismatches"] = [e for e in expected if not e["match"]]
    log["bank_mismatches"] = [{"file": s["file"], **s["mismatches"]} for s in log["steps"] if s.get("mismatches")]
    path = evidence_path("integration", pid)
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False))
    integrity = log["integrity"]
    if log["verdict_state_mismatches"] or log["bank_mismatches"] or integrity_failures(integrity):
        pq.transition(pid, "BLOCKED", role="S3", model=model, evidence_paths=[str(path)],
                      note="integration invariant failed; handed to F")
        return {"packet": pid, "state": "BLOCKED", "log": _rel(path), "integrity": integrity,
                "verdict_state_mismatches": log["verdict_state_mismatches"], "bank_mismatches": log["bank_mismatches"]}
    pq.transition(pid, "GRAPH_INTEGRATED", role="S3", model=model, evidence_paths=[str(path)],
                  note=f"graph nodes={integrity['nodes']} edges={integrity['edges']}")
    return {"packet": pid, "state": "GRAPH_INTEGRATED", "log": _rel(path), "integrity": integrity}


def s3_test(pid: str, model: str) -> dict:
    packet = _packet(pid, "GRAPH_INTEGRATED")
    review = packet["evidence"]["review"]
    log = {"packet": pid, "at": datetime.now(timezone.utc).isoformat(), "tested_by": {"role": "S3", "model": model}}
    tests = subprocess.run([str(PYTHON), "-m", "pytest", "knowledge/tests", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
    log["knowledge_tests"] = {"command": ".venv-jax/bin/python -m pytest knowledge/tests -q -p no:cacheprovider",
                              "exit_code": tests.returncode, "tail": tests.stdout[-1500:]}
    from kg.harvest_extraction import apply_verification
    replay = []
    for f in review["files"]:
        path = ROOT / f["path"]
        if path.suffix == ".jsonl":
            again = subprocess.run([str(PYTHON), str(HERE / "bank_extraction.py"), str(path)], cwd=ROOT,
                                   capture_output=True, text=True, timeout=1800)
            replay.append({"file": f["path"], "exit_code": again.returncode,
                           "already_banked": "All results already banked" in again.stdout})
        else:
            verdicts = json.loads(path.read_text())
            if isinstance(verdicts, dict) and "duplicate_retirements" in verdicts:
                from kg.harvest_extraction import retire_duplicate_equations
                stats = retire_duplicate_equations(verdicts)
                replay.append({"file": f["path"], "stats": stats,
                               "idempotent": stats["retired"] == 0 and not stats["refused"]
                               and stats["unchanged"] == len(verdicts["duplicate_retirements"])})
                continue
            if isinstance(verdicts, dict):
                from kg.store import index_by
                from kg.enrichment import review_replay_state
                rows = index_by("enrichments")
                documents = index_by("manifest")
                checks = []
                for item in verdicts["parser_reviews"]:
                    selected = [r for r in rows.values() if r["id"] == item["candidate_id"]
                                or r.get("run_id") == item["candidate_id"]]
                    evidence = dict(item.get("evidence") or {}, reviewer=verdicts["reviewer"],
                                    review_file=f["path"], review_file_sha256=f["sha256"])
                    checks.append(bool(selected) and all(
                        r.get("source_state") == "CURRENT" and bool(r.get("source_sha256"))
                        and r["source_sha256"] == documents.get(r.get("document_id"), {}).get("sha256")
                        and review_replay_state(r, state=item["state"], reason=item["reason"], evidence=evidence)
                        for r in selected))
                replay.append({"file": f["path"], "idempotent": all(checks)})
                continue
            stats = apply_verification(verdicts)
            replay.append({"file": f["path"], "stats": stats,
                           "idempotent": (not verification_failures(stats)
                                          and stats.get("unchanged", 0) + stats.get("retired_or_noncurrent", 0)
                                          == len(verdicts))})
    log["replay"] = replay
    log["integrity"] = _graph_integrity()
    ok = (tests.returncode == 0 and all(r.get("already_banked", r.get("idempotent")) for r in replay)
          and not integrity_failures(log["integrity"]))
    path = evidence_path("tests", pid)
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False))
    if ok:
        pq.transition(pid, "TESTED", role="S3", model=model, evidence_paths=[str(path)], test_result="pass",
                      note=tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "")
        return {"packet": pid, "state": "TESTED", "log": _rel(path)}
    pq.transition(pid, "BLOCKED", role="S3", model=model, evidence_paths=[str(path)],
                  note="tests, replay or integrity failed; handed to F")
    return {"packet": pid, "state": "BLOCKED", "log": _rel(path), "replay": replay,
            "tests_exit": tests.returncode}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("s1-run", "s2-prepare", "s3-integrate", "s3-test"))
    ap.add_argument("packet")
    ap.add_argument("--model", required=True, help="acting model identity as known to the worker")
    ap.add_argument("--timeout", type=int, default=480, help="per-page Docling timeout in seconds (s1-run)")
    ns = ap.parse_args(argv)
    try:
        if ns.command == "s1-run":
            out = s1_run(ns.packet, ns.model, ns.timeout)
        elif ns.command == "s2-prepare":
            out = s2_prepare(ns.packet, ns.model)
        elif ns.command == "s3-integrate":
            out = s3_integrate(ns.packet, ns.model)
        else:
            out = s3_test(ns.packet, ns.model)
    except (ValueError, PermissionError) as exc:
        print(json.dumps({"state": "REFUSED", "packet": ns.packet, "error": str(exc)}))
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
