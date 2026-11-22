#!/usr/bin/env python3
"""Check source integrity and registry coverage; never certify model validity."""
from __future__ import annotations

import csv
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    sources = json.loads((ROOT / "catalog/sources.json").read_text())["sources"]
    crosswalk = json.loads((ROOT / "catalog/model_crosswalk.json").read_text())
    source_ids = [s["id"] for s in sources]
    errors, gaps, observations, queue = [], [], [], []
    if len(set(source_ids)) != len(source_ids):
        errors.append("Duplicate catalog source IDs")
    statuses, evidence_kinds, hashes, csv_shapes = Counter(), Counter(), {}, []
    file_count = payload_bytes = pdf_count = new_downloads = local_copies = fixtures = mirror_copies = 0
    for source in sources:
        sid = source["id"]
        path = ROOT / "catalog/receipts" / (sid + ".json")
        if not path.is_file():
            errors.append(f"Missing receipt: {sid}")
            continue
        r = json.loads(path.read_text())
        if r["source_id"] != sid:
            errors.append(f"Receipt identity mismatch: {sid}")
        statuses[r["status"]] += 1
        evidence_kinds[r["evidence_kind"]] += 1
        if r.get("training_eligible") is not False or source.get("training_eligible") is not False:
            errors.append(f"Acquisition-only source unexpectedly training-eligible: {sid}")
        if r.get("hardware_qualified") is not False:
            errors.append(f"Unexpected hardware qualification claim: {sid}")
        if r["status"] != "ACQUIRED":
            gaps.append({"source_id": sid, "status": r["status"],
                         "next_action": r.get("next_action", r.get("error"))})
        for f in r["files"]:
            p = (ROOT / f["path"]).resolve()
            if not p.is_relative_to(ROOT.resolve()):
                errors.append(f"Path escapes collection: {sid}: {f['path']}")
                continue
            if not p.is_file():
                errors.append(f"Missing payload: {f['path']}")
                continue
            actual = sha256(p)
            if actual != f["sha256"] or p.stat().st_size != f["bytes"]:
                errors.append(f"Payload hash/size mismatch: {f['path']}")
            file_count += 1
            if f["origin"] != "unreviewed_pdf_text_derivative":
                payload_bytes += p.stat().st_size
            hashes.setdefault(actual, []).append(f["path"])
            if p.suffix == ".pdf":
                pdf_count += 1
                if not p.read_bytes().startswith(b"%PDF-"):
                    errors.append(f"Not a PDF: {f['path']}")
                if f["origin"] == "publisher_report_pdf":
                    new_downloads += 1
                if f["origin"] == "existing_local_literature_copy":
                    local_copies += 1
                if f["origin"] == "original_paper_on_third_party_host":
                    mirror_copies += 1
            if p.suffix == ".csv":
                with p.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    bad = sum(None in row or any(v is None for v in row.values()) for row in rows)
                    if bad:
                        errors.append(f"Malformed CSV rows: {f['path']}: {bad}")
                    csv_shapes.append({"source_id": sid, "path": f["path"], "rows": len(rows),
                                       "columns": reader.fieldnames,
                                       "review_state": "STRUCTURE_CHECKED_VALUES_NOT_REVALIDATED",
                                       "origin_scope": source["evidence_kind"]})
        if source.get("repository_files") and r["status"] == "ACQUIRED":
            fixtures += 1
            manifests = [f for f in r["files"] if f.get("original_path", "").endswith(".json")]
            for f in manifests:
                data = json.loads((ROOT / f["path"]).read_text())
                if data.get("data", {}).get("format") == "inline":
                    observations.append({"source_id": sid,
                                         "inline_rows": len(data["data"].get("rows", [])),
                                         "origin_scope": source["evidence_kind"]})
                if "anchors" in data:
                    observations.append({"source_id": sid, "rounded_anchors": len(data["anchors"]),
                                         "uncertainty_note": "Project comparison tolerances are not measured experimental uncertainties."})
                if data.get("source", {}).get("sha256") and data["source"].get("local_path"):
                    original = REPO / data["source"]["local_path"]
                    if original.is_file() and sha256(original) != data["source"]["sha256"]:
                        errors.append(f"Existing fixture source hash mismatch: {sid}")
        text_file = ROOT / "derived" / sid / "source.txt"
        locators = []
        if text_file.exists():
            for page, text in enumerate(text_file.read_text(errors="replace").split("\f"), 1):
                for line in text.splitlines():
                    line = line.strip()
                    if line.upper().startswith(("TABLE ", "TABLEI", "TABLEII", "TABLEIV")):
                        locators.append({"pdf_page_1_based": page, "unreviewed_text": line[:180]})
        queue.append({"source_id": sid, "acquisition_state": r["status"],
                      "source_review_state": "NOT_REVIEWED_FOR_TRAINING",
                      "locator_candidates": locators[:20],
                      "locator_status": "TEXT_SEARCH_HINTS_NOT_VERIFIED_OR_COMPLETE",
                      "next_action": "Review one bounded original-page case, definitions, uncertainty and conditions before numeric extraction.",
                      "training_eligible": False})
    current_registry = runpy.run_path(str(REPO / "src/lrekit/verification/registry.py"))["_MODELS"]
    current_ids = {m.model_id for m in current_registry}
    mapped_ids = [m["model_id"] for m in crosswalk["models"]]
    if set(mapped_ids) != current_ids or len(mapped_ids) != len(current_ids):
        errors.append("Crosswalk does not cover the current registry exactly once")
    if crosswalk["registry_sha256"] != sha256(REPO / "src/lrekit/verification/registry.py"):
        errors.append("Registry snapshot hash is stale; review and refresh the crosswalk")
    for m in crosswalk["models"]:
        if set(m["candidate_source_ids"]) - set(source_ids):
            errors.append(f"Unknown candidate source in crosswalk: {m['model_id']}")
        if m.get("training_eligible") is not False or m.get("hardware_qualified") is not False:
            errors.append(f"Unexpected model eligibility/qualification claim: {m['model_id']}")
    plan = json.loads((ROOT / "catalog/workspace.json").read_text())["primary_plan"]
    plan_path = REPO / plan["path"]
    if not plan_path.is_file() or sha256(plan_path) != plan["sha256"]:
        errors.append("Governing plan bytes are missing or changed since the recorded snapshot")
    result = {"status": "PASS" if not errors else "FAIL",
              "scope": "File integrity, fixture structure, source references and registry coverage only; no scientific validation or training eligibility.",
              "catalog_sources": len(sources), "source_status_counts": dict(statuses),
              "evidence_kind_counts": dict(evidence_kinds), "verified_files": file_count,
              "source_payload_bytes_excluding_derivative_text": payload_bytes,
              "pdf_count": pdf_count, "newly_downloaded_pdfs": new_downloads,
              "copied_local_pdfs": local_copies, "imported_fixture_bundles": fixtures,
              "downloaded_third_party_paper_copies": mirror_copies,
              "crosswalk_models": len(mapped_ids),
              "crosswalk_subsystems": len({m.subsystem for m in current_registry}),
              "csv_shapes": csv_shapes, "inline_fixture_counts": observations,
              "raw_instrument_exports_acquired": 0, "training_eligible_sources": 0,
              "duplicate_byte_groups": [v for v in hashes.values() if len(v) > 1],
              "acquisition_gaps": gaps, "errors": errors}
    save_json(ROOT / "catalog/status.json", result)
    save_json(ROOT / "catalog/extraction_queue.json", {"schema_version": 1, "queue": queue})
    lines = ["# Research-source acquisition status", "", f"Collection checks: **{result['status']}**.", "",
             "These checks establish file identity and coverage, not source correctness or physical validation.", "",
             f"- {new_downloads} downloaded NASA report PDFs; {local_copies} local-library PDF copies.",
             f"- {mirror_copies} original-paper PDF copy/copies downloaded from a third-party host; origin and unresolved reuse rights retained.",
             f"- {fixtures} imported numeric-fixture bundles; no new raw instrument exports acquired.",
             f"- {len(mapped_ids)} registered models across {result['crosswalk_subsystems']} subsystems mapped.",
             "- Zero sources marked eligible for training; original-page review and case-level extraction remain outstanding.", "",
             "| Source | Acquisition | Evidence kind | Files |", "|---|---|---|---|"]
    for s in sources:
        p = ROOT / "catalog/receipts" / (s["id"] + ".json")
        if p.exists():
            r = json.loads(p.read_text())
            pdfs = [f for f in r["files"] if f["path"].endswith(".pdf")]
            link = f"[PDF](../{pdfs[0]['path']})" if pdfs else f"[receipt](receipts/{s['id']}.json)"
            lines.append(f"| {s['id']} | {r['status']} | {s['evidence_kind']} | {link} |")
    lines += ["", "## Numeric fixtures", ""]
    for shape in csv_shapes:
        lines.append(f"- `{shape['source_id']}`: {shape['rows']} CSV rows. Mixed-origin columns retain their fixture definitions.")
    for item in observations:
        lines.append(f"- `{item['source_id']}`: {item.get('inline_rows', item.get('rounded_anchors'))} "
                     + ("inline case(s)." if "inline_rows" in item else "rounded anchor(s); project tolerances are not measurement uncertainty."))
    lines += ["", "## Outstanding work", "",
              "Review original source pages and determine whether useful raw/supplementary data exist. "
              "Digitize/transcribe reports only with observation definitions, uncertainty and source locators. "
              "A download is not a reviewed training dataset. See the parent README and extraction queue.", ""]
    for g in gaps:
        lines.append(f"- `{g['source_id']}`: {g['status']}. {g['next_action']}")
    if errors:
        lines += ["", "## Check failures", ""] + ["- " + e for e in errors]
    (ROOT / "catalog/STATUS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "catalog_sources", "source_status_counts", "verified_files",
                                            "newly_downloaded_pdfs", "copied_local_pdfs", "imported_fixture_bundles",
                                            "crosswalk_models", "crosswalk_subsystems", "errors"]}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
