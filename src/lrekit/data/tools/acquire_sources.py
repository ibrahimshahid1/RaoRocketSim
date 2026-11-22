#!/usr/bin/env python3
"""Acquire research sources without importing or changing LREKit physics.

Catalog entries describe candidate uses, not a completed scientific review.
Receipts record acquisition and byte identity, never training eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def fetch(url, dest, *, pdf=False):
    req = urllib.request.Request(url, headers={"User-Agent": "LREKit-research-source-acquisition/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read()
        headers = {k: response.headers.get(k) for k in ["Content-Type", "ETag", "Last-Modified"]}
        final_url = response.geturl()
    if pdf and not payload.startswith(b"%PDF-"):
        raise ValueError("Response is not a PDF; refusing to save an error page as source.pdf")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return {"requested_url": url, "resolved_url": final_url, "headers": headers}


def file_record(path, origin, **extra):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest(path), "origin": origin, **extra}


def acquire(source):
    sid = source["id"]
    folder = ROOT / "raw" / sid
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "catalog/receipts" / (sid + ".json")
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if previous and previous.get("status") == "ACQUIRED":
        if all((ROOT / f["path"]).is_file() and digest(ROOT / f["path"]) == f["sha256"]
               for f in previous.get("files", [])) and previous.get("files"):
            print(sid, "already acquired (hashes match)", flush=True)
            return previous
        raise ValueError(f"{sid}: previous payload missing or changed; resolve explicitly before re-acquisition")
    receipt = {"source_id": sid, "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
               "evidence_kind": source["evidence_kind"], "status": "PENDING", "files": [],
               "review_state": "NOT_REVIEWED_FOR_TRAINING", "training_eligible": False,
               "hardware_qualified": False, "limitations": source.get("limitations", [])}
    if previous:
        receipt["previous_attempts"] = previous.get("previous_attempts", []) + [
            {"acquired_at_utc": previous.get("acquired_at_utc"), "status": previous.get("status"),
             "error": previous.get("error")}]
    try:
        if source.get("ntrs_id"):
            url = "https://ntrs.nasa.gov/api/citations/" + source["ntrs_id"]
            metadata_path = folder / "ntrs_metadata.json"
            transport = fetch(url, metadata_path)
            metadata = json.loads(metadata_path.read_text())
            if str(metadata.get("id")) != source["ntrs_id"]:
                raise ValueError("NTRS record identifier mismatch")
            receipt["files"].append(file_record(metadata_path, "publisher_metadata", **transport))
            receipt["publisher_title"] = metadata.get("title")
            receipt["publisher_copyright"] = metadata.get("copyright")
            receipt["publisher_distribution"] = metadata.get("distribution")
            downloads = [d for d in metadata.get("downloads", []) if d.get("links", {}).get("pdf")]
            if not downloads:
                receipt["publisher_fulltext_available"] = False
                if source.get("alternate_pdf_url"):
                    pdf_path = folder / "source.pdf"
                    transport = fetch(source["alternate_pdf_url"], pdf_path, pdf=True)
                    receipt["files"].append(file_record(pdf_path, "original_paper_on_third_party_host", **transport))
                    receipt["alternate_copy_identity_check"] = source.get("alternate_copy_identity_check")
                    receipt["alternate_copy_reuse_rights"] = "NOT_ESTABLISHED; local research copy only; no license inferred from hosting"
                    receipt["status"] = "ACQUIRED"
                else:
                    receipt["status"] = "METADATA_ONLY_NO_PUBLISHER_DOWNLOAD"
                    receipt["next_action"] = "Find an authorized original-paper copy; do not substitute a bibliography."
            else:
                download_url = urllib.parse.urljoin("https://ntrs.nasa.gov", downloads[0]["links"]["pdf"])
                pdf_path = folder / "source.pdf"
                transport = fetch(download_url, pdf_path, pdf=True)
                receipt["files"].append(file_record(pdf_path, "publisher_report_pdf", **transport))
                receipt["status"] = "ACQUIRED"
        elif source.get("local_path"):
            src = REPO / source["local_path"]
            if not src.is_file():
                raise FileNotFoundError(str(src))
            dest = folder / "source.pdf"
            if not src.read_bytes().startswith(b"%PDF-"):
                raise ValueError("Local source is not a PDF")
            shutil.copyfile(src, dest)
            receipt["files"].append(file_record(dest, "existing_local_literature_copy",
                                                original_path=source["local_path"]))
            receipt["status"] = "ACQUIRED"
        elif source.get("repository_files"):
            for relative in source["repository_files"]:
                src = REPO / relative
                dest = folder / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                receipt["files"].append(file_record(dest, "existing_repository_fixture", original_path=relative))
            receipt["status"] = "ACQUIRED"
            receipt["review_state"] = "EXISTING_FIXTURE_IMPORTED_NOT_REVALIDATED"
        else:
            receipt["status"] = "UNAVAILABLE"
            receipt["next_action"] = source.get("next_action", "Acquire an applicable measurement dataset.")
        for f in list(receipt["files"]):
            p = ROOT / f["path"]
            if p.suffix == ".pdf":
                result = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True)
                if result.returncode:
                    raise ValueError("PDF parser rejected downloaded source: " + result.stderr[:300])
                receipt["pdf_info"] = result.stdout
                text_path = ROOT / "derived" / sid / "source.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(["pdftotext", "-layout", str(p), str(text_path)], capture_output=True, text=True)
                receipt["text_extraction"] = {"status": "DERIVATIVE_TEXT_UNREVIEWED" if result.returncode == 0 else "FAILED",
                                               "warning_excerpt": result.stderr[:400]}
                if result.returncode == 0:
                    receipt["files"].append(file_record(text_path, "unreviewed_pdf_text_derivative"))
    except Exception as exc:
        receipt["status"] = "ACQUISITION_FAILED"
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
    write_json(receipt_path, receipt)
    print(sid, receipt["status"], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Catalog source IDs; omit to acquire all")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "catalog/sources.json").read_text())
    selected = set(args.only) if args.only is not None else None
    known = {s["id"] for s in catalog["sources"]}
    if selected is not None and selected - known:
        parser.error("Unknown source IDs: " + ", ".join(sorted(selected - known)))
    results = [acquire(s) for s in catalog["sources"] if selected is None or s["id"] in selected]
    if any(r["status"] == "ACQUISITION_FAILED" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
