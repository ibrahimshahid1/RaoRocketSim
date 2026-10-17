"""Checkpointed native-text sampling; isolates malformed PDFs with timeouts.

Run with the PDF environment, e.g. .venv-kg/bin/python knowledge/tools/triage_pdfs.py.
This is a conversion triage sample, not a complete OCR census or accuracy score.
No OCR, model inference, corpus edits, or network requests are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METHOD = "native-first-middle-last@1"


def sample(path: Path) -> dict:
    import pymupdf
    with pymupdf.open(path) as pdf:
        pages = []
        for n in sorted({0, len(pdf) // 2, len(pdf) - 1}) if len(pdf) else []:
            text = pdf[n].get_text("text").strip()
            pages.append({"page": n + 1, "native_chars": len(text),
                          "replacement_chars": text.count("\ufffd"), "text_sparse": len(text) < 60})
        return {"page_count": len(pdf), "sampled_pages": pages}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_report(output: Path, rows: list[dict], documents: list[dict]) -> None:
    summary = {
        "canonical_documents": len(documents), "documents_assessed": len(rows),
        "canonical_pages_in_manifest": sum(d.get("page_count") or 0 for d in documents),
        "pages_sampled": sum(len(r.get("sampled_pages", [])) for r in rows),
        "documents_with_sampled_sparse_pages": sum(any(p["text_sparse"] for p in r.get("sampled_pages", [])) for r in rows),
        "sampled_sparse_pages": sum(p["text_sparse"] for r in rows for p in r.get("sampled_pages", [])),
        "documents_with_sampled_replacement_chars": sum(any(p["replacement_chars"] for p in r.get("sampled_pages", [])) for r in rows),
        "errors_or_timeouts": sum(bool(r.get("error")) for r in rows),
    }
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "method": METHOD,
              "limitations": "First/middle/last native text only. Sparse pages can be blank covers/figures; damaged embedded OCR can have abundant text. This is not an OCR-needed census or parser benchmark.",
              "summary": summary, "records": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    temporary.replace(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge/reports/pdf_triage.json")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds per PDF; one isolated worker at a time")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        try:
            result = sample(args.worker)
        except Exception as exc:
            result = {"sampled_pages": [], "error": type(exc).__name__}
        print(json.dumps(result))
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    manifest = json.loads((ROOT / "knowledge/manifest.json").read_text())
    docs = [d for d in manifest["records"] if not d.get("alternate_copy_of")]
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    cache = {r["doc_id"]: r for r in previous.get("records", [])} if previous.get("method") == METHOD else {}
    rows = []
    for d in docs:
        path = ROOT / d["source_pdf"]
        row = {"doc_id": d["id"], "source_pdf": d["source_pdf"], "sampled_pages": [],
               "conversion_status": d["conversion"]["status"], "timeout_seconds": args.timeout}
        try:
            sha = digest(path)
            row["source_sha256"] = sha
            old = cache.get(d["id"], {})
            if old.get("source_sha256") == sha and not (args.retry_errors and old.get("error")):
                row = {**old, "conversion_status": row["conversion_status"]}
            else:
                worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                                        capture_output=True, text=True, timeout=args.timeout)
                if worker.returncode:
                    row["error"] = "WORKER_FAILED"
                else:
                    row.update(json.loads(worker.stdout.strip().splitlines()[-1]))
        except subprocess.TimeoutExpired:
            row["error"] = "PDF_SAMPLE_TIMEOUT"
        except (OSError, ValueError, IndexError) as exc:
            row["error"] = type(exc).__name__
        rows.append(row)
        write_report(args.output, rows, docs)
        if row.get("error") or len(rows) % 25 == 0 or len(rows) == len(docs):
            print(f"[{len(rows)}/{len(docs)}] {d['id']}: {row.get('error', 'sampled')}", flush=True)
    print(json.dumps(json.loads(args.output.read_text())["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
