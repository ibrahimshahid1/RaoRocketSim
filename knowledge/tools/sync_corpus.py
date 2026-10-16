"""Synchronise ``propulsion_texts/*.pdf`` with ``propulsion_texts_for_agents/markdown``.

Idempotent.  Run it repeatedly: unchanged sources are not reconverted, and a
run with nothing to do rewrites nothing.

Mapping evidence, strongest first:
1. source PDF sha256 recorded in the conversion manifest,
2. explicit ``source_pdf`` path recorded in the manifest,
3. the generated Markdown's own ``lrekit-kg-conversion`` provenance block,
4. filename, used only as a last resort and reported as weak evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdf_to_markdown import CONVERTER_NAME, CONVERTER_VERSION, convert_pdf, sha256_file  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "propulsion_texts"
AGENTS = CORPUS / "propulsion_texts_for_agents"
MARKDOWN = AGENTS / "markdown"
MANIFEST = AGENTS / "_conversion_manifest.json"
VALIDATION = AGENTS / "_validation_report.json"

_PROV_RE = re.compile(r"<!-- lrekit-kg-conversion\n(.*?)-->", re.DOTALL)


def load_manifest() -> list[dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return []


def discover_pdfs() -> list[Path]:
    return sorted(
        p for p in CORPUS.rglob("*.pdf")
        if AGENTS.name not in p.relative_to(CORPUS).parts
    )


def output_path_for(pdf: Path) -> Path:
    rel = pdf.relative_to(CORPUS)
    return MARKDOWN / rel.with_suffix(".md")


def scan(verbose: bool = False) -> dict:
    """Return the current PDF <-> Markdown synchronisation state."""
    manifest = load_manifest()
    by_sha: dict[str, list[dict]] = {}
    for row in manifest:
        by_sha.setdefault(row.get("sha256", ""), []).append(row)

    pdfs = discover_pdfs()
    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdf_count": len(pdfs),
        "manifest_rows": len(manifest),
        "synchronized": [],
        "missing_markdown": [],
        "stale_markdown": [],
        "orphan_markdown": [],
        "mirrored_reference_notes": [],
        "weak_mapping": [],
        "duplicate_payloads": {},
    }
    seen_outputs: set[str] = set()
    sha_to_pdfs: dict[str, list[str]] = {}

    for pdf in pdfs:
        rel = str(pdf.relative_to(REPO))
        sha = sha256_file(pdf)
        sha_to_pdfs.setdefault(sha, []).append(rel)
        rows = by_sha.get(sha, [])
        out = output_path_for(pdf)
        if rows:
            # Manifest knows this payload.  Confirm the file it names exists.
            usable = [r for r in rows if r.get("output_markdown")
                      and (AGENTS / r["output_markdown"]).is_file()
                      and r.get("extraction_status") != "pdf_conversion_failed"]
            target = AGENTS / usable[0]["output_markdown"] if usable else None
            if target is not None:
                m = _PROV_RE.search(target.read_text(encoding="utf-8", errors="replace")[:4000])
                if m and f"source_sha256: {sha}" not in m.group(1):
                    state["stale_markdown"].append({"pdf": rel, "markdown": str(target.relative_to(REPO)),
                                                    "sha256": sha, "reason": "Markdown provenance differs from manifest"})
                    continue
                seen_outputs.add(str(target.resolve()))
                state["synchronized"].append({"pdf": rel, "markdown": str(target.relative_to(REPO)),
                                              "sha256": sha, "evidence": "manifest_sha256"})
            else:
                state["missing_markdown"].append({"pdf": rel, "sha256": sha,
                                                  "reason": "manifest output is absent or conversion failed"})
            continue
        if out.exists():
            text = out.read_text(encoding="utf-8", errors="replace")[:4000]
            m = _PROV_RE.search(text)
            if m and f"source_sha256: {sha}" in m.group(1):
                # The derivative is current but the manifest never recorded it
                # (e.g. a conversion run that was killed).  Re-register it.
                state["missing_markdown"].append({"pdf": rel, "sha256": sha,
                                                  "reason": "generated Markdown carries matching provenance but no manifest row; will re-register"})
            elif m:
                state["stale_markdown"].append({"pdf": rel, "markdown": str(out.relative_to(REPO)),
                                                "sha256": sha,
                                                "reason": "generated Markdown records a different source checksum"})
            else:
                # A Markdown file exists under the mirrored name but nothing ties it to
                # THIS payload (no manifest row, no provenance checksum).  The PDF was
                # probably replaced.  Treat the derivative as stale and regenerate it.
                state["stale_markdown"].append({"pdf": rel, "markdown": str(out.relative_to(REPO)),
                                                "sha256": sha,
                                                "reason": "existing Markdown has no checksum/manifest association with this PDF payload (filename match only); regenerating"})
            continue
        state["missing_markdown"].append({"pdf": rel, "sha256": sha, "reason": "no generated Markdown found"})

    state["duplicate_payloads"] = {s: v for s, v in sha_to_pdfs.items() if len(v) > 1}

    by_output = {str((AGENTS / r.get("output_markdown", "")).resolve()): r for r in manifest}
    for md in sorted(MARKDOWN.rglob("*.md")):
        if str(md.resolve()) in seen_outputs:
            continue
        note_source = CORPUS / md.relative_to(MARKDOWN)
        if md.name == "PINTLE_LITERATURE_CATALOG.md" and note_source.is_file() and note_source.read_bytes() == md.read_bytes():
            state["mirrored_reference_notes"].append({"markdown": str(md.relative_to(REPO)),
                "source_note": str(note_source.relative_to(REPO)), "sha256": sha256_file(note_source),
                "authority": "human-authored bibliography; not a PDF-derived scientific assertion"})
            continue
        row = by_output.get(str(md.resolve()))
        if row and row.get("source_kind") and row.get("source_kind") != "file":
            container = CORPUS / row.get("source_container", "")
            if not container.is_file():
                state["stale_markdown"].append({"pdf": f"{row.get('source_container')}::{row.get('archive_member')}",
                                                "markdown": str(md.relative_to(REPO)),
                                                "reason": "source archive is unavailable; source bytes cannot be verified"})
                continue
            try:
                actual = hashlib.sha256()
                with zipfile.ZipFile(container) as archive, archive.open(row.get("archive_member", "")) as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        actual.update(block)
                if actual.hexdigest() != row.get("sha256"):
                    raise ValueError("archive member checksum differs from conversion manifest")
            except (OSError, KeyError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
                state["stale_markdown"].append({"pdf": f"{row.get('source_container')}::{row.get('archive_member')}",
                                                "markdown": str(md.relative_to(REPO)),
                                                "reason": f"source archive member cannot be verified: {exc}"})
                continue
            state["synchronized"].append({"pdf": f"{row.get('source_container')}::{row.get('archive_member')}",
                                          "markdown": str(md.relative_to(REPO)), "sha256": row.get("sha256"),
                                          "evidence": "archive_member_sha256"})
            continue
        rel_md = md.relative_to(MARKDOWN)
        candidate = CORPUS / rel_md.with_suffix(".pdf")
        if not candidate.exists():
            state["orphan_markdown"].append({"markdown": str(md.relative_to(REPO)),
                                             "reason": ("manifest row exists but its source PDF is no longer on disk"
                                                        if row else "no source PDF at the mirrored path and no manifest row"),
                                             "manifest_row": bool(row)})
    return state


def _convert_in_subprocess(pdf: Path, out: Path, allow_ocr: bool, timeout_s: int) -> dict:
    """Run one conversion in a child interpreter so PyMuPDF memory is released
    between documents and a crash on one PDF cannot take down the batch."""
    import subprocess

    cmd = [sys.executable, str(Path(__file__).resolve().parent / "pdf_to_markdown.py"), str(pdf), str(out)]
    if not allow_ocr:
        cmd.append("--no-ocr")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"extraction_status": "pdf_conversion_failed",
                "issues": [f"Conversion exceeded {timeout_s}s and was stopped; no text was invented."]}
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        row = json.loads(line)
    except Exception:
        return {"extraction_status": "pdf_conversion_failed",
                "issues": ["Converter produced no parseable result"],
                "stderr_tail": proc.stderr[-2000:]}
    if proc.returncode != 0 and row.get("extraction_status") != "pdf_conversion_failed":
        row["extraction_status"] = "pdf_conversion_failed"
    return row


def _write_manifest(manifest: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(MANIFEST)


def convert_missing(state: dict, limit: int | None = None, allow_ocr: bool = True,
                    timeout_s: int = 1800) -> list[dict]:
    todo = state["missing_markdown"] + state["stale_markdown"]
    # smallest first so a memory-hungry giant cannot block the many small papers
    todo.sort(key=lambda it: (REPO / it["pdf"]).stat().st_size)
    if limit:
        todo = todo[:limit]
    manifest = load_manifest()
    new_rows: list[dict] = []
    for i, item in enumerate(todo, 1):
        pdf = REPO / item["pdf"]
        out = output_path_for(pdf)
        print(f"[{i}/{len(todo)}] {item['pdf']} ({pdf.stat().st_size/1e6:.1f} MB)", flush=True)
        row = _convert_in_subprocess(pdf, out, allow_ocr, timeout_s)
        row.setdefault("sha256", item["sha256"])
        row.setdefault("converter", CONVERTER_NAME)
        row.setdefault("converter_version", CONVERTER_VERSION)
        row.setdefault("converted_at", datetime.now(timezone.utc).isoformat())
        row.setdefault("issues", [])
        if row.get("extraction_status") == "pdf_conversion_failed":
            print(f"    FAILED: {row['issues'][:1]}", flush=True)
        # normalise paths for manifest storage
        row["source_pdf"] = str(pdf.relative_to(CORPUS))
        row["output_markdown"] = str(out.relative_to(AGENTS))
        row["source_container"] = pdf.name
        row["source_kind"] = "file"
        dup = next((r["source_pdf"] for r in manifest
                    if r.get("sha256") == row["sha256"] and r.get("source_pdf") != row["source_pdf"]), "")
        row["duplicate_of"] = dup
        # replace any earlier row for the same source path (re-registration / reconversion)
        manifest = [r for r in manifest if r.get("source_pdf") != row["source_pdf"]]
        manifest.append(row)
        _write_manifest(manifest)  # incremental: a killed run loses nothing
        new_rows.append(row)
        print(f"    -> {row.get('extraction_status')} ({row.get('method','?')}, "
              f"{row.get('page_count','?')}p, {row.get('markdown_chars',0)} chars)", flush=True)
    return new_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true", help="convert missing/stale sources")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--report", default=None, help="write the sync state to this JSON path")
    ap.add_argument("--timeout", type=int, default=1800, help="per-document seconds")
    ns = ap.parse_args(argv)

    state = scan()
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v)
                      for k, v in state.items() if k != "generated_at"}, indent=1))
    if ns.convert:
        rows = convert_missing(state, limit=ns.limit, allow_ocr=not ns.no_ocr, timeout_s=ns.timeout)
        print(f"converted {len(rows)} document(s)")
        state = scan()
    if ns.report:
        Path(ns.report).parent.mkdir(parents=True, exist_ok=True)
        Path(ns.report).write_text(json.dumps(state, indent=1))
    return 1 if state["missing_markdown"] or state["stale_markdown"] else 0


if __name__ == "__main__":
    sys.exit(main())
