"""Run one bounded local Docling formula/table enrichment and bank derivatives.

No cloud inference; remote services are disabled. Model-weight downloads may
require network on the first run. Existing reviewed derivatives are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from kg.enrichment import integrate, sha256, validate_metadata_binding
from kg.locks import LockBusyError, heavy_job
from kg.store import index_by, load


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_revisions():
    """Observe local cache references; this is not runtime model attestation."""
    hub = Path(os.environ["HF_HOME"]) / "hub"
    return {str(ref.relative_to(hub)): ref.read_text().strip()
            for ref in sorted(hub.glob("models--*/refs/*")) if ref.is_file()}


def input_identity(meta):
    return {key: meta[key] for key in ("document_id", "source_pdf", "source_sha256", "page_range",
                                       "options", "versions", "model_revisions")}


def cached_attempt(dest, descriptor, *, root=ROOT):
    """Locate a byte-verified matching attempt, retaining its QA state."""
    wanted = _digest(input_identity(descriptor))
    rows = load("enrichments")
    matches = []
    for path in dest.glob("*.meta.json"):
        try:
            meta = json.loads(path.read_text())
            binding = validate_metadata_binding(meta, path, root / meta["raw_artifact"])
            if _digest(input_identity(meta)) != wanted:
                continue
            raw = root / meta["raw_artifact"]
            if not raw.is_file() or sha256(raw) != meta["raw_sha256"]:
                continue
            if meta.get("conversion_status") not in ("success", "partial_success"):
                continue
            if meta.get("markdown_artifact"):
                md = root / meta["markdown_artifact"]
                if not md.is_file() or sha256(md) != meta.get("markdown_sha256"):
                    continue
            if binding["state"] == "VERIFIED":
                metadata_sha = sha256(path)
                if any(r.get("metadata_sha256") and r["metadata_sha256"] != metadata_sha
                       and r.get("raw_artifact") == meta["raw_artifact"]
                       and r.get("raw_sha256") == meta["raw_sha256"] for r in rows):
                    continue
            rejected = any(r.get("type") == "GeneratedDocument" and r.get("review_state") == "REJECTED"
                           and r.get("raw_artifact") == meta["raw_artifact"]
                           and r.get("raw_sha256") == meta["raw_sha256"] for r in rows)
            matches.append((meta.get("attempt_created_at", ""), str(path), raw, path, rejected))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # corrupt/legacy metadata cannot authorize cache reuse
    # Prefer a nonrejected attempt; historical rejected attempts remain on disk.
    good = [r for r in matches if not r[-1]]
    chosen = max(good or matches, default=None)
    return {"raw": chosen[2], "meta": chosen[3], "rejected": chosen[4]} if chosen else None


def _write_immutable(path, data):
    """Publish a complete file without replacing a previously reviewed artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)  # exclusive publication; no partial-reader window
        except FileExistsError:
            if path.read_bytes() != data:
                raise FileExistsError(f"Immutable Docling artifact differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def publish_attempt(raw_bytes, markdown_bytes, metadata, *, dest, root=ROOT):
    """Content-address source/configuration/model/output/attempt metadata together."""
    meta = dict(metadata)
    for key in ("raw_artifact", "markdown_artifact", "metadata_artifact", "artifact_identity_sha256"):
        meta.pop(key, None)
    meta.setdefault("attempt_id", uuid.uuid4().hex)
    meta.setdefault("attempt_created_at", datetime.now(timezone.utc).isoformat())
    meta.update(artifact_schema_version=2,
                raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
                input_identity_sha256=_digest(input_identity(meta)))
    address = _digest(meta)
    first, last = meta["page_range"]
    mode = "regions" if meta["options"].get("mode") == "visually_located_formula_regions" else f"p{first}-{last}"
    stem = f"{meta['source_sha256'][:12]}-{mode}-{address}"
    raw, md, meta_path = [dest / (stem + suffix) for suffix in (".json", ".md", ".meta.json")]
    meta.update(artifact_identity_sha256=address, raw_artifact=str(raw.relative_to(root)),
                markdown_artifact=str(md.relative_to(root)), metadata_artifact=str(meta_path.relative_to(root)))
    _write_immutable(raw, raw_bytes)
    _write_immutable(md, markdown_bytes)
    # Metadata is the completion marker; an interrupted export cannot be reused.
    _write_immutable(meta_path, (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode())
    return raw, meta_path


def publish_document(document, metadata, *, dest, root=ROOT):
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".export-") as directory:
        raw, md = Path(directory) / "raw.json", Path(directory) / "raw.md"
        document.save_as_json(raw)
        document.save_as_markdown(md)
        return publish_attempt(raw.read_bytes(), md.read_bytes(), metadata, dest=dest, root=root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("document_id")
    ap.add_argument("--pages", help="inclusive original PDF page range, e.g. 6:8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--reprocess", action="store_true", help="Run a new distinct immutable attempt, including for rejected pages")
    ap.add_argument("--device", choices=("cpu",), default="cpu",
                    help="CodeFormulaV2 Transformers engine supports CPU here; MPS is unavailable for this enrichment model")
    ns = ap.parse_args(argv)
    d = index_by("manifest").get(ns.document_id)
    if not d:
        ap.error("unknown document ID")
    pages = tuple(map(int, ns.pages.split(":"))) if ns.pages else (1, d["page_count"])
    if len(pages) != 2 or not 1 <= pages[0] <= pages[1] <= d["page_count"]:
        ap.error("page range is outside the source PDF")
    pdf = ROOT / d["source_pdf"]
    if sha256(pdf) != d["sha256"]:
        ap.error("source PDF changed; synchronize first")
    try:
        # At most one heavyweight Docling inference process on this machine.
        with heavy_job(purpose=f"docling_enrich {ns.document_id} pages {pages[0]}:{pages[1]}"):
            return _enrich(ns, d, pages, pdf)
    except LockBusyError as exc:
        print(json.dumps({"state": "DOCLING_BUSY", "document_id": ns.document_id, "pages": list(pages),
                          "holder": exc.holder, "recovery": "Retry after the running Docling job finishes."}))
        return BUSY_EXIT


BUSY_EXIT = 75  # EX_TEMPFAIL: nothing was attempted


def _enrich(ns, d, pages, pdf):
    # Keep all downloaded weights/cache inside a rebuildable local directory.
    os.environ.setdefault("HF_HOME", str(ROOT / "knowledge/.work/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(ns.threads))
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_formula_enrichment = True
    options.do_code_enrichment = False
    options.enable_remote_services = False
    options.allow_external_plugins = False
    options.do_table_structure = True
    options.ocr_options = TesseractCliOcrOptions(lang=["eng"])
    options.document_timeout = ns.timeout
    options.accelerator_options = AcceleratorOptions(num_threads=ns.threads, device=ns.device)
    # In pinned Docling2.139.0 the formula stage passes a literal2048 to its
    # engine, ignoring this model-spec field. Keep the recorded configuration
    # consistent with that source-inspected cap; worker timeouts bound jobs.
    options.code_formula_options.model_spec.max_new_tokens = 2048
    # Wide crops crossed Lee's narrow column gutter and hallucinated surrounding
    # prose. Retain the raw pilot; use a smaller margin for subsequent candidates.
    options.code_formula_options.expansion_factor = 0.02
    versions = {p: importlib.metadata.version(p) for p in (
        "docling", "docling-core", "docling-ibm-models", "docling-parse", "torch", "transformers")}
    serial_options = options.model_dump(mode="json")
    dest = ROOT / "propulsion_texts/propulsion_texts_for_agents/docling" / ns.document_id
    dest.mkdir(parents=True, exist_ok=True)
    descriptor = {"document_id": ns.document_id, "source_pdf": d["source_pdf"], "source_sha256": d["sha256"],
                  "page_range": list(pages), "options": serial_options, "versions": versions,
                  "model_revisions": model_revisions()}
    cached = cached_attempt(dest, descriptor, root=ROOT) if not ns.reprocess else None
    if cached:
        if cached["rejected"]:
            print(json.dumps({"state": "REJECTED_ARTIFACT_REPROCESS_REQUIRED", "document_id": ns.document_id,
                              "pages": list(pages), "recovery": "Use --reprocess for a distinct attempt."}))
            return 2
        print(json.dumps(integrate(cached["raw"], cached["meta"])))
        return 0
    started = time.monotonic()
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(pdf, page_range=pages)
    meta = {**descriptor, "model_revisions": model_revisions(),
            "model_revision_scope": "Observed local cache refs; not runtime model attestation.",
            "conversion_status": result.status.value,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "observed_pages": sorted(result.document.pages)}
    raw, meta_path = publish_document(result.document, meta, dest=dest, root=ROOT)
    print(json.dumps(integrate(raw, meta_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
