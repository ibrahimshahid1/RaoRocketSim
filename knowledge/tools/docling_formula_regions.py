"""Re-enrich visually located PDF formula regions using Docling's local stage.

Use when automatic layout split a fraction or missed the equation. Region JSON
contains original page numbers and PDF-point BOTTOMLEFT boxes, never expected
formula text. The raw output remains a candidate requiring original-page review.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from kg.enrichment import integrate, sha256
from kg.locks import LockBusyError, heavy_job
from kg.store import index_by
from tools.docling_enrich import cached_attempt, model_revisions, publish_document


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("document_id")
    ap.add_argument("regions", type=Path)
    ap.add_argument("--reprocess", action="store_true", help="Bank a distinct new attempt; preserve prior raw/metadata and reviews")
    ns = ap.parse_args(argv)
    document = index_by("manifest")[ns.document_id]
    pdf_path = ROOT / document["source_pdf"]
    if sha256(pdf_path) != document["sha256"]:
        ap.error("source PDF changed; synchronize before enriching")
    regions = json.loads(ns.regions.read_text())
    if not isinstance(regions, list) or not regions:
        ap.error("provide at least one original-PDF formula region")
    try:
        # At most one heavyweight Docling inference process on this machine.
        with heavy_job(purpose=f"docling_formula_regions {ns.document_id} {ns.regions}"):
            return _regions(ap, ns, document, pdf_path, regions)
    except LockBusyError as exc:
        print(json.dumps({"state": "DOCLING_BUSY", "document_id": ns.document_id,
                          "holder": exc.holder, "recovery": "Retry after the running Docling job finishes."}))
        return 75


def _regions(ap, ns, document, pdf_path, regions):
    os.environ.setdefault("HF_HOME", str(ROOT / "knowledge/.work/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from docling_core.types.doc import DoclingDocument, DocItemLabel, ProvenanceItem
    from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
    from docling.datamodel.base_models import ItemAndImageEnrichmentElement
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel
    import pypdfium2

    scale = 3
    options = CodeFormulaVlmOptions.from_preset("codeformulav2", expansion_factor=0, extract_code=False)
    versions = {p: importlib.metadata.version(p) for p in (
        "docling", "docling-core", "docling-ibm-models", "docling-parse", "torch", "transformers")}
    settings = {"mode": "visually_located_formula_regions", "regions": regions,
                "region_locator_sha256": sha256(ns.regions), "formula_options": options.model_dump(mode="json"),
                "remote_services": False, "scale": scale, "accelerator": {"device": "cpu", "num_threads": 4}}
    pages = sorted({region["page"] for region in regions})
    descriptor = {"document_id": ns.document_id, "source_pdf": document["source_pdf"],
                  "source_sha256": document["sha256"], "page_range": [min(pages), max(pages)],
                  "options": settings, "versions": versions, "model_revisions": model_revisions()}
    dest = ROOT / "propulsion_texts/propulsion_texts_for_agents/docling" / ns.document_id
    dest.mkdir(parents=True, exist_ok=True)
    cached = cached_attempt(dest, descriptor, root=ROOT) if not ns.reprocess else None
    if cached:
        if cached["rejected"]:
            print(json.dumps({"state": "REJECTED_ARTIFACT_REPROCESS_REQUIRED", "document_id": ns.document_id,
                              "recovery": "Use --reprocess for a distinct attempt."}))
            return 2
        print(json.dumps(integrate(cached["raw"], cached["meta"])))
        return 0
    source = pypdfium2.PdfDocument(pdf_path)
    result = DoclingDocument(name=ns.document_id + " selected formula regions")
    batches = []
    for region in regions:
        n = region["page"]
        if not 1 <= n <= len(source):
            ap.error("region page outside PDF")
        page = source[n - 1]
        width, height = page.get_size()
        box = region["bbox"]
        if not (0 <= box["l"] < box["r"] <= width and 0 <= box["b"] < box["t"] <= height):
            ap.error("region box outside PDF")
        if n not in result.pages:
            result.add_page(n, Size(width=width, height=height))
        item = result.add_text(label=DocItemLabel.FORMULA, text="", orig="",
            prov=ProvenanceItem(page_no=n, bbox=BoundingBox(**box, coord_origin=CoordOrigin.BOTTOMLEFT), charspan=(0, 0)))
        image = page.render(scale=scale).to_pil().crop((box["l"] * scale, (height - box["t"]) * scale,
                                                       box["r"] * scale, (height - box["b"]) * scale))
        batches.append(ItemAndImageEnrichmentElement(item=item, image=image))
    started = time.monotonic()
    stage = CodeFormulaVlmModel(enabled=True, enable_remote_services=False, artifacts_path=None,
                                options=options, accelerator_options=AcceleratorOptions(device="cpu", num_threads=4))
    list(stage(result, batches))
    pages = sorted(result.pages)
    meta = {**descriptor, "observed_pages": pages, "model_revisions": model_revisions(),
            "model_revision_scope": "Observed local cache refs; not runtime model attestation.",
            "conversion_status": "success", "elapsed_seconds": time.monotonic() - started}
    raw, meta_path = publish_document(result, meta, dest=dest, root=ROOT)
    print(json.dumps(integrate(raw, meta_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
