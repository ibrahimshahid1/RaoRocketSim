"""Faithful PDF -> agent-readable Markdown conversion for the LREKit propulsion corpus.

Design contract
---------------
* The source PDF stays authoritative.  This module produces a *derivative*
  Markdown rendering and records the ``source PDF -> generated Markdown``
  relationship, the source checksum, the converter identity, and honest
  extraction diagnostics.
* Nothing is invented.  When a page yields no reliable text the page marker is
  still emitted and the failure is recorded; no plausible text is synthesised.
* Output format matches the pre-existing ``propulsion_texts_for_agents`` corpus
  (``<!-- PDF page N -->`` markers, optional ``> [!WARNING]`` callout header)
  so old and new conversions are consumed identically.

Statuses recorded per document: ``good``, ``needs_review``,
``partial_conversion``, ``pdf_conversion_failed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CONVERTER_NAME = "lrekit-kg-pdf2md"
CONVERTER_VERSION = "1.0.1"

# Characters that signal displayed mathematics in raw PDF text.
_EQ_CHARS = "=≈≅≡≤≥∂∫∑∏√∞±∓·×÷αβγδεζηθκλμνξπρστφχψωΓΔΘΛΞΠΣΦΨΩ∇⋅"
_EQ_NUMBER_RE = re.compile(r"\(\s*\d{1,3}\s*[-.–]?\s*\d{0,3}\s*\)\s*$", re.MULTILINE)
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:\*\*|_|#+\s*)*\s*TABLE\s+[0-9IVXA-Z]", re.IGNORECASE | re.MULTILINE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:\*\*|_|#+\s*)*\s*(?:FIG(?:URE)?\.?)\s+[0-9IVXA-Z]", re.IGNORECASE | re.MULTILINE)
_REFERENCES_RE = re.compile(r"^\s*(?:\*\*|_|#+\s*)*\s*(REFERENCES|BIBLIOGRAPHY)\b", re.IGNORECASE | re.MULTILINE)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*(?:\*\*|_|#+\s*)*\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*(?:\*\*|_)*\s*$")
_REPLACEMENT_RE = re.compile(r"[�]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _equation_signals(text: str) -> int:
    """Count lines that look like displayed mathematics.

    This is a *signal* count used only to compare source against output, never
    an assertion that an equation was correctly reconstructed.
    """
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        math_chars = sum(1 for ch in stripped if ch in _EQ_CHARS)
        if math_chars >= 2:
            count += 1
        elif math_chars >= 1 and len(stripped) <= 80:
            count += 1
    count += len(_EQ_NUMBER_RE.findall(text))
    return count


@dataclass
class PageResult:
    number: int
    markdown: str
    native_chars: int
    ocr_used: bool = False
    error: str = ""


@dataclass
class ConversionResult:
    source_pdf: str
    output_markdown: str
    sha256: str
    source_bytes: int
    page_count: int
    method: str
    converter: str = CONVERTER_NAME
    converter_version: str = CONVERTER_VERSION
    converted_at: str = ""
    scanned: bool = False
    low_confidence: bool = False
    corrupted: bool = False
    native_chars: int = 0
    markdown_chars: int = 0
    extraction_status: str = "needs_review"
    issues: list = field(default_factory=list)
    equation_signals: int = 0
    source_equation_signals: int = 0
    markdown_tables: int = 0
    parser_tables: int = 0
    source_table_captions: int = 0
    figure_captions: int = 0
    source_figure_captions: int = 0
    pdf_title: str = ""
    pdf_author: str = ""
    pdf_subject: str = ""
    pdf_keywords: str = ""
    pdf_creation_date: str = ""
    source_identifier: str = ""
    duplicate_of: str = ""
    extractor_detail: str = ""
    ocr_pages: int = 0
    failed_pages: int = 0
    source_kind: str = "file"
    source_container: str = ""
    archive_member: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _detect_repeated_furniture(pages: list[PageResult], min_share: float = 0.5) -> set[str]:
    """Find short lines that repeat on most pages (running heads/feet)."""
    if len(pages) < 6:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.markdown.splitlines()]
        edge = [ln for ln in (lines[:3] + lines[-3:]) if ln]
        for ln in set(edge):
            if 0 < len(ln) <= 120:
                counts[ln] += 1
    threshold = max(3, int(len(pages) * min_share))
    return {ln for ln, c in counts.items() if c >= threshold}


def _strip_page_furniture(page_md: str, furniture: set[str]) -> str:
    lines = page_md.splitlines()
    keep: list[str] = []
    n = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        near_edge = idx < 3 or idx >= n - 3
        if near_edge and stripped:
            if stripped in furniture:
                continue
            if _PAGE_NUMBER_ONLY_RE.match(stripped):
                continue
        keep.append(line)
    return "\n".join(keep)


def _extract_source_identifier(meta: dict, raw_text: str) -> str:
    subject = (meta.get("subject") or "") + " " + (meta.get("keywords") or "")
    doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", subject)
    if doi:
        return doi.group(0).rstrip(".,;")
    head = raw_text[:6000]
    doi = re.search(r"\b(?:doi:?\s*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", head, re.IGNORECASE)
    if doi:
        return doi.group(1).rstrip(".,;")
    nasa = re.search(r"\bNASA\s+(?:SP|CR|TM|TN|TP|RP)[- ]?[0-9\-]+", head)
    if nasa:
        return re.sub(r"\s+", " ", nasa.group(0))
    return ""


def convert_pdf(pdf_path: Path, out_path: Path, *, allow_ocr: bool = True,
                ocr_char_threshold: int = 60) -> ConversionResult:
    import pymupdf
    import pymupdf4llm

    stat = pdf_path.stat()
    result = ConversionResult(
        source_pdf=str(pdf_path),
        output_markdown=str(out_path),
        sha256=sha256_file(pdf_path),
        source_bytes=stat.st_size,
        page_count=0,
        method="",
        converted_at=datetime.now(timezone.utc).isoformat(),
        source_container=pdf_path.name,
    )

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # pragma: no cover - depends on corpus
        result.corrupted = True
        result.extraction_status = "pdf_conversion_failed"
        result.issues = [f"PDF could not be opened: {exc!r}"]
        result.method = "none"
        return result

    result.page_count = doc.page_count
    meta = doc.metadata or {}
    result.pdf_title = (meta.get("title") or "").strip()
    result.pdf_author = (meta.get("author") or "").strip()
    result.pdf_subject = (meta.get("subject") or "").strip()
    result.pdf_keywords = (meta.get("keywords") or "").strip()
    result.pdf_creation_date = (meta.get("creationDate") or "").strip()

    # ---- raw source text (for diagnostics that must reference the source) ----
    raw_parts: list[str] = []
    per_page_native: list[int] = []
    for pno in range(doc.page_count):
        try:
            txt = doc[pno].get_text("text")
        except Exception:
            txt = ""
        raw_parts.append(txt)
        per_page_native.append(len(txt.strip()))
    raw_text = "\n".join(raw_parts)
    result.native_chars = len(raw_text.strip())
    result.source_equation_signals = _equation_signals(raw_text)
    result.source_table_captions = len(_TABLE_CAPTION_RE.findall(raw_text))
    result.source_figure_captions = len(_FIGURE_CAPTION_RE.findall(raw_text))
    source_has_references = bool(_REFERENCES_RE.search(raw_text))
    result.source_identifier = _extract_source_identifier(meta, raw_text)

    sparse_pages = sum(1 for c in per_page_native if c < ocr_char_threshold)
    scanned = doc.page_count > 0 and sparse_pages / doc.page_count >= 0.5
    result.scanned = scanned

    # ---- markdown extraction ----
    issues: list[str] = []
    pages: list[PageResult] = []
    parser_tables = 0
    method = "pymupdf4llm-classic"
    pymupdf4llm.use_layout(False)  # match the pre-existing corpus extraction mode
    try:
        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, write_images=False,
                                         show_progress=False)
        for chunk in chunks:
            md = chunk.get("text", "") or ""
            pno = int(chunk.get("metadata", {}).get("page", len(pages) + 1))
            parser_tables += len(chunk.get("tables", []) or [])
            pages.append(PageResult(number=pno, markdown=md,
                                    native_chars=per_page_native[pno - 1] if 0 < pno <= len(per_page_native) else 0))
    except Exception as exc:
        issues.append(f"pymupdf4llm markdown extraction failed ({exc!r})")
        method = "pymupdf-layout-fallback"
        pages = [PageResult(number=i + 1, markdown="", native_chars=per_page_native[i])
                 for i in range(doc.page_count)]

    # A parser can omit an image-only page entirely.  Keep every source page in
    # the conversion so it can receive OCR and retains its marker if unreadable.
    present_pages = {page.number for page in pages}
    missing_pages = [i + 1 for i in range(doc.page_count) if i + 1 not in present_pages]
    if missing_pages:
        pages.extend(PageResult(number=pno, markdown="", native_chars=per_page_native[pno - 1])
                     for pno in missing_pages)
        pages.sort(key=lambda page: page.number)
        issues.append(f"Markdown parser omitted {len(missing_pages)} page(s); source page markers were restored")

    # The Markdown parser silently returns nothing for some scanned-over-text and
    # older two-column scans.  Where the page demonstrably HAS native text, rebuild
    # it from span geometry rather than emitting a blank page.
    from text_layout import page_to_markdown  # local helper, same directory

    rebuilt = 0
    for page in pages:
        native = page.native_chars
        if native >= ocr_char_threshold and len(page.markdown.strip()) < max(40, 0.25 * native):
            try:
                md = page_to_markdown(doc[page.number - 1])
            except Exception as exc:
                page.error = f"layout fallback failed: {exc!r}"
                continue
            if len(md.strip()) > len(page.markdown.strip()):
                page.markdown = md
                rebuilt += 1
    if rebuilt:
        method += "+layout-rebuild"
        issues.append(
            f"{rebuilt} page(s) were rebuilt from PDF span geometry because the Markdown parser "
            "returned no text; table and equation layout on those pages is not reconstructed"
        )

    # ---- OCR only for pages the native extractor could not read ----
    # Eligibility is page-local: a scanned appendix needs OCR even if most of
    # the document has native text.  `scanned` remains a document diagnostic.
    ocr_pages = 0
    failed_pages = 0
    if allow_ocr and sparse_pages:
        tessdata = os.environ.get("TESSDATA_PREFIX") or "/opt/homebrew/share/tessdata"
        os.environ.setdefault("TESSDATA_PREFIX", tessdata)
        for page in pages:
            if page.native_chars >= ocr_char_threshold:
                continue
            try:
                tp = doc[page.number - 1].get_textpage_ocr(flags=0, dpi=220, full=True)
                text = doc[page.number - 1].get_text("text", textpage=tp)
                if len(text.strip()) > max(page.native_chars, 20):
                    page.markdown = text
                    page.ocr_used = True
                    ocr_pages += 1
            except Exception as exc:
                page.error = f"OCR failed: {exc!r}"
                failed_pages += 1
        if ocr_pages:
            method += "+tesseract-ocr"
            issues.append(
                f"Sparse source pages: {ocr_pages} page(s) reconstructed by OCR; symbols, equations, "
                "subscripts, and reading order require verification against the PDF"
            )
        if failed_pages:
            issues.append(f"OCR failed on {failed_pages} page(s); missing text was not reconstructed")

    result.ocr_pages = ocr_pages
    result.failed_pages = failed_pages
    result.method = method
    result.parser_tables = parser_tables

    furniture = _detect_repeated_furniture(pages)
    body: list[str] = []
    empty_pages = 0
    for page in pages:
        cleaned = _strip_page_furniture(page.markdown, furniture).strip()
        if not cleaned:
            empty_pages += 1
        body.append(f"<!-- PDF page {page.number} -->\n\n{cleaned}\n")
    markdown_body = "\n".join(body).rstrip() + "\n"

    result.equation_signals = _equation_signals(markdown_body)
    result.markdown_tables = markdown_body.count("\n|")
    result.figure_captions = len(_FIGURE_CAPTION_RE.findall(markdown_body))
    result.markdown_chars = len(markdown_body)

    # ---- honest diagnostics ----
    if furniture:
        sample = "; ".join(sorted(furniture)[:3])
        issues.append(f"Repeated page-edge text was removed: {sample}")
    if empty_pages:
        issues.append(f"{empty_pages} page(s) produced no extractable text and are represented by an empty page marker")
    if result.source_equation_signals and result.equation_signals < 0.6 * result.source_equation_signals:
        issues.append(
            f"Possible missing/broken equations ({result.equation_signals} output signals vs "
            f"{result.source_equation_signals} source signals)"
        )
    if result.source_table_captions and result.parser_tables == 0 and result.markdown_tables == 0:
        issues.append("Table captions exist in the source but no table structure was reconstructed")
    if result.source_figure_captions and result.figure_captions < 0.6 * result.source_figure_captions:
        issues.append(
            f"Possible missing figure captions ({result.figure_captions} output vs "
            f"{result.source_figure_captions} source)"
        )
    if source_has_references and not _REFERENCES_RE.search(markdown_body):
        issues.append("References heading detected in source but not preserved as a Markdown heading")
    if _REPLACEMENT_RE.search(markdown_body):
        issues.append("Invalid/replacement characters may affect symbols or units")
    if result.native_chars and result.markdown_chars < 0.4 * result.native_chars and not scanned:
        issues.append("Markdown is substantially shorter than the native PDF text; reading order may be incomplete")
    heading_count = len(re.findall(r"^#{1,6}\s", markdown_body, re.MULTILINE))
    if heading_count < 2 and doc.page_count > 4:
        issues.append("Section hierarchy may be incomplete (fewer than two Markdown headings)")

    low_conf = bool(
        scanned
        or result.ocr_pages
        or result.failed_pages
        or empty_pages > 0.2 * max(doc.page_count, 1)
        or (result.source_equation_signals and result.equation_signals < 0.4 * result.source_equation_signals)
    )
    result.low_confidence = low_conf

    if result.failed_pages and result.failed_pages >= 0.5 * max(doc.page_count, 1):
        status = "partial_conversion"
    elif result.markdown_chars < 200 and doc.page_count > 0:
        status = "partial_conversion"
        issues.append("Almost no text was recovered from this source")
    elif issues:
        status = "needs_review"
    else:
        status = "good"
    result.extraction_status = status
    result.issues = issues

    header = ""
    if low_conf or status in {"partial_conversion", "pdf_conversion_failed"}:
        header = (
            "> [!WARNING]\n"
            "> Extraction confidence is low; compare critical passages, equations, tables, "
            "and numerical values with the source PDF.\n\n"
        )
    provenance = (
        "<!-- lrekit-kg-conversion\n"
        f"source_pdf: {pdf_path}\n"
        f"source_sha256: {result.sha256}\n"
        f"converter: {CONVERTER_NAME} {CONVERTER_VERSION} ({method})\n"
        f"converted_at: {result.converted_at}\n"
        f"extraction_status: {status}\n"
        "-->\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + provenance + markdown_body, encoding="utf-8")
    doc.close()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert one PDF to agent-readable Markdown.")
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--no-ocr", action="store_true")
    ns = ap.parse_args(argv)
    try:
        res = convert_pdf(Path(ns.pdf), Path(ns.out), allow_ocr=not ns.no_ocr)
    except Exception:
        print(json.dumps({"source_pdf": ns.pdf, "extraction_status": "pdf_conversion_failed",
                          "traceback": traceback.format_exc()}))
        return 1
    print(json.dumps(res.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
