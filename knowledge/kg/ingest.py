"""Document manifest construction and incremental chunk ingestion.

Pipeline (idempotent):
    conversion manifest (propulsion_texts_for_agents/_conversion_manifest.json)
      + paper_index.json (prior automated bibliographic pass)
      + PDF metadata recorded at conversion time
    -> knowledge/manifest.json           one Document record per unique PDF payload
    -> knowledge/vector_index/chunks.sqlite   chunks for every canonical, readable document
    -> ingestion_state.json              per-document state, honest failure states

Duplicate PDF payloads (identical sha256 under different filenames) become ONE
Document with ``alternate_paths``; they never get independent scientific
authority or duplicate chunks.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import AGENT_CORPUS_ROOT, CORPUS_ROOT, MARKDOWN_ROOT, REPO_ROOT
from .chunker import chunk_markdown, markdown_sha256
from .schema import EXTRACTION_VERSION
from .store import connect, load, replace_doc_chunks, save

NTRS_RE = re.compile(r"^((?:19|20)\d{9})(?:\s*\(\d+\))?\.pdf$", re.IGNORECASE)
DOI_FILE_RE = re.compile(r"^(10\.\d{4,9}[_@/][-._;()/:A-Za-z0-9]+?)(?:\s*\(\d+\))?\.pdf$", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
NASA_REPORT_RE = re.compile(r"\bNASA\s+(SP|CR|TM|TN|TP|RP)[- ]?([0-9][0-9\-]*)", re.IGNORECASE)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()


def derive_document_id(rel_pdf: str, source_identifier: str, sha: str) -> tuple[str, str]:
    """Return (document_id, id_basis)."""
    name = Path(rel_pdf).name
    m = NTRS_RE.match(name)
    if m:
        return f"DOC_NTRS_{m.group(1)}", "ntrs_accession_from_filename"
    if source_identifier and DOI_RE.match(source_identifier):
        return f"DOC_DOI_{_slug(source_identifier)}", "doi_from_pdf_metadata_or_text"
    m = DOI_FILE_RE.match(name)
    if m:
        return f"DOC_DOI_{_slug(m.group(1).replace('@', '/').replace('_', '/', 1))}", "doi_from_filename"
    # A NASA report number found in body text is usually a *citation*, not the
    # document's identity - never derive an id from it.
    stem = re.sub(r"\s*\(\d+\)$", "", Path(name).stem)
    return f"DOC_{_slug(stem)[:60]}", "filename"


def _topic_from_path(rel_pdf: str) -> list[str]:
    parts = Path(rel_pdf).parts
    topics = []
    if "fuel_pump_design" in parts:
        topics.append("pump")
    if "materials_science" in parts:
        topics.append("materials")
    if "pintle_injector" in parts:
        topics.append("injector")
    return topics


def build_manifest(verbose: bool = True) -> list[dict]:
    conv = json.loads((AGENT_CORPUS_ROOT / "_conversion_manifest.json").read_text())
    paper_index_path = AGENT_CORPUS_ROOT / "paper_index.json"
    paper_index = json.loads(paper_index_path.read_text()) if paper_index_path.exists() else []
    pi_by_pdf = {p.get("PDF path", ""): p for p in paper_index}

    by_sha: dict[str, list[dict]] = {}
    for row in conv:
        by_sha.setdefault(row.get("sha256", ""), []).append(row)

    now = datetime.now(timezone.utc).isoformat()
    docs: list[dict] = []
    used_ids: dict[str, str] = {}
    for sha, rows in by_sha.items():
        if not sha:
            continue
        rows = sorted(rows, key=lambda r: (len(r["source_pdf"]), r["source_pdf"]))
        canon = rows[0]
        rel_pdf = canon["source_pdf"]
        pdf_path = CORPUS_ROOT / rel_pdf
        md_path = AGENT_CORPUS_ROOT / canon["output_markdown"]
        pi = pi_by_pdf.get(rel_pdf) or next((pi_by_pdf.get(r["source_pdf"]) for r in rows if r["source_pdf"] in pi_by_pdf), None)
        source_identifier = canon.get("source_identifier", "") or ""
        doc_id, id_basis = derive_document_id(rel_pdf, source_identifier, sha)
        if doc_id in used_ids and used_ids[doc_id] != sha:
            doc_id = f"{doc_id}_{sha[:8].upper()}"
            id_basis += "+sha_suffix(collision)"
        used_ids[doc_id] = sha

        title = (pi or {}).get("Title") or canon.get("pdf_title") or ""
        authors = (pi or {}).get("Authors") or canon.get("pdf_author") or ""
        year = (pi or {}).get("Year")
        identifier = (pi or {}).get("DOI or report number") or (source_identifier if DOI_RE.match(source_identifier or "") else "")
        doi = identifier if DOI_RE.match(identifier or "") else ""
        identifier_basis = ("paper_index" if (pi or {}).get("DOI or report number") else
                            ("pdf_metadata_or_first_page" if doi else "none"))
        ntrs = ""
        m = NTRS_RE.match(Path(rel_pdf).name)
        if m:
            ntrs = m.group(1)
        report_number = ""
        rm = NASA_REPORT_RE.search(f"{canon.get('pdf_subject','')} {title}")
        if rm and (ntrs or "NASA" in (title or "").upper()):
            report_number = f"NASA {rm.group(1).upper()}-{rm.group(2)}"
        meta_conf = "medium" if pi else ("low" if (title or authors) else "unknown")
        if title and canon.get("pdf_title") and title.strip().lower() == canon.get("pdf_title", "").strip().lower():
            meta_conf = "high"

        topics = list((pi or {}).get("Topics", []) or []) + _topic_from_path(rel_pdf)
        doc = {
            "id": doc_id,
            "type": "Document",
            "id_basis": id_basis,
            "filename": Path(rel_pdf).name,
            "source_pdf": str(Path("propulsion_texts") / rel_pdf),
            "source_kind": canon.get("source_kind", "file"),
            "source_container": str(Path("propulsion_texts") / canon["source_container"])
                if canon.get("source_kind") == "zip_member" else "",
            "archive_member": canon.get("archive_member", ""),
            "alternate_paths": [str(Path("propulsion_texts") / r["source_pdf"]) for r in rows[1:]],
            "generated_markdown": str(md_path.relative_to(REPO_ROOT)) if md_path.exists() else "",
            "markdown_exists": md_path.exists(),
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "ntrs_accession": ntrs,
            "report_number": report_number,
            "identifier": identifier,
            "identifier_basis": identifier_basis,
            "document_type": "standard" if "ASME" in Path(rel_pdf).name or "Y14" in Path(rel_pdf).name else
                             ("nasa_report" if ntrs else ("journal_or_conference" if doi else "unknown")),
            "topics": sorted(set(t for t in topics if t)),
            "keywords": list((pi or {}).get("Keywords", []) or [])[:25],
            "page_count": canon.get("page_count"),
            "sha256": sha,
            "source_bytes": canon.get("source_bytes"),
            "conversion": {
                "status": canon.get("extraction_status"),
                "method": canon.get("method"),
                "converter": canon.get("converter", "prior-corpus-pass" if "converter" not in canon else canon["converter"]),
                "converter_version": canon.get("converter_version", ""),
                "converted_at": canon.get("converted_at", ""),
                "scanned": canon.get("scanned"),
                # Older OCR rows incorrectly marked confidence as high. A scan
                # or recorded OCR still requires source-PDF verification.
                "low_confidence": bool(canon.get("low_confidence") or canon.get("scanned") or canon.get("ocr_pages")),
                "issues": canon.get("issues", []),
                "equation_signals": canon.get("equation_signals"),
                "source_equation_signals": canon.get("source_equation_signals"),
                "markdown_tables": canon.get("markdown_tables"),
                "source_table_captions": canon.get("source_table_captions"),
            },
            "bibliographic_confidence": meta_conf,
            "bibliographic_basis": "paper_index.json (prior automated pass) + PDF metadata" if pi else "PDF metadata only",
            "source_authority": "primary_literature" if not ("ASME" in Path(rel_pdf).name) else "standard",
            "known_limitations": canon.get("issues", []),
            "extraction_version": EXTRACTION_VERSION,
            "registered_at": now,
        }
        docs.append(doc)

    # Bibliographic identity: two different payloads (different scans/editions)
    # carrying the same DOI or NTRS accession are alternate copies of ONE
    # source.  The better-converted copy is canonical; the other is retained
    # as a Document but not chunked, so the corpus never double-counts it.
    groups: dict[str, list[dict]] = {}
    isbn_re = re.compile(r"isbn1[03]\s*(\d{10,13})", re.IGNORECASE)
    for d in docs:
        im = isbn_re.search(d["filename"])
        if im:
            d["isbn"] = im.group(1)
        key = d["doi"].lower() if d["doi"] else (f"ntrs:{d['ntrs_accession']}" if d["ntrs_accession"] else
                                               (f"isbn:{d['isbn']}" if d.get("isbn") else ""))
        if key:
            groups.setdefault(key, []).append(d)
    rank = {"good": 0, "needs_review": 1, "partial_conversion": 2, "pdf_conversion_failed": 3, None: 4}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda d: (rank.get(d["conversion"]["status"], 4),
                                    -(d["conversion"].get("equation_signals") or 0), d["id"]))
        canon = members[0]
        for other in members[1:]:
            other["alternate_copy_of"] = canon["id"]
            other["alternate_copy_basis"] = f"same bibliographic identifier ({key}) with a different PDF payload"
            canon.setdefault("alternate_copies", []).append(other["id"])
    for d in docs:
        d.setdefault("alternate_copy_of", "")
        d.setdefault("alternate_copies", [])

    # generated-markdown derivative records live alongside, one per canonical doc
    save("manifest", docs, meta={"generated_at": now, "n_documents": len(docs),
                                 "n_conversion_rows": len(conv),
                                 "authority_note": "The PDF is authoritative; generated Markdown is a derivative (DERIVED_FROM)."})
    if verbose:
        print(f"[manifest] {len(docs)} unique documents from {len(conv)} conversion rows")
    return docs


def ingest_chunks(force: bool = False, verbose: bool = True) -> dict:
    """Chunk every canonical document whose Markdown changed since last ingest."""
    docs = load("manifest")
    conn = connect()
    state_rows = {r["doc_id"]: dict(r) for r in conn.execute("SELECT * FROM doc_state")}
    now = datetime.now(timezone.utc).isoformat()
    summary = {"ingested": 0, "unchanged": 0, "skipped_no_markdown": 0, "skipped_failed_conversion": 0,
               "skipped_empty_source": 0, "skipped_empty_extraction": 0,
               "n_chunks_total": 0, "removed": 0, "states": []}
    live = set()
    for doc in docs:
        doc_id = doc["id"]
        status = doc["conversion"]["status"]
        if doc.get("alternate_copy_of"):
            summary["skipped_alternate_copy"] = summary.get("skipped_alternate_copy", 0) + 1
            summary["states"].append({"doc_id": doc_id, "state": "ALTERNATE_COPY_NOT_CHUNKED",
                                      "canonical": doc["alternate_copy_of"]})
            continue
        # A conversion artifact is not source content. In particular, PyMuPDF
        # can open a broken payload reporting zero pages without raising.
        if doc.get("page_count") == 0:
            live.add(doc_id)
            path = REPO_ROOT / doc.get("generated_markdown", "")
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            replace_doc_chunks(conn, doc_id, [], {
                "doc_id": doc_id, "source_sha256": doc["sha256"], "markdown_sha256": markdown_sha256(text),
                "extraction_version": EXTRACTION_VERSION, "n_chunks": 0, "ingested_at": now,
            })
            summary["skipped_empty_source"] += 1
            summary["states"].append({"doc_id": doc_id, "state": "SOURCE_UNREADABLE",
                                      "reason": "EMPTY_SOURCE", "n_chunks": 0,
                                      "source_pdf": doc.get("source_pdf"),
                                      "remediation": "Restore the original PDF: the tracked payload has zero pages; enrichment cannot recover absent pages."})
            continue
        if not doc["markdown_exists"] or not doc.get("generated_markdown") or not (REPO_ROOT / doc["generated_markdown"]).is_file():
            summary["skipped_no_markdown"] += 1
            summary["states"].append({"doc_id": doc_id, "state": "SOURCE_NOT_INGESTED",
                                      "reason": "no generated Markdown"})
            continue
        if status in {"pdf_conversion_failed"}:
            summary["skipped_failed_conversion"] += 1
            summary["states"].append({"doc_id": doc_id, "state": "PDF_CONVERSION_FAILED"})
            continue
        live.add(doc_id)
        md_path = REPO_ROOT / doc["generated_markdown"]
        text = md_path.read_text(encoding="utf-8", errors="replace")
        md_sha = markdown_sha256(text)
        prev = state_rows.get(doc_id)
        if prev and not force and prev["markdown_sha256"] == md_sha and prev["extraction_version"] == EXTRACTION_VERSION and prev["source_sha256"] == doc["sha256"]:
            if not prev["n_chunks"]:
                summary["skipped_empty_extraction"] += 1
                summary["states"].append({"doc_id": doc_id, "state": "SOURCE_NOT_INGESTED",
                                          "reason": "EMPTY_EXTRACTED_TEXT", "n_chunks": 0,
                                          "remediation": "Inspect original PDF pages and rerun extraction or enrichment; no source text was indexed."})
                continue
            summary["unchanged"] += 1
            summary["n_chunks_total"] += prev["n_chunks"]
            summary["states"].append({"doc_id": doc_id, "state": "LOW_EXTRACTION_CONFIDENCE"
                                      if doc["conversion"].get("low_confidence") or status == "partial_conversion"
                                      else "INGESTED", "n_chunks": prev["n_chunks"], "unchanged": True})
            continue
        chunks = chunk_markdown(text, doc_id)
        rows = [{
            "chunk_id": c.chunk_id, "doc_id": doc_id, "seq": c.seq,
            "page_start": c.page_start, "page_end": c.page_end, "section": c.section,
            "kind": c.kind, "char_start": c.char_start, "char_end": c.char_end,
            "n_chars": len(c.text), "equation_signals": c.equation_signals,
            "has_table": int(c.has_table), "extraction_version": EXTRACTION_VERSION,
            "source_sha256": doc["sha256"], "text": c.text,
        } for c in chunks]
        replace_doc_chunks(conn, doc_id, rows, {
            "doc_id": doc_id, "source_sha256": doc["sha256"], "markdown_sha256": md_sha,
            "extraction_version": EXTRACTION_VERSION, "n_chunks": len(rows), "ingested_at": now,
        })
        if not rows:
            summary["skipped_empty_extraction"] += 1
            summary["states"].append({"doc_id": doc_id, "state": "SOURCE_NOT_INGESTED",
                                      "reason": "EMPTY_EXTRACTED_TEXT", "n_chunks": 0,
                                      "remediation": "Inspect original PDF pages and rerun extraction or enrichment; no source text was indexed."})
            continue
        summary["ingested"] += 1
        summary["n_chunks_total"] += len(rows)
        st = "LOW_EXTRACTION_CONFIDENCE" if doc["conversion"].get("low_confidence") or status == "partial_conversion" else "INGESTED"
        summary["states"].append({"doc_id": doc_id, "state": st, "n_chunks": len(rows)})
        if verbose:
            print(f"[ingest] {doc_id}: {len(rows)} chunks ({st})")
    # drop chunks for documents no longer in the manifest
    for doc_id in list(state_rows):
        if doc_id not in live:
            with conn:
                conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                conn.execute("DELETE FROM doc_state WHERE doc_id = ?", (doc_id,))
            summary["removed"] += 1
            current = next((r for r in summary["states"] if r["doc_id"] == doc_id), None)
            if current is not None:
                current["stale_index_removed"] = True
                current["previous_index_state"] = state_rows[doc_id]
            else:
                summary["states"].append({"doc_id": doc_id, "state": "REMOVED_STALE"})
    summary["generated_at"] = now
    conn.close()
    save("ingestion_state", summary["states"], key="doc_id", meta={"generated_at": now})
    return summary
