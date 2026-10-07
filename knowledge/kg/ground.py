"""Deterministic grounding of LLM-extracted scientific objects against source text.

An extracted equation or claim enters the knowledge stores only if its
``quote`` (a short verbatim anchor the extractor must supply) can be located
in the ingested chunk text of the stated document on the stated page (±2
pages, because page markers sit at PDF-page granularity and layout rebuilds
can shift a line). Matching preserves word order and requires a contiguous
anchor after whitespace/markup normalization. Token overlap is not proof
that a source states a claim.

Outcomes:
    GROUNDED               anchor found on the stated page
    GROUNDED_PAGE_SHIFTED  anchor found within ±2 pages (page corrected)
    UNGROUNDED             anchor not found -> object is NOT stored, only reported
"""

from __future__ import annotations

import re
import unicodedata

from .store import connect, doc_chunks

_STRIP_RE = re.compile(r"[*_`#|<>\[\]{}()\\^~$]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _STRIP_RE.sub(" ", text)
    text = text.replace("­", "").replace("-\n", "")
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if len(t) > 1]


class DocGrounder:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        conn = connect()
        self.chunks = doc_chunks(conn, doc_id)
        conn.close()
        self.by_page: dict[int, list[dict]] = {}
        for c in self.chunks:
            for p in range(c["page_start"], c["page_end"] + 1):
                self.by_page.setdefault(p, []).append(c)
        self._norm_cache: dict[str, str] = {}
        self.max_page = max((c["page_end"] for c in self.chunks), default=0)

    def _norm(self, c: dict) -> str:
        cid = c["chunk_id"]
        if cid not in self._norm_cache:
            self._norm_cache[cid] = normalize(c["text"])
        return self._norm_cache[cid]

    def locate(self, quote: str, page: int | None, radius: int = 2, min_overlap: float = 0.8) -> dict:
        q = normalize(quote)
        qt = _tokens(quote)
        if len(q) < 12 or len(qt) < 3:
            return {"state": "UNGROUNDED", "reason": "anchor too short to be verifiable"}
        pages: list[int]
        if page:
            pages = [page] + [p for d in range(1, radius + 1) for p in (page - d, page + d) if p > 0]
        else:
            pages = sorted(self.by_page)
        for p in pages:
            for c in self.by_page.get(p, []):
                hay = self._norm(c)
                if q in hay:
                    state = "GROUNDED" if p == page else "GROUNDED_PAGE_SHIFTED"
                    return {"state": state, "chunk_id": c["chunk_id"], "page": p, "match": "exact", "overlap": 1.0}
        # last resort: search whole document for an exact normalized hit (page unknown to extractor)
        if page:
            for c in self.chunks:
                if q in self._norm(c):
                    return {"state": "GROUNDED_PAGE_SHIFTED", "chunk_id": c["chunk_id"], "page": c["page_start"],
                            "match": "exact_other_page", "overlap": 1.0}
        return {"state": "UNGROUNDED", "reason": "anchor not found on stated page ±%d" % radius}
