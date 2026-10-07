"""Structure-aware chunking of generated corpus Markdown.

Chunks follow the document's own structure - page markers, headings,
paragraphs, tables, fenced blocks - rather than a fixed token window, so an
equation stays with the sentence that introduces it and a table is never cut
in half.  Every chunk keeps the page span, heading path, document order and
character offsets needed to escalate back to the PDF.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

PAGE_RE = re.compile(r"<!--\s*PDF page\s+(\d+)\s*-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
PROV_RE = re.compile(r"<!-- lrekit-kg-conversion.*?-->\s*", re.DOTALL)
WARN_RE = re.compile(r"^> \[!WARNING\]\n(?:>.*\n?)+", re.MULTILINE)
REFERENCES_RE = re.compile(r"^\s*(references|bibliography|literature cited)\b", re.IGNORECASE)

_EQ_CHARS = "=≈≅≡≤≥∂∫∑∏√∞±∓·×÷αβγδεζηθκλμνξπρστφχψωΓΔΘΛΞΠΣΦΨΩ∇⋅"

TARGET_CHARS = 1400
MAX_CHARS = 2600
MIN_CHARS = 200


def equation_signals(text: str) -> int:
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = sum(1 for ch in s if ch in _EQ_CHARS)
        if m >= 2 or (m >= 1 and len(s) <= 80):
            n += 1
    return n


@dataclass
class Block:
    kind: str            # paragraph | heading | table | fence | equation
    text: str
    page: int
    char_start: int
    char_end: int
    level: int = 0       # heading level


@dataclass
class Chunk:
    doc_id: str
    seq: int
    page_start: int
    page_end: int
    section: str
    kind: str
    text: str
    char_start: int
    char_end: int
    equation_signals: int = 0
    has_table: bool = False

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#c{self.seq:05d}"


def _blocks(markdown: str) -> list[Block]:
    """Split Markdown into typed blocks, tracking the current PDF page."""
    blocks: list[Block] = []
    page = 0
    pos = 0
    lines = markdown.splitlines(keepends=True)
    buf: list[str] = []
    buf_start = 0
    buf_kind = "paragraph"
    in_fence = False

    def flush(end_pos: int):
        nonlocal buf, buf_kind, buf_start
        if buf:
            text = "".join(buf).strip("\n")
            if text.strip():
                blocks.append(Block(buf_kind, text, page, buf_start, end_pos))
        buf = []
        buf_kind = "paragraph"

    for line in lines:
        start = pos
        pos += len(line)
        stripped = line.strip()

        if in_fence:
            buf.append(line)
            if stripped.startswith("```"):
                in_fence = False
                flush(pos)
            continue

        m = PAGE_RE.match(stripped)
        if m:
            flush(start)
            page = int(m.group(1))
            continue

        if stripped.startswith("```"):
            flush(start)
            in_fence = True
            buf_kind = "fence"
            buf_start = start
            buf.append(line)
            continue

        hm = HEADING_RE.match(stripped)
        if hm:
            flush(start)
            blocks.append(Block("heading", hm.group(2).strip(), page, start, pos, level=len(hm.group(1))))
            continue

        if stripped.startswith("|"):
            if buf_kind != "table":
                flush(start)
                buf_kind = "table"
                buf_start = start
            buf.append(line)
            continue
        elif buf_kind == "table":
            flush(start)

        if not stripped:
            flush(start)
            continue

        if not buf:
            buf_start = start
            buf_kind = "paragraph"
        buf.append(line)
    flush(pos)
    return blocks


def chunk_markdown(markdown: str, doc_id: str) -> list[Chunk]:
    markdown = PROV_RE.sub("", markdown, count=1)
    markdown = WARN_RE.sub("", markdown, count=1)
    blocks = _blocks(markdown)

    chunks: list[Chunk] = []
    section_path: list[tuple[int, str]] = []
    cur: list[Block] = []
    cur_len = 0
    in_references = False

    def section_label() -> str:
        return " > ".join(t for _, t in section_path[-3:])

    def emit():
        nonlocal cur, cur_len
        if not cur:
            return
        text = "\n\n".join(b.text for b in cur)
        kinds = {b.kind for b in cur}
        kind = "table" if "table" in kinds and len(kinds) == 1 else ("references" if in_references else "text")
        if kind == "text" and equation_signals(text) >= 3:
            kind = "equation_rich"
        chunks.append(Chunk(
            doc_id=doc_id, seq=len(chunks),
            page_start=cur[0].page, page_end=cur[-1].page,
            section=section_label(), kind=kind, text=text,
            char_start=cur[0].char_start, char_end=cur[-1].char_end,
            equation_signals=equation_signals(text),
            has_table="table" in kinds,
        ))
        cur = []
        cur_len = 0

    for b in blocks:
        if b.kind == "heading":
            emit()
            while section_path and section_path[-1][0] >= b.level:
                section_path.pop()
            section_path.append((b.level, b.text))
            in_references = bool(REFERENCES_RE.match(b.text))
            continue
        blen = len(b.text)
        # tables and fences are atomic; give them their own chunk when large
        if b.kind in {"table", "fence"} and blen > TARGET_CHARS:
            emit()
            cur = [b]
            cur_len = blen
            emit()
            continue
        if cur_len + blen > MAX_CHARS or (cur_len >= TARGET_CHARS and b.kind == "paragraph"):
            emit()
        # page change with a substantial buffer -> split so page spans stay tight
        if cur and b.page != cur[-1].page and cur_len >= MIN_CHARS:
            emit()
        cur.append(b)
        cur_len += blen + 2
    emit()

    # merge tiny trailing fragments into their predecessor when same section/page
    merged: list[Chunk] = []
    for c in chunks:
        if merged and len(c.text) < MIN_CHARS and merged[-1].section == c.section \
                and merged[-1].page_end == c.page_start and len(merged[-1].text) + len(c.text) < MAX_CHARS:
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + c.text
            prev.page_end = c.page_end
            prev.char_end = c.char_end
            prev.equation_signals = equation_signals(prev.text)
            prev.has_table = prev.has_table or c.has_table
            continue
        merged.append(c)
    # Drop page furniture that survived conversion (download stamps, running
    # heads).  These carry no science and would otherwise pollute retrieval.
    merged = [c for c in merged if not _is_furniture(c.text)]
    for i, c in enumerate(merged):
        c.seq = i
    return merged


FURNITURE_RE = re.compile(
    r"^(?:downloaded by|purchased from|copyright ©|©\s*\d{4}|all rights reserved|journal homepage|"
    r"this article was downloaded|redistribution subject to|see the terms and conditions|"
    r"\*?\*?journal of propulsion and power\*?\*?$|\*?_?article_?\*?$|_?marshall space flight center_?$)",
    re.IGNORECASE,
)


def _is_furniture(text: str) -> bool:
    t = text.strip()
    if len(t) < 60 and "|" not in t and equation_signals(t) == 0:
        return True
    head = t.splitlines()[0].strip().lower()
    return bool(FURNITURE_RE.match(head)) and len(t) < 400


def markdown_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
