"""Structure-preserving fallback renderer for PDFs the Markdown parser cannot read.

Some corpus PDFs (notably scanned pages carrying an invisible text layer, and
some older AIAA/ASME scans) yield an empty result from ``pymupdf4llm`` even
though PyMuPDF's own text extractor reads them fine.  Rather than silently
emitting an empty document, the converter falls back to this renderer, which
rebuilds reading order and a conservative amount of Markdown structure from the
raw span geometry.

Nothing here reconstructs mathematics.  Ambiguous layout is emitted as-is.
"""

from __future__ import annotations

import statistics
from collections import Counter


def _spans_of(page, clip=None):
    d = page.get_text("dict", clip=clip)
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        yield block


def body_font_size(page) -> float:
    sizes: Counter[float] = Counter()
    for block in _spans_of(page):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if len(text) >= 3:
                    sizes[round(span.get("size", 0.0), 1)] += len(text)
    if not sizes:
        return 0.0
    return sizes.most_common(1)[0][0]


def _render_span(span) -> str:
    text = span.get("text", "")
    core = text.strip()
    if not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    flags = span.get("flags", 0)
    bold = bool(flags & 2 ** 4)
    italic = bool(flags & 2 ** 1)
    if bold and italic:
        core = f"**_{core}_**"
    elif bold:
        core = f"**{core}**"
    elif italic:
        core = f"_{core}_"
    return lead + core + trail


def _block_to_markdown(block, body_size: float) -> str:
    lines_out: list[str] = []
    max_size = 0.0
    for line in block.get("lines", []):
        parts = []
        for span in line.get("spans", []):
            max_size = max(max_size, span.get("size", 0.0))
            parts.append(_render_span(span))
        text = "".join(parts).rstrip()
        if text.strip():
            lines_out.append(text)
    if not lines_out:
        return ""
    text = "\n".join(lines_out)
    plain = " ".join(ln.strip() for ln in lines_out).strip()
    # Conservative heading detection: short, larger than body text, not a sentence.
    if body_size and max_size >= body_size * 1.15 and len(plain) <= 90 and not plain.endswith((".", ";", ",")):
        level = "##" if max_size >= body_size * 1.4 else "###"
        clean = plain.replace("**", "").replace("_", "")
        return f"{level} {clean}"
    return text


def page_to_markdown(page, *, use_columns: bool = True) -> str:
    """Render one page to Markdown-ish text preserving reading order."""
    body_size = body_font_size(page)
    clips = None
    if use_columns:
        try:
            from pymupdf4llm.helpers.multi_column import column_boxes

            boxes = column_boxes(page, no_image_text=False, ignore_images=True)
            if boxes and len(boxes) > 1:
                clips = list(boxes)
        except Exception:
            clips = None

    chunks: list[str] = []
    if clips:
        for clip in clips:
            for block in _spans_of(page, clip=clip):
                md = _block_to_markdown(block, body_size)
                if md.strip():
                    chunks.append(md)
    else:
        blocks = sorted(_spans_of(page), key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
        for block in blocks:
            md = _block_to_markdown(block, body_size)
            if md.strip():
                chunks.append(md)
    return "\n\n".join(chunks)
