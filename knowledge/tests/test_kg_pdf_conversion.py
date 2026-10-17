"""Corpus-free checks of page-level OCR eligibility and honest diagnostics."""

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import pdf_to_markdown


class FakePage:
    def __init__(self, native_text, *, ocr_text="", ocr_error=None):
        self.native_text = native_text
        self.ocr_text = ocr_text
        self.ocr_error = ocr_error
        self.ocr_calls = []
        self.textpage = object()

    def get_text(self, kind, *, textpage=None):
        assert kind == "text"
        if textpage is not None:
            assert textpage is self.textpage
            return self.ocr_text
        return self.native_text

    def get_textpage_ocr(self, **kwargs):
        self.ocr_calls.append(kwargs)
        if self.ocr_error is not None:
            raise self.ocr_error
        return self.textpage


class FakeDocument:
    metadata = {"title": "Synthetic mixed PDF", "author": "Test fixture"}

    def __init__(self, pages):
        self.pages = pages
        self.page_count = len(pages)
        self.closed = False

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


def native_page(number):
    return FakePage(
        f"# Source page {number}\n\n"
        + f"Native paragraph uniquely identifying source page {number}. " * 4
    )


@pytest.fixture
def fake_pdf(tmp_path, monkeypatch):
    """Stub both optional PDF packages; no PDF parser or OCR runtime is needed."""
    def install(pages, *, omitted_pages=()):
        doc = FakeDocument(pages)
        pymupdf = ModuleType("pymupdf")
        pymupdf.open = lambda path: doc
        pymupdf4llm = ModuleType("pymupdf4llm")
        pymupdf4llm.use_layout = lambda enabled: None

        def to_markdown(opened_doc, **kwargs):
            assert opened_doc is doc
            assert kwargs == {"page_chunks": True, "write_images": False,
                              "show_progress": False}
            return [{"metadata": {"page": i}, "text": page.native_text}
                    for i, page in enumerate(pages, 1) if i not in omitted_pages]

        pymupdf4llm.to_markdown = to_markdown
        monkeypatch.setitem(sys.modules, "pymupdf", pymupdf)
        monkeypatch.setitem(sys.modules, "pymupdf4llm", pymupdf4llm)
        # Ensure no external text-layout machinery is involved in these cases.
        layout = ModuleType("text_layout")
        layout.page_to_markdown = lambda page: page.native_text
        monkeypatch.setitem(sys.modules, "text_layout", layout)
        monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "unused-tessdata"))
        pdf = tmp_path / "source.pdf"
        pdf.write_bytes(b"Synthetic payload; parsed only by the in-memory fake.")
        output = tmp_path / "derivative.md"
        return doc, pdf, output

    return install


@pytest.mark.parametrize("omit_sparse_chunk", [False, True])
def test_mixed_pdf_recovers_sparse_page_and_marks_ocr_low_confidence(fake_pdf, omit_sparse_chunk):
    pages = [native_page(i) for i in range(1, 5)]
    pages.append(FakePage("", ocr_text="Recovered appendix prose requiring source verification."))
    doc, pdf, output = fake_pdf(pages, omitted_pages=(5,) if omit_sparse_chunk else ())

    result = pdf_to_markdown.convert_pdf(pdf, output)

    assert result.scanned is False  # Four native pages, one sparse scanned page.
    assert result.ocr_pages == 1 and result.failed_pages == 0
    assert not any(page.ocr_calls for page in pages[:4])
    assert pages[4].ocr_calls == [{"flags": 0, "dpi": 220, "full": True}]
    assert result.low_confidence is True
    assert result.extraction_status == "needs_review"
    assert result.method.endswith("+tesseract-ocr")
    markdown = output.read_text()
    assert markdown.startswith("> [!WARNING]\n")
    assert "Recovered appendix prose" in markdown
    assert f"source_pdf: {pdf}" in markdown
    assert f"source_sha256: {hashlib.sha256(pdf.read_bytes()).hexdigest()}" in markdown
    assert f"converter: {pdf_to_markdown.CONVERTER_NAME} {pdf_to_markdown.CONVERTER_VERSION}" in markdown
    for i in range(1, 6):
        assert markdown.count(f"<!-- PDF page {i} -->") == 1
    assert all(page.native_text.strip() in markdown for page in pages[:4])
    assert any("symbols, equations, subscripts, and reading order require verification" in issue
               for issue in result.issues)
    assert pdf.read_bytes().startswith(b"Synthetic payload") and doc.closed


def test_no_ocr_cli_preserves_empty_page_marker_without_attempting_ocr(fake_pdf, capsys):
    pages = [native_page(i) for i in range(1, 5)]
    pages.append(FakePage("", ocr_text="This content must not be requested."))
    doc, pdf, output = fake_pdf(pages)

    assert pdf_to_markdown.main([str(pdf), str(output), "--no-ocr"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ocr_pages"] == 0 and result["failed_pages"] == 0
    assert not any(page.ocr_calls for page in pages)
    assert "tesseract-ocr" not in result["method"]
    markdown = output.read_text()
    assert "must not be requested" not in markdown
    assert markdown.split("<!-- PDF page 5 -->", 1)[1].strip() == ""
    assert any("empty page marker" in issue for issue in result["issues"])
    assert doc.closed


def test_mixed_pdf_attempts_all_sparse_pages_and_reports_partial_ocr_failure(fake_pdf):
    recovered = FakePage("", ocr_text="Recovered sparse-page prose requiring manual verification.")
    failed = FakePage("", ocr_error=RuntimeError("synthetic OCR unavailable"))
    pages = [native_page(1), recovered, native_page(3), failed, native_page(5), native_page(6)]
    doc, pdf, output = fake_pdf(pages)

    result = pdf_to_markdown.convert_pdf(pdf, output)

    assert result.scanned is False
    assert len(recovered.ocr_calls) == len(failed.ocr_calls) == 1
    assert not any(page.ocr_calls for page in pages if page not in (recovered, failed))
    assert result.ocr_pages == result.failed_pages == 1
    assert result.low_confidence is True
    assert any("OCR failed on 1 page(s)" in issue for issue in result.issues)
    markdown = output.read_text()
    assert "Recovered sparse-page prose" in markdown
    assert markdown.split("<!-- PDF page 4 -->", 1)[1].split("<!-- PDF page 5 -->", 1)[0].strip() == ""
    assert "synthetic OCR unavailable" not in markdown
    assert doc.closed


def test_native_pdf_needs_no_ocr_or_confidence_warning(fake_pdf):
    pages = [native_page(i) for i in range(1, 5)]
    doc, pdf, output = fake_pdf(pages)

    result = pdf_to_markdown.convert_pdf(pdf, output)

    assert not any(page.ocr_calls for page in pages)
    assert result.ocr_pages == result.failed_pages == 0
    assert result.scanned is result.low_confidence is False
    assert result.extraction_status == "good"
    assert result.method == "pymupdf4llm-classic"
    assert "[!WARNING]" not in output.read_text()
    assert doc.closed
