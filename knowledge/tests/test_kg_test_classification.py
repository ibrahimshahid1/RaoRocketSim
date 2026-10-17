"""Function evidence must not inherit a module's unrelated literature claim."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_repo, store
from kg.schema import node_id


PDF = "propulsion_texts/reference.pdf"
SOURCE = '''"""Module references propulsion_texts/reference.pdf; not all tests reproduce it."""
import pytest

def test_bookkeeping():
    assert 2.0 == pytest.approx(2.0)

class TestPublished:
    def test_published_value(self):
        """Published value: propulsion_texts/reference.pdf, page 2."""
        expected = 3.0
        assert 3.0 == pytest.approx(expected, rel=1e-6)

class TestOther:
    def test_published_value(self):
        assert 4 == 4

async def test_snapshot_contract():
    assert True
'''


@pytest.fixture
def harvested(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(harvest_repo, "REPO_ROOT", tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mixed.py").write_text(SOURCE)
    paths = ["test_bookkeeping", "TestPublished::test_published_value", "TestOther::test_published_value",
             "test_snapshot_contract", "MissingClass::test_published_value", "test_no_longer_present"]
    records = []
    for suffix in paths:
        path = "tests/test_mixed.py::" + suffix
        records.append({"id": node_id("TEST", path), "type": "Test", "path": path,
                        "file": "tests/test_mixed.py", "verification_kind": "published_benchmark_reproduction",
                        "source_document_id": "DOC_INHERITED", "source_document_ids": ["DOC_INHERITED"],
                        "cited_pdfs": ["propulsion_texts/unrelated.pdf"], "model_ids": ["MODEL_RETAINED"],
                        "review_note": "Keep the survey annotation"})
    store.save("tests", records)
    return {r["path"]: r for r in harvest_repo._tests_harvest("now", {PDF: "DOC_REFERENCE"})}


def test_module_metadata_does_not_promote_unrelated_functions(harvested):
    module = harvested["tests/test_mixed.py"]
    assert module["classification_scope"] == "module"
    assert module["benchmark_candidate"] is True
    assert module["verification_kind"] != "published_benchmark_reproduction"
    record = harvested["tests/test_mixed.py::test_bookkeeping"]
    assert record["verification_kind"] == "unit_test"
    assert record["source_document_id"] == ""
    assert record["source_document_ids"] == record["cited_pdfs"] == []
    assert record["model_ids"] == ["MODEL_RETAINED"]
    assert record["review_note"] == "Keep the survey annotation"


def test_exact_class_qualified_method_gets_only_its_local_evidence(harvested):
    published = harvested["tests/test_mixed.py::TestPublished::test_published_value"]
    other = harvested["tests/test_mixed.py::TestOther::test_published_value"]
    assert published["verification_kind"] == "published_benchmark_reproduction"
    assert published["source_document_ids"] == ["DOC_REFERENCE"]
    assert published["classification_scope"] == "test_function"
    assert published["is_experimental_validation"] is False
    assert other["verification_kind"] == "unit_test"
    assert other["source_document_ids"] == []


def test_async_function_and_unlocated_records_are_conservative(harvested):
    assert harvested["tests/test_mixed.py::test_snapshot_contract"]["verification_kind"] == "snapshot_contract"
    for suffix in ("MissingClass::test_published_value", "test_no_longer_present"):
        record = harvested["tests/test_mixed.py::" + suffix]
        assert record["verification_kind"] == "unit_test"
        assert record["source_document_ids"] == record["cited_pdfs"] == []
        assert "not located" in record["verification_kind_basis"]


def test_citation_and_tolerance_words_without_actual_comparison_do_not_promote():
    text = '''def test_published():
    """propulsion_texts/reference.pdf mentions tolerance and pytest.approx."""
    assert True
'''
    result = harvest_repo._test_classification("test_published", text, {PDF: "DOC_REFERENCE"}, function=True)
    assert result["verification_kind"] == "unit_test"
    assert result["source_document_ids"] == ["DOC_REFERENCE"]


def test_malformed_source_has_no_guessed_function_matches():
    assert harvest_repo._test_function_sources("def test_missing(:\n") == {}
