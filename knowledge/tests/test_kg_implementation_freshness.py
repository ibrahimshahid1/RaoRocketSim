"""Exact AST locator regressions; no inspected code is imported or executed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import implementation_freshness as freshness, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    (tmp_path / "pkg").mkdir()
    return tmp_path


def component(symbol, *, name="SW_TEST", **extra):
    return {"id": name, "type": "SoftwareComponent", "file": "pkg/example.py",
            "qualname": "pkg.example:" + symbol, "line": 99, "verified": True,
            "model_ids": ["MODEL_EXISTING"], "equation_ids": ["EQ_EXISTING"],
            "provenance_id": "PROV_" + name, **extra}


def bank(records):
    store.save("software_links", records)
    store.save("provenance", [{"id": r["provenance_id"], "kind": "CODE_STATES",
                               "subject": r["id"], "file": r["file"], "line": r["line"]} for r in records])
    store.save("models", [{"id": "MODEL_EXISTING", "source_support_state": "SUPPORTED",
                           "source_support_sha256": "literature-source"}])


def test_exact_class_methods_and_nested_symbols_never_use_nearest_terminal_name(isolated):
    (isolated / "pkg/example.py").write_text(
        "raise RuntimeError('must never execute inspected source')\n"
        "class First:\n    def run(self):\n        return 1\n"
        "class Second:\n    def run(self):\n        return 2\n"
        "def outer():\n    def inner():\n        return 3\n    return inner\n")
    bank([component("Second.run", name="SW_EXACT"), component("run", name="SW_BARE"),
          component("outer.inner", name="SW_NESTED")])
    result = freshness.reconcile(repo_root=isolated)
    rows = store.index_by("software_links")
    assert result["verified"] == 2 and result["unavailable"] == 1
    assert rows["SW_EXACT"]["line"] == 6 and rows["SW_EXACT"]["end_line"] == 7
    assert rows["SW_NESTED"]["line"] == 9
    assert rows["SW_BARE"]["implementation_locator_state"] == "AMBIGUOUS_SYMBOL"
    assert rows["SW_BARE"]["active"] is False
    assert result["unavailable_components"][0]["candidates"] == ["First.run", "Second.run"]
    assert rows["SW_EXACT"]["model_ids"] == ["MODEL_EXISTING"]
    assert rows["SW_EXACT"]["implementation_scientific_comparison_state"] == "NOT_RECHECKED"


def test_comment_line_shift_updates_locators_without_invalidating_known_body(isolated):
    path = isolated / "pkg/example.py"
    path.write_text("def solve(value):\n    return value + 1\n")
    bank([component("solve")])
    freshness.reconcile(repo_root=isolated)
    before, = store.load("software_links")
    before["implementation_scientific_comparison_state"] = "PREVIOUS_REVIEW_RECORDED"
    store.save("software_links", [before])
    path.write_text("# Comment-only edit shifts definitions.\n\n" + path.read_text())
    result = freshness.reconcile(repo_root=isolated)
    after, = store.load("software_links")
    assert result["changed_definitions"] == 0 and result["locators_corrected"] == 1
    assert after["line"] == 3 and after["end_line"] == 4
    assert before["definition_sha256"] == after["definition_sha256"]
    assert before["source_sha256"] != after["source_sha256"]
    assert after["implementation_scientific_comparison_state"] == "PREVIOUS_REVIEW_RECORDED"
    prov, = store.load("provenance")
    assert prov["line"] == 3 and prov["source_sha256"] == after["source_sha256"]
    contents = store.store_path("software_links").read_bytes()
    assert freshness.reconcile(repo_root=isolated)["records_updated"] == 0
    assert store.store_path("software_links").read_bytes() == contents


def test_changed_body_requires_comparison_review_without_changing_model_verdict(isolated):
    path = isolated / "pkg/example.py"
    path.write_text("def solve(value):\n    return value + 1\n")
    bank([component("solve")])
    freshness.reconcile(repo_root=isolated)
    before, = store.load("software_links")
    before["implementation_scientific_comparison_state"] = "PREVIOUS_REVIEW_RECORDED"
    store.save("software_links", [before])
    models_before = store.store_path("models").read_bytes()
    path.write_text("def solve(value):\n    return value + 2\n")
    result = freshness.reconcile(repo_root=isolated)
    after, = store.load("software_links")
    assert result["changed_definition_ids"] == ["SW_TEST"]
    assert after["implementation_scientific_comparison_state"] == "NOT_RECHECKED"
    assert after["implementation_scientific_comparison_history"][-1]["state"] == "PREVIOUS_REVIEW_RECORDED"
    assert after["implementation_locator_history"][-1]["definition_sha256"] == before["definition_sha256"]
    assert after["model_ids"] == before["model_ids"] and after["equation_ids"] == before["equation_ids"]
    assert store.store_path("models").read_bytes() == models_before


def test_missing_duplicate_and_wrong_module_locators_preserve_inactive_history(isolated):
    path = isolated / "pkg/example.py"
    path.write_text("if flag:\n    def duplicated():\n        return 1\nelse:\n    def duplicated():\n        return 2\n")
    bank([component("missing", name="SW_MISSING"), component("duplicated", name="SW_DUPLICATE"),
          component("duplicated", name="SW_WRONG", qualname="pkg.other:duplicated")])
    result = freshness.reconcile(repo_root=isolated)
    rows = store.index_by("software_links")
    assert result["unavailable"] == 3
    assert rows["SW_MISSING"]["implementation_locator_state"] == "SYMBOL_NOT_FOUND"
    assert rows["SW_DUPLICATE"]["implementation_locator_state"] == "AMBIGUOUS_SYMBOL"
    assert rows["SW_WRONG"]["implementation_locator_state"] == "QUALNAME_FILE_MISMATCH"
    assert all(not r["active"] and r["implementation_locator_history"] for r in rows.values())
    before = store.store_path("software_links").read_bytes()
    assert freshness.reconcile(repo_root=isolated)["records_updated"] == 0
    assert store.store_path("software_links").read_bytes() == before
    path.write_text("def missing():\n    return 1\n")
    freshness.reconcile(repo_root=isolated)
    recovered = store.index_by("software_links")["SW_MISSING"]
    assert recovered["active"] and recovered["implementation_scientific_comparison_state"] == "NOT_RECHECKED"


def test_read_only_audit_and_module_scope_do_not_mutate_stores(isolated):
    (isolated / "pkg/example.py").write_text("DATA = [1, 2]\nDATA[index] = value\n")
    bank([component("<module>", name="SW_MODULE"), component("index", name="SW_INDEX")])
    before = {name: store.store_path(name).read_bytes() for name in ("software_links", "provenance", "models")}
    result = freshness.reconcile(repo_root=isolated, write=False)
    assert result["verified"] == 1 and result["unavailable"] == 1
    assert {name: store.store_path(name).read_bytes() for name in before} == before
    assert not result["write"]
    # Pre-existing inactive records from another owner are never revived.
    record = component("<module>", active=False, source_state="MANUAL_REVIEW_REQUIRED")
    store.save("software_links", [record])
    assert freshness.reconcile(repo_root=isolated)["skipped_inactive"] == 1
    assert store.load("software_links") == [record]
