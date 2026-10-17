"""A resolved report key cannot validate an engineering threshold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg import store, graph, harvest_constraints


def test_report_key_is_citation_not_threshold_support(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(graph, "GRAPH_DIR", tmp_path / "graph")
    monkeypatch.setattr(harvest_constraints, "REPO_ROOT", tmp_path)
    source = tmp_path / "raosim/mdo/constraints.py"
    source.parent.mkdir(parents=True)
    source.write_text('ConstraintSpec("example", None, "margin", "hard", "structures", 1, "Pa", "NASA-SP-125")\n')
    monkeypatch.setattr(harvest_constraints, "_load_manifest", lambda: ([{
        "name": "example", "engine_key": None, "kind": "margin", "optimizer_role": "hard",
        "category": "structures", "scale": 1, "units": "Pa", "source_id": "NASA-SP-125", "applicability": "always"}], "fixture"))
    store.save("manifest", [{"id": "DOC_NTRS_19710019929", "filename": "paper.pdf", "source_pdf": "paper.pdf",
                              "title": "Synthetic report identity", "conversion": {"status": "good"}}])
    harvest_constraints.harvest()
    row, = store.load("requirements")
    assert row["citation_resolution_state"] == "CITATION_RESOLVED"
    assert row["basis_state"] == row["source_support_state"] == "NOT_CHECKED"
    edges = [e for e in graph.assemble()["edges"] if e["from"] == row["id"]]
    assert any(e["type"] == "CITES" for e in edges)
    assert not any(e["type"] == "SUPPORTED_BY" for e in edges)


def test_ast_fallback_reads_keyword_basis_and_labels_non_literature_rows(tmp_path, monkeypatch):
    """The JAX-free fallback must map ``basis_class=`` by keyword, never into
    the positional ``applicability`` slot, and a declared non-literature row
    without a source is labelled rather than reported as an unknown basis."""
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(graph, "GRAPH_DIR", tmp_path / "graph")
    monkeypatch.setattr(harvest_constraints, "REPO_ROOT", tmp_path)
    source = tmp_path / "raosim/mdo/constraints.py"
    source.parent.mkdir(parents=True)
    source.write_text('''CONSTRAINT_MANIFEST = (
    ConstraintSpec("land_fit", "land_min", "inequality", "hard",
                   "geometry", 5.0e-4, "m", None,
                   basis_class="geometric_definition"),
    ConstraintSpec("chug", "chug_margin_min", "inequality", "hard",
                   "injector", 0.2, None,
                   "NASA-SP-125-1967;"
                   "NASA-SP-194-1972", "always", basis_class="literature"),
    ConstraintSpec("legacy", None, "inequality", "hard", "geometry", 1.0, None, None),
)
''')
    rows = harvest_constraints._parse_manifest(source.read_text())
    assert [r["name"] for r in rows] == ["land_fit", "chug", "legacy"]
    assert rows[0]["applicability"] == "always"
    assert rows[0]["basis_class"] == "geometric_definition"
    assert rows[1]["source_id"] == "NASA-SP-125-1967;NASA-SP-194-1972"
    assert rows[2]["basis_class"] is None
    monkeypatch.setattr(harvest_constraints, "_load_manifest", lambda: (rows, "fixture"))
    store.save("manifest", [])
    harvest_constraints.harvest()
    by_name = {r["name"]: r for r in store.load("requirements")}
    assert by_name["land_fit"]["basis_state"] == "DECLARED_GEOMETRIC_DEFINITION"
    assert by_name["land_fit"]["declared_basis_class"] == "geometric_definition"
    assert by_name["legacy"]["basis_state"] == "SCIENTIFIC_BASIS_UNKNOWN"
    assert by_name["chug"]["basis_state"] == "SOURCE_NOT_INGESTED"
