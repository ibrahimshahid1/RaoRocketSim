"""Source-free retrieval regressions; all index/graph/store writes are temporary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import api, graph, index, store  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_knowledge(tmp_path, monkeypatch):
    api.reset_caches()
    vector_dir = tmp_path / "vector_index"
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    monkeypatch.setattr(store, "CHUNK_DB", vector_dir / "chunks.sqlite")
    monkeypatch.setattr(index, "VECTOR_DIR", vector_dir)
    monkeypatch.setattr(graph, "GRAPH_DIR", graph_dir)
    store.save("manifest", [{"id": "DOC_TEST", "title": "Synthetic retrieval fixture",
                             "source_pdf": "synthetic.pdf", "conversion": {"status": "converted"}}])
    yield
    api.reset_caches()


def add_chunks(*texts):
    conn = store.connect()
    try:
        with conn:
            for seq, text in enumerate(texts):
                chunk_id = f"DOC_TEST#c{seq}"
                conn.execute("INSERT INTO chunks (chunk_id, doc_id, seq, page_start, page_end, section, kind, text) "
                             "VALUES (?, 'DOC_TEST', ?, ?, ?, 'Fixture', 'paragraph', ?)",
                             (chunk_id, seq, seq + 1, seq + 1, text))
                conn.execute("INSERT INTO chunks_fts (text, section, doc_id, chunk_id) "
                             "VALUES (?, 'Fixture', 'DOC_TEST', ?)", (text, chunk_id))
    finally:
        conn.close()


def test_missing_vectors_fall_back_to_lexical_with_explicit_status():
    add_chunks("A pump has an impeller.", "A nozzle expands gas.")
    result = api.semantic_search("impeller")
    assert result["state"] == "OK"
    assert [hit["chunk_id"] for hit in result["hits"]] == ["DOC_TEST#c0"]
    assert result["hits"][0]["page_start"] == 1
    assert result["hits"][0]["source_pdf"] == "synthetic.pdf"
    assert result["retrieval"]["mode"] == "LEXICAL_ONLY"
    assert result["retrieval"]["semantic"]["state"] == "UNAVAILABLE"
    assert result["retrieval"]["semantic"]["reason"] == "VECTOR_INDEX_NOT_BUILT"
    assert "lsa_rank" not in result["hits"][0]


def test_missing_vectors_status_survives_document_and_task_apis():
    add_chunks("Impeller pumping fixture")
    documents = api.search_documents("impeller")
    context = api.retrieve_for_task("impeller")
    assert documents["documents"][0]["pages"] == [1]
    assert documents["retrieval"]["semantic"]["state"] == "UNAVAILABLE"
    assert context["source_context"][0]["chunk_id"] == "DOC_TEST#c0"
    assert context["retrieval"]["semantic"]["state"] == "UNAVAILABLE"
    assert any("lexical search only" in warning for warning in context["implementation_warnings"])


def test_unmatched_lexical_query_does_not_claim_semantic_coverage():
    add_chunks("Impeller pumping fixture")
    result = api.semantic_search("zebrawidget")
    assert result["hits"] == []
    assert result["state"] == "NO_RELEVANT_SOURCE_FOUND"
    assert result["retrieval"]["semantic"]["state"] == "UNAVAILABLE"


@pytest.mark.parametrize("texts,reason", [([], "EMPTY_CORPUS"), (["the and of"], "NO_INDEXABLE_TERMS"),
                                        (["impeller"], "NO_INDEXABLE_TERMS")])
def test_build_empty_semantic_space_keeps_lexical_retrieval(texts, reason):
    add_chunks(*texts)
    meta = index.build(verbose=False)
    vi = index.VectorIndex()
    assert meta["k"] == 0
    assert meta["state"] == "UNAVAILABLE"
    assert meta["reason"] == reason
    assert vi.status["reason"] == reason
    assert vi.search("impeller") == []
    assert vi.vectors.shape == (len(texts), 0)
    if texts == ["impeller"]:
        assert api.semantic_search("impeller")["hits"][0]["chunk_id"] == "DOC_TEST#c0"


@pytest.mark.parametrize("texts", [["impeller pumping"], ["impeller", "impeller"]])
def test_build_single_row_or_column_uses_valid_semantic_projection(texts):
    add_chunks(*texts)
    meta = index.build(min_df=1, max_df_frac=1.0, verbose=False)
    vi = index.VectorIndex()
    assert meta["k"] == 1
    assert vi.status == {"state": "OK"}
    assert len(vi.search("impeller")) == len(texts)
    assert all(score > 0.99 for _, score in vi.search("impeller"))
    assert vi.search("zebrawidget") == []


def test_tiny_corpus_uses_sparse_svd_and_hybrid_search():
    add_chunks("impeller pumping", "nozzle expansion", "impeller nozzle")
    meta = index.build(k=2, min_df=1, max_df_frac=1.0, verbose=False)
    result = api.semantic_search("impeller")
    assert meta["k"] == 2
    assert result["retrieval"]["mode"] == "HYBRID"
    assert result["hits"]
    assert all(hit["lsa_cos"] > 0 for hit in result["hits"] if "lsa_cos" in hit)


def test_partial_index_files_degrade_to_lexical():
    add_chunks("impeller")
    index.build(min_df=1, max_df_frac=1.0, verbose=False)
    (index.VECTOR_DIR / "components.npy").unlink()
    result = api.semantic_search("impeller")
    assert result["hits"]
    assert result["retrieval"]["semantic"]["reason"] == "VECTOR_INDEX_NOT_BUILT"


def test_empty_rebuild_replaces_previous_semantic_index():
    add_chunks("impeller")
    index.build(min_df=1, max_df_frac=1.0, verbose=False)
    conn = store.connect()
    try:
        with conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_fts")
    finally:
        conn.close()
    index.build(verbose=False)
    vi = index.VectorIndex()
    assert vi.status["reason"] == "EMPTY_CORPUS"
    assert vi.chunk_ids == []
    assert vi.search("impeller") == []


def test_semantic_search_excludes_nonpositive_and_nonfinite_scores(monkeypatch):
    vi = index.VectorIndex()
    vi.chunk_ids = ["opposite", "orthogonal", "positive", "closest", "invalid"]
    vi.vectors = np.array([[-1, 0], [0, 1], [0.5, 0.5], [1, 0], [np.nan, 0]], dtype=np.float32)
    monkeypatch.setattr(vi, "embed", lambda _: np.array([1, 0], dtype=np.float32))
    assert vi.search("synthetic query") == [("closest", 1.0), ("positive", 0.5)]
    assert vi.search("synthetic query", limit=1) == [("closest", 1.0)]


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limits_return_no_hits(limit):
    add_chunks("impeller")
    index.build(min_df=1, max_df_frac=1.0, verbose=False)
    assert index.VectorIndex().search("impeller", limit=limit) == []
    assert index.hybrid_search("impeller", limit=limit) == []
    assert api.semantic_search("impeller", limit=limit)["hits"] == []


@pytest.mark.parametrize("task,subsystem", [("pump", "pump"), ("npsh", "pump"), ("cavitation", "pump"),
                                          ("bearing", "pump"), ("shaft", "pump"), ("benchmark", "verification"),
                                          ("parity", "verification"), ("validation", "verification"),
                                          ("missing evidence", "verification")])
def test_subsystem_hints_include_original_and_extended_terms(task, subsystem):
    assert subsystem in api._infer_subsystems(task)


def implementation_graph():
    return graph.Graph({"nodes": [{"id": "EQ", "type": "Equation"}, {"id": "MODEL", "type": "Model"},
                                  {"id": "CLAIM", "type": "Claim"}, {"id": "VALIDITY", "type": "Claim"},
                                  {"id": "SW", "type": "SoftwareComponent", "file": "synthetic.py", "line": 4}],
                        "edges": [{"from": "MODEL", "to": "EQ", "type": "IMPLEMENTS", "provenance": "equation"},
                                  {"from": "SW", "to": "MODEL", "type": "IMPLEMENTS", "provenance": "model"},
                                  {"from": "SW", "to": "EQ", "type": "IMPLEMENTS", "provenance": "direct"},
                                  {"from": "CLAIM", "to": "MODEL", "type": "SUPPORTS", "provenance": "claim"},
                                  {"from": "VALIDITY", "to": "MODEL", "type": "CONSTRAINS", "provenance": "validity"}]})


@pytest.mark.parametrize("object_id", ["EQ", "MODEL", "CLAIM", "VALIDITY"])
def test_implementation_links_return_unique_software_with_correct_edge_direction(object_id, monkeypatch):
    g = implementation_graph()
    # Retain the cached function itself so fixture cache cleanup remains valid.
    monkeypatch.setattr(api, "Graph", lambda: g)
    result = api.find_implementation(object_id)
    assert result["state"] == "OK"
    assert [(item["id"], item["type"], item["file"]) for item in result["implementations"]] == [
        ("SW", "SoftwareComponent", "synthetic.py")]
    if object_id == "EQ":
        assert result["implementations"][0]["provenance"] == "direct"


def test_model_alone_is_not_reported_as_software(monkeypatch):
    g = implementation_graph()
    g = graph.Graph({"nodes": list(g.nodes.values()), "edges": [g.edges[0]]})
    monkeypatch.setattr(api, "Graph", lambda: g)
    result = api.find_implementation("EQ")
    assert result["state"] == "NO_IMPLEMENTATION_LINK"
    assert result["implementations"] == []


@pytest.mark.parametrize("column,value", [("text", "changed text"), ("page_start", 7),
                                         ("source_sha256", "changed-source-hash")])
def test_corpus_fingerprint_detects_same_count_changes(column, value):
    add_chunks("impeller pumping")
    meta = index.build(min_df=1, max_df_frac=1.0, verbose=False)
    conn = store.connect()
    try:
        assert index.corpus_fingerprint(conn) == meta["corpus_fingerprint"]
        with conn:
            conn.execute(f"UPDATE chunks SET {column} = ?", (value,))
        assert index.corpus_fingerprint(conn) != meta["corpus_fingerprint"]
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    finally:
        conn.close()


def save_model(**extra):
    store.save("models", [{"id": "MODEL", "registry_id": "fixture_model", "quantity": "impeller",
                           "subsystem": "pump", "relation": "fixture", **extra}])


@pytest.mark.parametrize("support", [None, "NOT_CHECKED", "PARTIALLY_SUPPORTED", "NOT_FOUND_IN_SOURCE"])
def test_task_context_preserves_model_audit_and_verification(support):
    model_audit = {"source_support_state": support} if support else {}
    save_model(**model_audit, source_support_notes="Synthetic audit note")
    store.save("equations", [{"id": "EQ", "label": "impeller", "document_id": "DOC_TEST",
                              "verification": "NEEDS_REVIEW"}])
    store.save("claims", [{"id": "CLM", "subject": "impeller", "document_id": "DOC_TEST",
                           "verification": "VERIFIED"}])
    result = api.retrieve_for_task("impeller")
    assert result["models"][0]["source_support_notes"] == "Synthetic audit note"
    if support:
        assert result["models"][0]["source_support_state"] == support
    assert result["equations"][0]["verification"] == "NEEDS_REVIEW"
    assert result["claims"][0]["verification"] == "VERIFIED"
    assert any(f"source_support_state={support or 'NOT_CHECKED'}" in warning
               for warning in result["implementation_warnings"])


def test_stale_records_are_excluded_but_remain_inspectable():
    save_model(source_support_state="NOT_CHECKED")
    stale = {"source_state": "STALE_SOURCE", "document_id": "DOC_TEST", "model_ids": ["MODEL"]}
    store.save("equations", [{"id": "EQ", "label": "impeller", **stale}])
    store.save("claims", [{"id": "CLM", "subject": "impeller", "claim_type": "validity_statement", **stale}])
    store.save("entities", [{"id": "ASSUME", "label": "impeller", "type": "Assumption", **stale}])
    store.save("validity", [{"id": "VAL", "prose": "impeller", "model_id": "MODEL", **stale}])
    assert api.find_equations("impeller")["equations"] == []
    assert api.find_claims("impeller")["claims"] == []
    assert api.find_validity_conditions("MODEL")["validity"] == []
    result = api.retrieve_for_task("impeller")
    for field in ("equations", "claims", "assumptions", "validity_envelopes"):
        assert result[field] == []
    assert any("equations=1, claims=1, entities=1, validity=1" in warning
               for warning in result["implementation_warnings"])
    assert api.get_equation("EQ")["source_state"] == "STALE_SOURCE"


def test_entity_search_excludes_stale_graph_nodes_but_direct_inspection_retains_them(monkeypatch):
    g = graph.Graph({"nodes": [{"id": "STALE", "type": "Assumption", "label": "impeller",
                               "source_state": "STALE_SOURCE"},
                              {"id": "CURRENT", "type": "Assumption", "label": "impeller"}], "edges": []})
    monkeypatch.setattr(api, "Graph", lambda: g)
    assert [n["id"] for n in api.find_entity("impeller", limit=1)["nodes"]] == ["CURRENT"]
    assert api.get_entity("STALE")["node"]["source_state"] == "STALE_SOURCE"


def test_validity_association_supports_plural_and_legacy_model_ids():
    save_model()
    store.save("validity", [{"id": "VAL_MULTI", "model_ids": ["OTHER", "MODEL"],
                             "model_id": "OTHER", "prose": "Shared limit"},
                            {"id": "VAL_LEGACY", "model_id": "MODEL", "prose": "Legacy limit"}])
    expected = {"VAL_MULTI", "VAL_LEGACY"}
    assert {v["id"] for v in api.find_validity_conditions("MODEL")["validity"]} == expected
    result = api.retrieve_for_task("impeller")
    assert {v["id"] for v in result["validity_envelopes"]} == expected
    multi = next(v for v in result["validity_envelopes"] if v["id"] == "VAL_MULTI")
    assert multi["model_ids"] == ["OTHER", "MODEL"]


@pytest.mark.parametrize('corruption', ['shape', 'json'])
def test_interrupted_vector_artifacts_fall_back_to_lexical(corruption):
    add_chunks('impeller pumping')
    index.build(min_df=1, max_df_frac=1.0, verbose=False)
    if corruption == 'shape':
        np.save(index.VECTOR_DIR / 'chunk_vectors.npy', np.zeros((5, 3)))
    else:
        (index.VECTOR_DIR / 'index_meta.json').write_text('{unfinished')
    result = api.semantic_search('impeller')
    assert result['hits'][0]['chunk_id'] == 'DOC_TEST#c0'
    assert result['retrieval']['semantic']['reason'] == 'VECTOR_INDEX_INCONSISTENT'


def test_refuted_assumptions_are_excluded_from_search_but_inspectable_by_id():
    """Regression (S3 report 2026-10-16): a refuted assumption was still returned by retrieval."""
    base = {"type": "Assumption", "document_id": "DOC_TEST", "page": 1, "source_state": "CURRENT",
            "provenance_kind": "SOURCE_STATES", "model_ids": []}
    store.save("entities", [
        {**base, "id": "ASM_REFUTED", "label": "single impeller motor drives both pumps",
         "prose": "single impeller motor drives both pumps", "verification": "REFUTED"},
        {**base, "id": "ASM_CURRENT", "label": "single impeller motor drives the pump",
         "prose": "single impeller motor drives the pump"}])
    graph.assemble()
    api.reset_caches()
    found = {n["id"] for n in api.find_entity("single impeller motor")["nodes"]}
    assert "ASM_CURRENT" in found and "ASM_REFUTED" not in found
    node = api.get_entity("ASM_REFUTED")
    assert node["node"]["verification"] == "REFUTED"
    package = api.retrieve_for_task("single impeller motor assumption")
    ids = {a["id"] for a in package["assumptions"]}
    assert "ASM_REFUTED" not in ids
