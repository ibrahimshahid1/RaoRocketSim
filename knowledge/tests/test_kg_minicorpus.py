"""Run the real KG rebuild/retrieval pipeline without local literature or JAX.

PDFs and derivative text below are authored synthetic fixtures. Only the external
project-science catalogs are replaced; sync, grounding, stores, chunking, vector
indexing, repository harvest, graph, validation and cache reset remain real.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg import (api, enrichment, graph, harvest_artifacts, harvest_constraints,
                harvest_extraction, harvest_registry, harvest_repo, index, ingest,
                pipeline, report, store, validate, implementation_freshness)
from kg.schema import node_id
from raosim import model_registry
from tools import sync_corpus


def _pdf(text: str | None) -> bytes:
    """A valid small PDF, including a deliberate zero-page source, using stdlib."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    if text is None:
        objects.append(b"<< /Type /Pages /Count 0 /Kids [] >>")
    else:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = ("BT /F1 10 Tf 30 700 Td (" + escaped + ") Tj ET").encode("ascii")
        objects += [b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
                    b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
                    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(payload)
    payload += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    payload += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    payload += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(payload)


@pytest.fixture
def minicorpus(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    corpus = tmp_path / "propulsion_texts"
    agents = corpus / "propulsion_texts_for_agents"
    markdown = agents / "markdown"
    markdown.mkdir(parents=True)
    vector = knowledge / "vector_index"
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    for name, value in (("KNOWLEDGE_ROOT", knowledge), ("VECTOR_DIR", vector),
                        ("CHUNK_DB", vector / "chunks.sqlite"),
                        ("GRAPH_DIR", knowledge / "graph"), ("REPORT_DIR", knowledge / "reports")):
        monkeypatch.setattr(store, name, value)
    for module in (api, enrichment, harvest_registry, harvest_repo, harvest_constraints, ingest, validate, implementation_freshness):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ingest, "CORPUS_ROOT", corpus)
    monkeypatch.setattr(ingest, "AGENT_CORPUS_ROOT", agents)
    monkeypatch.setattr(validate, "CORPUS_ROOT", corpus)
    monkeypatch.setattr(validate, "AGENT_CORPUS_ROOT", agents)
    monkeypatch.setattr(validate, "VECTOR_DIR", vector)
    monkeypatch.setattr(validate, "REPORT_DIR", store.REPORT_DIR)
    monkeypatch.setattr(index, "VECTOR_DIR", vector)
    monkeypatch.setattr(graph, "GRAPH_DIR", store.GRAPH_DIR)
    monkeypatch.setattr(report, "REPORT_DIR", store.REPORT_DIR)
    for name, value in (("REPO", tmp_path), ("CORPUS", corpus), ("AGENTS", agents),
                        ("MARKDOWN", markdown), ("MANIFEST", agents / "_conversion_manifest.json"),
                        ("VALIDATION", agents / "_validation_report.json")):
        monkeypatch.setattr(sync_corpus, name, value)
    # Replace scientific inputs, not the harvesters or deterministic pipeline.
    model = SimpleNamespace(
        local_source="propulsion_texts/fixture_0.pdf", source="Synthetic authored fixture",
        subsystem="verification", quantity="synthetic transport", relation="flux = area * speed",
        equation_ref="synthetic relation on page 1", validity="Synthetic test only",
        status="published_equation", verification="synthetic unit test",
        validation_level="not_physically_validated", notes="No real source science is asserted.")
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {"fixture.transport": model})
    monkeypatch.setattr(harvest_constraints, "_load_manifest", lambda: ([], "synthetic test catalog"))
    original_artifact_harvest = harvest_artifacts.harvest
    monkeypatch.setattr(harvest_artifacts, "harvest", lambda: original_artifact_harvest(repo_root=tmp_path))
    quote = "Synthetic transport relation: flux = area * speed."
    rows = []
    for number in range(7):
        # The first two groups each have exactly three chunks: frequency filters
        # retain real semantic terms at min_df=3 and max_df_frac=0.5.
        words = ("alpha calibration invariant conservative transport" if number < 3 else
                 "beta geometry interpolation reference sampling")
        text = f"Synthetic document {number}. {words}. {quote} Authored only for deterministic pipeline testing."
        source = _pdf(text if number < 6 else None)
        filename = f"fixture_{number}.pdf"
        (corpus / filename).write_bytes(source)
        (markdown / f"fixture_{number}.md").write_text(
            "<!-- PDF page 1 -->\n\n# Synthetic fixture\n\n" + text + "\n" if number < 6 else
            "Synthetic zero-page placeholder; no PDF source content.\n")
        rows.append({"source_pdf": filename, "output_markdown": f"markdown/fixture_{number}.md",
                     "sha256": hashlib.sha256(source).hexdigest(), "source_bytes": len(source),
                     "source_kind": "file", "page_count": int(number < 6),
                     "method": "authored-synthetic-fixture", "extraction_status": "good",
                     "pdf_title": f"Synthetic fixture {number}", "issues": []})
    (agents / "_conversion_manifest.json").write_text(json.dumps(rows))
    code = tmp_path / "raosim" / "toy.py"
    code.parent.mkdir()
    code.write_text('SOURCE = "propulsion_texts/fixture_0.pdf"\n\ndef transport(area, speed):\n    return area * speed\n')
    (code.parent / "mdo").mkdir()
    (code.parent / "mdo" / "constraints.py").write_text("CONSTRAINT_MANIFEST = ()  # Synthetic catalog; no engineering constraints.\n")
    api.reset_caches()
    yield {"root": tmp_path, "quote": quote}
    api.reset_caches()


def test_full_synthetic_rebuild_preserves_banked_review_and_source_routes(minicorpus, capsys):
    assert pipeline.main([]) == 0
    first = json.loads((store.REPORT_DIR / "latest_ingestion.json").read_text())
    assert first["steps"]["index"]["state"] == "OK"
    assert first["steps"]["index"]["k"] > 0
    empty = next(r for r in first["document_states"] if r["doc_id"] == "DOC_FIXTURE_6")
    assert empty["state"] == "SOURCE_UNREADABLE" and empty["reason"] == "EMPTY_SOURCE"
    assert validate.status()["documents"]["chunked"] == 6
    assert validate.status()["documents"]["doc_state_entries"] == 7
    assert api.find_equations("synthetic")["equations"] == []  # warm the empty cache

    model_id = node_id("MODEL", "fixture.transport")
    extraction = {"target": model_id, "doc_id": "DOC_FIXTURE_0",
                  "supports_model": {"state": "SUPPORTED", "equation_labels": ["synthetic transport"]},
                  "equations": [{"label": "synthetic transport", "form": "flux = area * speed",
                                 "description": "Authored toy equation; no physical qualification.",
                                 "quote": minicorpus["quote"], "page": 1, "confidence": "high"}]}
    assert harvest_extraction.harvest([extraction], [])["n_equations"] == 1
    equation, = store.load("equations")
    eq_id = equation["id"]
    assert harvest_extraction.apply_verification([{
        "object_id": eq_id, "refuted": False, "confidence": "high",
        "source_sha256": equation["source_sha256"],
        "reason": "Matches the independently authored synthetic fixture; test evidence only.",
    }])["confirmed"] == 1
    store.upsert("software_links", [{"id": "SW_TOY", "type": "SoftwareComponent",
                                     "qualname": "raosim.toy.transport", "file": "raosim/toy.py", "line": 3,
                                     "model_ids": [model_id], "equation_ids": [eq_id],
                                     "cited_document_ids": ["DOC_FIXTURE_0"], "provenance_id": "PROV_TOY"}])
    store.upsert("provenance", [{"id": "PROV_TOY", "kind": "CODE_STATES", "subject": "SW_TOY",
                                "asserted_by": "raosim/toy.py:3", "statement": "Synthetic test implementation."}])
    # Rebanking and rebuilds must retain stable identity and reviewed evidence.
    harvest_extraction.harvest([extraction], [])
    for _ in range(2):
        assert pipeline.main([]) == 0
        rebuilt = json.loads((store.REPORT_DIR / "latest_ingestion.json").read_text())
        assert rebuilt["steps"]["index"] == {"skipped": "no chunk changes"}
        stored, = store.load("equations")
        assert stored["id"] == eq_id and stored["verification"] == "CONFIRMED"
        assert store.load("models")[0]["source_support_state"] == "SUPPORTED"
        assert api.find_equations("synthetic")["equations"][0]["id"] == eq_id
        g = graph.Graph()
        assert g.meta["n_dangling_dropped"] == 0
        assert all(e["from"] in g.nodes and e["to"] in g.nodes for e in g.edges)
        assert any(e["type"] == "SUPPORTED_BY" and e["from"] == model_id for e in g.edges)
        basis = api.find_scientific_basis("raosim/toy.py")
        toy = next(c for c in basis["components"] if c["component"]["id"] == "SW_TOY")
        assert toy["supported_source_documents"][0]["id"] == "DOC_FIXTURE_0"

    search = api.semantic_search("alpha calibration")
    assert search["hits"] and search["retrieval"]["mode"] == "HYBRID"
    context = api.get_source_context(equation["chunk_id"])
    assert context["escalation"]["pdf"] == "propulsion_texts/fixture_0.pdf"
    assert context["escalation"]["pages"] == [1, 1]
    assert any(e["type"] == "CITES" and e["to"] == "DOC_FIXTURE_0" for e in graph.Graph().edges)
    # A missing vector artifact must yield a working, explicitly labelled fallback.
    (store.VECTOR_DIR / "components.npy").unlink()
    api.reset_caches()
    fallback = api.semantic_search("alpha calibration")
    assert fallback["hits"] and fallback["retrieval"]["mode"] == "LEXICAL_ONLY"
    assert fallback["retrieval"]["semantic"]["state"] == "UNAVAILABLE"
    capsys.readouterr()
