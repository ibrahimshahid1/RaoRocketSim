"""Tests for the LREKit knowledge system.

Run:  .venv/bin/python -m pytest knowledge/tests -q

Two layers:
* deterministic-unit tests (chunker, grounding, symbol resolution, sync scan, stores)
* acceptance queries from the master instructions (§46): each must return a
  provenance-backed, id-addressable context package - or an explicit state
  (never a silent empty list).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

KNOWLEDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KNOWLEDGE))
sys.path.insert(0, str(KNOWLEDGE.parent))

from kg import api, store  # noqa: E402
from kg.chunker import chunk_markdown  # noqa: E402
from kg.ground import DocGrounder, normalize  # noqa: E402
from kg.quantities import resolve_symbol  # noqa: E402
from kg.schema import ENGINEERING_STATES, RETRIEVAL_STATES, SCHEMA  # noqa: E402

pytestmark = pytest.mark.skipif(not (KNOWLEDGE / "vector_index" / "chunks.sqlite").exists(),
                                reason="knowledge stores not built (run knowledge/kg.sh rebuild)")


# --------------------------------------------------------------------------
# deterministic layers
# --------------------------------------------------------------------------

def test_chunker_respects_pages_headings_and_tables():
    md = ("<!-- PDF page 1 -->\n\n# Intro\n\nPara one about thrust.\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n<!-- PDF page 2 -->\n\n## Method\n\nq = h_g (T_aw - T_wg) is the heat flux.\n"
          + ("More text. " * 40) + "\n")
    chunks = chunk_markdown(md, "DOC_T")
    assert chunks, "no chunks"
    assert all(c.chunk_id.startswith("DOC_T#c") for c in chunks)
    pages = {c.page_start for c in chunks}
    assert pages == {1, 2}
    assert any(c.has_table for c in chunks)
    assert any("Method" in c.section for c in chunks)
    for c in chunks:
        assert c.page_start <= c.page_end
        assert c.char_start < c.char_end


def test_symbol_resolution_preserves_ambiguity():
    assert resolve_symbol("k")["state"] == "AMBIGUOUS_SYMBOL"
    assert {c["qid"] for c in resolve_symbol("k")["candidates"]} >= {"QTY_GAMMA", "QTY_KCOND"}
    assert resolve_symbol("Pc")["state"] == "RESOLVED"
    assert resolve_symbol("k", subsystem="heat_transfer")["candidates"][0]["qid"] == "QTY_KCOND"
    assert resolve_symbol("zz_not_a_symbol")["state"] == "UNRESOLVED_ENTITY"


def test_grounding_rejects_unanchored_text():
    docs = store.load("manifest")
    doc = next(d for d in docs if d["id"] == "DOC_DOI_10_2514_8_7324")  # Rao 1958
    g = DocGrounder(doc["id"])
    assert g.chunks
    real = g.chunks[5]["text"].strip().splitlines()[0][:120]
    if len(normalize(real)) >= 12:
        assert g.locate(real, g.chunks[5]["page_start"])["state"].startswith("GROUNDED")
    assert g.locate("this sentence was never in any propulsion paper whatsoever", 2)["state"] == "UNGROUNDED"
    assert g.locate("short", 2)["state"] == "UNGROUNDED"


def test_every_stored_scientific_object_is_grounded_and_has_provenance():
    prov = {p["id"] for p in store.load("provenance")}
    docs = {d["id"] for d in store.load("manifest")}
    for name in ("equations", "claims"):
        for rec in store.load(name):
            assert rec["grounded"] in ("GROUNDED", "GROUNDED_PAGE_SHIFTED"), rec["id"]
            assert rec["provenance_id"] in prov, rec["id"]
            assert rec["document_id"] in docs, rec["id"]
            assert rec["quote"] and rec["page"]


def test_manifest_dedupes_payloads_and_alternate_copies():
    docs = store.load("manifest")
    shas = [d["sha256"] for d in docs]
    assert len(shas) == len(set(shas)), "duplicate payload registered twice"
    for d in docs:
        if d.get("alternate_copy_of"):
            assert d["alternate_copy_of"] in {x["id"] for x in docs}
        assert d["conversion"]["status"] in SCHEMA["conversion_states"] or d["conversion"]["status"] is None


def test_models_never_claim_hardware_qualification():
    for m in store.load("models"):
        assert m["hardware_qualified"] is False
        assert m["basis_class"] in SCHEMA["basis_classes"]
        assert m["promotion_status"] in SCHEMA["model_promotion"]


def test_constraint_kinds_are_not_flattened():
    reqs = store.load("requirements")
    kinds = {r["kind"] for r in reqs}
    assert {"optimizer_constraint", "postsolve_gate", "applicability_gate"} <= kinds
    for r in reqs:
        assert r["kind"] in SCHEMA["restriction_kinds"]
        if r["kind"] == "optimizer_constraint":
            assert r["optimizer_role"] == "hard"
        if r["kind"] in ("postsolve_gate", "applicability_gate"):
            assert r["optimizer_role"] == "final_gate"


def test_software_components_point_at_real_definitions():
    root = KNOWLEDGE.parent
    comps = [s for s in store.load("software_links") if s.get("type") == "SoftwareComponent"][:200]
    for c in comps:
        p = root / c["file"]
        assert p.exists(), c["file"]
        if c["qualname"].endswith("<module>"):
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        sym = c["qualname"].split(":")[-1].split(".")[-1]
        line = lines[c["line"] - 1]
        assert sym in line, f"{c['id']}: {sym!r} not on {c['file']}:{c['line']}"


def test_unknown_basis_constants_are_labelled_not_promoted():
    consts = [s for s in store.load("software_links") if s.get("type") == "EmpiricalConstant"]
    assert consts
    for k in consts:
        assert k["basis_class"] in SCHEMA["basis_classes"]
        # UNKNOWN_BASIS constants may carry a verbatim comment (context), but the
        # classification is the agent's verified judgement that the numeric value
        # itself has no cited source; the value is never upgraded here.
        assert k["verified"] is True


# --------------------------------------------------------------------------
# acceptance queries (§46) - each must be provenance-backed or explicit
# --------------------------------------------------------------------------

ACCEPTANCE_TASKS = [
    "What equations and source passages support the Rao contour methods currently present in LREKit?",
    "What assumptions differ between the TOP approximation and the variational Rao solver?",
    "What source and criterion determine whether a Rao solution is physically valid?",
    "Which LREKit functions implement the MOC characteristic relations?",
    "What equations determine gas-side heat flux, coolant-side heat transfer, and wall temperature?",
    "Which cooling correlations are empirical and what are their validity conditions?",
    "How does cooling-jacket pressure drop propagate into pump sizing?",
    "What differs between the traditional and differentiable cooling implementations?",
    "What literature supports the pintle injector hydraulic model?",
    "Which discharge coefficients are literature values versus calibrated quantities?",
    "What conditions invalidate the injector model?",
    "What literature supports the current pump meanline relations?",
    "What assumptions govern inducer NPSH calculations?",
    "Which quantities are optimizer-enforced constraints versus final gates?",
    "Which JAX models have NumPy reference oracles and parity tests?",
    "Which current fidelity gaps are intentionally unavailable?",
    "What evidence would be required before a generated pump component could be described as machineable?",
    "What is needed to bind a CAD artifact to a resolved engine configuration?",
    "What literature is still missing before implementing the bearing seal rotordynamics work?",
    "What published benchmarks exist for the nozzle MOC model?",
]

ALLOWED_STATES = set(RETRIEVAL_STATES) | set(ENGINEERING_STATES) | {"OK"}


@pytest.mark.parametrize("task", ACCEPTANCE_TASKS)
def test_retrieve_for_task_is_source_backed(task):
    ctx = api.retrieve_for_task(task)
    assert ctx["state"] in ALLOWED_STATES
    assert ctx["subsystems"], f"no subsystem inferred for {task!r}"
    assert ctx["source_context"], "no source chunks"
    docs = {d["id"] for d in store.load("manifest")}
    for h in ctx["source_context"]:
        assert h["doc_id"] in docs and h["chunk_id"] and h["page_start"] >= 1
    for e in ctx["equations"]:
        assert e["id"].startswith("EQ_") and e["document_id"] in docs and e["page"]
    for c in ctx["claims"]:
        assert c["id"].startswith("CLM_") and c["document_id"] in docs
    for m in ctx["models"]:
        assert m["basis_class"] in SCHEMA["basis_classes"]
    # implementations must be real file:line pairs
    for s in ctx["existing_implementations"]:
        assert (KNOWLEDGE.parent / s["file"]).exists()
    assert isinstance(ctx["implementation_warnings"], list)


def test_optimizer_vs_gate_question_answers_from_manifest():
    ctx = api.retrieve_for_task("Which quantities are optimizer-enforced constraints versus final gates?")
    kinds = {r["kind"] for r in ctx["requirements_and_gates"]}
    assert "optimizer_constraint" in kinds and ("postsolve_gate" in kinds or "applicability_gate" in kinds)


def test_jax_parity_question_finds_oracle_links():
    ctx = api.retrieve_for_task("Which JAX models have NumPy reference oracles and parity tests?")
    sw = [s for s in ctx["existing_implementations"] if s.get("variant_of_id")]
    assert sw, "no JAX components with variant_of_id"


def test_scientific_basis_of_registry_backed_component():
    r = api.find_scientific_basis("raosim/physics.py")
    assert r["state"] == "OK"
    assert any(c["models"] and c["source_documents"] for c in r["components"]), "no registry citation route for physics.py"
    for component in r["components"]:
        if component["state"] == "OK":
            assert component["model_support"]
            assert all(s["state"] == "SUPPORTED" for s in component["model_support"])


def test_no_validity_is_reported_not_invented():
    r = api.find_validity_conditions("MODEL_THAT_DOES_NOT_EXIST")
    assert r["state"] == "NO_VALIDITY_RANGE_LOCATED"
    assert r["validity"] == []
    assert "not evidence that none exists" in r["note"]


def test_unknown_document_is_explicit():
    assert api.get_document("DOC_NOT_INGESTED")["state"] == "SOURCE_NOT_INGESTED"
    assert api.get_source_context("DOC_NOT_INGESTED#c00001")["state"] == "INSUFFICIENT_SOURCE_CONTEXT"


def test_source_escalation_path_reaches_pdf():
    eqs = store.load("equations")
    if not eqs:
        pytest.skip("no extracted equations yet")
    ctx = api.get_source_context(eqs[0]["chunk_id"])
    assert ctx["escalation"]["pdf"].endswith(".pdf")
    assert ctx["escalation"]["pages"][0] >= 1
    assert "authoritative" in ctx["escalation"]["note"]
