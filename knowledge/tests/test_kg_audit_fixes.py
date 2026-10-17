"""Regression tests for the 2026-10-16 independent audit (Opus A findings A-01..A-11, N-04).

Corpus-free: stores, graph and code live in a temporary tree; grounding is stubbed
as in test_kg_review_provenance.py.
"""

import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import api, graph, harvest_extraction, implementation_freshness, index, locks, store  # noqa: E402
from kg.ground import form_key  # noqa: E402
from tools import agentic_packet  # noqa: E402

MODEL = {"id": "MODEL_TEST", "registry_id": "fixture.gas_side_coefficient", "subsystem": "nozzle",
         "quantity": "gas-side coefficient", "relation": "h = 0.026 x^0.8", "basis_class": "LITERATURE_DERIVED",
         "promotion_status": "screening", "validation_level": "not_physically_validated",
         "registry_status": "available", "source_document_id": "DOC_TEST"}
REVIEWER = {"role": "F", "model": "opus"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "knowledge"
    vector_dir = root / "vector_index"
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(store, "CHUNK_DB", vector_dir / "chunks.sqlite")
    monkeypatch.setattr(index, "VECTOR_DIR", vector_dir)
    monkeypatch.setattr(graph, "GRAPH_DIR", root / "graph")
    monkeypatch.setattr(implementation_freshness, "REPO_ROOT", tmp_path)

    class Grounder:
        def __init__(self, doc_id):
            self.chunks = [{"chunk_id": "DOC_TEST#c1", "source_sha256": "src"}]

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "DOC_TEST#c1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "DocGrounder", Grounder)
    monkeypatch.setattr(harvest_extraction, "_current_grounder", lambda doc_id, sha: Grounder(doc_id))
    store.save("manifest", [{"id": "DOC_TEST", "filename": "test.pdf", "source_pdf": "propulsion_texts/test.pdf",
                             "title": "Synthetic source", "sha256": "src", "conversion": {"status": "good"}}])
    store.save("models", [dict(MODEL)])
    code = tmp_path / "pkg" / "physics.py"
    code.parent.mkdir()
    code.write_text("def h(x):\n    return 0.026 * x ** 0.8\n\n\ndef mu(t):\n    return 1.18e-7 * t ** 0.6\n")
    api.reset_caches()
    yield tmp_path
    api.reset_caches()


def _eq(number, form, quote=None):
    return {"label": f"Fixture Eq. ({number}) long descriptive label", "equation_number": number, "page": 3,
            "quote": quote or f"anchor sentence for equation {number} on the page", "form": form, "confidence": "high"}


def _model_result(fidelity, labels, comparison, equations=()):
    return {"doc_id": "DOC_TEST", "target": "MODEL_TEST", "source_sha256": "src", "reviewer": REVIEWER,
            "supports_model": {"found": True, "state": "SUPPORTED", "equation_labels": labels,
                               "comparison_equation_labels": comparison},
            "model_review": {"fidelity": fidelity, "implementation": [{"file": "pkg/physics.py", "symbol": "h"}],
                             "source_pages": [3]},
            "equations": list(equations), "claims": []}


def _window(*equations):
    return {"doc_id": "DOC_TEST", "target": "pages3-3", "source_sha256": "src", "reviewer": REVIEWER,
            "equations": list(equations), "claims": []}


def _ids_by_number():
    return {e["equation_number"]: e for e in store.load("equations")}


def test_short_review_tags_resolve_to_printed_numbers_and_unmatched_labels_are_reported(env):
    """A-01(a): 'Eq. (1)' must select the stored long-labelled Eq. (1); 'Table 2' must be reported."""
    harvest_extraction.harvest([], [_window(_eq("2", "q = h*(T_aw - T_w)"))])
    report = harvest_extraction.harvest([_model_result("EQUIVALENT", ["Eq. (1)", "Eq. (2)", "Table 2"],
                                                       ["Eq. (1)", "Eq. (2)"], [_eq("1", "h = 0.026*x^0.8")])], [])
    eqs = _ids_by_number()
    assert eqs["1"]["model_ids"] == ["MODEL_TEST"] and eqs["2"]["model_ids"] == ["MODEL_TEST"]
    assert eqs["1"]["model_link_basis"] == "model_review:EQUIVALENT"
    assert [u["label"] for u in report["unresolved_support_links"]] == ["Table 2"]
    model = store.index_by("models")["MODEL_TEST"]
    assert sorted(model["source_comparison_equation_ids"]) == sorted([eqs["1"]["id"], eqs["2"]["id"]])
    assert "unresolved_comparison_links" not in report


def test_ambiguous_printed_number_is_reported_not_guessed(env):
    harvest_extraction.harvest([], [_window(_eq("29", "m = P/dP", "first anchor for twenty nine"),
                                            _eq("29", "m = P_out/(dP*eta)", "second anchor for twenty nine"))])
    report = harvest_extraction.harvest([_model_result("EQUIVALENT", ["Eqs. (26)-(29)"], ["Eq. (29)"])], [])
    assert all(not e.get("model_ids") for e in store.load("equations"))
    unresolved = " ".join(" ".join(u["unresolved"]) for u in report["unresolved_support_links"])
    assert "Eq. 29: 2 current matches" in unresolved and "Eq. 26: 0 current matches" in unresolved
    assert report["unresolved_comparison_links"][0]["label"] == "Eq. (29)"


def test_different_formula_review_cites_but_never_implements(env):
    """A-01(b): a DIFFERENT_FORMULA review creates CITES only; naming support labels is refused."""
    report = harvest_extraction.harvest([_model_result("DIFFERENT_FORMULA", ["Eq. (1)"], ["Eq. (1)"],
                                                       [_eq("1", "Nu = C1*Re^0.8*Pr^0.4")])], [])
    eq = _ids_by_number()["1"]
    assert not eq.get("model_ids")
    assert report["support_labels_refused"][0]["model_id"] == "MODEL_TEST"
    assembled = graph.assemble()
    kinds = {e["type"] for e in assembled["edges"] if e["from"] == "MODEL_TEST" and e["to"] == eq["id"]}
    assert kinds == {"CITES"}


def test_review_state_is_visible_in_task_context_and_flips_after_a_code_change(env):
    """A-01(c): fidelity and current review state reach graph nodes, model lookups,
    find_implementation and retrieve_for_task; a structural code edit flips the state."""
    tmp_path = env
    harvest_extraction.harvest([_model_result("EQUIVALENT", ["Eq. (1)"], ["Eq. (1)"], [_eq("1", "h = 0.026*x^0.8")])], [])
    eq = _ids_by_number()["1"]
    assembled = graph.assemble()
    api.reset_caches()
    node = {n["id"]: n for n in assembled["nodes"]}["MODEL_TEST"]
    assert node["source_review_state"] == "CURRENT" and node["source_review_fidelity"] == "EQUIVALENT"
    assert {e["type"] for e in assembled["edges"] if e["from"] == "MODEL_TEST" and e["to"] == eq["id"]} >= {"IMPLEMENTS", "CITES"}
    assert api.get_model("MODEL_TEST")["source_review_state"]["state"] == "CURRENT"
    found = api.find_implementation(eq["id"])
    assert found["state"] == "OK"
    assert [(r["file"], r["symbol"], r["fidelity"], r["review_state"]) for r in found["reviewed_implementations"]] == \
        [("pkg/physics.py", "h", "EQUIVALENT", "CURRENT")]
    package = api.retrieve_for_task("fixture gas-side coefficient")
    model, = [m for m in package["models"] if m["id"] == "MODEL_TEST"]
    assert model["source_review"]["state"] == "CURRENT" and model["source_review"]["fidelity"] == "EQUIVALENT"
    (tmp_path / "pkg" / "physics.py").write_text("def h(x):\n    return 0.023 * x ** 0.8\n\n\ndef mu(t):\n    return 1.18e-7 * t ** 0.6\n")
    api.reset_caches()
    assert api.get_model("MODEL_TEST")["source_review_state"]["state"] == "IMPLEMENTATION_CHANGED_REVIEW_REQUIRED"
    package = api.retrieve_for_task("fixture gas-side coefficient")
    assert any("IMPLEMENTATION_CHANGED_REVIEW_REQUIRED" in w for w in package["implementation_warnings"])


def test_duplicate_forms_differing_only_in_spacing_are_detected_and_retired_with_history(env):
    """A-02: whitespace around operators does not make a second transcription distinct."""
    assert form_key("m_inv = P_inv_out/delta_P_inv") == form_key("m_inv = P_inv_out / delta_P_inv")
    assert form_key("m = P/dP") != form_key("m = P/dQ")
    store.save("enrichments", [{"id": "CAND_X", "type": "ExtractionCandidate", "candidate_kind": "formula",
                                "document_id": "DOC_TEST", "source_sha256": "src", "pages": [3],
                                "source_state": "CURRENT", "location_state": "LOCATED", "active": True}])
    harvest_extraction.harvest([], [_window(_eq("29", "m_inv = P_inv_out / delta_P_inv", "legacy anchor for the inverter mass"),
                                            {**_eq("29", "m_inv = P_inv_out/delta_P_inv", "second anchor for the inverter mass"),
                                             "enrichment_candidate_ids": ["CAND_X"]})])
    canon, dup = sorted(store.load("equations"), key=lambda e: "second" in e["quote"])
    models = store.load("models")
    models[0]["source_comparison_equation_ids"] = [dup["id"]]
    store.save("models", models)
    payload = {"reviewer": REVIEWER, "duplicate_retirements": [
        {"duplicate_id": dup["id"], "canonical_id": canon["id"], "reason": "same page, number and form"}]}
    with pytest.raises(ValueError):
        harvest_extraction.retire_duplicate_equations({**payload, "reviewer": {"role": "S3", "model": "sonnet"}})
    stats = harvest_extraction.retire_duplicate_equations(payload)
    assert stats["retired"] == 1 and not stats["refused"]
    rows = store.index_by("equations")
    assert rows[dup["id"]]["active"] is False and rows[dup["id"]]["duplicate_of"] == canon["id"]
    assert rows[canon["id"]]["enrichment_candidate_ids"] == ["CAND_X"]
    assert store.index_by("models")["MODEL_TEST"]["source_comparison_equation_ids"] == [canon["id"]]
    again = harvest_extraction.retire_duplicate_equations(payload)
    assert again["unchanged"] == 1 and again["retired"] == 0 and again["stores_saved"] is False


def test_retirement_refuses_records_that_differ_beyond_whitespace(env):
    harvest_extraction.harvest([], [_window(_eq("5", "a = b + c", "first anchor for equation five"),
                                            _eq("5", "a = b - c", "second anchor for equation five"))])
    first, second = store.load("equations")
    stats = harvest_extraction.retire_duplicate_equations({"reviewer": REVIEWER, "duplicate_retirements": [
        {"duplicate_id": first["id"], "canonical_id": second["id"], "reason": "x"}]})
    assert stats["retired"] == 0 and "forms differ beyond whitespace" in stats["refused"][0]["reasons"]


def test_supersession_requires_a_recorded_carry_forward_decision(env):
    """A-03: a correction may not silently drop the superseded record's assumptions, validity or model links."""
    old = {"id": "EQ_TEST_OLD", "type": "Equation", "document_id": "DOC_TEST", "equation_number": "4-7", "page": 3,
           "form": "Ln = garbled", "confidence": "low", "source_sha256": "src", "source_state": "CURRENT",
           "extraction_note": "Equation OCR is garbled.", "model_ids": ["MODEL_TEST"],
           "assumptions": ["conical nozzle contour"], "validity": ["R from 0.5 to 1.5 R_t"]}
    store.save("equations", [old])
    fixed = {**_eq("4-7", "L_n = [R_t*(sqrt(eps) - 1) + R*(sec(a) - 1)]/tan(a)"), "supersedes_ids": ["EQ_TEST_OLD"]}
    report = harvest_extraction.harvest([], [_window(fixed)])
    undecided, = report["supersession_carry_forward_undecided"]
    assert undecided["old_id"] == "EQ_TEST_OLD"
    assert set(undecided["dropped"]) == {"assumptions", "validity", "model_ids"}
    decided = {**fixed, "description": "re-banked with a decision",
               "carry_forward": {"assumptions": True, "validity": False, "model_ids": False,
                                 "reason": "assumption page-verified; ranges are typical designs; model does not implement it"}}
    report = harvest_extraction.harvest([], [_window(decided)])
    assert "supersession_carry_forward_undecided" not in report
    new = store.index_by("equations")[store.index_by("equations")["EQ_TEST_OLD"]["superseded_by"]]
    assert new["assumptions"] == ["conical nozzle contour"] and not new.get("validity") and not new.get("model_ids")
    assert new["carry_forward_decisions"][0]["from_id"] == "EQ_TEST_OLD"


def test_refuted_record_is_replaceable_by_a_reviewed_correction(env):
    """N-04: the REFUTED branch of supersession (repair seq 51)."""
    store.save("equations", [{"id": "EQ_TEST_REF", "type": "Equation", "document_id": "DOC_TEST", "equation_number": "9",
                              "page": 3, "form": "x = y^2", "confidence": "high", "source_sha256": "src",
                              "source_state": "CURRENT", "verification": "REFUTED"}])
    report = harvest_extraction.harvest([], [_window({**_eq("9", "x = y^3"), "supersedes_ids": ["EQ_TEST_REF"]})])
    rows = store.index_by("equations")
    assert report["superseded_incomplete_transcriptions"] == 1
    assert rows["EQ_TEST_REF"]["active"] is False and rows["EQ_TEST_REF"]["source_state"] == "SUPERSEDED_EXTRACTION"


def test_refuted_validity_is_excluded_from_task_context_but_inspectable(env):
    """N-04: the validity branch of repair seq 345."""
    base = {"type": "ValidityEnvelope", "document_id": "DOC_TEST", "page": 3, "source_state": "CURRENT",
            "provenance_kind": "SOURCE_STATES", "model_ids": ["MODEL_TEST"]}
    store.save("validity", [{**base, "id": "VAL_REFUTED", "prose": "valid for all expansion ratios", "verification": "REFUTED"},
                            {**base, "id": "VAL_CURRENT", "prose": "valid for expansion ratios up to 50"}])
    graph.assemble()
    api.reset_caches()
    ids = {v["id"] for v in api.find_validity_conditions("MODEL_TEST")["validity"]}
    assert "VAL_CURRENT" in ids and "VAL_REFUTED" not in ids
    package = api.retrieve_for_task("fixture gas-side coefficient validity")
    assert "VAL_REFUTED" not in {v.get("id") for v in package["validity_envelopes"]}
    assert store.index_by("validity")["VAL_REFUTED"]["verification"] == "REFUTED"


def test_unchanged_verdict_replay_does_not_rewrite_the_stores(env):
    """A-10: a replay of an applied verdict leaves every store byte-identical."""
    harvest_extraction.harvest([], [_window(_eq("1", "h = 0.026*x^0.8"))])
    eq = store.load("equations")[0]
    verdict = [{"object_id": eq["id"], "refuted": False, "confidence": "high", "reason": "page check",
                "source_sha256": "src", "reviewer": {"role": "A", "model": "auditor"}}]
    assert harvest_extraction.apply_verification(verdict)["confirmed"] == 1
    paths = [store.store_path(n) for n in ("equations", "claims", "entities", "validity", "models")]
    before = {p: p.stat().st_mtime_ns for p in paths if p.exists()}
    time.sleep(0.01)
    stats = harvest_extraction.apply_verification(verdict)
    assert stats["unchanged"] == 1 and stats["stores_saved"] is False
    assert {p: p.stat().st_mtime_ns for p in paths if p.exists()} == before


def test_atomic_store_write_keeps_the_published_file_mode(env):
    """A-05: atomic publication must not narrow a store to 0600."""
    store.save("claims", [{"id": "A"}])
    path = store.store_path("claims")
    os.chmod(path, 0o644)
    store.save("claims", [{"id": "B"}])
    assert path.stat().st_mode & 0o777 == 0o644
    fresh = store.store_path("validity")
    assert not fresh.exists()
    store.save("validity", [])
    assert fresh.stat().st_mode & 0o777 == 0o666 & ~store._UMASK


def _hold_writer(lock_root, ready, seconds):
    from kg import locks as child_locks, store as child_store
    child_store.KNOWLEDGE_ROOT = Path(lock_root)
    with child_locks.writer(timeout=5, purpose="test holder"):
        ready.set()
        time.sleep(seconds)


def test_store_writer_lock_blocks_a_second_process_upsert(env):
    """A-06: while another process holds store-writer, upsert waits (or refuses at its timeout)."""
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_hold_writer, args=(str(store.KNOWLEDGE_ROOT), ready, 1.5))
    child.start()
    try:
        assert ready.wait(10)
        with pytest.raises(locks.LockBusyError):
            with locks.writer(timeout=0.2):
                pass
        start = time.monotonic()
        store.upsert("claims", [{"id": "AFTER_RELEASE"}])
        assert time.monotonic() - start >= 0.5
    finally:
        child.join(10)
    assert [r["id"] for r in store.load("claims")] == ["AFTER_RELEASE"]


def test_store_writer_lock_serializes_threads_of_one_process(env):
    """A-06: reentrancy belongs to the holding thread; another thread waits."""
    held, release = threading.Event(), threading.Event()

    def holder():
        with locks.writer(timeout=5):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(5)
        with pytest.raises(locks.LockBusyError):
            with locks.writer(timeout=0.2):
                pass
    finally:
        release.set()
        thread.join(5)
    with locks.writer(timeout=5):
        with locks.writer(timeout=5):  # nested in the same thread still re-enters
            pass


def test_s3_integrity_failures_include_dangling_edges():
    """A-07: dangling edges block S3 integration and testing like the other invariants."""
    clean = {"duplicate_node_ids": [], "duplicate_edges": [], "dangling_edges": 0, "edges_without_provenance": 0}
    assert agentic_packet.integrity_failures(clean) == []
    assert agentic_packet.integrity_failures({**clean, "dangling_edges": 2}) == ["dangling_edges"]


def test_attempt_evidence_never_overwrites_fingerprinted_files(tmp_path, monkeypatch):
    """A-11: a re-run writes <pid>.attempt-N.json; earlier evidence keeps its bytes."""
    monkeypatch.setattr(agentic_packet, "SESSION_DIR", tmp_path)
    first = agentic_packet.evidence_path("integration", "P2-X")
    first.write_text(json.dumps({"attempt": 1}))
    second = agentic_packet.evidence_path("integration", "P2-X")
    second.write_text(json.dumps({"attempt": 2}))
    third = agentic_packet.evidence_path("integration", "P2-X")
    assert (first.name, second.name, third.name) == ("P2-X.json", "P2-X.attempt-2.json", "P2-X.attempt-3.json")
    assert json.loads(first.read_text()) == {"attempt": 1}
