"""Reviewer identity, self-verification refusal and fingerprinted model reviews.

Corpus-free: grounding is stubbed exactly as in test_kg_extraction.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_extraction, implementation_freshness, store  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    monkeypatch.setattr(store, "CHUNK_DB", tmp_path / "knowledge/chunks.sqlite")
    monkeypatch.setattr(implementation_freshness, "REPO_ROOT", tmp_path)

    class Grounder:
        def __init__(self, doc_id):
            self.chunks = [{"chunk_id": "DOC_TEST#c1", "source_sha256": "src"}]

        def locate(self, quote, page):
            return {"state": "GROUNDED", "chunk_id": "DOC_TEST#c1", "page": page, "match": "exact"}

    monkeypatch.setattr(harvest_extraction, "DocGrounder", Grounder)
    monkeypatch.setattr(harvest_extraction, "_current_grounder", lambda doc_id, sha: Grounder(doc_id))
    store.save("manifest", [{"id": "DOC_TEST", "sha256": "src"}])
    store.save("models", [{"id": "MODEL_TEST", "subsystem": "thermal"}])
    code = tmp_path / "pkg" / "physics.py"
    code.parent.mkdir()
    code.write_text("def h(x):\n    return 0.026 * x ** 0.8\n")
    return tmp_path


def _result(**extra):
    anchor = {"page": 3, "quote": "The coefficient is given by the following relation."}
    return {"doc_id": "DOC_TEST", "target": "MODEL_TEST", "source_sha256": "src",
            "supports_model": {"found": True, "state": "PARTIALLY_SUPPORTED", "equation_labels": ["Eq. 1"]},
            "equations": [{**anchor, "label": "Eq. 1", "form": "h = 0.026 x^0.8", "confidence": "high"}],
            "claims": [], **extra}


def test_transcriber_identity_is_recorded_and_cannot_self_verify(isolated):
    harvest_extraction.harvest([_result(reviewer={"role": "F", "model": "opus"})], [])
    eq = store.load("equations")[0]
    assert eq["transcribed_by"] == {"role": "F", "model": "opus"}
    prov = {p["id"]: p for p in store.load("provenance")}[eq["provenance_id"]]
    assert "role F (opus)" in prov["asserted_by"]
    stats = harvest_extraction.apply_verification([{"object_id": eq["id"], "refuted": False, "confidence": "high",
                                                    "reason": "own transcription", "reviewer": {"role": "F", "model": "opus"}}])
    assert stats["self_verification_refused"] == 1 and stats["confirmed"] == 0
    assert "verification" not in store.load("equations")[0]
    # An unsigned verdict cannot show independence from the recorded transcriber,
    # and a mechanical worker role never makes a scientific verdict (audit A-04).
    stats = harvest_extraction.apply_verification([{"object_id": eq["id"], "refuted": False, "confidence": "high",
                                                    "reason": "anonymous"}])
    assert stats["unsigned_refused"] == 1 and stats["confirmed"] == 0
    stats = harvest_extraction.apply_verification([{"object_id": eq["id"], "refuted": True, "confidence": "high",
                                                    "reason": "worker opinion", "reviewer": {"role": "S2", "model": "sonnet"}}])
    assert stats["worker_role_refused"] == 1 and stats["refuted"] == 0
    assert "verification" not in store.load("equations")[0]
    stats = harvest_extraction.apply_verification([{"object_id": eq["id"], "refuted": False, "confidence": "high",
                                                    "reason": "independent original-page check",
                                                    "reviewer": {"role": "A", "model": "opus-auditor"}}])
    eq = store.load("equations")[0]
    assert stats["confirmed"] == 1 and eq["verification"] == "CONFIRMED"
    assert (eq["verifier_role"], eq["verifier_model"]) == ("A", "opus-auditor")


def test_reviewer_without_model_identity_is_refused(isolated):
    with pytest.raises(ValueError):
        harvest_extraction.harvest([_result(reviewer={"role": "F"})], [])


def _review(fidelity="DIFFERENT_FORMULA"):
    return {"fidelity": fidelity, "implementation": [{"file": "pkg/physics.py", "symbol": "h"}],
            "source_pages": [3], "comparison": [{"aspect": "coefficient", "source": "0.026", "code": "0.026"}]}


def test_model_review_binds_source_and_implementation_fingerprints(isolated):
    tmp_path = isolated
    with pytest.raises(ValueError):
        harvest_extraction.harvest([_result(reviewer={"role": "F", "model": "opus"}, model_review=_review("CLOSE"))], [])
    harvest_extraction.harvest([_result(reviewer={"role": "F", "model": "opus"}, model_review=_review())], [])
    model = store.load("models")[0]
    review = model["source_review"]
    assert review["fidelity"] == "DIFFERENT_FORMULA" and review["reviewed_source_sha256"] == "src"
    assert review["implementation_fingerprints"][0]["state"] == "VERIFIED"
    assert review["physical_validation"] == "NOT_ESTABLISHED" and review["sensitivity"] == "NOT_MEASURED"
    doc = {"sha256": "src"}
    assert harvest_extraction.model_review_state(model, doc, repo_root=tmp_path)["state"] == "CURRENT"
    code = tmp_path / "pkg" / "physics.py"
    code.write_text("# comment only\ndef h(x):\n    return 0.026 * x ** 0.8\n")
    assert harvest_extraction.model_review_state(model, doc, repo_root=tmp_path)["state"] == "CURRENT"
    code.write_text("def h(x):\n    return 0.023 * x ** 0.8\n")
    assert (harvest_extraction.model_review_state(model, doc, repo_root=tmp_path)["state"]
            == "IMPLEMENTATION_CHANGED_REVIEW_REQUIRED")
    assert harvest_extraction.model_review_state(model, {"sha256": "other"}, repo_root=tmp_path)["state"] == "STALE_SOURCE"
    # A model review must locate an exact definition; guessing a nearby symbol is refused.
    bad = _review()
    bad["implementation"] = [{"file": "pkg/physics.py", "symbol": "missing"}]
    with pytest.raises(ValueError):
        harvest_extraction.harvest([_result(reviewer={"role": "F", "model": "opus"}, model_review=bad)], [])


def test_variable_statistics_separate_current_from_superseded_history():
    from kg.validate import equation_variable_stats
    eqs = [{"id": "EQ_NEW", "variables": [{"resolution_state": "RESOLVED"}, {"resolution_state": "UNRESOLVED_ENTITY"}]},
           {"id": "EQ_OLD", "active": False, "source_state": "SUPERSEDED_EXTRACTION",
            "variables": [{"resolution_state": "AMBIGUOUS_SYMBOL"}, {"resolution_state": "UNRESOLVED_ENTITY"}]}]
    stats = equation_variable_stats(eqs)
    assert (stats["total"], stats["unresolved"], stats["ambiguous"], stats["resolved"]) == (2, 1, 0, 1)
    assert stats["historical_or_inactive"]["ambiguous"] == 1
    assert stats["all_records"]["total"] == 4


def test_self_declared_garbled_forms_are_incomplete_but_recovery_notes_are_not():
    from kg.graph import equation_formulation_state
    garbled_low = {"form": "_(Pc_ )in/(Pc)ns = _y-1_ (lt y Mj 2)", "confidence": "low",
                   "extraction_note": "Equation OCR is garbled (exponent and denominator structure unreadable)."}
    garbled_medium = {"form": "minimum values of AP inj /Pcn = 5 to 20 percent", "confidence": "medium",
                      "extraction_note": "OCR garbles the symbols; the '*' is an unreadable relational symbol"}
    recovered = {"form": "Vc = Wtc V ts", "confidence": "medium",
                 "extraction_note": "The equation body is not rendered in the Markdown; form transcribed from a render of PDF page 95"}
    marker = {"form": "[equation NOT RENDERED]", "confidence": "high"}
    assert equation_formulation_state(garbled_low) == "INCOMPLETE_TRANSCRIPTION"
    assert equation_formulation_state(garbled_medium) == "INCOMPLETE_TRANSCRIPTION"
    assert equation_formulation_state(marker) == "INCOMPLETE_TRANSCRIPTION"
    assert equation_formulation_state(recovered) == "NOT_FLAGGED_INCOMPLETE"


def test_reviewed_correction_supersedes_a_self_declared_garbled_transcription(isolated):
    old = {"id": "EQ_TEST_OLD", "type": "Equation", "document_id": "DOC_TEST", "equation_number": "1-14",
           "form": "_(Pc_ )in/(Pc)ns = _y-1_", "confidence": "low", "source_sha256": "src", "source_state": "CURRENT",
           "extraction_note": "Equation OCR is garbled.", "model_ids": ["MODEL_TEST"], "page": 3}
    store.save("equations", [old])
    fixed = _result(reviewer={"role": "F", "model": "opus"})
    fixed["equations"][0].update(equation_number="1-14", form="(Pc)inj/(Pc)ns = (1 + gamma*M^2)/(1 + (gamma-1)/2*M^2)^(gamma/(gamma-1))",
                                 supersedes_ids=["EQ_TEST_OLD"])
    report = harvest_extraction.harvest([fixed], [])
    rows = store.index_by("equations")
    assert report.get("superseded_incomplete_transcriptions") == 1
    assert rows["EQ_TEST_OLD"]["active"] is False and rows["EQ_TEST_OLD"]["source_state"] == "SUPERSEDED_EXTRACTION"
    new_id = rows["EQ_TEST_OLD"]["superseded_by"]
    assert rows[new_id]["corrects_ids"] == ["EQ_TEST_OLD"] and rows[new_id]["transcribed_by"]["role"] == "F"
    # The review decides what the superseded record carried forward (audit A-03):
    # here the model link is re-established by the correction itself, nothing is dropped.
    assert rows[new_id]["model_ids"] == ["MODEL_TEST"]
    assert "supersession_carry_forward_undecided" not in report


def test_standards_are_held_only_with_a_current_mapped_document(isolated):
    from kg import standards
    store.save("manifest", [{"id": "DOC_ASME_Y14_5_2018_DIMENSIONING_AND_TOLERANCING_8_2_THE_AMERICA",
                             "sha256": "abc", "page_count": 347}])
    store.save("capabilities", [{"id": "CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H6_ACHIEVABLE_TOLERANCE_AND_FITS_BY_PROCESS"}])
    out = standards.harvest()
    rows = store.index_by("standards")
    assert rows["STD_ASME_Y14_5"]["state"] == "HELD" and rows["STD_ASME_Y14_5"]["edition"] == "2018"
    assert rows["STD_ASME_Y14_41"]["state"] == "BLOCKED_ON_STANDARD"  # mapped PDF absent from this manifest
    assert rows["STD_ASME_Y14_41"]["held_document_state"] == "MAPPED_DOCUMENT_MISSING_OR_EMPTY"
    assert rows["STD_ISO_286"]["state"] == "BLOCKED_ON_STANDARD" and rows["STD_ISO_286"]["edition"] == "UNSPECIFIED"
    assert all(r["compliance_state"] == "NOT_ASSESSED" for r in rows.values())
    nodes, edges = standards.graph_records()
    assert {"from": "CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H6_ACHIEVABLE_TOLERANCE_AND_FITS_BY_PROCESS",
            "to": "STD_ISO_286", "type": "BLOCKED_ON", "provenance": "PROJECT_DERIVES:kg.standards"} in edges
    assert out["held"] == 1


def test_literature_constants_and_assumptions_can_be_independently_verified(isolated):
    res = _result(reviewer={"role": "F", "model": "opus"})
    res["empirical_constants"] = [{"name": "inverter power density", "value": "60", "units": "kW/kg", "page": 3,
                                   "quote": "The coefficient is given by the following relation."}]
    res["assumptions"] = [{"statement": "no mechanical loss between pump and motor", "page": 3,
                           "quote": "The coefficient is given by the following relation."}]
    harvest_extraction.harvest([res], [])
    ents = {e["type"]: e for e in store.load("entities")}
    const, assumption = ents["EmpiricalConstant"], ents["Assumption"]
    stats = harvest_extraction.apply_verification([
        {"object_id": const["id"], "refuted": False, "confidence": "high", "reason": "page value",
         "reviewer": {"role": "A", "model": "opus-auditor"}},
        {"object_id": assumption["id"], "refuted": True, "confidence": "high", "reason": "not stated",
         "reviewer": {"role": "A", "model": "opus-auditor"}}])
    rows = store.index_by("entities")
    assert stats["confirmed"] == 1 and stats["refuted"] == 1
    assert rows[const["id"]]["verification"] == "CONFIRMED" and rows[assumption["id"]]["verification"] == "REFUTED"
    # Repository-derived records cannot be "confirmed" against a source page.
    store.save("entities", list(rows.values()) + [{"id": "CONST_PROJECT", "type": "EmpiricalConstant",
                                                   "provenance_kind": "PROJECT_DERIVES", "document_id": "DOC_TEST"}])
    stats = harvest_extraction.apply_verification([{"object_id": "CONST_PROJECT", "refuted": False,
                                                    "confidence": "high", "reviewer": {"role": "A", "model": "x"}}])
    assert stats["unknown_ids"] == 1


def test_verification_queue_counts_constants_assumptions_and_source_validity(isolated):
    """Regression (S3 report 2026-10-16): the queue counted only equations and claims."""
    from kg import work_queue
    res = _result(reviewer={"role": "F", "model": "opus"})
    res["empirical_constants"] = [{"name": "inverter power density", "value": "60", "units": "kW/kg", "page": 3,
                                   "quote": "The coefficient is given by the following relation."}]
    res["assumptions"] = [{"statement": "no mechanical loss between pump and motor", "page": 3,
                           "quote": "The coefficient is given by the following relation."}]
    res["validity_statements"] = [{"statement": "tested between 1 and 3 MPa", "page": 3,
                                   "quote": "The coefficient is given by the following relation."}]
    harvest_extraction.harvest([res], [])
    store.save("models", [{**m, "registry_id": "test.model"} for m in store.load("models")])
    # Registry prose envelopes are not source transcriptions and are never queued.
    store.save("validity", store.load("validity") + [{"id": "VAL_REGISTRY", "provenance_kind": "PROJECT_DERIVES",
                                                       "model_id": "MODEL_TEST"}])
    summary = work_queue.build()["summary"]
    by_store = summary["pending_verifications_by_store"]
    assert by_store.get("equations") == 1 and by_store.get("entities") == 2 and by_store.get("validity") == 1
    assert summary["pending_verifications"] == sum(by_store.values())
    entity = next(e for e in store.load("entities") if e["type"] == "EmpiricalConstant")
    harvest_extraction.apply_verification([{"object_id": entity["id"], "refuted": False, "confidence": "high",
                                            "reason": "page value", "reviewer": {"role": "A", "model": "opus"}}])
    assert work_queue.build()["summary"]["pending_verifications_by_store"]["entities"] == 1
