"""Static call evidence must remain separate from runtime and validation claims."""

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg import execution_reach, harvest_registry, store
from raosim import model_registry


ROOT = Path(__file__).resolve().parents[2]


def test_cooling_models_reach_optimizer_callables_from_inspected_calls():
    inspected = execution_reach.StaticReach(ROOT)
    for mid in ("thermal.bartz", "thermal.sieder_tate"):
        result = inspected.model(mid)
        assert result["optimizer_reach_state"] == "STATIC_PATH_FOUND"
        route, = [r for r in result["routes"] if r["scope"] == "optimizer"]
        assert len(route["call_sites"]) >= 3
        for call in route["call_sites"]:
            assert call["source_sha256"] == hashlib.sha256((ROOT / call["file"]).read_bytes()).hexdigest()
            assert call["line"] > 0
        assert result["runtime_trace_state"] == "NOT_RUN"
        assert result["optimizer_leverage_state"] == "NOT_MEASURED"


def test_spray_api_paths_do_not_become_optimizer_coupling():
    inspected = execution_reach.StaticReach(ROOT)
    result = inspected.model("spray.schiller_naumann_drag")
    assert result["optimizer_reach_state"] == "NO_STATIC_PATH_FOUND"
    assert {r["scope"] for r in result["routes"]} == {"explicit_analysis_api"}
    assert result["dynamic_dispatch_state"] == "UNKNOWN"
    unknown = inspected.model("unmapped.model")
    assert unknown["state"] == unknown["optimizer_reach_state"] == "UNKNOWN"
    assert unknown["routes"] == []


def test_dynamic_calls_cannot_prove_reach(tmp_path, monkeypatch):
    path = tmp_path / "raosim/mdo/nlp.py"
    path.parent.mkdir(parents=True)
    path.write_text("def target():\n    pass\n\ndef root(callback):\n    callback()\n")
    monkeypatch.setattr(execution_reach, "MODEL_SYMBOLS", {"fixture": ("raosim.mdo.nlp.target",)})
    monkeypatch.setattr(execution_reach, "ROOTS", {"optimizer": ("raosim.mdo.nlp.root",)})
    result = execution_reach.StaticReach(tmp_path).model("fixture")
    assert result["state"] == "UNKNOWN"
    assert result["runtime_trace_state"] == "NOT_RUN"
    assert result["optimizer_reach_state"] == "NO_STATIC_PATH_FOUND"


def test_registry_reports_states_without_inventing_runs_or_physical_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    selected = model_registry.MODEL_REGISTRY["thermal.bartz"]
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {selected.model_id: selected})
    store.save("manifest", [{"id": "DOC_BARTZ", "source_pdf": selected.local_source}])
    result = harvest_registry.harvest()
    model, = store.load("models")
    assert model["promotion_status"] == "screening"
    assert "heuristic" in model["promotion_status_basis"]
    assert model["execution_reach"]["optimizer_reach_state"] == "STATIC_PATH_FOUND"
    assert model["software_verification_state"] == "REPORTED_NOT_REEXECUTED"
    assert model["physical_validation_state"] == "REPORTED_NOT_VALIDATED"
    assert model["experimentally_validated"] is False
    source_links = [link for link in result["links"] if link["to"] == "DOC_BARTZ"]
    assert {link["edge"] for link in source_links} == {"CITES"}
