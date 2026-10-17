from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg import constant_basis, store


def test_solver_tuning_is_not_physical_calibration_and_legacy_labels_are_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path / "knowledge")
    path = tmp_path / "raosim/rao_variational.py"
    path.parent.mkdir()
    path.write_text("\n".join(p for probes in constant_basis.NUMERICAL_TUNING.values() for p in probes))
    rows = [{"id": key, "type": "EmpiricalConstant", "file": "raosim/rao_variational.py",
             "basis_class": "EMPIRICALLY_CALIBRATED", "verified": True}
            for key in constant_basis.NUMERICAL_TUNING]
    rows.append({"id": "CONST_LEGACY", "type": "EmpiricalConstant", "basis_class": "LITERATURE_DERIVED"})
    metadata = {"generated_at": "fixture", "n": 3, "parameter_harvester": {"state": "fixture"}}
    store.save("software_links", rows, meta=metadata)
    assert constant_basis.reconcile(repo_root=tmp_path)["numerical_tuning_classifications"] == 2
    values = store.load("software_links")
    assert all(r["basis_class"] == "IMPLEMENTATION_CHOICE" and r["reported_basis_class"] == "EMPIRICALLY_CALIBRATED"
               for r in values if r["id"] in constant_basis.NUMERICAL_TUNING)
    legacy = next(r for r in values if r["id"] == "CONST_LEGACY")
    assert legacy["basis_class"] == "LITERATURE_DERIVED"
    assert legacy["scientific_basis_review_state"] == "NOT_CHECKED"
    assert legacy["value_freshness_state"] == "NOT_RECHECKED"
    constant_basis.reconcile(repo_root=tmp_path)
    assert store.load("software_links") == values
    assert store.load_meta("software_links") == metadata
