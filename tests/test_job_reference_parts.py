"""Unified jobs can export reference CAD with an honest part manifest.

``parts='reference_cad'`` routes the authoritative re-evaluation's STEP into
the run directory and records the engine part manifest from
``raosim.mdo.build``.  That module records path-only artifacts as unverified,
so the package is always partial and no part may claim geometry fidelity.
"""

from __future__ import annotations

import hashlib

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from raosim.mdo.build import ENGINE_PART_MANIFEST
from raosim.mdo.capabilities import JOB_CAPABILITIES
from raosim.mdo.execute import execute_engine_job
from raosim.mdo.job import (
    ArchitectureSelection, EngineJobSpec, JobValidationError, MissionRequest,
    OutputPolicy,
)
from raosim.mdo.snapshot import FieldComparison, SnapshotComparison


def _policy(tmp_path, **overrides):
    values = dict(parts="reference_cad", cad_formats=("step",),
                  output_dir=str(tmp_path), allow_partial_package=True)
    values.update(overrides)
    return OutputPolicy(**values)


def _spec(policy):
    return EngineJobSpec(
        mission=MissionRequest(thrust="13 kN"),
        architecture=ArchitectureSelection(liner_material="GRCop-84"),
        output_policy=policy,
    )


def test_reference_cad_capability_is_advertised():
    rows = {(row.kind, row.id) for row in JOB_CAPABILITIES}
    assert ("parts", "reference_cad") in rows
    assert ("cad_format", "step") in rows
    # No fidelity beyond L0 is offered: the manifest reports what is reached.
    assert {row.id for row in JOB_CAPABILITIES if row.kind == "cad_fidelity"} == {
        "L0_numerical_only"
    }


@pytest.mark.parametrize("overrides,match", [
    ({"cad_formats": ()}, "STEP only"),
    ({"cad_formats": ("iges",)}, "unsupported cad_format"),
    ({"cad_formats": ("step", "step")}, "STEP only"),
    ({"output_dir": None}, "requires output_dir"),
    ({"allow_partial_package": False}, "PARTIAL"),
    ({"parts": "none"}, "require parts='reference_cad'"),
    ({"cad_fidelity": "L2_valid_reference_cad"}, "unsupported cad_fidelity"),
])
def test_reference_cad_policy_is_validated(tmp_path, overrides, match):
    with pytest.raises(JobValidationError, match=match):
        _policy(tmp_path, **overrides)


def test_valid_reference_cad_policy_is_accepted(tmp_path):
    policy = _policy(tmp_path)
    assert policy.cad_formats == ("step",)
    assert policy.parts == "reference_cad"


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Known numerical outputs; the real job, run-record and build plumbing."""
    import raosim.mdo.engine as engine
    import raosim.mdo.postprocess as postprocess
    import raosim.mdo.state as state

    gates = {name: SimpleNamespace(value=True) for name in (
        "optimizer_constraints_feasible", "numerical_validity", "physics_feasible",
        "requirements_feasible", "workflow_readiness_feasible",
    )}
    snapshot = SimpleNamespace(
        performance={"specific_impulse_delivered_s": SimpleNamespace(value=260.0)},
        constraints_gates=gates, warnings=(),
        to_dict=lambda: {"performance": {}},
    )
    comparison = SnapshotComparison("mdo", "traditional", {
        "performance.isp_delivered": FieldComparison(260, 260, 0, 0, "within_tolerance", 0)
    }, {}, 1e-9, 1e-6)
    observed = {}

    def solve(design, mission, **kwargs):
        return SimpleNamespace(performance=SimpleNamespace(
            OF=mission.OF, mdot_total=3.0, Isp_delivered=260.0, eta_cstar=0.91,
        ))

    def reeval(design, mission, **kwargs):
        observed["handoff"] = kwargs
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        wall = out / "thrust_chamber_wall.step"
        wall.write_text("ISO-10303-21;")
        return SimpleNamespace(
            authoritative_snapshot=snapshot, mdo_snapshot=snapshot,
            variational_contour_validation=None, warnings=(),
            comparison=comparison,
            metadata={"input_integrity": {"status": "pass", "differences": [], "error": None}},
            result=SimpleNamespace(files={"step": wall}),
        )

    monkeypatch.setattr(engine, "chamber_surfaces_for", lambda mission: None)
    monkeypatch.setattr(state, "solve_engine_state", solve)
    monkeypatch.setattr(postprocess, "reevaluate", reeval)
    return observed


def test_reference_cad_job_routes_step_and_writes_partial_manifest(stub_pipeline, tmp_path):
    result = execute_engine_job(_spec(_policy(tmp_path)))
    run_dir = Path(result.output_directory)
    handoff = stub_pipeline["handoff"]
    assert handoff["cad"] == "step"
    assert Path(handoff["output_dir"]) == run_dir / "parts"

    manifest_path = Path(result.part_manifest)
    assert manifest_path.parent == run_dir / "parts"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is False
    assert manifest["achieved_fidelity"] == "l0_numerical_only"
    assert len(manifest["parts"]) == len(ENGINE_PART_MANIFEST)
    # Never overclaim: no part is built, and the uniform-wall body is not
    # attributed to the liner (or any other enumerated part).
    assert all(part["status"] == "unavailable" for part in manifest["parts"])
    assert all(not part["artifacts"] for part in manifest["parts"])
    assert result.part_manifest_summary["unattributed_artifacts"] == ["step"]
    assert result.part_artifacts["step"].endswith("thrust_chamber_wall.step")
    assert result.statuses["part_manifest"] == "partial"

    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert run_manifest["parts"]["path"] == "parts/engine_build_manifest.json"
    assert run_manifest["parts"]["status"] == "partial"
    assert run_manifest["parts"]["hardware_release"] is False
    # The STEP no part claims is still inside the run's integrity chain.
    step_record = run_manifest["parts"]["unattributed_artifacts"]["step"]
    step_bytes = Path(result.part_artifacts["step"]).read_bytes()
    assert step_record["attribution"] == "unattributed"
    assert step_record["sha256"] == hashlib.sha256(step_bytes).hexdigest()
    assert run_manifest["package_status"] == "partial"
    assert run_manifest["hardware_release"] is False
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["part_manifest"] == result.part_manifest


def test_job_without_parts_writes_no_manifest(stub_pipeline, tmp_path, monkeypatch):
    import raosim.mdo.postprocess as postprocess

    seen = {}

    def reeval(design, mission, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after handoff")

    monkeypatch.setattr(postprocess, "reevaluate", reeval)
    result = execute_engine_job(_spec(OutputPolicy(
        reports=("coverage",), output_dir=str(tmp_path), allow_partial_package=True,
    )))
    assert "cad" not in seen and "output_dir" not in seen
    assert result.part_manifest is None


@pytest.mark.slow
def test_reference_cad_job_exports_real_step(tmp_path):
    """Real solve + authoritative re-evaluation + CadQuery STEP export."""
    pytest.importorskip("cadquery")
    result = execute_engine_job(_spec(_policy(tmp_path)))
    run_dir = Path(result.output_directory)
    step = run_dir / "parts" / "thrust_chamber_wall.step"
    assert step.is_file() and step.stat().st_size > 0
    manifest = json.loads(Path(result.part_manifest).read_text())
    assert manifest["complete"] is False
    assert not any(part["status"] == "built" for part in manifest["parts"])
    assert result.part_artifacts["step"] == str(step)
