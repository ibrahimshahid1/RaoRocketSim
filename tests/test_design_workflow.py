import os
import subprocess
import sys

import numpy as np
import pytest

from raosim.design import NozzleDesignRequest, design_nozzle
from raosim.export import (
    _offset_contour,
    export_step,
    export_stl,
    step_representation,
)
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.validation import evaluate_design_gates


def test_bezier_contour_has_reliability_metadata():
    contour = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)

    assert contour["method"] == "bezier"
    assert contour["design_status"] == "preliminary_top_geometry"
    assert contour["hardware_qualified"] is False
    assert "qualification_note" in contour
    assert isinstance(contour["warnings"], list)


def test_rao_wrapper_keeps_requested_exit_radius_or_warns():
    contour = bell_nozzle_contour(
        Rt=0.020, epsilon=10.0, method="rao", length_pct=80.0, gamma=1.4
    )

    assert contour["design_status"] == "experimental_variational_geometry"
    assert contour["y"][-1] == pytest.approx(contour["Re"], rel=5e-3)
    assert isinstance(contour["warnings"], list)


def test_design_gates_record_bad_cad_thickness():
    contour = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
    report = evaluate_design_gates(
        contour, Pc=45e5, Pa=101325, gamma=1.23, wall_thickness=-0.001
    )

    assert not report.passed
    assert any(check.name == "wall_thickness" and not check.passed
               for check in report.checks)


def test_step_and_solid_stl_exports(tmp_path):
    contour = bell_nozzle_contour(Rt=0.020, epsilon=6.0, length_pct=80.0)
    step_path = export_step(
        contour["x"], contour["y"], tmp_path / "nozzle.step",
        n_angular=16, wall_thickness=0.002,
        metadata={"design_status": contour["design_status"]},
    )
    stl_path = export_stl(
        contour["x"], contour["y"], tmp_path / "nozzle.stl",
        n_angular=16, wall_thickness=0.002,
    )

    assert step_path.exists()
    assert "ISO-10303-21" in step_path.read_text(encoding="utf-8", errors="ignore")
    assert stl_path.exists()
    assert stl_path.stat().st_size > 84


def test_station_wise_normal_wall_offset_and_step_export(tmp_path):
    contour = bell_nozzle_contour(Rt=0.020, epsilon=6.0, length_pct=80.0)
    x, y = contour["x"], contour["y"]
    thickness = np.linspace(0.0007, 0.0015, len(x))
    xo, yo = _offset_contour(x, y, thickness)
    # The displacement is normal to the polyline and has the requested
    # magnitude at every station (not a constant radial shift).
    assert np.hypot(xo - x, yo - y) == pytest.approx(thickness, rel=1e-10)
    path = export_step(
        x, y, tmp_path / "variable_wall.step",
        n_angular=12, wall_thickness=thickness,
    )
    assert path.exists()
    assert step_representation(path) in {"brep", "faceted_brep"}


def test_require_brep_rejects_faceted_fallback(tmp_path, monkeypatch):
    import raosim.export as export_module
    contour = bell_nozzle_contour(Rt=0.020, epsilon=4.0, length_pct=80.0)
    monkeypatch.setattr(export_module, "_export_step_with_cadquery",
                        lambda profile, path: False)
    with pytest.raises(RuntimeError, match="CadQuery/OpenCascade"):
        export_module.export_step(
            contour["x"], contour["y"], tmp_path / "must_be_brep.step",
            wall_thickness=0.001, require_brep=True,
        )
    assert not (tmp_path / "must_be_brep.step").exists()


def test_design_nozzle_api_writes_report_and_step(tmp_path):
    request = NozzleDesignRequest(
        propellant_name="LOX/RP-1",
        Pc=100e5,
        Pa=101325,
        Rt=0.020,
        epsilon=4.0,
        method="bezier",
        output_dir=tmp_path,
        cad="step",
        wall_thickness=0.002,
        angular_points=16,
    )
    result = design_nozzle(request)

    assert result.design_status == "preliminary_top_geometry"
    assert result.files["csv"].exists()
    assert result.files["step"].exists()
    assert result.files["design_report"].exists()
    assert result.gate_report.to_dict()["hardware_qualified"] is False


@pytest.mark.parametrize("method", ["bezier", "moc", "rao"])
def test_batch_cli_methods_complete_without_theta_metadata_crash(method, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    cmd = [
        sys.executable,
        "main.py",
        "--propellant", "LOX/RP-1",
        "--Pc", "45",
        "--Rt", "20",
        "--epsilon", "6",
        "--method", method,
        "--no-plot",
        "--output", f"cli_{method}.csv",
    ]

    result = subprocess.run(
        cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True,
        timeout=180,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "unsupported format string passed to NoneType" not in combined
