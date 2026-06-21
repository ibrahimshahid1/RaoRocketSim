"""
Smoke test for the main runner (scripts/run_nozzle.py): the
solve -> contour -> wall -> regen pipeline produces all the expected
artifacts.  Uses ``--max-nfev 0`` (seed contour only) so it runs fast
without the host-scale LM solve.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def test_coolant_specific_cli_temperature_defaults():
    from scripts.run_nozzle import _default_coolant_inlet_temperature

    assert _default_coolant_inlet_temperature("methane") == 120.0
    assert _default_coolant_inlet_temperature("lh2") == 25.0
    assert _default_coolant_inlet_temperature("rp1") == 300.0


@pytest.mark.smoke
def test_seed_geometry_exports_only_after_geometry_gates_pass(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py",
         "--max-nfev", "0", "--regen", "--channels", "20",
         "--material", "grcop-84",
         "--l-star", "0.9", "--contraction-ratio", "3.0",
         "--shoulder-radius-factor", "0.2",
         "--minimum-cylindrical-length", "0.01",
         "--cad", "step",
         "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    assert "RaoRocketSim" in proc.stdout
    summary = json.loads((out / "summary.json").read_text())
    checks = summary["chamber"]["geometry_checks"]
    assert checks["axial_coordinates_monotonic"] is True
    assert checks["slope_continuity"] is True
    assert checks["seam_watertight"] is True
    assert checks["offset_self_intersections"] is False
    assert summary["wall_geometry"]["stl_watertight"] is True
    assert summary["wall_geometry"]["stl_boundary_edge_count"] == 0
    for name in ("contour.csv", "profile.png", "wall.stl", "wall.step",
                 "regen.stl", "regen_3d.png", "summary.json"):
        assert (out / name).exists(), f"missing {name}\n{proc.stdout}"


def test_size_wall_requires_a_material(tmp_path):
    """--size-wall needs the structural properties a --material supplies;
    without one it errors cleanly (rather than sizing on bad defaults)."""
    out = tmp_path / "sw"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--max-nfev", "0",
         "--regen", "--size-wall", "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 2
    assert "--size-wall needs --material" in proc.stdout


def test_list_materials_flag_prints_catalog_and_exits():
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--list-materials"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 0
    for name in ("GRCop-84", "NARloy-Z", "Inconel 718", "Stainless 316L"):
        assert name in proc.stdout, name


def test_tags_flag_lists_run_tags_and_exits():
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--tags"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 0
    # The grouped flag overview is shown.
    for tag in (
        "--epsilon", "--l-star", "--contraction-ratio", "--regen",
        "--helix-turns", "--thermal", "--backend",
    ):
        assert tag in proc.stdout, tag
