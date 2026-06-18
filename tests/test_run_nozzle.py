"""
Smoke test for the main runner (scripts/run_nozzle.py): the
solve -> contour -> wall -> regen pipeline produces all the expected
artifacts.  Uses ``--max-nfev 0`` (seed contour only) so it runs fast
without the host-scale LM solve.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


@pytest.mark.smoke
def test_runner_contour_plus_regen_pipeline(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py",
         "--max-nfev", "0", "--regen", "--channels", "20",
         "--material", "grcop-84",
         "--cad", "step",
         "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("contour.csv", "profile.png", "wall.stl", "wall.step",
                 "regen.stl", "regen_3d.png", "summary.json"):
        assert (out / name).exists(), f"missing {name}\n{proc.stdout}"
    # The regen STL is a non-trivial binary mesh.
    assert (out / "regen.stl").stat().st_size > 1000
    # The CLI prints the banner, the build plan, and the results panel.
    assert "RaoRocketSim" in proc.stdout
    assert "Regen geometry" in proc.stdout and "channels" in proc.stdout
    assert "Done" in proc.stdout
    # The chosen material flows into the build plan and the summary, and
    # sets the wall conductivity from the catalog (GRCop-84 k=285).
    assert "GRCop-84" in proc.stdout
    import json
    summary = json.loads((out / "summary.json").read_text())
    assert summary["material"]["name"] == "GRCop-84"
    assert summary["material"]["conductivity_W_mK"] == 285.0
    assert summary["wall_geometry"]["offset"] == "surface_normal"
    assert summary["wall_geometry"]["step"] in {"brep", "faceted_brep"}


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
    for tag in ("--epsilon", "--regen", "--helix-turns", "--thermal", "--backend"):
        assert tag in proc.stdout, tag
