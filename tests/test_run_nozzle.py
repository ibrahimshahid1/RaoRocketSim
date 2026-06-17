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
         "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("contour.csv", "profile.png", "wall.stl",
                 "regen.stl", "regen_3d.png", "summary.json"):
        assert (out / name).exists(), f"missing {name}\n{proc.stdout}"
    # The regen STL is a non-trivial binary mesh.
    assert (out / "regen.stl").stat().st_size > 1000
    # The CLI prints the banner, the build plan, and the results panel.
    assert "RaoRocketSim" in proc.stdout
    assert "Regen geometry" in proc.stdout and "channels" in proc.stdout
    assert "Done" in proc.stdout


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
