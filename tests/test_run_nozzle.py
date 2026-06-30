"""
Smoke test for the main runner (scripts/run_nozzle.py): the
solve -> contour -> wall -> regen pipeline produces all the expected
artifacts.  Uses ``--max-nfev 0`` (seed contour only) so it runs fast
without the host-scale LM solve.
"""
from __future__ import annotations

import json
import argparse
import importlib
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from raosim.throat_geometry import (
    throat_discharge_coefficient_hall,
    upstream_radius_ratio_for_discharge_coefficient,
)


REPO = Path(__file__).resolve().parent.parent


def test_coolant_specific_cli_temperature_defaults():
    from scripts.run_nozzle import _default_coolant_inlet_temperature

    assert _default_coolant_inlet_temperature("methane") == 120.0
    assert _default_coolant_inlet_temperature("lh2") == 25.0
    assert _default_coolant_inlet_temperature("rp1") == 300.0


def test_current_runner_rejects_legacy_single_injector_pressure_drop(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--injector-pressure-drop", "1400000", "--out", str(tmp_path / "run")],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "--injector-pressure-drop is deprecated" in proc.stderr


def test_split_injector_dp_defaults_are_independent_and_drive_regen_boundary():
    from raosim.run_nozzle import _apply_split_injector_pressure_model

    args = SimpleNamespace(
        fuel_injector_dp_fraction=0.33,
        oxidizer_injector_dp_fraction=None,
        pc=8.0e6,
        coolant="RP-1",
        _fuel_name="rp1",
        coolant_outlet_pressure=None,
    )
    _apply_split_injector_pressure_model(args, argparse.ArgumentParser())

    assert args.fuel_injector_dp_fraction == pytest.approx(0.33)
    assert args.oxidizer_injector_dp_fraction == pytest.approx(0.2)
    assert args._fuel_injector_pressure_drop == pytest.approx(0.33 * 8.0e6)
    assert args._oxidizer_injector_pressure_drop == pytest.approx(0.2 * 8.0e6)
    assert args._regen_injector_pressure_drop == pytest.approx(0.33 * 8.0e6)
    assert args._regen_pressure_boundary_source == (
        "fuel_injector_dp_fraction_split_model"
    )


def test_legacy_main_cli_exposes_only_split_injector_dp_flags():
    parser = importlib.import_module("main").build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--injector-pressure-drop" not in options
    assert "--fuel-injector-dp-fraction" in options
    assert "--oxidizer-injector-dp-fraction" in options


@pytest.mark.smoke
def test_seed_geometry_exports_only_after_geometry_gates_pass(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py",
         "--max-nfev", "0", "--regen", "--channels", "20",
         "--wall-sizing", "scalar",
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
    assert "LREKit" in proc.stdout
    summary = json.loads((out / "summary.json").read_text())
    checks = summary["chamber"]["geometry_checks"]
    assert checks["axial_coordinates_monotonic"] is True
    assert checks["slope_continuity"] is True
    assert checks["seam_watertight"] is True
    assert checks["offset_self_intersections"] is False
    assert summary["wall_geometry"]["stl_watertight"] is True
    assert summary["wall_geometry"]["stl_boundary_edge_count"] == 0
    assert summary["wall_geometry"]["selected_sizing_mode"] == "scalar"
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


def test_wall_sizing_regen_implies_regen_sizing_and_requires_material(tmp_path):
    out = tmp_path / "ws"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--max-nfev", "0",
         "--wall-sizing", "regen", "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 2
    assert "--size-wall needs --material" in proc.stdout


def test_wall_sizing_scalar_rejects_size_wall_conflict(tmp_path):
    out = tmp_path / "conflict"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--wall-sizing", "scalar",
         "--size-wall", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "--wall-sizing scalar cannot be combined with --size-wall" in proc.stderr


def test_cd_target_derives_ru_factor_and_is_reported(tmp_path):
    out = tmp_path / "cd"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--max-nfev", "0", "--cd-target", "0.99",
         "--n-control", "8", "--n-kernel", "12",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

    summary = json.loads((out / "summary.json").read_text())
    throat = summary["throat_geometry"]
    expected_ru = upstream_radius_ratio_for_discharge_coefficient(
        0.99, summary["gamma"]
    )
    assert throat["upstream_radius_source"] == "cd_target_hall_sp8120"
    assert throat["upstream_radius_ratio"] == pytest.approx(expected_ru)
    assert throat["discharge_coefficient_hall"] == pytest.approx(
        throat_discharge_coefficient_hall(expected_ru, summary["gamma"])
    )
    assert throat["discharge_coefficient_hall"] == pytest.approx(0.99)


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
        "--wall-sizing", "--helix-turns", "--thermal", "--backend",
    ):
        assert tag in proc.stdout, tag


@pytest.mark.smoke
def test_electric_pump_cli_exports_shared_bus_and_visualization(tmp_path):
    out = tmp_path / "pump"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py",
         "--no-banner",
         "--max-nfev", "0",
         "--n-control", "8",
         "--n-kernel", "12",
         "--injector", "pintle",
         "--injector-cad", "none",
         "--electric-pump",
         "--pump-visualize",
         "--oxidizer", "LOX",
         "--fuel", "RP-1",
         "--fuel-tank-pressure", "500000",
         "--oxidizer-tank-pressure", "600000",
         "--allow-infeasible-injector",
         "--allow-infeasible-pump",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=360,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    assert "final injector/chamber interface" in proc.stdout
    assert "injector/chamber interface auto-sized:" not in proc.stdout

    pump = json.loads((out / "pump.json").read_text())
    assert pump["assumptions"]["electric_bus_architecture"] == "shared_pack_bus"
    assert pump["assumptions"]["selected_bus_voltage_source"].startswith("shared_")
    voltages = {
        line["drive"]["voltage_v"]
        for line in pump["lines"].values()
        if line["drive"] is not None
    }
    assert len(voltages) == 1
    assert pump["battery"]["voltage_v"] == pytest.approx(next(iter(voltages)))
    assert (out / "pump_particles.gif").exists()


def test_package_module_entrypoint_uses_current_runner():
    proc = subprocess.run(
        [sys.executable, "-m", "raosim", "--tags"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 0
    assert "--wall-sizing" in proc.stdout
    assert "Rao Bell Nozzle Design Toolbox" not in proc.stdout


def test_lrekit_help_does_not_wake_matplotlib_cache():
    proc = subprocess.run(
        [sys.executable, "-m", "lrekit", "--help"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 0
    assert "Matplotlib" not in proc.stderr
    assert "Fontconfig" not in proc.stderr
