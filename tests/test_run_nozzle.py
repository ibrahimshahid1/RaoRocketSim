"""
Smoke test for the main runner (scripts/run_nozzle.py): the
solve -> contour -> wall -> regen pipeline produces all the expected
artifacts.  Uses ``--max-nfev 0`` (seed contour only) so it runs fast
without the host-scale LM solve.
"""
from __future__ import annotations

import json
import hashlib
import argparse
import importlib
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from raosim.throat_geometry import (
    REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS,
    throat_discharge_coefficient_hall,
    upstream_radius_ratio_for_discharge_coefficient,
)


REPO = Path(__file__).resolve().parent.parent


def _write_frozen_cli_table(
    tmp_path,
    *,
    pressure_pa=8.0e6,
    temperature_k=3571.0,
    mixture_ratio=2.27,
):
    from raosim.frozen_flow import MODEL_ID

    payload = {
        "schema_version": 2,
        "model": MODEL_ID,
        "molecular_weight_kg_mol": 0.0219,
        "composition_mass_fractions": {"manufactured_products": 1.0},
        "temperature_nodes_k": [150.0, 500.0, 1000.0, 1800.0, 2600.0, 3300.0, 3800.0],
        "cp_nodes_j_kg_k": [1180.0, 1300.0, 1480.0, 1710.0, 1900.0, 2040.0, 2120.0],
        "source": "manufactured CLI variable-cp regression fixture",
        "freeze_basis": "chamber_equilibrium_snapshot",
        "composition_state_pressure_pa": pressure_pa,
        "composition_state_temperature_k": temperature_k,
        "mixture_ratio": mixture_ratio,
        "generator": "pytest manufactured CLI table builder",
        "generator_version": "1",
        "thermo_database": "manufactured piecewise-linear cp",
        "source_artifact_sha256": hashlib.sha256(
            b"manufactured frozen CLI fixture"
        ).hexdigest(),
    }
    path = tmp_path / "frozen_cli.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_cli_release_hard_gate_requires_evidence_manifest(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--require-release-evidence", "--out", str(tmp_path / "blocked")],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "requires --release-evidence-manifest" in proc.stderr
    assert not (tmp_path / "blocked").exists()


def test_cli_spray_cstar_coupling_requires_explicit_scope_and_efficiencies():
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--spray-cstar-coupling", "--injector", "none"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "requires --injector pintle" in proc.stderr


def test_cli_spray_regen_requires_cycle_fuel_as_coolant(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--propellant", "LOX/RP-1", "--injector", "pintle",
         "--spray-cstar-coupling", "--spray-eta-mixing", "0.98",
         "--spray-eta-combustion", "0.99", "--regen",
         "--coolant", "water", "--out", str(tmp_path / "blocked")],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "requires --coolant to be the cycle fuel" in proc.stderr


def test_movable_cd_map_parser_is_strict():
    from raosim.run_nozzle import _parse_opening_cd_map

    assert _parse_opening_cd_map("0:0.62,0.5:0.70,1:0.76") == (
        (0.0, 0.62),
        (0.5, 0.70),
        (1.0, 0.76),
    )
    with pytest.raises(argparse.ArgumentTypeError, match="at least two"):
        _parse_opening_cd_map("0:0.62")
    with pytest.raises(argparse.ArgumentTypeError, match="opening_fraction:Cd"):
        _parse_opening_cd_map("not-a-map")


@pytest.mark.smoke
def test_cli_son_movable_map_reports_real_travel_and_axial_controller(tmp_path):
    out = tmp_path / "son_movable"
    command = [
        sys.executable,
        "scripts/run_nozzle.py",
        "--no-banner",
        "--propellant", "LOX/RP-1",
        "--pc", "7000000",
        "--rt", "0.02",
        "--injector", "pintle",
        "--no-electric-pump",
        "--injector-cad", "none",
        "--injector-architecture", "son_continuous_movable",
        "--injector-sizing", "auto",
        "--pintle-radial-exit", "continuous_radial_gap",
        "--pintle-radial-stream", "fuel",
        "--pintle-deflector-angle", "20",
        "--movable-post-diameter", "0.020",
        "--movable-post-thickness", "0.001",
        "--movable-center-gap-diameter", "0.012",
        "--movable-pintle-rod-diameter", "0.008",
        "--movable-cd-map", "0:0.62,0.5:0.70,1:0.76",
        "--movable-cd-source", "configuration-controlled CLI fixture",
        "--movable-cd-sha256", "b" * 64,
        "--movable-cd-geometry-sha256",
        "61c914edb9bfe93f80f36278e711acb5917b8ba333762c8abadd8bb15dcf4d22",
        "--movable-cd-fluid", "RP-1",
        "--movable-cd-re-min", "1",
        "--movable-cd-re-max", "1000000000",
        "--movable-cd-dp-min", "1",
        "--movable-cd-dp-max", "1000000000",
        "--movable-cd-temperature-min", "200",
        "--movable-cd-temperature-max", "400",
        "--movable-cd-cavitation-min", "0",
        "--movable-cd-cavitation-max", "100",
        "--movable-position-tolerance", "0.000001",
        "--movable-position-feedback-resolution", "0.000001",
        "--movable-backlash", "0.000001",
        "--movable-metrology-source", "configuration-controlled metrology fixture",
        "--movable-metrology-sha256", "c" * 64,
        "--movable-closed-leakage-area", "0",
        "--movable-leakage-source", "configuration-controlled leakage fixture",
        "--movable-leakage-sha256", "d" * 64,
        "--movable-unbalanced-pressure-area", "0.00002",
        "--movable-spring-preload-force", "5",
        "--movable-seal-friction-force", "4",
        "--movable-moving-mass", "0.2",
        "--movable-maximum-acceleration", "50",
        "--movable-actuator-force-capacity", "500",
        "--movable-stem-diameter", "0.006",
        "--movable-stem-allowable-stress", "200000000",
        "--movable-actuator-source", "configuration-controlled actuator fixture",
        "--movable-actuator-sha256", "e" * 64,
        "--movable-sheet-thickness", "0.000125",
        "--movable-sheet-thickness-method", "vof",
        "--movable-sheet-thickness-source", "configuration-controlled VOF fixture",
        "--movable-sheet-thickness-sha256", "a" * 64,
        "--movable-sheet-geometry-sha256",
        "61c914edb9bfe93f80f36278e711acb5917b8ba333762c8abadd8bb15dcf4d22",
        "--movable-sheet-thickness-fluid", "RP-1",
        "--movable-sheet-opening-min", "0.000001",
        "--movable-sheet-opening-max", "0.002",
        "--movable-sheet-dp-min", "1",
        "--movable-sheet-dp-max", "1000000000",
        "--movable-sheet-mass-flow-min", "0.000000001",
        "--movable-sheet-mass-flow-max", "10",
        "--throttle-map", "0.2,0.6,1.0",
        "--out", str(out),
    ]
    proc = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(REPO),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-5000:]

    summary = json.loads((out / "summary.json").read_text())
    injector = summary["injector"]
    throttle = summary["injector_throttle_map"]
    assert injector["architecture"] == "son_continuous_movable"
    assert injector["actuation"]["sheet_thickness_method"] == "vof"
    assert injector["actuation"]["opening_distance_m"] != pytest.approx(
        injector["actuation"]["sheet_thickness_m"]
    )
    assert throttle["actuator_stroke_available"] is True
    assert throttle["schedule_semantics"] == (
        "fixed_hardware_center_pintle_plus_upstream_annulus_controller"
    )
    assert all(
        point["annulus_area_command_fraction"] == pytest.approx(1.0)
        for point in throttle["points"]
    )
    assert all(
        point["required_axial_controller_dp_fraction"] is not None
        for point in throttle["points"]
    )
    assert "Lopen[mm]" in proc.stdout
    assert "O/F+TMR held by the sleeve" not in proc.stdout

    blocked_command = [
        *command,
        "--movable-axial-controller-dp-fraction-min", "0.25",
        "--movable-axial-controller-dp-fraction-max", "0.50",
        "--out", str(tmp_path / "unreachable_controller"),
    ]
    blocked = subprocess.run(
        blocked_command,
        cwd=REPO,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(REPO),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl_blocked"),
            "XDG_CACHE_HOME": str(tmp_path / "cache_blocked"),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        timeout=300,
    )
    assert blocked.returncode == 2
    assert "requested throttle map is not reachable" in blocked.stdout


@pytest.mark.smoke
def test_cli_spray_cstar_fixed_point_updates_mass_flow_and_report(tmp_path):
    out = tmp_path / "spray_loop"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--propellant", "LOX/RP-1", "--pc", "1500000",
         "--injector", "pintle", "--injector-cad", "none",
         "--no-electric-pump", "--spray-cstar-coupling",
         "--spray-eta-mixing", "0.98",
         "--spray-eta-combustion", "0.99",
         "--spray-evaporation-constant", "0.0001",
         "--fuel-injector-dp-fraction", "0.7",
         "--oxidizer-injector-dp-fraction", "0.7",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    summary = json.loads((out / "summary.json").read_text())
    coupling = summary["spray_cstar_coupling"]
    injector = summary["injector"]
    injected = (
        injector["annulus"]["mdot_kg_s"]
        + injector["slots"]["mdot_kg_s"]
    )
    assert coupling["converged"] is True
    assert coupling["scope"] == "injector_and_cycle_mass_flow_no_regen"
    assert injected == pytest.approx(summary["performance"]["mdot_total_kg_s"])
    assert summary["physical_release_readiness"]["blocked"] is True
    assert summary["hardware_qualified"] is False


@pytest.mark.smoke
def test_cli_spray_regen_outer_loop_serializes_final_cooling_and_feed_duty(
    tmp_path,
):
    out = tmp_path / "spray_regen_loop"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--propellant", "LOX/RP-1", "--thermo-mode", "constant-gamma",
         "--pc", "1500000", "--mixture-ratio", "0.4",
         "--injector", "pintle", "--injector-cad", "none",
         "--allow-infeasible-injector", "--no-electric-pump",
         "--regen", "--thermal", "--coolant", "rp1",
         "--coolant-inlet-temperature", "220", "--coolant-mdot", "0.1",
         "--channels", "40", "--channel-width", "0.0008",
         "--channel-height", "0.0025", "--wall-thickness", "0.001",
         "--fuel-injector-dp-fraction", "0.9",
         "--oxidizer-injector-dp-fraction", "3.0",
         "--spray-cstar-coupling", "--spray-eta-mixing", "0.98",
         "--spray-eta-combustion", "0.99",
         "--spray-evaporation-constant", "0.0001",
         "--cad", "none", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-5000:]
    summary = json.loads((out / "summary.json").read_text())
    coupling = summary["spray_cstar_coupling"]
    final = coupling["final_state_summary"]
    assert coupling["scope"] == "spray_cycle_regen_wall_feed_and_pump_duty"
    assert final["coolant_mass_flow_kg_s"] == pytest.approx(
        final["fuel_mass_flow_kg_s"]
    )
    assert final["regen_fuel_relative_flow_error"] == pytest.approx(0.0)
    assert summary["cooling"]["outer_loop_state"] == (
        "final_spray_cstar_regen_iterate"
    )
    assert summary["cooling"]["coolant_mass_flow_kg_s"] == pytest.approx(
        final["fuel_mass_flow_kg_s"]
    )
    assert final["feed_and_pump_duty_by_role"]["fuel"][
        "regen_loss_pa"
    ] == pytest.approx(final["coolant_pressure_drop_pa"])


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
    assert summary["contour_method"] == "bezier"
    assert summary["thrust_closure"]["export_gate_passed"] is True
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


def test_experimental_bvp_failure_blocks_artifacts_without_override(tmp_path):
    out = tmp_path / "blocked_bvp"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--contour-method", "rao-bvp", "--backend", "numpy",
         "--max-nfev", "0", "--n-control", "8", "--n-kernel", "8",
         "--injector", "none", "--no-electric-pump", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )

    assert proc.returncode == 2
    assert "failed the export gate" in proc.stdout
    assert not (out / "contour.csv").exists()
    assert not (out / "wall.stl").exists()


@pytest.mark.parametrize("mode", ("--engine-mdo", "--engine-mdo-optimize"))
def test_mdo_preliminary_host_contour_rejects_manufacturing_cad(mode, capsys):
    from raosim.run_nozzle import main

    return_code = main(
        [
            "--no-banner",
            mode,
            "--mdo-export",
            "--contour-method",
            "rao-bvp",
            "--cad",
            "step",
        ]
    )

    assert return_code == 2
    output = capsys.readouterr().out
    assert "preliminary numerical post-analysis" in output
    assert "cannot emit manufacturing CAD" in output


@pytest.mark.parametrize("mode", ("--engine-mdo", "--engine-mdo-optimize"))
def test_mdo_rao_selector_requires_post_analysis_export(mode, capsys):
    from raosim.run_nozzle import main

    return_code = main(
        [
            "--no-banner",
            mode,
            "--contour-method",
            "rao-bvp",
        ]
    )

    assert return_code == 2
    output = capsys.readouterr().out
    assert "requires --mdo-export" in output
    assert "fixed-topology Rao/TOP chart wall" in output


def test_mdo_rao_handoff_matches_traditional_cli_solver_controls():
    from raosim.run_nozzle import _mdo_authoritative_contour_handoff

    method, options = _mdo_authoritative_contour_handoff(
        SimpleNamespace(
            contour_method="rao-bvp",
            n_control=18,
            n_kernel=20,
            max_nfev=321,
            theta_b_guess=34.0,
            backend="numpy",
        )
    )

    assert method == "rao_variational_moc"
    assert options == {
        "n_control": 18,
        "n_kernel": 20,
        "max_nfev": 321,
        "evaluate_moc": True,
        "theta_n_guess_deg": 34.0,
        "starting_line_method": "kliegel_levine",
        "solver_backend": "numpy",
        "wall_method": "bde",
        "kernel_d_fraction_max": 0.7,
        "physics_weight": 1.0,
    }


def test_bezier_chart_extrapolation_requires_diagnostic_override(tmp_path):
    blocked = tmp_path / "blocked_chart"
    common = [
        sys.executable, "scripts/run_nozzle.py", "--no-banner",
        "--length-pct", "101", "--injector", "none", "--no-electric-pump",
    ]
    proc = subprocess.run(
        [*common, "--out", str(blocked)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 2
    assert "Rao/TOP chart extrapolation" in proc.stdout
    assert not (blocked / "contour.csv").exists()

    diagnostic = tmp_path / "diagnostic_chart"
    proc = subprocess.run(
        [*common, "--allow-chart-extrapolation", "--out", str(diagnostic)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    summary = json.loads((diagnostic / "summary.json").read_text())
    assert summary["contour_reliability"]["rao_chart_extrapolated"] is True
    assert summary["thrust_closure"]["diagnostic_override"] is True


def test_explicit_gamma_is_authoritative_across_cli_physics(tmp_path):
    out = tmp_path / "gamma"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--propellant", "LOX/RP-1", "--gamma", "1.30",
         "--injector", "none", "--no-electric-pump", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=300,
    )

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gamma"] == pytest.approx(1.30)
    assert summary["performance"]["gamma"] == pytest.approx(1.30)
    assert "all use the overridden value" in proc.stdout


def test_cli_rejects_gamma_override_with_cea():
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--thermo-mode", "cea", "--propellant", "LOX/RP-1",
         "--gamma", "1.30"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )

    assert proc.returncode == 2
    assert "--gamma cannot override RocketCEA" in proc.stderr


@pytest.mark.smoke
def test_cli_frozen_variable_cp_runs_bezier_target_thrust_and_serializes_gates(
    tmp_path,
):
    table = _write_frozen_cli_table(tmp_path)
    out = tmp_path / "frozen_cli_run"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_nozzle.py",
            "--no-banner",
            "--propellant",
            "LOX/RP-1",
            "--mixture-ratio",
            "2.27",
            "--pc",
            "8000000",
            "--epsilon",
            "8",
            "--target-thrust",
            "13000",
            "--nozzle-expansion-model",
            "frozen-variable-cp",
            "--frozen-gas-table",
            str(table),
            "--injector",
            "none",
            "--no-electric-pump",
            "--cad",
            "none",
            "--out",
            str(out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(REPO),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        timeout=300,
    )

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-5000:]
    summary = json.loads((out / "summary.json").read_text())
    performance = summary["performance"]
    assert performance["expansion_model"] == "frozen_variable_cp_q1d"
    assert performance["thrust_N"] == pytest.approx(13_000.0, rel=2e-12)
    assert performance["frozen_expansion"]["closures"]["all_pass"] is True
    assert performance["gamma_throat"] != pytest.approx(
        performance["gamma_exit"], rel=1e-4
    )
    assert summary["flow_model_gates"][
        "frozen_q1d_conservation_closure"
    ] is True
    assert summary["flow_model_gates"][
        "frozen_property_and_performance_benchmark"
    ] is False
    assert summary["flow_model_gates"][
        "variable_property_bartz_boundary_layer_regen"
    ] is False
    assert summary["throat_geometry"][
        "discharge_coefficient_model_applicable"
    ] is False
    assert summary["contour_reliability"]["quasi_1d_expansion_model"] == (
        "frozen_variable_cp_q1d"
    )


def test_cli_frozen_variable_cp_requires_property_table_and_explicit_model(
    tmp_path,
):
    common = [
        sys.executable,
        "scripts/run_nozzle.py",
        "--no-banner",
        "--propellant",
        "LOX/RP-1",
        "--injector",
        "none",
        "--no-electric-pump",
    ]
    missing = subprocess.run(
        [*common, "--nozzle-expansion-model", "frozen-variable-cp"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert missing.returncode == 2
    assert "requires --frozen-gas-table" in missing.stderr

    table = _write_frozen_cli_table(tmp_path)
    stale = subprocess.run(
        [*common, "--frozen-gas-table", str(table)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert stale.returncode == 2
    assert "requires --nozzle-expansion-model" in stale.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (("--gamma", "1.3"), "--gamma cannot be supplied"),
        (("--contour-method", "rao-bvp"), "currently Bezier-only"),
        (("--regen",), "blocks the constant-gamma"),
        (("--cd-target", "0.98"), "--cd-target is unavailable"),
    ),
)
def test_cli_frozen_variable_cp_rejects_incompatible_physics(
    tmp_path, extra, message
):
    table = _write_frozen_cli_table(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_nozzle.py",
            "--no-banner",
            "--propellant",
            "LOX/RP-1",
            "--nozzle-expansion-model",
            "frozen-variable-cp",
            "--frozen-gas-table",
            str(table),
            *extra,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert message in proc.stderr


def test_cli_frozen_variable_cp_rejects_stale_composition_state(tmp_path):
    table = _write_frozen_cli_table(tmp_path, pressure_pa=7.5e6)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_nozzle.py",
            "--no-banner",
            "--propellant",
            "LOX/RP-1",
            "--mixture-ratio",
            "2.27",
            "--pc",
            "8000000",
            "--nozzle-expansion-model",
            "frozen-variable-cp",
            "--frozen-gas-table",
            str(table),
            "--injector",
            "none",
            "--no-electric-pump",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "snapshot pressure" in proc.stderr


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
         "--allow-throat-radius-extension",
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
        0.99,
        summary["gamma"],
        min_ratio=REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS[0],
        max_ratio=REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS[1],
    )
    assert throat["upstream_radius_source"] == (
        "cd_target_hall_repository_extension"
    )
    assert throat["upstream_radius_ratio"] == pytest.approx(expected_ru)
    assert throat["discharge_coefficient_hall"] == pytest.approx(
        throat_discharge_coefficient_hall(expected_ru, summary["gamma"])
    )
    assert throat["discharge_coefficient_hall"] == pytest.approx(0.99)


def test_cd_target_above_sp8120_range_requires_explicit_extension(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", "--no-banner",
         "--cd-target", "0.99", "--injector", "none",
         "--no-electric-pump", "--out", str(tmp_path / "cd_rejected")],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=120,
    )
    assert proc.returncode == 2
    assert "SP-8120 range capability" in proc.stderr


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
        "--contour-method",
    ):
        assert tag in proc.stdout, tag


@pytest.mark.smoke
def test_complete_package_defaults_export_nozzle_pintle_and_pump(tmp_path):
    out = tmp_path / "complete"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py",
         "--no-banner",
         "--complete-package",
         "--max-nfev", "0",
         "--n-control", "8",
         "--n-kernel", "12",
         "--injector-cad", "none",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=360,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["run_defaults"]["complete_package"] is True
    assert summary["injector"]["feasible"] is True
    assert summary["injector"]["radial_exit_style"] == "holes"
    assert "hole_diameter" in summary["injector"]["slots"]["detail"]
    assert summary["electric_pump"]["feasible"] is True
    for name in ("contour.csv", "profile.png", "wall.stl",
                 "pintle.json", "pump.json", "pump_bom.json",
                 "pump_reference_geometry.json", "summary.json"):
        assert (out / name).exists(), f"missing {name}\n{proc.stdout}"
    for name in ("pump_parameters.json", "pump_dimensions.csv",
                 "pump_reference_assembly.stl"):
        assert (out / "pump" / name).exists(), f"missing pump/{name}"
    assert (out / "pump" / "pump_parts" / "fuel_impeller.stl").exists()


@pytest.mark.smoke
def test_lrekit_args_file_full_engine_sample(tmp_path):
    out = tmp_path / "lox_rp1_13kn"
    sample = REPO / "examples/cli/lox_rp1_13kn_copper_316l_full_engine.args"
    proc = subprocess.run(
        [sys.executable, "scripts/run_nozzle.py", f"@{sample}",
         "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "mpl"),
             "XDG_CACHE_HOME": str(tmp_path / "cache"),
             "PATH": __import__("os").environ.get("PATH", "")},
        timeout=480,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]

    summary = json.loads((out / "summary.json").read_text())
    perf = summary["performance"]
    assert perf["propellant"] == "LOX/RP-1"
    assert perf["rt_from_target_thrust"] is True
    assert perf["target_thrust_N"] == pytest.approx(13_000.0)
    assert perf["thrust_N"] == pytest.approx(13_000.0)
    assert summary["material"]["name"] == "OFHC Copper"
    assert summary["wall_profile"]["jacket_material"] == "Stainless 316L"
    assert summary["injector"]["feasible"] is True
    assert summary["electric_pump"]["feasible"] is True
    assert summary["wall_geometry"]["selected_sizing_mode"] == (
        "regen_thermostructural"
    )
    for name in (
        "contour.csv", "profile.png", "wall.stl", "jacket.stl",
        "wall.step", "jacket.step", "regen.stl", "regen_3d.png",
        "pintle.json", "pump.json", "summary.json",
    ):
        assert (out / name).exists(), f"missing {name}\n{proc.stdout}"


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
    assert (out / "pump_bom.json").exists()
    assert (out / "pump_reference_geometry.json").exists()


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
