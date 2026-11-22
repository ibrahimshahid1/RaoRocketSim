import json
import math
import os
import subprocess
import sys

import pytest

from lrekit.verification.benchmarks import (
    list_benchmark_cases,
    load_benchmark_case,
    load_reference_curves,
    run_benchmark,
)


EXPECTED_CASES = {
    "lea_top_schomberg_2014",
    "rao_scarfed_moc_1990",
    "vulcain_s1_separation_similarity",
    "rao1958_nozzle_b",
}


def test_benchmark_manifests_and_curves_are_valid():
    cases = set(list_benchmark_cases())
    assert EXPECTED_CASES.issubset(cases)

    for case_id in EXPECTED_CASES:
        case = load_benchmark_case(case_id)
        assert case["source"]["pdf"].endswith(".pdf")
        assert case["inputs"]["Rt"] > 0.0
        assert case["inputs"]["epsilon"] > 1.0
        assert case["reference"]
        assert case.get("expected_physics_gaps")

        for loaded in load_reference_curves(case):
            spec = loaded["spec"]
            rows = loaded["rows"]
            assert spec["uncertainty"] >= 0.0
            assert spec["axis_calibration"]
            x_key = spec["x_column"]
            y_key = spec["y_column"]
            xs = [row[x_key] for row in rows]
            ys = [row[y_key] for row in rows]
            assert all(math.isfinite(value) for value in xs + ys)
            assert all(b > a for a, b in zip(xs, xs[1:]))


def test_lea_top_bezier_benchmark_passes_explicit_geometry(tmp_path):
    case = load_benchmark_case("lea_top_schomberg_2014")
    assert case["inputs"]["epsilon"] == pytest.approx(5.51 ** 2)
    assert case["inputs"]["length_pct"] == pytest.approx(89.1183566844)

    result = run_benchmark(
        "lea_top_schomberg_2014", "bezier", report_path=tmp_path
    )

    assert result["overall_status"] == "pass"
    geometry = {
        metric["name"]: metric for metric in result["metrics"]
        if metric["category"] == "geometry"
    }
    assert geometry["exit_radius_over_rt"]["status"] == "pass"
    assert geometry["length_over_rt"]["status"] == "pass"
    assert geometry["theta_n_deg"]["status"] == "pass"
    assert geometry["theta_e_deg"]["status"] == "pass"


def test_benchmark_report_contains_source_status_and_metrics(tmp_path):
    result = run_benchmark(
        "rao_scarfed_moc_1990", "bezier", report_path=tmp_path
    )

    json_path = tmp_path / "rao_scarfed_moc_1990_bezier_benchmark.json"
    md_path = tmp_path / "rao_scarfed_moc_1990_bezier_benchmark.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["source"]["pdf"] == "19900015790.pdf"
    assert payload["overall_status"] == result["overall_status"]
    assert any(metric["source_ref"] for metric in payload["metrics"])
    assert "Physics Gaps" in markdown
    assert "19900015790.pdf" in markdown


def test_vulcain_s1_wall_pressure_trend_is_report_only(tmp_path):
    result = run_benchmark(
        "vulcain_s1_separation_similarity", "bezier", report_path=tmp_path
    )

    trend = [
        metric for metric in result["metrics"]
        if metric["name"] == "wall_pressure_trend"
    ][0]
    assert result["overall_status"] == "report"
    assert trend["status"] == "report"
    assert "boundary-layer" in " ".join(result["physics_gaps"])


def test_benchmark_cli_writes_reports(tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    report_dir = tmp_path / "reports"
    cmd = [
        sys.executable,
        "-m", "lrekit.legacy.toolbox",
        "--benchmark-case", "lea_top_schomberg_2014",
        "--benchmark-method", "bezier",
        "--benchmark-report", str(report_dir),
        "--no-plot",
    ]

    result = subprocess.run(
        cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True, timeout=120
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "[  PASS] bezier" in combined
    assert (report_dir / "lea_top_schomberg_2014_bezier_benchmark.json").exists()
    assert (report_dir / "lea_top_schomberg_2014_bezier_benchmark.md").exists()


def test_scarfed_case_is_explicitly_unsupported_by_axisymmetric_moc(tmp_path):
    result = run_benchmark(
        "rao_scarfed_moc_1990", "moc", report_path=tmp_path
    )
    assert result["overall_status"] == "unsupported"
    metric = result["metrics"][0]
    assert metric["category"] == "capability"
    assert metric["status"] == "unsupported"
    assert "scarfed" in metric["message"].lower()


def test_current_bvp_passes_strict_rao1958_nozzle_b_tables_2_and_3(tmp_path):
    case = load_benchmark_case("rao1958_nozzle_b")
    assert case["source"]["local_path"] == "propulsion_texts/rao1958.pdf"
    assert len(case["source"]["sha256"]) == 64
    result = run_benchmark(
        "rao1958_nozzle_b", "rao", report_path=tmp_path
    )
    assert result["overall_status"] == "pass"
    assert result["contour_design_status"] == "geometric_approximation"
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics["contour_rms_error"]["status"] == "pass"
    assert metrics["full_cde_thrust_coefficient"]["status"] == "pass"
    assert metrics["full_cde_thrust_coefficient"]["message"].startswith(
        "Computed by direct integration"
    )
    assumptions = result["model_assumptions"]
    assert assumptions["exact_boundary_reproduction"] is False
    assert assumptions["source_worked_contour"]["downstream_radius_over_rt"] == pytest.approx(0.45)
    assert assumptions["modeled_contour"]["downstream_radius_over_rt"] == pytest.approx(0.382)
    markdown = open(result["report_paths"]["markdown"]).read()
    assert "## Model Assumptions" in markdown
    assert '"exact_boundary_reproduction": false' in markdown
    assert result["solver_gate_report"]["promotion"]["reliability"] == result["contour_design_status"]
    assert "## Solver Gate Report" in markdown


def test_current_bvp_benchmark_forwards_explicit_throat_and_attachment_options(monkeypatch):
    from types import SimpleNamespace
    from lrekit.verification.benchmarks import _build_contour
    import lrekit.components.nozzle.solvers.variational as rv

    case = load_benchmark_case("rao1958_nozzle_b")
    case["solver_options"]["rao"].update(
        throat_downstream_radius_factor=0.45,
        throat_upstream_radius_factor=1.2,
        exact_d_attachment=True,
        exact_exit_attachment=True, kernel_max_rrcs=1000,
        kernel_seed_max_iter=25, kernel_seed_abs_tol=1e-8,
        kernel_dtheta_limit_deg=0.0625, physics_weight=1.0,
    )
    captured = {}

    def solve(config):
        captured["config"] = config
        def export(**kwargs):
            captured["export"] = kwargs
            return {}
        return SimpleNamespace(
            to_contour_dict=export,
            control_surface=SimpleNamespace(M=[2.6]),
            reliability=SimpleNamespace(value="geometric_approximation"),
        )

    monkeypatch.setattr(rv, "solve_rao_bvp", solve)
    _build_contour(case, "rao")
    config = captured["config"]
    assert config.throat_downstream_radius_factor == pytest.approx(0.45)
    assert config.throat_upstream_radius_factor == pytest.approx(1.2)
    assert config.exact_d_attachment is True
    assert config.exact_exit_attachment is True
    assert config.kernel_max_rrcs == 1000
    assert config.kernel_seed_max_iter == 25
    assert config.kernel_seed_abs_tol == 1e-8
    assert config.kernel_dtheta_limit_deg == 0.0625
    assert config.physics_weight == 1.0
    assert captured["export"]["Ru_factor"] == pytest.approx(1.2)
    assert captured["export"]["Rd_factor"] == pytest.approx(0.45)
