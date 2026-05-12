import json
import math
import os
import subprocess
import sys

import pytest

from raosim.benchmarks import (
    list_benchmark_cases,
    load_benchmark_case,
    load_reference_curves,
    run_benchmark,
)


EXPECTED_CASES = {
    "lea_top_schomberg_2014",
    "rao_scarfed_moc_1990",
    "vulcain_s1_separation_similarity",
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
        "main.py",
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


@pytest.mark.xfail(
    reason="Current MOC solver does not yet reproduce the published Rao/scarfed Mach and thrust references.",
    strict=False,
)
def test_moc_literature_benchmark_eventually_passes(tmp_path):
    result = run_benchmark(
        "rao_scarfed_moc_1990", "moc", report_path=tmp_path
    )
    assert result["overall_status"] == "pass"


@pytest.mark.xfail(
    reason="Current variational Rao path does not yet expose a validated exit Mach trace.",
    strict=False,
)
def test_rao_literature_benchmark_eventually_passes(tmp_path):
    result = run_benchmark(
        "lea_top_schomberg_2014", "rao", report_path=tmp_path
    )
    assert result["overall_status"] == "pass"
