"""
benchmarks.py - Literature-backed diagnostic benchmark runner.

The benchmark suite is intentionally evidence-first: it records what published
cases say, compares the current solver against those references, and separates
strict pass/fail checks from diagnostic xfail/report-only physics gaps.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)
from raosim.nozzle_geometry import bell_nozzle_contour


DATA_ROOT = Path(__file__).with_name("benchmark_data")
CASES_DIR = DATA_ROOT / "cases"
CURVES_DIR = DATA_ROOT / "curves"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent / "builds" / "benchmarks"

_VALID_METHODS = {"bezier", "moc", "rao"}
_VALID_MODES = {"strict", "xfail", "report"}


def list_benchmark_cases() -> list[str]:
    """Return available benchmark case ids."""
    if not CASES_DIR.exists():
        return []
    return sorted(path.stem for path in CASES_DIR.glob("*.json"))


def load_benchmark_case(case_id: str) -> dict[str, Any]:
    """Load and validate a benchmark manifest by case id."""
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        available = ", ".join(list_benchmark_cases()) or "none"
        raise ValueError(f"Unknown benchmark case '{case_id}'. Available: {available}")

    case = json.loads(path.read_text(encoding="utf-8"))
    _validate_case(case, path)
    return case


def load_reference_curves(case: str | dict[str, Any]) -> list[dict[str, Any]]:
    """Load every curve referenced by a benchmark case."""
    case_dict = _case_dict(case)
    loaded = []
    for spec in case_dict.get("reference", {}).get("curves", []):
        loaded.append({
            "spec": spec,
            "rows": _load_curve(spec["path"]),
        })
    return loaded


def run_benchmark(
    case_id: str,
    method: str,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run one benchmark case/method pair and write JSON + Markdown reports.

    Returns a serializable result dictionary with an overall status:
    pass, fail, xfail, xpass, or report.
    """
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError("method must be one of: bezier, moc, rao")

    case = load_benchmark_case(case_id)
    contour = _build_contour(case, method)

    metrics = []
    metrics.extend(compare_contour_to_reference(contour, case, method=method))
    metrics.extend(compare_performance_to_reference(contour, case, method=method))
    metrics.extend(_solver_metrics(contour, case, method))
    metrics.extend(_wall_pressure_metrics(contour, case, method))

    result = {
        "case_id": case["case_id"],
        "title": case["title"],
        "method": method,
        "source": case["source"],
        "status_policy": case.get("status_policy", {}),
        "overall_status": _overall_status(metrics),
        "metrics": metrics,
        "physics_gaps": case.get("expected_physics_gaps", []),
        "contour_design_status": contour.get("design_status", "unknown"),
        "warnings": contour.get("warnings", []),
    }
    _write_reports(result, report_path)
    return result


def compare_contour_to_reference(
    contour: dict[str, Any],
    case: str | dict[str, Any],
    *,
    method: str | None = None,
) -> list[dict[str, Any]]:
    """Compare generated contour geometry against manifest reference values."""
    case_dict = _case_dict(case)
    method_name = method or str(contour.get("method", "bezier"))
    inputs = case_dict["inputs"]
    geometry = case_dict.get("reference", {}).get("geometry", {})

    Rt = float(inputs["Rt"])
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    actuals = {
        "exit_radius_over_rt": float(y[-1] / Rt),
        "length_over_rt": float(contour.get("Ln", x[-1] - x[0]) / Rt),
        "theta_n_deg": float(contour.get("theta_n", math.nan)),
        "theta_e_deg": float(contour.get("theta_e", math.nan)),
        "area_ratio": float(contour.get("epsilon", inputs["epsilon"])),
    }

    metrics = []
    for name, spec in geometry.items():
        if name not in actuals:
            continue
        mode = _mode_for_metric(case_dict, method_name, spec.get("mode", "strict"))
        metrics.append(_numeric_metric(
            "geometry",
            name,
            actuals[name],
            spec.get("value"),
            spec.get("tolerance"),
            mode,
            spec.get("source_ref", ""),
        ))

    for curve in _curves_of_kind(case_dict, "contour"):
        rows = _load_curve(curve["path"])
        x_ref = np.array([row[curve["x_column"]] for row in rows], dtype=float)
        y_ref = np.array([row[curve["y_column"]] for row in rows], dtype=float)
        x_actual = x / Rt
        y_actual = y / Rt
        mask = (x_ref >= np.min(x_actual)) & (x_ref <= np.max(x_actual))
        if not np.any(mask):
            metrics.append(_message_metric(
                "geometry", "contour_curve_overlap", False,
                _mode_for_metric(case_dict, method_name, curve.get("mode", "report")),
                curve.get("source_ref", ""),
                "Reference curve has no overlap with generated contour domain.",
            ))
            continue
        y_interp = np.interp(x_ref[mask], x_actual, y_actual)
        err = y_interp - y_ref[mask]
        rms = float(np.sqrt(np.mean(err * err)))
        max_abs = float(np.max(np.abs(err)))
        mode = _mode_for_metric(case_dict, method_name, curve.get("mode", "report"))
        tol = float(curve.get("uncertainty", 0.0))
        metrics.append(_numeric_metric(
            "geometry", f"{curve['kind']}_rms_error",
            rms, 0.0, tol, mode, curve.get("source_ref", ""),
            message=f"max_abs_error={max_abs:.6g}",
        ))

    return metrics


def compare_performance_to_reference(
    contour: dict[str, Any],
    case: str | dict[str, Any],
    *,
    method: str | None = None,
) -> list[dict[str, Any]]:
    """Compare ideal-gas thrust coefficient predictions to reference values."""
    case_dict = _case_dict(case)
    method_name = method or str(contour.get("method", "bezier"))
    ref_perf = case_dict.get("reference", {}).get("performance", {})
    entries = ref_perf.get("thrust_coefficients", [])
    if not entries:
        return []

    gamma = float(case_dict["inputs"]["gamma"])
    epsilon = float(contour.get("epsilon", case_dict["inputs"]["epsilon"]))
    Me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
    pe_pc = isentropic_pressure_ratio(Me, gamma)

    metrics = []
    for entry in entries:
        pa_pc = float(entry.get("pa_over_pc", 0.0))
        cf = thrust_coefficient(Me, gamma, pe_pc, pa_pc, epsilon)
        mode = _mode_for_metric(case_dict, method_name, entry.get("mode", "report"))
        metrics.append(_numeric_metric(
            "performance",
            entry["name"],
            cf,
            entry.get("value"),
            entry.get("tolerance"),
            mode,
            entry.get("source_ref", ""),
            message="Computed with current 1-D inviscid thrust coefficient model.",
        ))
    return metrics


def _build_contour(case: dict[str, Any], method: str) -> dict[str, Any]:
    inputs = case["inputs"]
    Rt = float(inputs["Rt"])
    epsilon = float(inputs["epsilon"])
    length_pct = float(inputs.get("length_pct", 80.0))
    gamma = float(inputs.get("gamma", 1.4))

    if method == "bezier":
        return bell_nozzle_contour(
            Rt,
            epsilon,
            inputs.get("theta_n"),
            inputs.get("theta_e"),
            length_pct,
            gamma=gamma,
        )

    if method == "moc":
        from raosim.rao_optimizer import moc_bell_nozzle

        opts = dict(case.get("solver_options", {}).get("moc", {}))
        return moc_bell_nozzle(
            Rt, epsilon, gamma=gamma, length_pct=length_pct, **opts
        )

    if method == "rao":
        from raosim.rao_variational import rao_variational_contour

        opts = dict(case.get("solver_options", {}).get("rao", {}))
        return rao_variational_contour(
            Rt, epsilon, gamma=gamma, length_pct=length_pct, **opts
        )

    raise ValueError(f"Unsupported benchmark method '{method}'")


def _solver_metrics(
    contour: dict[str, Any],
    case: dict[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    metrics = []
    inputs = case["inputs"]
    gamma = float(inputs["gamma"])
    epsilon = float(inputs["epsilon"])
    ideal_me = mach_from_area_ratio(epsilon, gamma, supersonic=True)

    if method in {"moc", "rao"}:
        mode = _mode_for_metric(case, method, "strict")
        if "exit_M_mean" in contour:
            metrics.append(_numeric_metric(
                "solver",
                "exit_mach_consistency",
                float(contour["exit_M_mean"]),
                ideal_me,
                0.25 * ideal_me,
                mode,
                "Ideal area-Mach consistency check",
            ))
        else:
            metrics.append(_message_metric(
                "solver",
                "exit_mach_trace_present",
                False,
                mode,
                "Ideal area-Mach consistency check",
                "No solver exit Mach trace is available for this experimental method.",
            ))
    else:
        metrics.append(_message_metric(
            "solver",
            "exit_mach_trace_present",
            True,
            "report",
            "Ideal area-Mach consistency check",
            "Bezier/TOP path does not trace a characteristic exit Mach profile.",
        ))

    solver_ref = case.get("reference", {}).get("solver", {})
    design_me = solver_ref.get("design_exit_mach")
    if design_me:
        mode = _mode_for_metric(case, method, design_me.get("mode", "strict"))
        value = float(contour["exit_M_mean"]) if "exit_M_mean" in contour else ideal_me
        metrics.append(_numeric_metric(
            "solver",
            "published_design_exit_mach",
            value,
            design_me.get("value"),
            design_me.get("tolerance"),
            mode,
            design_me.get("source_ref", ""),
        ))

    throat_mach = solver_ref.get("throat_mach")
    if throat_mach:
        metrics.append(_message_metric(
            "solver",
            "published_throat_mach_input",
            False,
            _mode_for_metric(case, method, throat_mach.get("mode", "report")),
            throat_mach.get("source_ref", ""),
            "Current public contour API assumes sonic throat and cannot inject published throat Mach.",
        ))

    curved_sonic = solver_ref.get("curved_sonic_line_required")
    if curved_sonic:
        metrics.append(_message_metric(
            "solver",
            "curved_sonic_line_model",
            False,
            _mode_for_metric(case, method, curved_sonic.get("mode", "report")),
            curved_sonic.get("source_ref", ""),
            "Current starting-line models are approximate and not benchmarked against the published curved sonic line.",
        ))

    ideal_exit_mach = solver_ref.get("ideal_exit_mach")
    if ideal_exit_mach:
        metrics.append(_numeric_metric(
            "solver",
            "published_ideal_exit_mach",
            ideal_me,
            ideal_exit_mach.get("value"),
            ideal_exit_mach.get("tolerance"),
            _mode_for_metric(case, method, ideal_exit_mach.get("mode", "report")),
            ideal_exit_mach.get("source_ref", ""),
        ))

    return metrics


def _wall_pressure_metrics(
    contour: dict[str, Any],
    case: dict[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    metrics = []
    wall_ref = case.get("reference", {}).get("wall_pressure_trend")
    wall_curves = _curves_of_kind(case, "wall_pressure")
    if not wall_ref and not wall_curves:
        return metrics

    inputs = case["inputs"]
    Pc = float(inputs.get("Pc", 1.0))
    Pa = float(inputs.get("Pa", 0.0))
    gamma = float(inputs["gamma"])
    wp = _wall_pressure_distribution(contour, Pc, gamma)

    if wall_ref:
        expected = wall_ref.get("expected", "")
        if expected == "monotonic_decreasing_attached":
            passed = bool(wp["monotonic"])
            message = "Wall pressure should decrease downstream for attached low-altitude trend."
        else:
            passed = True
            message = (
                "Literature requires pressure-gradient similarity; current check is "
                "report-only until boundary-layer/separation models exist."
            )
        metrics.append(_message_metric(
            "flow",
            "wall_pressure_trend",
            passed,
            _mode_for_metric(case, method, wall_ref.get("mode", "report")),
            wall_ref.get("source_ref", ""),
            message,
        ))

    Rt = float(inputs["Rt"])
    x_actual = np.asarray(wp["x"], dtype=float) / Rt
    p_over_pc = np.asarray(wp["p_over_Pc"], dtype=float)
    for curve in wall_curves:
        rows = _load_curve(curve["path"])
        x_ref = np.array([row[curve["x_column"]] for row in rows], dtype=float)
        y_ref = np.array([row[curve["y_column"]] for row in rows], dtype=float)
        mask = (x_ref >= np.min(x_actual)) & (x_ref <= np.max(x_actual))
        if not np.any(mask):
            metrics.append(_message_metric(
                "flow", f"{curve['kind']}_curve_overlap", False,
                _mode_for_metric(case, method, curve.get("mode", "report")),
                curve.get("source_ref", ""),
                "Reference pressure curve has no overlap with generated contour domain.",
            ))
            continue

        if curve["y_column"] == "p_over_pa":
            pa_pc = float(curve.get("pa_over_pc") or (Pa / Pc if Pc else 0.0))
            if pa_pc <= 0.0:
                metrics.append(_message_metric(
                    "flow", "wall_pressure_curve_scaling", False,
                    _mode_for_metric(case, method, curve.get("mode", "report")),
                    curve.get("source_ref", ""),
                    "Cannot compare p/Pa curve without positive Pa/Pc.",
                ))
                continue
            y_actual = p_over_pc / pa_pc
        else:
            y_actual = p_over_pc

        interp = np.interp(x_ref[mask], x_actual, y_actual)
        err = interp - y_ref[mask]
        rms = float(np.sqrt(np.mean(err * err)))
        max_abs = float(np.max(np.abs(err)))
        metrics.append(_numeric_metric(
            "flow",
            f"{curve['kind']}_rms_error",
            rms,
            0.0,
            curve.get("uncertainty"),
            _mode_for_metric(case, method, curve.get("mode", "report")),
            curve.get("source_ref", ""),
            message=f"max_abs_error={max_abs:.6g}",
        ))

    return metrics


def _wall_pressure_distribution(
    contour: dict[str, Any],
    Pc: float,
    gamma: float,
) -> dict[str, Any]:
    """Small local pressure evaluator to avoid plotting imports in benchmarks."""
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    At = math.pi * Rt * Rt
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    mach = np.zeros_like(y)
    p = np.zeros_like(y)
    for i, radius in enumerate(y):
        ar = max(math.pi * float(radius) ** 2 / At, 1.0)
        try:
            mach[i] = mach_from_area_ratio(
                ar, gamma, supersonic=bool(i > throat_idx)
            )
        except Exception:
            mach[i] = 1.0
        p[i] = Pc * isentropic_pressure_ratio(float(mach[i]), gamma)

    downstream = p[throat_idx:]
    violations = np.where(np.diff(downstream) > 0)[0] + throat_idx
    return {
        "x": x,
        "p": p,
        "p_over_Pc": p / Pc if Pc else p,
        "M": mach,
        "monotonic": len(violations) == 0,
        "violation_indices": violations,
        "throat_idx": throat_idx,
    }


def _case_dict(case: str | dict[str, Any]) -> dict[str, Any]:
    return load_benchmark_case(case) if isinstance(case, str) else case


def _validate_case(case: dict[str, Any], path: Path) -> None:
    for key in ("case_id", "title", "source", "status_policy", "inputs", "reference"):
        if key not in case:
            raise ValueError(f"{path.name}: missing required key '{key}'")
    for key in ("Rt", "epsilon", "gamma"):
        if key not in case["inputs"]:
            raise ValueError(f"{path.name}: inputs missing '{key}'")
    if not any(case["reference"].get(key) for key in (
        "geometry", "performance", "solver", "wall_pressure_trend", "curves"
    )):
        raise ValueError(f"{path.name}: benchmark must define at least one reference")
    for method, mode in case.get("status_policy", {}).items():
        if method not in _VALID_METHODS or mode not in _VALID_MODES:
            raise ValueError(f"{path.name}: invalid status policy {method}={mode}")
    for curve in case.get("reference", {}).get("curves", []):
        curve_path = DATA_ROOT / curve["path"]
        if not curve_path.exists():
            raise ValueError(f"{path.name}: missing curve file {curve_path}")


def _curves_of_kind(case: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        curve for curve in case.get("reference", {}).get("curves", [])
        if curve.get("kind") == kind
    ]


def _load_curve(path: str) -> list[dict[str, float]]:
    csv_path = DATA_ROOT / path
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if len(rows) < 2:
        raise ValueError(f"Curve {path} must contain at least two rows")
    for row in rows:
        if any(not math.isfinite(value) for value in row.values()):
            raise ValueError(f"Curve {path} contains non-finite values")
    first_key = next(iter(rows[0]))
    values = [row[first_key] for row in rows]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"Curve {path} x values must be strictly increasing")
    return rows


def _mode_for_metric(case: dict[str, Any], method: str, requested: str) -> str:
    if requested not in _VALID_MODES:
        requested = "report"
    policy = case.get("status_policy", {}).get(method, "report")
    if requested == "report" or policy == "report":
        return "report"
    if policy == "xfail" or requested == "xfail":
        return "xfail"
    return "strict"


def _numeric_metric(
    category: str,
    name: str,
    value: float | int | None,
    reference: float | int | None,
    tolerance: float | int | None,
    mode: str,
    source_ref: str,
    *,
    message: str = "",
) -> dict[str, Any]:
    passed = False
    delta = None
    if value is not None and reference is not None:
        value_f = float(value)
        ref_f = float(reference)
        delta = abs(value_f - ref_f)
        passed = delta <= float(tolerance or 0.0)
    status = _metric_status(passed, mode)
    return {
        "category": category,
        "name": name,
        "value": _clean_number(value),
        "reference": _clean_number(reference),
        "delta": _clean_number(delta),
        "tolerance": _clean_number(tolerance),
        "mode": mode,
        "status": status,
        "source_ref": source_ref,
        "message": message,
    }


def _message_metric(
    category: str,
    name: str,
    passed: bool,
    mode: str,
    source_ref: str,
    message: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "value": bool(passed),
        "reference": True,
        "delta": 0 if passed else 1,
        "tolerance": 0,
        "mode": mode,
        "status": _metric_status(passed, mode),
        "source_ref": source_ref,
        "message": message,
    }


def _metric_status(passed: bool, mode: str) -> str:
    if mode == "report":
        return "report"
    if mode == "xfail":
        return "xpass" if passed else "xfail"
    return "pass" if passed else "fail"


def _overall_status(metrics: list[dict[str, Any]]) -> str:
    statuses = [metric["status"] for metric in metrics]
    if "fail" in statuses:
        return "fail"
    if "xfail" in statuses:
        return "xfail"
    if "xpass" in statuses:
        return "xpass"
    if "pass" in statuses:
        return "pass"
    return "report"


def _write_reports(result: dict[str, Any], report_path: str | Path | None) -> None:
    json_path, markdown_path = _resolve_report_paths(
        result["case_id"], result["method"], report_path
    )
    result["report_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_ready(result), indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_format_markdown(result), encoding="utf-8")


def _resolve_report_paths(
    case_id: str,
    method: str,
    report_path: str | Path | None,
) -> tuple[Path, Path]:
    if report_path is None:
        base = DEFAULT_REPORT_DIR
        return (
            base / f"{case_id}_{method}_benchmark.json",
            base / f"{case_id}_{method}_benchmark.md",
        )

    path = Path(report_path)
    if path.suffix.lower() == ".json":
        return path, path.with_suffix(".md")
    if path.suffix.lower() in {".md", ".markdown"}:
        return path.with_suffix(".json"), path
    return (
        path / f"{case_id}_{method}_benchmark.json",
        path / f"{case_id}_{method}_benchmark.md",
    )


def _format_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark: {result['title']}",
        "",
        f"- Case: `{result['case_id']}`",
        f"- Method: `{result['method']}`",
        f"- Overall status: `{result['overall_status']}`",
        f"- Source PDF: `{result['source'].get('pdf', '')}`",
        "",
        "## Metrics",
        "",
        "| Category | Metric | Status | Value | Reference | Tolerance | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for metric in result["metrics"]:
        lines.append(
            "| {category} | {name} | {status} | {value} | {reference} | "
            "{tolerance} | {source_ref} |".format(
                category=metric["category"],
                name=metric["name"],
                status=metric["status"],
                value=_fmt(metric.get("value")),
                reference=_fmt(metric.get("reference")),
                tolerance=_fmt(metric.get("tolerance")),
                source_ref=metric.get("source_ref", ""),
            )
        )
    lines.extend(["", "## Physics Gaps", ""])
    for gap in result.get("physics_gaps", []):
        lines.append(f"- {gap}")
    if result.get("warnings"):
        lines.extend(["", "## Solver Warnings", ""])
        for warning in result["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _clean_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
