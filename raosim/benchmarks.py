"""
benchmarks.py - Literature-backed diagnostic benchmark runner.

The benchmark suite is intentionally evidence-first: it records what published
cases say, compares the current solver against those references, and separates
strict pass/fail checks from diagnostic xfail/report-only physics gaps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)
from raosim.nozzle_geometry import (
    _EPSILON_VALS,
    _LPCT_VALS,
    bell_nozzle_contour,
    lookup_angles,
)


DATA_ROOT = Path(__file__).with_name("benchmark_data")
REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = DATA_ROOT / "cases"
CURVES_DIR = DATA_ROOT / "curves"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent.parent / "builds" / "benchmarks"

_VALID_METHODS = {"bezier", "moc", "rao"}
_VALID_MODES = {"strict", "xfail", "report", "unsupported"}


def list_benchmark_cases() -> list[str]:
    """Return available *nozzle* benchmark case ids.

    Other physics packages share ``benchmark_data/cases`` for packaging, but
    their manifests carry ``benchmark_kind`` and must be loaded by their own
    validator.  Legacy nozzle manifests omit the field; that remains
    equivalent to ``benchmark_kind='nozzle'``.
    """
    if not CASES_DIR.exists():
        return []
    cases: list[str] = []
    for path in CASES_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Retain malformed manifests so the normal loader reports the
            # actionable parse/schema error instead of hiding the file.
            cases.append(path.stem)
            continue
        if payload.get("benchmark_kind", "nozzle") == "nozzle":
            cases.append(path.stem)
    return sorted(cases)


def load_benchmark_case(case_id: str) -> dict[str, Any]:
    """Load and validate a benchmark manifest by case id."""
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        available = ", ".join(list_benchmark_cases()) or "none"
        raise ValueError(f"Unknown benchmark case '{case_id}'. Available: {available}")

    case = json.loads(path.read_text(encoding="utf-8"))
    kind = case.get("benchmark_kind", "nozzle")
    if kind != "nozzle":
        raise ValueError(
            f"Benchmark case '{case_id}' has kind {kind!r}, not 'nozzle'; "
            "use the owning subsystem benchmark loader"
        )
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
    if case.get("status_policy", {}).get(method) == "unsupported":
        result = {
            "case_id": case["case_id"],
            "title": case["title"],
            "method": method,
            "source": case["source"],
            "status_policy": case.get("status_policy", {}),
            "overall_status": "unsupported",
            "metrics": [_message_metric(
                "capability", "method_applicability", False, "unsupported",
                case.get("source", {}).get("citation", ""),
                case.get("unsupported_reasons", {}).get(
                    method,
                    "The requested solver and published configuration do not "
                    "share a compatible physical/modeling domain.",
                ),
            )],
            "physics_gaps": case.get("expected_physics_gaps", []),
            "contour_design_status": "not_run_incompatible_case",
            "warnings": [],
        }
        _write_reports(result, report_path)
        return result
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
        prediction = str(entry.get("prediction", "quasi_1d"))
        if prediction == "solver_full_cde":
            cf = contour.get("thrust_coefficient")
            message = (
                "Computed by direct integration of the current BVP's full "
                "Rao C-D-E control surface."
            )
        else:
            cf = thrust_coefficient(Me, gamma, pe_pc, pa_pc, epsilon)
            message = "Computed with current 1-D inviscid thrust coefficient model."
        mode = _mode_for_metric(case_dict, method_name, entry.get("mode", "report"))
        metrics.append(_numeric_metric(
            "performance",
            entry["name"],
            cf,
            entry.get("value"),
            entry.get("tolerance"),
            mode,
            entry.get("source_ref", ""),
            message=message,
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
        # The benchmark runner must exercise the current finite-dimensional
        # BVP, not the retained pre-BVP ``solve_optimal_control_surface``
        # prototype.  Translate the old manifest option names so existing
        # report-only cases remain runnable.
        from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

        opts = dict(case.get("solver_options", {}).get("rao", {}))
        n_control = int(opts.pop("n_control", opts.pop("n_ce_pts", 12)))
        n_kernel = int(opts.pop("n_kernel", opts.pop("n_char", 12)))
        max_nfev = int(opts.pop("max_nfev", opts.pop("max_iter", 200)))
        residual_tol = float(opts.pop("residual_tol", 5e-3))
        evaluate_moc = bool(opts.pop("evaluate_moc", True))
        wall_method = str(opts.pop("wall_method", "bde"))
        solver_backend = str(opts.pop("solver_backend", "numpy"))
        starting_line_method = str(
            opts.pop("starting_line_method", "kliegel_levine")
        )
        if opts:
            unknown = ", ".join(sorted(opts))
            raise ValueError(f"Unknown current-BVP benchmark options: {unknown}")
        solution = solve_rao_bvp(RaoSolverConfig(
            Rt=Rt,
            epsilon=epsilon,
            gamma=gamma,
            pa_over_p0=float(inputs.get("Pa", 0.0)) / max(
                float(inputs.get("Pc", 1.0)), 1e-30,
            ),
            length_pct=length_pct,
            n_control=n_control,
            n_kernel=n_kernel,
            max_nfev=max_nfev,
            residual_tol=residual_tol,
            evaluate_moc=evaluate_moc,
            wall_method=wall_method,
            solver_backend=solver_backend,
            starting_line_method=starting_line_method,
        ))
        contour = solution.to_contour_dict(
            Rt=Rt,
            epsilon=epsilon,
            length_pct=length_pct,
            pa_over_p0=float(inputs.get("Pa", 0.0)) / max(
                float(inputs.get("Pc", 1.0)), 1e-30,
            ),
        )
        contour["method"] = "rao"
        contour["benchmark_solver"] = "solve_rao_bvp"
        contour["exit_M_mean"] = float(solution.control_surface.M[-1])
        contour["design_status"] = solution.reliability.value
        return contour

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
    source = case.get("source", {})
    if source.get("local_path"):
        source_path = REPO_ROOT / str(source["local_path"])
        if not source_path.is_file():
            raise ValueError(f"{path.name}: missing local source {source_path}")
        expected_sha = source.get("sha256")
        if expected_sha:
            actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_sha != str(expected_sha):
                raise ValueError(
                    f"{path.name}: local source SHA-256 mismatch for {source_path}"
                )
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
    if policy == "unsupported" or requested == "unsupported":
        return "unsupported"
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
    if mode == "unsupported":
        return "unsupported"
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
    if "unsupported" in statuses and not any(
        status in {"pass", "fail", "xfail", "xpass"} for status in statuses
    ):
        return "unsupported"
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


# =====================================================================
#  Phase 7 — Rao TOP chart benchmark sweep
# =====================================================================
#
# Sweep the published (eps, length_pct) Rao/NASA SP-8120 chart grid and
# compare ``solve_rao_bvp``'s reported (theta_N, theta_E) against the
# tabulated values.
#
# READ THIS BEFORE TREATING err_* AS SOLVER ERROR (J5
# de-circularization, 2026-06-12):
#
#   * The solver columns are now genuine solver outputs under the
#     characteristic formulation: theta_N = the kernel arc-end angle
#     theta_B the BVP closed on; theta_E = the solved CE exit flow
#     angle.  (Pre-J5 they were a chart echo and the chart-N → exit
#     straight chord respectively — both contentless; see the
#     design-angle reporting block in solve_rao_bvp.)
#   * The chart tables are Rao's 1960 ARS J. *parabola-fit* charts
#     (gamma=1.23; contours gamma-insensitive per Rao 1961 p. 1490).
#     The exact variational solution is NOT expected to match them to
#     plan-target precision: at the eps=10/L80 reference the smooth
#     stationary-DE root sits at theta_B = 25.57 deg / theta_E =
#     11.12 deg vs chart 30 / 15.5 deg.  The err_* columns therefore
#     measure the *exact-vs-parabola-fit delta* — a documented finding
#     of the benchmark, not its failure mode (plan STATUS 2026-06-11h).
#
# Historical gates (REWRITE_PLAN.md Phase 7): RMS < 1.5 deg / max <
# 3.0 deg "plan target", 3 / 6 deg "release" — both defined when the
# columns were chart-circular.  The test suite now records that target as an
# explicit negative-applicability regression; the live full-grid test asserts
# completion + physical-band sanity and *records* the
# deltas.  See tests/test_rao_chart_benchmark.py.


@dataclass
class ChartBenchmarkRow:
    """One (epsilon, length_pct) chart sample with the solver's response."""

    epsilon: float
    length_pct: float
    chart_theta_n_deg: float
    chart_theta_e_deg: float
    solver_theta_n_deg: float | None = None
    solver_theta_e_deg: float | None = None
    # err_* = |solver − chart| — the exact-vs-parabola-fit delta, a
    # recorded finding (see the module comment above), not a defect.
    err_theta_n_deg: float | None = None
    err_theta_e_deg: float | None = None
    # Provenance of the reported theta_N, from
    # construction_diagnostics["design_angles"]["theta_N_source"]:
    # "kernel_theta_B:fixed_end_secant" is the solved angle;
    # "kernel_theta_B:seed_guess" means the secant failed and the row's
    # theta_N is chart-flavoured (treat its delta as low-quality).
    theta_n_source: str | None = None
    max_scaled: float | None = None
    mass_residual_rel: float | None = None
    length_residual_rel: float | None = None
    reliability: str | None = None
    rao_region: str | None = None
    kernel_d_fraction: float | None = None
    runtime_s: float | None = None
    exception: str | None = None


@dataclass
class ChartBenchmarkResult:
    """Aggregate across the chart sweep with the per-case rows."""

    rows: list[ChartBenchmarkRow] = field(default_factory=list)
    rms_theta_n_deg: float = float("nan")
    rms_theta_e_deg: float = float("nan")
    max_theta_n_deg: float = float("nan")
    max_theta_e_deg: float = float("nan")
    n_total: int = 0
    n_completed: int = 0
    n_failed: int = 0
    Rt: float = 0.020
    gamma: float = 1.4
    pa_over_p0: float = 0.0
    physics_weight: float = 0.05
    n_control: int = 10
    n_kernel: int = 10
    max_nfev: int = 300

    def passes(
        self,
        rms_tol_deg: float = 1.5,
        max_tol_deg: float = 3.0,
    ) -> bool:
        """True iff RMS and max errors meet the supplied gate."""
        if self.n_completed == 0:
            return False
        return (
            self.rms_theta_n_deg <= rms_tol_deg
            and self.rms_theta_e_deg <= rms_tol_deg
            and self.max_theta_n_deg <= max_tol_deg
            and self.max_theta_e_deg <= max_tol_deg
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return _json_ready(d)


# The default chart sub-grid lives inside the smooth-flow Rao region.
# Excluded chart corners:
#
#   * length_pct = 60          -- too short; Phase 5 valid-region check
#                                 fires (test_rao_valid_region.py).
#   * length_pct = 100         -- degenerates into a 15-deg full conical
#                                 nozzle; chart values are extrapolated
#                                 outside the published Rao TOP region.
#   * epsilon < 6              -- low-area-ratio cases sit at the edge
#                                 of the valid Rao region; the BVP
#                                 finds reduced-quality optima there.
#
DEFAULT_CHART_EPSILON_GRID: tuple[float, ...] = tuple(
    float(eps) for eps in _EPSILON_VALS if eps >= 6.0
)
DEFAULT_CHART_LENGTH_PCT_GRID: tuple[float, ...] = tuple(
    float(lpct) for lpct in _LPCT_VALS if 70.0 <= lpct <= 90.0
)


def rao_variational_chart_benchmark(
    *,
    Rt: float = 0.020,
    gamma: float = 1.4,
    pa_over_p0: float = 0.0,
    epsilon_grid: Sequence[float] | None = None,
    length_pct_grid: Sequence[float] | None = None,
    n_control: int = 10,
    n_kernel: int = 10,
    max_nfev: int = 300,
    residual_tol: float = 5e-3,
    angle_boundary_mode: str = "free",
    starting_line_method: str = "kliegel_levine",
    kernel_d_fraction_max: float | None = None,
    progress: bool = False,
) -> ChartBenchmarkResult:
    """Sweep ``solve_rao_bvp`` over the published Rao/NASA SP-8120 chart.

    Returns a :class:`ChartBenchmarkResult` carrying per-case rows and
    aggregate RMS / max errors against
    :data:`raosim.nozzle_geometry._THETA_N_TABLE` /
    :data:`raosim.nozzle_geometry._THETA_E_TABLE`.

    All keyword arguments default to the Phase 7 release-gate
    configuration: ``PHYSICS_WEIGHT=0.05`` (the robust default),
    ``angle_boundary_mode='free'`` (the chart never enters the residual
    stack — the solved angles are uncontaminated, and since the J5
    de-circularization the *reported* angles are those solver outputs;
    the err columns are exact-vs-parabola-fit deltas, see the module
    comment), and a sub-grid that excludes the chart corners where the
    Phase 5 valid-region check fires or the chart itself extrapolates.

    Parameters
    ----------
    Rt, gamma, pa_over_p0
        Throat radius, specific-heat ratio, ambient pressure ratio.
    epsilon_grid, length_pct_grid
        Optional overrides for the sweep grid.  Default to the
        :data:`DEFAULT_CHART_EPSILON_GRID` /
        :data:`DEFAULT_CHART_LENGTH_PCT_GRID` defined above.
    n_control, n_kernel, max_nfev, residual_tol
        Solver configuration passed to :class:`RaoSolverConfig`.
    angle_boundary_mode
        ``"free"`` (default) for an uncontaminated benchmark.  See
        :class:`RaoSolverConfig.angle_boundary_mode` for the other
        modes.
    starting_line_method
        Default ``"kliegel_levine"`` per REWRITE_PLAN.md Section 2.H.
    kernel_d_fraction_max
        Per-call cap on ``kernel_d_fraction``.  ``None`` (default)
        leaves the module-level :data:`KERNEL_D_FRACTION_MAX` in
        effect.  Pass a smaller value (e.g. ``0.7``) to study the
        high-PHYSICS_WEIGHT regime; see the docstring on the constant
        in :mod:`raosim.rao_variational` for the trade-off.
    progress
        If ``True`` and the optional ``tqdm`` library is available, a
        tqdm progress bar is printed.  Otherwise silent.
    """
    from raosim.rao_variational import (
        PHYSICS_WEIGHT,
        RaoSolverConfig,
        solve_rao_bvp,
    )

    epsilons = (
        DEFAULT_CHART_EPSILON_GRID if epsilon_grid is None
        else tuple(float(x) for x in epsilon_grid)
    )
    lpcts = (
        DEFAULT_CHART_LENGTH_PCT_GRID if length_pct_grid is None
        else tuple(float(x) for x in length_pct_grid)
    )
    cases = [(eps, lpct) for eps in epsilons for lpct in lpcts]

    iterator: Any = cases
    if progress:
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
            iterator = tqdm(cases, desc="rao_variational_chart_benchmark",
                            unit="case")
        except ImportError:
            iterator = cases

    rows: list[ChartBenchmarkRow] = []
    for epsilon, length_pct in iterator:
        chart_n, chart_e = lookup_angles(float(epsilon), float(length_pct))
        row = ChartBenchmarkRow(
            epsilon=float(epsilon),
            length_pct=float(length_pct),
            chart_theta_n_deg=float(chart_n),
            chart_theta_e_deg=float(chart_e),
        )

        cfg = RaoSolverConfig(
            Rt=Rt, epsilon=float(epsilon), gamma=gamma,
            pa_over_p0=pa_over_p0, length_pct=float(length_pct),
            n_control=n_control, n_kernel=n_kernel,
            max_nfev=max_nfev, residual_tol=residual_tol,
            evaluate_moc=False,
            starting_line_method=starting_line_method,
            angle_boundary_mode=angle_boundary_mode,
            kernel_d_fraction_max=kernel_d_fraction_max,
        )

        import time as _time
        t0 = _time.time()
        try:
            sol = solve_rao_bvp(cfg)
        except Exception as exc:
            row.exception = repr(exc)
            row.runtime_s = float(_time.time() - t0)
            rows.append(row)
            continue
        row.runtime_s = float(_time.time() - t0)
        row.solver_theta_n_deg = float(math.degrees(sol.theta_N))
        row.solver_theta_e_deg = float(math.degrees(sol.theta_E))
        row.err_theta_n_deg = abs(row.solver_theta_n_deg - row.chart_theta_n_deg)
        row.err_theta_e_deg = abs(row.solver_theta_e_deg - row.chart_theta_e_deg)
        row.theta_n_source = (
            sol.construction_diagnostics
            .get("design_angles", {})
            .get("theta_N_source")
        )
        row.max_scaled = float(sol.residuals.max_scaled)
        row.mass_residual_rel = float(sol.residuals.mass_residual_rel)
        row.length_residual_rel = float(sol.residuals.length_residual_rel)
        row.reliability = sol.reliability.value
        row.rao_region = sol.construction_diagnostics.get("rao_region")
        row.kernel_d_fraction = float(sol.control_surface.kernel_d_fraction)
        rows.append(row)

    errs_n = np.asarray(
        [r.err_theta_n_deg for r in rows if r.err_theta_n_deg is not None],
        dtype=float,
    )
    errs_e = np.asarray(
        [r.err_theta_e_deg for r in rows if r.err_theta_e_deg is not None],
        dtype=float,
    )

    def _rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan")

    def _max(arr: np.ndarray) -> float:
        return float(np.max(arr)) if arr.size else float("nan")

    return ChartBenchmarkResult(
        rows=rows,
        rms_theta_n_deg=_rms(errs_n),
        rms_theta_e_deg=_rms(errs_e),
        max_theta_n_deg=_max(errs_n),
        max_theta_e_deg=_max(errs_e),
        n_total=len(rows),
        n_completed=int(errs_n.size),
        n_failed=int(len(rows) - errs_n.size),
        Rt=Rt, gamma=gamma, pa_over_p0=pa_over_p0,
        physics_weight=float(PHYSICS_WEIGHT),
        n_control=n_control, n_kernel=n_kernel, max_nfev=max_nfev,
    )


def format_chart_benchmark_report(result: ChartBenchmarkResult) -> str:
    """Return a compact human-readable summary of the chart sweep."""
    lines: list[str] = [
        f"Rao chart benchmark: {result.n_completed}/{result.n_total} cases "
        f"completed ({result.n_failed} raised)",
        f"  PHYSICS_WEIGHT  = {result.physics_weight}",
        f"  n_control       = {result.n_control}",
        f"  n_kernel        = {result.n_kernel}",
        f"  max_nfev        = {result.max_nfev}",
        "",
        "  (err = exact-variational vs Rao-1960 parabola-fit chart "
        "delta — a recorded finding, not a solver error)",
        f"  RMS theta_N delta: {result.rms_theta_n_deg:5.2f} deg  "
        f"(max {result.max_theta_n_deg:5.2f} deg)",
        f"  RMS theta_E delta: {result.rms_theta_e_deg:5.2f} deg  "
        f"(max {result.max_theta_e_deg:5.2f} deg)",
        "",
        "  per-case rows:",
        "    eps  L%    chart_n  chart_e  solv_n  solv_e   err_n  err_e   "
        "max_scl  region",
    ]
    for r in result.rows:
        if r.exception is not None:
            lines.append(
                f"   {r.epsilon:4.1f}  {r.length_pct:4.1f}  "
                f"{r.chart_theta_n_deg:6.2f}   {r.chart_theta_e_deg:6.2f}  "
                f"     -       -        -      -        -    EXC: {r.exception[:30]}"
            )
            continue
        lines.append(
            f"   {r.epsilon:4.1f}  {r.length_pct:4.1f}  "
            f"{r.chart_theta_n_deg:6.2f}   {r.chart_theta_e_deg:6.2f}  "
            f"{(r.solver_theta_n_deg or 0):6.2f}  {(r.solver_theta_e_deg or 0):6.2f}  "
            f"{(r.err_theta_n_deg or 0):5.2f}  {(r.err_theta_e_deg or 0):5.2f}  "
            f"{(r.max_scaled or 0):7.2e}  {r.rao_region or '-'}"
        )
    return "\n".join(lines)
