#!/usr/bin/env python3
"""Compare current Python Rao/MOC outputs against NASA/JHU reference files.

This is a visibility tool, not a pass/fail gate.  It reports the current
baseline mismatch so algorithm ports can tighten one file family at a time.
Unavailable metrics are reported explicitly instead of silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raosim.legacy_io import (  # noqa: E402
    LegacyTable,
    parse_center_out,
    parse_kernel_out,
    parse_last_kernel_out,
    parse_rao_dat,
    parse_summary_out,
    parse_tt_prime_out,
    parse_uncropped_kernel_out,
    parse_wall_out,
)


NASA_OUT_DEFAULT = (
    REPO_ROOT
    / "Three-Dimensional-Nozzle-Design-Code-master"
    / "MOC_Grid_BDE"
    / "outputs_M3.5Perf"
)

NASA_TT_PRIME_PROVENANCE_DOC = "docs/nasa_tt_prime_provenance.md"
FIXTURE_GENERATOR_PROVENANCE = "unresolved"
CANONICAL_REFERENCE_TRACK = "visible_source_port"
HISTORICAL_OVERLAY_TRACK = "historical_fixture_overlay"
REQUIRED_FIXTURE_FILES = (
    "wall.out",
    "center.out",
    "rao.dat",
    "summary.out",
    "TT'.out",
    "TT'BF_Kernel.out",
    "BFE_Kernel.out",
    "LastKernel.out",
    "UncroppedKernel.out",
)


def _comparison_diagnostics(nasa_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_FIXTURE_FILES
        if not (nasa_dir / name).exists()
    ]
    fixture_overlay_available = not missing
    blockers = [
        "visible-source port parity is not certified by this fixture-overlay harness",
    ]
    if missing:
        blockers.append("required fixture files are missing")
    return {
        "canonical_reference_track": CANONICAL_REFERENCE_TRACK,
        "comparison_track": HISTORICAL_OVERLAY_TRACK,
        "source_port_candidate": args.starting_line_method,
        "source_port_matched": None,
        "source_port_match_status": "not_evaluated_by_historical_fixture_overlay",
        "source_port_workflow_complete": False,
        "fixture_overlay_available": bool(fixture_overlay_available),
        "fixture_overlay_is_promotion_authority": False,
        "fixture_overlay_missing_files": missing,
        "fixture_generator_provenance": FIXTURE_GENERATOR_PROVENANCE,
        "fixture_generator_provenance_doc": NASA_TT_PRIME_PROVENANCE_DOC,
        "nasa_reference_matched_eligible": False,
        "nasa_reference_matched_blockers": blockers,
        "historical_fixture_overlay_notes": [
            "M3.5Perf fixture deltas are diagnostics only",
            "unresolved TT' provenance does not define canonical source-port parity",
        ],
        "note": (
            "This report overlays current Python results on historical NASA/JHU "
            "sample outputs. NASA_REFERENCE_MATCHED promotion is governed by "
            "the visible-source port track, not by orphaned TT' fixture parity."
        ),
    }


def _metric(
    name: str,
    value: float | None,
    *,
    n: int = 0,
    status: str = "ok",
    reason: str | None = None,
    units: str = "dimensionless",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "value": None if value is None else float(value),
        "n": int(n),
        "units": units,
        "reason": reason,
    }


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mask = np.isfinite(diff)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def _station_rms(ref: np.ndarray, candidate: np.ndarray, *, n: int = 200) -> tuple[float, int]:
    ref = np.asarray(ref, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if ref.size < 2 or candidate.size < 2:
        return float("nan"), 0
    t = np.linspace(0.0, 1.0, n)
    ref_i = np.interp(t, np.linspace(0.0, 1.0, ref.size), ref)
    cand_i = np.interp(t, np.linspace(0.0, 1.0, candidate.size), candidate)
    return _rms(ref_i, cand_i), n


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros_like(unique, dtype=float)
    np.add.at(sums, inverse, y)
    return unique, sums / counts


def _rms_by_x(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    *,
    n: int = 200,
) -> tuple[float, int, str | None]:
    rx, ry = _unique_xy(ref_x, ref_y)
    cx, cy = _unique_xy(candidate_x, candidate_y)
    if rx.size < 2 or cx.size < 2:
        return float("nan"), 0, "not enough monotonic x samples"
    lo = max(float(rx.min()), float(cx.min()))
    hi = min(float(rx.max()), float(cx.max()))
    if not hi > lo:
        return float("nan"), 0, "no overlapping x/R* interval"
    grid = np.linspace(lo, hi, n)
    return _rms(np.interp(grid, rx, ry), np.interp(grid, cx, cy)), n, None


def _compare_station_series(
    metrics: list[dict[str, Any]],
    prefix: str,
    ref: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> None:
    for name in names:
        if name not in ref:
            metrics.append(_metric(f"{prefix}.{name}.station_rms", None, status="unavailable",
                                   reason="reference series not available"))
            continue
        if name not in candidate:
            metrics.append(_metric(f"{prefix}.{name}.station_rms", None, status="unavailable",
                                   reason="current Python series not available"))
            continue
        value, n = _station_rms(ref[name], candidate[name])
        metrics.append(_metric(f"{prefix}.{name}.station_rms", value, n=n))


def _station_diff_rows(
    ref: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    names: tuple[str, ...],
    *,
    n: int = 200,
) -> list[dict[str, float]]:
    available = [name for name in names if name in ref and name in candidate]
    if not available:
        return []
    rows: list[dict[str, float]] = []
    t = np.linspace(0.0, 1.0, n)
    ref_t = np.linspace(0.0, 1.0, np.asarray(ref[available[0]]).size)
    cand_t = np.linspace(0.0, 1.0, np.asarray(candidate[available[0]]).size)
    for idx, tau in enumerate(t):
        row: dict[str, float] = {"station": float(tau), "index": float(idx)}
        for name in available:
            ref_values = np.asarray(ref[name], dtype=float)
            cand_values = np.asarray(candidate[name], dtype=float)
            ref_i = float(np.interp(tau, ref_t, ref_values))
            cand_i = float(np.interp(tau, cand_t, cand_values))
            row[f"ref_{name}"] = ref_i
            row[f"python_{name}"] = cand_i
            row[f"diff_{name}"] = cand_i - ref_i
        rows.append(row)
    return rows


def _by_x_diff_rows(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    *,
    n: int = 200,
) -> list[dict[str, float]]:
    rx, ry = _unique_xy(ref_x, ref_y)
    cx, cy = _unique_xy(candidate_x, candidate_y)
    if rx.size < 2 or cx.size < 2:
        return []
    lo = max(float(rx.min()), float(cx.min()))
    hi = min(float(rx.max()), float(cx.max()))
    if not hi > lo:
        return []
    grid = np.linspace(lo, hi, n)
    ref_i = np.interp(grid, rx, ry)
    cand_i = np.interp(grid, cx, cy)
    return [
        {
            "X_over_Rstar": float(x),
            "ref_R_over_Rstar": float(r_ref),
            "python_R_over_Rstar": float(r_py),
            "diff_R_over_Rstar": float(r_py - r_ref),
        }
        for x, r_ref, r_py in zip(grid, ref_i, cand_i)
    ]


def _table_series(table: LegacyTable, mapping: dict[str, str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for out_name, column_name in mapping.items():
        if column_name in table.columns:
            out[out_name] = table.column(column_name)
    return out


def _kernel_series_from_nodes(nodes) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([node.x for node in nodes], dtype=float),
        "r": np.asarray([node.r for node in nodes], dtype=float),
        "mach": np.asarray([node.M for node in nodes], dtype=float),
        "theta_deg": np.degrees([node.theta for node in nodes]),
    }


def _kernel_flat_series(kernel) -> dict[str, np.ndarray]:
    nodes = [node for rrc in kernel.rrcs for node in rrc]
    return _kernel_series_from_nodes(nodes)


def _row_flat_series(rows) -> dict[str, np.ndarray]:
    nodes = [node for row in rows for node in row]
    return _kernel_series_from_nodes(nodes)


def _bfe_wall_series(kernel, region) -> dict[str, np.ndarray]:
    wall_nodes = [rrc[0] for rrc in kernel.rrcs if rrc]
    wall_nodes.extend(getattr(region, "wall_contour", ()) or ())
    return _kernel_series_from_nodes(wall_nodes)


def _kernel_mass_max_rel_error(kernel) -> float | None:
    if kernel is None or not getattr(kernel, "massflow", None):
        return None
    mdot0 = float(kernel.massflow[0][0])
    if mdot0 <= 0.0:
        return None
    return float(max(
        abs(float(mass[0]) - mdot0) / mdot0
        for mass in kernel.massflow
    ))


def _wall_state_from_solution(solution) -> dict[str, np.ndarray]:
    diagnostics = getattr(solution, "construction_diagnostics", {}) or {}
    states: dict[str, np.ndarray] = {}
    if "wall_mach" in diagnostics:
        states["mach"] = np.asarray(diagnostics["wall_mach"], dtype=float)
    if "wall_theta" in diagnostics:
        states["theta_deg"] = np.degrees(np.asarray(diagnostics["wall_theta"], dtype=float))
    return states


def _centerline_from_solution(solution) -> dict[str, np.ndarray]:
    rows = getattr(solution, "characteristic_net", []) or []
    axes = [row.axis for row in rows if getattr(row, "axis", None) is not None]
    if not axes:
        return {}
    return {
        "x": np.asarray([p.x for p in axes], dtype=float),
        "r": np.asarray([p.r for p in axes], dtype=float),
        "mach": np.asarray([p.M for p in axes], dtype=float),
        "theta_deg": np.degrees([p.theta for p in axes]),
    }


def _solve_current(args: argparse.Namespace, nasa_wall: LegacyTable):
    if args.skip_solve:
        return None, "skipped by --skip-solve"
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    epsilon = (
        float(args.epsilon)
        if args.epsilon is not None
        else float(nasa_wall.column("R_over_Rstar")[-1]) ** 2
    )
    config = RaoSolverConfig(
        Rt=float(args.rt),
        epsilon=epsilon,
        gamma=float(args.gamma),
        pa_over_p0=float(args.pa_over_p0),
        length_pct=float(args.length_pct),
        throat_downstream_radius_factor=float(args.rd_rt),
        thetaN_guess_deg=float(args.theta_b_deg),
        n_control=int(args.n_control),
        n_kernel=int(args.solve_n_kernel),
        max_nfev=int(args.max_nfev),
        evaluate_moc=bool(args.evaluate_moc),
        starting_line_method=args.starting_line_method,
    )
    try:
        return solve_rao_bvp(config), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def _build_current_kernel(args: argparse.Namespace):
    from raosim.nasa_moc import build_kernel

    try:
        kernel = build_kernel(
            Rt=1.0,
            Rd=float(args.rd_rt),
            theta_B=math.radians(float(args.theta_b_deg)),
            gamma=float(args.gamma),
            n_kernel=int(args.kernel_n),
            starting_line_method=args.starting_line_method,
            mdot_tol=float(args.mdot_tol),
        )
        return kernel, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def _build_current_source_contour(args: argparse.Namespace, kernel, nasa_wall: LegacyTable):
    if kernel is None:
        return None, "kernel unavailable"
    if bool(getattr(kernel, "fallback_used", False)):
        return None, "kernel used fallback BD construction"
    if not bool(getattr(kernel, "reached_wall", False)):
        return None, "kernel did not reach the throat-arc wall"

    from raosim.nasa_moc import build_source_contour_from_kernel

    x_e = float(nasa_wall.column("X_over_Rstar")[-1])
    r_e = float(nasa_wall.column("R_over_Rstar")[-1])
    epsilon = float(args.epsilon) if args.epsilon is not None else r_e * r_e
    try:
        contour = build_source_contour_from_kernel(
            kernel,
            x_E=x_e,
            r_E=r_e,
            epsilon=epsilon,
            pa_over_p0=float(args.pa_over_p0),
            n_de_points=int(args.de_n),
        )
        region = contour.bfe
        if not region.complete_remaining_mesh:
            return contour, "CalcRemainingMesh stopped before all BFE rows were built"
        if not region.wall_contour_complete:
            return contour, "CalcWallContour did not mass-bracket every post-BD row"
        return contour, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"{type(exc).__name__}: {exc}"


def _kernel_report_status(kernel, kernel_error: str | None) -> tuple[str, bool, list[str]]:
    """Classify the Python kernel for NASA-reference reporting."""
    if kernel_error:
        return "error", False, [f"current Python kernel failed: {kernel_error}"]
    if kernel is None:
        return "unavailable", False, ["current Python kernel unavailable"]

    blockers: list[str] = []
    fallback_used = bool(getattr(kernel, "fallback_used", False))
    reached_wall = bool(getattr(kernel, "reached_wall", False))
    if fallback_used:
        blockers.append("current Python kernel used fallback BD construction")
    if not reached_wall:
        blockers.append("current Python kernel did not reach the throat-arc wall")

    complete = not blockers
    return ("ok" if complete else "partial"), complete, blockers


def compare(args: argparse.Namespace) -> dict[str, Any]:
    nasa_dir = Path(args.nasa_dir)
    diagnostics = _comparison_diagnostics(nasa_dir, args)
    wall = parse_wall_out(nasa_dir / "wall.out")
    center = parse_center_out(nasa_dir / "center.out")
    rao = parse_rao_dat(nasa_dir / "rao.dat")
    summary = parse_summary_out(nasa_dir / "summary.out")
    tt_prime = parse_tt_prime_out(nasa_dir / "TT'.out")
    ttbf = parse_kernel_out(nasa_dir / "TT'BF_Kernel.out")
    bfe = parse_kernel_out(nasa_dir / "BFE_Kernel.out")
    last = parse_last_kernel_out(nasa_dir / "LastKernel.out")
    uncropped = parse_uncropped_kernel_out(nasa_dir / "UncroppedKernel.out")

    solution, solve_error = _solve_current(args, wall)
    kernel, kernel_error = _build_current_kernel(args)
    kernel_status, kernel_complete, kernel_blockers = _kernel_report_status(
        kernel, kernel_error,
    )
    source_contour, bfe_error = _build_current_source_contour(args, kernel, wall)
    bfe_region = None if source_contour is None else source_contour.bfe
    diagnostics["python_kernel_complete"] = bool(kernel_complete)
    diagnostics["python_source_contour_available"] = source_contour is not None
    diagnostics["python_source_contour_complete"] = (
        False if source_contour is None
        else bool(source_contour.diagnostics.get("source_contour_complete", False))
    )
    if source_contour is not None:
        diagnostics["python_source_contour"] = source_contour.diagnostics
    diagnostics["python_bfe_overlay_available"] = bfe_region is not None
    diagnostics["python_bfe_overlay_complete"] = (
        bfe_region is not None
        and bool(getattr(bfe_region, "complete_remaining_mesh", False))
        and bool(getattr(bfe_region, "wall_contour_complete", False))
    )
    diagnostics["source_port_workflow_complete"] = bool(
        kernel_complete and diagnostics["python_bfe_overlay_complete"]
    )
    if diagnostics["source_port_workflow_complete"]:
        diagnostics["source_port_workflow_status"] = "complete_not_certified"
    else:
        diagnostics["source_port_workflow_status"] = "incomplete"
    if bfe_error:
        diagnostics["python_bfe_overlay_reason"] = bfe_error
    if kernel_blockers:
        diagnostics["nasa_reference_matched_blockers"].extend(kernel_blockers)

    metrics: list[dict[str, Any]] = []
    artifacts: dict[str, list[dict[str, float]]] = {}

    if solution is None:
        for family in ("wall.out", "center.out", "rao.dat"):
            metrics.append(_metric(f"{family}.comparison", None, status="unavailable",
                                   reason=solve_error or "current solve unavailable"))
    else:
        wall_py = np.asarray(solution.wall_export, dtype=float)
        wall_candidate = {
            "x": wall_py[:, 0] / float(args.rt),
            "r": wall_py[:, 1] / float(args.rt),
        }
        wall_candidate.update(_wall_state_from_solution(solution))
        wall_ref = _table_series(
            wall,
            {
                "x": "X_over_Rstar",
                "r": "R_over_Rstar",
                "mach": "mach",
                "theta_deg": "theta_deg",
                "p": "Pressure_psia",
            },
        )
        _compare_station_series(
            metrics,
            "wall.out",
            wall_ref,
            wall_candidate,
            ("x", "r", "mach", "theta_deg", "p"),
        )
        artifacts["wall_out_station.csv"] = _station_diff_rows(
            wall_ref, wall_candidate, ("x", "r", "mach", "theta_deg", "p")
        )
        value, n, reason = _rms_by_x(
            wall_ref["x"], wall_ref["r"],
            wall_candidate["x"], wall_candidate["r"],
        )
        metrics.append(_metric(
            "wall.out.R_over_Rstar.by_x_rms",
            None if reason else value,
            n=n,
            status="unavailable" if reason else "ok",
            reason=reason,
        ))
        artifacts["wall_out_r_by_x.csv"] = _by_x_diff_rows(
            wall_ref["x"], wall_ref["r"],
            wall_candidate["x"], wall_candidate["r"],
        )

        rao_ref = _table_series(
            rao,
            {"r": "R_over_Rstar", "x": "X_over_Rstar", "theta_deg": "theta_deg"},
        )
        _compare_station_series(
            metrics,
            "rao.dat",
            rao_ref,
            wall_candidate,
            ("x", "r", "theta_deg"),
        )
        artifacts["rao_dat_station.csv"] = _station_diff_rows(
            rao_ref, wall_candidate, ("x", "r", "theta_deg")
        )
        value, n, reason = _rms_by_x(
            rao_ref["x"], rao_ref["r"],
            wall_candidate["x"], wall_candidate["r"],
        )
        metrics.append(_metric(
            "rao.dat.R_over_Rstar.by_x_rms",
            None if reason else value,
            n=n,
            status="unavailable" if reason else "ok",
            reason=reason,
        ))
        artifacts["rao_dat_r_by_x.csv"] = _by_x_diff_rows(
            rao_ref["x"], rao_ref["r"],
            wall_candidate["x"], wall_candidate["r"],
        )

        center_candidate = _centerline_from_solution(solution)
        center_ref = _table_series(
            center,
            {
                "x": "X_over_Rstar",
                "r": "R_over_Rstar",
                "mach": "Mach",
                "theta_deg": "Theta",
            },
        )
        _compare_station_series(
            metrics,
            "center.out",
            center_ref,
            center_candidate,
            ("x", "r", "mach", "theta_deg"),
        )
        artifacts["center_out_station.csv"] = _station_diff_rows(
            center_ref, center_candidate, ("x", "r", "mach", "theta_deg")
        )

    if kernel is None:
        for family in ("TT'.out", "TT'BF_Kernel.out", "LastKernel.out", "UncroppedKernel.out"):
            metrics.append(_metric(f"{family}.comparison", None, status="unavailable",
                                   reason=kernel_error or "current kernel unavailable"))
    else:
        tt_ref = _table_series(
            tt_prime,
            {"x": "X", "r": "R", "mach": "MACH", "theta_deg": "THETA"},
        )
        _compare_station_series(
            metrics,
            "TT'.out",
            tt_ref,
            _kernel_series_from_nodes(kernel.rrcs[0]),
            ("x", "r", "mach", "theta_deg"),
        )
        artifacts["tt_prime_station.csv"] = _station_diff_rows(
            tt_ref,
            _kernel_series_from_nodes(kernel.rrcs[0]),
            ("x", "r", "mach", "theta_deg"),
        )
        last_ref = _table_series(
            last,
            {"x": "x", "r": "r", "mach": "mach", "theta_deg": "theta"},
        )
        _compare_station_series(
            metrics,
            "LastKernel.out",
            last_ref,
            _kernel_series_from_nodes(kernel.bd),
            ("x", "r", "mach", "theta_deg"),
        )
        artifacts["lastkernel_station.csv"] = _station_diff_rows(
            last_ref,
            _kernel_series_from_nodes(kernel.bd),
            ("x", "r", "mach", "theta_deg"),
        )
        flat_candidate = _kernel_flat_series(kernel)
        for family, table in (
            ("TT'BF_Kernel.out", ttbf),
            ("UncroppedKernel.out", uncropped),
        ):
            ref = _table_series(
                table,
                {"x": "x", "r": "r", "mach": "mach", "theta_deg": "theta"},
            )
            if family == "UncroppedKernel.out":
                ref = _table_series(
                    table,
                    {"x": "x_in", "r": "r_in", "mach": "mach", "theta_deg": "theta"},
                )
            _compare_station_series(
                metrics,
                family,
                ref,
                flat_candidate,
                ("x", "r", "mach", "theta_deg"),
            )
            artifact_name = (
                "ttbf_kernel_station.csv"
                if family == "TT'BF_Kernel.out"
                else "uncroppedkernel_station.csv"
            )
            artifacts[artifact_name] = _station_diff_rows(
                ref,
                flat_candidate,
                ("x", "r", "mach", "theta_deg"),
            )
        bfe_ref = _table_series(
            bfe,
            {"x": "x", "r": "r", "mach": "mach", "theta_deg": "theta"},
        )
        if bfe_region is None or not getattr(bfe_region, "grid_rows", ()):
            metrics.append(_metric(
                "BFE_Kernel.out.comparison",
                None,
                status="unavailable",
                reason=bfe_error or "current BFE overlay unavailable",
                n=bfe.data.shape[0],
            ))
        else:
            bfe_candidate = _row_flat_series(bfe_region.grid_rows)
            _compare_station_series(
                metrics,
                "BFE_Kernel.out",
                bfe_ref,
                bfe_candidate,
                ("x", "r", "mach", "theta_deg"),
            )
            artifacts["bfe_kernel_station.csv"] = _station_diff_rows(
                bfe_ref,
                bfe_candidate,
                ("x", "r", "mach", "theta_deg"),
            )
            bfe_wall_candidate = _bfe_wall_series(kernel, bfe_region)
            wall_ref = _table_series(
                wall,
                {
                    "x": "X_over_Rstar",
                    "r": "R_over_Rstar",
                    "mach": "mach",
                    "theta_deg": "theta_deg",
                },
            )
            _compare_station_series(
                metrics,
                "BFE_wall_contour",
                wall_ref,
                bfe_wall_candidate,
                ("x", "r", "mach", "theta_deg"),
            )
            artifacts["bfe_wall_contour_station.csv"] = _station_diff_rows(
                wall_ref,
                bfe_wall_candidate,
                ("x", "r", "mach", "theta_deg"),
            )

    return {
        "nasa_dir": str(nasa_dir),
        "summary": {
            "nozzle_type": summary.fields.get("Nozzle Type"),
            "gamma": summary.fields.get("Gamma"),
            "theta_b_deg": float(args.theta_b_deg),
            "rd_over_rt": float(args.rd_rt),
        },
        "python": {
            "solve_status": "skipped" if args.skip_solve else ("error" if solve_error else "ok"),
            "solve_error": solve_error,
            "kernel_status": kernel_status,
            "kernel_error": kernel_error,
            "kernel_complete": bool(kernel_complete),
            "kernel_rrcs": None if kernel is None else len(kernel.rrcs),
            "kernel_initial_row_points": None if kernel is None else len(kernel.rrcs[0]),
            "kernel_final_row_points": None if kernel is None else len(kernel.rrcs[-1]),
            "kernel_mass_max_rel_error": _kernel_mass_max_rel_error(kernel),
            "kernel_fallback_used": None if kernel is None else bool(getattr(kernel, "fallback_used", False)),
            "kernel_reached_wall": None if kernel is None else bool(getattr(kernel, "reached_wall", False)),
            "source_contour_status": (
                "unavailable"
                if source_contour is None
                else (
                    "ok"
                    if source_contour.diagnostics.get("source_contour_complete", False)
                    else "partial"
                )
            ),
            "source_contour_length_closed": (
                None if source_contour is None
                else bool(source_contour.diagnostics.get("length_closed", False))
            ),
            "source_contour_exit_rel_error": (
                None if source_contour is None
                else source_contour.diagnostics.get("exit_rel_error")
            ),
            "source_contour_wall_points": (
                None if source_contour is None else len(source_contour.wall)
            ),
            "bfe_status": "unavailable" if bfe_region is None else ("partial" if bfe_error else "ok"),
            "bfe_error": bfe_error,
            "bfe_seed_rows": None if bfe_region is None else len(getattr(bfe_region, "rows", ())),
            "bfe_grid_rows": None if bfe_region is None else len(getattr(bfe_region, "grid_rows", ())),
            "bfe_wall_points": None if bfe_region is None else len(getattr(bfe_region, "wall_contour", ())),
            "bfe_complete_remaining_mesh": None if bfe_region is None else bool(getattr(bfe_region, "complete_remaining_mesh", False)),
            "bfe_wall_contour_complete": None if bfe_region is None else bool(getattr(bfe_region, "wall_contour_complete", False)),
        },
        "diagnostics": diagnostics,
        "metrics": metrics,
        "_artifacts": artifacts,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"NASA/JHU reference: {report['nasa_dir']}")
    diagnostics = report.get("diagnostics", {})
    if diagnostics:
        print(
            "Canonical reference track: "
            f"{diagnostics.get('canonical_reference_track')}"
        )
        print(f"Comparison track: {diagnostics.get('comparison_track')}")
        print(
            "Fixture provenance: "
            f"{diagnostics.get('fixture_generator_provenance')}"
        )
        print(
            "Fixture promotion authority: "
            f"{diagnostics.get('fixture_overlay_is_promotion_authority')}"
        )
        print(
            "Source workflow: "
            f"{diagnostics.get('source_port_workflow_status')}"
        )
        print(
            "Source port matched: "
            f"{diagnostics.get('source_port_match_status')}"
        )
        print(
            "NASA_REFERENCE_MATCHED eligible: "
            f"{diagnostics.get('nasa_reference_matched_eligible')}"
        )
    print(
        "Python kernel: "
        f"{report['python']['kernel_status']}, "
        f"rrcs={report['python']['kernel_rrcs']}, "
        f"fallback={report['python']['kernel_fallback_used']}, "
        f"reached_wall={report['python']['kernel_reached_wall']}"
    )
    if report["python"].get("kernel_mass_max_rel_error") is not None:
        print(
            "Kernel mass max rel error: "
            f"{report['python']['kernel_mass_max_rel_error']:.6g}"
        )
    print(f"Python solve: {report['python']['solve_status']}")
    if report["python"].get("solve_error"):
        print(f"  solve_error: {report['python']['solve_error']}")
    if report["python"].get("kernel_error"):
        print(f"  kernel_error: {report['python']['kernel_error']}")
    print()

    current_family = None
    for metric in report["metrics"]:
        parts = metric["name"].split(".")
        family = ".".join(parts[:2]) if len(parts) > 1 else metric["name"]
        if family != current_family:
            current_family = family
            print(f"{family}")
        if metric["status"] == "ok":
            print(f"  {metric['name']}: {metric['value']:.6g} (n={metric['n']})")
        else:
            print(f"  {metric['name']}: unavailable ({metric['reason']})")


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "_artifacts"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(report: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    public = _public_report(report)
    written: list[Path] = []

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    written.append(report_path)

    metrics_path = output_dir / "metrics.csv"
    _write_csv(metrics_path, public["metrics"])
    written.append(metrics_path)

    for filename, rows in sorted(report.get("_artifacts", {}).items()):
        if not rows:
            continue
        path = output_dir / filename
        _write_csv(path, rows)
        written.append(path)

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# NASA Comparison Artifacts\n\n"
        "Generated by `scripts/compare_nasa_reference.py`.\n\n"
        "- `report.json`: machine-readable summary, provenance diagnostics, and RMS metrics.\n"
        "- `metrics.csv`: one row per reported RMS/unavailable metric.\n"
        "- `*_station.csv`: normalized station-wise reference/Python diffs.\n"
        "- `*_r_by_x.csv`: wall-radius diffs on a common `X/R*` grid.\n"
        "\n"
        "The report uses a historical fixture overlay track. It does not claim\n"
        "that the visible NASA/JHU source generated the checked-in TT' fixture.\n"
    )
    written.append(readme_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nasa-dir", type=Path, default=NASA_OUT_DEFAULT)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write report.json, metrics.csv, and diff CSVs to this directory",
    )
    parser.add_argument("--skip-solve", action="store_true", help="only compare build_kernel outputs")
    parser.add_argument("--evaluate-moc", action="store_true", help="ask solve_rao_bvp to keep MOC rows")
    parser.add_argument("--rt", type=float, default=0.0254, help="Python throat radius in metres")
    parser.add_argument("--rd-rt", type=float, default=1.0, help="downstream throat radius ratio")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--pa-over-p0", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--length-pct", type=float, default=100.0)
    parser.add_argument("--theta-b-deg", type=float, default=15.2196)
    parser.add_argument("--n-control", type=int, default=20)
    parser.add_argument("--solve-n-kernel", type=int, default=20)
    parser.add_argument("--kernel-n", type=int, default=101)
    parser.add_argument("--de-n", type=int, default=24)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--mdot-tol", type=float, default=0.05)
    parser.add_argument("--starting-line-method", default="kliegel_levine")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = compare(args)
    if args.output_dir is not None:
        written = write_artifacts(report, args.output_dir)
        report["artifacts_written"] = [str(path) for path in written]
    if args.json:
        print(json.dumps(_public_report(report), indent=2, sort_keys=True))
    else:
        _print_human(_public_report(report))
        if args.output_dir is not None:
            print()
            print(f"Wrote artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
