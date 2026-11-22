"""Reproduce one MOC gate profile, optionally checking common-coordinate refinement."""
from __future__ import annotations

from lrekit.paths import package_root as _package_root, repository_root as _repository_root

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from lrekit.verification.benchmarks import load_benchmark_case
from lrekit.components.nozzle.solvers.variational import (
    RaoSolverConfig,
    _physical_bde_field_arrays,
    compare_rao_solutions_common_coordinates,
    solve_rao_bvp,
)
from lrekit.core.gas_dynamics import isentropic_pressure_ratio


def profile_config(*, fine: bool = False, exact_exit: bool = False,
                   seed_max_iter: int = 8, seed_abs_tol: float = 1e-6) -> RaoSolverConfig:
    return RaoSolverConfig(
        Rt=1.0, epsilon=4.973, gamma=1.23, pa_over_p0=0.0,
        length_pct=64.0452248702277, n_control=24 if fine else 16,
        n_kernel=32 if fine else 24, kernel_dtheta_limit_deg=0.125 if fine else 0.25,
        max_nfev=1000, residual_tol=0.005, evaluate_moc=True,
        wall_method="bde", solver_backend="numpy", exact_d_attachment=True,
        physics_weight=1.0, throat_upstream_radius_factor=1.5,
        throat_downstream_radius_factor=0.382,
        exact_exit_attachment=exact_exit, kernel_seed_max_iter=seed_max_iter,
        kernel_seed_abs_tol=seed_abs_tol,
    )


def published_wall_comparison(config, solution) -> dict:
    artifacts = solution.construction_diagnostics["bde_artifacts"]
    nodes = [row[0] for row in artifacts["kernel"].rrcs if row]
    nodes.extend(artifacts["bde_region"].wall_contour)
    data = np.asarray(sorted(
        (float(node.x) / config.Rt, float(node.M), math.degrees(float(node.theta)))
        for node in nodes
    ))
    _, first = np.unique(data[:, 0], return_index=True)
    data = data[first]
    source = _package_root() / "benchmark_data/curves/rao1958_nozzle_b_table2.csv"
    samples = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            x = float(row["x_over_rt"])
            available = bool(data[0, 0] <= x <= data[-1, 0])
            mach = float(np.interp(x, data[:, 0], data[:, 1])) if available else None
            theta = float(np.interp(x, data[:, 0], data[:, 2])) if available else None
            reference_mach = float(row["mach"])
            reference_theta = float(row["wall_slope_deg"])
            samples.append({
                "x_over_rt": x, "available": available,
                "published_M": reference_mach, "computed_M": mach,
                "computed_minus_published_M": None if mach is None else mach - reference_mach,
                "published_theta_deg": reference_theta, "computed_theta_deg": theta,
                "computed_minus_published_theta_deg": None if theta is None else theta - reference_theta,
            })
    return {
        "classification": "report_only_no_acceptance_tolerance",
        "source": "Rao 1958 Table 2, PDF page5, printed381",
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "interpolation": "linear in actual kernel/BDE wall-node x/Rt; first duplicate x retained; no extrapolation",
        "limitations": "Source throat/start-line assumptions differ and source values are rounded; these differences are not new gates.",
        "samples": samples,
    }


def summarize(config, solution) -> dict:
    diagnostics = solution.construction_diagnostics
    kernel = diagnostics["bde_artifacts"]["kernel"]
    return {
        "config": asdict(config),
        "solver_mesh_context": diagnostics["solver_mesh_context"],
        "gate_report": diagnostics["gate_report"],
        "Cf": float(solution.thrust_coefficient),
        "converged": bool(solution.converged),
        "reliability": solution.reliability.value,
        "export": diagnostics.get("export"),
        "endpoint_mismatch": diagnostics.get("endpoint_mismatch"),
        "exact_exit_domain": diagnostics.get("exact_exit_domain"),
        "kernel_seed_search": diagnostics.get("kernel_seed_search"),
        "raw_exit_m": solution.wall_raw[-1].tolist(),
        "target_exit_x_m": float(solution.control_surface.length_target),
        "warnings": list(solution.warnings),
        "published_wall_comparison": published_wall_comparison(config, solution),
        "kernel_seed": {
            "theta_B_deg": math.degrees(kernel.theta_B),
            "theta_B_provenance": kernel.theta_b_provenance,
            "reached_wall": bool(kernel.reached_wall),
            "fallback_used": bool(kernel.fallback_used),
            "row_count": len(kernel.rrcs), "max_rrcs": kernel.max_rrcs,
        },
    }


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def worst_field_differences(coarse, fine, *, Rt, gamma):
    """Report locations with the same bidirectional interpolation as the gate."""
    from scipy.interpolate import LinearNDInterpolator

    fields = [_physical_bde_field_arrays(solution, Rt=Rt) for solution in (coarse, fine)]
    if any(field is None for field in fields):
        return {"available": False}
    values = [np.column_stack((M, theta, isentropic_pressure_ratio(M, gamma)))
              for _, M, theta in fields]
    records = {}
    for target, other in ((0, 1), (1, 0)):
        xy = fields[target][0]
        predicted = np.asarray(LinearNDInterpolator(fields[other][0], values[other])(xy))
        valid = np.all(np.isfinite(predicted), axis=1)
        for column, (name, scale, unit) in enumerate((
            ("mach", 1.0, "Mach"), ("theta", 180.0 / math.pi, "deg"),
            ("pressure_ratio", 1.0, "p/p0"),
        )):
            differences = (predicted[valid, column] - values[target][valid, column]) * scale
            if not differences.size:
                continue
            row = int(np.argmax(np.abs(differences)))
            record = {
                "coordinate_x_over_rt_r_over_rt": xy[valid][row].tolist(),
                "target_mesh": "coarse" if target == 0 else "fine",
                "interpolated_mesh": "fine" if target == 0 else "coarse",
                "target_value": float(values[target][valid, column][row] * scale),
                "interpolated_value": float(predicted[valid, column][row] * scale),
                "signed_interpolated_minus_target": float(differences[row]), "unit": unit,
            }
            if name not in records or abs(differences[row]) > abs(records[name]["signed_interpolated_minus_target"]):
                records[name] = record
    return {"available": True, "classification": "report_only_location_of_existing_comparison_maximum", "fields": records}


def save_progress(directory, result, level):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "progress.json").write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_level": level, "promotion": result[level]["gate_report"]["promotion"]}), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("builds/moc_reference_check"))
    parser.add_argument("--exact-exit", action="store_true", help="eliminate the prescribed exit boundary unknowns")
    parser.add_argument("--seed-max-iter", type=int, default=8)
    parser.add_argument("--seed-abs-tol", type=float, default=1e-6)
    level_flags = parser.add_mutually_exclusive_group()
    level_flags.add_argument("--refine", action="store_true", help="run two fixed refinement levels")
    level_flags.add_argument("--levels", type=int, choices=(1, 2, 3), help="number of fixed refinement levels")
    args = parser.parse_args(argv)
    levels = args.levels or (2 if args.refine else 1)
    profile_options = dict(exact_exit=args.exact_exit, seed_max_iter=args.seed_max_iter,
                           seed_abs_tol=args.seed_abs_tol)
    coarse_config = profile_config(**profile_options)
    coarse = solve_rao_bvp(coarse_config)
    result = {
        "schema_version": 1,
        "scope": "single inviscid constant-gamma configuration; no domain-wide or physical validation",
        "model_assumptions": load_benchmark_case("rao1958_nozzle_b")["model_assumptions"],
        "coarse": summarize(coarse_config, coarse),
        "levels": levels,
        "acceptance_scope": "single_level_internal_gates" if levels == 1 else "last_pair_internal_gates_and_common_coordinate_policy",
    }
    passes = bool(coarse.converged)
    save_progress(args.output_dir, result, "coarse")
    if levels >= 2:
        fine_config = profile_config(fine=True, **profile_options)
        fine = solve_rao_bvp(fine_config)
        evidence = compare_rao_solutions_common_coordinates(
            coarse, fine, Rt=coarse_config.Rt, gamma=coarse_config.gamma,
        )
        result["fine"] = summarize(fine_config, fine)
        result["refinement"] = evidence.to_dict()
        result["refinement_worst_field_locations"] = worst_field_differences(
            coarse, fine, Rt=coarse_config.Rt, gamma=coarse_config.gamma,
        )
        passes = bool(passes and fine.converged and evidence.applicable and evidence.passes)
        result["acceptance_context"] = "fine"
        save_progress(args.output_dir, result, "fine")
    if levels == 3:
        finest_config = replace(
            fine_config, n_control=32, n_kernel=48, kernel_dtheta_limit_deg=0.0625,
            kernel_max_rrcs=1000,
        )
        finest = solve_rao_bvp(finest_config)
        evidence = compare_rao_solutions_common_coordinates(
            fine, finest, Rt=coarse_config.Rt, gamma=coarse_config.gamma,
        )
        result["finest"] = summarize(finest_config, finest)
        result["refinement_fine_to_finest"] = evidence.to_dict()
        result["refinement_fine_to_finest_worst_field_locations"] = worst_field_differences(
            fine, finest, Rt=coarse_config.Rt, gamma=coarse_config.gamma,
        )
        passes = bool(fine.converged and finest.converged and evidence.applicable and evidence.passes)
        result["acceptance_context"] = "finest"
        save_progress(args.output_dir, result, "finest")
    repo_root = _repository_root()
    result["source_sha256"] = {
        name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
        for name in (
            "raosim/rao_variational.py", "raosim/nasa_moc.py",
            "raosim/rao_residuals.py", "raosim/transonic_kernel.py",
            "tests/experiments/nozzle/audit_moc_reference_profile.py",
        )
    }
    result["exit_code"] = 0 if passes else 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(path.resolve()), "exit_code": result["exit_code"]}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
