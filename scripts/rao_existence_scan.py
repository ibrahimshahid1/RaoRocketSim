#!/usr/bin/env python3
"""Run a NumPy-only Rao D-attachment existence scan.

Example:
    PYTHONPATH=. python scripts/rao_existence_scan.py \
      --Rt 0.020 --epsilon 10 --length-pct 80 \
      --theta-b-center-deg 24 --theta-b-span-deg 12 --theta-b-count 31 \
      --kdf-min 0.05 --kdf-max 0.85 --kdf-count 31 \
      --models smooth,position,fan --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from raosim.rao_existence_scan import (
    ExistenceScanConfig,
    MODELS,
    SCAN_MODES,
    STOP_MODES,
    THIRD_COMPONENTS,
    plot_scan_heatmaps,
    refine_from_scan_best,
    resolution_convergence,
    scan_existence,
    stationarity_derivation_summary,
    write_root_results,
    write_scan_tables,
)


def _csv_models(value: str) -> tuple[str, ...]:
    models = tuple(v.strip() for v in value.split(",") if v.strip())
    unknown = set(models).difference(MODELS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {sorted(unknown)}; valid: {MODELS}"
        )
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map smooth/position/fan D-attachment residual fields without LM/JAX.",
    )
    parser.add_argument("--Rt", type=float, default=0.020)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--pa-over-p0", type=float, default=0.01)
    parser.add_argument("--length-pct", type=float, default=80.0)
    parser.add_argument("--theta-b-center-deg", type=float, default=24.0)
    parser.add_argument("--theta-b-span-deg", type=float, default=12.0)
    parser.add_argument("--theta-b-count", type=int, default=25)
    parser.add_argument("--theta-b-values-deg", type=str, default=None,
                        help="comma-separated theta_B values; overrides center/span/count")
    parser.add_argument("--kdf-min", type=float, default=0.05)
    parser.add_argument("--kdf-max", type=float, default=0.85)
    parser.add_argument("--kdf-count", type=int, default=25)
    parser.add_argument("--kdf-values", type=str, default=None,
                        help="comma-separated kdf values; overrides min/max/count")
    parser.add_argument("--models", type=_csv_models, default=MODELS,
                        help="comma-separated subset of smooth,position,fan")
    parser.add_argument("--stop-at", choices=STOP_MODES, default="mass",
                        help="mass mirrors current topology; radius/length expose mass residual directly")
    parser.add_argument("--scan-mode", choices=SCAN_MODES, default="geometry",
                        help="geometry: radius+length; stationarity: radius+length+sigma_E; diagnostic: report only")
    parser.add_argument("--third-residual", choices=THIRD_COMPONENTS, default="auto")
    parser.add_argument("--n-kernel", type=int, default=16)
    parser.add_argument("--n-de-points", type=int, default=24)
    parser.add_argument("--position-theta-span-deg", type=float, default=10.0)
    parser.add_argument("--position-theta-count", type=int, default=9)
    parser.add_argument("--position-mach-down", type=float, default=0.5)
    parser.add_argument("--position-mach-up", type=float, default=1.5)
    parser.add_argument("--position-mach-count", type=int, default=9)
    parser.add_argument("--fan-turn-min-deg", type=float, default=0.0)
    parser.add_argument("--fan-turn-max-deg", type=float, default=12.0)
    parser.add_argument("--fan-turn-count", type=int, default=13)
    parser.add_argument("--max-mach", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("debug_outputs/rao_existence_scan"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--compare-solver", action="store_true",
                        help="Optional: run the current Rao BVP once and write solver_compare.json")
    parser.add_argument("--solver-backend", choices=("jax", "numpy"), default="jax")
    parser.add_argument("--solver-max-nfev", type=int, default=500)
    parser.add_argument("--root-refine", action="store_true",
                        help="Refine theta_B/kdf roots from each model's best grid cell")
    parser.add_argument("--root-models", type=_csv_models, default=None,
                        help="comma-separated root models; default smooth,fan when present")
    parser.add_argument("--root-maxfev", type=int, default=80)
    parser.add_argument("--resolution-convergence", type=str, default=None,
                        help="comma-separated n_kernel values; requires or reuses root-refine seed")
    parser.add_argument("--print-stationarity-note", action="store_true")
    return parser


def _parse_float_list(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    vals = [float(v.strip()) for v in value.split(",") if v.strip()]
    return np.asarray(vals, dtype=float)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _write_solver_compare(config: ExistenceScanConfig, args: argparse.Namespace) -> None:
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    solver_cfg = RaoSolverConfig(
        Rt=config.Rt,
        epsilon=config.epsilon,
        gamma=config.gamma,
        pa_over_p0=config.pa_over_p0,
        length_pct=config.length_pct,
        throat_downstream_radius_factor=config.throat_downstream_radius_factor,
        throat_upstream_radius_factor=config.throat_upstream_radius_factor,
        n_control=max(8, config.n_de_points),
        n_kernel=config.n_kernel,
        max_nfev=args.solver_max_nfev,
        evaluate_moc=False,
        solver_backend=args.solver_backend,
        formulation="characteristic",
    )
    solution = solve_rao_bvp(solver_cfg)
    design_angles = solution.construction_diagnostics.get("design_angles", {})
    payload = {
        "backend": args.solver_backend,
        "max_nfev": args.solver_max_nfev,
        "optimizer_success": bool(solution.control_surface.optimizer_success),
        "solver_message": solution.control_surface.solver_message,
        "residual_max_scaled": float(solution.residuals.max_scaled),
        "residual_rms_scaled": float(solution.residuals.rms_scaled),
        "mass_residual_rel": float(solution.residuals.mass_residual_rel),
        "length_residual_rel": float(solution.residuals.length_residual_rel),
        "kernel_d_fraction": float(solution.control_surface.kernel_d_fraction),
        "theta_N_reported_deg": float(np.degrees(solution.theta_N)),
        "theta_E_reported_deg": float(np.degrees(solution.theta_E)),
        "theta_N_source": design_angles.get("theta_N_source"),
        "theta_E_source": design_angles.get("theta_E_source"),
        "D_theta_deg": float(np.degrees(solution.control_surface.theta[0])),
        "D_M": float(solution.control_surface.M[0]),
    }
    with (args.output_dir / "solver_compare.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = ExistenceScanConfig(
        Rt=args.Rt,
        epsilon=args.epsilon,
        gamma=args.gamma,
        pa_over_p0=args.pa_over_p0,
        length_pct=args.length_pct,
        theta_b_values_deg=_parse_float_list(args.theta_b_values_deg),
        kdf_values=_parse_float_list(args.kdf_values),
        theta_b_center_deg=args.theta_b_center_deg,
        theta_b_span_deg=args.theta_b_span_deg,
        theta_b_count=args.theta_b_count,
        kdf_min=args.kdf_min,
        kdf_max=args.kdf_max,
        kdf_count=args.kdf_count,
        models=args.models,
        stop_at=args.stop_at,
        scan_mode=args.scan_mode,
        third_residual=args.third_residual,
        n_kernel=args.n_kernel,
        n_de_points=args.n_de_points,
        position_theta_span_deg=args.position_theta_span_deg,
        position_theta_count=args.position_theta_count,
        position_mach_down=args.position_mach_down,
        position_mach_up=args.position_mach_up,
        position_mach_count=args.position_mach_count,
        fan_turn_min_deg=args.fan_turn_min_deg,
        fan_turn_max_deg=args.fan_turn_max_deg,
        fan_turn_count=args.fan_turn_count,
        max_mach=args.max_mach,
    )

    result = scan_existence(config)
    write_scan_tables(result, args.output_dir)
    if args.plot:
        plot_scan_heatmaps(result, args.output_dir)
    if args.print_stationarity_note:
        with (args.output_dir / "stationarity_note.txt").open("w", encoding="utf-8") as f:
            f.write(stationarity_derivation_summary() + "\n")
    root_results = {}
    if args.root_refine or args.resolution_convergence:
        requested = args.root_models
        if requested is None:
            requested = tuple(m for m in ("smooth", "fan") if m in result.closures)
        for model in requested:
            if model == "position" or model not in result.closures:
                continue
            root_results[model] = refine_from_scan_best(
                result, model=model, maxfev=args.root_maxfev,
            )
        if root_results:
            write_root_results(root_results, args.output_dir)
    if args.resolution_convergence:
        n_values = _parse_int_list(args.resolution_convergence)
        if n_values:
            conv_payload = {}
            seed_source = root_results or {}
            requested = args.root_models or tuple(m for m in ("smooth", "fan") if m in result.closures)
            for model in requested:
                if model == "position":
                    continue
                if model in seed_source:
                    seed = seed_source[model]
                    theta_seed = seed.theta_B_deg
                    kdf_seed = seed.kdf
                    fan_turn = seed.fan_turn_deg
                elif model in result.closures:
                    best = result.closures[model].best_summary()
                    theta_seed = float(best["theta_B_deg"])
                    kdf_seed = float(best["kdf"])
                    fan_turn = float(best.get("fan_turn_deg", 0.0))
                else:
                    continue
                rows = resolution_convergence(
                    config,
                    n_kernel_values=n_values,
                    model=model,
                    theta_B_seed_deg=theta_seed,
                    kdf_seed=kdf_seed,
                    fan_turn_deg=fan_turn,
                    maxfev=args.root_maxfev,
                )
                conv_payload[model] = [row.to_dict() for row in rows]
            with (args.output_dir / "resolution_convergence.json").open("w", encoding="utf-8") as f:
                json.dump(conv_payload, f, indent=2, sort_keys=True)
    if args.compare_solver:
        _write_solver_compare(config, args)

    print(json.dumps(result.summary()["best"], indent=2, sort_keys=True))
    print(f"Wrote scan outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
