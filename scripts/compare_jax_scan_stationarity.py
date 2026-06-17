#!/usr/bin/env python3
"""Compare one Rao BVP solution against independent scan stationarity metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from raosim.rao_existence_scan import (  # noqa: E402
    ExistenceScanConfig,
    MODEL_SMOOTH,
    SCAN_STATIONARITY,
    evaluate_closure_point,
    stationarity_derivation_summary,
)
from raosim.rao_variational import (  # noqa: E402
    RaoSolverConfig,
    _pack_bvp,
    _scaled_rao_bvp_residual,
    solve_rao_bvp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Side-by-side JAX BVP residual vs NumPy scan sigma_E.",
    )
    parser.add_argument("--Rt", type=float, default=0.020)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--pa-over-p0", type=float, default=0.01)
    parser.add_argument("--length-pct", type=float, default=80.0)
    parser.add_argument("--n-control", type=int, default=24)
    parser.add_argument("--n-kernel", type=int, default=8)
    parser.add_argument("--n-de-points", type=int, default=24)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--solver-backend", choices=("jax", "numpy"), default="jax")
    parser.add_argument("--output", type=Path,
                        default=Path("debug_outputs/rao_stationarity_compare.json"))
    return parser


def _summarize_vector(vec: np.ndarray) -> dict:
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return {"size": 0, "max_abs": 0.0, "rms": 0.0}
    return {
        "size": int(vec.size),
        "max_abs": float(np.max(np.abs(vec))),
        "rms": float(np.sqrt(np.mean(vec ** 2))),
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    solver_cfg = RaoSolverConfig(
        Rt=args.Rt,
        epsilon=args.epsilon,
        gamma=args.gamma,
        pa_over_p0=args.pa_over_p0,
        length_pct=args.length_pct,
        n_control=args.n_control,
        n_kernel=args.n_kernel,
        max_nfev=args.max_nfev,
        evaluate_moc=False,
        solver_backend=args.solver_backend,
        formulation="characteristic",
    )
    solution = solve_rao_bvp(solver_cfg)
    kernel_bd = tuple(p.to_flow_node() for p in solution.kernel_points)
    residual_cfg = replace(solver_cfg, kernel_bd=kernel_bd)
    ce = solution.control_surface
    u = _pack_bvp(ce, ce.lambda2, ce.lambda3, ce.log_C)
    numpy_residual = np.asarray(_scaled_rao_bvp_residual(u, ce.r, residual_cfg))

    jax_residual_summary = None
    try:
        import jax.numpy as jnp
        from raosim.jax import assembly

        sp = assembly.params_from_config(residual_cfg)
        fn = assembly.make_residual(sp)
        jax_residual = np.asarray(fn(jnp.asarray(u)))
        jax_residual_summary = _summarize_vector(jax_residual)
    except Exception as exc:
        jax_residual_summary = {"error": str(exc)}

    scan_cfg = ExistenceScanConfig(
        Rt=args.Rt,
        epsilon=args.epsilon,
        gamma=args.gamma,
        pa_over_p0=args.pa_over_p0,
        length_pct=args.length_pct,
        n_kernel=args.n_kernel,
        n_de_points=args.n_de_points,
        scan_mode=SCAN_STATIONARITY,
        models=(MODEL_SMOOTH,),
    )
    theta_B_deg = float(math.degrees(solution.theta_N))
    kdf = float(ce.kernel_d_fraction)
    scan_eval = evaluate_closure_point(
        scan_cfg,
        model=MODEL_SMOOTH,
        theta_B_deg=theta_B_deg,
        kdf=kdf,
    )
    design_angles = solution.construction_diagnostics.get("design_angles", {})
    payload = {
        "stationarity_note": stationarity_derivation_summary(),
        "solver": {
            "backend": args.solver_backend,
            "optimizer_success": bool(ce.optimizer_success),
            "solver_message": ce.solver_message,
            "reported_theta_B_deg": theta_B_deg,
            "theta_N_source": design_angles.get("theta_N_source"),
            "reported_theta_E_deg": float(math.degrees(solution.theta_E)),
            "theta_E_source": design_angles.get("theta_E_source"),
            "kdf": kdf,
            "D": {
                "theta_deg": float(math.degrees(ce.theta[0])),
                "M": float(ce.M[0]),
            },
            "residual_report": {
                "max_scaled": float(solution.residuals.max_scaled),
                "rms_scaled": float(solution.residuals.rms_scaled),
                "mass_residual_rel": float(solution.residuals.mass_residual_rel),
                "length_residual_rel": float(solution.residuals.length_residual_rel),
            },
            "numpy_assembled_residual": _summarize_vector(numpy_residual),
            "jax_assembled_residual": jax_residual_summary,
        },
        "independent_scan_at_solver_coordinate": {
            "theta_B_deg": theta_B_deg,
            "kdf": kdf,
            "radius_residual": float(scan_eval["radius_residual"]),
            "length_residual": float(scan_eval["length_residual"]),
            "mass_residual": float(scan_eval["mass_residual"]),
            "sigma_E_rad": float(scan_eval["sigma_E_rad"]),
            "rao_exit_residual_relative": float(scan_eval["rao_exit_residual"]),
            "performance_residual_diagnostic": float(scan_eval["performance_residual"]),
            "D": {
                "x": float(scan_eval["d_x"]),
                "r": float(scan_eval["d_r"]),
                "theta_pre_deg": float(scan_eval["d_theta_pre_deg"]),
                "theta_post_deg": float(scan_eval["d_theta_post_deg"]),
                "M_pre": float(scan_eval["d_mach_pre"]),
                "M_post": float(scan_eval["d_mach_post"]),
            },
            "E": {
                "x": float(scan_eval["exit_x"]),
                "r": float(scan_eval["exit_r"]),
                "theta_deg": float(scan_eval["exit_angle_deg"]),
                "M": float(scan_eval["exit_mach"]),
                "theta_Rao_E_deg": float(scan_eval["theta_rao_E_deg"]),
            },
        },
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
