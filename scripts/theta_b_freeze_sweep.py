"""THE decisive experiment for the flare/optimum question (2026-06-11f).

Sweeps ``theta_b_freeze_deg`` with FULL D-state continuity
(pin_d_theta + pin_d_mach) at the reference point (eps=10, L80, gamma
1.4).  The theta_N reconciliation (plan STATUS; Rao ARS J. 1961
pp. 1490-1491 + the in-repo chart) predicts:

  * the full-pin stationarity floor (5.7e-2 when the kernel was frozen
    at the fixed-end angle ~25.5 deg) collapses near theta_B ~ 30 deg
    (chart theta_N at eps=10/L80);
  * at the best theta_B the gate (2e-3) closes WITH full continuity;
  * the BDE wall peaks ~theta_B right after the throat arc (mid-bell
    flare gone) and the exit angle drops toward the chart theta_E
    (15.5 deg).

If the floor does NOT collapse anywhere in the band, the
Guderley-discontinuity hypothesis (optimum genuinely discontinuous at
D for fixed-L nozzles) becomes the live branch.

Run:  PYTHONPATH=. python scripts/theta_b_freeze_sweep.py
      (optionally: --band 26 31 --step 1.0)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig

OUT = Path("builds/theta_b_freeze_sweep.json")

rv.PHYSICS_WEIGHT = 1.0


def config_at(theta_b_deg: float, *, wall: bool = False) -> RaoSolverConfig:
    return RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=24, n_kernel=24, n_wall=12,
        max_nfev=4000, residual_tol=2e-3,
        evaluate_moc=bool(wall),
        wall_method="bde" if wall else "coupled",
        couple_wall=False, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=theta_b_deg,
        theta_b_freeze_deg=theta_b_deg,
        pin_d_theta=True, pin_d_mach=True,
        jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--band", nargs=2, type=float, default=(26.0, 31.0))
    ap.add_argument("--step", type=float, default=1.0)
    args = ap.parse_args()

    grid = np.arange(args.band[0], args.band[1] + 1e-9, args.step)
    rows = []
    print(f"theta_B sweep {grid[0]:.1f}..{grid[-1]:.1f} step {args.step}, "
          f"FULL pins, characteristic+ladder")
    for tb in grid:
        sol = rv.solve_rao_bvp(config_at(float(tb)))
        r = sol.residuals
        cs = sol.control_surface
        groups = {g["name"]: g["max"] for g in r.group_summaries}
        rows.append({
            "theta_b_deg": float(tb),
            "max_scaled": float(r.max_scaled),
            "stationarity_max": float(
                groups.get("algebraic_stationarity", float("nan"))),
            "kdf": float(cs.kernel_d_fraction),
            "mass": float(r.mass_residual_rel),
            "len": float(r.length_residual_rel),
            "converged": bool(cs.converged),
        })
        print(f"  tb={tb:5.2f}  max_scaled={r.max_scaled:.4e}  "
              f"stat={rows[-1]['stationarity_max']:.4e}  "
              f"kdf={cs.kernel_d_fraction:.4f}", flush=True)

    best = min(rows, key=lambda d: d["max_scaled"])
    print(f"\nbest: theta_B={best['theta_b_deg']:.2f}  "
          f"max_scaled={best['max_scaled']:.4e}  gate(2e-3): "
          f"{best['max_scaled'] <= 2e-3}")

    # Wall shape at the best theta_B.
    wall_row = None
    sol = rv.solve_rao_bvp(config_at(best["theta_b_deg"], wall=True))
    w = sol.wall_raw
    if w is not None and len(w) > 5:
        w = np.asarray(w)
        ang = np.degrees(np.arctan2(np.diff(w[:, 1]), np.diff(w[:, 0])))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(w[:, 0]),
                                                      np.diff(w[:, 1])))])
        ipk = int(np.argmax(ang))
        wall_row = {
            "peak_deg": float(ang.max()),
            "peak_pos_frac": float(s[ipk] / s[-1]),
            "exit_deg": float(ang[-1]),
            "monotone_violations": int((np.diff(ang[ipk:]) > 0.25).sum()),
            "exit_xy": [float(w[-1, 0]), float(w[-1, 1])],
        }
        print(f"wall @best: peak={wall_row['peak_deg']:.2f} deg at "
              f"{wall_row['peak_pos_frac']:.1%}, exit="
              f"{wall_row['exit_deg']:.2f} deg, "
              f"viol={wall_row['monotone_violations']}")
        np.save("builds/theta_b_freeze_best_wall.npy", w)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "best": best, "wall_at_best": wall_row,
        "chart_theta_n": 30.0, "chart_theta_e": 15.5,
    }, indent=2))
    print(f"wrote {OUT}")
    return 0 if best["max_scaled"] <= 2e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
