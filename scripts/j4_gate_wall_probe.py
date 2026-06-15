"""J4 gate replication + solved-CE BDE wall shape, in a single solve.

Replicates ``test_j4_gate_passes_with_position_only_attachment``'s config
(characteristic formulation, JAX backend, constraint-weight ladder,
position-only D attachment) with ``wall_method="bde"`` so the same solve
also yields the BDE wall.  Prints the gate verdict and the wall-slope
profile, and writes a JSON checkpoint + the wall polyline.

Purpose (Phase 12.4 follow-through): the kernel-march mass-integral fix
changed the seed — ``set_theta_b`` now converges the fixed-(L, ε)
topology exactly (θ_B ≈ 25.5° for the ε=10/L80 reference) instead of
fail-bracketing at the old ~24.2° cap — so both the gate value and the
solved-CE wall shape need re-measuring.  The bell criterion is the TOP
shape: slope peaks near θ_N just after the throat arc and decreases
monotonically (chart θ_N(ε=10, L80) ≈ 21.9°, θ_E ≈ 8.3°); the historic
defect was a 35.6° flare at 60% length driven by the ΔM ≈ 0.66 state
jump that position-only attachment leaves at D.

Run:  PYTHONPATH=. python scripts/j4_gate_wall_probe.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig

OUT_JSON = Path("builds/j4_gate_wall_probe.json")
OUT_WALL = Path("builds/j4_solved_bde_wall.npy")


def main() -> int:
    rv.PHYSICS_WEIGHT = 1.0
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=24, n_kernel=24, n_wall=12,
        max_nfev=4000, residual_tol=2e-3, evaluate_moc=False,
        couple_wall=False, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=21.87, solver_backend="jax",
        formulation="characteristic", pin_d_theta=False,
        jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
        wall_method="bde",
    )
    print("solving (J4 gate config + BDE wall)...", flush=True)
    sol = rv.solve_rao_bvp(cfg)
    r = sol.residuals
    cs = sol.control_surface
    gate = r.max_scaled <= 2e-3
    print(f"GATE: max_scaled={r.max_scaled:.4e} (<=2e-3: {gate}) "
          f"converged={cs.converged}")
    print(f"  mass={r.mass_residual_rel:.2e} len={r.length_residual_rel:.2e} "
          f"kdf={cs.kernel_d_fraction:.4f}")
    for g in sorted(r.group_summaries, key=lambda g: -abs(g["max"]))[:4]:
        print(f"  {g['name']:28s} max={g['max']:.3e} n={g['count']}")

    out = {
        "max_scaled": float(r.max_scaled),
        "gate": bool(gate),
        "converged": bool(cs.converged),
        "kdf": float(cs.kernel_d_fraction),
        "mass": float(r.mass_residual_rel),
        "len": float(r.length_residual_rel),
    }

    w = sol.wall_raw
    if w is not None and len(w) > 5:
        w = np.asarray(w)
        ang = np.degrees(np.arctan2(np.diff(w[:, 1]), np.diff(w[:, 0])))
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(w[:, 0]),
                                                      np.diff(w[:, 1])))])
        ipk = int(np.argmax(ang))
        viol = int((np.diff(ang[ipk:]) > 0.25).sum())
        print(f"WALL: n={len(w)} exit=({w[-1, 0]:.6f},{w[-1, 1]:.6f})")
        print(f"  slope peak={ang.max():.2f} deg at s/stot={s[ipk] / s[-1]:.2%} "
              f"exit angle={ang[-1]:.2f} deg")
        print(f"  monotone-after-peak violations: {viol}")
        for i in np.linspace(0, len(ang) - 1, 12).astype(int):
            print(f"    s={s[i] / s[-1]:.3f}  slope={ang[i]:7.3f}")
        out.update({
            "wall_peak_deg": float(ang.max()),
            "wall_peak_pos": float(s[ipk] / s[-1]),
            "wall_exit_deg": float(ang[-1]),
            "wall_monotone_violations": viol,
        })
        OUT_WALL.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUT_WALL, w)
        print(f"wall polyline -> {OUT_WALL}")
    else:
        print("WALL: unavailable (wall_raw missing/short)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"checkpoint -> {OUT_JSON}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
