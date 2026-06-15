"""Phase 12.4 step 2: full-continuity solve with a theta_B Picard refresh.

Full D-state continuity (``pin_d_theta=True`` + ``pin_d_mach=True``) with
the kernel march now clearing the historic ~24.2 deg cap.  Outer loop:

    solve BVP (characteristic, full D pins, ladder, jax backend)
      -> refresh theta_B: secant on the kernel-BD Rao stationarity at the
         solved (kdf, log_C)   [cheap kernel rebuilds only]
      -> reseed at the refreshed theta_B and re-solve
    until |d theta_B| < 0.05 deg.

Measured in the 2026-06-11 session (sandbox, n_control=24, ladder
(1, 10, 30, 100)):

    iter0  theta_B=21.87  ->  max_scaled 5.67e-2, kdf 0.0882, refresh -> 28.10
    iter1  theta_B=28.10  ->  max_scaled 5.70e-2, kdf 0.0865, refresh -> 28.17

i.e. the Picard *converges* in theta_B (the march cap no longer binds) but
the inner floor is theta_B-insensitive at ~5.7e-2 with an all-node
stationarity ramp and kdf collapsing toward B.  Interpretation (see the
STATUS block): with the kernel/theta_B frozen per solve, fixed-(L, eps) +
full D continuity stays overdetermined — DE is fully determined by D and
the stationarity+C+ ODE pair, so hitting (r_E, L) needs kdf *and* theta_B
live inside the iteration (J3b), or fixed-length transversality blocks,
or a Guderley-style discontinuous optimum at D.

Note: ``solve_rao_bvp``'s seed path runs ``set_theta_b`` (its own inner
secant on *length*), so ``thetaN_guess_deg`` only initialises that secant;
the kernel each solve actually freezes sits near the fixed-end topology's
theta_B (~25.5 deg for the reference).  The outer refresh here therefore
probes the seed basin, not a hard theta_B override.

Run:  PYTHONPATH=. python scripts/theta_b_picard_probe.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.gas_dynamics import mstar_from_M
from raosim.nasa_moc import build_kernel, calc_mdot_bd_grid

rv.PHYSICS_WEIGHT = 1.0

RT = 0.020
GAMMA = 1.4
RD = 0.382 * RT
RU = 1.5 * RT
OUT_JSON = Path("builds/theta_b_picard_probe.json")


def reference_config(theta_b_deg: float) -> RaoSolverConfig:
    return RaoSolverConfig(
        Rt=RT, epsilon=10.0, gamma=GAMMA, pa_over_p0=0.01,
        length_pct=80.0, n_control=24, n_kernel=24, n_wall=12,
        max_nfev=4000, residual_tol=2e-3, evaluate_moc=False,
        couple_wall=False, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=theta_b_deg, solver_backend="jax",
        formulation="characteristic",
        pin_d_theta=True, pin_d_mach=True,
        jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
    )


def kernel_stationarity_residual(theta_b_rad: float, kdf: float,
                                 log_C: float) -> float | None:
    """Rao stationarity at the kernel-BD point D(kdf), minus log_C."""
    k = build_kernel(RT, RD, theta_b_rad, GAMMA, 24,
                     starting_line_method="kliegel_levine", Ru=RU)
    if not k.reached_wall:
        return None
    _, D, _, _ = calc_mdot_bd_grid(k.bd, k.massflow[-1], kdf)
    M = max(float(D.M), 1.000001)
    mu = math.asin(1.0 / M)
    return (math.log(mstar_from_M(M, GAMMA))
            + math.log(abs(math.cos(float(D.theta) - mu)))
            - math.log(math.cos(mu)) - log_C)


def refresh_theta_b(theta_b_deg: float, kdf: float, log_C: float) -> float:
    t0 = math.radians(theta_b_deg)
    t1 = t0 + math.radians(0.5)
    f0 = kernel_stationarity_residual(t0, kdf, log_C)
    f1 = kernel_stationarity_residual(t1, kdf, log_C)
    if f0 is None or f1 is None:
        return theta_b_deg
    for _ in range(20):
        if abs(f1 - f0) < 1e-14:
            break
        t2 = t1 - f1 * (t1 - t0) / (f1 - f0)
        t2 = min(max(t2, math.radians(10.0)), math.radians(35.0))
        f2 = kernel_stationarity_residual(t2, kdf, log_C)
        if f2 is None:
            t2 = 0.5 * (t1 + t2)
            f2 = kernel_stationarity_residual(t2, kdf, log_C)
            if f2 is None:
                break
        t0, f0, t1, f1 = t1, f1, t2, f2
        if abs(f1) < 1e-10:
            break
    return math.degrees(t1)


def summarize(sol, tag: str) -> None:
    r = sol.residuals
    cs = sol.control_surface
    print(f"[{tag}] max_scaled={r.max_scaled:.4e} converged={cs.converged} "
          f"kdf={cs.kernel_d_fraction:.4f} mass={r.mass_residual_rel:.2e} "
          f"len={r.length_residual_rel:.2e}")
    for g in sorted(r.group_summaries, key=lambda g: -abs(g["max"]))[:4]:
        print(f"    {g['name']:28s} max={g['max']:.3e} n={g['count']}")
    sys.stdout.flush()


def main() -> int:
    theta_b = 21.87
    history = []
    final = None
    for it in range(4):
        print(f"\n=== Picard iter {it}: theta_B = {theta_b:.4f} deg ===",
              flush=True)
        final = rv.solve_rao_bvp(reference_config(theta_b))
        summarize(final, f"iter{it}")
        kdf = float(final.control_surface.kernel_d_fraction)
        log_C = float(final.control_surface.log_C)
        new_theta = refresh_theta_b(theta_b, kdf, log_C)
        d = new_theta - theta_b
        history.append({
            "theta_b": theta_b,
            "max_scaled": float(final.residuals.max_scaled),
            "kdf": kdf, "log_C": log_C, "refreshed": new_theta,
        })
        print(f"    refresh: theta_B {theta_b:.4f} -> {new_theta:.4f} "
              f"(d={d:+.4f} deg)", flush=True)
        theta_b = new_theta
        if abs(d) < 0.05:
            break

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "history": history,
        "final_theta_b": theta_b,
        "final_max_scaled": float(final.residuals.max_scaled),
        "gate": bool(final.residuals.max_scaled <= 2e-3),
    }, indent=2))
    print(f"\ncheckpoint -> {OUT_JSON}")
    return 0 if final.residuals.max_scaled <= 2e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
