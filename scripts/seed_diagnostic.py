#!/usr/bin/env python3
"""
seed_diagnostic.py  —  Seed-vs-optimizer falsification test for the Rao
coupled-wall convergence stall (max_scaled ~ 8, gate 2e-3).

WHY THIS EXISTS
---------------
The JAX plan's priority-1 thesis is that the stall is caused by scipy's
*finite-difference Jacobian noise* near the sonic / Mach-line singularities, so
exact autodiff (jacrev) Jacobians will fix it.  The competing hypothesis (the
repo's own Phase-6 xfail text, docs/legacy_code_audit.md, and the JHU/APL
report's statement that the initial-line shape drives convergence) is that the
stall is the SEED/KERNEL: the kernel BD cannot carry the throat target mass, so
the residual has NO root reachable from this seed and no optimizer can descend.

These make OPPOSITE, MEASURABLE predictions at the stalled point x*:

  SEED hypothesis      : x* is a (near-)stationary point of 0.5||r||^2 with
                         ||r|| ~ 8.  ||J^T r|| ~ 0 even with an ACCURATE
                         Jacobian; scipy stops on gtol; one block (mass /
                         moc_cminus) holds the floor; a more accurate (3-point)
                         Jacobian does NOT move the floor.  => exact JAX
                         Jacobians cannot fix convergence.

  OPTIMIZER hypothesis : x* is NOT stationary.  ||J^T r|| is large, a
                         Gauss-Newton step predicts a big cost drop, scipy
                         stopped on ftol/xtol, and a 3-point Jacobian escapes
                         the floor.  => exact Jacobians may help.

We test the REAL NumPy residual (no JAX re-port), so there is no port-mismatch
confound.  No JAX required — runs in the existing numpy/scipy env.

RUN
---
    python scripts/seed_diagnostic.py
    # ~1-3 min: it runs the real ref-case solve, then probes x*.

Paste the final "SUMMARY (paste this back)" block to me and I'll fold it in.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import raosim.rao_variational as rv
from raosim.rao_variational import (
    RaoSolverConfig, solve_rao_bvp, _scaled_rao_bvp_residual,
)
from scipy.optimize import least_squares as _real_ls
from scipy.optimize._numdiff import approx_derivative


def banner(s): print("\n" + "=" * 74 + "\n" + s + "\n" + "=" * 74, flush=True)


# ---- capture the solver's exact (fun, x0, args, bounds, result) -----------
CAP = {}
def _capturing_ls(fun, x0, **kw):
    res = _real_ls(fun, x0, **kw)
    CAP.update(fun=fun, x0=np.asarray(x0, float).copy(),
               args=kw.get("args", ()), bounds=kw.get("bounds", (-np.inf, np.inf)),
               result=res)
    return res


def main():
    rv.least_squares = _capturing_ls
    rv.PHYSICS_WEIGHT = 1.0
    # The exact Phase-6 xfail reference case (tests/test_phase6_coupled_wall.py:394).
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01, length_pct=80.0,
        n_control=12, n_kernel=12, n_wall=12, max_nfev=800, residual_tol=2e-3,
        evaluate_moc=False, couple_wall=True, kernel_d_fraction_max=0.7,
    )

    banner("STEP 1 — reproduce the reference-case coupled-wall stall")
    t0 = time.time()
    sol = solve_rao_bvp(cfg)
    res = CAP["result"]; fun, args, bounds = CAP["fun"], CAP["args"], CAP["bounds"]
    x = np.asarray(res.x, float)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    r = np.asarray(fun(x, *args), float)
    cost = 0.5 * float(r @ r); maxabs = float(np.max(np.abs(r)))
    print(f"solve wall-time     : {time.time()-t0:.1f} s")
    print(f"reliability         : {sol.reliability}")
    print(f"unknowns / residuals: {x.size} / {r.size}")
    print(f"max_scaled |r|_inf  : {maxabs:.4f}      (gate <= 2e-3)")
    print(f"cost 0.5||r||^2     : {cost:.4f}")
    print(f"scipy success       : {res.success}")
    print(f"scipy status        : {res.status}   (0=maxfev 1=gtol 2=ftol 3=xtol 4=ftol&xtol)")
    print(f"scipy .optimality   : {res.optimality:.3e}   <== ||proj grad||_inf the SOLVER saw")
    print(f"nfev / njev         : {res.nfev} / {res.njev}")
    print(f"message             : {res.message}")

    banner("STEP 2 — per-block residual decomposition at x*  (where is the floor?)")
    g = _scaled_rao_bvp_residual(x, args[0], args[1], return_groups=True)
    rows = [(s["name"], s["count"], s.get("max", 0.0), s.get("rms", 0.0))
            for s in g.summaries() if s.get("count", 0)]
    rows.sort(key=lambda t: abs(t[2]), reverse=True)
    print(f"{'block':<22}{'n':>4}{'max|r|':>14}{'rms':>14}")
    dominant = rows[0][0] if rows else "?"
    for nm, c, mx, rms in rows:
        print(f"{nm:<22}{c:>4}{mx:>14.4f}{rms:>14.4f}{'   <== FLOOR' if abs(mx) > 1 else ''}")

    banner("STEP 3 — accurate 3-point Jacobian at x*: is it a stationary point?")
    J = approx_derivative(lambda u: fun(u, *args), x, method="3-point", bounds=bounds)
    grad = J.T @ r
    at_lo, at_hi = x <= lo + 1e-9, x >= hi - 1e-9
    pg = grad.copy(); pg[at_lo & (grad > 0)] = 0.0; pg[at_hi & (grad < 0)] = 0.0
    dx, *_ = np.linalg.lstsq(J, -r, rcond=None)
    pred_cost = 0.5 * float((r + J @ dx) @ (r + J @ dx))
    red = 100 * (cost - pred_cost) / max(cost, 1e-30)
    sv = np.linalg.svd(J, compute_uv=False)
    rank_def = int(np.sum(sv < 1e-8 * sv[0]))
    g_inf = float(np.max(np.abs(grad))); pg_inf = float(np.max(np.abs(pg)))
    print(f"||r||_2                          : {np.linalg.norm(r):.4f}")
    print(f"||J^T r||_inf (objective grad)   : {g_inf:.3e}")
    print(f"||proj grad||_inf (with bounds)  : {pg_inf:.3e}")
    print(f"relative ||J^T r|| / ||r||       : {g_inf/max(np.linalg.norm(r),1e-30):.3e}")
    print(f"GN 1-step predicted cost         : {pred_cost:.4f}  (now {cost:.4f}, reduction {red:.1f}%)")
    print(f"Jacobian sing.values max / min   : {sv[0]:.2e} / {sv[-1]:.2e}   cond={sv[0]/max(sv[-1],1e-30):.2e}")
    print(f"rank-deficient directions (<1e-8): {rank_def} / {sv.size}")
    print(f"unknowns pinned at a bound       : {int(np.sum(at_lo|at_hi))} / {x.size}")

    banner("STEP 4 — re-solve from SAME seed with accurate 3-point Jacobian")
    t1 = time.time()
    res3 = _real_ls(fun, CAP["x0"], bounds=bounds, args=args, x_scale="jac",
                    ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000, jac="3-point")
    r3 = np.asarray(fun(res3.x, *args), float); m3 = float(np.max(np.abs(r3)))
    moved = abs(m3 - maxabs)
    print(f"re-solve wall-time  : {time.time()-t1:.1f} s")
    print(f"2-point (default)   : max|r|={maxabs:.4f}  cost={cost:.4f}  status={res.status}  nfev={res.nfev}")
    print(f"3-point (accurate)  : max|r|={m3:.4f}  cost={res3.cost:.4f}  status={res3.status}  nfev={res3.nfev}  optimality={res3.optimality:.2e}")
    print(f"floor moved by Jac  : {moved:.4f}  -> "
          f"{'NEGLIGIBLE (not Jacobian-noise)' if moved < 0.5 else 'SIGNIFICANT (Jacobian quality matters)'}")

    # ---- auto-verdict -----------------------------------------------------
    banner("VERDICT")
    stationary = (res.status == 1) or (res.optimality < 1e-4) or (pg_inf < 1e-4)
    floor_block = dominant in ("mass", "moc_cminus", "ce_geometry", "length")
    jac_doesnt_help = moved < 0.5
    small_gn = red < 25.0
    seed_score = sum([stationary, floor_block and maxabs > 1, jac_doesnt_help, small_gn, rank_def > 0])
    print(f"stationary point (gtol / tiny proj-grad)   : {stationary}")
    print(f"floor held by a structural block ({dominant:<14}): {floor_block and maxabs>1}")
    print(f"accurate Jacobian fails to move the floor  : {jac_doesnt_help}")
    print(f"Gauss-Newton predicts little reduction     : {small_gn}")
    print(f"Jacobian rank-deficient (degenerate geom)  : {rank_def > 0}")
    print(f"\nSEED-evidence score: {seed_score}/5")
    if seed_score >= 3:
        verdict = ("SEED / KERNEL.  Exact (JAX) Jacobians will NOT fix this stall — "
                   "the residual has no nearby root from this seed. Fix the kernel "
                   "seed (sauer_modified start line + CalcRRCsAlongArc BD mass).")
    elif seed_score <= 1:
        verdict = ("OPTIMIZER.  There is a real descent direction the FD solver "
                   "missed — exact Jacobians (the JAX port) could help. Proceed J4.")
    else:
        verdict = ("MIXED — inspect the numbers above; partly structural, partly "
                   "optimizer. Report the block floor + ||proj grad|| + floor-moved.")
    print("=> " + verdict)

    banner("SUMMARY (paste this back)")
    print(f"max_scaled={maxabs:.4f} cost={cost:.4f} status={res.status} "
          f"scipy_optimality={res.optimality:.2e} nfev={res.nfev}")
    print(f"dominant_block={dominant} block_max={rows[0][2]:.4f} "
          f"second_block={rows[1][0] if len(rows)>1 else '-'}:{rows[1][2] if len(rows)>1 else 0:.4f}")
    print(f"accurate_grad_inf={g_inf:.2e} proj_grad_inf={pg_inf:.2e} "
          f"GN_reduction_pct={red:.1f} cond={sv[0]/max(sv[-1],1e-30):.2e} "
          f"rank_def={rank_def}/{sv.size} bounds_pinned={int(np.sum(at_lo|at_hi))}/{x.size}")
    print(f"floor_3pt={m3:.4f} floor_moved={moved:.4f} seed_score={seed_score}/5")
    print(f"VERDICT={verdict.split('.')[0]}")


if __name__ == "__main__":
    main()
