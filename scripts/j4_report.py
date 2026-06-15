"""One-shot J0-J4 status report on the reference case (eps=10, L80, gamma=1.4).

Usage:
    python scripts/j4_report.py            # kernel + topology + JAX solve (~20-30 s)
    python scripts/j4_report.py --scipy    # also run the scipy backend (slow, minutes)

What to look for:
  [1] kernel march: rrcs == 58 on the NASA M3.5Perf geometry (oracle match)
  [2] Rao-case kernel: rrcs >> 1 (pre-fix failure mode was exactly 1)
  [3] topology: mass_BD == mass_DE (nonzero), E.r pinned to Re, DE nodes >= 12
  [4] JAX solve: max_scaled < 0.8, physics blocks (stationarity/moc_c+/-) < 0.2
      -- residual should concentrate in length / wall_endpoint (the open
      variational tension documented in JAX_DIFFERENTIABLE_PLAN.md STATUS).
"""
from __future__ import annotations

import math
import sys
import time

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main():
    run_scipy = "--scipy" in sys.argv

    # [1] NASA oracle: M3.5Perf kernel march
    section("[1] kernel march vs NASA M3.5Perf oracle")
    from raosim.nasa_moc import build_kernel
    t0 = time.time()
    k = build_kernel(1.0, 1.0, math.radians(15.2196), 1.4, 101,
                     starting_line_method="nasa_visible_kliegel_levine")
    bd = k.bd
    print(f"rrcs={len(k.rrcs)} (NASA: 58)   "
          f"B=(x{bd[0].x:.5f}, r{bd[0].r:.5f}, M{bd[0].M:.4f})  "
          f"(NASA: 0.26252, 1.03507, 1.6392)   [{time.time()-t0:.1f}s]")

    # [2] Rao-case kernel (the geometry that used to silently fall back)
    section("[2] Rao-geometry kernel (Rt=0.02, Rd=0.382Rt, Ru=1.5Rt)")
    t0 = time.time()
    k2 = build_kernel(0.020, 0.382 * 0.020, math.radians(30.0), 1.4, 24,
                      starting_line_method="kliegel_levine", Ru=1.5 * 0.020)
    print(f"rrcs={len(k2.rrcs)} (pre-fix: 1)   "
          f"BD wall theta={math.degrees(k2.bd[0].theta):.2f} deg   "
          f"axis M={k2.bd[-1].M:.3f}   [{time.time()-t0:.1f}s]")

    # [3] fixed-end topology seed
    section("[3] NASA fixed-end topology (set_theta_b + calc_lrc_de)")
    from raosim.nasa_moc import set_theta_b
    Rt, eps, lpct = 0.020, 10.0, 80.0
    Re = math.sqrt(eps) * Rt
    L = rv._target_length(Rt, eps, lpct)
    t0 = time.time()
    topo, kern = set_theta_b(Rt, eps, lpct, 1.4, 0.01,
                             theta_b_init_deg=24.0, n_kernel=24,
                             n_de_points=16,
                             starting_line_method="kliegel_levine",
                             Ru=1.5 * Rt)
    print(f"theta_B={math.degrees(kern.theta_B):.3f} deg   "
          f"D=(r{topo.D.r:.5f}, M{topo.D.M:.3f})   DE nodes={len(topo.DE)}")
    print(f"mass_BD={topo.mass_BD:.6g}  mass_DE={topo.mass_DE:.6g}   "
          f"E=(x{topo.E.x:.5f}, r{topo.E.r:.5f})  target=({L:.5f}, {Re:.5f})")
    print(f"rel length err={(topo.E.x-L)/L:+.3f} (expect ~+0.09 plateau)   "
          f"rel r_E err={(topo.E.r-Re)/Re:+.2e} (expect ~1e-9)   "
          f"[{time.time()-t0:.1f}s]")

    # [4] the J4 reference solve
    def solve(backend):
        original = rv.PHYSICS_WEIGHT
        try:
            rv.PHYSICS_WEIGHT = 1.0
            cfg = RaoSolverConfig(
                Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
                length_pct=80.0, n_control=12, n_kernel=24, n_wall=12,
                max_nfev=800, residual_tol=2e-3, evaluate_moc=False,
                couple_wall=True, kernel_d_fraction_max=0.7,
                thetaN_guess_deg=24.0, solver_backend=backend,
            )
            t0 = time.time()
            sol = rv.solve_rao_bvp(cfg)
            dt = time.time() - t0
        finally:
            rv.PHYSICS_WEIGHT = original
        section(f"[4] J4 reference solve, backend={backend!r}  ({dt:.0f}s)")
        r = sol.residuals
        print(f"max_scaled={r.max_scaled:.4g}   (history: scipy+degenerate "
              "kernel ~8; gate 2e-3; regression floor 0.8)")
        print(f"mass={r.mass_residual_rel:+.3g}  "
              f"length={r.length_residual_rel:+.3g}  "
              f"kdf={sol.control_surface.kernel_d_fraction:.3f}  "
              f"reliability={sol.reliability.value}")
        for s in r.group_summaries:
            if s["count"] and s["max"] > 1e-3:
                print(f"   {s['name']:24s} max={s['max']:.3e}")

    solve("jax")
    if run_scipy:
        solve("numpy")
    else:
        print("\n(re-run with --scipy to add the scipy-backend comparison; slow)")


if __name__ == "__main__":
    main()
