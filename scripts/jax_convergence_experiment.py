"""Experiment driver for the J4 gate: JAX LM strategies on the Phase-6 case.

Usage:  python scripts/jax_convergence_experiment.py [strategy]
  strategy in {barrier, homotopy, sigmoid}  (default: homotopy)

Prints per-rung max_scaled so stalls are visible immediately.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.jax import assembly
import jax
import jax.numpy as jnp
import optimistix as optx


def build_problem():
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01, length_pct=80.0,
        n_control=12, n_kernel=12, n_wall=12, max_nfev=800, residual_tol=2e-3,
        evaluate_moc=False, couple_wall=True, kernel_d_fraction_max=0.7,
    )
    ce0, kbd, topo, _kern = rv._initial_ce_from_kernel(cfg)
    sc = replace(cfg, kernel_bd=tuple(kbd))
    ce0.log_C = rv._seed_log_C_from_ce(ce0, cfg.gamma)
    wall = rv._initial_wall_guess(sc, ce0, topo)
    ce0.pair_fractions = np.linspace(0.0, 1.0, cfg.n_control)
    u0 = rv._pack_bvp(ce0, -0.5, 0.01, ce0.log_C, wall=wall)

    from raosim.gas_dynamics import mach_from_area_ratio
    n, n_w = cfg.n_control, cfg.n_wall
    Me = mach_from_area_ratio(cfg.epsilon, cfg.gamma, supersonic=True)
    L = rv._target_length(cfg.Rt, cfg.epsilon, cfg.length_pct)
    Re = math.sqrt(cfg.epsilon) * cfg.Rt
    lower = np.concatenate([
        np.full(n, 1.001), np.full(n, math.radians(-10.0)), np.full(n, 0.0),
        np.full(n_w, 1.001), np.full(n_w, 0.0), np.full(n_w, 0.0),
        np.full(n_w, cfg.Rt),
        np.array([-1e3, -1e3, -10.0, 0.0]), np.zeros(n)])
    upper = np.concatenate([
        np.full(n, max(12.0, 1.5 * Me)), np.full(n, math.radians(65.0)),
        np.full(n, 1.05 * Re),
        np.full(n_w, max(12.0, 1.5 * Me)), np.full(n_w, math.radians(45.0)),
        np.full(n_w, 1.2 * L), np.full(n_w, 1.05 * Re),
        np.array([1e3, 1e3, 10.0, 0.7]), np.ones(n)])
    return sc, u0, lower, upper


def report(tag, f_pure, u, lower, upper, t0):
    r = np.asarray(f_pure(jnp.asarray(u)))
    u_np = np.asarray(u)
    oob = float(np.sum(np.maximum(lower - u_np, 0)) +
                np.sum(np.maximum(u_np - upper, 0)))
    print(f"{tag}: max_scaled={np.max(np.abs(r)):.4g} rms={np.sqrt(np.mean(r**2)):.4g} "
          f"oob={oob:.2g} t={time.time()-t0:.0f}s", flush=True)
    return float(np.max(np.abs(r)))


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "homotopy"
    rungs = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                                else ["0.05", "0.25", "1.0"])]
    sc, u0, lower, upper = build_problem()
    lo, hi = jnp.asarray(lower), jnp.asarray(upper)
    span = hi - lo
    t0 = time.time()
    solver = optx.LevenbergMarquardt(rtol=1e-9, atol=1e-12)

    def make(w):
        sp = assembly.params_from_config(sc, physics_weight=w)
        f = assembly.make_residual(sp)

        def fn_barrier(u, args):
            barrier = jnp.concatenate([
                10.0 * jnp.maximum(lo - u, 0.0) / span,
                10.0 * jnp.maximum(u - hi, 0.0) / span])
            return jnp.concatenate([f(u), barrier])

        def fn_sigmoid(z, args):
            return f(lo + span * jax.nn.sigmoid(z))

        return f, fn_barrier, fn_sigmoid

    if strategy in ("homotopy", "barrier"):
        weights = rungs if strategy == "homotopy" else [1.0]
        u = jnp.asarray(u0)
        for w in weights:
            f_pure, fn, _ = make(w)
            for it in range(2):
                sol = optx.least_squares(fn, solver, u, args=None,
                                         max_steps=1500, throw=False)
                u = sol.value
                m = report(f"w={w:g} it={it}", f_pure, u, lower, upper, t0)
                if m < 2e-3:
                    break
        u_np = np.clip(np.asarray(u), lower, upper)
        f_pure, _, _ = make(1.0)
        report("FINAL(clipped, w=1)", f_pure, jnp.asarray(u_np), lower, upper, t0)
    elif strategy == "sigmoid":
        f_pure, _, fn_s = make(1.0)
        frac0 = np.clip((u0 - lower) / (upper - lower), 1e-2, 1 - 1e-2)
        z = jnp.asarray(np.log(frac0) - np.log1p(-frac0))
        for it in range(4):
            sol = optx.least_squares(fn_s, solver, z, args=None,
                                     max_steps=1500, throw=False)
            z = sol.value
            u = lo + span * jax.nn.sigmoid(z)
            report(f"sig it={it}", f_pure, u, lower, upper, t0)
    else:
        raise SystemExit(f"unknown strategy {strategy}")


if __name__ == "__main__":
    main()
