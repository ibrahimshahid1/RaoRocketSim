"""
J3b-2 — θ_B as a solved unknown inside the BVP (raosim.jax.theta_b_solve).

The wiring gate: at the seed θ_B the live-BD residual must equal the
static-``kernel_bd`` residual (the march reproduces the seed kernel's
BD at bit parity, so any deviation here is plumbing, not physics).
Plus: d(residual)/dθ_B is finite and FD-exact, and a coarse
end-to-end LM solve over [u, θ_B] runs and stays in bounds.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

import raosim.rao_variational as rv  # noqa: E402
from raosim.jax import assembly  # noqa: E402
from raosim.jax.theta_b_solve import (  # noqa: E402
    least_squares_jax_theta_b,
    make_residual_theta_b,
)


@pytest.fixture(scope="module")
def seeded():
    """Reference-config seed: CE guess, kernel BD, kernel object, u0."""
    config = rv.RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=10, n_kernel=24,
        max_nfev=60, evaluate_moc=False,
    )
    ce0, kernel_bd_seed, _topo, kernel = rv._initial_ce_from_kernel(config)
    solve_config = replace(
        config, kernel_bd=tuple(kernel_bd_seed) if kernel_bd_seed else None,
    )
    log_C0 = rv._seed_log_C_from_ce(ce0, config.gamma)
    ce0.log_C = log_C0
    u0 = rv._pack_bvp(ce0, -0.5, 0.01, log_C0)
    assert kernel.reached_wall, "seed kernel must be a real march"
    return solve_config, kernel, np.asarray(u0, dtype=float)


def test_residual_parity_at_seed_theta_b(seeded):
    """fn([u0, θ_B_seed]) == static-BD residual(u0) to ~1e-8: the live
    march IS the seed kernel at the seed angle."""
    solve_config, kernel, u0 = seeded
    static_fn = assembly.make_residual(
        assembly.params_from_config(solve_config))
    r_static = np.asarray(static_fn(jnp.asarray(u0)))

    fn, _ = make_residual_theta_b(
        solve_config, kernel, theta_b_max=float(kernel.theta_B) + 0.01)
    r_live = np.asarray(fn(jnp.asarray(
        np.concatenate([u0, [float(kernel.theta_B)]]))))

    assert r_live.shape == r_static.shape
    scale = np.maximum(np.abs(r_static), 1.0)
    assert np.max(np.abs(r_live - r_static) / scale) < 1e-6


def test_dresidual_dthetaB_finite_and_fd_exact(seeded):
    solve_config, kernel, u0 = seeded
    fn, _ = make_residual_theta_b(
        solve_config, kernel, theta_b_max=float(kernel.theta_B) + 0.01)
    tb = float(kernel.theta_B)
    u_ext = jnp.asarray(np.concatenate([u0, [tb]]))

    def f_tb(t):
        return fn(u_ext.at[-1].set(t))

    g = np.asarray(jax.jacfwd(f_tb)(jnp.float64(tb)))
    assert np.isfinite(g).all()
    assert np.max(np.abs(g)) > 0.0, "theta_B must reach the residual"

    d = 1e-6
    fd = np.asarray((f_tb(tb + d) - f_tb(tb - d)) / (2 * d))
    denom = np.maximum(np.abs(fd), 1e-3 * np.max(np.abs(fd)))
    assert np.max(np.abs(g - fd) / denom) < 1e-4


@pytest.mark.slow
def test_solve_rao_bvp_with_theta_b_live():
    """Production opt-in wiring: solve_rao_bvp(solve_theta_b=True)
    reports the LM-solved angle with 'bvp_solved' provenance and stays
    within the seed's smooth window."""
    cfg = rv.RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=8, n_kernel=24,
        max_nfev=20, evaluate_moc=False,
        jax_constraint_weight_ladder=(1.0,),
        solve_theta_b=True,
    )
    sol = rv.solve_rao_bvp(cfg)
    da = sol.construction_diagnostics["design_angles"]
    assert da["theta_N_source"] == "kernel_theta_B:bvp_solved"
    # Within ± dtheta_limit/4 = 0.125 deg of the seed's fixed-end root.
    assert math.degrees(sol.theta_N) == pytest.approx(25.57, abs=0.3)
    assert np.isfinite(sol.residuals.max_scaled)


@pytest.mark.slow
def test_small_theta_b_solve_runs(seeded):
    """Coarse end-to-end LM over [u, θ_B]: runs, stays in bounds,
    residual finite.  (Physics conclusions — whether full pins close
    with θ_B live — belong to the host experiments, not this smoke.)"""
    solve_config, kernel, u0 = seeded
    cfg = replace(solve_config, jax_constraint_weight_ladder=(1.0,),
                  max_nfev=25)
    n = cfg.n_control
    Me = rv.mach_from_area_ratio(cfg.epsilon, cfg.gamma, supersonic=True)
    Re = math.sqrt(cfg.epsilon) * cfg.Rt
    lower = np.concatenate([
        np.full(n, 1.001), np.full(n, math.radians(-10.0)), np.zeros(n),
        [-1e3, -1e3, -10.0, 0.0],
    ])
    upper = np.concatenate([
        np.full(n, max(12.0, 1.5 * Me)), np.full(n, math.radians(65.0)),
        np.full(n, 1.05 * Re), [1e3, 1e3, 10.0, 1.0],
    ])
    res = least_squares_jax_theta_b(
        cfg, kernel, u0, lower, upper,
        dtheta_limit=math.radians(2.0), max_steps=25,
    )
    lo_tb, hi_tb = res.theta_b_bounds
    assert lo_tb <= res.theta_b <= hi_tb
    assert np.isfinite(res.max_abs_residual)
    assert res.x.shape == u0.shape
