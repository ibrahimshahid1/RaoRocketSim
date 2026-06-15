"""
J2 gate (JAX_DIFFERENTIABLE_PLAN.md §7): assembled-residual parity.

``raosim.jax.assembly.make_residual`` must reproduce the NumPy
``_scaled_rao_bvp_residual`` (which folds in ``_unpack_bvp``,
``_rao_bvp_residual_groups``, ``_coupled_wall_residuals``, and
``RaoResidualGroups.flat()``) to 1e-8 on identical packed unknown
vectors.  States are the *real* Phase-6 seed (kernel-seeded CE + Bezier
wall) for the reference case, plus random in-bounds perturbations.

Passing this proves the JAX port changed numerics, not physics — every
downstream convergence result is then attributable to the optimizer
(exact Jacobian LM), not to a different residual.

Also covers J3 wiring on the real system: jacfwd/jacrev of the assembled
residual are finite and match finite differences.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import raosim.rao_variational as rv  # noqa: E402
from raosim.rao_variational import RaoSolverConfig  # noqa: E402
from raosim.jax import assembly  # noqa: E402

PARITY_ATOL = 1e-8
PARITY_RTOL = 1e-8


# --------------------------------------------------------------------------- #
# seed construction — mirrors solve_rao_bvp's setup byte for byte              #
# --------------------------------------------------------------------------- #
def _phase6_seed(couple_wall: bool = True):
    config = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=12, n_kernel=12, n_wall=12,
        max_nfev=800, residual_tol=2e-3,
        evaluate_moc=False, couple_wall=couple_wall,
        kernel_d_fraction_max=0.7,
    )
    ce0, kernel_bd_seed, topology_seed = rv._initial_ce_from_kernel(config)
    solve_config = replace(
        config, kernel_bd=tuple(kernel_bd_seed) if kernel_bd_seed else None,
    )
    log_C0 = rv._seed_log_C_from_ce(ce0, config.gamma)
    ce0.log_C = log_C0
    n = len(ce0.r)
    if couple_wall:
        wall_seed = rv._initial_wall_guess(solve_config, ce0, topology_seed)
        ce0.pair_fractions = np.linspace(0.0, 1.0, n)
        u0 = rv._pack_bvp(ce0, -0.5, 0.01, log_C0, wall=wall_seed)
    else:
        u0 = rv._pack_bvp(ce0, -0.5, 0.01, log_C0)
    return u0, ce0, solve_config


def _bounds_like_solver(solve_config, n, n_w):
    """The solve_rao_bvp bounds box (used to keep perturbations in-bounds)."""
    from raosim.gas_dynamics import mach_from_area_ratio

    Me = mach_from_area_ratio(solve_config.epsilon, solve_config.gamma,
                              supersonic=True)
    L = rv._target_length(solve_config.Rt, solve_config.epsilon,
                          solve_config.length_pct)
    Re = math.sqrt(solve_config.epsilon) * solve_config.Rt
    lower = [np.full(n, 1.001), np.full(n, math.radians(-10.0)), np.full(n, 0.0)]
    upper = [np.full(n, max(12.0, 1.5 * Me)), np.full(n, math.radians(65.0)),
             np.full(n, 1.05 * Re)]
    if n_w:
        lower += [np.full(n_w, 1.001), np.full(n_w, 0.0), np.full(n_w, 0.0),
                  np.full(n_w, solve_config.Rt)]
        upper += [np.full(n_w, max(12.0, 1.5 * Me)),
                  np.full(n_w, math.radians(45.0)),
                  np.full(n_w, max(1.2 * L, 1e-9)), np.full(n_w, 1.05 * Re)]
    kdf_cap = solve_config.kernel_d_fraction_max or 1.0
    lower.append(np.array([-1e3, -1e3, -10.0, 0.0]))
    upper.append(np.array([1e3, 1e3, 10.0, kdf_cap]))
    if n_w:
        lower.append(np.zeros(n))
        upper.append(np.ones(n))
    return np.concatenate(lower), np.concatenate(upper)


def _numpy_residual(u, ce0, solve_config):
    return np.asarray(
        rv._scaled_rao_bvp_residual(np.asarray(u, dtype=float), ce0.r,
                                    solve_config),
        dtype=float,
    )


# --------------------------------------------------------------------------- #
# parity on the coupled-wall Phase 6 reference seed                            #
# --------------------------------------------------------------------------- #
class TestCoupledParity:
    @pytest.fixture(scope="class")
    def coupled(self):
        u0, ce0, solve_config = _phase6_seed(couple_wall=True)
        sp = assembly.params_from_config(solve_config)
        fn = assembly.make_residual(sp)
        return u0, ce0, solve_config, sp, fn

    def test_residual_lengths_match(self, coupled):
        u0, ce0, solve_config, _, fn = coupled
        r_np = _numpy_residual(u0, ce0, solve_config)
        r_jx = np.asarray(fn(jnp.asarray(u0)))
        assert r_jx.shape == r_np.shape

    def test_seed_state_parity(self, coupled):
        u0, ce0, solve_config, _, fn = coupled
        r_np = _numpy_residual(u0, ce0, solve_config)
        r_jx = np.asarray(fn(jnp.asarray(u0)))
        np.testing.assert_allclose(r_jx, r_np, rtol=PARITY_RTOL,
                                   atol=PARITY_ATOL)

    def test_perturbed_states_parity(self, coupled):
        """20 random in-bounds states around the seed."""
        u0, ce0, solve_config, sp, fn = coupled
        lower, upper = _bounds_like_solver(solve_config, sp.n_ce, sp.n_wall)
        rng = np.random.default_rng(7)
        for _ in range(20):
            scale = 0.05 * (upper - lower)
            scale[~np.isfinite(scale)] = 0.1
            u = np.clip(u0 + rng.normal(0.0, 1.0, u0.shape) * scale,
                        lower, upper)
            r_np = _numpy_residual(u, ce0, solve_config)
            r_jx = np.asarray(fn(jnp.asarray(u)))
            np.testing.assert_allclose(r_jx, r_np, rtol=PARITY_RTOL,
                                       atol=PARITY_ATOL)

    def test_physics_weight_is_respected(self, coupled):
        """Monkeypatched PHYSICS_WEIGHT (the weight-ramp studies) flows in."""
        u0, ce0, solve_config, _, _ = coupled
        original = rv.PHYSICS_WEIGHT
        try:
            rv.PHYSICS_WEIGHT = 1.0
            sp1 = assembly.params_from_config(solve_config)
            fn1 = assembly.make_residual(sp1)
            r_np = _numpy_residual(u0, ce0, solve_config)
            r_jx = np.asarray(fn1(jnp.asarray(u0)))
            np.testing.assert_allclose(r_jx, r_np, rtol=PARITY_RTOL,
                                       atol=PARITY_ATOL)
        finally:
            rv.PHYSICS_WEIGHT = original

    def test_jit_matches_eager(self, coupled):
        """XLA fusion may reorder float ops; agreement to 1e-12 required."""
        u0, _, _, _, fn = coupled
        r_eager = np.asarray(fn(jnp.asarray(u0)))
        r_jit = np.asarray(jax.jit(fn)(jnp.asarray(u0)))
        np.testing.assert_allclose(r_jit, r_eager, rtol=1e-12, atol=1e-12)

    # ----------------------------- J3 on the real system ------------------- #
    def test_jacobian_finite_and_matches_fd(self, coupled):
        u0, ce0, solve_config, sp, fn = coupled
        lower, upper = _bounds_like_solver(solve_config, sp.n_ce, sp.n_wall)
        # nudge strictly inside the box so central differences stay in-bounds
        span = np.where(np.isfinite(upper - lower), upper - lower, 1.0)
        u = np.clip(u0, lower + 1e-3 * span, upper - 1e-3 * span)

        J = np.asarray(jax.jacfwd(fn)(jnp.asarray(u)))
        assert np.all(np.isfinite(J)), "autodiff Jacobian has non-finite entries"

        Jr = np.asarray(jax.jacrev(fn)(jnp.asarray(u)))
        np.testing.assert_allclose(Jr, J, rtol=1e-9, atol=1e-12)

        rng = np.random.default_rng(3)
        cols = rng.choice(u.size, size=12, replace=False)
        eps = 1e-7
        for k in cols:
            up = u.copy(); up[k] += eps
            um = u.copy(); um[k] -= eps
            fd = (_numpy_residual(up, ce0, solve_config)
                  - _numpy_residual(um, ce0, solve_config)) / (2 * eps)
            np.testing.assert_allclose(
                J[:, k], fd, rtol=2e-4, atol=5e-6,
                err_msg=f"column {k} (autodiff vs NumPy central difference)",
            )


# --------------------------------------------------------------------------- #
# parity on the uncoupled (legacy) layout                                      #
# --------------------------------------------------------------------------- #
def test_uncoupled_seed_parity():
    u0, ce0, solve_config = _phase6_seed(couple_wall=False)
    sp = assembly.params_from_config(solve_config)
    fn = assembly.make_residual(sp)
    r_np = _numpy_residual(u0, ce0, solve_config)
    r_jx = np.asarray(fn(jnp.asarray(u0)))
    assert r_jx.shape == r_np.shape
    np.testing.assert_allclose(r_jx, r_np, rtol=PARITY_RTOL, atol=PARITY_ATOL)


# --------------------------------------------------------------------------- #
# guard rails                                                                  #
# --------------------------------------------------------------------------- #
def test_rejects_missing_kernel_bd():
    config = RaoSolverConfig(Rt=0.02, epsilon=10.0, couple_wall=True)
    with pytest.raises(ValueError, match="kernel_bd"):
        assembly.params_from_config(config)


def test_rejects_unsupported_blocks():
    _, _, solve_config = _phase6_seed(couple_wall=False)
    bad = replace(solve_config,
                  residual_blocks=("mass", "length", "stationarity"))
    with pytest.raises(NotImplementedError, match="stationarity"):
        assembly.params_from_config(bad)


def test_rejects_chart_anchor_modes():
    _, _, solve_config = _phase6_seed(couple_wall=False)
    bad = replace(solve_config, angle_boundary_mode="chart_soft")
    with pytest.raises(NotImplementedError, match="angle_boundary_mode"):
        assembly.params_from_config(bad)
