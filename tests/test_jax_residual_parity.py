"""
J2 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the JAX residual leaves reproduce the
NumPy residual functions on a fixed CE/wall state.

Covers the pure node-wise leaves the BVP differentiates through:
  - rao_residuals: C+/C- axisym, left-Mach, wall tangency, C+ child position
  - rao_variational CE blocks: axisym C+/C- groups, algebraic stationarity,
    differential (fd) stationarity, left-Mach /ds, smoothness regularization

The full grouped/weighted assembly (mass/length/stationarity integrals) lands in
J3 — it is coupled to the marching construction, not a pure leaf — so it is out of
scope here by design.

Skips cleanly if JAX is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402

from raosim.moc import FlowNode  # noqa: E402
from raosim import rao_residuals as rr  # noqa: E402
from raosim import rao_variational as rv  # noqa: E402
from raosim.jax import residuals as jr  # noqa: E402

GAMMA = 1.4
TOL = 1e-10


def _fixed_ce():
    """A realistic, monotonic geometry-backed CE state (x given)."""
    n = 14
    r = np.linspace(0.30, 1.00, n)
    M = np.linspace(2.20, 3.50, n)
    theta = np.radians(np.linspace(28.0, 6.0, n))   # turning toward axis-aligned exit
    x = np.linspace(0.50, 4.20, n)
    return x, r, M, theta


def _ce_obj(x, r, M, theta, log_C=0.137):
    phi = theta + math.radians(0.5)
    return rv.ControlSurface(r=r.copy(), M=M.copy(), theta=theta.copy(),
                             phi=phi, x=x.copy(), log_C=log_C)


def _nodes(x, r, M, theta):
    return [FlowNode(float(x[i]), float(r[i]), max(float(M[i]), 1.001), float(theta[i]))
            for i in range(len(x))]


# --------------------------------------------------------------------------- #
# node-pair leaves                                                            #
# --------------------------------------------------------------------------- #
def test_cplus_cminus_leaf_parity():
    x, r, M, theta = _fixed_ce()
    nodes = _nodes(x, r, M, theta)
    np_cp = np.array([rr.residual_Cplus_axisym(nodes[i], nodes[i + 1], GAMMA)
                      for i in range(len(nodes) - 1)])
    np_cm = np.array([rr.residual_Cminus_axisym(nodes[i], nodes[i + 1], GAMMA)
                      for i in range(len(nodes) - 1)])
    jx_cp = np.asarray(jr.residual_Cplus_axisym(x, r, M, theta, GAMMA))
    jx_cm = np.asarray(jr.residual_Cminus_axisym(x, r, M, theta, GAMMA))
    assert np.allclose(jx_cp, np_cp, rtol=0, atol=TOL)
    assert np.allclose(jx_cm, np_cm, rtol=0, atol=TOL)


def test_left_mach_and_wall_tangency_leaf_parity():
    x, r, M, theta = _fixed_ce()
    nodes = _nodes(x, r, M, theta)
    np_lm = np.array([rr.residual_left_mach_geometry(nodes[i], nodes[i + 1])
                      for i in range(len(nodes) - 1)])
    np_wt = np.array([rr.residual_wall_tangency(nodes[i], nodes[i + 1])
                      for i in range(len(nodes) - 1)])
    assert np.allclose(np.asarray(jr.residual_left_mach_geometry(x, r, M, theta)),
                       np_lm, rtol=0, atol=TOL)
    assert np.allclose(np.asarray(jr.residual_wall_tangency(x, r, theta)),
                       np_wt, rtol=0, atol=TOL)


def test_cplus_child_position_leaf_parity():
    x, r, M, theta = _fixed_ce()
    nodes = _nodes(x, r, M, theta)
    for i in range(len(nodes) - 1):
        p, c = nodes[i], nodes[i + 1]
        np_val = rr.residual_cplus_child_position(p, c)
        jx_val = float(jr.residual_cplus_child_position(
            (p.x, p.r, p.M, p.theta), (c.x, c.r, c.M, c.theta)))
        assert jx_val == pytest.approx(np_val, rel=0, abs=TOL)


# --------------------------------------------------------------------------- #
# CE-array blocks (exact scaling parity)                                      #
# --------------------------------------------------------------------------- #
def test_ce_axisym_compatibility_groups_parity():
    x, r, M, theta = _fixed_ce()
    ce = _ce_obj(x, r, M, theta)
    np_cp, np_cm = rv._ce_axisymmetric_compatibility_residual_groups(ce, GAMMA)
    jx_cp, jx_cm = jr.ce_axisymmetric_compatibility_groups(x, r, M, theta, GAMMA)
    assert np.allclose(np.asarray(jx_cp), np_cp, rtol=0, atol=TOL)
    assert np.allclose(np.asarray(jx_cm), np_cm, rtol=0, atol=TOL)


def test_ce_algebraic_stationarity_parity():
    x, r, M, theta = _fixed_ce()
    ce = _ce_obj(x, r, M, theta, log_C=0.137)
    np_stat = rv._rao_algebraic_stationarity_residuals(ce, GAMMA)
    jx_stat = np.asarray(jr.ce_algebraic_stationarity(x, r, M, theta, 0.137, GAMMA))
    assert np.allclose(jx_stat, np_stat, rtol=0, atol=TOL)


def test_ce_left_mach_parity():
    x, r, M, theta = _fixed_ce()
    ce = _ce_obj(x, r, M, theta)
    np_lm = rv._rao_left_mach_geometry_residuals(ce)
    jx_lm = np.asarray(jr.ce_left_mach(x, r, M, theta))
    assert np.allclose(jx_lm, np_lm, rtol=0, atol=TOL)


def test_ce_smoothness_parity():
    x, r, M, theta = _fixed_ce()
    ce = _ce_obj(x, r, M, theta)
    np_reg = rv._ce_smoothness_regularization(ce, GAMMA)
    jx_reg = np.asarray(jr.ce_smoothness_regularization(M, theta, GAMMA))
    assert np.allclose(jx_reg, np_reg, rtol=0, atol=TOL)


def test_fd_stationarity_matches_numpy_loop():
    x, r, M, theta = _fixed_ce()
    nodes = _nodes(x, r, M, theta)
    np_fd = np.array([rr_fd(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)])
    jx_fd = np.asarray(jr.rao_stationarity_fd_residual(M, theta, GAMMA))
    assert np.allclose(jx_fd, np_fd, rtol=0, atol=TOL)


def rr_fd(p0, p1):
    return rv.rao_stationarity_fd_residual(p0, p1, GAMMA)


# --------------------------------------------------------------------------- #
# differentiability sanity (the whole point of the port)                      #
# --------------------------------------------------------------------------- #
def test_residual_block_is_differentiable_and_jittable():
    x, r, M, theta = _fixed_ce()

    def scalar(theta_arr):
        cp, cm = jr.ce_axisymmetric_compatibility_groups(x, r, M, theta_arr, GAMMA)
        return float(np_sumsq(cp) + np_sumsq(cm))

    import jax.numpy as jnp
    g = jax.jit(jax.grad(lambda th: jnp.sum(
        jr.ce_axisymmetric_compatibility_groups(x, r, M, th, GAMMA)[0] ** 2
        + jr.ce_axisymmetric_compatibility_groups(x, r, M, th, GAMMA)[1] ** 2
    )))(jnp.asarray(theta))
    assert np.all(np.isfinite(np.asarray(g)))
    assert g.shape == theta.shape


def np_sumsq(a):
    import jax.numpy as jnp
    return jnp.sum(a ** 2)
