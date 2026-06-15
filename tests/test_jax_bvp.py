"""
J3 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the differentiable solve infrastructure.

Demonstrates, on a *real* Rao residual built from the J2-validated leaves:
  1. Optimistix Levenberg-Marquardt + exact autodiff Jacobian converges the
     algebraic Rao stationarity system (REWRITE_PLAN.md §2.B closed-form optimum
     condition) to machine precision.
  2. The exact jacrev Jacobian matches a finite-difference Jacobian.
  3. Gradients through the converged solution (implicit function theorem) are
     finite and FD-consistent — the J6 sensitivity mechanism.
  4. The §2.D mass-closure integral matches an independent NumPy implementation
     and behaves sensibly.
  5. pack/unpack round-trips.

This is the solve *infrastructure*.  The full coupled-wall grouped residual and
the Phase-6 ``max_scaled < 2e-3`` gate (J4) require the MOC marching grid
(J3b) and are intentionally out of scope here.

Skips cleanly if JAX/Optimistix are absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optimistix")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from raosim.jax import pack as jpack  # noqa: E402
from raosim.jax import residuals as jr  # noqa: E402
from raosim.jax.bvp import least_squares_solve, make_differentiable_solution  # noqa: E402

GAMMA = 1.4


# --------------------------------------------------------------------------- #
# pack/unpack                                                                  #
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip():
    n = 9
    st = jpack.CEState(
        x=jnp.linspace(0.5, 4.0, n),
        r=jnp.linspace(0.3, 1.0, n),
        M=jnp.linspace(2.0, 3.5, n),
        theta=jnp.radians(jnp.linspace(25.0, 5.0, n)),
        log_C=jnp.asarray(0.137),
    )
    u = jpack.pack(st)
    assert u.shape[0] == jpack.n_unknowns(n)
    st2 = jpack.unpack(u, n)
    for a, b in zip(st, st2):
        assert np.allclose(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------- #
# Rao stationarity solve (the real residual)                                  #
# --------------------------------------------------------------------------- #
def _stationarity_residual(u, args):
    """Unknowns u = [theta(n), log_C]. Residual: per-node Rao stationarity + θ0 anchor."""
    M, theta0, gamma, n = args
    theta = u[:n]
    log_C = u[n]
    stat = jr.rao_stationarity_residual(M, theta, log_C, gamma)   # (n,)
    anchor = jnp.array([theta[0] - theta0])
    return jnp.concatenate([stat, anchor])


def test_lm_converges_rao_stationarity_system():
    n = 12
    M = jnp.linspace(2.0, 3.6, n)
    theta0 = math.radians(26.0)
    args = (M, theta0, GAMMA, n)

    u0 = jnp.concatenate([jnp.radians(jnp.linspace(26.0, 6.0, n)), jnp.array([0.0])])
    res = least_squares_solve(_stationarity_residual, u0, args=args)

    assert res.success
    assert float(res.max_abs) < 1e-9      # machine-precision closure

    # Recovered control surface really is Rao-stationary: M*·cos(θ−α)/cos(α)
    # is constant across all nodes.
    theta = np.asarray(res.u[:n])
    Mn = np.asarray(M)
    alpha = np.arcsin(1.0 / Mn)
    Mstar = np.sqrt((GAMMA + 1) * Mn**2 / (2 + (GAMMA - 1) * Mn**2))
    C = Mstar * np.cos(theta - alpha) / np.cos(alpha)
    assert np.std(C) / np.mean(C) < 1e-8   # constant to 1e-8 relative


def test_exact_jacobian_matches_finite_difference():
    n = 8
    M = jnp.linspace(2.0, 3.4, n)
    args = (M, math.radians(24.0), GAMMA, n)
    u = jnp.concatenate([jnp.radians(jnp.linspace(24.0, 6.0, n)), jnp.array([0.05])])

    J_exact = np.asarray(jax.jacfwd(lambda uu: _stationarity_residual(uu, args))(u))

    # central differences
    u_np = np.asarray(u)
    eps = 1e-6
    J_fd = np.zeros_like(J_exact)
    for k in range(u_np.size):
        up = u_np.copy(); up[k] += eps
        um = u_np.copy(); um[k] -= eps
        rp = np.asarray(_stationarity_residual(jnp.asarray(up), args))
        rm = np.asarray(_stationarity_residual(jnp.asarray(um), args))
        J_fd[:, k] = (rp - rm) / (2 * eps)
    assert np.allclose(J_exact, J_fd, rtol=1e-5, atol=1e-7)


def test_implicit_diff_through_converged_solution():
    """d(θ_mid*)/d(θ0) via implicit function theorem, checked against FD."""
    n = 10
    M = jnp.linspace(2.0, 3.5, n)
    mid = n // 2

    def fn(u, params):
        theta0 = params
        return _stationarity_residual(u, (M, theta0, GAMMA, n))

    def u0_fn(params):
        return jnp.concatenate([jnp.radians(jnp.linspace(26.0, 6.0, n)), jnp.array([0.0])])

    sol_fn = make_differentiable_solution(fn, u0_fn, lambda u: u[mid])

    theta0 = jnp.asarray(math.radians(26.0))
    g = float(jax.grad(sol_fn)(theta0))
    assert math.isfinite(g)

    h = 1e-6
    g_fd = (float(sol_fn(theta0 + h)) - float(sol_fn(theta0 - h))) / (2 * h)
    assert g == pytest.approx(g_fd, rel=1e-4, abs=1e-6)


# --------------------------------------------------------------------------- #
# mass-closure integral (§2.D)                                                #
# --------------------------------------------------------------------------- #
def _numpy_curve_mass_flux(x, r, M, theta, gamma):
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dr = r[i + 1] - r[i]
        ds = math.hypot(dx, dr)
        beta = math.atan2(dr, dx)
        Mavg = 0.5 * (max(M[i], 1.001) + max(M[i + 1], 1.001))
        thavg = 0.5 * (theta[i] + theta[i + 1])
        ravg = max(0.5 * (r[i] + r[i + 1]), 1e-9)
        T = 1.0 / (1.0 + 0.5 * (gamma - 1) * Mavg**2)
        rho = T ** (1.0 / (gamma - 1))
        V = Mavg * math.sqrt(gamma * T)
        total += 2 * math.pi * ravg * rho * V * abs(math.sin(beta - thavg)) * ds
    return total


def test_curve_mass_flux_matches_numpy():
    n = 11
    x = np.linspace(0.5, 4.0, n)
    r = np.linspace(0.3, 1.0, n)
    M = np.linspace(2.0, 3.5, n)
    theta = np.radians(np.linspace(25.0, 5.0, n))
    jx = float(jr.curve_mass_flux(x, r, M, theta, GAMMA))
    ref = _numpy_curve_mass_flux(x, r, M, theta, GAMMA)
    assert jx == pytest.approx(ref, rel=1e-10, abs=1e-12)
    assert jx > 0.0


def test_curve_mass_flux_is_differentiable():
    n = 8
    x = jnp.linspace(0.5, 4.0, n)
    r = jnp.linspace(0.3, 1.0, n)
    M = jnp.linspace(2.0, 3.5, n)
    theta = jnp.radians(jnp.linspace(25.0, 5.0, n))
    g = jax.grad(lambda rr: jr.curve_mass_flux(x, rr, M, theta, GAMMA))(r)
    assert np.all(np.isfinite(np.asarray(g)))
