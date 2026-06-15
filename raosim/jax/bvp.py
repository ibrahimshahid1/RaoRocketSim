"""
raosim.jax.bvp — Optimistix Levenberg-Marquardt least-squares solve (J3).

Replaces ``scipy.optimize.least_squares`` (rao_variational.py:2965/3559) for the
JAX backend.  The LM step uses the *exact* autodiff Jacobian (Optimistix calls
``jax.jacfwd`` internally), and the converged solution ``u*(params)`` is
differentiable via the implicit function theorem (Optimistix's default) — no
backprop through the iterations.  That implicit diff is what powers the J6
gradient API (sensitivity fields).

This module is solver-agnostic: callers pass a residual ``fn(u, args) -> array``.
J3b assembles the full grouped coupled-wall residual on top; J4 then checks the
``max_scaled < 2e-3`` Phase-6 gate on the ε=10/L80/γ=1.4 reference case.

Inner secant loops (θ_B, point D) will likewise be wrapped as
``optimistix.root_find`` so their gradients also come from the implicit function
theorem (REWRITE_PLAN.md §2.F / JAX_DIFFERENTIABLE_PLAN.md §4.5).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import raosim.jax  # noqa: F401  -- enables x64
import jax.numpy as jnp
import optimistix as optx


class LSResult(NamedTuple):
    u: jnp.ndarray          # solution vector
    residual: jnp.ndarray   # residual at the solution
    max_abs: jnp.ndarray    # max |residual|
    rms: jnp.ndarray        # rms residual
    steps: jnp.ndarray      # iterations taken
    success: bool           # converged within tolerances


def least_squares_solve(
    fn: Callable,
    u0: jnp.ndarray,
    args=None,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int = 256,
) -> LSResult:
    """
    Solve ``min_u ||fn(u, args)||²`` with Levenberg-Marquardt + exact Jacobian.

    Returns an :class:`LSResult`.  ``throw=False`` so non-convergence is reported
    in ``success`` rather than raised (the caller decides how to gate reliability).
    """
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    sol = optx.least_squares(
        fn, solver, u0, args=args, max_steps=max_steps, throw=False,
    )
    res = fn(sol.value, args)
    res = jnp.asarray(res)
    return LSResult(
        u=sol.value,
        residual=res,
        max_abs=jnp.max(jnp.abs(res)) if res.size else jnp.asarray(0.0),
        rms=jnp.sqrt(jnp.mean(res ** 2)) if res.size else jnp.asarray(0.0),
        steps=sol.stats.get("num_steps", jnp.asarray(-1)),
        success=(sol.result == optx.RESULTS.successful),
    )


def make_differentiable_solution(
    fn: Callable,
    u0_fn: Callable,
    readout: Callable,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_steps: int = 256,
) -> Callable:
    """
    Build ``params -> readout(u*(params))`` where ``u*`` is the converged LM
    solution of ``fn(u, params)``.  Differentiable via the implicit function
    theorem, so ``jax.grad`` of the returned callable gives exact design
    sensitivities (the J6 mechanism, previewed here).

    Parameters
    ----------
    fn       : residual ``fn(u, params) -> array``
    u0_fn    : ``params -> u0`` initial guess
    readout  : ``u* -> scalar`` quantity of interest
    """
    solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)

    def solution(params):
        sol = optx.least_squares(
            fn, solver, u0_fn(params), args=params,
            max_steps=max_steps, throw=False,
        )
        return readout(sol.value)

    return solution


__all__ = ["LSResult", "least_squares_solve", "make_differentiable_solution"]
