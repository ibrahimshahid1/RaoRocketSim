"""
raosim.jax.api — public entry points for the JAX backend.

``solve_rao_bvp_jax`` lands in **J4**; ``rao_sensitivities`` (the gradient API)
lands in **J6** (see JAX_DIFFERENTIABLE_PLAN.md §6).  Both return the same
structures as the NumPy path so downstream consumers (export, plotting, CLI) are
unchanged.  This backend stays opt-in (``solver_backend="jax"``) and non-default
until the J5 chart-benchmark gate passes.
"""

from __future__ import annotations

import raosim.jax  # noqa: F401


def solve_rao_bvp_jax(config):
    """Differentiable Rao BVP solve.  Returns a RaoSolution-shaped result. (J4)"""
    raise NotImplementedError("J4: solve_rao_bvp_jax — see JAX_DIFFERENTIABLE_PLAN.md")


def rao_sensitivities(config):
    """Exact gradients of Cf/Isp/etc. w.r.t. design params and nodes. (J6)"""
    raise NotImplementedError("J6: gradient API — see JAX_DIFFERENTIABLE_PLAN.md §6")
