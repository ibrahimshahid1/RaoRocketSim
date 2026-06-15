"""
raosim.jax.pack — flat unknown-vector <-> structured CE state (J3).

Mirrors the role of ``_pack_bvp`` / ``_unpack_bvp`` (rao_variational.py:1787/1844)
for the JAX backend.  Packing is a pure, static-shape transform so ``jit`` and
``jacrev`` apply.

The CE unknown layout (this phase) is::

    u = [ x(n) | r(n) | M(n) | theta(n) | log_C(1) ]

Wall + characteristic-net unknowns (the full Phase-6 coupled vector,
REWRITE_PLAN.md §2.F) extend this in J3b once the MOC march is ported; the
layout is deliberately block-contiguous so appending wall blocks is additive.
"""

from __future__ import annotations

from typing import NamedTuple

import raosim.jax  # noqa: F401  -- enables x64
import jax.numpy as jnp


class CEState(NamedTuple):
    """Geometry-backed control-surface state (a JAX pytree)."""
    x: jnp.ndarray        # (n,)
    r: jnp.ndarray        # (n,)
    M: jnp.ndarray        # (n,)
    theta: jnp.ndarray    # (n,)
    log_C: jnp.ndarray    # scalar


def pack(state: CEState) -> jnp.ndarray:
    """Flatten a CEState into the BVP unknown vector."""
    return jnp.concatenate([
        jnp.asarray(state.x, dtype=jnp.float64),
        jnp.asarray(state.r, dtype=jnp.float64),
        jnp.asarray(state.M, dtype=jnp.float64),
        jnp.asarray(state.theta, dtype=jnp.float64),
        jnp.atleast_1d(jnp.asarray(state.log_C, dtype=jnp.float64)),
    ])


def unpack(u: jnp.ndarray, n: int) -> CEState:
    """Reconstruct a CEState from a flat unknown vector (``n`` CE nodes)."""
    u = jnp.asarray(u, dtype=jnp.float64)
    return CEState(
        x=u[0:n],
        r=u[n:2 * n],
        M=u[2 * n:3 * n],
        theta=u[3 * n:4 * n],
        log_C=u[4 * n],
    )


def n_unknowns(n: int) -> int:
    """Length of the packed vector for ``n`` CE nodes."""
    return 4 * n + 1


__all__ = ["CEState", "pack", "unpack", "n_unknowns"]
