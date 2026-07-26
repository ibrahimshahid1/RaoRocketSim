"""
tests/test_mdo_cooling_gate.py — Phase-4a AD-vs-FD acceptance gate.

Total derivatives of converged cooling outputs (throat wall temperature, jacket
Δp, coolant exit temperature) with respect to the differentiable design inputs
(P_c, ṁ_cool) must come from the implicit differentiation of the Newton root
(``solve_cooling``) and agree with re-solved central differences to ~1e-4 in the
smooth regime (plan §11 Phase-4a gate, §12.1 rule 4).  This is the property that
makes the block usable inside a gradient-based MDF optimizer: exact derivatives
through the converged state, free of finite-difference step-size noise.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec
from raosim.mdo.grid import build_station_grid
from raosim.mdo.cooling import solve_cooling


_M = MissionSpec()
_G = build_station_grid(jnp.asarray(0.025), jnp.asarray(8.0), _M)
_TI = _G.throat_index
_BASE = dict(gamma=jnp.asarray(1.24), Tc=jnp.asarray(3550.0),
             c_star_del=jnp.asarray(1750.0))


def _outputs_of_Pc(Pc):
    T_wg, march = solve_cooling(_G, Pc=Pc, mdot_cool=jnp.asarray(1.02),
                                mission=_M, **_BASE)
    return jnp.stack([T_wg[_TI], march.dp_total, march.T_coolant_exit])


def _outputs_of_mdot(mdot):
    T_wg, march = solve_cooling(_G, Pc=jnp.asarray(3.0e6), mdot_cool=mdot,
                                mission=_M, **_BASE)
    return jnp.stack([T_wg[_TI], march.dp_total, march.T_coolant_exit])


def _central_fd(fn, x0, rel=1e-5):
    h = rel * x0
    return (np.asarray(fn(jnp.asarray(x0 + h)))
            - np.asarray(fn(jnp.asarray(x0 - h)))) / (2.0 * h)


@pytest.mark.parametrize("fn, x0", [(_outputs_of_Pc, 3.0e6),
                                    (_outputs_of_mdot, 1.02)])
def test_ad_matches_central_difference(fn, x0):
    ad = np.asarray(jax.jacfwd(fn)(jnp.asarray(x0)))
    fd = _central_fd(fn, x0)
    # relative agreement per output, with a small absolute floor
    denom = np.maximum(np.abs(fd), 1e-8)
    rel_err = np.abs(ad - fd) / denom
    assert np.max(rel_err) < 1e-4, (
        f"AD-vs-FD rel err {rel_err} too large\n ad={ad}\n fd={fd}")


def test_reverse_equals_forward_mode():
    """Adjoint (reverse) and direct (forward) total derivatives agree — the
    unified-derivatives identity (plan §4.3)."""
    x = jnp.asarray(3.0e6)
    fwd = np.asarray(jax.jacfwd(_outputs_of_Pc)(x))
    rev = np.asarray(jax.jacrev(_outputs_of_Pc)(x))
    np.testing.assert_allclose(fwd, rev, rtol=1e-8, atol=1e-8)


def test_block_is_jittable():
    """The converged-state evaluation is jit-safe (no host callbacks)."""
    jf = jax.jit(_outputs_of_Pc)
    out = np.asarray(jf(jnp.asarray(3.0e6)))
    ref = np.asarray(_outputs_of_Pc(jnp.asarray(3.0e6)))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
