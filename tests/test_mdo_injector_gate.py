"""
tests/test_mdo_injector_gate.py — Phase-5 AD-vs-FD acceptance gate.

The pintle block is closed-form (no implicit state), so total derivatives are
plain forward/reverse AD; they must still match re-evaluated central differences
to ~1e-4 and be jit-safe, so the block drops straight into the constraint
Jacobian of the MDF optimizer (plan §12.1 rule 4).
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec
from raosim.mdo.injector import injector_readouts

_M = MissionSpec()
_MDOT = 3.37
_MF = _MDOT / (1.0 + _M.OF)
_MO = _MDOT * _M.OF / (1.0 + _M.OF)
_BASE = dict(Pc=3.0e6, chi_f=0.22, chi_o=0.18, D_pintle=0.020,
             mdot_fuel=_MF, mdot_ox=_MO)


def _outputs(**over):
    kw = {**_BASE, **over}
    r = injector_readouts(Pc=jnp.asarray(kw["Pc"]), chi_f=jnp.asarray(kw["chi_f"]),
                          chi_o=jnp.asarray(kw["chi_o"]),
                          D_pintle=jnp.asarray(kw["D_pintle"]),
                          mdot_fuel=jnp.asarray(kw["mdot_fuel"]),
                          mdot_ox=jnp.asarray(kw["mdot_ox"]), mission=_M)
    return jnp.stack([r.momentum_ratio, r.spray_half_angle_deg,
                      r.blockage_factor, r.transition_margin, r.tip_opening])


def _fn_of(name):
    return lambda x: _outputs(**{name: x})


def _central_fd(fn, x0, rel=1e-6):
    h = rel * abs(x0)
    return (np.asarray(fn(jnp.asarray(x0 + h)))
            - np.asarray(fn(jnp.asarray(x0 - h)))) / (2.0 * h)


@pytest.mark.parametrize("name, x0", [("chi_f", 0.22), ("chi_o", 0.18),
                                      ("D_pintle", 0.020), ("Pc", 3.0e6)])
def test_ad_matches_central_difference(name, x0):
    fn = _fn_of(name)
    ad = np.asarray(jax.jacfwd(fn)(jnp.asarray(x0)))
    fd = _central_fd(fn, x0)
    denom = np.maximum(np.abs(fd), 1e-8)
    assert np.max(np.abs(ad - fd) / denom) < 1e-4, f"{name}: ad={ad} fd={fd}"


def test_forward_equals_reverse():
    fn = _fn_of("chi_f")
    x = jnp.asarray(0.22)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(fn)(x)),
                               np.asarray(jax.jacrev(fn)(x)), rtol=1e-9, atol=1e-9)


def test_jittable():
    fn = jax.jit(_fn_of("chi_f"))
    out = np.asarray(fn(jnp.asarray(0.22)))
    ref = np.asarray(_fn_of("chi_f")(jnp.asarray(0.22)))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)
