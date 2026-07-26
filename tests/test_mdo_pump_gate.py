"""
tests/test_mdo_pump_gate.py — Phase-6 AD-vs-FD acceptance gate.

Total derivatives of pump/feed outputs (specific speed, efficiency, electric
power, tip speed, battery mass) w.r.t. the differentiable inputs (Δp rise, pump
speed, mass flow) match re-evaluated central differences to ~1e-4 and are
jit-safe — so the block, and in particular the *smooth* η(Ns) that replaced the
binned estimator, drops into the MDF constraint Jacobian (plan §6.4, §12.1).
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec
from raosim.mdo.pump import pump_stream, electric_feed

_M = MissionSpec()
_MDOT = 3.37
_MF = _MDOT / (1.0 + _M.OF)
_MO = _MDOT * _M.OF / (1.0 + _M.OF)
_DPF = 3.0e6 * 1.2 - _M.P_tank_fuel
_DPO = 3.0e6 * 1.2 - _M.P_tank_ox
_BASE = dict(mdot=_MF, dp_rise=_DPF, N_rpm=30000.0)


def _stream_out(**over):
    kw = {**_BASE, **over}
    s = pump_stream(mdot=jnp.asarray(kw["mdot"]), dp_rise=jnp.asarray(kw["dp_rise"]),
                    rho=_M.rho_fuel, p_inlet=jnp.asarray(_M.P_tank_fuel),
                    p_vapor=_M.p_vapor_fuel, N_rpm=jnp.asarray(kw["N_rpm"]),
                    mission=_M)
    return jnp.stack([s.specific_speed, s.efficiency, s.P_electric,
                      s.tip_speed, s.suction_specific_speed])


def _fn_of(name):
    return lambda x: _stream_out(**{name: x})


def _central_fd(fn, x0, rel=1e-6):
    h = rel * abs(x0)
    return (np.asarray(fn(jnp.asarray(x0 + h)))
            - np.asarray(fn(jnp.asarray(x0 - h)))) / (2.0 * h)


@pytest.mark.parametrize("name, x0", [("dp_rise", _DPF), ("N_rpm", 30000.0),
                                      ("mdot", _MF)])
def test_ad_matches_central_difference(name, x0):
    fn = _fn_of(name)
    ad = np.asarray(jax.jacfwd(fn)(jnp.asarray(x0)))
    fd = _central_fd(fn, x0)
    denom = np.maximum(np.abs(fd), 1e-8)
    assert np.max(np.abs(ad - fd) / denom) < 1e-4, f"{name}: ad={ad} fd={fd}"


def test_battery_mass_differentiable_through_feed():
    """Battery power-limited mass has an exact, FD-matching derivative w.r.t.
    pump speed (through Ns → η → P_elec)."""
    def m_batt(N):
        ef = electric_feed(mdot_fuel=jnp.asarray(_MF), mdot_ox=jnp.asarray(_MO),
                           dp_rise_fuel=jnp.asarray(_DPF), dp_rise_ox=jnp.asarray(_DPO),
                           N_rpm=N, mission=_M)
        return ef.battery.power_limited_mass
    x = jnp.asarray(30000.0)
    ad = float(jax.grad(m_batt)(x))
    h = 1e-6 * 30000.0
    fd = (float(m_batt(x + h)) - float(m_batt(x - h))) / (2.0 * h)
    assert ad == pytest.approx(fd, rel=1e-4)


def test_forward_equals_reverse_and_jittable():
    fn = _fn_of("dp_rise")
    x = jnp.asarray(_DPF)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(fn)(x)),
                               np.asarray(jax.jacrev(fn)(x)), rtol=1e-9, atol=1e-9)
    jf = jax.jit(fn)
    np.testing.assert_allclose(np.asarray(jf(x)), np.asarray(fn(x)),
                               rtol=1e-10, atol=1e-10)
