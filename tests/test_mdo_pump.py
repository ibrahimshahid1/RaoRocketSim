"""
tests/test_mdo_pump.py — Phase-6 parity + C¹-efficiency gate for
``raosim.mdo.pump``.

Two things are checked:

1. **Duty/power parity** — Q, head, ω, Ns, Nss and the electrical chain match
   the closed-form ``raosim.pumps`` conventions (``pumps.py`` L2410 Nss form,
   the ω=2πN/60 / Ns=ω√Q/(g0H)^¾ definitions) to ~1e-10.
2. **The new η(Ns) is C¹** where the shipped ``pumps._estimate_pump_efficiency``
   is not: the binned estimator jumps at its Q bin edges, the surrogate is
   smooth (finite ``jax.grad`` everywhere, small adjacent-sample deltas) and
   sits in the SP-125 60–85 % band, peaked at Ns_opt.
"""

from __future__ import annotations

import math
from dataclasses import replace
import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim import pumps as oracle
from raosim.mdo.schema import MissionSpec
from raosim.mdo.pump import (
    pump_stream, pump_efficiency, battery_masses, electric_feed,
)


def _case():
    m = MissionSpec()
    mdot = 3.37
    mf = mdot / (1.0 + m.OF)
    mo = mdot * m.OF / (1.0 + m.OF)
    Pc = 3.0e6
    dpf = Pc * 1.2 - m.P_tank_fuel
    dpo = Pc * 1.2 - m.P_tank_ox
    return m, dict(mf=mf, mo=mo, dpf=dpf, dpo=dpo, N=30000.0)


def test_pump_duty_parity_forms():
    """Q/head/ω/Ns/Nss match the pumps.py closed forms to ~1e-10."""
    m, d = _case()
    s = pump_stream(mdot=jnp.asarray(d["mf"]), dp_rise=jnp.asarray(d["dpf"]),
                    rho=m.rho_fuel, p_inlet=jnp.asarray(m.P_tank_fuel),
                    p_vapor=m.p_vapor_fuel, N_rpm=jnp.asarray(d["N"]), mission=m)
    g0 = m.g0
    Q = d["mf"] / m.rho_fuel
    head = d["dpf"] / (m.rho_fuel * g0)
    omega = 2.0 * math.pi * d["N"] / 60.0
    Ns = omega * math.sqrt(Q) / (g0 * head) ** 0.75
    npsh = max(m.P_tank_fuel - m.p_vapor_fuel, 0.0) / (m.rho_fuel * g0)
    Nss = omega * math.sqrt(Q) / (g0 * npsh) ** 0.75   # pumps.py L2410 form
    assert float(s.Q) == pytest.approx(Q, rel=1e-10)
    assert float(s.head) == pytest.approx(head, rel=1e-10)
    assert float(s.omega) == pytest.approx(omega, rel=1e-10)
    assert float(s.specific_speed) == pytest.approx(Ns, rel=1e-10)
    assert float(s.suction_specific_speed) == pytest.approx(Nss, rel=1e-10)


def test_pump_power_chain():
    """P_hyd/P_shaft/P_elec are consistent with η and the drive efficiencies."""
    m, d = _case()
    s = pump_stream(mdot=jnp.asarray(d["mf"]), dp_rise=jnp.asarray(d["dpf"]),
                    rho=m.rho_fuel, p_inlet=jnp.asarray(m.P_tank_fuel),
                    p_vapor=m.p_vapor_fuel, N_rpm=jnp.asarray(d["N"]), mission=m)
    P_hyd = d["mf"] * d["dpf"] / m.rho_fuel
    assert float(s.P_hydraulic) == pytest.approx(P_hyd, rel=1e-10)
    assert float(s.P_shaft) == pytest.approx(P_hyd / float(s.efficiency), rel=1e-10)
    assert float(s.P_electric) == pytest.approx(
        float(s.P_shaft) / (m.eta_motor * m.eta_inverter), rel=1e-10)


@pytest.mark.parametrize("eta_inverter", [0.85, 0.96, 1.0])
def test_inverter_output_density_matches_numpy_and_is_independent_of_losses(eta_inverter):
    """Lee PDF p. 7 Eqs. (28)-(29): inverter mass uses its output power."""
    base, d = _case()
    m = replace(base, eta_inverter=eta_inverter, electric_technology="custom")
    ef = electric_feed(mdot_fuel=jnp.asarray(d["mf"]), mdot_ox=jnp.asarray(d["mo"]),
                       dp_rise_fuel=jnp.asarray(d["dpf"]),
                       dp_rise_ox=jnp.asarray(d["dpo"]), N_rpm=jnp.asarray(d["N"]),
                       mission=m)
    shaft_power = float(ef.fuel.P_shaft + ef.ox.P_shaft)
    inverter_output = shaft_power / m.eta_motor
    assert float(ef.inverter_mass) == pytest.approx(
        inverter_output / m.inverter_power_density, rel=1e-12)
    spec = oracle.PumpSizingSpec(drive=oracle.ElectricDriveSpec(
        motor_efficiency=m.eta_motor, inverter_efficiency=m.eta_inverter,
        motor_power_density=m.motor_power_density,
        inverter_power_density=m.inverter_power_density))
    numpy_drive = oracle._drive_sizing("combined", shaft_power, d["N"], spec,
                                       bus_voltage=48.0)
    assert float(ef.inverter_mass) == pytest.approx(numpy_drive.inverter_mass, rel=1e-12)
    assert float(ef.P_electric_total) == pytest.approx(inverter_output / eta_inverter)


def test_efficiency_is_c1_where_binned_estimator_jumps():
    """C¹ ⟺ a finite central difference converges to a well-defined derivative.
    The binned estimator fails this across its Q bin edge (FD blows up); the
    surrogate passes everywhere (FD matches the analytic ``jax.grad``)."""
    # old binned estimator: central FD across the Q = 1e-4 edge is unbounded —
    # i.e. no derivative exists there (a genuine C0 discontinuity).
    h = 1e-9
    fd_binned = (oracle._estimate_pump_efficiency(1.0e-4 + h, 400.0)[0]
                 - oracle._estimate_pump_efficiency(1.0e-4 - h, 400.0)[0]) / (2 * h)
    assert abs(fd_binned) > 1.0e4

    # surrogate: finite analytic gradient that a central difference converges to
    m = MissionSpec()
    grad = jax.grad(lambda x: pump_efficiency(x, m))
    for Ns in (0.1, 0.2, 0.35, 0.55, 0.9, 1.5, 2.2):
        hh = 1e-6 * Ns
        fd = (float(pump_efficiency(jnp.asarray(Ns + hh), m))
              - float(pump_efficiency(jnp.asarray(Ns - hh), m))) / (2 * hh)
        assert float(grad(jnp.asarray(Ns))) == pytest.approx(fd, rel=1e-5, abs=1e-9)
    # and the derivative itself is finite and continuous across a fine sweep
    dyn = np.array([float(grad(jnp.asarray(x)))
                    for x in np.linspace(0.05, 2.5, 400)])
    assert np.all(np.isfinite(dyn))


def test_efficiency_band_and_peak():
    """η in the SP-125 rocket-pump band over realistic Ns, peaked at Ns_opt."""
    m = MissionSpec()
    peak = float(pump_efficiency(jnp.asarray(m.pump_ns_opt), m))
    assert peak == pytest.approx(m.pump_eta_peak, rel=1e-12)
    for Ns in (0.2, 0.35, 0.55, 0.9, 1.2):
        assert 0.60 <= float(pump_efficiency(jnp.asarray(Ns), m)) <= 0.85
    # monotone up below the optimum, down above it (sign of dη/dNs flips)
    g = jax.grad(lambda x: pump_efficiency(x, m))
    assert float(g(jnp.asarray(0.30))) > 0.0
    assert float(g(jnp.asarray(0.90))) < 0.0


def test_battery_epigraph_exposes_both_branches():
    """Lee-2021 power- and energy-limited masses are returned separately (the
    NLP takes the max as an epigraph; no max() in the block)."""
    m = MissionSpec()
    b = battery_masses(jnp.asarray(16.6e3), m)
    e = 16.6e3 * m.burn_time / m.eta_discharge / m.battery_energy_density
    p = 16.6e3 / m.battery_power_density
    assert float(b.energy_limited_mass) == pytest.approx(e, rel=1e-10)
    assert float(b.power_limited_mass) == pytest.approx(p, rel=1e-10)
    # at 120 s burn the power branch governs — but both are exposed, not maxed
    assert float(b.power_limited_mass) > float(b.energy_limited_mass)


def test_nss_and_tipspeed_margins_signed():
    """Suction (SP-8052) and tip-speed (SP-8109) screens are exposed as
    margins with the documented sign."""
    m, d = _case()
    ef = electric_feed(mdot_fuel=jnp.asarray(d["mf"]), mdot_ox=jnp.asarray(d["mo"]),
                       dp_rise_fuel=jnp.asarray(d["dpf"]),
                       dp_rise_ox=jnp.asarray(d["dpo"]), N_rpm=jnp.asarray(d["N"]),
                       mission=m)
    for s in (ef.fuel, ef.ox):
        assert float(s.nss_margin) == pytest.approx(
            m.pump_nss_max - float(s.suction_specific_speed), rel=1e-10)
        assert float(s.tip_speed_margin) == pytest.approx(
            m.pump_tip_speed_max - float(s.tip_speed), rel=1e-10)
