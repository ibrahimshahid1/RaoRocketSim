"""
tests/test_mdo_engine.py — Phase-7 coupled whole-engine gate.

Pins the integration milestone: all four blocks solve as one differentiable
evaluation, the §5 hydraulic edge (cooling Δp → pump rise) is genuinely closed,
the η_c* combustion loop is a real (optional) fixed point with a measurable
ablation, and a single gradient flows from x through every block and both IFT
solves and matches finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec, DesignVector
from raosim.mdo.grid import build_station_grid
from raosim.mdo.cooling import solve_cooling
from raosim.mdo.injector import injector_readouts
from raosim.mdo.pump import electric_feed
from raosim.mdo.engine import (
    solve_engine, engine_outputs, ablation_delta, eta_cstar_coupled,
)


def _x():
    return DesignVector(Pc=jnp.asarray(3.0e6), eps=jnp.asarray(8.0),
                        dp_f_frac=jnp.asarray(0.2), dp_o_frac=jnp.asarray(0.2))


def test_engine_converges_and_reports_constraints():
    m = MissionSpec()
    r = solve_engine(_x(), m)
    assert abs(float(r.thrust_residual)) < 1e-9      # thrust closed
    assert float(r.Rt) > 0 and float(r.mdot) > 0
    assert 150.0 < float(r.Isp) < 400.0              # sane sea-level Isp
    for key in ("separation_margin", "coking_margin_min", "chug_margin_min",
                "pintle_transition_margin", "nss_margin_min",
                "tip_speed_margin_min"):
        assert key in r.constraints


def test_hydraulic_edge_is_closed():
    """Cooling jacket Δp feeds the fuel pump rise (previously a 0 placeholder);
    the fuel rise exceeds the ox rise by exactly the regen Δp."""
    m = MissionSpec()
    r = solve_engine(_x(), m)
    assert float(r.dp_regen) > 0.0
    # dp_rise_fuel = Pc(1+χ_f) + Δp_regen − P_tank_f ; dp_rise_ox has no regen
    expected_fuel = (float(_x().Pc) * 1.2 + float(r.dp_regen) - m.P_tank_fuel)
    assert float(r.dp_rise_fuel) == pytest.approx(expected_fuel, rel=1e-9)
    gap = float(r.dp_rise_fuel) - float(r.dp_rise_ox)
    assert gap == pytest.approx(float(r.dp_regen) + (m.P_tank_ox - m.P_tank_fuel),
                                rel=1e-9)


def test_integration_parity_with_standalone_blocks():
    """The engine's sub-results equal the blocks called standalone at the
    engine's converged (Rt, mdot) — integration changes no block physics."""
    m = MissionSpec()
    r = solve_engine(_x(), m)
    mdot_f = float(r.mdot) / (1.0 + m.OF)
    mdot_o = float(r.mdot) * m.OF / (1.0 + m.OF)
    grid = build_station_grid(r.Rt, _x().eps, m)
    _, cool = solve_cooling(grid, Pc=_x().Pc, gamma=jnp.asarray(m.gamma),
                            Tc=jnp.asarray(m.Tc),
                            c_star_del=r.eta_cstar * jnp.asarray(m.c_star_ideal()),
                            mdot_cool=jnp.asarray(m.cooling_fraction * mdot_f),
                            mission=m)
    assert float(cool.dp_total) == pytest.approx(float(r.dp_regen), rel=1e-9)
    inj = injector_readouts(Pc=_x().Pc, chi_f=_x().dp_f_frac, chi_o=_x().dp_o_frac,
                            D_pintle=jnp.asarray(m.pintle_diameter),
                            mdot_fuel=jnp.asarray(mdot_f), mdot_ox=jnp.asarray(mdot_o),
                            mission=m)
    assert float(inj.momentum_ratio) == pytest.approx(
        float(r.injector.momentum_ratio), rel=1e-9)


def test_end_to_end_differentiable_through_closed_edge():
    """AD of package mass w.r.t. Pc — which flows Pc → state → cooling → Δp_regen
    → pump → battery — matches central differences (both IFT solves included)."""
    m = MissionSpec()

    def pkg(pc):
        xa = jnp.array([pc, 8.0, 0.2, 0.2])
        return engine_outputs(xa, m, outputs=("package_mass",))[0]

    x0 = 3.0e6
    ad = float(jax.grad(pkg)(jnp.asarray(x0)))
    h = 1e-6 * x0
    fd = (float(pkg(jnp.asarray(x0 + h))) - float(pkg(jnp.asarray(x0 - h)))) / (2 * h)
    assert ad == pytest.approx(fd, rel=2e-4)
    assert abs(ad) > 0.0                              # the edge actually couples


def test_isp_ad_matches_fd():
    m = MissionSpec()
    fn = lambda e: engine_outputs(jnp.array([3.0e6, e, 0.2, 0.2]), m,
                                  outputs=("Isp",))[0]
    ad = float(jax.grad(fn)(jnp.asarray(8.0)))
    h = 1e-6 * 8.0
    fd = (float(fn(jnp.asarray(8.0 + h))) - float(fn(jnp.asarray(8.0 - h)))) / (2 * h)
    assert ad == pytest.approx(fd, rel=1e-4)


def test_eta_cstar_frozen_by_default_and_ablation_measurable():
    m = MissionSpec()
    r_off = solve_engine(_x(), m, couple_eta_cstar=False)
    assert float(r_off.eta_cstar) == pytest.approx(m.eta_cstar, rel=1e-12)
    r_on = solve_engine(_x(), m, couple_eta_cstar=True)
    # coupled η_c* is the self-consistent fixed point of the surrogate at TMR*
    tmr = float(r_on.injector.momentum_ratio)
    assert float(r_on.eta_cstar) == pytest.approx(
        float(eta_cstar_coupled(jnp.asarray(tmr), m)), rel=1e-9)
    # ablation is nonzero and matches the direct Isp difference
    d = float(ablation_delta(_x(), m, "Isp"))
    assert abs(d) > 1e-3
    assert d == pytest.approx(float(r_on.Isp) - float(r_off.Isp), rel=1e-6)


def test_coupled_solve_is_jittable():
    m = MissionSpec()
    xa = _x().to_array()
    f = jax.jit(lambda a: engine_outputs(a, m, outputs=("Isp", "package_mass")))
    out = np.asarray(f(xa))
    ref = np.asarray(engine_outputs(xa, m, outputs=("Isp", "package_mass")))
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)
