"""
tests/test_mdo_injector.py — Phase-5 parity + branch-structure gate for
``raosim.mdo.injector``.

Parity oracle: the Son movable-pintle geometry is checked against the *audited*
``raosim.movable_pintle`` functions themselves (center_gap_area,
son_minimum_tip_area, opening_for_tip_area), and the orifice/TMR/blockage
algebra against the ``raosim.injector`` conventions, to ~1e-9.  Also pins the
plan-rule-8 structure: the movable minimum area is exposed as TWO branches with
a consistency inequality, never collapsed with a differentiable ``min()``.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim import movable_pintle as mp
from raosim.mdo.schema import MissionSpec
from raosim.mdo.injector import (
    injector_readouts, son_tip_area, opening_for_tip_area, center_gap_area,
)


def _case():
    m = MissionSpec()
    mdot = 3.37
    mf = mdot / (1.0 + m.OF)
    mo = mdot * m.OF / (1.0 + m.OF)
    d = dict(Pc=3.0e6, chi_f=0.22, chi_o=0.18, D_pintle=0.020,
             mdot_fuel=mf, mdot_ox=mo)
    return m, d


def _numpy_injector(m, d):
    """Independent NumPy oracle: orifice/TMR/BF inline (injector.py forms) +
    the audited movable_pintle geometry functions."""
    Pc, chi_f, chi_o = d["Pc"], d["chi_f"], d["chi_o"]
    Dp, mf, mo = d["D_pintle"], d["mdot_fuel"], d["mdot_ox"]
    dp_f, dp_o = chi_f * Pc, chi_o * Pc
    G_f = m.injector_cd_fuel * math.sqrt(2.0 * m.rho_fuel * dp_f)
    G_o = m.injector_cd_ox * math.sqrt(2.0 * m.rho_ox * dp_o)
    v_f, v_o = G_f / m.rho_fuel, G_o / m.rho_ox
    A_f, A_o = mf / G_f, mo / G_o
    m_r, m_a = mf * v_f, mo * v_o
    TMR = m_r / m_a
    delta = math.radians(m.pintle_deflector_angle_deg)
    spray = math.degrees(math.atan2(m_r * math.cos(delta),
                                    m_a + m_r * math.sin(delta)))
    slot_w = math.sqrt(A_f / (m.pintle_slot_count * m.pintle_slot_aspect_ratio))
    BF = m.pintle_slot_count * slot_w / (math.pi * Dp)
    A_cg = mp.center_gap_area(m.pintle_center_gap_diameter, m.pintle_rod_diameter)
    L = mp.opening_for_tip_area(A_f, post_diameter=Dp,
                                post_thickness=m.pintle_post_thickness,
                                tip_angle_deg=m.pintle_tip_angle_deg)
    return dict(dp_fuel=dp_f, dp_ox=dp_o, v_fuel=v_f, v_ox=v_o,
                area_fuel=A_f, area_ox=A_o, momentum_ratio=TMR,
                spray_half_angle_deg=spray, slot_width=slot_w,
                blockage_factor=BF, tip_opening=L, area_tip_branch=A_f,
                area_center_gap=A_cg,
                transition_margin=m.pintle_transition_area_fraction * A_cg - A_f,
                chug_margin_fuel=chi_f - m.injector_dp_stability_min,
                chug_margin_ox=chi_o - m.injector_dp_stability_min)


def test_injector_parity_numpy():
    """JAX injector_readouts ≈ audited NumPy/movable_pintle oracle to ~1e-9."""
    m, d = _case()
    r = injector_readouts(Pc=jnp.asarray(d["Pc"]), chi_f=jnp.asarray(d["chi_f"]),
                          chi_o=jnp.asarray(d["chi_o"]),
                          D_pintle=jnp.asarray(d["D_pintle"]),
                          mdot_fuel=jnp.asarray(d["mdot_fuel"]),
                          mdot_ox=jnp.asarray(d["mdot_ox"]), mission=m)
    ref = _numpy_injector(m, d)
    for k, v in ref.items():
        assert float(getattr(r, k)) == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_son_tip_area_roundtrip():
    """opening_for_tip_area inverts son_tip_area on the monotone branch, and
    both match the audited movable_pintle scalar functions."""
    m, d = _case()
    r_f = 0.5 * d["D_pintle"] - m.pintle_post_thickness
    A_target = 4.0e-5
    L = opening_for_tip_area(jnp.asarray(A_target), r_f, m.pintle_tip_angle_deg)
    A_back = son_tip_area(L, r_f, m.pintle_tip_angle_deg)
    assert float(A_back) == pytest.approx(A_target, rel=1e-10)
    # cross-check against the NumPy oracle geometry
    L_np = mp.opening_for_tip_area(A_target, post_diameter=d["D_pintle"],
                                   post_thickness=m.pintle_post_thickness,
                                   tip_angle_deg=m.pintle_tip_angle_deg)
    assert float(L) == pytest.approx(L_np, rel=1e-10)


def test_son_tip_area_nonzero_angle_matches_oracle():
    """The general (θ>0) Son Eq.(1) branch also matches movable_pintle."""
    m, d = _case()
    Dp, t, theta = d["D_pintle"], m.pintle_post_thickness, 12.0
    r_f = 0.5 * Dp - t
    for L in (2e-4, 5e-4, 9e-4):
        a_jax = float(son_tip_area(jnp.asarray(L), r_f, theta))
        a_np = mp.son_minimum_tip_area(L, post_diameter=Dp, post_thickness=t,
                                       tip_angle_deg=theta)
        assert a_jax == pytest.approx(a_np, rel=1e-10)


def test_two_branch_minimum_area_not_min_collapsed():
    """Plan rule 8: both branches are exposed with a consistency inequality —
    the effective minimum is NOT a differentiable min() inside the block."""
    m, d = _case()
    r = injector_readouts(Pc=jnp.asarray(d["Pc"]), chi_f=jnp.asarray(d["chi_f"]),
                          chi_o=jnp.asarray(d["chi_o"]),
                          D_pintle=jnp.asarray(d["D_pintle"]),
                          mdot_fuel=jnp.asarray(d["mdot_fuel"]),
                          mdot_ox=jnp.asarray(d["mdot_ox"]), mission=m)
    A_tip = float(r.area_tip_branch)
    A_cg = float(r.area_center_gap)
    frac = m.pintle_transition_area_fraction
    # both branches present and distinct
    assert A_tip > 0.0 and A_cg > 0.0 and A_tip != A_cg
    # consistency inequalities have the documented sign
    assert float(r.transition_margin) == pytest.approx(frac * A_cg - A_tip, rel=1e-12)
    assert float(r.branch_consistency) == pytest.approx(A_cg - A_tip, rel=1e-12)
    # baseline sits on the tip-controlled branch
    assert float(r.transition_margin) > 0.0


def test_transition_margin_exposed_when_violated():
    """Forcing a large radial area drives the tip area past the center-gap cap;
    the transition_margin goes negative (exposed), not clamped."""
    m, d = _case()
    # huge fuel flow at low Δp → large required tip area
    r = injector_readouts(Pc=jnp.asarray(1.5e6), chi_f=jnp.asarray(0.12),
                          chi_o=jnp.asarray(0.18), D_pintle=jnp.asarray(0.020),
                          mdot_fuel=jnp.asarray(6.0), mdot_ox=jnp.asarray(2.3),
                          mission=m)
    assert float(r.transition_margin) < 0.0
    assert float(r.branch_consistency) < 0.0


def test_spray_angle_is_arctan_tmr_at_zero_deflector():
    """δ=0 ⇒ the leading-order kinematic θ = arctan(TMR) (Freeberg 2019)."""
    m, d = _case()
    assert m.pintle_deflector_angle_deg == 0.0
    r = injector_readouts(Pc=jnp.asarray(d["Pc"]), chi_f=jnp.asarray(d["chi_f"]),
                          chi_o=jnp.asarray(d["chi_o"]),
                          D_pintle=jnp.asarray(d["D_pintle"]),
                          mdot_fuel=jnp.asarray(d["mdot_fuel"]),
                          mdot_ox=jnp.asarray(d["mdot_ox"]), mission=m)
    assert float(r.spray_half_angle_deg) == pytest.approx(
        math.degrees(math.atan(float(r.momentum_ratio))), rel=1e-10)
