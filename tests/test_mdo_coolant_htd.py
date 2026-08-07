"""
tests/test_mdo_coolant_htd.py — supercritical heat-transfer-deterioration screen.

Source: Nasuti & Pizzarelli, J. Supercritical Fluids 168:105066 (2021),
``propulsion_texts/nasuti2021.pdf``, Eq. (9) and its K = 0.187 threshold.

The point of these tests is as much about what the screen *refuses* to do as
what it computes.  Evaluating Eq. (9) with the constant coolant properties the
cooling march currently carries would produce a flat number that misses the
(beta/cp) peak the criterion exists to detect — so the screen must come back
unavailable, loudly, rather than pass.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.coolant_htd import (
    HTD_THRESHOLD, htd_availability, htd_group, htd_margin, screen,
)


def test_threshold_matches_the_paper():
    assert HTD_THRESHOLD == pytest.approx(0.187)


def test_group_is_dimensionless():
    """q_w/(G f_w) is J/kg; (beta/cp)_b is kg/J.  Scaling the pair inversely
    must leave the group unchanged — the check that the units were assembled
    the way Eq. (9) writes them."""
    base = htd_group(heat_flux=2.0e7, mass_flux=4.0e3,
                     friction_factor=0.02, beta_over_cp=3.0e-6)
    scaled = htd_group(heat_flux=2.0e8, mass_flux=4.0e4,
                       friction_factor=0.02, beta_over_cp=3.0e-6)
    assert float(base) == pytest.approx(float(scaled))


def test_group_matches_a_hand_evaluation():
    q, G, f, bcp = 2.0e7, 4.0e3, 0.02, 3.0e-6
    assert float(htd_group(q, G, f, bcp)) == pytest.approx(
        q / (G * f) * bcp, rel=1e-12)


def test_deterioration_rises_with_heat_flux_and_falls_with_mass_flux():
    """The paper's own reading of Eq. (9): deterioration "occurs for large heat
    flux q_w and low mass flux G"."""
    kw = dict(mass_flux=4.0e3, friction_factor=0.02, beta_over_cp=3.0e-6)
    assert float(htd_group(heat_flux=4.0e7, **kw)) > float(
        htd_group(heat_flux=2.0e7, **kw))

    kw2 = dict(heat_flux=2.0e7, friction_factor=0.02, beta_over_cp=3.0e-6)
    assert float(htd_group(mass_flux=2.0e3, **kw2)) > float(
        htd_group(mass_flux=8.0e3, **kw2))


def test_deterioration_rises_where_beta_over_cp_is_large():
    """"heat transfer deterioration occurs in the tube sections where
    (beta/cp)_b is sufficiently large" — the term that peaks at the Widom
    line, which is why a constant-property evaluation is useless."""
    kw = dict(heat_flux=2.0e7, mass_flux=4.0e3, friction_factor=0.02)
    near_widom = float(htd_group(beta_over_cp=9.0e-6, **kw))
    far = float(htd_group(beta_over_cp=1.0e-6, **kw))
    assert near_widom > far


def test_margin_sign_convention_matches_the_repo():
    """>= 0 feasible, like every other margin in the constraint set."""
    kw = dict(mass_flux=4.0e3, friction_factor=0.02, beta_over_cp=3.0e-6)
    safe = htd_margin(heat_flux=1.0e6, **kw)
    bad = htd_margin(heat_flux=1.0e9, **kw)
    assert float(safe) > 0.0
    assert float(bad) < 0.0


def test_margin_crosses_zero_exactly_at_the_threshold():
    q, G, f = 2.0e7, 4.0e3, 0.02
    bcp_crit = HTD_THRESHOLD * (G * f) / q
    assert float(htd_margin(q, G, f, bcp_crit)) == pytest.approx(0.0, abs=1e-15)


def test_group_is_differentiable():
    import jax

    g = jax.grad(lambda q: htd_group(q, 4.0e3, 0.02, 3.0e-6))(2.0e7)
    assert np.isfinite(float(g)) and float(g) > 0.0


# --------------------------------------------------------------------------- #
# Availability — the part that must not lie                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coolant", ["methane", "LCH4", "LH2", "hydrogen",
                                     "propane"])
def test_htd_prone_coolants_are_unavailable_without_real_fluid_properties(
        coolant):
    available, reason = htd_availability(coolant, has_real_fluid_properties=False)
    assert available is False
    assert "beta" in reason and "CoolProp" in reason


@pytest.mark.parametrize("coolant", ["RP-1", "rp1", "ethanol"])
def test_hydrocarbon_coolants_defer_to_the_coking_limit(coolant):
    """RP-1 is not HTD-governed; SP-8087 coking is its wall constraint.  The
    screen reports available with the reason naming the mechanism that does
    govern — silence would be indistinguishable from an unmodelled risk."""
    available, reason = htd_availability(coolant, has_real_fluid_properties=False)
    assert available is True
    assert "coking" in reason


def test_htd_prone_coolant_becomes_available_with_real_properties():
    available, reason = htd_availability("methane",
                                         has_real_fluid_properties=True)
    assert available is True
    assert reason == ""


def test_screen_returns_nan_not_a_fabricated_margin():
    """The failure mode this module exists to prevent: an unmodelled risk
    presented as a satisfied constraint."""
    s = screen("methane")
    assert s.available is False
    assert np.isnan(float(s.margin))
    assert np.isnan(float(s.group))
    assert "beta" in s.reason


def test_screen_computes_when_given_the_real_fluid_term():
    s = screen("methane", heat_flux=2.0e7, mass_flux=4.0e3,
               friction_factor=0.02, beta_over_cp=9.0e-6)
    assert s.available is True
    assert np.isfinite(float(s.margin))
    assert float(s.margin) == pytest.approx(
        HTD_THRESHOLD - float(htd_group(2.0e7, 4.0e3, 0.02, 9.0e-6)))


@pytest.mark.parametrize("combo,expected_K", [
    ("LOX/RP-1", 300.0),
    ("LOX/LCH4", 120.0),
    ("LOX/LH2", 25.0),
])
def test_jacket_inlet_temperature_comes_from_the_central_resolver(
        combo, expected_K):
    """Regression pin for a real drift bug.

    ``raosim.physics.default_coolant_inlet_temperature`` is described in its own
    docstring as "the central preliminary inlet-temperature default", and the
    traditional pipeline uses it.  ``MissionSpec.for_propellant`` did not, so
    the MDO ran *every* propellant at the class default of 320 K — an RP-1
    number.  That put methane into the jacket at T/T_c = 1.68 instead of 0.63
    and hydrogen at 9.7 instead of 0.75: a different fluid state, with the
    coolant enthalpy rise starting from the wrong place and the wall-temperature
    and coking margins computed off it.

    Same failure mode as R0 — a central convention one pipeline honoured and the
    other silently did not.
    """
    from raosim.mdo.schema import MissionSpec

    m = MissionSpec.for_propellant(combo, 20.0e3)
    assert m.coolant_temperature == pytest.approx(expected_K)
    assert m.coolant_temperature != 320.0 or expected_K == 320.0


def test_cryogenic_coolants_enter_the_jacket_below_their_critical_point():
    """The physical statement behind the previous test: a regen jacket is fed
    *liquid*.  If an inlet ever lands above T_c the coolant is a supercritical
    gas at station 0 and the whole march is describing something else."""
    from raosim.mdo.schema import MissionSpec

    # NIST/CoolProp critical temperatures.
    for combo, T_crit in (("LOX/LCH4", 190.6), ("LOX/LH2", 33.1)):
        m = MissionSpec.for_propellant(combo, 20.0e3)
        assert m.coolant_temperature < T_crit, combo


def test_current_mission_defaults_cannot_evaluate_the_screen():
    """Integration pin: with today's constant-property MissionSpec, every
    HTD-prone propellant must report unavailable.  If this ever starts passing
    silently, someone has wired constants into Eq. (9)."""
    from raosim.mdo.propellants import get_propellant

    for combo in ("lox/lch4", "lox/lh2"):
        coolant = get_propellant(combo).coolant_name
        available, reason = htd_availability(
            coolant, has_real_fluid_properties=False)
        assert available is False, combo
        assert reason
