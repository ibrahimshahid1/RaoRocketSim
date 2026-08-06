"""
tests/test_mdo_propellants.py — propellant as a first-class MDO input.

Pins that the MDO is a *design tool*, not a single-engine script: the chamber
gases, L*, densities, coolant properties and the coolant wall limit all follow
the selected propellant combination, and hydrogen correctly carries **no**
coking limit (no carbon) so the gas-side material limit governs instead.
"""

from __future__ import annotations

import pytest

import raosim.jax  # noqa: F401  -- float64

from raosim.mdo.schema import MissionSpec
from raosim.mdo.propellants import get_propellant, available, PROPELLANTS


def test_table_covers_the_repo_propellants():
    names = available()
    for expect in ("lox/rp-1", "lox/lch4", "lox/lh2", "n2o4/mmh"):
        assert expect in names
    # aliases resolve
    assert get_propellant("kerolox").name == "LOX/RP-1"
    assert get_propellant("hydrolox").name == "LOX/LH2"
    assert get_propellant("METHALOX").name == "LOX/LCH4"
    with pytest.raises(KeyError):
        get_propellant("unobtainium/handwavium")


def test_lstar_matches_sp125_table_4_1():
    """L* values come from SP-125 Table 4-1 (in inches, converted)."""
    inch = 0.0254
    assert get_propellant("lox/rp-1").l_star == pytest.approx(45.0 * inch)
    # SP-125: LOX/LH2 30-40 in (LH2 injection); N2O4/hydrazine-base 30-35 in
    assert 30 * inch <= get_propellant("lox/lh2").l_star <= 40 * inch
    assert 30 * inch <= get_propellant("n2o4/mmh").l_star <= 35 * inch


def test_coolant_wall_limits_follow_sp8087():
    """SP-8087: RP-1 850 °F (728 K); hydrazine family / alcohols 600 °F (589 K);
    hydrogen cannot coke at all."""
    assert get_propellant("lox/rp-1").coolant_wall_limit_K == pytest.approx(728.0)
    assert get_propellant("n2o4/mmh").coolant_wall_limit_K == pytest.approx(589.0)
    assert get_propellant("lox/lh2").coolant_wall_limit_K is None


def test_estimates_are_flagged_not_hidden():
    """Methane and N2O/ethanol post-date SP-125/SP-8087 — their L*/wall limits
    are engineering estimates and must say so."""
    assert get_propellant("lox/lch4").estimated is True
    assert get_propellant("n2o/ethanol").estimated is True
    assert get_propellant("lox/rp-1").estimated is False
    for p in PROPELLANTS.values():
        if p.estimated:
            assert "ESTIMATE" in p.notes.upper()


def test_mission_for_propellant_drives_the_constants():
    """The mission's gases, L*, densities and coolant properties change with
    the selected combination — the whole point of the refactor."""
    rp1 = MissionSpec.for_propellant("lox/rp-1", 13.0e3)
    lh2 = MissionSpec.for_propellant("lox/lh2", 13.0e3)
    assert rp1.propellant_name == "LOX/RP-1"
    assert lh2.propellant_name == "LOX/LH2"
    assert lh2.Tc != rp1.Tc and lh2.gamma != rp1.gamma
    assert lh2.R_gas > rp1.R_gas            # light exhaust ⇒ larger R
    assert lh2.rho_cool < rp1.rho_cool      # LH2 is far less dense
    assert lh2.l_star < rp1.l_star          # SP-125: 35 in vs 45 in


def test_mdo_inputs_and_efficiency_split_match_traditional_propellants():
    """The MDO defaults/factory do not drift from repository performance data."""
    from raosim.propellants import get_propellant as get_traditional

    base = MissionSpec()
    traditional = get_traditional("LOX/RP-1")
    assert base.OF == pytest.approx(traditional.OF)
    assert base.R_gas == pytest.approx(traditional.R_gas)
    assert base.Tc == pytest.approx(traditional.Tc)
    assert base.eta_cstar == pytest.approx(traditional.eta_cstar)
    assert base.eta_CF == pytest.approx(traditional.eta_CF)

    for name in ("lox/rp-1", "lox/lch4", "lox/lh2", "n2o4/mmh"):
        mdo = get_propellant(name)
        legacy = get_traditional(mdo.name)
        mission = MissionSpec.for_propellant(name, 13.0e3)
        assert mission.eta_cstar == pytest.approx(legacy.eta_cstar)
        assert mission.eta_CF == pytest.approx(legacy.eta_CF)
        assert mdo.eta_cstar == pytest.approx(legacy.eta_cstar)
        assert mdo.eta_CF == pytest.approx(legacy.eta_CF)


def test_hydrogen_has_no_binding_coking_screen():
    """No carbon ⇒ no coking limit; the screen is set beyond any attainable
    wall temperature so the gas-side material constraint governs."""
    lh2 = MissionSpec.for_propellant("lox/lh2", 13.0e3)
    assert lh2.rp1_coking_wall_temp_K > 5.0e3
    rp1 = MissionSpec.for_propellant("lox/rp-1", 13.0e3)
    assert rp1.rp1_coking_wall_temp_K == pytest.approx(728.0)


def test_film_capacity_is_explicitly_sized_for_sp8087_reserve():
    m = MissionSpec()
    assert m.film_capacity_margin == pytest.approx(2.0)
    assert m.film_system_capacity_fraction == pytest.approx(0.60)


def test_explicit_overrides_win():
    m = MissionSpec.for_propellant("lox/rp-1", 13.0e3, OF=2.9, burn_time=42.0)
    assert m.OF == pytest.approx(2.9)
    assert m.burn_time == pytest.approx(42.0)
