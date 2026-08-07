"""
tests/test_mdo_bounds.py — design bounds derived from the architecture.

The defect these replace: ``Pc in [1.5, 6.0] MPa`` and ``eps in [3, 40]`` were
hard-coded from the 13 kN LOX/RP-1 baseline and then applied unchanged to every
propellant, ambient and thrust class.

Two properties are pinned here.

**One source of truth.**  A bound and the constraint that screens the same
limit must come from the same data.  ``eps`` is bounded by the Rao chart grid
that ``chart_domain_margin`` uses; extend the chart and the design space widens
with it, automatically.

**Sourced or labelled.**  Every chamber-pressure window either cites Yang 2004 /
SP-125, or is flagged ``literature=False`` so it cannot be quoted as sourced.
"""

from __future__ import annotations

import pytest

from raosim.mdo.bounds import (
    ArchitecturePressureLimit,
    PRESSURE_LIMITS,
    chamber_pressure_bounds,
    expansion_ratio_bounds,
    expansion_ratio_reference,
)
from raosim.mdo.schema import MissionSpec


# --------------------------------------------------------------------------- #
# eps: the Rao chart's own tabulated box                                       #
# --------------------------------------------------------------------------- #
def test_eps_bounds_are_read_from_the_chart_grid_not_restated():
    """Reading ``_EPS_GRID`` rather than hard-coding its endpoints is what
    makes the bound track the data."""
    from raosim.mdo.grid import _EPS_GRID

    lo, hi = expansion_ratio_bounds()
    assert lo == pytest.approx(float(_EPS_GRID[0]))
    assert hi == pytest.approx(float(_EPS_GRID[-1]))


def test_old_hardcoded_eps_box_was_wrong_in_both_directions():
    """Regression pin for the specific defect.

    The chart is tabulated over eps 4-50.  The retired box was [3, 40]: its
    lower end admitted designs the chart cannot evaluate (which the domain
    constraint then had to reject), and its upper end capped the optimiser 20 %
    below what the model actually supports.
    """
    lo, hi = expansion_ratio_bounds()
    assert lo > 3.0, "old lower bound admitted off-chart designs"
    assert hi > 40.0, "old upper bound capped below the chart"
    assert (lo, hi) == pytest.approx((4.0, 50.0))


def test_every_eps_in_the_box_is_chart_admissible():
    """The bound and ``rao_chart_domain_violation`` must agree exactly: no
    interior point may be rejected, and the endpoints must sit on the edge."""
    import numpy as np
    import jax.numpy as jnp
    from raosim.mdo.grid import rao_chart_domain_violation

    lo, hi = expansion_ratio_bounds()
    for eps in np.linspace(lo, hi, 25):
        v = np.asarray(rao_chart_domain_violation(jnp.asarray(eps),
                                                  jnp.asarray(80.0)))
        assert np.max(v[:2]) <= 1e-9, f"eps={eps} inside bounds but off-chart"


def test_eps_reference_follows_ambient_not_propellant():
    """SP-125 ch. III: the Alpha upper stages "operate in the vacuum and can
    use the largest practical expansion area ratio"; the sea-level A-1 cannot.
    So the seed is an ambient question."""
    sl = expansion_ratio_reference(MissionSpec(Pa=101325.0))
    vac = expansion_ratio_reference(MissionSpec(Pa=1.0))
    assert vac > sl
    lo, hi = expansion_ratio_bounds()
    for ref in (sl, vac):
        assert lo <= ref <= hi


def test_eps_bounds_do_not_re_encode_separation():
    """Separation is already ``separation_margin``, evaluated at the mission's
    ambient.  If the bounds also moved with ambient there would be two sources
    of truth for one limit."""
    assert (expansion_ratio_bounds(MissionSpec(Pa=101325.0))
            == expansion_ratio_bounds(MissionSpec(Pa=1.0)))


# --------------------------------------------------------------------------- #
# Pc: an architecture property, with its basis attached                        #
# --------------------------------------------------------------------------- #
def test_pressure_ceilings_differ_by_cycle():
    """Yang 2004 gives a different ceiling AND a different mechanism for each
    cycle; a single box cannot be right for all of them."""
    pf = chamber_pressure_bounds("pressure_fed")
    gg = chamber_pressure_bounds("gas_generator")
    ex = chamber_pressure_bounds("expander")
    sc = chamber_pressure_bounds("staged_combustion")
    assert pf.upper < ex.upper < gg.upper < sc.upper
    assert len({p.mechanism for p in (pf, gg, ex, sc)}) == 4


@pytest.mark.parametrize("arch,ceiling_mpa", [
    ("expander", 10.0),            # Yang: heat-transfer limited
    ("gas_generator", 15.0),       # Yang: 10-15 MPa performance optimum
    ("staged_combustion", 25.0),   # Yang: hardware limited 20-25 MPa
])
def test_yang_ceilings_match_the_monograph(arch, ceiling_mpa):
    assert chamber_pressure_bounds(arch).upper == pytest.approx(
        ceiling_mpa * 1.0e6)


def test_pressure_fed_ceiling_is_consistent_with_the_sp125_a4_engine():
    """SP-125 tank pressures run 100-400 psia; the worked A-4 engine sits at
    100 psia nozzle stagnation from a 165 psia tank.  The ceiling must be above
    that worked point and below the 400 psia tank pressure itself."""
    pf = chamber_pressure_bounds("pressure_fed")
    a4_pc = 100.0 * 6894.757          # psia -> Pa
    tank_max = 400.0 * 6894.757
    assert a4_pc < pf.upper < tank_max


def test_unsourced_windows_are_labelled_as_such():
    """The electric-pump window is a repository thermal-feasibility finding,
    not a published cycle limit.  It must not be quotable as literature."""
    ep = chamber_pressure_bounds("electric_pump")
    assert ep.literature is False
    assert "not a published limit" in ep.source
    for name, limit in PRESSURE_LIMITS.items():
        if limit.literature:
            assert ("Yang" in limit.source or "SP-125" in limit.source), name


def test_unknown_architecture_raises_rather_than_falling_back():
    """A silent fallback to the electric-pump box is the original defect."""
    with pytest.raises(KeyError, match="no chamber-pressure basis"):
        chamber_pressure_bounds("nuclear_thermal")


def test_all_windows_are_ordered_and_positive():
    for name, lim in PRESSURE_LIMITS.items():
        assert isinstance(lim, ArchitecturePressureLimit)
        assert 0.0 < lim.lower < lim.upper, name
        assert lim.mechanism and lim.source, name


# --------------------------------------------------------------------------- #
# Wiring into the design space                                                 #
# --------------------------------------------------------------------------- #
def test_design_space_uses_the_architecture_window():
    m = MissionSpec.for_thrust(5.0e3)
    spec = {s.name: s for s in m.scaled_design_space()}
    ep = chamber_pressure_bounds(m.feed_architecture)
    assert spec["Pc"].lower == pytest.approx(ep.lower)
    assert spec["Pc"].upper == pytest.approx(ep.upper)


def test_design_space_eps_is_the_chart_box_at_every_thrust_and_propellant():
    """The bound that used to be kerolox-13kN-specific must now be identical
    across thrust classes and propellants, because the chart is."""
    lo, hi = expansion_ratio_bounds()
    for m in (MissionSpec.for_thrust(5.0e3),
              MissionSpec.for_thrust(3.0e6),
              MissionSpec.for_propellant("LOX/LH2", 150.0e3),
              MissionSpec.for_propellant("N2O4/MMH", 7.5e3)):
        spec = {s.name: s for s in m.scaled_design_space()}
        assert spec["eps"].lower == pytest.approx(lo)
        assert spec["eps"].upper == pytest.approx(hi)


def test_reference_points_stay_inside_their_own_bounds():
    for m in (MissionSpec.for_thrust(5.0e3),
              MissionSpec.for_thrust(5.0e3, Pa=1.0),
              MissionSpec.for_propellant("LOX/LH2", 150.0e3, Pa=1.0)):
        for s in m.scaled_design_space():
            assert s.lower <= s.ref() <= s.upper, s.name
