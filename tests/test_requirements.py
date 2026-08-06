"""
tests/test_requirements.py — Layer 0 gate: requirements in, honest coverage out.

Two things are under test, and the second matters more than the first.

1. **Mapping.** An :class:`~raosim.requirements.EngineRequirement` stated in
   NASA SP-125 §2.1 terms produces the right ``MissionSpec`` and the right
   constraint selection.

2. **Honesty.** A requirement that is only partially screened, or not screened
   at all, is *reported as such* and can never be laundered into a claim that
   the requirement is met.  These are the tests that would fail if someone
   later "simplified" ``requirements_met`` into a plain boolean.

The SP-125 conventions pinned here are quoted in ``raosim/requirements.py``;
the source is ``propulsion_texts/19710019929.pdf`` §2.1, printed p. 31.
"""

from __future__ import annotations

import numpy as np
import pytest

from raosim.requirements import (
    Coverage,
    EngineRequirement,
    resolve_requirement,
)

_THRUST = 5.0e3


def _req(**kw) -> EngineRequirement:
    base = dict(thrust=_THRUST, thrust_condition="sea_level",
                isp_min=230.0, flight_duration=30.0, propellant="LOX/RP-1")
    base.update(kw)
    return EngineRequirement(**base)


def _cov(resolved, name):
    for c in resolved.coverage:
        if c.requirement == name:
            return c
    raise AssertionError(f"{name!r} not in coverage: "
                         f"{[c.requirement for c in resolved.coverage]}")


# --------------------------------------------------------------------------- #
# SP-125 §2.1 conventions                                                      #
# --------------------------------------------------------------------------- #
def test_thrust_condition_sets_ambient_sea_level():
    """SP-125 §2.1: booster thrust "usually quoted for sea-level conditions"."""
    assert resolve_requirement(_req()).mission.Pa == pytest.approx(101325.0)


def test_thrust_condition_vacuum_is_small_but_not_zero():
    """Upper-stage thrust is "quoted for that environment" (near-vacuum).

    Not exactly zero: the separation and collapse screens compare against
    ambient, and a hard zero would make vacuum a separate code path.
    """
    Pa = resolve_requirement(_req(thrust_condition="vacuum")).mission.Pa
    assert 0.0 < Pa <= 1.0


def test_thrust_condition_altitude_matches_atmosphere_model():
    from raosim.atmosphere import pressure

    h = 12_000.0
    got = resolve_requirement(_req(thrust_condition=("altitude", h))).mission.Pa
    assert got == pytest.approx(pressure(h), rel=1e-12)


def test_unknown_thrust_condition_rejected():
    with pytest.raises(ValueError, match="thrust_condition"):
        _req(thrust_condition="orbit")


def test_duration_is_two_numbers_and_qualification_is_the_longer():
    """SP-125 §2.1: qualification runs "many times the comparatively short
    rated flight duration", and *those* specs "govern most engine design
    considerations".  A cumulative duration shorter than one flight is a
    misuse of the field, not a valid input."""
    with pytest.raises(ValueError, match="cumulative"):
        _req(flight_duration=30.0, qualification_duration=10.0)


def test_flight_duration_drives_battery_sizing():
    assert resolve_requirement(_req(flight_duration=45.0)
                               ).mission.burn_time == pytest.approx(45.0)


def test_thrust_is_carried_and_scales_the_architecture():
    """MissionSpec.for_propellant derives the architecture from the thrust, so
    the requirement must not be silently applied to 13 kN-class defaults."""
    m = resolve_requirement(_req(thrust=50.0e3)).mission
    assert m.thrust == pytest.approx(50.0e3)
    assert m.propellant_name == "LOX/RP-1"


# --------------------------------------------------------------------------- #
# Coverage classification — the honesty tests                                  #
# --------------------------------------------------------------------------- #
def test_thrust_is_fully_enforced_without_a_constraint_row():
    c = _cov(resolve_requirement(_req()), "thrust")
    assert c.coverage is Coverage.ENFORCED
    assert c.sp125_item == 1


def test_isp_engine_system_basis_is_unsupported():
    """SP-125: "It is important to state whether a specified value of I_s
    refers to the complete engine system, or to the thrust chamber only."
    The solver reports thrust-chamber Isp, so the other basis must not be
    quietly accepted."""
    c = _cov(resolve_requirement(_req(isp_basis="engine_system")), "isp_min")
    assert c.coverage is Coverage.UNSUPPORTED
    assert "thrust-chamber" in (c.reason or "")


def test_isp_thrust_chamber_basis_is_enforced():
    c = _cov(resolve_requirement(_req()), "isp_min")
    assert c.coverage is Coverage.ENFORCED
    assert c.constraint == "isp_epsilon"
    assert resolve_requirement(_req()).isp_floor == pytest.approx(230.0)


def test_qualification_duration_is_unsupported_and_says_why():
    c = _cov(resolve_requirement(_req(qualification_duration=180.0)),
             "qualification_duration")
    assert c.coverage is Coverage.UNSUPPORTED
    assert c.missing        # must name what is missing, not just decline


def test_burnout_mass_is_partial_because_the_screen_is_a_lower_bound():
    c = _cov(resolve_requirement(_req(burnout_mass_max=30.0)),
             "burnout_mass_max")
    assert c.coverage is Coverage.PARTIALLY_ENFORCED
    assert c.constraint == "dry_mass_partial"
    assert not c.satisfied_implies_met
    # The excluded subsystems must be enumerated, matching HARDWARE_MASS_LEDGER.
    assert {"injector hardware", "valves", "lines", "gimbal"} <= set(c.missing)


def test_envelope_is_partial_because_the_flange_is_host_side():
    r = resolve_requirement(_req(envelope_diameter_max=0.30,
                                 envelope_length_max=0.60))
    d = _cov(r, "envelope_diameter_max")
    assert d.coverage is Coverage.PARTIALLY_ENFORCED
    assert d.sp125_item == 6
    assert any("flange" in m for m in d.missing)
    assert _cov(r, "envelope_length_max").coverage is (
        Coverage.PARTIALLY_ENFORCED)


def test_operability_requirements_are_carried_not_dropped():
    r = resolve_requirement(_req(throttle_range=(0.6, 1.0), restarts=2,
                                 reusable_cycles=20))
    for name in ("throttle_range", "restarts", "reusable_cycles"):
        c = _cov(r, name)
        assert c.coverage is Coverage.UNSUPPORTED
        assert c.sp125_item is None     # honestly not on SP-125's §2.1 list
        assert c.reason


def test_of_is_partial_without_cea_surfaces():
    """Without a CEA table gamma/Tc/R are flat in O/F, so the cooling-vs-
    performance mixture-ratio trade SP-125 describes cannot be resolved."""
    c = _cov(resolve_requirement(_req()), "of")
    assert c.coverage is Coverage.PARTIALLY_ENFORCED
    assert "FLAT IN O/F" in (c.reason or "")


def test_of_is_enforced_once_cea_surfaces_are_supplied():
    r = resolve_requirement(
        _req(mission_overrides={"cea_table_path": "builds/fake_table.npz"}))
    assert _cov(r, "of").coverage is Coverage.ENFORCED


def test_pinned_of_is_enforced_but_flagged_as_removing_a_lever():
    c = _cov(resolve_requirement(_req(of=2.4)), "of")
    assert c.coverage is Coverage.ENFORCED
    assert "removes a" in (c.reason or "")
    assert resolve_requirement(_req(of=2.4)).mission.OF == pytest.approx(2.4)


def test_fully_screened_is_false_when_anything_is_partial():
    assert not resolve_requirement(_req(burnout_mass_max=30.0)).fully_screened


def test_partial_and_unsupported_are_separately_addressable():
    r = resolve_requirement(_req(envelope_diameter_max=0.3, restarts=1))
    assert {c.requirement for c in r.partial} >= {"envelope_diameter_max"}
    assert {c.requirement for c in r.unsupported} >= {"restarts"}


def test_report_renders_every_requirement():
    r = resolve_requirement(_req(burnout_mass_max=30.0, restarts=1))
    text = r.report()
    for c in r.coverage:
        assert c.requirement in text


# --------------------------------------------------------------------------- #
# Architectures that would change what is being designed must hard-fail        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch", ["gas_generator", "staged_combustion",
                                  "pressure_fed", "expander"])
def test_unimplemented_feed_architecture_raises_and_names_the_source(arch):
    with pytest.raises(NotImplementedError) as e:
        resolve_requirement(_req(feed_architecture=arch))
    msg = str(e.value)
    assert "SP-8110" in msg and "SP-8081" in msg


def test_unimplemented_objective_raises():
    with pytest.raises(NotImplementedError, match="objective"):
        resolve_requirement(_req(objective="max_isp"))


def test_unknown_propellant_lists_the_available_set():
    with pytest.raises(ValueError) as e:
        resolve_requirement(_req(propellant="LOX/unobtanium"))
    assert "lox/rp-1" in str(e.value)


@pytest.mark.parametrize("kw", [
    {"thrust": 0.0}, {"isp_min": -1.0}, {"flight_duration": 0.0},
    {"burnout_mass_max": 0.0}, {"envelope_diameter_max": -0.1},
    {"throttle_range": (1.2, 0.6)},
])
def test_nonsense_requirements_rejected(kw):
    with pytest.raises(ValueError):
        _req(**kw)


# --------------------------------------------------------------------------- #
# The verdict must not collapse "screened subset feasible" into "met"          #
# --------------------------------------------------------------------------- #
class _FakeNLP:
    def __init__(self, feasible):
        self.feasible = feasible


def test_verdict_is_none_when_feasible_but_only_partially_screened():
    """The property this whole module exists to protect.

    A feasible optimum against a lower-bound screen does NOT prove the
    requirement is met.  Returning ``True`` here would be the "fake zero"
    failure mode in a new costume.
    """
    from raosim.requirements import RequirementResult

    r = RequirementResult(
        resolved=resolve_requirement(_req(burnout_mass_max=30.0)),
        nlp=_FakeNLP(feasible=True))
    assert r.requirements_met is None
    assert r.requirements_met is not True


def test_verdict_is_false_when_infeasible_regardless_of_coverage():
    from raosim.requirements import RequirementResult

    r = RequirementResult(resolved=resolve_requirement(_req()),
                          nlp=_FakeNLP(feasible=False))
    assert r.requirements_met is False


def test_verdict_is_true_only_when_feasible_and_fully_screened():
    from raosim.requirements import RequirementResult

    # Needs CEA surfaces: without them O/F is only partially screened (see
    # test_no_requirement_set_is_fully_screened_without_cea_surfaces), so this
    # is currently the *only* shape of requirement that can reach a plain True.
    resolved = resolve_requirement(
        _req(mission_overrides={"cea_table_path": "builds/fake_table.npz"}))
    assert resolved.fully_screened
    r = RequirementResult(resolved=resolved, nlp=_FakeNLP(feasible=True))
    assert r.requirements_met is True


def test_no_requirement_set_is_fully_screened_without_cea_surfaces():
    """A consequence worth pinning rather than discovering later.

    O/F is on SP-125's §2.1 list, and without a sampled CEA table the property
    surfaces are flat in it — so *every* requirement set is at best partially
    screened until ``scripts/sample_cea_surface.py`` has been run.  That is the
    honest state of the tool, and it should fail loudly if someone weakens the
    O/F coverage rule to make reports look cleaner.
    """
    assert not resolve_requirement(_req()).fully_screened
    assert not resolve_requirement(
        _req(isp_min=None, thrust_condition="vacuum")).fully_screened
