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
    MixtureRatioMode,
    RequirementAnalysisConfig,
    resolve_requirement,
    solve_requirement,
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


def _write_property_table(path, *, oxidizer="LOX", fuel="RP-1", flat=False):
    from raosim.mdo.properties import save_tables

    Pc = np.linspace(1.5e6, 6.0e6, 5)
    OF = np.linspace(1.8, 3.0, 5)
    P, O = np.meshgrid(Pc, OF, indexing="ij")
    Tc = 3500.0 + 0.0 * P if flat else 3500.0 - 100.0 * O
    save_tables(
        str(path),
        {
            "Pc_grid": Pc,
            "OF_grid": OF,
            "gamma": 1.24 + 0.0 * P,
            "Tc": Tc,
            "R_gas": 380.0 + 0.0 * P,
        },
        oxidizer=oxidizer,
        fuel=fuel,
    )
    return path


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


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"thrust": np.inf}, "thrust"),
        ({"flight_duration": np.inf}, "flight_duration"),
        ({"isp_min": np.nan}, "isp_floor"),
        ({"qualification_duration": np.inf}, "qualification_duration"),
        ({"burnout_mass_max": np.inf}, "mass_max"),
        ({"restarts": -1}, "restart_count"),
        ({"restarts": 1.5}, "restart_count"),
        ({"reusable_cycles": 2.5}, "reusable_cycles"),
        ({"thrust_condition": ("altitude", np.inf)}, "altitude"),
        ({"thrust_condition": ("altitude", np.nan)}, "altitude"),
    ],
)
def test_requirement_api_rejects_nonfinite_and_nonintegral_values(
    overrides, field
):
    with pytest.raises(ValueError, match=field):
        _req(**overrides)


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


def test_nominal_of_is_explicitly_pinned_without_cea_surfaces():
    """NOMINAL is an explicit intent, not an inferred failed optimization."""
    c = _cov(resolve_requirement(_req()), "of")
    assert c.coverage is Coverage.ENFORCED
    assert "NOMINAL intent" in (c.reason or "")
    assert resolve_requirement(_req()).mission.of_is_pinned is True


def test_of_is_enforced_in_explicit_optimize_mode(tmp_path):
    path = _write_property_table(tmp_path / "lox_rp1.npz")
    r = resolve_requirement(_req(
        mixture_ratio_mode=MixtureRatioMode.OPTIMIZE,
        analysis_config=RequirementAnalysisConfig(
            chamber_property_table=path
        ),
    ))
    assert _cov(r, "of").coverage is Coverage.ENFORCED
    assert r.mission.of_is_pinned is False


def test_pinned_of_is_enforced_but_flagged_as_removing_a_lever():
    c = _cov(resolve_requirement(_req(of=2.4)), "of")
    assert c.coverage is Coverage.ENFORCED
    assert "removes a" in (c.reason or "")
    assert resolve_requirement(_req(of=2.4)).mission.OF == pytest.approx(2.4)


def test_pinned_of_survives_cea_surfaces_being_available(tmp_path):
    """A pin must not be silently overridden once a table exists.

    ``of_design_space`` promotes O/F to a design variable as soon as
    ``cea_table_path`` is set; without the pin flag the optimiser would discard
    the value the user explicitly asked for.
    """
    from raosim.mdo.schema import default_design_space

    path = _write_property_table(tmp_path / "lox_rp1.npz")
    r = resolve_requirement(_req(
        mixture_ratio_mode=MixtureRatioMode.PINNED,
        mixture_ratio=2.4,
        analysis_config=RequirementAnalysisConfig(
            chamber_property_table=path
        ),
    ))
    assert r.mission.of_is_pinned is True
    assert "OF" not in [s.name for s in default_design_space(r.mission)]
    assert r.mission.OF == pytest.approx(2.4)


def test_unpinned_of_becomes_a_design_variable_once_surfaces_exist(tmp_path):
    from raosim.mdo.schema import default_design_space

    path = _write_property_table(tmp_path / "t.npz")
    r = resolve_requirement(_req(
        mixture_ratio_mode=MixtureRatioMode.OPTIMIZE,
        analysis_config=RequirementAnalysisConfig(
            chamber_property_table=path
        ),
    ))
    assert r.mission.of_is_pinned is False
    assert "OF" in [s.name for s in default_design_space(r.mission)]
    assert _cov(r, "of").coverage is Coverage.ENFORCED


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
    with pytest.raises(ValueError, match="objective"):
        _req(objective="max_isp")


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


@pytest.mark.parametrize("field", [
    "thrust", "Pa", "burn_time", "OF", "of_is_pinned",
    "dry_mass_max", "envelope_diameter_max", "envelope_length_max",
])
def test_mission_overrides_cannot_replace_requirement_owned_fields(field):
    """Coverage must describe the values actually handed to the solver."""

    with pytest.raises(ValueError, match="mission_overrides cannot change"):
        _req(mission_overrides={field: 1.0})


def test_resolved_constraint_set_exactly_tracks_active_requirement_rows():
    resolved = resolve_requirement(_req(
        burnout_mass_max=40.0,
        envelope_diameter_max=0.5,
        envelope_length_max=1.0,
    ))
    names = set(resolved.required_constraint_names)
    assert {
        "isp_epsilon", "dry_mass_partial", "envelope_diameter",
        "envelope_length",
    } <= names
    # Constant fallback grids are interpolation scaffolding rather than a
    # physical validity claim, so their property-domain row is inapplicable.
    assert "property_domain" not in names


def test_no_isp_floor_removes_only_the_isp_constraint():
    with_floor = resolve_requirement(_req()).required_constraint_names
    without_floor = resolve_requirement(_req(isp_min=None)).required_constraint_names
    assert "isp_epsilon" in with_floor
    assert "isp_epsilon" not in without_floor
    assert set(without_floor) == set(with_floor) - {"isp_epsilon"}


def test_requirement_driver_rejects_constraint_ablation():
    with pytest.raises(TypeError, match="constraint ablation"):
        solve_requirement(_req(), enforced=("isp_epsilon",), maxiter=1)


def test_mixture_ratio_intent_cannot_be_ambiguous():
    with pytest.raises(ValueError, match="only valid in PINNED"):
        _req(mixture_ratio=2.4)
    with pytest.raises(ValueError, match="requires a finite positive"):
        _req(mixture_ratio_mode=MixtureRatioMode.PINNED)
    with pytest.raises(ValueError, match="cannot be combined"):
        _req(mixture_ratio_mode=MixtureRatioMode.OPTIMIZE, of=2.4)


def test_flat_property_table_cannot_enable_of_optimization(tmp_path):
    path = _write_property_table(tmp_path / "flat.npz", flat=True)
    with pytest.raises(ValueError, match="meaningful O/F dependence"):
        resolve_requirement(_req(
            mixture_ratio_mode=MixtureRatioMode.OPTIMIZE,
            analysis_config=RequirementAnalysisConfig(
                chamber_property_table=path
            ),
        ))


def test_wrong_propellant_property_table_is_rejected(tmp_path):
    path = _write_property_table(
        tmp_path / "wrong_pair.npz", oxidizer="LOX", fuel="LCH4"
    )
    with pytest.raises(ValueError, match="propellant identity mismatch"):
        resolve_requirement(_req(
            analysis_config=RequirementAnalysisConfig(
                chamber_property_table=path
            ),
        ))


# --------------------------------------------------------------------------- #
# The verdict must not collapse "screened subset feasible" into "met"          #
# --------------------------------------------------------------------------- #
class _FakeNLP:
    def __init__(self, feasible, *, physics_status=None, success=True):
        self.feasible = feasible
        self.physics_status = physics_status
        self.success = success


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


def test_verdict_is_unknown_when_required_physics_is_unavailable():
    from raosim.requirements import RequirementResult

    r = RequirementResult(
        resolved=resolve_requirement(_req()),
        nlp=_FakeNLP(feasible=False, physics_status="unknown"),
    )
    assert r.requirements_met is None


def test_verdict_is_true_only_when_feasible_and_fully_screened():
    from raosim.requirements import RequirementResult

    # NOMINAL explicitly pins the catalog O/F, so a minimal RP-1 requirement
    # can be fully screened without claiming that an O/F optimization occurred.
    resolved = resolve_requirement(_req())
    assert resolved.fully_screened
    r = RequirementResult(resolved=resolved, nlp=_FakeNLP(feasible=True))
    assert r.requirements_met is True


def test_failed_solver_cannot_issue_requirements_met():
    from raosim.requirements import RequirementResult

    resolved = resolve_requirement(_req())
    assert resolved.fully_screened
    r = RequirementResult(
        resolved=resolved,
        nlp=_FakeNLP(feasible=True, physics_status="pass", success=False),
    )
    assert r.requirements_met is None


def test_optimize_of_requires_a_valid_property_table():
    """A path-free OPTIMIZE request must fail, not degrade to nominal O/F."""

    with pytest.raises(ValueError, match="requires.*chamber_property_table"):
        resolve_requirement(_req(
            mixture_ratio_mode=MixtureRatioMode.OPTIMIZE
        ))
