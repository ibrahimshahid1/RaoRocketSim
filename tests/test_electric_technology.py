"""One electric-drive technology set for both pipelines (audit §1.3 item 19).

The traditional ``raosim.pumps`` defaults, the MDO ``MissionSpec`` electric
fields and the MDO→traditional handoff must all read one registered set, and a
mission may never carry numbers its declared set does not hold.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from raosim.electric_technology import (
    CUSTOM_ELECTRIC_TECHNOLOGY_ID,
    DEFAULT_ELECTRIC_TECHNOLOGY,
    DEFAULT_ELECTRIC_TECHNOLOGY_ID,
    ELECTRIC_TECHNOLOGY_SETS,
    LEE2021_IJASS_ELECTRIC_PUMP,
    MISSION_FIELD_MAP,
    REPOSITORY_SCREENING_V1,
    SCREENING_DEFAULT_KEYS,
    ElectricTechnologySet,
    get_electric_technology,
    identify_electric_technology,
)
from raosim.pumps import SCREENING_DEFAULTS, BatterySpec, ElectricDriveSpec


def test_default_set_is_the_former_traditional_screening_values():
    # The values raosim.pumps used before the registry existed.
    assert DEFAULT_ELECTRIC_TECHNOLOGY_ID == "repository_screening_v1"
    assert REPOSITORY_SCREENING_V1.screening_defaults() == {
        "motor_efficiency": 0.90,
        "inverter_efficiency": 0.96,
        "motor_power_density": 2500.0,
        "inverter_power_density": 15000.0,
        "battery_energy_density": 250.0 * 3600.0,
        "battery_power_density": 3000.0,
        "battery_discharge_efficiency": 0.95,
        "battery_structural_margin": 1.20,
    }
    assert REPOSITORY_SCREENING_V1.basis_class == "repository_screening_assumption"


def test_traditional_defaults_come_from_the_default_set():
    for key in SCREENING_DEFAULT_KEYS:
        assert SCREENING_DEFAULTS[key] == getattr(DEFAULT_ELECTRIC_TECHNOLOGY, key)
    drive, battery = ElectricDriveSpec(), BatterySpec()
    assert drive.motor_power_density == DEFAULT_ELECTRIC_TECHNOLOGY.motor_power_density
    assert drive.inverter_efficiency == DEFAULT_ELECTRIC_TECHNOLOGY.inverter_efficiency
    assert battery.power_density == DEFAULT_ELECTRIC_TECHNOLOGY.battery_power_density
    assert battery.discharge_efficiency == (
        DEFAULT_ELECTRIC_TECHNOLOGY.battery_discharge_efficiency
    )


def test_mission_defaults_come_from_the_same_set():
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()
    assert mission.electric_technology == DEFAULT_ELECTRIC_TECHNOLOGY_ID
    for mission_name, set_name in MISSION_FIELD_MAP:
        assert getattr(mission, mission_name) == getattr(
            DEFAULT_ELECTRIC_TECHNOLOGY, set_name
        )
    # Thrust-scaled and propellant-scaled missions keep the declared set.
    assert MissionSpec.for_thrust(5.0e3).electric_technology == DEFAULT_ELECTRIC_TECHNOLOGY_ID
    scaled = MissionSpec.for_propellant("LOX/RP-1", 20.0e3)
    assert scaled.battery_power_density == DEFAULT_ELECTRIC_TECHNOLOGY.battery_power_density


def test_mission_refuses_numbers_its_declared_set_does_not_hold():
    from raosim.mdo.schema import MissionSpec

    with pytest.raises(ValueError, match="disagree with electric_technology"):
        replace(MissionSpec(), battery_power_density=1.3e3)
    with pytest.raises(ValueError, match="unknown electric technology"):
        MissionSpec(electric_technology="no_such_set")
    custom = replace(MissionSpec(), battery_power_density=1.3e3,
                     electric_technology=CUSTOM_ELECTRIC_TECHNOLOGY_ID)
    assert custom.battery_power_density == 1.3e3
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        replace(MissionSpec(), eta_inverter=1.5,
                electric_technology=CUSTOM_ELECTRIC_TECHNOLOGY_ID)


def test_with_electric_technology_replaces_every_field_atomically():
    from raosim.mdo.schema import MissionSpec

    lee = MissionSpec().with_electric_technology("lee2021_ijass_electric_pump")
    assert lee.electric_technology == "lee2021_ijass_electric_pump"
    assert {m: getattr(lee, m) for m, _ in MISSION_FIELD_MAP} == (
        LEE2021_IJASS_ELECTRIC_PUMP.mission_fields()
    )
    with pytest.raises(ValueError):
        MissionSpec().with_electric_technology(CUSTOM_ELECTRIC_TECHNOLOGY_ID)


def test_lee_set_is_page_located_and_closes_lee_table4():
    lee = LEE2021_IJASS_ELECTRIC_PUMP
    assert lee.basis_class == "literature"
    assert {s.pdf_page for s in lee.sources} == {7, 8, 10, 11}
    assert all(s.pdf_path.endswith("10.1007_s42405-020-00325-z.pdf")
               for s in lee.sources)
    assert lee.discharge_efficiency_burn_time_s == 600.0
    # Lee Table 4 (PDF p.12): motor 451.2 g sets P_m,out; the battery is
    # power-limited at 985.6 g (Eqs. 27, 29, 30 with these Table 2 values).
    shaft = 0.4512 * lee.motor_power_density
    battery_power = shaft / lee.motor_efficiency / lee.inverter_efficiency
    battery_mass = (lee.battery_structural_margin * battery_power
                    / lee.battery_power_density)
    assert battery_mass == pytest.approx(0.9856, abs=5e-4)


def test_screening_set_records_where_it_sits_in_the_literature():
    positions = REPOSITORY_SCREENING_V1.bracket_positions()
    assert set(positions) == set(SCREENING_DEFAULT_KEYS)
    assert positions["inverter_power_density"] == "below"
    assert positions["inverter_efficiency"] == "above"
    assert positions["battery_power_density"] == "inside"
    assert positions["battery_structural_margin"] == "inside"


def test_identify_and_registry_contract():
    assert identify_electric_technology(SCREENING_DEFAULTS) == "repository_screening_v1"
    assert identify_electric_technology(
        LEE2021_IJASS_ELECTRIC_PUMP.screening_defaults()
    ) == "lee2021_ijass_electric_pump"
    assert identify_electric_technology(
        {**SCREENING_DEFAULTS, "battery_power_density": 1.0}
    ) == CUSTOM_ELECTRIC_TECHNOLOGY_ID
    assert set(ELECTRIC_TECHNOLOGY_SETS) == {
        "repository_screening_v1", "lee2021_ijass_electric_pump",
    }
    with pytest.raises(ValueError):
        get_electric_technology("custom")
    with pytest.raises(ValueError, match="cite its source"):
        replace(LEE2021_IJASS_ELECTRIC_PUMP, id="uncited", sources=())
    assert isinstance(DEFAULT_ELECTRIC_TECHNOLOGY, ElectricTechnologySet)


def test_mdo_to_traditional_handoff_passes_the_mission_set():
    from raosim.mdo.postprocess import _pump_spec
    from raosim.mdo.schema import MissionSpec

    for technology_id in ELECTRIC_TECHNOLOGY_SETS:
        mission = MissionSpec().with_electric_technology(technology_id)
        spec = _pump_spec({"N_rpm": 30000.0}, mission, None)
        technology = get_electric_technology(technology_id)
        values = {
            "motor_efficiency": spec.drive.motor_efficiency,
            "inverter_efficiency": spec.drive.inverter_efficiency,
            "motor_power_density": spec.drive.motor_power_density,
            "inverter_power_density": spec.drive.inverter_power_density,
            "battery_energy_density": spec.battery.energy_density,
            "battery_power_density": spec.battery.power_density,
            "battery_discharge_efficiency": spec.battery.discharge_efficiency,
            "battery_structural_margin": spec.battery.structural_margin,
        }
        assert values == technology.screening_defaults()
        assert identify_electric_technology(values) == technology_id
        assert spec.burn_time == mission.burn_time


def test_resolved_inputs_record_the_technology():
    import dataclasses

    from raosim.mdo.resolved_inputs import FeedInputs

    default = next(f.default for f in dataclasses.fields(FeedInputs)
                   if f.name == "electric_technology")
    assert default == DEFAULT_ELECTRIC_TECHNOLOGY_ID


def test_cli_technology_reaches_both_pipelines():
    from raosim import run_nozzle

    args = SimpleNamespace(electric_technology="lee2021_ijass_electric_pump",
                           **{name: None for name in run_nozzle._ELECTRIC_FLAG_FIELDS})
    run_nozzle._apply_electric_technology(args)
    assert args.battery_power_density == 650.0
    assert args._electric_technology_id == "lee2021_ijass_electric_pump"
    overrides = run_nozzle._mdo_electric_overrides(args)
    assert overrides["electric_technology"] == "lee2021_ijass_electric_pump"
    assert overrides["battery_power_density"] == 650.0

    explicit = SimpleNamespace(electric_technology=None,
                               **{name: None for name in run_nozzle._ELECTRIC_FLAG_FIELDS})
    explicit.battery_power_density = 2000.0
    run_nozzle._apply_electric_technology(explicit)
    assert explicit.motor_power_density == DEFAULT_ELECTRIC_TECHNOLOGY.motor_power_density
    assert explicit._electric_technology_id == CUSTOM_ELECTRIC_TECHNOLOGY_ID
    custom = run_nozzle._mdo_electric_overrides(explicit)
    assert custom["electric_technology"] == CUSTOM_ELECTRIC_TECHNOLOGY_ID
    assert custom["battery_power_density"] == 2000.0

    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec(**custom)
    assert mission.battery_power_density == 2000.0


def test_lee_set_warns_outside_its_stated_burn_time():
    import warnings

    from raosim.electric_technology import ElectricTechnologyValidityWarning
    from raosim.mdo.schema import MissionSpec

    with pytest.warns(ElectricTechnologyValidityWarning, match="600 s burn"):
        MissionSpec().with_electric_technology("lee2021_ijass_electric_pump")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ElectricTechnologyValidityWarning)
        replace(MissionSpec(), burn_time=600.0).with_electric_technology(
            "lee2021_ijass_electric_pump"
        )
        MissionSpec()   # the screening set states no burn-time point
