import sys
import types

import pytest

from raosim.design import (
    CoolingSpec,
    DesignInput,
    InterfaceSpec,
    ManufacturingSpec,
    MaterialSpec,
    MissionAmbientSpec,
    ThermoSpec,
    design_nozzle_v2,
)
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, boundary_layer_displacement, regenerative_cooling_screen, structural_screen
from raosim.propellants import get_propellant


def _prelim_input(**kwargs):
    base = DesignInput(
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
        ),
        Pc=80e5,
        Rt=0.020,
        epsilon=8.0,
        mode="preliminary",
        method="bezier",
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def test_preliminary_v2_constant_gamma_runs():
    result = design_nozzle_v2(_prelim_input())

    assert result.design_status == "preliminary_top_geometry"
    assert result.report_sections["thermochemistry"]["source"] == "built_in_constant_gamma"
    assert result.report_sections["boundary_layer"]["effective_epsilon"] < result.input.epsilon
    assert result.contour["hardware_qualified"] is False


def test_validated_requires_cea():
    request = _prelim_input(mode="validated")

    with pytest.raises(RuntimeError, match="requires CEA"):
        design_nozzle_v2(request)


def test_experimental_methods_block_validated_and_manufacturing():
    validated = _prelim_input(mode="validated", method="moc")
    with pytest.raises(ValueError, match="bezier"):
        design_nozzle_v2(validated)

    manufacturing = _prelim_input(
        method="rao",
        manufacturing=ManufacturingSpec(wall_thickness=0.002, cad="step"),
    )
    with pytest.raises(ValueError, match="Manufacturing CAD"):
        design_nozzle_v2(manufacturing)


def test_target_thrust_loses_to_explicit_rt():
    request = _prelim_input(target_thrust=5000.0)
    result = design_nozzle_v2(request)

    assert result.input.Rt == pytest.approx(0.020)
    assert any("explicit Rt is used" in warning for warning in result.warnings)


def test_schema_rejects_bad_regen_dimensions():
    request = _prelim_input(
        cooling=CoolingSpec(method="regenerative", channel_count=12)
    )

    with pytest.raises(ValueError, match="channel_width"):
        design_nozzle_v2(request)


def test_v2_step_writes_metadata_and_ipt_is_deferred(tmp_path):
    request = _prelim_input(
        manufacturing=ManufacturingSpec(
            wall_thickness=0.002,
            cad="step",
            output_dir=tmp_path,
        ),
        interface=InterfaceSpec(flange_od=0.20, flange_length=0.010),
    )

    result = design_nozzle_v2(request)

    assert result.files["step"].exists()
    assert result.files["design_report"].exists()
    report = result.files["design_report"].read_text(encoding="utf-8")
    assert '"authoritative_cad": "STEP"' in report
    assert '"native_ipt": "deferred"' in report

    ipt_request = _prelim_input(
        manufacturing=ManufacturingSpec(wall_thickness=0.002, cad="ipt")
    )
    with pytest.raises(ValueError, match="IPT"):
        design_nozzle_v2(ipt_request)


def test_mocked_rocketcea_feeds_validated_thermochemistry(monkeypatch):
    class FakeCEA:
        def __init__(self, **_kwargs):
            pass

        def get_Chamber_MolWt_gamma(self, Pc, MR):
            return 22.0, 1.21

        def get_Tcomb(self, Pc, MR):
            return 3600.0

        def get_Cstar(self, Pc, MR):
            return 1750.0

    rocketcea_mod = types.ModuleType("rocketcea")
    cea_units_mod = types.ModuleType("rocketcea.cea_obj_w_units")
    cea_units_mod.CEA_Obj = FakeCEA
    monkeypatch.setitem(sys.modules, "rocketcea", rocketcea_mod)
    monkeypatch.setitem(sys.modules, "rocketcea.cea_obj_w_units", cea_units_mod)

    request = _prelim_input(
        mode="validated",
        thermo=ThermoSpec(
            mode="cea_frozen",
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
        ),
        Pc=200e5,
        epsilon=4.0,
        manufacturing=ManufacturingSpec(wall_thickness=0.004, cad="none"),
        cooling=CoolingSpec(
            method="regenerative",
            channel_count=120,
            channel_width=0.0015,
            channel_height=0.003,
            coolant_mass_flow=2.0,
            max_wall_temperature=2500.0,
        ),
        material=MaterialSpec(
            yield_strength=2.0e9,
            max_temperature=3000.0,
            max_heat_flux=100e6,
            conductivity=20.0,
        ),
        ambient=MissionAmbientSpec(Pa=5_000.0),
    )

    result = design_nozzle_v2(request)

    assert result.validated is True
    assert result.propellant.gamma == pytest.approx(1.21)
    assert result.propellant.c_star == pytest.approx(1750.0)
    assert result.report_sections["thermochemistry"]["source"] == "rocketcea"


def test_physics_trends_are_sensible():
    contour = bell_nozzle_contour(0.020, 8.0, length_pct=80.0)
    prop = get_propellant("LOX/RP-1")
    heat = bartz_heat_flux(contour, 80e5, prop)
    throat_x = contour["x"][abs(contour["y"] - contour["Rt"]).argmin()]

    assert abs(heat["x_q_max"] - throat_x) < 0.010

    bl = boundary_layer_displacement(contour, 80e5, prop)
    assert bl["effective_epsilon"] < contour["epsilon"]

    material = MaterialSpec()
    low_area = regenerative_cooling_screen(
        heat, contour,
        CoolingSpec(method="regenerative", channel_count=20, channel_width=0.0005,
                    channel_height=0.001, coolant_mass_flow=0.5),
        material, 0.002,
    )
    high_area = regenerative_cooling_screen(
        heat, contour,
        CoolingSpec(method="regenerative", channel_count=120, channel_width=0.0015,
                    channel_height=0.003, coolant_mass_flow=0.5),
        material, 0.002,
    )
    assert high_area["estimated_wall_temperature"] < low_area["estimated_wall_temperature"]

    low_pressure = structural_screen(contour, 40e5, 101325, prop, material, 0.002, heat, high_area)
    high_pressure = structural_screen(contour, 120e5, 101325, prop, material, 0.002, heat, high_area)
    assert high_pressure["stress_margin"] < low_pressure["stress_margin"]
