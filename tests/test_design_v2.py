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
from raosim.coolants import canonical_coolant_name
from raosim.physics import (
    resolve_coolant_inlet_temperature,
    resolve_coolant_properties,
)
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


def test_cooling_temperature_is_optional_and_resolved_centrally():
    methane = CoolingSpec(coolant="methane")
    hydrogen = CoolingSpec(coolant="lh2")
    explicit = CoolingSpec(coolant="methane", coolant_inlet_temperature=135.0)

    assert methane.coolant_inlet_temperature is None
    assert resolve_coolant_inlet_temperature(methane) == 120.0
    assert resolve_coolant_inlet_temperature(hydrogen) == 25.0
    assert resolve_coolant_inlet_temperature(explicit) == 135.0


@pytest.mark.parametrize("alias", ["methane", "ch4", "lch4"])
def test_methane_aliases_share_temperature_and_properties(alias):
    cooling = CoolingSpec(coolant=alias)
    properties = resolve_coolant_properties(cooling)
    assert canonical_coolant_name(alias) == "methane"
    assert resolve_coolant_inlet_temperature(cooling) == 120.0
    assert properties["rho"] == pytest.approx(423.0)
    assert properties["cp"] == pytest.approx(3450.0)


@pytest.mark.parametrize("alias", ["lh2", "hydrogen", "h2"])
def test_hydrogen_aliases_share_temperature_and_properties(alias):
    cooling = CoolingSpec(coolant=alias)
    properties = resolve_coolant_properties(cooling)
    assert canonical_coolant_name(alias) == "hydrogen"
    assert resolve_coolant_inlet_temperature(cooling) == 25.0
    assert properties["rho"] == pytest.approx(71.0)
    assert properties["cp"] == pytest.approx(9800.0)


def test_v2_backend_uses_central_methane_temperature_default():
    result = design_nozzle_v2(_prelim_input(
        cooling=CoolingSpec(
            method="regenerative",
            coolant="methane",
            channel_count=40,
            channel_width=0.0008,
            channel_height=0.0025,
            coolant_mass_flow=10.0,
            coolant_property_backend="constant",
        ),
        manufacturing=ManufacturingSpec(wall_thickness=0.001),
    ))

    cooling = result.report_sections["cooling"]
    assert cooling["coolant_inlet_temperature"] == 120.0
    assert cooling["coolant_inlet_temperature_source"] == (
        "central_coolant_default"
    )


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
    assert '"cad_body_scope": "single_revolved_uniform_wall_body"' in report
    assert result.files["step"].name == "thrust_chamber_wall.step"

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

    # A physically survivable regen-cooled copper engine.  Both the
    # throat heat flux (full Bartz, ~58 MW/m² at 70 bar) and the cooling
    # (coupled Sieder-Tate / 1-D conduction) are now real, so the
    # fixture must be a real high-performance regen wall: thin copper,
    # small fast channels (high coolant velocity -> high h_c).  Probed
    # margins: peak gas-side wall ~1020 K, cooling margin ~1.1.  The
    # pre-2026-06 fixture (20 MPa through a 4 mm k=20 wall) only passed
    # because the old screening under-predicted flux ~1000× and the old
    # cooling film model was ad-hoc.  This test checks CEA
    # thermochemistry feed-through, not the thermal margins themselves.
    request = _prelim_input(
        mode="validated",
        thermo=ThermoSpec(
            mode="cea_frozen",
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
        ),
        Pc=70e5,
        epsilon=4.0,
        manufacturing=ManufacturingSpec(wall_thickness=0.001, cad="none"),
        cooling=CoolingSpec(
            method="regenerative",
            coolant="rp1",
            channel_count=100,           # fits the 20 mm throat (60 of 126 mm)
            channel_width=0.0006,
            channel_height=0.0030,
            coolant_mass_flow=12.0,
            coolant_inlet_temperature=300.0,
            max_wall_temperature=1200.0,
        ),
        material=MaterialSpec(
            yield_strength=2.0e9,
            max_temperature=1300.0,
            max_heat_flux=120e6,
            conductivity=350.0,
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

    # Real coupled Sieder-Tate solve (gas side = full Bartz): more
    # coolant flow MUST lower the wall temperature (more cooling
    # capacity + faster coolant -> higher h_c).  A thin copper wall and
    # adequate flow keep the temperatures physical.
    material = MaterialSpec(conductivity=350.0, max_temperature=1300.0)

    def _cool(mdot):
        return regenerative_cooling_screen(
            heat, contour,
            CoolingSpec(method="regenerative", coolant="rp1",
                        channel_count=100, channel_width=0.0008,
                        channel_height=0.0025, coolant_mass_flow=mdot,
                        max_wall_temperature=1100.0),
            material, 0.001, prop, 80e5,
        )

    low_flow = _cool(8.0)
    high_flow = _cool(20.0)
    assert high_flow["model"] == "sieder_tate_1d_regen"
    assert (high_flow["estimated_wall_temperature"]
            < low_flow["estimated_wall_temperature"])
    # Coolant heats up along the channel (energy balance).
    assert high_flow["coolant_outlet_temperature"] > 293.0

    low_pressure = structural_screen(contour, 40e5, 101325, prop, material, 0.002, heat, high_flow)
    high_pressure = structural_screen(contour, 120e5, 101325, prop, material, 0.002, heat, high_flow)
    assert high_pressure["stress_margin"] < low_pressure["stress_margin"]
