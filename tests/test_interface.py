import math

import numpy as np
import pytest

from raosim.design import (
    DesignInput,
    InterfaceSpec,
    ManufacturingSpec,
    MaterialSpec,
    ThermoSpec,
    design_nozzle_v2,
)
from raosim.interface import (
    resolve_bolted_interface_geometry,
    screen_composite_regen_wall,
    screen_injector_chamber_interface,
    size_bolted_interface,
    validate_bolted_interface_geometry,
)
from raosim.regen_profile import RegenWallProfile


def _gate(ledger, name):
    return next(g for g in ledger.gates if g.name == name)


def test_resolve_bolted_interface_geometry_matches_flange_and_face():
    resolution = resolve_bolted_interface_geometry(
        chamber_pressure=8.0e6,
        chamber_radius=0.035,
        wall_thickness=0.002,
        material_yield_strength=900e6,
        min_tool_diameter=0.0005,
    )

    assert resolution.flange_outer_diameter == pytest.approx(
        resolution.face_outer_diameter
    )
    assert resolution.bolt_count >= 8
    assert resolution.bolt_hole_diameter >= 0.006
    assert resolution.inner_edge_distance >= (
        resolution.edge_distance_requirement - 1e-12
    )
    assert resolution.outer_edge_distance >= (
        resolution.edge_distance_requirement - 1e-12
    )
    assert resolution.bolt_pitch >= resolution.pitch_requirement - 1e-12
    assert resolution.face_thickness >= resolution.bolt_hole_diameter * 2.0
    assert resolution.auto_sized_fields["flange_od"] == "auto_sized"


def test_bolt_pattern_rejects_degenerate_counts_but_allows_three_explicit():
    with pytest.raises(ValueError, match="bolt_count"):
        resolve_bolted_interface_geometry(
            chamber_radius=0.035, bolt_count=2
        )
    triangular = resolve_bolted_interface_geometry(
        chamber_radius=0.035, bolt_count=3
    )
    assert triangular.bolt_count == 3
    automatic = resolve_bolted_interface_geometry(
        chamber_radius=0.035, default_bolt_count=3
    )
    assert automatic.bolt_count >= 4 and automatic.bolt_count % 2 == 0


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan")])
def test_invalid_joint_separation_factor_is_rejected(value):
    with pytest.raises(ValueError, match="joint_separation_factor"):
        resolve_bolted_interface_geometry(
            chamber_radius=0.035, joint_separation_factor=value
        )
    with pytest.raises(ValueError, match="joint_separation_factor"):
        size_bolted_interface(
            chamber_radius=0.035,
            chamber_pressure=3.0e6,
            joint_separation_factor=value,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chamber_pressure", float("nan")),
        ("structural_fos", float("inf")),
        ("edge_distance_factor", 0.0),
        ("pitch_factor", -1.0),
        ("material_yield_strength", float("nan")),
        ("face_thickness", float("inf")),
    ],
)
def test_interface_screen_rejects_nonfinite_or_nonpositive_contract_values(
    name, value
):
    kwargs = {
        "chamber_pressure": 7.0e6,
        "chamber_radius": 0.035,
        name: value,
    }
    with pytest.raises(ValueError, match=name):
        screen_injector_chamber_interface(**kwargs)


def test_interface_screen_rejects_degenerate_explicit_bolt_count():
    with pytest.raises(ValueError, match="bolt_count"):
        screen_injector_chamber_interface(
            chamber_pressure=7.0e6,
            chamber_radius=0.035,
            bolt_count=2,
        )


def test_cad_contract_rejects_a_mutated_degenerate_resolution():
    from dataclasses import replace

    valid = resolve_bolted_interface_geometry(chamber_radius=0.035)
    validate_bolted_interface_geometry(valid)
    with pytest.raises(ValueError, match="bolt_count"):
        validate_bolted_interface_geometry(replace(valid, bolt_count=2))


def test_bolt_pattern_boundary_roundoff_does_not_fail():
    ledger = screen_injector_chamber_interface(
        chamber_pressure=7.0e6,
        chamber_radius=0.033466401061363026,
        wall_thickness=0.002,
        face_outer_diameter=0.1312596721478973,
        face_thickness=0.02050656591523077,
        bolt_count=18,
        bolt_circle_diameter=0.1072596721478973,
        bolt_hole_diameter=0.006,
        material_yield_strength=900e6,
    )

    gate = _gate(ledger, "bolt_pattern_lands")
    assert gate.value < 0.0
    assert gate.value == pytest.approx(0.0, abs=1e-12)
    assert gate.status == "pass"


def test_interface_faceplate_and_bolt_equations():
    ledger = screen_injector_chamber_interface(
        chamber_pressure=7.0e6,
        chamber_radius=0.035,
        wall_thickness=0.003,
        face_outer_diameter=0.13,
        face_thickness=0.002,
        bolt_count=8,
        bolt_circle_diameter=0.10,
        bolt_hole_diameter=0.006,
        bolt_diameter=0.005,
        material_yield_strength=900e6,
        bolt_allowable_stress=600e6,
        structural_fos=1.5,
    )

    assert ledger.separating_force == pytest.approx(
        7.0e6 * math.pi * 0.035**2
    )
    allowable = 900e6 / 1.5
    expected_t = 0.035 * math.sqrt(0.75 * 7.0e6 / allowable)
    assert ledger.face_required_thickness == pytest.approx(expected_t)
    assert _gate(ledger, "injector_faceplate_bending").status == "fail"

    per_bolt = 1.5 * ledger.separating_force / 8
    tensile_area = 0.75 * math.pi * 0.005**2 / 4.0
    assert ledger.bolt_stress == pytest.approx(per_bolt / tensile_area)
    assert _gate(ledger, "bolt_joint_separation").status == "pass"
    assert _gate(ledger, "bolt_pattern_lands").status == "pass"


def test_interface_missing_bolt_data_is_informational():
    ledger = screen_injector_chamber_interface(
        chamber_pressure=7.0e6,
        chamber_radius=0.035,
        wall_thickness=None,
        material_yield_strength=900e6,
    )

    assert _gate(ledger, "chamber_wall_hoop_pressure").status == "info"
    assert _gate(ledger, "bolt_joint_separation").status == "info"
    assert ledger.feasible is True


def test_composite_regen_wall_screen_uses_common_strain():
    contour = {
        "x": np.array([0.0, 0.1]),
        "y": np.array([0.05, 0.05]),
        "Rt": 0.05,
    }
    profile = RegenWallProfile.uniform(
        contour,
        channel_count=80,
        channel_width=0.001,
        channel_height=0.003,
        land_width=0.001,
        t_hot=0.001,
        t_jacket=0.002,
    )
    liner = MaterialSpec(
        name="copper liner",
        yield_strength=200e6,
        conductivity=300.0,
        elastic_modulus=100e9,
        thermal_expansion=17e-6,
        poisson_ratio=0.33,
    )
    jacket = MaterialSpec(
        name="jacket alloy",
        yield_strength=900e6,
        conductivity=15.0,
        elastic_modulus=200e9,
        thermal_expansion=13e-6,
        poisson_ratio=0.29,
    )

    screen = screen_composite_regen_wall(
        chamber_pressure=4.0e6,
        wall_profile=profile,
        liner_material=liner,
        jacket_material=jacket,
        structural_fos=1.0,
        gas_side_wall_temperature=np.array([500.0, 500.0]),
        coolant_side_wall_temperature=np.array([400.0, 400.0]),
        coolant_temperature=np.array([330.0, 330.0]),
        coolant_pressure=np.array([5.0e6, 5.0e6]),
        liner_pressure_differential=np.array([1.0e6, 1.0e6]),
        heat_flux=np.array([1.0e6, 1.0e6]),
    )

    t_cu_eq = 0.001 + 0.5 * 0.003
    dT_cu = 0.5 * (500.0 + 400.0) - 293.15
    dT_j = 0.5 * (400.0 + 330.0) - 293.15
    eps = (
        100e9 * t_cu_eq * 17e-6 * dT_cu
        + 200e9 * 0.002 * 13e-6 * dT_j
    ) / (100e9 * t_cu_eq + 200e9 * 0.002)
    expected_liner_global = 100e9 * (eps - 17e-6 * dT_cu)

    assert screen.t_liner_equivalent_min == pytest.approx(t_cu_eq)
    assert screen.land_fraction_min == pytest.approx(0.5)
    assert screen.liner_global_membrane_stress == pytest.approx(
        expected_liner_global
    )
    assert screen.global_residual_pressure == pytest.approx(0.0)
    assert screen.global_residual_membrane_load == pytest.approx(0.0)
    assert screen.min_margin > 1.0
    assert screen.status == "pass"


def test_interface_replaces_scalar_wall_gate_with_composite_screen():
    contour = {
        "x": np.array([0.0, 0.1]),
        "y": np.array([0.05, 0.05]),
        "Rt": 0.05,
    }
    profile = RegenWallProfile.uniform(
        contour,
        channel_count=80,
        channel_width=0.001,
        channel_height=0.003,
        land_width=0.001,
        t_hot=0.001,
        t_jacket=0.002,
    )
    material = MaterialSpec(
        name="screen alloy",
        yield_strength=900e6,
        conductivity=300.0,
        elastic_modulus=120e9,
        thermal_expansion=16e-6,
        poisson_ratio=0.3,
    )
    composite = screen_composite_regen_wall(
        chamber_pressure=2.0e6,
        wall_profile=profile,
        liner_material=material,
        jacket_material=material,
        structural_fos=1.0,
        screen_station_index=0,
        screen_selection="injector_face_chamber_station",
    )

    ledger = screen_injector_chamber_interface(
        chamber_pressure=2.0e6,
        chamber_radius=0.05,
        wall_thickness=0.001,
        material_yield_strength=900e6,
        composite_wall_screen=composite,
    )
    names = {gate.name for gate in ledger.gates}
    assert "composite_regen_wall_hoop" in names
    assert "chamber_wall_hoop_pressure" not in names
    assert ledger.to_dict()["composite_wall"]["min_margin"] == pytest.approx(
        composite.min_margin
    )
    assert (
        ledger.to_dict()["composite_wall"]["screen_selection"]
        == "injector_face_chamber_station"
    )


def test_v2_report_contains_injector_interface_screen():
    result = design_nozzle_v2(
        DesignInput(
            thermo=ThermoSpec(mode="constant_gamma", propellant_name="LOX/RP-1"),
            Pc=8.0e6,
            Rt=0.020,
            epsilon=8.0,
            material=MaterialSpec(
                name="test alloy",
                yield_strength=900e6,
                elastic_modulus=190e9,
                poisson_ratio=0.29,
            ),
            manufacturing=ManufacturingSpec(wall_thickness=0.003),
            interface=InterfaceSpec(
                flange_od=0.14,
                flange_length=0.020,
                bolt_count=8,
                bolt_circle_diameter=0.11,
                bolt_hole_diameter=0.006,
                bolt_diameter=0.005,
                injector_face_od=0.14,
                injector_face_thickness=0.006,
            ),
        )
    )

    iface = result.report_sections["injector_interface"]
    assert iface["separating_force_n"] > 0.0
    assert iface["face_required_thickness_m"] > 0.0
    assert {g["name"] for g in iface["gates"]} >= {
        "injector_faceplate_bending",
        "bolt_joint_separation",
        "bolt_pattern_lands",
    }
