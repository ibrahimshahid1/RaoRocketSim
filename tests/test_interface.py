import math

import pytest

from raosim.design import (
    DesignInput,
    InterfaceSpec,
    ManufacturingSpec,
    MaterialSpec,
    ThermoSpec,
    design_nozzle_v2,
)
from raosim.interface import screen_injector_chamber_interface


def _gate(ledger, name):
    return next(g for g in ledger.gates if g.name == name)


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

