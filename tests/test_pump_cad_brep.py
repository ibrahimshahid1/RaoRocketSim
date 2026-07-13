"""True B-rep pump CAD (raosim.pump_cad_brep): named assemblies, re-import
validity, and meanline dimension fidelity.  CadQuery-gated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raosim.injector import FeedLineLedger, FeedSystemLedger
from raosim.pump_cad import pump_reference_geometry
from raosim.pump_cad_brep import (
    audit_meanline_geometry_fidelity,
    audit_pump_component_interference,
    audit_pump_clearances,
    audit_volute_flow_passage,
    build_pump_assembly,
    build_pump_parts,
    cadquery_available,
    export_pump_brep_package,
    inspect_pump_step,
)
from raosim.pumps import size_electric_pumps

pytestmark = pytest.mark.skipif(
    not cadquery_available(), reason="CadQuery/OpenCascade not installed")

_GRAVITY = 9.80665
_MM = 1000.0
# OCC bounding boxes carry a small inflation tolerance; solved diameters are
# also exceeded by the chordal corner overshoot of flat swept blade sections
# (~t^2/8r, microns).
_BBOX_TOL_MM = 0.05


def _feed_line(role: str, rho: float, mdot: float,
               tank_pressure: float = 4.0e5) -> FeedLineLedger:
    required = 3.2e6
    Q = mdot / rho
    rise = max(required - tank_pressure, 0.0)
    head = rise / (rho * _GRAVITY)
    return FeedLineLedger(
        role=role,
        chamber_pressure=2.0e6,
        injector_dp=6.0e5,
        manifold_loss=1.0e5,
        manifold_screen_loss=0.0,
        regen_loss=2.0e5 if role == "fuel" else 0.0,
        line_valve_loss=1.5e5,
        control_margin=1.5e5,
        required_outlet_pressure=required,
        available_outlet_pressure=None,
        pressure_margin=None,
        density=rho,
        viscosity=1.7e-3,
        vapor_pressure=2.0e3,
        volumetric_flow=Q,
        required_pressure_rise=rise,
        required_pump_head=head,
        ideal_pump_power=Q * rise / 0.6,
        flow_capacity=None,
        capacity_margin=None,
        npsh_available=tank_pressure - 2.0e3,
        npsh_required=None,
        npsh_margin=None,
        status="info",
    )


@pytest.fixture(scope="module")
def pump():
    ledger = FeedSystemLedger(
        architecture="pump_fed",
        lines={
            "fuel": _feed_line("fuel", 810.0, 0.35),
            "oxidizer": _feed_line("oxidizer", 1141.0, 0.80),
        },
        governing_required_pressure=3.2e6,
        notes=[],
    )
    return size_electric_pumps(ledger)


@pytest.fixture(scope="module")
def package(pump, tmp_path_factory):
    out = tmp_path_factory.mktemp("pump_brep")
    return export_pump_brep_package(pump, out)


def test_every_step_reimports_valid(package):
    step_keys = [k for k in package["files"] if k.endswith("_step")]
    assert len(step_keys) >= 17, sorted(package["files"])
    for key in step_keys:
        info = package["diagnostics"][key]
        assert info["valid"], (key, info)
        assert info["volume_mm3"] > 0.0, (key, info)
        assert info["representation"] == "open_cascade_brep"
        # Per-part bodies are single solids; assemblies carry one per part.
        for key, info in package["diagnostics"].items():
            if key.endswith("_step") and not key.endswith("_assembly_step"):
                assert info["solid_count"] == 1, (key, info)


def test_expected_parts_and_stls_present(package):
    names = {Path(p).name for p in package["files"].values()}
    for role in ("fuel", "oxidizer"):
        for part in ("impeller", "inducer", "diffuser_ring",
                     "volute_body", "volute_front_cover", "shaft", "motor"):
            assert f"{role}_{part}.step" in names
            assert f"{role}_{part}.stl" in names
        assert f"{role}_pump.step" in names
    assert "shared_battery_pack.step" in names


def test_named_assembly_children(pump):
    assembly = build_pump_assembly(pump, "fuel")
    assert assembly.name == "fuel_pump"
    child_names = {child.name for child in assembly.children}
    assert {"impeller", "inducer", "diffuser_ring", "volute_body",
            "volute_front_cover", "shaft"} <= child_names


def test_key_dimensions_match_meanline(pump, package):
    geom = pump_reference_geometry(pump)
    for role, line in pump.lines.items():
        ref = line.reference_geometry
        comp = geom["components"][role]

        # Manifest values equal the meanline reference geometry exactly.
        assert (comp["impeller"]["outer_diameter_m"]
                == ref.impeller_disk["outer_diameter_m"])
        assert (comp["impeller"]["outlet_width_m"]
                == ref.impeller_disk["outlet_width_m"])
        assert (comp["inducer"]["diameter_m"]
                == ref.inducer_helix["diameter_m"])
        assert comp["inducer"]["pitch_m"] == ref.inducer_helix["pitch_m"]

        # Solid-level: the exported bodies measure the solved diameters.
        d2_mm = _MM * ref.impeller_disk["outer_diameter_m"]
        info = package["diagnostics"][f"{role}_impeller_step"]
        assert info["bounding_box_mm"]["x"] == pytest.approx(
            d2_mm, abs=_BBOX_TOL_MM)
        assert info["bounding_box_mm"]["y"] == pytest.approx(
            d2_mm, abs=_BBOX_TOL_MM)

        d_ind_mm = _MM * ref.inducer_helix["diameter_m"]
        length_mm = _MM * (
            ref.inducer_helix["pitch_m"]
            * ref.inducer_helix["wrap_angle_deg"] / 360.0
        )
        info = package["diagnostics"][f"{role}_inducer_step"]
        assert info["bounding_box_mm"]["x"] == pytest.approx(
            d_ind_mm, abs=_BBOX_TOL_MM)
        assert info["bounding_box_mm"]["z"] == pytest.approx(
            length_mm, abs=_BBOX_TOL_MM)


def test_bore_fit_is_resolved_upstream_without_cad_mutation(pump):
    parts, notes = build_pump_parts(pump, "fuel")
    assert not any("hub radius increased" in note for note in notes)
    assert not any("NPSH revalidation required" in note for note in notes)
    assert not any("shaft bore skipped" in note for note in notes)
    assert any("downstream splitters" in note for note in notes)
    for name, solid in parts.items():
        assert solid.isValid(), name
        assert len(solid.Solids()) == 1, name
    audit = audit_pump_component_interference(parts)
    assert audit["passed"]
    assert audit["maximum_overlap_mm3"] <= audit["tolerance_mm3"]
    comp = pump_reference_geometry(pump)["components"]["fuel"]
    clearance = audit_pump_clearances(parts, comp)
    assert clearance["passed"]
    assert min(clearance["clearances"].values()) > 0.0
    assert min(clearance["axial_shaft_engagements"].values()) > 0.0


def test_inducer_uses_solved_leading_edge_thickness(pump):
    geom = pump_reference_geometry(pump)
    for role, line in pump.lines.items():
        solved = line.reference_geometry.inducer_helix[
            "leading_edge_thickness_m"
        ]
        assert solved > 0.0
        assert geom["components"][role]["inducer"][
            "leading_edge_thickness_m"
        ] == pytest.approx(solved)


def test_casing_void_is_connected_from_inlet_to_outlet(pump):
    import cadquery as cq

    geom = pump_reference_geometry(pump)
    for role, comp in geom["components"].items():
        gate = audit_volute_flow_passage(
            cq, comp["diffuser_volute"], comp["ports"], comp["shaft"]
        )
        assert gate["passed"], (role, gate)
        assert gate["single_connected_solid"]
        assert min(gate["handoff_overlaps"].values()) > 1.0e-6


def test_package_records_flow_and_interference_gates(package):
    for role in ("fuel", "oxidizer"):
        assert package["assembly_gates"][
            f"{role}_component_interference"
        ]["passed"]
        assert package["assembly_gates"][
            f"{role}_casing_flow_passage"
        ]["passed"]
        assert package["assembly_gates"][
            f"{role}_nominal_clearances"
        ]["passed"]
        fidelity = package["assembly_gates"][
            f"{role}_meanline_geometry_fidelity"
        ]
        assert fidelity["passed"]
        assert fidelity["status"] == "pass"
        assert not fidelity["deviations"]
        split = package["assembly_gates"][
            f"{role}_split_casing_manufacturability"
        ]
        assert split["passed"]
        assert split["material_overlap_mm3"] <= 1e-6
        assert split["relative_volume_closure_error"] <= 1e-6
        assert split["scroll_tool_clearance_mm"] >= 0.0
    assert package["cold_flow_release_ready"] is False
    assert package["hardware_qualified"] is False
    assert package["external_release_blockers"]


def test_meanline_fidelity_gate_is_identity_after_coupled_solve(pump):
    geom = pump_reference_geometry(pump)
    for comp in geom["components"].values():
        gate = audit_meanline_geometry_fidelity(comp)
        assert gate["status"] == "pass"
        assert gate["passed"]
        assert not gate["deviations"]
        for feature in gate["all_features"]:
            assert feature["hub_radius_increase_m"] == pytest.approx(0.0)
            assert feature["inlet_area_reduction_m2"] == pytest.approx(0.0)


def test_inspection_bbox_unions_all_imported_solids(tmp_path):
    import cadquery as cq

    assembly = cq.Assembly(name="bbox_probe")
    assembly.add(cq.Workplane("XY").box(2.0, 2.0, 2.0), name="left")
    assembly.add(
        cq.Workplane("XY").box(2.0, 2.0, 2.0).translate((100.0, 0.0, 0.0)),
        name="right",
    )
    path = tmp_path / "bbox_probe.step"
    assembly.export(str(path))
    info = inspect_pump_step(path)
    assert info["solid_count"] == 2
    assert info["bounding_box_mm"]["x"] == pytest.approx(102.0, abs=0.01)


def test_reimport_gate_rejects_missing_file(tmp_path):
    with pytest.raises(Exception):
        inspect_pump_step(tmp_path / "missing.step")


def test_engine_assembly_places_pump_packages(pump, package, tmp_path):
    """Phase 3: engine-level assembly aggregates the gated pump STEPs."""
    from raosim.engine_cad import export_engine_assembly

    brep_dir = Path(package["dir"])
    info = export_engine_assembly(
        tmp_path / "engine_assembly.step",
        {
            "fuel_pump": brep_dir / "fuel_pump.step",
            "oxidizer_pump": brep_dir / "oxidizer_pump.step",
            "shared_battery_pack": brep_dir / "shared_battery_pack.step",
            "wall": tmp_path / "not_there.step",
        },
        pump_result=pump,
    )
    assert set(info["children"]) == {
        "fuel_pump", "oxidizer_pump", "shared_battery_pack"}
    assert info["diagnostics"]["valid"]
    assert info["diagnostics"]["volume_mm3"] > 0.0
    assert any("wall skipped" in n for n in info["notes"])
    assert info["assembly_gates"]["component_interference"]["passed"]
    assert info["hardware_qualified"] is False
    assert Path(info["unit_sidecar"]).exists()
    assert info["units"]["neutral_file_linear_unit"] == "mm"
    boxes = info["component_bounding_boxes_mm"]
    clearance = info["placement_clearance_mm"]
    assert boxes["shared_battery_pack"]["ymax"] <= (
        min(boxes["fuel_pump"]["ymin"], boxes["oxidizer_pump"]["ymin"])
        - clearance + 0.01
    )
    screens = info["pump_mount_flange_screen"]
    assert set(screens) == {"fuel", "oxidizer"}
