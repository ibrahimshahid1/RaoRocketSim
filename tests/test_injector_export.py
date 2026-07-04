"""Tests for raosim.injector_export - pintle reference geometry + output package."""

import json
import os

import pytest

from raosim.injector import (
    InjectorSpec,
    PintleMechanicalSpec,
    PintleGeometrySpec,
    PropellantFeedSpec,
    evaluate_pintle_injector,
)
from raosim.injector_export import (
    export_pintle_package,
    pintle_reference_geometry,
)

PC, MR = 7.0e6, 2.6
MDOT = 1.8116689741766580
MDOT_F = MDOT / (1.0 + MR)
MDOT_O = MR * MDOT_F


def _inj():
    # Explicit property overrides so the test needs no CoolProp.
    fuel = PropellantFeedSpec(role="fuel", name="rp-1", inlet_temperature=298.0,
                              density=810.0, viscosity=1.6e-3,
                              surface_tension=2.3e-2, vapor_pressure=2.0e3)
    ox = PropellantFeedSpec(role="oxidizer", name="lox", inlet_temperature=90.0,
                            density=1140.0, viscosity=1.9e-4,
                            surface_tension=1.3e-2, vapor_pressure=1.0e5)
    spec = InjectorSpec(
        type="pintle", sizing="auto", fuel=fuel, oxidizer=ox,
        geometry=PintleGeometrySpec(pintle_diameter=0.02, slot_count=24,
                                    radial_stream="fuel", deflector_angle=15.0,
                                    face_od=0.10))
    inj = evaluate_pintle_injector(
        spec, mdot_fuel=MDOT_F, mdot_oxidizer=MDOT_O, Pc=PC, mixture_ratio=MR,
        chamber_radius=0.035, chamber_length=0.13, gamma=1.2, Tc=3500.0,
        R_gas=350.0, fuel_name="rp-1", oxidizer_name="lox")
    return inj, spec


class TestReferenceGeometry:
    def test_groups_and_symbols(self):
        inj, spec = _inj()
        geom = pintle_reference_geometry(inj, spec=spec)
        syms = {d["symbol"] for d in geom["dimensions"]}
        # the core labeled set the CLI deliverable promises
        for s in ("D_pr", "D_ann_o", "h_ann", "D_ob", "N_slot", "w_slot",
                  "h_slot", "BF", "theta_s", "x_wall", "D_c", "L_c",
                  "L_open", "D_cg"):
            assert s in syms
        assert geom["architecture"] == "fixed_annulus_radial_slots"

    def test_movable_dims_marked_not_applicable(self):
        inj, spec = _inj()
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {d["symbol"]: d for d in geom["dimensions"]}
        assert by["L_open"]["kind"] == "not_applicable"
        assert by["D_cg"]["kind"] == "not_applicable"
        assert by["L_open"]["value_si"] is None

    def test_solved_values_match_result(self):
        inj, spec = _inj()
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {d["symbol"]: d for d in geom["dimensions"]}
        assert by["D_pr"]["value_si"] == pytest.approx(inj.pintle_diameter)
        assert by["D_ann_o"]["value_si"] == pytest.approx(
            inj.annulus.detail["outer_diameter"])
        assert by["BF"]["value_si"] == pytest.approx(inj.blockage_factor)
        # mm convenience field is consistent with SI
        assert by["D_pr"]["value_mm"] == pytest.approx(
            inj.pintle_diameter * 1e3)

    def test_construction_values_flagged_schematic(self):
        inj, spec = _inj()
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {d["symbol"]: d for d in geom["dimensions"]}
        assert by["t_wall"]["kind"] == "schematic"
        assert by["D_ob"]["kind"] == "schematic"

    def test_nonfinite_values_export_as_json_null(self):
        inj, spec = _inj()
        inj.spray_wall_axial_distance = float("inf")
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {d["symbol"]: d for d in geom["dimensions"]}
        assert by["x_wall"]["value_si"] is None
        assert "Infinity" not in json.dumps(geom)


class TestPackage:
    def test_mandatory_files_written(self, tmp_path):
        inj, spec = _inj()
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="none")
        for f in ("pintle_parameters.json", "pintle_dimensions.csv",
                  "pintle_schematic.svg", "pintle_cross_section.png"):
            assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0
        # parameters JSON round-trips and carries the dimension list
        data = json.loads((tmp_path / "pintle_parameters.json").read_text())
        assert data["architecture"] == "fixed_annulus_radial_slots"
        assert len(data["dimensions"]) >= 15

    def test_dimensions_csv_has_header_and_rows(self, tmp_path):
        inj, spec = _inj()
        export_pintle_package(inj, tmp_path, spec=spec, cad="none")
        lines = (tmp_path / "pintle_dimensions.csv").read_text().splitlines()
        assert lines[0].startswith("group,symbol,name")
        assert len(lines) > 15

    def test_dxf_profile_written(self, tmp_path):
        inj, spec = _inj()
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="reference",
                                    cad_format="dxf")
        dxf = tmp_path / "pintle_reference.dxf"
        assert dxf.exists()
        text = dxf.read_text()
        assert "ENTITIES" in text and "pintle_rod_tip" in text

    def test_step_without_cadquery_degrades(self, tmp_path):
        from raosim.injector_cad import cadquery_available
        inj, spec = _inj()
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="parts",
                                    cad_format="step")
        # mandatory deliverables always present
        assert (tmp_path / "pintle_schematic.svg").exists()
        if not cadquery_available():
            assert any("CadQuery" in n for n in res["notes"])
            assert not (tmp_path / "pintle_reference.step").exists()

    def test_machined_mode_writes_manufacturing_report(self, tmp_path):
        from raosim.injector_cad import cadquery_available
        inj, spec = _inj()
        spec.mechanical = PintleMechanicalSpec(
            bolt_count=6,
            bolt_circle_diameter=0.085,
            bolt_hole_diameter=0.004,
            faceplate_outer_diameter=0.10,
            faceplate_thickness=0.010,
            fuel_inlet_count=2,
            oxidizer_inlet_count=2,
            min_tool_diameter=0.0005,
            tolerance=0.00005,
        )
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="machined",
                                    cad_format="step")
        report = tmp_path / "injector_manufacturing_report.json"
        assert report.exists() and report.stat().st_size > 0
        data = json.loads(report.read_text())
        assert data["status"] == \
            "preliminary_machined_layout_requires_cold_flow_validation"
        assert data["resolved"]["slot_cut_depth_m"] >= \
            data["resolved"]["pintle_wall_thickness_m"]
        assert data["resolved"]["bolt_count"] == 6
        assert data["flow_continuity"]["status"] == \
            "coaxial_circuits_sealed_until_chamber_exit"
        assert data["coaxial"]["inner_role"] == "fuel"
        assert data["coaxial"]["outer_role"] == "oxidizer"
        assert "manufacturing_report" in res["files"]
        assert any("NASA SP-8089" in s for s in data["literature_basis"])
        assert data["cad_export"]["architecture"] == \
            "coaxial_five_part_center_bore_annular_sheet"
        assert data["cad_export"]["flow_separation_audit"]["circuits_sealed"]
        if not cadquery_available():
            assert data["cad_export"]["status"] == "cadquery_unavailable"

    def test_auto_cad_selects_machined_package(self, tmp_path):
        from raosim.injector_cad import cadquery_available
        inj, spec = _inj()
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="auto",
                                    cad_format="stl")
        report = tmp_path / "injector_manufacturing_report.json"
        assert report.exists() and "manufacturing_report" in res["files"]
        assert any("machined STEP package" in n for n in res["notes"])
        data = json.loads(report.read_text())
        assert data["status"] == \
            "preliminary_machined_layout_requires_cold_flow_validation"
        assert data["architecture"] == \
            "coaxial_five_part_center_bore_annular_sheet"
        if cadquery_available():
            assert "machined_assembly" in res["files"]
            assert (tmp_path / "injector_assembly_machined.step").exists()
        else:
            assert data["cad_export"]["status"] == "cadquery_unavailable"

    def test_coaxial_layout_sizes_distribution_and_seals_circuits(self):
        from raosim.injector_coaxial_cad import (
            analytic_flow_separation,
            resolve_coaxial_layout,
        )
        inj, spec = _inj()
        layout = resolve_coaxial_layout(inj, spec=spec)
        S = layout["coaxial"]
        assert S["inner_role"] == "fuel"
        assert S["outer_role"] == "oxidizer"
        assert S["orifice_plate_open_area_m2"] >= \
            1.25 * layout["roles"]["oxidizer"]["stream_area_m2"]
        audit = analytic_flow_separation(layout)
        assert audit["circuits_sealed"]
        assert audit["minimum_radial_separation_m"] > 0.0

    def test_machined_wrapper_accepts_radial_exit_styles(self, tmp_path):
        from raosim.injector_cad import export_machined_pintle_cad
        inj, spec = _inj()
        res = export_machined_pintle_cad(
            inj, tmp_path, spec=spec, radial_style="slots")
        report = tmp_path / "injector_manufacturing_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["cad_export"]["radial_exit_style"] == "slots"
        assert data["cad_export"]["flow_separation_audit"]["circuits_sealed"]
        assert "manufacturing_report" in res["files"]

    def test_machined_layout_screens_ports_and_slots(self):
        from raosim.injector_cad import resolve_machined_pintle_layout
        inj, spec = _inj()
        spec.mechanical = PintleMechanicalSpec(min_tool_diameter=0.0004)
        layout = resolve_machined_pintle_layout(inj, spec=spec)
        names = {g["name"] for g in layout["manufacturing_gates"]}
        assert "slot_cut_through" in names
        assert "fuel_inlet_velocity" in names
        assert "oxidizer_manifold_velocity" in names
        assert "faceplate_manifold_pocket_depth" in names
        assert "bolt_manifold_clearance" in names
        assert layout["roles"]["fuel"]["inlet_count"] == 2

    def test_machined_layout_grows_thin_face_and_bolt_circle(self):
        from raosim.injector_cad import resolve_machined_pintle_layout
        inj, spec = _inj()
        spec.mechanical = PintleMechanicalSpec(
            faceplate_thickness=0.004,
            fuel_manifold_depth=0.016,
            oxidizer_manifold_depth=0.020,
            bolt_count=8,
            bolt_circle_diameter=0.040,
            bolt_hole_diameter=0.004,
        )
        layout = resolve_machined_pintle_layout(inj, spec=spec)
        r = layout["resolved"]
        assert r["faceplate_thickness_m"] >= \
            r["faceplate_minimum_thickness_m"]
        assert r["bolt_circle_diameter_m"] >= \
            r["bolt_circle_minimum_diameter_m"]
        gates = {g["name"]: g for g in layout["manufacturing_gates"]}
        assert gates["faceplate_manifold_pocket_depth"]["status"] == "warn"
        assert gates["bolt_manifold_clearance"]["status"] == "warn"

    def test_machined_step_face_round_trips_single_solid(self, tmp_path):
        from raosim.injector_cad import (
            cadquery_available,
            export_machined_pintle_cad,
        )
        if not cadquery_available():
            pytest.skip("CadQuery not installed")
        inj, spec = _inj()
        spec.mechanical = PintleMechanicalSpec(
            faceplate_outer_diameter=0.10,
            faceplate_thickness=0.01287,
            fuel_manifold_depth=0.016,
            oxidizer_manifold_depth=0.020,
            bolt_count=8,
            bolt_circle_diameter=0.085,
            bolt_hole_diameter=0.004,
            min_tool_diameter=0.0005,
            tolerance=0.00005,
        )
        res = export_machined_pintle_cad(inj, tmp_path, spec=spec)
        face_path = tmp_path / "faceplate.step"
        assert "faceplate" in res["files"]
        assert face_path.exists() and face_path.stat().st_size > 1000
        assert res["layout"]["cad_export"]["inspection"][
            "faceplate"]["single_solid"]
        import cadquery as cq
        imported = cq.importers.importStep(str(face_path))
        solids = [solid for shape in imported.vals() for solid in shape.Solids()]
        assert len(solids) == 1
        assert solids[0].isValid()

    def test_machined_step_is_millimetre_scale(self, tmp_path):
        """The STEP is written in millimetres (repo convention), so the pintle
        post OD matches pintle_diameter*1e3 -- guards the metres-as-mm bug that
        collapsed the injector to a speck in the engine assembly."""
        from raosim.injector_cad import (
            cadquery_available,
            export_machined_pintle_cad,
        )
        if not cadquery_available():
            pytest.skip("CadQuery not installed")
        inj, spec = _inj()
        export_machined_pintle_cad(inj, tmp_path, spec=spec)
        import cadquery as cq
        post = cq.importers.importStep(
            str(tmp_path / "pintle_body.step"))
        faces = [f for v in post.vals() for f in v.Faces()]
        bb = cq.Compound.makeCompound(faces).BoundingBox()
        expected_od_mm = inj.pintle_diameter * 1.0e3
        # post OD is the transverse (x/y) extent; allow for a chamfer/flat.
        assert bb.xlen == pytest.approx(expected_od_mm, rel=0.05)
        assert bb.ylen == pytest.approx(expected_od_mm, rel=0.05)
        # a real ~10-40 mm pintle, never the sub-mm speck of the old bug.
        assert 3.0 < bb.xlen < 200.0


class TestReferenceLayoutSingleSource:
    """The schematic, DXF profile, and 3-D reference assembly all consume
    resolve_reference_pintle_layout - these invariants keep them honest."""

    def test_tip_cone_honors_deflector_angle(self):
        import math
        from raosim.injector_cad import resolve_reference_pintle_layout
        inj, spec = _inj()
        lay = resolve_reference_pintle_layout(inj, spec)
        flank = math.degrees(math.atan2(
            lay["tip_length_m"],
            lay["pintle_radius_m"] - lay["tip_flat_radius_m"]))
        assert flank == pytest.approx(lay["deflector_angle_deg"], rel=1e-9)
        assert (lay["body_straight_m"] + lay["tip_length_m"]
                == pytest.approx(lay["body_length_m"], rel=1e-12))

    def test_stations_are_ordered(self):
        from raosim.injector_cad import resolve_reference_pintle_layout
        inj, spec = _inj()
        lay = resolve_reference_pintle_layout(inj, spec)
        assert lay["skip_length_m"] >= 0.0
        assert lay["z_sleeve_exit_m"] < lay["z_slot_top_m"]
        assert (lay["z_slot_top_m"] + lay["slot_height_m"]
                <= lay["body_straight_m"] + 1e-12)
        # bore closes at the post/tip joint (screwed-on tip practice)
        assert lay["bore_end_m"] == pytest.approx(lay["body_straight_m"])
        assert lay["bore_radius_m"] < lay["pintle_radius_m"]

    def test_dxf_profile_matches_layout(self):
        from raosim.injector_cad import (
            _pintle_profile_loops,
            resolve_reference_pintle_layout,
        )
        inj, spec = _inj()
        lay = resolve_reference_pintle_layout(inj)
        loops = _pintle_profile_loops(inj)
        assert set(loops) == {"pintle_rod_tip", "pintle_tip_cone",
                              "annular_sleeve", "injector_face"}
        x_nose, r_nose = loops["pintle_tip_cone"][2]
        assert x_nose == pytest.approx(lay["body_length_m"] * 1e3, rel=1e-9)
        assert r_nose == pytest.approx(lay["tip_flat_radius_m"] * 1e3,
                                       rel=1e-9)

    def test_reference_solids_valid_and_conical(self):
        from raosim.injector_cad import cadquery_available
        if not cadquery_available():
            pytest.skip("CadQuery not installed")
        from raosim.injector_cad import (
            build_pintle_parts,
            resolve_reference_pintle_layout,
        )
        inj, spec = _inj()
        lay = resolve_reference_pintle_layout(inj)
        parts = build_pintle_parts(inj)
        for name, wp in parts.items():
            solid = wp.val() if hasattr(wp, "val") else wp
            assert solid.isValid(), name
        tip = parts["pintle_tip"].val()
        bb = tip.BoundingBox()
        assert bb.zlen == pytest.approx(lay["tip_length_m"], rel=1e-6)
        assert bb.xlen == pytest.approx(2.0 * lay["pintle_radius_m"],
                                        rel=1e-6)
        assert bb.zmin == pytest.approx(lay["body_straight_m"], rel=1e-6)
