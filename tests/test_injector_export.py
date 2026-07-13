"""Tests for raosim.injector_export - pintle reference geometry + output package."""

import copy
import json
import math
import os

import pytest

from raosim.injector import (
    InjectorSpec,
    MovablePintleSpec,
    PintleMechanicalSpec,
    PintleGeometrySpec,
    PropellantFeedSpec,
    evaluate_pintle_injector,
)
from raosim.injector_export import (
    export_pintle_package,
    pintle_reference_geometry,
)
from raosim.movable_pintle import movable_geometry_fingerprint

PC, MR = 7.0e6, 2.6
MDOT = 1.8116689741766580
MDOT_F = MDOT / (1.0 + MR)
MDOT_O = MR * MDOT_F


def _inj(radial_exit_style="slots"):
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
                                    radial_exit_style=radial_exit_style,
                                    radial_stream="fuel", deflector_angle=15.0,
                                    face_od=0.10))
    inj = evaluate_pintle_injector(
        spec, mdot_fuel=MDOT_F, mdot_oxidizer=MDOT_O, Pc=PC, mixture_ratio=MR,
        chamber_radius=0.035, chamber_length=0.13, gamma=1.2, Tc=3500.0,
        R_gas=350.0, fuel_name="rp-1", oxidizer_name="lox")
    return inj, spec


def _fixed_hole_inj(diameter=0.0002, min_tool=0.0005):
    baseline, spec = _inj("holes")
    spec = copy.deepcopy(spec)
    spec.sizing = "fixed"
    spec.geometry.radial_hole_diameter = diameter
    spec.geometry.radial_hole_length = max(2.0 * diameter, 0.002)
    spec.geometry.annulus_gap = baseline.annulus.detail["gap"]
    spec.mechanical = PintleMechanicalSpec(min_tool_diameter=min_tool)
    inj = evaluate_pintle_injector(
        spec, mdot_fuel=MDOT_F, mdot_oxidizer=MDOT_O, Pc=PC,
        mixture_ratio=MR, chamber_radius=0.035, chamber_length=0.13,
        gamma=1.2, Tc=3500.0, R_gas=350.0,
        fuel_name="rp-1", oxidizer_name="lox",
    )
    return inj, spec


def _movable_inj():
    fuel = PropellantFeedSpec(
        role="fuel", name="rp-1", inlet_temperature=298.0,
        density=810.0, viscosity=1.6e-3, surface_tension=2.3e-2,
        vapor_pressure=2.0e3,
    )
    oxidizer = PropellantFeedSpec(
        role="oxidizer", name="lox", inlet_temperature=90.0,
        density=1140.0, viscosity=1.9e-4, surface_tension=1.3e-2,
        vapor_pressure=1.0e5,
    )
    movable = MovablePintleSpec(
        post_diameter=0.020,
        post_thickness=0.001,
        center_gap_diameter=0.012,
        pintle_rod_diameter=0.008,
        cd_vs_opening_fraction=(
            (0.0, 0.62), (0.5, 0.70), (1.0, 0.76)
        ),
        cd_calibration_source="configuration-controlled cold-flow map",
        cd_calibration_artifact_sha256="b" * 64,
        cd_reynolds_range=(1.0, 1.0e9),
        cd_pressure_drop_range=(1.0e5, 2.0e6),
        cd_temperature_range=(280.0, 320.0),
        cd_cavitation_number_range=(0.0, 1.0e12),
        cd_fluid_name="rp-1",
        position_tolerance=1.0e-6,
        position_feedback_resolution=1.0e-6,
        backlash=1.0e-6,
        closed_leakage_area=0.0,
        metrology_source="configuration-controlled metrology fixture",
        metrology_artifact_sha256="c" * 64,
        leakage_source="configuration-controlled leakage fixture",
        leakage_artifact_sha256="d" * 64,
        unbalanced_pressure_area=2.0e-5,
        spring_preload_force=5.0,
        seal_friction_force=4.0,
        moving_mass=0.2,
        maximum_acceleration=50.0,
        actuator_force_capacity=500.0,
        stem_diameter=0.006,
        stem_allowable_stress=200.0e6,
        actuator_source="configuration-controlled actuator/material fixture",
        actuator_artifact_sha256="e" * 64,
        sheet_thickness=0.125e-3,
        sheet_thickness_method="vof",
        sheet_thickness_source="configuration-controlled VOF fixture",
        sheet_thickness_artifact_sha256="a" * 64,
        sheet_thickness_fluid_name="rp-1",
        sheet_thickness_opening_range=(1.0e-6, 2.0e-3),
        sheet_thickness_pressure_drop_range=(1.0e5, 2.0e6),
        sheet_thickness_mass_flow_range=(0.1, 1.0),
    )
    geometry_digest = movable_geometry_fingerprint(
        movable, tip_angle_deg=20.0
    )
    movable.cd_geometry_fingerprint_sha256 = geometry_digest
    movable.sheet_thickness_geometry_fingerprint_sha256 = geometry_digest
    spec = InjectorSpec(
        type="pintle",
        architecture="son_continuous_movable",
        sizing="auto",
        fuel=fuel,
        oxidizer=oxidizer,
        geometry=PintleGeometrySpec(
            pintle_diameter=0.020,
            slot_count=1,
            radial_exit_style="continuous_radial_gap",
            radial_stream="fuel",
            deflector_angle=20.0,
            face_od=0.10,
        ),
        movable_pintle=movable,
    )
    inj = evaluate_pintle_injector(
        spec, mdot_fuel=MDOT_F, mdot_oxidizer=MDOT_O, Pc=PC,
        mixture_ratio=MR, chamber_radius=0.035, chamber_length=0.13,
        gamma=1.2, Tc=3500.0, R_gas=350.0,
        fuel_name="rp-1", oxidizer_name="lox",
    )
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

    def test_round_hole_geometry_is_not_reported_as_slots(self):
        inj, spec = _inj("holes")
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {d["symbol"]: d for d in geom["dimensions"]}

        assert geom["architecture"] == "fixed_annulus_radial_holes"
        assert geom["operating_point"]["radial_exit_style"] == "holes"
        assert by["d_hole"]["value_si"] == pytest.approx(
            inj.slots.detail["hole_diameter"]
        )
        assert by["L_hole"]["value_si"] == pytest.approx(
            inj.slots.detail["hole_length"]
        )
        assert "w_slot" not in by

    def test_reference_flow_volume_uses_solved_round_hole_topology(self):
        from raosim.injector_cad import build_pintle_assembly

        inj, _spec = _inj("holes")
        assembly = build_pintle_assembly(inj, include_flow_volumes=True)
        names = {child.name for child in assembly.children}
        assert "radial_hole_network" in names
        assert "radial_slot_network" not in names

    def test_movable_report_keeps_metering_opening_and_sheet_evidence_distinct(self):
        inj, spec = _movable_inj()
        geom = pintle_reference_geometry(inj, spec=spec)
        by = {item["symbol"]: item for item in geom["dimensions"]}

        assert geom["architecture"] == "son_continuous_movable"
        assert geom["model_id"] == "son2017_continuous_radial_gap"
        assert by["D_post"]["value_si"] == pytest.approx(0.020)
        assert by["t_post"]["value_si"] == pytest.approx(0.001)
        assert by["D_cg"]["value_si"] == pytest.approx(0.012)
        assert by["D_pr"]["value_si"] == pytest.approx(0.008)
        assert by["L_open"]["value_si"] == pytest.approx(
            inj.actuation.opening_distance
        )
        assert by["L_transition"]["value_si"] == pytest.approx(
            inj.actuation.transition_opening
        )
        assert by["A_tip"]["value_si"] == pytest.approx(
            inj.actuation.tip_minimum_area
        )
        assert by["A_cg"]["value_si"] == pytest.approx(
            inj.actuation.center_gap_area
        )
        assert by["delta_sheet"]["value_si"] == pytest.approx(0.125e-3)
        assert by["delta_sheet"]["value_si"] != pytest.approx(
            by["L_open"]["value_si"]
        )
        assert geom["evidence"]["sheet_thickness"]["source"] == (
            "configuration-controlled VOF fixture"
        )
        assert geom["evidence"]["sheet_thickness"]["method"] == "vof"
        assert geom["evidence"]["sheet_thickness"]["opening_range_m"] == [
            1.0e-6, 2.0e-3
        ]
        assert geom["evidence"]["sheet_thickness"]["artifact_sha256"] == (
            "a" * 64
        )
        assert geom["evidence"]["discharge_coefficient"][
            "artifact_sha256"
        ] == "b" * 64
        assert geom["evidence"]["discharge_coefficient"][
            "geometry_fingerprint_sha256"
        ] == inj.actuation.resolved_geometry_fingerprint_sha256
        assert geom["evidence"]["discharge_coefficient"][
            "calibration_fluid_name"
        ] == "rp-1"
        assert geom["evidence"]["position_metrology"][
            "artifact_sha256"
        ] == "c" * 64
        assert geom["evidence"]["closed_stop_leakage"][
            "artifact_sha256"
        ] == "d" * 64
        assert geom["evidence"]["actuator_and_material"][
            "artifact_sha256"
        ] == "e" * 64
        assert geom["cad_status"]["available"] is False

    def test_movable_report_rejects_calibration_from_another_geometry(self):
        inj, spec = _movable_inj()
        object.__setattr__(
            inj.actuation,
            "discharge_coefficient_geometry_fingerprint_sha256",
            "d" * 64,
        )
        with pytest.raises(ValueError, match="geometry fingerprint"):
            pintle_reference_geometry(inj, spec=spec)


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

    def test_package_resolves_cad_style_from_solved_hydraulics(self, tmp_path):
        inj, spec = _inj("holes")
        res = export_pintle_package(inj, tmp_path, spec=spec, cad="none")
        assert res["geometry"]["architecture"] == "fixed_annulus_radial_holes"
        with pytest.raises(ValueError, match="does not match"):
            export_pintle_package(
                inj, tmp_path / "bad", spec=spec, cad="none",
                radial_style="slots",
            )

    def test_movable_report_only_package_writes_evidence_without_cad_claim(self,
                                                                            tmp_path):
        inj, spec = _movable_inj()
        result = export_pintle_package(
            inj, tmp_path, spec=spec, cad="none"
        )

        for name in (
            "pintle_parameters.json",
            "pintle_dimensions.csv",
            "pintle_schematic.svg",
            "pintle_cross_section.png",
        ):
            assert (tmp_path / name).exists()
        report = json.loads((tmp_path / "pintle_parameters.json").read_text())
        assert report["architecture"] == "son_continuous_movable"
        assert report["operating_point"]["movable_actuation"][
            "opening_distance_m"
        ] == pytest.approx(inj.actuation.opening_distance)
        assert report["evidence"]["hardware_qualified"] is False
        assert result["cad_audit"] is None

    @pytest.mark.parametrize(
        ("cad", "cad_format"),
        [
            ("reference", "dxf"),
            ("reference", "step"),
            ("parts", "step"),
            ("machined", "step"),
            ("auto", "step"),
            ("step", "step"),
        ],
    )
    def test_every_movable_cad_request_fails_closed(self, tmp_path, cad,
                                                    cad_format):
        inj, spec = _movable_inj()
        with pytest.raises(NotImplementedError, match="swept moving-pintle"):
            export_pintle_package(
                inj,
                tmp_path / cad,
                spec=spec,
                cad=cad,
                cad_format=cad_format,
            )

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

    def test_orifice_plate_pattern_is_independent_of_flange_bolt_count(self):
        from raosim.injector_coaxial_cad import (
            build_coaxial_bodies,
            resolve_coaxial_layout,
        )

        inj, spec = _inj()
        spec.mechanical = PintleMechanicalSpec(
            faceplate_outer_diameter=0.30,
            bolt_count=48,
            bolt_circle_diameter=0.275,
            bolt_hole_diameter=0.006,
            min_tool_diameter=0.0005,
        )
        layout = resolve_coaxial_layout(inj, spec=spec)
        S = layout["coaxial"]

        # A flange bolt pattern is not a hydraulic distribution pattern.  The
        # previous 2*N_bolt coupling produced 96 overlapping bores and split
        # this plate into dozens of disconnected solids in the full-engine
        # sample.
        assert S["n_bolt"] == 48
        assert S["orifice_plate_hole_count"] == max(12, inj.slot_count)
        assert S["orifice_plate_minimum_web_m"] >= S["min_tool"]
        gates = {g["name"]: g for g in layout["manufacturing_gates"]}
        assert gates["orifice_plate_hole_ligament"]["status"] == "pass"

        bodies = build_coaxial_bodies(inj, spec=spec, layout=layout)
        plate = bodies["orifice_plate"].val()
        assert plate.isValid()
        assert len(plate.Solids()) == 1

    def test_round_hole_cad_consumes_exact_hydraulic_diameter(self, tmp_path):
        from raosim.injector_coaxial_cad import resolve_coaxial_layout
        from raosim.injector_cad import export_machined_pintle_cad

        # Deliberately below the selected tool floor: the manufacturing gate
        # must fail, but CAD must never enlarge the metering bore and change
        # the hydraulically delivered mass flow.
        diameter = 0.0002
        inj, spec = _fixed_hole_inj(diameter=diameter, min_tool=0.0005)
        layout = resolve_coaxial_layout(inj, spec=spec)
        S = layout["coaxial"]
        expected_area = inj.slot_count * math.pi * diameter**2 / 4.0
        assert S["radial_exit_style"] == "holes"
        assert S["hole_d"] == pytest.approx(diameter)
        assert S["radial_opening_area_m2"] == pytest.approx(expected_area)
        assert S["radial_opening_area_m2"] == pytest.approx(inj.slots.area)
        assert abs(S["radial_opening_area_error_fraction"]) < 1.0e-12
        gates = {g["name"]: g for g in layout["manufacturing_gates"]}
        assert gates["minimum_tool_diameter"]["status"] == "fail"

        result = export_machined_pintle_cad(inj, tmp_path, spec=spec)
        report = json.loads(
            (tmp_path / "injector_manufacturing_report.json").read_text()
        )
        assert report["cad_export"]["radial_exit_style"] == "holes"
        assert report["cad_export"]["status"] == "step_written"
        assert report["cad_export"]["component_interference_audit"]["passed"]
        assert "machined_assembly" in result["files"]

    def test_export_rejects_radial_topology_override(self, tmp_path):
        from raosim.injector_cad import export_machined_pintle_cad

        inj, spec = _inj("holes")
        with pytest.raises(ValueError, match="disagrees with solved"):
            export_machined_pintle_cad(
                inj, tmp_path, spec=spec, radial_style="slots"
            )

    def test_seal_glands_retention_and_component_clearance_are_modeled(self):
        from raosim.injector_coaxial_cad import (
            audit_component_interference,
            audit_flow_connectivity,
            audit_nominal_clearances,
            build_coaxial_bodies,
            resolve_coaxial_layout,
        )

        inj, spec = _inj()
        layout = resolve_coaxial_layout(inj, spec=spec)
        S = layout["coaxial"]
        features = layout["mechanical_features"]
        assert features["pintle_retention"]["flange_modeled"]
        assert features["pintle_retention"]["fastener_holes_modeled"]
        assert features["replaceable_tip"]["spigot_socket_modeled"]
        assert features["seals"]["chamber_joint_gland_modeled"]
        assert features["seals"]["post_face_gland_modeled"]
        assert features["seals"]["plate_outer_glands_modeled"]
        assert (
            S["joint_seal_center"] + 0.5 * S["seal_w"]
            < S["bolt_c"] - S["bolt_r"]
        )
        assert (
            S["post_seal_center"] + 0.5 * S["seal_w"]
            < S["retention_bolt_c"] - 0.5 * S["retention_bolt_d"]
        )
        bodies = build_coaxial_bodies(inj, spec=spec, layout=layout)
        assert audit_component_interference(bodies)["passed"]
        connectivity = audit_flow_connectivity(layout)
        assert connectivity["passed"], connectivity
        assert connectivity["inner_single_connected_void"]
        assert connectivity["outer_single_connected_void"]
        assert min(
            connectivity["handoff_overlap_volumes_m3"].values()
        ) > connectivity["minimum_required_overlap_m3"]
        clearances = audit_nominal_clearances(layout)
        assert clearances["passed"]
        assert min(clearances["clearances"].values()) > 0.0

        # Removing the O-ring specification removes four modeled glands and
        # therefore increases each owning component's material volume.
        dry = copy.deepcopy(spec)
        dry.mechanical.seal_type = "none"
        dry_layout = resolve_coaxial_layout(inj, spec=dry)
        dry_bodies = build_coaxial_bodies(inj, spec=dry, layout=dry_layout)
        for name in (
            "pintle_body", "pintle_tip", "injector_body", "orifice_plate",
            "faceplate",
        ):
            assert dry_bodies[name].val().Volume() > bodies[name].val().Volume()

    def test_machined_report_does_not_claim_hardware_release(self, tmp_path):
        from raosim.injector_cad import export_machined_pintle_cad

        inj, spec = _inj()
        result = export_machined_pintle_cad(inj, tmp_path, spec=spec)
        cad = result["layout"]["cad_export"]
        assert cad["flow_connectivity_audit"]["passed"]
        assert cad["nominal_clearance_audit"]["passed"]
        assert cad["component_interference_audit"]["passed"]
        assert cad["cold_flow_release_ready"] is False
        assert cad["hot_fire_release_ready"] is False
        assert cad["hardware_qualified"] is False
        assert cad["external_release_blockers"]

        package = export_pintle_package(
            inj, tmp_path / "package", spec=spec, cad="machined"
        )
        assert package["cad_audit"]["flow_connectivity_audit"]["passed"]
        assert package["cad_audit"]["hardware_qualified"] is False

    def test_machined_cad_gate_failure_is_not_returned_as_success(
        self, tmp_path, monkeypatch
    ):
        import raosim.injector_coaxial_cad as coaxial
        from raosim.injector_cad import export_machined_pintle_cad

        inj, spec = _inj()

        def fail_build(*_args, **_kwargs):
            raise RuntimeError("synthetic invalid solid")

        monkeypatch.setattr(coaxial, "build_coaxial_bodies", fail_build)
        with pytest.raises(RuntimeError, match="required geometry/export gate"):
            export_machined_pintle_cad(inj, tmp_path, spec=spec)
        report = json.loads(
            (tmp_path / "injector_manufacturing_report.json").read_text()
        )
        assert report["cad_export"]["status"] == "step_export_failed"
        assert "synthetic invalid solid" in report["cad_export"]["error"]

    def test_reference_and_parts_steps_use_mm_and_unit_sidecar(self, tmp_path):
        from raosim.injector_cad import cadquery_available

        if not cadquery_available():
            pytest.skip("CadQuery not installed")
        inj, spec = _inj()
        result = export_pintle_package(
            inj, tmp_path, spec=spec, cad="parts", cad_format="step"
        )
        units = json.loads((tmp_path / "pintle_cad_units.json").read_text())
        assert units["neutral_file_linear_unit"] == "mm"
        assert all(unit == "mm" for unit in units["files"].values())
        assert "cad_units" in result["files"]

        import cadquery as cq

        part = cq.importers.importStep(
            str(tmp_path / "pintle_parts" / "pintle_rod.step")
        )
        solids = [s for shape in part.vals() for s in shape.Solids()]
        assert len(solids) == 1 and solids[0].isValid()
        assert solids[0].BoundingBox().xlen == pytest.approx(
            inj.pintle_diameter * 1.0e3, rel=0.05
        )

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
        expected_od_mm = inj.pintle_diameter * 1.0e3
        # The tip is a pure Dp cylinder (no flange), so its transverse extent
        # is the pintle OD exactly.
        tip = cq.importers.importStep(str(tmp_path / "pintle_tip.step"))
        faces = [f for v in tip.vals() for f in v.Faces()]
        bb = cq.Compound.makeCompound(faces).BoundingBox()
        assert bb.xlen == pytest.approx(expected_od_mm, rel=0.05)
        assert bb.ylen == pytest.approx(expected_od_mm, rel=0.05)
        # The body carries the retention flange, so it is wider than Dp but
        # still the same order — never the sub-mm speck of the old bug.
        post = cq.importers.importStep(str(tmp_path / "pintle_body.step"))
        pfaces = [f for v in post.vals() for f in v.Faces()]
        pbb = cq.Compound.makeCompound(pfaces).BoundingBox()
        assert expected_od_mm <= pbb.xlen < 6.0 * expected_od_mm
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
