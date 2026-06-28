"""Tests for raosim.injector_export - pintle reference geometry + output package."""

import json
import os

import pytest

from raosim.injector import (
    InjectorSpec,
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
