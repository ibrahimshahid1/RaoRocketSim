"""True B-rep pump CAD (raosim.pump_cad_brep): named assemblies, re-import
validity, and meanline dimension fidelity.  CadQuery-gated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raosim.injector import FeedLineLedger, FeedSystemLedger
from raosim.pump_cad import pump_reference_geometry
from raosim.pump_cad_brep import (
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
    assert len(step_keys) >= 15, sorted(package["files"])
    for key in step_keys:
        info = package["diagnostics"][key]
        assert info["valid"], (key, info)
        assert info["volume_mm3"] > 0.0, (key, info)
        assert info["representation"] == "open_cascade_brep"
    # Per-part bodies are single solids; assemblies carry one per part.
    for key, info in package["diagnostics"].items():
        if not key.endswith("_assembly_step"):
            assert info["solid_count"] == 1, (key, info)


def test_expected_parts_and_stls_present(package):
    names = {Path(p).name for p in package["files"].values()}
    for role in ("fuel", "oxidizer"):
        for part in ("impeller", "inducer", "diffuser_ring",
                     "volute_casing", "shaft", "motor"):
            assert f"{role}_{part}.step" in names
            assert f"{role}_{part}.stl" in names
        assert f"{role}_pump.step" in names
    assert "shared_battery_pack.step" in names


def test_named_assembly_children(pump):
    assembly = build_pump_assembly(pump, "fuel")
    assert assembly.name == "fuel_pump"
    child_names = {child.name for child in assembly.children}
    assert {"impeller", "inducer", "diffuser_ring", "volute_casing",
            "shaft"} <= child_names


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


def test_bore_fit_notes_are_honest(pump):
    parts, notes = build_pump_parts(pump, "fuel")
    # This screening-size pump solves a minimum shaft larger than the
    # SP-8052 inducer hub: CAD must say so instead of severing the rotor.
    assert any("inducer shaft bore skipped" in note for note in notes)
    for name, solid in parts.items():
        assert solid.isValid(), name
        assert len(solid.Solids()) == 1, name


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
    screens = info["pump_mount_flange_screen"]
    assert set(screens) == {"fuel", "oxidizer"}
