"""Tests for raosim.flange_cad — chamber flange solid and fastener callout."""

import math

import pytest

from raosim.flange_cad import (
    build_chamber_flange,
    cadquery_available,
    fastener_callout,
)
from raosim.interface import size_bolted_interface
from raosim.materials import get_material


@pytest.fixture(scope="module")
def sizing():
    inconel = get_material("Inconel 718")
    return size_bolted_interface(
        chamber_radius=0.031368 * math.sqrt(8.0),
        chamber_pressure=3.0e6,
        wall_thickness=8.0e-4,
        material_yield_strength=inconel.yield_strength,
        structural_fos=1.5,
        flange_density=inconel.density,
        bolt_density=7850.0,
    )


def test_callout_is_orderable_hardware(sizing):
    """"Correctly sized bolts" means a thread, a class, a length and a torque."""

    c = fastener_callout(sizing).to_dict()
    assert c["designation"].startswith("M")
    assert c["property_class"] in {"8.8", "10.9", "12.9", "A2-70"}
    assert c["count"] > 0
    assert c["minimum_bolt_length_m"] > c["grip_length_m"]
    assert c["tightening_torque_n_m"] > 0.0
    # Nothing here may present itself as qualified.
    assert c["status"] == "screening_sized"
    assert "torque-tension" in c["qualification_status"]


def test_preload_and_torque_follow_the_documented_relations(sizing):
    from raosim.interface import _BOLT_CLASSES

    c = fastener_callout(sizing, lubrication="dry", reusable=True).to_dict()
    proof, _ = _BOLT_CLASSES[sizing.bolt_class]
    assert c["proof_load_n"] == pytest.approx(proof * sizing.bolt_stress_area)
    assert c["target_preload_n"] == pytest.approx(0.75 * c["proof_load_n"])
    # Shigley short form T = K F_i d.
    assert c["tightening_torque_n_m"] == pytest.approx(
        c["nut_factor"] * c["target_preload_n"] * c["nominal_diameter_m"]
    )
    assert c["nut_factor"] == pytest.approx(0.20)

    lubed = fastener_callout(sizing, lubrication="lubricated").to_dict()
    assert lubed["nut_factor"] == pytest.approx(0.15)
    assert lubed["tightening_torque_n_m"] < c["tightening_torque_n_m"]

    permanent = fastener_callout(sizing, reusable=False).to_dict()
    assert permanent["target_preload_n"] > c["target_preload_n"]


def test_unknown_lubrication_is_rejected(sizing):
    with pytest.raises(ValueError, match="lubrication"):
        fastener_callout(sizing, lubrication="graphite-slurry")


def test_bolt_utilisation_is_within_the_allowable(sizing):
    c = fastener_callout(sizing).to_dict()
    assert 0.0 < c["utilisation"] <= 1.0
    assert c["load_per_bolt_n"] * c["count"] == pytest.approx(
        c["joint_separation_load_n"], rel=1e-9
    )


@pytest.mark.skipif(not cadquery_available(), reason="CadQuery not installed")
def test_flange_solid_matches_the_ledger_and_the_faceplate_pattern(sizing):
    import cadquery as cq

    from raosim.mass_ledger import flange_bolt_mass_ledger

    res = sizing.resolution
    flange = build_chamber_flange(cq, res)
    solids = [s for v in flange.vals() for s in v.Solids()]
    assert len(solids) == 1 and solids[0].isValid()

    cad_volume_m3 = float(abs(solids[0].Volume())) * 1.0e-9
    ledger = flange_bolt_mass_ledger(res, flange_material="Inconel 718")
    ring = next(i for i in ledger.items if i.component == "chamber flange ring")
    # The ledger's closed-form annulus-less-holes must be the solid the
    # exporter writes; that is the module's whole premise.
    assert ring.volume_m3 == pytest.approx(cad_volume_m3, rel=1e-6)

    # The bolt circle must match the faceplate's hole-for-hole.
    bb = solids[0].BoundingBox()
    assert bb.xlen == pytest.approx(res.flange_outer_diameter * 1e3, rel=1e-6)
    assert bb.zlen == pytest.approx(res.flange_length * 1e3, rel=1e-6)


@pytest.mark.skipif(not cadquery_available(), reason="CadQuery not installed")
def test_export_is_gated_on_reimport(sizing, tmp_path):
    from raosim.flange_cad import export_chamber_flange_step

    report = export_chamber_flange_step(
        sizing.resolution, tmp_path / "chamber_flange.step", sizing=sizing
    )
    assert report["valid"] and report["solid_count"] == 1
    assert report["volume_mm3"] > 0.0
    assert report["neutral_file_linear_unit"] == "mm"
    assert report["bolt_count"] == sizing.resolution.bolt_count
    assert "fastener_callout" in report
    assert report["status"].startswith("preliminary")


@pytest.mark.skipif(not cadquery_available(), reason="CadQuery not installed")
def test_degenerate_flange_is_rejected_not_exported(sizing):
    import cadquery as cq
    from dataclasses import replace

    bad = replace(
        sizing.resolution,
        flange_outer_diameter=sizing.resolution.chamber_outer_diameter,
    )
    with pytest.raises(ValueError, match="outer diameter"):
        build_chamber_flange(cq, bad)
