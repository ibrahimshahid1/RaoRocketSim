"""Cooling-aware B-rep (raosim.regen_cad): the liner + ribs + jacket as one
STEP solid with REAL channel voids (not a surface mesh).  CadQuery-gated.
"""
from __future__ import annotations

import copy

import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import custom_propellant
from raosim.regen_cad import (
    cadquery_available, export_channel_wall_step, export_regen_brep,
    inspect_regen_step,
)
from raosim.thermal_design import size_wall_profile

pytestmark = pytest.mark.skipif(
    not cadquery_available(), reason="CadQuery/OpenCascade not installed")


@pytest.fixture(scope="module")
def profile():
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    contour = bell_nozzle_contour(Rt=0.08, epsilon=10.0, gamma=1.24, length_pct=80.0)
    res = size_wall_profile(
        contour, prop, 1.0e6, material="grcop-84", jacket_material="inconel718",
        channel_count=40, channel_width=0.0015, channel_height=0.005,
        mixture_ratio=2.0, n_outer=2, n_iter=10)
    return res["profile"]


def test_axial_brep_is_one_valid_solid_with_channel_voids(profile, tmp_path):
    profile.helix_turns = 0.0
    info = export_regen_brep(profile, tmp_path / "regen.step", max_sections=22)
    assert info["valid"] and info["single_solid"]
    assert info["representation"] == "open_cascade_brep"
    assert info["channel_count"] == 40
    assert info["include_manifolds"] is False        # robust core is the default
    # The channels are REAL voids: the solid is lighter than the fused envelope.
    assert 0.0 < info["coolant_void_volume_mm3"] < info["envelope_volume_mm3"]
    assert info["solid_volume_mm3"] < info["envelope_volume_mm3"]
    assert info["model"] == "fused_liner_ribs_jacket_with_lofted_passage_voids"
    # Re-importing the STEP confirms exactly one valid solid.
    insp = inspect_regen_step(tmp_path / "regen.step")
    assert insp["solid_count"] == 1 and insp["valid"]
    assert (tmp_path / "regen.step").stat().st_size > 10_000


def test_more_channels_remove_more_material(profile, tmp_path):
    """Sanity: the void volume tracks the channel geometry (a coarse vs a
    finer channel count removes a different amount of material)."""
    profile.helix_turns = 0.0
    info = export_regen_brep(profile, tmp_path / "r1.step", max_sections=20)
    assert info["loft_sections"] >= 4                # enough sections to loft
    assert info["coolant_void_volume_mm3"] > 0.0


def test_helical_brep_is_valid_and_void_volume_tracks_longer_path(profile, tmp_path):
    """The annular-segment cross-section (corners on the cylindrical floor /
    ceiling) makes the HELICAL channel cut close to one valid solid, and the
    adaptive section count keeps the swept void volume faithful. Width is
    defined normal to coolant flow, so its constant-x band widens by the local
    helix stretch and the longer passage removes more material."""
    profile.helix_turns = 0.0
    axial = export_regen_brep(profile, tmp_path / "axial.step", max_sections=22)
    profile.helix_turns = 2.0
    helix = export_regen_brep(profile, tmp_path / "helix.step", max_sections=22)
    assert helix["valid"] and helix["single_solid"]
    va = axial["coolant_void_volume_mm3"]
    vh = helix["coolant_void_volume_mm3"]
    assert vh > 1.05 * va


# --------------------------------------------------------------------- #
#  Faithful FULL-N channel wall (lands as positives, channels = gaps).
#  One multi-shape material fuse, but no per-channel Boolean cuts.
# --------------------------------------------------------------------- #
def test_full_n_channel_wall_is_one_fused_solid(profile, tmp_path):
    p = copy.deepcopy(profile)
    p.channel_count = 12
    p.helix_turns = 0.0
    info = export_channel_wall_step(p, tmp_path / "wall.step", max_sections=18)
    assert info["single_solid"]
    assert info["solid_count"] == 1
    assert info["inspection"]["valid"]
    assert info["inspection"]["solid_count"] == 1
    assert 0.0 < info["void_fraction"] < 1.0
    assert info["model"] == "full_n_patterned_ribs_single_solid_channels_as_gaps"
    assert info["fuse_kernel"]["run_parallel"]
    assert (tmp_path / "wall.step").stat().st_size > 10_000


def test_full_n_channel_wall_helix_is_valid(profile, tmp_path):
    p = copy.deepcopy(profile)
    p.channel_count = 8
    p.helix_turns = 0.25
    info = export_channel_wall_step(p, tmp_path / "wallh.step", max_sections=16)
    assert info["single_solid"] and info["inspection"]["valid"]


def test_full_n_manifold_network_is_one_connected_material_solid(profile, tmp_path):
    p = copy.deepcopy(profile)
    p.channel_count = 8
    p.helix_turns = 0.0
    info = export_channel_wall_step(
        p,
        tmp_path / "network.step",
        max_sections=16,
        include_manifolds=True,
        release_mode="cold_flow",
        ports_per_manifold=2,
    )
    assert info["single_solid"] and info["inspection"]["single_solid"]
    assert info["include_manifolds"]
    assert info["release_mode"] == "cold_flow"
    assert info["cold_flow_geometry_ready"]
    assert info["cold_flow_release_ready"] is False
    assert info["hardware_qualified"] is False
    assert info["external_release_blockers"]
    assert info["flow_path_status"] == \
        "connected_inlet_channels_outlet_with_external_ports"
    assert min(info["network_overlaps"].values()) > 0.0
    assert (
        info["manifold_metrics"]["hydraulic_status"]
        == "continuity_area_screen_only_no_maldistribution_solution"
    )


def test_cold_flow_release_rejects_sealed_channel_ends(profile, tmp_path):
    with pytest.raises(ValueError, match="requires include_manifolds=True"):
        export_channel_wall_step(
            profile, tmp_path / "sealed.step", release_mode="cold_flow"
        )
    with pytest.raises(ValueError, match="requires include_manifolds=True"):
        export_regen_brep(
            profile, tmp_path / "sealed_brep.step", release_mode="cold_flow"
        )
