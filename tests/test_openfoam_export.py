"""Determinism, provenance, geometry, and fail-closed OpenFOAM exports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from raosim.openfoam import (
    MovablePintleVOFGeometry,
    OpenFOAMExportError,
    OpenFOAMWedgeControls,
    build_radhakrishnan2018_sheet_vof_case,
    write_openfoam_case,
)


MANIFEST = "raosim_openfoam_manifest.json"


def _artifacts(package) -> str:
    return "\n".join(
        package.files[path]
        for path in sorted(package.files)
        if path != MANIFEST
    )


def test_exports_foundation_v13_water_only_vof_tree_without_running_tools():
    package = build_radhakrishnan2018_sheet_vof_case("case_1")
    expected = {
        "0/U",
        "0/alpha.water",
        "0/epsilon",
        "0/k",
        "0/nut",
        "0/p_rgh",
        "Allclean",
        "Allrun",
        "README.md",
        "constant/g",
        "constant/momentumTransport",
        "constant/phaseProperties",
        "constant/physicalProperties.air",
        "constant/physicalProperties.water",
        "system/blockMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        MANIFEST,
    }
    assert set(package.files) == expected
    manifest = package.manifest
    assert manifest["solver"] == {
        "distribution": "OpenFOAM Foundation",
        "major_version": 13,
        "patch_tag": "20260624",
        "runner": "foamRun",
        "module": "incompressibleVoF",
        "source_url": "https://github.com/OpenFOAM/OpenFOAM-13/tree/20260624",
        "runtime_version_verified": False,
    }
    assert "solver          incompressibleVoF;" in package.files[
        "system/controlDict"
    ]
    assert "runApplication foamRun" in package.files["Allrun"]
    assert "runApplication checkMesh -allTopology -allGeometry" in package.files[
        "Allrun"
    ]
    assert manifest["benchmark"]["paper_stage"].startswith("Section 2.3")
    assert "airInlet" not in _artifacts(package)
    assert "3.03 g/s" in package.files["README.md"]
    assert not manifest["gates"]["solver_executed"]


def test_case_is_byte_deterministic_and_hashes_cover_every_nonmanifest_file():
    first = build_radhakrishnan2018_sheet_vof_case("case_2")
    second = build_radhakrishnan2018_sheet_vof_case("case_2")
    assert first.fingerprint == second.fingerprint
    assert dict(first.files) == dict(second.files)
    hashes = first.manifest["artifact_sha256"]
    assert set(hashes) == set(first.files) - {MANIFEST}
    for path, expected in hashes.items():
        assert hashlib.sha256(first.files[path].encode("utf-8")).hexdigest() == expected
    parsed = json.loads(first.files[MANIFEST])
    assert parsed["case_fingerprint_sha256"] == first.fingerprint
    assert all(text.endswith("\n") and "\r" not in text for text in first.files.values())


def test_changed_mesh_input_changes_fingerprint_without_host_or_timestamp_data():
    first = build_radhakrishnan2018_sheet_vof_case(
        controls=OpenFOAMWedgeControls(radial_cells=120)
    )
    second = build_radhakrishnan2018_sheet_vof_case(
        controls=OpenFOAMWedgeControls(radial_cells=121)
    )
    assert first.fingerprint != second.fingerprint
    manifest_text = first.files[MANIFEST]
    assert str(Path.cwd()) not in manifest_text
    assert "generated_at" not in manifest_text


def test_mechanical_opening_is_input_and_author_vof_sheet_is_output_target():
    package = build_radhakrishnan2018_sheet_vof_case("case_4")
    mapping = package.manifest["benchmark"]["paper_stage_mapping"]
    assert mapping["mechanical_opening_input"]["value_m"] == pytest.approx(0.8e-3)
    output = mapping["author_vof_sheet_thickness_output_target"]
    assert output["value_m"] == pytest.approx(0.568e-3)
    assert output["prescribed_to_case"] is False
    assert package.manifest["geometry"]["opening_distance_m"] == pytest.approx(0.8e-3)
    assert package.manifest["template"]["geometry_fidelity"].startswith(
        "reduced_external_gap"
    )
    assert not package.manifest["gates"]["internal_injector_geometry_resolved"]


def test_wave_constants_and_lagrangian_air_are_provenance_only_not_vof_inputs():
    package = build_radhakrishnan2018_sheet_vof_case("case_1")
    mapping = package.manifest["benchmark"]["paper_stage_mapping"]
    assert mapping["lagrangian_air_mass_flow"]["prescribed_to_case"] is False
    assert mapping["wave_constants"]["prescribed_to_case"] is False
    foam_text = _artifacts(package)
    assert "wave_b0" not in foam_text
    assert "wave_b1" not in foam_text
    assert "mdot_air" not in foam_text


@pytest.mark.parametrize("case_id", ("case_1", "case_2", "case_3", "case_4"))
def test_full_annulus_and_wedge_liquid_mass_flux_close(case_id):
    package = build_radhakrishnan2018_sheet_vof_case(case_id)
    flux = package.manifest["boundary_flux"]
    expected_wedge = flux["liquid_mass_flow_360_kg_s"] * flux["wedge_fraction"]
    integrated = (
        package.manifest["mesh_and_numerics"]["water"]["density_kg_m3"]
        * flux["liquid_inlet_area_wedge_m2"]
        * flux["liquid_radial_velocity_m_s"]
    )
    assert flux["liquid_mass_flow_wedge_kg_s"] == pytest.approx(expected_wedge)
    assert integrated == pytest.approx(expected_wedge, rel=1e-12)
    assert flux["relative_mass_residual"] <= 1e-12


def test_openfoam_kinematic_viscosities_are_mu_over_rho():
    package = build_radhakrishnan2018_sheet_vof_case()
    numerics = package.manifest["mesh_and_numerics"]
    for phase in ("water", "air"):
        values = numerics[phase]
        expected = values["dynamic_viscosity_pa_s"] / values["density_kg_m3"]
        assert values["kinematic_viscosity_m2_s"] == pytest.approx(expected)
        text = package.files[f"constant/physicalProperties.{phase}"]
        match = re.search(r"\nnu\s+([^;]+);", text)
        assert match
        assert float(match.group(1)) == pytest.approx(expected, rel=1e-11)


def test_wedge_mesh_is_symmetric_curved_and_positive_volume():
    package = build_radhakrishnan2018_sheet_vof_case()
    text = package.files["system/blockMeshDict"]
    vertex_section = text.split("vertices\n(\n", 1)[1].split("\n);", 1)[0]
    vertices = [
        tuple(float(value) for value in match.groups())
        for match in re.finditer(
            r"\(([-+0-9.eE]+) ([-+0-9.eE]+) ([-+0-9.eE]+)\)",
            vertex_section,
        )
    ]
    assert len(vertices) == 16
    for station in range(4):
        base = 4 * station
        assert vertices[base][0:2] == pytest.approx(vertices[base + 2][0:2])
        assert vertices[base][2] == pytest.approx(-vertices[base + 2][2])
        assert vertices[base + 1][0:2] == pytest.approx(
            vertices[base + 3][0:2]
        )
        assert vertices[base + 1][2] == pytest.approx(-vertices[base + 3][2])
    assert text.count("    arc ") == 8
    assert re.search(r"\([^\n]+\)\n    \([0-9]+ [0-9]+ 1\)", text)
    # A representative local Jacobian for each block: axial x radial x azimuthal.
    for i in range(3):
        a = vertices[4 * i]
        b = vertices[4 * (i + 1)]
        radial = vertices[4 * i + 1]
        azimuthal = vertices[4 * i + 2]
        e1 = tuple(b[j] - a[j] for j in range(3))
        e2 = tuple(radial[j] - a[j] for j in range(3))
        e3 = tuple(azimuthal[j] - a[j] for j in range(3))
        cross = (
            e2[1] * e3[2] - e2[2] * e3[1],
            e2[2] * e3[0] - e2[0] * e3[2],
            e2[0] * e3[1] - e2[1] * e3[0],
        )
        jacobian = sum(e1[j] * cross[j] for j in range(3))
        assert jacobian > 0.0


def test_every_field_names_every_mesh_patch_once():
    package = build_radhakrishnan2018_sheet_vof_case()
    patches = (
        "upstreamAmbient",
        "downstreamOutlet",
        "innerWallUpstream",
        "waterInlet",
        "innerWallDownstream",
        "outerAtmosphere",
        "wedgeFront",
        "wedgeBack",
    )
    for path in ("0/U", "0/alpha.water", "0/p_rgh", "0/k", "0/epsilon", "0/nut"):
        body = package.files[path].split("boundaryField", 1)[1]
        for patch in patches:
            assert len(re.findall(rf"^    {patch}$", body, re.MULTILINE)) == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"wedge_angle_deg": 12.0}, "wedge_angle"),
        ({"opening_cells": 3}, "opening_cells"),
        ({"max_alpha_courant": 0.3}, "max_alpha_courant"),
        ({"radial_expansion_ratio": 21.0}, "radial_expansion_ratio"),
        ({"initial_delta_t_s": math.nan}, "initial_delta_t"),
        ({"radial_cells": True}, "radial_cells"),
        ({"radial_cells": 1000, "max_total_cells": 10}, "exceeds"),
    ],
)
def test_controls_reject_unsafe_or_unbounded_values(kwargs, match):
    with pytest.raises(OpenFOAMExportError, match=match):
        OpenFOAMWedgeControls(**kwargs)


def test_geometry_contract_requires_full_movable_pintle_dimensions():
    with pytest.raises(OpenFOAMExportError, match="rod diameter"):
        MovablePintleVOFGeometry(
            post_diameter_m=8e-3,
            center_gap_diameter_m=4.55e-3,
            pintle_rod_diameter_m=5e-3,
            pintle_tip_diameter_m=8e-3,
            annular_gap_thickness_m=0.5e-3,
            post_angle_deg=30,
            pintle_tip_angle_deg=40,
            pintle_tip_thickness_m=1e-3,
            post_recess_length_m=3e-3,
            post_thickness_m=0.5e-3,
            opening_distance_m=0.2e-3,
            axial_domain_length_m=80e-3,
            radial_domain_radius_m=120e-3,
        )


def test_writer_is_atomic_idempotent_and_refuses_nonidentical_destination(tmp_path):
    package = build_radhakrishnan2018_sheet_vof_case()
    destination = tmp_path / "case"
    first = write_openfoam_case(package, destination)
    assert first.written
    assert (destination / MANIFEST).is_file()
    assert (destination / "Allrun").stat().st_mode & 0o111
    second = write_openfoam_case(package, destination)
    assert not second.written
    (destination / "README.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(OpenFOAMExportError, match="not the exact same"):
        write_openfoam_case(package, destination)


def test_writer_rejects_symlinked_destination_components(tmp_path):
    package = build_radhakrishnan2018_sheet_vof_case()
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(OpenFOAMExportError, match="symlinked"):
        write_openfoam_case(package, linked / "case")


def test_manifest_never_promotes_static_export_to_validation():
    gates = build_radhakrishnan2018_sheet_vof_case().manifest["gates"]
    required_false = {
        "internal_injector_geometry_resolved",
        "exact_openfoam_runtime_verified",
        "block_mesh_executed",
        "check_mesh_passed",
        "solver_executed",
        "mesh_convergence_verified",
        "time_step_convergence_verified",
        "author_vof_component_targets_reproduced",
        "experimental_smd_validated",
        "vof_to_lagrangian_handoff_verified",
        "reacting_spray_validated",
        "lox_gch4_applicable",
        "hardware_qualified",
    }
    assert all(gates[name] is False for name in required_false)
