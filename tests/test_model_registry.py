"""Provenance registry contracts for physical equations and heuristics."""

import json
from pathlib import Path

from raosim.model_registry import (
    MODEL_REGISTRY,
    audit_model_registry,
    get_model_provenance,
    model_provenance_dict,
)


SPRAY_MODEL_IDS = {
    "spray.prescribed_one_way_carrier",
    "spray.schiller_naumann_drag",
    "spray.seeded_discrete_random_walk",
    "spray.wave_kh_radhakrishnan_2018",
    "spray.wave_kh_radhakrishnan_2021",
    "spray.rayleigh_taylor_openfoam",
    "spray.spalding_evaporation_eq16",
    "spray.weighted_smd_rosin_rammler",
    "spray.benchmark_radhakrishnan2018_water_air",
    "spray.benchmark_radhakrishnan2021_water_air",
    "spray.benchmark_radhakrishnan2021_tables_7_8",
    "spray.primary_geometry_dispatch",
    "spray.deterministic_parcel_march",
    "spray.typed_cycle_handoff",
    "spray.openfoam13_external_gap_vof",
    "spray.vof_to_lagrangian_handoff",
}


def test_registry_covers_injector_pump_and_chamber():
    subsystems = {entry.subsystem for entry in MODEL_REGISTRY.values()}
    assert {
        "injector", "pump", "chamber", "performance", "throat", "nozzle",
        "thermal", "structure", "materials", "interface", "separation", "cad",
        "spray",
    } <= subsystems


def test_repository_heuristics_are_not_misrepresented_as_literature_laws():
    heuristic_ids = {
        "injector.auto_pintle_diameter",
        "injector.primary_breakup_15dh",
        "chamber.shoulder_fill_fraction",
        "pump.auto_efficiency_flow_head",
    }
    for model_id in heuristic_ids:
        entry = get_model_provenance(model_id)
        assert "heuristic" in entry.status
        assert entry.validity
        assert entry.verification


def test_injector_registry_serializes_and_exposes_validity():
    data = model_provenance_dict(subsystem="injector")
    json.dumps(data)
    assert data["injector.hinze_stable_drop"]["validity"]
    assert "transcritical" in data["injector.hinze_stable_drop"]["validity"]


def test_son2017_movable_pintle_registry_preserves_geometry_and_limits():
    entry = get_model_provenance(
        "injector.son2017_movable_continuous_gap"
    )

    assert "A_tip=pi/sin(theta_pt)" in entry.relation
    assert "theta_pt->0" in entry.relation
    assert "A_cg=pi/4*(D_cg^2-D_pr^2)" in entry.relation
    assert "A_eff=min(A_tip,A_cg)" in entry.relation
    assert "0.417946" in entry.verification
    assert "0.454243" in entry.verification
    assert "0.568310" in entry.verification
    assert "Cd(opening,Re)" in entry.validity
    assert "actuator" in entry.validity
    assert "sheet thickness" in entry.validity
    assert "mechanical opening is not liquid-sheet thickness" in entry.validity
    assert entry.local_source == "propulsion_texts/pintle_injector/son2017.pdf"
    assert "no swept" in entry.notes
    assert entry.validation_level.endswith("not_hydraulic_actuator_sheet_or_hardware_validated")


def test_rao_bvp_registry_records_the_strict_literature_benchmark():
    entry = get_model_provenance("nozzle.rao_variational_bvp")
    assert "Rao 1958 Nozzle B Table II" in entry.verification
    assert entry.validation_level == \
        "strict_literature_benchmark_passed_not_cfd_validated"


def test_frozen_variable_cp_registry_keeps_the_applicability_boundary_explicit():
    entry = get_model_provenance("performance.frozen_variable_cp_quasi1d")

    assert "cp/T" in entry.relation
    assert "A/A*=G*/G" in entry.relation
    assert "c*=p0/G*" in entry.relation
    assert "fixed ideal-gas composition" in entry.validity
    assert "Bezier-only" in entry.validity
    assert "MOC/Rao" in entry.validity
    assert "equilibrium chemistry" in entry.validity
    assert "Bartz" in entry.validity
    assert "Hall-Cd" in entry.validity
    assert "Property-grid refinement" in entry.notes
    assert "software identities only" in entry.notes
    assert entry.local_source.endswith("5f36b7c4ded79bb3e90754d0f81682f7a68014be.pdf")
    assert entry.validation_level == (
        "software_verified_conservation_only_not_property_or_hardware_validated"
    )


def test_spray_registry_covers_each_new_layer_without_blending_claims():
    assert SPRAY_MODEL_IDS <= set(MODEL_REGISTRY)
    spray = model_provenance_dict(subsystem="spray")
    assert set(spray) == SPRAY_MODEL_IDS
    json.dumps(spray)


def test_spray_carrier_drag_and_dispersion_state_one_way_closures_honestly():
    carrier = get_model_provenance("spray.prescribed_one_way_carrier")
    assert "one_way" in carrier.validity
    assert "feedback" in carrier.validity
    assert "energy" in carrier.validity
    assert "not_cfd_or_physically_validated" in carrier.validation_level

    drag = get_model_provenance("spray.schiller_naumann_drag")
    assert "0.687" in drag.relation
    assert "one_way" in drag.validity
    assert "deformation" in drag.validity
    assert drag.local_source.endswith("Tesi_dottorato_Cavalieri.pdf")
    assert "not_spray_drag_physically_validated" in drag.validation_level

    dispersion = get_model_provenance("spray.seeded_discrete_random_walk")
    assert "tau_e=C_L*k/epsilon" in dispersion.relation
    assert "explicit local RNG seed" in dispersion.relation
    assert "one_way" in dispersion.validity
    assert "publishes no seed" in dispersion.notes
    assert dispersion.local_source.endswith("radhakrishnan2021.pdf")
    assert "not_dispersion_physically_validated" in dispersion.validation_level


def test_wave_registry_versions_coefficients_and_keeps_rt_optional():
    wave_2018 = get_model_provenance("spray.wave_kh_radhakrishnan_2018")
    wave_2021 = get_model_provenance("spray.wave_kh_radhakrishnan_2021")
    rt = get_model_provenance("spray.rayleigh_taylor_openfoam")

    assert "0.4*Ta" in wave_2018.relation
    assert "0.87*We_g" in wave_2018.relation
    assert "0.4/0.87" in wave_2018.notes
    assert "0.4/0.865" in wave_2018.notes
    assert wave_2018.local_source.endswith("radhakrishnan2018 (1).pdf")

    assert "0.45*Ta" in wave_2021.relation
    assert "0.87*We_g" in wave_2021.relation
    assert "h=half" in wave_2021.validity
    assert "Tables 7/8 are author CFD, not experiment" in wave_2021.validity
    assert wave_2021.local_source.endswith("radhakrishnan2021.pdf")

    assert "optional and disabled by default" in rt.validity
    assert "absent from the Radhakrishnan" in rt.validity
    assert rt.local_source is None
    assert "not_pinned_or_physically_validated" in rt.validation_level


def test_spalding_and_weighted_statistics_registry_exposes_all_closures():
    evaporation = get_model_provenance("spray.spalding_evaporation_eq16")
    assert "Sh*D/d" in evaporation.relation
    assert "Bm=(Y_s-Y_inf)/(1-Y_s)" in evaporation.relation
    assert "no droplet-temperature/energy equation" in evaporation.validity
    assert "carrier-energy feedback" in evaporation.validity
    assert "Eq. 16" in evaporation.equation_ref
    assert "not_evaporation_or_energy_physically_validated" in (
        evaporation.validation_level
    )

    statistics = get_model_provenance("spray.weighted_smd_rosin_rammler")
    assert "sum(N_i*d_i^3)/sum(N_i*d_i^2)" in statistics.relation
    assert "mass survival" in statistics.relation
    assert "not an atomization or combustion model" in statistics.validity
    assert "decreasing mass-survival" in statistics.notes
    assert "not_spray_physically_validated" in statistics.validation_level


def test_spray_fixture_registry_separates_experiment_from_author_cfd():
    original = get_model_provenance(
        "spray.benchmark_radhakrishnan2018_water_air"
    )
    revision = get_model_provenance(
        "spray.benchmark_radhakrishnan2021_water_air"
    )
    tables = get_model_provenance(
        "spray.benchmark_radhakrishnan2021_tables_7_8"
    )

    assert "experimental targets" in original.validity
    assert "distinct publication revision" in original.notes
    assert original.validation_level == (
        "experimental_target_fixture_not_end_to_end_model_validation"
    )
    assert revision.validation_level == (
        "experimental_target_fixture_distinct_revision_not_end_to_end_model_validation"
    )
    assert original.local_source != revision.local_source

    assert "literature_reproduction_only" in tables.validity
    assert "not experimental" in tables.validity
    assert "never be promoted as physical validation" in tables.notes
    assert tables.validation_level == (
        "literature_reproduction_only_not_experimental_validation"
    )


def test_openfoam_registry_separates_static_vof_export_from_handoff_authority():
    exporter = get_model_provenance("spray.openfoam13_external_gap_vof")
    handoff = get_model_provenance("spray.vof_to_lagrangian_handoff")

    assert "mechanical opening is prescribed" in exporter.validity
    assert "internal center-gap" in exporter.validity
    assert "3.03 g/s" in exporter.notes
    assert "not_openfoam_run" in exporter.validation_level
    assert "mass/momentum" in handoff.relation
    assert "does not itself run or validate CFD" in handoff.validity
    assert "fail_closed_interface" in handoff.validation_level


def test_registry_local_sources_and_required_metadata_are_auditable():
    root = Path(__file__).resolve().parents[1]
    audit = audit_model_registry(root)
    assert audit["entry_count"] >= 25
    assert audit["missing_local_sources"] == []
    assert audit["published_without_local_source"] == []
    assert audit["incomplete_entries"] == []
    assert audit["unlabeled_repository_heuristics"] == []
    assert audit["passed"] is True
