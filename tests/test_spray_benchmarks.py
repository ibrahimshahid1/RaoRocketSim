"""Provenance and readiness contracts for the Radhakrishnan spray fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from raosim.spray.benchmarks import (
    SprayBenchmarkError,
    benchmark_readiness_report,
    compare_smd_to_benchmark,
    list_spray_benchmark_cases,
    load_spray_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "raosim" / "benchmark_data"
EXPECTED_CASES = {
    "radhakrishnan2018_water_air",
    "radhakrishnan2021_water_air_validation",
    "radhakrishnan2021_variable_area_lox_gch4",
}


def test_lists_only_spray_manifests_and_sha_checks_sources():
    assert set(list_spray_benchmark_cases()) == EXPECTED_CASES

    expected_hashes = {
        "radhakrishnan2018_water_air": (
            "512f152720e604770e28e5e684700d9d22e7450a065e066f58fe305b0273e266"
        ),
        "radhakrishnan2021_water_air_validation": (
            "b2550805f9cd9837fcfac67243812665a3a3be832fb32d018638722c63c2c784"
        ),
        "radhakrishnan2021_variable_area_lox_gch4": (
            "b2550805f9cd9837fcfac67243812665a3a3be832fb32d018638722c63c2c784"
        ),
    }
    for case_id, expected_hash in expected_hashes.items():
        dataset = load_spray_benchmark(case_id)
        assert dataset.source_path.is_file()
        assert dataset.source_sha256 == expected_hash


def test_2018_water_air_rows_preserve_experiment_and_author_outputs():
    dataset = load_spray_benchmark("radhakrishnan2018_water_air")
    assert dataset.row_schema == "radhakrishnan2018_water_air_v1"
    assert len(dataset.rows) == 4
    assert dataset.validation_role == "physical_cold_flow_validation_target"

    case_1 = dataset.row("case_1")
    assert case_1["lopen_mm"] == pytest.approx(0.2)
    assert case_1["mdot_air_g_s"] == pytest.approx(3.03)
    assert case_1["mdot_water_g_s"] == pytest.approx(22.9)
    assert case_1["sheet_thickness_full_mm"] == pytest.approx(0.2)
    assert case_1["wave_b0"] == pytest.approx(4.92)
    assert case_1["wave_b1"] == pytest.approx(0.989)
    assert case_1["spray_half_angle_experiment_deg"] == pytest.approx(24.4)
    assert case_1["smd_experiment_um"] == pytest.approx(108.10)
    assert case_1["smd_experiment_uncertainty_um"] == pytest.approx(3.14)
    assert dataset.row("case_4")["sheet_thickness_full_mm"] == pytest.approx(
        0.568
    )

    origins = dataset.manifest["column_provenance"]
    assert origins["smd_experiment_um"]["data_origin"] == "experiment"
    assert origins["sheet_thickness_full_mm"]["data_origin"] == (
        "author_vof_simulation"
    )
    assert dataset.manifest["equation_variant"][
        "wave_wavelength_second_correction_coefficient"
    ] == pytest.approx(0.4)
    assert dataset.manifest["equation_variant"][
        "wave_weber_denominator_coefficient"
    ] == pytest.approx(0.87)


def test_2021_water_air_is_a_separate_publication_revision():
    original = load_spray_benchmark("radhakrishnan2018_water_air").row("case_1")
    revision = load_spray_benchmark(
        "radhakrishnan2021_water_air_validation"
    )
    row = revision.row("water_air_validation")

    assert len(revision.rows) == 1
    assert row["sheet_thickness_full_mm"] == pytest.approx(0.125)
    assert row["wave_b0"] == pytest.approx(3.059)
    assert row["wave_b1"] == pytest.approx(1.26)
    assert row["spray_half_angle_experiment_deg"] == pytest.approx(24.4)
    assert row["spray_half_angle_author_simulation_deg"] == pytest.approx(24.1)
    assert row["smd_experiment_um"] == pytest.approx(108.1)
    assert row["smd_author_simulation_um"] == pytest.approx(113.16)
    assert row["smd_experiment_uncertainty_um"] == pytest.approx(3.13)

    assert row["sheet_thickness_full_mm"] != original[
        "sheet_thickness_full_mm"
    ]
    assert revision.manifest["revision_relationship"]["must_remain_separate"]
    assert revision.manifest["equation_variant"][
        "wave_wavelength_second_correction_coefficient"
    ] == pytest.approx(0.45)
    assert revision.manifest["equation_variant"][
        "wave_weber_denominator_coefficient"
    ] == pytest.approx(0.87)


def test_2021_tables_3_7_8_are_literature_reproduction_not_experiment():
    dataset = load_spray_benchmark(
        "radhakrishnan2021_variable_area_lox_gch4"
    )
    assert len(dataset.rows) == 6
    assert dataset.validation_role == "literature_reproduction_only"

    case_4 = dataset.row("case_4")
    assert case_4["pc_mpa"] == pytest.approx(2.0)
    assert case_4["sheet_thickness_full_mm"] == pytest.approx(0.152)
    assert case_4["sheet_breakup_length_mm"] == pytest.approx(3.677)
    assert case_4["wave_b0"] == pytest.approx(3.97)
    assert case_4["wave_b1"] == pytest.approx(8.58)
    assert case_4["smd_author_simulation_um"] == pytest.approx(70.0)
    assert dataset.row("case_5")["pintle_tip_angle_deg"] == pytest.approx(0.0)

    provenance = dataset.manifest["column_provenance"]
    for column in (
        "sheet_thickness_full_mm",
        "sheet_breakup_length_mm",
        "wave_b0",
        "wave_b1",
        "smd_author_simulation_um",
    ):
        assert "experiment" not in provenance[column]["data_origin"]
    assert provenance["smd_author_simulation_um"]["data_origin"] == (
        "author_lagrangian_simulation"
    )
    uncertainty = dataset.manifest["uncertainty"]["smd_author_simulation_um"]
    assert uncertainty["plus_minus_min_um"] == pytest.approx(4.0)
    assert uncertainty["plus_minus_max_um"] == pytest.approx(8.0)


@pytest.mark.parametrize("case_id", sorted(EXPECTED_CASES))
def test_readiness_blocks_strict_end_to_end_smd_validation(case_id):
    report = benchmark_readiness_report(case_id)
    assert report.source_sha256_verified
    assert not report.strict_end_to_end_smd_validation_ready
    assert report.missing_publication_data
    assert report.blockers
    joined = " ".join(report.blockers).lower()
    assert "carrier" in joined
    assert "parcel" in joined
    json.dumps(report.as_dict())


def test_tables_7_8_readiness_explicitly_says_not_experimental():
    report = benchmark_readiness_report(
        "radhakrishnan2021_variable_area_lox_gch4"
    )
    assert report.tables_7_8_are_experimental is False
    assert "not experimental" in " ".join(report.blockers).lower()


def test_smd_comparison_preserves_experiment_and_author_cfd_meaning():
    experimental = compare_smd_to_benchmark(
        "radhakrishnan2018_water_air", "case_1", 108.10e-6
    )
    assert experimental.target_origin == "experiment"
    assert experimental.within_published_uncertainty is True
    assert experimental.component_target_agreement is True
    assert experimental.strict_end_to_end_validated is False
    assert experimental.physical_validation_credit is False

    author_cfd = compare_smd_to_benchmark(
        "radhakrishnan2021_variable_area_lox_gch4",
        "case_1",
        27.0e-6,
        target_kind="author_simulation",
    )
    assert author_cfd.target_origin == "author_lagrangian_simulation"
    assert author_cfd.absolute_error_um == pytest.approx(0.0)
    assert author_cfd.within_published_uncertainty is None
    assert author_cfd.physical_validation_credit is False
    assert "not experimental" in " ".join(author_cfd.blockers).lower()
    with pytest.raises(SprayBenchmarkError, match="no experimental SMD"):
        compare_smd_to_benchmark(
            "radhakrishnan2021_variable_area_lox_gch4", "case_1", 27e-6
        )


def test_loader_rejects_source_hash_mismatch(tmp_path):
    data_root = _copy_case_to_tmp(tmp_path, "radhakrishnan2018_water_air")
    manifest_path = data_root / "cases" / "radhakrishnan2018_water_air.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SprayBenchmarkError, match="SHA-256 mismatch"):
        load_spray_benchmark(
            "radhakrishnan2018_water_air",
            data_root=data_root,
            repo_root=REPO_ROOT,
        )


def test_loader_rejects_duplicate_row_ids(tmp_path):
    data_root = _copy_case_to_tmp(tmp_path, "radhakrishnan2018_water_air")
    csv_path = data_root / "curves" / "radhakrishnan2018_water_air.csv"
    text = csv_path.read_text(encoding="utf-8").replace(
        "case_2,0.4", "case_1,0.4", 1
    )
    csv_path.write_text(text, encoding="utf-8")

    with pytest.raises(SprayBenchmarkError, match="duplicate row case_id"):
        load_spray_benchmark(
            "radhakrishnan2018_water_air",
            data_root=data_root,
            repo_root=REPO_ROOT,
        )


def test_loader_rejects_unit_drift(tmp_path):
    case_id = "radhakrishnan2021_water_air_validation"
    data_root = _copy_case_to_tmp(tmp_path, case_id)
    manifest_path = data_root / "cases" / f"{case_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["units"]["smd_experiment_um"] = "mm"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SprayBenchmarkError, match="units do not exactly match"):
        load_spray_benchmark(case_id, data_root=data_root, repo_root=REPO_ROOT)


def test_loader_rejects_experimental_relabeling_of_tables_7_8(tmp_path):
    case_id = "radhakrishnan2021_variable_area_lox_gch4"
    data_root = _copy_case_to_tmp(tmp_path, case_id)
    manifest_path = data_root / "cases" / f"{case_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["column_provenance"]["smd_author_simulation_um"][
        "data_origin"
    ] = "experiment"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SprayBenchmarkError, match="cannot be classified"):
        load_spray_benchmark(case_id, data_root=data_root, repo_root=REPO_ROOT)


def _copy_case_to_tmp(tmp_path: Path, case_id: str) -> Path:
    data_root = tmp_path / "benchmark_data"
    cases_dir = data_root / "cases"
    curves_dir = data_root / "curves"
    cases_dir.mkdir(parents=True)
    curves_dir.mkdir(parents=True)
    source_manifest = DATA_ROOT / "cases" / f"{case_id}.json"
    shutil.copy2(source_manifest, cases_dir / source_manifest.name)

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest["data"]["format"] == "csv":
        source_csv = DATA_ROOT / manifest["data"]["path"]
        shutil.copy2(source_csv, curves_dir / source_csv.name)
    return data_root
