"""Phase-2 held-out CEA value/gradient gate without a RocketCEA dependency."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("jax")

from raosim.mdo.properties import (
    DEFAULT_CEA_VALIDATION_TOLERANCES,
    PROPERTY_TABLE_SCHEMA_VERSION,
    load_chamber_surfaces,
    rocketcea_card_notes,
    rocketcea_card_names,
    sample_cea_tables,
    save_tables,
    validate_cea_surface_holdout,
)
from scripts.sample_cea_surface import main as sampler_main


def _linear_backend(*, Pc: float, OF: float, oxidizer: str, fuel: str):
    """Smooth manufactured RocketCEA oracle, linear in both fit axes."""

    pressure_mpa = Pc / 1.0e6
    return {
        "gamma": 1.12 + 0.002 * pressure_mpa + 0.015 * OF,
        "Tc": 3050.0 + 12.0 * pressure_mpa + 85.0 * OF,
        "R_gas": 315.0 + 0.25 * pressure_mpa + 7.0 * OF,
    }


def _write_table(path, *, wrong_of_slope: bool = False):
    Pc = np.linspace(1.5e6, 6.0e6, 7)
    OF = np.linspace(2.0, 4.0, 8)
    tables = sample_cea_tables(
        Pc, OF, oxidizer="LOX", fuel="LCH4", sampler=_linear_backend
    )
    if wrong_of_slope:
        tables["Tc"] = tables["Tc"] + 300.0 * (
            OF[None, :] - OF[0]
        )
    save_tables(
        str(path), tables, oxidizer="LOX", fuel="LCH4"
    )
    return path


def test_heldout_gate_checks_values_gradients_and_retains_table_identity(tmp_path):
    path = _write_table(tmp_path / "methalox.npz")
    report = validate_cea_surface_holdout(
        str(path),
        expected_propellant="LOX/LCH4",
        sampler=_linear_backend,
        max_points=11,
        seed=91,
    )

    assert report.passed
    assert report.table_schema_version == PROPERTY_TABLE_SCHEMA_VERSION
    assert report.table_content_sha256 == load_chamber_surfaces(
        str(path)
    ).content_sha256
    assert report.rocketcea_fuel == "CH4"
    assert any("no stock LCH4" in note for note in report.backend_notes)
    assert all(summary.passed for summary in report.fields.values())
    assert all(summary.value.checked == 11 for summary in report.fields.values())

    surfaces = load_chamber_surfaces(str(path))
    pc_nodes = np.asarray(surfaces.gamma.xg)
    of_nodes = np.asarray(surfaces.gamma.yg)
    for pc, of in report.holdout_points:
        assert not np.any(pc == pc_nodes)
        assert not np.any(of == of_nodes)

    serialized = report.as_dict()
    assert serialized["table"]["content_sha256"] == report.table_content_sha256
    assert serialized["table"]["schema_version"] == PROPERTY_TABLE_SCHEMA_VERSION
    assert set(serialized["tolerances"]) == {"gamma", "Tc", "R_gas"}
    for field in serialized["tolerances"].values():
        assert set(field) == {
            "value_rtol", "value_atol", "dPc_rtol", "dPc_atol",
            "dOF_rtol", "dOF_atol",
        }


def test_gate_rejects_a_hash_valid_table_with_wrong_value_and_gradient(tmp_path):
    path = _write_table(tmp_path / "wrong.npz", wrong_of_slope=True)
    report = validate_cea_surface_holdout(
        str(path), sampler=_linear_backend, max_points=9, seed=7
    )

    assert not report.passed
    assert report.fields["Tc"].value.failures > 0
    assert report.fields["Tc"].dOF.failures > 0
    # The file itself is internally valid and hash-verified: this failure is
    # specifically the independent held-out backend comparison.
    assert report.table_content_sha256 == load_chamber_surfaces(
        str(path)
    ).content_sha256


def test_cli_returns_nonzero_and_writes_hash_linked_report_on_gate_failure(
    tmp_path,
):
    path = _write_table(tmp_path / "wrong_cli.npz", wrong_of_slope=True)
    report_path = tmp_path / "validation.json"
    exit_code = sampler_main(
        [
            "--oxidizer", "LOX",
            "--fuel", "LCH4",
            "--out", str(path),
            "--validate-only",
            "--holdout-count", "5",
            "--report", str(report_path),
        ],
        sampler=_linear_backend,
    )

    assert exit_code != 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["table"]["content_sha256"] == load_chamber_surfaces(
        str(path)
    ).content_sha256
    assert payload["propellant"]["rocketcea_fuel"] == "CH4"


def test_cli_can_sample_and_pass_with_injected_backend(tmp_path):
    path = tmp_path / "generated.npz"
    report_path = tmp_path / "generated.validation.json"
    exit_code = sampler_main(
        [
            "--oxidizer", "LOX", "--fuel", "LCH4",
            "--pc-min", "1500000", "--pc-max", "6000000", "--n-pc", "6",
            "--of-min", "2.0", "--of-max", "4.0", "--n-of", "7",
            "--out", str(path), "--holdout", "--holdout-count", "6",
            "--report", str(report_path),
        ],
        sampler=_linear_backend,
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["table"]["schema_version"] == PROPERTY_TABLE_SCHEMA_VERSION


def test_catalog_names_use_stock_rocketcea_cards_without_changing_identity():
    assert rocketcea_card_names("LOX", "RP-1") == ("LOX", "RP1")
    assert rocketcea_card_names("LOX", "LCH4") == ("LOX", "CH4")
    assert rocketcea_card_names("LOX", "LH2") == ("LOX", "LH2")
    assert rocketcea_card_names("N2O4", "MMH") == ("N2O4", "MMH")
    assert rocketcea_card_names("N2O", "Ethanol") == ("N2O", "Ethanol")
    assert any(
        "N2O_nbp" in note for note in rocketcea_card_notes("N2O", "Ethanol")
    )


def test_explicit_holdout_point_cannot_reuse_a_fit_node(tmp_path):
    path = _write_table(tmp_path / "nodes.npz")
    with pytest.raises(ValueError, match="fit-grid node"):
        validate_cea_surface_holdout(
            str(path), sampler=_linear_backend,
            points=[(2.25e6, 2.2)],
        )


def test_zero_reference_with_pure_relative_gradient_tolerance_is_safe(tmp_path):
    def constant_backend(*, Pc: float, OF: float, oxidizer: str, fuel: str):
        return {"gamma": 1.2, "Tc": 3200.0, "R_gas": 360.0}

    Pc = np.linspace(1.5e6, 6.0e6, 5)
    OF = np.linspace(2.0, 4.0, 5)
    exact = sample_cea_tables(
        Pc, OF, oxidizer="LOX", fuel="LCH4", sampler=constant_backend
    )
    exact_path = tmp_path / "zero_exact.npz"
    save_tables(
        str(exact_path), exact, oxidizer="LOX", fuel="LCH4"
    )
    pure_relative = dict(DEFAULT_CEA_VALIDATION_TOLERANCES)
    pure_relative["gamma"] = replace(
        pure_relative["gamma"], dPc_atol=0.0, dPc_rtol=0.1
    )

    exact_report = validate_cea_surface_holdout(
        str(exact_path),
        sampler=constant_backend,
        points=[(2.1e6, 2.3)],
        tolerances=pure_relative,
    )
    assert exact_report.fields["gamma"].dPc.passed
    assert exact_report.fields["gamma"].dPc.max_tolerance_ratio == 0.0

    wrong = {name: np.asarray(value).copy() for name, value in exact.items()}
    wrong["gamma"] += 0.01 * (
        (Pc[:, None] - Pc[0]) / (Pc[-1] - Pc[0])
    )
    wrong_path = tmp_path / "zero_mismatch.npz"
    save_tables(
        str(wrong_path), wrong, oxidizer="LOX", fuel="LCH4"
    )
    wrong_report = validate_cea_surface_holdout(
        str(wrong_path),
        sampler=constant_backend,
        points=[(2.1e6, 2.3)],
        tolerances=pure_relative,
    )
    metric = wrong_report.fields["gamma"].dPc
    assert not metric.passed
    assert np.isfinite(metric.max_tolerance_ratio)
    assert metric.max_tolerance_ratio == np.finfo(float).max
