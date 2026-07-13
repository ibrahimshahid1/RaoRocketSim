"""Provenance-first spray benchmark fixtures and readiness gates.

This module deliberately keeps experimental validation targets separate from
author CFD outputs.  In particular, Radhakrishnan et al. (2021) Tables 7 and 8
are VOF/Lagrangian simulation results.  They are useful literature-reproduction
fixtures, but they are never promoted to experimental evidence here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "raosim" / "benchmark_data"
CASES_DIR = DATA_ROOT / "cases"

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_POLICY_ROLES = {
    "physical_cold_flow_validation_target",
    "literature_reproduction_only",
}

_ROW_COLUMNS: dict[str, tuple[str, ...]] = {
    "radhakrishnan2018_water_air_v1": (
        "case_id",
        "lopen_mm",
        "p_liquid_bar",
        "p_gas_bar",
        "mdot_air_g_s",
        "mdot_water_g_s",
        "tmr",
        "sheet_thickness_full_mm",
        "water_velocity_m_s",
        "wave_b0",
        "wave_b1",
        "spray_half_angle_experiment_deg",
        "spray_half_angle_experiment_uncertainty_deg",
        "smd_experiment_um",
        "smd_experiment_uncertainty_um",
    ),
    "radhakrishnan2021_water_air_validation_v1": (
        "case_id",
        "lopen_mm",
        "mdot_air_g_s",
        "mdot_water_g_s",
        "sheet_thickness_full_mm",
        "wave_b0",
        "wave_b1",
        "spray_half_angle_experiment_deg",
        "spray_half_angle_author_simulation_deg",
        "smd_experiment_um",
        "smd_experiment_uncertainty_um",
        "smd_author_simulation_um",
    ),
    "radhakrishnan2021_variable_area_lox_gch4_v1": (
        "case_id",
        "lopen_mm",
        "pintle_tip_angle_deg",
        "pc_mpa",
        "mdot_lox_g_s",
        "mdot_gch4_g_s",
        "tmr",
        "throttle_percent",
        "sheet_thickness_full_mm",
        "sheet_breakup_length_mm",
        "wave_b0",
        "wave_b1",
        "smd_author_simulation_um",
    ),
}

_EXPECTED_UNITS: dict[str, dict[str, str]] = {
    "radhakrishnan2018_water_air_v1": {
        "lopen_mm": "mm",
        "p_liquid_bar": "bar",
        "p_gas_bar": "bar",
        "mdot_air_g_s": "g/s",
        "mdot_water_g_s": "g/s",
        "tmr": "1",
        "sheet_thickness_full_mm": "mm",
        "water_velocity_m_s": "m/s",
        "wave_b0": "1",
        "wave_b1": "1",
        "spray_half_angle_experiment_deg": "deg",
        "spray_half_angle_experiment_uncertainty_deg": "deg",
        "smd_experiment_um": "um",
        "smd_experiment_uncertainty_um": "um",
    },
    "radhakrishnan2021_water_air_validation_v1": {
        "lopen_mm": "mm",
        "mdot_air_g_s": "g/s",
        "mdot_water_g_s": "g/s",
        "sheet_thickness_full_mm": "mm",
        "wave_b0": "1",
        "wave_b1": "1",
        "spray_half_angle_experiment_deg": "deg",
        "spray_half_angle_author_simulation_deg": "deg",
        "smd_experiment_um": "um",
        "smd_experiment_uncertainty_um": "um",
        "smd_author_simulation_um": "um",
    },
    "radhakrishnan2021_variable_area_lox_gch4_v1": {
        "lopen_mm": "mm",
        "pintle_tip_angle_deg": "deg",
        "pc_mpa": "MPa",
        "mdot_lox_g_s": "g/s",
        "mdot_gch4_g_s": "g/s",
        "tmr": "1",
        "throttle_percent": "percent",
        "sheet_thickness_full_mm": "mm",
        "sheet_breakup_length_mm": "mm",
        "wave_b0": "1",
        "wave_b1": "1",
        "smd_author_simulation_um": "um",
    },
}

_TABLES_7_8_SCHEMA = "radhakrishnan2021_variable_area_lox_gch4_v1"
_TABLES_7_8_SIMULATION_COLUMNS = {
    "sheet_thickness_full_mm",
    "sheet_breakup_length_mm",
    "wave_b0",
    "wave_b1",
    "smd_author_simulation_um",
}


class SprayBenchmarkError(ValueError):
    """Raised when a spray fixture fails provenance or schema validation."""


@dataclass(frozen=True)
class SprayBenchmarkRow:
    """One unit-explicit, provenance-indexed row from a spray fixture."""

    case_id: str
    values: Mapping[str, float]

    def __getitem__(self, field: str) -> float:
        return self.values[field]

    def as_dict(self) -> dict[str, float | str]:
        return {"case_id": self.case_id, **dict(self.values)}


@dataclass(frozen=True)
class SprayBenchmarkDataset:
    """Validated manifest, source identity, and typed numeric rows."""

    case_id: str
    row_schema: str
    manifest: Mapping[str, Any]
    rows: tuple[SprayBenchmarkRow, ...]
    manifest_path: Path
    source_path: Path
    source_sha256: str
    data_path: Path | None

    @property
    def validation_role(self) -> str:
        return str(self.manifest["validation_policy"]["role"])

    def row(self, case_id: str) -> SprayBenchmarkRow:
        for item in self.rows:
            if item.case_id == case_id:
                return item
        available = ", ".join(item.case_id for item in self.rows)
        raise KeyError(f"Unknown row '{case_id}'. Available: {available}")


@dataclass(frozen=True)
class SprayBenchmarkReadinessReport:
    """Evidence statement for what a fixture can and cannot validate."""

    case_id: str
    source_sha256_verified: bool
    validation_role: str
    strict_end_to_end_smd_validation_ready: bool
    available_for: tuple[str, ...]
    missing_publication_data: tuple[str, ...]
    blockers: tuple[str, ...]
    tables_7_8_are_experimental: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_sha256_verified": self.source_sha256_verified,
            "validation_role": self.validation_role,
            "strict_end_to_end_smd_validation_ready": (
                self.strict_end_to_end_smd_validation_ready
            ),
            "available_for": list(self.available_for),
            "missing_publication_data": list(self.missing_publication_data),
            "blockers": list(self.blockers),
            "tables_7_8_are_experimental": self.tables_7_8_are_experimental,
        }


@dataclass(frozen=True)
class SpraySMDComparison:
    """Origin-aware comparison to one published SMD target.

    Component agreement is not promoted to end-to-end validation when the
    fixture readiness report says the publication is incomplete.
    """

    dataset_case_id: str
    row_case_id: str
    target_kind: str
    target_origin: str
    predicted_smd_um: float
    target_smd_um: float
    uncertainty_um: float | None
    absolute_error_um: float
    relative_error: float
    within_published_uncertainty: bool | None
    component_target_agreement: bool | None
    strict_end_to_end_validated: bool
    physical_validation_credit: bool
    status: str
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_case_id": self.dataset_case_id,
            "row_case_id": self.row_case_id,
            "target_kind": self.target_kind,
            "target_origin": self.target_origin,
            "predicted_smd_um": self.predicted_smd_um,
            "target_smd_um": self.target_smd_um,
            "uncertainty_um": self.uncertainty_um,
            "absolute_error_um": self.absolute_error_um,
            "relative_error": self.relative_error,
            "within_published_uncertainty": self.within_published_uncertainty,
            "component_target_agreement": self.component_target_agreement,
            "strict_end_to_end_validated": self.strict_end_to_end_validated,
            "physical_validation_credit": self.physical_validation_credit,
            "status": self.status,
            "blockers": list(self.blockers),
        }


def list_spray_benchmark_cases(
    *, data_root: str | Path | None = None,
) -> list[str]:
    """Return only manifests explicitly tagged as spray benchmarks."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    cases_dir = root / "cases"
    if not cases_dir.is_dir():
        return []
    found: list[str] = []
    for path in sorted(cases_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SprayBenchmarkError(
                f"Cannot inspect benchmark manifest {path}: {exc}"
            ) from exc
        if payload.get("benchmark_kind") == "spray":
            found.append(path.stem)
    return found


def load_spray_benchmark(
    case_id: str,
    *,
    data_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> SprayBenchmarkDataset:
    """Load a spray fixture and enforce its schema, units, and PDF SHA-256."""
    if not _CASE_ID_RE.fullmatch(case_id):
        raise SprayBenchmarkError(f"Invalid spray benchmark case id: {case_id!r}")

    data_root_path = Path(data_root) if data_root is not None else DATA_ROOT
    repo_root_path = Path(repo_root) if repo_root is not None else REPO_ROOT
    manifest_path = _resolve_under(
        data_root_path, Path("cases") / f"{case_id}.json"
    )
    if not manifest_path.is_file():
        available = ", ".join(
            list_spray_benchmark_cases(data_root=data_root_path)
        ) or "none"
        raise SprayBenchmarkError(
            f"Unknown spray benchmark '{case_id}'. Available: {available}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SprayBenchmarkError(
            f"Cannot parse spray benchmark {manifest_path}: {exc}"
        ) from exc

    _validate_manifest(manifest, manifest_path)
    if manifest["case_id"] != case_id:
        raise SprayBenchmarkError(
            f"{manifest_path.name}: case_id does not match filename"
        )

    source = manifest["source"]
    source_path = _resolve_under(repo_root_path, Path(source["local_path"]))
    if not source_path.is_file():
        raise SprayBenchmarkError(
            f"{manifest_path.name}: missing local source {source_path}"
        )
    actual_sha = _sha256(source_path)
    if actual_sha != source["sha256"]:
        raise SprayBenchmarkError(
            f"{manifest_path.name}: local source SHA-256 mismatch for "
            f"{source_path}; expected {source['sha256']}, got {actual_sha}"
        )

    data = manifest["data"]
    data_path: Path | None = None
    if data["format"] == "csv":
        data_path = _resolve_under(data_root_path, Path(data["path"]))
        raw_rows = _load_csv_rows(data_path, data["row_schema"])
    else:
        raw_rows = data["rows"]

    rows = _validate_rows(
        raw_rows,
        row_schema=data["row_schema"],
        expected_count=data["expected_row_count"],
        manifest_name=manifest_path.name,
    )
    return SprayBenchmarkDataset(
        case_id=case_id,
        row_schema=data["row_schema"],
        manifest=_freeze_top_level(manifest),
        rows=rows,
        manifest_path=manifest_path,
        source_path=source_path,
        source_sha256=actual_sha,
        data_path=data_path,
    )


def benchmark_readiness_report(
    case: str | SprayBenchmarkDataset,
    *,
    data_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> SprayBenchmarkReadinessReport:
    """Build the explicit release/validation-readiness statement for a fixture."""
    dataset = (
        load_spray_benchmark(case, data_root=data_root, repo_root=repo_root)
        if isinstance(case, str)
        else case
    )
    readiness = dataset.manifest["readiness"]
    missing = tuple(str(value) for value in readiness["missing_publication_data"])
    strict_ready = bool(readiness["strict_end_to_end_smd_validation_ready"])
    blockers = tuple(f"Publication omits: {value}." for value in missing)

    tables_flag: bool | None = None
    if dataset.row_schema == _TABLES_7_8_SCHEMA:
        tables_flag = False
        blockers += (
            "Radhakrishnan 2021 Tables 7 and 8 are author VOF/Lagrangian "
            "simulation outputs, not experimental measurements.",
        )
    if not strict_ready and not blockers:
        blockers = (
            "The manifest blocks strict end-to-end SMD validation without "
            "sufficient publication evidence.",
        )

    return SprayBenchmarkReadinessReport(
        case_id=dataset.case_id,
        source_sha256_verified=True,
        validation_role=dataset.validation_role,
        strict_end_to_end_smd_validation_ready=strict_ready,
        available_for=tuple(str(value) for value in readiness["available_for"]),
        missing_publication_data=missing,
        blockers=blockers,
        tables_7_8_are_experimental=tables_flag,
    )


def compare_smd_to_benchmark(
    case: str | SprayBenchmarkDataset,
    row_case_id: str,
    predicted_smd_m: float,
    *,
    target_kind: str = "experiment",
) -> SpraySMDComparison:
    """Compare SMD while preserving experiment-versus-author-CFD origin."""

    dataset = load_spray_benchmark(case) if isinstance(case, str) else case
    predicted_m = float(predicted_smd_m)
    if not math.isfinite(predicted_m) or predicted_m <= 0.0:
        raise SprayBenchmarkError("predicted_smd_m must be finite and > 0")
    target_kind = str(target_kind).strip().lower()
    if target_kind not in {"experiment", "author_simulation"}:
        raise SprayBenchmarkError(
            "target_kind must be 'experiment' or 'author_simulation'"
        )
    row = dataset.row(row_case_id)
    values = row.values
    uncertainty: float | None = None
    if target_kind == "experiment":
        if "smd_experiment_um" not in values:
            raise SprayBenchmarkError(
                f"{dataset.case_id} has no experimental SMD target; Tables 7/8 "
                "are author simulation outputs"
            )
        target = float(values["smd_experiment_um"])
        uncertainty_value = values.get("smd_experiment_uncertainty_um")
        uncertainty = (
            None if uncertainty_value is None else float(uncertainty_value)
        )
        origin = "experiment"
        status = "experimental_component_target_comparison_only"
    else:
        if "smd_author_simulation_um" not in values:
            raise SprayBenchmarkError(
                f"{dataset.case_id} has no author-simulation SMD target"
            )
        target = float(values["smd_author_simulation_um"])
        origin = "author_lagrangian_simulation"
        status = "literature_reproduction_only_not_experimental_validation"

    predicted_um = predicted_m * 1.0e6
    error = abs(predicted_um - target)
    within = None if uncertainty is None else error <= uncertainty
    readiness = benchmark_readiness_report(dataset)
    blockers = readiness.blockers
    if target_kind == "author_simulation":
        blockers += (
            "Agreement with an author CFD output is not experimental validation.",
        )
    return SpraySMDComparison(
        dataset_case_id=dataset.case_id,
        row_case_id=row_case_id,
        target_kind=target_kind,
        target_origin=origin,
        predicted_smd_um=predicted_um,
        target_smd_um=target,
        uncertainty_um=uncertainty,
        absolute_error_um=error,
        relative_error=error / target,
        within_published_uncertainty=within,
        component_target_agreement=within,
        strict_end_to_end_validated=False,
        physical_validation_credit=False,
        status=status,
        blockers=blockers,
    )


def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    required = {
        "schema_version",
        "benchmark_kind",
        "case_id",
        "title",
        "dataset_revision",
        "source",
        "equation_variant",
        "definitions",
        "data",
        "units",
        "column_provenance",
        "uncertainty",
        "validation_policy",
        "readiness",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise SprayBenchmarkError(
            f"{path.name}: missing required keys {', '.join(missing)}"
        )
    if manifest["schema_version"] != 1:
        raise SprayBenchmarkError(f"{path.name}: unsupported schema_version")
    if manifest["benchmark_kind"] != "spray":
        raise SprayBenchmarkError(f"{path.name}: benchmark_kind must be spray")
    if not _CASE_ID_RE.fullmatch(str(manifest["case_id"])):
        raise SprayBenchmarkError(f"{path.name}: invalid case_id")
    if not str(manifest["dataset_revision"]).strip():
        raise SprayBenchmarkError(f"{path.name}: dataset_revision is empty")

    source = manifest["source"]
    for key in ("pdf", "local_path", "sha256", "doi", "citation", "references"):
        if not source.get(key):
            raise SprayBenchmarkError(f"{path.name}: source missing {key}")
    if not str(source["pdf"]).lower().endswith(".pdf"):
        raise SprayBenchmarkError(f"{path.name}: source.pdf must be a PDF")
    if not _SHA256_RE.fullmatch(str(source["sha256"])):
        raise SprayBenchmarkError(f"{path.name}: invalid source SHA-256")
    if not str(source["doi"]).startswith("10."):
        raise SprayBenchmarkError(f"{path.name}: invalid DOI")
    if not isinstance(source["references"], list) or not source["references"]:
        raise SprayBenchmarkError(f"{path.name}: source references are empty")
    for ref in source["references"]:
        if not isinstance(ref.get("pdf_page_1_based"), int) or (
            ref["pdf_page_1_based"] < 1
        ):
            raise SprayBenchmarkError(
                f"{path.name}: every source reference needs a positive PDF page"
            )
        if not str(ref.get("printed_page", "")).strip():
            raise SprayBenchmarkError(
                f"{path.name}: every source reference needs a printed page"
            )
        if not str(ref.get("data_origin", "")).strip():
            raise SprayBenchmarkError(
                f"{path.name}: every source reference needs data_origin"
            )

    equation = manifest["equation_variant"]
    for key in ("id", "source_ref"):
        if not str(equation.get(key, "")).strip():
            raise SprayBenchmarkError(
                f"{path.name}: equation_variant missing {key}"
            )
    for key in (
        "wave_wavelength_second_correction_coefficient",
        "wave_weber_denominator_coefficient",
    ):
        value = equation.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SprayBenchmarkError(
                f"{path.name}: equation_variant {key} must be finite and positive"
            )
    if "sheet_thickness_full_mm" not in manifest["definitions"]:
        raise SprayBenchmarkError(
            f"{path.name}: full-vs-half sheet-thickness definition is required"
        )

    data = manifest["data"]
    if data.get("format") not in {"csv", "inline"}:
        raise SprayBenchmarkError(f"{path.name}: data format must be csv or inline")
    row_schema = data.get("row_schema")
    if row_schema not in _ROW_COLUMNS:
        raise SprayBenchmarkError(f"{path.name}: unknown row_schema {row_schema!r}")
    if not isinstance(data.get("expected_row_count"), int) or (
        data["expected_row_count"] < 1
    ):
        raise SprayBenchmarkError(f"{path.name}: invalid expected_row_count")
    if data["format"] == "csv" and not str(data.get("path", "")).strip():
        raise SprayBenchmarkError(f"{path.name}: CSV data path is missing")
    if data["format"] == "inline" and not isinstance(data.get("rows"), list):
        raise SprayBenchmarkError(f"{path.name}: inline rows are missing")

    expected_units = _EXPECTED_UNITS[row_schema]
    if manifest["units"] != expected_units:
        raise SprayBenchmarkError(
            f"{path.name}: units do not exactly match schema {row_schema}"
        )
    provenance = manifest["column_provenance"]
    if set(provenance) != set(expected_units):
        raise SprayBenchmarkError(
            f"{path.name}: column_provenance must cover every numeric column"
        )
    for column, entry in provenance.items():
        if not str(entry.get("data_origin", "")).strip() or not str(
            entry.get("source_ref", "")
        ).strip():
            raise SprayBenchmarkError(
                f"{path.name}: incomplete provenance for {column}"
            )

    policy = manifest["validation_policy"]
    if policy.get("role") not in _VALID_POLICY_ROLES:
        raise SprayBenchmarkError(f"{path.name}: invalid validation policy role")
    if not isinstance(policy.get("prohibited_claims"), list) or not policy[
        "prohibited_claims"
    ]:
        raise SprayBenchmarkError(f"{path.name}: prohibited_claims are required")

    readiness = manifest["readiness"]
    if not isinstance(readiness.get("strict_end_to_end_smd_validation_ready"), bool):
        raise SprayBenchmarkError(f"{path.name}: readiness flag must be boolean")
    if readiness["strict_end_to_end_smd_validation_ready"]:
        raise SprayBenchmarkError(
            f"{path.name}: publication fixtures cannot claim strict end-to-end "
            "SMD readiness while carrier/property/parcel data are unavailable"
        )
    if not readiness.get("missing_publication_data"):
        raise SprayBenchmarkError(
            f"{path.name}: missing_publication_data must state the blockers"
        )
    if not readiness.get("available_for"):
        raise SprayBenchmarkError(f"{path.name}: available_for is empty")

    if row_schema == _TABLES_7_8_SCHEMA:
        if policy["role"] != "literature_reproduction_only":
            raise SprayBenchmarkError(
                f"{path.name}: Tables 7/8 must be literature_reproduction_only"
            )
        for column in _TABLES_7_8_SIMULATION_COLUMNS:
            origin = str(provenance[column]["data_origin"]).lower()
            if "experiment" in origin:
                raise SprayBenchmarkError(
                    f"{path.name}: Tables 7/8 column {column} cannot be "
                    "classified as experimental"
                )


def _load_csv_rows(path: Path, row_schema: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise SprayBenchmarkError(f"Missing spray benchmark CSV {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = list(_ROW_COLUMNS[row_schema])
            if reader.fieldnames != expected:
                raise SprayBenchmarkError(
                    f"{path.name}: columns do not match schema {row_schema}"
                )
            return [dict(row) for row in reader]
    except OSError as exc:
        raise SprayBenchmarkError(f"Cannot read {path}: {exc}") from exc


def _validate_rows(
    raw_rows: list[Mapping[str, Any]],
    *,
    row_schema: str,
    expected_count: int,
    manifest_name: str,
) -> tuple[SprayBenchmarkRow, ...]:
    columns = _ROW_COLUMNS[row_schema]
    if len(raw_rows) != expected_count:
        raise SprayBenchmarkError(
            f"{manifest_name}: expected {expected_count} rows, got {len(raw_rows)}"
        )
    parsed: list[SprayBenchmarkRow] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows, start=1):
        if set(raw) != set(columns):
            raise SprayBenchmarkError(
                f"{manifest_name}: row {index} fields do not match {row_schema}"
            )
        row_id = str(raw["case_id"]).strip()
        if not _CASE_ID_RE.fullmatch(row_id):
            raise SprayBenchmarkError(
                f"{manifest_name}: row {index} has invalid case_id {row_id!r}"
            )
        if row_id in seen:
            raise SprayBenchmarkError(
                f"{manifest_name}: duplicate row case_id {row_id!r}"
            )
        seen.add(row_id)

        values: dict[str, float] = {}
        for column in columns[1:]:
            raw_value = raw[column]
            if isinstance(raw_value, bool) or raw_value is None or raw_value == "":
                raise SprayBenchmarkError(
                    f"{manifest_name}: {row_id}.{column} must be numeric"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise SprayBenchmarkError(
                    f"{manifest_name}: {row_id}.{column} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise SprayBenchmarkError(
                    f"{manifest_name}: {row_id}.{column} is non-finite"
                )
            if column == "pintle_tip_angle_deg":
                if value < 0.0:
                    raise SprayBenchmarkError(
                        f"{manifest_name}: {row_id}.{column} must be non-negative"
                    )
            elif value <= 0.0:
                raise SprayBenchmarkError(
                    f"{manifest_name}: {row_id}.{column} must be positive"
                )
            if column == "throttle_percent" and value > 100.0:
                raise SprayBenchmarkError(
                    f"{manifest_name}: {row_id}.throttle_percent exceeds 100"
                )
            values[column] = value
        parsed.append(SprayBenchmarkRow(
            case_id=row_id,
            values=MappingProxyType(values),
        ))
    return tuple(parsed)


def _resolve_under(root: Path, relative: Path) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise SprayBenchmarkError(
            f"Benchmark path escapes configured root: {relative}"
        ) from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_top_level(value: dict[str, Any]) -> Mapping[str, Any]:
    """Prevent accidental replacement of top-level manifest sections."""
    return MappingProxyType(value)


__all__ = [
    "SprayBenchmarkDataset",
    "SprayBenchmarkError",
    "SprayBenchmarkReadinessReport",
    "SprayBenchmarkRow",
    "SpraySMDComparison",
    "benchmark_readiness_report",
    "compare_smd_to_benchmark",
    "list_spray_benchmark_cases",
    "load_spray_benchmark",
]
