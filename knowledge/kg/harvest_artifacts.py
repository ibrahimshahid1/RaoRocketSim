"""Observe existing design and injector CAD reports without creating test evidence.

Only two producer formats present in this checkout are recognized. A serialized
input/layout is a *knowledge snapshot*, not a producer configuration identity.
An existing STEP and its hash prove current bytes only. Stored CAD inspections
and manufacturing gates remain report assertions; no kernel or physical check
is performed here. ManufacturingProcess records describe reference plans, never
executed manufacturing, and no Measurement/Experiment/Calibration is inferred.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from . import REPO_ROOT
from .store import load, load_meta, save

HARVESTER = "kg.harvest_artifacts@1"
REPORT_NAMES = ("design_report_v2.json", "injector_manufacturing_report.json")
CAD_SUFFIXES = {".step", ".stp", ".stl", ".iges", ".igs", ".brep", ".ipt"}


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(prefix: str, *parts: str) -> str:
    return prefix + "_" + _digest(json.dumps(parts, ensure_ascii=False).encode())


def _canonical_digest(payload: dict) -> str:
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode())


def _nonfinite_locations(value, pointer: str = "") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [pointer]
    if isinstance(value, dict):
        return [p for key, item in value.items()
                for p in _nonfinite_locations(item, pointer + "/" + str(key))]
    if isinstance(value, list):
        return [p for index, item in enumerate(value)
                for p in _nonfinite_locations(item, pointer + "/" + str(index))]
    return []


def _safe_assertion(value):
    """Retain a nonfinite legacy assertion as an unavailable token, not a number."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"state": "UNAVAILABLE", "source_token": str(value),
                "reason": "nonfinite value in legacy producer report"}
    if isinstance(value, dict):
        return {key: _safe_assertion(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_assertion(item) for item in value]
    return value


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root)


def _discover(root: Path) -> list[Path]:
    """Scan known output reports, including ignored builds, without symlink walks."""
    directory = root / "builds"
    if not directory.is_dir() or not _inside(directory, root):
        return []
    result = []
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(d for d in directories
                                if not (Path(parent) / d).is_symlink())
        for name in REPORT_NAMES:
            if name in files:
                result.append(Path(parent) / name)
    return sorted(result)


def _source_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _reference_path(value: str, source: Path, root: Path) -> tuple[Path | None, str]:
    reference = Path(value)
    if reference.is_absolute():
        if not _inside(reference, root):
            return None, "OUTSIDE_REPOSITORY"
        return reference, "absolute_repository_path"
    root_candidate, local_candidate = root / reference, source.parent / reference
    existing = [p for p in (root_candidate, local_candidate)
                if _inside(p, root) and p.is_file()]
    unique = {str(p.resolve()): p for p in existing}
    if len(unique) > 1:
        return None, "AMBIGUOUS_RELATIVE_PATH"
    if unique:
        return next(iter(unique.values())), "existing_relative_path"
    # The existing producer emits root-relative builds/... paths; bare filenames
    # are relative to the report. Missing files keep that declared association.
    candidate = root_candidate if reference.parts[:1] == ("builds",) else local_candidate
    if not _inside(candidate, root):
        return None, "OUTSIDE_REPOSITORY"
    return candidate, "root_relative_path" if candidate == root_candidate else "report_relative_path"


def collect(paths: Iterable[str | Path] | None = None, *, repo_root: Path = REPO_ROOT) -> dict:
    """Return deterministic records/provenance/report without writing any store.

    ``paths=None`` discovers recognized filenames under builds. Explicit paths
    form a bounded refresh. All paths, including symlink targets, must remain in
    repo_root. A missing or invalid input is reported, never replaced by guesses.
    """
    root = Path(repo_root).resolve()
    full_scan = paths is None
    candidates = _discover(root) if full_scan else [_source_path(p, root) for p in paths]
    candidates = sorted(set(candidates), key=str)
    records: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    sources, issues = [], []

    def add(record: dict, source_info: dict, locator: str, statement: str) -> str:
        record = _safe_assertion(record)
        pid = "PROV_" + record["id"]
        record.update(harvester=HARVESTER, source_path=source_info["path"],
                      source_sha256=source_info["sha256"], json_pointer=locator,
                      provenance_id=pid, source_state="CURRENT", active=True)
        records[record["id"]] = record
        provenance[pid] = {
            "id": pid, "type": "ProvenanceRecord", "kind": "PROJECT_DERIVES",
            "subject": record["id"], "asserted_by": source_info["path"] + "#" + locator,
            "source_path": source_info["path"], "source_sha256": source_info["sha256"],
            "json_pointer": locator, "harvester": HARVESTER,
            "statement": statement, "verification": "json_locator_and_current_file_hash_only",
            "physical_validation": False,
        }
        if record.get("payload_selection"):
            provenance[pid]["payload_selection"] = record["payload_selection"]
        return record["id"]

    def link(src: str, dst: str, edge: str, source_info: dict, locator: str, **attrs) -> None:
        add({"id": _identity("ARTLINK", src, dst, edge), "type": "ArtifactLink",
             "from": src, "to": dst, "edge": edge, **attrs}, source_info, locator,
            "Report association only; no independently verified generation-time binding.")

    for source in candidates:
        # Check target before reading; external symlinks and traversal cannot
        # turn discovery into an arbitrary filesystem read.
        lexical = Path(os.path.abspath(source))
        relative = lexical.relative_to(root).as_posix() if lexical.is_relative_to(root) else None
        if not _inside(source, root):
            issues.append({"path": str(source), "state": "OUTSIDE_REPOSITORY"})
            if relative is not None:
                # A previously observed report replaced by an external symlink
                # must also lose active support on a bounded refresh.
                sources.append({"path": relative, "state": "OUTSIDE_REPOSITORY"})
            continue
        if relative is None:
            issues.append({"path": str(source), "state": "OUTSIDE_REPOSITORY"})
            continue
        info = {"path": relative, "state": "SOURCE_UNAVAILABLE"}
        sources.append(info)
        if source.name not in REPORT_NAMES:
            info["state"] = "UNRECOGNIZED_REPORT_FORMAT"
            continue
        try:
            raw = source.read_bytes()
            info["sha256"] = _digest(raw)
            # Legacy reports contain Infinity in unharvested structural output.
            # Keep that limitation visible; serialized configuration hashing
            # below still refuses nonfinite payloads.
            data = json.loads(raw)
        except OSError as exc:
            info["detail"] = type(exc).__name__
            continue
        except (ValueError, UnicodeError) as exc:
            info.update(state="INVALID_JSON", detail=type(exc).__name__)
            continue
        if not isinstance(data, dict):
            info["state"] = "UNRECOGNIZED_REPORT_FORMAT"
            continue
        nonfinite = _nonfinite_locations(data)
        if nonfinite:
            issues.append({"path": relative, "state": "NONFINITE_REPORT_VALUES",
                           "json_pointers": nonfinite})
        if source.name == "design_report_v2.json":
            if not isinstance(data.get("input"), dict) or not isinstance(data.get("metadata"), dict):
                info["state"] = "UNRECOGNIZED_REPORT_FORMAT"
                continue
            payload, scope, pointer = data["input"], "serialized_design_input", "/input"
            selection = {"kind": "json_pointer", "pointer": pointer}
        else:
            if not all(isinstance(data.get(k), dict)
                       for k in ("mechanical_spec", "resolved", "cad_export")):
                info["state"] = "UNRECOGNIZED_REPORT_FORMAT"
                continue
            keys = ("hydraulic_basis", "mechanical_spec", "resolved", "roles", "coaxial", "architecture")
            payload = {key: data[key] for key in keys if key in data}
            scope, pointer = "injector_layout_only", ""
            selection = {"kind": "top_level_fields", "fields": list(payload)}
        try:
            payload_sha = _canonical_digest(payload)
        except (ValueError, TypeError):
            info["state"] = "NONFINITE_CONFIGURATION"
            continue
        info["state"] = "CURRENT"
        manifest_id = add({
            "id": _identity("ARTMAN", relative, info["sha256"]), "type": "ArtifactManifest",
            "name": relative, "kind": "producer_report_snapshot", "state": "REPORT_ONLY",
            "report_format": source.name, "size_bytes": len(raw),
            "simulation_output": True, "hardware_qualified": False,
            "qualification_state": "NOT_QUALIFIED",
            "declared_status": data.get("status") or data.get("metadata", {}).get("design_status"),
        }, info, "", "Existing producer report observed; report contents are repository assertions.")
        metadata = data.get("metadata", {})
        config_id = add({
            "id": _identity("CFG", relative, info["sha256"], payload_sha),
            "type": "EngineConfiguration", "name": relative + "#" + pointer,
            "kind": scope, "scope": scope, "state": "REPORT_ONLY", "payload": payload,
            "snapshot_sha256": payload_sha, "snapshot_hash_schema": "kg.configuration_snapshot@1",
            "payload_selection": selection,
            "identity_basis": "canonical_serialized_payload_snapshot_only",
            "producer_configuration_id": metadata.get("configuration_id"),
            "producer_digest": None, "producer_binding_state": "UNKNOWN",
            "full_engine_identity_state": "UNKNOWN", "hardware_qualified": False,
            "qualification_state": "NOT_QUALIFIED",
        }, info, pointer, "Serialized repository input/layout snapshot; canonical hash is not a producer configuration digest.")
        link(config_id, manifest_id, "FOUND_IN", info, pointer)
        if source.name == "design_report_v2.json":
            # cad='none' and authoritative_cad='STEP' cannot prove any CAD exists.
            continue

        process_id = add({
            "id": _identity("MFG", relative, info["sha256"]),
            "type": "ManufacturingProcess", "name": relative + " machining reference",
            "kind": "machining_reference_layout", "state": "REPORT_ONLY",
            "execution_state": "UNAVAILABLE", "execution_evidence": None,
            "hardware_qualified": False, "qualification_state": "NOT_QUALIFIED",
            "declared_status": data.get("status"), "mechanical_spec": data["mechanical_spec"],
            "declared_manufacturing_gates": data.get("manufacturing_gates", []),
            "declared_flow_continuity": data.get("flow_continuity"),
            "unresolved_literature_basis": data.get("literature_basis", []),
            "verification_state": "UNKNOWN",
        }, info, "/mechanical_spec", "Reference machining layout and software gates; no performed manufacturing or physical inspection evidence.")
        link(process_id, config_id, "BOUND_TO_CONFIGURATION", info, "/mechanical_spec",
             binding_state="UNKNOWN", binding_basis="same_report_layout_snapshot")
        link(process_id, manifest_id, "FOUND_IN", info, "/mechanical_spec")
        cad = data["cad_export"]
        files = cad.get("files", {})
        if not isinstance(files, dict):
            issues.append({"path": relative, "state": "INVALID_CAD_FILE_MAP"})
            continue
        inspections = cad.get("inspection", {})
        for role, declared in sorted(files.items()):
            if not isinstance(declared, str) or Path(declared).suffix.lower() not in CAD_SUFFIXES:
                continue
            path, resolution = _reference_path(declared, source, root)
            observation = {"state": "UNAVAILABLE", "observed_sha256": None,
                           "size_bytes": None, "path": None}
            if path is not None:
                observation["path"] = path.resolve().relative_to(root).as_posix()
                try:
                    content = path.read_bytes()
                    observation.update(state="REPORT_ONLY", observed_sha256=_digest(content),
                                       size_bytes=len(content))
                except OSError as exc:
                    observation["unavailable_reason"] = type(exc).__name__
            else:
                observation["unavailable_reason"] = resolution
            if observation["state"] == "UNAVAILABLE":
                issues.append({"path": relative, "role": role,
                               "state": observation.get("unavailable_reason", "UNAVAILABLE")})
            inspection = inspections.get(role) if isinstance(inspections, dict) else None
            artifact_id = add({
                "id": _identity("CAD", relative, info["sha256"], str(role),
                                observation["observed_sha256"] or "UNAVAILABLE"),
                "type": "CADArtifact", "name": str(role), "kind": "reference_cad",
                "format": Path(declared).suffix[1:].lower(), "declared_path": declared,
                "path_resolution": resolution, **observation,
                "declared_cad_status": cad.get("status"),
                "declared_representation": cad.get("representation"),
                "declared_geometry_id": cad.get("geometry_id"),
                "declared_inspection": inspection,
                "inspection_verification_state": "UNKNOWN", "units_verification_state": "UNKNOWN",
                "generation_time_sha256": None, "generation_time_integrity_state": "UNKNOWN",
                "binding_state": "UNKNOWN", "binding_basis": "same_report_layout_snapshot",
                "hardware_qualified": False, "qualification_state": "NOT_QUALIFIED",
            }, info, "/cad_export/files/" + str(role),
                "CAD path declared by report; current bytes hashed separately. Stored inspection is unverified and is not a measurement.")
            link(artifact_id, config_id, "BOUND_TO_CONFIGURATION", info, "/cad_export/files/" + str(role),
                 binding_state="UNKNOWN", binding_basis="same_report_layout_snapshot")
            link(artifact_id, manifest_id, "FOUND_IN", info, "/cad_export/files/" + str(role))

    nodes = sorted(records.values(), key=lambda row: row["id"])
    report = {"harvester": HARVESTER, "full_scan": full_scan, "discovered_reports": len(candidates),
              "accepted_reports": sum(s["state"] == "CURRENT" for s in sources),
              "sources": sources, "issues": issues,
              "records_by_type": dict(sorted(Counter(r["type"] for r in nodes).items())),
              "physical_evidence_created": False}
    return {"records": nodes, "provenance": sorted(provenance.values(), key=lambda p: p["id"]),
            "report": report}


def harvest(paths: Iterable[str | Path] | None = None, *, repo_root: Path = REPO_ROOT) -> dict:
    """Refresh observed records, retaining old evidence as explicitly inactive.

    A bounded refresh only changes records sourced from attempted paths. A full
    refresh also invalidates reports which disappeared. Previous snapshots and
    their provenance remain inspectable; inactive links must not support active
    configuration claims.
    """
    result = collect(paths, repo_root=repo_root)
    current = {r["id"]: r for r in result["records"]}
    source_states = {s["path"]: s["state"] for s in result["report"]["sources"]}
    source_hashes = {s["path"]: s.get("sha256") for s in result["report"]["sources"]}
    existing = {r["id"]: r for r in load("artifacts")}
    for rid, record in existing.items():
        if rid in current or record.get("harvester") != HARVESTER:
            continue
        source = record.get("source_path")
        if not result["report"]["full_scan"] and source not in source_states:
            continue
        state = source_states.get(source, "SOURCE_UNAVAILABLE")
        if state == "CURRENT":
            state = "STALE_ARTIFACT_BYTES" if source_hashes[source] == record.get("source_sha256") else "STALE_SOURCE"
        record = dict(record, active=False, source_state=state)
        current[rid] = record
    # Records belonging to another producer and unrefreshed bounded sources are
    # preserved exactly, including any independently curated provenance.
    merged = {**existing, **current}
    save("artifacts", merged.values(), meta={"harvester": HARVESTER,
                                             "n": len(merged), "last_collection": result["report"]})
    prov = {p["id"]: p for p in load("provenance")}
    prov.update({p["id"]: p for p in result["provenance"]})
    save("provenance", prov.values(), meta={**load_meta("provenance"), "n": len(prov)})
    return result["report"]
