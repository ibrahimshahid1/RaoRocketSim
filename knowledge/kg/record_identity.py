"""Preserve duplicate survey facets without manufacturing scientific support.

Only an exact identity can share a record. Historical list unions are explicitly
separate from active model/test/constant associations. Ambiguous ID collisions
raise with the original records attached; no guessed ID/link remapping occurs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FACET_FIELDS = {"survey_variants", "survey_conflicts", "historical_survey_associations",
                "survey_variants_scope", "survey_identity_state", "survey_recovery"}
ASSOCIATION_FIELDS = ("model_ids", "registry_ids", "test_ids", "constant_ids", "cited_document_ids",
                      "equation_ids", "gate_ids")
SCOPE = "Retained historical survey assertions only; not current scientific support, executed tests, or validated implementation associations."


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def identity(record: dict) -> tuple:
    """Exact identity; free-text constant names are not silently normalized."""
    kind = record.get("type")
    if kind == "SoftwareComponent":
        return kind, record.get("file"), record.get("qualname")
    if kind == "EmpiricalConstant":
        if record.get("logical_key"):
            return kind, record.get("harvest_owner"), record["logical_key"]
        return kind, record.get("file"), record.get("component_id"), record.get("name")
    if kind == "SoftwareLink":
        return kind, record.get("from"), record.get("to"), record.get("edge")
    return kind, _canonical({k: v for k, v in record.items() if k not in FACET_FIELDS})


class IdentityCollisionError(ValueError):
    def __init__(self, collisions: list[dict]):
        self.collisions = copy.deepcopy(collisions)
        super().__init__("Unresolved duplicate record identities: " + ", ".join(c["id"] for c in collisions)
                         + ". Original records retained in exception; no IDs or links were guessed.")


def consolidate(records: list[dict], *, preferred: dict[str, dict] | None = None,
                origin: dict | None = None) -> tuple[list[dict], dict]:
    """One record per exact identity, with inspectable original survey facets.

    ``preferred`` protects current survivors during historical recovery: their
    values, basis classes, locators, active associations and histories win in
    every top-level field. Otherwise the last input facet remains the primary
    assertion, matching the previous survivor policy but retaining other facets.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record["id"]].append(record)
    collisions = []
    for rid, group in groups.items():
        identities = {identity(r) for r in group}
        survivor = (preferred or {}).get(rid)
        if len(identities) != 1 or (survivor is not None and identity(survivor) not in identities):
            collisions.append({"id": rid, "identities": [list(v) for v in sorted(identities, key=repr)],
                               "current_identity": list(identity(survivor)) if survivor else None,
                               "records": group})
    if collisions:
        raise IdentityCollisionError(collisions)

    out = []
    stats = Counter()
    for rid, group in sorted(groups.items()):
        base = copy.deepcopy((preferred or {}).get(rid, group[-1]))
        if len(group) == 1 and not group[0].get("survey_variants"):
            out.append(base)
            continue
        stats["duplicate_groups"] += int(len(group) > 1)
        stats["extra_records"] += max(0, len(group) - 1)
        variants = {}
        for r in [base, *group]:
            for variant in r.get("survey_variants", []):
                variants[variant["record_sha256"]] = copy.deepcopy(variant)
        occurrences = Counter()
        snapshots = {}
        for r in group:
            snapshot = {k: v for k, v in r.items() if k not in FACET_FIELDS}
            sha = _sha(snapshot)
            snapshots[sha] = snapshot
            occurrences[sha] += 1
        for sha, snapshot in snapshots.items():
            variant = variants.setdefault(sha, {"record_sha256": sha, "record": copy.deepcopy(snapshot),
                                                "state": "HISTORICAL_SURVEY_ASSERTION", "origins": []})
            variant["occurrences"] = max(variant.get("occurrences", 0), occurrences[sha])
            if origin is not None and origin not in variant["origins"]:
                variant["origins"].append(copy.deepcopy(origin))
        base["survey_variants"] = [variants[k] for k in sorted(variants)]
        base["survey_variants_scope"] = SCOPE
        base["survey_identity_state"] = "SAME_EXACT_IDENTITY_MULTIPLE_SURVEY_FACETS"
        conflicts = {}
        fields = sorted({k for v in variants.values() for k in v["record"]})
        absent = {"state": "FIELD_NOT_PRESENT"}
        for key in fields:
            values = {}
            for sha, variant in variants.items():
                value = variant["record"].get(key, absent)
                bucket = values.setdefault(_canonical(value), {"value": value, "variant_sha256": []})
                bucket["variant_sha256"].append(sha)
            if len(values) > 1:
                conflicts[key] = [values[k] for k in sorted(values)]
        base["survey_conflicts"] = conflicts
        associations = {"state": "HISTORICAL_SURVEY_ASSERTIONS_ONLY", "scope": SCOPE}
        for key in ASSOCIATION_FIELDS:
            values = {_canonical(v): v for variant in variants.values() for v in variant["record"].get(key, [])}
            if values:
                associations[key] = [values[k] for k in sorted(values)]
        base["historical_survey_associations"] = associations
        stats["distinct_variants"] += len(variants)
        stats["conflicting_fields"] += len(conflicts)
        out.append(base)
    return out, dict(stats)


def recover_legacy_duplicates(archive: Path | str | dict | None = None, *, write: bool = True) -> dict:
    """Restore archived facets onto matching current survivors only.

    Historical identity collisions are quarantined in the archive and reported;
    they cannot overwrite current values/basis or create active support edges.
    No missing current node is recreated from a historical scientific assertion.
    """
    from . import store
    if archive is None:
        archive = store.REPORT_DIR / "legacy_duplicate_records_2026-10-15.json"
    payload = archive if isinstance(archive, dict) else json.loads(Path(archive).read_text())
    if payload.get("authority") != "HISTORICAL_RECOVERY_ONLY":
        raise ValueError("Recovery archive must be explicitly HISTORICAL_RECOVERY_ONLY.")
    rows = store.load("software_links")
    # Existing duplicates must be resolved explicitly, never dictionary-collapsed.
    current = {r["id"]: r for r in store._unique_records(rows, "id")}
    recovered, missing, quarantined = [], [], []
    origin = {"kind": "HISTORICAL_RECOVERY_ONLY", **payload.get("source", {})}
    for group in payload.get("groups", []):
        rid = group["id"]
        if rid not in current:
            missing.append(rid)
            continue
        try:
            merged, _ = consolidate(group["records"], preferred={rid: current[rid]}, origin=origin)
        except IdentityCollisionError as exc:
            quarantined.append({"id": rid, "state": "HISTORICAL_IDENTITY_COLLISION_QUARANTINED",
                                "identities": exc.collisions[0]["identities"],
                                "current_identity": list(identity(current[rid]))})
            continue
        current[rid] = merged[0]
        recovered.append(rid)
    updated = [current[r["id"]] for r in rows]
    changed = sum(a != b for a, b in zip(rows, updated))
    if write and changed:
        store.save("software_links", updated, meta=store.load_meta("software_links"))
    return {"authority": "HISTORICAL_RECOVERY_ONLY", "write": write, "archive_groups": len(payload.get("groups", [])),
            "recovered_groups": len(recovered), "records_updated": changed, "recovered_ids": recovered,
            "missing_current_ids": missing, "quarantined_groups": quarantined,
            "scope": SCOPE}
