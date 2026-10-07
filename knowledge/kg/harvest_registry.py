"""Harvest ``raosim.model_registry`` into Model records and software links.

The registry is the repository's own statement of what each implemented model
is, where its source is, and how far it has been verified.  Its authority is
therefore ``PROJECT_DERIVES`` - it is the implementation-state authority, not
the scientific authority.  The scientific authority remains the cited PDF,
which is resolved here to a ``Document`` id from ``knowledge/manifest.json``
by exact path match (never by name similarity).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT
from .schema import EXTRACTION_VERSION, node_id
from .store import load, save

sys.path.insert(0, str(REPO_ROOT))


def _status_to_basis(status: str) -> str:
    s = status.lower()
    if "heuristic" in s or "policy" in s or "schedule" in s or "packaging" in s:
        return "REPOSITORY_HEURISTIC"
    if "derived" in s:
        return "PROJECT_DERIVED"
    if "calibrat" in s:
        return "EMPIRICALLY_CALIBRATED"
    if "published" in s or "source_port" in s:
        return "LITERATURE_DERIVED"
    if "external" in s or "interface" in s or "verification" in s:
        return "IMPLEMENTATION_CHOICE"
    return "UNKNOWN_BASIS"


def _promotion(status: str, validation_level: str) -> str:
    s = f"{status} {validation_level}".lower()
    if "not_implemented" in s or "planned" in s:
        return "planned_not_implemented"
    if "screen" in s:
        return "screening"
    if "one_way" in s or "report" in s:
        return "report_only"
    if "oracle" in s:
        return "reference_oracle"
    return "screening" if "assumption_only" in s else "optimizer_coupled"


def _verification_kind(verification: str, validation_level: str) -> list[str]:
    text = f"{verification} {validation_level}".lower()
    kinds = []
    if "parity" in text or "jax" in text:
        kinds.append("numpy_jax_parity")
    if "benchmark" in text or "reference_matched" in text or "literature_reproduction" in text or "reproduc" in text:
        kinds.append("published_benchmark_reproduction")
    if "cold_flow" in text and "validated" in text and "not" not in text:
        kinds.append("experimental_validation")
    if "identity" in text or "unit" in text or "closure" in text or "regression" in text or "test" in text:
        kinds.append("unit_test")
    if "topology" in text or "round" in text:
        kinds.append("cad_round_trip")
    return kinds or ["unit_test"]


def harvest() -> dict:
    from raosim.model_registry import MODEL_REGISTRY

    docs_by_path = {}
    for d in load("manifest"):
        canonical = d.get("alternate_copy_of") or d["id"]   # ground against the chunked copy
        docs_by_path[d["source_pdf"]] = canonical
        for alt in d.get("alternate_paths", []):
            docs_by_path[alt] = canonical
    now = datetime.now(timezone.utc).isoformat()
    existing_models = {m["id"]: m for m in load("models")}

    models, links, provenance, unresolved = [], [], [], []
    for mid, m in MODEL_REGISTRY.items():
        model_id = node_id("MODEL", mid)
        src_doc = ""
        src_kind = ""
        local = m.local_source or ""
        if local:
            if local in docs_by_path:
                src_doc = docs_by_path[local]
                src_kind = "corpus_pdf"
            elif local.endswith(".md") and (REPO_ROOT / local).exists():
                src_doc = node_id("RDOC", local)
                src_kind = "repository_document"
            else:
                unresolved.append({"model_id": model_id, "local_source": local,
                                   "state": "SOURCE_NOT_INGESTED" if (REPO_ROOT / local).exists() else "LOCAL_SOURCE_MISSING"})
        elif m.source:
            unresolved.append({"model_id": model_id, "source": m.source, "state": "NO_LOCAL_SOURCE_ROUTE"})

        basis = _status_to_basis(m.status)
        rec = {
            # Literature extraction and verification own additional fields on
            # these same records. Refresh registry fields without deleting the
            # independent evidence gathered since the previous rebuild.
            **existing_models.get(model_id, {}),
            "id": model_id,
            "type": "Model",
            "registry_id": mid,
            "subsystem": m.subsystem,
            "quantity": m.quantity,
            "relation": m.relation,
            "cited_source_text": m.source,
            "local_source": local,
            "source_document_id": src_doc,
            "source_kind": src_kind,
            "equation_ref": m.equation_ref,
            "validity_text": m.validity,
            "registry_status": m.status,
            "verification_text": m.verification,
            "validation_level": m.validation_level,
            "notes": m.notes,
            "basis_class": basis,
            "promotion_status": _promotion(m.status, m.validation_level),
            "verification_kinds": _verification_kind(m.verification, m.validation_level),
            "hardware_qualified": False,
            "experimentally_validated": ("cold_flow_validated" in m.validation_level or "hot_fire_validated" in m.validation_level)
                                        and "not_" not in m.validation_level,
            "provenance_kind": "PROJECT_DERIVES",
            "provenance_note": "raosim/model_registry.py entry; implementation-state authority. Scientific authority is the cited source document.",
            "extraction_version": EXTRACTION_VERSION,
            "harvested_at": now,
        }
        previous = existing_models.get(model_id, {})
        changed = [key for key in ("relation", "local_source", "source_document_id", "equation_ref")
                   if key in previous and previous[key] != rec[key]]
        if changed and previous.get("source_support_state"):
            rec.setdefault("source_support_history", []).append({
                "state": previous["source_support_state"], "notes": previous.get("source_support_notes"),
                "source_sha256": previous.get("source_support_sha256"),
                "invalidated_because": "registry changed: " + ", ".join(changed),
            })
            rec["source_support_state"] = "NOT_CHECKED"
            rec["source_support_notes"] = "Previous source review is stale: registry changed " + ", ".join(changed)
            rec.pop("source_support_state_verified", None)
            rec.pop("source_support_verifier_note", None)
        models.append(rec)

        prov_id = node_id("PROV", f"registry {mid}")
        provenance.append({
            "id": prov_id, "type": "ProvenanceRecord", "kind": "PROJECT_DERIVES",
            "asserted_by": "raosim/model_registry.py", "subject": model_id,
            "statement": f"registry entry {mid}: {m.relation}",
            "source_document_id": src_doc, "local_source": local,
            "confidence": "high" if src_doc else "medium",
        })

        if src_doc:
            links.append({
                "id": node_id("LINK", f"{mid} supported_by {src_doc}"),
                "type": "SoftwareLink", "edge": "SUPPORTED_BY",
                "from": model_id, "to": src_doc,
                "basis": basis, "provenance": prov_id,
                "note": f"registry local_source={local}; equation_ref={m.equation_ref}",
            })
        # validity envelope as its own node so it can be retrieved/queried
        if m.validity:
            val_id = node_id("VAL", mid)
            links.append({
                "id": node_id("LINK", f"{mid} has_validity {val_id}"),
                "type": "SoftwareLink", "edge": "HAS_VALIDITY_ENVELOPE",
                "from": model_id, "to": val_id, "basis": basis, "provenance": prov_id,
                "note": m.validity,
            })

    registry_validity_ids = {
        node_id("VAL", m["registry_id"]) for m in existing_models.values()
        if m.get("registry_id")
    } | {node_id("VAL", m["registry_id"]) for m in models}
    validity = [v for v in load("validity")
                if v.get("provenance_kind") == "SOURCE_STATES"
                or v["id"] not in registry_validity_ids]
    for m in models:
        if m["validity_text"]:
            validity.append({
                "id": node_id("VAL", m["registry_id"]), "type": "ValidityEnvelope",
                "model_id": m["id"], "subsystem": m["subsystem"],
                "prose": m["validity_text"],
                "machine_readable": {},   # populated only when a source states numeric bounds
                "machine_readable_state": "NOT_EXTRACTED",
                "provenance_kind": "PROJECT_DERIVES",
                "source_document_id": m["source_document_id"],
            })

    save("models", models, meta={"generated_at": now, "source": "raosim/model_registry.py", "n": len(models)})
    save("validity", validity, meta={"generated_at": now, "n": len(validity)})
    return {"models": len(models), "links": links, "provenance": provenance, "validity": len(validity),
            "unresolved": unresolved}


if __name__ == "__main__":
    import json
    r = harvest()
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()}, indent=1))
    print(json.dumps(r["unresolved"], indent=1))
