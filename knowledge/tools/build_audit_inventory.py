"""Reproducible implementation/source inventory and provisional evidence-debt queue.

This inventories every stored component; it does not certify each component's
science. Missing runtime leverage, source reviews and hardware evidence remain
explicit. Priorities are review scheduling assessments, never physical gates.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
sys.path.insert(0, str(ROOT))
from kg.coverage import summary
from kg.graph import model_source_support
from kg.store import load, index_by


def main():
    docs = index_by("manifest")
    models = index_by("models")
    rows = []
    for m in models.values():
        effective, reason = model_source_support(m, docs.get(m.get("source_document_id")))
        rows.append({"id": m["id"], "type": "Model", "registry_id": m["registry_id"],
            "audit_state": "CURRENT_SOURCE_REVIEW" if effective in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "SOURCE_REVIEW_REQUIRED",
            "declared_source_support": m.get("source_support_state", "NOT_CHECKED"),
            "effective_source_support": effective, "reason": reason, "source_document_id": m.get("source_document_id"),
            "execution_reach": m.get("execution_reach"), "physical_validation_state": m.get("physical_validation_state", "UNKNOWN")})
    constants = []
    for r in load("software_links"):
        if r.get("type") not in ("SoftwareComponent", "EmpiricalConstant"):
            continue
        active = r.get("active", True) and r.get("source_state", "CURRENT") == "CURRENT"
        row = {k: r.get(k) for k in ("id", "type", "file", "line", "end_line", "qualname", "name", "value", "units", "basis_class", "source_sha256", "definition_sha256")}
        row.update(active=active, model_ids=r.get("model_ids", []),
                   audit_state=("MECHANICAL_LOCATOR_VERIFIED_SCIENCE_NOT_RECHECKED" if r.get("implementation_locator_state") == "VERIFIED" else "NOT_REVIEWED")
                   if r["type"] == "SoftwareComponent" else "BASIS_NOT_INDEPENDENTLY_REVIEWED")
        if r["type"] == "EmpiricalConstant":
            constants.append(r)
        rows.append(row)
    for r in load("requirements"):
        rows.append({k: r.get(k) for k in ("id", "type", "file", "line", "name", "kind", "optimizer_role", "source_id_text", "source_document_ids", "basis_state")}
                    | {"audit_state": "DECLARATION_INVENTORIED_THRESHOLD_SOURCE_NOT_REVIEWED"})
    from raosim.mdo.build import ENGINE_PART_MANIFEST
    for p in ENGINE_PART_MANIFEST:
        rows.append({"id": p.part_id, "type": "EnginePart", "subsystem": p.subsystem,
                     "declared_supported_fidelity": int(p.supported_fidelity), "limitation": p.limitation,
                     "audit_state": "DECLARATION_INVENTORIED_CONFIGURATION_BINDING_NOT_ESTABLISHED"})
    coverage = summary(include_documents=True)
    for r in coverage["by_document"]:
        d = docs[r["document_id"]]
        rows.append({"id": d["id"], "type": "Document", "source_pdf": d["source_pdf"], "sha256": d["sha256"],
                     "page_count": d.get("page_count"), "coverage": r,
                     "audit_state": "EMPTY_SOURCE" if d.get("page_count") == 0 else "PARTIAL_STRUCTURED_EXTRACTION" if not r["extraction_gap"] else "STRUCTURED_EXTRACTION_NOT_STARTED"})
    debt = []
    for r in constants:
        if (r.get("basis_class") != "UNKNOWN_BASIS" or not r.get("active", True)
                or r.get("source_state", "CURRENT") != "CURRENT"):
            continue
        consumers = r.get("consumers", [])
        critical = r.get("harvest_owner") == "kg.harvest_parameters@1" and "MODEL_PUMP_ELECTRIC_DRIVE" in r.get("model_ids", [])
        optimizer_path = any(models.get(mid, {}).get("execution_reach", {}).get("optimizer_reach_state") == "STATIC_PATH_FOUND" for mid in r.get("model_ids", []))
        priority = "P0" if critical else "P1" if optimizer_path else "P2_PROVISIONAL"
        debt.append({"id": r["id"], "file": r.get("file"), "line": r.get("line"), "name": r.get("name"),
            "value": r.get("value"), "units": r.get("units"), "basis_class": "UNKNOWN_BASIS", "review_priority": priority,
            "priority_basis": "Current electric-drive default/consumer observation and primary-review priority" if critical else
                              "Associated model has a bounded static optimizer path" if optimizer_path else
                              "Unranked scientific basis gap; reach/severity need review (P2 is provisional queue placement)",
            "consumers": consumers, "model_ids": r.get("model_ids", []),
            "runtime_trace_state": "NOT_RUN", "optimizer_leverage_state": "NOT_MEASURED", "decision_severity": "NOT_INDEPENDENTLY_ASSESSED"})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    path = ROOT / "knowledge/reports/audit_inventory_2026-10-15.json"
    from collections import Counter
    generated_at = datetime.now(timezone.utc).isoformat()
    counts = dict(Counter(r["type"] for r in rows))
    path.write_text(json.dumps({"generated_at": generated_at, "commit": commit, "working_tree": "local uncommitted remediation",
        "scope": "Full mechanical inventory; scientific completeness NOT_ESTABLISHED",
        "counts": counts, "records": rows, "coverage": coverage}, indent=1, default=str) + "\n")
    (path.parent / "credibility_debt_2026-10-15.json").write_text(json.dumps({
        "generated_at": generated_at,
        "scope": "Provisional review priorities, not numerical risk scores or measured optimizer leverage",
        "by_priority": dict(Counter(r["review_priority"] for r in debt)), "records": debt}, indent=1) + "\n")
    print(json.dumps({"counts": counts, "unknown_basis_queue": len(debt)}))


if __name__ == "__main__":
    main()
