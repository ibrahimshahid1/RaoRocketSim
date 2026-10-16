"""Harvest the MDO constraint contract into Requirement / Constraint / Gate records.

Source of truth: ``raosim.mdo.constraints.CONSTRAINT_MANIFEST`` - the repo's
authoritative ordered manifest.  Its ``optimizer_role`` decides the
restriction kind; the knowledge system never re-labels a row:

    optimizer_role="hard"        -> optimizer_constraint   (enforced inside the solve)
    optimizer_role="final_gate"  -> postsolve_gate         (mandatory check after solving)
    optimizer_role="report_only" -> diagnostic
    kind="availability_gate"     -> applicability_gate     (decides whether a prediction may be made)
    category="requirement"       -> also tagged requirement (what the requested engine must satisfy)

``source_id`` strings are mapped to Documents ONLY when a report number in the
string matches a Document's title/identifier; unmatched strings are preserved as
``source_id_text`` with ``source_state="UNRESOLVED_SOURCE_KEY"``.  ``None`` becomes
``SCIENTIFIC_BASIS_UNKNOWN`` - the row is software policy with no cited basis.

The manifest module is imported with the JAX-free fallback: if the import fails
(e.g. an environment without jax), the harvester reads the ``ConstraintSpec(...)``
literals from the source with ``ast`` (positional AND keyword arguments, string
concatenation included; nothing is executed) instead and says so.

Each row's ``basis_class`` (literature, geometric_definition,
model_validity_domain, repository_rule, numerical_contract, requirement) is
carried as ``declared_basis_class``.  A row without a ``source_id`` whose
declared basis is not literature is labelled ``DECLARED_<BASIS>`` instead of
``SCIENTIFIC_BASIS_UNKNOWN``: the repository states what kind of limit it is,
which is a label, not source support.
"""

from __future__ import annotations

import ast
import re
import sys
from datetime import datetime, timezone

from . import REPO_ROOT
from .schema import node_id
from .store import load, save

sys.path.insert(0, str(REPO_ROOT))

# NASA monograph -> NTRS accession.  Each mapping records its evidence; loose
# title-word matching is deliberately NOT used (it mis-resolved SP-194 and
# SP-8109 to unrelated papers in an earlier pass).
NASA_REPORT_DOCS = {
    "NASA-SP-125":  ("DOC_NTRS_19710019929", "first pages print 'NASA SP-125'; title 'DESIGN OF LIQUID PROPELLANT ROCKET ENGINES'"),
    "NASA-SP-194":  ("DOC_NTRS_19720026079", "first pages print 'SP-194'; title 'LIQUID PROPELLANT ROCKET COMBUSTION INSTABILITY'"),
    "NASA-SP-8052": ("DOC_NTRS_19710025474", "first pages print 'SP-8052' (turbopump inducers monograph)"),
    "NASA-SP-8109": ("DOC_NTRS_19740020848", "first pages print 'SP-8109'; title 'LIQUID ROCKET ENGINE CENTRIFUGAL FLOW TURBOPUMPS'"),
    "NASA-SP-8120": ("DOC_NTRS_19770009165", "title 'Liquid Rocket Engine Nozzles' = SP-8120 title; same file is the registry's nozzle local_source (repository assertion)"),
    "NASA-SP-8087": ("DOC_NTRS_19730022965", "title 'LIQUID ROCKET ENGINE FLUID-COOLED COMBUSTION CHAMBERS' = SP-8087 title; constraints.py docstring names it (repository assertion)"),
    "NASA-SP-8089": ("DOC_NTRS_19760023196", "title 'Liquid Rocket Engine Injectors' = SP-8089 title; registry local_source for SP-8089 (repository assertion)"),
}
DOI_KEYS = {
    "Nasuti-Pizzarelli-2021": "10.1016/j.supflu.2020.105066",
    "Mirzamoghadam-1991": "10.2514/6.1991-1982",
    "Carlile-1992": "10.2514/6.1992-3154",
    "Wadel-1997": "10.2514/6.1997-2913",
    "Freeberg-2019": "10.2514/6.2019-0152",
}
_FIELDS = ("name", "engine_key", "kind", "optimizer_role", "category", "scale",
           "units", "source_id", "applicability", "basis_class")
_DEFAULTS = {"applicability": "always", "basis_class": None}


def _resolve_source_key(key: str, docs: list[dict]) -> tuple[str, str]:
    by_id = {d["id"]: d for d in docs}
    for prefix, (did, evidence) in NASA_REPORT_DOCS.items():
        if key.upper().startswith(prefix.upper()):
            d = by_id.get(did)
            if d:
                return d.get("alternate_copy_of") or did, f"report_key_mapped:{evidence}"
            return "", "SOURCE_NOT_INGESTED"
    for prefix, doi in DOI_KEYS.items():
        if key.startswith(prefix):
            for d in docs:
                if (d.get("doi") or "").lower() == doi.lower():
                    return d.get("alternate_copy_of") or d["id"], f"doi_mapped:{doi}"
            return "", "SOURCE_NOT_INGESTED"
    return "", "UNRESOLVED_SOURCE_KEY"


def _parse_manifest(text: str) -> list[dict]:
    """``ConstraintSpec(...)`` calls inside ``CONSTRAINT_MANIFEST``, via ``ast``.

    Positional arguments map onto the dataclass field order and keywords by
    name, so a keyword-only ``basis_class=`` can never be mistaken for the
    positional ``applicability``.  Values must be literals (``literal_eval``).
    """
    rows = []
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "ConstraintSpec"):
            continue
        try:
            row = dict(_DEFAULTS)
            row.update(zip(_FIELDS, (ast.literal_eval(a) for a in node.args)))
            row.update({k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in _FIELDS})
        except (ValueError, SyntaxError):
            continue
        if all(field in row for field in _FIELDS[:8]):
            rows.append(row)
    return rows


def _load_manifest():
    try:
        from raosim.mdo.constraints import CONSTRAINT_MANIFEST  # type: ignore
        return [dict(name=s.name, engine_key=s.engine_key, kind=s.kind, optimizer_role=s.optimizer_role,
                     category=s.category, scale=s.scale, units=s.units, source_id=s.source_id,
                     applicability=s.applicability, basis_class=getattr(s, "basis_class", None))
                for s in CONSTRAINT_MANIFEST], "import"
    except Exception:
        text = (REPO_ROOT / "raosim/mdo/constraints.py").read_text()
        return _parse_manifest(text), "source_ast_parse"


def harvest() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    rows, how = _load_manifest()
    docs = load("manifest")
    src_text = (REPO_ROOT / "raosim/mdo/constraints.py").read_text().splitlines()
    out = []
    for i, r in enumerate(rows):
        if r["kind"] == "availability_gate":
            rkind = "applicability_gate"
        elif r["optimizer_role"] == "hard":
            rkind = "optimizer_constraint"
        elif r["optimizer_role"] == "final_gate":
            rkind = "postsolve_gate"
        else:
            rkind = "diagnostic"
        line = next((n + 1 for n, ln in enumerate(src_text) if f'ConstraintSpec("{r["name"]}"' in ln or f"ConstraintSpec(\n" in ln and False), None)
        if line is None:
            line = next((n + 1 for n, ln in enumerate(src_text) if f'"{r["name"]}"' in ln), None)
        src_docs, states = [], []
        if r["source_id"]:
            for key in str(r["source_id"]).split(";"):
                did, st = _resolve_source_key(key.strip(), docs)
                if did:
                    src_docs.append(did)
                states.append({"key": key.strip(), "state": st, "document_id": did})
        else:
            states.append({"key": None, "state": "SCIENTIFIC_BASIS_UNKNOWN", "document_id": ""})
        declared = r.get("basis_class")
        if src_docs:
            basis_state = "NOT_CHECKED"
        elif r["source_id"]:
            basis_state = "SOURCE_NOT_INGESTED"
        elif declared and declared != "literature":
            basis_state = "DECLARED_" + str(declared).upper()
        else:
            basis_state = "SCIENTIFIC_BASIS_UNKNOWN"
        rid = node_id("CST", r["name"])
        out.append({
            "id": rid, "type": "Constraint", "name": r["name"], "engine_key": r["engine_key"],
            "kind": rkind, "is_requirement": r["category"] == "requirement",
            "manifest_kind": r["kind"], "optimizer_role": r["optimizer_role"], "category": r["category"],
            "units": r["units"], "scale": r["scale"], "applicability": r["applicability"], "order": i,
            "source_id_text": r["source_id"], "source_resolution": states,
            "source_document_ids": sorted(set(src_docs)),
            "basis_state": basis_state, "declared_basis_class": declared,
            "citation_resolution_state": "CITATION_RESOLVED" if src_docs else "UNRESOLVED",
            "source_support_state": "NOT_CHECKED",
            "source_support_scope": "A resolved report key is a citation route, not page-verified support for this constraint or threshold.",
            "subsystem": {"thermal": "cooling", "feed": "pump", "aerodynamics": "performance", "geometry": "nozzle_contour",
                          "structures": "structures", "injector": "injector", "manufacturing": "manufacturing",
                          "requirement": "requirements", "numerical": "mdo", "properties": "thermochemistry"}.get(r["category"], "mdo"),
            "enforced_by": "raosim/mdo/nlp.py (SLSQP hard rows)" if rkind == "optimizer_constraint" else
                           ("raosim/mdo/nlp.py (mandatory post-solve gate, no relaxation)" if rkind in ("postsolve_gate", "applicability_gate") else "reported only"),
            "file": "raosim/mdo/constraints.py", "line": line,
            "detail": f"{r['category']} {r['kind']} row; optimizer_role={r['optimizer_role']}; applicability={r['applicability']}",
            "quantity": r["engine_key"] or r["name"], "model_ids": [], "quantity_ids": [],
            "provenance_id": node_id("PROV", f"constraint manifest {r['name']}"), "harvest_method": how, "harvested_at": now,
        })
    save("requirements", out, meta={"generated_at": now, "source": "raosim/mdo/constraints.py CONSTRAINT_MANIFEST", "n": len(out), "method": how})
    prov = {p["id"]: p for p in load("provenance")}
    for r in out:
        prov[r["provenance_id"]] = {"id": r["provenance_id"], "type": "ProvenanceRecord", "kind": "CODE_STATES",
                                    "asserted_by": f"raosim/mdo/constraints.py:{r['line']}", "subject": r["id"],
                                    "statement": f"CONSTRAINT_MANIFEST row {r['name']} optimizer_role={r['optimizer_role']}"}
    save("provenance", prov.values(), meta={"generated_at": now, "n": len(prov)})
    from collections import Counter
    return {"rows": len(out), "by_kind": dict(Counter(r["kind"] for r in out)),
            "by_basis": dict(Counter(r["basis_state"] for r in out)), "method": how}


if __name__ == "__main__":
    import json
    print(json.dumps(harvest(), indent=1))
