"""Turn the agent survey of LREKit source into verified SoftwareComponent records.

Grounding rule: an agent's report is a *lead*, not a fact.  Each reported
component is verified against the working tree before it is stored:

* the file must exist;
* the qualname's terminal symbol must be defined in that file (``def``/``class``),
  and the reported line is corrected to the actual definition line;
* each hard-coded constant's ``line`` must exist and its ``value`` must appear
  in the file within a few lines of the reported line, else it is stored with
  ``verified=false`` and excluded from the graph;
* each ``model_registry_ids`` entry must be a real registry id;
* each cited corpus path must exist on disk.

Unverifiable items are kept in the ingestion report as ``UNGROUNDED`` so the
loss is visible, never silently dropped.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT
from .schema import node_id
from .store import load, save
from .record_identity import consolidate

DEF_RE = r"^\s*(?:(?:async\s+def|def|class)\s+{name}\b|{name}\s*(?::[^=]+)?=)"

SUBSYSTEM_MAP = {
    "nozzle-rao": "nozzle_contour", "nozzle-moc": "moc", "nozzle-jax": "mdo",
    "thermal-cooling": "cooling", "thermal-mdo": "mdo", "structures-materials": "structures",
    "injector-pintle": "injector", "spray": "spray", "pumps-feed": "pump",
    "performance-gasdyn": "performance", "cad-manufacturing": "cad", "mdo-core": "mdo",
    "requirements-gates": "verification", "engine-integration": "configuration",
}


def _find_def(lines: list[str], symbol: str, hint_line: int | None) -> int | None:
    pat = re.compile(DEF_RE.format(name=re.escape(symbol)))
    cands = [i + 1 for i, ln in enumerate(lines) if pat.match(ln)]
    if not cands:
        return None
    if hint_line:
        return min(cands, key=lambda n: abs(n - hint_line))
    return cands[0]


def _value_near(lines: list[str], line: int | None, value: str, radius: int = 6) -> bool:
    if not line or not value:
        return False
    v = value.strip()
    # accept the numeric literal or the first token of a longer description
    probes = [v]
    m = re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", v)
    if m:
        probes.append(m.group(0))
    lo, hi = max(0, line - 1 - radius), min(len(lines), line + radius)
    window = "\n".join(lines[lo:hi])
    return any(p in window for p in probes if p)


def harvest(survey_results: list[dict], registry_ids: set[str]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    docs_by_path = {}
    for d in load("manifest"):
        canonical = d.get("alternate_copy_of") or d["id"]
        docs_by_path[d["source_pdf"]] = canonical
        for alt in d.get("alternate_paths", []):
            docs_by_path[alt] = canonical
    models_by_reg = {m["registry_id"]: m["id"] for m in load("models")}

    components, constants, links, provenance = [], [], [], []
    report = {"verified": 0, "ungrounded_components": [], "ungrounded_constants": [],
              "unknown_registry_ids": [], "unresolved_citations": [], "line_corrections": 0}
    file_cache: dict[str, list[str]] = {}

    def lines_of(rel: str) -> list[str] | None:
        if rel in file_cache:
            return file_cache[rel]
        p = REPO_ROOT / rel
        if not p.exists():
            return None
        file_cache[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return file_cache[rel]

    qual_to_id: dict[str, str] = {}
    pending_variants: list[tuple[str, str]] = []

    for survey in survey_results:
        subsystem = SUBSYSTEM_MAP.get(survey.get("subsystem", ""), survey.get("subsystem", ""))
        for comp in survey.get("components", []):
            rel = comp.get("file", "").replace("\\", "/")
            lines = lines_of(rel)
            symbol = re.sub(r"\s*\(.*\)$", "", comp.get("qualname", "").split(":")[-1]).split(".")[-1].strip()
            if lines is None or not symbol:
                report["ungrounded_components"].append({"qualname": comp.get("qualname"), "file": rel, "reason": "file missing or no symbol"})
                continue
            actual = _find_def(lines, symbol, comp.get("line"))
            if actual is None:
                report["ungrounded_components"].append({"qualname": comp.get("qualname"), "file": rel, "reason": f"no def/class {symbol} in file"})
                continue
            if comp.get("line") != actual:
                report["line_corrections"] += 1
            cid = node_id("SW", f"{rel}:{symbol}")
            qual_to_id[comp["qualname"]] = cid
            qual_to_id[f"{rel}:{symbol}"] = cid
            prov_id = node_id("PROV", f"survey {rel}:{symbol}")
            provenance.append({"id": prov_id, "type": "ProvenanceRecord", "kind": "CODE_STATES",
                               "asserted_by": "agent survey of source, verified against working tree",
                               "subject": cid, "file": rel, "line": actual, "verified_at": now})

            reg_ids, model_ids = [], []
            for rid in comp.get("model_registry_ids", []) or []:
                if rid in registry_ids:
                    reg_ids.append(rid)
                    model_ids.append(models_by_reg[rid])
                else:
                    report["unknown_registry_ids"].append({"component": cid, "registry_id": rid})

            cited_docs, cited_text = [], []
            for cit in comp.get("cited_sources", []) or []:
                cited_text.append(cit)
                for m in re.finditer(r"propulsion_texts/[^\s'\"`)\]]+\.pdf", cit):
                    path = m.group(0)
                    if path in docs_by_path:
                        cited_docs.append(docs_by_path[path])
                    else:
                        report["unresolved_citations"].append({"component": cid, "path": path})

            const_ids = []
            for k in comp.get("hardcoded_constants", []) or []:
                kline = k.get("line")
                ok = _value_near(lines, kline, str(k.get("value", "")))
                kid = node_id("CONST", f"{rel}:{symbol}:{k.get('name','')}")
                rec = {"id": kid, "type": "EmpiricalConstant", "component_id": cid, "name": k.get("name"),
                       "value": str(k.get("value")), "units": k.get("units", ""), "file": rel, "line": kline,
                       "stated_basis": k.get("stated_basis", ""), "basis_class": k.get("basis_class", "UNKNOWN_BASIS"),
                       "verified": ok, "provenance_id": prov_id}
                if ok:
                    constants.append(rec)
                    const_ids.append(kid)
                else:
                    report["ungrounded_constants"].append({"component": cid, "name": k.get("name"), "line": kline, "value": k.get("value")})

            refusals = []
            for r in comp.get("refusals", []) or []:
                refusals.append({"condition": r.get("condition"), "behavior": r.get("behavior"),
                                 "line": r.get("line"), "message": r.get("message", "")})

            comp_rec = {
                "id": cid, "type": "SoftwareComponent", "qualname": comp["qualname"], "file": rel, "line": actual,
                "reported_line": comp.get("line"), "subsystem": subsystem, "survey_area": survey.get("subsystem"),
                "role": comp.get("role", ""), "relations": comp.get("relations", []) or [],
                "relations_text": " ".join(comp.get("relations", []) or []),
                "inputs": comp.get("inputs", []) or [], "outputs": comp.get("outputs", []) or [],
                "registry_ids": reg_ids, "model_ids": model_ids,
                "cited_sources_text": cited_text, "cited_document_ids": sorted(set(cited_docs)),
                "validity_statements": comp.get("validity_statements", []) or [],
                "refusals": refusals, "constant_ids": const_ids,
                "tests_text": comp.get("tests", []) or [],
                "variant_of_text": comp.get("variant_of", "") or "",
                "verified": True, "provenance_id": prov_id, "harvested_at": now,
            }
            components.append(comp_rec)
            report["verified"] += 1
            if comp_rec["variant_of_text"]:
                pending_variants.append((cid, comp_rec["variant_of_text"]))

    # resolve variant siblings to ids when the sibling was also surveyed
    comp_by_id = {c["id"]: c for c in components}
    for cid, text in pending_variants:
        m = re.search(r"([\w./]+\.py):(\w+)|([\w.]+):(\w+)", text)
        target = None
        if m:
            if m.group(1):
                target = qual_to_id.get(f"{m.group(1)}:{m.group(2)}")
            else:
                target = qual_to_id.get(f"{m.group(3)}:{m.group(4)}")
        if target and target in comp_by_id:
            comp_by_id[cid]["variant_of_id"] = target
            tl = text.lower()
            comp_by_id[cid]["variant_edge"] = "PARITY_CHECKED_AGAINST" if ("oracle" in tl or "numpy" in tl or "parity" in tl) else "COMPARED_WITH"

    # tests referenced by components become Test records (existence-verified)
    tests = {}
    for c in components:
        tids = []
        for t in c["tests_text"]:
            path = t.split("::")[0]
            if (REPO_ROOT / path).exists():
                tid = node_id("TEST", t)
                tests.setdefault(tid, {"id": tid, "type": "Test", "path": t, "file": path,
                                       "subsystem": c["subsystem"], "verification_kind": "unit_test",
                                       "verification_kind_basis": "default; refined by tests survey",
                                       "model_ids": [], "provenance_id": c["provenance_id"]})
                for mid in c["model_ids"]:
                    if mid not in tests[tid]["model_ids"]:
                        tests[tid]["model_ids"].append(mid)
                tids.append(tid)
        c["test_ids"] = tids
        del c["tests_text"]

    previous = load("software_links")
    existing_links = [s for s in previous
                      if s.get("type") not in ("SoftwareComponent", "EmpiricalConstant")
                      or s.get("harvest_owner") == "kg.harvest_parameters@1"]
    incoming = components + constants
    incoming_ids = {r["id"] for r in incoming}
    prior_facets = [s for s in previous if s["id"] in incoming_ids
                    and s.get("harvest_owner") != "kg.harvest_parameters@1"]
    # Keep current locator fingerprints/history so a subsequent exact AST
    # reconciliation can detect changed bodies rather than losing its baseline.
    preferred = {r["id"]: dict(r) for r in incoming}
    protected = {"line", "end_line", "source_sha256", "definition_sha256", "ast_qualname",
                 "active", "source_state", "verified"}
    for old in prior_facets:
        for key, value in old.items():
            if key in protected or key.startswith("implementation_"):
                preferred[old["id"]][key] = value
    consolidated, identity_report = consolidate(existing_links + prior_facets + incoming,
                                                 preferred=preferred,
                                                 origin={"kind": "AGENT_SURVEY_ASSERTION"})
    report["record_identity"] = identity_report
    save("software_links", consolidated,
         meta={"generated_at": now, "n_components": len(components), "n_constants": len(constants)})
    # merge tests
    prior_tests = {t["id"]: t for t in load("tests")}
    for tid, t in tests.items():
        if tid in prior_tests:
            prior_tests[tid]["model_ids"] = sorted(set(prior_tests[tid].get("model_ids", [])) | set(t["model_ids"]))
        else:
            prior_tests[tid] = t
    save("tests", prior_tests.values(), meta={"generated_at": now, "n": len(prior_tests)})
    prior_prov = {p["id"]: p for p in load("provenance")}
    for p in provenance:
        prior_prov[p["id"]] = p
    save("provenance", prior_prov.values(), meta={"generated_at": now, "n": len(prior_prov)})
    report["n_components"] = len(components)
    report["n_constants"] = len(constants)
    report["n_tests"] = len(tests)
    return report
