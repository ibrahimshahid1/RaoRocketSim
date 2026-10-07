"""Store grounded literature extractions (equations, claims, validity, constants).

Input: the structured results of the extraction workflow.  Every object is
passed through :class:`kg.ground.DocGrounder`; objects whose verbatim anchor
cannot be found in the indexed source text are NOT stored - they are listed in
the returned report under ``ungrounded`` so the loss is visible.

Model links: a model-driven extraction that reports ``supports_model.state``
is recorded on the Model (``source_support_state``) and, for SUPPORTED /
PARTIALLY_SUPPORTED, its equations get ``model_ids`` with
``model_link_basis = "registry_extraction"``.  ``NOT_FOUND_IN_SOURCE`` is a
first-class finding: the registry's citation could not be confirmed in the
local copy, and the model keeps that state rather than a fabricated link.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from .ground import DocGrounder, normalize
from .quantities import resolve_symbol
from .schema import EXTRACTION_VERSION, node_id
from .store import load, save


def _short_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:8]


def _subsystem_for_model(models_by_id: dict, mid: str) -> str:
    return models_by_id.get(mid, {}).get("subsystem", "")


def _merge_list(old: list | None, new: list | None) -> list:
    result = list(old or [])
    for value in new or []:
        if value not in result:
            result.append(value)
    return result


def _current_record(record: dict, source_sha256: str) -> dict:
    """Old verification cannot vouch for an extraction from a changed source."""
    record = dict(record)
    changed = record.get("source_state") == "STALE_SOURCE" or (
        bool(record.get("source_sha256")) and record["source_sha256"] != source_sha256)
    if changed:
        for key in list(record):
            if key.startswith(("verifier_", "stale_")) or key in (
                "verification", "verified_at", "model_ids_before_refutation", "model_ids", "model_link_basis",
                "source_state_reason", "assumptions", "validity"):
                record.pop(key, None)
    return record


def harvest(model_results: list[dict], window_results: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    models_by_id = {m["id"]: m for m in load("models")}
    source_hashes = {d["id"]: d.get("sha256", "") for d in load("manifest")}
    grounders: dict[str, DocGrounder] = {}

    def grounder(doc_id: str) -> DocGrounder:
        if doc_id not in grounders:
            grounders[doc_id] = DocGrounder(doc_id)
        return grounders[doc_id]

    equations: dict[str, dict] = {e["id"]: e for e in load("equations")}
    claims: dict[str, dict] = {c["id"]: c for c in load("claims")}
    entities: dict[str, dict] = {e["id"]: e for e in load("entities")}
    validity: dict[str, dict] = {v["id"]: v for v in load("validity")}
    provenance: dict[str, dict] = {p["id"]: p for p in load("provenance")}
    report = {"grounded": 0, "page_shifted": 0, "ungrounded": [], "per_doc": {}, "model_support": {},
              "n_equations": 0, "n_claims": 0, "n_validity": 0, "n_constants": 0, "n_assumptions": 0}

    def bump(doc_id, key):
        report["per_doc"].setdefault(doc_id, {"grounded": 0, "page_shifted": 0, "ungrounded": 0})[key] += 1

    def ground(doc_id: str, obj: dict, kind: str) -> dict | None:
        g = grounder(doc_id)
        if not g.chunks:
            report["ungrounded"].append({"doc_id": doc_id, "kind": kind, "reason": "SOURCE_NOT_INGESTED",
                                         "label": obj.get("label") or obj.get("statement") or obj.get("name")})
            bump(doc_id, "ungrounded")
            return None
        loc = g.locate(obj.get("quote", ""), obj.get("page"))
        if loc["state"] == "UNGROUNDED":
            report["ungrounded"].append({"doc_id": doc_id, "kind": kind, "reason": loc.get("reason"),
                                         "page": obj.get("page"), "quote": obj.get("quote", "")[:120],
                                         "label": obj.get("label") or obj.get("statement") or obj.get("name")})
            bump(doc_id, "ungrounded")
            return None
        if loc["state"] == "GROUNDED_PAGE_SHIFTED":
            report["page_shifted"] += 1
            bump(doc_id, "page_shifted")
        else:
            report["grounded"] += 1
            bump(doc_id, "grounded")
        return loc

    def prov(subject: str, doc_id: str, page: int, chunk_id: str, quote: str, extractor: str) -> str:
        pid = node_id("PROV", f"{subject}")
        provenance[pid] = {"id": pid, "type": "ProvenanceRecord", "kind": "SOURCE_STATES", "subject": subject,
                           "source_document_id": doc_id, "page": page, "chunk_id": chunk_id, "quote": quote,
                           "source_sha256": source_hashes.get(doc_id, ""), "source_state": "CURRENT",
                           "asserted_by": extractor, "verified_by": "kg.ground (verbatim anchor located in indexed source text)",
                           "extraction_version": EXTRACTION_VERSION, "recorded_at": now}
        return pid

    def handle(result: dict, extractor: str, model_id: str | None):
        doc_id = result["doc_id"]
        source = {"source_sha256": source_hashes.get(doc_id, ""), "source_state": "CURRENT"}
        sub = _subsystem_for_model(models_by_id, model_id) if model_id else ""
        # An equation is linked to the registry model ONLY when the extractor found
        # support and (if it named which equations support it) this is one of them.
        sm = result.get("supports_model") or {}
        support_state = sm.get("state", "")
        supporting_labels = {normalize(x) for x in (sm.get("equation_labels") or [])}

        def links_model(label: str) -> bool:
            if not model_id or support_state not in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                return False
            return not supporting_labels or normalize(label) in supporting_labels

        # ---- equations ----
        for e in result.get("equations", []) or []:
            loc = ground(doc_id, e, "equation")
            if not loc:
                continue
            page = loc["page"]
            eid = f"EQ_{doc_id[4:]}_{_short_hash(e.get('form','') + e.get('quote',''))}".upper()
            eq_model_id = model_id if links_model(e.get("label", "")) else None
            variables = []
            for v in e.get("variables", []) or []:
                res = resolve_symbol(v.get("symbol", ""), sub or None)
                variables.append({"symbol": v.get("symbol"), "meaning": v.get("meaning"), "units": v.get("units", ""),
                                  "role": v.get("role", "unknown"), "resolution_state": res["state"],
                                  "quantity_candidates": [c["qid"] for c in res["candidates"]][:3]})
            rec = _current_record(equations.get(eid, {}), source["source_sha256"])
            model_ids = sorted(set(rec.get("model_ids", [])) | ({eq_model_id} if eq_model_id else set()))
            if rec.get("verification") == "REFUTED":
                model_ids = []
            pid = prov(eid, doc_id, page, loc["chunk_id"], e.get("quote", ""), extractor)
            equations[eid] = {
                **rec,
                **source,
                "id": eid, "type": "Equation", "label": e.get("label"), "equation_number": e.get("equation_number", ""),
                "form": e.get("form"), "description": e.get("description", ""), "variables": variables,
                "variables_text": " ".join(f"{v['symbol']} {v['meaning']}" for v in variables),
                "concepts_text": f"{e.get('label','')} {e.get('description','')}",
                "assumptions": _merge_list(rec.get("assumptions"), e.get("assumptions")),
                "validity": _merge_list(rec.get("validity"), e.get("validity")),
                "document_id": doc_id, "page": page, "reported_page": e.get("page"), "chunk_id": loc["chunk_id"],
                "quote": e.get("quote"), "grounded": loc["state"], "match": loc.get("match"),
                "confidence": e.get("confidence"), "extraction_note": e.get("extraction_note", ""),
                "model_ids": model_ids, "model_link_basis": ("registry_extraction:" + support_state) if eq_model_id else rec.get("model_link_basis", ""),
                "extracted_for_model": model_id or rec.get("extracted_for_model", ""),
                "provenance_id": pid, "provenance_kind": "SOURCE_STATES", "subsystem": sub or rec.get("subsystem", ""),
                "equivalent_to": rec.get("equivalent_to", []), "competing_with": rec.get("competing_with", []),
                "extraction_version": EXTRACTION_VERSION, "recorded_at": now,
            }
            report["n_equations"] += 1
        # ---- claims ----
        for c in result.get("claims", []) or []:
            loc = ground(doc_id, c, "claim")
            if not loc:
                continue
            page = loc["page"]
            cid = f"CLM_{doc_id[4:]}_{_short_hash(c.get('statement','') + c.get('quote',''))}".upper()
            rec = _current_record(claims.get(cid, {}), source["source_sha256"])
            cm = model_id if (model_id and support_state in ("SUPPORTED", "PARTIALLY_SUPPORTED")) else None
            model_ids = sorted(set(rec.get("model_ids", [])) | ({cm} if cm else set()))
            if rec.get("verification") == "REFUTED":
                model_ids = []
            pid = prov(cid, doc_id, page, loc["chunk_id"], c.get("quote", ""), extractor)
            qids = []
            for word in re.findall(r"[A-Za-z_*][A-Za-z0-9_*]+", f"{c.get('subject','')} {c.get('object','')}"):
                r = resolve_symbol(word)
                if r["state"] == "RESOLVED":
                    qids.append(r["candidates"][0]["qid"])
            claims[cid] = {
                **rec,
                **source,
                "id": cid, "type": "Claim", "claim_type": c.get("claim_type"), "subject": c.get("subject"),
                "predicate": c.get("predicate"), "object": c.get("object"), "statement": c.get("statement"),
                "qualifiers": c.get("qualifiers", []) or [], "document_id": doc_id, "page": page,
                "reported_page": c.get("page"), "chunk_id": loc["chunk_id"], "quote": c.get("quote"),
                "grounded": loc["state"], "confidence": c.get("confidence"), "model_ids": model_ids,
                "quantity_ids": sorted(set(qids)), "contradicts": rec.get("contradicts", []),
                "provenance_id": pid, "provenance_kind": "SOURCE_STATES", "subsystem": sub or rec.get("subsystem", ""),
                "extraction_version": EXTRACTION_VERSION, "recorded_at": now,
            }
            report["n_claims"] += 1
        # ---- validity statements (source-stated) ----
        for v in result.get("validity_statements", []) or []:
            loc = ground(doc_id, v, "validity")
            if not loc:
                continue
            page = loc["page"]
            vid = f"VAL_{doc_id[4:]}_{_short_hash(v.get('statement','') + v.get('quote',''))}".upper()
            pid = prov(vid, doc_id, page, loc["chunk_id"], v.get("quote", ""), extractor)
            bounds = [b for b in (v.get("bounds") or []) if b.get("variable")]
            rec = _current_record(validity.get(vid, {}), source["source_sha256"])
            linked_model = (model_id if support_state in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "") or ""
            validity[vid] = {
                **rec,
                **source,
                "id": vid, "type": "ValidityEnvelope",
                "model_id": rec.get("model_id") or linked_model, "subsystem": sub or rec.get("subsystem", ""),
                "model_ids": sorted(set(rec.get("model_ids", [])) | {m for m in (rec.get("model_id"), linked_model) if m}),
                "extracted_for_model": model_id or rec.get("extracted_for_model", ""),
                "prose": v.get("statement"), "applies_to": v.get("applies_to"),
                "machine_readable": {"bounds": bounds} if bounds else rec.get("machine_readable", {}),
                "machine_readable_state": "SOURCE_STATED_BOUNDS" if bounds else rec.get("machine_readable_state", "PROSE_ONLY"),
                "document_id": doc_id, "page": page, "chunk_id": loc["chunk_id"], "quote": v.get("quote"),
                "grounded": loc["state"], "provenance_id": pid, "provenance_kind": "SOURCE_STATES",
                "extraction_version": EXTRACTION_VERSION,
            }
            report["n_validity"] += 1
        # ---- assumptions ----
        for a in result.get("assumptions", []) or []:
            loc = ground(doc_id, a, "assumption")
            if not loc:
                continue
            aid = f"ASM_{doc_id[4:]}_{_short_hash(a.get('statement','') + a.get('quote',''))}".upper()
            pid = prov(aid, doc_id, loc["page"], loc["chunk_id"], a.get("quote", ""), extractor)
            rec = _current_record(entities.get(aid, {}), source["source_sha256"])
            entities[aid] = {**rec, **source, "id": aid, "type": "Assumption", "label": a.get("statement", "")[:160], "prose": a.get("statement"),
                             "document_id": doc_id, "page": loc["page"], "chunk_id": loc["chunk_id"], "quote": a.get("quote"),
                             "grounded": loc["state"], "model_ids": sorted(set(rec.get("model_ids", [])) | ({model_id} if (model_id and support_state in ("SUPPORTED", "PARTIALLY_SUPPORTED")) else set())),
                             "subsystem": sub or rec.get("subsystem", ""), "provenance_id": pid, "provenance_kind": "SOURCE_STATES"}
            report["n_assumptions"] += 1
        # ---- empirical constants stated by the literature ----
        for k in result.get("empirical_constants", []) or []:
            loc = ground(doc_id, k, "constant")
            if not loc:
                continue
            kid = f"CONST_{doc_id[4:]}_{_short_hash(k.get('name','') + str(k.get('value','')) + k.get('quote',''))}".upper()
            pid = prov(kid, doc_id, loc["page"], loc["chunk_id"], k.get("quote", ""), extractor)
            rec = _current_record(entities.get(kid, {}), source["source_sha256"])
            entities[kid] = {**rec, **source, "id": kid, "type": "EmpiricalConstant", "label": f"{k.get('name')} = {k.get('value')} {k.get('units','')}".strip(),
                             "name": k.get("name"), "value": str(k.get("value")), "units": k.get("units", ""), "context": k.get("context", ""),
                             "document_id": doc_id, "page": loc["page"], "chunk_id": loc["chunk_id"], "quote": k.get("quote"),
                             "grounded": loc["state"], "model_ids": sorted(set(rec.get("model_ids", [])) | ({model_id} if (model_id and support_state in ("SUPPORTED", "PARTIALLY_SUPPORTED")) else set())), "subsystem": sub or rec.get("subsystem", ""),
                             "basis_class": "LITERATURE_DERIVED", "provenance_id": pid, "provenance_kind": "SOURCE_STATES"}
            report["n_constants"] += 1

    for r in model_results:
        mid = r.get("target")
        if mid not in models_by_id:
            mid = None
        handle(r, "extraction workflow (model-driven)", mid)
        sm = r.get("supports_model") or {}
        if mid:
            model = models_by_id[mid]
            source_changed = model.get("source_support_sha256") not in (None, "", source_hashes.get(r["doc_id"], ""))
            if source_changed or model.get("source_support_state") == "NOT_CHECKED":
                for key in ("source_support_state_verified", "source_support_verifier_note", "source_support_state_extracted"):
                    model.pop(key, None)
            model["source_support_state_extracted"] = sm.get("state", "UNKNOWN")
            model["source_support_state"] = model.get("source_support_state_verified") or sm.get("state", "UNKNOWN")
            models_by_id[mid]["source_support_notes"] = sm.get("mismatch_notes", "")
            models_by_id[mid]["source_support_equation_labels"] = sm.get("equation_labels", [])
            models_by_id[mid]["source_support_sha256"] = source_hashes.get(r["doc_id"], "")
            report["model_support"][mid] = sm.get("state", "UNKNOWN")
    for r in window_results:
        handle(r, "extraction workflow (document-driven)", None)

    save("equations", equations.values(), meta={"generated_at": now, "n": len(equations)})
    save("claims", claims.values(), meta={"generated_at": now, "n": len(claims)})
    save("entities", entities.values(), meta={"generated_at": now, "n": len(entities)})
    save("validity", validity.values(), meta={"generated_at": now, "n": len(validity)})
    save("provenance", provenance.values(), meta={"generated_at": now, "n": len(provenance)})
    save("models", models_by_id.values(), meta={"generated_at": now, "source": "raosim/model_registry.py + extraction support states", "n": len(models_by_id)})
    report["n_ungrounded"] = len(report["ungrounded"])
    return report


# --------------------------------------------------------------------------
# grouped extraction (one agent per subsystem group) + cross-document relations
# --------------------------------------------------------------------------

_REL_EDGE = {
    "EQUIVALENT_TO": ("EQUIVALENT_TO", "equivalent_to"),
    "NOTATION_VARIANT": ("EQUIVALENT_TO", "equivalent_to"),
    "POSSIBLY_EQUIVALENT": ("POSSIBLY_EQUIVALENT", "possibly_equivalent"),
    "APPROXIMATES": ("APPROXIMATES", "approximates"),
    "DIFFERENT_ASSUMPTIONS": ("COMPETING_FORMULATION_WITH", "competing_with"),
    "CONTRADICTS": ("CONTRADICTS", "competing_with"),
}


def harvest_groups(groups: list[dict]) -> dict:
    """Flatten grouped results, harvest them, then attach cross-document relations.

    Relations are attached only when BOTH labels resolve to a stored (grounded)
    equation of the stated document; unresolved relations are reported, not
    guessed.  A relation never merges two equations - both keep their own
    provenance; the edge itself carries ``SYSTEM_INFERS``-level provenance
    (an extraction agent's comparison), which is what the graph records.
    """
    model_results, window_results = [], []
    skipped = []
    for g in groups:
        for r in g.get("results", []) or []:
            (model_results if str(r.get("target", "")).startswith("MODEL_") else window_results).append(r)
        skipped.extend({"group": g.get("group") or g.get("name"), "item": s} for s in (g.get("skipped") or []))
    report = harvest(model_results, window_results)
    report["skipped"] = skipped

    eqs = {e["id"]: e for e in load("equations")}
    by_doc_label: dict[tuple[str, str], str] = {}
    for e in eqs.values():
        by_doc_label[(e["document_id"], normalize(e.get("label", "")))] = e["id"]
    rels_store = {r["id"]: r for r in load("relationships")}
    attached, unresolved = 0, []
    now = datetime.now(timezone.utc).isoformat()
    for g in groups:
        for rel in g.get("cross_document_relations", []) or []:
            a = by_doc_label.get((rel.get("a_doc_id"), normalize(rel.get("a_label", ""))))
            b = by_doc_label.get((rel.get("b_doc_id"), normalize(rel.get("b_label", ""))))
            kind = rel.get("relation", "")
            if not a or not b or kind not in _REL_EDGE or a == b:
                unresolved.append({"group": g.get("group") or g.get("name"), **rel,
                                   "reason": "label(s) did not resolve to a stored grounded equation" if not (a and b) else "unsupported relation"})
                continue
            edge, field = _REL_EDGE[kind]
            rid = node_id("REL", f"{a} {edge} {b}")
            rels_store[rid] = {"id": rid, "type": "Relationship", "from": a, "to": b, "edge": edge,
                               "source_relation": kind, "note": rel.get("note", ""),
                               "provenance_kind": "SYSTEM_INFERS",
                               "provenance_note": "cross-document comparison by the extraction agent for this subsystem group; both equations keep their own SOURCE_STATES provenance",
                               "recorded_at": now}
            lst = eqs[a].setdefault(field, [])
            if b not in lst:
                lst.append(b)
            if edge in ("EQUIVALENT_TO", "POSSIBLY_EQUIVALENT", "COMPETING_FORMULATION_WITH", "CONTRADICTS"):
                lst_b = eqs[b].setdefault(field, [])
                if a not in lst_b:
                    lst_b.append(a)
            attached += 1
    save("equations", eqs.values(), meta={"generated_at": now, "n": len(eqs)})
    save("relationships", rels_store.values(), meta={"generated_at": now, "n": len(rels_store)})
    report["cross_document_relations"] = {"attached": attached, "unresolved": unresolved}
    return report


# --------------------------------------------------------------------------
# adversarial verification results
# --------------------------------------------------------------------------

def apply_verification(verdicts: list[dict]) -> dict:
    """Record skeptic verdicts on equations/claims.

    A refuted object is NOT deleted (that would hide the loss); it is marked
    ``verification="REFUTED"`` with the skeptic's reason, detached from any
    model link, and excluded from ``retrieve_for_task`` equation lists.
    Confirmed objects get ``verification="CONFIRMED"`` plus any assumptions or
    validity limits the skeptic found the extractor had dropped (each such
    addition must itself carry a verbatim quote and is grounded here).
    """
    now = datetime.now(timezone.utc).isoformat()
    eqs = {e["id"]: e for e in load("equations")}
    cls = {c["id"]: c for c in load("claims")}
    models = {m["id"]: m for m in load("models")}
    grounders: dict[str, DocGrounder] = {}
    stats = {"confirmed": 0, "refuted": 0, "unknown_ids": 0, "added_assumptions": 0, "added_validity": 0,
             "ungrounded_additions": 0, "model_links_removed": 0, "unchanged": 0, "stale": 0, "uncertain": 0}
    for v in verdicts:
        oid = v.get("object_id", "")
        rec = eqs.get(oid) or cls.get(oid)
        if not rec:
            stats["unknown_ids"] += 1
            continue
        if rec.get("source_state") == "STALE_SOURCE":
            stats["stale"] += 1
            continue
        if not isinstance(v.get("refuted"), bool) or v.get("confidence") not in ("high", "medium", "low"):
            raise ValueError(f"Verification {oid} requires refuted: boolean and confidence")
        verdict_hash = hashlib.sha256(json.dumps({"verdict": v, "source": rec.get("source_sha256")},
                                                 sort_keys=True).encode()).hexdigest()
        if rec.get("verifier_result_hash") == verdict_hash:
            stats["unchanged"] += 1
            continue
        rec["verifier_result_hash"] = verdict_hash
        rec["verification"] = ("REFUTED" if v["refuted"] else
                               "UNCONFIRMED" if v["confidence"] == "low" else "CONFIRMED")
        rec["verifier_reason"] = v.get("reason", "")
        rec["verifier_confidence"] = v["confidence"]
        rec["verified_at"] = now
        linked_models = rec.get("model_ids", []) or rec.get("model_ids_before_refutation", [])
        override = v.get("model_support_override")
        if override in ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_FOUND_IN_SOURCE") and v["confidence"] != "low":
            for mid in linked_models:
                if mid in models:
                    models[mid].setdefault("source_support_state_extracted", models[mid].get("source_support_state"))
                    models[mid]["source_support_state_verified"] = override
                    models[mid]["source_support_state"] = override
                    models[mid]["source_support_verifier_note"] = v.get("reason", "")
        if v["refuted"]:
            stats["refuted"] += 1
            for mid in linked_models:
                if mid in models:
                    ids = models[mid].setdefault("refuted_equation_ids", [])
                    if oid not in ids:
                        ids.append(oid)
                        stats["model_links_removed"] += 1
            rec["model_ids_before_refutation"] = linked_models
            rec["model_ids"] = []
            continue
        if v["confidence"] == "low":
            stats["uncertain"] += 1
            continue
        stats["confirmed"] += 1
        doc_id = rec["document_id"]
        if doc_id not in grounders:
            grounders[doc_id] = DocGrounder(doc_id)
        g = grounders[doc_id]
        for add in v.get("missing_assumptions", []) or []:
            loc = g.locate(add.get("quote", ""), add.get("page"))
            if loc["state"] == "UNGROUNDED":
                stats["ungrounded_additions"] += 1
                continue
            rec["assumptions"] = _merge_list(rec.get("assumptions"), [add.get("statement")])
            rec["verifier_additions"] = _merge_list(rec.get("verifier_additions"), [{"kind": "assumption", **add, "page": loc["page"], "chunk_id": loc["chunk_id"], "grounded": loc["state"]}])
            stats["added_assumptions"] += 1
        for add in v.get("missing_validity", []) or []:
            loc = g.locate(add.get("quote", ""), add.get("page"))
            if loc["state"] == "UNGROUNDED":
                stats["ungrounded_additions"] += 1
                continue
            rec["validity"] = _merge_list(rec.get("validity"), [add.get("statement")])
            rec["verifier_additions"] = _merge_list(rec.get("verifier_additions"), [{"kind": "validity", **add, "page": loc["page"], "chunk_id": loc["chunk_id"], "grounded": loc["state"]}])
            stats["added_validity"] += 1
    save("equations", eqs.values(), meta={"generated_at": now, "n": len(eqs)})
    save("claims", cls.values(), meta={"generated_at": now, "n": len(cls)})
    save("models", models.values(), meta={"generated_at": now, "n": len(models),
                                          "source": "raosim/model_registry.py + extraction support states + verification"})
    return stats
