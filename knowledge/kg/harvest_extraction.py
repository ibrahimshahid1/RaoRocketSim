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
from .store import connect, load, save


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


def _current_grounder(doc_id: str, source_sha256: str) -> DocGrounder:
    """Refuse to relabel text from an old/mixed index as current PDF evidence.

    A rebuild can stop after replacing the manifest but before replacing the
    index. Every chunk and its document ingestion state must identify the same
    PDF as the current manifest before its text can ground a scientific object.
    An empty index retains the existing SOURCE_NOT_INGESTED handling.
    """
    g = DocGrounder(doc_id)
    if not g.chunks:
        return g
    conn = connect()
    try:
        state = conn.execute("SELECT source_sha256 FROM doc_state WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    if (not source_sha256 or not state or state["source_sha256"] != source_sha256
            or any(c.get("source_sha256") != source_sha256 for c in g.chunks)):
        raise ValueError(f"SOURCE_INDEX_IDENTITY_MISMATCH: {doc_id}; synchronize and ingest the current PDF before banking")
    return g


def harvest(model_results: list[dict], window_results: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    models_by_id = {m["id"]: m for m in load("models")}
    source_hashes = {d["id"]: d.get("sha256", "") for d in load("manifest")}
    grounders: dict[str, DocGrounder] = {}

    def grounder(doc_id: str) -> DocGrounder:
        if doc_id not in grounders:
            grounders[doc_id] = _current_grounder(doc_id, source_hashes.get(doc_id, ""))
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
        handled_equations = []
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
                res = resolve_symbol(v.get("symbol", ""), sub or None, units=v.get("units"), meaning=v.get("meaning"))
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
                "enrichment_candidate_ids": e.get("enrichment_candidate_ids", rec.get("enrichment_candidate_ids", [])),
                "corrects_candidate_ids": e.get("corrects_candidate_ids", rec.get("corrects_candidate_ids", [])),
                "model_ids": model_ids, "model_link_basis": ("registry_extraction:" + support_state) if eq_model_id else rec.get("model_link_basis", ""),
                "extracted_for_model": model_id or rec.get("extracted_for_model", ""),
                "provenance_id": pid, "provenance_kind": "SOURCE_STATES", "subsystem": sub or rec.get("subsystem", ""),
                "equivalent_to": rec.get("equivalent_to", []), "competing_with": rec.get("competing_with", []),
                "extraction_version": EXTRACTION_VERSION, "recorded_at": now,
            }
            report["n_equations"] += 1
            handled_equations.append((eid, e))
            for old_id in e.get("supersedes_ids", []):
                old = equations.get(old_id)
                old_number = str(old.get("equation_number", "")).strip("() ") if old else ""
                new_number = str(e.get("equation_number", "")).strip("() ")
                same_number = old and old_number == new_number
                if old and not same_number:
                    # A damaged composite, e.g. 9.26-9.28, may be recovered as
                    # individual 9.26, 9.27, 9.28a/b records. Match only an
                    # explicit inclusive range within one chapter; never
                    # disjoint lists, reversed ranges or another chapter.
                    composite = re.fullmatch(r"(\d+)\.(\d+)\s*[-–]\s*(\d+)\.(\d+)", old_number)
                    child = re.fullmatch(r"(\d+)\.(\d+)[a-z]?", new_number, re.I)
                    same_number = bool(composite and child
                        and int(composite[1]) == int(composite[3]) == int(child[1])
                        and int(composite[2]) <= int(child[2]) <= int(composite[4]))
                damaged = old and re.search(r"not\s+rendered|ocr[-\s]+garbled", old.get("form", ""), re.I)
                if old_id != eid and old and old.get("document_id") == doc_id and same_number and damaged and source["source_sha256"] and old.get("source_sha256") == source["source_sha256"]:
                    first_child = old.get("superseded_by") or eid
                    children = _merge_list(old.get("superseded_by_ids"), [first_child, eid])
                    old.update(active=False, source_state="SUPERSEDED_EXTRACTION",
                               superseded_by=first_child, superseded_by_ids=children)
                    equations[eid]["corrects_ids"] = _merge_list(equations[eid].get("corrects_ids"), [old_id])
                    report["superseded_incomplete_transcriptions"] = report.get("superseded_incomplete_transcriptions", 0) + 1
                else:
                    report.setdefault("unresolved_supersessions", []).append({"equation_id": eid, "old_id": old_id})
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

        # Only explicit extraction references create equation-level assumption
        # links. Similar wording or a shared symbol is not enough to infer scope.
        for eid, original in handled_equations:
            for statement in original.get("assumption_refs", []):
                matches = [a["id"] for a in entities.values() if a.get("type") == "Assumption"
                           and a.get("document_id") == doc_id and a.get("source_state") == "CURRENT"
                           and normalize(a.get("prose", "")) == normalize(statement)]
                if len(matches) == 1:
                    equations[eid]["assumption_ids"] = _merge_list(equations[eid].get("assumption_ids"), matches)
                    equations[eid]["assumption_link_basis"] = "explicit extraction reference to grounded same-document assumption"
                else:
                    report.setdefault("unresolved_assumption_links", []).append({"equation_id": eid, "statement": statement})

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
            # Comparing a source formulation with a different implementation
            # creates citation paths, never an assertion that code implements it.
            compared = []
            for label in sm.get("comparison_equation_labels", []):
                matches = [e["id"] for e in equations.values()
                           if e.get("document_id") == r["doc_id"]
                           and e.get("source_sha256") == source_hashes.get(r["doc_id"])
                           and e.get("source_state") == "CURRENT"
                           and normalize(e.get("label", "")) == normalize(label)]
                if len(matches) == 1:
                    compared.extend(matches)
                else:
                    report.setdefault("unresolved_comparison_links", []).append({"model_id": mid, "label": label})
            model["source_comparison_equation_ids"] = sorted(set(compared))
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
    "DERIVED_FROM": ("DERIVED_FROM", "derived_from"),
    "EQUIVALENT_TO": ("EQUIVALENT_TO", "equivalent_to"),
    "NOTATION_VARIANT": ("EQUIVALENT_TO", "equivalent_to"),
    "POSSIBLY_EQUIVALENT": ("POSSIBLY_EQUIVALENT", "possibly_equivalent"),
    "APPROXIMATES": ("APPROXIMATES", "approximates"),
    "DIFFERENT_ASSUMPTIONS": ("COMPETING_FORMULATION_WITH", "competing_with"),
    "CONTRADICTS": ("CONTRADICTS", "competing_with"),
}


def _relationship_id(a: str, edge: str, b: str) -> str:
    """Address the complete directed tuple; readable node_id truncates long IDs."""
    payload = json.dumps([a, edge, b], ensure_ascii=False, separators=(",", ":")).encode()
    return "REL_" + hashlib.sha256(payload).hexdigest()


def _relationship_tuple(record: dict) -> tuple[str, str, str]:
    identity = tuple(record.get(key) for key in ("from", "edge", "to"))
    if not all(isinstance(value, str) and value for value in identity):
        raise ValueError("Relationship identity requires nonempty from, edge and to strings")
    return identity


def _relationship_snapshot(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "history"}


def _relationship_assertion(record: dict) -> dict:
    return {key: value for key, value in record.items()
            if key not in {"history", "legacy_ids", "recorded_at"}}


def _migrate_relationship_ids(records: list[dict]) -> dict[str, dict]:
    """Retain legacy assertions and aliases while keeping one row per tuple."""
    migrated = {}
    for original in records:
        identity = _relationship_tuple(original)
        rid = _relationship_id(*identity)
        row = dict(original, id=rid)
        history = list(row.get("history", []))
        legacy_ids = set(row.get("legacy_ids", []))
        if original["id"] != rid:
            legacy_ids.add(original["id"])
            snapshot = _relationship_snapshot(original)
            if snapshot not in history:
                history.append(snapshot)
        if legacy_ids:
            row["legacy_ids"] = sorted(legacy_ids)
        if history:
            row["history"] = history
        previous = migrated.get(rid)
        if previous is not None:
            if _relationship_tuple(previous) != identity:
                raise ValueError(f"Relationship ID collision for distinct tuples: {rid}")
            combined_history = list(previous.get("history", []))
            for snapshot in history + [_relationship_snapshot(row)]:
                if snapshot not in combined_history and snapshot != _relationship_snapshot(previous):
                    combined_history.append(snapshot)
            previous.update(legacy_ids=sorted(set(previous.get("legacy_ids", [])) | legacy_ids),
                            history=combined_history)
        else:
            migrated[rid] = row
    return migrated


def harvest_groups(groups: list[dict]) -> dict:
    """Flatten grouped results, harvest them, then attach cross-document relations.

    Explicit relations are attached only when BOTH labels uniquely resolve to
    active, current, grounded, non-refuted equations of the stated documents
    with source fingerprints matching the current manifest. Unresolved or
    ambiguous labels are reported, never guessed. No derivations are inferred
    from equation order or label similarity. A relation never merges equations;
    both keep their own
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
    source_hashes = {d["id"]: d.get("sha256") for d in load("manifest")}
    by_doc_label: dict[tuple[str, str], list[str]] = {}
    for e in eqs.values():
        current_sha = source_hashes.get(e.get("document_id"))
        if (not e.get("active", True) or e.get("source_state") != "CURRENT"
                or e.get("verification") == "REFUTED"
                or e.get("grounded") not in {"GROUNDED", "GROUNDED_PAGE_SHIFTED"}
                or not current_sha or e.get("source_sha256") != current_sha):
            continue
        label = normalize(e.get("label", ""))
        if label:
            by_doc_label.setdefault((e["document_id"], label), []).append(e["id"])
    rels_store = _migrate_relationship_ids(load("relationships"))
    attached, unresolved = 0, []
    now = datetime.now(timezone.utc).isoformat()
    for g in groups:
        for rel in g.get("cross_document_relations", []) or []:
            a_matches = by_doc_label.get((rel.get("a_doc_id"), normalize(rel.get("a_label", ""))), [])
            b_matches = by_doc_label.get((rel.get("b_doc_id"), normalize(rel.get("b_label", ""))), [])
            a = a_matches[0] if len(a_matches) == 1 else None
            b = b_matches[0] if len(b_matches) == 1 else None
            kind = rel.get("relation", "")
            if not a or not b or kind not in _REL_EDGE or a == b:
                unresolved.append({"group": g.get("group") or g.get("name"), **rel,
                                   "reason": "label(s) did not uniquely resolve to a current grounded equation with matching source fingerprint" if not (a and b) else "unsupported relation"})
                continue
            edge, field = _REL_EDGE[kind]
            rid = _relationship_id(a, edge, b)
            previous = rels_store.get(rid)
            if previous is not None and _relationship_tuple(previous) != (a, edge, b):
                raise ValueError(f"Relationship ID collision for distinct tuples: {rid}")
            relation = {"id": rid, "type": "Relationship", "from": a, "to": b, "edge": edge,
                        "source_relation": kind, "note": rel.get("note", ""),
                        "provenance_kind": "SYSTEM_INFERS",
                        "provenance_note": "cross-document comparison by the extraction agent for this subsystem group; both equations keep their own SOURCE_STATES provenance",
                        "recorded_at": now}
            if previous is not None:
                if _relationship_assertion(previous) == _relationship_assertion(relation):
                    relation = previous  # replay preserves identity, timestamps and history
                else:
                    history = list(previous.get("history", []))
                    snapshot = _relationship_snapshot(previous)
                    if snapshot not in history:
                        history.append(snapshot)
                    relation["history"] = history
                    if previous.get("legacy_ids"):
                        relation["legacy_ids"] = previous["legacy_ids"]
            rels_store[rid] = relation
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
    source_hashes = {d["id"]: d.get("sha256") for d in load("manifest")}
    grounders: dict[str, DocGrounder] = {}
    stats = {"confirmed": 0, "refuted": 0, "unknown_ids": 0, "added_assumptions": 0, "added_validity": 0,
             "ungrounded_additions": 0, "model_links_removed": 0, "unchanged": 0, "stale": 0, "uncertain": 0,
             "retired_or_noncurrent": 0}
    for v in verdicts:
        oid = v.get("object_id", "")
        rec = eqs.get(oid) or cls.get(oid)
        if not rec:
            stats["unknown_ids"] += 1
            continue
        supplied_sha = v.get("source_sha256")
        current_sha = source_hashes.get(rec.get("document_id"))
        if (rec.get("source_state") == "STALE_SOURCE" or not current_sha
                or rec.get("source_sha256") != current_sha
                or (supplied_sha and supplied_sha != current_sha)):
            stats["stale"] += 1
            continue
        if not rec.get("active", True) or rec.get("source_state", "CURRENT") != "CURRENT":
            # A historical verdict must not change current model support via
            # model_ids retained on a superseded extraction for audit history.
            stats["retired_or_noncurrent"] += 1
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
        rec["verifier_source_sha256"] = supplied_sha or rec.get("source_sha256")
        # A replacement verdict cannot inherit evidence the new reviewer did
        # not supply. Completed verdict journals retain the preceding audit.
        rec.pop("verifier_evidence", None)
        rec.pop("verifier_scope_note", None)
        if v.get("evidence"):
            rec["verifier_evidence"] = v["evidence"]
        if v.get("scope_note"):
            rec["verifier_scope_note"] = v["scope_note"]
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
        if not (v.get("missing_assumptions") or v.get("missing_validity")):
            continue
        if doc_id not in grounders:
            grounders[doc_id] = _current_grounder(doc_id, current_sha)
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
