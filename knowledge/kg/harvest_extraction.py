"""Store grounded literature extractions (equations, claims, validity, constants).

Input: the structured results of the extraction workflow.  Every object is
passed through :class:`kg.ground.DocGrounder`; objects whose verbatim anchor
cannot be found in the indexed source text are NOT stored - they are listed in
the returned report under ``ungrounded`` so the loss is visible.

Model links: per-source citation assessments and fingerprinted implementation
reviews are retained independently. Only an equivalent structured review may
create IMPLEMENTS or claim SUPPORTS; comparison formulations remain CITES.
Historical single-source records stay readable. A secondary assessment cannot
erase a prior source's state, and a citation never establishes physical validity.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from .graph import equation_formulation_state
from .ground import DocGrounder, normalize
from .locks import writes_stores
from .quantities import resolve_symbol
from .schema import EXTRACTION_VERSION, node_id
from .store import connect, load, save
from . import verification_freshness
from . import candidate_links


def _short_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:8]


def claim_quantity_ids(claim: dict) -> list[str]:
    """Resolve explicit tokens without interpreting English connectors as symbols."""
    connectors = {"a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
                  "is", "of", "on", "or", "the", "to", "with"}
    found = set()
    for word in re.findall(r"[A-Za-z_*][A-Za-z0-9_*]+",
                           f"{claim.get('subject', '')} {claim.get('object', '')}"):
        if word in connectors:
            continue
        resolved = resolve_symbol(word)
        if resolved["state"] == "RESOLVED":
            found.add(resolved["candidates"][0]["qid"])
    return sorted(found)


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


FIDELITY_CLASSES = ("EXACT", "EQUIVALENT", "CONDITIONALLY_EQUIVALENT", "SUPPORTED_APPROXIMATION",
                    "REPOSITORY_EXTENSION", "DIFFERENT_FORMULA", "BAD_ATTRIBUTION", "INSUFFICIENT_EVIDENCE",
                    "UNRESOLVED")
# Only a reviewed equivalence lets a model IMPLEMENT a source equation; every
# other class is a citation path (CITES) and never an implementation claim.
IMPLEMENTS_FIDELITY = ("EXACT", "EQUIVALENT", "CONDITIONALLY_EQUIVALENT")
# Mechanical worker roles never make scientific verdicts (AGENTIC_ROLES_2026-10-16.md).
WORKER_ROLES = ("S1", "S2", "S3")

_EQ_ID = re.compile(r"\bEQ_[A-Z0-9_]+\b")
_EQ_RANGE = re.compile(r"[\(\[]\s*(\d+)\s*[\)\]]\s*[-–]\s*[\(\[]\s*(\d+)\s*[\)\]]")
_EQ_BRACKETED = re.compile(r"[\(\[]\s*(\d+(?:[.\-–]\d+)?[a-z]?)\s*[\)\]]", re.I)
_EQ_BARE = re.compile(r"\bEqs?\.?\s*(\d+(?:[.\-–]\d+)?[a-z]?)\b", re.I)


def _equation_number_key(value) -> str:
    """'(4-23)', '4–23', '[7]' and ' 7 ' compare as printed numbers, not text."""
    return re.sub(r"[\s()\[\]]", "", str(value or "")).replace("–", "-").lower()


def _label_selector(label: str) -> dict:
    """Read one support/comparison label as exact label, equation ids and printed numbers.

    Short review tags ("Eq. (1)", "Eq. [7]", "Eq. (4-23)", "Eqs. (26)-(32)") name
    printed equation numbers; an exact stored label or an EQ_ id selects directly.
    """
    text = str(label or "")
    numbers = set()
    for low, high in _EQ_RANGE.findall(text):
        if int(low) <= int(high) <= int(low) + 50:
            numbers.update(str(n) for n in range(int(low), int(high) + 1))
    numbers.update(_equation_number_key(n) for n in _EQ_BRACKETED.findall(text) + _EQ_BARE.findall(text))
    return {"label": label, "exact": normalize(text), "ids": set(_EQ_ID.findall(text)), "numbers": numbers}


def _selects(selector: dict, equation: dict, *, by_number: bool = True) -> bool:
    if equation.get("id") in selector["ids"] or normalize(equation.get("label", "")) == selector["exact"]:
        return True
    return by_number and bool(equation.get("equation_number")) and \
        _equation_number_key(equation["equation_number"]) in selector["numbers"]


def _resolve_labels(selector: dict, candidates: list[dict]) -> tuple[list[str], list[str]]:
    """Resolve a label to stored equations: exact label or id first, else one per printed number.

    Returns (equation ids, unresolved reasons). Ambiguity is never resolved by guessing.
    """
    direct = [e["id"] for e in candidates if _selects(selector, e, by_number=False)]
    if direct:
        return (direct, []) if len(direct) == 1 or selector["ids"] else ([], [f"ambiguous label ({len(direct)} matches)"])
    ids, unresolved = [], []
    for number in sorted(selector["numbers"]):
        hits = [e["id"] for e in candidates if e.get("equation_number")
                and _equation_number_key(e["equation_number"]) == number]
        if len(hits) == 1:
            ids.extend(hits)
        else:
            unresolved.append(f"Eq. {number}: {len(hits)} current matches")
    if not selector["numbers"]:
        unresolved.append("no exact label, equation id or printed number matched")
    return ids, unresolved


def _model_review_record(result: dict, source_sha256: str, now: str) -> dict:
    """Bind a model-versus-source comparison to source and implementation fingerprints.

    A structural code change or a replaced PDF makes the comparison stale; the
    review never certifies execution reach, sensitivity or physical validity.
    """
    from .implementation_freshness import definition_fingerprint
    review = dict(result["model_review"])
    if review.get("fidelity") not in FIDELITY_CLASSES:
        raise ValueError(f"{result.get('target')}: model_review.fidelity must be one of {FIDELITY_CLASSES}")
    refs = review.get("implementation") or []
    if not refs or not all(isinstance(x, dict) and x.get("file") and x.get("symbol") for x in refs):
        raise ValueError(f"{result.get('target')}: model_review.implementation needs exact file/symbol pairs")
    if not (result.get("reviewer") or {}).get("role"):
        raise ValueError(f"{result.get('target')}: model reviews require reviewer identity")
    fingerprints = [definition_fingerprint(x["file"], x["symbol"]) for x in refs]
    missing = [f for f in fingerprints if f["state"] != "VERIFIED"]
    if missing:
        raise ValueError(f"{result.get('target')}: implementation definition not uniquely located: {missing}")
    review.update(implementation_fingerprints=fingerprints, reviewed_source_sha256=source_sha256,
                  reviewed_document_id=result["doc_id"], reviewer=result["reviewer"], recorded_at=now,
                  citation_is_not_support=True, physical_validation=review.get("physical_validation", "NOT_ESTABLISHED"),
                  sensitivity=review.get("sensitivity", "NOT_MEASURED"))
    payload = {k: v for k, v in review.items() if k not in ("recorded_at",)}
    review["review_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return review


def model_source_reviews(model: dict) -> list[dict]:
    """Current review per source; old stores keep their single-review contract."""
    reviews = {r.get("reviewed_document_id"): r for r in model.get("source_reviews", [])
               if r.get("reviewed_document_id")}
    legacy = model.get("source_review") or {}
    if legacy.get("reviewed_document_id"):
        reviews.setdefault(legacy["reviewed_document_id"], legacy)
    return [reviews[key] for key in sorted(reviews)]


def _single_review_state(review: dict, document: dict | None, *, repo_root=None) -> dict:
    from .implementation_freshness import definition_fingerprint
    if document is None:
        return {"state": "SOURCE_NOT_INGESTED", "reason": "Reviewed source is absent from the manifest."}
    if not review.get("reviewed_source_sha256") or not review.get("implementation_fingerprints"):
        return {"state": "NOT_CHECKED", "reason": "Historical review lacks source or implementation fingerprints."}
    if document.get("sha256") != review.get("reviewed_source_sha256"):
        return {"state": "STALE_SOURCE", "reason": "Source bytes differ from the reviewed fingerprint."}
    changed = []
    for old in review.get("implementation_fingerprints", []):
        now = definition_fingerprint(old["file"], old["symbol"], repo_root=repo_root)
        if now.get("definition_sha256") != old.get("definition_sha256"):
            changed.append({"file": old["file"], "symbol": old["symbol"], "state": now["state"]})
    if changed:
        return {"state": "IMPLEMENTATION_CHANGED_REVIEW_REQUIRED", "changed": changed,
                "reason": "An implementation definition changed structurally after the comparison."}
    return {"state": "CURRENT", "fidelity": review.get("fidelity"),
            "reason": "Source and implementation fingerprints match the recorded comparison."}


def model_review_state(model: dict, document: dict | None = None, *, documents=None, repo_root=None) -> dict:
    """Aggregate retained source reviews without allowing a new source to hide a stale one.

    Individual fidelity classes remain visible. MIXED is an aggregate label only,
    never a new scientific verdict or an assertion of physical validation.
    """
    reviews = model_source_reviews(model)
    if not reviews:
        return {"state": "NOT_CHECKED", "reason": "No structured model-versus-source review is recorded."}
    if documents is None:
        documents = {d["id"]: d for d in load("manifest")}
        if document is not None:
            # Historical callers pass the primary document positionally.
            primary = model.get("source_review") or reviews[0]
            documents[primary.get("reviewed_document_id")] = document
    states = [{**review, **_single_review_state(review, documents.get(review.get("reviewed_document_id")),
                                               repo_root=repo_root)} for review in reviews]
    for item in states:
        changed = (model.get("source_registry_review_required") or {}).get(item["reviewed_document_id"])
        if changed:
            item.update(state="NOT_CHECKED", reason="Registry metadata changed after this source comparison: " + ", ".join(changed))
    priority = ("IMPLEMENTATION_CHANGED_REVIEW_REQUIRED", "STALE_SOURCE", "SOURCE_NOT_INGESTED", "NOT_CHECKED")
    state = next((value for value in priority if any(r["state"] == value for r in states)), "CURRENT")
    fidelities = sorted({r.get("fidelity", "UNRESOLVED") for r in states})
    fidelity = fidelities[0] if len(fidelities) == 1 else "MIXED"
    selected = next((r for r in states if r["state"] == state), states[0])
    return {"state": state, "reason": selected.get("reason"), "fidelity": fidelity,
            "fidelities": fidelities, "source_reviews": states,
            "changed": [change for r in states for change in r.get("changed", [])],
            "physics_change_required": any(r.get("physics_change_required") for r in states)}


def model_source_assessments(model: dict) -> list[dict]:
    """Per-document citation assessments with source-bound verifier overrides."""
    assessments = {r.get("document_id"): dict(r) for r in model.get("source_assessments", [])
                   if r.get("document_id")}
    primary = model.get("source_document_id") or (model.get("source_review") or {}).get("reviewed_document_id")
    if not primary and len(assessments) == 1:
        primary = next(iter(assessments))
    if primary and primary not in assessments and model.get("source_support_sha256"):
        assessments[primary] = {"document_id": primary, "source_sha256": model["source_support_sha256"],
                                "state": model.get("source_support_state", "NOT_CHECKED"),
                                "notes": model.get("source_support_notes", ""),
                                "comparison_equation_ids": model.get("source_comparison_equation_ids", [])}
    # Historical single-source verdicts identify the primary source via its
    # support fingerprint. Explicit identities on newer verdicts must also fit.
    if (primary in assessments and model.get("source_support_state_verified")
            and model.get("source_support_sha256") == assessments[primary].get("source_sha256")
            and model.get("source_support_verifier_document_id", primary) == primary
            and model.get("source_support_verifier_source_sha256", model.get("source_support_sha256"))
                == assessments[primary].get("source_sha256")):
        assessments[primary]["state"] = model["source_support_state_verified"]
    for verdict in model.get("source_support_verification_overrides", []):
        assessment = assessments.get(verdict.get("document_id"))
        if (assessment and verdict.get("source_sha256")
                and verdict["source_sha256"] == assessment.get("source_sha256")):
            assessment.update(state=verdict["state"], verifier_note=verdict.get("reason", ""))
    for did in model.get("source_registry_review_required") or {}:
        if did in assessments:
            assessments[did]["state"] = "NOT_CHECKED"
    return [assessments[key] for key in sorted(assessments)]


def aggregate_source_support(assessments: list[dict], documents: dict) -> tuple[str, str]:
    """Conservative citation support over every retained source, with live fingerprints."""
    if not assessments:
        return "NOT_CHECKED", "No source assessment is recorded."
    states = []
    for assessment in assessments:
        doc = documents.get(assessment.get("document_id"))
        state = assessment.get("state", "UNKNOWN")
        if doc is None:
            state = "SOURCE_NOT_INGESTED"
        elif not assessment.get("source_sha256"):
            state = "NOT_CHECKED"
        elif doc.get("sha256") != assessment.get("source_sha256"):
            state = "STALE_SOURCE"
        states.append(state)
    if len(set(states)) == 1:
        state = states[0]
    elif any(s in ("SUPPORTED", "PARTIALLY_SUPPORTED") for s in states):
        state = "PARTIALLY_SUPPORTED"
    elif "STALE_SOURCE" in states:
        state = "STALE_SOURCE"
    else:
        state = "NOT_CHECKED"
    notes = "; ".join(f"{a['document_id']}: {state_value}" + (f" ({a['notes']})" if a.get("notes") else "")
                      for a, state_value in zip(assessments, states))
    return state, notes


@writes_stores("harvest extraction results")
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
    parser_candidates = {c["id"]: c for c in load("enrichments")}
    # Capture before in-place link/carry-forward edits; only source assertion
    # content participates in the independent-verification fingerprint.
    previous_assertions = json.loads(json.dumps({"equations": equations, "claims": claims,
                                                "entities": entities, "validity": validity}))
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
        reviewer = result.get("reviewer") or {}
        if reviewer:
            if not reviewer.get("role") or not reviewer.get("model"):
                raise ValueError(f"{result.get('target')}: reviewer identity requires role and model")
            # The transcribing role is recorded so it cannot later count as its own independent verifier.
            source["transcribed_by"] = {"role": reviewer["role"], "model": reviewer["model"]}
            extractor = f"{extractor}; original-page transcription by role {reviewer['role']} ({reviewer['model']})"

        def candidate_fields(obj: dict, object_id: str, page: int) -> dict:
            """Keep only explicitly reviewed, identity-bound parser references."""
            fields = {}
            for field, relation in candidate_links.FIELDS.items():
                if field not in obj:
                    continue  # existing references remain inspectable; the graph rechecks identity
                requested = obj[field]
                if not isinstance(requested, list) or any(not isinstance(cid, str) for cid in requested):
                    report.setdefault("unresolved_candidate_links", []).append(
                        {"object_id": object_id, "field": field, "reason": "CANDIDATE_IDS_MUST_BE_STRING_LIST"})
                    fields[field] = []
                    continue
                fields[field] = []
                for cid in dict.fromkeys(requested):
                    problem = ("EXTRACTION_SOURCE_MISMATCH" if not result.get("source_sha256")
                               or result["source_sha256"] != source["source_sha256"] else
                               candidate_links.link_problem(
                                   parser_candidates.get(cid, {}), document_id=doc_id, page=page,
                                   source_sha256=source["source_sha256"], reviewer=reviewer,
                                   correction=relation == "CORRECTS"))
                    if problem:
                        report.setdefault("unresolved_candidate_links", []).append(
                            {"object_id": object_id, "candidate_id": cid, "field": field,
                             "document_id": doc_id, "page": page, "reason": problem})
                    else:
                        fields[field].append(cid)
            return fields

        sub = _subsystem_for_model(models_by_id, model_id) if model_id else ""
        # An equation is linked to the registry model ONLY when the extractor found
        # support and (if it named which equations support it) this is one of them.
        sm = result.get("supports_model") or {}
        support_state = sm.get("state", "")
        handled_equations = []
        new_review = bool(result.get("model_review"))
        retained_review = next((r for r in model_source_reviews(models_by_id.get(model_id, {}))
                                if r.get("reviewed_document_id") == doc_id), {})
        review = result.get("model_review") or retained_review
        inherited_review = bool(review and not new_review)
        if inherited_review:
            support_state = review.get("support_state", support_state)
            support_labels = review.get("resolved_implementation_equation_ids")
            if support_labels is None and "equation_links" in review:
                support_labels = [row["equation_id"] for row in review["equation_links"]
                                  if row.get("relation") == "IMPLEMENTS" and row.get("equation_id")]
            if support_labels is None:
                support_labels = sm.get("equation_labels") or []
        else:
            support_labels = sm.get("equation_labels") or []
        support_selectors = [_label_selector(x) for x in support_labels]
        # A replacement review owns this model/source pair. Old links cannot
        # survive a narrowed equivalence or a new DIFFERENT_FORMULA verdict.
        if model_id and review:
            if new_review or review.get("fidelity") not in IMPLEMENTS_FIDELITY:
                for equation in equations.values():
                    if equation.get("document_id") == doc_id and model_id in equation.get("model_ids", []):
                        equation["model_ids"] = [mid for mid in equation["model_ids"] if mid != model_id]
            if review.get("fidelity") not in IMPLEMENTS_FIDELITY:
                for claim in claims.values():
                    if claim.get("document_id") == doc_id and model_id in claim.get("model_ids", []):
                        claim["model_ids"] = [mid for mid in claim["model_ids"] if mid != model_id]
                        claim["comparison_model_ids"] = _merge_list(claim.get("comparison_model_ids"), [model_id])

        def may_implement() -> bool:
            if not model_id or support_state not in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                return False
            # A structured review links only what it names, and only a reviewed
            # equivalence; a different formula is cited (CITES), never implemented.
            if inherited_review and _single_review_state(
                    review, {"sha256": source["source_sha256"]})["state"] != "CURRENT":
                return False
            return not review or review.get("fidelity") in IMPLEMENTS_FIDELITY

        def links_model(equation: dict) -> bool:
            if not may_implement():
                return False
            if not support_selectors:
                return not review  # legacy extraction semantics: every equation of a supporting result
            return any(_selects(s, equation) for s in support_selectors)

        # ---- equations ----
        for e in result.get("equations", []) or []:
            loc = ground(doc_id, e, "equation")
            if not loc:
                continue
            page = loc["page"]
            eid = f"EQ_{doc_id[4:]}_{_short_hash(e.get('form','') + e.get('quote',''))}".upper()
            eq_model_id = model_id if links_model({"id": eid, "label": e.get("label", ""),
                                                   "equation_number": e.get("equation_number", "")}) else None
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
                **candidate_fields(e, eid, page),
                "model_ids": model_ids,
                "model_link_basis": ((("model_review:" + str(review.get("fidelity"))) if review else ("registry_extraction:" + support_state))
                                     if eq_model_id else rec.get("model_link_basis", "")),
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
                # Replaceable: a flagged incomplete transcription or one an independent review refuted.
                damaged = old and (equation_formulation_state(old) == "INCOMPLETE_TRANSCRIPTION"
                                   or old.get("verification") == "REFUTED")
                if old_id != eid and old and old.get("document_id") == doc_id and same_number and damaged and source["source_sha256"] and old.get("source_sha256") == source["source_sha256"]:
                    first_child = old.get("superseded_by") or eid
                    children = _merge_list(old.get("superseded_by_ids"), [first_child, eid])
                    old.update(active=False, source_state="SUPERSEDED_EXTRACTION",
                               superseded_by=first_child, superseded_by_ids=children)
                    equations[eid]["corrects_ids"] = _merge_list(equations[eid].get("corrects_ids"), [old_id])
                    report["superseded_incomplete_transcriptions"] = report.get("superseded_incomplete_transcriptions", 0) + 1
                    # What the superseded record carried (page-stated assumptions,
                    # validity, model links) is never dropped silently: the
                    # correcting review decides, with a reason, what carries over.
                    new_rec = equations[eid]
                    dropped = {k: [x for x in (old.get(k) or []) if x not in (new_rec.get(k) or [])]
                               for k in ("assumptions", "validity", "model_ids")}
                    decision = e.get("carry_forward")
                    if isinstance(decision, dict) and str(decision.get("reason") or "").strip():
                        for k in ("assumptions", "validity"):
                            if decision.get(k) is True:
                                new_rec[k] = _merge_list(new_rec.get(k), old.get(k))
                        if decision.get("model_ids") is True and new_rec.get("verification") != "REFUTED":
                            new_rec["model_ids"] = sorted(set(new_rec.get("model_ids") or []) | set(old.get("model_ids") or []))
                        new_rec["carry_forward_decisions"] = _merge_list(new_rec.get("carry_forward_decisions"),
                                                                         [{**decision, "from_id": old_id}])
                    elif any(dropped.values()):
                        report.setdefault("supersession_carry_forward_undecided", []).append(
                            {"equation_id": eid, "old_id": old_id, "dropped": {k: v for k, v in dropped.items() if v}})
                else:
                    report.setdefault("unresolved_supersessions", []).append({"equation_id": eid, "old_id": old_id})
        # ---- review-named implementation links ----
        # Support labels may name equations banked by earlier results; every
        # label must resolve to current equations of this document or it is
        # reported (an S3 integration mismatch), never silently dropped.
        if model_id and support_selectors:
            if not may_implement() and not inherited_review:
                report.setdefault("support_labels_refused", []).append(
                    {"model_id": model_id, "labels": list(sm.get("equation_labels") or []),
                     "reason": f"support {support_state or 'UNSTATED'} with review fidelity "
                               f"{review.get('fidelity')} does not permit IMPLEMENTS"})
            elif may_implement():
                current = [r for r in equations.values() if r.get("document_id") == doc_id
                           and r.get("source_sha256") == source["source_sha256"]
                           and r.get("source_state", "CURRENT") == "CURRENT" and r.get("active", True)]
                for selector in support_selectors:
                    ids, unresolved = _resolve_labels(selector, current)
                    for linked in ids:
                        linked_rec = equations[linked]
                        if linked_rec.get("verification") == "REFUTED":
                            unresolved.append(f"{linked} is independently refuted")
                            continue
                        if model_id not in (linked_rec.get("model_ids") or []):
                            linked_rec["model_ids"] = sorted(set(linked_rec.get("model_ids") or []) | {model_id})
                            linked_rec["model_link_basis"] = "model_review:" + str(review.get("fidelity") or support_state)
                    if unresolved:
                        report.setdefault("unresolved_support_links", []).append(
                            {"model_id": model_id, "label": selector["label"], "unresolved": unresolved})
        # ---- claims ----
        for c in result.get("claims", []) or []:
            loc = ground(doc_id, c, "claim")
            if not loc:
                continue
            page = loc["page"]
            cid = f"CLM_{doc_id[4:]}_{_short_hash(c.get('statement','') + c.get('quote',''))}".upper()
            rec = _current_record(claims.get(cid, {}), source["source_sha256"])
            cm = model_id if may_implement() else None
            model_ids = sorted(set(rec.get("model_ids", [])) | ({cm} if cm else set()))
            comparison_model_ids = _merge_list(rec.get("comparison_model_ids"),
                                               [model_id] if model_id and not cm else [])
            if cm:
                comparison_model_ids = [mid for mid in comparison_model_ids if mid != cm]
            if rec.get("verification") == "REFUTED":
                model_ids = []
            pid = prov(cid, doc_id, page, loc["chunk_id"], c.get("quote", ""), extractor)
            qids = claim_quantity_ids(c)
            claims[cid] = {
                **rec,
                **source,
                "id": cid, "type": "Claim", "claim_type": c.get("claim_type"), "subject": c.get("subject"),
                "predicate": c.get("predicate"), "object": c.get("object"), "statement": c.get("statement"),
                "qualifiers": c.get("qualifiers", []) or [], "document_id": doc_id, "page": page,
                "reported_page": c.get("page"), "chunk_id": loc["chunk_id"], "quote": c.get("quote"),
                "grounded": loc["state"], "confidence": c.get("confidence"), "model_ids": model_ids,
                "comparison_model_ids": comparison_model_ids, "extracted_for_model": model_id or rec.get("extracted_for_model", ""),
                "quantity_ids": sorted(set(qids)), "contradicts": rec.get("contradicts", []),
                **candidate_fields(c, cid, page),
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
                **candidate_fields(v, vid, page),
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
                             **candidate_fields(a, aid, loc["page"]),
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
                             **candidate_fields(k, kid, loc["page"]),
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
            if r.get("model_review") and model.get("source_registry_review_required"):
                # A newly fingerprinted comparison renews this source only.
                model["source_registry_review_required"].pop(r["doc_id"], None)
                if not model["source_registry_review_required"]:
                    model.pop("source_registry_review_required")
            doc_id, source_sha = r["doc_id"], source_hashes.get(r["doc_id"], "")
            assessments = {a["document_id"]: a for a in model_source_assessments(model)}
            previous = assessments.get(doc_id, {})
            # Comparing a source formulation with a different implementation
            # creates citation paths, never an assertion that code implements it.
            compared = []
            current_doc = [e for e in equations.values()
                           if e.get("document_id") == doc_id and e.get("source_sha256") == source_sha
                           and e.get("source_state") == "CURRENT" and e.get("active", True)
                           and e.get("verification") != "REFUTED"]
            for label in sm.get("comparison_equation_labels", []):
                ids, unresolved = _resolve_labels(_label_selector(label), current_doc)
                compared.extend(ids)
                if unresolved:
                    report.setdefault("unresolved_comparison_links", []).append(
                        {"model_id": mid, "label": label, "unresolved": unresolved})
            actual_implements = sorted(e["id"] for e in current_doc if mid in e.get("model_ids", []))
            retained = next((review for review in model_source_reviews(model)
                             if review.get("reviewed_document_id") == doc_id), {})
            preserve_review = bool(retained and not r.get("model_review")
                                   and retained.get("reviewed_source_sha256") == source_sha)
            if preserve_review:
                valid_ids = {e["id"] for e in current_doc}
                compared.extend(eid for eid in retained.get("resolved_comparison_equation_ids",
                                                           previous.get("comparison_equation_ids", []))
                                if eid in valid_ids)
            compared = sorted(set(compared))
            primary = model.get("source_document_id") or (model.get("source_review") or {}).get("reviewed_document_id") or doc_id
            if primary == doc_id:
                source_changed = model.get("source_support_sha256") not in (None, "", source_sha)
                if source_changed or model.get("source_support_state") == "NOT_CHECKED":
                    for key in ("source_support_state_verified", "source_support_verifier_note"):
                        model.pop(key, None)
                model["source_support_state_extracted"] = sm.get("state", "UNKNOWN")
                model["source_support_equation_labels"] = sm.get("equation_labels", [])
                model["source_support_sha256"] = source_sha
            assessment = {"document_id": doc_id, "source_sha256": source_sha,
                          "state": ((model.get("source_support_state_verified") if primary == doc_id else None)
                                    or (retained.get("support_state") if preserve_review else None)
                                    or sm.get("state", "UNKNOWN")),
                          "notes": (retained.get("support_notes", previous.get("notes", "")) if preserve_review
                                    else sm.get("mismatch_notes", "")),
                          "support_equation_ids": actual_implements, "comparison_equation_ids": compared,
                          "comparison_claim_ids": sorted(c["id"] for c in claims.values()
                                                         if c.get("document_id") == doc_id
                                                         and mid in c.get("comparison_model_ids", []))}
            if previous and previous != assessment:
                model["source_assessment_history"] = _merge_list(model.get("source_assessment_history"), [previous])
            assessments[doc_id] = assessment
            model["source_assessments"] = [assessments[key] for key in sorted(assessments)]
            # Legacy fingerprint fields identify the registry's primary source.
            # Aggregate support cannot erase another source's negative finding.
            aggregate, notes = aggregate_source_support(model_source_assessments(model),
                                                        {key: {"sha256": value} for key, value in source_hashes.items()})
            model["source_support_state"] = aggregate
            model["source_support_notes"] = notes
            model["source_comparison_equation_ids"] = sorted({eid for a in assessments.values()
                                                               for eid in a.get("comparison_equation_ids", [])})
            report["model_support"][mid] = aggregate
            if r.get("model_review"):
                raw_review = r["model_review"]
                if "equation_links" in raw_review:
                    declared = raw_review.get("equation_links") or []
                    expected_implements = sorted({link.get("equation_id") for link in declared
                                                   if link.get("relation") == "IMPLEMENTS" and link.get("equation_id")})
                    expected_cites = sorted({link.get("equation_id") for link in declared
                                              if link.get("relation") in ("CITES", "CITES_ONLY")
                                              and link.get("equation_id")})
                    actual_cites = sorted(set(compared) - set(actual_implements))
                    invalid = [link for link in declared if not link.get("equation_id")
                               or link.get("relation") not in ("IMPLEMENTS", "CITES", "CITES_ONLY")]
                    if expected_implements != actual_implements or expected_cites != actual_cites or invalid:
                        report.setdefault("model_review_equation_link_mismatches", []).append(
                            {"model_id": mid, "doc_id": doc_id,
                             "declared_implements": expected_implements, "actual_implements": actual_implements,
                             "declared_cites": expected_cites, "actual_cites": actual_cites, "invalid": invalid})
                review = _model_review_record(r, source_sha, now)
                # Resolved links are fingerprinted with the review, so a replay
                # cannot silently claim the same review while changing its scope.
                review.update(support_state=assessment["state"], support_notes=assessment["notes"],
                              resolved_implementation_equation_ids=actual_implements,
                              resolved_comparison_equation_ids=compared,
                              resolved_comparison_claim_ids=assessment["comparison_claim_ids"])
                review["review_hash"] = hashlib.sha256(json.dumps(
                    {key: value for key, value in review.items() if key not in ("recorded_at", "review_hash")},
                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                reviews = {item["reviewed_document_id"]: item for item in model_source_reviews(model)}
                old = reviews.get(doc_id)
                if old and old.get("review_hash") == review["review_hash"]:
                    review = old
                elif old:
                    model["source_review_history"] = _merge_list(model.get("source_review_history"), [old])
                reviews[doc_id] = review
                model["source_reviews"] = [reviews[key] for key in sorted(reviews)]
                if (not model.get("source_review") or primary == doc_id
                        or model["source_review"].get("reviewed_document_id") == doc_id):
                    model["source_review"] = review
    for r in window_results:
        handle(r, "extraction workflow (document-driven)", None)

    invalidated = []
    for name, records in (("equations", equations), ("claims", claims),
                          ("entities", entities), ("validity", validity)):
        for oid, rec in records.items():
            if verification_freshness.reconcile_record(previous_assertions[name].get(oid, {}), rec, now=now):
                invalidated.append(oid)
    report["content_verifications_invalidated"] = sorted(invalidated)
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


@writes_stores("harvest grouped extraction results")
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

@writes_stores("apply independent verification verdicts")
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
    # Literature constants, assumptions and validity statements are source
    # assertions too; they are verified with the same identity rules.
    ents = {e["id"]: e for e in load("entities")}
    vals = {v["id"]: v for v in load("validity")}
    models = {m["id"]: m for m in load("models")}
    source_hashes = {d["id"]: d.get("sha256") for d in load("manifest")}
    grounders: dict[str, DocGrounder] = {}
    stats = {"confirmed": 0, "refuted": 0, "unknown_ids": 0, "added_assumptions": 0, "added_validity": 0,
             "ungrounded_additions": 0, "model_links_removed": 0, "unchanged": 0, "stale": 0, "uncertain": 0,
             "retired_or_noncurrent": 0, "self_verification_refused": 0, "unsigned_refused": 0,
             "worker_role_refused": 0, "content_mismatch": 0, "stale_legacy_replay": 0,
             "content_hash_required": 0}
    dirty = False  # stores are rewritten only when a verdict changed something
    for v in verdicts:
        oid = v.get("object_id", "")
        rec = eqs.get(oid) or cls.get(oid) or ents.get(oid) or vals.get(oid)
        if rec is not None and rec.get("provenance_kind") not in (None, "SOURCE_STATES"):
            rec = None  # only source-stated assertions can be confirmed against a source page
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
        reviewer = v.get("reviewer") or {}
        transcriber = rec.get("transcribed_by") or {}
        if reviewer.get("role") in WORKER_ROLES:
            # Mechanical workers (S1-S3) never make scientific verdicts.
            stats["worker_role_refused"] += 1
            continue
        if transcriber and not (reviewer.get("role") and reviewer.get("model")):
            # An anonymous verdict cannot show independence from a recorded transcriber.
            stats["unsigned_refused"] += 1
            continue
        if reviewer.get("role") and reviewer.get("role") == transcriber.get("role"):
            # The role that transcribed an object cannot be its independent verifier.
            stats["self_verification_refused"] += 1
            continue
        inspected_content = verification_freshness.content_sha256(rec)
        verdict_hash = hashlib.sha256(json.dumps({"verdict": v, "source": rec.get("source_sha256")},
                                                 sort_keys=True).encode()).hexdigest()
        if (rec.get("verifier_result_hash") == verdict_hash
                and rec.get("verifier_content_sha256") in (None, inspected_content)):
            stats["unchanged"] += 1
            continue
        if v.get("content_sha256") and v["content_sha256"] != inspected_content:
            stats["content_mismatch"] += 1
            continue
        # A legacy verdict carried only object/source identity. It must not be
        # replayed to certify content added after its original inspection.
        if (not v.get("content_sha256")
                and verdict_hash in verification_freshness.historical_verdict_hashes(rec)):
            stats["stale_legacy_replay"] += 1
            continue
        if (not v.get("content_sha256") and rec.get("verifier_result_hash") == verdict_hash):
            verification_freshness.archive(rec, reason="CONTENT_CHANGED_REVIEW_REQUIRED", now=now)
            dirty = True
            stats["stale_legacy_replay"] += 1
            continue
        if (not v.get("content_sha256") and (rec.get("verification_state") in
                ("CONTENT_CHANGED_REVIEW_REQUIRED", "STALE_SOURCE")
                or verification_freshness.state(rec).get("state") == "CONTENT_CHANGED_REVIEW_REQUIRED")):
            # A fresh review of a modified assertion must identify what it
            # inspected; changing a legacy reason string is not such evidence.
            stats["content_hash_required"] += 1
            continue
        dirty = True
        if rec.get("verification"):
            verification_freshness.archive(rec, reason="SUPERSEDED_VERDICT", now=now)
        rec["verifier_result_hash"] = verdict_hash
        rec["verification"] = ("REFUTED" if v["refuted"] else
                               "UNCONFIRMED" if v["confidence"] == "low" else "CONFIRMED")
        rec["verifier_reason"] = v.get("reason", "")
        rec["verifier_confidence"] = v["confidence"]
        rec["verifier_source_sha256"] = supplied_sha or rec.get("source_sha256")
        rec["verifier_content_sha256"] = inspected_content
        rec["verifier_content_version"] = verification_freshness.CONTENT_VERSION
        rec["verifier_content_binding"] = ("REVIEWER_SUPPLIED" if v.get("content_sha256")
                                            else "LEGACY_INPUT_BOUND_AT_APPLICATION")
        rec.pop("verification_state", None)
        for key in ("verifier_role", "verifier_model"):
            rec.pop(key, None)
        if reviewer:
            rec["verifier_role"], rec["verifier_model"] = reviewer.get("role"), reviewer.get("model")
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
                if mid not in models:
                    continue
                model = models[mid]
                did, sha = rec["document_id"], rec["source_sha256"]
                assessment = next((row for row in model_source_assessments(model)
                                   if row["document_id"] == did and row.get("source_sha256") == sha), None)
                if assessment is None:
                    # A retained model link cannot bind a verdict to a different
                    # source assessment or revive a missing one.
                    continue
                row = {"document_id": did, "source_sha256": sha, "state": override,
                       "object_id": oid, "verdict_hash": verdict_hash, "reason": v.get("reason", ""),
                       "reviewer": reviewer, "verified_at": now}
                overrides = {(item["document_id"], item["source_sha256"]): item
                             for item in model.get("source_support_verification_overrides", [])}
                previous = overrides.get((did, sha))
                if previous and previous != row:
                    model["source_support_verification_override_history"] = _merge_list(
                        model.get("source_support_verification_override_history"), [previous])
                overrides[(did, sha)] = row
                model["source_support_verification_overrides"] = [overrides[key] for key in sorted(overrides)]
                primary = model.get("source_document_id") or (model.get("source_review") or {}).get("reviewed_document_id")
                if not primary and len(model.get("source_assessments", [])) == 1:
                    primary = model["source_assessments"][0]["document_id"]
                if primary == did and model.get("source_support_sha256") == sha:
                    model.setdefault("source_support_state_extracted", model.get("source_support_state"))
                    model["source_support_state_verified"] = override
                    model["source_support_verifier_document_id"] = did
                    model["source_support_verifier_source_sha256"] = sha
                    model["source_support_verifier_note"] = v.get("reason", "")
                aggregate, notes = aggregate_source_support(model_source_assessments(model),
                                                            {key: {"sha256": value} for key, value in source_hashes.items()})
                model["source_support_state"], model["source_support_notes"] = aggregate, notes
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
        # Grounded additions were explicitly supplied by this verdict and are
        # part of the assertion the reviewer inspected, not a later re-bank.
        rec["verifier_content_sha256"] = verification_freshness.content_sha256(rec)
    if dirty:
        save("equations", eqs.values(), meta={"generated_at": now, "n": len(eqs)})
        save("claims", cls.values(), meta={"generated_at": now, "n": len(cls)})
        save("entities", ents.values(), meta={"generated_at": now, "n": len(ents)})
        save("validity", vals.values(), meta={"generated_at": now, "n": len(vals)})
        save("models", models.values(), meta={"generated_at": now, "n": len(models),
                                              "source": "raosim/model_registry.py + extraction support states + verification"})
    stats["stores_saved"] = dirty
    return stats


@writes_stores("retire duplicate equation records")
def retire_duplicate_equations(payload: dict) -> dict:
    """Retire an exact duplicate equation record in favour of a canonical one, keeping history.

    Both records must be current, from the same document bytes, page and printed
    equation number, with forms identical apart from whitespace and case. The
    duplicate's parser-candidate links, model links and model comparison links
    move to the canonical record; the duplicate stays inspectable by id with
    ``duplicate_of``/``superseded_by``. Only reviewer F may decide this, with a reason.
    """
    from .ground import form_key
    reviewer = payload.get("reviewer") or {}
    if reviewer.get("role") != "F" or not reviewer.get("model"):
        raise ValueError("duplicate retirement requires reviewer role F with a model identity")
    now = datetime.now(timezone.utc).isoformat()
    eqs = {e["id"]: e for e in load("equations")}
    models = {m["id"]: m for m in load("models")}
    stats = {"retired": 0, "unchanged": 0, "refused": []}
    dirty = False
    for row in payload.get("duplicate_retirements", []) or []:
        dup, canon = eqs.get(row.get("duplicate_id")), eqs.get(row.get("canonical_id"))
        if dup and canon and dup.get("duplicate_of") == canon["id"] and not dup.get("active", True):
            stats["unchanged"] += 1
            continue
        reasons = []
        if not dup or not canon:
            reasons.append("unknown equation id")
        else:
            if dup["id"] == canon["id"]:
                reasons.append("duplicate and canonical are the same record")
            for r in (dup, canon):
                if not r.get("active", True) or r.get("source_state", "CURRENT") != "CURRENT":
                    reasons.append(f"{r['id']} is not current")
            if (dup.get("document_id") != canon.get("document_id")
                    or dup.get("source_sha256") != canon.get("source_sha256")):
                reasons.append("different document or source bytes")
            if dup.get("page") != canon.get("page"):
                reasons.append("different page")
            if _equation_number_key(dup.get("equation_number")) != _equation_number_key(canon.get("equation_number")):
                reasons.append("different printed equation number")
            if form_key(dup.get("form")) != form_key(canon.get("form")):
                reasons.append("forms differ beyond whitespace")
            if canon.get("verification") == "REFUTED":
                reasons.append("canonical record is independently refuted")
        if not str(row.get("reason") or "").strip():
            reasons.append("no recorded reason")
        if reasons:
            stats["refused"].append({"duplicate_id": row.get("duplicate_id"), "canonical_id": row.get("canonical_id"),
                                     "reasons": reasons})
            continue
        canon["enrichment_candidate_ids"] = _merge_list(canon.get("enrichment_candidate_ids"), dup.get("enrichment_candidate_ids"))
        canon["corrects_candidate_ids"] = _merge_list(canon.get("corrects_candidate_ids"), dup.get("corrects_candidate_ids"))
        if dup.get("verification") != "REFUTED":
            canon["model_ids"] = sorted(set(canon.get("model_ids") or []) | set(dup.get("model_ids") or []))
        canon["merged_duplicate_ids"] = _merge_list(canon.get("merged_duplicate_ids"), [dup["id"]])
        for model in models.values():
            compared = model.get("source_comparison_equation_ids") or []
            if dup["id"] in compared:
                model["source_comparison_equation_ids"] = sorted((set(compared) - {dup["id"]}) | {canon["id"]})
        dup.update(active=False, source_state="SUPERSEDED_EXTRACTION", superseded_by=canon["id"],
                   duplicate_of=canon["id"],
                   duplicate_retirement={"reason": row["reason"], "reviewer": reviewer, "at": now})
        stats["retired"] += 1
        dirty = True
    if dirty:
        save("equations", eqs.values(), meta={"generated_at": now, "n": len(eqs)})
        save("models", models.values(), meta={"generated_at": now, "n": len(models),
                                              "source": "raosim/model_registry.py + extraction support states + verification"})
    stats["stores_saved"] = dirty
    return stats
