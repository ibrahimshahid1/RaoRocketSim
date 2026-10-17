"""Stable programmatic interface for LREKit development agents.

The storage layout may change; these function names and return shapes are
the contract.  Every returned object keeps its ``id`` so the caller can drill
down (``get_equation``, ``get_source_context``) instead of loading the corpus.

States are explicit.  Empty results are reported with a ``state`` such as
``NO_RELEVANT_SOURCE_FOUND`` rather than silently returning ``[]``.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import REPO_ROOT
from .graph import Graph, equation_formulation_state, model_source_support
from .index import VectorIndex, hybrid_search
from .quantities import BY_ID, resolve_symbol
from .store import connect, doc_chunks, get_chunk, index_by, load

# --------------------------------------------------------------------------
# cached loads
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _graph() -> Graph:
    return Graph()


@lru_cache(maxsize=1)
def _vi() -> VectorIndex:
    return VectorIndex()


@lru_cache(maxsize=1)
def _docs() -> dict[str, dict]:
    return index_by("manifest")


@lru_cache(maxsize=1)
def _models() -> dict[str, dict]:
    return index_by("models")


@lru_cache(maxsize=1)
def _equations() -> dict[str, dict]:
    return index_by("equations")


@lru_cache(maxsize=1)
def _claims() -> dict[str, dict]:
    return index_by("claims")


@lru_cache(maxsize=1)
def _software() -> dict[str, dict]:
    return {s["id"]: s for s in load("software_links") if s.get("type") == "SoftwareComponent" and _current_source(s)}


@lru_cache(maxsize=1)
def _constants() -> dict[str, dict]:
    return {s["id"]: s for s in load("software_links") if s.get("type") == "EmpiricalConstant" and _current_source(s)}


@lru_cache(maxsize=1)
def _tests() -> dict[str, dict]:
    return index_by("tests")


@lru_cache(maxsize=1)
def _caps() -> dict[str, dict]:
    return index_by("capabilities")


@lru_cache(maxsize=1)
def _reqs() -> dict[str, dict]:
    return index_by("requirements")


@lru_cache(maxsize=1)
def _entities() -> dict[str, dict]:
    return index_by("entities")


@lru_cache(maxsize=1)
def _repo_docs() -> dict[str, dict]:
    return index_by("repo_documents")


def reset_caches() -> None:
    for f in (_graph, _vi, _docs, _models, _equations, _claims, _software, _constants, _tests, _caps, _reqs, _entities, _repo_docs):
        f.cache_clear()


# --------------------------------------------------------------------------
# documents & search
# --------------------------------------------------------------------------

def _doc_brief(doc_id: str) -> dict:
    d = _docs().get(doc_id)
    if not d:
        return {"id": doc_id, "state": "SOURCE_NOT_INGESTED"}
    return {"id": doc_id, "title": d.get("title"), "authors": d.get("authors"), "year": d.get("year"),
            "identifier": d.get("identifier"), "source_pdf": d["source_pdf"],
            "source_kind": d.get("source_kind", "file"),
            "source_container": d.get("source_container"), "archive_member": d.get("archive_member"),
            "generated_markdown": d.get("generated_markdown"),
            "conversion_status": d["conversion"]["status"],
            "low_confidence": bool(d["conversion"].get("low_confidence"))}


def get_document(document_id: str) -> dict:
    d = _docs().get(document_id)
    if not d:
        return {"id": document_id, "state": "SOURCE_NOT_INGESTED"}
    out = dict(d)
    g = _graph()
    out["equations"] = [n["node"]["id"] for n in g.neighbors(document_id, {"FOUND_IN"}, "in") if n["node"] and n["node"]["type"] == "Equation"]
    out["claims"] = [n["node"]["id"] for n in g.neighbors(document_id, {"FOUND_IN"}, "in") if n["node"] and n["node"]["type"] == "Claim"]
    out["supports_models"] = [n["node"]["id"] for n in g.neighbors(document_id, {"SUPPORTED_BY"}, "in") if n["node"]]
    citations = g.neighbors(document_id, {"CITES"}, "in")
    out["cited_by_software"] = [n["node"]["id"] for n in citations if n["node"] and n["node"]["type"] == "SoftwareComponent"]
    out["cited_by_models"] = [n["node"]["id"] for n in citations if n["node"] and n["node"]["type"] == "Model"]
    return out


def search_documents(query: str, limit: int = 10) -> dict:
    search = semantic_search(query, limit=limit * 4)
    hits = search["hits"]
    by_doc: dict[str, dict] = {}
    for h in hits:
        b = by_doc.setdefault(h["doc_id"], {"doc": _doc_brief(h["doc_id"]), "score": 0.0, "pages": set()})
        b["score"] += h["score"]
        b["pages"].add(h["page_start"])
    ranked = sorted(by_doc.values(), key=lambda b: -b["score"])[:limit]
    for b in ranked:
        b["pages"] = sorted(b["pages"])
        b["score"] = round(b["score"], 5)
    return {"query": query, "state": "OK" if ranked else "NO_RELEVANT_SOURCE_FOUND", "documents": ranked,
            "retrieval": search["retrieval"]}


def _expand_query(query: str) -> list[str]:
    extra = []
    ql = query.lower()
    for q in BY_ID.values():
        if q.canonical in ql or any(a.lower() in ql for a in q.aliases):
            extra.extend(s for s in q.symbols[:3])
            extra.extend(q.code_symbols[:2])
    return extra


def semantic_search(query: str, limit: int = 20) -> dict:
    vi = _vi()
    hits = hybrid_search(query, limit=limit, expand_terms=_expand_query(query), vector_index=vi)
    for h in hits:
        d = _docs().get(h["doc_id"], {})
        h["document"] = d.get("title") or d.get("filename")
        h["source_pdf"] = d.get("source_pdf")
        h["low_confidence"] = bool(d.get("conversion", {}).get("low_confidence"))
        h["text"] = h["text"][:1200]
    return {"query": query, "state": "OK" if hits else "NO_RELEVANT_SOURCE_FOUND", "hits": hits,
            "retrieval": {"mode": "HYBRID" if vi.status["state"] == "OK" else "LEXICAL_ONLY",
                          "semantic": dict(vi.status)}}


def get_source_context(chunk_id_or_ref: str, window: int = 1) -> dict:
    """Return a chunk with its neighbours and the PDF escalation path."""
    conn = connect()
    c = get_chunk(conn, chunk_id_or_ref)
    if not c:
        conn.close()
        return {"ref": chunk_id_or_ref, "state": "INSUFFICIENT_SOURCE_CONTEXT"}
    allc = doc_chunks(conn, c["doc_id"])
    conn.close()
    i = c["seq"]
    ctx = [dict(x) for x in allc[max(0, i - window): i + window + 1]]
    d = _docs().get(c["doc_id"], {})
    return {
        "chunk": c, "context": ctx,
        "document": _doc_brief(c["doc_id"]),
        "escalation": {
            "pdf": d.get("source_pdf"), "pages": [c["page_start"], c["page_end"]],
            "markdown": d.get("generated_markdown"),
            "source_container": d.get("source_container"), "archive_member": d.get("archive_member"),
            "note": "PDF is authoritative. Verify equations/values against the PDF page(s) before load-bearing use.",
        },
    }


# --------------------------------------------------------------------------
# entities / graph
# --------------------------------------------------------------------------

def find_entity(name: str, limit: int = 15) -> dict:
    g = _graph()
    hits = [n for n in g.find(name, limit=len(g.nodes)) if _active_science(n)][:limit]
    sym = resolve_symbol(name)
    return {"query": name, "state": "OK" if hits or sym["candidates"] else "UNRESOLVED_ENTITY",
            "symbol_resolution": sym, "nodes": hits}


def get_entity(entity_id: str) -> dict:
    g = _graph()
    n = g.nodes.get(entity_id)
    if not n:
        return {"id": entity_id, "state": "UNRESOLVED_ENTITY"}
    return {"node": n, "neighbors": g.neighbors(entity_id)}


def find_artifacts(query: str = "", limit: int = 15) -> dict:
    """Find active observed engineering records; historical snapshots use IDs."""
    terms = re.findall(r"[a-z0-9_]+", query.lower())
    hits = []
    for rec in load("artifacts"):
        if rec.get("type") == "ArtifactLink" or not rec.get("active", True):
            continue
        hay = " ".join(str(rec.get(k, "")) for k in
                       ("id", "type", "name", "kind", "scope", "source_path", "path")).lower()
        score = sum(term in hay for term in terms)
        if not terms or score:
            hits.append((score, rec))
    hits.sort(key=lambda hit: (-hit[0], hit[1]["id"]))
    return {"query": query, "state": "OK" if hits else "NO_RELEVANT_SOURCE_FOUND",
            "records": [rec for _, rec in hits[:max(0, limit)]],
            "authority": "Repository report assertions and observed file hashes; inspect binding and qualification states."}


def get_artifact(artifact_id: str) -> dict:
    rec = index_by("artifacts").get(artifact_id)
    if not rec:
        return {"id": artifact_id, "state": "UNRESOLVED_ENTITY"}
    return {"record": rec, "provenance": index_by("provenance").get(rec.get("provenance_id")),
            "neighbors": get_neighbors(artifact_id),
            "state": rec.get("source_state", "UNKNOWN"),
            "observation_note": "Snapshot from the last rebuild; current hashes do not establish generation-time or physical validation."}


def get_neighbors(entity_id: str, edge_types: list[str] | None = None) -> list[dict]:
    return _graph().neighbors(entity_id, set(edge_types) if edge_types else None)


def find_path(a: str, b: str) -> dict:
    g = _graph()
    ra = a if a in g.nodes else (g.find(a, limit=1) or [{}])[0].get("id")
    rb = b if b in g.nodes else (g.find(b, limit=1) or [{}])[0].get("id")
    if not ra or not rb:
        return {"state": "UNRESOLVED_ENTITY", "a": a, "b": b}
    p = g.find_path(ra, rb)
    return {"state": "OK" if p else "NO_PATH", "from": ra, "to": rb, "path": p}


# --------------------------------------------------------------------------
# equations / claims / models
# --------------------------------------------------------------------------

def _matches(text: str, rec: dict, fields: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(t in str(rec.get(f, "")).lower() for f in fields)


def _current_source(rec: dict) -> bool:
    return rec.get("active", True) and rec.get("source_state", "CURRENT") == "CURRENT"


def _active_science(rec: dict) -> bool:
    """Current and not independently refuted; refuted records stay inspectable by ID only."""
    return _current_source(rec) and rec.get("verification") != "REFUTED"


def content_verification_summary(rec: dict) -> dict:
    """Current assertion fingerprint, separate from historical source verdicts."""
    from . import verification_freshness
    return verification_freshness.state(rec, _docs().get(rec.get("document_id"))) | {
        "content_sha256": verification_freshness.content_sha256(rec),
        "content_version": verification_freshness.CONTENT_VERSION,
    }


def _validity_model_ids(rec: dict) -> set[str]:
    ids = set(rec.get("model_ids", []))
    if rec.get("model_id"):
        ids.add(rec["model_id"])
    return ids


def find_equations(concept: str, limit: int = 20) -> dict:
    fields = ("label", "form", "description", "concepts_text", "variables_text")
    eqs = [e for e in _equations().values() if _current_source(e) and _matches(concept, e, fields)
           and e.get("verification") != "REFUTED"]
    terms = [w for w in re.findall(r"[a-z][a-z_*]{2,}", concept.lower())]

    def rank(e):
        label = str(e.get("label", "")).lower()
        hits = sum(1 for w in terms if w in label) * 3 + sum(1 for w in terms if w in str(e.get("form", "")).lower())
        penalty = 2 if ("sample calculation" in label or "numeric" in label or "worked example" in label) else 0
        return -(hits - penalty + (1 if e.get("model_ids") else 0))

    eqs.sort(key=rank)
    if not eqs:
        # fall back to retrieval: equation-rich chunks
        hits = [h for h in semantic_search(concept, limit=40)["hits"] if h["kind"] == "equation_rich"][:limit]
        return {"query": concept, "state": "NO_EXTRACTED_EQUATION_MATCH" if not hits else "CHUNK_FALLBACK",
                "equations": [], "equation_rich_chunks": hits,
                "note": "No extracted Equation record matched. Chunks flagged equation_rich are returned for manual inspection; this is not a claim that no such equation exists in the corpus."}
    selected = [{**e, "transcription_state": equation_formulation_state(e),
                 "content_verification": content_verification_summary(e)} for e in eqs[:limit]]
    incomplete = [e["id"] for e in selected if e["transcription_state"] == "INCOMPLETE_TRANSCRIPTION"]
    return {"query": concept, "state": "INCOMPLETE_TRANSCRIPTION" if len(incomplete) == len(selected) else "OK",
            "equations": selected, "incomplete_formula_ids": incomplete}


def get_equation(equation_id: str) -> dict:
    e = _equations().get(equation_id)
    if not e:
        return {"id": equation_id, "state": "UNRESOLVED_ENTITY"}
    out = dict(e)
    out["transcription_state"] = equation_formulation_state(e)
    out["content_verification"] = content_verification_summary(e)
    out["document"] = _doc_brief(e["document_id"])
    out["neighbors"] = _graph().neighbors(equation_id)
    return out


def find_claims(subject: str, predicate: str | None = None, obj: str | None = None, limit: int = 30) -> dict:
    res = []
    for c in _claims().values():
        if not _current_source(c) or c.get("verification") == "REFUTED":
            continue
        if subject and not _matches(subject, c, ("subject", "object", "statement", "quote")):
            continue
        if predicate and predicate.lower() not in str(c.get("predicate", "")).lower():
            continue
        if obj and obj.lower() not in str(c.get("object", "")).lower():
            continue
        res.append(c)
    return {"query": {"subject": subject, "predicate": predicate, "object": obj},
            "state": "OK" if res else "NO_RELEVANT_SOURCE_FOUND", "claims": [
                {**c, "content_verification": content_verification_summary(c)} for c in res[:limit]]}


def get_claim_sources(claim_id: str) -> dict:
    c = _claims().get(claim_id)
    if not c:
        return {"id": claim_id, "state": "UNRESOLVED_ENTITY"}
    return {"claim": {**c, "content_verification": content_verification_summary(c)},
            "document": _doc_brief(c["document_id"]),
            "source_context": get_source_context(c["chunk_id"]) if c.get("chunk_id") else None}


def model_review_summary(model: dict) -> dict:
    """Expose every retained source comparison and its live source/code freshness."""
    from .harvest_extraction import model_review_state
    current = model_review_state(model, documents=_docs())
    if not current.get("source_reviews"):
        return current
    reviews = current["source_reviews"]
    primary = model.get("source_review") or reviews[0]
    return {"state": current["state"], "reason": current.get("reason"), "changed": current.get("changed", []),
            "fidelity": current.get("fidelity"), "fidelities": current.get("fidelities", []),
            "physics_change_required": current.get("physics_change_required"),
            "reviewed_document_id": primary.get("reviewed_document_id"),
            "reviewed_source_sha256": primary.get("reviewed_source_sha256"),
            "equation_ids": sorted({eid for review in reviews for eid in review.get("equation_ids", [])}),
            "implementation": [{k: f.get(k) for k in ("file", "symbol", "line", "state")}
                               for f in primary.get("implementation_fingerprints", [])],
            "source_reviews": reviews,
            "reviewer": primary.get("reviewer"), "physical_validation": "NOT_ESTABLISHED",
            "scope": "Source-versus-code comparison only; not execution reach, sensitivity or physical validation."}


def find_models(concept: str, limit: int = 20) -> dict:
    ms = [m for m in _models().values() if _matches(concept, m, ("registry_id", "quantity", "relation", "subsystem", "cited_source_text", "equation_ref"))]
    return {"query": concept, "state": "OK" if ms else "NO_RELEVANT_SOURCE_FOUND",
            "models": [{**m, "source_review_state": model_review_summary(m)} for m in ms[:limit]]}


def get_model(model_id: str) -> dict:
    m = _models().get(model_id)
    if not m:
        return {"id": model_id, "state": "UNRESOLVED_ENTITY"}
    out = dict(m)
    out["source_review_state"] = model_review_summary(m)
    g = _graph()
    out["neighbors"] = g.neighbors(model_id)
    out["source_document"] = _doc_brief(m["source_document_id"]) if m.get("source_document_id") else {"state": "NO_LOCAL_SOURCE_ROUTE"}
    return out


def find_validity_conditions(model: str) -> dict:
    ms = find_models(model)["models"] if model not in _models() else [_models()[model]]
    vals = load("validity")
    out = []
    for m in ms:
        for v in vals:
            if _active_science(v) and m["id"] in _validity_model_ids(v) and v not in out:
                out.append(v)
        for c in _claims().values():
            if (_current_source(c) and c.get("verification") != "REFUTED" and m["id"] in c.get("model_ids", [])
                    and c.get("claim_type") in ("validity_statement", "assumption", "constraint")):
                out.append({"id": c["id"], "type": "Claim", "claim_type": c["claim_type"], "statement": c.get("statement"),
                            "document_id": c["document_id"], "page": c.get("page"), "quote": c.get("quote")})
    return {"query": model, "state": "OK" if out else "NO_VALIDITY_RANGE_LOCATED",
            "validity": out,
            "note": "" if out else "No validity range was located in the ingested corpus or registry; this is not evidence that none exists."}


def find_requirements(query: str, limit: int = 30) -> dict:
    rs = [r for r in _reqs().values() if _matches(query, r, ("name", "quantity", "kind", "detail"))]
    return {"query": query, "state": "OK" if rs else "NO_RELEVANT_SOURCE_FOUND", "requirements": rs[:limit]}


def compare_sources(query: str, limit: int = 12) -> dict:
    """Surface competing/contradicting formulations plus per-document hits."""
    g = _graph()
    hits = semantic_search(query, limit=limit * 3)["hits"]
    per_doc: dict[str, list] = {}
    for h in hits:
        per_doc.setdefault(h["doc_id"], []).append({"chunk_id": h["chunk_id"], "page": h["page_start"], "section": h["section"], "text": h["text"][:600]})
    conflicts = []
    for e in g.edges:
        if e["type"] in ("CONTRADICTS", "COMPETING_FORMULATION_WITH"):
            a, b = g.nodes.get(e["from"]), g.nodes.get(e["to"])
            if a and b and (_matches(query, a, ("label", "form")) or _matches(query, b, ("label", "form"))):
                conflicts.append({"a": a, "b": b, "edge": e["type"], "provenance": e.get("provenance")})
    return {"query": query, "state": "CONFLICTING_FORMULATIONS" if conflicts else ("OK" if per_doc else "NO_RELEVANT_SOURCE_FOUND"),
            "documents": [{"document": _doc_brief(d), "passages": p} for d, p in list(per_doc.items())[:limit]],
            "recorded_conflicts": conflicts}


# --------------------------------------------------------------------------
# software <-> science
# --------------------------------------------------------------------------

def find_implementation(scientific_object_id: str) -> dict:
    g = _graph()
    impls = [n for n in g.neighbors(scientific_object_id, {"IMPLEMENTS"}, "in")
             if n["node"] and n["node"]["type"] == "SoftwareComponent"]
    # Model -> Equation IMPLEMENTS; Claim -> Model SUPPORTS/CONSTRAINS.
    models = g.neighbors(scientific_object_id, {"IMPLEMENTS"}, "in") + \
             g.neighbors(scientific_object_id, {"SUPPORTS", "CONSTRAINS"}, "out")
    via_model = []
    for m in models:
        if m["node"] and m["node"]["type"] == "Model":
            via_model += [n for n in g.neighbors(m["node"]["id"], {"IMPLEMENTS"}, "in") if n["node"] and n["node"]["type"] == "SoftwareComponent"]
    all_hits = list({n["node"]["id"]: n for n in via_model + impls}.values())
    # A structured review names the exact symbols it compared with this
    # equation; those are reviewed implementations, with their current state.
    reviewed = []
    for m in models:
        if m["node"] and m["node"]["type"] == "Model" and m.get("edge") == "IMPLEMENTS":
            record = _models().get(m["node"]["id"], {})
            summary = model_review_summary(record)
            for review in summary.get("source_reviews", []):
                # A model may implement equations from several documents with
                # distinct scopes; return only the source review for this object.
                equation = _equations().get(scientific_object_id, {})
                if review.get("reviewed_document_id") != equation.get("document_id"):
                    continue
                reviewed += [{"model_id": record["id"], "fidelity": review.get("fidelity"),
                              "review_state": review["state"], "reviewed_document_id": review["reviewed_document_id"],
                              **{k: f.get(k) for k in ("file", "symbol", "line", "state")}}
                             for f in review.get("implementation_fingerprints", [])]
    return {"id": scientific_object_id, "state": "OK" if (all_hits or reviewed) else "NO_IMPLEMENTATION_LINK",
            "implementations": [{"id": n["node"]["id"], "type": n["node"]["type"], "file": n["node"].get("file"),
                                 "line": n["node"].get("line"), "label": n["node"].get("label"), "provenance": n.get("provenance")}
                                for n in all_hits],
            "reviewed_implementations": reviewed}


def find_scientific_basis(software_ref: str) -> dict:
    """software_ref: a SoftwareComponent id, 'file.py', 'file.py:Symbol' or a model registry id."""
    g = _graph()
    comps = []
    if software_ref in g.nodes:
        comps = [g.nodes[software_ref]]
    else:
        ref = software_ref.replace("\\", "/")
        for n in _software().values():
            if n["file"] == ref or n["qualname"].endswith(ref) or ref in n["qualname"] or n["file"].endswith(ref.split(":")[0]) and (":" not in ref or ref.split(":")[1] in n["qualname"]):
                comps.append(n)
        if not comps:
            for m in _models().values():
                if m["registry_id"] == software_ref:
                    comps.append(g.nodes[m["id"]])
    if not comps:
        return {"ref": software_ref, "state": "UNRESOLVED_ENTITY"}
    out = []
    for c in comps:
        cid = c["id"]
        models = ([c] if c.get("type") == "Model" else
                  [n["node"] for n in g.neighbors(cid, {"IMPLEMENTS"}, "out") if n["node"] and n["node"]["type"] == "Model"])
        equations = [n["node"] for n in g.neighbors(cid, {"IMPLEMENTS"}, "out") if n["node"] and n["node"]["type"] == "Equation"]
        docs = [n["node"] for n in g.neighbors(cid, {"CITES", "SUPPORTED_BY"}, "out") if n["node"] and n["node"]["type"] == "Document"]
        support = []
        supported_docs = set()
        for m in models:
            # Registry citation paths stay visible even when their support is
            # absent, partial, unreviewed, or stale. Recheck against the stores
            # so a pre-rebuild graph cannot turn a stale review into support.
            record = _models().get(m["id"], m)
            did = record.get("source_document_id")
            effective, note = model_source_support(record, _docs().get(did))
            support.append({"model_id": m["id"], "document_id": did,
                            "declared_state": record.get("source_support_state", "NOT_CHECKED"),
                            "state": effective, "notes": record.get("source_support_notes", ""),
                            "evidence_note": note, "source_review": model_review_summary(record)})
            if did:
                if did in _docs():
                    docs.append(_docs()[did])
                if effective in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
                    supported_docs.add(did)
            docs += [n["node"] for n in g.neighbors(m["id"], {"CITES", "SUPPORTED_BY"}, "out") if n["node"] and n["node"]["type"] == "Document"]
            equations += [n["node"] for n in g.neighbors(m["id"], {"IMPLEMENTS"}, "out") if n["node"] and n["node"]["type"] == "Equation"]
        constants = [n["node"] for n in g.neighbors(cid, {"USES"}, "out") if n["node"] and n["node"]["type"] == "EmpiricalConstant"]
        refusals = [n["node"] for n in g.neighbors(cid, {"REFUSES_WHEN"}, "out") if n["node"]]
        tests = [n["node"] for n in g.neighbors(cid, {"TESTED_WITH"}, "out") if n["node"]]
        states = [s["state"] for s in support]
        if states and all(s == "SUPPORTED" for s in states):
            basis_state = "OK"
        elif any(s in {"SUPPORTED", "PARTIALLY_SUPPORTED"} for s in states):
            basis_state = "PARTIALLY_SUPPORTED"
        elif states:
            basis_state = next((s for s in ("STALE_SOURCE", "NOT_FOUND_IN_SOURCE", "SOURCE_NOT_INGESTED", "SOURCE_UNREADABLE") if s in states), "NOT_CHECKED")
        else:
            basis_state = "NOT_CHECKED" if (equations or docs) else "SCIENTIFIC_BASIS_UNKNOWN"
        # All source_documents are routes to inspect, not a claim that each
        # source supports the implemented formulation.
        doc_ids = {d["id"] for d in docs} | {s["document_id"] for s in support if s["document_id"]}
        out.append({"component": c, "state": basis_state, "models": models, "equations": equations,
                    "source_documents": [_doc_brief(did) for did in sorted(doc_ids)],
                    "supported_source_documents": [_doc_brief(did) for did in sorted(supported_docs)],
                    "model_support": support,
                    "support_scope": "Citation support only; does not verify implementation correctness or physical validity.",
                    "constants": constants, "refusals": refusals, "tests": tests,
                    "unknown_basis_constants": [k for k in constants if k.get("basis_class") == "UNKNOWN_BASIS"]})
    component_states = {c["state"] for c in out}
    aggregate = ("OK" if component_states == {"OK"} else
                 "PARTIALLY_SUPPORTED" if component_states & {"OK", "PARTIALLY_SUPPORTED"} else
                 next(iter(component_states)) if len(component_states) == 1 else "INSUFFICIENT_SOURCE_CONTEXT")
    # Keep state=OK as the historical operation-success contract; basis_state
    # and each component's state describe the scientific support assessment.
    return {"ref": software_ref, "state": "OK", "basis_state": aggregate, "components": out}


# --------------------------------------------------------------------------
# retrieve_for_task - the primary interface
# --------------------------------------------------------------------------

_SUBSYSTEM_HINTS = {
    "cooling": ["cool", "regen", "coolant", "jacket", "channel", "wall temperature", "heat flux", "bartz", "chf"],
    "heat_transfer": ["heat transfer", "nusselt", "convect", "radiat", "bartz"],
    "nozzle_contour": ["nozzle", "rao", "contour", "bell", "top ", "parabolic", "expansion ratio", "exit angle", "theta_b", "transversality", "stationarity"],
    "moc": ["characteristic", "moc", "kernel", "transonic", "sauer", "characteristics"],
    "injector": ["injector", "pintle", "orifice", "discharge coefficient", "momentum ratio", "tmr", "skip"],
    "spray": ["spray", "atomiz", "droplet", "smd", "breakup", "evaporat"],
    "pump": ["pump", "impeller", "inducer", "npsh", "cavitation", "specific speed", "head coefficient", "volute", "diffuser",
             "bearing", "seal", "rotordynamic", "shaft"],
    "electric_drive": ["motor", "battery", "electric"],
    "structures": ["stress", "buckl", "fatigue", "liner", "yield", "factor of safety", "hoop"],
    "materials": ["material", "alloy", "copper", "inconel", "conductivity", "properties"],
    "performance": ["isp", "specific impulse", "thrust coefficient", "c-star", "cstar", "c*", "thrust", "altitude", "separation"],
    "thermochemistry": ["cea", "equilibrium", "frozen", "combustion temperature", "gamma", "mixture ratio", "o/f"],
    "gas_dynamics": ["mach", "isentropic", "prandtl", "quasi-1d", "area ratio", "shock"],
    "mdo": ["mdo", "optimiz", "constraint", "objective", "nlp", "jax", "differentiab", "gradient", "adjoint", "implicit"],
    "cad": ["cad", "step", "stl", "b-rep", "solid", "cadquery", "geometry export"],
    "manufacturing": ["machin", "manufactur", "tolerance", "gd&t", "y14", "inspection", "datum"],
    "chamber": ["chamber", "l-star", "l*", "contraction", "combustion chamber"],
    "throat": ["throat"],
    "trajectory": ["trajectory", "mission", "delta-v", "ascent"],
    "mass": ["mass", "dry mass", "ledger"],
    "verification": ["benchmark", "parity", "test", "validate", "validation", "oracle", "fidelity", "gap", "unavailable",
                     "unsupported", "blocked", "missing", "qualified", "evidence", "machineable"],
    "configuration": ["configuration", "fingerprint", "identity", "artifact", "bound"],
    "feed_system": ["feed", "line", "valve", "tank pressure", "bearing", "seal", "rotordynamic", "shaft", "thrust balance"],
}
_TOPIC_TO_SUBSYSTEM = {"pump": "pump", "injector": "injector", "materials": "materials", "Regenerative cooling": "cooling",
                       "Heat transfer": "heat_transfer", "Nozzle": "nozzle_contour", "Structures": "structures",
                       "Combustion": "injector", "Turbomachinery": "pump", "CFD": "verification"}


REGISTRY_SUBSYSTEM_ALIASES = {
    "cooling": {"thermal", "cooling"}, "heat_transfer": {"thermal", "heat_transfer"},
    "structures": {"structure", "structures", "materials"}, "materials": {"materials", "structure"},
    "gas_dynamics": {"flow", "gas_dynamics", "performance"}, "performance": {"performance", "altitude", "separation", "flow"},
    "configuration": {"interface", "configuration"}, "nozzle_contour": {"nozzle", "nozzle_contour", "throat"},
    "throat": {"throat", "nozzle"}, "moc": {"nozzle", "moc"}, "mdo": {"mdo"}, "injector": {"injector"},
    "spray": {"spray", "injector"}, "pump": {"pump", "feed_system"}, "feed_system": {"pump", "feed_system"},
    "chamber": {"chamber"}, "cad": {"cad"}, "manufacturing": {"cad", "manufacturing"},
    "thermochemistry": {"thermochemistry", "performance"}, "trajectory": {"trajectory"}, "mass": {"mass"},
    "verification": {"verification"}, "requirements": {"requirements"}, "electric_drive": {"pump"},
}


def _sub_match(subsystems: list[str], value: str) -> bool:
    v = str(value or "").lower()
    if not v:
        return False
    for s in subsystems:
        if v == s or v in REGISTRY_SUBSYSTEM_ALIASES.get(s, {s}):
            return True
    return False


def _infer_subsystems(task: str, hits: list[dict] | None = None) -> list[str]:
    t = task.lower().replace("-", " ")
    scored = []
    for sub, hints in _SUBSYSTEM_HINTS.items():
        n = sum(1 for h in hints if h in t)
        if n:
            scored.append((n, sub))
    out = [s for _, s in sorted(scored, reverse=True)][:4]
    if not out and hits:
        # fall back to the topics of the documents retrieval found
        from collections import Counter
        c = Counter()
        for h in hits[:8]:
            for topic in _docs().get(h["doc_id"], {}).get("topics", []):
                if topic in _TOPIC_TO_SUBSYSTEM:
                    c[_TOPIC_TO_SUBSYSTEM[topic]] += 1
        out = [s for s, _ in c.most_common(2)]
    return out


def _infer_quantities(task: str) -> list[dict]:
    t = task.lower().replace("-", " ")
    out = []
    for q in BY_ID.values():
        if q.canonical in t or any(a.lower() in t for a in q.aliases) or any(re.search(rf"\b{re.escape(s.lower())}\b", t) for s in q.code_symbols if len(s) > 2):
            out.append({"qid": q.qid, "canonical": q.canonical, "si_unit": q.si_unit, "symbols": list(q.symbols)})
    return out


def retrieve_for_task(task: str, max_chunks: int = 12, max_items: int = 15) -> dict:
    """Task-aware context package for an LREKit development task."""
    g = _graph()
    search = semantic_search(task, limit=max_chunks)
    hits = search["hits"]
    subsystems = _infer_subsystems(task, hits)
    quantities = _infer_quantities(task)
    doc_ids = list(dict.fromkeys(h["doc_id"] for h in hits))

    task_terms = [w for w in re.findall(r"[a-zA-Z][a-zA-Z_*]{3,}", task.lower().replace("-", " "))
                  if w not in {"repair", "implement", "model", "models", "used", "using", "with", "that", "this", "from", "into", "which", "what", "does", "make", "used"}]

    def term_hits(rec, fields):
        hay = " ".join(str(rec.get(f, "")) for f in fields).lower()
        return sum(1 for w in task_terms if w in hay)

    mfields = ("registry_id", "quantity", "relation", "equation_ref", "validity_text", "cited_source_text")
    scored_models = sorted(_models().values(),
                           key=lambda m: -(term_hits(m, mfields) * 3 + (2 if _sub_match(subsystems, m.get("subsystem")) else 0)))
    models = [m for m in scored_models if term_hits(m, mfields) or _sub_match(subsystems, m.get("subsystem"))][:max_items]

    equations = sorted((e for e in _equations().values() if _current_source(e) and e.get("verification") != "REFUTED"),
                       key=lambda e: -term_hits(e, ("label", "form", "description", "concepts_text", "variables_text")))
    equations = [e for e in equations if term_hits(e, ("label", "form", "description", "concepts_text", "variables_text")) or e["document_id"] in doc_ids][:max_items]
    equations = [{**e, "transcription_state": equation_formulation_state(e)} for e in equations]
    claims = sorted((c for c in _claims().values() if _current_source(c) and c.get("verification") != "REFUTED"),
                    key=lambda c: -term_hits(c, ("subject", "predicate", "object", "statement")))
    claims = [c for c in claims if term_hits(c, ("subject", "predicate", "object", "statement")) >= 1 or c["document_id"] in doc_ids][:max_items * 2]
    assumptions = [c for c in claims if c.get("claim_type") in ("assumption",)] + \
                  [a for a in _entities().values() if _active_science(a) and a.get("type") == "Assumption" and (term_hits(a, ("label", "prose")) or a.get("document_id") in doc_ids)][:max_items]
    stored_validity = load("validity")
    selected_model_ids = {m["id"] for m in models}
    validity = [c for c in claims if c.get("claim_type") == "validity_statement"] + \
               [v for v in stored_validity if _active_science(v) and _validity_model_ids(v) & selected_model_ids]
    sfields = ("qualname", "role", "relations_text", "file")
    sw = [s for s in _software().values() if term_hits(s, sfields) >= 1]
    sw = sorted(sw, key=lambda s: -(term_hits(s, sfields) * 2 + (1 if _sub_match(subsystems, s.get("subsystem")) else 0)))[:max_items]
    constants = [k for k in _constants().values() if any(k.get("component_id") == s["id"] for s in sw)]
    tests = sorted([t for t in _tests().values() if term_hits(t, ("path", "detail")) >= 1 or
                    (_sub_match(subsystems, t.get("subsystem")) and t.get("verification_kind") == "published_benchmark_reproduction")],
                   key=lambda t: -(term_hits(t, ("path", "detail")) * 2 + (1 if t.get("verification_kind") == "published_benchmark_reproduction" else 0)))[:max_items]
    benchmarks = [t for t in tests if t.get("verification_kind") == "published_benchmark_reproduction"]
    caps = sorted([c for c in _caps().values() if term_hits(c, ("name", "detail", "file")) >= 1 or
                   (subsystems and any(_sub_match(subsystems, x) for x in c.get("subsystems", [])))],
                  key=lambda c: -term_hits(c, ("name", "detail", "file")))[:max_items]
    reqs = [r for r in _reqs().values() if _sub_match(subsystems, r.get("subsystem")) or term_hits(r, ("name", "quantity", "detail", "category"))][:max_items]
    repo_docs = [r for r in _repo_docs().values() if term_hits(r, ("path", "title", "summary"))][:8]
    artifacts = find_artifacts(" ".join(task_terms), limit=max_items)["records"] if task_terms else []

    conflicts = []
    eq_ids = {e["id"] for e in equations}
    cl_ids = {c["id"] for c in claims}
    for e in g.edges:
        if e["type"] in ("CONTRADICTS", "COMPETING_FORMULATION_WITH") and (e["from"] in eq_ids | cl_ids or e["to"] in eq_ids | cl_ids):
            conflicts.append(e)

    warnings = []
    incomplete_formulas = [e["id"] for e in equations if e["transcription_state"] == "INCOMPLETE_TRANSCRIPTION"]
    if incomplete_formulas:
        warnings.append("INCOMPLETE_TRANSCRIPTION: formula bodies remain missing/garbled; quote grounding does not verify the mathematics. "
                        "Excluded from implementation support edges: " + ", ".join(incomplete_formulas))
    if search["retrieval"]["semantic"]["state"] != "OK":
        warnings.append("Semantic index UNAVAILABLE; source retrieval used lexical search only. "
                        + search["retrieval"]["semantic"]["recovery"])
    stale_counts = {}
    for name, records, fields in (
            ("equations", _equations().values(), ("label", "form", "description", "concepts_text", "variables_text")),
            ("claims", _claims().values(), ("subject", "predicate", "object", "statement")),
            ("entities", _entities().values(), ("label", "prose")),
            ("validity", stored_validity, ("prose", "statement"))):
        count = sum(1 for rec in records if not _current_source(rec) and (
            term_hits(rec, fields) or rec.get("document_id") in doc_ids
            or _validity_model_ids(rec) & selected_model_ids))
        if count:
            stale_counts[name] = count
    if stale_counts:
        warnings.append("Excluded noncurrent source records (including superseded history and stale evidence): "
                        + ", ".join(f"{name}={count}" for name, count in stale_counts.items()))
    refuted_counts = {}
    for name, records, fields in (
            ("equations", _equations().values(), ("label", "form", "description")),
            ("claims", _claims().values(), ("subject", "predicate", "object", "statement")),
            ("entities", _entities().values(), ("label", "prose")),
            ("validity", stored_validity, ("prose", "statement"))):
        count = sum(1 for rec in records if _current_source(rec) and rec.get("verification") == "REFUTED" and (
            term_hits(rec, fields) or rec.get("document_id") in doc_ids
            or _validity_model_ids(rec) & selected_model_ids))
        if count:
            refuted_counts[name] = count
    if refuted_counts:
        warnings.append("Excluded independently refuted source records (inspectable by ID): "
                        + ", ".join(f"{name}={count}" for name, count in refuted_counts.items()))
    review_summaries = {m["id"]: model_review_summary(m) for m in models}
    for m in models:
        summary = review_summaries[m["id"]]
        if summary["state"] in ("IMPLEMENTATION_CHANGED_REVIEW_REQUIRED", "STALE_SOURCE", "SOURCE_NOT_INGESTED"):
            warnings.append(f"{m['registry_id']}: structured source review {summary['state']} - {summary.get('reason')}")
        if summary["state"] != "NOT_CHECKED" and summary.get("fidelity") not in ("EXACT", "EQUIVALENT"):
            warnings.append(f"{m['registry_id']}: structured source review fidelity={summary.get('fidelity')}"
                            + ("; physics change required (owner decision)" if summary.get("physics_change_required") else ""))
        support = m.get("source_support_state", "NOT_CHECKED")
        if support != "SUPPORTED":
            warning = f"{m['registry_id']}: source_support_state={support}"
            if m.get("source_support_notes"):
                warning += "; " + str(m["source_support_notes"])
            warnings.append(warning)
        if m.get("basis_class") in ("REPOSITORY_HEURISTIC", "UNKNOWN_BASIS"):
            warnings.append(f"{m['registry_id']}: {m['basis_class']} - not a literature-validated criterion")
        if "not_" in m.get("validation_level", "") or "screening" in m.get("validation_level", ""):
            warnings.append(f"{m['registry_id']}: validation_level={m['validation_level']}")
    for k in constants:
        if k.get("basis_class") == "UNKNOWN_BASIS":
            warnings.append(f"constant {k.get('name')} at {k.get('file')}:{k.get('line')} has UNKNOWN_BASIS")
    for h in hits:
        if h.get("low_confidence"):
            warnings.append(f"chunk {h['chunk_id']} comes from a LOW_EXTRACTION_CONFIDENCE conversion; verify against PDF")
    unavailable = [c for c in caps if str(c.get("state", "")).upper() in ("UNAVAILABLE", "NOT_IMPLEMENTED", "BLOCKED_ON_LITERATURE", "BLOCKED_ON_STANDARD", "PLANNED_NOT_IMPLEMENTED")]

    state = "OK"
    if not hits and not models and not sw and not artifacts:
        state = "NO_RELEVANT_SOURCE_FOUND"
    elif not equations and not claims and not artifacts:
        state = "INSUFFICIENT_SOURCE_CONTEXT"

    def slim(rec, keys):
        return {k: rec.get(k) for k in keys if rec.get(k) not in (None, "", [], {})}

    return {
        "task": task,
        "state": state,
        "retrieval": search["retrieval"],
        "subsystems": subsystems,
        "quantities": quantities,
        "source_documents": [_doc_brief(d) for d in doc_ids],
        "source_context": [slim(h, ("chunk_id", "doc_id", "page_start", "page_end", "section", "kind", "score", "text", "low_confidence")) for h in hits],
        "equations": [slim(e, ("id", "label", "form", "document_id", "page", "variables", "assumptions", "validity", "model_ids", "confidence", "grounded", "verification", "transcription_state")) | {"content_verification": content_verification_summary(e)} for e in equations],
        "claims": [slim(c, ("id", "claim_type", "subject", "predicate", "object", "statement", "document_id", "page", "model_ids", "confidence", "verification")) | {"content_verification": content_verification_summary(c)} for c in claims],
        "assumptions": [slim(a, ("id", "label", "prose", "statement", "document_id", "page")) | {"content_verification": content_verification_summary(a)} for a in assumptions],
        "validity_envelopes": [slim(v, ("id", "model_id", "model_ids", "prose", "statement", "machine_readable", "machine_readable_state", "document_id", "page")) | {"content_verification": content_verification_summary(v)} for v in validity],
        "models": [slim(m, ("id", "registry_id", "subsystem", "quantity", "relation", "basis_class", "basis_class_basis", "promotion_status", "promotion_status_basis", "execution_reach", "software_verification_state", "physical_validation_state", "validation_level", "source_document_id", "equation_ref", "validity_text", "source_support_state", "source_support_notes")) | {"source_review": review_summaries[m["id"]]} for m in models],
        "existing_implementations": [slim(s, ("id", "qualname", "file", "line", "role", "model_ids", "variant_of_id", "verified")) for s in sw],
        "empirical_constants": [slim(k, ("id", "name", "value", "units", "basis_class", "stated_basis", "file", "line")) for k in constants],
        "requirements_and_gates": [slim(r, ("id", "kind", "name", "quantity", "enforced_by", "file", "line")) for r in reqs],
        "tests": [slim(t, ("id", "path", "verification_kind", "subsystem", "source_document_id", "expected")) for t in tests],
        "benchmarks": [slim(t, ("id", "path", "source_document_id", "expected")) for t in benchmarks],
        "unavailable_capabilities": [slim(c, ("id", "name", "state", "enforced_by", "unblock_requires")) for c in unavailable],
        "repository_documents": [slim(r, ("id", "path", "kind", "date", "status_head")) for r in repo_docs],
        "engineering_artifacts": [slim(r, ("id", "type", "name", "kind", "scope", "state", "source_state", "source_path", "source_sha256", "json_pointer", "path", "observed_sha256", "binding_state", "producer_binding_state", "generation_time_integrity_state", "execution_state", "qualification_state", "provenance_id")) for r in artifacts],
        "conflicting_sources": conflicts,
        "implementation_warnings": warnings,
        "protocol": [
            "Inspect equations and their assumptions/validity BEFORE coding.",
            "Escalate to the PDF page for any load-bearing value (kg sources <id>).",
            "Existing code is not scientific authority; registry basis_class tells you what it is.",
            "UNKNOWN / UNAVAILABLE are valid results - do not replace them with estimates.",
        ],
    }
