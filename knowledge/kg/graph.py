"""Graph assembly, traversal and export.

The graph is *assembled* from the JSON stores (it is a view, never the
primary record) so that a node always traces back to the store that owns it.
Every edge carries a ``provenance`` field; edges without provenance are
reported by validation, not silently kept.

Exports: ``knowledge/graph/graph.json`` and ``knowledge/graph/graph.graphml``.
"""

from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from .schema import ONTOLOGY_VERSION
from .store import GRAPH_DIR, load


def _node(nid: str, ntype: str, label: str, **attrs) -> dict:
    d = {"id": nid, "type": ntype, "label": label}
    d.update({k: v for k, v in attrs.items() if v not in (None, "", [], {})})
    return d


def model_source_support(model: dict, document: dict | None) -> tuple[str, str]:
    """Return current reviewed support, separately from a registry citation.

    A legacy assessment without a source fingerprint remains inspectable, but
    cannot establish that the assessment applies to the current source bytes.
    This checks citation support only, never physical validation of the model.
    """
    declared = model.get("source_support_state") or "NOT_CHECKED"
    if model.get("source_state") == "STALE_SOURCE":
        return "STALE_SOURCE", "Model source-support assessment is stale."
    if not model.get("active", True) or model.get("source_state", "CURRENT") != "CURRENT":
        return "NOT_CHECKED", "Model source-support assessment is inactive or noncurrent; its declared result is retained as history."
    reviewed_sha = model.get("source_support_sha256")
    if reviewed_sha and document and reviewed_sha != document.get("sha256"):
        return "STALE_SOURCE", "Source bytes differ from the reviewed source fingerprint."
    if declared not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
        return declared, model.get("source_support_notes", "")
    if document is None:
        return "SOURCE_NOT_INGESTED", "Cited source is absent from the current document manifest."
    if not reviewed_sha or not document.get("sha256"):
        return "NOT_CHECKED", "Legacy support assessment has no current source fingerprint."
    return declared, model.get("source_support_notes", "")


def equation_formulation_state(record: dict) -> str:
    """Flag explicitly missing/garbled bodies without judging intact formulas.

    Inspect the current form, not historical extraction notes: an equation
    recovered from a PDF may legitimately describe earlier Markdown damage.
    """
    if re.search(r"not\s+rendered|ocr[-\s]+garbled", record.get("form") or "", re.I):
        return "INCOMPLETE_TRANSCRIPTION"
    return "NOT_FLAGGED_INCOMPLETE"


def assemble() -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    seen_edges: set[tuple] = set()
    invalid_science = {r["id"] for name in ("equations", "claims", "entities", "validity")
                       for r in load(name) if not r.get("active", True) or r.get("source_state", "CURRENT") != "CURRENT" or r.get("verification") == "REFUTED"}
    incomplete_equations = {e["id"] for e in load("equations")
                            if equation_formulation_state(e) == "INCOMPLETE_TRANSCRIPTION"}
    inactive_artifacts = {r["id"] for r in load("artifacts") if not r.get("active", True)}
    inactive_software = {r["id"] for r in load("software_links")
                         if not r.get("active", True) or r.get("source_state", "CURRENT") != "CURRENT"}

    def add_edge(src, dst, etype, provenance="", **attrs):
        if not src or not dst or (etype != "CORRECTS" and (src in invalid_science | inactive_artifacts | inactive_software or dst in invalid_science | inactive_artifacts | inactive_software)):
            return
        if etype in {"IMPLEMENTS", "IMPLEMENTED_IN"} and (src in incomplete_equations or dst in incomplete_equations):
            return
        key = (src, dst, etype)
        if key in seen_edges:
            return
        seen_edges.add(key)
        e = {"from": src, "to": dst, "type": etype, "provenance": provenance}
        e.update({k: v for k, v in attrs.items() if v not in (None, "", [], {})})
        edges.append(e)

    # ---- documents ----
    documents = {d["id"]: d for d in load("manifest")}
    for d in documents.values():
        nodes[d["id"]] = _node(d["id"], "Document", d.get("title") or d["filename"],
                               year=d.get("year"), authors=d.get("authors"), doi=d.get("doi"),
                               ntrs=d.get("ntrs_accession"), report_number=d.get("report_number"),
                               source_pdf=d["source_pdf"], conversion_status=d["conversion"]["status"],
                               topics=d.get("topics"), document_type=d.get("document_type"))
        if d.get("generated_markdown"):
            mid = "MD_" + d["id"][4:]
            nodes[mid] = _node(mid, "GeneratedDocument", d["generated_markdown"],
                               path=d["generated_markdown"], conversion_status=d["conversion"]["status"],
                               converter=d["conversion"].get("converter"))
            add_edge(mid, d["id"], "DERIVED_FROM", provenance="conversion_manifest")
        if d.get("alternate_copy_of"):
            add_edge(d["id"], d["alternate_copy_of"], "EQUIVALENT_TO", provenance="bibliographic_identity",
                     note=d.get("alternate_copy_basis"))

    # Docling output remains derivative extraction evidence. Formula/table
    # candidates are not reviewed Equation nodes or scientific support.
    from .enrichment import graph_records
    enrichment_nodes, enrichment_edges = graph_records()
    nodes.update({n["id"]: n for n in enrichment_nodes})
    for enrichment_edge in enrichment_edges:
        edge = dict(enrichment_edge)
        add_edge(edge.pop("from"), edge.pop("to"), edge.pop("type"), **edge)

    # ---- quantities ----
    for q in load("quantities"):
        nodes[q["qid"]] = _node(q["qid"], "PhysicalQuantity", q["canonical"], si_unit=q["si_unit"],
                                dimension=q["dimension"], symbols=q["symbols"], subsystems=q["subsystems"])

    # ---- subsystems ----
    from .schema import SUBSYSTEMS
    for s in SUBSYSTEMS:
        sid = f"SUB_{s.upper()}"
        nodes[sid] = _node(sid, "Subsystem", s)

    # ---- models (registry) ----
    equations_by_id = {e["id"]: e for e in load("equations")}
    for m in load("models"):
        support_state, support_note = model_source_support(m, documents.get(m.get("source_document_id")))
        nodes[m["id"]] = _node(m["id"], "Model", m["registry_id"], subsystem=m["subsystem"],
                               quantity=m["quantity"], relation=m["relation"], basis_class=m["basis_class"],
                               promotion_status=m["promotion_status"], validation_level=m["validation_level"],
                               registry_status=m["registry_status"], source_support_state=m.get("source_support_state"),
                               active=m.get("active", True), source_state=m.get("source_state"),
                               source_support_notes=m.get("source_support_notes"),
                               promotion_status_basis=m.get("promotion_status_basis"),
                               basis_class_basis=m.get("basis_class_basis"), execution_reach=m.get("execution_reach"),
                               software_verification_state=m.get("software_verification_state"),
                               physical_validation_state=m.get("physical_validation_state"),
                               effective_source_support_state=support_state, source_support_evidence_note=support_note,
                               source_document_id=m.get("source_document_id"),
                               source_support_sha256=m.get("source_support_sha256"))
        add_edge(m["id"], f"SUB_{m['subsystem'].upper()}" if f"SUB_{m['subsystem'].upper()}" in nodes else "SUB_" + _sub_alias(m["subsystem"]),
                 "PART_OF", provenance="model_registry")
        if m.get("source_document_id"):
            add_edge(m["id"], m["source_document_id"], "CITES", provenance="model_registry",
                     basis=m["basis_class"], equation_ref=m.get("equation_ref"),
                     support_state=support_state, support_notes=support_note)
            if support_state in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
                add_edge(m["id"], m["source_document_id"], "SUPPORTED_BY", provenance="model_source_review",
                         support_state=support_state, support_notes=support_note,
                         source_sha256=m.get("source_support_sha256"),
                         scope="citation support only; not implementation or physical validation")
                for eid in m.get("source_comparison_equation_ids", []):
                    equation = equations_by_id.get(eid, {})
                    if (equation.get("document_id") == m["source_document_id"]
                            and equation.get("source_sha256") == m.get("source_support_sha256")):
                        add_edge(m["id"], eid, "CITES", provenance="model_source_review",
                                 scope="explicit source/implementation comparison; no IMPLEMENTS assertion",
                                 support_notes=support_note)
    for v in load("validity"):
        nodes[v["id"]] = _node(v["id"], "ValidityEnvelope", v["prose"][:120], prose=v["prose"],
                               machine_readable_state=v.get("machine_readable_state"), source_state=v.get("source_state"))
        if v.get("source_state") == "STALE_SOURCE":
            continue
        for mid in v.get("model_ids", []) or [v.get("model_id")]:
            add_edge(mid, v["id"], "HAS_VALIDITY_ENVELOPE", provenance=v.get("provenance_id") or v.get("provenance_kind", ""))

    # ---- equations / claims / entities (LLM-extracted, grounded) ----
    for e in load("equations"):
        nodes[e["id"]] = _node(e["id"], "Equation", e.get("label") or e.get("form", "")[:80],
                               form=e.get("form"), page=e.get("page"), document_id=e.get("document_id"),
                               confidence=e.get("confidence"), grounded=e.get("grounded"),
                               source_state=e.get("source_state"), verification=e.get("verification"),
                               unresolved_variables=[v for v in e.get("variables", [])
                                                     if v.get("resolution_state") != "RESOLVED"],
                               transcription_state=equation_formulation_state(e))
        if e.get("source_state") == "STALE_SOURCE" or e.get("verification") == "REFUTED":
            continue
        for old_id in e.get("corrects_ids", []):
            add_edge(e["id"], old_id, "CORRECTS", provenance=e.get("provenance_id", ""), scope="superseded extraction history")
        for candidate_id in e.get("corrects_candidate_ids", []):
            candidate = nodes.get(candidate_id, {})
            if candidate.get("document_id") == e["document_id"] and candidate.get("source_sha256") == e.get("source_sha256"):
                add_edge(e["id"], candidate_id, "CORRECTS", provenance=e.get("provenance_id", ""),
                         scope="Parser notation corrected by original-PDF inspection; not model validation")
        add_edge(e["id"], e["document_id"], "FOUND_IN", provenance=e.get("provenance_id", ""), page=e.get("page"))
        for candidate_id in e.get("enrichment_candidate_ids", []):
            candidate = nodes.get(candidate_id, {})
            if candidate.get("active", True) and candidate.get("document_id") == e["document_id"] and candidate.get("source_sha256") == e.get("source_sha256"):
                add_edge(e["id"], candidate_id, "DERIVED_FROM", provenance=e.get("provenance_id", ""),
                         note="Reviewed source transcription used this parser candidate; parser alone does not establish correctness.")
        for v in e.get("variables", []):
            if v.get("resolution_state") != "RESOLVED":
                continue  # candidate identities are not physical dependencies
            for qid in v.get("quantity_candidates", []):
                add_edge(e["id"], qid, "INPUT_VARIABLE" if v.get("role") != "output" else "OUTPUT_VARIABLE",
                         provenance=e.get("provenance_id", ""), symbol=v.get("symbol"),
                         resolution=v.get("resolution_state"))
        for a in e.get("assumption_ids", []):
            add_edge(e["id"], a, "ASSUMES", provenance=e.get("provenance_id", ""))
        for mid in e.get("model_ids", []):
            add_edge(mid, e["id"], "IMPLEMENTS", provenance=e.get("provenance_id", ""),
                     link_basis=e.get("model_link_basis"))
    for r in load("relationships"):
        if (not r.get("active", True) or r.get("source_state", "CURRENT") != "CURRENT"
                or r.get("verification") == "REFUTED"):
            continue  # retain the stored assertion/history without an active graph edge
        add_edge(r["from"], r["to"], r["edge"], provenance="SYSTEM_INFERS:" + r["id"], note=r.get("note"),
                 source_relation=r.get("source_relation"))
    for c in load("claims"):
        nodes[c["id"]] = _node(c["id"], "Claim", f"{c.get('subject','')} {c.get('predicate','')} {c.get('object','')}"[:120],
                               claim_type=c.get("claim_type"), page=c.get("page"), document_id=c.get("document_id"),
                               confidence=c.get("confidence"), grounded=c.get("grounded"),
                               source_state=c.get("source_state"), verification=c.get("verification"))
        if c.get("source_state") == "STALE_SOURCE" or c.get("verification") == "REFUTED":
            continue
        add_edge(c["id"], c["document_id"], "FOUND_IN", provenance=c.get("provenance_id", ""), page=c.get("page"))
        for mid in c.get("model_ids", []):
            add_edge(c["id"], mid, "SUPPORTS" if c.get("claim_type") != "validity_statement" else "CONSTRAINS",
                     provenance=c.get("provenance_id", ""))
        for other in c.get("contradicts", []):
            add_edge(c["id"], other, "CONTRADICTS", provenance=c.get("provenance_id", ""))
        for qid in c.get("quantity_ids", []):
            add_edge(c["id"], qid, "DESCRIBES", provenance=c.get("provenance_id", ""))
    for a in load("entities"):
        nodes[a["id"]] = _node(a["id"], a.get("type", "Concept"), a.get("label", a["id"]),
                               **{k: a[k] for k in ("subsystem", "document_id", "page", "prose", "source_state") if k in a})
        if a.get("source_state") == "STALE_SOURCE":
            continue
        if a.get("document_id"):
            add_edge(a["id"], a["document_id"], "FOUND_IN", provenance=a.get("provenance_id", ""), page=a.get("page"))
        for mid in a.get("model_ids", []):
            add_edge(mid, a["id"], "ASSUMES" if a.get("type") == "Assumption" else "USES",
                     provenance=a.get("provenance_id", ""))

    # ---- software components / tests / capabilities / repo docs ----
    for s in load("software_links"):
        if s.get("type") == "SoftwareComponent":
            nodes[s["id"]] = _node(s["id"], "SoftwareComponent", s["qualname"], file=s["file"], line=s["line"],
                                   subsystem=s.get("subsystem"), role=s.get("role"), verified=s.get("verified"),
                                   active=s.get("active", True), source_state=s.get("source_state", "CURRENT"))
            for mid in s.get("model_ids", []):
                add_edge(s["id"], mid, "IMPLEMENTS", provenance=s.get("provenance_id", ""))
            for eid in s.get("equation_ids", []):
                add_edge(s["id"], eid, "IMPLEMENTS", provenance=s.get("provenance_id", ""))
            for did in s.get("cited_document_ids", []):
                add_edge(s["id"], did, "CITES", provenance=s.get("provenance_id", ""))
            for t in s.get("test_ids", []):
                add_edge(s["id"], t, "TESTED_WITH", provenance=s.get("provenance_id", ""))
            if s.get("variant_of_id"):
                add_edge(s["id"], s["variant_of_id"], s.get("variant_edge", "COMPARED_WITH"),
                         provenance=s.get("provenance_id", ""))
            for g in s.get("gate_ids", []):
                add_edge(s["id"], g, "GATED_BY", provenance=s.get("provenance_id", ""))
            for r in s.get("refusals", []):
                digest = hashlib.sha256(r.get('condition', '').encode()).hexdigest()[:16]
                rid = f"{s['id']}::REFUSES::{digest}"
                nodes[rid] = _node(rid, "ValidationGate", r.get("condition", "")[:100], behavior=r.get("behavior"),
                                   message=r.get("message"), line=r.get("line"), kind="applicability_gate")
                add_edge(s["id"], rid, "REFUSES_WHEN", provenance=s.get("provenance_id", ""))
            for k in s.get("constant_ids", []):
                add_edge(s["id"], k, "USES", provenance=s.get("provenance_id", ""))
        elif s.get("type") == "SoftwareLink":
            if s["id"] in inactive_software:
                continue
            # Old registry harvests asserted support merely from local_source.
            # Rebuilds and legacy stores must both preserve this as a citation.
            if (s.get("edge") == "SUPPORTED_BY" and s.get("from", "").startswith("MODEL_")
                    and (s.get("provenance", "").startswith("PROV_REGISTRY_")
                         or "registry local_source=" in s.get("note", ""))):
                add_edge(s["from"], s["to"], "CITES", provenance=s.get("provenance", ""),
                         note=s.get("note"), basis=s.get("basis"))
                continue
            add_edge(s["from"], s["to"], s["edge"], provenance=s.get("provenance", ""), note=s.get("note"),
                     basis=s.get("basis"))
        elif s.get("type") == "EmpiricalConstant":
            nodes[s["id"]] = _node(s["id"], "EmpiricalConstant", s.get("name"), value=s.get("value"),
                                   units=s.get("units"), basis_class=s.get("basis_class"),
                                   reported_basis_class=s.get("reported_basis_class"),
                                   basis_class_basis=s.get("basis_class_basis"),
                                   scientific_basis_review_state=s.get("scientific_basis_review_state"),
                                   value_freshness_state=s.get("value_freshness_state"),
                                   verified_scope=s.get("verified_scope"),
                                   stated_basis=s.get("stated_basis"), file=s.get("file"), line=s.get("line"),
                                   active=s.get("active", True), source_state=s.get("source_state", "CURRENT"))
    for t in load("tests"):
        nodes[t["id"]] = _node(t["id"], "Benchmark" if t.get("verification_kind") == "published_benchmark_reproduction" else "Test",
                               t.get("path", t["id"]), verification_kind=t.get("verification_kind"),
                               subsystem=t.get("subsystem"), source_document_id=t.get("source_document_id"),
                               expected=t.get("expected"))
        if t.get("source_document_id"):
            add_edge(t["id"], t["source_document_id"], "BENCHMARKED_AGAINST", provenance=t.get("provenance_id", ""))
        for mid in t.get("model_ids", []):
            add_edge(t["id"], mid, "VALIDATES" if t.get("verification_kind") == "published_benchmark_reproduction" else "TESTED_WITH",
                     provenance=t.get("provenance_id", ""))
    for c in load("capabilities"):
        nodes[c["id"]] = _node(c["id"], "Capability", c.get("name", c["id"]), state=c.get("state"),
                               enforced_by=c.get("enforced_by"), unblock_requires=c.get("unblock_requires"))
        for did in c.get("blocked_on_document_requests", []):
            add_edge(c["id"], did, "BLOCKED_ON_LITERATURE", provenance=c.get("provenance_id", ""))
        for sid in c.get("subsystems", []):
            add_edge(c["id"], f"SUB_{sid.upper()}", "PART_OF", provenance=c.get("provenance_id", ""))
    for r in load("repo_documents"):
        nodes[r["id"]] = _node(r["id"], "RepositoryDocument", r.get("path"), date=r.get("date"), kind=r.get("kind"),
                               status_head=r.get("status_head"), authority="implementation_state_not_science")
        for rel in r.get("relations", []):
            add_edge(r["id"], rel["target_id"], rel["edge"], provenance=r.get("provenance_id", ""), quote=rel.get("quote"))
    for req in load("requirements"):
        ntype = {"optimizer_constraint": "Constraint", "postsolve_gate": "ValidationGate",
                 "applicability_gate": "ValidationGate", "diagnostic": "Constraint"}.get(req.get("kind"), "Requirement")
        if req.get("is_requirement"):
            ntype = "Requirement"
        nodes[req["id"]] = _node(req["id"], ntype, req.get("name", req["id"]), kind=req.get("kind"),
                                 optimizer_role=req.get("optimizer_role"), quantity=req.get("quantity"),
                                 units=req.get("units"), applicability=req.get("applicability"),
                                 basis_state=req.get("basis_state"), enforced_by=req.get("enforced_by"),
                                 file=req.get("file"), line=req.get("line"))
        for did in req.get("source_document_ids", []):
            add_edge(req["id"], did, "CITES", provenance=req.get("provenance_id", ""),
                     note=req.get("source_id_text"))
        sub = f"SUB_{str(req.get('subsystem','')).upper()}"
        if sub in nodes:
            add_edge(req["id"], sub, "PART_OF", provenance=req.get("provenance_id", ""))
        for mid in req.get("model_ids", []):
            add_edge(req["id"], mid, "CONSTRAINS", provenance=req.get("provenance_id", ""))
        for qid in req.get("quantity_ids", []):
            add_edge(req["id"], qid, "CONSTRAINS", provenance=req.get("provenance_id", ""))

    # ---- observed engineering report/configuration/CAD snapshots ----
    # Inactive snapshots remain inspectable, but their old associations cannot
    # establish current configuration binding or artifact existence.
    for artifact in load("artifacts"):
        if artifact.get("type") == "ArtifactLink":
            if artifact.get("active", True):
                add_edge(artifact["from"], artifact["to"], artifact["edge"],
                         provenance=artifact.get("provenance_id", ""),
                         binding_state=artifact.get("binding_state"),
                         binding_basis=artifact.get("binding_basis"))
            continue
        nodes[artifact["id"]] = _node(
            artifact["id"], artifact["type"], artifact.get("name", artifact["id"]),
            **{k: artifact[k] for k in (
                "kind", "scope", "state", "active", "source_state", "source_path",
                "source_sha256", "json_pointer", "provenance_id", "path", "format",
                "observed_sha256", "snapshot_sha256", "binding_state", "binding_basis",
                "producer_binding_state", "generation_time_integrity_state",
                "execution_state", "qualification_state", "hardware_qualified",
            ) if k in artifact})

    # drop dangling edges but report them
    dangling = [e for e in edges if e["from"] not in nodes or e["to"] not in nodes]
    edges = [e for e in edges if e["from"] in nodes and e["to"] in nodes]
    graph = {
        "meta": {"ontology_version": ONTOLOGY_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
                 "n_nodes": len(nodes), "n_edges": len(edges), "n_dangling_dropped": len(dangling)},
        "nodes": list(nodes.values()), "edges": edges, "dangling": dangling[:200],
    }
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (GRAPH_DIR / "graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False))
    _write_graphml(graph)
    return graph


def _sub_alias(sub: str) -> str:
    aliases = {"thermal": "COOLING", "structure": "STRUCTURES", "flow": "GAS_DYNAMICS", "altitude": "PERFORMANCE",
               "interface": "CONFIGURATION", "separation": "GAS_DYNAMICS", "throat": "THROAT", "nozzle": "NOZZLE_CONTOUR",
               "chamber": "CHAMBER", "cad": "CAD", "pump": "PUMP", "spray": "SPRAY", "injector": "INJECTOR",
               "materials": "MATERIALS", "performance": "PERFORMANCE", "thermochemistry": "THERMOCHEMISTRY",
               "trajectory": "TRAJECTORY"}
    return aliases.get(sub, sub.upper())


def _write_graphml(graph: dict) -> None:
    keys = {}
    def key_for(name):
        if name not in keys:
            keys[name] = f"d{len(keys)}"
        return keys[name]
    node_lines, edge_lines = [], []
    for n in graph["nodes"]:
        attrs = "".join(f'<data key="{key_for(k)}">{escape(str(v))}</data>' for k, v in n.items() if k != "id")
        node_lines.append(f'<node id="{escape(n["id"])}">{attrs}</node>')
    for i, e in enumerate(graph["edges"]):
        attrs = "".join(f'<data key="{key_for("e_" + k)}">{escape(str(v))}</data>' for k, v in e.items() if k not in ("from", "to"))
        edge_lines.append(f'<edge id="e{i}" source="{escape(e["from"])}" target="{escape(e["to"])}">{attrs}</edge>')
    key_lines = [f'<key id="{kid}" for="{"edge" if name.startswith("e_") else "node"}" attr.name="{escape(name[2:] if name.startswith("e_") else name)}" attr.type="string"/>'
                 for name, kid in keys.items()]
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n' + "\n".join(key_lines) +
           '\n<graph id="lrekit" edgedefault="directed">\n' + "\n".join(node_lines) + "\n" + "\n".join(edge_lines) +
           "\n</graph>\n</graphml>\n")
    (GRAPH_DIR / "graph.graphml").write_text(xml)


class Graph:
    def __init__(self, data: dict | None = None):
        if data is None:
            data = json.loads((GRAPH_DIR / "graph.json").read_text())
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        for e in self.edges:
            self.out[e["from"]].append(e)
            self.inc[e["to"]].append(e)
        self.meta = data.get("meta", {})

    def neighbors(self, nid: str, edge_types: set[str] | None = None, direction: str = "both") -> list[dict]:
        res = []
        if direction in ("both", "out"):
            for e in self.out.get(nid, []):
                if not edge_types or e["type"] in edge_types:
                    res.append({"direction": "out", "edge": e["type"], "node": self.nodes.get(e["to"]),
                                "provenance": e.get("provenance"), "attrs": {k: v for k, v in e.items() if k not in ("from", "to", "type", "provenance")}})
        if direction in ("both", "in"):
            for e in self.inc.get(nid, []):
                if not edge_types or e["type"] in edge_types:
                    res.append({"direction": "in", "edge": e["type"], "node": self.nodes.get(e["from"]),
                                "provenance": e.get("provenance"), "attrs": {k: v for k, v in e.items() if k not in ("from", "to", "type", "provenance")}})
        return res

    HUB_TYPES = {"Subsystem"}

    def find_path(self, a: str, b: str, max_depth: int = 6, allow_hubs: bool = False) -> list[dict] | None:
        """BFS shortest path.  Subsystem hub nodes are avoided first (a path through
        'SUB_COOLING' says nothing scientific); if no hub-free path exists the search
        is repeated allowing them."""
        if a not in self.nodes or b not in self.nodes:
            return None
        prev: dict[str, tuple[str, dict] | None] = {a: None}
        q = deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            if cur == b:
                break
            if d >= max_depth:
                continue
            for e in self.out.get(cur, []) + self.inc.get(cur, []):
                nxt = e["to"] if e["from"] == cur else e["from"]
                if not allow_hubs and nxt != b and self.nodes.get(nxt, {}).get("type") in self.HUB_TYPES:
                    continue
                if nxt not in prev:
                    prev[nxt] = (cur, e)
                    q.append((nxt, d + 1))
        if b not in prev and not allow_hubs:
            return self.find_path(a, b, max_depth, allow_hubs=True)
        if b not in prev:
            return None
        path = []
        cur = b
        while prev[cur]:
            p, e = prev[cur]
            path.append({"from": p, "to": cur, "edge": e["type"], "reversed": e["from"] != p, "provenance": e.get("provenance")})
            cur = p
        return list(reversed(path))

    def find(self, text: str, types: set[str] | None = None, limit: int = 20) -> list[dict]:
        t = text.lower()
        hits = []
        for n in self.nodes.values():
            if types and n["type"] not in types:
                continue
            hay = " ".join(str(v) for k, v in n.items() if k != "id").lower()
            if t in hay or t in n["id"].lower():
                hits.append(n)
        return hits[:limit]
