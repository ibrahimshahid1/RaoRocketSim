"""Deterministic harvest of repository-side knowledge.

* ``docs/*.md`` -> RepositoryDocument records with dates, STATUS heads and
  explicit precedence language (SUPERSEDES / CORRECTS / UPDATES / DEPRECATES /
  BLOCKS), each relation carrying the verbatim line that asserts it.
  Repository documents are implementation-state authority, never scientific
  authority; that distinction is stored on every record.
* ``tests/*.py`` -> Test records classified by verification kind from their
  names and contents; tests that reference a corpus PDF or a named published
  case are flagged ``published_benchmark_reproduction`` candidates and linked
  to the Document.  Classification is heuristic and says so.
* ``raosim/ lrekit/ scripts/`` -> Capability records for explicit
  NotImplementedError / unavailable / blocked paths, with file:line.
* ``docs/LITERATURE_REQUESTS*.md`` -> BLOCKED_ON_LITERATURE capability records.
* every ``propulsion_texts/...pdf`` string in code/docs/tests -> CITES edges
  (path must exist on disk; unresolved paths are reported).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT
from .schema import node_id
from .store import load, save

DATE_IN_NAME = re.compile(r"(20\d{2}-\d{2}-\d{2})")
DATE_IN_TEXT = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
STATUS_HEAD = re.compile(r"^\s*(?:#+\s*)?\**\s*STATUS\b[^\n]*", re.IGNORECASE | re.MULTILINE)
DOC_REF = re.compile(r"(?:docs/)?([A-Za-z0-9_.\-]+\.md)")
PRECEDENCE = [
    ("SUPERSEDES", re.compile(r"\bsupersedes\b", re.I)),
    ("SUPERSEDED_BY", re.compile(r"\bsuperseded by\b", re.I)),
    ("CORRECTS", re.compile(r"\bcorrects\b", re.I)),
    ("CORRECTED_BY", re.compile(r"\bcorrected by\b", re.I)),
    ("UPDATES", re.compile(r"\bupdates\b", re.I)),
    ("DEPRECATES", re.compile(r"\bdeprecates\b|\bdeprecated\b", re.I)),
    ("BLOCKS", re.compile(r"\bblocks\b|\bblocked by\b|\bblocked on\b", re.I)),
    ("AUDITS", re.compile(r"\baudits\b|\baudit of\b", re.I)),
]
STALENESS_WORDS = re.compile(r"\b(stale|historical|retained only|no longer true|use .{3,60} instead|landed|deferred|blocked)\b", re.I)
PDF_REF = re.compile(r"propulsion_texts/[^\s'\"`)\]\\>]+?\.pdf")
UNAVAILABLE = [
    ("NOT_IMPLEMENTED", re.compile(r"raise\s+NotImplementedError\b")),
    ("UNAVAILABLE", re.compile(r"(?:status|state|verdict|availability|return|raise|=)\s*[^\n]{0,40}[\"']unavailable[\"']|\bUNAVAILABLE\b")),
    ("BLOCKED_ON_LITERATURE", re.compile(r"BLOCKED_ON_LITERATURE|blocked on literature|literature request", re.I)),
    ("BLOCKED_ON_STANDARD", re.compile(r"BLOCKED_ON_STANDARD|blocked on standard", re.I)),
    ("NOT_QUALIFIED", re.compile(r"\bnot_qualified\b|\bNOT_QUALIFIED\b|hardware[- ]qualified", re.I)),
    ("UNSUPPORTED", re.compile(r"[\"']unsupported[\"']|\bnot supported\b", re.I)),
]

TEST_KIND_RULES = [
    ("numpy_jax_parity", re.compile(r"parity|numpy.*jax|jax.*numpy|oracle", re.I)),
    ("autodiff_finite_difference", re.compile(r"finite[_ -]?diff|\bfd\b|check_grads|jvp|vjp|gradient", re.I)),
    ("cad_round_trip", re.compile(r"\bstep\b|\bstl\b|brep|cadquery|solid|round[_ -]?trip", re.I)),
    ("snapshot_contract", re.compile(r"snapshot|contract|schema|manifest", re.I)),
    ("published_benchmark_reproduction", re.compile(r"benchmark|reference_case|nasa_reference|reproduc|published|literature", re.I)),
    ("model_vs_model_comparison", re.compile(r"compar|cross[_ -]?check|versus|_vs_", re.I)),
    ("numerical_regression", re.compile(r"regression|convergence|residual|tolerance", re.I)),
]


def _docs_harvest(now: str) -> tuple[list[dict], list[dict]]:
    docs_dir = REPO_ROOT / "docs"
    from raosim.model_registry import MODEL_REGISTRY
    registry_notes = [REPO_ROOT / m.local_source for m in MODEL_REGISTRY.values()
                      if m.local_source and m.local_source.endswith(".md")]
    files = sorted({f for f in list(docs_dir.rglob("*.md")) + [REPO_ROOT / "README.md"] + registry_notes
                    if f.is_file()})
    by_name = {f.name: node_id("RDOC", str(f.relative_to(REPO_ROOT))) for f in files}
    records, unresolved = [], []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rel = str(f.relative_to(REPO_ROOT))
        rid = node_id("RDOC", rel)
        date = None
        m = DATE_IN_NAME.search(f.name)
        if m:
            date = m.group(1)
            date_basis = "filename"
        else:
            dates = DATE_IN_TEXT.findall(text[:4000])
            date = max(dates) if dates else None
            date_basis = "first 4000 chars of content" if date else "unknown"
        title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")), f.name)
        sh = STATUS_HEAD.search(text)
        status_head = ""
        if sh:
            start = text[:sh.start()].count("\n")
            status_head = "\n".join(lines[start:start + 12]).strip()[:1200]
        kind = "plan" if "PLAN" in f.name.upper() else ("audit" if "AUDIT" in f.name.upper() else
               ("literature_request" if "LITERATURE_REQUEST" in f.name.upper() else
                ("provenance" if "PROVENANCE" in f.name.upper() or "REGISTRY" in f.name.upper() or "BASIS" in f.name.upper() else
                 ("evidence" if "EVIDENCE" in f.name.upper() else ("guide" if "GUIDE" in f.name.upper() or f.name == "README.md" else "note")))))
        relations = []
        for i, ln in enumerate(lines):
            for edge, rx in PRECEDENCE:
                if rx.search(ln):
                    for m2 in DOC_REF.finditer(ln):
                        target = m2.group(1)
                        if target == f.name or target not in by_name:
                            if target != f.name and target.endswith(".md"):
                                unresolved.append({"doc": rel, "line": i + 1, "target": target, "edge": edge})
                            continue
                        e = edge
                        src, dst = rid, by_name[target]
                        if edge.endswith("_BY"):
                            e = edge[:-3]
                            src, dst = dst, rid
                        relations.append({"edge": e, "from": src, "target_id": dst, "line": i + 1, "quote": ln.strip()[:240]})
        staleness = [{"line": i + 1, "quote": ln.strip()[:200]} for i, ln in enumerate(lines) if STALENESS_WORDS.search(ln)][:25]
        records.append({
            "id": rid, "type": "RepositoryDocument", "path": rel, "title": title[:160], "kind": kind,
            "date": date, "date_basis": date_basis, "status_head": status_head,
            "authority": "implementation_state_not_science",
            "summary": " ".join(lines[1:12])[:600],
            "relations": [r for r in relations if r["from"] == rid],
            "incoming_relations": [r for r in relations if r["from"] != rid],
            "staleness_signals": staleness, "n_lines": len(lines),
            "cited_pdfs": sorted(set(PDF_REF.findall(text))),
            "provenance_id": node_id("PROV", f"repo doc {rel}"), "harvested_at": now,
        })
    # incoming relations (X SUPERSEDED_BY Y means Y SUPERSEDES X) attach to the source record
    by_id = {r["id"]: r for r in records}
    for r in records:
        for rel in r.pop("incoming_relations"):
            src = by_id.get(rel["from"])
            if src:
                src["relations"].append(rel)
    return records, unresolved


def _tests_harvest(now: str, docs_by_path: dict[str, str]) -> list[dict]:
    tests_dir = REPO_ROOT / "tests"
    prior = {t["id"]: t for t in load("tests")}
    out = {}
    for f in sorted(tests_dir.glob("test_*.py")):
        text = f.read_text(encoding="utf-8", errors="replace")
        rel = str(f.relative_to(REPO_ROOT))
        name = f.stem.lower()
        # Conservative classification: the file NAME decides, content only confirms.
        # A test is a benchmark reproduction only if it cites a corpus PDF AND compares
        # against a tolerance; everything else defaults to unit_test.
        kinds = []
        pdfs = sorted(set(PDF_REF.findall(text)))
        doc_ids = sorted({docs_by_path[p] for p in pdfs if p in docs_by_path})
        has_tol = bool(re.search(r"\b(rtol|atol|approx|tolerance|expected)\b", text))
        if doc_ids and has_tol:
            kinds.append("published_benchmark_reproduction")
        if re.search(r"parity|oracle", name) or (re.search(r"\bjax\b", name) and "allclose" in text):
            kinds.append("numpy_jax_parity")
        if re.search(r"check_grads|finite[_-]?diff|central[_-]?diff", text):
            kinds.append("autodiff_finite_difference")
        if re.search(r"\b(cad|brep|step|stl)\b", name):
            kinds.append("cad_round_trip")
        if re.search(r"snapshot|contract|schema", name):
            kinds.append("snapshot_contract")
        if re.search(r"regression|convergence|resolution", name):
            kinds.append("numerical_regression")
        if re.search(r"compare|comparison|parity", name) and "numpy_jax_parity" not in kinds:
            kinds.append("model_vs_model_comparison")
        primary = kinds[0] if kinds else "unit_test"
        n_tests = len(re.findall(r"^\s*def test_", text, re.M))
        subsystem = _subsystem_from_name(name)
        tid = node_id("TEST", rel)
        # named expected values are captured only as text snippets, never interpreted
        expected = [ln.strip()[:160] for ln in text.splitlines() if re.search(r"\b(expected|reference|published)\b.*=\s*[-\d.]", ln)][:6]
        rec = {"id": tid, "type": "Test", "path": rel, "file": rel, "n_tests": n_tests,
               "verification_kind": primary, "verification_kinds_all": kinds or ["unit_test"],
               "verification_kind_basis": "heuristic classification from test name/content; not a physical-validation claim",
               "subsystem": subsystem, "cited_pdfs": pdfs, "source_document_id": doc_ids[0] if doc_ids else "",
               "source_document_ids": doc_ids, "expected": expected, "detail": text[:300],
               "model_ids": sorted(set(prior.get(tid, {}).get("model_ids", []))),
               "is_experimental_validation": False,
               "provenance_id": node_id("PROV", f"tests harvest {rel}"), "harvested_at": now}
        out[tid] = rec
    # keep the per-test-function records from the survey (TEST ids for path::name) and attach file-level kind
    for tid, t in prior.items():
        if tid not in out:
            file_rec = out.get(node_id("TEST", t.get("file", "")))
            if file_rec:
                t["verification_kind"] = file_rec["verification_kind"]
                t["verification_kind_basis"] = file_rec["verification_kind_basis"]
                t["source_document_id"] = file_rec["source_document_id"]
            out[tid] = t
    return list(out.values())


def _subsystem_from_name(name: str) -> str:
    table = [("cool", "cooling"), ("thermal", "cooling"), ("regen", "cooling"), ("wall", "cooling"), ("bartz", "heat_transfer"),
             ("rao", "nozzle_contour"), ("nozzle", "nozzle_contour"), ("moc", "moc"), ("nasa", "moc"), ("theta_b", "nozzle_contour"),
             ("inject", "injector"), ("pintle", "injector"), ("spray", "spray"), ("pump", "pump"), ("inducer", "pump"),
             ("mdo", "mdo"), ("jax", "mdo"), ("cad", "cad"), ("brep", "cad"), ("step", "cad"), ("material", "materials"),
             ("cea", "thermochemistry"), ("gas_dyn", "gas_dynamics"), ("perform", "performance"), ("altitude", "performance"),
             ("separation", "performance"), ("mass", "mass"), ("trajectory", "trajectory"), ("requirement", "requirements"),
             ("release", "verification"), ("registry", "verification"), ("design", "configuration"), ("engine", "configuration"),
             ("manufact", "manufacturing"), ("chamber", "chamber"), ("structure", "structures"), ("buckl", "structures")]
    for k, s in table:
        if k in name:
            return s
    return ""


def _capabilities_harvest(now: str, docs_by_path: dict[str, str]) -> tuple[list[dict], list[dict]]:
    caps = []
    cites = []
    roots = [REPO_ROOT / "raosim", REPO_ROOT / "lrekit", REPO_ROOT / "scripts"]
    for root in roots:
        for f in sorted(root.rglob("*.py")):
            rel = str(f.relative_to(REPO_ROOT))
            if rel == "raosim/model_registry.py":
                continue  # its local_source paths are harvested as Model SUPPORTED_BY edges, not CITES
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, ln in enumerate(lines):
                for state, rx in UNAVAILABLE:
                    if rx.search(ln):
                        ctx = " ".join(l.strip() for l in lines[max(0, i - 2): i + 2])[:300]
                        name = re.sub(r"\s+", " ", ln.strip())[:140]
                        caps.append({"id": node_id("CAP", f"{rel}:{i+1}"), "type": "Capability", "name": name,
                                     "state": state, "file": rel, "line": i + 1, "quote": ln.strip()[:240],
                                     "context": ctx, "enforced_by": f"{rel}:{i+1}",
                                     "subsystems": [_subsystem_from_name(rel.lower())] if _subsystem_from_name(rel.lower()) else [],
                                     "detail": ctx, "unblock_requires": "", "blocked_on_document_requests": [],
                                     "provenance_id": node_id("PROV", f"capability scan {rel}:{i+1}"), "harvested_at": now})
                for m in PDF_REF.finditer(ln):
                    path = m.group(0)
                    cites.append({"file": rel, "line": i + 1, "path": path, "document_id": docs_by_path.get(path, ""),
                                  "resolved": path in docs_by_path, "quote": ln.strip()[:200]})
    # literature requests
    for f in sorted((REPO_ROOT / "docs").glob("LITERATURE_REQUESTS*.md")):
        rel = str(f.relative_to(REPO_ROOT))
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        cur = None
        for i, ln in enumerate(lines):
            if ln.startswith("#") and not ln.startswith("# "):
                title = ln.lstrip("# ").strip()
                # keep only headings that read as a request for a named source/topic,
                # not narrative sections ("What I checked and concluded ...")
                looks_like_request = bool(re.match(r"^(?:[A-Z]\d+|\d+[a-z]?\.|Priority|Tier|Request|Needed|Missing|H\d)", title)) \
                    or re.search(r"\b(NASA|SP-|CR-|TN|TM|DOI|AIAA|ASME|standard|monograph|paper|report|handbook)\b", title)
                narrative = re.search(r"\b(what i checked|concluded|headline|summary|overview|context|scope|how to)\b", title, re.I)
                if len(title) > 6 and looks_like_request and not narrative:
                    cur = {"id": node_id("CAP", f"litreq {rel} {title}"), "type": "Capability", "name": title[:160],
                           "state": "BLOCKED_ON_LITERATURE", "file": rel, "line": i + 1, "quote": ln.strip()[:240],
                           "detail": "", "enforced_by": rel, "subsystems": [_subsystem_from_name(title.lower())] if _subsystem_from_name(title.lower()) else [],
                           "unblock_requires": "", "blocked_on_document_requests": [],
                           "provenance_id": node_id("PROV", f"literature request {rel}:{i+1}"), "harvested_at": now}
                    caps.append(cur)
            elif cur is not None and ln.strip() and len(cur["detail"]) < 600:
                cur["detail"] += ln.strip() + " "
    return caps, cites


def harvest() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    docs_by_path = {}
    for d in load("manifest"):
        canonical = d.get("alternate_copy_of") or d["id"]
        docs_by_path[d["source_pdf"]] = canonical
        for alt in d.get("alternate_paths", []):
            docs_by_path[alt] = canonical
    repo_docs, unresolved_doc_refs = _docs_harvest(now)
    tests = _tests_harvest(now, docs_by_path)
    caps, cites = _capabilities_harvest(now, docs_by_path)

    # attach unresolved doc citations to a report; write CITES edges as SoftwareLinks
    links = [s for s in load("software_links") if not (s.get("type") == "SoftwareLink" and s.get("edge") == "CITES" and s.get("basis") == "path_literal")]
    sw_by_file = {}
    for s in links:
        if s.get("type") == "SoftwareComponent":
            sw_by_file.setdefault(s["file"], []).append(s)
    n_cite_edges = 0
    for c in cites:
        if not c["resolved"]:
            continue
        comps = sw_by_file.get(c["file"], [])
        # the nearest component defined above the citing line, else the file itself
        owner = None
        for s in sorted(comps, key=lambda s: s["line"]):
            if s["line"] <= c["line"]:
                owner = s
        src = owner["id"] if owner else node_id("SW", f"{c['file']}:module")
        if not owner and not any(s.get("id") == src for s in links):
            links.append({"id": src, "type": "SoftwareComponent", "qualname": f"{c['file']}:<module>", "file": c["file"], "line": 1,
                          "subsystem": _subsystem_from_name(c["file"].lower()), "role": "module-level citation holder",
                          "model_ids": [], "cited_document_ids": [], "verified": True,
                          "provenance_id": node_id("PROV", f"citation scan {c['file']}")})
        links.append({"id": node_id("LINK", f"{src} cites {c['document_id']} L{c['line']}"), "type": "SoftwareLink", "edge": "CITES",
                      "from": src, "to": c["document_id"], "basis": "path_literal", "provenance": node_id("PROV", f"citation scan {c['file']}:{c['line']}"),
                      "note": f"{c['file']}:{c['line']}: {c['quote']}"})
        n_cite_edges += 1
    save("repo_documents", repo_docs, meta={"generated_at": now, "n": len(repo_docs)})
    save("tests", tests, meta={"generated_at": now, "n": len(tests)})
    save("capabilities", caps, meta={"generated_at": now, "n": len(caps)})
    save("software_links", links, meta={"generated_at": now, "n": len(links)})
    prov = {p["id"]: p for p in load("provenance")}
    for r in repo_docs:
        prov[r["provenance_id"]] = {"id": r["provenance_id"], "type": "ProvenanceRecord", "kind": "PROJECT_DERIVES",
                                    "asserted_by": r["path"], "subject": r["id"], "statement": "repository document (implementation-state authority)"}
    for c in caps:
        prov[c["provenance_id"]] = {"id": c["provenance_id"], "type": "ProvenanceRecord", "kind": "CODE_STATES",
                                    "asserted_by": f"{c['file']}:{c['line']}", "subject": c["id"], "quote": c["quote"]}
    for t in tests:
        prov[t["provenance_id"]] = {"id": t["provenance_id"], "type": "ProvenanceRecord", "kind": "CODE_STATES",
                                    "asserted_by": t["path"], "subject": t["id"], "statement": t.get("verification_kind_basis", "")}
    for c in cites:
        pid = node_id("PROV", f"citation scan {c['file']}:{c['line']}")
        prov[pid] = {"id": pid, "type": "ProvenanceRecord", "kind": "CODE_STATES", "asserted_by": f"{c['file']}:{c['line']}",
                     "subject": c["document_id"] or c["path"], "quote": c["quote"], "resolved": c["resolved"]}
    save("provenance", prov.values(), meta={"generated_at": now, "n": len(prov)})
    return {"repo_documents": len(repo_docs), "doc_relations": sum(len(r["relations"]) for r in repo_docs),
            "unresolved_doc_refs": unresolved_doc_refs[:30], "tests": len(tests),
            "tests_by_kind": dict(Counter(t["verification_kind"] for t in tests)),
            "benchmark_candidates": [t["path"] for t in tests if t["verification_kind"] == "published_benchmark_reproduction" and t.get("source_document_id")],
            "capabilities": len(caps), "capabilities_by_state": dict(Counter(c["state"] for c in caps)),
            "citations": len(cites), "citations_unresolved": [c for c in cites if not c["resolved"]][:40], "cite_edges": n_cite_edges}


if __name__ == "__main__":
    import json
    print(json.dumps(harvest(), indent=1)[:6000])
