"""Human-readable ingestion / validation report (Markdown) from the JSON reports."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from .store import REPORT_DIR, load


def _j(name: str) -> dict:
    p = REPORT_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def render() -> str:
    ing = _j("latest_ingestion.json")
    val = _j("latest_validation.json")
    ext = _j("latest_extraction.json")
    sync = _j("latest_sync.json")
    docs = load("manifest")
    models = load("models")
    eqs = load("equations")
    cls = load("claims")
    now = datetime.now(timezone.utc).isoformat()
    lines = [f"# LREKit knowledge system — ingestion report", f"", f"Generated {now}", ""]
    lines += ["## Corpus", "",
              f"- PDFs on disk: {sync.get('pdf_count', '?')}; conversion rows: {sync.get('manifest_rows', '?')}; "
              f"synchronized: {len(sync.get('synchronized', []))}; missing Markdown: {len(sync.get('missing_markdown', []))}; "
              f"stale: {len(sync.get('stale_markdown', []))}; orphan Markdown: {len(sync.get('orphan_markdown', []))}; "
              f"duplicate payload groups: {len(sync.get('duplicate_payloads', {}))}",
              f"- Unique documents: {len(docs)}; alternate copies: {sum(1 for d in docs if d.get('alternate_copy_of'))}",
              f"- Conversion status: {dict(Counter(d['conversion']['status'] for d in docs))}",
              f"- Low-confidence conversions: {sum(1 for d in docs if d['conversion'].get('low_confidence'))}", ""]
    ch = ing.get("steps", {}).get("chunks", {})
    idx = ing.get("steps", {}).get("index", {})
    lines += ["## Chunks and index", "", f"- {ch}", f"- {idx}", ""]
    lines += ["## Knowledge stores", ""]
    for name in ("models", "validity", "equations", "claims", "entities", "software_links", "tests", "capabilities", "requirements", "repo_documents", "relationships", "provenance"):
        lines.append(f"- {name}: {len(load(name))}")
    lines.append("")
    lines += ["## Model registry ↔ source support (extraction pass)", ""]
    sup = Counter(m.get("source_support_state", "NOT_CHECKED") for m in models)
    lines.append(f"- {dict(sup)}")
    for m in models:
        if m.get("source_support_state") in ("NOT_FOUND_IN_SOURCE", "PARTIALLY_SUPPORTED", "SOURCE_UNREADABLE"):
            lines.append(f"  - `{m['registry_id']}` → {m['source_support_state']}: {str(m.get('source_support_notes',''))[:220]}")
    lines.append("")
    lines += ["## Extraction grounding", "",
              f"- grounded: {ext.get('grounded')}, page-shifted: {ext.get('page_shifted')}, ungrounded (dropped): {ext.get('n_ungrounded')}",
              f"- equations: {len(eqs)} (confirmed {sum(1 for e in eqs if e.get('verification')=='CONFIRMED')}, refuted {sum(1 for e in eqs if e.get('verification')=='REFUTED')}), "
              f"claims: {len(cls)}, validity statements from sources: {sum(1 for v in load('validity') if v.get('provenance_kind')=='SOURCE_STATES')}",
              ""]
    if ext.get("ungrounded"):
        lines.append("Dropped (anchor not found):")
        for u in ext["ungrounded"][:25]:
            lines.append(f"  - {u.get('doc_id')} p{u.get('page')} [{u.get('kind')}] {str(u.get('label'))[:90]}")
        lines.append("")
    lines += ["## Validation", "", f"- issues: {val.get('summary', {}).get('n_issues')} — {val.get('summary', {}).get('issues_by_check')}",
              f"- stats: {json.dumps(val.get('summary', {}).get('stats', {}), default=str)[:1200]}", ""]
    caps = load("capabilities")
    lines += ["## Explicitly unavailable / blocked", "", f"- {dict(Counter(c['state'] for c in caps))}", ""]
    consts = [s for s in load("software_links") if s.get("type") == "EmpiricalConstant"]
    lines += ["## Hard-coded constants by basis", "", f"- {dict(Counter(k['basis_class'] for k in consts))}", ""]
    unk = [k for k in consts if k["basis_class"] == "UNKNOWN_BASIS"]
    lines.append(f"UNKNOWN_BASIS constants ({len(unk)}), first 30:")
    for k in unk[:30]:
        lines.append(f"  - {k['file']}:{k['line']} `{k['name']}` = {k['value']}")
    return "\n".join(lines) + "\n"


def write() -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = render()
    (REPORT_DIR / "latest_report.md").write_text(text)
    return text
