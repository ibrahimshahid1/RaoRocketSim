"""Write the session completion report from the packet queue, integration logs and journal.

    .venv-jax/bin/python knowledge/tools/session_completion_report.py \
        --session knowledge/workflows/results/agentic-2026-10-16 \
        --fresh-checkout <fresh_checkout.json> --out knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md

Every count in the generated tables is recomputed from those records, so the
report cannot drift from what was actually integrated; the narrative sections
are passed in by the reviewer (``--narrative``) and kept verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from kg import packet_queue as pq, store  # noqa: E402


SESSION_START = "2026-10-16T18:39"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect(session: Path) -> dict:
    packets = pq.load()["packets"]
    session_packets = {pid: p for pid, p in packets.items()
                       if any(h.get("at", "") >= "2026-10-16" for h in p.get("history", []))}
    by_state = Counter(p["state"] for p in session_packets.values())
    banked = Counter()
    parser = Counter()
    per_packet = []
    for pid, p in sorted(session_packets.items(), key=lambda kv: (-kv[1]["priority"], kv[0])):
        row = {"packet": pid, "state": p["state"], "document": p["document_id"], "pages": p["pages"]}
        # every attempt's log: the first is <pid>.json, later ones <pid>.attempt-N.json
        logs = sorted((session / "integration").glob(f"{pid}.attempt-*.json"),
                      key=lambda path: int(path.stem.rsplit("-", 1)[1]))
        if (session / "integration" / f"{pid}.json").exists():
            logs.insert(0, session / "integration" / f"{pid}.json")
        row["integration_attempts"] = len(logs)
        for log_path in logs:
            log = json.loads(log_path.read_text())
            for step in log.get("steps", []):
                # bank and parser steps keep their output at the step's top level;
                # apply_verification nests its counts under "stats"
                stats = step.get("stats") or step
                if step.get("action") == "bank_extraction" and isinstance(step.get("counts"), dict):
                    # structured report (bank_extraction --report-out)
                    if step["counts"].get("already_banked"):
                        row["rebanked_no_change"] = True
                    for key in ("n_equations", "n_claims", "n_constants", "n_assumptions", "n_validity"):
                        if isinstance(step["counts"].get(key), int):
                            banked[key] += step["counts"][key]
                            row[key] = row.get(key, 0) + step["counts"][key]
                elif step.get("action") == "bank_extraction":
                    # older logs: bank_extraction printed its counts as the first JSON object on stdout;
                    # the object can contain nested objects, so decode it rather than
                    # cutting at the first closing brace
                    tail = stats.get("stdout_tail", "")
                    start = tail.find("{")
                    try:
                        report = json.JSONDecoder().raw_decode(tail[start:])[0] if start >= 0 else {}
                    except json.JSONDecodeError:
                        report = {}
                    if not isinstance(report, dict):
                        report = {}
                    if not report and "already banked" in tail:
                        row["rebanked_no_change"] = True
                    for key in ("n_equations", "n_claims", "n_constants", "n_assumptions", "n_validity", "n_ungrounded"):
                        if isinstance(report.get(key), int):
                            banked[key] += report[key]
                            row[key] = row.get(key, 0) + report[key]
                elif step.get("action") == "record_parser_reviews":
                    for res in stats.get("results", []):
                        parser[res.get("state", "UNKNOWN")] += res.get("reviewed_records", 0)
                        row["parser_" + res.get("state", "UNKNOWN")] = row.get("parser_" + res.get("state", "UNKNOWN"), 0) + res.get("reviewed_records", 0)
                elif step.get("action") == "apply_verification":
                    for k in ("confirmed", "refuted", "unchanged"):
                        if isinstance(stats.get(k), int):
                            row["verdicts_" + k] = row.get("verdicts_" + k, 0) + stats[k]
                            banked["verdicts_" + k] += stats[k]
            row["mismatches"] = row.get("mismatches", 0) + len(log.get("verdict_state_mismatches", [])) \
                + len(log.get("bank_mismatches", []))
        per_packet.append(row)
    # Verdict outcomes from the stores (the authority), not from logs a re-run may have replaced.
    verified = Counter()
    for name in ("equations", "claims", "entities", "validity"):
        for rec in store.load(name):
            if rec.get("verified_at", "") >= SESSION_START and rec.get("verification"):
                verified[(rec.get("verifier_role") or "UNSIGNED", rec["verification"])] += 1
    journal = _load_jsonl(store.REPORT_DIR / "agentic_work_journal.jsonl")
    roles = Counter((r["role"], r["model"]) for r in journal)
    fixes = [r for r in journal if r["state"] == "FIXED_VERIFIED"]
    return {"by_state": dict(by_state), "banked": dict(banked), "parser": dict(parser), "packets": per_packet,
            "roles": roles, "fixes": fixes, "journal_rows": len(journal), "verified": verified}


def render(data: dict, fresh: dict | None, narrative: str) -> str:
    out = ["# Knowledge-graph completion — 2026-10-16 agentic continuation", "",
           "Generated by `knowledge/tools/session_completion_report.py` from the packet queue, the",
           "S3 integration logs and the append-only work journal; the narrative below the tables is",
           "the reviewer's (Opus F) and is kept verbatim.", ""]
    out += ["## Packets touched this session", "", "| State | Packets |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in sorted(data["by_state"].items())]
    out += ["", "## Who did what (work journal)", "", f"Journal rows: {data['journal_rows']}.", "",
            "| Role | Model | Entries |", "|---|---|---|"]
    out += [f"| {role} | {model} | {n} |" for (role, model), n in sorted(data["roles"].items())]
    out += ["", "## Knowledge-system repairs (journal state FIXED_VERIFIED)", "", "| Seq | When (UTC) | Repair |", "|---|---|---|"]
    out += [f"| {r['seq']} | {r['at'][:16]} | {r['task']} |" for r in data["fixes"]]
    b, p = data["banked"], data["parser"]
    out += ["", "## Totals over integrated packets", "",
            f"- banked objects: equations {b.get('n_equations', 0)}, claims {b.get('n_claims', 0)}, constants {b.get('n_constants', 0)}, "
            f"assumptions {b.get('n_assumptions', 0)}, validity statements {b.get('n_validity', 0)}; ungrounded {b.get('n_ungrounded', 0)}",
            "- independent verification recorded in the stores since session start, by verifier role: "
            + (", ".join(f"{role} {state} {n}" for (role, state), n in sorted(data["verified"].items())) or "none"),
            "  (current store state; a record re-verified later shows only its latest verdict)",
            f"- parser candidates reviewed against the page: " + ", ".join(f"{k} {v}" for k, v in sorted(p.items())),
            "- counts are the per-log record counts printed by bank_extraction; a packet that re-banks an identical record "
            "(the L-/R- linkage and correction packets) counts it again, so these totals can exceed the number of distinct new objects"]
    integrated_states = {"GRAPH_INTEGRATED", "TESTED", "FINAL_AUDIT_ELIGIBLE"}
    integrated = [r for r in data["packets"] if r["state"] in integrated_states]
    pending = [r for r in data["packets"] if r["state"] not in integrated_states]
    out += ["", "## Integrated packets", "", "| Packet | State | Document | Pages | eq | claims | consts | assum | valid | parser compared/rejected | verdicts confirmed/refuted/unchanged | mismatches |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in integrated:
        name = r["packet"] + (" (re-integration; objects already banked)" if r.get("rebanked_no_change") else "")
        out.append(f"| {name} | {r['state']} | {r['document']} | {r['pages'][0]}-{r['pages'][1]} | "
                   f"{r.get('n_equations', '')} | {r.get('n_claims', '')} | {r.get('n_constants', '')} | "
                   f"{r.get('n_assumptions', '')} | {r.get('n_validity', '')} | "
                   f"{r.get('parser_SOURCE_PAGE_COMPARED', '')}/{r.get('parser_REJECTED', '')} | "
                   f"{r.get('verdicts_confirmed', '')}/{r.get('verdicts_refuted', '')}/{r.get('verdicts_unchanged', '')} | {r.get('mismatches', '')} |")
    out += ["", "Counts are new records per integration log. A re-integrated packet's log counts only what was new in that run.",
            "", "## Packets touched this session but not integrated (resumable)", "",
            "| Packet | State | Document | Pages |", "|---|---|---|---|"]
    out += [f"| {r['packet']} | {r['state']} | {r['document']} | {r['pages'][0]}-{r['pages'][1]} |" for r in pending]
    if fresh:
        out += ["", "## Fresh-checkout proof", "",
                f"- HEAD `{fresh['head'][:12]}` alone: `{fresh['head_only_knowledge_tests']['summary']}`",
                f"- HEAD plus this working tree's changes: `{fresh['head_plus_changes_knowledge_tests']['summary']}`",
                f"- proof: **{fresh['proof']}**",
                f"- files a commit must add: {len(fresh['untracked_to_add'])} untracked, {len(fresh['modified_tracked'])} modified, "
                f"{len(fresh['deleted_tracked'])} deleted (full lists in the JSON next to this report)"]
    out += ["", narrative.strip(), ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--fresh-checkout", type=Path)
    ap.add_argument("--narrative", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    fresh = json.loads(ns.fresh_checkout.read_text()) if ns.fresh_checkout else None
    text = render(collect(ns.session), fresh, ns.narrative.read_text())
    ns.out.write_text(text)
    print(f"wrote {ns.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
