"""Command-line access to the persistent source-packet state machine.

    .venv-jax/bin/python knowledge/tools/packet_queue.py summary
    .venv-jax/bin/python knowledge/tools/packet_queue.py list --state QUEUED
    .venv-jax/bin/python knowledge/tools/packet_queue.py show PACKET_ID
    .venv-jax/bin/python knowledge/tools/packet_queue.py next QUEUED DOCLING_RUNNING --role S1 --model MODEL
    .venv-jax/bin/python knowledge/tools/packet_queue.py transition PACKET_ID RAW_CAPTURED --role S1 \
        --model MODEL --run-id ENR_... --note "..."

Every write is atomic and journaled with the acting role and model. This tool
never starts agents, never runs Docling and never edits scientific stores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from kg import packet_queue as pq  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    p = sub.add_parser("list")
    p.add_argument("--state")
    p.add_argument("--kind")
    p = sub.add_parser("show")
    p.add_argument("packet_id")
    p = sub.add_parser("add")
    p.add_argument("packet_id")
    p.add_argument("--kind", required=True, choices=sorted(pq.KINDS))
    p.add_argument("--document-id", required=True)
    p.add_argument("--pages", required=True, help="FIRST:LAST original PDF pages")
    p.add_argument("--target", action="append", default=[])
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--reason", required=True)
    p.add_argument("--depends-on", action="append", default=[])
    p.add_argument("--role", required=True)
    p.add_argument("--model", required=True)
    p = sub.add_parser("transition")
    p.add_argument("packet_id")
    p.add_argument("to_state", choices=pq.STATES)
    p.add_argument("--role", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--evidence", action="append", default=[], help="file whose sha256 is recorded")
    p.add_argument("--run-id", action="append", default=[], help="Docling GeneratedDocument run ID")
    p.add_argument("--verdict", action="append", default=[], help="reviewed verdict/result file")
    p.add_argument("--test-result", choices=("pass", "fail"))
    p = sub.add_parser("reprioritize")
    p.add_argument("packet_id")
    p.add_argument("priority", type=int)
    p.add_argument("--role", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--note", required=True)
    p = sub.add_parser("next")
    p.add_argument("from_state", choices=pq.STATES)
    p.add_argument("to_state", choices=pq.STATES)
    p.add_argument("--role", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--kind")
    p.add_argument("--min-priority", type=int)
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "summary":
            out = pq.summary()
        elif ns.cmd == "list":
            out = [{k: p[k] for k in ("id", "kind", "state", "document_id", "pages", "priority", "attempts")}
                   for p in sorted(pq.load()["packets"].values(), key=lambda p: (-p["priority"], p["id"]))
                   if (not ns.state or p["state"] == ns.state) and (not ns.kind or p["kind"] == ns.kind)]
        elif ns.cmd == "show":
            out = pq.load()["packets"].get(ns.packet_id) or {"state": "UNKNOWN_PACKET"}
        elif ns.cmd == "add":
            first, last = map(int, ns.pages.split(":"))
            out = pq.add(ns.packet_id, kind=ns.kind, document_id=ns.document_id, pages=(first, last),
                         targets=ns.target, priority=ns.priority, reason=ns.reason, role=ns.role,
                         model=ns.model, dependencies=ns.depends_on)
        elif ns.cmd == "transition":
            out = pq.transition(ns.packet_id, ns.to_state, role=ns.role, model=ns.model, note=ns.note,
                                evidence_paths=ns.evidence, run_ids=ns.run_id, verdict_paths=ns.verdict,
                                test_result=ns.test_result)
        elif ns.cmd == "reprioritize":
            out = pq.reprioritize(ns.packet_id, ns.priority, role=ns.role, model=ns.model, note=ns.note)
        else:
            out = pq.claim_next(ns.from_state, ns.to_state, role=ns.role, model=ns.model, kind=ns.kind,
                                min_priority=ns.min_priority)
            out = out or {"state": "NO_ELIGIBLE_PACKET"}
    except (ValueError, PermissionError) as exc:
        print(json.dumps({"state": "REFUSED", "error": str(exc)}))
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
