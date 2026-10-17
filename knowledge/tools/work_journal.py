"""Append to (or show) the agentic work journal.

    .venv-jax/bin/python knowledge/tools/work_journal.py append --role S2 --model MODEL \
        --task "structure packet P" --state CANDIDATES_PREPARED --packet P \
        --input path --output path --note "..."
    .venv-jax/bin/python knowledge/tools/work_journal.py tail --n 20

Mechanical worker roles (S1-S3) cannot record authoritative scientific states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from kg import work_journal  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("append")
    p.add_argument("--role", required=True, choices=sorted(work_journal.ROLES))
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--state", required=True, choices=sorted(work_journal.STATES))
    p.add_argument("--packet")
    p.add_argument("--input", action="append", default=[])
    p.add_argument("--output", action="append", default=[])
    p.add_argument("--note", default="")
    p = sub.add_parser("tail")
    p.add_argument("--n", type=int, default=20)
    ns = ap.parse_args(argv)
    if ns.cmd == "tail":
        for row in work_journal.read()[-ns.n:]:
            print(json.dumps(row, ensure_ascii=False))
        return 0
    try:
        entry = work_journal.append(role=ns.role, model=ns.model, task=ns.task, state=ns.state,
                                    packet_id=ns.packet, inputs=ns.input, outputs=ns.output, note=ns.note)
    except (ValueError, PermissionError) as exc:
        print(json.dumps({"state": "REFUSED", "error": str(exc)}))
        return 2
    print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
