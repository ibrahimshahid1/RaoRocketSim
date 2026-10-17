# Agentic roles for the 2026-10-16 knowledge continuation

Repository: `/Users/ibrahimshahid/Documents/Github/RaoRocketSim`. Python for every
command below: `.venv-jax/bin/python` (the `.venv` named elsewhere is absent).
Docling runs only inside `knowledge/tools/docling_batch.py`, which uses
`.venv-docling` and a machine-wide lock: **at most one Docling process exists at
any time**. Never start Docling any other way.

Every state change goes through `knowledge/tools/packet_queue.py` or
`knowledge/tools/agentic_packet.py`; both journal the acting role and model to
`knowledge/reports/agentic_work_journal.jsonl`. Pass `--model` exactly as your
own runtime identifies you (for example `claude-sonnet-5-5`); never claim
another model or role.

Packet path: `QUEUED -> DOCLING_RUNNING -> RAW_CAPTURED -> CANDIDATES_PREPARED ->
OPUS_SOURCE_REVIEW -> VERIFIED | REJECTED | BLOCKED -> GRAPH_INTEGRATED -> TESTED
-> FINAL_AUDIT_ELIGIBLE`. The queue refuses out-of-order or wrong-role moves.

## Common prohibitions (all Sonnet roles)

- Do not edit any file under `knowledge/kg/`, `knowledge/tools/`, `knowledge/tests/`,
  `raosim/`, `lrekit/`, any store `knowledge/*.json`, or any reviewed verdict/result file.
- Do not decide whether a formula, claim, symbol or source statement is correct,
  equivalent, supported or physically valid. Do not write verdicts.
- Do not run `knowledge/kg.sh rebuild`, `sync --convert`, git commands that change
  state, or any Docling command other than through `agentic_packet.py s1-run`.
- If a command fails twice in the same way, or output looks systematically
  damaged, stop and report the exact command and output. Do not work around it.
- Keep a short factual report: commands run, packet IDs, states, failures.

## S1 - Docling orchestration and corpus processing

Loop until no eligible packet remains or your packet budget is reached:

```bash
.venv-jax/bin/python knowledge/tools/packet_queue.py next QUEUED DOCLING_RUNNING --role S1 --model <you>
.venv-jax/bin/python knowledge/tools/agentic_packet.py s1-run <PACKET> --model <you>
```

Repeat `s1-run` for the same packet while it reports `DOCLING_RUNNING_PARTIAL`
(one page per call, about 1-8 minutes each; use a Bash timeout of 600000 ms).
`RAW_CAPTURED`, `QUEUED` (retry/busy) and `BLOCKED` end that packet for you.
Produce `RAW_EXTRACTION_READY` artifacts only. Before each new packet, check
memory with `vm_stat | head -5`; if free+inactive pages are below 2 GB worth,
pause and report instead of starting Docling.

## S2 - Candidate structuring and source metadata

For each packet in `RAW_CAPTURED` (list with `packet_queue.py list --state RAW_CAPTURED`):

```bash
.venv-jax/bin/python knowledge/tools/agentic_packet.py s2-prepare <PACKET> --model <you>
```

Then open the bundle it names and report mechanical observations only: empty or
flagged candidates, unbalanced brackets, unresolved quantity mappings, quote
anchors not located, duplicate forms, missing targets. Never infer a missing
formula or upgrade a candidate.

## S3 - Verified graph integration and deterministic QA

For each packet in `VERIFIED` or `REJECTED`:

```bash
.venv-jax/bin/python knowledge/tools/agentic_packet.py s3-integrate <PACKET> --model <you>
.venv-jax/bin/python knowledge/tools/agentic_packet.py s3-test <PACKET> --model <you>
```

Integration banks only the exact reviewer-signed files whose fingerprints the
queue recorded. A `BLOCKED` result is a handoff to F: report it with the log
path. Do not change graph semantics, stores or tests to make checks pass.

## Opus F (main session) owns

Scheduling packets, opening `OPUS_SOURCE_REVIEW`, original-page review and
verdict/result files (signed `"reviewer": {"role": "F", "model": ...}`), code
repairs and regression tests, `FINAL_AUDIT_ELIGIBLE`, and full rebuilds.

## Limits of the role controls (recorded after the 2026-10-16 audit, note N-10)

Roles are self-declared: `--role` and `reviewer.role` are what the caller states. They are not authenticated. The state machine, the journal and `apply_verification` refuse role-inappropriate actions *as declared*:
- S-roles cannot open or decide a source review;
- worker roles cannot journal scientific acceptance;
- S1–S3 verdicts and unsigned verdicts on transcribed records are refused;
- a transcriber cannot verify itself.

These rules are an honour system for cooperating agents, backed by the append-only journal and by an independent audit (Opus A) that re-checks attribution against the record. They are not a security boundary against a dishonest caller.

Verdicts from the independent auditor (role A) are applied by F directly with `apply_verification` and journaled, because packet review files accept only reviewer role F.
