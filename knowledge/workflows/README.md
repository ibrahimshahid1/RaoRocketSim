# Literature review workflows

The current workflow is sequential and checkpointed: `kg.sh pending` →
`kg.sh --json packet TARGET` → inspect source/PDF → save one result JSONL line →
`tools/bank_extraction.py`. See [the guide](../README.md#resume-extraction-in-small-steps)
and [the result schema](extraction.schema.json).

Reviewing a source and independently checking an extracted equation are separate
jobs. The queue reports both; no command here starts agents or calls paid APIs.
Verbatim anchoring checks location only. Preserve uncertainties and never infer
absence from a bounded reading packet.

`recovered_results.jsonl` contains the ten already-banked Claude source reviews.
`results/` holds new completed work. Replaying a result is safe; banked result
hashes prevent double counting. Store snapshots before the continuation are
retained locally under `knowledge/.work/pre-continuation-2026-09-20/`.

[archive/](archive/README.md) contains the recovered original specification and
legacy Claude workflow scripts. Those scripts depend on Claude's Workflow
runtime and must not be run as ordinary Node.js programs. The old
`extraction_verify.js` is also a legacy-runtime reference. The Python verification
entry point is `kg.harvest_extraction.apply_verification`.
