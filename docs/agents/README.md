# docs/agents

Instructions and prompts addressed to coding agents working in this repository.

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [AGENTS.md](AGENTS.md) | active | agent_prompt | cross-cutting | Agent guidance: read `knowledge/AGENTS.md`, use `knowledge/kg.sh`, treat the original PDF as the authority, follow the packet protocol. |
| [TEST_PROMPT.md](TEST_PROMPT.md) | active | agent_prompt | cross-cutting | Paste-in prompt for the standard 13 kN regression check on `examples/cli/test_13kn_sealevel_regen_allstl.args`, ending in a PASS/FAIL/CHANGE report. |

Notes:

- Until 2026-11-05 `AGENTS.md` and `TEST_PROMPT.md` sat at the repository root. Agent tools that look for a root
  `AGENTS.md` no longer find it there. The knowledge-system rules are in
  [../../knowledge/AGENTS.md](../../knowledge/AGENTS.md), which did not move.
- `AGENTS.md` (line 10) gives a `.venv/bin/python` test command; on 2026-11-05 and again on 2026-11-09 the checkout
  held `.venv-jax`, `.venv-kg` and `.venv-docling` but no `.venv`.
- Step 4 of `TEST_PROMPT.md` compares against `builds/tests/13kn_sl_baseline_summary.json` and initializes it when
  absent; no such file existed anywhere under `builds/` on 2026-11-05 or 2026-11-09.

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
