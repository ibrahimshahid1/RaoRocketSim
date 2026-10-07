# Knowledge-system implementation audit — 2026-09-20

The local scientific-memory foundation was already implemented. Literature
extraction and independent source verification were incomplete. This continuation
recovered the prior work, repaired maintenance and retrieval failures, and resumed
one bounded source review. The full original specification is **not complete**.

## What Claude actually completed

The working tree contained an uncommitted `knowledge/` implementation and a
`.gitignore` change. Its source files, JSON stores, vector artifacts, and saved
Claude workflow journals were inspected. The original 35 knowledge tests passed.

| Component | Observed state before continuation |
|---|---|
| PDF manifest | 252 document records; 280 PDFs on disk; four alternate document copies |
| Ingestion | 248 canonical documents, 19,565 searchable chunks |
| Retrieval | SQLite FTS5 plus a 256-dimensional local LSA index; Python API and CLI |
| Knowledge graph | 4,493 nodes and 5,632 edges in the latest graph, with two dropped dangling links |
| Registry | 61 models; only 10 had source-support checks |
| Literature objects | 28 equations, 174 claims, 33 source validity statements, 44 constants, 28 assumptions |
| Source coverage | Extracted objects from six documents; remaining PDFs were searchable text, not completed scientific extraction |
| Repository traceability | 1,479 software-link records, 1,061 test records, 98 capability records, 31 requirements/gates |
| Independent verification | No recorded confirmed/refuted verdicts for the 28 equations or 174 claims |

The previous report was stale: it showed 4,183 graph nodes and zero equation
variables even though later stores contained 4,493 nodes and 100 equation variables.
The 35 acceptance tests established basic retrieval behavior, not scientific
completeness or freedom from maintenance regressions.

### Recovery evidence

Project-specific Claude session `9bc3be5e-dcba-4772-b4d3-2db545a28995` contained
workflow journals. The original extraction script launched an agent per registry
model and per document window; a later script grouped work by subsystem.

- `wf_a21c0f54-6ed`: 179 start events, 160 failure events, ten result events.
- `wf_5a5cac1c-3c9`: 88 start events, 83 failure events, zero result events.
- `wf_3875d965-4cf`: 39 start events, 39 failure events, zero result events.
- `wf_f57ff58c-ba7`: 11 start events, two results consisting of an empty/unknown result and duplicate D2 material.

These are journal event counts, not unique papers, unique agents, or a token-usage
accounting. The ten useful completed reviews were already banked. No additional
completed batch was hidden in those journals. The recovered result-only journal,
legacy scripts, and original specification are retained under `workflows/`.
The old `.work` files were task inputs, not completed extraction outputs.

## Work completed in this continuation

- Unified `kg.sh rebuild` with the complete pipeline. It now refreshes the
  repository harvest, requirements, validation and reports as well as the index.
- Preserved literature-owned model metadata, validity records, verifier findings,
  and links during registry refresh and extraction replay. Changes to a model's
  cited source or relation invalidate the old source review.
- Made extraction banking resumable with result fingerprints, unique-object
  counts, visible corrupt-journal failures and source-hash checks.
- Added PDF/Markdown freshness tracking. Changed evidence remains stored as
  `STALE_SOURCE` and cannot silently support current models. Chunk deletion or
  source replacement updates the derived index.
- Replaced token-overlap grounding with contiguous normalized anchors. Validation
  now rechecks stored anchors against the indexed source text.
- Made missing, empty, tiny, partial and inconsistent vector artifacts explicit;
  lexical retrieval remains available. Vector corpus fingerprints detect stale
  derived artifacts. Fixed overwritten subsystem hints and science-to-code traversal.
- Exposed source-support state, uncertainty and verification status in task
  context; excluded stale/refuted evidence from active scientific support.
- Made graph refusal IDs deterministic and included registry-cited repository
  notes, resolving the two dangling graph links.
- Added a persistent review queue and bounded reading packets. They never start
  agents or call paid APIs. Added root agent guidance and a user-facing guide.
- Visually inspected Rao 1958 PDF pages 2–6 and read the Rao text in its six-page
  Markdown. Banked nine equations, five claims, one validity statement and two
  assumptions. All 17 anchors grounded without page shifts. The source supports
  the registered variational method; this does not validate the numerical solver.

The Rao record lives in `workflows/results/rao1958-2026-09-20.jsonl`. Its equation
forms were transcribed from the original PDF because the Markdown mathematics
is damaged. Independent verification remains a separate pending task.

## Current coverage and remaining work

There are now 37 equations, 179 claims, 34 source validity statements and 74
literature entities from seven documents. Eleven of 61 registry models have
source-support checks: two supported, eight partially supported, one not found
in the cited local source. Fifty remain unchecked: 42 have a local PDF route and
eight require repository-basis review or source routing. There are 216 equations
and claims awaiting independent verification. These counts are also regenerated
by `kg.sh pending`; that queue is the current authority after future updates.

The original specification also requests features that remain incomplete:

1. Finish model source reviews and independent equation/claim verification,
   starting with load-bearing models. Review one source at a time and bank each
   result. Model-source support, extraction coverage, and independent verification
   are separate completion criteria.
2. Extend extraction beyond the seven reviewed documents. The remaining 241
   canonical documents have no extracted scientific objects; searchable passages
   alone are not a completed knowledge graph. Cross-source disagreement and
   equivalence relations also remain sparse/unpopulated.
3. Implement native, configuration-bound artifact, measurement, comparison and
   versioned calibration records. Their ontology names and repository references
   exist, but they are not a complete ingestion/retrieval layer. Use actual
   configuration fingerprints, raw-data hashes and uncertainty from LREKit's
   existing contracts; do not invent measurements or qualification evidence.
4. Add reliable structured table/figure records with page provenance. Current
   chunks preserve table text, but do not constitute a complete typed table and
   figure store. PDF mathematics and OCR need selective visual verification.
5. Improve semantic relevance assessment and entity resolution. LSA is the local
   implementation; no neural embedding backend or remote retrieval service was
   built. Many equation symbols and empirical coefficients still need reviewed
   meanings/bases. UNKNOWN_BASIS values must remain explicitly unknown.

Outstanding validation findings are two Markdown files without conversion
manifest rows (one is a human-authored catalog) and the already recorded
unconfirmed CAD-source attribution. These were retained as findings, not erased
or promoted to successful validation. Conversion quality remains predominantly
`needs_review`, including 14 flagged low-confidence document records.

## Validation and preservation

Validation result: **97 tests passed** (35 original corpus tests and 62 added
regression cases). Run `.venv/bin/python -m pytest knowledge/tests -q`. The suite covers local-corpus
acceptance plus source-free regressions for rebuild preservation, corpus changes,
strict grounding, replay, source checks, retrieval fallback, uncertainty, graph
traversal and bounded queue behavior. Full rebuilds succeeded against the
real local corpus. The repeat rebuild skipped the unchanged vector index; all
original scientific-object IDs and ten prior source-support reviews survived.
The resulting graph has 4,514 nodes, 5,685 edges and zero dangling links. No propulsion solver or engineering constants
were changed; the broader simulation suite was outside this maintenance change.

The original stores were snapshotted under
`.work/pre-continuation-2026-09-20/` before rebuilding. Work remains in the working
tree; no commit, push, or external publication was made.
