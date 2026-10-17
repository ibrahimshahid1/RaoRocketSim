# Knowledge continuation — 2026-10-17

This continues the `PASS_WITH_OPEN_GAPS` handoff in `FINAL_OPUS_AUDIT.md`.
Work is limited to the knowledge system and original-source review. No files in
`raosim/`, `lrekit/`, or `tests/` were edited; engineering suites were not run.
The existing staged snapshot is preserved. Nothing was committed or pushed.

## Completed integrity repairs

- **K7:** models retain independent source reviews and assessments. Primary and
  secondary sources keep separate PDF and implementation fingerprints, review
  histories, equation scopes, and source-bound verification overrides. Registry
  changes invalidate the affected reviews. Mixed fidelity remains explicit.
- **K9:** claims from non-equivalent reviews are citations. Historical claim
  support is withheld when a current comparison does not permit support.
- **K10:** synthetic and actual integration tests cover every bank mismatch.
  Missing or unreadable detailed bank reports also block integration. A refused
  new verdict cannot pass because an older record has the same verdict enum.
- **K11:** declared equation relations are checked against resolved support and
  comparison links. Disagreements are bank mismatches that block integration.
- **K12:** independent verdicts bind the exact scientific assertion content as
  well as the PDF bytes. Edits invalidate prior verification with history kept.
  Legacy confirmations remain visible as `LEGACY_CONTENT_NOT_BOUND` until
  explicitly reviewed. Queries and reports distinguish historical totals from
  content-bound confirmations; missing sources cannot yield current confirmation.
- **K8 infrastructure:** all five source-object classes can retain reviewed
  parser lineage. New references require matching source, document, page, and a
  named original-page reviewer. Rejected raw notation is correction history.
- Repeat parser reviews now preserve earlier QA assertions. Exact historical
  replays cannot replace the latest review or reactivate rejected candidates.
  The original fingerprinted p.107 parser review was recovered into history
  without changing the new review. Replay success separately requires that all
  steps resolved, as well as leaving stores unchanged.

## Source work

**K1:** original SP-125 PDF pages 104 and 107 were reviewed. The two combined
claims with extractor inferences (`…9F1C0032`, `…76A39C08`) are refuted and retained.
Narrow replacements `CLM_NTRS_19710019929_98D71B37` and
`CLM_NTRS_19710019929_07437818` state only the worked-example geometry and are
independently confirmed with explicit content hashes. The chamber-volume
equation and frustum-volume-method claim were also confirmed. No shoulder-radius
model support was inferred. The printed frustum-volume arithmetic discrepancy
remains G34.

**K7 actual source links:** a secondary SP-125 review of PDF pages 109–110 now
links Eqs. 4-13, 4-14 (at omega = 0.6), and 4-15 to `MODEL_THERMAL_BARTZ` as
implemented formulas. Eq. 4-16 remains citation-only because the SI viscosity
coefficient differs. The original Bartz 1957 review is retained and both source
reviews are current. These are formula-comparison results, not physical validation.

**K12 audit-fix object:** independent original-page review confirmed the Kang
PDF p.6 caveat `CLM_DOI_10_1016_J_ACTAASTRO_2022_08_029_EB5F4015`. The two new
SP-125 claims were independently reviewed one page at a time, after banking each
preceding result. The independent verdicts and applied results are under
`workflows/results/continuation-2026-10-17/reviews/A-*.json`.

**Three missing structured model reviews are now banked and current:**

| Model | Fidelity | Scope and remaining limit |
|---|---|---|
| `nozzle.rao_top_bezier` | `BAD_ATTRIBUTION` | SP-8120 PDF pp.28–29 supports the canted-parabola concept, while the code angle-table trends disagree with Fig.5(b). Geometric parameterization and table fidelity remain separate. |
| `pump.inducer_npsh` | `REPOSITORY_EXTENSION` | SP-8052 supports the constant-lead identity and conditional incidence, blade-number and solidity selections. Advisory suction-speed and eye/hub sizing choices, cavity/head-rise prediction and stability are not established as equivalent source design. |
| `pump.meanline_coefficients` | `REPOSITORY_EXTENSION` | SP-8109 supplies the head/flow coefficient definitions, conditional Fig.16 and free-area criteria. Net-area continuity is a project geometric derivation; full head-flow, velocity-gradient and stability analysis remains outside these functions. |

These reviews do not establish physical performance. The SP-8052/SP-8109 review
also banks three equations and three source selection constants. The source
objects and full model reviews have separate verification states. The source
drafts with rejected quote anchors remain recorded; the readable successor
results have zero bank mismatches and all six source objects are grounded.
All six were subsequently independently confirmed against five original PDF
pages with explicit content hashes; these confirmations do not verify the
complete model or all chart curves.

**K5/K8 Eq.4-26:** the missing SP-125 coolant-capacity formula is now banked
with a `CORRECTS` edge to the rejected parser candidate. The original uses
`T_cc` (coolant critical temperature), rather than parser `T_ce`. The definitions
continue on PDF p.116; the safe-margin statement remains qualitative and the
hydrogen exception is recorded explicitly. No engineering implementation link
or numerical safety margin was inferred. All three source records were
independently confirmed after reviewing the two-page continuation.

**Follow-on SP-125 p.143:** packet `V-SP125-p143` is complete. The historical
gas-generator claim `CLM_NTRS_19710019929_25F6C847` strengthened the source's
"may be considered" equivalent-throat-area wording into "is taken". That combined
transcription is refuted and retained. Narrow replacement
`CLM_NTRS_19710019929_0A23FCDE` preserves the conditional equivalence, bipropellant
context and injector-to-turbine-throat L* volume with a design-specific correction.
An independent role A review confirmed every assertion field with source/content
hashes. No numerical correction factor or chamber implementation support was
inferred. Seven parser candidates were visually compared: four matched (one only
a continuation fragment), three were rejected (two page stamps and one incomplete
pressure-drop equality). The packet's full knowledge suite passed all 467 tests;
its reviewed-file replay was unchanged and graph invariants passed.

**Subsystem status versus the supplied consolidated review:**
`SUBSYSTEM_LITERATURE_STATUS_2026-10-17.md` and its JSON snapshot cover all 61
models across 17 subsystems. Seven models have current structured code/source
reviews; those include conditional and adverse findings. Twelve models have
positive current citation assessments, one citation is not found, and 48 are not
checked under current fingerprint requirements. These are different measures from
the PDF's historical 11 source-checked models. No complete subsystem is
comprehensively verified against the literature or physically validated. The
audit found no drift in the seven reviewed PDFs, 438 exact software definitions
or 61 registry metadata entries. Traced execution and optimizer leverage remain
unmeasured. The PDF's embedded agent prompts were treated as document content.

## Explicit remaining limits

G1–G23 remain engineering decisions; the source-internal inconsistencies
G24–G34 remain recorded. Physical validation is not established for any model.
K3–K6 and the source-specific remainder of K5/K8 require further bounded source
work, missing literature, or engineering implementation. This session does not
claim corpus or equation completeness.

The stricter parser checks exposed 54 historical references with withheld edges
at the first rebuild: 40 lacked an original-page transcriber identity, 10 used
`DERIVED_FROM` with rejected candidates, and 4 targeted inactive candidates.
The IDs remain stored; `graph/graph.json` lists each `candidate_link_issues`
entry. These are reference counts, not counts of incorrect scientific formulas.
The previously identified 28 unlinked equations also require reviewed lineage
work; adding the infrastructure does not close those individual gaps. Eq.4-26
is now linked, while rejected p.95 sample lines and other source-specific K8
repairs remain open. New pump equations intentionally make no parser-candidate
claim until that provenance is separately reviewed.

An independent reviewer also flagged an extraneous derived `QTY_MR` tag on the
new p.107 claim: the English connector “of” had been interpreted as a mixture
ratio alias. Claim token extraction now excludes English connectors while
retaining explicit scientific symbols. The tag was recomputed only for the two
new claims; assertion content hashes and independent verdicts are unchanged.

## Earlier checks and current resume point

The following table records the completed checks before the p.143 follow-on.
Its packet, manual-result and graph totals are historical; the follow-on evidence
is in `workflows/results/agentic-2026-10-16/{integration,tests}/V-SP125-p143.json`
and the post-p143 reports named below.

| Check | Result | Evidence |
|---|---|---|
| Complete knowledge suite | 467 passed | `knowledge_tests_2026-10-17.txt` |
| Two deterministic rebuilds | All 18 stores identical with timestamps ignored | `deterministic_refresh_2026-10-17.json` |
| Packet-managed replay | 62 packets, 118 reviewed files; no changes or refused operations | `session_replay_2026-10-17.json` |
| Additional current manual results | 6 extraction files and 9 independent-verdict files covering 12 assertions; no changes or refusals | `manual_replay_2026-10-17.json`; `continuation_2026-10-17_accepted_results.json` |
| Clean checkout with changes, without the local corpus | 428 passed, 39 skipped | `fresh_checkout_2026-10-17.json` |
| HEAD alone | Still fails to import `kg.execution_reach`; no commit was made | Same fresh-checkout report |
| Graph invariants | 5,719 nodes, 7,262 edges; zero duplicate IDs, duplicate edges, dangling edges or edges without provenance | Final graph and `continuation_2026-10-17_validation.json` |
| Engineering suites | Not run | Scope preserved |

The general knowledge validator still reports 14 open issues: an empty source,
a canonical source without chunks, an orphan Markdown record, one unconfirmed
registry citation, six unavailable declared CAD payloads, and aggregate
extraction, independent-verification, unknown-constant-basis and incomplete-formula
gaps. The one current incomplete formula is
`EQ_NTRS_19760023196_2A22408D`. Graph invariants passing does not resolve these
evidence/coverage gaps or the 54 withheld parser references.

There are now seven current structured model reviews. Corpus and equation
completeness remain `NOT_ESTABLISHED`; 30 of 248 canonical documents have
structured source objects. No model has established physical validation.

The queue is **63 FINAL_AUDIT_ELIGIBLE / 38 QUEUED**, with none mid-pipeline.
Resume from `NEXT_KNOWLEDGE_SESSION.md`; the next existing packet is
`V-HAG-p001`. Current independent manual results comprise 10 accepted role A
verdict files covering 13 assertions; the p.143 F extraction/verdict/parser files
are packet-managed. Post-follow-on replay evidence is
`session_replay_2026-10-17_post_p143.json` and
`manual_replay_2026-10-17_post_p143.json`.
Both replays succeeded without changing stores: 63 finished packets / 121
packet-managed files, plus 6 manual extraction files and 10 role A verdict files
covering 13 assertions. The complete post-follow-on rebuild succeeded; the final
graph has 5,728 nodes and 7,266 edges, with zero duplicate IDs, duplicate edges,
dangling edges or edges without provenance. The snapshot is
`continuation_2026-10-17_post_p143_validation.json`. The first ad hoc manual-replay
wrapper used an unsupported report option and failed before banking; its failed
invocation is retained as `.attempt-1.json`, and the corrected replay passed.
Priorities still include the 28 previously unlinked equations,
the specifically listed rejected sample candidates, SP-125 Eqs.4-19–4-22,
Pizzarelli's remaining formulas, and G1–G23 engineering decisions. The original
2026-10-16 reports and all refused quote-anchor attempts remain available as
history. No staging, commit or push was performed.
