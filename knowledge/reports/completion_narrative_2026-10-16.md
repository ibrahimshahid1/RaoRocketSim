## Narrative (Opus F)

### Authority and constraints
This session followed `LREKit_Knowledge_Graph_Agentic_Continuation_2026-10-16.md` as amended by the owner. Where the amendment conflicts, the amendment wins.

- HEAD 550d5c2's MDO engineering work is intended behaviour and was not repaired.
- `raosim/`, `lrekit/` and `tests/` were not edited. Every code-versus-source discrepancy is recorded in `REMAINING_SCIENTIFIC_GAPS.md`.
- Nothing was downloaded.
- Docling ran as one process at a time, and no more than two Sonnet workers were active while Docling or pytest ran.
- Only the knowledge suite and targeted regressions were run; the engineering suites were not.
- Nothing is committed. The owner commits.

### Agent mechanics (recorded as the amendment required)
- **Batch 1:** fresh per-packet workers through the Workflow tool (run `wf_af06278b-de0`). An interrupted Docling packet from that run (`V-SP125-p104`) was requeued through the negative path, with a note.
- **Batch 2 onward:** persistent role agents launched with the Agent tool and continued with SendMessage:
  - S1, Docling orchestration (Sonnet);
  - S2, candidate structuring (Sonnet);
  - S3, graph integration and QA (Sonnet).
  Subagents cannot message each other, so F routed every packet.
- **Opus F** (main session) did all source reviews, verdicts, code fixes and coordination.
- **Opus A** was launched with the Agent tool (model opus) after the final integration, rebuild and test pass, for an independent, read-only audit. The owner stopped it once and resumed the same agent. Its verdict and its recheck are recorded in `FINAL_OPUS_AUDIT.md`. This report quotes them only as summarized in the section "Independent audit and repairs" below.
- Each role and model is recorded per entry in `agentic_work_journal.jsonl`; the table above summarizes it. The packet state machine refuses role-inappropriate transitions: S-roles cannot open or decide a source review, and only F verifies. The journal refuses ACCEPTED_SCIENCE from worker roles.

### What changed in the knowledge system (beyond the table)
1. **Fresh checkout.** The `.gitignore` line `knowledge/` (added in 550d5c2) hid 9 imported `kg` modules, test modules, Docling tools, stores and review journals. A clean export of HEAD fails collection with `ModuleNotFoundError: kg.execution_reach`. With the line replaced, HEAD plus the working tree passes the corpus-free suite; `knowledge/tools/fresh_checkout_check.py` proves this repeatably and lists the exact files to add.
2. **Concurrency safety.**
   - An inter-process writer lock serializes every store writer (`kg/locks.py`, reentrant).
   - Atomic private temp files replace the shared `.tmp` name.
   - A machine-wide single-Docling lock refuses a second process with exit 75. That refusal is not recorded as page-attempt history.
3. **Packet protocol.**
   - `kg/packet_queue.py` is a role-gated state machine QUEUED → … → FINAL_AUDIT_ELIGIBLE. It binds the reviewed-file SHA-256 at VERIFIED and re-checks it at GRAPH_INTEGRATED.
   - Requeues need a reason, and attempt history is kept.
   - `kg/work_journal.py` is the append-only role/model journal.
   - `tools/agentic_packet.py` holds the deterministic S1/S2/S3 operations.
4. **Review provenance.**
   - The transcriber identity is stored, and that transcriber cannot verify its own object.
   - Reviewer identity requires a model name.
   - Independent verification now covers constants, assumptions and source-stated validity, not just equations and claims.
   - Structured Section 17 model reviews are bound to source and AST implementation fingerprints, with a CURRENT / IMPLEMENTATION_CHANGED / STALE_SOURCE state.
5. **Damaged-formula handling.** Self-declared "OCR garbled" low-confidence forms are now INCOMPLETE_TRANSCRIPTION, and a page-verified correction supersedes them while keeping history. This session corrected SP-125 Eqs. 1-14, 4-6 and 4-7 that way.
6. **Repairs reported by workers and fixed with regression tests:**
   - S2 bundle captions, table grids and model targets;
   - U+FFFD and glyph-run flags on parser text;
   - the pending-verification metric undercount;
   - the S3 verdict-state check, which ignored entities and validity and falsely BLOCKED a packet;
   - retrieval returning a refuted assumption.
7. **Standards as sources** (§10.4 item 8): 8 StandardSource records, HELD or BLOCKED_ON_STANDARD.
8. **Completion and audit tooling:**
   - `kg/completion.py` writes CORPUS_COMPLETENESS / EQUATION_FIDELITY_AUDIT / MODEL_SOURCE_AUDIT;
   - `tools/session_completion_report.py` writes this report;
   - `tools/trace_flow.py` traces an object end to end.

### Scientific work (all against original PDF pages)
- **P1, Rao/TOP angle charts.**
  - SP-125 Fig. 4-14 (p.100) and SP-8120 Fig. 5(b) (p.29) were digitized. The digitization was validated at SP-125's worked point (p.107: ε 14, L_f 0.8 → θn 27.4°, θe 9.8°).
  - `raosim/nozzle_geometry.py` tables are BAD_ATTRIBUTION: θe rises with ε where both charts fall, and θn is high for L_f < 100 %. See gaps G1–G3.
- **P2, electric-drive technology sets.**
  - Every cited value was confirmed on the page.
  - Complete battery, motor and inverter sets with stated power bases: Lee 2021, Kwak 2018 and Pavlov Rachov 2013. Rachov's motor density is on shaft output (Eq. 35) and its inverter density on output (Eq. 37).
  - Battery only: Casalino 2019, Solda-Lentini 2008.
  - Park 2023 has motor and inverter values with unstated basis. SP-8107 covers turbopump assemblies only.
  - The default set `repository_screening_v1` lies outside held evidence (G4).
- **P3, constraint provenance.**
  - Mirzamoghadam gives 755 K, 839 K (material-dependent) and 700 K, never 800 K. The code labels its 800 K a repository rule, which is consistent.
  - The BF band is supported to 0.7 for single-row tips. Values above that come from two-row pintles, whose BF definition differs (Hwang Eq. 3) and where tip damage occurred at 86 %.
  - Son's pintle rows are supported as criteria. Son's printed Eq. (2) has a typo, which the paper's own Table 2 exposes; the code follows the consistent form.
- **P4.**
  - Pizzarelli 2011 Eqs. 1-21, 36-38 and 40-45 are banked, with validity: AR 1–16 CFD comparison, weak at low AR; Vulcain AR 4.5–8.5.
  - Nasuti & Pizzarelli Eq. 9, K = 0.187.
  - SP-8087 film cooling: sec. 3.3 2× capacity, Table XI procedure, orifice minimum.
  - Throttling requirements: Gromski, Tokudome and Hu.
- **P5, Section 17 reviews.**
  - `injector.son2017_movable_continuous_gap`: EQUIVALENT.
  - `pump.electric_drive`: CONDITIONALLY_EQUIVALENT (smooth max in the objective).
  - `thermal.bartz`: EQUIVALENT for h_g, σ and Pr. The viscosity function has a conversion error of +0.745 % (exact coefficient 1.184081e-7); the code also has two wrong SP-125 locators and does not enforce the validity limits.
  - `thermal.sieder_tate`: DIFFERENT_FORMULA. SP-125 Eq. 4-23 has Pr^0.4 and a coolant-specific C1 (RP-1 0.0214), against the code's 0.027 Pr^(1/3). The code's h_c is 7–26 % higher.
  - `nozzle.rao_top_bezier`: angle tables classified in P1, no structured review banked.
  - `pump.inducer_npsh`, `pump.meanline_coefficients`: not reviewed.
  - Supporting source packets were banked late in the session:
    - SP-125's own presentation of the Bartz workflow, Eqs. 4-10 to 4-18 with the A-1 sample values (PDF pp.109-112);
    - the RP-1 coolant form of Eq. 4-23 with sample Eqs. (a)-(c) (PDF p.119).
    They are document-driven on purpose. A model-targeted result would replace the comparison links of the Bartz and Sieder–Tate reviews, because a model holds comparison links from one document only (gap K7).
  - The printed expanded Eq. (b) on p.119 omits π, and SP-125's own tube arithmetic on p.120 does not reproduce (gap G33).
- **Self-corrections before banking.** F caught and fixed three of its own transcription errors on the page:
  - Solda subscripts "eo" → "ep";
  - SP-8107 J-2/SSME row labels;
  - Pizzarelli partial-derivative notation.
  The journal records them.

### Previously completed (Codex, 2026-10-14/15) work: kept, not redone
Present at the start of the session, hidden from git by the `.gitignore` line:
- `kg/execution_reach.py`, `harvest_artifacts.py`, `harvest_parameters.py`, `record_identity.py`, `variable_links.py`, `constant_basis.py`, `coverage.py`, `enrichment.py`, `implementation_freshness.py` (F added `definition_fingerprint`);
- the Docling tools (F added the heavy-job lock and cache reuse);
- the White derivation links and supersession history;
- the September and October review journals.
These were re-verified through the full knowledge suite, the fresh-checkout proof and S3's exact snapshot diffs. They were not rewritten.

### Independent audit and repairs (Opus A, then F)
- **First verdict: FAIL_WITH_FINDINGS** (`FINAL_OPUS_AUDIT.md`).
  - A independently confirmed 145 of 145 sampled F transcriptions against the original pages, with 0 refuted. F applied that verdict file with `apply_verification`.
  - A reproduced all four Section 17 reviews from code and pages.
  - A found one major knowledge-system defect and ten minor ones.
- **A-01 (major).** Review support labels were short tags such as "Eq. (1)", while stored labels are long, so they silently matched nothing. As a result no reviewed model IMPLEMENTED its source equations, and no API read the structured review.
  - **Fixed in `kg/harvest_extraction.py`.** Labels now resolve by exact label, equation id or printed number, including ranges. An unmatched or ambiguous label is reported, and S3 blocks on it. Only EXACT/EQUIVALENT/CONDITIONALLY_EQUIVALENT reviews create IMPLEMENTS edges; every other class only cites.
  - **Fixed in `kg/graph.py` and `kg/api.py`.** Model nodes, `get_model`/`find_models`, `find_implementation`, `find_scientific_basis` and `retrieve_for_task` now show the review fidelity and its current state. A non-current or non-equivalent review raises a task warning.
  - **Re-banked** through correction packets with explicit per-equation links: R-SON2017-p003, R-BARTZ-p001-005, R-SP125-p115, R2-LEE-p006-008.
- **A-02 to A-11:**
  - whitespace-insensitive duplicate detection, and reviewed retirement of the Lee Eq. (29) duplicate with its history kept;
  - recorded carry-forward decisions on supersession, and the SP-125 corrections now carrying their page-verified assumptions and ranges;
  - refusal of unsigned and worker-role verdicts;
  - atomic writes that preserve file mode, with the 21 narrowed stores reset to 0644;
  - a store-writer lock serialized across threads, with process and thread tests;
  - dangling edges now blocking S3;
  - an exact report-tool file reference;
  - no store rewrite on a no-op verdict or parser replay;
  - attempt-numbered evidence that never overwrites;
  - verdict totals counted from the stores.
  The P2-LEE first BLOCKED log and bundle were overwritten earlier in the session and cannot be recovered; this is recorded, not repaired.
- **Notes taken up:**
  - N-01: resume commands now use `.venv-jax`.
  - N-02: per-symbol Bartz findings and `equation_ids`.
  - N-03 and N-11: journal rows.
  - N-04: tests for the REFUTED supersession branch and refuted validity.
  - N-05: G5 wording.
  - N-06: structured bank report.
  - N-07: new gap G34.
  - N-08: banked Kang caveat.
  - N-09: committed `tools/session_replay_check.py`.
  - N-10: documented that roles are self-declared.
  - N-12: regenerated reports.
  - N-13: the tracked `validation_20*.json` files are left as they are.
- Regression tests for every finding are in `knowledge/tests/test_kg_audit_fixes.py`, plus two extended tests in `test_kg_review_provenance.py`.

### Not done (honest scope)
- Corpus-wide coverage was not attempted. Most documents still have no extracted objects (`CORPUS_COMPLETENESS.json`).
- Three P5 models have no structured review:
  - `nozzle.rao_top_bezier`, classified through P1 only;
  - `pump.inducer_npsh`;
  - `pump.meanline_coefficients`.
- 28 equations F banked early in the session still have no Docling-candidate link (listed in `NEXT_KNOWLEDGE_SESSION.md`).
- No new machine-readable validity bounds were generated.
- No Experiment/Measurement objects were created.
- Cross-source CONTRADICTS edges were not added; the comparisons live in model reviews and gaps.
- The 14 low-confidence conversions were not revisited.
- `V-SP125-p104`, `p107` and `p143`, and the remaining queued batch-1 packets, are left resumable.
