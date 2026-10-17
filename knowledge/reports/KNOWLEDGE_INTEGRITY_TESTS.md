# Knowledge-system integrity tests: 2026-10-16 agentic continuation

Written by Opus F from commands run in this session, including the repair round after Opus A's audit. Times are UTC.

"Knowledge suite" means `.venv-jax/bin/python -m pytest knowledge/tests -q -p no:cacheprovider`. The owner's engineering suites (`tests/test_mdo_*.py` and the rest of `tests/`) were not run, as instructed.

## 1. Test-suite results

| When | What | Result |
|---|---|---|
| 2026-10-16 18:39 (journal seq 1) | knowledge suite at HEAD 550d5c2, before any change | 287 passed |
| 2026-10-16 18:39 (journal seq 1-2) | `git archive HEAD` export, knowledge suite | import failure `ModuleNotFoundError: kg.execution_reach` (the `.gitignore` line `knowledge/` hid imported modules) |
| 18:52 to 03:31, every integrated packet | `agentic_packet.py s3-test <packet>` (S3), 60 runs | all exit 0. 43 runs at 316 passed, 7 at 331 (the audit-correction packets), 1 at 314, 6 at 312, 3 at 310; the count rose as F added regression tests. Logs: `knowledge/workflows/results/agentic-2026-10-16/tests/<packet>.json` |
| 2026-10-17 (Opus A audit) | knowledge suite, run by A | 316 passed (before the repair round) |
| 2026-10-17 03:32 | knowledge suite, final, after the final rebuild | **331 passed in 8.97 s** |
| 2026-10-17 03:34 | `knowledge/tools/fresh_checkout_check.py --output knowledge/reports/fresh_checkout_2026-10-16.json` | proof **PASS**: HEAD alone `2 errors in 0.26s` (collection); HEAD plus this working tree `292 passed, 39 skipped`. The 39 skips are corpus-dependent tests, which skip when the local PDFs and vector index are absent. |

The suite grew from 287 to 331 test cases (+44):
- 28 in four new files from the first part of the session: `test_kg_agentic_controls.py` (8), `test_kg_agentic_packet_ops.py` (8), `test_kg_review_provenance.py` (9), `test_kg_repository_tracking.py` (3);
- 1 in `test_kg_retrieval.py`;
- 15 in `test_kg_audit_fixes.py` (audit findings A-01 to A-11 and N-04). Two existing tests in `test_kg_review_provenance.py` were extended for A-03 and A-04.

## 2. Rebuild, determinism, replay and fingerprints

1. **Rebuilds 1 and 2** (01:16 and 01:17, before the audit): identical apart from timestamps; the only difference was `manifest.json` `registered_at`. Rebuild 1 changed no scientific record, timestamps ignored. Its additions were derived only: 72 build-artifact provenance rows, 2 literature-request rows and 2 locator verifications.
2. **Final rebuild** (03:32:25, exit 0, after the audit-correction packets). It rewrote `CORPUS_COMPLETENESS.json`, `EQUATION_FIDELITY_AUDIT.json` and `MODEL_SOURCE_AUDIT.json`.
3. **Whole-session replay**, committed as `knowledge/tools/session_replay_check.py`; output in `knowledge/reports/session_replay_2026-10-16.json`. It covered 60 FINAL_AUDIT_ELIGIBLE packets and 112 reviewed files:
   - every reviewed-file SHA-256 equals the hash bound at VERIFIED;
   - 60 extraction files: "already banked; no changes";
   - 48 verdicts: unchanged;
   - 373 parser reviews: re-recorded, with no store write;
   - 1 duplicate retirement: unchanged;
   - stores changed, timestamps ignored: **none** (`no_op: true`).
4. **Opus A's verdict file replayed** (`reviews/A-audit-verdicts.json`): 144 unchanged and 1 retired_or_noncurrent (the retired Lee duplicate). The store files were not rewritten; mtimes are unchanged (audit A-10).
5. **Completion-report fingerprints.** All 11 recorded input SHA-256s in each of the three completion JSONs equal the current files, checked after the replay (audit A-10).
6. **Final graph** (`knowledge/graph/graph.json`): 5695 nodes and 7267 edges. Duplicate node ids 0, duplicate edges 0, dangling edges 0, edges without provenance 0.
7. **Per-packet S3 checks**, one log per packet in `integration/`, `tests/` and `bank_reports/`. Each log records:
   - before and after snapshots, with integrity counts including dangling edges;
   - replay idempotency;
   - store and graph diffs between integration and test;
   - structured bank reports with no mismatch keys;
   - for model packets, field-level model baselines.

## 3. Negative paths: tests that exist and pass

| Negative path | Test(s), all passing in the final run |
|---|---|
| Rejected OCR or parser candidates | `test_kg_enrichment.py::test_rejected_parser_run_is_history_not_active_evidence`; `test_kg_docling_attempts.py::test_explicit_new_attempt_does_not_overwrite_or_inherit_rejected_review`; `test_kg_agentic_packet_ops.py::test_s2_flags_a_damaged_text_layer_behind_a_clean_transcription` |
| Stale or changed source | `test_kg_agentic_controls.py::test_changed_source_blocks_further_progress`; `test_kg_extraction.py::test_changed_source_does_not_reuse_old_verification`; `test_kg_retrieval.py::test_stale_records_are_excluded_but_remain_inspectable`; `test_kg_citation_support.py::test_direct_model_lookup_and_pre_rebuild_graph_do_not_hide_stale_review` |
| Changed implementation | `test_kg_audit_fixes.py::test_review_state_is_visible_in_task_context_and_flips_after_a_code_change` (new); `test_kg_review_provenance.py::test_model_review_binds_source_and_implementation_fingerprints` |
| Superseded equation | `test_kg_review_provenance.py::test_reviewed_correction_supersedes_a_self_declared_garbled_transcription`; `test_kg_completeness.py::test_superseded_damage_is_history_not_current_transcription_gap`; `test_kg_audit_fixes.py::test_supersession_requires_a_recorded_carry_forward_decision` and `::test_refuted_record_is_replaceable_by_a_reviewed_correction` (new) |
| Unsupported citation | `test_kg_citation_support.py::test_unreviewed_or_unsupported_citation_is_preserved_without_support`; `test_kg_audit_fixes.py::test_different_formula_review_cites_but_never_implements` (new) |
| Missing original | `test_kg_derivation_links.py::test_missing_manifest_source_cannot_establish_derivation`; `test_kg_pipeline.py::test_queue_skips_completed_reviews_and_lists_missing_source_separately`; `test_kg_pipeline.py::test_sync_does_not_call_missing_archive_synchronized` |
| Unresolved or ambiguous symbol or label | `test_kg_system.py::test_symbol_resolution_preserves_ambiguity`; `test_kg_citation_support.py::test_ambiguous_glyph_does_not_assert_two_physical_dependencies`; `test_kg_extraction.py::test_unresolved_assumption_reference_never_fabricates_entity`; `test_kg_audit_fixes.py::test_short_review_tags_resolve_to_printed_numbers_and_unmatched_labels_are_reported` and `::test_ambiguous_printed_number_is_reported_not_guessed` (new) |
| Relationship or identity collision, duplicates | `test_kg_derivation_identity.py::test_hash_collision_between_distinct_tuples_raises_before_relation_store_write`; `test_kg_record_identity.py::test_store_rejects_conflicting_duplicates_before_write_and_deduplicates_exact`; `test_kg_audit_fixes.py::test_duplicate_forms_differing_only_in_spacing_are_detected_and_retired_with_history` and `::test_retirement_refuses_records_that_differ_beyond_whitespace` (new) |
| Interrupted Docling job | `test_kg_agentic_controls.py::test_interrupted_docling_job_returns_to_queue_with_attempt_history`; `test_kg_agentic_controls.py::test_busy_docling_refusal_is_not_attempted_page_history` |
| Self-verification, unsigned and worker verdicts | `test_kg_review_provenance.py::test_transcriber_identity_is_recorded_and_cannot_self_verify` (extended for A-04) |
| Refuted-record exclusion | `test_kg_retrieval.py::test_refuted_assumptions_are_excluded_from_search_but_inspectable_by_id`; `test_kg_extraction.py::test_refutation_updates_effective_support_and_repeat_keeps_original_links`; `test_kg_audit_fixes.py::test_refuted_validity_is_excluded_from_task_context_but_inspectable` (new) |
| Role violations and reviewed bytes | `test_kg_agentic_controls.py::test_packet_state_machine_enforces_order_roles_and_reviewed_bytes`; `test_kg_agentic_controls.py::test_worker_roles_cannot_journal_scientific_acceptance` |
| Concurrent writers (processes and threads) | `test_kg_agentic_controls.py::test_writer_lock_is_reentrant_and_refuses_a_second_process`; `test_kg_audit_fixes.py::test_store_writer_lock_blocks_a_second_process_upsert` and `::test_store_writer_lock_serializes_threads_of_one_process` (new) |
| Store permissions, no-op rewrites, evidence overwrite, dangling edges | `test_kg_audit_fixes.py::test_atomic_store_write_keeps_the_published_file_mode`, `::test_unchanged_verdict_replay_does_not_rewrite_the_stores`, `::test_attempt_evidence_never_overwrites_fingerprinted_files`, `::test_s3_integrity_failures_include_dangling_edges` (new) |

## 4. Traced flow (one object, end to end)

Command: `.venv-jax/bin/python knowledge/tools/trace_flow.py EQ_DOI_10_2514_1_B36301_821FB00A --query "Son 2017 movable pintle minimum tip area continuous gap"`. It exits 0 with `complete: true`.

Object: Son et al. 2017, Eq. (1), the movable-pintle minimum orifice area.

Hops 6 and 7 are stricter than in the first version of this report. Hop 6 requires a model with a CURRENT structured review, reached through the graph relation. Hop 7 requires that review to be visible in the task package.

| Hop | State | Evidence |
|---|---|---|
| 1 original PDF page | OK | `propulsion_texts/pintle_injector/son2017.pdf`, sha256 `bc10dd3b…a2b1a7`, PDF p.2 |
| 2 raw parser candidate | OK | `CAND_425b13206dc22bdb09ffb549` (Docling, p.2), SOURCE_PAGE_COMPARED by F; DERIVED_FROM edge |
| 3 reviewed object | OK | SOURCE_STATES, quote GROUNDED, `transcribed_by` F |
| 4 independent verification | OK | CONFIRMED by role A ("claude-opus-5-5 (Opus A, independent auditor)"), independent of the transcriber F |
| 5 graph | OK | FOUND_IN document, DERIVED_FROM candidate, CITES and IMPLEMENTS from the model |
| 6 implementation association and verdict | OK | `MODEL_INJECTOR_SON2017_MOVABLE_CONTINUOUS_GAP`, relations {CITES, IMPLEMENTS}, review CURRENT / EQUIVALENT. `find_implementation` is OK, with 7 reviewed symbols: 4 in `raosim/movable_pintle.py` and 3 in `raosim/mdo/injector.py` |
| 7 evidence-aware query | OK | `retrieve_for_task` returns the equation, and its model entry carries `source_review.state = CURRENT` |

Contrast cases, from the same API:
- Bartz Eq. [9] (cited only, conversion error) and SP-125 Eq. 4-23 (DIFFERENT_FORMULA) both return `NO_IMPLEMENTATION_LINK` from `find_implementation`.
- `find_scientific_basis("raosim/movable_pintle.py:son_minimum_tip_area")` returns Son Eq. (1) with review CURRENT / EQUIVALENT.

## 5. Not run, or not established

- The engineering suites (`tests/`) were not run (owner instruction).
- No Docling process is running. The 14 low-confidence conversions were not reprocessed.
- No CI. The fresh-checkout proof is local: `git archive HEAD`, the working tree overlaid, then the corpus-free suite.
- Physical validation is not established for any model (`MODEL_SOURCE_AUDIT.json`: `physical_validation_established` 0). A passing test is software verification, not physical validation.
- Roles are self-declared (`knowledge/workflows/AGENTIC_ROLES_2026-10-16.md`, note N-10). The controls are an honour system backed by the journal and the independent audit.
