# Final independent audit — 2026-10-16 LREKit knowledge-graph continuation

Auditor: **Opus A** (role `A`, model `claude-opus-5-5 (Opus A, independent auditor)`).
Audited work: Opus F (main session) and its Sonnet workers S1–S3.
Brief: `knowledge/reports/OPUS_A_AUDIT_BRIEF_2026-10-16.md`.
Date: 2026-10-16 local (2026-10-17 UTC). HEAD `550d5c2` throughout; nothing committed.

**Verdict: FAIL_WITH_FINDINGS.** The science checks out: 145 of 145 sampled F transcriptions were
confirmed against the original pages, the four Section 17 reviews' numbers and fingerprints are
correct, the fresh-checkout proof passes and the knowledge suite passes. The failure comes from one
major knowledge-system defect (A-01) and ten minor defects. All of them can be fixed. Details are in §9.

**Final verdict after recheck (2026-10-17T03:53Z): PASS_WITH_OPEN_GAPS.** All eleven findings are
fixed and verified; see the "Recheck" section at the end.

---

## 1. Scope and method

### 1.1 Constraints honoured

- I wrote only this file and `knowledge/workflows/results/agentic-2026-10-16/reviews/A-audit-verdicts.json`.
- I did not edit `raosim/`, `lrekit/`, `tests/` or any store. I did not commit, push or download.
- I did not run the engineering suites, Docling or `kg.sh rebuild`.
- I ran one pytest process at a time. `fresh_checkout_check.py` runs its two suites one after the other.
- I rendered PDF pages with Ghostscript into the scratchpad only, at
  `/private/tmp/claude-501/-Users-ibrahimshahid-Documents-Github-RaoRocketSim/b769a29f-6c0c-4b07-8fd9-06e5cdf5f611/scratchpad/auditA/`.
- Two probes exercised `apply_verification`. Both ran on temporary or copied stores in the scratchpad. MD5 sums of the repository stores were unchanged afterwards.
- I did not apply my verdicts.

### 1.2 Commands run and results

| Command | Result |
|---|---|
| `.venv-jax/bin/python -m pytest knowledge/tests -q -p no:cacheprovider` | **316 passed in 5.82 s**. It wrote no repository files; the journal was still 635 rows afterwards. |
| `.venv-jax/bin/python knowledge/tools/fresh_checkout_check.py --output <scratch>/auditA/fresh.json` | **proof PASS**. HEAD alone: exit 2, `2 errors in 0.25s` (`ModuleNotFoundError: kg.execution_reach` in `test_kg_pipeline.py` and `test_kg_registry.py`). HEAD plus changes: `277 passed, 39 skipped in 4.21s`. |
| File list from the same run | 34 modified, 447 untracked, 0 deleted. F's list had 445; the 2 extra are `KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md` and `KNOWLEDGE_INTEGRITY_TESTS.md`, written after F's run. Total size 13.3 MB; largest file `knowledge/enrichments.json` at 3.2 MB. Types: .json, .jsonl, .py, .md, .txt, .csv and one pip `.lock`; no binaries. A secret-pattern scan found only prose ("GitHub secrets"). |
| `git -c core.excludesFile=<HEAD .gitignore> check-ignore --no-index` on three knowledge paths | All three are ignored under HEAD's rules and none under the current rules. `test_kg_repository_tracking.py` would therefore fail on HEAD. |
| `gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pnggray -r130 …`, plus 300/600 dpi crops | Pages listed in §1.3. Every source PDF's SHA-256 equals its manifest `sha256`. |
| Python checks against `raosim` (import only) | `movable_pintle.opening_for_tip_area` gives 0.417946 / 0.454243 / 0.568310 mm; printed Eq. (2) gives 0.365703 / 0.395456 / 0.492309 mm. `nozzle_geometry.lookup_angles(14, 80)` gives (31.6°, 16.7°). The exact SI Bartz coefficient is 1.184081e-7. |
| `kg.harvest_extraction.model_review_state` and `implementation_freshness.definition_fingerprint` on all 22 fingerprinted symbols | All four reviews are `CURRENT`. All 22 symbols are `VERIFIED` with the same line ranges and unchanged definition hashes. |
| `api.find_implementation`, `api.find_scientific_basis`, `api.retrieve_for_task`, `knowledge/tools/trace_flow.py EQ_DOI_10_2514_1_B36301_821FB00A` | See A-01. |
| Probe `<scratch>/auditA/bin/sim_unsigned_verdict.py` (temporary root) | A verdict signed by F on an F transcription is refused. An **unsigned** verdict is accepted (`CONFIRMED`, no verifier identity). An **S2-signed** verdict is accepted (`REFUTED`, model link removed). See A-04. |
| Dry run `<scratch>/auditA/bin/dryrun_apply.py` (copied stores) | My verdict file gives `confirmed 145`; replaying it gives `unchanged 145`. All 145 objects end `CONFIRMED` with verifier `A`. |
| Recomputation of the completion totals from `integration/*.json` | Totals reproduce: 122/92/126/4/2/0 banked, 291/82 parser reviews, 35/0/13 verdicts. The logs themselves undercount, because one was overwritten (A-11). |
| Journal and packet-queue analysis | See §7. |

### 1.3 Pages rendered and read

| Source | PDF pages |
|---|---|
| Son 2017 | 2, 3, 4, 10 |
| SP-125 | 15, 98, 99, 100, 107, 109, 110, 115, 119, 120 (300 dpi crops of pp.15, 99, 110, 115, 119; 600 dpi crop of p.115 Eq. 4-26) |
| Bartz 1957 | 1–3 |
| Pizzarelli 2011 | 7, 9, 18 |
| Lee 2021 | 7, 10, 11 |
| Pavlov Rachov 2013 | 4, 5 |
| Kwak 2018 | 9 |
| Mirzamoghadam 1991 | 4–7 |
| Hwang 2022 | 4 |
| Freeberg 2019 | 6 |
| Kang 2022 | 6 |
| Nasuti & Pizzarelli 2021 | 6 |
| SP-8120 | 29 |
| SP-8107 | 16, 17 |

### 1.4 Changes observed during the audit

- The owner stopped this audit and later resumed it (journal seq 636–637).
- At 22:02–22:03 local, F edited `KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md`, `KNOWLEDGE_INTEGRITY_TESTS.md`, `NEXT_KNOWLEDGE_SESSION.md` and `completion_narrative_2026-10-16.md` to describe that stop and resume. I re-checked every line I cite against the current files.
- At 22:03:49 local, the git index was updated: **all 481 session files are now staged** (34 `M`, 447 `A`). No journal row records who staged them (N-11).
- The stores have not changed since 21:19:45 local.

---

## 2. Findings

Severity vocabulary: blocker / major / minor / note. Line numbers refer to the current working tree.

| ID | Severity | Location | Evidence | Required action |
|---|---|---|---|---|
| A-01 | **major** | `knowledge/kg/harvest_extraction.py:206-220` (`links_model`); `knowledge/kg/graph.py:173-182`; `knowledge/kg/api.py:713` (model fields returned by `retrieve_for_task`); `reviews/{M-SON2017-p003,X-BARTZ-p001-005,V-LEE-p006-008,M-SP125-p115}.extraction.jsonl`; `knowledge/reports/KNOWLEDGE_INTEGRITY_TESTS.md:75-76` | **(a) Support labels silently match nothing.** The four P5 results give `supports_model.equation_labels` as short tags ("Eq. (1)", "Eq. [7]", "Eq. (4-23)", "Eqs. (26)-(32)"). The stored labels are long ("Son 2017 Eq. (1) movable-pintle minimum orifice area"), so `links_model` matched nothing, and no warning was emitted (only comparison links are reported). As a result, `MODEL_INJECTOR_SON2017_MOVABLE_CONTINUOUS_GAP` and `MODEL_THERMAL_BARTZ` have **0 IMPLEMENTS edges**, only `CITES` edges scoped "explicit source/implementation comparison; no IMPLEMENTS assertion", although both reviews say EQUIVALENT.<br>**(b) The reviewed basis is not reachable.** `api.find_implementation("EQ_DOI_10_2514_1_B36301_821FB00A")` and `("EQ_DOI_10_2514_8_12572_14DF9A7B")` both return `NO_IMPLEMENTATION_LINK`. `find_scientific_basis("raosim/movable_pintle.py:son_minimum_tip_area")` returns `equations: []`.<br>**(c) The structured review appears nowhere in the graph or retrieval API.** The fidelity class, review state (`CURRENT` / `IMPLEMENTATION_CHANGED_REVIEW_REQUIRED` / `STALE_SOURCE`), `physics_change_required` and the fingerprints are absent from `retrieve_for_task` and from model graph nodes; `api.py` never reads `source_review`. Only `completion.py` (`MODEL_SOURCE_AUDIT.json`) and `trace_flow.py` show them. A later code change would therefore never reach task context, contrary to `knowledge/AGENTS.md` §11 ("separate, visible states in task context").<br>**(d) The trace overstates.** The integrity report marks hops 6–7 "OK", but the query returns the equation without the review. | 1. Report unmatched `supports_model.equation_labels` in the harvest (as already done for comparison labels) and treat them as an S3 mismatch, or accept equation IDs.<br>2. Record the intended link per equation and re-bank: Son Eq. (1) and Bartz Eqs. [5], [7], [8] IMPLEMENTS; Son printed Eq. (2), Bartz Eq. [9] and SP-125 Eq. 4-23 CITES only.<br>3. Expose `model_review_state()` and the fidelity class in `retrieve_for_task`, the model summaries and the model graph nodes.<br>4. Add tests: the review state is visible in task context; it flips to `IMPLEMENTATION_CHANGED_REVIEW_REQUIRED` after a structural code edit; `find_implementation` returns the reviewed symbol for an EQUIVALENT equation.<br>5. Correct hops 6–7 in `KNOWLEDGE_INTEGRITY_TESTS.md` until this is done. |
| A-02 | minor | `EQ_DOI_10_1007_S42405_020_00325_Z_00D2425E_C452A65C` vs `…_D022B465`; `knowledge/kg/validate.py:161`; `knowledge/tools/agentic_packet.py:287` with `knowledge/kg/ground.py:28-33` | Two **current, active** records of Lee Eq. (29): same document, page 7 and number, and forms identical except for whitespace ("P_inv_out/delta_P_inv" vs "P_inv_out / delta_P_inv"). D022B465 (legacy) is CONFIRMED, carries IMPLEMENTS and has assumptions and validity. C452A65C (F, V-LEE-p006-008) is unverified, carries the model's comparison link and has neither. Both duplicate detectors normalize whitespace runs but not spaces around operators, so neither the validation nor the R-LEE bundle flagged the pair. | Supersede one record and keep the history. Make both duplicate checks whitespace-insensitive (or compare equation number and page) and add a test. |
| A-03 | minor | `EQ_NTRS_19710019929_C43FA39B`, `_0BBBD3E9`, `_2AD5D2F0` (corrections) vs `_98E8ACF4`, `_7A8070BC`, `_16E1DC3A`; `reviews/V-SP125-p014-015.extraction.jsonl`, `V-SP125-p098-099.extraction.jsonl` (`assumptions: []`, `validity: []`); `REMAINING_SCIENTIFIC_GAPS.md:73` (K2 "RESOLVED") | The corrected forms are right (confirmed at 300 dpi). However, the corrections dropped the page-stated assumptions and validity carried by the superseded records:<br>• 1-14: "v_inj = 0, (p_c)inj = p_inj" (p.15).<br>• 4-6: "injector face excluded" (p.98).<br>• 4-7: conical contour assumption; R 0.5–1.5 R_t; convergent 20–45°; divergent 12–18° (pp.98–99).<br>They also dropped the model link: `MODEL_CHAMBER_SHOULDER_FILL_FRACTION` now has **no current equation link**. Nothing records this as a decision, and K2 does not mention it. | Carry the page-anchored assumptions and validity onto the corrected records, or record a reasoned decision against it. State the model-link change in K2. Add an assertion to `test_reviewed_correction_supersedes_a_self_declared_garbled_transcription` that the review decides what is carried forward. |
| A-04 | minor | `knowledge/kg/harvest_extraction.py:635-640` (`apply_verification`) | The self-verification refusal applies only when a verdict names a reviewer. The probe showed that an **unsigned** verdict confirms an F transcription and records no verifier identity, and that an **S2-signed** verdict refutes one. Role F is enforced only by `packet_queue._verdict_summary`. No unsigned or worker verdict was applied this session: all 48 this-session verifications are by F on legacy objects. | Refuse verdicts on `transcribed_by` records that carry no reviewer identity. Refuse worker roles (S1–S3) in `apply_verification`. Add both cases to `test_transcriber_identity_is_recorded_and_cannot_self_verify`. |
| A-05 | minor | `knowledge/kg/store.py:103-115` (`_atomic_write_json`) | `NamedTemporaryFile` creates mode 0600 and `os.replace` keeps it. As a result, all 17 `knowledge/*.json` stores rewritten since the fix, plus `packet_queue.json`, `CORPUS_COMPLETENESS.json`, `EQUATION_FIDELITY_AUDIT.json` and `MODEL_SOURCE_AUDIT.json`, are now `-rw-------`. Git does not record these bits, but other local users or service accounts can no longer read the stores. | Before the replace, set the mode from the existing file or the umask (e.g. `os.fchmod`). Add a test. |
| A-06 | minor | `knowledge/tests/test_kg_agentic_controls.py:48-66`; `knowledge/kg/locks.py:55-68` | Repair seq 3 names "no writer serialization" as the defect. The test, however, proves cross-process refusal only for the `docling-heavy` lock. Nothing shows that `store.save`/`upsert` block or time out while another process holds `store-writer`. Separately, reentrancy is keyed per process, so a second **thread** passes straight through. | Add a spawn-process test that holds `store-writer` and asserts `store.upsert` waits or raises `LockBusyError`. Document or fix the per-thread semantics. |
| A-07 | minor | `knowledge/tools/agentic_packet.py:406-408` (`s3_integrate`) and `:449-451` (`s3_test`) | `dangling_edges` is computed (line 338) but is missing from both pass/block conditions. Dangling edges would therefore pass S3 integration. The final graph has 0, so there was no harm this session. | Add `dangling_edges` to both conditions and to the S3 regression test. |
| A-08 | minor | `knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md:248-252`; `completion_narrative_2026-10-16.md:84-88` | The report says "The journal records them" of three self-corrections. Only the Solda one is journaled (seq 430, 444). No journal row mentions the SP-8107 J-2/SSME row labels or the Pizzarelli partial-derivative notation. I checked the SP-8107 Table I values on p.17; they are correct as banked. | Journal the two corrections (what was wrong, what changed, which file hash), or amend the sentence. |
| A-09 | minor | `knowledge/reports/KNOWLEDGE_INTEGRITY_TESTS.md:75` | The table says the 7 fingerprints are `raosim/movable_pintle.py` and `raosim/mdo/pintle.py` symbols. They are `raosim/mdo/injector.py` symbols (`son_tip_area`, `opening_for_tip_area`, `center_gap_area`), as `models.json` and `trace_flow.py` show. | Correct the file name. |
| A-10 | minor | `knowledge/reports/{CORPUS_COMPLETENESS,EQUATION_FIDELITY_AUDIT,MODEL_SOURCE_AUDIT}.json` `inputs`; `knowledge/kg/harvest_extraction.py:711-716` | The reports were generated at 01:17:57Z. At 01:19:42Z an `apply_verification` run re-saved five stores (one shared `meta.generated_at`), and `enrichments.json` was re-saved at 01:19:45Z. This is consistent with the whole-session replay, since `apply_verification` saves all five of its stores even when every verdict is `unchanged`. It is also later than the "final" 01:18 suite run that `KNOWLEDGE_INTEGRITY_TESTS.md` §1 places after the replay. Six of the eleven recorded input SHA-256s (equations, claims, entities, validity, models, enrichments) no longer match the files, so the reports cannot be byte-checked against the stores. | Skip saves when nothing changed, then regenerate `completion.write_all()` (or rebuild) after the final replay so the fingerprints match. |
| A-11 | minor | `knowledge/tools/agentic_packet.py:320, 403, 453` (fixed paths `candidates/`, `integration/`, `tests/<pid>.json`); `packet_queue.json` P2-LEE-p010-011; `KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md:48` | Re-running S2 or S3 overwrites the evidence a history step fingerprinted. For P2-LEE-p010-011, the BLOCKED integration log (sha `6aabb423…`) and the first candidate bundle no longer exist. The overwritten log applied 12 confirmations and 1 refutation (`ASM_DOI_10_1007_S42405_020_00325_Z_00D2425E_D002C669`), so the completion report's "**confirmed 35, refuted 0**" is wrong: the verdict files and stores show **47 confirmed, 1 refuted** this session. | Write attempt-numbered logs and never overwrite fingerprinted evidence. Count verdict outcomes from the verdict files or stores. Correct line 48. |
| N-01 | note | `knowledge/reports/NEXT_KNOWLEDGE_SESSION.md:33` | The resume commands use `.venv/bin/python`; `AGENTIC_ROLES_2026-10-16.md` says `.venv` is absent. | Use `.venv-jax`. |
| N-02 | note | `models.json` `MODEL_THERMAL_BARTZ.source_review` | The top-level class EQUIVALENT covers 7 fingerprinted symbols. These include `combustion_gas_viscosity` (CONVERSION_ERROR, +0.745 %) and `recovery_temperature` (REPOSITORY_EXTENSION), so a reader of `fidelity` alone misses both. All four reviews also have `equation_ids: []`. | Consider per-symbol fidelity, or keep those two symbols out of the EQUIVALENT set. Fill `equation_ids`. |
| N-03 | note | `agentic_work_journal.jsonl` seq 319, 320, 344, 345 | These FIXED_VERIFIED rows fingerprint only the test files, not the code they changed (`agentic_packet.py`, `work_queue.py`, `api.py`, `graph.py`). | Add follow-up rows with the code fingerprints. |
| N-04 | note | `harvest_extraction.py:270-271` (REFUTED records replaceable); `api.py` `find_validity_conditions` / `retrieve_for_task` validity filter | These branches of repairs 51 and 345 have no regression test; only the garbled-form and refuted-assumption paths are tested. | Add tests. |
| N-05 | note | `REMAINING_SCIENTIFIC_GAPS.md:33` (G5) vs `P2-electric-technology-sets-comparison.json` P2-F5 | G5 gives SP-8107 Table I as "3.65–179 kW/kg". The 179 is the SSME row, which footnote 3 on p.17 marks as not operational. P2-F5 gives 3.65–35.5 kW/kg for operational assemblies. | Align the wording. |
| N-06 | note | `knowledge/tools/session_completion_report.py:52-57` | Counts are parsed from the first `{` of a 3000-character stdout tail. A longer output would drop or misparse them silently. No log was truncated this session. | Have `bank_extraction` write a structured result and read that instead. |
| N-07 | note | `EQ_NTRS_19710019929_2DC4BFBD` (SP-125 p.107) | The page prints 7760 in³ for the convergent frustum; its own inputs give 7780 in³ (+0.26 %). This is a source-arithmetic slip like G32/G33, and it is not recorded. | Add it to section B of the gaps list. |
| N-08 | note | `CONST_DOI_10_1016_J_ACTAASTRO_2022_08_029_4B9DC35D` (Kang p.6) | The value 1868 K is correct. The same page says the computed temperatures "were somewhat over predicted because the chemical equilibrium combustion model was applied"; the record does not carry this caveat. | Attach the caveat. |
| N-09 | note | `KNOWLEDGE_INTEGRITY_TESTS.md` §2.3 | The whole-session replay used an uncommitted scratch script and kept no output artifact, so it cannot be re-run from the repository. The per-packet S3 replays are recorded. | Commit the script or its output. |
| N-10 | note | `knowledge/kg/packet_queue.py`, `knowledge/kg/work_journal.py` | Roles are self-declared (`--role`): an honour system, not authentication. The recorded attribution is consistent (§7). | Document the limit. |
| N-11 | note | `.git/index` (22:03:49 local) | All 481 session files are staged and no journal row records it. The staged snapshot predates this audit's two files and any fixes. HEAD is unchanged. | Whoever staged it: journal it. Re-stage after the fixes. |
| N-12 | note | `KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md:185` vs `completion_narrative_2026-10-16.md:21` | The generated report (22:02:41) says the audit was stopped "before it produced a report"; the narrative (22:03:20) says it was resumed. The report was not regenerated. | Regenerate after the audit. |
| N-13 | note | `.gitignore` last line | `knowledge/reports/validation_20*.json` is now ignored, but five such files remain tracked from HEAD. This is harmless. | Untrack them or accept the inconsistency. |

---

## 3. Independent verification sample (`A-audit-verdicts.json`)

- **145 verdicts: 145 confirmed, of which 144 at high and 1 at medium confidence; 0 refuted; 0 low-confidence.**
- The sample covers 145 of the 342 current F-transcribed SOURCE_STATES objects (42 %): 53 equations, 41 claims and 51 constants or assumptions, across 14 documents.
- The generating script checked that every target is current, `SOURCE_STATES`, has `transcribed_by.role == "F"` and sits on the stated page.
- Every verdict carries the manifest `source_sha256`, reviewer `{"role": "A", "model": "claude-opus-5-5 (Opus A, independent auditor)"}` and the page.
- Applied to copied stores, the file gives `confirmed 145`, then `unchanged 145` on replay.

All items the brief requires are included and confirmed:

| Required item | Object(s) | Page and check |
|---|---|---|
| Son 2017 Eq. (1) | `EQ_DOI_10_2514_1_B36301_821FB00A` | p.2; term-by-term. This completes hop 4 of the traced flow once applied. |
| Pizzarelli equation and Table 2 | Eqs. (6), (9), (10), (11), (14), (15), (16); all 10 Table 2 values (`CONST_…578011_036694A8` … `_7B43C087`) | pp.7, 9, 18 |
| SP-125 Eq. 4-23 and RP-1 C1 | `EQ_NTRS_19710019929_78B48BD5`; `CONST_NTRS_19710019929_7620B1E8` (C1 = 0.0214) | pp.115, 119 at 300 dpi |
| Lee electric-drive equation | Eq. (29) `…_C452A65C`, Eq. (25) `…_462E6827`, Eq. (38) `…_F56CC231` | pp.7, 10. C452A65C is the duplicate in A-02. |
| Corrected SP-125 equations | Eqs. 1-14, 4-6, 4-7 | pp.15, 98, 99. The forms are right; see A-03. |
| P3 constraint provenance | Mirzamoghadam 1050 °F, 800 °F, 900 °F, AR 7/8, 0.9 and four claims (pp.4–7); Hwang Eq. (3) (p.4); Freeberg 0.3 < BF < 0.7 (p.6); Kang 1868/1545 K (p.6); Son pp.2, 3, 4, 10 | as listed |
| Late SP-125 packets | Eq. 4-16 `EQ_NTRS_19710019929_D069B835` (p.110, 300 dpi); RP-1 Nusselt form `EQ_NTRS_19710019929_848A0FB1` (p.119, 300 dpi) | as listed |

Other observations from the pages:

- **Parser rejection confirmed.** F's rejection of `CAND_e5b9293d638533e465c7b8a9` (Eq. 4-26 read as "T_ce") is right: at 600 dpi the page prints T_cc. At 130 dpi I had misread it the same way.
- **G33 confirmed.** The printed expanded Eq. (b) omits π. The p.120 results do not reproduce: the equations give d = 0.847 in and N = 90.7, and for N = 94 they give d = 0.815 in.
- **No transcription error found** in the sample.

---

## 4. Model reviews (Section 17)

| Model | Class | Checked against code and pages | Result |
|---|---|---|---|
| `MODEL_INJECTOR_SON2017_MOVABLE_CONTINUOUS_GAP` | EQUIVALENT | Code: Eq. (1) expanded form equals the printed form (ratio 1.000000000 at 0/20/40°); transition openings 0.417946/0.454243/0.568310 mm reproduce Table 2 (0.418/0.454/0.568, p.3). Printed Eq. (2), with R_post, gives 0.3657/0.3955/0.4923 mm, so the typo claim holds. The registry author list (`raosim/model_registry.py:113`) is wrong, as the review says. 7 fingerprints point to `raosim/movable_pintle.py` and `raosim/mdo/injector.py` symbols; all current. | Correct. Integration gap: A-01. |
| `MODEL_PUMP_ELECTRIC_DRIVE` | CONDITIONALLY_EQUIVALENT | `raosim/mdo/pump.py:85-174`: P_shaft = mdot dp/(rho eta) (Eq. 20); motor mass from shaft power (26–27); inverter mass from (P_shaft/eta_m) (28–29); battery branches with P_elec = P_shaft/(eta_m eta_inv) (30–31). kappa_b is applied in `raosim/mdo/engine.py:379-390`. The objective uses `_smooth_max(…, 2.0)` = logsumexp(2v)/2 (`engine.py:90-93`); installed mass uses an exact `jnp.maximum`. Pump mass uses P_shaft/pump_specific_mass, so the claim "DIFFERENT_METHOD vs Eq. 25" holds. | Correct. |
| `MODEL_THERMAL_BARTZ` | EQUIVALENT | Bartz Eqs. [5], [7], [8], [9] (pp.1–2) and SP-125 Eqs. 4-13 to 4-16 (pp.109–110) agree with the code. The exact coefficient is 46.6e-10 × 17.857967 × 1.8^0.6 = **1.184081e-7** (F: 1.18407e-7); the code's 1.1929e-7 is +0.745 % in μ and +0.149 % in h_g, so the conversion-error claim holds. Locators: the code cites SP-125 "Eq. 4-18" for Pr (it is 4-15) and "Eq. 4-23" for μ (it is 4-16; 4-23 is the coolant Nusselt relation on p.115). Applicability: D*/r_c ≲ 3, angles within 50 % (p.2); chamber flux 40–50 % of measured (p.3). The MDO path passes `mission.mu_gas` (`raosim/mdo/cooling.py:339-357`). | Correct within rounding. See N-02 and A-01. |
| `MODEL_THERMAL_SIEDER_TATE` | DIFFERENT_FORMULA | SP-125 p.115 prints Nu = C1 Re^0.8 Pr^0.4 (μ/μ_w)^0.14 with a coolant-specific C1 and calls it "the Sieder-Tate equation". p.119 gives RP-1 C1 = 0.0214. The code (`raosim/physics.py:489`, `raosim/jax/thermal.py:74`) uses 0.027 Pr^(1/3). The ratio 1.2617·Pr^(−1/15) gives +26.2 % at Pr 1, +17.3 % at Pr 3 and +7.1 % at Pr 11.7; Pr 11.7 equals SP-125's RP-1 μCp/k = 11.69. The registry cites SP-125 (`raosim/model_registry.py:508-516`). | Correct; the non-conservative direction holds. |

Every one of the 22 implementation fingerprints is `VERIFIED` against the current tree, and each points to the named symbol.

---

## 5. The 15 journaled repairs, plus the reporting-tool fix

For each repair: did the behaviour change, and does a regression test fail without it?

| Seq | Behaviour change | Regression test |
|---|---|---|
| 2 | Yes; the fresh checkout passes | `test_kg_repository_tracking.py`: fails under HEAD's ignore rules (shown with `check-ignore`) |
| 3 | Yes | `test_save_uses_private_temporary_files` and `test_busy_docling_refusal_is_not_attempted_page_history` fail on old code. Store-writer serialization is untested (A-06). |
| 4 | New modules | state-machine, journal, interruption, changed-source and priority tests |
| 51 | Yes (`graph.py:61-74`, `harvest_extraction.py:268-271`) | Two tests fail on HEAD's form-only check. The REFUTED branch is untested (N-04). |
| 52 | Yes | `test_variable_statistics_separate_current_from_superseded_history` |
| 53 | Yes | identity, model-identity and fingerprinted-review tests. Unsigned and worker verdicts are uncovered (A-04). |
| 59 | Yes (3 HELD, 5 BLOCKED) | `test_standards_are_held_only_with_a_current_mapped_document` |
| 111 | Yes; I re-ran it: PASS | It is a tool; no unit test |
| 129, 152 | Yes | the S2 caption, model-target and table-grid tests |
| 145 | Yes | `test_literature_constants_and_assumptions_can_be_independently_verified` |
| 319 | Yes | `test_s2_flags_a_damaged_text_layer_behind_a_clean_transcription`. The journal row fingerprints the test only (N-03). |
| 320 | Yes | `test_verification_queue_counts_constants_assumptions_and_source_validity` |
| 344 | Yes | `test_s3_checks_verdicts_on_constants_assumptions_and_validity`: the old lookup BLOCKs |
| 345 | Yes for assumptions | `test_refuted_assumptions_are_excluded_from_search_but_inspectable_by_id`. Validity is untested (N-04). |
| Final fix to `session_completion_report.py` | Totals reproduce from the logs | No test, as F stated. The parsing is fragile (N-06) and the source logs undercount (A-11). |

---

## 6. Negative paths

All of these tests exist, and all passed in my run of the suite.

| Negative path | Tests |
|---|---|
| Rejected OCR or parser candidates | `test_kg_enrichment.py::test_rejected_parser_run_is_history_not_active_evidence`; `test_kg_docling_attempts.py::test_explicit_new_attempt_does_not_overwrite_or_inherit_rejected_review`; `test_kg_agentic_packet_ops.py::test_s2_flags_a_damaged_text_layer_behind_a_clean_transcription`, `::test_s3_applies_signed_parser_reviews_to_individual_candidates` |
| Stale source | `test_kg_agentic_controls.py::test_changed_source_blocks_further_progress`; `test_kg_extraction.py::test_changed_source_does_not_reuse_old_verification`; `test_kg_retrieval.py::test_stale_records_are_excluded_but_remain_inspectable`; `test_kg_citation_support.py::test_direct_model_lookup_and_pre_rebuild_graph_do_not_hide_stale_review` |
| Superseded equation | `test_kg_review_provenance.py::test_reviewed_correction_supersedes_a_self_declared_garbled_transcription`; `test_kg_completeness.py::test_superseded_damage_is_history_not_current_transcription_gap` |
| Unsupported citation | `test_kg_citation_support.py::test_unreviewed_or_unsupported_citation_is_preserved_without_support`; `test_kg_constraint_citations.py::test_report_key_is_citation_not_threshold_support` |
| Missing original | `test_kg_derivation_links.py::test_missing_manifest_source_cannot_establish_derivation`; `test_kg_pipeline.py::test_queue_skips_completed_reviews_and_lists_missing_source_separately`, `::test_sync_does_not_call_missing_archive_synchronized` |
| Unresolved or ambiguous symbol | `test_kg_system.py::test_symbol_resolution_preserves_ambiguity`; `test_kg_citation_support.py::test_ambiguous_glyph_does_not_assert_two_physical_dependencies`; `test_kg_extraction.py::test_unresolved_assumption_reference_never_fabricates_entity` |
| Relationship or identity collision | `test_kg_derivation_identity.py::test_hash_collision_between_distinct_tuples_raises_before_relation_store_write`; `test_kg_record_identity.py::test_store_rejects_conflicting_duplicates_before_write_and_deduplicates_exact` |
| Interrupted Docling job | `test_kg_agentic_controls.py::test_interrupted_docling_job_returns_to_queue_with_attempt_history`, `::test_busy_docling_refusal_is_not_attempted_page_history` |
| Self-verification refusal | `test_kg_review_provenance.py::test_transcriber_identity_is_recorded_and_cannot_self_verify`. Signed verdicts only; see A-04. |
| Refuted-record exclusion | `test_kg_retrieval.py::test_refuted_assumptions_are_excluded_from_search_but_inspectable_by_id`; `test_kg_extraction.py::test_refutation_updates_effective_support_and_repeat_keeps_original_links` |

---

## 7. Process integrity

**Packet state machine** (`packet_queue.json`): 94 packets, 53 FINAL_AUDIT_ELIGIBLE and 41 QUEUED.

- In every FINAL_AUDIT_ELIGIBLE packet's last cycle, F performed OPUS_SOURCE_REVIEW, VERIFIED and FINAL_AUDIT_ELIGIBLE, and S3 performed GRAPH_INTEGRATED and TESTED. No exception.
- S1 performed every DOCLING_RUNNING and RAW_CAPTURED step; S2 performed every CANDIDATES_PREPARED step.
- All 104 reviewed files still match the SHA-256 bound at VERIFIED.
- All current integration and test logs match their bound hashes.
- All 94 packet source hashes equal the manifest.
- Requeues carry reasons: V-SP125-p104 was interrupted; M-SP125-p109-112 and P2-SOLDA hit timeouts and were BLOCKED; P2-LEE-p010-011 was falsely BLOCKED.
- History-evidence drift exists in 9 packets:
  - comparison documents fingerprinted at OPUS_SOURCE_REVIEW were edited later (living documents);
  - P2-LEE-p010-011's first bundle and its BLOCKED log were overwritten (A-11).

**Journal** (637 rows at audit end):

- `seq` is contiguous and timestamps are monotonic.
- F: 289 rows, all `claude-opus-5-5 (Opus F, main session)`.
- S1 117, S2 124 and S3 107 rows, all `claude-sonnet-5-5`.
- **No worker row has ACCEPTED_SCIENCE, REJECTED_SCIENCE or FIXED_VERIFIED.** All 54 ACCEPTED_SCIENCE rows are F's VERIFIED transitions.
- A word scan of worker notes found no scientific judgement; the two hits are byte-integrity remarks.
- The 53 S3 test runs never overlapped (minimum spacing 14 s). They reported 43×316, 1×314, 6×312 and 3×310 passed, as stated.

**Unsigned and worker verdicts:** none were applied this session (A-04 is latent).

---

## 8. Honesty of claims

Accurate and backed by the record:

- 316 tests; fresh-checkout PASS.
- Graph 5694/7261 nodes/edges, with 0 duplicate, 0 dangling and 0 unprovenanced edges (recomputed).
- `physical_validation_established: 0`.
- Completeness `NOT_ESTABLISHED`.
- 8 standards (3 HELD).
- The 28 unlinked F equations.
- The P1 statement that the code already carried an adjudication note.
- P2 values (Rachov, Kwak, Lee and SP-8107 values checked on the pages).
- P3 answers (755 K and 700 K are printed; 839 K is a correct conversion of 1050 °F).
- The scope limits in the completion report ("Not done").

Gap entries spot-checked against code lines and pages, all accurate:

- G1, G2: `lookup_angles(14, 80)` = 31.6°/16.7°; Fig. 4-14 θe falls with ε.
- G4: `electric_technology.py:293-303`; η_E = 0.95 at 785 s.
- G5–G7, G9–G11, G13, G14, G16, G20: lines in `schema.py` / `constraints.py`.
- G15: arithmetic.
- G17: `cooling.py:339-357`.
- G19: registry lines 508–516.
- G24: arithmetic.
- G33: arithmetic.
- K8: three REJECTED candidates with no correcting record.

Inaccurate or overstated:

- the self-correction journal claim (A-08);
- verdict totals (A-11);
- the trace file name (A-09);
- hops 6–7 "OK" (A-01d);
- K2 "RESOLVED" without the dropped assumptions and validity (A-03);
- G5's upper bound (N-05);
- the stale generated report (N-12).

---

## 9. Verdict

**FAIL_WITH_FINDINGS.**

What passed:

- **Sources.** Every sampled scientific object matches its original page: 145/145 confirmed, 0 refuted. This covers the brief's required items, at 300–600 dpi where notation mattered.
- **Model reviews.** The fidelity classes and load-bearing numbers of the four Section 17 reviews reproduce from code and pages.
- **Tests and checkout.** The fresh-checkout fix is real and proven. The knowledge suite passes.
- **Process.** Role gating and attribution were respected. The reports are scoped honestly about what remains undone.

Why it fails:

- **A-01 (major).** The Section 17 reviews do not reach the system's own retrieval contract. EQUIVALENT-reviewed equations report `NO_IMPLEMENTATION_LINK`. Review state and fidelity never appear in `retrieve_for_task` or the graph, so a later code change would never be visible in task context. The harvest silently dropped the reviews' support labels.
- **Ten minor defects** affect correctness, integrity or honesty of the record:
  - a duplicate current equation;
  - corrections that dropped page-stated validity;
  - a self-verification bypass through unsigned or worker verdicts;
  - owner-only store permissions;
  - untested writer serialization;
  - an unenforced dangling-edge check;
  - an unjournaled self-correction claim;
  - a wrong file name in the trace table;
  - stale derived-report fingerprints;
  - overwritten evidence with understated verdict totals.

None of this invalidates the scientific content.

**Path to PASS_WITH_OPEN_GAPS:** fix A-01 through A-11, or have the owner explicitly accept specific items as open gaps, then apply `A-audit-verdicts.json` with `apply_verification`.

---

## Recheck

Rechecked by Opus A (same agent) at 2026-10-17T03:53Z, after F's repairs (journal seq 638–726).
The same rules applied: I wrote only this file; no git writes, no rebuild, no Docling, no
engineering suites, and one pytest at a time.

**Final verdict: PASS_WITH_OPEN_GAPS.** All eleven actionable findings are fixed, and I checked
each in code, data and tests. The notes were acted on, except N-13, which was accepted. What
remains is either a documented open gap or a residual note (R-1 to R-5 below); none of it is a
defect in the session's claims.

### Commands and results

| Check | Result |
|---|---|
| `.venv-jax/bin/python -m pytest knowledge/tests -q -p no:cacheprovider` | **331 passed in 9.30 s**: 316 plus 15 new in `test_kg_audit_fixes.py`. The journal was unchanged by the run. |
| `.venv-jax/bin/python knowledge/tools/fresh_checkout_check.py --output <scratch>/auditA/fresh_recheck.json` | **proof PASS**. HEAD alone: `2 errors in 0.25s`. HEAD plus changes: `292 passed, 39 skipped`. 49 files are new since the owner's staging. |
| Repair fingerprints in journal seq 712–725 | All 15 code/test SHA-256 prefixes equal the current files. |
| `trace_flow.py EQ_DOI_10_2514_1_B36301_821FB00A` | `complete: true`. Hop 4 is CONFIRMED by role A as an independent verdict. |
| My probe `sim_unsigned_verdict.py`, re-run on a temporary root | F-signed verdict: `self_verification_refused`. Unsigned verdict: **`unsigned_refused`**. S2-signed verdict: **`worker_role_refused`**. Before the fix the last two were applied. Repository stores unchanged (MD5). |
| `A-audit-verdicts.applied.json` | Bound to my file's SHA-256 (`486fec1f…`, unchanged). The stores hold 145 role-A CONFIRMED verdicts: 144 on current records, plus the retired duplicate `…_C452A65C`, kept as history. |
| Graph | 5695 nodes / 7267 edges, with 0 duplicate, 0 dangling and 0 unprovenanced edges (recomputed). |
| Packet queue | 60 FINAL_AUDIT_ELIGIBLE and 41 QUEUED. In every final cycle, F performed OPUS_SOURCE_REVIEW, VERIFIED and FINAL_AUDIT_ELIGIBLE, and S3 performed GRAPH_INTEGRATED and TESTED. All 112 reviewed files still match their bound SHA-256. |
| Journal | 726 rows. F 336 rows, all Opus; S1 131, S2 138 and S3 121 rows, all Sonnet. No worker row has an authoritative state. |
| Completion-report inputs | All 11 input fingerprints in each of the three JSON reports match the current files. |
| Store file modes | 0 files at 0600 outside `.work`; all 19 `knowledge/*.json` files are 0644. |

### Finding by finding

For each finding: what I checked, and whether a regression test fails on the old code.

| ID | Status | What I verified | Test fails on the old code? |
|---|---|---|---|
| A-01 | **fixed** | Labels resolve by exact label, EQ_ id or printed number (`harvest_extraction.py:103-145`). Unresolved, ambiguous or refused labels are reported, and `BANK_MISMATCH_KEYS` block S3 (`agentic_packet.py:59-60, 447`). The links now match my required set exactly: Son Eq. (1) IMPLEMENTS, printed Eq. (2) CITES only; Bartz [5], [7], [8] IMPLEMENTS, [9] CITES only; SP-125 4-23 and the p.119 RP-1 form CITES only; Lee 20 and 26–32 IMPLEMENTS, 25 CITES only. For all four models, each review's `equation_links` equals the actual IMPLEMENTS set, and no CITES_ONLY equation is linked. `find_implementation` now returns `reviewed_implementations` (CURRENT, EQUIVALENT) for Son Eq. (1) and Bartz Eq. [7], and `NO_IMPLEMENTATION_LINK` for Son Eq. (2), Bartz [9] and SP-125 4-23, as it should. `find_scientific_basis(son_minimum_tip_area)` returns Eq. (1) and the review. `retrieve_for_task` carries `source_review` (state, fidelity, `physics_change_required`) and warns for the DIFFERENT_FORMULA and CONDITIONALLY_EQUIVALENT reviews. Model graph nodes carry `source_review_*`. N-02 is addressed: Bartz `fidelity_scope` plus `implementation_findings` for the viscosity function and the recovery temperature. | Yes: `test_short_review_tags_…`, `test_ambiguous_printed_number_…`, `test_different_formula_review_…`, `test_review_state_is_visible_…`, which also shows the flip to IMPLEMENTATION_CHANGED after a code edit. |
| A-02 | **fixed** | `form_key` ignores all whitespace and is used in `validate.py:162-163` and `agentic_packet.py:315-322`. `…_C452A65C` is retired (inactive, `duplicate_of …_D022B465`, reasoned F record). The candidate link moved to D022B465 and the comparison link was dropped. No current duplicates remain. | Yes: both A-02 tests. |
| A-03 | **fixed** | Supersession now reports `supersession_carry_forward_undecided` unless a `carry_forward` decision is recorded. The three SP-125 corrections carry page-stated assumptions, and for 4-7 the conical-nozzle ranges, labelled "typical-design range, not a stated validity limit"; each has a recorded decision. I re-read the added text against PDF pp.15, 98 and 99: it is accurate and the labelling is fair. K2 now states that the model link is deliberately not carried. | Yes: `test_supersession_requires_a_recorded_carry_forward_decision` plus the extended correction test. |
| A-04 | **fixed** | `harvest_extraction.py:754-766`: worker roles are refused, and unsigned verdicts are refused when `transcribed_by` exists. Confirmed by my probe. | Yes: the extended `test_transcriber_identity_…` |
| A-05 | **fixed** | `store._atomic_write_json` sets the published or umask mode with `fchmod` before `os.replace`. All files are restored to 0644. | Yes: `test_atomic_store_write_keeps_the_published_file_mode` |
| A-06 | **fixed** | `locks.exclusive` keeps a per-path `threading.Lock` and is reentrant only for the holding thread. There are now a two-process upsert test and a thread test. | The thread test would fail on the old process-wide reentrancy. |
| A-07 | **fixed** | `integrity_failures()` includes `dangling_edges` and is used by both `s3_integrate` and `s3_test` (`agentic_packet.py:447, 498`). | Yes |
| A-08 | **fixed** | Journal seq 646–647 record both corrections and the reviewed-file hashes. | n/a (record) |
| A-09 | **fixed** | `KNOWLEDGE_INTEGRITY_TESTS.md` hop 6 names `raosim/mdo/injector.py`; hops 4–7 now match the trace output. | n/a |
| A-10 | **fixed** | `apply_verification`, duplicate retirement and `record_review` save only when dirty, and `record_review` holds the writer lock. The final rebuild ran after the last write, and all report input hashes match. | Yes: `test_unchanged_verdict_replay_does_not_rewrite_the_stores` |
| A-11 | **fixed** | `evidence_path()` writes `<pid>.attempt-N.json` and is used for candidates, bank reports, integration and tests. Structured bank reports are written. The completion report counts verdicts from the stores ("A CONFIRMED 145, F CONFIRMED 47, F REFUTED 1"; I recomputed the same) and records the lost P2-LEE log as unrecoverable (line 300). | Yes: `test_attempt_evidence_never_overwrites_fingerprinted_files` |

Notes:

- **Fixed or acted on:**
  - N-01: `.venv-jax` resume command.
  - N-02: see A-01.
  - N-03: seq 722–725, with the stated caveat that the fingerprints are of current versions.
  - N-04: two coverage tests.
  - N-05: G5 wording.
  - N-06: structured bank report.
  - N-07: G34.
  - N-08: caveat claim `CLM_DOI_10_1016_J_ACTAASTRO_2022_08_029_EB5F4015`, which agrees with Kang p.6 as I read it.
  - N-09: `tools/session_replay_check.py` and `session_replay_2026-10-16.json` (`no_op: true`).
  - N-10: role-limits section in `AGENTIC_ROLES_2026-10-16.md`.
  - N-11: seq 648.
  - N-12: reports regenerated.
- **Accepted:** N-13.

### Residual notes (non-blocking)

| ID | Location | Note | Suggested action |
|---|---|---|---|
| R-1 | `agentic_packet.py:447` (`log["bank_mismatches"]`); `bank_extraction.py --report-out` | The S3 block on bank mismatches is read in code but not exercised by a test. The harvest-level report keys are tested. | Factor the gate (as `integrity_failures` was) and unit-test it with a synthetic report. |
| R-2 | `models.json` `source_review.equation_links` | `equation_links` is descriptive. Links are created from `supports_model.equation_labels`, and nothing checks the two agree. They agree today for all four models. | Report any disagreement in the harvest. |
| R-3 | `harvest_extraction.py` (re-bank of an existing ID) | Re-banking can add assumptions, validity or description to a record without resetting its earlier independent verification (the verdict hash covers the verdict and source, not the record content). For the three SP-125 corrections I re-checked the added text on the pages and it is correct. | Consider flagging records whose content changed after verification. |
| R-4 | New F objects created by the fixes (e.g. `CLM_DOI_10_1016_J_ACTAASTRO_2022_08_029_EB5F4015`) | These are pending independent verification. I read the page and saw no discrepancy, but I did not add verdicts, so the applied verdict file stays unchanged. | Include them in the next verification pass. |
| R-5 | `.git/index` | The staged snapshot predates the fixes and both audit files. | The owner must re-stage before committing, including `FINAL_OPUS_AUDIT.md` and `reviews/A-audit-verdicts.json`. |

### Open gaps that remain (documented, not defects)

- K7: single-document comparison links. SP-125's Bartz equations 4-13 to 4-16 are still not linked to `MODEL_THERMAL_BARTZ`.
- K9: claims in model-targeted results SUPPORT regardless of fidelity.
- K1, K3–K6 and K8.
- Three P5 models have no structured review: `nozzle.rao_top_bezier`, `pump.inducer_npsh`, `pump.meanline_coefficients`.
- 41 QUEUED batch-1 packets.
- The code-versus-source discrepancies G1–G23 await owner decisions; `raosim/` was out of scope.
- Corpus and equation completeness are `NOT_ESTABLISHED`.
- No physical validation is established for any model.
- The overwritten P2-LEE BLOCKED evidence is unrecoverable; this is recorded.
