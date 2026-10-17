# Next knowledge session — resume point (written 2026-10-16)

## Continuation update — 2026-10-17 (takes precedence over the snapshot below)

Read `CONTINUATION_2026-10-17.md` for the current work and validation evidence.
K1, K7, K9, K10 and K11 are addressed; K12 content-freshness controls and the
pending Kang audit-fix verification are addressed. Parser lineage can now be
stored on all source-object classes, but individual K8 records still need
review. The three models listed in section 5 below now have current structured
reviews; this does not qualify their physics.

V-SP125-p104 and V-SP125-p107 are now FINAL_AUDIT_ELIGIBLE: the two inferred
claims were refuted and narrowed; replacements were independently confirmed.
V-SP125-p143 is also FINAL_AUDIT_ELIGIBLE: a conditional equivalent-area statement
was narrowed to preserve SP-125's wording; its replacement was independently
confirmed. The queue is 63 FINAL_AUDIT_ELIGIBLE and 38 QUEUED. The next queued
packet is V-HAG-p001. Continue one bounded packet at a time and bank each result.
Do not reopen the completed three model reviews merely because the historical
section 5 below lists them as pending.

Before further literature expansion, review `graph/graph.json`'s explicit
`candidate_link_issues`: the strengthened checks withhold 54 historical
references (at this update) without deleting their stored IDs. The prior 28
unlinked equations, the remaining K5 formulas, and G1–G23 engineering
decisions remain open. No physical validation is established.

SP-125 Eq.4-26 is now banked with its `T_cc` correction, p.116 definitions and
hydrogen exception; see the current continuation report. The historical K5/K8
rows below predate that correction. The six new pump source objects are also
independently confirmed; full pump performance remains unvalidated.

For the supplied consolidated technical review, read
`SUBSYSTEM_LITERATURE_STATUS_2026-10-17.md` and its JSON evidence snapshot.
Only 7/61 models have current structured code/source reviews (including adverse
findings); 12 have positive current citation assessments, one citation was not
found, and 48 are not checked under current fingerprint requirements. No complete
subsystem has comprehensive literature verification or established physical
validation. A verified source claim does not renew an entire model review.

New independent verdicts should include the assertion `content_sha256`, not
only the source PDF hash. Historical stored confirmations without this binding
remain separately visible as LEGACY_CONTENT_NOT_BOUND. Repeat parser QA retains
history; use the updated session replay tool's `success` result, which requires
both no mutation and all steps resolved.

No staging, commits or pushes were performed in this continuation. The old
staged snapshot remains stale. Recheck the current working-tree path list
before any eventual staging; do not use the 2026-10-16 list as an exact new list.

Read this file, then `knowledge/reports/REMAINING_SCIENTIFIC_GAPS.md` and
`knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md`. The session record
is `knowledge/workflows/results/agentic-2026-10-16/`. The append-only role/model
journal is `knowledge/reports/agentic_work_journal.jsonl`, and the packet state
machine is `knowledge/reports/packet_queue.json`.

## 1. Resume commands

```bash
# environment: knowledge tools run with the repository's JAX venv
.venv-jax/bin/python knowledge/tools/packet_queue.py summary
.venv-jax/bin/python knowledge/tools/packet_queue.py list --state QUEUED        # remaining packets, highest priority first
.venv-jax/bin/python knowledge/tools/packet_queue.py list --state RAW_CAPTURED  # awaiting S2
.venv-jax/bin/python knowledge/tools/packet_queue.py list --state VERIFIED      # awaiting S3
knowledge/kg.sh status
knowledge/kg.sh pending --limit 3

# one packet through the roles (S1 -> S2 -> F -> S3); see knowledge/workflows/AGENTIC_ROLES_2026-10-16.md
.venv-jax/bin/python knowledge/tools/packet_queue.py next QUEUED DOCLING_RUNNING --role S1 --model <model> --min-priority <p>
.venv-jax/bin/python knowledge/tools/agentic_packet.py s1-run <PACKET> --model <model>
.venv-jax/bin/python knowledge/tools/agentic_packet.py s2-prepare <PACKET> --model <model>
#   F: write reviews/<PACKET>.extraction.jsonl (+ .parser.json, + .verdicts.json), then
.venv-jax/bin/python knowledge/tools/packet_queue.py transition <PACKET> OPUS_SOURCE_REVIEW --role F --model <model> --evidence <comparison.json>
.venv-jax/bin/python knowledge/tools/packet_queue.py transition <PACKET> VERIFIED --role F --model <model> --verdict <file> [--verdict <file> ...]
.venv-jax/bin/python knowledge/tools/agentic_packet.py s3-integrate <PACKET> --model <model>
.venv-jax/bin/python knowledge/tools/agentic_packet.py s3-test <PACKET> --model <model>
.venv-jax/bin/python knowledge/tools/packet_queue.py transition <PACKET> FINAL_AUDIT_ELIGIBLE --role F --model <model> --note "<log check>"

# full deterministic refresh, the knowledge suite, the replay no-op proof and the fresh-checkout proof
# (.venv is absent on this machine; the knowledge tools run with .venv-jax)
knowledge/kg.sh rebuild
.venv-jax/bin/python -m pytest knowledge/tests -q -p no:cacheprovider
.venv-jax/bin/python knowledge/tools/session_replay_check.py --since 2026-10-16 --output knowledge/reports/session_replay_2026-10-16.json
.venv-jax/bin/python knowledge/tools/fresh_checkout_check.py --output knowledge/reports/fresh_checkout_2026-10-16.json
```

Machine limits that applied this session (owner): one Docling process at a time; at
most two Sonnet workers while Docling or pytest is running; knowledge tests and
targeted regressions only (no engineering suites); no downloads. Docling page times
were 8–330 s; SP-125 pages took 115–240 s. Solda 2008 PDF p.3 needed a 1500 s
timeout. The laptop's clamshell sleep stops the worker timeout clock.

## 2. Packet queue state at handoff

See section 6 ("Queue snapshot"), which is regenerated at the end of the session.

## 3. Engineering unblock table

Status refers to literature only: whether the held, page-verified sources are
enough to implement the item. Engineering work and owner decisions remain separate.

| Item | Verified sources (ID, PDF pages) | Sources still missing | Status |
|---|---|---|---|
| (a) contour-dependent thrust | SP-125 `DOC_NTRS_19710019929` Fig. 4-14 p.100, worked example p.107; SP-8120 `DOC_NTRS_19770009165` Fig. 5(b) p.29 (both digitized, ±0.3° reading); Rao 1958 `DOC_DOI_10_2514_8_7324` method (no θ charts) | Rao 1960 (chart origin) | PARTLY: the charts are held and contradict the code tables (gaps G1–G3); the contour–thrust coupling itself needs the existing MOC/TOP machinery, not new literature |
| (b) multipoint missions | Gromski 2010 `DOC_GROMSKI2010` pp.2-3 (TR202 100–18.8 %, 4:1 from 75 %, Pc 700–130 psia, Isp 453–436 s, MR 6) and p.8 (5.3:1 overall, 98 % C* assumed); Tokudome 2010 `DOC_TOKUDOME2010` pp.1-3 (28 % demonstrated, >5 Hz; requirement 30 %, >1 Hz); Hu 2021 `DOC_DOI_10_1016_J_AST_2021_106729` pp.1-2 (constant injector Δp principle; DC-motor dynamics Eqs. 1-4); Son 2017 pp.9-10 (500 N 5:1 schedule, Table 7) | throttled-performance and stability maps for small pintle engines | PARTLY: requirement points and schedules are sourced; throttle-dependent η_c* and stability are not |
| (c) quasi-2-D cooling | Pizzarelli 2011 `DOC_DOI_10_1080_10407782_2011_578011` Eqs. 1-21 pp.6-10; validity pp.15-23 (Table 2: AR 1–16 errors; Vulcain AR 4.5–8.5) | Pizzarelli Eqs. 22-38 (pp.11-14, held, not yet banked); real-fluid EOS inputs (see d) | PARTLY |
| (d) real-fluid coolant properties and HTD | Nasuti & Pizzarelli 2021 `DOC_DOI_10_1016_J_SUPFLU_2020_105066` Eq. 9, K = 0.187 p.6; SP-125 Eq. 4-23 p.115, RP-1 C1 = 0.0214 p.119, supercritical Eq. 4-25 p.115; Bell 2014 `DOC_BELL2014` (CoolProp) and Huber 2009 `DOC_HUBER2009` (RP-1) held | coolant-specific C1 for methane/hydrogen/propane; Sieder & Tate 1936 | PARTLY: the HTD criterion is sourced; property surfaces are buildable from held Bell/Huber, which are not yet extracted |
| (e) valve and feed-line losses | Lee 2021 p.6 (elbow Eqs. 16-19; valve loss from commercial data), p.11 Table 2 (valve loss coefficient 0.579) | SP-8094, SP-8097, SP-8090, SP-8080 | BLOCKED (valve design literature not held) |
| (f) film-injector hardware | SP-8087 `DOC_NTRS_19730022965` pp.65-67 (orifice ring, selective sizing, barrier cooling, no convergent-section tubes), p.107 (2× capacity, +100 % flexibility), p.108 Table XI (procedure, η_m/η_g = 0.4), p.109 (minimum 0.381 mm orifices at 7.62 mm); Freeberg 2019 p.2 (pintle film via TMR/BF) | Hatch & Papell 1959 TN D-130 (Table XI basis; defines v_g/v_f); SP-8087 ref. 60 | PARTLY |
| (g) liner buckling as a hard constraint | SP-125 Eq. 4-29 (inelastic buckling; implemented earlier) | SP-8068 *Buckling Strength of Structural Plates* | BLOCKED |
| (h) configuration object binding | software only | — | N/A for literature |
| (i) 500 N pump path | Lee 2021 pp.6-8 (Eqs. 20-25, Cordier Eq. 23, empirical pump mass Eq. 25), p.10-11 (500 N design point); Park 2023 `DOC_DOI_10_1007_S42405_022_00561_5` pp.6, 10-11 (small cryogenic electric pump, Eqs. 10-11, 15-16); Son 2017 (500 N pintle) | low-specific-speed / partial-emission (Barske) pump texts | PARTLY |
| (j) sourced default electric-technology set | Lee 2021 Table 2 p.11 (+ Eqs. 26-32 pp.7-8, Eq. 38 p.10); Kwak 2018 pp.5, 8-9; Pavlov Rachov 2013 `DOC_DOI_10_2514_1_B34714` pp.4-5 (motor density on shaft output, Eq. 35; inverter on output, Eq. 37); Casalino 2019 and Solda 2008 (battery only); Park 2023 (motor/inverter, basis unstated) | none required for a choice | UNBLOCKED (literature); choosing and registering the default is an owner decision; the current default `repository_screening_v1` lies outside held evidence (G4) |
| (k) Rao/TOP chart tables | SP-125 Fig. 4-14 p.100, p.107 (θn 27.4°, θe 9.8° at ε 14, L_f 0.8); SP-8120 Fig. 5(b) p.29; digitized tables in `.../agentic-2026-10-16/p1/` | Rao 1960 (primary chart paper) | UNBLOCKED for replacing or annotating the tables with digitized reproductions (two held charts agree within 1.5°/0.8°); owner decision |

## 4. Section 10.4 (consolidated review) — twelve items

| # | Item | State | Evidence |
|---|---|---|---|
| 1 | Execution reach from code | PARTIAL (Codex) | `knowledge/kg/execution_reach.py` gives bounded AST evidence (61 models: 8 CODE_INSPECTED, 53 UNKNOWN); traced MDO run and Jacobian leverage are NOT_MEASURED |
| 2 | Credibility-debt report | PARTIAL (Codex) | `knowledge/reports/credibility_debt_2026-10-15.json` exists. This session page-verified the electric-drive constants behind most of the MDO objective (P2, gap G4), but the P0 items are not resolved (owner decision) |
| 3 | Remaining model reviews | PARTIAL (this session +4) | Fingerprinted Section 17 reviews written for `injector.son2017_movable_continuous_gap` (EQUIVALENT), `pump.electric_drive` (CONDITIONALLY_EQUIVALENT), `thermal.bartz` (EQUIVALENT, with locator, conversion and validity findings), `thermal.sieder_tate` (DIFFERENT_FORMULA). P1 classifies the `nozzle.rao_top_bezier` angle tables BAD_ATTRIBUTION. `pump.inducer_npsh` and `pump.meanline_coefficients` are NOT reviewed |
| 4 | Typed applicability | PARTIAL | Source-stated validity was banked as page-anchored statements (Pizzarelli AR table, Bartz D*/r_c ≤ ~3, Son 0.1 mm, SP-125 Eq. 4-23 regime); no new machine-readable bounds were generated this session |
| 5 | Validation objects | OPEN | No Experiment/Measurement/ComparisonRecord populated; held datasets (e.g. Pizzarelli Table 2, Kang Table 4/6, Gromski Fig. 2) are banked as constants or claims only |
| 6 | Configuration-bound objects | PARTIAL (Codex) | `knowledge/artifacts.json`: 17 EngineConfiguration, 70 CADArtifact; binding_state UNKNOWN where unproven |
| 7 | Manufacturing knowledge | PARTIAL (Codex) | 15 ManufacturingProcess reference plans; execution evidence unavailable |
| 8 | Standards as sources | NEWLY_FIXED | `knowledge/kg/standards.py`: 8 StandardSource records (ASME Y14.5/Y14.41/Y14.24 HELD; ISO 286, ISO 2768, ASTM E606, JCGM GUM, ASME PTC 19.1 BLOCKED_ON_STANDARD); ontology 1.2.0; test `test_standards_are_held_only_with_a_current_mapped_document` |
| 9 | Cross-source relations | PARTIAL | `relationships.json` holds 39 relations (37 DERIVED_FROM, 1 APPROXIMATES, 1 EQUIVALENT_TO). This session's cross-source comparisons (Bartz vs SP-125, Sieder–Tate vs SP-125, Rao charts vs code) are recorded in model reviews and gaps, not yet as CONTRADICTS/COMPETING edges |
| 10 | Tables and figures | PARTIAL | Typed Docling table/figure candidates with page bboxes and dereferenced captions; this session recorded 373 parser-candidate reviews against original pages across its packets. Digitized P1 curves exist as session files, not graph objects |
| 11 | Repository split | NEWLY_FIXED (tracking) / OPEN (CI) | `.gitignore` no longer hides `knowledge/`; `knowledge/tools/fresh_checkout_check.py` proves HEAD + changes passes the corpus-free suite. CI regeneration and release assets are not set up |
| 12 | Conversion quality | PARTIAL | Load-bearing pages of SP-125, SP-8120, SP-8087, Son, Lee, Bartz, Pizzarelli, Nasuti, Mirzamoghadam, Freeberg, Hwang, Kang, Park, Rachov, Kwak, Casalino and SP-8107 were visually verified this session; the 14 low-confidence conversions were not systematically revisited |

## 5. Pending model IDs (next Section 17 reviews, dependency-first)

1. `MODEL_NOZZLE_RAO_TOP_BEZIER`: bank a fingerprinted model review from the P1 record (`P1-rao-top-angle-chart-comparison.json`).
2. `MODEL_PUMP_INDUCER_NPSH`: SP-8052 `DOC_NTRS_19710025474`.
3. `MODEL_PUMP_MEANLINE_COEFFICIENTS`: SP-8109 `DOC_NTRS_19740020848` (page offset +12; see memory note).
4. Remaining NOT_CHECKED models: `knowledge/kg.sh pending`.

## 6. Queue snapshot (at handoff, after the final rebuild)

| State | Packets |
|---|---|
| FINAL_AUDIT_ELIGIBLE | 60 |
| QUEUED | 41 |

Open packets (not FINAL_AUDIT_ELIGIBLE), highest priority first:

| Priority | Packet | State | Document | Pages | Attempts | Reason |
|---|---|---|---|---|---|---|
| 97 | V-SP125-p104 | QUEUED | DOC_NTRS_19710019929 | 104-104 | 1 | independent original-page verification of pending September extraction objects: sample-calculation chamber vol |
| 96 | V-SP125-p107 | QUEUED | DOC_NTRS_19710019929 | 107-107 | 0 | independent original-page verification of pending September extraction objects: convergent volume and 80-perce |
| 95 | V-SP125-p143 | QUEUED | DOC_NTRS_19710019929 | 143-143 | 0 | independent original-page verification of pending September extraction objects: gas-generator turbine throat-a |
| 90 | V-HAG-p001 | QUEUED | DOC_DOI_10_2514_2_5354 | 1-1 | 0 | independent original-page verification of pending September extraction objects: altitude/flow adaptation claim |
| 89 | V-HAG-p003-005 | QUEUED | DOC_DOI_10_2514_2_5354 | 3-5 | 0 | independent original-page verification of pending September extraction objects: separation/side-load/dual-bell |
| 84 | M-SP125-p116-118 | QUEUED | DOC_NTRS_19710019929 | 116-118 | 0 | Section 17 model audit: SP-125 Eqs. 4-27..4-32 tube/liner stress and cooling-passage pressure drop for MODEL_S |
| 80 | V-WHITE-p822 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 822-822 | 0 | independent original-page verification of pending September extraction objects: boundary-layer thickness claim |
| 79 | V-WHITE-p826-829 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 826-829 | 0 | independent original-page verification of pending September extraction objects: momentum integral and displace |
| 78 | V-WHITE-p837-839 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 837-839 | 0 | independent original-page verification of pending September extraction objects: transition and turbulent flat- |
| 77 | V-WHITE-p841-843 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 841-843 | 0 | independent original-page verification of pending September extraction objects: one-seventh-power law and drag |
| 76 | V-WHITE-p846 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 846-846 | 0 | independent original-page verification of pending September extraction objects: flat-plate summary claim |
| 75 | V-WHITE-p905 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 905-905 | 0 | independent original-page verification of pending September extraction objects: alternate Prandtl relations cl |
| 70 | V-SP8089-p017-018 | QUEUED | DOC_NTRS_19760023196 | 17-18 | 0 | independent original-page verification of pending September extraction objects: stability sensitivity claims |
| 69 | V-SP8089-p022-023 | QUEUED | DOC_NTRS_19760023196 | 22-23 | 0 | independent original-page verification of pending September extraction objects: unlike-impinging element claim |
| 68 | V-SP8089-p032-034 | QUEUED | DOC_NTRS_19760023196 | 32-34 | 0 | independent original-page verification of pending September extraction objects: pintle element claims |
| 67 | V-SP8089-p036 | QUEUED | DOC_NTRS_19760023196 | 36-36 | 0 | independent original-page verification of pending September extraction objects: slot pintle claims |
| 66 | V-SP8089-p039 | QUEUED | DOC_NTRS_19760023196 | 39-39 | 0 | independent original-page verification of pending September extraction objects: triplet/doublet wall compatibi |
| 65 | V-SP8089-p043-045 | QUEUED | DOC_NTRS_19760023196 | 43-45 | 0 | independent original-page verification of pending September extraction objects: instability classes and pressu |
| 64 | V-SP8089-p046-048 | QUEUED | DOC_NTRS_19760023196 | 46-48 | 0 | independent original-page verification of pending September extraction objects: orifice discharge coefficient  |
| 63 | V-SP8089-p049-050 | QUEUED | DOC_NTRS_19760023196 | 49-50 | 0 | independent original-page verification of pending September extraction objects: orifice flow separation and L/ |
| 62 | V-SP8089-p053 | QUEUED | DOC_NTRS_19760023196 | 53-53 | 0 | independent original-page verification of pending September extraction objects: orifice drilling tolerance cla |
| 61 | V-SP8089-p087 | QUEUED | DOC_NTRS_19760023196 | 87-87 | 0 | independent original-page verification of pending September extraction objects: impingement-angle design crite |
| 60 | V-SP8089-p089 | QUEUED | DOC_NTRS_19760023196 | 89-89 | 0 | independent original-page verification of pending September extraction objects: pintle equivalent impingement  |
| 59 | V-SP8089-p091-093 | QUEUED | DOC_NTRS_19760023196 | 91-93 | 0 | independent original-page verification of pending September extraction objects: pressure-drop/stability design |
| 58 | V-SP8089-p099 | QUEUED | DOC_NTRS_19760023196 | 99-99 | 0 | independent original-page verification of pending September extraction objects: flow-distribution claim |
| 57 | V-SP8089-p102 | QUEUED | DOC_NTRS_19760023196 | 102-102 | 0 | independent original-page verification of pending September extraction objects: injector structural natural fr |
| 56 | V-WHITE-p750 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 750-750 | 0 | independent original-page verification of pending September extraction objects: incompressible orifice formula |
| 55 | V-WHITE-p1077-1079 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 1077-1079 | 0 | independent original-page verification of pending September extraction objects: sonic flow pressure ratio and  |
| 54 | V-WHITE-p1091-1092 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 1091-1092 | 0 | independent original-page verification of pending September extraction objects: adiabatic/isentropic critical- |
| 53 | V-WHITE-p1098 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 1098-1098 | 0 | independent original-page verification of pending September extraction objects: sonic throat claim |
| 52 | V-WHITE-p1100-1102 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 1100-1102 | 0 | independent original-page verification of pending September extraction objects: maximum mass flow and area-rat |
| 51 | V-WHITE-p1120-1122 | QUEUED | DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI | 1120-1122 | 0 | independent original-page verification of pending September extraction objects: converging nozzle example clai |
| 49 | V-RAD21-p007 | QUEUED | DOC_DOI_10_1080_13647830_2021_1937702 | 7-7 | 0 | independent original-page verification of pending September extraction objects: spray-combustion evaporation c |
| 48 | V-RAD21-p013 | QUEUED | DOC_DOI_10_1080_13647830_2021_1937702 | 13-13 | 0 | independent original-page verification of pending September extraction objects: Eq. 16 LOX droplet evaporation |
| 47 | V-RAD21-p027-029 | QUEUED | DOC_DOI_10_1080_13647830_2021_1937702 | 27-29 | 0 | independent original-page verification of pending September extraction objects: evaporation result claims |
| 44 | V-LEE-p010 | QUEUED | DOC_DOI_10_1007_S42405_020_00325_Z_00D2425E | 10-10 | 0 | independent original-page verification of pending September extraction objects: validation-scope claim |
| 43 | V-LEE-p012-013 | QUEUED | DOC_DOI_10_1007_S42405_020_00325_Z_00D2425E | 12-13 | 0 | independent original-page verification of pending September extraction objects: Table 4 mass and future-work c |
| 40 | V-CAD-p004 | QUEUED | DOC_CAD_04 | 4-4 | 0 | independent original-page verification of pending September extraction objects: Bezier/B-spline continuity cla |
| 39 | V-CAD-p026 | QUEUED | DOC_CAD_04 | 26-26 | 0 | independent original-page verification of pending September extraction objects: clamped knot sequence claim |
| 38 | V-CAD-p040-042 | QUEUED | DOC_CAD_04 | 40-42 | 0 | independent original-page verification of pending September extraction objects: convex hull claim and knot-cou |
| 36 | V-CAD-p116 | QUEUED | DOC_CAD_04 | 116-116 | 0 | independent original-page verification of pending September extraction objects: knot removal claim |

All 60 packets with session work are FINAL_AUDIT_ELIGIBLE, including the 7 audit-correction packets, and none is mid-pipeline. The 41 QUEUED packets are batch-1 verification and model-audit packets. They are left resumable, and the stopping rule did not require them.

## 7. Open items carried forward

### 7.1 F-banked equations without a Docling-candidate link (28)

Each of these is a verified F transcription. The missing piece is only the DERIVED_FROM or CORRECTS link to the parser candidate it was compared with. On 26 of them, formula candidates exist on the page. Write a linkage packet the same way as `L-FREEBERG-p002`: re-bank the identical record with `enrichment_candidate_ids` or `corrects_candidate_ids`.

On the two SP-8087 Table XI rows, `page` is the p.107 anchor page. The forms are printed on PDF p.108, and the description says so.

| Document | PDF page | Eq. | Equation ID | Candidates on page |
|---|---|---|---|---|
| DOC_DOI_10_1007_S42405_020_00325_Z_00D2425E | 10 | 38 | `EQ_DOI_10_1007_S42405_020_00325_Z_00D2425E_F56CC231` | formula candidates on page |
| DOC_DOI_10_1007_S42405_022_00561_5 | 10 | 16 | `EQ_DOI_10_1007_S42405_022_00561_5_398B2283` | formula candidates on page |
| DOC_DOI_10_1016_J_AST_2018_02_033 | 5 | 1 | `EQ_DOI_10_1016_J_AST_2018_02_033_0DE63135` | formula candidates on page |
| DOC_DOI_10_1016_J_AST_2018_02_033 | 5 | 2 | `EQ_DOI_10_1016_J_AST_2018_02_033_5DBFBEEC` | formula candidates on page |
| DOC_DOI_10_1016_J_AST_2018_02_033 | 5 | 3 | `EQ_DOI_10_1016_J_AST_2018_02_033_7C2BEECC` | formula candidates on page |
| DOC_DOI_10_1016_J_AST_2018_02_033 | 5 | 4 | `EQ_DOI_10_1016_J_AST_2018_02_033_C980062E` | formula candidates on page |
| DOC_DOI_10_2514_1_35074 | 3 | 27 | `EQ_DOI_10_2514_1_35074_AEA9A267` | formula candidates on page |
| DOC_DOI_10_2514_1_35074 | 3 | 29 | `EQ_DOI_10_2514_1_35074_AEBAAEAE` | formula candidates on page |
| DOC_DOI_10_2514_1_35074 | 3 | 30 | `EQ_DOI_10_2514_1_35074_8531FA93` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 33 | `EQ_DOI_10_2514_1_B34714_111EED25` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 34 | `EQ_DOI_10_2514_1_B34714_E0F0D66D` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 35 | `EQ_DOI_10_2514_1_B34714_83DE1395` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 36 | `EQ_DOI_10_2514_1_B34714_1947394A` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 37 | `EQ_DOI_10_2514_1_B34714_27608400` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 38 | `EQ_DOI_10_2514_1_B34714_C8BF2FB1` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 39 | `EQ_DOI_10_2514_1_B34714_C230EB1F` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 40 | `EQ_DOI_10_2514_1_B34714_E1732F38` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 41 | `EQ_DOI_10_2514_1_B34714_EC6957D1` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 42 | `EQ_DOI_10_2514_1_B34714_E1CE772C` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 46 | `EQ_DOI_10_2514_1_B34714_BBFDF57C` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 47 | `EQ_DOI_10_2514_1_B34714_8C420699` | formula candidates on page |
| DOC_DOI_10_2514_1_B34714 | 4 | 48 | `EQ_DOI_10_2514_1_B34714_5EAF9606` | formula candidates on page |
| DOC_DOI_10_3390_AEROSPACE6030036 | 5 | 14 | `EQ_DOI_10_3390_AEROSPACE6030036_E898278C` | formula candidates on page |
| DOC_DOI_10_3390_AEROSPACE6030036 | 5 | 15 | `EQ_DOI_10_3390_AEROSPACE6030036_5E4F7409` | formula candidates on page |
| DOC_DOI_10_3390_AEROSPACE6030036 | 6 | 17 | `EQ_DOI_10_3390_AEROSPACE6030036_C4F9C13A` | formula candidates on page |
| DOC_DOI_10_3390_AEROSPACE6030036 | 6 | 18 | `EQ_DOI_10_3390_AEROSPACE6030036_C403B415` | formula candidates on page |
| DOC_NTRS_19730022965 | 107 | Table XI gaseous region (1) | `EQ_NTRS_19730022965_E1C6F3BC` | no formula candidate on page |
| DOC_NTRS_19730022965 | 107 | Table XI liquid region (1) | `EQ_NTRS_19730022965_3242BDAD` | no formula candidate on page |

### 7.2 Rejected formula candidates with no corrected record (gap K8)

- SP-125 Eq. (4-26), `CAND_e5b9293d638533e465c7b8a9` (p.115).
- p.95 sample lines `CAND_69e6c31ad567d5200dd4ae0a` and `CAND_d266d7384140dede2256b853`.
- Sample-calculation lines rejected for unit or symbol damage on pp.111, 112 and 119. Their values are banked as constants, which cannot carry candidate links.

### 7.3 Order of work for the next session

1. Read `FINAL_OPUS_AUDIT.md`. The first verdict was FAIL_WITH_FINDINGS (1 major, 10 minor), all fixed in the repair round, and the same Opus A rechecked them; its Recheck section is the authority. Apply anything it left open. Use `knowledge/reports/OPUS_A_AUDIT_BRIEF_2026-10-16.md`, then apply its `A-audit-verdicts.json` with `apply_verification`; that completes hop 4 of the traced flow. Then fix findings and have the same agent recheck.
2. Write a `MODEL_NOZZLE_RAO_TOP_BEZIER` structured review from the P1 record. Then `MODEL_PUMP_INDUCER_NPSH` and `MODEL_PUMP_MEANLINE_COEFFICIENTS`.
3. Design a multi-source model review (gap K7), so SP-125's Bartz and RP-1 equations can be cited by their models without replacing the single-document comparison links.
4. Run the linkage packets for §7.1, then continue the QUEUED packets from the top of §6, one bounded packet at a time.

### 7.4 Residual notes from the final audit (Opus A recheck, PASS_WITH_OPEN_GAPS)

These are non-blocking and recorded as gaps K10–K12:
- R-1: add a regression test for the S3 bank-mismatch gate.
- R-2: check each review's `equation_links` against the links actually created.
- R-3: flag records whose content changed after independent verification.
- R-4: verify the objects F created in the repair round, for example the Kang p.6 caveat claim.
- R-5: the owner must re-stage before committing (`git add .gitignore docs/LITERATURE_REQUESTS.md knowledge`), because the staged snapshot predates the repair round.
