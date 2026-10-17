# Remaining scientific gaps — 2026-10-16 agentic continuation

**2026-10-17 continuation:** K1, K7, K9–K11 are addressed, and K12 now has
content-bound verification controls plus the independent Kang audit-fix verdict.
The three missing structured model reviews are banked and current. K8's storage
and identity-checking infrastructure is implemented; its individual source
lineage gaps remain open. See `CONTINUATION_2026-10-17.md` and the current
`NEXT_KNOWLEDGE_SESSION.md` update; the dated tables below preserve the original
audit snapshot. G1–G34 and all physical-validation limitations remain open.

Reviewer: Opus F (claude-opus-5-5, main session). Every entry was compared against
the original PDF page (rendered with Ghostscript and read by F); the knowledge-graph
records that carry the page-verified source statements are listed under each item.

**No code was changed.** The owner's amendment for this session forbids edits to
`raosim/`, `lrekit/` and `tests/`, so every code-versus-source discrepancy below is
recorded for an owner or engineering decision. HEAD 550d5c2's MDO engineering work is
treated as intended behaviour and was not re-audited beyond the comparisons requested
(P1, P2, P3 and the first four P5 models).

Classification vocabulary (Section 17): `EXACT`, `EQUIVALENT`, `CONDITIONALLY_EQUIVALENT`,
`SUPPORTED_APPROXIMATION`, `REPOSITORY_EXTENSION`, `DIFFERENT_FORMULA`,
`BAD_ATTRIBUTION`, `BAD_LOCATOR`, `CONVERSION_ERROR`, `REPOSITORY_RULE`,
`VALIDITY_NOT_ENFORCED`, `SOURCE_INTERNAL_INCONSISTENCY`. Severity is the reviewer's
judgement of design consequence, not a measured sensitivity (none was run).

Evidence records:
- P1: `knowledge/workflows/results/agentic-2026-10-16/P1-rao-top-angle-chart-comparison.json`
- P2: `.../P2-electric-technology-sets-comparison.json`
- P3: `.../P3-constraint-provenance-comparison.json`
- P5: model reviews in `.../reviews/{M-SON2017-p003,V-LEE-p006-008,X-BARTZ-p001-005,M-SP125-p115}.extraction.jsonl`

## A. Code-versus-source discrepancies (open; owner decision)

| ID | Code (file:line) | Source (PDF, page) | Finding | Class | Severity |
|---|---|---|---|---|---|
| G1 | `raosim/nozzle_geometry.py:76-91` `_THETA_E_TABLE` (used by `lookup_angles`, :104) | SP-125 `propulsion_texts/19710019929.pdf` Fig. 4-14 PDF p.100 (printed 91) and worked example PDF p.107; SP-8120 `propulsion_texts/19770009165.pdf` Fig. 5(b) PDF p.29 | The table rises with ε for every length; both held charts fall with ε. It exceeds both charts by 2.7–15.8° over ε 8–50. At the only source-stated point (ε 14, L_f 0.8 → θe 9.8°, p.107) the code gives 16.7°. | BAD_ATTRIBUTION | high: exit angle sets the bell contour and divergence loss |
| G2 | `raosim/nozzle_geometry.py:58-73` `_THETA_N_TABLE` | same | θn is +1.6 to +9.3° high for L_f 60–90 %, within about 2° at 100 %. At ε 14, L_f 0.8 the code gives 31.6° against the source's 27.4°. | BAD_ATTRIBUTION (L_f < 100 %) | high |
| G3 | `raosim/nozzle_geometry.py` provenance comment ("verified against propulsion_texts") | Rao 1960 not held | The cited chart paper is not in the corpus; the held reproductions contradict the tables (G1, G2). | BAD_ATTRIBUTION | attribution |
| G4 | `raosim/electric_technology.py:293-303` `repository_screening_v1` (default set) | Lee 2021 PDF p.10-11; Kwak 2018 PDF pp.5, 8-9; Pavlov Rachov 2013 PDF pp.4-5; Park 2023 PDF p.6 | Inverter power density of 15 kW/kg is below every held value (26.64–60 kW/kg); inverter efficiency of 0.96 is above every held value (0.80–0.85). The constant discharge efficiency of 0.95 matches Lee Eq. (38) only near a 785 s burn. | REPOSITORY_RULE outside held evidence | medium (the electric package dominates the MDO objective) |
| G5 | `raosim/mdo/schema.py:640` `pump_specific_mass = 2.5e3` W/kg | Kwak 2018 PDF p.8-9 (15, 20 kW/kg pump only); Rachov 2013 PDF p.5 (22 kW/kg); SP-8107 `19750012398.pdf` PDF p.17 Table I (3.65–35.5 kW/kg for operational turbopump assemblies; the 179 kW/kg SSME row is marked not operational in footnote 3) | Below every held pump-only value, and Lee's own pump mass is the diameter fit Eq. (25) (PDF p.7). The bases differ, so no value transfers directly. | REPOSITORY_RULE (screening assumption) | medium |
| G6 | `raosim/mdo/constraints.py:223-225`, `raosim/mdo/schema.py:586` `blockage_factor_max = 0.90` | Freeberg 2019 PDF pp.5-6, 10 (0.3 < BF < 0.7, single row); Hwang 2022 PDF pp.2, 4 (Eq. 3 two-row BF; tip damage at 86 %), p.8; Kang 2022 PDF pp.3, 6, 9-10 (86 % tip damaged at 0.7 s; 58 % completed 5 s) | No held source states 0.90. Single-row evidence ends at 0.7. Values above 0.7 come from two-row pintles, whose BF sums both rows (Hwang Eq. 3), and include reported tip damage at 86 %. The row is already labelled `repository_rule`. | REPOSITORY_RULE above held evidence | medium |
| G7 | `raosim/mdo/constraints.py:257-261`, `schema.py:394` `liner_T_wg_max = 800 K` | Mirzamoghadam 1991 `mirzamoghadam1991.pdf` PDF p.4 (900 F = 755 K), p.6 (800 F = 700 K nominal channel with 3 % film), p.7 Table 1 (900 F) and Table 2 (1050 F, "Depends on Material") | 800 K is not stated; the code labels it a repository rule and its comment matches the page-verified values. The source values apply to an H2/O2 copper-alloy GG-cycle chamber. | REPOSITORY_RULE (correctly labelled) | low (documentation consistent) |
| G8 | `raosim/mdo/constraints.py:211-215` (aspect_ratio 8, cites Mirzamoghadam) | Mirzamoghadam PDF p.5 (maximum depth/width 7, manufacturing) and p.7 Table 2 (7); Table 1 (former criteria) 8 | The citation supports 8 only as the former design value; the present criterion in the same paper is 7. | PARTIALLY_SUPPORTED | low |
| G9 | `raosim/mdo/schema.py:332-337` (AR cap "validated to AR ≈ 8 (Pizzarelli 2011 …)") versus `raosim/mdo/constraints.py:208-210` ("a MANUFACTURING bound, not a model-validity bound") | Pizzarelli 2011 `pizzarelli2011.pdf` Table 2 PDF p.18; text PDF pp.15, 20, 23 | The two comments give conflicting rationales. Pizzarelli compared AR 1, 2, 4, 8 and 16 against CFD: temperature-rise errors of 29.3, 19.0, 6.7, 1.2 and 3.7 %, pressure-drop errors of 18.8, 12.8, 2.2, 4.4 and 2.8 %. The Vulcain case spans AR 4.5–8.5. That quasi-2-D model is weakest at low AR, not above 8, and the repository's cooling march is not Pizzarelli's model. | BAD_ATTRIBUTION (validity rationale) | low–medium |
| G10 | `raosim/mdo/constraints.py:217-221` locator "Freeberg-2019-PDF10-conclusions" | Freeberg 2019 PDF p.10 | The 0.3–0.7 statement is in the discussion before the "VI. Conclusions" heading. | BAD_LOCATOR | low |
| G11 | `raosim/mdo/constraints.py:198-201` locator "Son-2017-…-Eq1-PDF4-PDF9" | Son 2017 `pintle_injector/son2017.pdf` PDF p.2 (Eq. 1, printed 859); PDF p.4 (Eq. 2 and the criterion) | Eq. (1) is on PDF p.2. | BAD_LOCATOR | low |
| G12 | `raosim/model_registry.py:113` (Son citation "Son, Hong, and Koo") | Son 2017 PDF p.1 | The authors are Min Son, Kanmaniraja Radhakrishnan, Jaye Koo, Oh Chae Kwon and Heuy Dong Kim. The equations and Table 2 benchmarks are correctly sourced. | BAD_ATTRIBUTION (citation metadata) | low |
| G13 | `raosim/mdo/schema.py:572-573` (quotes "the two master geometric knobs" from Hwang 2022) | Hwang 2022 `pintle_injector/s42405-022-00489-w.pdf`; Kang 2022 PDF p.3 | The quoted phrase does not occur in Hwang or Kang. The nearest wording is Kang p.3 "the most important factors in the design of the pintle injector". | QUOTE_NOT_IN_SOURCE | low |
| G14 | `raosim/mdo/schema.py:926-928` (BF mid-band sizing called "the literature design rule") | Freeberg PDF pp.5-6; Kang PDF p.2 | The mid-band of [0.30, 0.90] uses a repository upper bound, so the target is not a literature rule. 0.60 does lie inside TRW's 0.3–0.7 band. | OVERSTATED_ATTRIBUTION | low |
| G15 | `raosim/physics.py:110-127` `combustion_gas_viscosity` coefficient `1.1929e-7` | Bartz 1957 `technical-notes-1957.pdf` Eq. [9] PDF p.2; SP-125 Eq. 4-16 PDF p.110 (printed 101) | The exact SI conversion of 46.6e-10 lb/(in·s)·m^0.5·T_R^0.6 is 1.184081e-7 Pa·s·T_K^-0.6, so the code is 0.745 % high in μ (0.149 % in h_g). The docstring claims an exact conversion. The MDO path uses `mission.mu_gas` instead. | CONVERSION_ERROR | low |
| G16 | `raosim/physics.py:113` (cites "SP-125, Eq. 4-23"), `:104` (cites "SP-125, Eq. 4-18") | SP-125 PDF p.110: Eq. 4-15 (Pr) and Eq. 4-16 (μ); Eq. 4-18 is h_gc with a deposit; Eq. 4-23 is the coolant Nusselt relation (PDF p.115) | Wrong equation numbers. | BAD_LOCATOR | low |
| G17 | `raosim/physics.py:186-231`, `raosim/jax/thermal.py:44-55`, applied at every station in `raosim/mdo/cooling.py:339-357` | Bartz 1957 PDF p.2 (accuracy if the angles change by no more than 50 % from 30/15° and D*/r_c is not greater than about 3); PDF p.3 (chamber heat fluxes only 40–50 % of measured, RFNA-NH3) | The correlation transcription is correct (Eqs. [5], [7] EQUIVALENT). The source's applicability limits are not checked, and chamber stations use the same correlation that Bartz reports under-predicting chamber flux by half. | VALIDITY_NOT_ENFORCED | medium (cooling margins in the chamber) |
| G18 | `raosim/jax/thermal.py:58-66` `recovery_temperature` (r = Pr^(1/3)) | SP-125 PDF p.109 Eq. 4-10 (T_aw = (T_c)ns × recovery factor 0.90–0.98); PDF p.119 sample (0.923) | The standard compressible recovery form is used. Neither Bartz 1957 nor SP-125 states it; SP-125 defines a stagnation-referenced factor. | REPOSITORY_EXTENSION | low |
| G19 | `raosim/physics.py:466-492` `sieder_tate_coefficient`, `raosim/jax/thermal.py:69-76` `sieder_tate_hc`; registry `raosim/model_registry.py:508-516` | SP-125 Eq. 4-23 PDF p.115 (printed 106): Nu = C1 Re^0.8 Pr^0.4 (μ/μw)^0.14, C1 coolant-specific; RP-1 C1 = 0.0214 PDF p.119 | The code uses 0.027 Re^0.8 Pr^(1/3) (classical Sieder & Tate 1936, not held). Against SP-125's RP-1 form the code's h_c is higher by about 26 % at Pr 1, 17 % at Pr 3 and 7 % at Pr 11.7, which is non-conservative. The registry cites SP-125 as the local source for this form. | DIFFERENT_FORMULA / BAD_ATTRIBUTION | medium–high (coolant-side margin) |
| G20 | `raosim/mdo/schema.py:346` (SP-8087 "shall-language, p. 6082") | SP-8087 `19730022965.pdf` sec. 3.3, PDF p.107 (printed 91) | The locator is not a page of the source. The quoted text ("twice the estimated required quantity") is correct. | BAD_LOCATOR | low |
| G21 | `raosim/mdo/constraints.py:262-264` film_capacity source_id "NASA-SP-8087" | SP-8087 PDF p.107 | The citation has no section or page; the 2× criterion is sec. 3.3. The code applies 2× capacity on film fraction against `film_system_capacity_fraction` 0.60, a repository sizing assumption. | IMPRECISE_LOCATOR | low |
| G22 | `raosim/mdo/coolant_htd.py:13-19` (problem statement quoted to "sec. 2.2, source-PDF p. 6") | Nasuti & Pizzarelli 2021 `nasuti2021.pdf` PDF p.1 (abstract) | The quoted sentence is in the abstract. Eq. (9) and K = 0.187 are correctly on PDF p.6. | BAD_LOCATOR | low |
| G23 | Son model, MDO transition fraction `raosim/mdo/schema.py:564` (0.95) and module-diameter formula `raosim/mdo/pintle.py:207-209` | Son 2017 PDF pp.4, 9 | Both are repository choices, labelled as such. The equations agree with the source (EQUIVALENT). Printed Eq. (2) omits t_post (G24). | REPOSITORY_RULE | low |

## B. Source-internal inconsistencies (recorded; sources not corrected)

| ID | Source | Pages | Inconsistency |
|---|---|---|---|
| G24 | Son 2017 | PDF p.4 vs p.2-3 | Printed Eq. (2) uses R_post where Eq. (1) has R_post − t_post. The paper's own Table 2 transition openings (0.418/0.454/0.568 mm) follow the t_post form; the printed Eq. (2) would give 0.366/0.395/0.492 mm. The code follows the consistent form. |
| G25 | Son 2017 | PDF p.9 Table 5 vs p.10 Table 7 | At 100 % throttle the vaporization time is 8.83 ms and the distance 84.04 mm (Table 5), against 1.870 ms and 41.79 mm (Table 7). Table 5 repeats the velocity entries, probably a typesetting error. |
| G26 | Casalino 2019 | PDF p.5 vs p.6 Table 1 | Set A overall efficiency is 0.64 in the text and 0.68 in the table. |
| G27 | Kwak 2018 | Table 2 vs Table 3 | D-cell energy density is 198.5 vs 198 Wh/kg. |
| G28 | Kang 2022 | PDF p.3 Table 2 vs pp.9-10 | First-row AR is printed "3.1" in Table 2 but "3:1" in the text. |
| G29 | Kang 2022 | PDF p.9 | The published captions of Fig. 28 and Fig. 29 have lost their abscissa symbols (TMR, OFR on the axes). |
| G30 | Hwang 2022 | PDF pp.4, 8 | Eqs. (2), (6), (7) print '1' where Δ is meant (e.g. "2ρa1pa", "1TMR"); the prose uses Δ. |
| G31 | Pizzarelli 2011 | PDF pp.10-11 Eqs. (22)-(23) | The text says the velocity peak lies in the lower part for B > 0, but Eq. (22) gives u(h) − u(0) = B·u_mh, so the peak is at the top for B > 0. With asymmetric heating, Eq. (24) gives B < 0 and the peak at the hot bottom, as the CFD shows. Not used by LREKit. |
| G32 | SP-125 | PDF p.111 sample calculation | sqrt(22.5) is printed as 4.76 (4.743), giving μ = 4.18e-6 against 4.144e-6 lb/(in·s). This affects only the sample arithmetic. |
| G33 | SP-125 | PDF p.119 Eq. (b); PDF p.120 | The printed expanded form of Eq. (b), (0.8d + 24.93)/(d + 0.04), omits the factor π that SP-125 itself uses on p.120. The p.120 tube results do not reproduce from Eqs. (b) and (d): F's check gives d = 0.847 in and N = 90.7, against printed d = 0.85 and N = 94.5; for N = 94, Eq. (b) gives d = 0.815 in against the printed 0.855. p.120 is not in a reviewed packet, and the code does not use this example. |
| G34 | SP-125 | PDF p.107 sample calculation 4-2 (`EQ_NTRS_19710019929_2DC4BFBD`) | The convergent-frustum volume is printed as 7760 in³. Its own inputs give (π/3)·12.4·[15.75² + 12.45² + 15.75·12.45] = 7780 in³ (+0.26 %). This is a source arithmetic slip like G32. Found by Opus A and recomputed by F. |

## C. Knowledge-system scientific gaps still open

| ID | Gap | State |
|---|---|---|
| K1 | SP-125 claims `CLM_…76A39C08` and `CLM_…9F1C0032` contain extractor inferences beyond the source. | Pending; route through V-SP125-p104/p107 packets (not processed this session) |
| K2 | RESOLVED this session. SP-125 Eqs. 1-14, 4-6 and 4-7 (`EQ_NTRS_19710019929_98E8ACF4`, `_7A8070BC`, `_16E1DC3A`) are superseded by page-verified corrections (`_C43FA39B`, `_0BBBD3E9`, `_2AD5D2F0`); the old records are active false / SUPERSEDED_EXTRACTION. Audit A-03 found that the corrections had dropped the legacy assumptions, validity and model link. Packets R-SP125-p014-015 and R-SP125-p098-099 now carry the page-verified assumptions, and the p.98–99 conical-nozzle ranges labelled as typical-design ranges rather than validity limits, each with a recorded `carry_forward` decision. The `MODEL_CHAMBER_SHOULDER_FILL_FRACTION` link is deliberately not carried, so that model has **no current equation link**. It is a repository fillet-radius heuristic (Rs = 0.8 Rs,max) that implements none of these equations; the legacy link was extraction context. | V-SP125-p014-015, V-SP125-p098-099, R-SP125-p014-015, R-SP125-p098-099 |
| K3 | Pizzarelli's quasi-2-D model is page-verified (Eqs. 1-21, validity) but not implemented; the cooling march is 1-D with constant coolant properties. | Unblock item (c) PARTLY |
| K4 | HTD screen (Nasuti Eq. 9) is implemented but UNAVAILABLE without real-fluid β/c_p. Huber 2009 (RP-1 properties) is held but not extracted. | Unblock item (d) PARTLY |
| K5 | Not yet banked or defined: Pizzarelli Eqs. 22-38 (velocity profile, wall conduction), SP-125 Eqs. 4-19 to 4-22 (PDF pp.113-114) and Eq. 4-26, and the SP-8087 Table XI symbols v_g, v_f. Lee Eq. (25) pump mass is banked (`EQ_DOI_10_1007_S42405_020_00325_Z_00D2425E_462E6827`, V-LEE-p006-008). SP-125 Eqs. 4-10 to 4-18 and the p.119 RP-1 forms were banked late in this session. | Next session |
| K6 | Rao 1960 (Rao/TOP chart origin) and Sieder & Tate 1936 are not held; both are cited by code. | Literature request |
| K7 | A model-targeted extraction replaces a model's `source_comparison_equation_ids` and support state wholesale (`knowledge/kg/harvest_extraction.py`), so a model carries comparison links from one document only. Partly addressed after audit A-01: labels may now name equation ids or printed numbers, so same-document equations can be cited (the SP-125 p.119 RP-1 form now has a CITES link from MODEL_THERMAL_SIEDER_TATE). SP-125's own Bartz equations (Eqs. 4-13 to 4-16, another document) are still not linked to MODEL_THERMAL_BARTZ. | Multi-source model review in knowledge/kg |
| K8 | Rejected Docling formula candidates with no corrected record: SP-125 Eq. (4-26) `CAND_e5b9293d638533e465c7b8a9` (p.115) and the p.95 sample lines `CAND_69e6c31ad567d5200dd4ae0a` and `CAND_d266d7384140dede2256b853`. Only equations accept candidate links, so sample-calculation constants cannot record which candidate they correct. | Next session |
| K9 | Claims banked by a model-targeted result SUPPORT that model whatever the review's fidelity. For example, SP-125 p.115 claim `CLM_NTRS_19710019929_79A70B82` (coolant pressure 0.3–0.7 of critical) SUPPORTS MODEL_THERMAL_SIEDER_TATE although that review is DIFFERENT_FORMULA. Claim links should follow the same review gate as equation links. | Next session (knowledge/kg) |
| K10 | (audit R-1) The S3 block on bank mismatches (`knowledge/tools/agentic_packet.py`, `bank_mismatches` in `s3_integrate`) has no regression test. | Move the gate into its own function; test it with a synthetic bank report |
| K11 | (audit R-2) A review's `equation_links` list is descriptive: nothing checks it against the IMPLEMENTS/CITES links actually created. They agree today for all four reviewed models. | Have the harvest report disagreements as a bank mismatch |
| K12 | (audit R-3, R-4) Re-banking can add assumptions or validity to a record without resetting its earlier independent verification. Opus A re-checked the three affected SP-125 corrections. Objects F created in the repair round, e.g. the Kang caveat claim `CLM_DOI_10_1016_J_ACTAASTRO_2022_08_029_EB5F4015`, await independent verification. | Flag records whose content changed after verification; verify the new F objects |
