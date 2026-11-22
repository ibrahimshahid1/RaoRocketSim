# MOC and CFD plan: literature and implementation audit

21 November 2026. **The plan is useful, but should not be executed or promoted unchanged.** The review found concrete equation/monitoring errors and several acceptance rules that cannot establish the claims assigned to them. It also confirmed important parts of the proposed physics, including the KL coefficient correction. Missing evidence is distinguished from a demonstrated numerical defect throughout.

The requested order was followed: all nine supplied PDFs were copied, converted with local Docling, linked to figures/pages, indexed and checked through the knowledge graph **before** the scientific review agents began. This audit does not change production physics, coefficients, defaults or acceptance tolerances, and does not choose the plan's outstanding user decisions.

## Literature delivery

[Ingestion and read-back evidence](../../../knowledge/reports/moc_cfd_docling_import_2026-11-21.json) records source SHA-256 identities, Markdown hashes, page coverage, images, graph IDs, retrieval probes and completed tests.

| Supplied source | PDF pages | Detected figure crops |
|---|---:|---:|
| Giles–Pierce 2001 | 19 | 4 |
| Broer 1958 | 7 | 0 |
| Celik et al. 2008 | 4 | 2 |
| Lovely–Haimes 1999 | 9 | 22 |
| Giles–Ulbrich 2010, **Part 2** | 17 | 7 |
| Salas 1976 | 6 | 10 |
| Hall 1962 | 22 | 1 |
| Kliegel–Levine 1969 | 4 | 8 |
| Onofri–Paciorri, *Shock Fitting* (2017 edited volume) | 234 | 142 |
| **Total** | **322** | **196** |

There are also 322 original-page facsimiles: all 518 local figure/page links were read back, and all nine documents were successfully retrieved. These documents contributed 641 indexed chunks. Figure nodes link to images, parser runs and source Documents. A figure count is an automatic layout-detection count, not a certification that every diagram was segmented correctly; the complete page facsimiles cover missed figures.

Bulk Docling layout/OCR/table conversion disables formula-model enrichment. Parsed equations remain unreviewed. The original PDFs and page images remain the authority. Kliegel's first page contains preceding-article material, and the supplied Giles2010 file is Part 2 rather than Part 1. The supplied Celik paper has four PDF pages; page references from a longer hosted wrapper must not be copied onto it.

Reusable entry points are `knowledge/tools/docling_import.py` and `docling_register.py`; their use is documented in `knowledge/AGENTS.md`. Existing scientific support and verification states were preserved rather than automatically promoted by parsing.

## Most consequential findings

| Finding | What is wrong or missing | Required correction |
|---|---|---|
| **CFD-01** | The zero-viscosity shockFluid case enables standard `forcesCompressible`, whose pinned implementation asks for the transport model that this solver branch deliberately does not construct. | Replace it with a pressure-only, source-verified integration and include all monitors in a runtime dry run. This is a source-predicted incompatibility, not an observed OpenFOAM run. |
| **AV-01** | The plan puts the combustion-efficiency multiplier around the BL loss as well as inviscid performance; TDK's displayed source equation places it only on the first term. | Correct the transcription and state units/mass-flow/loss conventions before V0 implementation. Independently checked by two reviewers. |
| **CFD-03/04** | A raw two-level MOC delta is called an uncertainty bound, and different KL/full-convergent problems are judged using a discretization-only comparison band. | Separate refinement sensitivity, justified numerical error, starting-model error, and geometry/boundary mismatch. |
| **TS-01** | T1 has no independent admission budget for its approximate variable-gas transonic starting data. | Separate smooth-core verification from transonic initialization admission/benchmarking. Agreement of two approximate starts is insufficient. |
| **AV-03/04** | A scalar thrust correction does not define the new mass-flow closure or its ambient-pressure behavior. | Return consistent thrust and discharge/sonic flux; bind geometry/gas/start-line identity and apply ambient pressure through a declared control volume. |
| **CFD-05/06/07** | GCI can receive undefined/negative-order input; a same-triple asymptotic ratio is partly algebraic; fixed drift thresholds do not control iterative contamination of every observable. | Add explicit admissibility and unavailable states, order stability and per-observable iterative-error evidence. |
| **CFD-09/12** | The current constant-gas Nozzle-B writer does not generate TOP shocks or JANAF variable-cp cases and has no complete extraction implementation. | Add named case-family and extraction milestones rather than letting one CFD1 check stand for all regimes. |
| **CFD-10/11** | Mass-averaged Mach is compared on different surfaces; the wall-force cross-check uses the wrong sign and does not explicitly include inlet pressure. | Compare the same exit plane and implement the complete signed momentum balance. |

Other findings cover chemistry-coordinate normalization, property smoothing versus legacy parity, finite-rate equilibrium limits, shock admissibility, fold/onset interpretation, weak-coalescence policies, derivative convergence, BL feasibility labels, and capability-specific promotion. The specialist reports give exact plan/code locations, primary PDF pages, source restrictions, confidence and concrete repairs.

## Detailed reviews

- [Thermodynamics, transonic starting data and geometry](thermo_startline.md), with [structured findings](thermo_startline.json).
- [Shock fitting and differentiability](shock_differentiability.md), with [structured findings](shock_differentiability.json).
- [CFD, extraction and discretization uncertainty](cfd_uncertainty.md), with [structured findings](cfd_uncertainty.json).
- [Viscous coupling, engine architecture and integration](architecture_viscous.md), with [structured findings](architecture_viscous.json).

## Coverage of planned phases

| Phases | Assessment |
|---|---|
| C0, M0a/M0b | Keep the provider/refusal architecture; specify common thrust, mass, energy, reference area and loss accounting. A thrust-only table is incomplete. |
| G0, B0 | Explicit geometry conventions and source-bound benchmark assets are appropriate. LEA's inferred length definition remains an inference; approval cannot turn it into independent source evidence. |
| Q0, X0 | Preserve existing results and open axis/lip-row tests. A two-grid difference is refinement evidence, not automatically a remaining-error bound. |
| K0a/K0b | The original KL appendix confirms the sign correction and critical-speed normalization. Corrected-design reoptimization remains a separate unexecuted task. |
| N0, N1, N2 | A primal oracle and branch-aware derivative tests are appropriate. Include residual conditioning and numerical convergence; unchanged flags alone do not prove a valid derivative. |
| T0 | Consistent EOS identities are essential; smoothing the constitutive representation is a model change, so legacy/bitwise parity must be scoped. |
| T1 | Generalized smooth frozen-gas turning is supported. Starting-data error needs its own acceptance gate. |
| T2, T3 | Keep equilibrium and finite-rate acoustics separate. State chemistry coordinate units and the singular relaxation-limit test. |
| S0a, S0, S1 | Preserve fold refusal. Add front/admissibility/rotational-region closure and do not promote an unconverged discrete crossing to a resolved physical front. |
| S2 | Refusal of unsupported interactions and mixed-type regions is an appropriate initial scope. |
| V0 | The displaced-wall architecture is supported; E11b remains prescribed heuristic-displacement feasibility evidence. Repair the TDK loss factorization. |
| V1, V2 | Correlations remain onset screens with source validity masks; RSS/hysteresis/side-load histories require separate capabilities. |
| CFD0, CFD1 | Split runtime compatibility, extraction verification, matched smooth-flow comparison, full-convergent assessment, TOP shock case and variable-cp case. |
| A0, D0 | Admit a named capability/domain, with its own completed dependencies. A default smooth constant-gas branch need not wait for every reacting feature, but cannot imply their completion. |

## Recommended revision order

1. Repair the plan's confirmed source/algebra discrepancies, including the TDK factor, signed shock criterion, chemistry normalization and flux comparators. Update acquisition status and exact-copy page references.
2. Strengthen C0 around a single physical state and consistent thrust/mass/energy/area/loss contracts. Retain X0, source-qualified G0 and the opt-in corrected K0 work with its separate re-solve decision.
3. Repair the CFD monitors/runners and implement typed extraction. First verify a matched constant-gas smooth case; assess the full-convergent/KL model separately.
4. Establish N0 and T0, then N1/N2 and T1 with their own convergence and starting-data budgets. Keep conditioning, topology and property-representation limits explicit.
5. Add independent TOP-shock and variable-cp CFD case families. For S1 require admissible jump roots, front geometry and rotational downstream transport; quantify any weak-coalescence approximation separately.
6. Implement a real BL operator and its initialization/transition/thermal data before calling prescribed-displacement feasibility a V0 success. Integrate displacement and exit deficits without duplicate losses.
7. Couple accepted providers into both engine residuals, test downstream propagation, then admit each capability through its own A0/D0 gate.

These are proposed plan repairs, not new arbitrary numerical tolerances or choices on the user's behalf.

## Refactor and verification record

The user completed a source-layout refactor during this task. The final implementation lives in `src/lrekit`; `src/raosim` is a compatibility shim. Historical plan/logbook names intentionally remain readable through [the relocation map](../../architecture/relocations_2026-11-21.json). New review locators use current canonical paths and fingerprints. A move or old name by itself is not a physics defect.

Initial knowledge checks passed before the move. The post-move run exposed **24 failed, 444 passed, 1 skipped**, chiefly old source roots and stale implementation locators. The completed catch-up now passes **470 knowledge tests, 1 skipped**, including two new relocation regressions. The relocated writer static suite also passes **4 tests**. These are focused suites, not a claim that every repository test passes.

The repair translates 1,453 software/constant records and 1,075 test locators through the explicit relocation map while preserving IDs and history. All 438 SoftwareComponents resolve against current AST definitions; changed implementations remain `NOT_RECHECKED` for their scientific comparison. All nine source hashes, 322 pages, 196 Figure image URIs, 518 links and retrieval probes were reverified after the move. The graph rebuild reports zero dangling edges. See [refactor repair evidence](../../../builds/moc_cfd_literature_audit_2026-11-21/knowledge_relocation_repair_result.json), [final knowledge test result](../../../builds/moc_cfd_literature_audit_2026-11-21/knowledge_tests_relocation_repaired_result.json) and [relocated writer result](../../../builds/moc_cfd_literature_audit_2026-11-21/writer_tests_after_relocation.json). The implemented strategy is locator reconciliation, not a global scientific-identity migration.

No corrected BVP solve, OpenFOAM execution, mesh-convergence experiment, hot-fire test or physical qualification was performed. Primary-source review and algebraic diagnostics support the findings; passing package tests do not supply missing physical validation. The ingestion rebuild also reports source/CAD/provenance completeness warnings, which remain visible rather than being presented as a clean scientific validation.
