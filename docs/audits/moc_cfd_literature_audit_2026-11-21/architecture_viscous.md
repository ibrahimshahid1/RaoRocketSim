# Viscous coupling, engine architecture and phase integration review

Reviewed 21 November 2026. This is a review of the saved extension plan, its superseded default-solver roadmap, selected experiment scripts and current engine residuals. Findings about proposed work are requirements to repair the plan, not claims that an unimplemented solver has produced bad hardware. No production coefficients, defaults, acceptance tolerances or numerical results were changed.

## AV-01 — The displayed TDK delivered-performance equation changes the placement of the combustion-efficiency factor

**P1; confirmed transcription discrepancy; high confidence.**

Target: `moc_extensions_plan_2026-11-21/sections/06_g3.tex`, lines 5–8.

The plan writes `Isp_del = eta_DER Isp' (1 - DeltaF_BLM/F')`. Nickerson, Dang and Coats, *Engineering and Programming Manual: Two-Dimensional Kinetic Reference Computer Program (TDK)*, NASA CR-178628 (1985), NTRS 19860007470, PDF p.26, printed p.2-4, instead prints `Isp_del = eta_DER Isp_TDK' - DeltaF_BLM / mdot_TDK'`, in the source's specific-impulse/unit convention. PDF p.27, printed p.2-5, separately defines the exit boundary-layer thrust deficit and explains that it includes friction and heat-transfer effects.

Using `F' = mdot' Isp'` in that convention, the plan's expression additionally multiplies the boundary-layer subtraction by `eta_DER`. It agrees with the source only for special cases such as `eta_DER=1` or zero deficit. With specific impulse in seconds, include the declared `g0` conversion consistently. This is not merely a notation change.

**Repair:** transcribe the named source faithfully, with its unit and mass-flow conventions, or explicitly introduce a different project model and justify where its energy-release and viscous losses act. Test nonunity energy-release efficiency with a nonzero BL deficit. Keep a common ledger for displacement, exit deficit, wall heat transfer and the residual efficiency factor. The existing no-double-counting sentence is correct but does not resolve the incorrect displayed equation.

Original PDF used: `builds/moc_extensions_plan_2026-11-21/scout3_viscous_ad/downloads/tdk_19860007470.pdf`. Rendered source evidence: `builds/moc_cfd_literature_audit_2026-11-21/source_pages/tdk1985-p26.png` and `tdk1985-p27.png`.

## AV-02 — E11b does not establish that an integral boundary-layer/TDK coupling works at scale one

**P2; demonstrated evidence-scope mismatch; high confidence.**

Target: `sections/06_g3.tex`, lines 54–61 and 76; decision D9 in `sections/14_decisions.tex`. Code: `tests/experiments/nozzle/moc_ext_e11b_effective_throat.py` (formerly `scripts/moc_ext_e11b_effective_throat.py`), especially lines 55–67 and the `not_established` fields near the end.

The approach row calls the option “Integral BL plus displaced-wall second pass (TDK)” and says it “works at s=1.” E11b actually prescribes a displacement from the repository heuristic in `src/lrekit/components/cooling/screens.py:boundary_layer_displacement`, samples its effective-throat displacement at B, applies a project smoothing onset, and replays the inviscid march. It does not solve an integral BL, determine transition, compute the TDK momentum-thickness deficit, or close heat transfer. The script explicitly disclaims boundary-layer physics and loss magnitude. The plan partly preserves that limitation elsewhere; the approach table must preserve it too.

TDK PDF p.287, printed p.5-161, describes BLW as consuming displacement thickness **from BLM** and moving a given physical wall inward (or a designed real wall outward) normal to the contour, including the throat correction. That supports the architecture, not this heuristic profile. NASA TP-2725, *Experimental Evaluation of a High-Area-Ratio Rocket Nozzle* (1987), NTRS 19870015990, PDF p.7, printed p.5, additionally describes supplied transition and upstream BL/wall-temperature inputs. These are not supplied by a bare displaced contour.

**Repair:** call E11b a prescribed-displacement march-feasibility diagnostic. Separate V0 into a verified BL operator with initialization/transition/wall thermal inputs, geometry/edge-state coupling, and performance/energy reduction. Retain the fold result as a feasibility limitation of that diagnostic; do not infer that a physically calculated BL necessarily folds there or that a different coupling necessarily fixes it.

## AV-03 — A thrust-only MOC correction does not define a consistent engine mass-flow closure

**P1; missing contract required before M0 integration; high confidence.**

Target: `sections/09_geometry_arch.tex`, lines 33–50 and 59–61. Current code: `src/lrekit/optimization/engine.py:engine_residual`, lines 157–173.

The proposed M0a table supplies `lambda = CF_MOC/CF_1D`. The existing residual independently closes thrust with `eta_CF CF_ideal` and mass flow with `Pc At/(eta_cstar cstar_ideal)`. The plan itself reports different MOC discharge coefficients between start-line variants, so the mass-flow assumption cannot be left implicit when the thrust model changes.

This is a missing integration definition, not a claim that the present Q1D branch violates its own declared approximation. A future insertion of lambda only into the thrust residual would combine a MOC thrust with a separately assumed mass flow. Resulting Isp, coolant flow and pump sizing would not be the same MOC analysis.

**Repair:** C0/M0 must return or derive mutually consistent thrust, mass flow/sonic flux, reference throat area, gas state and loss conventions. For example, report both `CF_vac` and `CD`, with `CF_vac/CD` available as a thrust-per-mass diagnostic; define how physical combustion loss combines with geometric/transonic discharge. Do not count discharge a second time through a reinterpreted c-star efficiency. Verify both engine residuals and all downstream flow consumers under a geometry/start-line perturbation. The superseded default roadmap's shared flow/mass closure is more explicit than the replacement plan here and should be retained.

## AV-04 — The M0a table needs a declared geometry and ambient-pressure domain

**P2; missing model/domain definition; high confidence.**

Target: `sections/09_geometry_arch.tex`, line 49, and current `src/lrekit/optimization/engine.py` lines 166–167.

The listed table arguments are `(epsilon, length_pct, gamma, Rd)`. They omit upstream curvature `Ru/Rt`, even though the chosen start-line model depends on it, and do not identify arbitrary-wall controls versus the chart-defined TOP family. “Smooth” is not an established property across folds or physical branch changes. Recording the start-line method alone is insufficient to identify a flow result.

The E04 ratio is a vacuum comparison. If it is inserted as a multiplier on the existing **ambient** CF, the ambient term is also scaled. For an attached exit plane and shared area normalization, direct momentum balance gives `CF(pa)=CF_vac-epsilon*pa/Pc`. Therefore `lambda*CF_1D(pa)` differs from `lambda*CF_1D_vac-epsilon*pa/Pc` by `(1-lambda)*epsilon*pa/Pc`. This algebra is a project derivation, not a new literature correlation. Separated flow requires a different control volume/model.

**Repair:** declare the wall family, fixed versus variable curvature ratios, start-line/gas identity, normalization, flow branch and interpolation domain. Interpolate vacuum thrust and mass-flow quantities separately, then apply the supported ambient/control-volume relation. Refuse extrapolation and unsupported shock/separation branches; measure holdout and derivative errors within each accepted branch. A fixed Ru is acceptable if it is explicit in the table identity.

## AV-05 — The new roadmap weakens some integration requirements present in the superseded one

**P2; promotion-scope and acceptance gap; medium-high confidence.**

Target: `sections/12_roadmap.tex`, A0/D0 dependencies and `sections/09_geometry_arch.tex`, C0/M0/D0 acceptance rows. Compare the superseded `default_solver_roadmap_2026-11-21/sections/04_differentiability_and_mdo.tex` and `05_implementation_and_validation.tex`.

The newer C0 is described as typing/refusal and M0 as perturbation propagation/holdout/Taylor testing. The earlier roadmap also specifies shared flow/mass/thermal residuals, conditioning, solver tolerance refinement, physical wall versus core boundary, loss accounting, and independent admission of each physical regime. A0/D0 in the new dependency table do not explicitly require N2 or the thermodynamic/shock/viscous stages whose eventual promotion the title may suggest.

This does **not** mean every constant-gas smooth-flow release must wait for all reacting/separated-flow capabilities. It means “default” must name its supported capability and branch. An attached constant-gas release can be bounded and useful; it cannot imply that all G1–G4 capabilities are accepted.

**Repair:** make promotion capability-specific. Smooth-flow sensitivities require the N2 validity policy; frozen, equilibrium, finite-rate, fitted-shock and BL-coupled products each require their own completed prerequisites and source/benchmark evidence. Restore the explicit contract/conditioning/energy-ledger requirements instead of relying on a generic round-trip test. Retain current user-decision gates; this audit does not make those choices.

## AV-06 — A central cited source is still outside the searchable corpus

**P2; confirmed accessibility gap; high confidence.**

The nine supplied PDFs are now in the corpus and retrievable. However, a manifest search by NTRS 19860007470 and direct path inspection found the TDK manual only in ignored scout downloads under `builds/`, not as a canonical `propulsion_texts` Document. This manual supports important T1/T3/V0 statements. A subsequent agent searching only the documented corpus can miss it despite its “held” status in the plan.

**Repair:** in a follow-up corpus acquisition pass, ingest the already downloaded load-bearing scout PDFs with source identity and page/diagram provenance, especially TDK. Do not treat bibliography metadata or a scout's note as equivalent to corpus coverage. The present ingestion request was completed for all nine supplied documents; it did not silently relabel the rest of the scout downloads as ingested.

## Claims checked and retained

- The direction of given-wall versus inverse-design displacement is correctly distinguished by TDK BLW (PDF p.287) and the old roadmap.
- The TDK exit deficit displayed in the plan agrees with PDF p.27 under its stated conventions; the error is the separate delivered-performance factorization.
- Whitfield, Swafford and Jacocks (1981), DOI 10.2514/3.60066, PDF p.6, printed p.1320, Eq.(19), supports the quoted surface mass-source formula and `k=0/1` planar/axisymmetric distinction. It is part of an iterated inverse BL/Euler treatment, not evidence that simply changing an MOC wall condition already supplies its full separated-flow behavior.
- The existing displacement code explicitly labels its nozzle extension a repository heuristic. The audit does not promote it to an integral BL model.
- Empirical separation onset, RSS/hysteresis and directional side loads remain different capabilities. Excluding a side-load history from a steady axisymmetric field is justified.
- The plan correctly says the criterion sweep does not identify the physically correct criterion. Its 1.11 spread is not a statistically established uncertainty bound or proof that a 1.2 reserve covers model discrepancy.

## Coverage and limits

Read the complete extension-plan sections and the earlier architecture/BL/promotion sections; inspected the current Q1D engine residual, displacement heuristic, E11b and evidence labels. Used KG context/search across the corpus for TDK, BL displacement, heat/friction and separation, followed by the original TDK manual, Whitfield paper, SP-8120 and TP-2725. Visually checked TDK PDF pp.26,27,287; Whitfield p.6; TP-2725 p.7. The five source-page images are retained in the audit build directory. No full BL calculation, corrected BVP solve, CFD run or physical test was executed in this review.

## Working-tree observation

During review the repository was reorganized into `src/lrekit` (with a `raosim` compatibility shim) and categorized test/experiment directories. Code locators above were refreshed against the moved files; historical plan references remain unchanged. The baseline writer test and ingestion knowledge-suite results refer to their completed execution before that reorganization. No missing legacy import path is counted here as a scientific defect. Source fingerprints are recorded in `architecture_viscous.json`.
