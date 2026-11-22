# Thermodynamics and transonic-start audit

The main formulation choices hold, including the KL sign correction and critical-speed normalization. The strongest unresolved issue is T1 admission: the plan does not bound the error introduced by its approximate variable-gas starting data. Other findings concern units and acceptance definitions, not new demonstrated production-flow defects.

This review began after all nine supplied PDFs were converted and indexed. It inspected T0–T3, K0, X0 and the relevant G0 definitions in the current extension plan and compared the superseded default roadmap. Existing dirty-worktree content was preserved.

## TS-01 [P1] T1 needs a separate error gate for approximate transonic starting data

**Evidence class:** MISSING_ACCEPTANCE_EVIDENCE. **Confidence:** high.

**Plan:** `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:57`; `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:123`.

KL(gamma*) and zoned Sauer are approximate perfect-gas-based initializations. Their mutual agreement, a downstream exact-oracle convergence rate, or conservation of their input mass flux does not bound variable-cp transonic initialization bias. The extension plan itself says the Levine-Coats small-change estimate excludes cp(T); T1 acceptance does not close this acknowledged dependency. This is an admission-evidence gap, not proof that gamma* initialization is inaccurate for every case.

**Correction:** Split T1 into consistent smooth-core verification and start-data admission. Require matched variable-cp transonic Euler/reference comparisons of sonic surface, initial p,T,V, mass flux and downstream functionals, with independent start-location/series/property-grid refinement and a declared initialization-error budget. Until then mark KL(gamma*) provisional and restrict its calibrated domain.

**Sources:**
- Engineering and Programming Manual: Two-Dimensional Kinetic Reference Computer Program (TDK) — NASA CR-178628; NTRS19860007470 — `builds/moc_extensions_plan_2026-11-21/scout1_thermo/downloads/19860007470.pdf`, PDF pp. 41,42.
- Transonic Flow in Small Throat Radius of Curvature Nozzles — DOI 10.2514/3.5355 — `propulsion_texts/kliegel1969.pdf`, PDF pp. 2.

## TS-02 [P2] Restore TDK nondimensional coordinates before using its chemistry source

**Evidence class:** DOCUMENT_EQUATION_NORMALIZATION_AMBIGUITY. **Confidence:** high.

**Plan:** `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:46`.

TDK keeps dimensional production rates and inserts r* because xhat=x/r*. The plan retains r* in A and dci but omits the coordinate normalization. A is dimensionless; multiplying it by dimensional dx would be inconsistent. The neighboring smooth characteristic source has units 1/length. No finite-rate production implementation was identified, so this is a specification ambiguity with a concrete scale-factor failure mode.

**Correction:** Write xhat=x/r*, rhat=r/r* explicitly and keep the TDK normalized set throughout; or use dimensional x and remove r* from A and the species transport coefficient. Add a scale-invariance/dimensional-units check to T3.

**Sources:**
- TDK Engineering and Programming Manual — NASA CR-178628 — `builds/moc_extensions_plan_2026-11-21/scout1_thermo/downloads/19860007470.pdf`, PDF pp. 37,54,55.

## TS-03 [P2] Distinguish thermodynamic smoothing from legacy parity

**Evidence class:** ACCEPTANCE_SPECIFICATION_AMBIGUITY. **Confidence:** high.

**Plan:** `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:115`.

The existing piecewise-linear cp is a thermodynamically consistent representation; h and s are already C1 when cp is continuous. Smoothing cp across its knots can be useful for higher derivatives but changes the constitutive approximation. PCHIP through cp=[1000,1200,1300] at T=[300,500,700] changes integral h from470000 to473333.333 J/kg. Thus tight legacy parity is not guaranteed merely by keeping the nodes. Bitwise parity of an interpolated/integrated branch with an analytic closed form likewise needs explicit dispatch; physics only requires a controlled limiting agreement.

**Correction:** Specify which quantities need which derivative orders. Preserve the legacy piecewise-linear adapter for exact parity; identify a smoother backend as a separate property representation, derive h,s from it, and assess its error against the same EOS or a refined property reference. Require bitwise parity only for an explicit analytic constant-cp dispatch; otherwise declare numerical tolerance and property-grid/quadrature convergence.

**Sources:**
- Current implementation.
- Implementation counterexample (project calculation, not literature data).

## TS-04 [P2] Define the finite-rate equilibrium-limit test as a singular relaxation limit

**Evidence class:** INCOMPLETE_ACCEPTANCE_DEFINITION. **Confidence:** high.

**Plan:** `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:36`; `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/03_g1.tex:125`.

Broer distinguishes convergence of resolved flow toward equilibrium from the characteristic speeds of the unreduced finite-rate PDE. For every finite rate the characteristics remain frozen; the exact-equilibrium reduced equations have equilibrium speeds. The plan states the frozen result correctly, but leaves its equilibrium acceptance and refinement order undefined. Requiring frozen characteristic angles to converge to equilibrium angles would be wrong. No existing implementation defect is asserted.

**Correction:** Define separate tests for finite-rate frozen eigenvalues, relaxation conservation and resolved/macroscopic equilibrium convergence as rate and spatial/source-step resolution change. State the limiting variables, boundary data, order of limits and supported acoustic modes. Retain chemistry stiffness/error diagnostics.

**Sources:**
- Characteristics of the equations of motion of a reacting gas — DOI 10.1017/S0022112058000434 — `propulsion_texts/broer1958.pdf`, PDF pp. 1,5,6.

## TS-05 [P3] Apply CD<=1 at the geometric throat and bind curved-start flux to it

**Evidence class:** DERIVATION_QUALIFICATION. **Confidence:** high.

**Plan:** `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/04_kl.tex:32`; `docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/04_kl.tex:65`.

The physical upper bound holds for matched uniform stagnation conditions and the same minimum throat area: at the throat rho*u_n<=rho*|V|<=rho_star*a_star, then integrate over At. A curved C-minus surface has a different area, so its equality to throat mass flux additionally requires a closed steady mass balance, oriented flux, correct normalization and controlled quadrature/series error. calc_massflow_along_rrc sums absolute segment contributions. Its scalar value alone does not establish that control-volume equivalence. Independent source sign and normalization evidence still establishes the production defects; this caveat does not rescue them. Eq14 at gamma1.23,R1.5 gives CD=0.99442694, consistent with V3 about0.9944.

**Correction:** Use the throat-normal flux proof and list assumptions. Compare source Eq14, a throat-plane integral and signed curved-start flux under refinement, with a numerical allowance. Describe CD>1 as a mass-flux consistency failure pending attribution rather than a proof identifying an individual series coefficient.

**Sources:**
- Transonic Flow in Small Throat Radius of Curvature Nozzles — DOI 10.2514/3.5355 — `propulsion_texts/kliegel1969.pdf`, PDF pp. 3,4.

## Reviewed claims that hold

- **Axisymmetric v3 z*r^3 coefficient is negative; production positive sign is a real discrepancy.** Kliegel1969 PDFp4 A6 visually checked; src/lrekit/components/nozzle/solvers/transonic.py:88 has positive sign.
- **The KL velocity magnitude is critical-speed Mach M*, not local Mach M.** Kliegel1969 PDFp2 explicitly sonic-normalized velocities; TDK1985 PDFp42 also u/a*,v/a*. Constant-gamma energy relation gives the plan M*/M conversion.
- **Hall original is not an independent uncorrected gold table for all axisymmetric third-order coefficients.** Kliegel1969 PDFp4 expressly reports errors in Hall third-order velocity and discharge terms and supplies corrected appendix.
- **A thermodynamically consistent isentrope-based generalized turning function is the proper smooth frozen-gas extension; local gamma substitution into constant-gamma nu is unsupported.** Anderson PDF186–187; NACA1135/Guentert references routed through plan and build-source text, plus direct algebra. No new numerical claim beyond existing experiment was rerun.
- **Finite-rate characteristics use frozen sound speed; exact-equilibrium reduced equations require equilibrium acoustic derivatives.** Broer1958 PDF1,5,6 visually checked.
- **X0 correctly leaves the untested BVP _source_axisym axis-limit path open.** Plan08_verification explicitly distinguishes tested forward axis and solve_axis_point paths from residual path; no fabricated closure.
- **G0 needs explicit length conventions and a source-qualified LEA benchmark.** Current src/lrekit/optimization/grid.py:381 uses sharp cone while host path changed; plan09_geometry_arch:23–31 honestly labels xmn interpretation as inferred. Approval of inferred geometry cannot make it an independently established physical benchmark.

## Scope and evidence

Machine-readable findings, exact line ranges and source-page metadata are in `thermo_startline.json`. Retrieval outputs, the PCHIP counterexample, extracted source text and source-page renders are under `builds/moc_cfd_literature_audit_2026-11-21/`. The critical KL Appendix A6, Hall normalization (PDF p. 2), Anderson turning differential (PDF p. 186), Broer equilibrium discussion and TDK normalization/start-data/source pages were rendered and visually inspected.

The superseded roadmap is more careful than the extension plan about interpolation changing the property representation and requires gas-consistent transonic starting data. Its statement that finite-rate implementation need not await every equilibrium software feature is also sound: T2 is useful reference infrastructure, not a mathematical prerequisite to derive a finite-rate solver.

TDK, Guentert, NACA1135, Levine–Coats and Sivells were cited from scout downloads in the current plan, but several are absent from canonical corpus retrieval. This review did not silently ingest additional PDFs beyond the requested nine; the accessibility gap must remain visible.

No production code, coefficient, gate or default changed. No BVP re-solve, CFD run or new physical validation was performed.

## Final locator refresh after concurrent reorganization

The canonical package is now `src/lrekit`; `src/raosim` is a compatibility shim. Paths and line ranges were checked against `docs/architecture/relocations_2026-11-21.json` and the current symbols. This reorganization is not a scientific failure and does not change the findings. Plan paths above remain historical document references. Earlier path/hash observations are retained in `thermo_startline.json` under `locator_snapshot_history`. No solve was run.

The eleven affected coefficient, KL-state, mass-flux and frozen-property functions have the same normalized AST as their pre-refactor git-HEAD functions after stripping imports and docstrings only. This verifies their numerical expressions separately from renamed imports; it does not claim whole-file byte identity. Exact comparisons and AST fingerprints are recorded in `locator_snapshot.equation_preservation_checks`. The current expressions also match those inspected during this review.

- `src/lrekit/components/nozzle/solvers/transonic.py` — SHA-256 `41036fb81a406a2a04355acb6b1aacfbf080b444fd76d0d4f462c94e23277d46`.
- `src/lrekit/components/nozzle/solvers/moc.py` — SHA-256 `54805bc2acbdf82ab6d0a21ec93863578d22cbbb92de5f2597d899bfdef9634b`.
- `src/lrekit/thermochemistry/frozen_flow.py` — SHA-256 `ecf874d2df2c73b7c7a84a8b9c16126a58d6d75d7caac75c48634787acd28c24`.
- `src/lrekit/components/nozzle/solvers/nasa.py` — SHA-256 `1fffcebd6a3bc1ef0e9075ba2fea27cb504ed51cc19e06a4bab9c61418071bb4`.
- `tests/experiments/nozzle/moc_ext_e09b_startline_review.py` — SHA-256 `d5a9364eb852cc0a8e50379a4719ac9d7921b34b5e68d120194133265b8e6fe9`.
- `src/lrekit/optimization/grid.py` — SHA-256 `f158c21f0067dd7b8fe8c32bc1d4788ac849905ea60ca7924c7a78d178e156bb`.
