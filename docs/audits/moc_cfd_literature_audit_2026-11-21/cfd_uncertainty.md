# CFD and discretization-uncertainty review — 21 November 2026

All nine supplied PDFs were converted and embedded in the knowledge graph before this review began. This review changes no numerical code, plan text, defaults or scientific support records. It inspects the present E13 writer and its generated comparison contract; it does not claim that OpenFOAM has run. `foamRun`, `docker` and `podman` were not found on PATH in this review.

## Main result

The write-only CFD prototype is a useful scaffold, but it is not ready to be treated as a runnable benchmark package or as the uncertainty basis for all MOC extensions. The most direct defect is an inviscid-incompatible wall-force function object. Several acceptance rules also mix discretization sensitivity with model/geometry differences. The findings below distinguish source-level defects from proposed-method limitations.

## Scope and source checks

Primary papers: [Celik et al. (2008)](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/propulsion_texts/procedure-for-estimation-and-reporting-of-uncertainty-due-to-dis-2008.pdf), [Lovely and Haimes (1999)](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/propulsion_texts/lovely1999.pdf). PDF page numbers below refer to these exact copies.

The repository was reorganized during review. The writer moved from `scripts/moc_ext_e13_write_shockfluid_case.py` to `tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py`; current code links below follow the latter (two import lines were added). W-line references in reasoning retain the original read-time numbering. This relocation is not a scientific defect.

- Read the CFD, verification and dependency sections of the extension plan; relevant parts of the superseded default-solver roadmap; the complete E13 writer contract, mesh/BC/thermodynamic generation and static tests; generated Nozzle-B comparison JSON and four case manifests.
- Retrieved through the full rebuilt knowledge graph using CFD/nozzle/discretization/shock queries. Relevant routed sources include Celik2008, Lovely1999, the shock-fitting volume, Anderson and Betti/Leccese nozzle studies. Conclusions below use the primary Celik/Lovely PDFs, inspected code and explicit derivations.
- Celik: all four PDF pages read; pages1,2,4 rendered and visually inspected. Lovely: all nine pages searched/read; PDFpages2 and6 rendered and visually inspected. The equations are checked against the original image, not trusted from Docling formula text.
- OpenFOAM source was checked at the exact pinned commit, not inferred from a current tutorial. Downloaded files and SHA-256 values are in [the source manifest](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/builds/moc_cfd_literature_audit_2026-11-21/cfd_sources/manifest.json).
- The parent independently completed the existing writer static suite:4passed. That result checks text generation/determinism and synthetic geometry; it cannot exercise the runtime function-object conflict.

## Findings

### CFD-01 [P1] The inviscid case enables a force monitor that requires a transport model

**Classification:** confirmed source-level runtime incompatibility. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:649](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:649), [moc_ext_e13_write_shockfluid_case.py:536](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:536), [moc_ext_e13_write_shockfluid_case.py:1148](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1148).

**Claim:** E13 can monitor wallForces while shockFluid takes its zero-viscosity path and constructs no momentumTransport model.

The pinned shockFluid.C lines148–180 chooses inviscid when max(mu)<=0 and leaves momentumTransport null. Sutherland As=0 gives mu=0 for positive finite T (sutherlandTransportI.H mu implementation). forcesCompressible loads the standard forces object. In the exact pinned forcesBase.C, write() lines934–936 calls calcForcesMoments; its ordinary pressure/viscous branch lines828–839 unconditionally calls devTau(); lines250–254 then emits a fatal error when no recognized transport model is registered. Creating a momentumTransport dictionary does not instantiate the missing model. The generated object does not select directForceDensity. This is a deterministic source-path conflict, although no OpenFOAM process was run to capture its runtime log.

**Correction:** Use a pressure-only wall integration that does not call viscous-stress lookup, or a source-verified inviscid-compatible monitor. Retain raw pressure/face vectors for an independent integral. Include execution of this function object in the first dry run; static dictionary tests cannot close CFD0 runtime compatibility.

**Source anchors:** [pinned OpenFOAM source](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/65eda17ad4caa916bfd0606039db92509670f9f7/applications/modules/shockFluid/shockFluid.C); [pinned OpenFOAM source](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/65eda17ad4caa916bfd0606039db92509670f9f7/src/functionObjects/forces/forcesBase/forcesBase.C); [pinned OpenFOAM source](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/65eda17ad4caa916bfd0606039db92509670f9f7/src/thermophysicalModels/specie/transport/sutherland/sutherlandTransportI.H)

### CFD-02 [P2] The displayed shock criterion discards compression direction

**Classification:** confirmed plan/writer mismatch. Confidence: high.

**Locations:** [10_cfd.tex:28](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:28), [moc_ext_e13_write_shockfluid_case.py:1177](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1177).

**Claim:** The LaTeX plan identifies shock location through the absolute normal Mach number.

Lovely–Haimes stationary-shock section, PDFp2/printed297, uses the pressure-gradient direction and rejects expansions because its dot product with the Mach vector is negative. Taking an absolute value loses that rejection. The actual writer string already uses signed M_n and a pressure-gradient filter, so the defect is in the plan formula rather than the writer. The M_n=1 contour bounds a detected region rather than selecting a unique fitted front.

**Correction:** Write signed M_n=(u/a) dot grad(p)/|grad(p)|, apply the positive compression-region criterion with small-gradient refusal/filtering, and describe M_n=1 as a region boundary. Preserve the distinction between a numerical region and a front location.

**Source anchors:** LOVELY PDFp2 printed297; Filtering PDFp6 printed301.

### CFD-03 [P1] A two-level MOC delta is promoted into an uncertainty bound

**Classification:** unsupported uncertainty inference. Confidence: high.

**Locations:** [10_cfd.tex:32](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:32), [moc_ext_e13_write_shockfluid_case.py:1140](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1140), [moc_ext_e13_write_shockfluid_case.py:1196](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1196), [moc_ext_e13_write_shockfluid_case.py:1202](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1202).

**Claim:** The t19 coarse/fine difference is U_MOC and may be combined quadratically with a fine-grid CFD GCI to adjudicate agreement.

A two-level difference does not establish an observed order or remaining continuum error. Even for f(h)=f_exact+C h^p, the fine-grid error is |f_coarse-f_fine|/(r^p-1), not generally the raw difference; without a justified p it can exceed or fall below it. Celik PDFp2 Eq7 applies an apparent-order denominator and safety factor, and explicitly excludes modeling error. No probabilistic interpretation, covariance assumption, or consistent bound interpretation is supplied for root-sum-square mixing of these two different evidence types. It can be a declared comparison policy but is not a literature-established complete numerical uncertainty budget.

**Correction:** Keep t19 delta labeled a refinement sensitivity until an MOC error estimate is justified for the same observable and regime (additional matched refinement levels or verified-order evidence). State separately what U_CFD/U_MOC mean, what errors are excluded, and why any combination rule is appropriate. Preserve discrepancies for diagnosis rather than automatically interpreting a failed narrow band as model failure.

**Source anchors:** CELIK PDFpp1–2, Eq3–7; algebraic derivation stated here.

### CFD-04 [P1] The comparison tests different starting-flow problems using discretization-only tolerance

**Classification:** boundary/model mismatch in acceptance design. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1212](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1212), [moc_ext_e13_write_shockfluid_case.py:1215](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1215), [10_cfd.tex:19](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:19).

**Claim:** Nozzle-B CFD with a newly selected convergent is a same-model comparison to a prescribed KL start, and differences should fit discretization uncertainty alone.

The writer expressly adds a45-degree convergent with contraction2.5, whereas the MOC supplies transonic initial data from the KL series and a throat curvature. Both may solve inviscid perfect-gas equations, but they do not share the same initial/boundary data or necessarily the same continuum field. The writer also compares a faceted wall with a solved MOC wall representation. Those systematic differences survive h->0 and are absent from the stated root-sum-square budget. Celik PDFp1 specifically asks for inlet-condition sensitivity. The plan already acknowledges the convergent difference, but does not propagate it into acceptance interpretation.

**Correction:** Split CFD1 into a matched supersonic-boundary/geometry MOC discretization comparison and a full-convergent transonic/KL model assessment. For the latter declare starting-model/geometry discrepancies and run convergent/inlet sensitivity; do not require their difference to be explained by spatial discretization alone.

**Source anchors:** CELIK PDFp1 Preliminaries; CODE W1210–1227; problem-definition reasoning.

### CFD-05 [P2] The GCI prescription permits zero or negative apparent orders and singular relative errors

**Classification:** confirmed incomplete postprocessor specification. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1186](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1186), [moc_ext_e13_write_shockfluid_case.py:1189](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1189), [moc_ext_e13_write_shockfluid_case.py:1198](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1198).

**Claim:** Rejecting a negative successive-difference ratio is sufficient before applying the stated p/GCI expressions.

The planned formula has no stated branches for zero/near-zero differences, positive ratios<=1, nonfinite values, or f1 nearzero. For the monotonically ordered triple(f1,f2,f3)=(1,2,2.5), its formula gives p=-1 and GCI21=-2.5. Ratio1 gives p0 and a zero denominator. Celik PDFp2 Eq3a includes absolute values and explicitly warns that very small differences defeat the procedure; PDFp3 discusses high relative errors at near-zero velocity. No actual postprocessor exists here, so this is a specification defect, not a claim that bad CFD numbers have already been computed.

**Correction:** Define finite-data checks, an explicit observed-order/admissibility policy, zero-difference and near-zero-output cases, and absolute-error reporting with a declared reference scale. Mark unresolved convergence unavailable and request more levels rather than emitting negative/undefined uncertainty.

**Source anchors:** CELIK PDFp2 Eq3a–7 and limitations; PDFp3 near-zero velocity example; exact algebraic counterexample.

### CFD-06 [P2] The proposed asymptotic-ratio check is not independent evidence of the asymptotic range

**Classification:** confirmed weakness of acceptance test. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1190](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1190), [10_cfd.tex:47](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:47).

**Claim:** GCI32/(r^p GCI21) near1 establishes the asymptotic range for these equal-r triples.

If p=log(|epsilon32/epsilon21|)/log(r) is estimated from the same equal-r three-grid triple, then with the stated relative GCI definitions the ratio reduces algebraically to |f1/f2| (assuming admissible nonzero differences). With common normalization it is exactly1. A large output offset therefore makes this check nearly1 without showing that a leading-order error model dominates. The recorded counterexample f(h)=100+h^2+100h^6 on h=(.25,.5,1) has estimated p5.845 although its true asymptotic order is2; the ratio is0.98305. This does not invalidate GCI; it invalidates treating this check alone as sufficient evidence.

**Correction:** Retain it as a consistency diagnostic, but require an additional refinement level/order stability, agreement with a verified scheme limit, and evidence that iterative/geometry error is below the measured signal. Do not assign automatic asymptotic PASS solely from near1.

**Source anchors:** CELIK PDFp2 discussion of apparent versus formal order; independent algebraic derivation and manufactured sequence.

### CFD-07 [P2] Fixed mass-balance and thrust-drift thresholds do not control iterative error in every reported output

**Classification:** missing numerical-quality evidence. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1204](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1204), [moc_ext_e13_write_shockfluid_case.py:1077](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1077), [10_cfd.tex:47](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:47).

**Claim:** Mass imbalance<1e-4 and CF change<1e-5 over a flow-through time are sufficient prerequisites for GCI on all functionals and smooth samples.

These are sensible monitoring choices, but neither constrains local wall-pressure/Mach changes or slow convergence modes. Celik PDFp1 asks for residual trends for each equation and convergence of key variables, and PDFp4 recommends estimated iteration error at least one order below discretization error. It explicitly warns that small successive changes can be misleading when the convergence factor is near1. The writer supplies no per-observable iteration-error estimate or signal-based rule, and its fixed ten-flow-through endtime is an initial schedule, not proof of stationarity.

**Correction:** Add per-observable stationarity/iterative-error estimates, conservation/energy monitoring, extended-run checks, and residual histories suitable for this explicit conservative solver. Require iterative contamination below the discretization signal used in each GCI. If only physical steady state is compared, distinguish this from time-accurate transient qualification.

**Source anchors:** CELIK PDFp1 Preliminaries, PDFpp3–4 Appendix; CODE W997–1001.

### CFD-08 [P2] Wall faceting and wedge-angle error are outside the three-grid uncertainty estimate

**Classification:** uncontrolled systematic numerical effects. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1061](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1061), [moc_ext_e13_write_shockfluid_case.py:1209](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1209), [moc_ext_e13_write_shockfluid_case.py:1225](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1225).

**Claim:** Refining the emitted mesh supports a continuum smooth-nozzle comparison without additional geometry/wedge error control.

All three grids keep identical block vertices and the same polyline wall; refining cell counts converges toward that polygonal domain. A fixed0.125degree turning limit creates geometric kinks and associated local waves that do not vanish under this refinement. The0.2degree wedge is tested only on the coarse mesh, with no acceptance: it cannot isolate an angle error on the fine mesh from angle/grid interaction. The script already labels these effects, which is good, but they are not included in the disagreement budget or quantified at the accuracy demanded of CFD1.

**Correction:** Either make the MOC analyze the identical polygonal wall for a matched comparison, or perform separate wall-representation convergence against the intended smooth contour. Perform wedge-angle sensitivity at a mesh level appropriate to the accepted accuracy, with separate limits/reporting. Continue normalizing fluxes using the actual mesh throat area; that part is correct.

**Source anchors:** CELIK PDFp1 definition of the discrete problem and PDFp2 systematic refinement; CODE W351–355,W1059–1061,W1207–1227; geometry-convergence reasoning.

### CFD-09 [P2] CFD1 promises TOP shocks and variable-cp fields that the current case family cannot generate

**Classification:** missing deliverables and dependencies. Confidence: high.

**Locations:** [10_cfd.tex:27](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/10_cfd.tex:27), [12_roadmap.tex:42](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/12_roadmap.tex:42), [moc_ext_e13_write_shockfluid_case.py:517](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:517), [moc_ext_e13_write_shockfluid_case.py:1016](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1016).

**Claim:** The Nozzle-B E13 prototype provides the CFD1 references required by S1 and T1.

E13 hard-codes a Nozzle-B case identity/source and hConst/pureMixture/perfectGas thermodynamics with one constantCp and gamma. Its own cannot_establish list excludes variable-gamma effects. It has neither an accepted TOP shock case nor a JANAF variable-cp case/initial-state contract. Thus Nozzle-B success would not close either advertised reference. CFD1 currently depends only on CFD0 and X0; the additional geometry/gas/source matching tasks have no explicit deliverable or acceptance.

**Correction:** Split CFD1a smooth constant-gas Nozzle-B, CFD1b matched TOP internal shock, CFD1c frozen variable-cp/JANAF with matching gas tables and sound speed, and later viscous/RANS work. Add the corresponding geometry, thermodynamic and shock-extraction dependencies. A useful first limited case need not cover all regimes.

**Source anchors:** CODE W509–538,W1014,W1218–1220; PLAN10_cfd27–35 and12_roadmap42.

### CFD-10 [P2] Mass-averaged Mach on C-D-E is not the exit-plane mass-averaged Mach

**Classification:** confirmed invalid alternate comparator. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1164](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1164).

**Claim:** Either a C-D-E mass-averaged M or an exit-plane mass-averaged M is a valid counterpart to the CFD outlet-plane metric.

Mass flow and complete momentum-plus-pressure flux can be moved between suitable control surfaces using conservation. Mach number is not a conserved transported scalar; it changes through smooth expansion along a streamline. Therefore an oblique upstream C-D-E average need not equal an outlet-plane average even for an exact common flow, while the pressure/momentum control-surface equality can remain valid.

**Correction:** Compare the same x=x_E plane and use the same weighting: integral(M rho u_x dA)/integral(rho u_x dA). Keep C-D-E averages as separately named diagnostics. Verify interpolation coverage and quadrature on that exit plane.

**Source anchors:** CODE W1162–1163; conservation/observable-definition derivation.

### CFD-11 [P2] The wall-force cross-check needs an outward-normal sign and inlet pressure term

**Classification:** confirmed sign error plus ambiguous inlet term in planned control-volume cross-check. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:1155](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1155), [moc_ext_e13_write_shockfluid_case.py:649](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:649).

**Claim:** Outlet momentum-plus-pressure thrust equals inlet momentum flux plus the standard wallForces pressure integral.

With normals outward from the fluid, steady inviscid x-momentum gives J_out=J_in-W_x, where J_in=integral_inlet(rho u_x^2+p)dA and W_x=integral_wall p n_x dA. The pinned forcesBase.C lines853–858 returns +(p-pRef)Sf, so the raw pressure force has this W_x sign. The writer says plus wallForces, which is opposite. Its inlet momentum wording also does not explicitly include inlet pressure. Inlet pressure is material for the finite inlet radius. Main outlet integral is correct; the defect affects the independent cross-check specification.

**Correction:** Write the complete signed control-volume balance including inlet pressure, all included wall patches, any nonzero unsteady storage term, and explicit fluid/solid force convention. Export an inlet surface as well as outlet/wall data, or supply a tested inlet integration. Set pRef0 explicitly for reproducibility. Verify the identity on a uniform manufactured flow and a converging/diverging geometry.

**Source anchors:** [pinned OpenFOAM source](https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-13/65eda17ad4caa916bfd0606039db92509670f9f7/src/functionObjects/forces/forcesBase/forcesBase.C#L853-L858); steady Euler momentum integral derived here.

### CFD-12 [P2] The return-data plan has no implemented field/flux extractor or reproducible shock-region estimator

**Classification:** missing extraction contract. Confidence: high.

**Locations:** [moc_ext_e13_write_shockfluid_case.py:640](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:640), [moc_ext_e13_write_shockfluid_case.py:1177](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1177), [moc_ext_e13_write_shockfluid_case.py:1233](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/tests/experiments/nozzle/moc_ext_e13_write_shockfluid_case.py:1233).

**Claim:** The writer and schema suffice to deliver accepted numerical quantities after a user returns logs.

The script emits field samples, monitor templates, prose definitions and an illustrative handoff schema, but no implementation is present for the promised surface quadrature, signed inlet flux handling, field registration, GCI edge cases, connected shock-region localization, or threshold sensitivity. The axis sampling line is actually at r=1e-4Rt and excludes domain ends (W630–635), while the comparison describes r0 extrapolated cell-centre results; reconstruction policy is unspecified. Proposed gradient threshold0.05 is honestly labeled a policy, but no detection calibration or miss/false-positive study is specified. Lovely PDFp6 describes threshold selection as a substantive filter problem, not a universal constant.

**Correction:** Add CFD-extract as a distinct milestone before CFD1 adjudication: typed SI field/surface inputs, extraction-code hashes, exact station/normal/area conventions, robust convergence handling, and synthetic/exact detector checks including smooth expansion, smooth compression, constant flow and known captured shocks. Require explicit UNAVAILABLE for missing samples; do not infer all metrics from a checkMesh log or a short dry run.

**Source anchors:** LOVELY PDFp6 Filtering; CODE W630–674,W1175–1178,W1231–1255.

## Claims that survive this review

- A frozen-composition, constant-gamma, inviscid independent CFD discretization is a useful first comparison. Choosing an external solver is reasonable; rejecting an in-repo FV implementation here is a project cost/independence decision, not a law of verification.
- `As=0` does produce zero Sutherland viscosity for valid positive temperatures in the pinned implementation, and shockFluid then deliberately skips its transport models. The inviscid selection itself is supported; the force monitor is the conflict.
- The pinned standard force object defaults pRef to0, and for dimensional pressure its pressure contribution is the absolute pressure times the face area vector. The previous UNVERIFIED pRef item can be resolved by source review, although runtime compatibility remains open.
- The totalPressure configuration selects the compressible psi/gamma relation in the pinned dynamicPressure source. waveTransmissive uses normal advection plus sqrt(gamma/psi). A dictionary/source check still does not prove boundary-condition suitability or stable converged flow. No blanket claim that these BCs are invalid is made.
- The three cell counts have refinement factor4 in total cells and factor2 in each active meridional direction. Keeping geometry fixed is appropriate for studying discretization on that polygonal domain, provided geometry error is separately addressed.
- The actual flat-wedge throat area0.5 Rt^2 sin(alpha) and its full-revolution conversion factor are geometrically consistent. Using the actual mesh area is correct.
- For a finite positive admissible apparent order and nonzero normalized output, the stated equal-r extrapolation and fine-GCI formulas have the expected form. Refusing unresolved/oscillatory cases is a permissible conservative policy; Celik itself also discusses methods for oscillatory/noisy data, so this refusal should be identified as project policy.
- Smooth-sample/profile GCI is not forbidden by Celik; the paper explicitly presents profile error bars. Pointwise discontinuities and shock-region registration require separate treatment. The plan says only functionals in one place and functionals plus smooth samples elsewhere; unify the intended scope.
- Code-to-code agreement is not physical validation. The plan correctly keeps this boundary and does not claim executed CFD or a resolved viscous/RSS result.

## Additional precision and operational items

- The supplied Celik paper is4PDFpages, with equations1–2 onp1 and3–7 onp2. References toPDFpp5–7 belong to a different hosted policy-file wrapper. Cite the exact corpus copy and update all six authors; do not silently transfer page numbers.
- Celikp1 defines h according to the actual area/volume mesh, irrespective of physical dimensionality. E13 is an extruded one-cell 3D wedge, refined only in x/r, but quotes the2D Eq2 with r2. An effective meridional refinement coordinate is a reasonable project choice for this anisotropic family; label and justify it. Literal3D average-volume h instead has r=4^(1/3). A consistent reparameterization changes apparent p but leaves r^p and extrapolated error invariant for the same exact power-law sequence; this is chiefly attribution/order-interpretation precision, not proof of wrong CFD.
- `Allrun` lacks `set -e` or explicit `|| exit` after blockMesh/checkMesh. The pinned RunFunctions does not make the calling script fail-fast. Require explicit successful mesh checks before foamRun; the Slurm template already uses set-e.
- A stationary detector cannot be applied indiscriminately during the transient startup. Use converged stationary fields or the moving-shock extension; report a numerical region and threshold sensitivity. The writer currently promises a steady comparison, which is a sensible scope.
- The0.2degree coarse sensitivity case and the current proposed gradient cutoff are exploratory policies. Neither is a literature-certified error tolerance.

## Suggested CFD phase order

1. Repair pressure-only monitoring, complete signed flux extraction and fail-fast runners; execute a pinned-version dry run including all function objects.
2. Verify the extraction pipeline and solver on exact/manufactured smooth cases with matched geometry and boundaries.
3. Run the matched constant-gas Nozzle-B comparison with spatial, iterative, geometry and wedge sensitivities reported separately.
4. Run the independent full-convergent transonic/KL assessment, keeping starting-model differences separate from discretization.
5. Add a controlled TOP shock case and a calibrated region/front extraction contract for S0a/S1.
6. Add matched frozen variable-cp thermodynamics/JANAF cases for T1; then separately designed viscous/RANS cases.
7. Use explicit applicability/error requirements to adjudicate each regime. Keep unavailable/error-budget-deficient comparisons unresolved.

## Artifacts

- [Machine-readable findings](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/docs/audits/moc_cfd_literature_audit_2026-11-21/cfd_uncertainty.json).
- [Algebraic counterexamples](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/builds/moc_cfd_literature_audit_2026-11-21/cfd_algebra_checks.json).
- [Pinned source hashes](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/builds/moc_cfd_literature_audit_2026-11-21/cfd_sources/manifest.json).

## Limits

This is a source-level audit and small algebraic verification, not solver execution, a completed mesh study, or physical validation. The full shock-fitting-book methodology is owned by the shock/differentiability review; this CFD review does not claim to have independently verified every chapter. The source scans and renderer recovered the Celik PDF after an xref warning; the relevant rendered pages were readable. No unreviewed parser candidate is promoted to scientific support.
