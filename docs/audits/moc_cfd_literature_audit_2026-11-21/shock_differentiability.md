# Shock-fitting and differentiability plan audit

Shock fitting and differentiability plan audit, after the full supplied-paper Docling/KG ingestion gate. No plan, physics, default or graph edits.

Source PDFs remain authoritative. Findings distinguish errors in statements from missing acceptance requirements; none certifies a new solver capability.

## SA-01 [P2] Distinguish a discrete AD derivative from convergence to the shock-sensitive continuum derivative

**Class:** incorrect_generalization; confidence high.

**Locations:** docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/07_g4.tex:23,26,43,46

**Evidence:** Giles-Ulbrich Part 2 proves convergence for a modified Lax-Friedrichs scalar convex-flux scheme with viscosity of order h^alpha, 2/3 < alpha < 1, which makes shock width in physical units vanish while points across the shock increase. PDF p5/printed909 explains this requirement; PDF p3/printed907 demonstrates an apparently grid-converged adjoint with the wrong value. Giles-Pierce PDF p6/printed332 explains that certain numerical discretizations can recover correct adjoint behavior without explicit enforcement of the internal shock condition.

**Implication:** The table says plain AD lacks shock-position terms as a general property. AD gives the derivative of the chosen finite-dimensional capturing algorithm; whether that derivative converges to the intended continuum shape/output derivative depends on the scheme, shock resolution, objective and limiting process. The supplied papers do not support a universal impossibility claim.

**Correction:** State that plain AD of a capturing discretization does not by itself establish a converged continuum shock sensitivity. Add a separate shocked scalar/Q1D derivative benchmark and mesh-plus-smoothing study before claiming derivatives through shocks; keep N1 initially scoped to the smooth attached branch. Do not transfer the scalar theorem to axisymmetric reacting Euler without new evidence.

**Primary sources:** Michael B. Giles and Stefan Ulbrich (2010), Convergence of Linearized and Adjoint Approximations for Discontinuous Solutions of Conservation Laws. Part 2: Adjoint Approximations and Extensions, DOI 10.1137/09078078X, PDF pages 1,3,5,6; Michael B. Giles and Niles A. Pierce (2001), Analytic adjoint solutions for the quasi-one-dimensional Euler equations, DOI 10.1017/S0022112000002366, PDF pages 5,6

## SA-02 [P2] A finite-grid fold is not yet a physical shock-onset bound

**Class:** unsupported_physical_location_claim; confidence high.

**Locations:** docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/05_g2.tex:44,55,61,69,79; docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/12_roadmap.tex:23,38; src/lrekit/components/nozzle/solvers/bounded.py:393,418,441,455

**Evidence:** Salas PDF p3/printed585 explicitly describes his coalescence detector as empirical; he initiates a fitted front before mathematical coalescence, using an empirically chosen anticipation parameter. The book PDF p86/printed73 explicitly illustrates a fitted shock introduced upstream of the characteristic merging point. Current code detects a signed-cell-area reversal and stores the attempted node coordinate, rather than solving a conservative shock front.

**Implication:** The plan correctly reports E10 axial drift and no demonstrated convergence, but later says the first crossing bounds physical onset and S0a locates the envelope. Those later statements are stronger than the numerical evidence. A bound for a continuum simple-wave envelope does not automatically become a certified bound from a nonconverged axisymmetric discrete fold.

**Correction:** Call S0a a smooth-march refusal and candidate compression/coalescence locus. Require same-family identification, in-domain checks, refinement, upstream compression evidence and independent front/location comparison before an onset bound is claimed. Preserve the older roadmap distinction between fold, shock candidate and admissible solved shock. Never import Salas's anticipation threshold as a universal constant.

**Primary sources:** Manuel D. Salas (1976), Shock Fitting Method for Complicated Two-Dimensional Supersonic Flows, DOI 10.2514/3.61399, PDF pages 3,4; Renato Paciorri and Aldo Bonfiglioli, in Onofri/Paciorri (eds.), Shock Fitting (2017), Basic Elements of Unstructured Shock-Fitting: Results Achieved and Future Developments, DOI 10.1007/978-3-319-68427-7_3, PDF pages 75,78,86

## SA-03 [P1] S0 needs branch admissibility and model closure tests before its general-EOS result feeds S1

**Class:** acceptance_gap; confidence high.

**Locations:** docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/05_g2.tex:26,34,69,81

**Evidence:** The planned S0 acceptance lists only calorically perfect theta-beta-M and normal-shock limits. The superseded roadmap explicitly required a compressive entropy-admissible root, exclusion of the trivial unshocked root, dimensional upstream state and a named frozen/equilibrium/finite-rate gas mode. Salas PDF p4/printed586 computes the high-pressure side with both a downstream-reaching characteristic and Rankine-Hugoniot equations; the book Sec2.5 explains that jumps alone do not supply all interface information.

**Implication:** Agreement in a perfect-gas limiting test does not prove correct branch selection or thermodynamic closure on the general-EOS branch. In particular, the identical upstream/downstream state algebraically satisfies the conservation equalities; a Newton solver can return it unless a nontrivial compressive branch is selected. Taking the smallest beta above mu is a proposed weak-branch policy, not a general theorem for every EOS and downstream boundary condition.

**Correction:** Before S1, specify dimensional upstream p,T,velocity,composition, EOS/acoustic mode and front orientation; explicitly exclude the no-jump root, enforce compression and entropy admissibility, test weak/strong/attached-limit branches, and distinguish frozen jump plus relaxation from equilibrium jump. Retain conservation residuals and chemistry/element closure tests. Treat beta policy as conditional on the stated gas/topology and supporting downstream compatibility.

**Primary sources:** Manuel D. Salas (1976), Shock Fitting Method for Complicated Two-Dimensional Supersonic Flows, DOI 10.2514/3.61399, PDF pages 3,4; Renato Paciorri and Aldo Bonfiglioli, in Onofri/Paciorri (eds.), Shock Fitting (2017), Basic Elements of Unstructured Shock-Fitting: Results Achieved and Future Developments, DOI 10.1007/978-3-319-68427-7_3, PDF pages 75,78,86

## SA-04 [P1] N2 needs primal convergence and conditioning admission, even when topology flags stay fixed

**Class:** acceptance_gap; confidence high.

**Locations:** docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/07_g4.tex:14,16,43,56,62,63; src/lrekit/numerics/jax/bvp.py:69,97; src/lrekit/numerics/jax/sensitivities.py:20,22,141,149

**Evidence:** The plan itself writes F_z z_x = -F_x with invertible F_z. The older differentiability roadmap required root convergence, conditioning checks, tolerance/grid refinement and enough primal accuracy for Taylor tests. The present N2 label depends on fixed flags and accounting for E12 jumps, which are necessary but not sufficient. Direct derivative-error reasoning: errors in residual or its derivatives are amplified by the inverse state Jacobian even without a topology event.

**Implication:** An event-free tape can contain an unconverged or ill-conditioned continuous unit process. Therefore unchanged flags cannot certify the derivative. There is a concrete preview-wrapper example: make_differentiable_solution uses throw=False and immediately returns readout(sol.value), ignoring sol.result.

**Correction:** Make valid require accepted primal residuals, solver success, usable Jacobian/adjoint conditioning and derivative-error/Taylor evidence under tolerance and grid refinement, in addition to stable topology. Failed or insufficiently resolved solves must return unavailable derivatives. Retain the older roadmap's distinction between root derivatives, nonzero-residual least-squares optimum derivatives and finite-iteration derivatives.

**Focused probe:** PYTHONPATH=src .venv-jax/bin/python: fn(z,p)=z-[p], initial0, p=2, make_differentiable_solution(max_steps=0). unguarded readout 0.0; least_squares_solve reports success=False and residual[-2.0]. Tiny helper probe only; not a new MOC numerical solve. The helper is currently referenced only by its dedicated test, so this does not demonstrate an exposed production total-gradient bug.

## SA-05 [P2] A weak-shock entropy asymptotic does not justify marching through an arbitrary fold

**Class:** missing_acceptance_detail; confidence high.

**Locations:** docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/05_g2.tex:61,62,79; docs/subsystems/S06-rao-moc-bvp-contour/moc_extensions_plan_2026-11-21/sections/07_g4.tex:61,63; src/lrekit/components/nozzle/solvers/bounded.py:441,455

**Evidence:** The plan invokes third-order entropy loss in shock strength, but specifies no operational strength measure, conserved-flux reconstruction or output error budget for its weak-coalescence continuation. Salas PDF p4/printed586 deletes discontinuities below a chosen 1% Mach difference and explicitly says that feature is not required by the technique.

**Implication:** A small local entropy change does not alone control geometry, downstream pressure, accumulated wave error, thrust or sensitivity error. Signed-cell-area reversal does not even supply a shock strength. Salas's optional deletion rule is implementation evidence, not a validated transferable tolerance for this MOC nozzle.

**Correction:** Keep refusal as the default until the policy names its state-pair/strength definition, conserved quantities, loss budget, cumulative effect and tested output/derivative error versus a fitted or independently captured front. Preregister thresholds with the intended gas/domain; no automatic smoothing or fold bypass on the basis of entropy order alone.

**Primary sources:** Manuel D. Salas (1976), Shock Fitting Method for Complicated Two-Dimensional Supersonic Flows, DOI 10.2514/3.61399, PDF pages 3,4

## Claims that remain supported

- Stationary inviscid shock mass, normal momentum, normal energy plus continuous tangential velocity form is valid under stated conditions.
- Separate rotational downstream streamline data are necessary; supersonic-only S1 and typed refusal of axis/wall interactions are reasonable bounded first scope.
- The plan correctly distinguishes scalar one-dimensional sensitivity theory from axisymmetric full-system claims.
- The explicit report that E10 has not converged and E12 remains FAIL should remain.
- Shared-KL E07b parity is correctly identified as code-to-code evidence, not independent physical validation.

## Coverage and limits

- Book targeted chapters/sections selected from its contents and KG/corpus retrieval; not a cover-to-cover mathematical validation of every chapter.
- Giles2010 supplied PDF is Part2; Part1 equation1.4 remains unverified from this supplied set.
- No complete fitted-shock implementation exists in the inspected paths; acceptance gaps are plan defects, not demonstrated errors from a nonexistent solver.
- Concurrent source-tree relocation during audit moved legacy raosim paths to src/lrekit, with src/raosim retained as a compatibility shim. Current inspected source paths and hashes are recorded; relocation itself is not a scientific defect.
- No long MOC solves or CFD runs performed.

The JSON companion records source identities, exact locations, retrieval scope, inspected source hashes and the small helper probe.

Final locator refresh: the parent reread the canonical bounded-row fold guard, JAX least-squares wrapper and sensitivity scope in `src/lrekit` after the package rename. The cited line numbers are unchanged. Prior file paths/hashes are retained in the JSON snapshot records; only current locator identity was refreshed, not scientific support.
