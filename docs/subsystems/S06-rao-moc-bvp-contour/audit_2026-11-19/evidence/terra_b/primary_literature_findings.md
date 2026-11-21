# Terra B: independent primary-literature findings

Audit date: 2026-11-19 (user timezone). Role: original-literature researcher, reporting to Sol A. No implementation files changed. Source text/renders and machine-readable inventory live beside this report. The inventory records payload hashes, actual reviewed pages, and catalogue-only items; catalogue inclusion is not a claim of full review.

## Independent requirements and decisions

### TB-01 — Current local C+/C− pairing is supported; do not rewrite signs

**Classification:** correctly implemented local formulation; preserve regression protection.

**Source:** `DOC_5F36B7C4DED79BB3E90754D0F81682F7A68014BE`, Anderson third edition, PDF422 / printed406, §11.8, Eqs11.52–11.53 (render inspected).

- C−: d(theta+nu)=dr/[r(sqrt(M²−1)−cot(theta))].
- C+: d(theta−nu)=−dr/[r(sqrt(M²−1)+cot(theta))].
- nu is the Prandtl–Meyer angle; mu=asin(1/M) is Mach angle. Because dr=sin(theta∓mu)ds, these are algebraically equivalent to the repository S=sin(theta)sin(mu)/r, with +S on C− and −S on C+.
- `raosim/moc.py` governing equations match this. Label differences between references must be mapped by geometric slope, not by C+/C− names alone. MIT lecture PDF4 uses m+ for inclination theta−mu, whereas code C+ is inclination theta+mu. Both can be valid naming conventions.

**Disagreement preserved:** `DOC_NTRS_20030067852`, Rice PDF8 Eq2 and PDF9 Eq4 visibly print mu in d(theta∓mu), then Eq3 defines mu=asin(1/M). A literal reading conflicts with Anderson's derivation. Treat as source notation/typographical issue; do not propagate it into code. Rice remains valuable for topology and reference-port conventions.

### TB-02 — D is common boundary data, not a freely discontinuous join

**Classification:** literature-backed requirement; current soft duplicated boundary variables are a potential implementation deficiency.

**Primary evidence:** Rao1958 `DOC_DOI_10_2514_8_7324`, PDF3 / printed379, “Solution of the Problem”: CD is a left characteristic in the fixed kernel, D lies at the intersection of the kernel right characteristic through B with CDE, M and theta are continuous in the interior, and phi is required continuous along CDE. Eq11 is phi=theta+alpha. Eq15 is the characteristic compatibility relation on DE. PDF4 / printed380, Eqs17–19, places D on the kernel characteristic and transfers its M/theta to the DE construction subject to mass matching. PDF5 Figure4 fills the characteristic net between these same BD and DE boundaries and traces the wall.

**Independent corroboration:** Rice PDF9 says D lies on the last kernel RRC BF and a LRC is constructed *from D*; PDF10 iterates D and theta_B until E meets constraints and back-calculates the BDE region before finding B–E streamline.

**Requirement for Sol A specification:** identical x,r,M,theta at the kernel D and DE start in the accepted model. Finite numerical tolerance is necessary for floating-point code, but the continuum boundary condition itself is exact.

**Implementation rationale:** eliminate duplicate boundary degrees of freedom, reconstruct their shared values from D, or use an equivalent exact constrained parameterization. These are implementation choices justified by the required boundary identity. The literature does not prescribe a particular pack/unpack layout or penalty weight. An unconstrained least-squares penalty can trade away the boundary condition, so changing a weight alone gives weaker assurance than preserving the identity by construction.

### TB-03 — Residual scaling is numerical policy; raw compatibility remains independently testable

**Classification:** repository numerical-design issue, not a missing physical equation.

**Sources:** Anderson PDF422 Eqs11.52–53; Rao PDF3 Eq15; Rice PDF21 §3.4 (quality of kernel mesh affects downstream convergence).

The equations require continuum residual zero. No examined source defines the repository physics weight0.05, raw tolerance0.005deg, BVP residual tolerance, or a universal number of characteristic nodes. If a scaled objective accepts raw0.012deg while the independent raw gate accepts at most0.005deg, the objective acceptance and audit acceptance are inconsistent numerical policies. Correct the normalization/constraint architecture and demonstrate reduction of the independent raw residual; do not loosen the independent gate merely to report success. Conversely,0.005deg must not be called a literature-derived physical threshold. It is a recorded repository accuracy target.

Rice PDF21 also warns that increasing initial points excessively can make RRCs too close or crossing; refinement must vary relevant controls jointly and compare common coordinates, not assume every increase in n improves the mesh.

### TB-04 — Geometric viability is not a universal bound on neighboring-state increments

**Classification:** independent diagnostic requirement; the current decision to keep raw neighbor increments report-only is supportable.

Young2012 `DOC_YOUNG_THESIS`, PDF62–63 / printed51–52, Eq3.10 and listed tests inspect *geometric intersection/order* of characteristic segments. It explicitly requires checking after the non-simple region too, because a viable kernel alone does not establish a viable whole solution. Anderson PDF423 explains axisymmetric characteristic invariants vary with spatial location. None of these sources establishes a universal maximum |delta M| or |delta theta| per edge as a physical shock criterion.

Frey&Hagemann2000 `DOC_DOI_10_2514_2_5593`, PDF2–4 and6, explains internal shocks and particular flow structures in thrust-optimized contours. This supports the possibility of physical coalescence, but does not prove every numerical fold in this repository is a physical shock. Preserve geometry checks, contextual refinement, and explicit stop reasons; require convergence or independent CFD evidence before assigning a physical location.

### TB-05 — Published Nozzle-B comparison uses different throat data today

**Classification:** confirmed benchmark applicability/documentation gap; do not silently alter production defaults.

Rao1958 PDF3 bottom and PDF4 top give Ru=1.5Rt and Rd=0.45Rt and Sauer-based TT' with wall Mach1.103 for the worked construction. PDF5 says NozzleB is obtained with ME=2.6 and zero ambient pressure from this construction. Tables2–3 supply its shape and Cf.

`RaoSolverConfig` defaults Rd=.382Rt; current `rao1958_nozzle_b.json` does not override it, and `raosim/benchmarks.py` does not propagate a benchmark-specific throat radius to the config. It also uses the KL branch, not the original Sauer construction. The current comparison can validly assess the published shape/integral envelope within declared tolerance, but is not an exact boundary-data reproduction. The code comment attributing the .382 convention directly to Rao1958 is unsupported by this held source.

Potential bounded task: expose explicit benchmark throat options, retain .382 legacy profile, and add a source-profile comparison under .45 only after its starting-line assumptions are stated. A source-profile result still needs every internal gate; matching .45 alone is insufficient.

### TB-06 — Two M-star normalizations must not be conflated

**Classification:** valid stationarity scaling difference; transonic-velocity convention unresolved.

Rao1958 PDF4 Eq17 defines its parameter Mstar=M/sqrt(2+(gamma−1)M²). Anderson PDF94 / printed78 defines characteristic Mach Mstar=V/a*, and PDF98 / printed82 Eq3.37 gives M²=2Mstar²/[(gamma+1)−(gamma−1)Mstar²]. Anderson's Mstar is sqrt(gamma+1) times Rao's. At fixed gamma this constant can be absorbed in the stationarity constant C: this difference alone does not invalidate `gas_dynamics.mstar_from_M`.

`transonic_kernel.TransonicState` fields say `u_over_astar` and `v_over_astar`, but `kliegel_levine` returns M=hypot(U,V). If the stated normalization is accurate, these quantities are not the actual local Mach without Anderson Eq3.37 conversion. However the original KL coefficient source is not held, and source-visible NASA compatibility intentionally uses legacy conventions. Therefore do not change all branches or global defaults solely from these labels. Obtain original normalization/coefficient evidence, preserve a named compatibility branch, and separately validate any corrected branch.

### TB-07 — KL coefficient claims exceed available primary evidence

**Classification:** uncertain mathematical corrections / confirmed provenance gap.

Local inventory and manifest search found no original Kliegel–Levine1969 paper, Hall1962 paper, or Zucrow&Hoffman Vol2 monograph. Anderson and an independent KL experimental comparison are held. `transonic_kernel.py` nevertheless states exact “cross-check Kliegel–Levine1969 Table1” for selected coefficient changes and numeric applicability limits. The held Rice report supplies only high-level TT' description, not the coefficient table. A parity/power-order argument is useful investigation but cannot establish a disputed coefficient's exact sign/value.

`DOC_NTRS_19690063911` (Cuffel, Back, Massier1969; DOI10.2514/3.5349), PDF2 / printed1365, independently verifies the toroidal expansion parameter1/(1+rc/rt), reviews KL three-term predictions, and reports good throat-plane/reasonable centerline agreement but worse wall agreement for rc/rt=.625. It does not supply the disputed coefficient table. Thus current KL math should remain labeled partially justified until original evidence or an independently derived recurrence is available. This does not establish the coefficient edits are wrong.

### TB-08 — Conservation baseline must match transonic assumptions

**Classification:** requirement to preserve distinct model comparisons.

Cuffel etal PDF2 reports inviscid 2-D/ideal1-D mass-flow coefficient0.982 from KL theory versus measured0.985 for its rc/rt=.625 case. Therefore a nonuniform transonic start line need not carry precisely the ideal1-D sonic-throat mass. Downstream mass conservation should compare like-for-like fluxes within the same modeled flow (e.g., against its own starting line). Separate that conservation error from ideal1-D mass normalization/discharge coefficient. No source supports forcing every throat model to perfect1-D mass to make a gate pass.

Rao PDF4 Eq19 requires m(BD)=m(DE); it does not justify using a partial DE momentum integral as the full nozzle thrust. Rao PDF2 Eq2 and PDF3 setup use the complete CDE control surface. A mass-fraction scaling of a partial integral is not a substitute for reconstructing its missing segment.

### TB-09 — Source coverage and metadata are not exhausted by the graph

The independent original-file inventory identified load-bearing Anderson under a hash filename and the Cuffel experimental comparison inside `experimental_validation/`. These could be missed by searching only Rao/MOC filenames. Manifest metadata is not always publication metadata: Young's record title is an abstract fragment and identifierNASA CR-178628 although the actual front page is an Auburn MS thesis, August4,2012; the MIT record authors field is “Sapient” whereas the original identifies Prof.Manuel Martinez-Sanchez. No knowledge-store mutation was made in this research task. Inventory JSON preserves both original inspected title and existing manifest title so discrepancies remain reviewable.

## Supported scope versus possible enhancement

SP-8120 `DOC_NTRS_19770009165`, PDF27 §2.1.2.1.1 and PDF80 §3.1.2.1.1 distinguishes mathematical optimum and approximate transonic/gas-property input; variable equilibrium/nonequilibrium gas properties and boundary-layer corrections require their own modeling. PDF81 §3.1.2.1.2 recommends near-optimum parabolic contours under tolerable losses and where rigorous optimization does not exist. RP-1104 `DOC_NTRS_19830016278`, PDF6–7, concerns perfect bell nozzles and their truncation, not a universal table of Rao optimum parabolic angles. These comparisons must retain family and assumptions.

Young PDF94 Table4.2 distinguishes isentropic compatibility from finite-rate reacting/rotational compatibility with extra source terms. Adding variable gamma alone does not implement full chemically reacting MOC. Such extension and shock-fitting are enhancements beyond the immediate confirmed failures; neither should be introduced speculatively while fixing D attachment and acceptance consistency.

## Research limits

- No full audit of every page in every inventory item is claimed. Inspected page windows and exact hashes are in `literature_inventory.json`.
- No external literature download or web claims were needed. Original KL/Hall coefficient evidence remains unavailable locally.
- No universal finite residual, mesh-increment, or shock-location tolerance is inferred from prose.
- Boundary-elimination proposal is a mathematical-equivalence implementation choice anchored in exact source boundary conditions; regression/independent numerical validation remains Sol A/Sol B's responsibility.


## Targeted follow-up: endpoint constraints and invariant independence

### TB-10 — Free-exit transversality is not mandatory for a fixed-end problem

**Source:** Rao1958 PDF3 / printed379, “Solution of the Problem”, Eqs6,10,14 (original page rendered and inspected).

The derivation explicitly says only nozzle length is prescribed, so the exit-radius variation delta(y_E) is nonzero. The endpoint term in Eq6 is delta(y_E) [f1+lambda2 f2+lambda3 f3]_E. Because delta(y_E) is arbitrary in that free-exit problem, Eq10 follows; substitution of Eqs11–13 then gives Eq14.

If Rt and epsilon are fixed, y_E=Rt sqrt(epsilon) is fixed and delta(y_E)=0. Consequently Eq10/14 is not a necessary natural boundary condition for this different constrained problem. The current fixed-L/fixed-epsilon solver's default omission of its transversality block is therefore not, by itself, a missing capability or mathematical bug. Preserve the distinction in the final audit. Reproducing the source's *free-exit optimum* requires the original endpoint condition; it may be reported independently for a fixed-end run selected to coincide with a published optimum, but must not silently become a generic fixed-end acceptance requirement.

### TB-11 — Eq13 follows from Eq12 plus compatibility in smooth constant-gamma flow

**Source:** Rao1958 PDF3 / printed379 Eqs12,13,15 and PDF4 Eq18. The following is a project algebraic derivation, not a verbatim additional source equation.

Set q=sqrt(M²−1), A=1+(gamma−1)M²/2, mu=asin(1/M), and

- I1 = W cos(theta−mu)/cos(mu) (Eq12 invariant),
- I2 = y rho W² sin²(theta) tan(mu) (Eq13 invariant),
- R = dtheta − q/(M A) dM + sin(mu)sin(theta)/(y sin(theta+mu))dy (Eq15 differential residual).

Using W proportional to M/sqrt(A), rho proportional to A^(−1/(gamma−1)), dmu/dM=−1/(M q), and ordinary differentiation gives the exact differential identity

    d log(I2) = (1+q cot(theta)) d log(I1) + (q+cot(theta)) R.

For example, the logarithmic derivative coefficients of I2 are (2/M−gamma M/A−M/q²) for dM, 2cot(theta) for dtheta, and 1/y for dy; substitution on the right reproduces each coefficient. Therefore I1 constant and exact compatibility imply I2 constant on a smooth DE. Do not classify the absence of a separate Eq13 residual as an independent missing continuum equation. An independently computed I2 drift can still reveal discrete integration or under-resolution errors and is a useful diagnostic/refinement test. Adding it to a least-squares stack may alter finite-grid weighting and must be justified numerically rather than described as restoring absent physics.

Domain: constant gamma and smooth isentropic flow, y>0, M>1, nonzero sin(theta), and well-defined characteristic direction. The logarithmic identity needs limiting treatment at the axis or theta=0; it cannot be applied there by direct division. A constant factor between Rao/Anderson speed normalizations cancels in the logarithmic derivative.

Research derivation sanity check: central directional finite differences at 27 smooth states (gamma=1.2,1.4,5/3; M=1.3,2,3.5; theta=8,20,35 degrees) gave maximum absolute identity error 3.03e-09 using step1e-6. This checks the independent algebra, not the repository implementation or a physical validation gate.
