# Rao B3/B5 host evidence — 2026-08-24

## Scope and verdict

This note records observed host behavior for the new B3 opt-in post-solve
validator and the B5 $\varepsilon\times L_\%$ declared-domain infrastructure.
It is an implementation evidence record, not a solver qualification report.
The 2026-08-26 resolution-policy correction below supersedes the original
wording that attributed all three probes to one neighbor-smoothness failure.

- **B3:** the typed host validation contract has landed, but full qualification
  has **not**. The historical probes below returned `UNKNOWN` for two distinct
  reasons: the BDE probes failed the former raw neighbor-state-increment hard
  screen, while the coupled probe failed a separate characteristic-crossing
  audit. Raw increments are now diagnostics only; the corrected 24×24/32×32
  BDE profile passes the typed common-coordinate refinement comparison. That
  evidence is fixed-context, not universal B3 qualification.
- **B5:** the typed declared-domain schema, fail-closed lookup, NumPy
  D-attachment root refinement, kernel-resolution evidence, and Rao Eq. (5)
  classification have landed. The historical v1 H2 slice omitted the kernel
  angular-spacing and mass-flow policies and is rejected fail-closed. Its
  0.25°/2% schema-v2 replacement has now been regenerated, independently
  loaded, and hash-checked. B5 remains partial beyond that fixed context and
  does not establish full-BVP convergence.
- **Default path:** `reevaluate()` runs B3 only when given an explicit
  `VariationalPostSolveConfig`. Omitting it adds no variational solve and
  therefore zero added variational-solver runtime to the analytic TOP path.

## Host and timing context

The observations were collected on the project host: Apple M3 Pro (`arm64`),
macOS 15.7.5, with the CPU JAX environment (Python 3.12.13, JAX 0.6.2, NumPy
2.4.6, SciPy 1.17.1). Wall times are individual observations on that host, not
benchmark guarantees and not promises for another machine, cache state, or
solver configuration.

## Historical B3 real-probe record

All three probes reached a numerically converged BVP, an evaluable/valid Rao
Eq. (5) result, and available direct full-C-D-E performance. None is a B3 pass:
a downstream numerical/topology gate failed, so the typed outcome is `UNKNOWN`
and the geometry/performance discrepancy is not adjudication-eligible.

| Probe | Declared context | Observed wall time | BVP | Eq. (5) | Full C-D-E | Failing audit | Typed result |
|---|---|---:|---|---|---|---|---|
| 12×12 coupled | $\gamma=1.4$; coupled wall | 16.2 s cold; 12.5 s immediate repeat | pass | pass | available | characteristic-crossing audit; this path does **not** compute the BDE raw-neighbor screen | historical `UNKNOWN` |
| 24×24 BDE | $\gamma=1.4$, $P_a/P_0=0.01$, $\varepsilon=10$, $L=80\%$, $R_t=0.020$ m, `physics_weight=1.0`; characteristic formulation, BDE wall, `coupled_wall=False` | 10.93 s | pass | pass | available | former hard screen: raw neighbor $\max|\Delta M|=0.11813>0.10$; worst edge 4.179 mm | historical `UNKNOWN` |
| 32×32 BDE | same physical/numerical context as the 24×24 BDE row, except resolution | 14.96 s | pass | pass | available | former hard screen: raw neighbor $\max|\Delta M|=0.13752>0.10$; worst edge 4.759 mm | historical `UNKNOWN` |

The BDE cases passed characteristic compatibility and direction, zero-crossing,
cell-orientation, wall-tangency, axial-mass, seam/topology, and BVP checks. With
only the kernel angular-spacing limit changed from 0.5° to 0.25°, the raw maxima
fell to 0.06999 (24×24) and 0.06705 (32×32), and all existing BDE gates passed.
The independently interpolated 24×24/32×32 Mach fields differed by only 0.00188
at common physical coordinates. This evidence diagnoses under-resolution of an
adaptive characteristic edge; it is not evidence of a flow discontinuity.

The implemented B3 profile now solves 24×24 and 32×32 BDE rungs at the same
0.25°/2% kernel policy and compares only the wall-cropped physical B-D-E grids.
A fresh reproduction measured maximum common-coordinate differences of
$\Delta M=0.0019034$, $\Delta\theta=0.041943^\circ$, and
$\Delta(p/p_0)=1.1467\times10^{-4}$, with 93.74% minimum bidirectional field
overlap. The wall-radius maximum was $0.001663R_t$; $\theta_E$, kernel-D
fraction, and full-C-D-E thrust coefficient changed by $0.01037^\circ$,
$4.11\times10^{-5}$, and $6.04\times10^{-5}$ relative, respectively. The typed
repository profile is therefore $0.005$, $0.05^\circ$, $0.001$, 90% overlap,
$0.002R_t$, $0.05^\circ$ for each design angle, $5\times10^{-4}$ for D
fraction, and $10^{-3}$ relative thrust. These are explicit convergence-screen
values, not literature-derived physical criteria.

The former 0.10 neighbor-Mach hard limit is **repository screening policy**, not a tolerance
stated by Rao 1958/1999 or another qualification authority. The BDE bundle's
$|\Delta\theta|\leq2^\circ$ and $|\Delta(p/p_0)|\leq0.02$ limits have the same
status. The implementation now records the bundle as diagnostics rather than
including it in physical-topology `passes`. Resolution acceptance
must use spacing metadata and common-coordinate comparison between independently
refined, nonnested meshes. A universal $|\Delta M|/\Delta s$ cutoff would still
be a repository heuristic, not a shock criterion.

The literature-supported physical boundary is different. Rao 1958 requires
continuous interior variables and characteristic compatibility (source PDF
p. 3, Eq. 15). Rao, Beck, and Booth 1999 identify loss of positive radial
increment $\mathrm dR$ and the $\mathrm dR=0$ boundary as the limit of the
shock-free construction (source PDF pp. 3--4, Eqs. 5--6). Rice's 2003 JHU/NASA
report identifies `DTHETAB Max` as a user-selected grid control, gives
0.25°--0.5° as nominal values (§3.2.8, source PDF pp. 17--18), and explicitly
describes grid sufficiency as problem-dependent (§3.4, source PDF p. 21). None
defines a universal finite-edge $|\Delta M|$ limit.

Primary sources checked for this correction are
`propulsion_texts/rao1958.pdf`, `propulsion_texts/rao1999.pdf`, and
`propulsion_texts/20030067852.pdf`. Numerical source parity was checked against
`Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp`.

The checked-in JHU/NASA reference `MOC_Grid.plt` reaches a direct same-index
adjacency $|\Delta M|$ of 0.27294. Its topology/indexing is not identical to the
current BDE link graph, but it independently rules out 0.10 as a universal
NASA/JHU acceptance criterion.

### Separate coupled-wall crossing diagnosis

The 12×12 coupled audit remains fail-closed and is not repaired by the BDE
smoothness-policy change. A reproduced 0.25°/2% run reports 55 all-pairs
straight-segment intersections, while its same-target-row local-fold count is
zero. The retained samples are intersections between long C-minus chords to the
symmetry axis and characteristic links landing in earlier rows. This strongly
suggests that the present chord-based all-pairs audit conflates axis treatment
with a local folded cell, but that diagnosis is not yet a qualification result.
The implementation therefore records both statistics and continues to use the
conservative all-pairs count as the coupled-path gate. Resolving it requires a
separate axis-aware topology criterion and regression evidence; it must not be
hidden by the BDE common-coordinate convergence result.

`ContourDiscrepancyTolerances` likewise has no implicit defaults: each caller
must state its maximum/rms radius, divergent-length, and thrust-coefficient
error policy. Those limits are repository adjudication choices, not literature
qualification limits. Only after the appropriate resolution and topology gates
pass can TOP-versus-variational discrepancies be interpreted. An `UNKNOWN`
result must not be reported as a variational validation pass or replaced
silently by TOP geometry.

### Numerical-policy inventory

The following controls are collected by the typed Rao/MOC numerical policy and
must remain documented separately from Rao's physical valid-region criterion:

- kernel wall-angle spacing: 0.25° for the B3 validation profile; the original
  JHU/NASA implementation exposes it as input and historically defaults to
  0.5°;
- kernel mass-flow drift: the historical Python default was 5%, versus 2% in
  the visible JHU source; the typed validation/existence policy now explicitly
  uses 2%. Both values are implementation diagnostics rather than a physical
  criterion;
- BDE axial mass audit: seven cuts, 500 radial samples, 98% interpolation
  coverage, at least four valid cuts, and 0.5% relative error;
- Mach-line direction: 0.05°; wall tangency: 0.25°; crossing endpoint band:
  $10^{-3}$ of median edge length; cell-area band: $10^{-12}$ of median area;
- full-C-D-E reconstruction: $0.005R_t$ D-projection error,
  $|\Delta M_D|\leq0.02$, $|\Delta\theta_D|\leq0.5^\circ$, and 2% mass
  mismatch; full-surface thrust consistency: 5%;
- BVP residual policy: $2\times10^{-3}$ by default; Rao Eq. (5)'s $10^{-9}$
  radial-increment band is instead a numerical roundoff treatment around a
  physical sign boundary.
- B3 nonnested physical-grid convergence: maximum $|\Delta M|=0.005$,
  $|\Delta\theta|=0.05^\circ$, $|\Delta(p/p_0)|=0.001$, wall difference
  $0.002R_t$, design-angle difference $0.05^\circ$, D-fraction difference
  $5\times10^{-4}$, thrust-coefficient relative difference $10^{-3}$, and at
  least 90% overlap in each interpolation direction. These are repository
  acceptance values selected for the validated 24×24/32×32 profile.

## B5 fixed-slice artifacts, timing, and policy

Observed building-block times on the same host were approximately 1.16 s for an
11×11 internal existence scan, 2.2 s for one D-attachment geometry-root
refinement, and 5.3 s for one 8/12/16 kernel-resolution ladder. These are
component observations, not per-knot guarantees. The accepted fixed-slice policy
adds a 24-point kernel rung. The complete dense 65-knot generation took
approximately 68 minutes on this host; the earlier 11–20 minute planning estimate
was low.

The first H2 slice was generated under the following historical policy. Its
kernel march also inherited the then-implicit 0.5° angular limit and 5%
mass-flow diagnostic; because neither was recorded in v1 identity, the artifact
is provenance evidence only and is not accepted by schema-v2 lookup.

- external $\varepsilon$ axis:
  `4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50`;
- external $L_\%$ axis: `60, 70, 80, 90, 100` (65 knots total);
- fixed physical slice: $\gamma=1.4$, $P_a/P_0=0.01$,
  $R_t=0.020$ m, $R_d/R_t=0.382$, and $R_u/R_t=1.5$;
- closure/numerics: NumPy-host smooth D-attachment geometry root,
  Kliegel–Levine start line, 8/12/16/24 kernel ladder, 24 D-E points, mass
  stop, geometry scan, unit residual weights, and maximum Mach 12;
- per-sample evidence: root success and normalized geometry residual, kernel
  convergence in $\theta_B$ and `kdf`, and Rao Eq. (5)
  evaluability/validity/minimum $\mathrm{d}R/R$;
- lookup interpolation policy: maximum $\varepsilon$ spacing 5 and maximum
  length-percentage spacing 10, with all bracketing samples required to be
  converged, to pass the declared kernel-refinement policy, and to be Eq.
  (5)-valid.

The historical numerical thresholds are a normalized geometry-residual limit of
$10^{-6}$, final-two-rung $\theta_B$ change no greater than 0.05°, and final-
two-rung `kdf` change no greater than $5\times10^{-4}$. These values and the
lookup spacings are **repository policy**, not literature-derived validity
thresholds. They are stored explicitly in the artifact.

### Legacy artifact identity and measured classifications

- path:
  `raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json`;
- byte length: 101,932;
- canonical artifact SHA-256:
  `a85de663fffced5559e92760fe933a7edd0ca269fc227aae5908c0eccbf87b19`;
- source model:
  `raosim.rao_existence_scan:epsilon-length-fixed-context-d-attachment-domain:v1`;
- Eq. (5) authority:
  `Rao-Beck-Booth-AIAA-99-2584-1999-pdf-page-4-Eqs-2-3-5-6`.

This v1 artifact is now deliberately incompatible with the runtime contract.
It must not be silently assigned the new 0.25°/2% policy. Regeneration must use
schema/source model v2, a `_v2.json` filename, and a newly reported content hash.

The JSON is strict and contains no NaN or infinity tokens. Across its 65 knots,
40 are `converged_valid` and 25 are `not_converged`; `converged_invalid`,
`numerical_failure`, `under_resolved`, and `unevaluable` are all zero. Across
the 48 cell midpoints, fail-closed lookup returns 23 `supported`, 11
`refused_not_converged`, and 14 `unknown_under_resolved` results.

### Current schema-v2 artifact identity

- path:
  `raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v2.json`;
- byte length: 102,151;
- canonical artifact SHA-256:
  `2b2d1f4017f88d230cd6857dbc1e1a859ac5d3f611aa14a60f9e413d2c5949a9`;
- fixed-context identity SHA-256:
  `a3095bad66a56a24554f7e234b9133f967110ee4958998193d385f32d5d7d277`;
- source model:
  `raosim.rao_existence_scan:epsilon-length-fixed-context-d-attachment-domain:v2`;
- explicit kernel policies: `kernel_dtheta_limit_deg=0.25` and
  `kernel_massflow_rel_tol=0.02`.

The v2 artifact round-trips through the strict loader with its recorded hash.
Its 65-knot classification grid is unchanged from v1: 40 `converged_valid`, 25
`not_converged`, and zero in the other four classes. Its 48 midpoint lookups
likewise produce 23 `supported`, 11 `refused_not_converged`, and 14
`unknown_under_resolved`. The unchanged classifications are a numerical result,
not permission to retrofit the missing controls onto v1; only the v2 identity
is current evidence.

### Schema-v2 regeneration command

```bash
.venv-jax/bin/python scripts/rao_existence_scan.py \
  --Rt 0.020 --epsilon 10 --gamma 1.4 --pa-over-p0 0.01 --length-pct 80 \
  --kernel-dtheta-limit-deg 0.25 --kernel-massflow-rel-tol 0.02 \
  --theta-b-center-deg 24 --theta-b-span-deg 10 --theta-b-count 21 \
  --kdf-min 0.005 --kdf-max 0.90 --kdf-count 25 \
  --models smooth --stop-at mass --scan-mode geometry --third-residual auto \
  --n-kernel 8 --n-de-points 24 --root-maxfev 80 \
  --resolution-convergence 8,12,16,24 \
  --domain-theta-seed-policy rao_chart_centered \
  --domain-chart-theta-span-deg 40 --domain-chart-theta-count 21 \
  --domain-epsilon-values 4,5,6,8,10,15,20,25,30,35,40,45,50 \
  --domain-length-pct-values 60,70,80,90,100 --domain-model smooth \
  --domain-residual-limit 1e-6 --domain-theta-resolution-tol-deg 0.05 \
  --domain-kdf-resolution-tol 0.0005 \
  --domain-max-epsilon-spacing 5 --domain-max-length-pct-spacing 10 \
  --domain-output raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v2.json \
  --output-dir builds/rao_domain_h2_v2
```

This command has completed and the resulting v2 JSON was independently loaded,
hash-checked, and pinned by regression test. The product declares support only
for its exact context. It remains a NumPy
D-attachment root/kernel-refinement/Eq. (5) domain: passing its last-rung change
policy is **kernel-resolution evidence only**, not proof that every
mesh-dependent quantity or the full variational BVP is converged. Broader claims
require separately generated and reviewed slices for other $\gamma$, ambient
pressure ratio, throat geometry, closure model, numerical context, and a fuller
common-coordinate mesh-convergence study.

## Historical software verification

The pre-schema-v2 B3/B5 tree recorded on 2026-08-24 passed the repository-wide
non-slow gate:

```text
.venv-jax/bin/python -m pytest -q -m 'not slow'
2148 passed, 26 deselected, 3 warnings in 1586.04 s
```

The three warnings are the existing Matplotlib animation-lifetime warnings and
CadQuery `save` deprecation warning. The focused Rao/JAX regression slice also
passed 68 tests with 5 deselected. `compileall` and `git diff --check` passed.
A fresh isolated wheel build at that revision contained the 101,932-byte v1
domain JSON; reading that wheel member reproduced the historical artifact hash
and the 40/25 classification counts above. This historical packaging check does
not make v1 compatible with the current schema-v2 numerical identity.

## Current schema-v2 software verification

The completed 2026-08-26 tree passed the repository-wide non-slow gate:

```text
.venv-jax/bin/python -m pytest -q -m 'not slow'
2183 passed, 26 deselected, 3 warnings in 1542.89 s
```

The warnings are the same two Matplotlib animation-lifetime warnings and one
CadQuery `save` deprecation warning. `compileall` and `git diff --check` passed.
An isolated wheel contained the exact 102,151-byte v2 JSON. A real B3 run at
the documented fixed context loaded that identity as `supported`, solved the
24×24 and 32×32 BDE rungs, passed their common-coordinate comparison, and
reached the final `within_tolerances` result under an explicitly supplied
TOP-discrepancy policy.
