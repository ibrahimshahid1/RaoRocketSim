# Subsystem literature status — 2026-10-17

**The subsystems have not been comprehensively verified against literature.** Seven of
61 models have current structured implementation-versus-source reviews. Those seven
include adverse and conditional findings; they do not establish whole-subsystem
correctness. No model has established physical validation or hardware qualification.

This read-only audit compares current records and live code/source fingerprints with
[LREKit Consolidated Technical Review and Forward Program (1).pdf](</Users/ibrahimshahid/Downloads/LREKit Consolidated Technical Review and Forward Program (1).pdf>)
(October 7, PDF pp.11–15 and 28–37). Document proposals are audit content, not new
instructions. Existing source comparisons were checked for freshness, not repeated
from their governing PDF equations in this audit.

Snapshot: `2026-10-17T17:56:40.412102+00:00`; HEAD `550d5c26bc2679b43b027f612805eb18516e8343`. The model store
was harvested at `2026-10-17T17:43:40.419892+00:00`. Its
SHA-256 is `1ba7c1410cea0889203a02a3040013fb0d651c4edd81467bcdf08bc478ff7ff3`. The matching
[JSON snapshot](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/knowledge/reports/SUBSYSTEM_LITERATURE_STATUS_2026-10-17.json) contains all 61 model states, per-source review details,
source hashes and reproducible methods. Follow-on source checks are tracked separately;
renewing a claim does not create a structured model review.

## What the counts mean

The PDF's **11 source-checked models** (pp.11 and 35) meant 2 supported, 8 partial and
1 not found in the September 20 source-support snapshot. That historical measure did
not require the current combination of PDF fingerprint and exact implementation
fingerprints.

Today **7 models have current structured code/source reviews**, spanning 8 retained
source reviews because Bartz retains both the original paper and SP-125. Separately,
**12 models have positive current citation assessments**: 3 `SUPPORTED` and
9 `PARTIALLY_SUPPORTED`. Five of those 12 have no structured implementation comparison.
One further citation is `NOT_FOUND_IN_SOURCE`, and 48 are current `NOT_CHECKED`.
Seven historical positive statuses remain visible but lack current PDF binding.
These counts describe different evidence and freshness requirements; 11 versus 7 is
not a regression or a claim that four completed comparisons were lost.

## Coverage of all 17 registry subsystems

The source columns describe citation support. Even `SUPPORTED` does not establish
full implementation fidelity, source validity at the intended operating point, or
physical validation.

| Subsystem | Models | Current structured reviews | Source supported | Source partial | Not found | Not checked |
|---|---:|---:|---:|---:|---:|---:|
| altitude | 1 | 0 | 0 | 0 | 0 | 1 |
| cad | 2 | 0 | 0 | 0 | 1 | 1 |
| chamber | 2 | 0 | 0 | 0 | 0 | 2 |
| flow | 1 | 0 | 0 | 1 | 0 | 0 |
| injector | 11 | 1 | 1 | 1 | 0 | 9 |
| interface | 1 | 0 | 0 | 0 | 0 | 1 |
| materials | 1 | 0 | 0 | 0 | 0 | 1 |
| nozzle | 3 | 1 | 1 | 1 | 0 | 1 |
| performance | 3 | 0 | 1 | 1 | 0 | 1 |
| pump | 7 | 3 | 0 | 3 | 0 | 4 |
| separation | 1 | 0 | 0 | 0 | 0 | 1 |
| spray | 16 | 0 | 0 | 0 | 0 | 16 |
| structure | 2 | 0 | 0 | 0 | 0 | 2 |
| thermal | 7 | 2 | 0 | 2 | 0 | 5 |
| thermochemistry | 1 | 0 | 0 | 0 | 0 | 1 |
| throat | 1 | 0 | 0 | 0 | 0 | 1 |
| trajectory | 1 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **61** | **7** | **3** | **9** | **1** | **48** |

Spray and injector still contain 27 of 61 models. All 16 spray models lack a current
structured implementation-versus-source review, and only one of 11 injector models
has one. The pump reviews cover three of seven models; thermal two of seven; nozzle
one of three. Every other subsystem has zero structured reviews.

## Current structured review findings

| Model | Fidelity | Scope and limit |
|---|---|---|
| `injector.son2017_movable_continuous_gap` | `EQUIVALENT` | The movable-gap relation and consistent inversion agree; printed Eq. (2) has a thickness omission. Hydraulic, actuator and hardware behavior remain unvalidated. |
| `nozzle.rao_top_bezier` | `BAD_ATTRIBUTION` | The canted-parabola concept has support; the angle tables disagree with the held chart. This review finds a problem, not source equivalence. |
| `pump.electric_drive` | `CONDITIONALLY_EQUIVALENT` | Core Lee equations agree subject to the smooth maximum in the optimizer objective. The default technology set and per-stream/system extensions are separate assumptions. |
| `pump.inducer_npsh` | `REPOSITORY_EXTENSION` | Constant-lead geometry and conditional source selections have support; advisory suction-speed/eye sizing and cavity, head-rise and stability predictions lack whole-model equivalence. |
| `pump.meanline_coefficients` | `REPOSITORY_EXTENSION` | Head/flow coefficient definitions agree. Net-area continuity and remaining geometry choices are project-derived; the full source performance procedure is not established. |
| `thermal.bartz` | `EQUIVALENT` | Gas-side coefficient, sigma at the reviewed omega and Prandtl estimate agree within the recorded scope. Viscosity coefficient, locator, clipping/generalized branches and applicability caveats remain. |
| `thermal.sieder_tate` | `DIFFERENT_FORMULA` | The implemented coefficient and Prandtl exponent differ from held SP-125 RP-1 forms. The original Sieder & Tate paper is not held; the comparison is citation-only. |

The source comparison records, not the fidelity label alone, define their scope.
The non-equivalent TOP and Sieder–Tate reviews retain citation paths rather than
asserting implementation support. The Bartz review does not certify all gas-property
estimators or every use of the correlation.

## Source assessments without structured implementation review

| Model | Current citation assessment |
|---|---|
| `flow.boundary_layer_displacement` | `PARTIALLY_SUPPORTED` |
| `injector.compressible_orifice` | `PARTIALLY_SUPPORTED` |
| `nozzle.rao_variational_bvp` | `SUPPORTED` |
| `performance.frozen_variable_cp_quasi1d` | `PARTIALLY_SUPPORTED` |
| `performance.isentropic_area_mach` | `SUPPORTED` |

These are useful evidence routes, not five additional completed code/source reviews.
`cad.kernel_topology` is separately `NOT_FOUND_IN_SOURCE`: its cited NURBS lecture
contains no supporting OpenCascade topology, solid-validity, STEP-exchange,
re-import or volume-check treatment. That finding does not invalidate a kernel unit
test; it invalidates this citation as scientific support for the stated policy.

## Historical assessments without current PDF binding

| Model | Retained historical assessment | Current assessment |
|---|---|---|
| `altitude.attached_thrust` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |
| `chamber.lstar_volume` | `SUPPORTED` | `NOT_CHECKED` |
| `chamber.shoulder_fill_fraction` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |
| `injector.d2_evaporation` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |
| `injector.incompressible_orifice` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |
| `injector.momentum_resultant_angle` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |
| `injector.n_tau_stability` | `PARTIALLY_SUPPORTED` | `NOT_CHECKED` |

The missing fingerprints mean these old comparisons cannot establish current support.
They have not been refuted merely because the binding is absent. Any new page-level
SP-125 claim verification remains separate from a renewed `chamber.lstar_volume` or
`chamber.shoulder_fill_fraction` implementation comparison. No fillet-radius equation
or shoulder heuristic support is inferred from chamber-volume geometry.

## Comparison with the PDF's validation program

PDF pp.28–33 distinguish V0 mathematical verification, V1 independent implementation
comparison, V2 comparison with physical measurements and V3 coupled system physical
validation. The dataset tables identify potential and packaged evidence; they do not
show that every listed source has been mapped to inputs, run, compared with signed
uncertainty-aware deltas and accepted for the current implementation.

This audit establishes no new V0 or V1 execution result and no V2 or V3 validation.
It does not rerun the Rao, MOC, CEA, coolant, material or spray datasets. A source
transcription check, formula equivalence or software test cannot promote a model to
V2/V3. Physical validation remains unestablished for all 61 models:
`hardware_qualified=false` and `experimentally_validated=false` in every record.
The conservative registry-derived physical states are 44 `REPORTED_NOT_VALIDATED`
and 17 `UNKNOWN`.

PDF pp.34–37 ask for the remaining model comparisons, code-derived execution reach,
traced optimizer execution, leverage, typed applicability and validation/configuration
objects. The knowledge infrastructure and source checks have progressed, as recorded in
[CONTINUATION_2026-10-17.md](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/knowledge/reports/CONTINUATION_2026-10-17.md),
but the full program is unfinished. An artifact hash or reference manufacturing plan
is not a measurement, calibration, executed process or qualified article.

## Current code and source freshness

- All seven distinct PDFs underlying the eight structured source reviews match their
  manifest/review hashes. All retained structured source comparisons are `CURRENT`.
- All 61 registry-owned metadata records match the live `MODEL_REGISTRY`.
- Read-only AST reconciliation verified all 438 exact software locators across 104
  files, with zero changed definitions, corrected locators or updated records.
- Live bounded execution-reach analysis matches stored reach for all 61 models.

These checks detect no drift from the recorded comparisons. They do not prove that
the comparisons, models or original publications are physically correct.

The bounded AST mapping finds 8 `CODE_INSPECTED` models and leaves 53 `UNKNOWN`.
Only `thermal.bartz` and `thermal.sieder_tate` have a discovered static path from the
optimizer root. Optimizer reach states are 2 `STATIC_PATH_FOUND`,
8 `NO_STATIC_PATH_FOUND` and 51 `UNKNOWN`. The other located paths cover the opt-in
spray API and traditional-analysis coupling. `NO_STATIC_PATH_FOUND` is not proof of
unreachability: the mapping deliberately covers a limited set of files and direct calls.
Every runtime trace remains `NOT_RUN`, dynamic dispatch `UNKNOWN`, and optimizer
leverage `NOT_MEASURED`. Keyword-derived `promotion_status` is not a substitute for
these observations. The PDF's proposed traced MDO and Jacobian-based leverage work
is therefore still open.

## G1–G34 remain open engineering or source issues

No governing equation, coefficient, default, engineering gate or geometry was changed
by this status audit. The detailed source/code findings remain in
[REMAINING_SCIENTIFIC_GAPS.md](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/knowledge/reports/REMAINING_SCIENTIFIC_GAPS.md).
The grouped register below covers every G1–G34 item. Source-internal inconsistencies
are retained evidence; they are not all software defects or requests to alter the source.

| IDs | Kind | Current unresolved issue |
|---|---|---|
| G1–G3 | Engineering/source attribution | Rao/TOP exit and initial angle tables disagree with held charts; the cited origin paper is absent. No contour table was changed. |
| G4–G5 | Engineering assumptions | Default electric-drive technology values lie outside held evidence; pump specific-mass values have different assembly bases. No default was selected or changed. |
| G6 | Engineering gate | The 0.90 blockage ceiling exceeds the held single-row evidence, which stops at 0.7; two-row observations have different definitions and damage caveats. |
| G7–G9 | Engineering assumptions and attribution | 800 K remains a repository wall-temperature rule; the aspect-ratio limits and Pizzarelli validity rationale remain unresolved. |
| G10–G14 | Attribution and locators | Freeberg and Son locators, Son author list, an absent Hwang quotation, and the mid-band sizing attribution remain open in engineering source/comments. |
| G15–G18 | Thermal formula and applicability | The gas-viscosity SI constant differs; SP-125 equation-number comments differ; Bartz validity and chamber underprediction are not enforced; recovery-temperature support remains a separate extension. |
| G19 | Thermal formula | The implemented Sieder–Tate form differs from the held SP-125 RP-1 correlation and produces higher coolant coefficients over the compared Prandtl range. |
| G20–G22 | Locators and sizing context | SP-8087 film-cooling and Nasuti citation locators, plus the repository film-capacity sizing assumption, remain engineering/source-context issues. |
| G23 | Repository choices | Son transition fraction and module-diameter sizing are repository choices. The reviewed source equation equivalence does not validate those choices. |
| G24–G25 | Source-internal inconsistencies | Son printed Eq. (2) omits the post thickness; its tabulated vaporization data disagree. The code follows the internally consistent transition form, so G24 is not a requested code correction. |
| G26–G30 | Source-internal inconsistencies | Casalino efficiency, Kwak energy density, Kang table/axis notation and Hwang delta notation remain recorded inconsistencies in original publications. |
| G31–G34 | Source-internal inconsistencies | Pizzarelli velocity-profile sign/prose, SP-125 viscosity-sample arithmetic, tube-sample equations/results and frustum-volume arithmetic remain recorded source issues. |

## Reproducible method and scope

Startup used `knowledge/kg.sh status` and task context as required by
`knowledge/AGENTS.md`. The synchronized local stores were read through
`kg.store.load`; `kg.harvest_extraction.model_review_state` checked each retained
source/implementation comparison; `kg.graph.model_source_support` applied current
citation freshness. `kg.implementation_freshness.reconcile(write=False)` checked exact
AST locators without writing. `kg.execution_reach.StaticReach(REPO_ROOT).model(...)`
was recomputed and compared with each stored result. Registry-owned fields were
compared with `raosim.model_registry.MODEL_REGISTRY`, and original PDF bytes were
hashed independently of the manifest. The Python runtime was
`PYTHONPATH=knowledge .venv-jax/bin/python`; the expected `.venv/bin/python` is absent.

The provided PDF's relevant text was read from the page-numbered extraction at
`knowledge/.work/consolidated-review-comparison-2026-10-17/review-pages.json`.
The PDF and extraction hashes are retained in the JSON snapshot. PDF page numbers,
not printed publication page numbers, are used for this comparison.

The audit created only these two report files. It did not mutate scientific stores,
rebuild the graph, run engineering suites, stage files, commit or push. Knowledge
integrity checks and bounded follow-on source work are reported by their own results;
they do not establish subsystem completeness.
