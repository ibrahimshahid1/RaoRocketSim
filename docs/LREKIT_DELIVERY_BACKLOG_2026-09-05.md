# LREKit delivery backlog

Companion to [the readiness review and delivery plan](LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md). These are **local implementation issues**, not GitHub issues created. September 9 working-tree changes address D01/D02/D26 and portions of D03/D04/D24; [the implementation record](INPUT_EXECUTION_INTEGRITY_2026-09-09.md) states exact coverage and remaining acceptance work. Other items remain open. The named owner is a discipline, not a staffing assignment. Existing A–G, P0–P11 and H–M IDs are cross-references; D IDs are the common delivery index.

## Standard issue contract

Use this structure for every implementation issue and expose its outcomes through the same job/run record:

```yaml
id: Dxx
title: Concrete problem and resulting behavior
owner: application | geometry | thermal | mechanical | manufacturing | validation
status: open
priority: first | next | later
existing_tracks: []
depends_on: []
scope:
  architectures: []
  components: []
  intended_use: exploratory | component_article | engine_article
inputs_and_units: []
outputs_and_consumers: []
sources:
  - source_id: required_if_physics_or_process_based
    section_and_pdf_page: required_before_equation_implementation
    verification: metadata_only | original_checked
    applicability: explicit
acceptance:
  positive_cases: []
  negative_and_unknown_cases: []
  evidence_artifacts: []
remaining_unknowns: []
```

Do not close an issue just because a function exists. Demonstrate that the real CLI/API execution path consumes its inputs and emits the promised result. For new physical relations, separately record source transcription verification, numerical verification and experimental validation. Code review/CI cannot generate a physical test approval.

## First milestone: trustworthy requests and recoverable runs

### D01 — Consume or reject every accepted request field

**Owner:** application. **Priority:** first. **Cross-reference:** W01/W03, M1. **Dependencies:** none.

Compile altitude and explicit ambient conditions, single-point operating data, thrust/duration, mission-level limits and architecture selections to actual model inputs/constraints. Reject unknown schema versions; reject unsupported architecture choices with the missing capability. Resolve conflicts explicitly instead of silently overriding an operating point. Preserve supplied and resolved values and default provenance.

**Acceptance:** regress the 10 km and single-vacuum-point cases; equivalent explicit pressure yields the same resolved inputs. Unsupported pressure-fed/coaxial/ablative requests cannot run the unrelated default model under those labels. Every accepted schema field is consumed by a documented stage or produces an explicit unsupported result. No source literature is needed for the plumbing repair; any atmosphere conversion must use an identified model and declared range.

### D02 — Carry actual variable policies into optimization

**Owner:** optimization/application. **Priority:** first. **Cross-reference:** W02, S0. **Dependencies:** D01.

Build one design layout that substitutes fixed variables, intersects user bounds with applicable domains, handles discrete enumeration outside the continuous solve and maps solved variables back into the configuration. Route supported target constraints to the optimizer; explicitly distinguish postsolve-only checks. Do not merely preserve policies in JSON.

**Acceptance:** use a deliberately binding pin and narrow bound that the default optimizer would violate; verify the real solver receives and honors them. Empty bound intersections and unsupported enumeration fail before solving. Re-evaluation and CAD preserve the same pin. Verify feasibility/unknown outcomes for conflicting constraints.

### D03 — Make output policy execute the requested stages

**Owner:** application/CAD. **Priority:** first. **Cross-reference:** W03/W06, M. **Dependencies:** D01.

Wire requested pumps, parts, formats, reports and attainable fidelity to actual export services. Use one execution service for jobs, legacy CLI adapters and API. Keep old public verbs working. Unsupported formats/fidelity must be rejected or explicitly refused; output directory and partial-package policy must be operational.

**Acceptance:** a saved job requests component CAD and produces a manifest containing the requested verified files. CAD-disabled jobs do not run CAD. The full-design interview collects design values or shows explicit reference defaults. Unavailable maximize objectives are not offered as runnable choices. A capability matrix maps every exposed feature to a callable route.

### D04 — Persist stage outcomes and publish honest partial packages

**Owner:** application. **Priority:** first. **Cross-reference:** W04/W05, L2. **Dependencies:** D03 for export integration; failure-record shell can start independently.

Create a run record before expensive work. Model stage dependencies; persist expected/actual artifacts, scientific verdict and package status atomically. Stage files in run-specific directories; distinguish successful components, failed components and blocked dependents. Consume typed exporter validation rather than inferring CAD fidelity from file existence. Resolve the 130 kN bolt-sector feasibility omission before entering B-rep construction.

**Acceptance:** inject the 500 N sizing failure and the late oxidizer export failure. Earlier independent results survive with a summary; diagnostic partial directories contain an immediately visible status file and complete inventory. A process interruption is recoverable; stale files from another run are excluded. `fake.step` cannot earn L2. Missing required parts prevent complete-assembly readiness without suppressing individual artifact fidelity.

### D26 — Repair roughness units at the MDO/traditional boundary

**Owner:** thermal/application. **Priority:** first. **Cross-reference:** D6, I process bridge. **Dependencies:** none; integrate with D05.

`MissionSpec.channel_roughness` is labelled relative in `mdo/schema.py:415` and passed directly to `darcy_friction_factor(Re, rel_roughness)` in `mdo/cooling.py:413`. `resolved_inputs.py:519` and `postprocess.py:301` transfer it unchanged to `CoolingSpec`, documented in metres (`design.py:185`); `physics.py:1006` then divides it by hydraulic diameter. The zero default hides the inconsistency.

Choose one explicit canonical representation, preferably a named absolute hydraulic roughness length plus stationwise relative roughness derived from the actual hydraulic diameter. Distinguish surface-metrology Ra from the effective roughness used in the selected hydraulic correlation; any conversion needs its own basis. Version/migrate old inputs so their meaning is not silently changed.

**Acceptance:** compare nonzero-roughness cases at more than one hydraulic diameter through both real analysis paths. Match friction and pressure-drop inputs under the same correlation/regime; preserve differences in other thermal models as separate discrepancies. The unit regression must fail if either the divide-by-Dh is omitted or applied twice.

A bounded code probe at Re=100000, Dh=0.001 m and unchanged scalar 0.00001 produced friction factors 0.017924 (MDO interpretation) and 0.038751 (traditional interpretation) using the same MDO correlation. This is a dimensional-consistency demonstration, not a validated physical test point.

## Configuration and faithful geometry

### D05 — Compose immutable geometry, configuration and artifact identities

**Owner:** application/data. **Priority:** first/parallel. **Cross-reference:** integration §2.1, K0, S2. **Dependencies:** none for schema; adapters follow D01.

Preserve existing local hashes, introduce exact canonical definitions and explicit parent links, and make solved component state the input to every builder/analysis. Include schema and model/property/coefficient identities. Record code revision and dirty-patch identity for reproducibility. Separate semantic geometry identity from neutral-file byte integrity and from tolerance-equivalence reports. Deep-freeze nested data and reject unsupported/non-finite serialization.

**Acceptance:** changing a physical dimension, component selection or model version changes the corresponding identity; changing an output path does not change physical identity. Mixed-configuration parts fail assembly. Geometry comparison cannot collapse two article serials into one. Identical canonical inputs replay deterministically under the declared generation environment.

### D06 — Generate the applicable part graph and full accounting boundary

**Owner:** systems/application. **Priority:** next. **Cross-reference:** G, W04. **Dependencies:** D05.

Generate expected parts from architecture and intended article scope. Distinguish manufactured, purchased, reference-envelope and test-fixture items; include quantity, material, frame, interfaces, evidence and exclusions. Expand pump packages into their actual component records. Sum mass only from disjoint measured or explicitly labelled surrogate contributions; keep whole-engine mass unavailable when essential terms remain missing.

**Acceptance:** component cold-flow and engine-article jobs have different correct part graphs; no irrelevant igniter/gimbal requirement blocks a pump bench. Missing applicable parts are visible. Existing purchased components need selection and interface data, not invented internal CAD. One body represented in both a package and a component is counted once.

### D07 — Use one wall/flow-path definition in analysis, CAD and mass

**Owner:** geometry/thermal. **Priority:** next. **Cross-reference:** G2/G3, D6. **Dependencies:** D05.

Route the final `RegenWallProfile`, actual coolant voids, lands, liner, jacket, manifold/end-closure/joint regions and material assignment to all consumers. Replace the main path's uniform-wall export where the analyzed article is regenerative. Preserve separate provisional outputs when a complete builder is unavailable.

**Acceptance:** exported section dimensions, hydraulic areas, wall thickness, connectivity and disjoint material volumes agree with the resolved analysis inputs and mass ledger. Test a nonuniform wall and a helical case; agreement on a uniform smooth fixture alone is insufficient. Joint/manifold allowances are explicit, not silently inserted into mass.

### D08 — Resolve around a selected component without changing it

**Owner:** geometry/optimization. **Priority:** next. **Cross-reference:** K4, S4. **Dependencies:** D02/D05.

Consume `editable_reference_id` through typed component/feature references. Imported CAD needs units, frame, topology/feature mapping and validated dimensions; retain opaque imported geometry where inference is unavailable. State which degrees of freedom remain free and how the pinned component's performance is modeled.

**Acceptance:** pin an impeller or chamber feature and verify the geometric identity survives optimization/re-evaluation/export. Contradictory performance targets produce infeasible or unknown results with reasons; the pin is never overwritten. A file with unknown units cannot silently enter the assembly.

### D09 — Finish P1 and implement 3-D camber mathematics

**Owner:** pump hydraulics/geometry. **Priority:** next; pure mathematics can begin now. **Cross-reference:** P1/P2, H1. **Dependencies:** D05 interface for integration.

Establish impeller hub/mid/shroud inlet triangles, blockage and pitch checks, angle/sign conventions and common meridional parameterization. Integrate separate camber streamlines with declared stacking. Treat the inducer's hub/tip angles as inducer data. Preserve a named 2-D limit for regression rather than forcing a 3-D design to reproduce it everywhere.

**Acceptance:** recover the endpoint metal-angle convention by differentiation, monotonic wrap under valid assumptions, zero TE clocking differences for TE stacking, expected upstream wrap differences and continuity across span. Reject singular radius/angle/channel domains with reasons. Verify equations from the original source before coding numerical design coefficients.

### D10 — Build 3-D blade solids, edges and splitters with section audits

**Owner:** CAD/pump hydraulics. **Priority:** next. **Cross-reference:** P3–P5, H2/H4. **Dependencies:** D09.

Choose a robust planar-section loft or sewn-surface construction; parameterize thickness, leading/trailing edges and splitter start. Maintain pressure/suction orientation and local span coordinates. Recompute passage blockage and effective throats from the result.

**Acceptance:** one valid solid per blade/body after STEP re-import; no negative local thickness, inverted sections, unintended self-intersections or passage closures. Extract angles/areas/thickness from solid sections, not copied metadata. Check scale-sensitive cases and a deliberately invalid geometry. Tolerances have an explicit numerical/process basis.

### D11 — Replace diffuser and volute placeholders with resolved passages

**Owner:** pump hydraulics/CAD. **Priority:** next. **Cross-reference:** P6/P7, H5. **Dependencies:** D09/D10 interfaces.

Define actual diffuser vane profiles, inlet incidence, throat and downstream recovery geometry. Choose a source-scoped volute collection law; separate local collection area, collector radius, tongue and discharge transition from the external port. Reconcile pressure recovery, off-design incidence and casing loads with the hydraulic model. Evaluate blade/vane interactions only with verified equations and declared scope.

**Acceptance:** extract minimum throat and area progression from the solid and compare to design; conserve collected flow under the chosen law. Audit recirculation/dead ends and interfaces. A terminal port area cannot stand in for an unmodeled exit transition. Load estimates identify model uncertainty before bearing/shaft use.

### D12 — Complete applicable sweep, lean, shroud and finishing geometry

**Owner:** pump CAD/mechanics. **Priority:** next. **Cross-reference:** P8–P11, H6. **Dependencies:** D10/D11 and mechanical/process interfaces.

Use local coordinates for sweep/lean; include shroud and wear-ring interfaces where selected. Reserve root/edge radii early and perform fragile final fillets after topology stabilizes. Re-run hydraulic section and structural-clearance checks after these additions. Historical radii are scoped design guidance, not universal process release limits.

**Acceptance:** final part, including fillets, satisfies section/clearance checks and STEP round trip at declared representative scales. Failures produce a labelled diagnostic geometry, not an unfilleted part disguised as the final design. Selected shroud/process route remains assemblable and inspectable.

## Mechanical, manufacturing and test-article definition

### D13 — Close rotating load paths and select real interfaces

**Owner:** mechanical/rotordynamics. **Priority:** next/parallel. **Cross-reference:** J. **Dependencies:** D05 and evolving D09–D12 load/geometry interfaces.

Define shaft/coupling/retention, radial and axial hydraulic loads, balance arrangement, bearings, seals, wear rings, lubrication/cooling and leakage paths. Select applicable material and purchased-component data. Evaluate combined/thermal/fatigue loads, clearance changes and rotor critical/dynamic response over the operating schedule. Historical speed-based rules cannot alone select a balance device or a bearing.

**Acceptance:** every torque/force reaction reaches a support; residual axial load and bearing/seal assumptions are explicit. Real temperature/speed/load data support the selected elements. Mechanical margins and required external analysis are linked to the configuration. A placeholder shaft or bearing mass cannot satisfy the assembly gate.

### D14 — Implement a process-capability contract per feature

**Owner:** manufacturing/CAM/geometry. **Priority:** next/parallel. **Cross-reference:** I, G5. **Dependencies:** D05; final reach checks depend on D10–D12.

Represent stock/material/condition, process route, machines/axis travel, tool and holder geometry, fixture/access directions, feature minimum radii/walls, tool deflection/contact error, setups, surface specification and attainable tolerance. Separate analytic screens, CAM simulation and reviewed shop evidence. Distinguish machining, casting, additive and joining capabilities; route pressure/thermal models to the resulting material/roughness/joint data.

**Acceptance:** a deliberately inaccessible passage fails despite valid CAD; a ruled surface does not automatically pass flank machining. Tool/holder/fixture clearance is checked on final geometry. A standards table is never substituted for measured shop capability. Process limits and physics validity have different reason codes.

### D15 — Produce drawings, assembly and inspection definitions

**Owner:** manufacturing/metrology/mechanical. **Priority:** next. **Cross-reference:** I3/G5. **Dependencies:** D06/D13/D14.

Attach material/stock/heat-treatment/joining specifications, selected fasteners/seals, datum frames, functional tolerances and surface finish, assembly sequence/access and inspection features. Match manufacturing and assembly CAD to the same configuration. Include tolerance-stack inputs/results and inspection methods for hidden passages.

**Acceptance:** a reviewer can trace every critical flow area, fit, sealing land and load interface from requirement to dimension and measurement method. Inconsistent revisions and missing dimensional controls block that part's manufacturing status. Drawings and neutral CAD reference one immutable package.

### D16 — Design instrumentation and article-specific test readiness

**Owner:** test/mechanical/metrology. **Priority:** next/parallel. **Cross-reference:** J/K1, external release requirements. **Dependencies:** D05/D13; final package D15.

Define the intended component or engine test and the quantities needed to discriminate the models. Add instrument/pressure/temperature accommodations and fixtures to CAD before freezing fabrication. Record calibration, station conventions, planned operating/load envelope and article-specific external evidence. Include applicable material compatibility, cleanliness, joining, inspection and facility interfaces.

**Acceptance:** every proposed calibration quantity is observable or explicitly unavailable; sensor placement and response are represented in the observation model. Instrument provisions receive the same structural/process checks as other features. Readiness is based on the applicable article and reviewed evidence, never on a global CAD-success boolean. This issue does not supply universal proof, spin or firing procedures.

## Experimental feedback

### D17 — Ingest immutable observations bound to the as-built article

**Owner:** data/test. **Priority:** next/parallel. **Cross-reference:** K1/S3. **Dependencies:** D05; schema can start now.

Implement the measurement/article schema from the readiness plan; preserve original files and processing provenance. Support units, station/frame, acquisition timebase, absolute/gauge/total conventions, instrument certificates, conditions, uncertainty type/coverage/correlation and quality flags. Import as-built metrology and preserve deviations from nominal.

**Acceptance:** wrong units, stations or article identity are rejected; missing uncertainty remains explicit. A raw dataset is unchanged after reprocessing. As-built geometry changes produce a new linked analysis configuration without rewriting nominal design or the original test record.

### D18 — Compare predicted sensor observables to measurements

**Owner:** validation/physics/data. **Priority:** next. **Cross-reference:** K2/S3. **Dependencies:** D17.

Re-solve at measured conditions using the as-built configuration. Match time windows and stations; account for sensor observation/reduction models and correlations. Emit signed residuals, absolute/relative errors where defined, uncertainty and unresolved model-form effects. Use existing snapshot machinery for presentation, without labelling model parity as experiment.

**Acceptance:** synthetic known bias/time offset/unit conversions recover the expected residuals; a wrong observation location fails. Near-zero relative errors are unavailable rather than unbounded. A test-fluid similarity transformation is applied only under an explicit, supported similarity model, with unmet conditions reported.

### D19 — Fit, validate and promote declared coefficient sets

**Owner:** validation/modeling. **Priority:** next. **Cross-reference:** K3/S5. **Dependencies:** D18.

Generalize the existing source-bound Cd contract into a reusable coefficient set. Implement constrained fitting, uncertainty, identifiability, training/held-out partition, model/source applicability and immutable promotion/rollback. Separate article geometry changes from coefficient changes. Do not make automatic trim instructions the first deliverable.

**Acceptance:** recover an identifiable synthetic parameter and reject an underdetermined fit. A real component dataset supports repeatable fit/reduction, held-out assessment and provenance. Unimproved/overfit candidates cannot be labelled validated. Historical baseline predictions can be reproduced after promotion.

## Whole-engine closure and common access

### D20 — Qualify the resolved chamber/nozzle thermal and structural model

**Owner:** thermal/structures. **Priority:** later, after article scope chosen. **Cross-reference:** D1–D6/F1–F6/G2–G3 of the older completion plan. **Dependencies:** D07/D15/D18.

Reconcile thermal correlations and stations, add appropriate conjugate/hot-spot and transient load evaluations, and bind material/allowable/life evidence to manufactured state and joining route. Scope variable-property, viscous and separation models to the intended operating cases. Keep absent LCF/side-load evidence unavailable.

**Acceptance:** numerical refinement and cross-model discrepancies are distinguished from measured-data residuals. The analyzed wall is the built wall. Start/shutdown/operating load cases and thermal growth are represented where required; steady stress alone cannot satisfy a cyclic-life requirement.

### D21 — Close injector, film, ignition and stability coverage

**Owner:** combustion/injector/test. **Priority:** later. **Cross-reference:** G1, integration §2.4. **Dependencies:** D05/D15/D18.

Retain the five-part injector path; add the actual selected movable/film/igniter hardware and its hydraulics, actuation and thermal interfaces. Separate supplied combustion efficiency from prediction. Treat static chug/acoustic screens as limited screens; combustion-response and dynamic stability evidence need their own models/data.

**Acceptance:** modeled film flow has applicable physical injection hardware and accounted mass; an igniter port is not labelled a complete igniter. As-built metering geometry/Cd and measured conditions support prediction. Missing stability/ignition evidence remains a visible engine-article blocker.

### D22 — Resolve feed, electrical, mounts and facility interfaces

**Owner:** systems/feed/electrical/mechanical. **Priority:** later/selected components parallel. **Cross-reference:** valves/lines/brackets/gimbal plan. **Dependencies:** D06/D13/D15.

Select valves/fittings/electrical hardware, route lines, close pressure losses and heat/power/energy budgets, and design mounts/supports. Include wiring/control/instrument interfaces and the selected test fixture. Model gimbal/flexible lines only when required by scope. Keep subsystem and engine mass accounting boundaries explicit.

**Acceptance:** routed geometry matches the loss network and the assembly BOM; supplied torque-speed/electrical/thermal data support the operating schedule. Mount loads have a defined reaction path. A motor/inverter/battery envelope is never labelled vendor-internal manufacturing CAD.

### D23 — Evaluate one hardware configuration across operating conditions

**Owner:** systems/optimization/test. **Priority:** later. **Cross-reference:** C1–C4, L. **Dependencies:** D02/D13/D20–D22 as relevant.

Implement shared geometry with per-point controls, fluid states and pressure/temperature boundaries. Evaluate head/flow/efficiency/cavitation and separation/property domains per point. Integrate energy over the schedule. Include transient analyses through appropriate separate solvers/evidence; steady points cannot automatically qualify transients.

**Acceptance:** all points carry the same hardware identity; controls change without hidden resizing. Unsupported points are individually explained and prevent an unqualified full-envelope pass. Empirical sweep coverage and computed constraints are separate; solver nonconvergence is not a physical impossibility claim.

### D24 — Expose the full supported suite through one CLI/API/UI contract

**Owner:** application/UI. **Priority:** next once M0 stabilizes; finish with integrated features. **Cross-reference:** M/S6. **Dependencies:** D01–D05.

Build capability-driven guided input, saved jobs, previews, component pinning, stage progress, CAD/artifact inspection and test/calibration history. Thin terminal and graphical clients share the same service. Retain legacy compatibility, explicit research modes, units/default provenance and cancellation/resume behavior.

**Acceptance:** one saved job resolves identically through each client; no frontend independently recomputes physics. Every offered feature has an executed route and a supported output or explicit refusal. Failed components remain inspectable. UI does not present requested fidelity as achieved evidence.

### D25 — Maintain one source/evidence/domain release matrix

**Owner:** validation/documentation. **Priority:** first/continuous. **Cross-reference:** L/M and all plans. **Dependencies:** none.

Reconcile loose PDFs, index/ZIP entries, duplicates, extraction warnings and requests. Correct report identities; distinguish acquired source, reviewed equation, implemented model, verified software and validated domain. Link each D issue and capability to its evidence. Pin dependency versions for reproducible geometry/numerics where relevant.

**Acceptance:** missing/unindexed sources are not repeatedly requested as absent; metadata-only references cannot certify equations. Every published supported case has a reproducible configuration and validation scope. The release matrix identifies incomplete work and replaces contradictory completion claims across plans.

## Integration and first article exit gate

Start with D01–D05/D26, D09 pure mathematics, D17 schema and D25 literature in parallel. Begin process selection and mechanical interface definition early; run detailed checks on the final solid later. Keep the article scope bounded until one full evidence loop closes.

A component article is ready for its **specified** test only when its applicable BOM is complete, analyzed and manufactured geometry is configuration-bound, the process/inspection/assembly definition exists, mechanical and instrumentation dependencies are resolved, and the required external evidence has been reviewed. A whole-engine article additionally needs the corresponding combustion, feed, electrical, thermal/transient and mount evidence. Neither a code test count nor an exporter file count can replace those gates.
