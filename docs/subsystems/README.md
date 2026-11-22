# docs/subsystems

One folder per pipeline stage that has a dedicated physics or design-basis document, named `Sxx-<slug>` with the id
and slug of the stage (stage table: [../README.md](../README.md); definitions:
[../architecture/DATAFLOW.md](../architecture/DATAFLOW.md)). Eight of the fourteen stages have such a folder;
S01, S02, S03, S11, S12, S14 have none. The cross-cutting groups X01-X08 have no folders
either; see the end of this page.

Where a document and the code disagree, the code is the authority. None of these documents is a hardware
qualification; no code path sets `hardware_qualified` to true.

## S01 - CLI entry points, argument parsing and workflow dispatch

No dedicated document, so there is no `S01-cli-entry-and-dispatch/` folder, and no document in `docs/` has S01 as its primary stage.
The design intent of the stage's modules is in their header comments: see the S01 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

## S02 - Inputs, units, validation and job contracts

No dedicated design-basis document, so there is no `S02-inputs-and-contracts/` folder. Documents that concern S02 are filed in other folders (listed below).
The design intent of the stage's modules is in their header comments: see the S02 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

Related documents in other folders:

- [audits/LREKIT_WORKFLOW_AUDIT_2026-09-04.md](../audits/LREKIT_WORKFLOW_AUDIT_2026-09-04.md)
- [audits/CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md](../audits/CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md)
- [audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md)
- [literature/LITERATURE_REQUESTS_GENERALIZATION.md](../literature/LITERATURE_REQUESTS_GENERALIZATION.md)
- [architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](../architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md)

## S03 - Propellant identity and chamber thermochemistry

No dedicated design-basis document, so there is no `S03-propellant-thermochemistry/` folder. Documents that concern S03 are filed in other folders (listed below).
The design intent of the stage's modules is in their header comments: see the S03 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

Related documents in other folders:

- [audits/CEA_H1_HOST_EVIDENCE_2026-08-22.md](../audits/CEA_H1_HOST_EVIDENCE_2026-08-22.md)
- [subsystems/S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md](../subsystems/S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md)

## S04 - Throat sizing, ideal/delivered performance and off-design screens

Folder: [S04-performance-and-off-design/](S04-performance-and-off-design/)

- [FROZEN_VARIABLE_CP_EXPANSION.md](S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md) (active, subsystem_basis): Theory and claim boundary of `raosim/frozen_flow.py` (chemically frozen, variable-cp quasi-1-D expansion): equations, property-table provenance contract, unsupported couplings, promotion gates.

## S05 - Default Bezier/TOP nozzle contour, throat and chamber geometry

Folder: [S05-nozzle-and-chamber-geometry/](S05-nozzle-and-chamber-geometry/)

- [shoulder_radius_design_basis.md](S05-nozzle-and-chamber-geometry/shoulder_radius_design_basis.md) (active, subsystem_basis): Separates the throat-approach radius Ru (performance-governed, SP-8120) from the chamber shoulder fillet Rs (SP-125); records the Rs closure and the `--cd-target` Ru closure.

Related documents in other folders:

- [literature/chamber_design_literature.md](../literature/chamber_design_literature.md)
- [data/rao_nozzle.csv](../data/rao_nozzle.csv)

## S06 - Opt-in Rao variational / method-of-characteristics BVP contour

Folder: [S06-rao-moc-bvp-contour/](S06-rao-moc-bvp-contour/)

- [MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md](S06-rao-moc-bvp-contour/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md) (active, remediation_record): Shows that the old `moc.py` forward march was not a bounded characteristic net; Phase 0-4 plan. Its STATUS head says Phases 0-2 landed through `raosim/moc_bounded.py`.
- [REWRITE_PLAN.md](S06-rao-moc-bvp-contour/REWRITE_PLAN.md) (active, plan): Thirteen-phase repair plan for `rao_variational.py` and the MOC contour pipeline. Phase 10 (variable-gamma MOC) and Phase 11 (SU2 CFD cross-check) have not landed: the unit processes of `raosim/moc.py` take a scalar `gamma`, and no SU2 code exists.
- [nasa_tt_prime_provenance.md](S06-rao-moc-bvp-contour/nasa_tt_prime_provenance.md) (active, provenance): Provenance of the NASA/JHU M3.5Perf TT' fixture: the visible source-port workflow, not the fixture, is the authority for `NASA_REFERENCE_MATCHED`. Named by code constants.
- [rao_stationarity_transversality.md](S06-rao-moc-bvp-contour/rao_stationarity_transversality.md) (active, subsystem_basis): Terminology note for the Rao D-existence scans: fixed-endpoint problem, hard closure conditions, optional free-exit diagnostic, scan modes.
- [rice_jhu_moc_topology_map.md](S06-rao-moc-bvp-contour/rice_jhu_moc_topology_map.md) (active, subsystem_basis): Names the characteristic regions of NASA/JHU MOC_Grid_BDE and maps each routine to its Python analogue; lists ports still incomplete.

Related documents in other folders:

- [program/JAX_DIFFERENTIABLE_PLAN.md](../program/JAX_DIFFERENTIABLE_PLAN.md)
- [audits/RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md](../audits/RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md)
- [audits/legacy_code_audit.md](../audits/legacy_code_audit.md)
- [data/nasa_jhu_file_list.txt](../data/nasa_jhu_file_list.txt)

## S07 - Regenerative wall sizing, cooling analysis and wall structural screens

Folder: [S07-regen-cooling-and-wall-structures/](S07-regen-cooling-and-wall-structures/)

- [regen_wall_model.md](S07-regen-cooling-and-wall-structures/regen_wall_model.md) (active, subsystem_basis): Station-wise regenerative wall sizing: literature, implemented equations, input-to-thickness table, CAD export and qualification limits.

Related documents in other folders:

- [literature/thermofluid_literature_provenance.md](../literature/thermofluid_literature_provenance.md)

## S08 - Pintle injector sizing and optional spray/c* fixed point

Folder: [S08-injector-and-spray-coupling/](S08-injector-and-spray-coupling/)

- [injector_pintle_provenance.md](S08-injector-and-spray-coupling/injector_pintle_provenance.md) (active, subsystem_basis): Equation/source table for `raosim/injector.py` with an explicit not-validated list (cold flow and hot fire required).
- [LAGRANGIAN_SPRAY_MODEL.md](S08-injector-and-spray-coupling/LAGRANGIAN_SPRAY_MODEL.md) (active, subsystem_basis): Opt-in, non-reacting Lagrangian droplet model (`raosim/spray/`) and its fail-closed gates; the CLI keeps the correlation screen.
- [MOVABLE_PINTLE_MODEL.md](S08-injector-and-spray-coupling/MOVABLE_PINTLE_MODEL.md) (active, subsystem_basis): Son et al. (2017) continuous movable-pintle model (`raosim/movable_pintle.py`): area law, Cd evidence contract, throttle-map semantics, CLI flags; `hardware_qualified` stays false.
- [OPENFOAM_SPRAY_EXPORT.md](S08-injector-and-spray-coupling/OPENFOAM_SPRAY_EXPORT.md) (active, subsystem_basis): Deterministic OpenFOAM Foundation 13 VOF case export for the movable-pintle sheet stage (`export-openfoam-spray`) and the typed VOF-to-Lagrangian handoff.
- [PINTLE_DESIGN_EVALUATION.md](S08-injector-and-spray-coupling/PINTLE_DESIGN_EVALUATION.md) (active, subsystem_basis): Literature-grounded pintle equations, a line-by-line audit of `raosim/injector.py`, the CLI propagation map and recommendations.
- [pintle_injector_model_gap_analysis.md](S08-injector-and-spray-coupling/pintle_injector_model_gap_analysis.md) (active, subsystem_basis): Strengths and weaknesses of the pintle model against the literature; proposed design flow and implementation order.
- [SPRAY_CSTAR_COUPLING.md](S08-injector-and-spray-coupling/SPRAY_CSTAR_COUPLING.md) (active, subsystem_basis): Opt-in relaxed fixed point (`raosim/spray_coupling.py`) that closes the c* efficiency as vaporization times mixing times combustion efficiency, re-sizing the pintle each pass.

Related documents in other folders:

- [literature/pintle_injector_literature_inventory.md](../literature/pintle_injector_literature_inventory.md)

## S09 - Feed-pressure ledger to electric pump, motor, inverter and battery sizing

Folder: [S09-feed-system-and-electric-pumps/](S09-feed-system-and-electric-pumps/)

- [electric_pump_model_basis.md](S09-feed-system-and-electric-pumps/electric_pump_model_basis.md) (active, subsystem_basis): Basis of the electric-pump sizing screen from feed ledger to battery mass: meanline, assumptions, literature anchors, blade/channel geometry fidelity.
- [PUMP_ADVANCED_GEOMETRY_ROADMAP_2026-07-11.md](S09-feed-system-and-electric-pumps/PUMP_ADVANCED_GEOMETRY_ROADMAP_2026-07-11.md) (superseded, roadmap): Eight-item roadmap for next-tier impeller, inducer, diffuser and volute CAD, with a 2026-07-12 closure update; remaining items corrected by the 2026-07-13 plan.
- [PUMP_CAD_IMPLEMENTATION_PLAN.md](S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md) (active, plan): Plan and STATUS log for pump CAD B-rep parity, Phases 0-3; its STATUS head (2026-07-12) records them landed plus split-casing remediation.
- [PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md](S09-feed-system-and-electric-pumps/PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md) (active, subsystem_basis): Pump meanline relations, process rules and design criteria with a verification mark on each number and a D1-D12 discrepancy ledger against the code.

Related documents in other folders:

- [program/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md](../program/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md)
- [audits/LREKIT_PUMP_AUDIT_2026-09-05.md](../audits/LREKIT_PUMP_AUDIT_2026-09-05.md)
- [literature/LITERATURE_REQUESTS_HARDWARE_LOOP.md](../literature/LITERATURE_REQUESTS_HARDWARE_LOOP.md)

## S10 - Injector/chamber bolted interface and hardware mass ledger

Folder: [S10-interface-and-mass-ledger/](S10-interface-and-mass-ledger/)

- [HARDWARE_MASS_LEDGER.md](S10-interface-and-mass-ledger/HARDWARE_MASS_LEDGER.md) (active, subsystem_basis): Geometry-integrated mass ledger for chamber, interface, injector and pump BOM, with the rule that missing masses are None, never 0.0.
- [injector_interface_model.md](S10-interface-and-mass-ledger/injector_interface_model.md) (active, subsystem_basis): Basis for `raosim/interface.py`: separating load, hoop screens, faceplate bending, bolt separation and bolt-pattern heuristics, with limits.

Related documents in other folders:

- [program/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md](../program/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md)

## S11 - CAD, mesh and geometry deliverables

No dedicated design-basis document, so there is no `S11-cad-and-geometry-export/` folder. Documents that concern S11 are filed in other folders (listed below).
The design intent of the stage's modules is in their header comments: see the S11 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

Related documents in other folders:

- [subsystems/S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md](../subsystems/S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md)
- [subsystems/S07-regen-cooling-and-wall-structures/regen_wall_model.md](../subsystems/S07-regen-cooling-and-wall-structures/regen_wall_model.md)
- [architecture/PHYSICAL_RELEASE_GATES.md](../architecture/PHYSICAL_RELEASE_GATES.md)

## S12 - Figures, MOC construction diagrams and flow animations

No dedicated document, so there is no `S12-plots-and-flow-visualization/` folder, and no document in `docs/` has S12 as its primary stage.
The design intent of the stage's modules is in their header comments: see the S12 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

## S13 - Differentiable whole-engine MDO, optimizer and requirement-driven design

Folder: [S13-engine-mdo-and-requirements/](S13-engine-mdo-and-requirements/)

- [DIFFERENTIABLE_ENGINE_MDO_PLAN.md](S13-engine-mdo-and-requirements/DIFFERENTIABLE_ENGINE_MDO_PLAN.md) (active, plan): Living plan and phase/gate tracker for the differentiable engine MDO (`raosim/mdo/`): formulation, per-discipline physics, phases with status, decision log.
- [MDO_GUIDE.md](S13-engine-mdo-and-requirements/MDO_GUIDE.md) (active, guide): Reviewer's guide to `raosim/mdo/`: module map, design vector, constraint manifest, NLP robustness, run commands and a list of what is still soft.

Related documents in other folders:

- [architecture/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md](../architecture/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md)
- [architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](../architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md)
- [program/COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md](../program/COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md)
- [program/GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md](../program/GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md)
- [program/ROADMAP_REQUIREMENTS_TO_PARTS.md](../program/ROADMAP_REQUIREMENTS_TO_PARTS.md)
- [audits/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md](../audits/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md)
- [audits/LITERATURE_AUDIT_2026-07-24.md](../audits/LITERATURE_AUDIT_2026-07-24.md)
- [audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md)
- [literature/REMEDIATION_LITERATURE_TRACEABILITY.md](../literature/REMEDIATION_LITERATURE_TRACEABILITY.md)

## S14 - Authoritative re-evaluation, provenance, release gates and run records

No dedicated design-basis document, so there is no `S14-reevaluation-provenance-and-run-records/` folder. Documents that concern S14 are filed in other folders (listed below).
The design intent of the stage's modules is in their header comments: see the S14 section of [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).

Related documents in other folders:

- [architecture/MODEL_REGISTRY.md](../architecture/MODEL_REGISTRY.md)
- [architecture/PHYSICAL_RELEASE_GATES.md](../architecture/PHYSICAL_RELEASE_GATES.md)
- [architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](../architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md)
- [audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md)
- [audits/INPUT_EXECUTION_SMOKE_2026-09-09.json](../audits/INPUT_EXECUTION_SMOKE_2026-09-09.json)

## Cross-cutting groups X01-X08

These groups have no folder under `subsystems/`. Their documents are filed as follows.

- X01 Differentiable JAX core (`raosim/jax/`): [../program/JAX_DIFFERENTIABLE_PLAN.md](../program/JAX_DIFFERENTIABLE_PLAN.md).
- X02 Lagrangian spray mid-tier and OpenFOAM export (`raosim/spray/`, `raosim/openfoam/`): filed under S08 as
  [LAGRANGIAN_SPRAY_MODEL.md](S08-injector-and-spray-coupling/LAGRANGIAN_SPRAY_MODEL.md) and
  [OPENFOAM_SPRAY_EXPORT.md](S08-injector-and-spray-coupling/OPENFOAM_SPRAY_EXPORT.md). The CLI never runs the parcel
  solver (`raosim/spray_coupling.py` imports the spray package only for its handoff types); the one X02 entry point the
  CLI exposes is the `export-openfoam-spray` verb.
- X03 Legacy toolbox CLI (`main.py`, `RaoRocketSimLegacy`): no dedicated document; the root README describes it.
- X04 Vendored NASA/JHU MOC_Grid_BDE reference: S06 folder
  ([rice_jhu_moc_topology_map.md](S06-rao-moc-bvp-contour/rice_jhu_moc_topology_map.md),
  [nasa_tt_prime_provenance.md](S06-rao-moc-bvp-contour/nasa_tt_prime_provenance.md)),
  [../data/nasa_jhu_file_list.txt](../data/nasa_jhu_file_list.txt) and
  [../audits/legacy_code_audit.md](../audits/legacy_code_audit.md).
- X05 Standalone research scripts (`scripts/`): no dedicated document; header comments in
  [../architecture/CODE_HEADER_COMMENTS.md](../architecture/CODE_HEADER_COMMENTS.md).
- X06 Knowledge graph and source corpus: [../../knowledge/README.md](../../knowledge/README.md).
- X07 Test harness: [../agents/TEST_PROMPT.md](../agents/TEST_PROMPT.md) (standard 13 kN regression check).
- X08 Library code not reachable from the CLI: no dedicated document; listed in
  [../architecture/CODE_MAP.md](../architecture/CODE_MAP.md).

## Filing notes

- [FROZEN_VARIABLE_CP_EXPANSION.md](S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md) describes
  `raosim/frozen_flow.py`, which `CODE_MAP.md` assigns to S03; the triage filed the document under S04
  (performance). Use `CODE_MAP.md` for module ownership.
- `SPRAY_CSTAR_COUPLING.md`, `pintle_injector_model_gap_analysis.md` and
  `PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md` have no primary stage in their triage records; their folder was
  chosen in the triage.
- `JAX_DIFFERENTIABLE_PLAN.md` was triaged to S06 but filed in [../program/](../program/README.md) as the project's
  STATUS-head record.
