# LREKit / RaoRocketSim documentation

This folder holds the project's prose documentation. Since the cleanup of 2026-11-05 it is organized by the data flow
of the CLI. Each pipeline stage S01-S14 (defined in [architecture/DATAFLOW.md](architecture/DATAFLOW.md)) has a folder
under [subsystems/](subsystems/README.md) once it has a dedicated design-basis document; documents that span stages are
filed by kind. Every document kept its file name when it moved, so an old flat path `docs/<NAME>` is found by name in
one of the folders below. The complete old-to-new list is Appendix A of the repository guide
(`latex-report/RaoRocketSim_Repository_Guide/chapters/app_a_docs_inventory.tex`).

Nothing here is a hardware qualification record. Software verification, geometry verification and physical release
evidence are separate claims ([architecture/PHYSICAL_RELEASE_GATES.md](architecture/PHYSICAL_RELEASE_GATES.md)), and
no code path sets `hardware_qualified` to true (`raosim/run_nozzle.py` writes it as false). Where a document and the code disagree, the code is the authority;
known disagreements are tabulated in section 10 of [architecture/DATAFLOW.md](architecture/DATAFLOW.md).

## Start here

1. [../README.md](../README.md): installation, CLI usage, model summaries and known limitations.
2. [architecture/DATAFLOW.md](architecture/DATAFLOW.md): the CLI traced from the console script to the last file it
   writes; defines the stage ids used everywhere in `docs/`.
3. [program/REMAINING_WORK_AND_GOALS.md](program/REMAINING_WORK_AND_GOALS.md): what remains to be built and why.
4. [program/JAX_DIFFERENTIABLE_PLAN.md](program/JAX_DIFFERENTIABLE_PLAN.md): read the STATUS blockquotes at its head,
   the project's running status record. The newest is dated 2026-06-15 and covers the Rao/JAX contour and
   physics-coupled design work; later status sits at the head of dated documents such as
   [subsystems/S06-rao-moc-bvp-contour/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md](subsystems/S06-rao-moc-bvp-contour/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md)
   and [audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md](audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md).
5. [../knowledge/README.md](../knowledge/README.md): the local scientific memory (`knowledge/kg.sh`) that links source
   PDFs, equations, the model registry, code and tests.

Then use [architecture/CODE_MAP.md](architecture/CODE_MAP.md) (one row per code file) and the folder indexes below.

## Folder scheme

| Folder | Holds | Index |
|---|---|---|
| `architecture/` | Cross-stage contracts and maps: data flow, code map, header comments, model registry, release gates, MDO output contract, packaging | [architecture/README.md](architecture/README.md) |
| `subsystems/Sxx-<slug>/` | Physics and design basis of one pipeline stage (eight stages have a folder) | [subsystems/README.md](subsystems/README.md) |
| `program/` | Plans, roadmaps and backlogs spanning stages; the STATUS-head project record | [program/README.md](program/README.md) |
| `audits/` | Dated audits, remediation records, host-run evidence and the JSON evidence they cite | [audits/README.md](audits/README.md) |
| `literature/` | Literature request ledgers, corpus inventories, equation-to-source provenance | [literature/README.md](literature/README.md) |
| `data/` | Small reference data kept for reading (not loaded by code) | [data/README.md](data/README.md) |
| `agents/` | Instructions and prompts for coding agents | [agents/README.md](agents/README.md) |
| `archive/` | Historical, empty or superseded items that nothing cites as authority | [archive/README.md](archive/README.md) |

## Pipeline stages: code, documents and tests

Code modules per stage are those of [architecture/CODE_MAP.md](architecture/CODE_MAP.md). `CODE_MAP.md` files every
test module under group X07; the test column below assigns tests to stages by the modules each test imports and lists
the most specific suites only (all tests are in `tests/`). Many tests import modules from several stages.

| Stage | Name | Code modules | Documents | Main tests |
|---|---|---|---|---|
| S01 | CLI entry points, argument parsing and workflow dispatch | `lrekit/__init__.py`, `lrekit/__main__.py`, `lrekit/cli.py`, `raosim/__init__.py`, `raosim/__main__.py`, `raosim/cli.py`, `raosim/run_nozzle.py`, `scripts/run_nozzle.py` | no folder | `test_run_nozzle.py`, `test_cli_module_exit.py` |
| S02 | Inputs, units, validation and job contracts | `lrekit/guided.py`, `lrekit/job_cli.py`, `raosim/input_validation.py`, `raosim/manufacturing_limits.py`, `raosim/mdo/capabilities.py`, `raosim/mdo/job.py`, `raosim/mdo/metrics.py`, `raosim/units.py` | no folder; also [LREKIT_WORKFLOW_AUDIT_2026-09-04.md](audits/LREKIT_WORKFLOW_AUDIT_2026-09-04.md), [CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md](audits/CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md), [INPUT_EXECUTION_INTEGRITY_2026-09-09.md](audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md), [LITERATURE_REQUESTS_GENERALIZATION.md](literature/LITERATURE_REQUESTS_GENERALIZATION.md), [MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md) | `test_input_validation.py`, `test_units.py`, `test_lrekit_guided.py`, `test_lrekit_job_cli.py`, `test_mdo_job.py`, `test_mdo_metrics.py`, `test_fidelity_scope_gates.py` |
| S03 | Propellant identity and chamber thermochemistry | `raosim/cea.py`, `raosim/frozen_flow.py`, `raosim/propellants.py` | no folder; also [CEA_H1_HOST_EVIDENCE_2026-08-22.md](audits/CEA_H1_HOST_EVIDENCE_2026-08-22.md), [FROZEN_VARIABLE_CP_EXPANSION.md](subsystems/S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md) | `test_cea_pinned.py`, `test_cea_catalog.py`, `test_frozen_flow.py`, `test_cea_surface_validation.py` |
| S04 | Throat sizing, ideal/delivered performance and off-design screens | `raosim/atmosphere.py`, `raosim/engine.py`, `raosim/gas_dynamics.py`, `raosim/separation.py`, `raosim/wall_pressure.py` | [subsystems/S04-performance-and-off-design/](subsystems/S04-performance-and-off-design/) | `test_engine.py`, `test_gas_dynamics.py`, `test_v2_features.py` |
| S05 | Default Bezier/TOP nozzle contour, throat and chamber geometry | `raosim/chamber_geometry.py`, `raosim/nozzle_geometry.py`, `raosim/rao_top_chart.py`, `raosim/throat_geometry.py`, `raosim/validation.py` | [subsystems/S05-nozzle-and-chamber-geometry/](subsystems/S05-nozzle-and-chamber-geometry/); also [chamber_design_literature.md](literature/chamber_design_literature.md), [rao_nozzle.csv](data/rao_nozzle.csv) | `test_nozzle_geometry.py`, `test_chamber_geometry.py`, `test_design_workflow.py`, `test_rao_chart_benchmark.py` |
| S06 | Opt-in Rao variational / method-of-characteristics BVP contour | `raosim/moc.py`, `raosim/moc_bounded.py`, `raosim/moc_topology.py`, `raosim/nasa_moc.py`, `raosim/rao_existence_scan.py`, `raosim/rao_optimizer.py`, `raosim/rao_residuals.py`, `raosim/rao_variational.py`, `raosim/transonic_kernel.py`, `raosim/wall_model.py` | [subsystems/S06-rao-moc-bvp-contour/](subsystems/S06-rao-moc-bvp-contour/); also [JAX_DIFFERENTIABLE_PLAN.md](program/JAX_DIFFERENTIABLE_PLAN.md), [RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md](audits/RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md), [legacy_code_audit.md](audits/legacy_code_audit.md), [nasa_jhu_file_list.txt](data/nasa_jhu_file_list.txt) | `test_moc.py`, `test_moc_bounded_net.py`, `test_moc_topology.py`, `test_characteristic_pairing.py`, `test_transonic_kernel.py`, `test_phase6_coupled_wall.py`, `test_nasa_kernel.py`, `test_nasa_kernel_march_parity.py`, `test_nasa_moc_topology.py`, `test_nasa_port.py`, `test_rao_variational.py`, `test_rao_variational_moc.py`, `test_rao_stationarity.py`, `test_rao_residuals.py`, `test_rao_valid_region.py`, `test_rao_existence_scan.py`, `test_rao_convergence_domain.py`, `test_rao_resolution_convergence.py`, `test_rao_optimizer.py` |
| S07 | Regenerative wall sizing, cooling analysis and wall structural screens | `raosim/coolants.py`, `raosim/materials.py`, `raosim/physics.py`, `raosim/regen_profile.py`, `raosim/thermal_design.py`, `raosim/thermofluids.py` | [subsystems/S07-regen-cooling-and-wall-structures/](subsystems/S07-regen-cooling-and-wall-structures/); also [thermofluid_literature_provenance.md](literature/thermofluid_literature_provenance.md) | `test_regen_cooling.py`, `test_regen_profile.py`, `test_bartz_heat_flux.py`, `test_advanced_thermofluids.py`, `test_thermal_design.py`, `test_joint_design.py`, `test_materials.py`, `test_fatigue.py`, `test_wall_physics.py`, `test_wall_profile.py` |
| S08 | Pintle injector sizing and optional spray/c* fixed point | `raosim/injector.py`, `raosim/movable_pintle.py`, `raosim/spray_coupling.py` | [subsystems/S08-injector-and-spray-coupling/](subsystems/S08-injector-and-spray-coupling/); also [pintle_injector_literature_inventory.md](literature/pintle_injector_literature_inventory.md) | `test_injector.py`, `test_movable_pintle.py`, `test_spray_coupling.py` |
| S09 | Feed-pressure ledger to electric pump, motor, inverter and battery sizing | `raosim/electric_technology.py`, `raosim/pumps.py` | [subsystems/S09-feed-system-and-electric-pumps/](subsystems/S09-feed-system-and-electric-pumps/); also [PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md](program/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md), [LREKIT_PUMP_AUDIT_2026-09-05.md](audits/LREKIT_PUMP_AUDIT_2026-09-05.md), [LITERATURE_REQUESTS_HARDWARE_LOOP.md](literature/LITERATURE_REQUESTS_HARDWARE_LOOP.md) | `test_electric_technology.py`, `test_pumps.py` |
| S10 | Injector/chamber bolted interface and hardware mass ledger | `raosim/interface.py`, `raosim/mass_ledger.py`, `raosim/regen_volumes.py` | [subsystems/S10-interface-and-mass-ledger/](subsystems/S10-interface-and-mass-ledger/); also [PLAN_VALVES_LINES_BRACKETS_GIMBAL.md](program/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md) | `test_interface.py`, `test_mass_ledger.py`, `test_regen_volumes.py` |
| S11 | CAD, mesh and geometry deliverables | `raosim/engine_cad.py`, `raosim/export.py`, `raosim/injector_cad.py`, `raosim/injector_coaxial_cad.py`, `raosim/injector_export.py`, `raosim/pump_cad.py`, `raosim/pump_cad_brep.py`, `raosim/regen_cad.py`, `raosim/regen_geometry.py` | no folder; also [PUMP_CAD_IMPLEMENTATION_PLAN.md](subsystems/S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md), [regen_wall_model.md](subsystems/S07-regen-cooling-and-wall-structures/regen_wall_model.md), [PHYSICAL_RELEASE_GATES.md](architecture/PHYSICAL_RELEASE_GATES.md) | `test_regen_cad.py`, `test_regen_geometry.py`, `test_injector_export.py`, `test_pump_cad_brep.py`, `test_pumps.py` |
| S12 | Figures, MOC construction diagrams and flow animations | `raosim/flow_viz.py`, `raosim/injector_plots.py`, `raosim/moc_diagrams.py`, `raosim/plotting.py` | no folder | `test_plotting.py`, `test_flow_viz.py`, `test_moc_diagrams.py` |
| S13 | Differentiable whole-engine MDO, optimizer and requirement-driven design | 24 modules in `raosim/mdo/`: `__init__`, `assembly`, `bounds`, `constraints`, `convergence`, `coolant_htd`, `cooling`, `engine`, `envelope`, `grid`, `injector`, `mass`, `material_map`, `nlp`, `nozzle`, `objectives`, `pintle`, `propellants`, `properties`, `pump`, `scaling`, `schema`, `state`, `structures`, `raosim/requirements.py` | [subsystems/S13-engine-mdo-and-requirements/](subsystems/S13-engine-mdo-and-requirements/); also [IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md](architecture/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md), [MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md), [COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md](program/COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md), [GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md](program/GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md), [ROADMAP_REQUIREMENTS_TO_PARTS.md](program/ROADMAP_REQUIREMENTS_TO_PARTS.md), [DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md](audits/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md), [LITERATURE_AUDIT_2026-07-24.md](audits/LITERATURE_AUDIT_2026-07-24.md), [MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md](audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md), [REMEDIATION_LITERATURE_TRACEABILITY.md](literature/REMEDIATION_LITERATURE_TRACEABILITY.md) | `test_mdo_*.py` except the five listed under S14, `test_requirements.py` |
| S14 | Authoritative re-evaluation, provenance, release gates and run records | `raosim/design.py`, `raosim/mdo/build.py`, `raosim/mdo/contour_validation.py`, `raosim/mdo/execute.py`, `raosim/mdo/postprocess.py`, `raosim/mdo/resolved_inputs.py`, `raosim/mdo/run_record.py`, `raosim/mdo/snapshot.py`, `raosim/model_registry.py`, `raosim/release_readiness.py` | no folder; also [MODEL_REGISTRY.md](architecture/MODEL_REGISTRY.md), [PHYSICAL_RELEASE_GATES.md](architecture/PHYSICAL_RELEASE_GATES.md), [MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md), [INPUT_EXECUTION_INTEGRITY_2026-09-09.md](audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md), [INPUT_EXECUTION_SMOKE_2026-09-09.json](audits/INPUT_EXECUTION_SMOKE_2026-09-09.json) | `test_design_v2.py`, `test_model_registry.py`, `test_release_readiness.py`, `test_mdo_build.py`, `test_mdo_contour_validation.py`, `test_mdo_execute.py`, `test_mdo_snapshot.py`, `test_mdo_resolved_inputs.py`, `test_job_reference_parts.py`, `test_job_execution_integrity.py` |

The five `test_mdo_*.py` files listed under S14 are `test_mdo_build.py`, `test_mdo_contour_validation.py`,
`test_mdo_execute.py`, `test_mdo_snapshot.py` and `test_mdo_resolved_inputs.py`.

Cross-cutting groups (no `subsystems/` folder):

| Group | Name | Code | Documents | Main tests |
|---|---|---|---|---|
| X01 | Differentiable JAX core | `raosim/jax/` (except the five modules listed under X08) | [program/JAX_DIFFERENTIABLE_PLAN.md](program/JAX_DIFFERENTIABLE_PLAN.md) | `test_jax_primitives.py`, `test_jax_assembly_parity.py`, `test_jax_moc_kernel.py`, `test_jax_convergence.py`, `test_jax_theta_b_solve.py`, `test_jax_sensitivities.py` |
| X02 | Lagrangian spray mid-tier and OpenFOAM VOF case export | `raosim/spray/`, `raosim/openfoam/` | [subsystems/S08-injector-and-spray-coupling/](subsystems/S08-injector-and-spray-coupling/) | `test_spray_*.py` (except `test_spray_coupling.py`), `test_openfoam_*.py` |
| X03 | Legacy toolbox CLI | `main.py`, `raosim/{altitude_performance,benchmarks,build_log,conical,nozzle_comparison,trade_study}.py` | none | `test_benchmarks.py`, `test_conical.py`, `test_comparison.py`, `test_v2_features.py` |
| X04 | Vendored NASA/JHU MOC_Grid_BDE reference | `Three-Dimensional-Nozzle-Design-Code-master/`, `raosim/legacy_io.py`, `scripts/compare_nasa_reference.py`, `scripts/nasa_wall_probe.py` | [subsystems/S06-rao-moc-bvp-contour/](subsystems/S06-rao-moc-bvp-contour/), [data/](data/README.md) | `test_legacy_io.py`, `test_compare_nasa_reference.py`, `test_nasa_port.py` |
| X05 | Standalone research scripts and offline generators | `scripts/*.py` (standalone) | none | `test_cea_surface_validation.py`, `test_rao_existence_scan.py` |
| X06 | Knowledge graph, source corpus, packaged data | `knowledge/`, `raosim/data/` | [../knowledge/README.md](../knowledge/README.md) | `knowledge/tests/` |
| X07 | Test harness and regression arg files | `tests/`, `examples/cli/*.args`, `pytest.ini` | [agents/TEST_PROMPT.md](agents/TEST_PROMPT.md) | (the harness itself) |
| X08 | Library code not reachable from the CLI | `raosim/flange_cad.py`, `raosim/thermal_correlations.py`, `raosim/trajectory.py`, `raosim/jax/{bvp,design_opt,pack,reference,residuals}.py` | none | `test_flange_cad.py`, `test_trajectory.py`, `test_jax_bvp.py`, `test_jax_residual_parity.py`, `test_jax_thermal_design_opt.py` |

## Where to put a new document

1. **One stage.** A physics or design basis, a stage-scoped plan, or a provenance note for one stage's equations goes in
   `subsystems/Sxx-<slug>/`, using the id and slug from [architecture/DATAFLOW.md](architecture/DATAFLOW.md). Create the
   folder if the stage has none yet (S01, S02, S03, S11, S12 and S14 have none), and add the document to
   [subsystems/README.md](subsystems/README.md).
2. **Several stages, a contract or a map.** Interfaces, data-flow descriptions, code maps and explainers of a
   repository-wide mechanism (model registry, release gates, output contracts) go in `architecture/`.
3. **Plans and roadmaps across stages.** These go in `program/`. Put the date in the file name (`_YYYY-MM-DD`) and a
   status line at the head; record progress as dated STATUS blockquotes at the head, as the existing plans do.
4. **Audits and evidence.** Dated audits, remediation records, host-run evidence and the JSON they cite go in
   `audits/`, with the date in the file name. Do not rewrite an audit's findings later; add a dated status or closure
   note at its head (as `audits/REPO_PHYSICS_AUDIT_2026-07-02.md` does).
5. **Literature.** Request ledgers, corpus inventories and equation-to-source maps go in `literature/`. Name request
   ledgers `LITERATURE_REQUESTS*.md`; `knowledge/kg/harvest_repo.py` (line 291) harvests that pattern.
6. **Data.** Small reference files kept for reading go in `data/`. Data that code loads belongs in `raosim/data/` or
   `raosim/benchmark_data/` and must be declared as package data in `pyproject.toml`.
7. **Agent prompts** go in `agents/`.
8. **Archiving.** Move a document to `archive/` only when nothing cites it as an authority (search code, tests,
   `raosim/model_registry.py`, `knowledge/kg/` and the live documents), and add a row naming the reason and the
   replacement to [archive/README.md](archive/README.md). An out-of-date document that is still cited stays where it
   is, with a banner at its head.
9. **Not at the repository root.** Keep the root to code, packaging and `README.md`.
10. **After adding, moving or renaming**, update the folder index, check relative links, update any code that names
    the path (next section), and ask the knowledge-system owner to run `knowledge/kg.sh rebuild`: repository-document
    records are keyed by path (`knowledge/kg/harvest_repo.py`, lines 74 and 80), and `knowledge/repo_documents.json`
    still lists the pre-2026-11-05 flat paths until that rebuild runs.

## Document paths that code depends on

| Code location | Path it names | Effect if the document moves |
|---|---|---|
| `raosim/model_registry.py:662` (`local_source`) | `docs/literature/thermofluid_literature_provenance.md` | `audit_model_registry` lists it under `missing_local_sources` and fails; `raosim/run_nozzle.py:4973` writes that audit into the flag runner's `summary.json` |
| `raosim/model_registry.py:349` (source text) | `docs/subsystems/S05-nozzle-and-chamber-geometry/shoulder_radius_design_basis.md` | Stale text only (not existence-checked) |
| `raosim/rao_variational.py:347` (`NASA_REFERENCE_PROVENANCE_DOC`) | `docs/subsystems/S06-rao-moc-bvp-contour/nasa_tt_prime_provenance.md` | Stale provenance path written into contour metadata |
| `scripts/compare_nasa_reference.py:45` (`NASA_TT_PRIME_PROVENANCE_DOC`) | same | Stale provenance path in NASA comparison artifacts |
