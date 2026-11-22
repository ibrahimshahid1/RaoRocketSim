# docs/program

Plans, roadmaps and backlogs that span several pipeline stages, plus the project's running status record. A plan
scoped to one stage may live in that stage's folder under [../subsystems/](../subsystems/README.md) (for example
`PUMP_CAD_IMPLEMENTATION_PLAN.md`, `REWRITE_PLAN.md`, `DIFFERENTIABLE_ENGINE_MDO_PLAN.md`).

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [COMPLETION_PLAN.md](COMPLETION_PLAN.md) | active | roadmap | cross-cutting | Deduplicated backlog of remaining work (tracks A-G) with acceptance gates and host batches H1-H5; front matter says last updated 2026-08-26. |
| [COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md](COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md) | active | plan | S13 | PROPOSED plan with no executable code: evidence interfaces for component models, fitting outside the JAX MDO, a conical-pintle pilot, an OpenFOAM round trip. |
| [GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md](GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md) | active | plan | S13 | Plan to generalize the 13 kN LOX/RP-1 electric-pump tool across thrust class, propellant and feed architecture, with an acceptance suite and work orders. |
| [INTEGRATION_PROGRAM_2026-08-22.md](INTEGRATION_PROGRAM_2026-08-22.md) | active | roadmap | cross-cutting | Eight-item gap list from design candidates to one fabricable article, sequenced as Phases 0-7 with the literature that blocks each. |
| [JAX_DIFFERENTIABLE_PLAN.md](JAX_DIFFERENTIABLE_PLAN.md) | active | plan | S06 | Build plan for the JAX Rao BVP backend (`raosim/jax/`); its head STATUS blockquotes are the running project record (newest dated 2026-06-15). |
| [LREKIT_DELIVERY_BACKLOG_2026-09-05.md](LREKIT_DELIVERY_BACKLOG_2026-09-05.md) | active | roadmap | cross-cutting | Implementation backlog D01-D26 with an issue contract and milestones toward a first-article exit gate. |
| [LREKIT_HARDWARE_PROGRAM_2026-09-05.md](LREKIT_HARDWARE_PROGRAM_2026-09-05.md) | active | plan | cross-cutting | Draft tracks H-M from reference geometry to machinable, testable hardware; its head banner defers corrected claims to the readiness review. |
| [LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md](LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md) | active | plan | cross-cutting | Readiness review and delivery index: capability state, target architecture, milestones M0-M6. The root README points new readers here. |
| [PHYSICS_FIDELITY_ROADMAP.md](PHYSICS_FIDELITY_ROADMAP.md) | active | roadmap | none recorded | Seven-item fidelity roadmap (variable-gamma MOC, boundary-layer displacement, separation, thermal/structural coupling, Rao promotion, differentiability, CAD/GD&T). |
| [PLAN_VALVES_LINES_BRACKETS_GIMBAL.md](PLAN_VALVES_LINES_BRACKETS_GIMBAL.md) | active | plan | S10 | Unimplemented design plan for valves, lines, brackets and gimbal, the subsystems still missing from the mass ledger. |
| [PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md](PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md) | active | plan | S09 | Pump 3-D geometry sequence P0-P11 with corrections C1-C8 and an SP-8109/SP-8052 literature ledger; the Track H geometry backlog. |
| [ROADMAP_REQUIREMENTS_TO_PARTS.md](ROADMAP_REQUIREMENTS_TO_PARTS.md) | active | roadmap | S13 | Roadmap R0-R6 from requirements to parameterized parts. Its status table (clarified 2026-10-15) marks R3, the requirements layer, partly implemented; the 2026-08-25 audit had flagged the older not-started entry. |

How to read this folder:

- The STATUS blockquotes at the head of [JAX_DIFFERENTIABLE_PLAN.md](JAX_DIFFERENTIABLE_PLAN.md) are the project's
  running record for the Rao/JAX contour and physics-coupled design work. The newest of them is dated 2026-06-15;
  later status is kept at the head of the dated documents here and in [../audits/](../audits/README.md)
  (for example [../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md)).
- [LREKIT_HARDWARE_PROGRAM_2026-09-05.md](LREKIT_HARDWARE_PROGRAM_2026-09-05.md) opens with a banner that makes the
  readiness review and its D-series backlog the delivery index; treat its corrected claims accordingly. The verb
  family it proposes (lines 620-630: `job`, `component`, `build`, `test`, `calibrate`, `ui`) does not exist; the
  implemented job verbs are `guided`, `run`, `validate` and `catalog` (`lrekit/job_cli.py:29`).
- Several plans state their own maturity: `COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md` is PROPOSED with no
  executable code, and `PLAN_VALVES_LINES_BRACKETS_GIMBAL.md` is unimplemented.

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
