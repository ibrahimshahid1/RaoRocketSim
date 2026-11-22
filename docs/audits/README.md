# docs/audits

Dated audits, remediation records and host-run evidence notes, including the four files formerly in
`docs/audit_work/`. Each is a snapshot: counts, test totals and statuses are true as of its date, and later work may
have changed them. Where a dated document and the code disagree, the code is the authority.

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [CEA_H1_HOST_EVIDENCE_2026-08-22.md](CEA_H1_HOST_EVIDENCE_2026-08-22.md) | historical record | host_evidence | S03 | Host-run record: four held-out-validated RocketCEA chamber-property tables, the packaged CEA catalog, and a four-propellant 13 kN MDO survey. |
| [CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md](CONTRACT_INTEGRITY_REMEDIATION_2026-09-05.md) | historical record | remediation_record | S02 | Treats findings W01-W04 as one defect (parsing is not binding) and gives fixes and a totality gate; a 9 September note says the repairs landed. |
| [DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md](DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md) | historical record | audit | S13 | Feasibility check of the differentiable MDO plan against corpus and code; one deviation fixed the same day; ten amendments. |
| [FEATURE_DATAFLOW_AUDIT_2026-07-11.md](FEATURE_DATAFLOW_AUDIT_2026-07-11.md) | historical record | audit | cross-cutting | Whole-repository feature, data-flow, literature and CAD audit at commit `ba8fcb6` plus uncommitted work; several of its CLI data-flow claims no longer hold (row 11 of section 10 in [../architecture/DATAFLOW.md](../architecture/DATAFLOW.md)). The later whole-repository audit is `PROJECT_AUDIT_2026-08-25.md`. |
| [INPUT_EXECUTION_INTEGRITY_2026-09-09.md](INPUT_EXECUTION_INTEGRITY_2026-09-09.md) | active | remediation_record | none recorded | Remediation of request/solver/report boundary defects (D01-D04, D26, W01-W04): no silently dropped inputs, variable policies, unified report package. |
| [INPUT_EXECUTION_SMOKE_2026-09-09.json](INPUT_EXECUTION_SMOKE_2026-09-09.json) | historical record | data | S14 | JSON evidence of one `lrekit run` smoke run, cited by `INPUT_EXECUTION_INTEGRITY_2026-09-09.md`. |
| [LITERATURE_AUDIT_2026-07-24.md](LITERATURE_AUDIT_2026-07-24.md) | historical record | audit | S13 | Re-verifies each physical choice in `raosim/mdo/` against `propulsion_texts/` with VERIFIED/SURROGATE/GAP verdicts. |
| [LREKIT_LITERATURE_AUDIT_2026-09-05.md](LREKIT_LITERATURE_AUDIT_2026-09-05.md) | active | audit | cross-cutting | Corpus and acquisition audit of `propulsion_texts/` with issues LIT-01 to LIT-06 and a literature gate sequence. |
| [LREKIT_PUMP_AUDIT_2026-09-05.md](LREKIT_PUMP_AUDIT_2026-09-05.md) | active | audit | S09 | Finds the pump branch is preliminary sizing plus reference CAD, not released hardware; gives pump sequence A-H. |
| [LREKIT_WORKFLOW_AUDIT_2026-09-04.md](LREKIT_WORKFLOW_AUDIT_2026-09-04.md) | active | audit | S02 | Job/CLI/MDO audit with findings W01-W06, a proposed identity chain and its own stages S0-S7 (not the pipeline stages). |
| [MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md](MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md) | active | audit | S13 | Open MDO solver issues and ten blockers that keep the MDO out of the part-geometry workflow, with a repair-status table. |
| [PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md](PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md) | active | remediation_record | none recorded | Maps the 2026-07-02 audit findings to the implementation in 15 areas with the remaining boundary per row; states it is not CFD validation or hardware qualification. |
| [PROJECT_AUDIT_2026-08-25.md](PROJECT_AUDIT_2026-08-25.md) | historical record | audit | cross-cutting | Whole-repository audit at commit f763fe1: both pipelines, capability tiers, nine gaps to the endpoint, validation table, limitations. |
| [RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md](RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md) | historical record | host_evidence | S06 | Host timings and results of the opt-in B3 post-solve variational validator and the B5 declared-domain artifact. |
| [REPO_PHYSICS_AUDIT_2026-07-02.md](REPO_PHYSICS_AUDIT_2026-07-02.md) | historical record | audit | cross-cutting | First full-repository physics and CAD audit against `propulsion_texts/`; its head banner marks it a historical snapshot. |
| [legacy_code_audit.md](legacy_code_audit.md) | historical record | audit | S06 | Early audit of the NASA/JHU MOC_Grid_BDE port; its head banner marks it a historical snapshot. |

Notes:

- `REPO_PHYSICS_AUDIT_2026-07-02.md` and `legacy_code_audit.md` carry their own "historical snapshot" banners and
  point to [PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md](PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md) for current behaviour.
- The stages S0-S7 in `LREKIT_WORKFLOW_AUDIT_2026-09-04.md` are that audit's own proposal, not the pipeline stages
  S01-S14 used to organize `docs/`.
- The HTML rendering of the 2026-08-25 project audit is in [../archive/](../archive/README.md).

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
