# docs/literature

Literature request ledgers, corpus inventories and equation-to-source provenance notes. The source PDFs themselves
live in the local, gitignored `propulsion_texts/` corpus; the scientific memory that links them to equations, code and
tests is described in [../../knowledge/README.md](../../knowledge/README.md).

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [LITERATURE_REQUESTS.md](LITERATURE_REQUESTS.md) | active | literature_request | cross-cutting | Master ledger of 26 numbered literature requests by priority (valves/lines, structures, pump rotor, feed, TVC, igniter), a 2026-10-16 holdings check, and eight further requests raised that day. |
| [LITERATURE_REQUESTS_GENERALIZATION.md](LITERATURE_REQUESTS_GENERALIZATION.md) | active | literature_request | S02 | Sources G1-G14 needed to generalize across thrust class, propellant and feed cycle, with acquisition status. |
| [LITERATURE_REQUESTS_HARDWARE_LOOP.md](LITERATURE_REQUESTS_HARDWARE_LOOP.md) | active | literature_request | S09 | Gaps H1-H14 for pump manufacturing, assembly testing and measurement ingestion; resolves the SP-8109 reference list. |
| [REMEDIATION_LITERATURE_TRACEABILITY.md](REMEDIATION_LITERATURE_TRACEABILITY.md) | active | provenance | S13 | Source ledger and claim/decision matrix separating literature, repository calculation and software policy in the MDO remediation. |
| [chamber_design_literature.md](chamber_design_literature.md) | active | literature_inventory | S05 | Maps chamber and convergent-section geometry rules to held SP-125 and SP-8120 sections; acquisition checklist. |
| [pintle_injector_literature_inventory.md](pintle_injector_literature_inventory.md) | active | literature_inventory | S08 | First-pass map of `propulsion_texts/pintle_injector/`: priority sources, local inventory, papers to add or verify. |
| [thermofluid_literature_provenance.md](thermofluid_literature_provenance.md) | active | provenance | S07 | Equation-to-code map for thermofluid, regen-cooling and wall-structure models with sources and limits; a `local_source` of the model registry. |

Notes:

- `knowledge/kg/harvest_repo.py` (line 291) harvests every `LITERATURE_REQUESTS*.md` under `docs/` recursively as
  literature-request records; keep that file-name prefix for new request ledgers.
- `raosim/model_registry.py` (line 662) names `docs/literature/thermofluid_literature_provenance.md` as a
  `local_source`; `audit_model_registry` reports a missing `local_source` as a failure, so moving that file requires
  editing the registry.

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
