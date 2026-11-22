# docs/architecture

Cross-stage contracts and maps: how the CLI flows, which file belongs to which stage, how model provenance and
physical-release evidence are recorded, and the MDO output contract. Documents that describe a single stage live in
[../subsystems/](../subsystems/README.md) instead.

| Document | Status | Type | Stage | What it is |
|---|---|---|---|---|
| [REFACTOR_PLAN_2026-11-21.md](REFACTOR_PLAN_2026-11-21.md) | active (project record for the refactor) | plan | all | Agreed LREKit architecture and phased migration plan; its STATUS head records what has landed (src/ layout, import-identity fix, packaging) and what is open. |
| [relocations_2026-11-21.json](relocations_2026-11-21.json) | added 2026-11-21 | record | all | Every pre-move path and module name mapped to its current location, plus pre-move SHA-256 hashes. Historical records keep old paths; translate through this file. |
| [DATAFLOW.md](DATAFLOW.md) | added 2026-11-05 | architecture | all | Map of the repository by CLI data flow: stages S01-S14 in execution order, the modules each owns, the function and line of each step, files written, and where documents disagree with the code. Defines the stage ids and groups X01-X08; Markdown twin of Chapter 2 of the repository guide. |
| [CODE_MAP.md](CODE_MAP.md) | generated 2026-11-05; paths translated to the src/ layout 2026-11-21 | code map | all | One row per code file (Python, shell, vendored C++ headers, example arg files): stage, line count, one-line role. |
| [CODE_HEADER_COMMENTS.md](CODE_HEADER_COMMENTS.md) | generated 2026-11-05; pre-move paths, regeneration pending | code map | all | Verbatim module docstrings and leading comment blocks of every code file, grouped by stage. |
| [IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md](IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md) | active | architecture | S13 | Design checkpoint, not implemented: how an exact Rao/JAX contour would enter the MDO (pure-array `RaoState`, implicit KKT/IFT derivatives, MDO integration stages A to D, acceptance tests). |
| [MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md](MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md) | active | architecture | S13 | Two-layer MDO output contract (pure-JAX `EngineState`, host `EngineAnalysisSnapshot` 2.0.0), resolved inputs, `EngineJobSpec`, metric registry and the `lrekit` job verbs; remediation table with parity evidence and open solver-equivalence gaps. |
| [MODEL_REGISTRY.md](MODEL_REGISTRY.md) | active | architecture | cross-cutting | Explainer for `src/lrekit/verification/registry.py`: entry fields, what `audit_model_registry` enforces, and validation levels; `hardware_qualified` is never asserted. |
| [PHYSICAL_RELEASE_GATES.md](PHYSICAL_RELEASE_GATES.md) | active | architecture | cross-cutting | Keeps software verification, geometry verification and physical release evidence apart; defines the `lrekit.release_evidence.v1` manifest checked by `src/lrekit/verification/release.py`. |
| [pypi_release.md](pypi_release.md) | active | guide | cross-cutting | Checklist for publishing the `lrekit` package to TestPyPI/PyPI through GitHub Actions trusted publishing. |

Notes:

- `CODE_HEADER_COMMENTS.md` reproduces some `knowledge/` text verbatim, so two relative links inside it (to `AGENTS.md`
  and `reports/...`) resolve against this folder rather than `knowledge/`.
- `MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md` (lines 151 and 157) says `src/lrekit/job_cli.py` adds three verbs and lists
  catalog topics ending at `unavailable`. The code has four verbs (`guided`, `run`, `validate`, `catalog`;
  `src/lrekit/job_cli.py:29`) and eight topics including `capabilities` (lines 31-34); see row 6 of section 10 in
  [DATAFLOW.md](DATAFLOW.md). The code is the authority.
- `MODEL_REGISTRY.md` and `PHYSICAL_RELEASE_GATES.md` explain `src/lrekit/verification/registry.py` and
  `src/lrekit/verification/release.py`, which `CODE_MAP.md` assigns to stage S14.

Status, type and stage come from the per-document triage of 2026-11-05. Identifiers, counts and dates quoted in the descriptions were re-checked against each document on 2026-11-09. *Status*: active = current reference or open plan; historical record = dated snapshot kept for traceability (numbers and statuses are as of its date); superseded = replaced by a named later document; stub = empty, or data without provenance. *Stage* is the pipeline stage the document mainly serves (see [../README.md](../README.md)).
