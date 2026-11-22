# LREKit refactoring and architecture modernization plan

Agreed architecture and implementation sequence for turning LREKit (formerly RaoRocketSim)
into one modular, installable engineering application. The STATUS blocks below are the
record of what has landed (newest on top); the plan text after them is the agreed
specification as handed over on 2026-11-21, with prose sections condensed (tables and decisions
verbatim) and its "current checkpoint" section replaced by the STATUS blocks.

> **STATUS 2026-11-21 (later) — canonical package renamed `raosim` → `lrekit`.**
>
> *Decision (owner).* The package is named for the whole engine toolkit, not for Rao bells:
> "we have a full engine MDO now". Everything under `src/raosim/` moved to `src/lrekit/`
> with the same relative paths (312 files, none lost). `raosim` is now a compatibility
> package holding only `__init__.py` and `__main__.py`; [`src/lrekit/_compat.py`](../../src/lrekit/_compat.py)
> aliases `raosim.<x>` and the former flat names (`raosim.cea`, `raosim.mdo`, `lrekit.engine`,
> ...) to the one canonical module, and objects pickled under any old name still load.
> [`relocations_2026-11-21.json`](relocations_2026-11-21.json) now maps pre-move paths straight to
> `src/lrekit/` and records both steps.
>
> *Rewrite.* 287 files, 1,702 substitutions, each file verified to differ from its pre-rename
> text only by `raosim` → `lrekit`. Rewritten: import statements, code references to the
> `raosim` name, dotted names under a current subpackage (`raosim.components...`), and
> `src/raosim/` paths. Deliberately not rewritten: versioned identifiers that begin with
> `raosim.` (model ids such as `raosim.rao_existence_scan:...:v2`, format ids such as
> `raosim.cad_units.v1`, the `raosim.chamber-property-table.v1` table format inside the
> packaged CEA catalog, and hash salts); all 283 such strings are byte-identical before and
> after. Prose that names old flat modules is left as is (those names still import).
> Fixes the rename itself required: `importlib.resources.files("lrekit")` for the CEA catalog
> (it would have resolved to the near-empty shim); nine `tests/experiments` provenance filters
> over `sys.modules` (`startswith("raosim")` would have silently recorded empty source-hash
> sets) and one path classifier; the spray `local_source` pointers and the source-collection
> tool's registry path, both stale since the first step.
>
> *Packaging.* Console scripts target `lrekit.application.cli` / `lrekit.cli`; package data is
> declared under `lrekit`; the version is single-sourced from `lrekit.__version__` = 0.1.1
> (the code had reported 0.1.0 while the distribution said 0.1.1). The wheel ships the shim as
> `raosim/__init__.py` and `raosim/__main__.py` only. The 13 kN argfiles document
> `PYTHONPATH=src python -m lrekit @...` (same dispatch as `python -m raosim`).
>
> *Verification.* `test_module_identity.py` is now 84 tests; a `raosim` that shares
> `lrekit.__path__` (the original defect, mirrored) fails 78 of them. `test_packaging.py`
> (slow) passes. Full `-m "not slow"` run: 3,009 passed; the 46 failures are exactly the
> earlier set (22 pre-existing + 24 knowledge-graph), compared by test id per chunk, with no
> new failure. All four `scripts/*.py`, `run.sh`, and `python -m lrekit|raosim` run from a
> checkout with nothing installed.
>
> *Also fixed.* The step-1 doc translation had corrupted the README layout block
> (`src/src/lrekit/`, and "formerly ..." notes rewritten to their own targets); rewritten by hand.
>
> *Open.* (1) Knowledge-graph migration, unchanged and still awaiting the identity-policy
> decision; it should now target `lrekit` names directly, so one migration covers both steps.
> (2) The research-source collection is now at `src/lrekit/data/{raw,derived,catalog,tools}`
> and still belongs in top-level `data/`. (3) Product strings still say RaoRocketSim (the
> `RaoRocketSim*` console scripts, `raosim_openfoam_manifest.json`, CLI banners); rename
> deliberately if wanted. Items (4)–(7) of the block below are unchanged.

> **STATUS 2026-11-21 — Phase 0 and Phase 1 complete; Phase 2 file move landed and repaired;
> knowledge-graph path migration open (decision pending).**
>
> *What the interrupted session did.* A one-shot relocation script moved 264 paths: the
> flat `raosim/*.py` modules into component packages under `src/raosim/` (`components/`,
> `optimization/` ← `mdo/`, `numerics/jax/` ← `jax/`, `assembly/`, `thermochemistry/`, ...),
> `lrekit/` → `src/lrekit/`, `main.py` → `raosim.legacy.toolbox`, 44 research scripts →
> `tests/experiments/`, and 138 test modules into `tests/<area>/`. Every old → new path and
> module name is in [`relocations_2026-11-21.json`](relocations_2026-11-21.json); historical
> records (logbooks, review journals, `result.json` provenance, the MOC extension plan) keep
> the old spelling on purpose and are read through that file.
>
> *Phase 0 baseline.* The pre-move tree was HEAD `085beb5` plus uncommitted code edits, so a
> HEAD checkout is not the baseline. The relocation record carries the SHA-256 of all 482
> pre-move files. Inverting the script's import rewrite reproduces the pre-move hash for
> 421 of 489 files (pure renames); the other 67 carry only path/import edits, each reviewed.
> No file was lost: the 88 deletions outside the script are the earlier `docs/` reorganization
> and LaTeX build products. A byte-exact pre-move tree (452 files hash-verified, 30
> experiment scripts approximate) was rebuilt to classify every test failure below.
>
> *Phase 1, the identity defect.* The old `lrekit/__init__.py` extended `__path__` with
> `raosim.__path__`, so `lrekit.engine` executed `raosim/engine.py` a second time and every
> dataclass, enum and registry existed twice (confirmed in the pre-move tree). The
> replacement, [`src/raosim/_compat.py`](../../src/lrekit/_compat.py) (now `src/lrekit/_compat.py`), is a lazy meta-path alias:
> legacy flat names and `lrekit.*` return the canonical module object. Two latent faults were
> fixed: real modules now always win over an alias (a future `lrekit.components` package would
> otherwise have been hijacked by `raosim.components`), and children of an aliased package
> follow the parent's canonical identity. Canonical code imports nothing through an alias, so
> the shim stays removable. Pinned by
> [`tests/contracts/test_module_identity.py`](../../tests/contracts/test_module_identity.py):
> 81 tests, of which the old `__path__` facade fails 77.
>
> *Phase 1, packaging and paths.* Repairs to the move: `build_log` restored beside its only
> consumer (`raosim.legacy.build_log`; the script had archived it and broken
> `RaoRocketSimLegacy`); the runner no longer inserts a repository root into `sys.path` at
> import time (installed, that root falls back to the caller's cwd); package-data lookups in
> tests and experiments use `package_root()`; `run.sh` and the four `scripts/*.py` bootstrap
> `src/` themselves; the Cuffel–Back–Massier evidence JSON is now package data; the 13 kN
> argfiles say `PYTHONPATH=src`. The wheel is 4.7 MB, ships every runtime table, and excludes
> the 87 MB research corpus under `src/raosim/data/{raw,derived,catalog}`. A wheel installed
> into a fresh venv runs all six entry points (`lrekit`, `raosim`, `RaoRocketSim`,
> `RaoRocketSimLegacy`, `python -m lrekit|raosim`) outside the checkout:
> [`tests/contracts/test_packaging.py`](../../tests/contracts/test_packaging.py) (slow).
>
> *Environment.* `.venv-jax` had lost `pip/_internal/operations/build/` and the whole pypa
> `build` package at 14:34 — consistent with a sweep of directories named `build`. Both were
> repaired (pip 26.1 → 26.2.1 from the interpreter's bundled wheel, `build==1.5.0`); no
> other installed file is missing against its RECORD. The stale `RaoRocketSim 0.1.0`
> editable install (its finder pointed at the deleted root `raosim/`) and the root
> `lrekit.egg-info` were removed and `lrekit` reinstalled editable from `src/`.
>
> *Test state (`-m "not slow"`, run in parallel chunks; every failure re-run in the pre-move tree).*
> numerics 538/538; application 457 + 9 failed; components 836 + 15 failed; optimization
> 451 + 8 failed; assembly/mission/thermochemistry/verification 187 + 2 failed; knowledge
> 444 + 24 failed; new contracts 81/81 (+2 slow packaging tests).
> **Move-induced and fixed (12):** three legacy batch-CLI tests that ran `main.py`; four spray
> benchmark tests and two `rao_domains` artifact tests reading `<repo>/raosim/...`; the
> Cuffel evidence path; `tests.test_moc_bounded_net` imported by its old name; and the pinned
> `source_model_id` literal in `test_rao_convergence_domain.py`, which the rewrite had changed
> although the code and the packaged domain artifacts keep `raosim.rao_existence_scan:...`
> (model IDs are identifiers, not import paths; no other identifier literal was rewritten).
> **Pre-existing (22; same failures in the pre-move tree, so not caused by the move):** 15
> follow from an uncommitted pre-move change to the TOP chart: domain narrowed to ε 5–50 and
> chart outputs moved (tests still probe ε = 4, or pin pre-change angle and bell-length
> values, e.g. a pinned 0.12912 m bell length that now evaluates to 0.12992 m): `test_nozzle_geometry` ×4,
> `test_rao_chart_benchmark`, `test_design_v2` ×2, `test_design_workflow` ×2,
> `test_phase6_coupled_wall`, `test_run_nozzle` Bézier extrapolation, `test_mdo_bounds`,
> `test_mdo_engine` ×3); 4 `test_mdo_nlp_robustness`; `test_regen_cad` full-N manifold
> boolean; and the LEA/TOP Bézier benchmark pair in `test_benchmarks.py`.
> **Move-induced and open (24):** all in `knowledge/tests` — see item (1) below.
>
> *Open, in order.* (1) Knowledge-graph migration: `knowledge/kg` hard-codes ~35 pre-move
> paths/modules and derives module names from repo-relative paths, and record IDs embed paths
> (`SW_RAOSIM_PUMPS_PY_...`); a naive rebuild would mark ~440 verified locators unavailable
> and append "comparison requires review" history. Needs an identity policy (keep IDs,
> migrate `file`/qualname fields once, leave signed journals untouched) before rebuilding.
> (2) Move the research-source collection `src/raosim/data/{raw,derived,catalog,tools}` to
> top-level `data/` (§4.2); it is not runtime package data. (3) Canonical package name:
> implementation still lives in `raosim.*` with `lrekit.*` as the alias; §4.1 targets the
> reverse. (4) Experiments: provenance file lists hashed at run time still name pre-move
> paths (e.g. `audit_moc_refinement_pair.py`); `moc_ext_*` reproduction commands in the MOC
> plan read `scripts/...`. (5) Regenerate `CODE_HEADER_COMMENTS.md` (a verbatim snapshot with
> pre-move paths; `CODE_MAP.md` and `DATAFLOW.md` are already translated). (6) Pre-existing
> test debt listed above. (7) Phase 3: component and job contracts.

---

# Plan as agreed (2026-11-21)

**Project:** LREKit (formerly RaoRocketSim)
**Document type:** Software architecture and implementation specification
**Status:** Agreed architecture and proposed implementation sequence
**Reference configuration:** 13 kN LOX/RP-1, electric-pump-fed, regenerative cooling, fixed-pintle architecture

## 1. Executive Summary

LREKit shall transition from a collection of interconnected numerical analysis tools, research implementations, optimization routines, CAD exporters, and CLI scripts into a unified, modular engineering software platform.

The application shall provide a consistent interface for defining high-level requirements, resolving configurations, evaluating mathematical models, managing component representations, conducting numerical assessments, and producing traceable research outputs.

The refactor has five primary objectives:

1. **Codebase modernization:** Establish a professional, understandable, maintainable repository organized by component and responsibility.
2. **Application unification:** Consolidate CLI, configuration, numerical analysis, optimization, and reporting around a single application contract.
3. **Component modularity:** Give every major component a clearly owned software package with defined interfaces to the wider system.
4. **Geometry consistency:** Eliminate conflicting component representations across analysis, CAD, simulation, and assembly-level models.
5. **Verification and extensibility:** Preserve existing numerical capabilities and establish a foundation for reproducible simulation, experimental comparison, and future research.

The immediate refactor is a software architecture project, not a claim that the current numerical models or generated geometry are experimentally qualified.

The guiding principle shall be:

**One request → one resolved configuration → one coordinated set of component representations → one traceable result.**

## 2. Confirmed Project Decisions

| Decision | Agreed direction |
|---|---|
| Initial product scope | Engine-level analysis first |
| Reference configuration | 13 kN, LOX/RP-1, electric pump, regenerative cooling, fixed pintle |
| Additional reference engines | None for the first milestone |
| 500 N reference case | Excluded |
| Vehicle altitude/trajectory optimization | Deferred |
| Code organization | Component-oriented packages under `src/` |
| Repository structure | `src/`, `models/`, `tests/`, `archive/`, with supporting project directories |
| CLI | One principal interface with guided and advanced configuration |
| Optimization | Existing MDO framework retained and integrated |
| Geometry | Shared, versioned component definitions |
| CAD philosophy | Parametric, code-driven geometry informed by explicit design-intent records |
| Purchased components | Vendor representations when appropriate and available |
| Simulation | Modular external-solver interfaces, including OpenFOAM |
| Experimental data | Curated, traceable datasets used for model assessment |
| Deprecated code | Removed from active execution paths and archived or deleted |
| Compatibility | Incremental migration, preserving supported public interfaces |

The reference configuration is a fixed software-regression target. It does not imply that every physical aspect of that configuration has already been independently validated.

## 3. The Intended Application Experience

### 3.1 Guided user mode

The primary CLI should ask for a small set of meaningful engineering requirements. The application should distinguish user intent from the numerical variables used internally. Typical categories include performance requirements and reference operating conditions; duration or mission-related requirements supported by the selected workflow; propellant and architecture selection; packaging or other declared constraints; preferred optimization objectives; numerical assessment and reporting preferences.

The user should not need to understand every internal model, numerical method, or component-specific parameter before beginning an analysis. The user interface shall expose only supported functionality. Unsupported requirements should produce an explicit explanation rather than silently selecting assumptions.

### 3.2 Advanced user mode

An advanced user should be able to provide a configuration file containing additional model parameters, solver preferences, analysis settings, and explicit constraints. All parameter decisions should belong to a common policy system:

| Parameter policy | Meaning |
|---|---|
| Free | Application may select the value within the supported domain |
| Fixed | User explicitly specifies the value |
| Bounded | Value must remain inside a supplied interval |
| Preferred | Soft preference, distinct from a hard constraint |
| Discrete | Value selected from declared alternatives |
| Derived | Value calculated from other resolved quantities |

A preference must never be silently converted into a physical constraint.

### 3.3 One unified application contract

Guided CLI, advanced configuration files, programmatic APIs, and a future graphical interface shall all produce the same versioned job specification. The existing `EngineJobSpec` is the intended starting point. No front end should implement an independent version of the engine-analysis workflow.

### 3.4 Proposed user-facing command organization

| Command family | Responsibility |
|---|---|
| `lrekit new` | Create a configuration |
| `lrekit validate` | Check schema, supported capabilities, and requirements |
| `lrekit solve` | Run a supported numerical analysis job |
| `lrekit inspect` | Examine configurations, component states, and results |
| `lrekit simulate` | Manage supported numerical simulation jobs |
| `lrekit report` | Generate traceable analysis reports |
| `lrekit export` | Export supported research artifacts |

Legacy command-line options may remain temporarily through compatibility adapters, but they should not create competing execution pathways.

## 4. Proposed Repository Architecture

The repository should communicate its organization immediately to a new contributor: CLI, physics, component models, geometry representations, simulation adapters, and testing infrastructure should be identifiable without tracing hundreds of unrelated imports.

### 4.1 Target directory structure

```text
LREKit/
├── src/
│   ├── lrekit/
│   │   ├── __init__.py
│   │   ├── cli/  app/  core/
│   │   ├── components/
│   │   │   ├── nozzle/  chamber/  injector/  cooling/
│   │   │   ├── pumps/  feed_system/  electrical/  structure/
│   │   ├── geometry/  assembly/  physics/  optimization/
│   │   ├── simulation/
│   │   │   ├── openfoam/  structural/
│   │   ├── calibration/  data/  reporting/  exports/
│   └── raosim/
│       └── compatibility adapters
├── models/      registry/  metadata/  training_configs/
├── tests/       unit/  integration/  regression/  contracts/  fixtures/  experiments/
├── data/        catalog/  references/  benchmarks/
├── examples/    reference_case/
├── docs/        architecture/  components/  models/  validation/  decisions/
├── scripts/
├── archive/
├── pyproject.toml  pytest.ini  README.md  CONTRIBUTING.md  .gitignore
```

This is a target structure rather than a requirement to create every folder immediately. Empty architecture should not be generated merely to appear comprehensive.

### 4.2 Responsibilities of the top-level directories

- **`src/` — production software.** The installable packages and all actively supported application code.
- **`models/` — model artifacts and metadata.** Trained-model manifests, calibration configurations, model-selection records, reproducibility metadata. Implementation code belongs inside the installable package; large binary weights should not ordinarily be committed.
- **`tests/` — verification and reproducible experiments.** Automated tests, regression fixtures, integration checks, and explicitly labeled research experiments. Experimental scripts must not be confused with pass/fail tests; the ordinary suite excludes them by default.
- **`archive/` — inactive historical implementations.** Local retired code only when retaining it is useful; never imported by production code. `.gitignore` does not stop tracking files already committed, and Git history is generally preferable to preserving abandoned source trees.
- **`data/` — scientific data provenance.** Data catalogs, benchmark definitions, and references to reviewed experimental sources; large datasets use managed external storage when appropriate.
- **`docs/` — developer-facing documentation.** Architecture decisions, component interface specifications, mathematical assumptions, and validation status.

## 5. Component-Oriented Design

Each major component shall have its own software package owning the implementations specific to that component; generic infrastructure remains in shared modules.

### 5.1 Standard component architecture

A component package may contain `schemas.py`, `models.py`, `analysis.py`, `geometry.py`, `interfaces.py`, `validation.py`, `reporting.py` — only where the responsibilities are genuinely needed, without mechanically duplicating boilerplate.

### 5.2 Component boundaries

| Package | Software ownership |
|---|---|
| Nozzle | Contour representations, existing nozzle-method adapters, and numerical result definitions |
| Chamber | Chamber representations, numerical analysis, and geometry interfaces |
| Injector | Supported injector model families, parameter schemas, and validation interfaces |
| Cooling | Thermal and coolant-model interfaces and associated geometry representations |
| Pumps | Existing pump-analysis models, geometry interfaces, and capability declarations |
| Feed system | Abstract system connectivity, interfaces, and component inventory |
| Electrical | Electrical-system representations, selected-component metadata, and accounting interfaces |
| Structure | Structural model definitions and assessment interfaces |
| Assembly | Component placement, interface consistency, and consolidated model state |

Common mathematics, units, numerical utilities, and generic geometry operations should not be copied into these packages.

### 5.3 Component contracts

Every component should expose: identity and model version; input parameters and their provenance; supported analysis methods; geometry representation and geometry version; interfaces with neighboring components; assumptions and validity domains; evaluation results and uncertainty; validation and export status; dependencies and reasoned failure states. A component should not be able to claim successful evaluation simply because one numerical routine returned without an exception.

### 5.4 Dependency architecture

**Application → orchestration → component APIs → shared numerical infrastructure.** Components should not depend on CLI formatting, filesystem layout, or another component's private implementation; cross-component coordination belongs in the orchestration and assembly layers.

## 6. A Single Authoritative Geometry Model

Previous audits found analysis, detailed geometry, assembly, and export logic referring to related but nonidentical physical representations. This cannot be corrected through directory restructuring alone.

### 6.1 Geometry as a first-class software object

Each component should have a canonical geometry specification containing identity and version, declared geometric parameters, coordinate-frame information, named interfaces and reference surfaces, explicit assumptions and representations, relationships to other component objects, and provenance of the producing model. Derived representations reference the source specification and identify any approximation.

### 6.2 One geometry, multiple representations

The geometry layer distinguishes (1) design definition, (2) analysis representation, (3) CAD representation, (4) simulation representation, (5) assembly representation, (6) export artifact. They need not share mathematical complexity, but their differences must be explicit. An analysis should never quietly evaluate one geometry while the exporter writes a different one.

### 6.3 Geometry interfaces between components

The assembly layer maintains shared interface records. Two components sharing a boundary reference the same authoritative interface specification (e.g. the common chamber/nozzle throat interface) with shared identity, consistent units, declared coordinate systems, and explicit compatibility checks. An incompatible pair produces a diagnostic failure, not an apparently complete assembly.

### 6.4 CAD philosophy

**Parametric, code-driven custom geometry informed by explicit design-intent records, with externally supplied geometry for purchased components when appropriate.**

| Status | Meaning |
|---|---|
| Conceptual | Geometric placeholder or conceptual representation |
| Parametric | Generated from a declared parameter model |
| Kernel-validated | Passed supported CAD-kernel and topology checks |
| Interface-consistent | Passed declared inter-component compatibility checks |
| Independently assessed | Evaluated against independent geometric or engineering evidence |
| Review required | Additional engineering review remains outstanding |

CAD-kernel validity does not establish physical suitability, manufacturability, or operational qualification.

### 6.5 Previously identified geometry gaps

Tracked as existing issues, each to become a reproducible test with resolution criteria: divergent cooling-system analysis and detailed CAD representations; incomplete assembly-level interconnection; partial or placeholder geometry in some pump-related exporters; build manifests that establish file identity without proving cross-component consistency; optimized configurations a CAD adapter cannot represent faithfully; unresolved geometric changes introduced by export or representation conversion.

## 7. Optimization and Numerical Solver Architecture

### 7.1 Separate the numerical responsibilities

Requirement resolution; component numerical evaluation; coupled system evaluation; optimization; geometry construction; higher-fidelity simulation; model assessment; reporting and artifact generation — not all controlled through a single monolithic runner.

### 7.2 Solver interfaces

Methods are registered behind stable interfaces describing identity, required inputs, applicability domain, outputs, backend and dependencies, convergence and validity status, and model/implementation versions. No method is selected solely because another failed; a fallback must itself be applicable and assessed.

### 7.3 Nozzle-method policy

- **TOP/Bézier:** the fast contour-construction path and current MDO baseline. Chart domain: expansion ratio 5–50, bell length 60–100 %.
- **Variational Rao/MOC:** an alternate research implementation with its own assumptions, convergence requirements, and domain.
- **OpenFOAM or another CFD backend:** a separate assessment tool, not a replacement for contour construction or optimization.

Outside the TOP chart bounds the application performs capability-based routing; it must not silently clamp, extrapolate undeclared, or assume the variational implementation is automatically valid.

### 7.4 MDO integration

The optimizer consumes component evaluator interfaces; model versions and assumptions stay fixed within a run. Results include the resolved input configuration, the exact model set, numerical status and constraint outcomes, convergence information, the selected candidate's identity, and re-evaluation/assessment status. A mathematically feasible candidate is not automatically a validated design.

## 8. Simulation and Model-Feedback Architecture

Later-stage capability following trustworthy component contracts and geometry provenance. A simulation job identifies the component representation, a versioned simulation profile, the external solver and version, numerical assumptions and inputs, reproducibility settings, output locations and result schemas, and convergence/assessment status. Fast evaluation (existing models, MDO) is separated from the assessment layer (selected high-cost simulations, experimental comparison, controlled model updates); no training or model modification during an optimizer evaluation. Feedback creates an independently identifiable model revision; the previous model stays reproducible; a new version replaces a baseline only on quantitative comparison and recorded evidence. Results distinguish success from nonconvergence, unsupported domain, invalid inputs, missing validation evidence, inconsistent interfaces, exceeded budget, and insufficient improvement relative to uncertainty.

## 9. Experimental Data and Model Management

The catalog distinguishes: literature identified → source retrieved → inspected → numeric observations extracted → case reviewed → dataset accepted for a specified task → model comparison completed. Retrieval of a PDF does not establish extraction or validation. Initial source families: nozzle performance (NASA TP-2725, TP-3576); gas-side thermal (Santoro–Pal, Back–Massier–Gier); coolant-side thermal (Meyer NASA TM-106985); pintle performance (TRW, Austin, Sakaki); spray (Radhakrishnan experimental cases); pumps (NASA references, candidate measured maps). Accepted datasets preserve source identity, observation definitions and units, test conditions, uncertainty, extraction history, version, limitations, and intended task. Complex models are compared against simpler baselines with attention to leakage, independent groups, derived-variable dependence, numerical error, and uncertainty; models that do not improve the declared task do not replace simpler accepted ones.

## 10. Application Orchestration and Result Management

Canonical job lifecycle: user requirements → schema and capability validation → resolved job configuration → component model initialization → coordinated numerical evaluation → result and geometry consistency checks → optional independent assessment → versioned artifacts and reports. Every significant run has a stable run identifier, configuration hash, software version, model versions, geometry specification versions, input/output manifests, validation results, and a reproducible execution record; re-running never overwrites earlier results. The component inventory distinguishes active models, external/purchased specifications, conceptual representations, not-yet-implemented, and not-supported-for-this-operation.

## 11. Archive and Legacy-Code Policy

One active implementation pathway per supported responsibility. Each source file is classified **Active**, **Compatibility**, **Experimental**, **Deprecated**, or **Dead**. Migration rules: do not move code merely because its filename resembles a component name; identify imports, runtime behavior, and public APIs first; move one coherent group at a time; keep scientific corrections separate from behavior-preserving refactoring; preserve regression fixtures; remove compatibility layers after consumers migrate; retain history in Git rather than duplicate code trees. Compatibility namespaces may re-export existing objects but must not re-execute or independently load their source modules — particularly for dataclasses, enums, registries, and caches.

## 12. Testing and Verification Strategy

| Category | Purpose |
|---|---|
| Unit | Individual functions, classes, and mathematical contracts |
| Contract | Shared interfaces, schemas, and type identity |
| Integration | Interactions among application subsystems |
| Regression | Preserve established behavior and accepted reference outputs |
| Geometry | Verify consistency and supported geometric invariants |
| Numerical | Check convergence and supported numerical behavior |
| Packaging | Confirm installation, imports, and entry points |
| Experimental | Isolated research assessments, excluded from routine test runs |

The initial integration baseline uses only **13 kN / LOX-RP-1 / electric pump / regenerative cooling / fixed pintle**, as a software-regression target, not evidence of physical qualification. Required checks: public imports resolve; canonical and compatibility imports reference the same class objects; CLI entry points execute the same application logic; guided and advanced representations of equivalent jobs resolve consistently; reference outputs stay within declared tolerances; component records and geometry share expected identities; unsupported functionality is reported explicitly; artifact manifests are reproducible; clean-environment installation works; archived code is not required at runtime.

## 13. Implementation Roadmap

| Phase | Purpose | Exit criterion |
|---|---|---|
| 0 | Establish the migration baseline | All subsequent changes can be compared against a documented initial state |
| 1 | Repair package identity and installation | One implementation corresponds to one module identity, regardless of import route |
| 2 | Complete repository reorganization | A new contributor locates component functionality through the package structure |
| 3 | Establish component and job contracts | Components invoked and inspected through stable application-level interfaces |
| 4 | Restore the reference workflow | The reference job runs through the new architecture without the old monolithic path |
| 5 | Unify geometry representations | Reference components have consistently identified representations with documented approximations |
| 6 | Consolidate reporting and artifacts | A completed job produces a coherent, reproducible record |
| 7 | Numerical simulation integration | A supported assessment is submitted, executed, inspected, and reproduced through application contracts |
| 8 | Experimental model assessment | Model claims are supported by inspectable evidence |
| 9 | Broader application capabilities | Scheduled only after the relevant contracts are stable |

## 14. Immediate Implementation Priorities

| Priority | Work item | Completion evidence |
|---|---|---|
| P0 | Recover interrupted migration | Working tree inspected; changes preserved |
| P0 | Resolve import identity defects | No duplicated module or class identities |
| P0 | Validate installable source layout | Clean installation and supported CLI entries work |
| P1 | Establish source ownership | Active component modules have clear packages |
| P1 | Simplify application entry points | Central job interface handles supported execution |
| P1 | Establish shared component contracts | Contract tests pass |
| P1 | Restore reference case | Recorded outputs match accepted baseline |
| P2 | Unify geometry provenance | Analysis/CAD representations have consistent identities |
| P2 | Consolidate results and documentation | Reproducible job manifests and readable reports |
| Later | Simulation and experimental integration | Independent assessment evidence exists |

Prefer narrow changes with passing tests and independently reviewable outcomes; do not mix these into one large pull request.

## 15. Definition of Done for the Initial Refactor

1. Coherent, installable `src/`-based architecture. 2. Active component implementations organized by responsibility. 3. No unrelated duplicated utilities. 4. The CLI invokes one canonical workflow. 5. Guided and advanced configuration resolve consistently. 6. The 13 kN reference workflow is reproducible. 7. Component representations and interfaces trace to authoritative specifications. 8. CAD capabilities and limitations are reported accurately. 9. Deprecated and experimental code are separated from normal execution. 10. Test, packaging, and compatibility checks pass. 11. Research artifacts carry provenance sufficient to reproduce them. 12. Documentation explains structure, responsibilities, and capability boundaries.

This milestone does **not** require OpenFOAM-based refinement, trajectory optimization, experimental qualification, or hardware-ready manufacturing outputs.

## 16. Non-Goals for the Initial Refactor

A second reference engine; the 500 N case; vehicle-level apogee optimization; replacing every numerical method; training models during optimization; every component variant; rebuilding all CAD from scratch; a GUI before the application contract is stable; declaring generated geometry manufacturing-ready; physical assembly or propulsion testing.

## 17. Long-Term Architectural Principle

LREKit is a collection of independently testable scientific and engineering software components coordinated by a unified application layer. Each model states what it represents, its assumptions, where it applies, what evidence supports it, how it relates to other components, whether its outputs are consistent with the wider configuration, and what remains unresolved.

**The application must not merely produce an answer. It must make clear which models produced that answer, which geometry was represented, how the result was checked, and which claims remain unsupported.**
