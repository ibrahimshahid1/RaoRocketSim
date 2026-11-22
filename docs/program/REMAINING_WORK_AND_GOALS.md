# Remaining work and program goals

Status: active register. Written 2026-11-09 against `main` at `d636d85` plus the uncommitted working tree.
Companion of Chapter 14 of the repository guide
([`ch14_remaining_work.tex`](../../latex-report/RaoRocketSim_Repository_Guide/chapters/ch14_remaining_work.tex)), which
carries the same register with the same IDs.

This page answers two questions: what the program is now trying to achieve, and what is still needed. It consolidates
the open items of the plans in this folder, the audits in [`../audits/`](../audits/README.md), the subsystem documents in
[`../subsystems/`](../subsystems/README.md), [README.md](../../README.md) ("Known limitations" and "Remaining work"),
the knowledge-system reports, the per-stage defect sections of the repository guide, and two documents that are not
stored in the repository:

- **Review (7 Oct)**: *LREKit Consolidated Technical Review and Forward Program*, dated 7 October 2026, examining commit
  `6c04d92`. Page numbers below refer to its 65-page revision (exported 8 October 2026), which records the owner's
  decisions of 7 October; the earlier 46-page export does not.
- **Plan (3 Nov)**: *From Engine Design to Experimental Feedback: component data, multi-fidelity models, and the first
  implementation plan*, 3 November 2026, 17 pages, compiled at
  [`output/pdf/LREKit_Component_Data_and_Research_Plan.pdf`](../../output/pdf/LREKit_Component_Data_and_Research_Plan.pdf).
  Its Markdown source of recommendations is
  [COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md](COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md)
  (status: PROPOSED, no executable code).

Nothing here is a hardware qualification record. Every contour is stamped `hardware_qualified = False`; the default
physical-release report is BLOCKED with 17 external-evidence requirements missing
([`raosim/release_readiness.py`](../../raosim/release_readiness.py), 10 engine, 2 injector, 2 regen, 3 pump). Where a
document and the code disagree, the code wins, and the register says so.

**Priorities.** **P0** blocks trust in a result today (a crash without a record, a false feasible, a silent
substitution, a legal or release blocker). **P1** is near-term work inside the owner's first stage (credibility).
**P2** is later: after the credibility gate, research-track, or blocked on literature or hardware.

**IDs.** `RW-<group>-<nn>`. Groups are the pipeline stages S01-S14 and cross-cutting groups X01-X08 of
[`../architecture/DATAFLOW.md`](../architecture/DATAFLOW.md), plus `PRG` for program-level release and governance work.
IDs are stable: a closed item keeps its ID and moves to the closed table.

---

## A. Overall goals now (program level, in order)

The order is the owner's, recorded in the Review (7 Oct) §3 (p. 7) and §8 (p. 22): credibility first, then CAD with a
simulated article, then hardware.

| # | Goal | What "done" means | Source |
|---|---|---|---|
| G1 | Trustworthy, recoverable runs | Every request ends in a complete package or a typed refusal naming the blocking part, model or domain; no crash without a summary, no silent substitution, no false feasible | Readiness plan §4 and M0 ([LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md](LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md)); Review §11.7 p. 45 |
| G2 | One configuration-controlled engine definition | One `configuration_id` binds solve, CAD, mass and BOM; analyzed wall equals exported wall within a stated tolerance; every mass measured from built geometry or `None` with a reason | Review §8.1 P1 p. 23; [COMPLETION_PLAN.md](COMPLETION_PLAN.md) §13; delivery backlog D05 |
| G3 | Credibility release (owner's Stage 1) | Each output carries model, source basis, validity domain, numerical uncertainty, validation discrepancy and evidence state; the §8.3 scorecard passes; release tagged and archived privately until licensing is settled | Review §3 p. 7, §8.1 P2-P6 pp. 23-25, §8.3 pp. 26-27 |
| G4 | Simulated article (owner's Stage 2) | Chamber and nozzle assembly, then injector, then large pump parts, modelled to L4 from one configuration and ported to OpenFOAM and FEA, with signed deltas against the low-order models | Review §3 decision 2 p. 7, §8.1 C1-C4 pp. 24-25, §11.5 p. 43 |
| G5 | Experimental feedback that changes the design | One real pintle dataset, one reproducible hydraulic model and one engine comparison showing how that model changes sizing and coupled results; then the CFD round trip | Plan (3 Nov) p. 1 and §8; Review decision 3 (first physical article: water cold-flow pintle) |
| G6 | Hardware loop (owner's Stage 3) | Water cold-flow pintle, then pump water bench, then a chamber and nozzle article only under a test facility's approvals; measured data enter the same comparison records | Review §8.1 H1-H4 pp. 25-26; readiness plan M3-M5 |
| G7 | Breadth by declared domains, never by a scale factor | "Any thrust class" becomes "any request, answered or refused with reasons"; 500 N explained before more is built on it | Review §3 decision 5 p. 7, §8.2 p. 26; readiness plan §2 |
| G8 | Research questions on the gated tool | RQ1-RQ4 run on the released tool and cite its configuration hashes | Review §8.1 R p. 26; [COMPLETION_PLAN.md](COMPLETION_PLAN.md) §0, §7 |
| G9 | Public release, later | License, copyright purge, export-control determination and citation metadata in place before anything is published | Review decision 4 p. 7, §13 pp. 48-49 |

The standing constraint over all nine: the physical-release gate stays intact. No output claims fabrication release,
hardware qualification or L5 ([../architecture/PHYSICAL_RELEASE_GATES.md](../architecture/PHYSICAL_RELEASE_GATES.md);
Review §11.7).

---

## B. Open-work register

Columns: ID, item, why it matters, evidence, dependency, priority. "Guide ch. N" refers to the repository guide
chapter that verified the item in code on 2026-11-05. "Verified 11-09" means re-checked in code for this page.

### Closed since the 7 October review (kept for traceability)

| Former item | Closed by | Evidence |
|---|---|---|
| Review §7 item 26: `python -m raosim` discarded the CLI return code | `raosim/__main__.py` now calls `raise SystemExit(main())` | [`raosim/__main__.py`](../../raosim/__main__.py) (verified 11-09); `tests/test_cli_module_exit.py` |
| Review §7 item 7: `--burn-time` defaulted to 10 s and fed the MDO | Default is `None`; traditional pump sizing uses 10 s, MDO and requirements paths keep the `MissionSpec` 120 s | `raosim/run_nozzle.py:3142` (verified 11-09); [MDO issues §4](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md) |
| Review §7 item 16: no independent KKT check | `nlp.kkt_check` gates `success` and `workflow_status` | `raosim/mdo/nlp.py:452` (verified 11-09); MDO issues §4 |
| Review §7 item 15: no multi-start | `n_starts` with seeded scrambled Sobol starts | `raosim/mdo/nlp.py:404`, `:1135` (verified 11-09) |
| Review §7 item 27: one bad iterate aborts the NLP; Isp sweep all-or-nothing | `_GuardedProblem` backtracks or falls back; `pareto_frontier` isolates each floor and the CLI prints a FAILED row | `raosim/mdo/nlp.py:593`; MDO issues §4. The 225 s failure did not reproduce on 2026-10-16 |
| Review §7 item 28 (first half): two electric-drive constant sets | One registry, [`raosim/electric_technology.py`](../../raosim/electric_technology.py), read by both pipelines. The basis of the default set remains open (RW-S09-04) | MDO issues §4 row 1.3.19 |
| Review §7 item 21: contradictory status blocks in the MOC remediation document | The later STATUS block now states it supersedes the historical baseline | [MOC remediation](../subsystems/S06-rao-moc-bvp-contour/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md) head |
| Review §7 item 22 (part): roadmap listed R3 "not started"; `NEXT_SESSION_PROMPT.md` stale | R3 clarified 2026-10-15; prompt moved to [`../archive/`](../archive/README.md) | [ROADMAP_REQUIREMENTS_TO_PARTS.md](ROADMAP_REQUIREMENTS_TO_PARTS.md) status table |
| Review §7 item 25 (part): 0-byte `deep-research-report.md` at the root | Moved to `docs/archive/` in the 2026-11-05 cleanup | [`../archive/README.md`](../archive/README.md) |
| Review §10.4 item 8: standards as sources | 8 `StandardSource` records, HELD or BLOCKED_ON_STANDARD | [KG completion 2026-10-16](../../knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md) |
| Review §15.6: `D_pintle` ended exactly at its start value; Son transition row enforced on the slotted post | Pintle post mass in the objective; Son row applies only to the valve | MDO issues §0 |
| Delivery backlog D01, D02, D26 and parts of D03/D04 | September 9 input/execution integrity work | [INPUT_EXECUTION_INTEGRITY_2026-09-09.md](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md) |

### PRG: program, release and governance

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-PRG-01 | Settle the working tree: 18 modified `raosim/` modules plus untracked `raosim/rao_top_chart.py`, `raosim/thermal_correlations.py`, `raosim/data/catalog/`, `raosim/data/tools/` | No commit reproduces the CLI as it runs today. Committing `nozzle_geometry.py` without the untracked chart module breaks its import | `git status` (11-09); guide ch. 12 §"What this says about the working tree" | RW-S05-01 decision | P0 |
| RW-PRG-02 | Untrack and purge the 21 PDFs still tracked under `propulsion_texts/` | Copyrighted texts in history; release blocker. **Conflict:** README layout calls the corpus "gitignored" | `git ls-files propulsion_texts` = 21 (11-09); Review §7 item 1 p. 19, §13 p. 48 | Owner decision on purge timing (D-05) | P0 |
| RW-PRG-03 | Close public exposure while the project is personal: Review recommends making the repository private and deleting PyPI `lrekit` 0.1.0 and 0.1.1 | Review reports both public, PyPI files without a license and carrying the NASA-derived port without its agreement text. Not re-checked here. **Conflict:** README "Known limitations" item 10 says the first public PyPI release is pending | Review §7 items 2-3 pp. 19-20, §13 pp. 48-49; [README.md](../../README.md) | D-05 | P0 |
| RW-PRG-04 | License and NOSA 1.3 obligations: no `LICENSE`, no license field in `pyproject.toml`; `raosim/nasa_moc.py` lacks the agreement text, NASA copyright notice and dated modification record | Required before any publication; owner deferred the license choice to public release | Verified 11-09; Review §13 p. 48; [pypi_release.md](../architecture/pypi_release.md) step 1; guide ch. 10 §X04 | D-06 | P1 (P0 before publication) |
| RW-PRG-05 | Write `docs/CREDIBILITY_TARGET_V1.md`: the promoted path and the claims V1 makes | Phase P1 entry; scopes every later validation and provenance task | Review §8.1 P1 p. 23 (file does not exist, verified 11-09) | D-01 (V1 domain) | P1 |
| RW-PRG-06 | Export-control determination before Stage 2 outputs (process routes, tolerances, inspection plans) go to any collaborator, reviewer or the public; keep the manufacturing layer separable | Giving technical data to a foreign person can itself be an export; deferral fits personal use only while the data stays with the owner | Review §13 p. 49, §12 p. 47 | D-05 | P1 (gate for Stage 2) |
| RW-PRG-07 | Reconcile plan trackers and keep one release matrix (D25). Stale: [COMPLETION_PLAN.md](COMPLETION_PLAN.md) last updated 2026-08-26; [GENERALIZATION_PLAN](GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md) §6 item 1 says the CEA sampler run is outstanding while COMPLETION B1 records it done; [MDO_GUIDE](../subsystems/S13-engine-mdo-and-requirements/MDO_GUIDE.md) §9 item 13 says the chart tables are unchanged; [JAX_DIFFERENTIABLE_PLAN.md](JAX_DIFFERENTIABLE_PLAN.md) head is dated 2026-06-15; [MDO_REMEDIATION](../architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md) states schema 2, inputs 1.1.0 and 59 metrics against code 3, 1.2.0, 64 | Agents schedule work from these documents | Delivery backlog D25; guide ch. 9 §S14; this page | none | P1 |
| RW-PRG-08 | Scoped commits by concern; regenerated knowledge artifacts kept out of code commits | `6c04d92` added 206,266 lines around a 2,185-line MOC fix | Review §7 item 10 p. 20, §8.1 P0 p. 23 | D-08 | P1 |
| RW-PRG-09 | Credibility-release packaging: tagged release archived privately, then `CITATION.cff` and a persistent identifier at public release | Results must cite version, commit and configuration hash | Review §8.1 P6 p. 24, §13 p. 49 (`CITATION.cff` absent, verified 11-09) | G3 | P2 |
| RW-PRG-10 | External inputs: an external technical reviewer, a shop capability profile, CFD compute | Credibility sign-off, process registry and CHT/pump CFD cannot proceed without them | Review §12 pp. 46-48 | D-07, D-09 | P2 |

### S01: CLI entry points, argument parsing and dispatch

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S01-01 | Convert stage exceptions into typed component outcomes with a written `summary.json`: the chart `ValueError` in `_solve` and the unguarded `size_electric_pumps` call at `raosim/run_nozzle.py:6212` end in a traceback, exit 1 and no summary | Review's P0 containment step for the 500 N crash; delivery backlog D04 acceptance requires injected failures to preserve independent outputs | Verified 11-09; guide ch. 3 §S01 defects; Review §11.4 fix 1 p. 43; [INPUT_EXECUTION_INTEGRITY](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md) "Next work" | none | P0 |
| RW-S01-02 | One application service for flag runner, job verbs and API: `run_nozzle.main` holds parser, dispatch and pipeline in one function of more than 4,100 lines; the flag dialect does not compile to `EngineJobSpec` and uses different vocabulary (`--feed-architecture pump_fed` against `feed: electric_pump`) | Two contracts with different coverage; stages cannot be run, tested or failed in isolation | Guide ch. 3 §S01, §S02; readiness plan §4; delivery backlog D03, D24 | RW-S14-01 | P1 |
| RW-S01-03 | README and docstring drift: CLI section omits job verbs, `export-openfoam-spray` and the MDO flags; Outputs section describes `builds/vNNN_*` folders the runner never writes; bare `lrekit` shows a menu first; test status dated 11 July 2026 (1246 passed); `main.py` docstring flags; runner docstring "seed" examples | First thing a reader trusts | [DATAFLOW.md §10](../architecture/DATAFLOW.md) rows 1-3, 5, 8, 9, 11, 12 (open); [README.md](../../README.md) lines 93-125 | none | P1 |
| RW-S01-04 | Front-door defects: `-i --requirements` ignores menu choice 1; interactive `_prompt` casts to `float`, keeps the default on a parse failure and takes bare SI values | Silent input loss on the interactive path | Guide ch. 3 §S01 defects | RW-S02-01 | P2 |
| RW-S01-05 | Version drift: `pyproject.toml` 0.1.1 against `raosim.__version__ = "0.1.0"` | Release metadata disagree | Verified 11-09 | none | P2 |

### S02: inputs, units, validation and job contracts

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S02-01 | Extend the unit policy beyond the job dialect: flags, `@file.args` and both interviews take bare SI floats | The "1000x design error" `raosim/units.py` exists to prevent is still possible on the default command line | Guide ch. 3 §S02 defects | none | P1 |
| RW-S02-02 | Declare PyYAML: YAML is the documented job format but `pyproject.toml` does not list it | A clean install cannot read a documented job file | Verified 11-09 | none | P1 |
| RW-S02-03 | Unified jobs cannot select an electric technology set or the Son valve; `derived` pump-speed intent is refused | Job users cannot reach capabilities the flag path has | [MDO issues](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md) §4 last line; [INPUT_EXECUTION_INTEGRITY](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md) | RW-S09-04 | P1 |
| RW-S02-04 | Guided output is not a complete job: no `output_policy`, no run record, no `models` section, no screening opt-in; unreachable duplicate check in `_cmd_guided` | `lrekit guided --save F --run` writes only the job file | Guide ch. 3 §S02 defects | none | P2 |
| RW-S02-05 | Remaining request semantics: most target metrics remain post-solve checks (only the delivered-Isp floor reaches the optimizer); qualification duration has unknown coverage; additional architectures not implemented | Requirements read as enforced when they are only checked afterwards | [INPUT_EXECUTION_INTEGRITY](../audits/INPUT_EXECUTION_INTEGRITY_2026-09-09.md) table | RW-S13-05 | P2 |
| RW-S02-06 | Manufacturing limits are advisory and partial: notes only; aspect-ratio limit not checked against the search box; "no sharp changes" not encoded; Wadel AIAA number unfilled | Manufacturing screens look stronger than they are | Guide ch. 3 §S02 defects | RW-S11-06 | P2 |
| RW-S02-07 | Component data bindings in the job contract (`component_models`, `evidence_inputs`, `simulation_assessment`) need a versioned schema migration; the parser rejects unknown keys | Required before supplied data can affect a design run | Plan (3 Nov) §6 p. 10; [COMPONENT_DATA plan](COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md) §7 | RW-S14-02 | P2 |

### S03: propellant identity and chamber thermochemistry

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S03-01 | Built-in constant-gamma table ignores the requested O/F and deviates from RocketCEA: gamma 6-9 % high for every pair, LOX/LH2 molar mass 22.7 % low, Tc 7-9 % low for three pairs; no warning | Absolute constant-gamma numbers, especially LOX/LH2, are indicative only | Guide ch. 4 §S03 (check run 5 Nov) | none | P1 |
| RW-S03-02 | RocketCEA adapter defects: an efficiency override discards the RocketCEA c* (`run_nozzle.py:4213-4223`); `cea_frozen` labels a call RocketCEA makes with `frozen=0`; card names not mapped (`LCH4` fails and falls back with a warning only); CEA provenance not written to `summary.json` | Thermochemistry provenance and labels misstate what ran | Guide ch. 4 §S03 items 3-5, 7 | none | P1 |
| RW-S03-03 | Efficiency inputs: no domain check on `--eta-cstar`/`--eta-cf` (0 raises an uncaught `ZeroDivisionError`, >1 accepted); supplied efficiencies shown beside delivered performance without a supplied-versus-predicted label. Merged from three sources | A supplied coefficient reads as a prediction | Guide ch. 4 §S03 item 8; Review §7 item 24 p. 21; [INTEGRATION_PROGRAM](INTEGRATION_PROGRAM_2026-08-22.md) Phase 0c | none | P1 |
| RW-S03-04 | N2O/Ethanol: MDO catalog blocked pending the `N2O` versus `N2O_nbp` oxidizer-state choice, while the traditional CEA path accepts the pair | Same request, two behaviours | [CEA_H1_HOST_EVIDENCE](../audits/CEA_H1_HOST_EVIDENCE_2026-08-22.md) item 7; guide ch. 4 item 6 | D-11 | P2 |
| RW-S03-05 | Frozen variable-cp path: five promotion gates open (configuration-controlled thermochemistry fixture, property-grid refinement, published benchmarks, profile-aware thermal/BL/Cd, CFD plus test); no in-repo property data; `manufactured_composition` tables accepted for engine runs without warning | Software-verified only; must not be read as property evidence | [FROZEN_VARIABLE_CP_EXPANSION.md](../subsystems/S04-performance-and-off-design/FROZEN_VARIABLE_CP_EXPANSION.md) "Evidence required"; README remaining work 5 | none | P2 |
| RW-S03-06 | Independent re-evaluation of the final optimum on authoritative backends (D3, R5b), plus Cantera as an independent implementation (V1 evidence) | Thermochemistry is a P0-priority subsystem in the Review's coverage table | [COMPLETION_PLAN](COMPLETION_PLAN.md) D3; [ROADMAP_REQUIREMENTS_TO_PARTS](ROADMAP_REQUIREMENTS_TO_PARTS.md) R5b; Review §9.5 p. 31 | RW-S14-02 | P1 |
| RW-S03-07 | Equilibrium-chemistry option (F4) and RQ4 frozen-versus-equilibrium studies | Deferred until the credibility release | COMPLETION F4, E5; Review §8.4 p. 27 | G3 | P2 |
| RW-S03-08 | Predicted combustion efficiency: c* efficiency is an input; the opt-in spray loop is a screening knob; the 13 kN atomization screen predicts about 0.577 against the assumed 0.975 | Delivered thrust rests on an assumed efficiency | INTEGRATION_PROGRAM §2.2; [MDO issues](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md) item 13; guide ch. 12 §injector; Review §8.4 (follows M4) | RW-S14-02, literature L7 | P2 |

### S04: throat sizing, performance and off-design screens

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S04-01 | Thrust closure is blind to the contour: `Cf = eta_CF * Cf_ideal`; Rt is sized from the quasi-1-D Cf | Contour changes cannot move thrust in the optimizer | Review §7 item 8 p. 20; MDO issues item 9 (`raosim/mdo/engine.py:167`, `:319`); guide ch. 4 §S04 | RW-S06-06 | P1 |
| RW-S04-02 | The traditional run records no separation verdict: the screen exists only as a PNG at four fixed ambients, not the design ambient; `separation_margin` checks the raw flag | An off-design risk is drawn but never recorded | Guide ch. 4 §S04 item 2 | none | P1 |
| RW-S04-03 | Separation beyond FSS onset (F3 classifier); side loads stay unavailable (F8); registry entry cites Kalt-Badal/Schmucker and `hagemann1998.pdf`, not the four Ostlund forms implemented | TOP nozzles can show restricted shock separation that an onset ratio cannot represent | [COMPLETION_PLAN](COMPLETION_PLAN.md) F3, F8; [PHYSICS_FIDELITY_ROADMAP](PHYSICS_FIDELITY_ROADMAP.md) §3; guide ch. 4 item 3 | G3 | P2 |
| RW-S04-04 | Boundary-layer displacement is computed once and never fed back; throat Reynolds number is not computed; add the SP-8120 small-nozzle regime gate (F1a) before the displacement loop (F1) | SP-8120 forbids an inviscid core with a perturbation correction for small nozzles | INTEGRATION_PROGRAM §0 and Phases 0a, 2; COMPLETION F1 | none | P1 |
| RW-S04-05 | Silent numerical fallbacks: `mach_from_area_ratio` returns its last iterate without a convergence flag; `wall_pressure_distribution` replaces a failed inversion with M = 1 | Failed numerics become plausible numbers | Guide ch. 4 §S04 item 5 | none | P1 |
| RW-S04-06 | Provenance gaps: `atmosphere.py`, the wall-pressure screen, the built-in propellant table and the efficiency-split convention have no `ModelProvenance` entry; `gas_dynamics.py` cites an unidentified document | Unregistered models escape provenance closure | Guide ch. 4 §S04 item 7 | RW-X06-03 | P1 |
| RW-S04-07 | Atmosphere: one lapse rate over 20-47 km gives a 27 K step at 47 km; `requirements._ambient_pressure` accepts any altitude while jobs stop at 32 km | Altitude requests above 20 km carry a model defect | Guide ch. 4 §S04 item 6 | none | P2 |
| RW-S04-08 | Design ambient is a ratio (`--pa-over-p0 0.01`), so the default run is not a sea-level design and changing Pc silently changes Pa | Default runs answer a different question than users assume | Guide ch. 4 §S04 item 4 | none | P2 |

### S05: default Bezier/TOP contour, throat and chamber geometry

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S05-01 | Chart-domain change in the working tree: `lookup_angles` now raises outside eps 5-50 and 60-100 % length (SP-125 Fig. 4-14 re-digitization, `physical_validation: NOT_ESTABLISHED`). The shipped `test_13kn_sealevel_regen_allstl.args` and `test_13kn_movable_pintle.args` (eps 4.524) fail with a traceback; `--allow-chart-extrapolation` and `rao_chart_extrapolated` are unreachable; the documented MDO optimum at eps 4.0 lies outside the new domain. **Conflict:** MDO_GUIDE §9 item 13 says the tables are unchanged pending an owner decision | The default trusted path cannot run its own shipped benchmark | Verified 11-09 (`raosim/nozzle_geometry.py:53-68`, `raosim/rao_top_chart.py`); guide ch. 12 §"Attempt 1"; guide ch. 5 §S05; DATAFLOW §10 row 10 | D-02 | P0 |
| RW-S05-02 | Re-baseline everything computed with the superseded angles: archived Bezier builds, the TEST_PROMPT snapshot, the 13 kN MDO optimum (26.227 kg at eps 4.0 on 2026-10-16), test premises quoting 30/15.5 deg, the B5 seed identifier | References no longer describe the code | Guide ch. 12, ch. 10 §X01, ch. 11 §X05; [MDO issues](../audits/MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md) §4 | RW-S05-01 | P1 |
| RW-S05-03 | Label and source inconsistency: design status `preliminary_top_geometry` and registry level `benchmark_screened_not_hardware_validated` against the reliability `benchmark_validated` set by `_TrustedTopSolution`; registry cites SP-8120 while the chart is digitized from SP-125 | Maturity vocabulary must not overstate | Guide ch. 5 §S05; [latest_report](../../knowledge/reports/latest_report.md) (`nozzle.rao_top_bezier` PARTIALLY_SUPPORTED) | none | P1 |
| RW-S05-04 | Vacuous wall-offset gate on the CLI: `full_engine_contour` calls the geometry checks without `offset_distance` (`"offset_checked": false`) | A self-intersection check passes without running | Guide ch. 5 §S05 | none | P1 |
| RW-S05-05 | One cone-length helper: only the CLI Bezier uses the rounded-throat reference cone; six other modules use the sharp-corner length (0.62 % at eps 10, 1.2 % at 4.524) | Bezier and BVP walls end at different stations for the same flags | Guide ch. 5 §S05 | none | P1 |
| RW-S05-06 | Hall Cd is reported but not applied: `engine.py` has no discharge coefficient; `--cd-target` changes only Ru | Throat sized on geometric area | Guide ch. 5 §S05 | none | P2 |
| RW-S05-07 | Generic chamber defaults (L* = 1 m, contraction 2.5, 45 deg) and no Tier-2 shoulder-radius optimization; refuse the L*/contraction placeholder for hardware-oriented jobs | Placeholders flow into hardware geometry | Guide ch. 5 §S05; INTEGRATION_PROGRAM Phase 0b; [shoulder_radius_design_basis.md](../subsystems/S05-nozzle-and-chamber-geometry/shoulder_radius_design_basis.md) | none | P2 |
| RW-S05-08 | The chart interpolation is piecewise bilinear, not C1; derivatives switch at knots where the MDO optimum sat | Gradient-based optimum can sit on an interpolation artefact | MDO issues item 7 | RW-S05-01 | P2 |

### S06: Rao variational / MOC BVP contour

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S06-01 | The promoted TOP/Bezier wall carries an internal compression coalescence (`rrc_coalescence`); a shock-free MOC cannot validate it, so it needs a shock-capturing Euler/RANS computation. The quoted location (about 20 mm past N) is stale under the new chart (55.3 and 51.6 mm, outside the test band) | The default contour is not Rao's shock-free contour | [MOC remediation](../subsystems/S06-rao-moc-bvp-contour/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md); Review §7 item 8, §9.6 p. 33; guide ch. 5 §S06 | RW-X02-01 | P1 |
| RW-S06-02 | B3 qualification: coupled-crossing audit open; qualification outside the one fixed reference context open | TOP-versus-variational discrepancies cannot be interpreted | [COMPLETION_PLAN](COMPLETION_PLAN.md) B3; [RAO_B3_B5_HOST_EVIDENCE](../audits/RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md); MDO_GUIDE §9 item 11 | none | P1 |
| RW-S06-03 | B5 domain map is one slice (gamma 1.4, pa/p0 0.01, Rt 20 mm; 25 of 65 knots not converged); other gamma, ambient and full-BVP slices are separate evidence; `RAO_CHART_THETA_SEED_SOURCE_ID` still reads `rao-top-chart:v1` and the B5 eps axis starts at 4, so regeneration would raise | Declared domain cannot be claimed outside the slice | COMPLETION B5; guide ch. 5 §S06, ch. 11 §X05 | RW-S05-01 | P1 |
| RW-S06-04 | MOC Phase 3/4 items: `_construct_wall_from_ce` hard-codes `'area_ratio'`; `build_kernel` cannot build the NASA-visible line above 101 points; no M*/M convention flag (F11); `approximate_starting_line` accepts `sauer_modified` (F10); dead `_hall_throat_line` and `calc_arc_wall_point`; JAX interior point still clamps; no JAX bounded march; full-net promotion of `nozzle.moc` awaits a second oracle (M4Perfect) | Trusted subset stays one oracle case | MOC remediation STATUS "Open (Phase 3/4)"; Review §7 item 23 p. 21; guide ch. 5 §S06 | literature L12 | P2 |
| RW-S06-05 | `raosim/moc_bounded.py` imports 12 underscore-private functions from `raosim/nasa_moc.py` | Implicit internal API between production modules | Review §7 item 20 p. 20 (not re-counted here) | none | P2 |
| RW-S06-06 | No exact Rao provider in the MDO loop (design checkpoint only); `rao-bvp` refuses manufacturing CAD in MDO export (`run_nozzle.py:1877-1882`); the exported BVP wall meets the throat through a cubic bridge; below eps 5 the BVP chart angles fall back to the theta_B guess | The contour that is checked is not the contour that can be built | [IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md](../architecture/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md); MDO issues §2.5; guide ch. 5 §S06 | RW-S06-02 | P2 |
| RW-S06-07 | Total design derivatives (J6 v2 / F7): sensitivities stop at the control surface; frozen start line; live theta_B solve not routine | Gradients do not reach manufacturable wall coordinates | README remaining work 2-3; COMPLETION F7; guide ch. 10 §X01 | RW-S06-06 | P2 |
| RW-S06-08 | Rao-cliff track E: continuation (E2), sigma_min and lambda_min diagnostics with a mesh-convergence gate (E3), RQ3 (E4), RQ4 (E5) | Research track; the only track that can be cut without harming the tool | COMPLETION §7, risk 6 | G3, literature L13 | P2 |
| RW-S06-09 | Release-level promotion: independent review of the exact-variational chart sweep, published exact-versus-TOP deltas, evidence definition before `BENCHMARK_VALIDATED_AT_RELEASE` (False, `rao_variational.py:862`) could be enabled; no de-circularized J5 record exists | No contour is benchmark-validated by this repository | README remaining work 1; guide ch. 11 §X05 | RW-S05-02 | P2 |
| RW-S06-10 | REWRITE_PLAN Phase 8 deprecations absent; unvalidated `coupled`/`legacy` wall methods remain library defaults of `RaoSolverConfig` and `bell_nozzle_contour` | Library users get unvalidated defaults | Guide ch. 5 §S06; [REWRITE_PLAN.md](../subsystems/S06-rao-moc-bvp-contour/REWRITE_PLAN.md) | none | P2 |
| RW-S06-11 | Variable-gamma frozen MOC (F2); equilibrium reacting MOC deferred | Constant-gamma misplaces the Prandtl-Meyer turning | README limitation 2; COMPLETION F2; Review §8.4 | G3 | P2 |

### S07: regenerative wall sizing, cooling and wall structures

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S07-01 | Thermal verification ladder on the optimum: signed 1-D versus quasi-2-D land hot-spot delta (T1, D1, re-sited Phase 4b), then a Betti-style CHT pass as computational oracle (T3, D2) | Scorecard row "thermal verification: not run"; coking and wall-temperature rows are active at the optimum and will tighten | Review §8.3 p. 27, §9.6 p. 33; COMPLETION D1, D2 and risk 4 | RW-S14-01 | P1 |
| RW-S07-02 | Sieder-Tate form differs from its registry source: code 0.027 Re^0.8 Pr^(1/3) against SP-125 Eq. 4-23 C1 Re^0.8 Pr^0.4 (RP-1 C1 = 0.0214); code h_c 7-26 % higher; Sieder and Tate 1936 not held | Load-bearing coolant correlation is DIFFERENT_FORMULA | [KG completion](../../knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md) P5; [LITERATURE_REQUESTS.md](../literature/LITERATURE_REQUESTS.md) 2026-10-16 table | D-13, literature L9 | P1 |
| RW-S07-03 | Bartz: viscosity coefficient 1.1929e-7 against exact 1.184081e-7 (+0.745 %), two wrong SP-125 locators, validity limits not enforced; the untracked orphan `raosim/thermal_correlations.py` holds the exact constant and an applicability report | Small error, but the model reads "EQUIVALENT" only with these caveats | KG completion P5; guide ch. 10 §X08 | RW-X08-01 | P1 |
| RW-S07-04 | Curvature credit on the re-evaluation path: `regenerative_cooling_screen` applies Niino-Kumakawa enhancement by default (used by `design_nozzle_v2`, `--mdo-export`, `lrekit run`), which SP-8087 says not to credit without calibration | Optimistic wall temperatures on the authoritative path | Guide ch. 6 D1 | none | P1 |
| RW-S07-05 | The coupled march runs a fixed number of relaxed passes (12-30) with no residual computed or reported | Convergence is assumed | Guide ch. 6 D2 | none | P1 |
| RW-S07-06 | Unknown coolants silently become RP-1: `--coolant` has no choices and `resolve_coolant_properties` falls back to the RP-1 row (n2o, lox, jet-a get RP-1 density, cp, 700 K coking limit, 300 K inlet) | Silent substitution of physics | Guide ch. 6 D4 | none | P0 |
| RW-S07-07 | Silent conventions: spray loop ignores `--cooling-fraction`; with `--size-wall` any objective other than `min_dp` becomes `min_mass`; header-ring Reynolds number uses mu = 1e-3 Pa s for every coolant | Requested behaviour is replaced without notice | Guide ch. 6 D5, D8, D9 | none | P1 |
| RW-S07-08 | Film cooling is load-bearing (18 % at the 13 kN optimum) but has no film-injector hardware in either pipeline; nonzero-film thermal parity is unavailable (R5a); film-model sources (Hatch and Papell 1959; SP-8087 ref. 60) not held | The optimum depends on hardware that does not exist in the model | MDO issues item 16; MDO_GUIDE §9 item 8; ROADMAP R5a | literature L9 | P1 |
| RW-S07-09 | Liner buckling stays a post-solve gate because the implicit root is not robust at valid-box corners; hard enforcement needs SP-8068 (not held) | A structural limit is checked after optimization, not during it | COMPLETION A3, §10; MDO issues item 6 | literature L3 | P1 |
| RW-S07-10 | `regenerative_cooling_3d` divides half-pitch heat by full channel flow, halving the coolant temperature rise (65.6 K against 135.7 K); test-only, but a test may pass because of it | Defect hidden in a verification test | Guide ch. 6 D3 | none | P2 |
| RW-S07-11 | Jacket treatment inconsistent: `jacket_fos = 1.5` (no flag) against `closeout_structural_fos = 1.32` (SP-8087) and a thin-shell check the S07 sizer lacks; mass objective assumes jacket thickness = hot-wall thickness | Two paths size the same shell differently | Guide ch. 6 D7 | none | P2 |
| RW-S07-12 | Labels and figures: cross-section figure ignores the sized depth profile; eq. 4-29 margin uses 0.9 Sc without a source; mislabelled SP-125 equation numbers and dangling references | Reports and provenance disagree with the calculation | Guide ch. 6 D6, D10, D11 | none | P2 |
| RW-S07-13 | Low-cycle-fatigue life unavailable; single-point material values; no cyclic plasticity or creep; structural-life modelling deferred to after the credibility release | Static stress cannot satisfy a cyclic-life requirement | COMPLETION A4; Review §8.4; [regen_wall_model.md](../subsystems/S07-regen-cooling-and-wall-structures/regen_wall_model.md) | G3 | P2 |
| RW-S07-14 | Remaining thermofluid physics: HTD correlation with an enthalpy-coupled march; composition-driven radiation and soot; calibrated port/header losses; configurable turbulent correlations; cryogenic forced-flow CHF; validation against Carlile/Wadel/NASA data | Listed as missing physics by the provenance note | [thermofluid_literature_provenance.md](../literature/thermofluid_literature_provenance.md) "Remaining high-priority physics" | G3 | P2 |
| RW-S07-15 | CoolProp-sampled coolant property surface for the MDO (generalization item 9) and a live HTD constraint (item 3b); authoritative LOX/LCH4 and LOX/LH2 MDO runs stop at the coverage preflight | Methalox and hydrolox remain physics-unknown | [GENERALIZATION_PLAN](GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md) §7; MDO issues items 12, 2.10 | D-01 | P2 |

### S08: pintle injector and spray coupling

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S08-01 | Pintle pilot Stage 0: one component contract, data template and protocol; separate geometry adapters (conical seat valve versus Son radial gap; ASCEND uses the full cone angle); reconcile the test matrix (0.3/1/2 bar in the supplied plan against 0.3/1/3/4 bar in checked-in defaults) and the two cavitation-index conventions; locate the pilot data (the ASCEND experimental directory held no CSVs when reviewed) | First step of the owner's experimental-feedback goal; nothing downstream is defined until it exists | Plan (3 Nov) §4.1, §4.4, §8.1 pp. 6-8, 13; COMPONENT_DATA plan §5, §9 | D-04 | P1 |
| RW-S08-02 | Movable pintle: every CAD request fails closed; `to_design_input` and `--mdo-export` refuse `son_movable`; no swept assembly with stops, clearances, seals or guides; actuator dynamics, controller and feed compliance not modelled; the uncalibrated Cd fallback fails its gate by construction | An optimized, exported, calibrated movable component cannot yet be claimed | Guide ch. 7 §S08; [MOVABLE_PINTLE_MODEL.md](../subsystems/S08-injector-and-spray-coupling/MOVABLE_PINTLE_MODEL.md); MDO issues §2.7; Plan (3 Nov) §1.1 | RW-S14-01 | P1 |
| RW-S08-03 | Throttle map is outside any credible range (eta_c* about 0.06 at 50 % flow) and assumes a movable sleeve while the sized injector has fixed areas; the drawing labels fixed hardware with movable-pintle symbols | Misleading output in a default package | Guide ch. 12 §"Known defects" | none | P1 |
| RW-S08-04 | Spray coupling: the CLI never sets the `source` field, so only the legacy screen is reachable; default K is uncalibrated and often rejects the design; the Son sheet is inapplicable to the screen, so coupling and the movable architecture cannot combine | The c* fixed point cannot run on the architectures that matter | Guide ch. 7 §S08; [SPRAY_CSTAR_COUPLING.md](../subsystems/S08-injector-and-spray-coupling/SPRAY_CSTAR_COUPLING.md) | RW-S03-08 | P2 |
| RW-S08-05 | Two pump-power numbers in one summary (ledger with eta 0.7 against S09 meanline); deprecated `predicted_cstar_efficiency` alias still serialized; documentation drift in the argument-file comments and three stale statements in `injector_pintle_provenance.md` | Readers misread the outputs | Guide ch. 7 §S08 | none | P2 |
| RW-S08-06 | Injector film ring (L0), igniter (L0, no source in the corpus), combustion stability (response half unavailable; SP-8089 holds no n-tau equation); register stability as unavailable with the SP-8113 reason | Engine-article blockers that must stay visible | Review §11.1 p. 38; delivery backlog D21; INTEGRATION_PROGRAM Phase 0d; [latest_report](../../knowledge/reports/latest_report.md) (`injector.n_tau_stability`) | literature L7, L8 | P2 |

### S09: feed ledger, electric pumps and drive

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S09-01 | 500 N chain: compute specific speed and diameter from duty before any geometry and return OUTSIDE_VALIDITY (the 0.10-1.20 Ns screen runs after geometry, `pumps.py:3252`); bracket the speed instead of stepping (a 0.896 step jumped a window of about 23,600-25,750 rpm); size the inducer from its own suction criterion instead of the `0.45 * D2` floor (`pumps.py:2507`); all six constants in the chain lack a basis | The class the project started from cannot be sized; Review decision 5 says explain it before building more | Review §7 items 4, 19 pp. 19-20, §11.4 pp. 41-43; verified 11-09; containment is RW-S01-01 | RW-S01-01, D-03, literature L14 | P1 |
| RW-S09-02 | Pump B-rep summary fails whenever a split volute is built: `<role>_split_casing_layout` (no `valid` key) enters `diagnostics`; `run_nozzle.py:6275-6282` reads `info["valid"]`; the broad `except` records `pump_package: failed` and returns 2 after the files are written; no test covers it. This is not the oxidizer bolt-sector refusal of the 13 July sandbox, which its report records as fixed that day | A complete pump package is reported as failed; TEST_PROMPT step 2 cannot pass | Verified 11-09 (`pump_cad_brep.py:1490`); guide ch. 12 §"Known defects", ch. 8 §S11 | none | P0 |
| RW-S09-03 | Bolt-layout fit is not gated: `bolt_layout_fits_clear_arc` is computed (`pumps.py:1927`, `:1955`) but gates read only `bolt_screen_passed` (`pumps.py:3369`, `pump_cad_brep.py:915`); at 130 kN sizing reports feasible while the layout fails and a late export failure leaves partial STEP files | False feasible | Verified 11-09; Review §7 item 14 p. 20; [LREKIT_PUMP_AUDIT](../audits/LREKIT_PUMP_AUDIT_2026-09-05.md); delivery backlog D04 | none | P0 |
| RW-S09-04 | Electric-drive constants: one registry now, but the default `repository_screening_v1` lies outside held evidence; the page-verified `lee2021_ijass_electric_pump` set is selectable. The electric package was about 94 % of the 13 kN objective. **Conflict:** Review §15 (35.81 kg, two sets) is superseded by MDO issues §4 (26.227 kg, one set, eps 4.0), itself now out of the chart domain | The optimized mass is dominated by unsourced constants; the Review places them at the top of the credibility-debt P0 list | Review §15.6 p. 57; MDO issues §4 row 1.3.19; KG completion P2 (gap G4) | D-10 | P0 |
| RW-S09-05 | Technology-set validity: the 600 s burn warning fires only in `MissionSpec`; the traditional 10 s burn with `lee2021_ijass_electric_pump` applies the 600 s energy efficiency silently | Out-of-domain use without a flag | Guide ch. 7 §S09 | RW-S09-04 | P1 |
| RW-S09-06 | Pump inlet properties for a closed direct-regen fuel path come from the jacket-outlet state although the pump is upstream (no effect for constant-property RP-1; wrong for CoolProp coolants) | Wrong NPSH and flow for methane | Guide ch. 7 §S09 (code reading, no test) | none | P2 |
| RW-S09-07 | Fixed hydraulic assumptions (beta2 25 deg, psi 0.55, phi 0.08, efficiency cap 0.78, uncited loss constants) and the C1 efficiency surrogate in the MDO need sources | Pump efficiency and power rest on assumptions | Guide ch. 7 §S09; MDO issues item 14; [latest_report](../../knowledge/reports/latest_report.md) (`pump.meanline_coefficients` PARTIALLY_SUPPORTED) | RW-X06-03 | P2 |
| RW-S09-08 | Faithful 3-D pump geometry (M2: D09-D12, Track H): the impeller is still a planar camber ribbon extruded axially (`pump_cad_brep.py:106`, `:233`); collector items D1-D4 (machinability ceiling, process tip speed, constant-moment-of-momentum volute, flow-angle-matched vanes) not implemented | Owner's Stage 2 includes the large pump parts at L4 | Readiness plan M2; delivery backlog D09-D12; [PUMP_GEOMETRY_IMPLEMENTATION_PLAN](PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md); pump audit §1 | RW-S14-01, literature L14 | P2 |
| RW-S09-09 | Rotating assembly (D13, Track J): bearings and seals are mass placeholders (0.12 and 0.08 of shaft); no shaft retention, axial-thrust balance or rotordynamics | Blocked on SP-8048, SP-8101, SP-8121 and vendor data | INTEGRATION_PROGRAM §4; readiness plan §7; [PUMP_PHYSICS_AND_PROCESS_BASIS](../subsystems/S09-feed-system-and-electric-pumps/PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md) | literature L1 | P2 |
| RW-S09-10 | Required NPSH has no applicable correlation (margins null at 130 kN); velocity term deferred; cavitation not validated | NPSH margin stays unknown | Readiness plan §2; pump audit §1; [LITERATURE_AUDIT_2026-07-24](../audits/LITERATURE_AUDIT_2026-07-24.md) | literature L6 | P2 |
| RW-S09-11 | Feed architectures: `pressure_fed` end to end (not started; SP-125 Ch. V held); gas-generator and staged cycles (SP-8110, SP-8081 held, no turbine model); ox-versus-fuel efficiency split; low-specific-speed pump family for small engines | Requirements route supports only the electric pump (`raosim/requirements.py`) | GENERALIZATION_PLAN §4, §6 items 4, 6, 8; Review §11.3 item 5 p. 41 | D-03 | P2 |
| RW-S09-12 | Pump CAD default format stays `stl` until CI installs CadQuery | B-rep pump output is opt-in | [PUMP_CAD_IMPLEMENTATION_PLAN](../subsystems/S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md) "Deferred" | RW-X07-02 | P2 |

### S10: injector/chamber interface and mass ledger

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S10-01 | Interface material and factor of safety: both joint calls receive the liner `--material`; without `--bolt-allowable-stress` the bolt allowable is liner yield/1.5, an unlabelled proxy; `--structural-fos` is not passed | A bolt screen built on the wrong material | Guide ch. 8 §S10 | none | P1 |
| RW-S10-02 | `feasible` is true when every gate is `info`; two inner-edge definitions (bore against chamber OD); `design._injector_interface_screen` never uses the composite screen; CLI blocks catch every exception so a failure only drops a summary key | Feasible by default | Guide ch. 8 §S10 | none | P1 |
| RW-S10-03 | Whole-engine dry mass: the objective omits the injector body, faceplate and orifice plate and the flange and bolts (the interface hardware that dominates modelled mass, 18.1 kg at 13 kN per ROADMAP R2), and valves, lines and mounts; chamber mass from disjoint CAD partial (G2); configuration-controlled mass models (G3) and traditional pump BOM (G4) missing; `summary["wall_profile"]` omits `land_mass_kg` | The optimized quantity is a partial subtotal; close mass only as far as the canonical cases need | COMPLETION G2-G4; MDO issues item 20; [HARDWARE_MASS_LEDGER.md](../subsystems/S10-interface-and-mass-ledger/HARDWARE_MASS_LEDGER.md); Review §8.1 P5 | RW-S14-01 | P1 |
| RW-S10-04 | Feed-line losses: `line_dp_allowance = 0.0` and no valve model, so pump duty is understated | A known bias in pump power and battery mass | MDO issues item 15 (`raosim/mdo/schema.py:568`) | literature L2 | P1 |
| RW-S10-05 | Valves, lines, brackets and gimbal are not modelled; `total_engine_package_mass_kg` stays unavailable | Blocked on SP-8094/8097/8090/8080/8119/8123 | [PLAN_VALVES_LINES_BRACKETS_GIMBAL.md](PLAN_VALVES_LINES_BRACKETS_GIMBAL.md); delivery backlog D22 | literature L2 | P2 |
| RW-S10-06 | R2 flange/faceplate sizing is advisory only and not applied to exported CAD (faceplate 11.0 mm against 5.07 mm bending); emit a fastener callout (thread, grade, length, torque) | Sized bolts are not yet orderable bolts | ROADMAP R2, R4 item 1; guide ch. 10 §X08 (`flange_cad` callout) | RW-X08-01 | P2 |

### S11: CAD, mesh and geometry deliverables

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S11-01 | Analyzed wall is not the exported wall: the main path exports a uniform wall; the regen exporter should take the solved `RegenWallProfile` (D07, G2, Integration Phase 1d); MDO-analysed stations are not the exported wall | Review high-severity item; the definition of done requires agreement within a stated tolerance | Review §7 item 5 p. 19; MDO issues §2.2; delivery backlog D07 | RW-S14-01 | P0 |
| RW-S11-02 | Producer-validated artifacts: path-only records are unverified by design, but no typed exporter handoff (validity, units, part identity, configuration identity) exists; `execute._PART_ARTIFACT_KEYS` is empty, so unified-job STEP files stay unattributed and every package is partial | No part earns fidelity from a job | Verified 11-09 (`raosim/mdo/execute.py:398`); INPUT_EXECUTION_INTEGRITY; MDO issues §4 row 2.3 | RW-S14-01 | P1 |
| RW-S11-03 | Assembly defects: `export_engine_assembly` shifts each child to z = 0, so wall flange and jacket collide (3424 mm^3 in a synthetic probe); the CLI assembly list has no `regen_wall`; `test_13kn_movable_pintle.args` omits `--injector-cad none` and would hit `NotImplementedError` | Interference gates fail or are skipped on real packages | Guide ch. 8 §S11 | none | P1 |
| RW-S11-04 | Failure records and gates: a STEP failure exits without `summary.json`; coaxial separation audited on two idealized cylinders; pump interference recorded, not raised; the single-solid gate is promised but `_export_solid` checks validity and volume only | Failures and weak audits look like passes | Guide ch. 8 §S11; pump audit (`pump_cad_brep.py:1437`) | RW-S01-01 | P1 |
| RW-S11-05 | Expected-part graph by article scope (D06): 27 entries, 11 capped at L2, 4 at L1, 12 numerical only, none at L3 or L4; component articles must not demand igniter or gimbal | Completeness must be judged per article | Review §11 pp. 37-39; readiness plan §1; delivery backlog D06 | RW-S14-01 | P1 |
| RW-S11-06 | Process capability and inspection (Track I, D14): process registry keyed on solver quantities, B-rep interrogation, process outputs that feed physics (roughness, allowables), generated inspection requirements | Owner's Stage 2 C2 | Review §11.2 p. 40; [LREKIT_HARDWARE_PROGRAM](LREKIT_HARDWARE_PROGRAM_2026-09-05.md) Track I | D-07, RW-PRG-06 | P2 |
| RW-S11-07 | Tolerances and drawings (G5, D15): datum and feature-control proposals, tolerance-stack input contract, semantic PMI in STEP AP242 or a drawing set tested against the NIST PMI cases, sensitivity-based tolerance allocation from the MDO derivatives with a Monte Carlo stack | L4 needs tolerances, process and inspection; L5 stays unreachable | Review §11.2 p. 40; COMPLETION G5 (blocked on fastener selection) | literature L15 | P2 |
| RW-S11-08 | Stale messages and dead code: `_PUMP_STEP_UNAVAILABLE` says the B-rep path "has not landed yet"; `build_machined_pintle_bodies` has no caller; header drift in `export.py` and `regen_geometry.py` | Misleading messages | Guide ch. 8 §S11 | none | P2 |

### S12: figures, construction diagrams and animations

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S12-01 | Figure fidelity: `flowfield.png` and `anim_particles.gif` use the 3500 K default `args.chamber_temp`, never the thermochemistry Tc (3571 K in the fresh run); the structural dashboard's buckling panel is never drawn (no caller passes `buckling_profile`); the pintle schematic's claim that drawing, tables and solids agree does not cover the default machined STEP | Figures disagree with the numbers they illustrate | Guide ch. 8 §S12 | none | P2 |

### S13: differentiable MDO, optimizer and requirements

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S13-01 | Robustness matrix in CI: 500 N, 800 N, 13 kN and 130 kN cases, each required to end in a complete package or a typed refusal, never an exception | Review's P0 exit gate ("every robustness case ends in a written summary") | Review §8.1 P0 p. 23, §11.3 item 4 p. 41; readiness plan §8 | RW-S01-01, RW-S09-02, RW-S09-03 | P0 |
| RW-S13-02 | The MDO is opt-in and its geometry is not the CAD source: `--mdo-export` rebuilds the engine in `design_nozzle_v2` | Two pipelines produce the analysed and the exported engine | MDO issues §2.1-2.2 | RW-S14-01 | P1 |
| RW-S13-03 | Re-establish the 13 kN optimum under the new chart: it sat on the eps 4.0 chart edge, wall and post thickness floors, a zero-length barrel and blockage factor 0.90; eps 4.0 is now outside the chart domain | Bounds and coverage edges, not physics, set the design; the documented optimum is stale | MDO issues items 21, 1.1.7; guide ch. 9 §S13 | RW-S05-01, RW-S09-04 | P1 |
| RW-S13-04 | Requirements to parts (R3 remainder, R4): mass and envelope requirements refused as unavailable (`job.py:927`); the `pareto` workflow not wired (`execute.py:335`, `:394`); build package with BOM, materials, tolerances and gate report | The stated end state is "requirements in, parts out" | MDO issues §2.3; [ROADMAP_REQUIREMENTS_TO_PARTS](ROADMAP_REQUIREMENTS_TO_PARTS.md) R3, R4 | RW-S11-02 | P1 |
| RW-S13-05 | Multipoint mission (Phase 10, Track C1-C4): jobs refuse more than one operating point (`raosim/mdo/job.py:737`); throttle envelopes cannot be honoured | Required by the definition of done; Review orders it after P5 | COMPLETION §5; Review §7 item 17 p. 20, §8.2 p. 26; verified 11-09 | RW-S10-03, G3 | P2 |
| RW-S13-06 | Learned component models in the MDO: frozen evaluator with model identity, applicability and uncertainty in every result; baseline-versus-calibrated comparison; Cd that depends on sized geometry turns the area calculation implicit. The MDO injector Cd today comes from constant mission coefficients | Data must change the design for the right reason | Plan (3 Nov) §5 pp. 9-10, Stage 3 p. 13; COMPONENT_DATA plan §6 | RW-S08-01, RW-S14-02 | P2 |
| RW-S13-07 | Whole-engine domain map (Track L3) and an outer discrete loop over architecture choices; `job.ENUMERABLE_SELECTIONS` lists `channel_count` but nothing reads it | "Any thrust class" needs declared domains | Review §11.3 items 2-3 p. 41; guide ch. 9 §S14 | RW-S14-04 | P2 |
| RW-S13-08 | Fair solver benchmark (R5e): simultaneous differentiable MDO against FD NLP and block-coordinate on identical physics | No claim that the differentiable approach is better is supported until it exists | ROADMAP R5e | G3 | P2 |
| RW-S13-09 | Epigraph variables for the remaining hard extrema (R5c, gap 12.11) | Subgradient switching in the NLP | ROADMAP R5c | none | P2 |
| RW-S13-10 | Header drift (10-variable vector against 12 plus optional O/F; `mass.py` ratio-based closeout against hoop sizing; `constraints.py` eps range [4, 50]); two material mappers (`resolve_requirement` writes liner fields itself and records no material name or closeout) | Docstrings misdescribe the solver | Guide ch. 9 §S13 | none | P2 |

### S14: re-evaluation, provenance, release gates and run records

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-S14-01 | `ResolvedEngineConfiguration`: a Merkle composition of the existing input, injector, pintle and regen identities; `configuration_id` stamped on every artifact; composition across mismatched IDs refused; strict canonical digest (the `ResolvedEngineInputs.digest()` form is weaker than the injector and pintle identities). Does not exist (verified 11-09) | "Nothing in either track starts before P1": every validation delta and manufactured part must name its configuration | Review §7 items 5, 18 pp. 19-20, §8.1 P1 p. 23; delivery backlog D05; hardware program K0; INTEGRATION_PROGRAM Phase 1 | none | P0 |
| RW-S14-02 | One validation and comparison package (P3): case schema, metrics, uncertainty budget, preregistered policies, evidence grades V0-V3, observation operators; the same records take published experiments, CFD/FEA results and bench data; merges D17-D19, K1-K3 and the Plan (3 Nov) record types (component identity, dataset, measurement definition, model artifact, evidence requirement, simulation record) | One comparison layer serves all three owner stages | Review §8.1 P3 p. 24, §9.2 p. 28; Plan (3 Nov) §2 Table 1 p. 3, Stage 1 p. 13; readiness plan §7 | RW-S14-01, literature L5 | P1 |
| RW-S14-03 | Uncertainty as a first-class output: numerical, parameter and model-form components reported separately | Scorecard row "uncertainty: none" | Review §8.3 p. 27 | RW-S14-02 | P1 |
| RW-S14-04 | Typed applicability (Track L1): convert validity prose (95 envelopes) to machine-checkable bounds where a source states numbers, keep NO_VALIDITY_RANGE_LOCATED explicit; out-of-domain requests refuse with the blocking model named | Input Track L needs to refuse instead of crash | Review §10.4 item 4 p. 36, §11.3 item 1; hardware program L1-L2 | RW-X06-02 | P1 |
| RW-S14-05 | MET is effectively unreachable for jobs: `_model_consistency` returns unknown on any difference; the 2026-09-09 smoke recorded 17 differences and 79 missing comparisons; set field-specific parity tolerances and an adjudication rule (D5) and reconcile the two pipelines' thermal correlations and station geometry (D6) | A job is at best INDETERMINATE | Guide ch. 9 §S14; COMPLETION D5, D6; [MDO_REMEDIATION](../architecture/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md) §11 | none | P1 |
| RW-S14-06 | Release evidence is ignored on MDO paths: the MDO dispatch precedes the `--require-release-evidence` checks; `to_design_input` carries neither manifest nor `configuration_id` | The release report on MDO runs always shows the default blocked state | Guide ch. 9 §S14 | RW-S14-01 | P1 |
| RW-S14-07 | Registry coverage: none of the 61 entries covers an MDO surrogate (pump efficiency, film effectiveness, battery epigraph, HTD, closeout sizing); add verification level, validation datasets, domain, metrics and model-form error per entry | Surrogates escape provenance | Guide ch. 9 §S14; Review §9.1 p. 28 | RW-X06-02 | P1 |
| RW-S14-08 | Authoritative re-evaluation is another screening model; Phase 11 checks and the discrepancy-correction (trust-region) loop D4 are TODO | A delta beyond tolerance must flag the optimum, not be absorbed | COMPLETION D4, §6; guide ch. 9 §S14 | RW-S07-01, RW-S03-06 | P2 |
| RW-S14-09 | Two canonical case studies: Case A the LOX/RP-1 baseline, Case B a different in-domain scale; frozen bundles with configuration hashes, uncertainty and discrepancy ledgers, reproducible by one command | Phase P5 deliverable | Review §8.1 P5 p. 24 | RW-S14-01..03 | P2 |
| RW-S14-10 | Capability drift: `JOB_CAPABILITIES` names `raosim.mdo.feed` as the electric-pump implementer; no such module exists | Capability registry must point at callable code | Verified 11-09 (`raosim/mdo/capabilities.py:23`) | none | P2 |

### X01: differentiable JAX core

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X01-01 | Header drift and test-only capabilities: `__init__.py` calls JAX the default for `solve_rao_bvp` (not for a default CLI run); `thermal.py` names a test file that does not exist; stale line references; `rao_sensitivities` and `plot_sensitivity_field` only in tests; `solve_theta_b` has no CLI flag; plan direction "raosim/jax becomes the only core package" unmet | Documentation overstates what runs | Guide ch. 10 §X01 | none | P2 |

### X02: spray parcels and OpenFOAM handoff

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X02-01 | First simulation templates, each checked against a published case before use: shock-capturing compressible nozzle flow (against Back-Massier-Gier, TP-2725); internal hydraulic pintle profile with complementary pressure/flow condition and pressure-tap observables (Plan Stage 4); conjugate heat transfer (Betti-style case). Today `raosim/openfoam` has one template (external-gap water/air wedge, parameter-meshed, never executed, results not read back) | P4 oracle packs and the CFD round trip | Review §8.1 P4 p. 24, §11.5 pp. 43-45; Plan (3 Nov) §7 pp. 11-12; [OPENFOAM_SPRAY_EXPORT.md](../subsystems/S08-injector-and-spray-coupling/OPENFOAM_SPRAY_EXPORT.md); guide ch. 10 §X02 | RW-S14-02, D-09 | P1 |
| RW-X02-02 | Full simulation port (Stage 2 C3): mesh from the build-once CAD tagged with `configuration_id` (snappyHexMesh or Gmsh), solver-availability handling with UNAVAILABLE results, results read back as comparison records with mesh-convergence uncertainty, rotating pump flow, FEA adapter with a pinned code | Owner's simulated article | Review §11.5 infrastructure items 1-5 p. 45 | RW-X02-01, D-09 | P2 |
| RW-X02-03 | Spray cycle handoff cannot pass: two of nine gates are literal failures; `strict_target_benchmark` cannot pass with shipped fixtures; injector geometry (annulus plus slots or holes) is secondary-only for the parcel model | The handoff can never certify a case | Guide ch. 10 §X02; [LAGRANGIAN_SPRAY_MODEL.md](../subsystems/S08-injector-and-spray-coupling/LAGRANGIAN_SPRAY_MODEL.md) | none | P2 |

### X03: legacy toolbox CLI

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X03-01 | Tests write into the real `builds/`: `build_log.BUILDS_DIR` is hard-coded and `tests/test_design_workflow.py` runs `main.py` in the working directory (v455-v463 appeared on 2026-11-05) | Test runs pollute the build tree | Guide ch. 10 §X03, ch. 11 §X07, ch. 12 | none | P1 |
| RW-X03-02 | Legacy `rao` output is non-physical (exit angle 88.16 deg) and only weakly flagged; legacy `moc` reports NaN exit metrics; make both fail closed or remove them from `--method` | Non-physical output passes every gate except `method_status` | Guide ch. 10 §X03, ch. 12 | D-12 | P1 |
| RW-X03-03 | Build reproducibility: no build stores its command line; `_next_version` stops counting at v1000 | Builds cannot be reproduced | Guide ch. 10 §X03, ch. 12 | none | P2 |
| RW-X03-04 | `main.py` still packaged; its docstring and epilog advertise flags that `lrekit` no longer accepts | Legacy surface drifts | Review §7 item 25; DATAFLOW §10 row 8 | D-12 | P2 |

### X04: vendored NASA/JHU reference

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X04-01 | Generalize the reference port: `CropNozzleToLength` not ported for exported interior grids; historical `TT'.out` generator provenance unresolved; one certified case (M3.5Perf, gamma 1.4); the topology map predates `moc_bounded.py`. NOSA notices are RW-PRG-04 | The oracle certifies one configuration | README remaining work 4; [nasa_tt_prime_provenance.md](../subsystems/S06-rao-moc-bvp-contour/nasa_tt_prime_provenance.md); guide ch. 10 §X04 | none | P2 |

### X05: research scripts and offline generators

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X05-01 | Stale records and narrow products: archived J4/J5 outputs predate the corrections their headers describe; scripts quote 30/15.5 deg chart angles; CEA tables cover Pc 1.5-6 MPa only; `generate_contour.py` writes an open surface; `tvc_sizing_study.py` fixes Pc and writes to `/tmp`; scripts write into the archived `builds/` | Script evidence cannot be cited as current | Guide ch. 11 §X05 | RW-S05-02 | P2 |

### X06: knowledge system and source corpus

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X06-01 | Run `knowledge/kg.sh rebuild` after the docs reorganization: `repo_documents.json` lists 66 of 68 documents under old flat paths; capability records point at old paths; archived build CAD paths no longer resolve | Retrieval and provenance cite files that moved | Guide ch. 11 §X06; repository guide App. B "Follow-up actions" item 2 | owner (not run here) | P1 |
| RW-X06-02 | Model source reviews: 41 of 61 NOT_CHECKED (4 SUPPORTED, 15 PARTIALLY_SUPPORTED, 1 NOT_FOUND_IN_SOURCE). **Conflict:** Review counts 11 checked at 2026-09-20; the report of 2026-11-04 counts 20 | Scorecard target is 61 of 61 | [latest_report](../../knowledge/reports/latest_report.md) (2026-11-04); Review §8.3 | none | P1 |
| RW-X06-03 | Credibility debt: UNKNOWN_BASIS constants 257 in one section and 274 in the validation statistics of the same report (scope difference unstated); `credibility_debt_2026-10-15.json` lists 25 P0 and 216 provisional (its own caveat: "provisional review priorities, not numerical risk scores or measured optimizer leverage"); confirm whether keyword-derived `promotion_status`/basis labels are replaced by execution reach | Scorecard target: zero load-bearing UNKNOWN_BASIS on the promoted path | latest_report; [credibility_debt_2026-10-15.json](../../knowledge/reports/credibility_debt_2026-10-15.json); Review §7 item 11, §10.4 items 1-2 | RW-PRG-05 | P1 |
| RW-X06-04 | Corpus-backed knowledge tests skip whenever the vector index is absent (`knowledge/tests/test_kg_system.py:31`), so CI exercises only the corpus-free suite; add a synthetic mini-corpus fixture | "Knowledge pipeline exercised in CI" is a P0 exit gate in the Review | Verified 11-09; Review §7 item 9, §8.1 P0, §10.4 item 11 | D-08 | P1 |
| RW-X06-05 | Digitize the validation sources now held locally (TP-2725, TP-3576, TM-106985, Santoro-Pal, Back-Cuffel-Massier, GRCop-84 TM, CR-134806, Venters 2019) with recorded digitization uncertainty, as typed table and figure records | Held but not extracted; N2/N3 and T2 depend on them | File-name search of `propulsion_texts/` by NTRS id (11-09); Review §9.4 pp. 30-31, §10.4 item 10, §12 | RW-S14-02 | P1 |
| RW-X06-06 | Corpus conversion quality: 244 of 262 conversions `needs_review`; visually verify load-bearing pages of documents cited by promoted models, starting with low-confidence conversions | Retrieval can cite text that differs from the page | latest_report; Review §7 item 12, §10.4 item 12 | none | P2 |
| RW-X06-07 | Populate validation, configuration-bound and manufacturing objects (Experiment, Measurement, ComparisonRecord with signed deltas; EngineConfiguration and CADArtifact bound to `configuration_id`; ManufacturingProcess) and cross-source relations (CONTRADICTS, COMPETING_FORMULATION_WITH, APPROXIMATES) | Where the graph meets the configuration object and the comparison layer | Review §10.4 items 5-7, 9 p. 36; latest_report (17 EngineConfiguration, 70 CADArtifact records harvested from builds) | RW-S14-01, RW-S14-02 | P2 |
| RW-X06-08 | Knowledge documentation drift: CODE_MAP gives 0 lines for 54 knowledge files; `kg/cli.py` header omits six subcommands; `pipeline.py` header lists 10 of 20 steps; `AGENTS.md` names 3 of 20 tools; `NEXT_KNOWLEDGE_SESSION.md` queue counts stale; `thermal_correlations.py` cites a report that does not exist; `.venv/bin/python` commands name an environment that does not exist | Agents follow these instructions | Guide ch. 11 §X06; App. B follow-up item 6 | none | P2 |

### X07: test harness and CI

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X07-01 | The suite lags the working tree: the 5 November run ended with failures traced to the chart change (`test_bezier_chart_extrapolation_requires_diagnostic_override`, Bezier-chart tests in `test_nozzle_geometry.py`, `test_design_v2.py`, `test_mdo_engine.py`, `test_rao_chart_benchmark.py`). The current result is recorded by the editor in the guide (section "Test-suite status") | A red suite hides new regressions | Guide ch. 12 §"Attempt 1", ch. 11 §X07 | RW-S05-01 | P0 |
| RW-X07-02 | CI: two monolithic jobs (35 and 45 min) plus a knowledge job; the knowledge suite runs twice; the `cea` extra is never installed, so RocketCEA paths are untested; `publish.yml` can publish a tag the Test workflow has not passed; split into about six subsystem jobs | Green CI says little about subsystems; publication not gated on tests | [`.github/workflows/test.yml`](../../.github/workflows/test.yml) (verified 11-09); guide ch. 11 §X07; Review §7 item 13 | none | P1 |
| RW-X07-03 | Regression coverage of the shipped argument files: only the copper file is run by a test, and it passes BVP flags the Bezier path ignores, uses the repository-only Ru extension at the 7 MPa the sea-level header calls infeasible, and does not assert wall feasibility; the TEST_PROMPT baseline `builds/tests/13kn_sl_baseline_summary.json` has never existed and its `--max-nfev 4000` "full benchmark" is inert | The manual regression cannot detect drift | Guide ch. 11 §X07, ch. 12; [TEST_PROMPT.md](../agents/TEST_PROMPT.md) | RW-S05-02 | P1 |
| RW-X07-04 | Test hygiene: `test_jax_convergence.py` runs `max_nfev=4000` solves without the `slow` marker and its header contradicts its assertions; `test_transonic_kernel.py:195-196` cites a pre-reorganization path; `pytest.ini` has no `testpaths` | CI cost and stale references | Guide ch. 11 §X07; App. B follow-up item 3 | none | P2 |

### X08: library code not reachable from the CLI

| ID | Item | Why it matters | Evidence | Depends on | Pri |
|---|---|---|---|---|---|
| RW-X08-01 | Decide wire, keep or retire per module: `flange_cad` (wire the preload/torque callout, retire the solid); `thermal_correlations` (untracked, no test; wire report-only after adding a test); `trajectory` (keep as library, call it API-only); `jax/bvp`, `jax/pack`, `jax/residuals`, `jax/reference` (retire); `jax/design_opt` (keep until its two parity tests move). Also test-only: `regenerative_cooling_3d`, `build_machined_pintle_bodies`. Every `summary.json` lists the whole registry, including models the run never executed | Orphans carry stale claims (e.g. the 2026-08-25 audit's "every module imported and tested") | Guide ch. 10 §X08 table | D-12 | P2 |

### Conflicts between sources

| # | Topic | Source A | Source B | Reading |
|---|---|---|---|---|
| C1 | PyPI status | README limitation 10: first public release pending | Review §13: 0.1.0 and 0.1.1 public since 2026-06-29 | Not re-checked; README must change either way (RW-PRG-03) |
| C2 | Corpus tracking | README layout: `propulsion_texts/` is a gitignored corpus | `git ls-files`: 21 PDFs tracked | Code (git) wins (RW-PRG-02) |
| C3 | Rao/TOP chart | MDO_GUIDE §9 item 13 and MDO issues 1.1.7: tables unchanged pending owner decision | Working tree: `rao_top_chart.py` replaces them (uncommitted) | Working tree wins for behaviour; decision still needed on the domain below eps 5 (D-02) |
| C4 | 13 kN optimum | Review §15: 35.81 kg, two constant sets | MDO issues §4: 26.227 kg, one set, eps 4.0 | Both superseded: eps 4.0 is outside the new chart domain (RW-S13-03) |
| C5 | Model audits and unknown basis | Review: 11 of 61 checked, 223 UNKNOWN_BASIS (2026-09-20) | latest_report: 20 of 61 checked, 257 or 274 UNKNOWN_BASIS (2026-11-04) | Newer report governs (RW-X06-02, -03) |
| C6 | CEA sampler | GENERALIZATION_PLAN §6 item 1: sampler run outstanding | COMPLETION B1: done for four pairs; tables packaged | Code wins (RW-PRG-07) |
| C7 | Order of the pintle work | Review (7 Oct): water cold-flow pintle is Stage 3, after the simulated article | Plan (3 Nov) §1.2: an isolated pintle pilot can run in parallel now | Owner decision D-04 |
| C8 | Which pintle | Review decision 3: "a water cold-flow pintle" | Plan (3 Nov) §4.1: conical seat metering valve and Son radial-gap injector are different geometry families; engine sizes a fixed-slotted injector | Owner decision D-04 |
| C9 | NTRS identifiers | INTEGRATION_PROGRAM §4: SP-8048 `19710018568`, SP-8101 `19730012577`, SP-8121 `19780012205`, SP-8119 `19770009166`, SP-8080 `19730013806`, SP-8030 `19690023005` | LREKIT_LITERATURE_AUDIT: corrected to `19710018535`, `19740006328`, `19780022641`, `19770017247`, `19740002611`, `19710014805` | Use the corrected identifiers (section C) |
| C10 | Full benchmark contour | TEST_PROMPT: `--max-nfev 4000` gives the full benchmark contour | `run_nozzle._solve` Bezier branch ignores it | Code wins (RW-X07-03) |
| C11 | Pump B-rep | [PUMP_CAD_IMPLEMENTATION_PLAN](../subsystems/S09-feed-system-and-electric-pumps/PUMP_CAD_IMPLEMENTATION_PLAN.md) STATUS head: complete except the deferred default flip and optional NURBS | Pump audit and code: the impeller is a planar camber ribbon extruded axially | Both true at their scope: the reference B-rep exists; 3-D blades (M2) do not (RW-S09-08) |
| C12 | Module coverage | PROJECT_AUDIT_2026-08-25: every raosim module imported and tested | `thermal_correlations.py` and `jax/reference.py` have no test | Code wins (RW-X08-01) |

---

## C. Literature- and standards-blocked items

"Held" was checked on 2026-11-09 by searching `propulsion_texts/` file names for the NTRS identifier or title words; a
NOT HELD entry says nothing about whether the literature exists. Identifiers are the corrected ones from
[LREKIT_LITERATURE_AUDIT_2026-09-05.md](../audits/LREKIT_LITERATURE_AUDIT_2026-09-05.md).

| ID | Need | Sources | Held? | Unblocks |
|---|---|---|---|---|
| L1 | Rotating assembly | NASA SP-8048 bearings (19710018535), SP-8101 shafts and couplings (19740006328), SP-8121 shaft seals (19780022641), SP-8100 gears; vendor bearing, seal and motor data | Not held | RW-S09-09; delivery backlog D13 |
| L2 | Valves, lines, joints | SP-8094 (19740019163), SP-8097 (19740008129), SP-8090 (19730013807), SP-8080 (19740002611), SP-8119 (19770017247), SP-8123 (19780008146); RP-1228 fastener manual (19900009424) | Not held | RW-S10-04, RW-S10-05; D22 |
| L3 | Structures | SP-8068 buckling of plates (19710022285); SP-8083, SP-8019, SP-8007, SP-8040, Waters 1937 | Not held | RW-S07-09; flange and nozzle-collapse qualification |
| L4 | Transients and feed dynamics | SP-8030 transient loads (19710014805), SP-8112, SP-8055, SP-8072, SP-8088; NTRS 20040000363, 19700024650 | Not held | Integration Phase 5; D20, D23 |
| L5 | V&V and uncertainty methodology | JCGM 100:2008 (free); NASA-STD-7009B (free); ASME V&V 20-2009 (R2021, purchase); AIAA G-077-1998; Coleman and Steele; Kennedy and O'Hagan 2001 | Not held | RW-S14-02, RW-S14-03; D17-D19 |
| L6 | Test and measurement standards | ISO 9906:2012 pump acceptance; ISO 5167-1:2022 differential-pressure flow; a rocket test instrumentation reference (H14) | Not held | Pintle and pump benches (H1, H2); RW-S09-10 |
| L7 | Combustion efficiency and stability | ROCCID CR-187109/187110 (19910014917/8); JANNAF performance methodology; Bazarov and Yang 1998; pintle c*-efficiency data (a search task inside the held `propulsion_texts/pintle_injector/` folder first) | Not held (pintle folder held) | RW-S03-08, RW-S08-06 |
| L8 | Igniter | Tinker (20210000596); Aerospace 7(8):114, 2020 | Not held | RW-S08-06 |
| L9 | Thermal provenance | Sieder and Tate 1936; coolant-specific C1 for SP-125 Eq. 4-23; Hatch and Papell 1959 (NASA TN D-130); SP-8087 ref. 60 | Not held (SP-8087, Huber 2009 held) | RW-S07-02, RW-S07-08 |
| L10 | Chart provenance | Rao 1960, ARS Journal 30(6) | Not held | RW-S05-01, RW-S05-03 |
| L11 | Pintle blockage band primaries | Escher 1996 thesis; Ryu et al. 2017; Kang et al. 2020 | Not held | MDO blockage rows (RW-S13-03) |
| L12 | MOC start line | Kliegel and Levine 1969 | Not found by name | RW-S06-04 |
| L13 | Continuation | Keller 1977 | Not held | RW-S06-08 |
| L14 | Pump hydraulics and structure | Due 1966 (axial-thrust integrand); CR-72630 thrust balance; NACA Report 936 volute law; low-thrust pump CR; Katsanis TN D-2546 and D-5044; Zangeneh 1991; finite-tool flank milling; low-specific-speed (Barske-type) design texts; SP-8109 figures and Eqs. (20)-(21) digitized | Not held (SP-8109, SP-8052 held) | RW-S09-01, RW-S09-08, RW-S09-11 |
| L15 | Tolerancing and process | ISO 286; ISO 22081:2021 (replaces withdrawn ISO 2768-2); ASTM E606; selected fastener family standards; NIST PMI test cases; MMPDS (subscription); oxygen compatibility guide NASA/TM-2007-213740 | Not held (ASME Y14.5-2018, Y14.41-2003, Y14.24-2012 held, unindexed) | RW-S11-07, RW-S10-06 |

Held but not yet extracted, and therefore work rather than acquisition: the Review §9.4 validation sources (RW-X06-05),
SP-8110, SP-8081 and SP-8124 for other cycles (RW-S09-11), SP-125 Chapter V for pressure-fed feed (RW-S09-11), Huber
2009 for RP-1 properties (RW-S07-15).

---

## D. Validation and qualification path

**Evidence grades** (Review §9.1 p. 28): V0 mathematical verification against an exact or manufactured solution; V1
agreement with an independent implementation; V2 agreement with an independent physical measurement; V3 system-level
validation from one physical configuration. No model is V2 for LREKit hardware today; V3 waits for the M4 and M5
articles. CFD and FEA are computational evidence, never validation by themselves (Review §8.2 p. 26).

**Ladders to build first** (Review §9.6 p. 33):

| Ladder | Step | Data | Register |
|---|---|---|---|
| Nozzle | N1 keep NASA/JHU parity (Oracle B passes on the bounded march) | `Three-Dimensional-Nozzle-Design-Code-master/` | RW-S06-04 |
| Nozzle | N2 digitize the full Back-Cuffel-Massier transonic profiles | NTRS 19690063911 (held) | RW-X06-05 |
| Nozzle | N3 C_F and wall pressure against TP-2725 and TP-3576; shock-capturing Euler/RANS for the TOP internal shock | NTRS 19870015990, 19960042495 (held) | RW-S06-01, RW-X02-01 |
| Thermal | T0 exact conduction and fin cases | analytic | RW-S07-01 |
| Thermal | T1 1-D against quasi-2-D on the optimum, land hot spot as a signed delta | code exists (`physics.regenerative_cooling_2d`) | RW-S07-01 |
| Thermal | T2 coolant side against Meyer TM-106985; gas side against Back-Massier-Gier and Santoro-Pal | NTRS 19950024435, 19990025912 (held); Back-Massier-Gier 1964 not checked | RW-X06-05 |
| Thermal | T3 Betti-style CHT as a computational oracle | `betti2014.pdf` (held) | RW-S07-01, RW-X02-01 |
| Thermochemistry | Held-out CEA gate (done for four pairs); re-evaluate the final optimum; Cantera as V1 | packaged tables; RP-1311 (held) | RW-S03-06 |
| Materials | GRCop-84 and NARloy-Z properties | NASA/TM-2005-213566, CR-134806 (held) | RW-S07-13 |
| Injector | Water cold-flow Cd with held-out runs (Plan Stages 0-5) | pilot data to be located | RW-S08-01, RW-S14-02 |
| Pump | Water bench: head, flow, power, speed; NPSH where measured | none yet | RW-S09-10 |

**Simulation port** (owner's Stage 2): OpenFOAM Foundation 13 is already pinned by `raosim/openfoam`; meshing from CAD,
an FEA code and CFD compute are not yet chosen or wired (D-09). Each template is checked on a published case before it
judges new geometry (RW-X02-01, RW-X02-02).

**Release evidence** ([`raosim/release_readiness.py`](../../raosim/release_readiness.py)): the 17 requirements and the
activity that would produce each.

| Requirement | Produced by | State |
|---|---|---|
| engine.cfd_flow | Shock-capturing nozzle and chamber CFD (N3, C3) | Missing |
| engine.cht | CHT with mesh and time-step independence (T3, C3) | Missing |
| engine.structural_fea | FEA of liner, closeout and flange with CHT loads (C3) | Missing; FEA code not chosen |
| engine.material_allowables | Lot, process and temperature allowables | Missing; MMPDS not held |
| engine.drawings_gdt | Stage 2 C2 (RW-S11-07) | Missing |
| engine.manufacturing_review | Process route and inspection (RW-S11-06) with a shop profile | Missing |
| engine.proof_test | Proof and leak tests | Missing; facility needed |
| engine.cold_flow | Integrated cold flow (H1, H2) | Missing |
| engine.combustion_stability | Ignition, stability, acoustic response | Missing; blocked on L7, L8 |
| engine.hot_fire | Chamber and engine articles (H3, H4) under facility approvals | Missing |
| injector.void_connectivity | CAD fluid-volume and leak-path report | Missing as release evidence (a circuit-separation audit exists in the coaxial CAD) |
| injector.cold_flow_distribution | Water cold-flow pintle (H1) | Missing |
| regen.flow_distribution | Regen CFD or a calibrated manifold network | Missing |
| regen.hydroproof | Hydroproof and leak test | Missing |
| pump.fluid_volume | Pump fluid-volume CAD or CFD report | Missing |
| pump.rotordynamics | Rotordynamics, bearing and seal report | Missing; blocked on L1 |
| pump.performance_map | Pump water bench (H2) | Missing |

**Test-suite status.** The repository guide records the current full-suite result in Chapter 14, section "Test-suite
status" (filled in by the editor). The latest dated record before it is in
[RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md](../audits/RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md) (2183 passed, 26 deselected),
and README still quotes the 11 July 2026 run.

---

## E. Proposed roadmap: the next three milestones

**The owner's ordering** (quoted from the sources, not inferred):

- Review (7 Oct) §3, decisions recorded on 7 October (p. 7): "1. Order of work: credibility first, then CAD, then
  hardware. 2. A simulated article before any manufacturing ... Priority runs chamber and nozzle assembly first, then
  the injector, then the large pump components. 3. First physical article: a water cold-flow pintle. 4. Release:
  public eventually, personal use for now, with licensing and other legal work deferred. 5. The 500 N class must be
  explained before more is built on it."
- Plan (3 Nov), "The order of the work" (p. 1): first agree what the pintle geometry, measurements, uncertainty and
  acceptance question mean; then import real data, fit a model and show that it changes the engine calculation; next
  establish a small internal-flow CFD pipeline; after that compare experiment-only and fused models, test a proposed
  geometry, and extend the interface. §1.2 (p. 2): the plan "follows" the CLI, validation and component-export work
  for full engine integration; "in parallel, we can define the evidence schema and run an isolated pintle pilot."

**This page's inference** (not the owner's): three milestones inside Stage 1 that respect both orders. The pintle pilot
runs in parallel only as definition and data handling (Plan Stages 0-3); no hardware is built before Stage 3, which keeps
the Review's order. Decision D-04 can overturn this.

| Milestone | Scope (register IDs) | Exit evidence |
|---|---|---|
| **M-A Trustworthy baseline** (Review P0) | RW-PRG-01..03, RW-S05-01, RW-S01-01, RW-S09-02, RW-S09-03, RW-S07-06, RW-S13-01, RW-X07-01; then RW-S05-02, RW-X07-02, RW-X07-03, RW-X03-01, RW-S01-03, RW-S02-02, RW-X06-01, RW-X06-04 | Committed tree reproduces the CLI; full suite green; every robustness case (500 N, 800 N, 13 kN, 130 kN) ends in a written summary or a typed refusal; shipped examples run or are refused with a reason; README matches the code; 21 PDFs untracked |
| **M-B One identifiable, provenance-closed design** (Review P1 and P2; Plan Stages 0-1 in parallel) | RW-S14-01, RW-S11-01, RW-S13-02, RW-PRG-05, RW-S09-04, RW-S09-01, RW-S13-03, RW-S07-02, RW-S07-03, RW-S03-03, RW-S04-06, RW-S14-07, RW-X06-02, RW-X06-03, RW-S14-04; parallel: RW-S08-01, Stage 1 of RW-S14-02 | Every result and file names its `configuration_id`; analyzed and exported walls agree within a stated tolerance; V1 domain written; zero load-bearing UNKNOWN_BASIS on the promoted path; 13 kN optimum re-established inside the chart domain; 500 N answered by a refusal with numbers or a sized design; pintle contract, data template and one reproducible import |
| **M-C Comparison layer and first oracle evidence** (Review P3 and the start of P4; Plan Stages 2-4) | RW-S14-02, RW-S14-03, RW-S14-05, RW-X06-05, RW-S07-01, RW-S06-01, RW-X02-01, RW-S03-06; parallel: RW-S13-06 (Plan Stages 2-3) | One validation case runs in CI from manifest to signed-delta report; nozzle N2/N3 and thermal T1/T2 cases reported with signed deltas and uncertainty; first OpenFOAM template checked against a published case; pintle model fitted and shown to change one engine calculation |

After M-C, by the owner's order: P4 completion, P5 canonical cases (RW-S14-09), P6 credibility gate (RW-PRG-09), then
Stage 2 (C1 priority CAD, C2 process and tolerances, C3 simulation port, C4 simulated-article review) and Stage 3
(H1 water cold-flow pintle, H2 pump bench, H3 chamber article, H4 integrated engine). Multipoint (RW-S13-05) returns
after P5 (Review §8.2). Research (RW-S06-08, RW-S13-08) runs on the gated tool.

---

## F. Decisions the owner still has to make

The Review listed nine owner decisions (§14 pp. 49-52): two settled on 7 October, one partly settled, one deferred. The
table carries the seven that remain from that list and seven from other sources: five raised by later work (D-02,
D-04, D-10, D-12, D-13), one open since the CEA host evidence of August (D-11) and one from the Review's resource list
(D-14).

| # | Decision | Status | Options and recommendation on record | Blocks |
|---|---|---|---|---|
| D-01 | V1 declared domain (propellant pairs, thrust band, architecture) | Open (Review §14 row 3) | Review: one pair and one band around the 13 kN LOX/RP-1 baseline; a band reaching 500 N also needs D-03 | RW-PRG-05, RW-S07-15, RW-S14-04 |
| D-02 | Rao/TOP chart domain below eps 5 | New | Accept the SP-125 v2 re-digitization and move the shipped examples to eps 5 or above; or extend the chart from a second source (SP-8120 Fig. 5b differs by up to about 1.5 deg); or add explicit wall angles to the flag dialect. Re-baseline either way | RW-S05-01, RW-S05-02, RW-S13-03, RW-S06-03 |
| D-03 | Feed architectures and the 500 N class | Open (Review §14 row 6) | Pressure-fed through the requirements layer; lower chamber pressure for small engines; a sourced low-specific-speed pump family; or declare small classes out of domain | RW-S09-01, RW-S09-11 |
| D-04 | Which pintle is the first article, and when the data pilot runs | New (C7, C8) | Water cold-flow pintle (Review) against the conical metering-valve pilot (Plan 3 Nov) against the Son movable or fixed-slotted injector the engine sizes; pilot in parallel now (Plan) or after Stage 2 (Review) | RW-S08-01, RW-S13-06 |
| D-05 | Public scope and exposure | Partly decided: private for now | Review: make the repository private, delete both PyPI releases, purge PDFs before any public release; manufacturing layer separable pending the export-control determination | RW-PRG-02, -03, -06 |
| D-06 | License | Deferred to public release | NOSA 1.3 conditions on the ported code; separate module with its own license file | RW-PRG-04 |
| D-07 | Shop capability profile | Open (needed by Stage 2 C2) | The shop that will cut the first article, recorded as data with provenance | RW-S11-06, RW-PRG-10 |
| D-08 | Where generated knowledge artifacts live | Open | Review: keep reviews and provenance in git, regenerate the rest | RW-PRG-08, RW-X06-04 |
| D-09 | Simulation stack | Open | Pin one FEA code (CalculiX, code_aster or another) and its version; workstation or cloud for CFD | RW-X02-01, RW-X02-02 |
| D-10 | Default electric technology set | New | `repository_screening_v1` (outside held evidence) or the page-verified `lee2021_ijass_electric_pump` (600 s burn basis) | RW-S09-04, RW-S13-03 |
| D-11 | N2O/Ethanol oxidizer inlet state | Open since 2026-08-22 | RocketCEA `N2O` or `N2O_nbp` | RW-S03-04 |
| D-12 | Legacy and orphan code | New | Freeze `main.py` and make legacy `rao`/`moc` fail closed, or retire them; per-module X08 verdicts | RW-X03-02, RW-X03-04, RW-X08-01 |
| D-13 | Sieder-Tate form | New | Adopt SP-125 Eq. 4-23 with coolant-specific C1, or keep the classical form after acquiring Sieder and Tate 1936 | RW-S07-02 |
| D-14 | External reviewer and export-control route | Open | A faculty or industry reviewer; a university export-control office | RW-PRG-06, RW-PRG-10 |
