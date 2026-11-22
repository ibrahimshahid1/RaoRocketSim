# LREKit workflow, identity, and feedback audit — 2026-09-04

This audit inspected the working tree at commit `f763fe1` (2026-08-24), including the existing uncommitted nozzle/MOC work. It changes no source code. Source references below use working-tree line numbers. Generated `build/lib` copies are excluded: the authoritative implementation is `raosim/` and `lrekit/`.

## Assessment

The repository has made substantial progress toward a common input and analysis contract. It has not connected that contract to a complete artifact-producing workflow, a graphical interface, or a measurement-to-model return path. The next integration work must first repair accepted-but-unconsumed inputs. Otherwise a more polished interface will make promises that execution does not honor.

The user's earlier statement that configuration hashing and comparisons do not exist now needs qualification. `ResolvedEngineInputs` and its digest do exist; signed comparisons exist between two computational pipelines. Neither is the required identity of a manufactured article or a comparison against physical test measurements.

There is no defensible single percentage for completion: numerical capability, CAD fidelity, manufacturing readiness, assembly completeness, and physical evidence have different ceilings. The current common build contract explicitly caps most modeled metal at L2 reference CAD, electrical components at L1 envelopes, and numerous necessary components at L0 (`raosim/mdo/build.py:36`, `:67`). Completing software routes does not raise those physics and hardware ceilings.

## What has actually landed

| Capability | Evidence | Actual scope |
|---|---|---|
| Versioned, typed mission/job description | `raosim/mdo/job.py:280`, `:500` | Mission, architecture, models, fixed/free/bounded/enumerated variable declarations, targets, objective, operating points, output policy. Several declarations do not yet affect execution; see findings below. |
| Engineering-unit parsing and explicit unavailable quantities | `raosim/mdo/job.py:122`; `raosim/mdo/metrics.py:178` | Dimensional job inputs reject ambiguous bare numbers. Whole-engine dry mass, installed envelope, life, stability, restart and other unsupported claims are catalogued. |
| User-facing job verbs | `lrekit/cli.py:19`; `lrekit/job_cli.py:30` | `guided`, `validate`, `run`, `catalog`. Other arguments fall through to the traditional runner. |
| One resolved physical input contract | `raosim/mdo/resolved_inputs.py:245`, `:288`, `:293` | Propellant, flow split, thermochemistry/model identities, material, contour conventions, thermal, injector and feed inputs; content digest; fuel branch closure. This materially reduces MDO/traditional drift. |
| Common analysis execution and explicit verdicts | `raosim/mdo/execute.py:253`, `:423`; `raosim/input_validation.py:28` | Resolve, solve, reevaluate, compare constraints/targets; exit codes distinguish met, violation, invalid input, indeterminate, and solver failure. |
| Computational comparison layer | `raosim/mdo/snapshot.py:3414`, `:3477`, `:3554` | Signed and absolute scalar deltas, profile comparisons and unavailable states between MDO and traditional results. This is software/model parity, not experimental validation. |
| Expected-part ledger | `raosim/mdo/build.py:67`, `:182`, `:242` | Fixed expected part list, missing-part reasons, file hashes and limited fidelity metadata. Presently a library utility, not a production artifact orchestrator. |
| External release evidence | `raosim/release_readiness.py:24`, `:372`; `raosim/design.py:922` | Reviewed pass/fail records, local artifact integrity checks, optional expected configuration-ID agreement. These are useful evidence records; no measured value or uncertainty schema is present. |
| Some local geometry identity | `raosim/regen_cad.py:840`; `raosim/design.py:2041`, `:2645` | Regen profile hashes and injector mass/export geometry checks. These should become leaves of an engine configuration graph, not be discarded. |

Recent source history for these layers: `7120646` (August 10, MDO remediation/export/input validation), `4486a6f` (August 24, integration and fidelity gates), `f763fe1` (August 24, Rao postsolve validation/domain map). The existing dirty changes in `design.py` and `run_nozzle.py` primarily strengthen nozzle-domain/geometry restrictions; they do not close the workflow issues below.

## Findings that should become first-stage issues

### W01 — Accepted mission conditions do not all affect the solve

`MissionRequest.resolved()` parses altitude, optional Isp, dry-mass, envelope and qualification-duration requests (`job.py:293`). `resolve_engine_job()` only maps thrust, burn time and the special case `vacuum -> Pa=0` into the mission (`job.py:665`). A requested altitude is not converted to ambient pressure. A single `OperatingPointSpec` is accepted but not consumed; multiple points are explicitly rejected (`job.py:542`).

**Reproduced without a numerical solve:** both `thrust_condition='altitude', altitude='10 km'` and a single operating point with `ambient_pressure='0 Pa'` resolve to `mission.Pa == 101325.0`. The default sea-level case has the same value. No note identifies the ignored condition.

This affects physics directly: ambient pressure affects delivered thrust coefficient and the throat area sized from thrust. It is not merely missing report metadata. Mission-level `isp_min` is also distinct from `targets`: `_isp_floor()` only reads target records (`execute.py:241`). Optional mass/envelope requests must compile to an available target or be rejected with the existing unavailable-metric reason; parsing them is insufficient.

**Acceptance:** every accepted mission/point field maps to a documented solve input, target, evidence requirement or explicit unsupported error. Equivalent explicit ambient pressure and atmosphere-derived pressure produce equivalent resolved cases. No multi-condition job is represented by an undeclared single point.

### W02 — Optimization loses fixed and bounded design intent

Resolution records `fixed_variables`, clipped `bounds`, and `enumerations` (`job.py:735`). The optimize path calls `solve_min_mass(mission, isp_floor, objective=..., allow_incomplete_physics=..., contour_provider=...)` (`execute.py:347`), without those policies. `solve_min_mass()` constructs `default_design_space(mission)` (`nlp.py:207`, `:232`). Thus an optimized job's declared pin/bounds are not the optimizer's design space. Analyze jobs do use `job.design_vector()`.

**Acceptance:** compile a reduced design layout that substitutes fixed values, intersects bounded values with model domains, and expands discrete enumerations explicitly. Carry that exact layout through solve/state/postprocess. A requested pin is bitwise or tolerance-equivalent in output; every returned free value satisfies the actual requested bounds. Non-Isp target constraints either participate in optimization or the interface explicitly declares them postsolve checks. Unsupported target relations must never imply an optimizer constraint that does not exist.

### W03 — Architecture, schema and output contracts can be accepted without implementation

`ArchitectureSelection` accepts feed/injector/cooling strings (`job.py:335`), but the resolver only reads propellant and material fields. A pressure-fed/coaxial/ablative request was accepted and resolved with the ordinary pump/pintle/regen design variables, with no unsupported-architecture explanation.

`_build_spec()` allows top-level `schema_version` but never forwards it (`lrekit/job_cli.py:73`, `:121`). A supplied `UNSUPPORTED_V999` became the current default `1.1.0` in the audit probe.

`OutputPolicy` defines parts, CAD formats/fidelity, reports, directory and partial-package behavior (`job.py:488`). Repository source search found no execution consumer of these fields. The job CLI calls `execute_engine_job(spec)` (`job_cli.py:196`), whose default is `size_pumps=False` (`execute.py:256`); its `reevaluate()` call has no output directory or CAD settings (`execute.py:404`). The bridge defaults to `output_dir=None`, `cad='none'` (`postprocess.py:158`). Therefore a job's output request does not currently drive component CAD generation. `--out FILE` only writes `EngineJobResult.to_dict()`, a compact result rather than a part package (`job_cli.py:233`; `execute.py:110`).

**Acceptance:** a capability registry validates every architecture and output option before solve. Reject unknown schema versions. A requested report/package either appears with its manifest or returns an explicit stage failure. Make `allow_partial_package` govern actual artifact publication. Reject unsupported requested CAD fidelity instead of silently ignoring it.

### W04 — The build manifest is neither wired nor sufficient to certify its own labels

Production search found no calls to `build_engine_parts()` or `write_engine_build()` outside their definitions; all callers are in `tests/test_mdo_build.py`. `build_engine_parts()` takes paths to already-produced files rather than invoking exporters (`build.py:242`). It checks existence and hashes bytes (`:285`), then infers fidelity from requested fidelity and a static ceiling (`:297`). It does not parse CAD, consume an export/audit report, verify units, compare geometry identities or assign material/volume/mass.

**Reproduced:** a text file named `fake.step` containing `this is not a STEP file` is recorded as `built`, `L2_VALID_REFERENCE_CAD`. A manifest with only that one part gets overall L2, while `complete=False` correctly exposes missing parts. The calculation at `build.py:321` minimizes over built parts only; its note that the ceiling is set by the least-complete required part is misleading.

**Acceptance:** consume typed exporter results containing actual validation, identity, units, material, measured solid volume, artifact hashes and failure reason. Require positive evidence for each fidelity assertion. Calculate assembly readiness over applicable required parts; separately expose the best/lowest fidelity among produced artifacts. Generate the expected parts from the chosen architecture and test scope: a fixed whole-engine electric-pump list is unsuitable as the only contract for a component bench article or pressure-fed design. Distinguish manufactured parts, purchased components and reference envelopes.

### W05 — Failure persistence is inconsistent across entry points

The traditional runner writes summaries at several late branches (`run_nozzle.py:5892`, `:6023`, `:6657`); a pump sizing failure can still escape before those branches as described in the user's sweep. The MDO postprocessor already catches pump-sizing failures into `pump_error` (`postprocess.py:1090`), demonstrating an existing pattern to reuse. The job CLI catches execution exceptions but only prints and returns, before its optional result write (`job_cli.py:190` to `:213`).

**Acceptance:** allocate run identity and an initial manifest before computation. Update stage records atomically, including rejected inputs and solver exceptions. Export in staging directories and commit a package only after its policy is met; if diagnostic partial output is allowed, put `partial/failed`, per-part reasons and expected/actual artifact inventory next to the files. Preserve independent nozzle/chamber/injector results when pump sizing fails. Keep scientific verdict, requested-artifact completion and physical release readiness as separate fields.

### W06 — Two terminal front doors remain; the new interview overstates its coverage

Bare `lrekit` dispatches to the three-way traditional/MDO/MDO-optimize menu (`run_nozzle.py:1699`, `:3511`). `lrekit guided` uses a separate job interview (`guided.py:118`). No maintained graphical frontend or server application was found in repository source; the audit HTML file is a report, not an engine-design UI.

The guided interview offers “partial design” and “complete design point — analyze and generate artifacts” but never asks for the supplied partial/complete design or output settings. It collects mission, propellant/material, optional Isp and objective and then returns (`guided.py:175` to `:200`). A complete-point selection consequently analyzes reference defaults (with resolver notes), rather than a design entered in the interview. The objective list offers maximizing Isp (`guided.py:171`, `:194`) although objective validation/execution is minimization-only (`job.py:270`, `execute.py:333`). The preview describes a screening opt-in but the interview does not set it (`guided.py:227`).

**Acceptance:** one shared capability-driven form produces a saved `EngineJobSpec` for both terminal and future graphical clients. A partial design actually collects pins; a complete point exposes all required design values or marks explicit defaults for review; unavailable objectives cannot be selected; a preview accurately states achievable physics and artifact scope. Legacy argument invocations translate to the same job contract and keep compatibility tests.

## Identity and the experimental return path

`ResolvedEngineInputs.digest()` is valuable input identity, not final geometry identity. It does not contain a frozen collection of solved component geometries, placements, tolerances, process decisions, vendor selections, or the actual manufactured deviations. `EngineJobSpec.digest()` hashes request representation (including unit strings and output settings), so it should remain a request/replay identity rather than be mistaken for canonical physical identity. `configuration_id` in release evidence remains a caller-supplied string; local artifact integrity checks do not prove that the artifact belongs to the currently solved geometry.

Use distinct immutable records with explicit parent links:

1. **Request:** original user job and canonical SI interpretation, validation outcomes and explicit defaults.
2. **Resolved configuration:** final physical parameters, geometry definitions/frames, interfaces, per-part identities, selected model/coefficient versions, property-table hashes, software commit plus dirty-patch identity, solver settings, operating/load-case identities and manufacturing decisions. Compute a canonical content hash after resolution. Preserve component geometry hashes as children. CAD and analyses consume this configuration rather than independently rebuilding values from request flags.
3. **Build article:** configuration hash, part revision/serial, lot/material certificate, process route, inspection/as-built dimensions and deviations, assembly records and instrumentation. A machined deviation produces a linked as-built configuration; it does not silently rewrite the nominal design.
4. **Test run and measurement dataset:** article/configuration identity, raw-data hashes, sensor/channel identity, calibration certificate, measured quantity, units, physical station/frame, absolute-versus-gauge pressure convention, acquisition timebase, conditions, uncertainty model/covariance and quality flags. Keep this separate from `EvidenceRecord.passed`.
5. **Comparison:** selected matching operating window and sensor observation model, prediction version, measurement version, signed residual convention (`measurement - prediction`), absolute/relative residual where meaningful, and propagated uncertainty. Do not divide by a near-zero prediction or compare wall temperature at one station to a gas temperature elsewhere. Preserve time alignment and covariance for derived quantities.
6. **Calibration candidate:** explicitly declared adjustable coefficients, source/prior bounds, fitted values and uncertainties, identifiability diagnostics, training/held-out dataset lists, objective/weighting, validity envelope and improvement/regression report. Promotion creates a new coefficient version; it never mutates the baseline. Avoid using the same observations both to fit and to claim independent validation.

The existing computational snapshot delta types can be reused for serialization and display, but measurement alignment and uncertainty require a separate physical observation layer. `measured_part_volumes_m3` in injector CAD is a computed solid-volume measurement; it is not experimental engine data. `calibrate_wave_constants_from_vof` is simulation-derived breakup calibration, not a test ingest/calibrate workflow. `editable_reference_id` appears in pump BOM construction/serialization only (`pumps.py:602`, `:628`, `:2887`); no corresponding consumer was found.

Do not equate “pin geometry” with “pin a filename.” An imported or selected part needs units, coordinate system, topology/feature identity, inferred physical dimensions, provenance and compatibility checks. Re-solving around a pinned part must preserve it exactly and identify which remaining degrees of freedom are allowed to change. Candidate inconsistencies should produce explicit infeasibility or unknown outcomes rather than overwrite the pin.

## Proposed common entry point and interface

The current commands are `lrekit guided`, `lrekit validate JOB.yaml [--resolved-out FILE]`, `lrekit run JOB.yaml [--out FILE]`, and `lrekit catalog TOPIC`. Traditional options such as `--complete-package`, `--engine-mdo`, `--engine-mdo-optimize`, `--mdo-export` and `--cad` remain separate execution routes. Architecture enumeration exists through Python `execute_engine_job_set()` (`execute.py:557`), but the job CLI invokes only the single-job executor. Pareto jobs are explicitly rejected (`execute.py:273`).

A proposed command family, **not currently implemented**, is:

```text
lrekit job new / validate / run / inspect
lrekit component inspect / pin
lrekit build create / verify
lrekit test ingest / inspect / compare
lrekit calibrate fit / validate / promote
lrekit catalog models / components / processes / coefficients
lrekit ui
```

Names can change; the requirement is one application service and schema shared by terminal, Python API and UI. Preserve current public verbs as aliases. Full feature access should come from registries with capability, input, output, dependency, fidelity and provenance metadata rather than a manually duplicated menu.

The UI should present a saved project/run with five connected views: mission and pinned inputs; coverage and feasibility; solved components/assembly with per-part status and CAD preview; fabrication/inspection/evidence package; and test/comparison/calibration history. Display achieved fidelity and unknown quantities beside the values they qualify. Users should be able to inspect an independent successful component even when another component failed. A single large form containing hundreds of solver flags would not provide this workflow.

## Issue-ready sequence and release gates

| Stage | Work | Concrete completion test |
|---|---|---|
| S0 Contract integrity | W01–W03: accepted fields, optimization pins/bounds, unsupported schema/architecture/output rejection | Parameterized contract tests establish that every accepted field changes its intended resolved input or yields an explicit unsupported result; test actual optimizer handoff, not only resolver dictionaries. |
| S1 Persistent build execution | W04–W05: wire typed exporters, conditional part graph, atomic stage/artifact manifests | Inject one sizing failure and one late export failure; independent parts remain accessible and every file's package status is discoverable in its directory. Fake STEP cannot receive L2. |
| S2 Final configuration identity | Extend resolved inputs into final geometry/assembly identity and artifact consumers | Change any physical dimension/model coefficient/part transform and identity changes; changing only output directory does not change physical identity. Cross-configuration artifacts are refused. |
| S3 Measurement and comparison | Build article identity, raw-data ingest, sensor mapping, uncertainty and signed deltas | Synthetic known-offset and unit-conversion data recover expected residuals; wrong station/configuration/units are rejected; missing uncertainty is explicit. Real archived data remain unchanged. |
| S4 Geometry selection | Consume editable references and typed pinned part parameters | Pin an impeller/chamber feature, solve remaining variables, and demonstrate unchanged pinned geometry; incompatible requested performance yields reasoned infeasibility. |
| S5 Calibration and validation | Declared coefficient sets, fit/held-out comparisons, immutable promotion | Recover an identifiable synthetic coefficient, reject underdetermined fits, demonstrate held-out improvement within the declared envelope and reproducible rollback. |
| S6 Shared user access | Common application service, capability-driven terminal/UI, legacy adapters | The same saved job produces the same resolved/configuration identity and stage outcomes through CLI, API and UI; every offered feature has a callable route and preview. |
| S7 Test-article release integration | Link manufacturing, mechanical, instrumentation and external evidence results | Selected article's applicable parts are complete to required maturity; process/inspection packages and traceable evidence correspond to the as-built configuration. Numerical success alone cannot satisfy this gate. |

S2–S3 can proceed alongside pump 3-D geometry and mechanical development. The measurement plan must influence sensor accommodations before fabrication. Calibration should first target a narrow, identifiable component/quantity and a declared range rather than claim universal validity for arbitrary thrust.

## Verification performed

The repository `.venv` uses Python 3.14 and lacks JAX; the first test invocation failed at collection for that environment. Retried with the existing `.venv-jax` environment, without changing dependencies:

```text
.venv-jax/bin/python -m pytest -q \
  tests/test_lrekit_job_cli.py tests/test_lrekit_guided.py \
  tests/test_mdo_build.py tests/test_release_readiness.py \
  tests/test_mdo_job.py tests/test_mdo_resolved_inputs.py
84 passed in 41.63s
```

Read-only/runtime probes additionally demonstrated the altitude/single-point pressure loss, ignored architecture, discarded schema version and fake-STEP fidelity issues above. Those findings are not contradicted by passing existing tests: current tests cover accepted dictionaries, typical execution and manifest bookkeeping, but not these end-to-end input-consumption and artifact-validity properties. No tracked source or test was edited, and no full-engine hardware validity claim follows from this software test batch.
