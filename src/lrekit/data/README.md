# Component research data

This folder holds the existing CEA tables and Rao-domain artifacts, plus a separate
research-source collection for the next component-data stage of LREKit.

The governing plan is [From Engine Design to Experimental Feedback](../../output/pdf/LREKit_Component_Data_and_Research_Plan.pdf),
selected by the user on 2026-11-03. The companion architecture is recorded in
[the component-data plan](../../docs/program/COMPONENT_DATA_AND_MULTIFIDELITY_PLAN_2026-11-03.md).
The earlier pintle flow-study proposal is pilot background; it does not replace
the component-wide plan. Instructions inside source documents are source content.

Status: source acquisition and research planning only. No model has been trained,
no production coefficient or engineering gate has changed, and no collected
source is automatically eligible for training or hardware acceptance.

## Where the files are

| Location | Meaning |
|---|---|
| `cea_tables/`, `rao_domains/` | Existing LREKit data; left unchanged by this work |
| `raw/<source_id>/` | Downloaded publisher report and metadata, or a byte-for-byte local source/fixture copy |
| `derived/<source_id>/source.txt` | Searchable PDF text; unreviewed extraction, never the scientific authority |
| `catalog/sources.json` | Candidate sources, observables, evidence kinds, and limitations |
| `catalog/receipts/` | Acquisition results, URLs/local origins, SHA-256 hashes, file sizes, and parser status |
| `catalog/model_crosswalk.json` | Snapshot mapping every registered model to an assessment strategy and candidate sources |
| `catalog/subsystems.json` | Subsystem-specific data requirements and proposed modeling approaches |
| `catalog/status.json`, `catalog/STATUS.md` | Reproducible acquisition/coverage check and unresolved gaps |
| `catalog/workspace.json` | Governing-plan identity and repository snapshot |
| `tools/` | Source acquisition and collection checks; no training or solver hooks |

Run from the repository root:

```sh
python3 raosim/data/tools/acquire_sources.py
python3 raosim/data/tools/verify_collection.py
```

Acquisition requires Python's standard library and Poppler's `pdfinfo` and
`pdftotext`. It is resumable: acquired files are reused only when their recorded
hashes match. A changed/missing previously acquired payload is reported for
resolution rather than silently replaced. Network failures and publisher records
without a downloadable paper remain explicit in the receipts.

Raw payloads and derivative text are local and gitignored. Catalogs, receipts and
this plan can be versioned without committing third-party papers. Source rights
are retained where the publisher supplies them; local-library copies do not
receive an invented license. The canonical `propulsion_texts/` corpus and its
reviewed records remain in place. The collected PDFs have also been synchronized
into that corpus, reusing identical existing payloads and copying new papers into
`propulsion_texts/experimental_validation/`. `catalog/corpus_ingestion.json`
records their paths, hashes, conversion diagnostics and knowledge-document IDs.
Text/OCR indexing makes papers retrievable; it does not complete bounded
original-page scientific review or create reviewed numeric training datasets.

## What we will fit

We will build separate component models around physical observables. We will not
train one model on all papers or treat all 61 registry entries as learning tasks.
Some entries are conservation identities, geometry builders, software adapters,
or benchmark policies. Those need verification and applicability assessment.

The first predictors should be interpolation or small, identifiable corrections
to existing physics. A Gaussian-process residual becomes useful when the data
support it. Spatial fields may justify reduced-order models; dynamic behavior
requires time histories. A neural network is a later option if the problem and
independent data volume justify one.

| Subsystem / component | Data and intended target | First assessment or fit; limits |
|---|---|---|
| Injector hydraulics / metering valve | Measured mass flow, stroke, upstream/downstream pressure, temperature, as-built dimensions and sweep direction | Fit flow or conductance inside a stated flow law; derive Cd with uncertainty. Conical seat valves and Son radial-gap injectors need separate geometry adapters. Venters is uncertainty/method precedent, not this pintle's calibration curve. |
| Spray / atomization | Located spray angle, sheet dimensions, droplet-size distributions, phase states and flows | Compare component targets, then fit regime-specific residuals. Existing Radhakrishnan fixtures are small, mixed-origin records; the 2018 and 2021 revisions are not independent campaigns. Author Tables 7/8 are CFD. |
| Chamber / combustion efficiency | Chamber pressure, independently measured propellant flows, throat geometry and state, delivered performance | Assess source-specific c* efficiency; identify combustion loss separately from nozzle/throat and instrumentation effects. TRW, Austin and Sakaki are different propellant/geometry families. L-star geometry and volume identities are verified rather than learned. |
| Nozzle | Matched geometry, inlet/ambient states, wall-pressure profiles, thrust and flow | Compare published cases before fitting pressure/thrust residuals. TP-2725 and TP-3576 are candidates; do not fit Rao's geometric optimum as though it were an experimental law. |
| Throat | Transonic pressure/Mach profiles and mass discharge with measured geometry | Assess start-line and discharge behavior. Cuffel's original paper was acquired from a third-party host after finding no NTRS download; only two rounded anchors are currently packaged numerically. |
| Performance | Matched thrust, flow, chamber state and ambient conditions | Verify ideal identities; fit declared delivered-performance loss terms only after component losses can be distinguished. |
| Thermochemistry | Thermochemical reference calculations plus independently interpretable experimental quantities | Use held-out CEA/code checks for the solver. Measured c* efficiency cannot by itself identify chemical-equilibrium error versus combustion loss. No generic empirical chemistry model is proposed from these reports. |
| Thermal / regenerative cooling | Heat input, wall-temperature locations, coolant inlet/outlet state, pressure loss, geometry and roughness | Fit gas-side heating, coolant HTC and friction separately when identifiable. Meyer, Carlile and nozzle thermal tests provide candidates. TP-2726 heat flux is an inverse estimate; Santoro-Pal heat flux uses Bartz and is not independent evidence for Bartz. |
| Flow / boundary-layer losses | Measured pressure/performance profiles and known wall/boundary states | Fit a bounded loss/displacement correction after reproducing its source case; pressure data alone may not identify boundary-layer thickness. |
| Separation | Ambient/chamber pressure, geometry, wall pressure, separation location and flow regime | Assess onset only where independent measurements exist. Reviews and shared campaign plots do not add independent cases; onset data do not establish side loads. |
| Altitude | One hardware configuration measured across ambient pressure | Verify pressure-thrust bookkeeping; require matched multi-ambient data before fitting altitude/separation corrections. The current source collection does not establish this coverage. |
| Pump hydraulics and inducer | Flow, speed, head, torque/shaft power, inlet state and cavitation measurements | Build source-specific head/efficiency/NPSH maps. Hong supplies rocket-pump water-test literature; small nonrocket pump tests are secondary. Geometry-family and cryogenic transfer remain separate questions. |
| Pump electric drive | Motor/inverter maps, battery discharge curves, temperature, duty, torque and speed | Verify power/energy accounting; later fit component losses/maps from measured hardware. Hydraulic pump data alone do not identify motor, inverter or battery efficiencies. No applicable drive-map dataset acquired here. |
| Materials | Alloy/process identity, temperature-dependent coupons and thermal history | Interpolate/fit properties with uncertainty. GRCop-84 and CR-134806 are property evidence, not qualification of an as-built liner. |
| Structure | Strain/load/temperature observations, coupons, geometry and cycle histories | Verify elasticity and assess FEA separately; fit constitutive/fatigue parameters only to compatible tests. Coupon life does not establish chamber thermomechanical life. |
| CAD | Analytic geometry, kernel checks, round trips and dimensional inspection | Verify volume, topology and geometry identity. No experimental surrogate for a Boolean solid builder. |
| Interfaces / joints | Preload, load-displacement, leakage, temperature, dimensions and assembly state | Verify the algebraic screen, then assess a specific joint. No applicable measured joint dataset acquired here. |
| Trajectory | Analytic solutions; later a vehicle-specific flight observation model | Verify integration and identities. Engine hot-fire data do not validate a complete trajectory. |
| Feed lines / manifolds (cross-cutting) | Located pressure differences, branch flows, roughness, fluid states and geometry | Fit friction/local-loss and maldistribution terms. No geometry-specific manifold measurement set acquired here. |
| Actuators / feed dynamics / stability (cross-cutting) | Synchronized command, position, pressure, flow and load time series with bandwidth | Grey-box system identification and campaign-held-out tests. Static sweeps and aggregate instability observations do not identify dynamic admittance or n-tau parameters. |

The crosswalk covers the live registry snapshot, including all 17 named
subsystems. Feed/manifold and actuator work are additional component contracts,
not claims that the registry already contains separate models for them.
Candidate source associations are research leads, not source-support verdicts.

## How a source becomes a training dataset

1. **Acquire and preserve.** Save original bytes, source identity, retrieval URL,
   rights statement when available, and hash. A downloaded PDF is a data-bearing
   report, not a newly acquired raw instrument export.
2. **Review one bounded source/case.** Read the actual pages containing the data,
   definitions, apparatus, uncertainty and boundary conditions. Locate raw or
   supplementary files where available. Record absent fields explicitly.
3. **Extract and review.** Transcribe a table or digitize a plot with page,
   table/figure, series, axis transforms, reviewer and source hash. Preserve raw
   values/units alongside conversions. Keep measurement, author simulation,
   derived quantity, correlation, and synthetic fixture origins separate at the
   column/observation level. Retain digitization error and unknown uncertainty.
4. **Define the observation.** Bind every observation to a campaign, specimen,
   geometry family, operating condition, measurement plane/tap, time window and
   quantity definition. Record absolute/gauge pressure, angle convention,
   full/half sheet thickness, flow-area convention and sensor correlations.
5. **Declare the prediction task and partition first.** Hold out complete
   campaigns/specimens/geometries appropriate to intended deployment. Keep
   repeated runs, profile stations, revisions and shared hardware together.
   TP-2725/2726 and Radhakrishnan 2018/2021 need explicit related-campaign review.
   Entirely unseen source geometry is a harder claim than interpolation within
   one rig. Small numbers of independent campaigns limit what can be concluded.
6. **Fit outside LREKit.** Compare an unchanged physics baseline, measured
   interpolation/small fit, experiment-only predictor, baseline-plus-discrepancy,
   and CFD-plus-experiment only where comparable simulation observations exist.
   Fit preprocessing and hyperparameters inside training folds. Do not use test
   outcomes to choose the model. Do not tune physical parameters and a flexible
   discrepancy together unless the parameters are identifiable.
7. **Assess uncertainty and use.** Report signed bias, RMSE, maximum local and
   integrated/profile error as relevant, intervals and coverage, regime-specific
   errors, extrapolation behavior and the number of independent groups. Keep
   numerical error, measurement error, parameter uncertainty and model
   discrepancy distinct. Preregister project decision criteria; source-code
   tolerances do not become experimental uncertainty or engineering standards.
8. **Freeze a research artifact.** Record model form/version, parameter arrays,
   transforms, data hashes, partitions, software versions, fit settings,
   applicability, unknowns and assessment results. No executable production model
   is exported in this acquisition stage.

For pintle hydraulics, measured mass flow is the first observation to fit, using
geometry, stroke, temperature and pressure boundaries as inputs. Flow-derived
Reynolds number and Cd share measurement error; using measured Re as an ordinary
forward feature can leak the unknown flow into the predictor. The governing
component plan discusses a pressure-forcing coordinate or an implicit forward
solve. Neither missing absolute pressure nor missing reverse sweeps may become
invented cavitation or hysteresis evidence.

For multi-fidelity work, compare against the experiment-only model on the same
held-out groups and budget. A cheaper simulation is not automatically a useful
fidelity level. Start with a small matched pilot, preserve mesh/time convergence
and solver setup, and keep the simpler model if fusion adds no predictive value.
The current external-gap water/air VOF case prescribes liquid flow and omits the
internal valve passage; it cannot independently measure internal-valve Cd.

## Work order and later integration

1. Collect the candidate sources and complete their acquisition checks (this
   stage). Review numeric extraction candidates for nozzle/thermal work while
   defining the real pintle measurement contract. This does not start a training
   campaign or authorize a hardware test.
2. Run an isolated pintle hydraulic pilot once actual measurements and as-built
   geometry/instrument definitions are available. In parallel, build reviewed
   source-case datasets for nozzle performance, cooling and pump maps.
3. Train and assess component research models outside the engine solve. Record
   negative findings and data gaps. Follow with materials, spray and other
   components as their observation contracts and evidence become adequate.
4. After repository cleanup and the relevant component-design/input/export
   issues are resolved, integrate one frozen model through the ordinary engine
   job. The shared geometry identity, movable-pintle host handoff, MDO use of
   hydraulic coefficients, units, validity and numerical derivatives are concrete
   dependencies. Do not bypass an existing unavailable/refusal state.
5. Demonstrate a baseline-versus-calibrated sizing comparison with the same model
   identity in the solver, constraints, report and exported part. Check both
   predictions and derivatives where optimization needs them. A coefficient
   changed only in the report has not affected the design.
6. Extend the same evidence interface to other components. Only after the
   bounded component question works should an outer planner select the next
   experiment, repeat, CFD resolution or specimen using uncertainty and cost.

Research opportunities include whether a flow window transfers across pintle
geometries, whether channel roughness explains hydraulic and thermal residuals,
how pump maps transfer across speed, and whether spray corrections improve
component predictions. The tool should make those questions reproducible; the
answers and physical validity remain results to establish.
