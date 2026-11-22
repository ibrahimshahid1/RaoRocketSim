# Component data, multi-fidelity MDO, and simulation feedback

Date: 2026-11-03

Status: PROPOSED. This is an architecture and research plan. It adds no executable models, CLI commands, engineering gates, data, or solver results.

## 1. Recommendation

Give each component model a declared way to consume evidence: which quantities it can learn, which measurements it needs, how uncertainty enters, and where the resulting model can be used. Fit and assess those models outside the engine optimization, then freeze their versions for each MDO run. Let an outer research loop request experiments or simulations and produce the next model version.

The first demonstration should be the conical pintle metering-valve study. It has a measurable hydraulic response, a small geometric design space, an existing analysis workflow, and a clear question: does the approximately linear flow window persist across pressure and Reynolds-number conditions, and does CFD help predict an untested geometry?

The reusable deliverable is the component evidence interface. The pintle study establishes whether one particular model and data-fusion strategy deserves to use that interface. Different components will need different physical and statistical models.

Experiments should be optional inputs to an ordinary analysis job. A job requesting a result that specifically requires experimental support should report that support as missing when it is absent. A simulation assessment can satisfy a declared simulation requirement; hardware acceptance still needs the appropriate configuration-bound physical evidence.

## 2. What the inspected repositories already provide

These are implementation observations from the working trees on 2026-11-03, with RaoRocketSim at commit d636d85 plus existing local changes. They are not claims of physical validation.

| Existing code | Useful foundation | Extension needed |
|---|---|---|
| [EngineJobSpec](../../raosim/mdo/job.py), [job CLI](../../lrekit/job_cli.py) | One versioned input contract, model selection, operating points, output policy, and job digest | Component dataset/model bindings, evidence requirements, and simulation requests. The current parser rejects unknown top-level keys, so proposed configuration needs an intentional schema migration. |
| [MovablePintleSpec](../../raosim/movable_pintle.py), [model documentation](../subsystems/S08-injector-and-spray-coupling/MOVABLE_PINTLE_MODEL.md) | An opening-dependent Cd curve with artifact identity, exact geometry binding, fluid identity, and operating-domain declarations | Measurement import, uncertainty, multi-condition fitting, and a separate model for transfer across a geometric family |
| [MDO injector readouts](../../raosim/mdo/injector.py) | Differentiable hydraulic and geometric calculations | Cd currently comes from constant mission coefficients. A dataset accepted by another path does not automatically influence this optimization. |
| [Host handoff](../../raosim/mdo/postprocess.py) | Explicit refusal to hand an optimized movable-pintle design to a different injector builder | The movable architecture needs an authoritative shared contract before claiming an optimized, exported, calibrated part. |
| [Numerical state](../../raosim/mdo/state.py), [snapshots](../../raosim/mdo/snapshot.py), [run records](../../raosim/mdo/run_record.py) | Availability states, evaluator fingerprints, stored inputs and results | Include complete learned-model identities, applicability, predictive uncertainty, and evidence lineage. |
| [Part manifest](../../raosim/mdo/build.py) | Enumerated parts and achieved artifact fidelity | Attach quantity-specific evidence requirements and results. BuildFidelity describes artifacts; it should remain distinct from simulation fidelity and model validation. |
| [OpenFOAM exporter](../../raosim/openfoam/case.py), [VOF handoff](../../raosim/openfoam/handoff.py) | Deterministic case generation and a typed, gated sheet-to-parcel handoff | General component profiles, execution, extraction, result collection, and assimilation |

The present OpenFOAM profile is a Foundation v13 external-gap water/air VOF wedge tied to a literature fixture. It prescribes water flow and lacks the internal valve passage. It cannot establish internal hydraulic Cd simply by reading back the prescribed flow. Its documentation explicitly distinguishes static template checks from solver execution and validation.

The neighboring [aiaa_flow_linearity repository](../../../aiaa_flow_linearity) has geometry and flow calculations, window analysis, uncertainty helpers, experimental CSV import, and single- and multi-fidelity BO scaffolding. Its experimental directory currently contains a README and no measurement CSVs. This does not establish whether pilot measurements exist elsewhere.

The attachment reports a built flow loop and pilot calibration, while the pasted reconstruction describes an earlier proposed rig. Instrumentation and current capability need to be reconciled against the actual apparatus. Neither document supplies a completed experimental dataset for this review.

The local scientific-memory status reports a synchronized corpus. Its retrieved records still distinguish source support, software checks, model applicability, and physical validation. The new evidence system should preserve those distinctions.

## 3. Shared architecture

~~~mermaid
flowchart TD
    E["Raw experiments and instrument records"] --> D["Versioned datasets and measurement definitions"]
    S["Verified simulation runs and extractors"] --> D
    D --> F["Fit, assess, and version component models"]
    F --> B["Frozen model bundle and applicability rules"]
    B --> M["Engine analysis or MDO"]
    M --> A["Candidate design and evidence coverage"]
    A --> R["Research planner: next experiment or simulation"]
    R --> S
    R --> E
~~~

The research planner proposes hardware work; an actual measurement enters only after the experiment is performed. A requested simulation and a completed, usable simulation are also separate records.

### Evidence and model objects

Use a small shared set of records, with component-specific observable schemas:

| Record | Required meaning |
|---|---|
| Component identity | Component instance, geometry family and schema, nominal design, as-built specimen, material/process revision, dimensions, and relevant surface/metrology descriptors |
| Dataset | Immutable raw-file hashes, source kind, acquisition/simulation method, units, conditions, specimen and campaign identities, processing version, exclusions, and uncertainty |
| Measurement definition | What was actually observed: pressure-tap locations, absolute/gauge reference, mass-flow estimation, temperature location, averaging interval, optical measurement plane, and any filtering |
| Model artifact | Governing model and version, learned parameters or surrogate arrays, preprocessing, training dataset hashes, fit settings, software versions, applicability, validation results, and uncertainty representation |
| Evidence requirement | Component, quantity, operating conditions, required evidence kind, acceptance rule, tolerances, and source or project rationale |
| Simulation request/result | Frozen design and boundary conditions, solver profile/build, mesh and numerical settings, requested observables, extraction version, quality checks, and result hashes |

Keep source kinds explicit: experiment, simulation, literature observation, and synthetic fixture. Store separate descriptors for numerical fidelity and evidence quality. An experiment is a noisy observation of a specimen under particular conditions, rather than an automatically exact highest-fidelity label.

Retain both raw and processed data. Store pressure units and reference explicitly; retain uncertainty as standard uncertainty or a distribution with its coverage convention. Record sensor biases and correlations across a campaign. Nominal ADC resolution or a sensor's full-scale accuracy percentage is insufficient to define uncertainty at a particular operating point.

Keep measurement variability, manufacturing variability, parameter uncertainty, model discrepancy, and numerical error separately traceable. Repeated samples and mesh refinements of the same condition must retain their dependence rather than being counted as independent physical evidence.

The measurement definition matters as much as the column name. A CFD patch-mean pressure and a bench wall-tap pressure are different observation operators. Thermocouple placement and response similarly affect comparisons with wall-temperature fields. Define how each model produces the quantity that the instrument measured.

### Exact specimen and geometric-family models

Retain the present exact-geometry calibration rule. A curve measured on specimen A remains evidence about A.

Add a distinct family-model contract for predicting other geometries. It should name the family, descriptor definitions, supporting specimens, operating coverage, and held-out-geometry assessment. A new geometry receives a model prediction with uncertainty and a transfer status. Its own measurement and acceptance records remain outstanding.

This allows useful research across a design space without weakening exact-part provenance. A matching hash establishes identity; it does not establish measurement accuracy or physical adequacy. A rectangular box containing training inputs likewise does not establish support throughout that box.

## 4. Appropriate model families

The following are proposed model choices. Specific correlations, coefficients, and validity limits still require their own source review and relevant data.

| Component or question | Experimental inputs | First useful model | Later simulation/fusion model | Limit of the evidence |
|---|---|---|---|---|
| Metering valves and pintle hydraulics | Flow, opening, upstream/downstream pressure, temperature, measured geometry, repeated up/down sweeps | Flow law with a measured Cd/conductance map; weighted parameter fit or a small GP correction | Internal-flow laminar/RANS model plus learned discrepancy | Supports hydraulic behavior in the covered fluid and operating regime |
| Feed lines and manifolds | Pressure differences and branch flows across operating conditions | Friction/local-loss and flow-distribution corrections | 3D internal-flow correction for turning, junctions, and maldistribution | Does not establish transient stability or all manufactured surface conditions |
| Injector atomization | Flow split, spray imaging, angle, sheet thickness, drop-size distributions, measurement plane and method | Regime-specific corrections to spray/atomization models | Two-stream VOF and, where appropriate, a separately assessed parcel model | A valve Cd curve does not identify spray quality or combustion efficiency |
| Cooling passages and chamber walls | Flow, pressure loss, inlet/outlet states, heat input, wall temperatures and locations | Separate pressure-loss and heat-transfer corrections; later a reduced thermal model | Conjugate heat-transfer simulation with thermal and hydraulic discrepancy | Wall temperatures alone may not identify gas-side heat load, coolant HTC, and contact resistance separately |
| Pumps | Flow, speed, pressure rise, shaft torque/power, inlet state, temperature | Head and efficiency maps, with similarity-based coordinates where justified | Rotating-flow simulations corrected against measured maps | Affinity scaling does not establish cavitation, off-design stability, or cryogenic transfer |
| Nozzle performance | Wall pressure, thrust and mass flow, inlet state, ambient pressure | Corrections to specified performance or pressure observables | Compressible-flow simulation; reduced spatial basis if field data justify it | Cold-flow agreement does not establish reacting chemistry, heat load, or hot-wall performance |
| Chamber efficiency | Independently measured chamber state, propellant flows, throat geometry, thrust and test conditions | Carefully identified delivered-performance corrections | Reacting-flow studies with explicit chemistry/combustion uncertainty | Avoid confounding injector, combustion, throat, and nozzle losses in one fitted efficiency |
| Structure and materials | Temperature-dependent coupons, strain/load data, dimensional inspection, relevant cycle histories | Material/structural parameter estimation and uncertainty | FEA reduced models through a separate structural solver adapter | A proof or elastic strain test does not establish fatigue/creep life |
| Actuators and dynamic feed response | Time-resolved command, position, pressure, flow and load | Grey-box state-space/system-identification models; separate friction/backlash terms | Transient coupled simulation when supported | Static sweeps do not identify dynamic admittance or combustion stability |

Start with measured interpolation or a small weighted parameter fit when data are scarce. Use Bayesian parameter estimation when identifiable parameters and useful priors exist. Add a GP discrepancy when residual structure warrants it. Spatial reduced-order models become useful when the output is a field, while dynamic models need time histories.

A neural surrogate or PINN may become appropriate for a larger, well-defined problem. Neither is a prerequisite for the first component data interface.

### Multi-fidelity progression

For the pintle, compare increasingly complex predictors:

1. A declared analytic baseline.
2. An experiment-only model.
3. A baseline with an experiment-trained discrepancy.
4. CFD alone, with its numerical uncertainty.
5. A fused CFD/experiment model, adding an analytic level only if helpful.

A possible scalar hierarchy, on an appropriate output transform, is:

~~~text
u_CFD(x)      = a10 u_analytic(x) + discrepancy10(x)
u_physical(x) = a21 u_CFD(x)      + discrepancy21(x)
y_measured   = observation(u_physical) + campaign_bias + measurement_error
~~~

Autoregressive Gaussian-process models have an established basis in [Kennedy and O'Hagan (2000)](https://doi.org/10.1093/biomet/87.1.1). The original computer-code hierarchy needs an explicit observation model when extended to noisy experiments. Numerical error, physical model discrepancy, and instrument error should remain identifiable categories in the implementation.

Begin with a simple correction or constant inter-source scaling. Only adopt nonlinear source relationships when grouped predictive assessment supports the additional complexity; [Perdikaris et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5332612/) provide a nonlinear autoregressive approach.

Avoid simultaneously fitting many physical parameters and a flexible discrepancy from a small dataset. The resulting ambiguity is the calibration/discrepancy confounding examined by [Brynjarsdóttir and O'Hagan (2014)](https://doi.org/10.1088/0266-5611/30/11/114007). Independently measured roughness, dimensions, and sensor calibrations can inform priors. A fitted roughness should not be presented as a measured surface property.

The ASCEND code already uses optional BoTorch/GPyTorch tooling, and [Emukit documents multi-fidelity emulation](https://emukit.github.io/multifidelity-emulation/). Benchmark a small reference implementation before choosing a production fitting backend. Store the fitted numerical artifact and its semantics independently of the training library.

## 5. The pintle pilot: changes to the attached proposal

The main direction is sound: model a hydraulic response over geometry and operating conditions, then calculate the linearity window from predicted curves. Keep the following issues explicit.

### Geometry and conventions

The conical seat-valve study and the Son continuous radial-gap injector are different geometry families. Their area, length, angle, and opening definitions need separate adapters. The ASCEND code uses full cone angle theta and computes gap from its half-angle. Importing theta as the Son tip-angle parameter would change the physical interpretation.

Define an authoritative geometry mapping and reference flow area before deriving Cd or Reynolds number. Optimize manufacturable variables such as seat diameter, cone angle, and edge thickness. Dimensionless features can describe the surrogate, but they need a consistent forward map from those design variables and stroke.

The analytic level also needs honest status. The ASCEND physics module includes a logistic Cd model with chosen transition and separation parameters. Its existence as synthetic-data code is not experimental establishment of that correlation. Treat it as a declared baseline/prior until reviewed and assessed.

### Model flow without leaking the measured response into the input

For the declared incompressible, noncavitating branch:

~~~text
m_dot = Cd A sqrt(2 rho DeltaP)
Re    = rho (Q/A) Dh / mu
~~~

If both Cd and Re are derived from measured flow, their errors are correlated. At prediction time, flow is unknown, so an ordinary Cd(geometry, measured_Re) regressor is not a complete forward model.

The recommended first implementation fits measured mass flow as the observation, using geometry, stroke, temperature, and measured pressure boundary conditions as inputs. A latent positive Cd or conductance correction can sit inside the flow equation.

An alternative forcing coordinate is:

~~~text
Re_DeltaP = Dh sqrt(2 rho DeltaP) / mu
Re       = Cd Re_DeltaP
~~~

The second relation follows from the stated flow law using the same area/velocity convention. This coordinate avoids requiring measured flow as a forward input. If actual Re is retained instead, solve the resulting implicit flow equation and account for shared input/output measurement error.

Near closure, area uncertainty can dominate a derived Cd. Treat leakage, stroke-zero uncertainty, seating, and detection limits separately. Do not train a finite Cd at zero area from a computational sentinel. A reference-area convention also needs to precede any universal-looking bounds on a fitted Cd.

### Reconcile pressure and cavitation definitions

The PDF uses:

~~~text
sigma_A = (Pd - Pv) / (Pu - Pv)
~~~

The ASCEND implementation uses:

~~~text
sigma_B = (Pd - Pv) / (Pu - Pd)
sigma_A = sigma_B / (1 + sigma_B)
~~~

These are algebraically related conventions with different numerical thresholds. Store the definition ID alongside any value or criterion. Neither an unspecified sigma threshold nor a generic Reynolds cutoff establishes onset or a plateau for this valve.

The current experimental enrichment routine can estimate missing upstream pressure as 2.5 times DeltaP. That assumption cannot supply measured cavitation evidence. Missing absolute pressure should leave that assessment unavailable. Similarly, zero hysteresis and a favorable cavitation value must not be substituted for absent observations.

The checked-in ASCEND defaults use pressure levels 0.3, 1, 3, and 4 bar, while the supplied baseline plan uses 0.3, 1, and 2 bar. The current configuration also has a Re gate and differing cavitation settings across analysis and BO. Reconcile these against the intended protocol and actual rig rather than inheriting them as validated engineering criteria.

### Experimental design and statistical assessment

Use fixed-stroke pressure sweeps to investigate Reynolds dependence while holding restriction geometry fixed. Also collect repeated stroke sweeps and directions to estimate repeatability and hysteresis. Record actual temperature, measured dimensions, stroke zero, pressure stability, pressure references, and collection intervals.

Pressure expansion should follow achievable flow/pressure capability, metrology, and observed response changes. A higher pressure does not guarantee a Cd plateau, validity at crack-open, or an absence of cavitation. Single-phase and cavitating observations should remain distinguishable; a cavitating observation can be useful research data while being ineligible for a single-phase fit.

Keep full campaigns and specimen identities. Hundreds of rows from eight geometries provide detailed curves, but still only eight tested geometries for assessing geometric transfer. Adjacent stroke points also share geometry and instrument errors.

Validate by holding out entire geometry groups, with campaign/specimen checks where appropriate. Fit preprocessing, model hyperparameters, and model-choice decisions within the training partition. If the intended deployment has CFD available for a new geometry, allow that CFD in the declared validation scenario; also assess the distinct scenario of predicting with no CFD at the new geometry. Do not use held-out experimental responses.

Compare experiment-only, baseline-corrected, CFD-only, and fused predictions. Report flow/Cd error by stroke and regime, window-width error, geometry ranking, interval coverage and sharpness, uncertainty in the comparisons, and experiment/compute cost. A nominal 90% interval needs a coverage assessment with the small number of independent geometries acknowledged.

Generate correlated posterior flow curves and run the same fixed window algorithm on each curve. This propagates uncertainty into the window and its ranking. Independent pointwise draws would distort the curve and the derived window. Document stroke-grid resolution and the rule used to search candidate windows.

Keep the discontinuous window search in the outer research loop. The inner differentiable engine solve can consume a smooth hydraulic model. Any smooth approximation to the window objective would require its own approximation assessment.

The study's 2% nonlinearity, 0.4 mm window, and improvement targets remain proposed study endpoints. Confirm their definitions and whether measurement uncertainty can resolve them before using them as acceptance criteria. A selected design needs a prospective measurement after selection.

Initially use a small matched CFD/experiment pilot. Estimate cost and discrepancy before committing to hundreds of CFD geometries. When fusion fails to improve prediction, retain the simpler model and report the finding. The attachment's narrow novelty claim would need a dedicated literature review before publication.

## 6. Connecting learned models to the differentiable MDO

Provide a pure numerical evaluator for each component's supported quantities. At an operating point it should return the prediction, an uncertainty representation, availability, and applicability information. The host report can expand compact numerical reason codes into source and domain explanations.

Train outside JAX. A fitted small GP can export its kernel, transforms, training inputs and factorization for a JAX evaluator; a smooth table or low-order fit may be simpler. Validate the evaluator against the fitting implementation and check derivatives where the model is used by the optimizer. No fitting, file reads, library callbacks, or solver launches belong inside the traced engine calculation.

Freeze the complete model bundle during each solve. Its identity should cover all arrays, transforms, hyperparameters, uncertainty settings, applicability definitions, and source data. Extend the existing state/snapshot fingerprint checks so a state cannot be reported under a different calibrated model.

A calibrated model must participate in the quantities that determine sizing and constraints. For example, changing Cd must affect required area/opening, velocity and relevant feed requirements. Substituting an empirically corrected coefficient only in the final report would leave the design based on the old model.

When Cd depends on geometry being sized, the existing direct area calculation can become an implicit sizing problem. Resolve that dependency in the component residual/inversion and differentiate it consistently. Assess monotonicity, uniqueness, branch changes, and unavailable cases.

Start uncertainty propagation with a frozen set of parameter/model samples and repeated engine evaluations. Preserve shared samples across coupled quantities and operating points. Do not assume uncertainty in injector losses, feed demand, pump power, and cooling states is independent when they share a model or measurement source.

For a genuinely assessed constraint, a possible later policy is a stated probability that the physical margin exceeds zero. Its risk level and uncertainty interpretation must be declared; a GP confidence band alone does not create an engineering acceptance standard. First establish nominal behavior, sensitivity, and calibrated uncertainty.

Support across the complete requested operating envelope matters. A design can leave the learned domain during optimization or at a throttle point. Constrain ordinary supported runs to an assessed domain and preserve OUTSIDE_VALIDITY/UNKNOWN when support is missing. Research exploration can propose such points with a stated need for new evidence.

## 7. Proposed CLI behavior

Expose the same component bindings through guided input, YAML/JSON, and CLI options. Use the shared EngineJobSpec rather than a separate research-only definition of an engine.

Three useful policies are:

| Policy | Behavior |
|---|---|
| Baseline | Run the declared analytic model and report its evidence limits. |
| Prefer supplied model | Use an eligible pinned model; any fallback is explicit in the result and coverage report. |
| Require specified evidence | A missing, incompatible, or unsupported model/data requirement remains unmet; it cannot be turned into a pass by fallback. |

The following is a proposed extension fragment, not accepted current syntax:

~~~yaml
component_models:
  injector.radial_metering:
    policy: require
    artifact: models/pintle_family_v1/model.json
    outside_domain: unavailable
evidence_inputs:
  injector.radial_metering:
    - datasets/pintle_campaign_01/manifest.json
simulation_assessment:
  requirements: requirements/engine_pre_hardware.yaml
  backend: openfoam_foundation_13
~~~

Model fitting should be a separate operation from ordinary engine execution. Illustrative future verbs:

~~~text
lrekit evidence template --component pintle_valve --out dataset-template/
lrekit evidence import dataset/manifest.json
lrekit models fit fit-request.yaml
lrekit models assess model.json --validation validation-plan.yaml
lrekit validate engine.yaml
lrekit run engine.yaml
lrekit simulate plan engine-run/ --out simulation-requests/
lrekit simulate run simulation-requests/
lrekit simulate collect simulation-run/
~~~

The final report should show, per component and required quantity: the model used, source kind, operating coverage, prediction and uncertainty, available checks, missing evidence, and supported conclusion. Unsupported component schemas should say NOT_IMPLEMENTED. Archiving an arbitrary CSV is distinct from having a model capable of using it.

## 8. OpenFOAM round trip

Introduce simulation profiles tied to physical questions. The first hydraulic profile should contain the internal conical-valve flow passage and pressure observation locations. A separate injector-sheet profile can extend the current VOF work. Cooling, compressible nozzle flow, and reacting flow need their own supported profiles; structures require a structural solver.

The hydraulic profile must expose a measurable input/output pair. Pressure-driven flow can predict flow for a specified pressure condition. A prescribed-flow internal case can predict the pressure drop. Do not prescribe both the quantity being checked and its intended driving response, then report agreement as independent evidence.

Each request should be bound to a design snapshot, geometry and fluid-domain hashes, materials/properties, operating point, boundary conventions, solver distribution/build, mesh recipe, numerical settings, and requested quantities.

The requirement planner needs a profile-specific input/output contract:

| Proposed profile | Necessary inputs beyond the part surface | Extracted response |
|---|---|---|
| Internal valve/manifold hydraulics | Complete wetted passage, opening, tap positions, fluid state/properties, compatible inlet/outlet conditions, roughness and inlet-flow assumptions | Complementary flow or pressure response, branch distribution, consistently defined Cd, and hydraulic loads where resolved |
| Injector sheet/spray | Both flow passages where relevant, stream boundary states, phase properties including surface tension, surrounding domain, and measurement planes | Sheet geometry/velocity, spray observables, conservation and averaging evidence; droplet statistics only where resolved or through a separately assessed breakup model |
| Cooling/CHT | Coolant and solid domains, interfaces, temperature-dependent properties, coolant boundary states, hot-side heat-load condition, and external thermal conditions | Pressure loss, heat transfer, outlet state, wall-temperature/heat-flux fields; the imposed hot-side load remains an assumption if not independently established |
| Compressible nozzle | Internal flow domain, inlet total state/composition, ambient condition, wall treatment, and thermochemical closure | Mass flow, pressure field and thrust with defined integration/control surfaces |
| Rotating pump | Internal/rotating domains and interfaces, speed, fluid/inlet state, complementary flow or pressure condition, and supported phase treatment | Head, torque/power and hydraulic efficiency; cavitation quantities only for an appropriate supported model |

These profiles should declare missing inputs before launch. Simulation outputs become typed observations in the same evidence system, with simulation origin retained.

The execution and collection sequence should be:

1. Generate fluid and, where needed, solid domains with semantic inlet, outlet, wall, interface, and measurement labels. A STEP/STL part alone does not define those domains or boundary conditions.
2. Verify the installed solver/profile identity and execute the mesh and solver workflow with declared resource limits, restart state, and retained logs.
3. Extract quantities using versioned observation definitions. Match experimental tap locations, pressure definitions, averaging windows, and flow conventions.
4. Assess mesh quality, convergence/stationarity, conservation, and relevant refinement/sensitivity checks.
5. Write an immutable simulation result and match it to the component evidence requirements.
6. Generate a candidate updated model under a declared assimilation policy, assess it, and pin an eligible version into a subsequent engine run.

OpenFOAM's [v13 post-processing framework](https://doc.cfd.direct/openfoam/user-guide-v13/post-processing-functionality) supplies function objects for quantities such as averages, probes, flux-related processing, and wall heat flux. Extraction still needs profile-specific unit and reference checks. In particular, distinguish pressure in Pa from kinematic pressure, reconstruct the appropriate pressure from p_rgh, and distinguish volumetric from mass flux.

For representative geometries and regimes, plan at least three systematically refined meshes when estimating observed convergence order. Include gap resolution and a wall treatment consistent with the turbulence choice. Transient studies additionally need timestep and averaging-window assessment; domain and boundary-placement sensitivity can matter independently. Apply GCI only when its assumptions are supported, following an established procedure such as [NASA's spatial-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html). Solver completion and a checkMesh pass do not establish output accuracy.

Keep low-Re, separated, cavitating, and strongly three-dimensional regimes identifiable. A steady axisymmetric calculation cannot provide measured actuator hysteresis or resolve an omitted eccentric passage. Discrepancy can characterize supported residual error; it cannot establish omitted physics in an unobserved regime.

Automation should cover request generation, execution tracking, extraction, eligibility checking, and collection. Every refit produces a new model identity. A configured policy can select an assessed new version for the next solve while preserving the earlier run and model. Failed, incomplete, mismatched, or unsupported results remain recorded with those states.

Changes to geometry, boundary definitions, solver setup, or extraction invalidate dependent applicability checks and cached responses. Historical results remain evidence about their original configuration; they are not silently rebound to the new design.

Call the resulting deliverable a pre-hardware simulation assessment, with named requirements and quantities. [NASA's validation guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/valassess.html) ties validation to intended uses and experimental coverage. Simulation can strengthen the evidence supporting a design without certifying its manufactured hardware or operating history.

## 9. Prerequisites and implementation order

The full engine integration belongs after the current CLI, geometry, and validation gaps have been resolved sufficiently for the relevant component. Data definitions and an isolated pintle pilot can proceed while that work continues.

| Stage | Work | Exit evidence |
|---|---|---|
| 0. Define the pilot | Reconcile as-built rig, geometry/angle conventions, pressure matrix, observables, uncertainty, and study endpoints; separate conical valve from Son injector | One reviewed component contract, data template, and protocol with unresolved fields visible |
| 1. Build evidence import | Immutable manifests, units/reference handling, specimen/campaign grouping, uncertainty, source types, and applicability | One real dataset is processed reproducibly; missing pressure or metrology remains missing; synthetic fixtures retain their identity |
| 2. Fit the first model | Measured curve or small fit, experiment-only baseline, model artifact, grouped assessment, and frozen numerical evaluator | Reproducible predictions, uncertainty, correct unavailable cases, and fitting/JAX parity where required |
| 3. Demonstrate actual design influence | Connect the learned hydraulic model to shared resolved inputs, component sizing, mission points, and reporting | Baseline-versus-calibrated ablation explains changes to geometry and engine metrics; model identity follows through every output |
| 4. Establish the CFD round trip | Internal hydraulic profile, execution/extraction, quality evidence, and a small matched pilot | Reproducible eligible CFD observations bound to the right geometry/conditions; numerical uncertainty assessed |
| 5. Test family transfer and fusion | Grouped predictive comparisons, uncertainty propagation to windows, and prospective batch testing | Evidence that the selected model improves the intended prediction/decision at a measured resource cost, or a documented choice to keep the simpler model |
| 6. Generalize components | Reuse evidence contracts for cooling pressure loss/thermal data, then pump maps and other supported models | Each component has its own measurable outputs, applicability, model assessment, and requirement coverage |

Before Stage 3, stabilize the relevant engine input-to-analysis-to-export identity. The current movable-pintle host handoff refusal is a concrete integration dependency. A research surrogate should not bypass it.

Before declaring simulation requirements satisfied, define their acceptance rules and engineering basis. Some checks concern numerical adequacy, some require experiments, and others require material, inspection, assembly, proof, or operating evidence. Requirement coverage should show all applicable items, including unknown ones.

Meaningful software checks for the implementation include: unit/reference conversion, raw-to-derived uncertainty, source-type preservation, geometry/condition mismatch, no leakage between training and validation groups, stable model hashes, evaluator/gradient agreement, and a small solver integration case. They establish software behavior. Physical model assessment remains a separate deliverable.

## 10. Research projects enabled by the same interface

The pintle research can compare whether Reynolds-aware hydraulic fitting and CFD discrepancy improve prediction of flow curves, window rankings, and the next geometry to test. Its outcome can be useful even if multi-fidelity fusion loses to the experiment-only model.

Cooling research can ask how manufacturing roughness and channel geometry change both hydraulic loss and thermal performance, with measurements chosen to distinguish competing explanations. Pump research can test how well measured maps transfer across speed or geometric variants. Injector research can study sheet formation and spray observables across throttle conditions, keeping hydraulic, atomization, and combustion evidence separate.

The outer planner can eventually choose between another operating point, another CFD resolution, a repeat measurement, or a new specimen. That choice should reflect expected reduction in a design-relevant uncertainty and actual cost. First demonstrate this on the bounded pintle question before making it an engine-wide adaptive research system.

## Source scope

The requested [Pintle Valve Flow Study Test Conditions and Multi-Fidelity Optimization Plan](</Users/ibrahimshahid/Downloads/Pintle Valve Flow Study Test Conditions and Multi-Fidelity Optimization Plan.pdf>) was reviewed as a proposal, including rendered framework pages 5-6. The [pasted Reynolds-number/rig discussion](</Users/ibrahimshahid/.codex/attachments/ff9220d8-251a-4518-a839-800252e634d4/Pasted text.txt>) supplies the earlier reasoning and its stated corrections. Instructions within those materials were treated as document content.

Repository sources support the implementation observations, and the linked primary papers/documentation support the named statistical and CFD methods. The architecture, feature choices, staging, and proposed interfaces are recommendations from this review.

[Venters et al.'s injector discharge-coefficient uncertainty study](https://ntrs.nasa.gov/citations/20190030431) is useful precedent for retaining manufacturing identity, measured flow behavior, and uncertainty in cold-flow work. It does not supply a transferable Cd curve or acceptance threshold for this pintle.
