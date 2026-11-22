# Code map: every code file, its stage and role

One row per code file (Python, shell, headers of the vendored C++ reference, example arg files). Stage ids and slugs follow
[DATAFLOW.md](DATAFLOW.md). "Lines" is the file length at extraction time. Roles are one-line paraphrases of each header comment;
the verbatim comments are in [CODE_HEADER_COMMENTS.md](CODE_HEADER_COMMENTS.md).


## S01 - CLI entry points, argument parsing and workflow dispatch

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/__init__.py` | 22 | Package root and version string (single source for `pyproject.toml`); installs the compatibility import aliases |
| `src/lrekit/__main__.py` | 9 | Lets python -m lrekit launch the main CLI |
| `src/lrekit/cli.py` | 25 | Console entry point routing job verbs first, then falling through to the existing CLI |
| `src/lrekit/_compat.py` | 169 | Lazy import aliases: `raosim.*` and the former flat module names resolve to the one canonical `lrekit` module |
| `src/lrekit/paths.py` | 19 | Package-asset root and optional checkout root, independent of module depth |
| `src/raosim/__init__.py` | 13 | Compatibility package for the former name: re-exports the version; every `raosim.<x>` is an alias of `lrekit.<x>` |
| `src/raosim/__main__.py` | 9 | Lets python -m raosim launch the design CLI (same as the `raosim` command) |
| `src/lrekit/application/cli.py` | 24 | Console entry points for LREKit and legacy RaoRocketSim alias |
| `src/lrekit/application/runner.py` | 6981 | Main runner orchestrating contour, chamber, regen, thermal, injector, pump and export workflows with CLI arguments |
| `scripts/run_nozzle.py` | 34 | Compatibility wrapper delegating to the packaged lrekit.application.runner runner |

## S02 - Inputs, units, validation and job contracts

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/guided.py` | 300 | Interactive guided wizard collecting engine requirements with injectable prompting for testability |
| `src/lrekit/job_cli.py` | 416 | Job verbs guided, run, validate, catalog for the unified YAML workflow |
| `src/lrekit/core/input_validation.py` | 236 | Shared host-side elementary domain checks for engine-analysis inputs across all workflows |
| `src/lrekit/geometry/manufacturing.py` | 203 | Wadel-sourced milled cooling-channel fabrication limits, kept separate from physical validity limits |
| `src/lrekit/optimization/capabilities.py` | 69 | Executable job capabilities shared by validation and user-facing catalogs |
| `src/lrekit/optimization/job.py` | 1209 | Versioned EngineJobSpec input contract that wizard, CLI flags and JSON/YAML all compile into |
| `src/lrekit/optimization/metrics.py` | 382 | Registry of selectable objectives and targets, derived from constraint manifest and validated against snapshot fields. |
| `src/lrekit/core/units.py` | 224 | Engineering-notation unit parsing into canonical SI, dimension-checked, never guessing bare-number units |

## S03 - Propellant identity and chamber thermochemistry

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/thermochemistry/cea.py` | 466 | Optional RocketCEA thermochemistry integration with fallback to demo-grade propellant table |
| `src/lrekit/thermochemistry/frozen_flow.py` | 1005 | Thermally-perfect frozen-composition quasi-1D nozzle flow using tabulated cp(T) with closure residuals |
| `src/lrekit/thermochemistry/propellants.py` | 213 | Propellant combustion-product database with efficiency convention (eta_c*, eta_CF, eta_Isp) and derived gas constants |

## S04 - Throat sizing, ideal/delivered performance and off-design screens

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/core/atmosphere.py` | 79 | Simplified ISA atmosphere giving temperature, pressure and density versus altitude |
| `src/lrekit/performance/engine.py` | 240 | Computes thrust, Isp and mass flow from nozzle geometry and propellant thermochemistry |
| `src/lrekit/core/gas_dynamics.py` | 270 | Constant-gamma isentropic area-Mach, Prandtl-Meyer and related compressible-flow relations |
| `src/lrekit/components/nozzle/separation.py` | 311 | Empirical flow-separation criteria (Summerfield, Schilling, Kalt-Badal, Schmucker) with design-margin verdict for overexpanded nozzles |
| `src/lrekit/components/nozzle/wall_pressure.py` | 173 | Wall pressure distribution from area-Mach relation with SP-8120 monotonicity separation-risk check |

## S05 - Default Bezier/TOP nozzle contour, throat and chamber geometry

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/components/chamber/geometry.py` | 481 | Combustion chamber, convergent section and full thrust-chamber contour geometry |
| `src/lrekit/components/nozzle/geometry.py` | 403 | Rao thrust-optimized parabolic bell contour from circular arcs and quadratic Bezier section |
| `src/lrekit/components/nozzle/top_chart.py` | 16 | Rao/TOP angle data digitized from NASA SP-125 Figure 4-14 for default Bezier contour |
| `src/lrekit/components/nozzle/throat.py` | 134 | Shared geometric contract for the chamber/nozzle throat region |
| `src/lrekit/verification/validation.py` | 399 | Design-status labels and conservative engineering gate checks on generated nozzles |

## S06 - Opt-in Rao variational / method-of-characteristics BVP contour

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/components/nozzle/solvers/moc.py` | 890 | Axisymmetric method-of-characteristics unit processes and forward flow-field analysis entry point |
| `src/lrekit/components/nozzle/solvers/bounded.py` | 763 | Bounded axisymmetric MOC characteristic net marched between axis and a prescribed nozzle wall |
| `src/lrekit/components/nozzle/solvers/topology.py` | 305 | Explicit Rao construction topology object per REWRITE_PLAN section 11.7 |
| `src/lrekit/components/nozzle/solvers/nasa.py` | 3589 | Python port of NASA/JHU MOC_Grid_BDE producing the Rao B-D-E optimum-thrust characteristic topology |
| `src/lrekit/components/nozzle/solvers/existence.py` | 2926 | Solver-independent D-attachment existence scans for Rao B-D-E topology and convergence-domain classification |
| `src/lrekit/components/nozzle/solvers/wall_optimizer.py` | 433 | Experimental direct wall-shape thrust optimizer on the forward bounded MOC net (not CLI-reachable) |
| `src/lrekit/components/nozzle/solvers/residuals.py` | 160 | Residual primitives for the Rao variational/MOC coupling (one-line docstring) |
| `src/lrekit/components/nozzle/solvers/variational.py` | 7083 | Experimental Rao-style variational nozzle prototype: control-surface functionals, direct optimizer, CE-driven MOC wall |
| `src/lrekit/components/nozzle/solvers/transonic.py` | 303 | Kliegel-Levine transonic throat kernel for MOC starting line, with corrected NASA coefficient typos |
| `src/lrekit/components/nozzle/solvers/wall.py` | 173 | Spline wall geometry queried by MOC solver for radius, angle and characteristic intersection |

## S07 - Regenerative wall sizing, cooling analysis and wall structural screens

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/core/coolants.py` | 24 | Canonical coolant name normalization shared by all thermofluid paths |
| `src/lrekit/core/materials.py` | 374 | Thrust-chamber and nozzle construction materials catalog supplying wall thermal and structural properties |
| `src/lrekit/components/cooling/screens.py` | 2357 | Conservative screening physics for cooling, wall and structural checks during preliminary nozzle design |
| `src/lrekit/components/cooling/profile.py` | 430 | Station-wise regen wall and channel dimension profile, helical path length, normal-offset contour primitive |
| `src/lrekit/components/cooling/design.py` | 1460 | Cooling-coupled contour selection: throat curvature and channel mass-flux levers shaping geometry before fixing contour |
| `src/lrekit/components/cooling/thermofluids.py` | 569 | Higher-fidelity coolant network, real-fluid, radiation and boiling tools extending the axisymmetric regen solver |

## S08 - Pintle injector sizing and optional spray/c* fixed point

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/components/injector/design.py` | 4355 | Pintle injector hydraulic sizing, movable-gap branch, and screening spray surrogates |
| `src/lrekit/components/injector/movable.py` | 686 | Movable continuous-radial-gap pintle geometry (Son 2017) with calibrated Cd and static actuation screening |
| `src/lrekit/components/injector/coupling.py` | 320 | Opt-in fixed-point loop coupling injector vaporization efficiency to c-star and cycle mass flow |

## S09 - Feed-pressure ledger to electric pump, motor, inverter and battery sizing

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/components/electrical/technology.py` | 496 | Single registry of electric-drive technology sets shared by traditional pump sizing and MDO |
| `src/lrekit/components/pump/design.py` | 3976 | Electric pump-feed sizing on the injector feed ledger: motor, inverter, battery, inducer, impeller, volute screens |

## S10 - Injector/chamber bolted interface and hardware mass ledger

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/assembly/interfaces.py` | 1529 | Injector-to-chamber bolted joint screening: separation force, hoop stress, faceplate bending, bolt clamp |
| `src/lrekit/assembly/mass.py` | 1507 | Configuration-controlled hardware mass ledger for chamber, injector and electric feed system |
| `src/lrekit/components/cooling/volumes.py` | 317 | Shared quadrature of liner, land and closeout metal volumes, feeding the mass ledger and geometry IDs |

## S11 - CAD, mesh and geometry deliverables

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/assembly/cad.py` | 314 | Assembles topology-gated STEP parts (wall, jacket, pintle, pumps, battery) into one engine assembly |
| `src/lrekit/geometry/export.py` | 875 | Exports nozzle contours as CSV, STL meshes and STEP solids with topology validation |
| `src/lrekit/components/injector/cad.py` | 1521 | Named-body pintle injector CadQuery assembly with STEP and per-body STL export |
| `src/lrekit/components/injector/coaxial.py` | 1209 | Five-part coaxial TRW/Nardi pintle injector CAD, sealed by construction, driven by hydraulic sizing |
| `src/lrekit/components/injector/export.py` | 837 | Resolves pintle reference geometry and writes the pintle output package (JSON, CSV, SVG, PNG, STEP) |
| `src/lrekit/components/pump/cad.py` | 938 | Reference pump geometry manifest and mesh export for electric-pump feed hardware, not blade-to-blade design |
| `src/lrekit/components/pump/brep.py` | 1552 | CadQuery B-rep assemblies of pump package built from the meanline manifest, with export and validation gates |
| `src/lrekit/components/cooling/cad.py` | 1185 | CadQuery/OpenCascade builder of regenerative-wall B-rep with channels, plenums, ports, exported and re-imported as STEP |
| `src/lrekit/components/cooling/geometry.py` | 356 | NumPy/Matplotlib 3-D nozzle wall and axial or helical cooling-channel surfaces, with STL export |

## S12 - Figures, MOC construction diagrams and flow animations

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/components/nozzle/flow_visualization.py` | 894 | Plots MOC-net Mach, pressure, angle fields with characteristics and streamlines |
| `src/lrekit/components/injector/plots.py` | 742 | Pintle injector diagnostic figures: cross-section, spray, hydraulics, thermal, stability, gates, throttle map |
| `src/lrekit/components/nozzle/diagrams.py` | 1069 | Construction diagrams of the Rao/MOC nozzle build, complementing the solved-field flow visualizations |
| `src/lrekit/components/nozzle/plots.py` | 1338 | 2-D and 3-D matplotlib nozzle visualisation (one-line header only) |

## S13 - Differentiable whole-engine MDO, optimizer and requirement-driven design

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/optimization/__init__.py` | 142 | Package docstring for the differentiable multidisciplinary-feasible engine MDO layer, with build-phase tracker and rules |
| `src/lrekit/optimization/assembly.py` | 289 | Walking-skeleton coupled engine residual R(y,x)=0 stacking discipline blocks for Newton solve |
| `src/lrekit/optimization/bounds.py` | 258 | Design-variable bounds derived from architecture choices rather than hand-set |
| `src/lrekit/optimization/constraints.py` | 593 | Authoritative ordered constraint and feasibility manifest consumed by solve, NLP, state and snapshot |
| `src/lrekit/optimization/convergence.py` | 396 | Station-refinement grid-convergence evidence for the fixed-topology coupled solve |
| `src/lrekit/optimization/coolant_htd.py` | 289 | Supercritical coolant heat-transfer-deterioration screen |
| `src/lrekit/optimization/cooling.py` | 545 | Phase 4a 1-D plus fin regenerative-cooling residual block with counterflow upwind march |
| `src/lrekit/optimization/engine.py` | 829 | Phase 7 coupled whole-engine solve integrating nozzle, cooling, injector and pump blocks |
| `src/lrekit/optimization/envelope.py` | 204 | Differentiable thrust-chamber envelope geometry |
| `src/lrekit/optimization/grid.py` | 468 | Fixed-topology chamber/nozzle station grid as differentiable analytic inner-contour stand-in |
| `src/lrekit/optimization/injector.py` | 180 | Phase 5 differentiable pintle-injector block using closed-form algebra |
| `src/lrekit/optimization/mass.py` | 259 | Differentiable thrust-chamber hardware mass, JAX mirror of mass_ledger |
| `src/lrekit/optimization/material_map.py` | 223 | Atomic mapper from catalog material record to traced MDO wall properties; rejects partial or fallback application. |
| `src/lrekit/optimization/nlp.py` | 1456 | Epsilon-constraint hard-constrained NLP with SLSQP, guarded trials, KKT check, multi-start, Pareto sweep. |
| `src/lrekit/optimization/nozzle.py` | 202 | Typed contour-provider seam wrapping analytic TOP/Bezier grid with explicit failure outcome contract. |
| `src/lrekit/optimization/objectives.py` | 63 | Explicit objective identities as reporting contracts for the differentiable engine optimizer. |
| `src/lrekit/optimization/pintle.py` | 241 | Pintle injector element as an optimizable differentiable MDO component. |
| `src/lrekit/optimization/propellants.py` | 373 | Propellant combinations as first-class MDO input with differentiable thermochemistry property handling. |
| `src/lrekit/optimization/properties.py` | 1683 | C1 shape-preserving interpolated property surfaces for differentiable thermochemical and coolant data. |
| `src/lrekit/optimization/pump.py` | 174 | Differentiable closed-form pump plus electric-feed block for MDO mass and power. |
| `src/lrekit/optimization/scaling.py` | 60 | Affine variable and residual scaling to unit box for gradient-based optimization. |
| `src/lrekit/optimization/schema.py` | 1402 | Design variables, bounds and mission definition schema for the engine MDO. |
| `src/lrekit/optimization/state.py` | 1062 | Versioned pure-JAX numerical output contract (EngineResult) for discipline-internal engine evaluation. |
| `src/lrekit/optimization/structures.py` | 288 | Differentiable chamber shell structural screens based on NASA SP-8087. |
| `src/lrekit/application/requirements.py` | 831 | Layer-0 SP-125 mission requirements, coverage statuses, resolved into MissionSpec and constraint selection |

## S14 - Authoritative re-evaluation, provenance, release gates and run records

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/application/design.py` | 2855 | High-level design-gated nozzle workflow with request specs and authoritative evaluation (one-line header only) |
| `src/lrekit/optimization/build.py` | 363 | Expected-part manifest and single build/artifact orchestrator preventing falsely complete ledgers |
| `src/lrekit/optimization/contour_validation.py` | 1598 | Host-side validation of optimized TOP contour against Rao BVP, outside the differentiable inner loop |
| `src/lrekit/optimization/execute.py` | 899 | Single entry point running an engine job end to end across differentiable and traditional paths |
| `src/lrekit/optimization/postprocess.py` | 1475 | Non-differentiated bridge mapping an MDO optimum to authoritative traditional-pipeline outputs and reports. |
| `src/lrekit/optimization/resolved_inputs.py` | 741 | Single host-side resolved-input definition shared by both pipelines, with conventions and provenance. |
| `src/lrekit/optimization/run_record.py` | 146 | Atomic isolated report publication with explicit completeness manifest separate from design feasibility. |
| `src/lrekit/optimization/snapshot.py` | 3719 | Versioned host-side output contract converting JAX results into engine analysis snapshots. |
| `src/lrekit/verification/registry.py` | 787 | Machine-readable provenance and validity metadata distinguishing published equations from repository heuristics |
| `src/lrekit/verification/release.py` | 478 | Evidence-record based release gates for physical hardware; never claims hardware qualification |

## X01 - Differentiable JAX core (rao-bvp LM backend + MDO primitives/thermal)

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/numerics/jax/__init__.py` | 28 | Package marker enabling x64 and describing the JAX differentiable Rao solver backend |
| `src/lrekit/numerics/jax/api.py` | 164 | Public JAX entry points: Optimistix LM solve with sigmoid-bounded reparametrisation; solve_rao_bvp_jax sugar |
| `src/lrekit/numerics/jax/assembly.py` | 644 | Full assembled Rao BVP residual in JAX, parity-gated against the NumPy residual |
| `src/lrekit/numerics/jax/moc_kernel.py` | 828 | Differentiable throat-arc kernel characteristic march (Anderson-form unit processes) as lax.scan |
| `src/lrekit/numerics/jax/primitives.py` | 178 | Gas-dynamics leaf primitives in JAX mirroring NumPy closed forms with sonic-safe guards |
| `src/lrekit/numerics/jax/sensitivities.py` | 252 | J6 exact gradient API over solved BVP: node-level Cf tolerance fields and scalar sensitivities |
| `src/lrekit/numerics/jax/thermal.py` | 154 | Differentiable Bartz, Sieder-Tate, wall-temperature, Schmucker separation and Cf physics for design loop |
| `src/lrekit/numerics/jax/theta_b_solve.py` | 243 | Makes theta_B a solved BVP unknown by recomputing kernel BD in-graph each residual evaluation |

## X02 - Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export (export-openfoam-spray verb; parcel solver is API-only, imported by spray_coupling for handoff types)

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/simulation/openfoam/__init__.py` | 54 | OpenFOAM package facade exporting case builders and VOF handoff gates; never runs OpenFOAM |
| `src/lrekit/simulation/openfoam/case.py` | 1642 | Version-pinned OpenFOAM v13 VOF wedge case generation for movable-pintle sheet studies, text files only |
| `src/lrekit/simulation/openfoam/handoff.py` | 869 | Fail-closed evidence contract turning VOF extraction results into radial-sheet source for spray package |
| `src/lrekit/components/injector/spray/__init__.py` | 124 | Opt-in spray package facade re-exporting benchmark, breakup, carrier and solver APIs without altering legacy injector screen |
| `src/lrekit/components/injector/spray/benchmarks.py` | 735 | Provenance-first spray benchmark fixtures and readiness gates separating experimental targets from author CFD results |
| `src/lrekit/components/injector/spray/breakup.py` | 703 | Versioned WAVE/KH and optional Rayleigh-Taylor droplet breakup primitives with separated coefficient sets |
| `src/lrekit/components/injector/spray/carrier.py` | 354 | Explicit steady axisymmetric carrier gas velocity fields with bilinear interpolation, no extrapolation |
| `src/lrekit/components/injector/spray/dispersion.py` | 161 | Seeded discrete-random-walk turbulent dispersion for parcels using local random generator |
| `src/lrekit/components/injector/spray/domain.py` | 215 | Axisymmetric parcel-tracking domain with piecewise-linear wall and exact segment boundary event detection |
| `src/lrekit/components/injector/spray/drag.py` | 141 | One-way Schiller-Naumann spherical drag for parcels |
| `src/lrekit/components/injector/spray/evaporation.py` | 354 | Explicit Spalding evaporation primitives with caller-selected Sherwood closure and no default properties |
| `src/lrekit/components/injector/spray/handoff.py` | 513 | Typed gate-derived handoff from parcel results to future cycle coupling; currently always cycle-ineligible |
| `src/lrekit/components/injector/spray/ledger.py` | 223 | Mass and momentum accounting for represented droplets under one-way carrier assumption |
| `src/lrekit/components/injector/spray/primary.py` | 994 | Geometry-dispatched deterministic primary parcel source initialization, radial-sheet pintle path literature-calibrated |
| `src/lrekit/components/injector/spray/solver.py` | 944 | Deterministic one-way Lagrangian parcel march integrating breakup, dispersion, evaporation, events and ledgers |
| `src/lrekit/components/injector/spray/statistics.py` | 215 | Weighted droplet-cloud statistics: Sauter mean diameter and Rosin-Rammler mass survival fitting |
| `src/lrekit/components/injector/spray/types.py` | 342 | Dependency-light frozen SI data contracts and validation error for spray models with read-only arrays |

## X03 - Legacy toolbox CLI (RaoRocketSimLegacy -> src/lrekit/legacy/toolbox.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)

| File | Lines | Role |
|---|---:|---|
| `docs/archive/Rocket_nozzle_sim_phase2.py` | 375 | Legacy standalone interactive Rao bell nozzle prototype: area-Mach, angle optimisation, contour, 3D plot, ascent simulation |
| `src/lrekit/legacy/toolbox.py` | 1280 | Legacy interactive/batch toolbox CLI with parameter sweeps for Rao bell nozzles |
| `src/lrekit/performance/altitude.py` | 163 | Legacy altitude sweep of thrust, Isp and Cf with flow separation onset markers |
| `src/lrekit/verification/benchmarks.py` | 1159 | Literature-backed diagnostic benchmark runner comparing solver contours and performance to published references |
| `src/lrekit/legacy/build_log.py` | 139 | Versioned builds/vNNN output directories with metadata logging for legacy runs |
| `src/lrekit/components/nozzle/conical.py` | 187 | Conical nozzle contour and Rao divergence-loss correction used as bell-length baseline |
| `src/lrekit/components/nozzle/comparison.py` | 274 | Compares nozzle contour methods: 2D divergence loss, nozzle efficiency, comparison tables and overlay plots |
| `src/lrekit/application/trade_study.py` | 165 | Single-variable parameter sweeps and multi-panel trade-study plotting for the legacy toolbox |

## X04 - Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling

| File | Lines | Role |
|---|---:|---|
| `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCDlg.cpp` | 256 | MFC dialog for the 3D MOC tool, launches grid solve thread (inferred role; one-line header) |
| `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCGrid.cpp` | 2789 | 3D MOC class using reference-plane method (Armstrong AEDC-TR-78-68); nozzle geometry in, Tecplot 3D data out |
| `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCGrid.hpp` | 152 | Header declaring the C3D_MOCGrid class, body-type enums and Newton-Raphson helper functions |
| `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/newt.cpp` | 82 | Inferred: Numerical Recipes Newton-Raphson with line search used to locate body points |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/Chart.cpp` | 649 | Third-party MFC 2D chart control used for in-GUI plotting (not nozzle physics) |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp` | 3754 | Core 2D/axisymmetric MOC grid and nozzle contour calculation class (Rao, cone, perfect, fixed-end) |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.h` | 201 | Header declaring the MOC grid calculation class, nozzle geometry/type/parameter enums |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE_IO.cpp` | 760 | Output half of the MOC class: writes kernel, wall, summary, Tecplot and TDK rao.dat files |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridDlg.cpp` | 731 | MFC input dialog for the 2D MOC tool; runs the solver and launches STT2001 (header is a one-line wizard comment) |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/dummyStruct.h` | 37 | Dummy struct of double/int arrays used to return multiple values from functions |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/engineering_constants.hpp` | 144 | Shared engineering constants, min/max macros and status enums with JHU/APL banner |
| `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/readme.txt` | 2 | Note that the MOC GUI writes a $$.bat to launch STT2001; relative paths may need editing |
| `Three-Dimensional-Nozzle-Design-Code-master/README.md` | 20 | Top-level README describing the three MFC programs (MOC_GRID_BDE, STT2001, 3D_MOC) and the Rice JHU/APL report |
| `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Chart3d.h` | 65 | Third-party 3D chart class interface (Kris Jearakul), use-at-own-risk notice |
| `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Matrix.cpp` | 532 | Inferred: Matrix class implementation (constructors, extraction, arithmetic) for streamline tracing |
| `Three-Dimensional-Nozzle-Design-Code-master/STT2001/STT2001Dlg.cpp` | 3117 | Streamline Tracing Tool MFC dialog and implementation, traces streamlines on a MOC grid (inferred role; one-line header) |
| `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Vector.hpp` | 69 | Inferred: dense double Vector class with 2D/3D constructors, friend of Matrix, for linear algebra |
| `Three-Dimensional-Nozzle-Design-Code-master/STT2001/readme.txt` | 3 | Usage note for the Streamline Tracing Tool: manual dialog inputs or .inp file open/save |
| `Three-Dimensional-Nozzle-Design-Code-master/license.txt` | 241 | NASA Open Source Agreement 1.3 for LEW-20180; first 40 lines excerpted |
| `src/lrekit/verification/reference_io.py` | 415 | Parsers for NASA/JHU MOC_Grid_BDE whitespace tables and Tecplot files, used as a reference oracle |
| `tests/experiments/nozzle/compare_nasa_reference.py` | 904 | Visibility report comparing Python Rao/MOC outputs with NASA/JHU reference files |
| `tests/experiments/nozzle/nasa_wall_probe.py` | 72 | Phase 12.5 probe comparing the BDE wall constructor against NASA wall.out |

## X05 - Standalone research diagnostics and offline generators (J4/J5 gates, theta_B probes, CEA surface sampler, TVC study, contour/regen generators)

| File | Lines | Role |
|---|---:|---|
| `tests/experiments/nozzle/compare_jax_scan_stationarity.py` | 178 | Compares a Rao BVP solution against independent scan stationarity metrics |
| `tests/experiments/nozzle/d_attachment_diagnostic.py` | 160 | J4 diagnostic of D-attachment start-condition variants behind the stationarity floor |
| `scripts/generate_contour.py` | 177 | Parametric converged Rao BVP plus BDE wall contour generator writing CSV, STL, PNGs |
| `scripts/generate_regen_nozzle.py` | 92 | Generates 3-D regen-cooled nozzle with channels as STL and PNG, optionally thermally coloured |
| `tests/experiments/nozzle/j4_gate_wall_probe.py` | 121 | Replicates the J4 gate config and extracts the solved BDE wall shape |
| `tests/experiments/nozzle/j4_report.py` | 109 | One-shot J0-J4 status report on the eps=10 L80 reference case |
| `tests/experiments/nozzle/j5_chart_sweep.py` | 89 | J5 record run of the Rao chart sweep comparing solver output with 1960 charts |
| `tests/experiments/nozzle/jax_convergence_experiment.py` | 122 | Experiment driver testing JAX LM convergence strategies for the J4 gate |
| `tests/experiments/nozzle/plot_reference_topology.py` | 54 | Renders the Phase-12.6 reference topology and wall to PNG and CSV |
| `tests/experiments/nozzle/rao_existence_scan.py` | 401 | NumPy-only Rao D-attachment existence scan with B5 domain-map options |
| `scripts/sample_cea_surface.py` | 201 | Offline RocketCEA chamber-property table sampler with holdout validation for the differentiable layer |
| `tests/experiments/nozzle/seed_diagnostic.py` | 175 | Falsification test separating seed versus optimizer causes of the Rao BVP convergence stall |
| `tests/experiments/nozzle/theta_b_freeze_sweep.py` | 115 | Sweeps theta_B freezing with full D continuity around the smooth stationary-DE root |
| `tests/experiments/nozzle/theta_b_picard_probe.py` | 156 | Historical theta_B Picard refresh diagnostic retained for reproducibility |
| `tests/experiments/feed/tvc_sizing_study.py` | 443 | Gimbal TVC actuator torque, speed and inertia sizing study over generated contours |

## X06 - Knowledge graph, source corpus and packaged data assets (non-CLI)

| File | Lines | Role |
|---|---:|---|
| `knowledge/AGENTS.md` | 0 | Agent manual: authority, provenance contract and mandatory startup protocol |
| `knowledge/README.md` | 0 | Overview and quick-start commands for the local scientific memory |
| `knowledge/kg.sh` | 0 | Shell wrapper picking a venv python and launching the kg CLI module |
| `knowledge/kg/__init__.py` | 17 | Package docstring pointing to kg.api and kg.cli entry points |
| `knowledge/kg/__main__.py` | 3 | python -m kg shim delegating to kg.cli main (inferred) |
| `knowledge/kg/api.py` | 791 | Stable programmatic query interface with explicit states for development agents |
| `knowledge/kg/candidate_links.py` | 33 | Identity checks linking reviewed source objects to raw parser candidates |
| `knowledge/kg/chunker.py` | 241 | Structure-aware Markdown chunking preserving pages, headings, tables and offsets |
| `knowledge/kg/cli.py` | 246 | kg command-line interface: search, docs, equations, models, sync, rebuild, validate |
| `knowledge/kg/completion.py` | 438 | Deterministic completeness, equation-fidelity and model-audit report products |
| `knowledge/kg/constant_basis.py` | 46 | Separates legacy survey constant labels from independently reviewed source basis |
| `knowledge/kg/coverage.py` | 160 | Lightweight evidence coverage counts without claiming completeness |
| `knowledge/kg/enrichment.py` | 440 | Docling derivative handling as inspectable candidates, never automatic source support |
| `knowledge/kg/execution_reach.py` | 142 | Bounded static call-site evidence for model-to-symbol reach |
| `knowledge/kg/freshness.py` | 77 | Preserves historical extraction records while invalidating changed evidence |
| `knowledge/kg/graph.py` | 563 | Graph assembly from JSON stores, traversal and JSON/GraphML export |
| `knowledge/kg/ground.py` | 90 | Deterministic grounding of LLM-extracted objects against source chunk text |
| `knowledge/kg/harvest_artifacts.py` | 341 | Observes existing design and injector CAD reports as report assertions |
| `knowledge/kg/harvest_constraints.py` | 192 | Harvests MDO constraint manifest into requirement, constraint and gate records |
| `knowledge/kg/harvest_extraction.py` | 1243 | Stores grounded literature equations, claims, validity and constants with model links |
| `knowledge/kg/harvest_parameters.py` | 400 | AST observation of electric-feed technology defaults without importing models |
| `knowledge/kg/harvest_registry.py` | 230 | Harvests lrekit.verification.registry into Model records and software links |
| `knowledge/kg/harvest_repo.py` | 382 | Deterministic harvest of docs, tests, capabilities and corpus citations |
| `knowledge/kg/harvest_survey.py` | 245 | Verifies agent source surveys into SoftwareComponent records against working tree |
| `knowledge/kg/implementation_freshness.py` | 273 | Refreshes Python AST locators for implementations without certifying science |
| `knowledge/kg/index.py` | 292 | Hybrid FTS5 plus LSA retrieval index with reciprocal-rank fusion |
| `knowledge/kg/ingest.py` | 311 | Document manifest construction and idempotent incremental chunk ingestion |
| `knowledge/kg/locks.py` | 133 | Advisory flock locks for store writers and single heavy Docling job |
| `knowledge/kg/packet_queue.py` | 330 | Atomic role-gated state machine for bounded source review packets |
| `knowledge/kg/pipeline.py` | 131 | End-to-end rebuild of deterministic knowledge-system layers |
| `knowledge/kg/quantities.py` | 326 | Canonical physical quantity, symbol and unit normalisation table |
| `knowledge/kg/record_identity.py` | 167 | Exact-identity merging of duplicate survey facets without inventing support |
| `knowledge/kg/report.py` | 97 | Markdown ingestion and validation report rendering from JSON reports |
| `knowledge/kg/schema.py` | 245 | Versioned knowledge-system ontology, node/edge types and status vocabularies |
| `knowledge/kg/standards.py` | 90 | Standards registry: held documents versus BLOCKED_ON_STANDARD |
| `knowledge/kg/store.py` | 258 | Atomic sorted JSON stores plus SQLite FTS5 chunk store |
| `knowledge/kg/validate.py` | 262 | Quality-control checks writing validation reports |
| `knowledge/kg/variable_links.py` | 24 | Rebuilds conservative quantity aliases without rewriting source mathematics |
| `knowledge/kg/verification_freshness.py` | 110 | Binds source verification to fingerprints of inspected scientific content |
| `knowledge/kg/work_journal.py` | 99 | Append-only journal of role/model task attribution and acceptance state |
| `knowledge/kg/work_queue.py` | 114 | Deterministic resumable review queue, never invoking an LLM |
| `knowledge/tools/__init__.py` | 1 | Package marker for local knowledge-system command-line utilities |
| `knowledge/tools/agentic_packet.py` | 571 | Deterministic s1-s3 role operations moving one source packet through ingestion states |
| `knowledge/tools/bank_extraction.py` | 173 | Banks workflow journal extraction results into knowledge stores idempotently |
| `knowledge/tools/build_audit_inventory.py` | 90 | Builds implementation/source inventory and provisional evidence-debt review queue |
| `knowledge/tools/build_completion_reports.py` | 18 | Writes corpus completeness, equation fidelity and model source audit reports |
| `knowledge/tools/docling_batch.py` | 182 | Checkpointed bounded local Docling job planner and runner |
| `knowledge/tools/docling_enrich.py` | 0 | Runs one bounded local Docling formula/table enrichment and banks derivatives |
| `knowledge/tools/docling_formula_regions.py` | 0 | Re-enriches manually located PDF formula regions via local Docling |
| `knowledge/tools/fresh_checkout_check.py` | 0 | Proves a commit of knowledge changes yields a working fresh checkout |
| `knowledge/tools/kg_pipeline.py` | 0 | Compatibility entry point sharing the CLI pipeline |
| `knowledge/tools/packet_queue.py` | 0 | Command-line access to the persistent source-packet state machine |
| `knowledge/tools/pdf_to_markdown.py` | 0 | Faithful PDF to agent-readable Markdown converter for the propulsion corpus |
| `knowledge/tools/session_completion_report.py` | 0 | Writes session completion report from packet queue, integration logs and journal |
| `knowledge/tools/session_replay_check.py` | 0 | Replays reviewed session files proving the stores do not change |
| `knowledge/tools/sync_corpus.py` | 0 | Idempotently synchronises propulsion_texts PDFs with generated Markdown |
| `knowledge/tools/text_layout.py` | 0 | Structure-preserving fallback renderer for PDFs the Markdown parser cannot read |
| `knowledge/tools/trace_flow.py` | 0 | Traces one source object end to end through the knowledge system |
| `knowledge/tools/triage_pdfs.py` | 0 | Checkpointed native-text sampling isolating malformed PDFs with timeouts |
| `knowledge/tools/validate_audit_replay_2026_10_16.py` | 0 | Replays completed 2026-10-16 audit journals and verifies preserved content |
| `knowledge/tools/work_journal.py` | 0 | Appends to or shows the agentic work journal |
| `src/lrekit/data/tools/acquire_sources.py` | 165 | Standalone tool acquiring research sources with byte-identity receipts, independent of LREKit physics |
| `src/lrekit/data/tools/verify_collection.py` | 186 | Checks source integrity and registry coverage without certifying model validity |

## X07 - Test harness and regression arg files

| File | Lines | Role |
|---|---:|---|
| `.github/workflows/publish.yml` | 98 | CI workflow building and publishing the package to TestPyPI and PyPI on version tags |
| `.github/workflows/test.yml` | 95 | CI workflow running knowledge, non-slow and slow-nozzle test jobs |
| `examples/cli/lox_rp1_13kn_copper_316l_full_engine.args` | 73 | Full-engine CLI regression argument file for 13 kN LOX/RP-1 copper/316L |
| `examples/cli/test_13kn_movable_pintle.args` | 72 | Variant args exercising Son continuous-gap movable pintle on the 13 kN benchmark |
| `examples/cli/test_13kn_rao_bvp_flowfield.args` | 44 | Variant args running rao-bvp contour with flow-field render and construction diagrams |
| `examples/cli/test_13kn_sealevel_regen_allstl.args` | 156 | Authoritative 13 kN sea-level benchmark args with regen sizing, pintle STEP, pump CAD |
| `knowledge/tests/test_kg_agentic_controls.py` | 0 | Tests agentic workflow controls |
| `knowledge/tests/test_kg_agentic_packet_ops.py` | 0 | Tests agentic packet role operations |
| `knowledge/tests/test_kg_artifact_integration.py` | 0 | Tests engineering artifact integration into the graph |
| `knowledge/tests/test_kg_artifacts.py` | 0 | Tests engineering artifact records |
| `knowledge/tests/test_kg_audit_fixes.py` | 0 | Regression tests for audit-driven knowledge system fixes |
| `knowledge/tests/test_kg_bank_gate.py` | 0 | Tests the banking gate for extracted records |
| `knowledge/tests/test_kg_candidate_links.py` | 0 | Tests candidate link generation |
| `knowledge/tests/test_kg_citation_support.py` | 0 | Tests citation support checks |
| `knowledge/tests/test_kg_claim_quantity_tokens.py` | 0 | Tests quantity tokens in claims |
| `knowledge/tests/test_kg_completeness.py` | 0 | Tests corpus completeness reporting |
| `knowledge/tests/test_kg_constant_basis.py` | 0 | Inferred: tests numerical tuning constants are not treated as physical calibration |
| `knowledge/tests/test_kg_constraint_citations.py` | 0 | Tests resolved report keys cannot validate engineering thresholds |
| `knowledge/tests/test_kg_coverage_constants.py` | 0 | Tests coverage constants |
| `knowledge/tests/test_kg_derivation_identity.py` | 0 | Tests derivation identity |
| `knowledge/tests/test_kg_derivation_links.py` | 0 | Tests derivation links |
| `knowledge/tests/test_kg_docling_attempts.py` | 0 | Tests Docling attempt records |
| `knowledge/tests/test_kg_enrichment.py` | 0 | Tests enrichment records |
| `knowledge/tests/test_kg_execution_reach.py` | 0 | Tests execution reach of code |
| `knowledge/tests/test_kg_extraction.py` | 0 | Tests extraction |
| `knowledge/tests/test_kg_implementation_freshness.py` | 0 | Tests implementation freshness |
| `knowledge/tests/test_kg_minicorpus.py` | 0 | Mini-corpus end-to-end pipeline tests |
| `knowledge/tests/test_kg_multisource_reviews.py` | 0 | Tests multi-source reviews |
| `knowledge/tests/test_kg_parameters.py` | 0 | Tests parameter harvesting |
| `knowledge/tests/test_kg_parser_review_history.py` | 0 | Tests parser review history |
| `knowledge/tests/test_kg_pdf_conversion.py` | 0 | Tests PDF conversion |
| `knowledge/tests/test_kg_pipeline.py` | 0 | Tests the pipeline |
| `knowledge/tests/test_kg_record_identity.py` | 0 | Tests record identity |
| `knowledge/tests/test_kg_registry.py` | 0 | Tests model registry harvesting |
| `knowledge/tests/test_kg_relationship_states.py` | 0 | Tests relationship states |
| `knowledge/tests/test_kg_repository_tracking.py` | 0 | Tests repository tracking |
| `knowledge/tests/test_kg_retrieval.py` | 0 | Tests retrieval |
| `knowledge/tests/test_kg_review_provenance.py` | 0 | Tests review provenance |
| `knowledge/tests/test_kg_session_replay.py` | 0 | Tests session replay |
| `knowledge/tests/test_kg_system.py` | 0 | Two-layer system tests for the knowledge system |
| `knowledge/tests/test_kg_test_classification.py` | 0 | Tests test classification |
| `knowledge/tests/test_kg_variable_units.py` | 0 | Inferred: tests overloaded symbols stay distinct by units and meaning in variable links |
| `knowledge/tests/test_kg_verification_freshness.py` | 0 | Tests independent source verdicts are scoped to the assertion inspected |
| `tests/components/cooling/test_advanced_thermofluids.py` | 135 | Tests cyclic modulus, radiation, annular network, real-fluid backend, Zuber CHF and regen integration |
| `tests/application/test_bartz_heat_flux.py` | 168 | Pins full Bartz 1957 gas-side heat-transfer formula and physical trends |
| `tests/verification/test_benchmarks.py` | 152 | Benchmark manifests, curves, reports, CLI and Rao 1958 table checks |
| `tests/thermochemistry/test_cea_catalog.py` | 321 | Hash-pinned CEA catalog resources and sentinel wiring tests |
| `tests/thermochemistry/test_cea_pinned.py` | 88 | Contract tests for parity-aligned pinned chamber thermochemistry |
| `tests/thermochemistry/test_cea_surface_validation.py` | 217 | Held-out CEA value and gradient gate without RocketCEA |
| `tests/components/chamber/test_chamber_geometry.py` | 226 | Chamber shoulder factor, throat spec, L* volume and contour geometry tests |
| `tests/components/nozzle/test_characteristic_pairing.py` | 184 | Characteristic invariant pairing regression from the 2026-06-11 correction |
| `tests/application/test_cli_module_exit.py` | 19 | Ensures python -m raosim preserves CLI process exit status |
| `tests/contracts/test_module_identity.py` | 210 | One module object per implementation across canonical, legacy and lrekit import routes; lazy imports; entry points |
| `tests/contracts/test_packaging.py` | 115 | (slow) Wheel contents and a clean-venv install running every console script outside the checkout |
| `tests/components/nozzle/test_compare_nasa_reference.py` | 100 | Tests NASA reference comparison script provenance and artifacts |
| `tests/components/nozzle/test_comparison.py` | 94 | Tests for the nozzle comparison module |
| `tests/components/nozzle/test_conical.py` | 95 | Tests for the conical nozzle module |
| `tests/application/test_design_v2.py` | 1308 | Large suite for design_nozzle_v2 inputs, specs, hashing and solver wiring |
| `tests/application/test_design_workflow.py` | 203 | Design workflow and export profile tests including subprocess CLI runs |
| `tests/components/pump/test_electric_technology.py` | 218 | One electric-drive technology set shared by both pipelines |
| `tests/application/test_engine.py` | 58 | Tests for the engine performance module |
| `tests/application/test_fatigue.py` | 156 | Coffin-Manson low-cycle fatigue model and gating policy tests |
| `tests/application/test_fidelity_scope_gates.py` | 162 | Scope gates for separation topology and milling practice |
| `tests/assembly/test_flange_cad.py` | 127 | Chamber flange solid and fastener callout tests |
| `tests/application/test_flow_viz.py` | 175 | Steady flow-field visualisation tests built on the MOC net |
| `tests/thermochemistry/test_frozen_flow.py` | 728 | Thermally-perfect frozen-composition quasi-1-D flow contracts |
| `tests/application/test_gas_dynamics.py` | 105 | Tests for the gas dynamics module |
| `tests/components/injector/test_injector.py` | 1468 | Pintle injector hydraulic sizing and gate tests |
| `tests/components/injector/test_injector_export.py` | 935 | Pintle reference geometry and output package tests |
| `tests/verification/test_input_validation.py` | 60 | Workflow-independent engine input validation tests |
| `tests/assembly/test_interface.py` | 338 | Injector/chamber bolted interface tests via design_nozzle_v2 |
| `tests/numerics/test_jax_assembly_parity.py` | 225 | J2 gate: assembled JAX residual parity with reference |
| `tests/numerics/test_jax_bvp.py` | 180 | J3 gate: differentiable BVP solve infrastructure tests |
| `tests/numerics/test_jax_convergence.py` | 320 | J4 convergence of the differentiable backend |
| `tests/numerics/test_jax_moc_kernel.py` | 275 | J3b differentiable throat-arc kernel march tests |
| `tests/numerics/test_jax_primitives.py` | 115 | J1 gate: JAX primitives reproduce the NumPy references |
| `tests/numerics/test_jax_residual_parity.py` | 167 | J2 gate: JAX residual leaves reproduce references |
| `tests/numerics/test_jax_sensitivities.py` | 172 | J6 v1 smoke and known-sign sensitivity gates |
| `tests/numerics/test_jax_thermal_design_opt.py` | 223 | Constrained differentiable nozzle design with JAX thermal model |
| `tests/numerics/test_jax_theta_b_solve.py` | 148 | J3b-2 theta_B as a solved unknown inside the BVP |
| `tests/application/test_job_execution_integrity.py` | 334 | Regression checks at public job, CLI, verdict and report boundaries |
| `tests/application/test_job_reference_parts.py` | 184 | Unified jobs export reference CAD with an honest part manifest |
| `tests/assembly/test_joint_design.py` | 152 | Joint wall and channel optimizer in thermal_design |
| `tests/components/nozzle/test_legacy_io.py` | 135 | Legacy NASA/JHU file parsers (Tecplot, kernel and center-out tables) |
| `tests/application/test_lrekit_guided.py` | 115 | Guided interview tests: describe the engine, not the solver |
| `tests/application/test_lrekit_job_cli.py` | 174 | Job verbs and their compatibility guarantee tests |
| `tests/assembly/test_mass_ledger.py` | 796 | Mass ledger and its differentiable MDO mirror tests |
| `tests/application/test_materials.py` | 150 | Materials catalog and SP-125 eq. 4-31 wall-stress relation tests |
| `tests/optimization/test_mdo_bounds.py` | 191 | Design bounds derived from the architecture tests |
| `tests/optimization/test_mdo_build.py` | 164 | Tests engine part manifest and achieved-fidelity contract; partial assemblies never labelled complete |
| `tests/optimization/test_mdo_cea_surfaces.py` | 89 | Phase-2 gates for wiring CEA (P_c, O/F) property surfaces into the MDO |
| `tests/optimization/test_mdo_constraints.py` | 192 | Gates single constraint manifest and tri-state feasibility contract |
| `tests/optimization/test_mdo_contour_validation.py` | 833 | Tests post-solve validation of optimized TOP contour against variational Rao contour; fail-closed gates |
| `tests/optimization/test_mdo_convergence.py` | 122 | Station-refinement grid convergence study tests re-solving a fixed design at doubled station counts |
| `tests/optimization/test_mdo_coolant_htd.py` | 207 | Tests supercritical coolant heat-transfer-deterioration screen (Nasuti and Pizzarelli) |
| `tests/optimization/test_mdo_cooling.py` | 388 | Phase-4a acceptance gate for differentiable cooling block versus NumPy and host oracles |
| `tests/optimization/test_mdo_cooling_gate.py` | 78 | Phase-4a autodiff-versus-finite-difference gate for cooling outputs |
| `tests/optimization/test_mdo_custom_propellant.py` | 252 | Tests runtime custom-propellant registration and its wall-side mechanism |
| `tests/optimization/test_mdo_engine.py` | 242 | Phase-7 coupled whole-engine integration gate for the four differentiable blocks |
| `tests/optimization/test_mdo_envelope.py` | 178 | Tests SP-125 envelope (bounding cylinder) screen |
| `tests/optimization/test_mdo_execute.py` | 407 | Tests single execute entry point and four-valued verdicts across differentiable and traditional paths |
| `tests/optimization/test_mdo_injector.py` | 155 | Phase-5 parity and branch-structure gate for differentiable pintle injector block |
| `tests/optimization/test_mdo_injector_gate.py` | 72 | Phase-5 autodiff-versus-finite-difference gate for pintle injector block |
| `tests/optimization/test_mdo_job.py` | 381 | Tests the versioned engine-job contract all front ends compile into |
| `tests/optimization/test_mdo_material_map.py` | 166 | Tests typed catalog-to-traced material property mapper and its atomicity |
| `tests/optimization/test_mdo_metrics.py` | 124 | Tests selectable-output metric registry and derived constraint metrics |
| `tests/optimization/test_mdo_nlp.py` | 88 | Phase 8/9 gate for epsilon-constraint NLP, exact constraint Jacobians |
| `tests/optimization/test_mdo_nlp_robustness.py` | 568 | Tests fail-closed NLP evaluation, independent KKT check and multi-start using analytic engines |
| `tests/optimization/test_mdo_nozzle.py` | 199 | Gates typed MDO contour-provider adapter for nozzle geometry |
| `tests/optimization/test_mdo_objectives.py` | 37 | Checks mass-objective names match the hardware subtotal they select |
| `tests/optimization/test_mdo_of_variable.py` | 260 | Tests oxidizer/fuel ratio as an MDO design variable |
| `tests/optimization/test_mdo_pintle_component.py` | 307 | Tests pintle as optimizable MDO component with host-parity gates |
| `tests/optimization/test_mdo_propellants.py` | 113 | Tests propellant as first-class MDO input, not single-engine hardcoding |
| `tests/optimization/test_mdo_properties.py` | 206 | Phase-2 gates for C1 monotone interpolant values, gradients, shape and domain flags |
| `tests/optimization/test_mdo_pump.py` | 164 | Phase-6 duty/power parity and C1 efficiency gate for pump/feed block |
| `tests/optimization/test_mdo_pump_gate.py` | 84 | Phase-6 autodiff-versus-finite-difference gate for pump and feed outputs |
| `tests/optimization/test_mdo_resolved_inputs.py` | 328 | Tests single host-side resolved input contract (remediation item 9) |
| `tests/optimization/test_mdo_roughness_contract.py` | 204 | Tests hydraulic roughness k_s is a length, with legacy mission roughness handoff |
| `tests/optimization/test_mdo_schema.py` | 172 | Phase-1 gate that schema pytrees and scaling are jit/jacfwd/jacrev clean |
| `tests/optimization/test_mdo_skeleton.py` | 184 | Walking-skeleton gates: coupled residual converges and matches NumPy closed-form oracle |
| `tests/optimization/test_mdo_snapshot.py` | 1261 | End-to-end tests for versioned MDO/traditional output snapshot bridge |
| `tests/optimization/test_mdo_state.py` | 250 | Contract tests for fixed-shape pure-JAX EngineState |
| `tests/optimization/test_mdo_structures.py` | 159 | Differentiable structural-screen parity and constraint wiring tests |
| `tests/optimization/test_mdo_variable_policies.py` | 223 | Tests that pins and requested bounds govern the evaluated optimization problem |
| `tests/components/nozzle/test_moc.py` | 393 | Tests MOC unit processes, bounded-net flowfield entry point and deprecated front march |
| `tests/components/nozzle/test_moc_bounded_net.py` | 399 | Topology and oracle gates for the bounded characteristic net |
| `tests/components/nozzle/test_moc_diagrams.py` | 98 | Tests MOC/Rao construction diagrams: throat kernel fan, BDE topology and net |
| `tests/components/nozzle/test_moc_topology.py` | 152 | Phase 12.6 tests for full-form RaoTopology at the reference design point |
| `tests/verification/test_model_registry.py` | 234 | Tests provenance registry contracts for physical equations and heuristics |
| `tests/components/injector/test_movable_pintle.py` | 208 | Tests Son-geometry and static-actuation contracts for movable pintles |
| `tests/components/nozzle/test_nasa_kernel.py` | 475 | Phase 12.4 tests for NASA-style kernel build (CalcRRCsAlongArc port) |
| `tests/components/nozzle/test_nasa_kernel_march_parity.py` | 255 | NASA kernel-march oracle parity tests (kernel half of J6 gate) |
| `tests/components/nozzle/test_nasa_moc_topology.py` | 65 | Tests NASA-style BD mass flux and D-to-E left-Mach segment closure |
| `tests/components/nozzle/test_nasa_port.py` | 202 | Regression tests of the Python NASA MOC port against NASA/JHU ground-truth wall grids |
| `tests/components/nozzle/test_nozzle_geometry.py` | 121 | Geometric correctness tests for nozzle contour continuity, dimensions and monotonicity |
| `tests/application/test_openfoam_export.py` | 295 | Determinism, provenance and fail-closed tests for OpenFOAM spray case export |
| `tests/application/test_openfoam_handoff.py` | 288 | Inferred: tests for OpenFOAM VOF handoff evidence types and gates |
| `tests/application/test_phase6_coupled_wall.py` | 462 | Phase 6 tests for coupled wall physics inside the BVP |
| `tests/components/nozzle/test_plotting.py` | 374 | Phase 13 smoke tests for plotting outputs |
| `tests/components/pump/test_pump_cad_brep.py` | 295 | Tests for true B-rep pump CAD named assemblies and re-import |
| `tests/components/pump/test_pumps.py` | 808 | Pump CAD package gates: watertight meshes, meanline-consumed dimensions (Phase 0 plan) |
| `tests/components/nozzle/test_rao_chart_benchmark.py` | 423 | Phase 7 Rao TOP chart benchmark sweep tests |
| `tests/components/nozzle/test_rao_convergence_domain.py` | 679 | Inferred: tests of Rao existence-scan convergence domain, subprocess and radial-increment source |
| `tests/components/nozzle/test_rao_existence_scan.py` | 220 | Inferred: tests of the Rao existence scan module and CLI |
| `tests/components/nozzle/test_rao_optimizer.py` | 63 | Inferred: tests of rao_optimizer optimize_wall |
| `tests/components/nozzle/test_rao_residuals.py` | 77 | Inferred: tests of axisymmetric characteristic and wall-tangency residual functions |
| `tests/components/nozzle/test_rao_resolution_convergence.py` | 217 | Inferred: tests of Rao field convergence tolerances and contour reliability |
| `tests/components/nozzle/test_rao_stationarity.py` | 250 | Phase 3 tests of algebraic Rao stationarity and differential consistency |
| `tests/components/nozzle/test_rao_valid_region.py` | 432 | Tests for Rao 1999 radial-increment validity gate and diagnostic plots |
| `tests/components/nozzle/test_rao_variational.py` | 174 | Unit tests for the rao_variational module |
| `tests/components/nozzle/test_rao_variational_moc.py` | 1066 | Inferred: large tests of MOC-net links and Rao variational endpoint/residual ablations |
| `tests/components/cooling/test_regen_cad.py` | 138 | Tests for cooling-aware B-rep liner, ribs and jacket solid |
| `tests/components/cooling/test_regen_cooling.py` | 558 | Tests for Sieder-Tate coolant side and coupled 1-D wall regen cooling |
| `tests/components/cooling/test_regen_geometry.py` | 129 | Tests for 3-D regen nozzle wall-surface geometry |
| `tests/components/cooling/test_regen_profile.py` | 221 | Tests for station-wise RegenWallProfile and helix-length pressure drop |
| `tests/components/cooling/test_regen_volumes.py` | 105 | Contract tests for the shared regenerative metal-volume kernel |
| `tests/verification/test_release_readiness.py` | 165 | Inferred: tests of release-readiness evidence manifest and gate evaluation |
| `tests/application/test_requirements.py` | 471 | Layer 0 gate tests: requirements in, honest coverage out |
| `tests/application/test_run_nozzle.py` | 1203 | Smoke test of the main runner solve-contour-wall-regen pipeline artifacts |
| `tests/components/injector/test_spray_benchmarks.py` | 262 | Provenance and readiness contracts for Radhakrishnan spray fixtures |
| `tests/components/injector/test_spray_breakup_evaporation.py` | 464 | Inferred: tests of spray droplet breakup and evaporation models |
| `tests/components/injector/test_spray_carrier_drag.py` | 313 | Tests of spray carrier interpolation, drag and dispersion determinism |
| `tests/components/injector/test_spray_coupling.py` | 182 | Inferred: tests of spray c-star fixed-point coupling solver |
| `tests/components/injector/test_spray_domain.py` | 72 | Inferred: tests of axisymmetric spray domain containment and radius |
| `tests/components/injector/test_spray_handoff.py` | 179 | Tests of typed liquid-parcel and gas-carrier cycle handoff |
| `tests/components/injector/test_spray_ledger.py` | 80 | Tests that conservation ledgers separate parcel closure from carrier closure |
| `tests/components/injector/test_spray_primary.py` | 467 | Geometry dispatch and conservation tests for primary spray parcel sources |
| `tests/components/injector/test_spray_solver.py` | 278 | End-to-end deterministic parcel-march and conservation tests |
| `tests/components/injector/test_spray_statistics.py` | 86 | Tests of deterministic multiplicity-aware spray statistics |
| `tests/components/cooling/test_thermal_design.py` | 180 | Tests of cooling-coupled contour selection in thermal_design |
| `tests/mission/test_trajectory.py` | 46 | Inferred: tests that stage delta-v is burnout velocity change |
| `tests/components/nozzle/test_transonic_kernel.py` | 253 | Phase 9 Kliegel-Levine transonic kernel tests |
| `tests/application/test_units.py` | 102 | Engineering-notation parsing tests with canonical SI storage |
| `tests/application/test_v2_features.py` | 209 | Tests for v2 features: wall_pressure, separation, trade_study |
| `tests/application/test_wall_physics.py` | 149 | Tests of SP-125 longitudinal inelastic buckling and related wall structural equations |
| `tests/application/test_wall_profile.py` | 134 | Tests of variable hot-wall and jacket profile sizing |

## X08 - Library code not reachable from the CLI (test-only or orphan modules)

| File | Lines | Role |
|---|---:|---|
| `src/lrekit/assembly/flange.py` | 354 | Builds the chamber-side flange solid, bolt pattern and orderable fastener callout |
| `src/lrekit/numerics/jax/bvp.py` | 102 | Solver-agnostic Optimistix Levenberg-Marquardt least-squares wrapper with implicit-diff solution |
| `src/lrekit/numerics/jax/design_opt.py` | 460 | Constrained differentiable nozzle design maximizing Cf over expansion ratio and throat curvature (BFGS penalty) |
| `src/lrekit/numerics/jax/pack.py` | 62 | Pack/unpack flat unknown vector to structured CE state for JAX solve |
| `src/lrekit/numerics/jax/reference.py` | 38 | Canonical M3.5Perf reference constants sizing static padded march shapes |
| `src/lrekit/numerics/jax/residuals.py` | 261 | Vectorized node-wise Rao/MOC residual leaves ported from rao_residuals and rao_variational |
| `src/lrekit/components/cooling/correlations.py` | 100 | Source-scoped thermal correlation selection and applicability reports with Bartz guidance bounds |
| `src/lrekit/mission/trajectory.py` | 111 | Optional 1-D vertical-ascent trajectory integrator with gravity, drag and atmosphere |
