# Deep Technical Project Audit — LREKit / RaoRocketSim

**Audit date:** 2026-08-25
**Repository:** `/Users/ibrahimshahid/Documents/Github/RaoRocketSim` (`github.com/ibrahimshahid1/RaoRocketSim`)
**Audited commit:** `f763fe1` "Implement Rao B3 postsolve validation and B5 domain map" (2026-08-24), working tree clean
**Method:** direct inspection of source, docs, benchmark data, generated artifacts, and git history, plus independent execution of the test suites, the CLI, the MDO optimizer, and the CAD exporters on this host.

Everything marked "verified" below was run by the auditor during this session, not read from a status file.

---

## 0. What was independently executed

Everything marked *verified* in this report was run by the auditor on this host during this session, not read from a status file.

| Check | Invocation | Result |
|---|---|---|
| Non-slow suite | `pytest -q -m "not slow"` | **pass** 2148 passed, 26 deselected, 3 warnings, 1420 s |
| Slow suite | `pytest -q -m slow` | **pass** 26 passed, 2148 deselected, 636 s |
| Collected total | `pytest --collect-only` | 2174 tests across 120 modules |
| Full engine pipeline | `run_nozzle @…13kn_copper_316l_full_engine.args` | **pass** completed; wrote 55 artifacts (STL, STEP, CSV, PNG, JSON) |
| STEP re-import | `cadquery.importers.importStep` | **pass** `wall.step` 1 valid solid, 848.583 cm³; injector assembly 5 valid solids, 3088.368 cm³ |
| MDO single point | `--engine-mdo --pc 3e6 --epsilon 8 --film-frac 0.10` | **ran** full report, 31 margins, RQ1 ablation; physics verdict FAIL at that user-chosen point |
| MDO optimizer *(API)* | `solve_min_mass(MissionSpec(), 190.0)` | **feasible** max_violation 4.9e-11, Isp 224.77 s, objective 35.81 kg |
| MDO optimizer *(CLI)* | `--engine-mdo-optimize --isp-min 190` | **fails** isolated to the CLI's 10 s `--burn-time` default — see §12 |
| Rao variational BVP | `--contour-method rao-bvp --backend jax --max-nfev 400` | **converged, refused** max_scaled 1.450e-04; export blocked on MOC closure (correct) |
| Published benchmark | `run_benchmark("rao1958_nozzle_b","rao")` | **pass 7/7** contour RMS 0.00888 Rt (tol 0.015); C_F 1.5769 vs published 1.5829 |
| Requirements layer | `--requirements --target-thrust 5000 --isp-min 200` | **ran** max_violation 1.66e-11; verdict INDETERMINATE with itemized SP-125 coverage |
| Model registry audit | `audit_model_registry(".")` | **pass** 61 entries, 17 subsystems, 0 missing sources, 0 incomplete |

---

## 1. Identifying the project

### What kind of project this is

LREKit is a **hybrid** of four things genuinely present in the tree, not four labels applied to one thing:

1. **Engineering design software.** A parameterized liquid-rocket-engine preliminary design tool that takes an operating point and emits sized geometry, a mass ledger, and manufacturable-format CAD.
2. **A simulation/modeling codebase.** Quasi-1-D gas dynamics, an axisymmetric method-of-characteristics (MOC) solver, a finite-dimensional Rao variational boundary-value problem, a 1-D+fin regenerative-cooling march, a Lagrangian spray parcel solver.
3. **A differentiable-optimization research project.** A JAX/Optimistix multidisciplinary-feasible (MDF) engine model whose converged implicit states are differentiated through the implicit function theorem, driven by SLSQP with exact reverse-mode Jacobians.
4. **A computational research investigation.** A study of the *Rao nozzle existence boundary* — the region of (expansion ratio, length, γ, ambient ratio) space where no shock-free thrust-optimum contour exists — and whether whole-engine optima migrate toward it.

It is **not** a wrapper around an external solver. Every numerical kernel — the MOC unit processes, the variational BVP residual assembly, the cooling march, the pintle hydraulics, the pump meanline, the B-rep construction — is implemented in this repository. JAX, Optimistix, SciPy, CadQuery/OpenCascade, CoolProp, and RocketCEA provide autodiff, root-finding, linear algebra, geometry Booleans, and fluid/thermochemistry backends; they provide none of the engineering models.

### What problem it addresses

Preliminary liquid-rocket-engine design is a *coupled* problem conventionally solved sequentially: pick chamber pressure and expansion ratio, size a contour, check cooling, size an injector, size the feed system, iterate by hand. Each discipline uses low-order screens, and the coupling — the cooling-jacket pressure drop that sets the pump rise, the film-cooling flow that buys wall temperature at the cost of c*, the wall thickness that trades conductance against structure — is resolved by iteration rather than simultaneous solution.

LREKit does two distinct things about that. The **traditional path** (`raosim/design.py:design_nozzle_v2`) runs the sequential chain but instruments it: every model carries a provenance record, every screen carries an applicability scope, and a gap that cannot be computed is reported as `unavailable` with a reason rather than defaulted. The **MDO path** (`raosim/mdo/`) turns the same physics into one differentiable residual system, solves the whole engine at once, and hands exact total derivatives to a gradient optimizer under 31 named constraints.

### What goes in, what comes out

**Inputs (traditional).** Chamber pressure, ambient pressure, throat radius *or* target thrust, expansion ratio, bell length percentage, propellant pair (or explicit oxidizer/fuel + O/F), contour method, chamber `L*`/contraction ratio, wall and jacket materials, coolant and channel geometry, injector architecture and discharge coefficients, feed architecture with tank pressures and line losses, electric drive and battery constants, CAD options. The CLI exposes **328 distinct flags**.

**Inputs (MDO).** A `MissionSpec` (thrust, propellant, ambient, burn time, materials) plus a 10-variable (11 with O/F) design vector: `Pc`, `eps`, `dp_f_frac`, `dp_o_frac`, `D_pintle`, `N_rpm`, `channel_width`, `channel_height`, `film_frac`, `t_wall`.

**Outputs.** A contour CSV; profile, cooling, separation, structural, injector, and spray diagnostic PNGs; watertight STL; unit-declared neutral B-rep STEP for nozzle wall, jacket, regen channels, five-part machined pintle injector, and pump package; a pintle dimension table and machining-gate report; a pump BOM and reference geometry; and a `summary.json` with 39 top-level sections (226 KB on the 13 kN sample) carrying the reliability record, model-provenance audit, physical-release readiness, and every screen's result.

### Major components

| Layer | Location | Role |
|---|---|---|
| Requirements (Layer 0) | `raosim/requirements.py` | SP-125 §2.1 requirement set → constraint rows, with coverage verdicts |
| Job / CLI | `run_nozzle.py` (6641 ln), `lrekit/job_cli.py`, `lrekit/guided.py` | 328-flag CLI, job verbs (`guided`/`run`/`validate`/`catalog`), interactive interview |
| Sequential orchestrator | `design.py:562 design_nozzle_v2` | the traditional design chain and its gates |
| Contour solvers | `rao_variational.py` (6179), `moc.py`, `nasa_moc.py` (3472), `conical.py`, `chamber_geometry.py` | Rao/TOP Bézier, direct MOC wall optimization, Rao variational BVP, NASA/JHU port |
| Thermofluids | `physics.py` (2343), `thermofluids.py`, `thermal_design.py` (1460), `regen_*.py` | Bartz, Sieder–Tate, fin efficiency, Darcy, coolant properties, structural + fatigue screens |
| Injector | `injector.py` (4349), `movable_pintle.py`, `spray/` (14 files) | pintle hydraulics, Son-2017 movable branch, Lagrangian spray parcel solver |
| Feed system | `pumps.py` (3916) | centrifugal meanline, inducer/NPSH, electric drive, battery |
| Mass | `mass_ledger.py` (1507) | configuration-controlled hardware mass, CAD-linked where geometry exists |
| CAD | `regen_cad.py`, `injector_coaxial_cad.py`, `pump_cad_brep.py` (1552), `engine_cad.py`, `export.py` | STL + B-rep STEP with re-import validity gates |
| Differentiable core | `raosim/jax/` (13 files, 3380 ln) | JAX primitives, MOC kernel, thermal, IFT wrappers, sensitivities |
| MDO layer | `raosim/mdo/` (30 files, 18 745 ln) | schema, property surfaces, grid, cooling, injector, pump, coupled solve, NLP, snapshot, postprocess |
| Provenance | `model_registry.py`, `validation.py`, `release_readiness.py` | 61-entry model registry, reliability labels, release-evidence gates |

### What distinguishes it from a simpler implementation

A minimal "Rao nozzle tool" is a few hundred lines: interpolate θ_n/θ_e off the Rao chart, fit a quadratic Bézier, compute C_F from isentropic relations, plot. That path exists here and is deliberately the *default*, labeled the trusted baseline. What sits around it:

- A **second, independent contour construction** that solves Rao's variational problem as a boundary-value problem — stationarity, characteristic compatibility, mass and length closure, D-state continuity — and reproduces Rao's own 1958 published Nozzle B coordinates to 0.0089 R_t RMS.
- **Two backends** for that solve (SciPy oracle, JAX/Optimistix Levenberg–Marquardt) with parity at ~1e-10, so gradients can be trusted against a non-differentiable reference.
- A **port of NASA/JHU's `MOC_Grid_BDE` C++ code** with bit-parity of the throat-arc kernel march (max |ΔM| ≈ 5e-10 over 120 nodes).
- A **coupled MDO layer** where the cooling-jacket Δp actually feeds the pump rise, and a single `jax.grad` flows through two nested implicit solves.
- A **refusal architecture.** Ten engine outputs are registered *unavailable with a stated reason* rather than estimated. Contour export is blocked when MOC closure fails. The multipoint job path *raises* rather than returning a single-point answer labeled as a throttle envelope.
- **Manufacturable-format output.** The injector export builds five solids once, measures each body's volume from the OpenCascade kernel, writes STEP, re-imports, and compares — so the reported injector mass is measured from the geometry that was exported, keyed by a `geometry_id` content hash.

---

## 2. The technical workflow as it exists today

### 2.1 Traditional design chain (the default path)

```text
CLI flags / .args file / job YAML
  → input_validation            bounds, unit, consistency checks
  → cea / propellants           γ, T_c, R, c*_ideal, densities
        (built-in constant-γ table | RocketCEA snapshot | strict frozen cp(T) table)
  → throat sizing               Rt from F/(C_F·Pc), or Rt given directly
  → contour construction        one of four methods (§2.2)
  → chamber_geometry            L*, contraction ratio, shoulder fillet (auto or scalar),
                                convergent cone, Ru/Rd throat arcs (Ru from Hall Cd target)
  → wall_pressure               quasi-1-D wall pressure along the contour
  → separation                  Summerfield / Kalt–Badal / Schmucker onset screens
                                (declares free-shock-only topology scope)
  → boundary_layer_displacement  δ* computed and REPORTED — not fed back
  → regen_profile / thermal_design  station-wise wall + channel sizing
  → regenerative_cooling_analysis   Bartz h_g (σ factor) → series wall circuit
                                (land fin efficiency tanh(mH)/mH) → Sieder–Tate h_c
                                (+ Shah–London laminar branch) → Darcy Δp march
  → structural_screen           SP-125 eq. 4-31 combined stress, buckling, Coffin–Manson
  → injector                    pintle orifice sizing, TMR, blockage factor, spray angle,
                                chug screen, atomization; feed-pressure ledger
  → pumps                       meanline duty, Ns/Nss/NPSH, inducer, tip speed,
                                motor / inverter / battery sizing
  → mass_ledger                 hardware masses, CAD-linked where geometry was built
  → export / *_cad.py           STL + B-rep STEP, re-import validity gates
  → release_readiness           physical-release evidence manifest check
  → summary.json + artifact set
```

Manual intervention required: the user supplies `Pc`, ε, `L*`, contraction ratio, channel geometry, wall thickness, pintle diameter, pump RPM, and roughly thirty other coupled decisions. Iteration between disciplines is the user's job on this path — precisely the gap the MDO and requirements layers address.

Key assumptions entering: calorically perfect constant-γ expansion (or fixed-composition variable-c_p on the Bézier path only); inviscid characteristic net; flat-plate boundary-layer displacement; empirical separation onset; one-dimensional coolant march with a land-fin correction; screening structural allowables.

### 2.2 The four contour methods

| Method | Construction | Status in code |
|---|---|---|
| `bezier` *(default)* | Upstream + downstream throat arcs, then a quadratic Bézier bell anchored to Rao-chart θ_n/θ_e interpolated on the digitized chart | **promoted** `preliminary_top_geometry`, `benchmark_validated`; deterministic and endpoint-exact |
| conical | Straight-cone reference | reference / comparison only |
| `moc` | Axisymmetric characteristic march coupled to a monotone spline wall under SLSQP / Nelder–Mead | **experimental** `experimental_moc_geometry` |
| `rao-bvp` | Rao variational/MOC BVP: NASA-topology seed → control-surface unknowns → stationarity + characteristic compatibility + mass/length closure + D-state continuity | **experimental** exports only when every gate passes |

The `rao-bvp` residual assembly (`rao_variational.py:3354`) stacks fifteen named residual groups — mass, length, transversality, stationarity, algebraic stationarity, left-Mach geometry, C+ and C− axisymmetric compatibility, CE geometry, smoothness regularization, penalties, and four coupled-wall blocks — each individually scaled and individually switchable, with a configurable `physics_weight` on the derivative-form groups. That structure is what makes the solve auditable: a failing design says *which* physical closure failed.

### 2.3 The differentiable MDO chain

```text
MissionSpec.for_thrust(F) / .for_propellant(name, F)
    architecture derived from physics, not hard-coded:
    At = F/(C_F·Pc); chamber length from L*/CR; channel count ∝ circumference at
    fixed manufacturable pitch; D_pintle sized to mid-band blockage factor;
    pump speed from specific speed ω = Ns(g0H)^¾/√Q  (bigger engines run slower)
  → default_design_space() → ScaledSpace unit-box affine scaling
  → solve_engine(x, mission):
        OUTER Optimistix Newton root on (Rt, mdot)                  [IFT-differentiated]
          → mdo.properties  C¹ Fritsch–Carlson / tensor-Hermite chamber surfaces
                            γ(Pc,O/F), T_c, c*_ideal — hash-pinned packaged CEA tables
          → mdo.nozzle      typed ContourProvider (analytic TOP) → station grid
          → mdo.cooling     INNER Newton root on stationwise T_wg vector    [IFT]
                            Bartz h_g (0.026, σ factor) → land-fin series circuit →
                            Sieder–Tate h_c → counterflow upwind finite-volume march;
                            liquid-film phase-change model, η_fc derived from
                            Knuth's critical flow-per-circumference
          → mdo.injector    orifice metering, TMR, blockage, Son-2017 two-branch
                            minimum area (never a differentiable min())
          → CLOSE THE HYDRAULIC EDGE:
                Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank
          → mdo.pump        meanline duty, C¹ η(Ns) surrogate, Nss/tip-speed screens,
                            Lee-2021 battery epigraph (energy- and power-limited
                            branches kept separate — no max() in the core)
          → mdo.mass        Pappus/shell chamber mass from the same station grid
          → mdo.constraints 31 rows: 23 hard, 7 mandatory final gates, 1 report-only
  → mdo.nlp.solve_min_mass  jacrev exact constraint Jacobian → SLSQP ε-constraint solve
  → mdo.nlp.pareto_frontier warm-started ε-sweep
  → mdo.state / snapshot    versioned fixed-shape output contract
  → mdo.postprocess.reevaluate → design_nozzle_v2 with pinned chamber thermochemistry
                            → authoritative contour, pump sizing, report, CAD
  → compare_snapshots       common-scalar and profile parity between pipelines
```

> **Verified on this host**
>
> `solve_engine` at Pc 3 MPa / ε 8 / film 10 % converges the outer state to residual 1e-11 and the cooling residual to 9.2e-12, closes the jacket→pump edge (Δp_regen 1.03 bar → fuel pump rise 33.03 bar), and reports all 31 margins. `solve_min_mass` at `MissionSpec()` and Isp floor 190 s reaches `max_violation` 4.9e-11 with `film_frac` = 0.183 active — the optimum is genuinely thermal-limited, as intended.

### 2.4 Where the two pipelines meet

`mdo/snapshot.py` (3703 lines) defines one versioned host contract that both the MDO solve and the traditional `design_nozzle_v2` run map into, with a field manifest, normalized profiles, and `compare_snapshots` for common-field parity. `mdo/postprocess.reevaluate` pins the MDO's exact solved γ/T_c/R/c* and property-surface fingerprint into the traditional workflow so the authoritative re-evaluation cannot silently drift onto different thermochemistry.

### 2.5 The research workflow (Rao existence boundary)

`rao_existence_scan.py` (2843 lines) is deliberately kept out of the JAX stack — pure NumPy, sandbox-runnable. It samples (θ_B, `kernel_d_fraction`) pairs, builds the NASA-style kernel, marches the DE characteristic from point D under three named D-attachment closures (`smooth`, `position`, `fan`), and records exit residuals plus Rao's Eq. (5) radial-increment validity per segment.

The generated artifact `rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json` (101 932 bytes, SHA-256 `a85de663…`) is a 65-knot (13 ε × 5 L%) map at fixed γ = 1.4, P_a/P_0 = 0.01, R_t = 0.020 m: **40 knots `converged_valid`, 25 `not_converged`**, zero `converged_invalid`/`numerical_failure`/`under_resolved`/`unevaluable`. Fail-closed lookup over 48 cell midpoints returns 23 `supported`, 11 `refused_not_converged`, 14 `unknown_under_resolved`.

---

## 3. What it can actually do today

### **A** Implemented and demonstrated

- **Rao/TOP Bézier contour generation** with interpolated chart angles, a chart-domain gate, and endpoint-exact construction. Verified by execution.
- **Full traditional engine sizing** from an operating point through contour, chamber, regen wall, thermal, structural, pintle injector, and electric pump to a 55-artifact package. Verified end-to-end.
- **B-rep CAD export with independent re-import validation.** Verified: generated STEP files re-import as valid solids with expected solid counts and volumes. Machining gates (radial cut-through, corner radius, minimum tool diameter, faceplate pocket depth, bolt land, bolt-manifold clearance, inlet velocity) evaluate and report pass/warn per feature.
- **The Rao variational/MOC BVP solve** on both backends, reaching `max_scaled` ≈ 1.4e-4 with converged mass/length closure. Verified.
- **Strict published-data benchmark against Rao 1958 Nozzle B**: geometry, contour RMS, direct-integration C_F, and design exit Mach all inside declared tolerances. Verified.
- **Coupled differentiable engine solve** with exact total derivatives through two nested IFT solves, matched to re-solved central differences. Verified.
- **ε-constraint NLP reaching a feasible KKT-scale point** with coking hard-enforced via film cooling. Verified through the Python API.
- **The requirements layer**: SP-125 §2.1 items mapped to constraint rows with `enforced`/`partially_enforced` classification, explicit "missing components" lists, and three verdict states including `INDETERMINATE`. Verified via CLI.
- **The model-provenance registry**: 61 entries, 17 subsystems, machine-audited for local source routing and heuristic labeling. Verified: audit passes.
- **A 224-file Markdown mirror of 190 unique PDFs (12 047 source pages)** with a per-paper extraction manifest and confidence flags — the citation substrate the docs route claims through.
- **Packaging**: `lrekit` 0.1.1 wheel and sdist built; four console entry points; CI running the non-slow suite on Python 3.12; a trusted-publishing release workflow; a `v0.1.0` tag.

### **B** Implemented but preliminary or approximate

- **All thermal, structural, and life models are screens.** Bartz + Sieder–Tate + fin is a 1-D+fin circuit, not conjugate heat transfer. SP-125 eq. 4-31 combined stress is static. The liner buckling model is explicitly labeled a rib-supported long-plate screening assumption, not SP-125 eq. 4-29 or a qualification calculation.
- **Separation is empirical onset only** — and the code says so: `separation_pattern_scope == "free_shock_separation_only"`, `restricted_shock_separation == "not_modelled"`, citing Frey & Hagemann for why an onset pressure ratio cannot represent RSS in exactly the contour family this tool generates.
- **The pump efficiency surface** is a smooth log-Gaussian η(N_s) surrogate calibrated to the SP-125 60–85 % band. The band is literature-verified; the peaked *shape* is a screening fit, and the code says so.
- **Chamber property surfaces cover four propellant pairs** (LOX/RP-1, LOX/LCH4, LOX/LH2, N2O4/MMH) at the chamber only. Coolant (T,p)/CoolProp surfaces are unwired, so real-fluid coolant HTD coverage for cryogens is registered unavailable and authoritative LOX/LCH4 and LOX/LH2 optimization stops at model-coverage preflight.
- **The film-cooling model** correctly rejects the classical gaseous ε = C(X/VR)^−0.8Re^0.2 correlations (an RP-1 film runs at VR ≈ 10⁻³, three orders below their calibration band) and implements a liquid phase-change model instead, with η_fc derived from Knuth's critical flow-per-circumference rather than fitted. Its own error bars are stated: Woodmansee & Hanratty measured Γ_cr ≈ 3× lower than Knuth; Stechman's model errs −20 %…+13 %.
- **The Lagrangian spray solver** is explicitly one-way: the carrier is prescribed and does not consume parcel source terms; droplet temperature and carrier energy are not solved.
- **Pump and injector CAD** is reference geometry for layout and trade review. The legacy triangle-mesh pump path self-labels `schematic_only`.

### **C** Partially implemented

- **Boundary-layer displacement is computed and never fed back.** `physics.boundary_layer_displacement` is called once in `design.py:716`, the result is reported, and the wall is not moved. Throat Reynolds number is not computed anywhere for the nozzle. No CLI flag exposes a viscous correction; `INTEGRATION_PROGRAM_2026-08-22.md` §0 documents this precisely, quoting the SP-8120 procedure the tool is missing.
- **The B3 post-solve contour validator** has a landed typed contract but no qualification. All three real probes (12×12 coupled, 24×24 BDE, 32×32 BDE) return `UNKNOWN`: the BVP converges, Rao Eq. (5) passes, full C-D-E performance is available, but forward-MOC neighbor-Mach smoothness measures 0.11810 and 0.13748 against a repository-policy limit of 0.10.
- **The B5 convergence-domain map** exists for exactly one fixed physical slice. 25 of its 65 knots did not converge. It is kernel-refinement evidence, not proof of full mesh or full-BVP convergence.
- **J6 v2 total design derivatives** remain deferred (`jax/sensitivities.py:24`). Sensitivities reach the control surface, not manufacturable wall coordinates.
- **Chamber mass from disjoint liner/ribs/jacket CAD** is partial; MDO injector mass and chamber/nozzle CAD-linked qualification remain unavailable, so whole-engine dry mass stays `None` plus a reason.
- **The requirements layer's three-layer design is one-third built.** Requirement mapping (layer 3) is implemented and CLI-reachable; O/F as a variable (part of layer 2) is implemented; architecture enumeration as an outer discrete loop and multi-start (layer 1) do not exist — `grep` finds no multi-start or architecture-enumeration code anywhere.

### **D** Currently under active implementation

Evidence from the last four commits (2026-08-06 → 2026-08-24) and the dated status heads:

- **Track B, contour and property closure.** B2 (typed `mdo/nozzle.py` adapter) and B4 (Rao valid/invalid boundary on export) landed 2026-08-22; B3 and B5 landed as contracts 2026-08-24 and are in the qualification stage now.
- **Track G, hardware mass and CAD.** G1 (real CadQuery volume/STEP round-trip for the injector) landed 2026-08-22, verified in this audit. G2 is in progress.
- **Honesty/scope gates.** The most recent commits added `manufacturing_limits.py` (milled-channel practice limits from Wadel 1997), the separation topology scope declaration, and the `mdo/metrics.py` unavailable-metric registry — all shipped as gates before the corresponding models.
- **Fidelity roadmap items §3.1, §4.1, §5.1** landed as scope/validity gates per the roadmap's 2026-08-24 status head; remaining items are analysis-only.

### **E** Planned or documented but not implemented

From `COMPLETION_PLAN.md`'s deduplicated backlog (Tracks A–G), explicitly TODO with no implementation in the tree:

- `mdo/multipoint.py` (Phase 10) — the largest single item; `mdo/job.py:542` currently *raises* on more than one operating point.
- `mdo/continuation.py` — pseudo-arclength continuation (Keller 1977) for tracking through folds.
- σ_min(R_u) and λ_min(H_red) diagnostics with a mesh-convergence gate.
- Quasi-2-D cooling (Phase 4b) as a post-optimum check; Betti-style conjugate coupling validation.
- Independent held-out CEA/CoolProp re-evaluation of the final optimum; discrepancy-correction/trust-region loop.
- Boundary-layer displacement iteration; variable-γ frozen MOC; FSS/RSS topology classifier; equilibrium chemistry; compressible integral BL; PNS validation layer; Rao adjoint (J6 v2).
- Traditional pump BOM mass model; configuration-controlled chamber/nozzle + injector mass models.
- A `ResolvedEngineConfiguration` object binding solved state → exported geometry → mass ledger → BOM by content hash (identified as the highest-leverage missing item).
- Igniter model **blocked** (no source in the corpus). Combustion-response function for stability **half-blocked** (SP-8113 gives the damping half, not the response half).
- L4 datum/feature-control scaffolding and tolerance-stack input contract **blocked on standards**.
- Side-load estimate — deliberately staying unavailable.

---

## 4. What was actually built

Single-author repository: 72 commits, all by Ibrahim Shahid, 2026-02-22 → 2026-08-24 (plus one 2025-06-22 file upload). Ten of the 72 are GitHub merge commits from five early feature branches. There is no evidence of any other human contributor in the artifacts.

#### Numerical method implementation

- Implemented Rao's 1958 optimum-thrust variational problem as a finite-dimensional BVP: control-surface unknowns, stationarity residuals from the thrust/mass-flow integrands with Lagrange multipliers λ₂/λ₃, transversality at the exit, axisymmetric C+/C− compatibility, mass and length closure, and continuity with an upstream characteristic kernel.
- Implemented an axisymmetric MOC solver (interior, axis, wall unit processes) and a Kliegel–Levine/Sauer transonic starting-line kernel.
- Ported NASA/JHU's `MOC_Grid_BDE` C++ throat-arc kernel and Rao B-D-E topology routines to stagnation-normalized Python (`build_kernel`, `calc_massflow_along_rrc`, `nasa_deriv`/`nasa_runge_kutta`, `find_point_e`, `calc_lrc_de`, `set_theta_b`).
- Reimplemented that kernel march in JAX with a padded grow-by-one row policy reproducing NASA's exact decision/width/pairing logic at bit parity (max |ΔM| ≈ 5e-10) while keeping static shapes, so d(BD)/dθ_B is exact to ~1e-8 versus finite differences.

#### Model derivation and correction

- Corrected the characteristic compatibility invariants across the whole codebase (commit `c9261ae`): enforced C+: d(θ−ν) = −S ds and C−: d(θ+ν) = +S ds with S = sinθ·sinμ/r, removing a spurious cosμ/cos(θ±μ) factor mirrored through `rao_residuals`, `moc`, and both JAX implementations.
- Identified and fixed the length-constraint formulation (commit `44953ee`): Rao's constraint is the exit station L = z_C + ∫cot(φ)dr, whereas the legacy Σdx double-counted x_D against the x_E pin, making the two constraints contradictory and driving `kernel_d_fraction` to the throat plane. The code comment records the observed symptom (kdf → 0.02 drift) and why it was invisible before the kernel fix.
- Removed a degenerate optimization basin (commit `3fa01b0`) by making left-Mach geometry exact by construction — reconstructing `ce.x` from dx/dr = 1/tan(θ+μ) and dropping x from the packed unknown vector (4n → 3n), so the solver could no longer satisfy geometry by moving x.
- Zeroed the CE Mach-monotonicity penalty under the characteristic formulation after determining the control surface crosses streamlines, so M need not be monotone along it, and that the penalty was strong enough to displace the correct solution branch.
- Fixed the folded-RRC mass integral (commit `1be73e2`) with a regularized product form so mass stays finite and monotone when dr = 0.
- Replaced the binned `pumps.py:_estimate_pump_efficiency` (a C0 step in Q, undifferentiable at bin edges) with a C¹ η(N_s) surrogate, demonstrating the property in a test where finite differences converge to the analytic gradient at a point where the binned estimator's FD diverges.
- Corrected the film-cooling model from a gaseous-coolant correlation family to a liquid phase-change model, retaining the gaseous VR as a pinned diagnostic so the inapplicability stays documented.
- Corrected the separation criteria to Östlund's Eq. 29/30 forms and relabeled the criterion attribution.
- Corrected the load-bearing literature attribution in the MDO plan (Rao 1958 vs Rao 1999) after reading the source PDFs rather than inferring from filenames.

#### System architecture and integration

- Designed and built the `raosim/mdo/` layer (30 modules) as a pure-numerical differentiable core with an explicit rule set: never substitute a Bézier contour for a failed Rao solve; pin the c* convention in one place; differentiate converged states via the IFT, never through solver iterations; keep discrete variables discrete in an outer enumeration; no `max()` in the differentiable core; every JAX block keeps a NumPy oracle with a ~1e-10 parity test; CAD generation is post-optimum only.
- Closed the §5 hydraulic edge — replacing a `regen_dp_allowance = 0` placeholder with the actual cooling-jacket Δp feeding the pump rise — and verified a single gradient flows through it.
- Built a versioned snapshot contract shared by both pipelines, with strict handoff validation, solved-chamber-state pinning, and digest-linked report/CAD metadata.
- Built the 31-row constraint manifest as a single owned object carrying row order, applicability, availability, scaling, optimizer role, and source identifier per row.
- Built a 61-entry model-provenance registry with a machine audit that fails when a declared local source is absent, a published entry has no local source route, or a repository-created heuristic is not labeled as one.
- Built the requirements layer mapping SP-125 §2.1's nine design parameters onto constraint rows with coverage classification.
- Built a job/CLI layer (`guided`, `run`, `validate`, `catalog`) as an additive translator over the existing 328-flag CLI, motivated in its own docstring by the observation that O/F optimization was implemented long before it was reachable from a command line.

#### Geometry and CAD

- Rebuilt the pintle injector CAD from a two-ring-manifold machined scheme to the coaxial five-part TRW/Nardi architecture after determining the old scheme could not realize a pintle's flow topology — an outer-radius ring manifold can neither feed the central bore nor stay sealed from it. The replacement carries a flow-separation audit asserting the circuits stay sealed until chamber exit.
- Built B-rep constructors for nozzle wall, regen channels with manifolds and ports, pump impeller/inducer/diffuser/volute with log-spiral camber blades, engine assembly, and flange interfaces — each with export → re-import → `isValid` / solid-count / volume gates.
- Implemented CAD-linked mass: the injector exporter builds each solid once, measures its volume from OpenCascade, and keys the ledger to a `geometry_id` content hash, with a STEP round-trip volume comparison.

#### Data and provenance infrastructure

- Assembled a 255-PDF literature corpus and converted 190 unique PDFs (12 047 pages) into a 224-file Markdown mirror with a per-paper extraction manifest, confidence flags, page-level traceability markers, and an index — so claims cite line numbers into extracted text rather than filenames.
- Built seven benchmark case manifests with source PDF SHA-256 hashes, per-metric tolerances and modes, and explicit `expected_physics_gaps`, plus eight digitized reference curves.
- Sampled and packaged four hash-pinned CEA property tables with an every-cell held-out value-and-gradient gate against RocketCEA.

#### Debugging and investigation

- Investigated the provenance of NASA's checked-in `TT'.out` reference fixture, traced the upstream commit timeline, and concluded the generating executable is not present in the local tree or the public history — then demoted that fixture from promotion authority to a historical overlay and made the visible source-port workflow the canonical reference track.
- Diagnosed why coking was infeasible throughout the old design box: an empirical probe established the wall is **coolant-enthalpy-limited, not film-coefficient-limited** (T_wc tracks coolant exit temperature; the full channel-geometry box moved it only 1241 → 1124 K), which identified film cooling — not channel geometry — as the lever. Adding `film_frac` made the constraint satisfiable at 7.7 % film and turned the Pareto front genuinely thermal-limited.
- Found and fixed two defects in the cooling march while completing it: an off-by-one/double-counted segment in the march quadrature, and a silent `land` clamp hiding the channel-fit constraint (now exposed as `land_min`).
- Found that `jacfwd` silently drops the tangent of a geometry-only `jnp.min` margin through jitted nested Optimistix root-finds, verified it against finite differences, and switched the constraint Jacobian to reverse mode with a documented warning not to revert.
- Found two dead design variables (zero Jacobian column *and* zero objective gradient) and removed or fixed them: `film_slot_height` (physically inert for a liquid film) removed; `D_pintle` made live by constraining blockage factor. The guide ships the re-check snippet.
- Found that `jax/primitives.py:mach_from_area_ratio` seeded its subsonic branch with a scalar, breaking array inputs, and fixed it to a shape-matched seed with scalar results bit-identical.
- Found that a 130 kN run had *no feasible point in the box* with hand-tuned 13 kN defaults, and derived the whole architecture from physics instead.

---

## 5. Technical depth

### 5.1 Solving Rao's variational problem as a BVP rather than reading his chart

**The problem.** Rao's 1958 optimum-thrust contour is the solution of a constrained variational problem: maximize thrust at fixed mass flow with the control surface forced to be a characteristic surface, via Lagrange multipliers. Rao states the optimality condition "is implicit in the solution" of the governing equations. Standard practice — including the default path here — replaces that with the Rao/TOP parabola: a chart of θ_n and θ_e fitted to the family of exact solutions.

**Why it is nontrivial.** The exact solve is a coupled two-point BVP over a control surface whose geometry, flow state, and multipliers are simultaneously unknown, closed by mass flow, length, transversality, and continuity with an upstream characteristic kernel. It has degenerate basins (the x-freedom removed in `3fa01b0`), contradictory constraint pairs when the length convention is wrong (fixed in `44953ee`), and regions where no shock-free solution exists at all.

**How it is addressed.** Fifteen named residual groups, each independently scaled and switchable, over a packed unknown vector; a NASA-topology seed from the ported kernel; two solver backends with parity tests; a constraint-weight ladder; and export gates that refuse geometry failing MOC closure, endpoint tangency, or Rao's Eq. (5) validity.

**How complete.** Genuinely solved for the strict benchmark case: Rao 1958 Nozzle B reproduces at 0.0089 R_t contour RMS with C_F 1.5769 against Rao's published 1.5829. Not solved as a promotable general path: `BENCHMARK_VALIDATED_AT_RELEASE = False`, MOC-net compatibility blocks export on the general case run in this audit, and the B3 validator returns `UNKNOWN` on all three real probes.

### 5.2 Exact total derivatives through a coupled engine with nested implicit solves

**The problem.** To hand a gradient optimizer exact derivatives of engine performance and constraints, the derivatives must flow through two nested root-finds — an outer Newton on (R_t, ṁ) and an inner Newton on a stationwise wall-temperature vector — plus an explicit injector/pump/battery chain.

**Why it is nontrivial.** Differentiating through solver iterations (unrolling) is numerically poor and memory-expensive; the correct approach is the implicit function theorem on the converged state. Doing that inside JAX requires the residual to be a pure function with static shapes, no host callbacks, and no non-smooth branches — which rules out `max()`, binned lookups, integer relaxation, and CAD Booleans, all of which appear naturally in this physics.

**How it is addressed.** Every implicit state is closed with `optx.root_find` and IFT-differentiated. The battery's two limiting branches are kept as separate epigraph quantities instead of a `max()`. The movable-pintle minimum area is kept as two exposed branches with a consistency inequality instead of a `min()`. Discrete counts (channels, slots, stages, blades, materials, bus voltage class) stay in an outer enumeration. The pump efficiency step function was replaced with a C¹ surrogate. Every JAX block keeps a NumPy oracle at ~1e-10 parity, and inner residuals are converged ≥100× tighter than optimizer feasibility.

**How complete.** Working and gated: AD versus re-solved central differences matches to <1e-4 (plan gate) and in practice ~1e-5 on the skeleton; forward and reverse agree; jit-safe. One documented sharp edge: forward mode drops a geometry-only `jnp.min` tangent through the jitted nested root-finds, so the constraint Jacobian is reverse-mode.

### 5.3 The Rao existence boundary as a computed object

**The problem.** Rao 1999 (AIAA 99-2584) Eqs. (2), (3), and (5) give a per-segment radial-increment criterion: the simultaneous solution of the two finite-difference compatibility equations cannot continue unless dR is positive, and Eq. (5)'s zero is the boundary. This is a *physical* existence statement, distinct from numerical non-convergence and distinct from optimality.

**Why it is nontrivial.** Three different failure modes look identical from outside a solver: the physical boundary (no shock-free solution exists), numerical solvability (σ_min of the residual Jacobian), and optimality (λ_min of the reduced Hessian). Conflating them produces false conclusions in either direction. The plan explicitly names this as the failure to avoid.

**How it is addressed.** `rao_radial_increment_validity` evaluates Eq. (5) directly by midpoint quadrature, returning per-segment predicted dR/R, the limiting segment, evaluability, and a specific reason for each unevaluable segment — refusing to clamp or substitute when M ≤ 1 or the radius is non-positive. Eq. (6) is retained only as a labeled legacy diagnostic. Contour export refuses invalid or unevaluable results.

**How complete.** Diagnostic 1 (physical validity) is done. Diagnostics 2 and 3 (σ_min, λ_min with a mesh-convergence gate) are not implemented. The domain map covers one fixed slice with 40/65 knots converged. Continuation through folds is not implemented.

### 5.4 Making a low-order screen's error bar visible instead of absorbing it

**The problem.** Gradient optimization against low-order screens drives designs into exactly the regions where those screens are least accurate. This is the plan's own stated risk.

**Why it is nontrivial.** The mitigations — constraining designs to a screen's calibration envelope, adding discrepancy margins, trust-region correction — all require knowing where the envelope is, which is a literature question, not a coding one.

**How it is addressed today, mostly by gates rather than models:** a chart-domain constraint preventing clamping from being read as extrapolation; a property-domain constraint keeping the optimizer inside the sampled table; a `film_slot_validity` diagnostic pinned by a test that keeps the gaseous-correlation inapplicability documented; a separation-topology scope field; milled-channel practice limits from Wadel 1997; a coolant-HTD applicability screen returning `unknown` for cryogens; and ten registered-unavailable outputs.

**How complete.** The gates are shipped. The higher-fidelity verification layers that would quantify the deltas (quasi-2-D cooling, conjugate coupling, held-out CEA/CoolProp re-evaluation, trust-region correction) are all TODO.

### 5.5 Hardware geometry that the mass number actually comes from

**The problem.** A mass ledger built from profile integrals and a CAD package built from the same parameters can silently disagree, and the optimizer will happily minimize the integral.

**How it is addressed.** For the injector: *build once* — the exporter constructs the five solids, measures each body's volume from OpenCascade, writes STEP, re-imports, compares volumes, and stamps a `geometry_id` hash. `cad_linked_mass_eligible` gates whether the ledger may use those numbers. Verified in this audit: measured vs round-tripped volumes agree to ~1e-9 relative on four of five parts and ~3e-5 on the injector body.

**How complete.** Done for the injector (plan item G1). Chamber mass from disjoint liner/ribs/jacket CAD is partial; the pump BOM mass model and configuration-controlled chamber/nozzle mass models are TODO; consequently whole-engine dry mass is `None` with a reason rather than a number.

---

## 6. Technical maturity

The right question for this project: *how far can its outputs be carried, and by whom?*

#### As engineering-design software

Its outputs are a credible starting point for detailed engineering — a solid model you can open in a CAD system, a channel layout passing millability checks, a mass breakdown, and a documented list of what has *not* been checked. They are not a basis for fabrication. Every contour is stamped `hardware_qualified = False`; the physical-release gate on the 13 kN sample blocked with 17 missing external-evidence requirements (CFD, CHT, FEA, drawings, proof test, cold flow, hot fire). The tool is explicit that it can *record* that evidence and cannot *create* it.

#### As a simulation/modeling codebase

The verification is unusually strong for a single-author research tool and the physical validation is thin — and the code says so in those words. Verified today: 2174 tests green; NumPy↔JAX parity at ~1e-10 across blocks; AD-vs-FD gates; a bit-parity port of a NASA reference kernel; one strict published-geometry benchmark passing on seven metrics. Absent: any CFD comparison, any conjugate heat-transfer comparison, any FEA, any experimental data. The credible regime is inviscid, calorically-perfect, attached axisymmetric flow at moderate area ratio for geometry, and screening-level for everything thermal, structural, and combustion-related.

#### As a differentiable-MDO research artifact

The machinery works and is gated. A single solve reaches a feasible point with scaled `max_violation` ~1e-11 and exact reverse-mode Jacobians. What is missing to make it a *research result* rather than a working machine: multipoint (Phase 10), independent higher-fidelity verification of the optimum (Phase 11 D1–D4), and the KKT stationarity computation the plan's own Phase-9 gate still asks for — the current code treats SciPy success plus feasibility as a KKT pass.

#### As a tool another person could use

Partly. It installs, has four console entry points, a guided interview, job verbs, a catalog, argument files, and 7600 words of README. But a user must still supply ~40 coupled decisions on the traditional path, the requirements layer covers only a screened subset of what it would need to answer "give me 5 kN at 250 s under 30 kg," and — as this audit found — the documented optimizer CLI invocation fails on its own default burn time.

#### As a research investigation into the Rao boundary

Early. Diagnostic 1 is a computed object; the domain map is one slice with 40/65 knots; diagnostics 2 and 3 and continuation do not exist; RQ3 and RQ4 have not been run. The prior-art caveat is already recorded in the plan: Rao 1999 Fig. 7 shows length-limited vacuum optima sitting *on* the valid boundary, so the novelty claim would have to be engine-level masking, not the existence of the effect.

---

## 7. Intended system, current build, and "done"

### **A** What exists now, as pieces of the eventual system

- The **physics layer** — contour, thermal, structural, injector, spray, feed — is the layer the eventual system consumes rather than replaces.
- The **differentiable core** is architecturally the right shape for the endpoint: `solve_engine(x, mission)` is a pure function, exactly what a `vmap`-over-operating-points multipoint layer needs.
- The **output contract** (`state.py` → `snapshot.py` → `postprocess.py`) is the seam through which higher-fidelity verification layers would report deltas, and it already carries the comparison machinery.
- The **constraint manifest** is a single owned object with per-row applicability and availability, so new constraints and gates attach at one place.
- The **CAD layer with re-import gates and geometry IDs** is the substrate for the configuration-controlled dataset the endpoint requires.
- The **provenance layer** is the substrate for every claim the endpoint would make.

None of these read as throwaway experiments. Each is a load-bearing piece of the target.

### **B** What is being built right now

**Contour closure (Track B).** B3's promotion gate is forward-MOC compatibility on real solves: neighbor-Mach smoothness is 0.118 and 0.137 against a 0.10 repository-policy limit. Until that passes, TOP-versus-variational discrepancies are not adjudication-eligible, and the analytic TOP provider remains the only promoted inner-loop contour. B5 needs slices at other γ, ambient ratio, throat geometry, and closure model, plus a fuller mesh-convergence study.

**Hardware mass and CAD (Track G).** G1 landed. G2 (chamber mass from disjoint liner/ribs/jacket CAD), G3 (configuration-controlled chamber/nozzle + injector mass models), and G4 (traditional pump BOM) are what stand between the current partial subtotal and a real dry mass. The plan promotes this above multipoint on the explicit reasoning that "a design tool that optimizes a partial mass subtotal is not a design tool."

**Scope and honesty gates.** Shipping continuously and ahead of their corresponding models, by rule.

### **C** What remains to reach the intended system

1. **Multipoint (Phase 10).** Shared hardware x_h, per-point controls x_k and states y_k, one stacked NLP, with separation/property/pump-efficiency surfaces evaluated at every point and the battery epigraph summed across points. The `job.py` guard comes out last.
2. **Whole-engine dry mass.** Every reported mass measured from built geometry or `None` with a reason — requiring G2–G4 plus the valves/lines/brackets/gimbal models specified in `PLAN_VALVES_LINES_BRACKETS_GIMBAL.md`.
3. **Post-optimum verification with signed deltas (Phase 11).** Quasi-2-D cooling on the candidate optimum reporting T_wg,land − T_wg,1D; a Betti-style conjugate pass; independent held-out CEA/CoolProp re-evaluation; field-specific parity tolerances and an adjudication rule written down *before* it is needed; and a discrepancy-correction/trust-region loop when they disagree.
4. **A resolved-configuration object.** A frozen `ResolvedEngineConfiguration` with a content hash over the solved state, consumed rather than re-derived by thermal, structures, mass, every CAD builder, BOM, and assembly, with `build_engine_parts` refusing to assemble parts carrying different hashes. This is what makes "the analyzed wall and the exported wall are the same wall" checkable rather than asserted.
5. **The fidelity ladder (Track F).** Boundary-layer displacement iteration with a throat-Reynolds validity gate; variable-γ frozen MOC via a `ThermoProvider` interface; an equilibrium option behind the same interface; an FSS/RSS topology classifier; a compressible integral BL; a PNS validation layer; the Rao adjoint / start-line→wall map.
6. **The Rao cliff track (Track E).** σ_min and λ_min diagnostics with mesh convergence; pseudo-arclength continuation; then RQ3 and RQ4 as runs on the finished tool.
7. **The experiment-to-model return path.** Absent from every plan until `INTEGRATION_PROGRAM` identified it. Without it, predicted combustion efficiency is not derivable from the corpus — SP-8089's own Table II records the pintle's status as "no correlations for level of mixing and spray size."

### What "done" concretely looks like

> **From COMPLETION_PLAN.md §13**
>
> The tool is complete when, for an arbitrary propellant and thrust class within its declared domain: every reported mass is measured from built geometry or is `None` plus a reason; every constraint states its applicability scope and its source; the optimum is verified by at least one independent higher-fidelity layer per discipline, with signed deltas reported; multipoint throttle envelopes are honored rather than refused; the contour provider declares where it is valid and refuses where Rao says no shock-free solution exists; and no number in any report is a screening surrogate without saying so.
>
> Concretely, the user-facing workflow at that endpoint: a user states requirements — "5 kN at sea level, Isp ≥ 250 s, under 30 kg, 300 mm envelope, 60–100 % throttle, 20 reuses, LOX/RP-1" — and the tool enumerates discrete architectures in an outer loop, solves the continuous MDO at every operating point in the throttle envelope simultaneously, verifies the candidate optimum against quasi-2-D cooling and conjugate thermal checks and authoritative CEA/CoolProp, reports every disagreement as a signed delta, and emits a configuration-hashed parts package — contour, chamber, jacket, injector, pumps, drives, battery — where every mass came from a measured body and every screen states its scope. RQ1–RQ4 are then run *on* that tool, inheriting its error bars.

### Alignment between what exists and what "done" requires

Aligned. The existing components are architecturally relevant rather than exploratory: `solve_engine`'s purity makes multipoint a `vmap` plus a shared-variable layer, not a rewrite; the snapshot/postprocess seam is where verification layers attach and already carries `compare_snapshots`; the manifest's per-row applicability already supports constraints active only at some operating points; the unavailable-metric registry is already the mechanism by which "verified by a higher-fidelity layer or `None`" would be enforced; and the `ContourProvider` seam exists specifically so an exact-variational provider can replace the analytic TOP one without touching the solve.

One architectural tension worth naming: nothing currently binds the solved state to the exported geometry to the mass ledger to the BOM, and `INTEGRATION_PROGRAM` §2.1 calls the missing `ResolvedEngineConfiguration` "a precondition for items 4, 7, and 8 being meaningful." That is an addition, not a redesign — the geometry IDs and snapshot digests are already partial implementations — but it should land before the mass and verification tracks, not after.

---

## 8. Where the functioning chain stops

```text
requirements → thermochemistry → contour → coupled engine model →
optimization → verification → resolved configuration → CAD/parts →
fabrication → test → model calibration → design iteration
```

| Stage | State today |
|---|---|
| Requirements | **partial** mapping layer implemented and CLI-reachable; architecture enumeration and multi-start absent; coverage is a screened subset, and the tool says so |
| Thermochemistry | **four pairs** chamber only; frozen constant-γ through the expansion; no equilibrium; no coolant real-fluid surfaces |
| Contour | **implemented** TOP Bézier promoted; variational BVP works and benchmarks against Rao 1958, but is not promoted and cannot yet be validated against forward MOC |
| Coupled engine model | **implemented** one differentiable solve, one genuinely closed feedback edge, one optional two-way loop with an ablation |
| Optimization | **implemented** exact Jacobians, feasible KKT-scale point; single-point only; independent KKT stationarity not yet computed |
| Verification | **not reached** no higher-fidelity layer runs on the optimum today |
| Resolved configuration | **not reached** geometry IDs and snapshot digests exist; the binding object does not |
| CAD / parts | **implemented** for the modeled subsystems, with valid B-rep and machining gates; not a released drawing package |
| Fabrication | **not reached** |
| Test | **not reached** |
| Model calibration from test | **not reached** and not yet planned anywhere except `INTEGRATION_PROGRAM` §2.5 |

> **Where it stops**
>
> The functioning chain stops at a **verified-by-construction single-point optimized design with exported, kernel-validated B-rep geometry and a partial mass ledger**, blocked from physical release by 17 named missing external-evidence requirements. The intended chain ends at a fabricated, hot-fired article whose test data flows back into the model.

---

## 9. Gaps between the current project and its endpoint

### 1. Multipoint operation

**Missing** — `mdo/multipoint.py` — shared hardware, per-point controls and states, one stacked NLP.
**Why** — Every throttle, altitude, and duty-cycle requirement is currently unanswerable. The tool refuses rather than pretending, but refusing is not the endpoint.
**Status** — **planned** Track C / plan Phase 10, not started.
**Where** — Inside the project.
**Closure** — A ≥3-point job converging with all margins reported per point, a deep-throttle case exercising both the separation screen and the off-BEP pump surface without silent extrapolation, and the `job.py:542` guard removed.

### 2. Whole-engine dry mass

**Missing** — Configuration-controlled chamber/nozzle and injector mass models, a traditional pump BOM, and mass models for valves, lines, brackets, and the gimbal.
**Why** — The differentiable objective is a partial subtotal. An optimizer minimizing a subtotal is optimizing the wrong thing — the plan promotes this above multipoint for exactly that reason.
**Status** — **active** G1 landed and verified; G2 partial; G3/G4 planned.
**Where** — Inside the project, plus fabrication data for the valve/line/fitting classes.
**Closure** — `masses.engine_dry_mass` moving off the unavailable registry with every component traced to a measured body or a sourced model.

### 3. Verification of the optimum at higher fidelity

**Missing** — Quasi-2-D cooling on the candidate optimum, a conjugate coupling pass, independent held-out CEA/CoolProp re-evaluation, field-specific parity tolerances with an adjudication rule, and a discrepancy-correction loop.
**Why** — Nothing currently checks whether the gradient optimizer walked into the region where its screens are least accurate — the plan's own headline risk.
**Status** — **planned** Track D, not started. `physics.regenerative_cooling_2d` already returns the land-midpoint hot spot the 1-D circuit averages away, so D1 is a wiring job, not a modeling job.
**Where** — Inside the project for D1/D2; on the authoritative backends for D3.
**Closure** — A signed T_wg,land − T_wg,1D delta and a Δp delta reported on a candidate optimum, with the optimum flagged when the spread exceeds the wall-temperature constraint margin.

### 4. Rao contour promotion (B3)

**Missing** — Forward-MOC compatibility on real variational solves.
**Why** — Without it, no TOP-versus-variational discrepancy is adjudication-eligible, the exact-Rao inner-loop provider cannot be promoted, and the existence-boundary research cannot use the exact contour as its physical reference.
**Status** — **active** typed contract landed 2026-08-24; the gate is the named next step.
**Where** — Inside the project — a numerics problem (neighbor-Mach smoothness 0.118 / 0.137 vs a 0.10 policy limit).
**Closure** — A real BDE probe returning a typed pass with recorded discrepancies against a declared tolerance policy.

### 5. Convergence-domain coverage (B5)

**Missing** — Slices at other γ, P_a/P_0, throat geometry, closure model, and numerical context; a fuller mesh-convergence study; convergence at the 25 currently-`not_converged` knots.
**Why** — The fail-closed lookup refuses outside its declared context, so the map's coverage bounds where an exact-Rao path could ever be used.
**Status** — **partial** one slice generated, hashed, and packaged.
**Where** — Inside the project; sandbox-runnable NumPy, so it does not compete for host time. Cost is known: the 65-knot slice took ~68 minutes on this host.
**Closure** — A multi-γ, multi-ambient family of slices with a stated interpolation policy and a mesh-convergence study beyond kernel-resolution evidence.

### 6. Viscous correction and its validity gate

**Missing** — The δ* → wall feedback loop, and a throat-Reynolds gate returning `unknown` in SP-8120's small-nozzle regime.
**Why** — Every contour, C_F, and separation result is inviscid. SP-8120 states the missing procedure verbatim and also states where it breaks down (throat Re < 100).
**Status** — **planned** F1, with F1a — the cheaper Reynolds gate — newly separated out as independently shippable.
**Where** — Inside the project for the displacement loop; higher-fidelity CFD for validation.
**Closure** — A `--boundary-layer-correction {off,displacement}` flag that reports the correction magnitude and refuses with `unknown` when the throat Reynolds number puts the design outside the method's domain.

### 7. The resolved-configuration binding

**Missing** — A frozen object with a content hash over the solved state, consumed by every downstream consumer, with assembly refusing mismatched hashes.
**Why** — It makes "the analyzed wall is the exported wall" checkable, and is a precondition for the mass, CAD, and test-return tracks being meaningful.
**Status** — **specified** `INTEGRATION_PROGRAM` §2.1, not started. No literature dependency.
**Where** — Inside the project; ASME Y14.41 (already in the corpus) is the reference for an L4 configuration-controlled digital dataset.
**Closure** — Every artifact stamping the same hash, and `build_engine_parts` raising on a mismatch.

### 8. Combustion efficiency, ignition, and stability

**Missing** — A predictive η_c*, any igniter model, and a combustion-response function.
**Why** — `--eta-cstar` is an input, so delivered c* is not a prediction. There is no igniter model at all — only a port diameter and depth cut through the pintle.
**Status** — **blocked** ignition has no source in the corpus. Stability is half-blocked: SP-8113 supplies calculable acoustic damping; no source supplies a pintle combustion-response function. η_c* prediction depends on the test-return path, not the reverse — SP-8089 Table II records "no correlations for level of mixing and spray size" for the pintle element family.
**Where** — Literature acquisition, then experimental measurement.
**Closure** — Near term: registering combustion stability as unavailable with that reason and gating delivered c* as non-predictive when η_c* was supplied. Long term: a calibrated configuration-specific decomposition fitted from test data.

### 9. Experiment-to-model return path

**Missing** — Everything. Absent from every plan until 2026-08-22.
**Why** — It closes the loop the whole tool aims at, and gaps 8 and 3 both depend on it.
**Status** — **newly identified** as a genuinely new workstream.
**Where** — Fabrication and hot-fire testing, plus a data-ingestion contract inside the project.
**Closure** — A measured hot-fire c* and wall-temperature history ingested against a configuration hash and used to calibrate the model that predicted them.

---

## 10. Validation and evidence

| Mechanism | Validates | Does not validate | Strength |
|---|---|---|---|
| 2174 tests, all green *(verified)* | equations, units, identities, trends, failure behavior, refusal paths | any physical correctness | Strong for software correctness. Raw counts are inflated by parametrization: `test_jax_primitives.py` alone is 460 cases from 11 test functions swept over 6 γ values and Mach ranges. |
| NumPy↔JAX parity ~1e-10 | that the differentiable implementation computes the same function as the audited reference | that either is physically right | Strong, and load-bearing: it is what makes gradients trustworthy |
| AD vs re-solved central FD (<1e-4 gate) | that total derivatives through the IFT solves are exact, not approximations of an unrolled solver | model fidelity | Strong for the specific claim |
| NASA `MOC_Grid_BDE` bit-parity port | that the throat-arc kernel reproduces an independent published implementation (max \|ΔM\| ≈ 5e-10 over 120 nodes) | that either implementation matches experiment | Strong for the port; the reference is itself a code, not data |
| **Rao 1958 Nozzle B benchmark** *(verified 7/7)* | that the variational solver reproduces published optimum-thrust geometry and C_F for one case | any other point in (ε, L%, γ, P_a/P_0) space; any viscous or real-gas effect | **The strongest physical-validation evidence in the repository.** It is one case. |
| Held-out CEA property-surface gate | that the packaged C¹ surfaces reproduce RocketCEA at points not used in the fit (every cell, value and gradient) | that RocketCEA's chamber model matches a real chamber | Strong for the surface fit; four propellant pairs only |
| STEP export → re-import → volume *(verified)* | that generated geometry is topologically valid B-rep and survives a kernel round trip | that the geometry is manufacturable as drawn, or correct as engineering | Strong for the specific claim |
| Model registry audit *(verified 61/61)* | that every declared model has a reachable local source and every repository heuristic is labeled | that any model is correct or applicable | Moderate — a bookkeeping gate, and valuable as one |
| Benchmark manifests with source SHA-256 | that reference values were taken from a specific document revision | the underlying data quality | Strong for provenance |
| Schomberg LEA TOP, Vulcain/S1, Radhakrishnan pintle | geometry (strict) and trends (report-only) | most of the physics, deliberately — these carry `expected_physics_gaps` | Honest but limited; several are explicitly report-only until BL physics exists |
| Physical-release evidence gate *(verified: blocked, 17)* | that the tool refuses to imply hardware readiness | anything about the design | Strong as a refusal mechanism |

### Outputs with little or no independent validation

- **Every thermal result.** No conjugate heat-transfer comparison, no experimental wall temperature. The 1-D+fin circuit's central approximation — averaging away the land-midpoint hot spot — is known and quantifiable with code that already exists (`regenerative_cooling_2d`) and has not been run against an optimum.
- **Every structural and life result.** No FEA, no cyclic plasticity, no creep, no material allowables from qualification data. LCF is registered unavailable rather than reported.
- **Every separation and side-load result.** Empirical onset only; restricted shock separation not modeled; side load registered unavailable.
- **Every pump result.** No measured pump map, no cavitation test, no rotordynamics.
- **Every injector performance result.** No cold flow, no hot fire. The spray solver's coupling to the cycle is opt-in and gated precisely because it is not validated.
- **The variational contour outside the single Rao 1958 case.** The general path run in this audit converged numerically and was refused at export.

### Validation needed now vs. later

*Needed now,* for components that already function and produce numbers people could act on: an independent CFD comparison for C_F, wall pressure, exit Mach, and flow angle on the promoted Bézier contour; the quasi-2-D land hot-spot delta on a candidate optimum; and a held-out re-evaluation of a final optimum on the authoritative backends.

*Only relevant once the corresponding subsystem exists:* multipoint constraint satisfaction, side-load prediction, cycle-life qualification, combustion-stability verdicts, and anything requiring hot-fire data.

---

## 11. Important technical decisions and project evolution

#### Feb–Apr 2026 — prototype to package

The project starts as `Rocket_nozzle_sim_phase2.py`, a single interactive script that solves ε from Pc/Pa, optimizes θ_n/θ_e, and plots a reduced-length bell. Within weeks it becomes a package (`raosim`) with a CLI, tests, and assets. The April commits are mostly correctness work on the MOC path: intersection-based exit-plane sampling, corrected exit thrust integration, a refactor of `optimize_wall` from an ad-hoc routine to a SciPy constrained optimizer, and criterion-based MOC tests.

#### Apr 16 — the first architectural correction

"Fix MOC and implement Rao backward MOC," then "Use quadratic Bézier for bell contour." The quadratic Bézier becomes the trusted default and stays it — a decision that holds through everything after.

#### May 2026 — the variational path and the NASA reference

"Fixing Rao Variational Method," then "Add coupled wall solver," then "Added in NASA code for MOC generation." Bringing in NASA/JHU's C++ source is the pivot that makes the rest possible: it supplies an independent oracle for the kernel march, which is what later makes bit-parity a meaningful gate.

#### Jun 15 2026 — the densest technical day in the history

31 commits that month, most on this date. A cluster of corrections that materially changed what the solver computes:

- *Characteristic invariants corrected.* A spurious cosμ/cos(θ±μ) factor had been mirrored through four modules; removing it changed the physics of every characteristic march.
- *Left-Mach geometry made exact by construction.* Removing x from the unknown vector eliminated a degenerate basin where the solver satisfied geometry by moving x rather than by getting the flow right.
- *Length constraint corrected to Rao's exit-station form.* The legacy Σdx double-counted x_D against the x_E pin; the two constraints were contradictory, and the solver resolved the contradiction by driving `kernel_d_fraction` to the throat plane. The code comment records that this was invisible before the kernel fix because the degenerate BD held D at x ≈ 0.
- *CE Mach-monotonicity penalty zeroed* under the characteristic formulation, after determining the control surface crosses streamlines so M need not be monotone along it, and that the penalty was displacing the correct branch.
- *Folded-RRC mass integral regularized* so mass stays finite when dr = 0.
- *Design angles de-circularized:* reported θ_n/θ_e became solver outputs with provenance metadata, and Rao chart angles became comparison data — the `err_*` columns explicitly redefined as exact-vs-parabola deltas rather than errors.
- *J4 gate closed*, moving `max_scaled` from 8 to 1.157e-3, and the BDE wall oracle-validated against NASA `wall.out` at RMS 1.8e-4 against a 1e-3 gate.
- *JAX differentiable core added*, with an LM solver, a differentiable MOC kernel, and a sensitivities API.

The pattern across that day is consistent: each change is a *correction to a formulation*, each is accompanied by a test that pins the corrected behavior, and each records the symptom that revealed it.

#### Jun 28–Jul 2026 — packaging, injector, and pumps

Package renamed to LREKit; PyPI release workflow; pintle injector integrated into the v2 workflow with a feed-system ledger; shoulder radius moved from a 0.25 placeholder to a geometric auto-sizing derivation; Cd target → Ru derivation via Hall's transonic relation; a shared pump bus. The injector CAD is then rebuilt from a machined two-ring scheme to the coaxial five-part TRW/Nardi architecture after the old scheme was found topologically incapable of a pintle's flow.

#### Jul 2026 — the MDO layer, in phases, in one month

The walking skeleton lands 2026-07-22; the Phase 4a cooling block, Phase 5 injector block, Phase 6 pump block, and Phase 7 coupled solve all land 2026-07-23; Phase 7b (film cooling) and Phase 8/9 (the NLP) the same day. Two decisions in that burst are load-bearing:

- *The coupling is nested-IFT, not a monolithic block-sparse Newton*, on the explicit reasoning that enlarging the nonlinear system to *look* coupled is dishonest when the physics is a one-way chain.
- *Film cooling was added because an empirical probe proved the wall is coolant-enthalpy-limited*, not because it is a standard feature. Before that probe, coking was infeasible throughout the design box and the natural (wrong) response would have been to relax the constraint.

#### Aug 2026 — mass, requirements, generalization, and honesty gates

The hardware mass ledger; the requirements layer (SP-125 §2.1); propellant and thrust generalization via `MissionSpec.for_thrust`/`for_propellant` (forced by the discovery that a 130 kN run had no feasible point in the hand-tuned 13 kN box); bounds and HTD screens; the O/F variable; input validation; then the 2026-08-22/24 pair of commits landing the CEA held-out gate, the typed contour provider, the Rao Eq. (5) export refusal, the B3 validator contract, and the B5 domain map.

> **A recurring decision pattern**
>
> When a gap is found, the gate ships before or with the model: LCF became "unavailable" rather than being reported as static stress; side load became "unavailable" rather than correlated; liner buckling became a mandatory final gate rather than an inner-loop constraint while its root robustness is unproven; separation gained a topology-scope field rather than a new model; and cryogenic HTD coverage became a preflight refusal rather than an extrapolation. This is codified as a rule — *"a fidelity gap must be visible as `unknown`, never absorbed into a passing number"* — and it is followed.

---

## 12. Failures, limitations, and unresolved problems

### Major limitations

Things that change what the current implementation can claim.

> **1 · Found in this audit**
>
> **The documented MDO optimizer CLI invocation fails on its own default burn time.** `lrekit --engine-mdo-optimize --isp-min 200` — the example in `MDO_GUIDE.md` §8 — aborts with an Equinox/Lineax "linear solver received non-finite input" error propagated out of a `jax.pure_callback`. The cause is isolated: `--burn-time` defaults to 10.0 s in the CLI, whereas `MissionSpec`'s own default is 120 s and `tests/test_mdo_nlp.py` uses the latter.

| burn_time | `solve_min_mass(MissionSpec.for_thrust(13e3, of_is_pinned=True, burn_time=…), 190.0)` |
|---|---|
| 10 s *(CLI default)* | **fails** XlaRuntimeError — non-finite linear solve |
| 30 s | **ok** feasible, max_violation 8.7e-11, Isp 224.8 |
| 60 s | **ok** feasible, max_violation 1.0e-10, Isp 224.8 |
| 120 s *(module default)* | **ok** feasible, max_violation 1.0e-12, Isp 224.8 |

The failure is caught and reported as "MDO solver failed" rather than crashing, and the test suite does not cover it because the fixture uses the module default. At a 10 s burn the battery becomes power-limited rather than energy-limited, which is the plausible mechanism. **Effect on claims:** the optimizer is reachable and correct from the Python API, and currently unreachable from its own documented command line at default settings.

1. **Rao contour promotion is blocked and the fallback is the approximation the exact solve was meant to replace.** All B3 real probes return `UNKNOWN`. The bulk MDO contour is the analytic Rao/TOP Bézier — a validated approximation to the optimum, not the optimum. The plan is explicit that this is acceptable for Pareto sweeps and not for existence-boundary work, which is exactly the research question.
2. **Whole-engine dry mass does not exist.** The differentiable objective is `dry_mass_partial`: pumps, motors, inverters, a smooth battery governing-branch surrogate, and screening liner/land/closeout chamber masses. Injector hardware, manifolds, valves, lines, gimbal, and mounts are absent. `masses.engine_dry_mass` is registered unavailable. Any mass requirement is therefore checked against a **lower bound**, and the requirements layer says so in those words.
3. **All contour and performance results are inviscid.** δ* is computed once and reported, never fed back. Throat Reynolds number is not computed for the nozzle. There is no CLI flag for a viscous correction.
4. **Nozzle expansion is constant-γ.** The CEA surfaces give γ at the chamber; the value entering the characteristic net is a single frozen number. At high area ratio this systematically misplaces the Prandtl–Meyer turning and therefore the wall angles. The frozen variable-c_p path exists but is Bézier-only and not physically validated.
5. **Multipoint operation is refused, not approximated.** Correct behavior, and a hard limit on what the tool can answer.
6. **No physical validation of any thermal, structural, separation, injector, or pump result.** Enumerated in §10.
7. **Cryogenic coverage is incomplete.** Authoritative LOX/LCH4 and LOX/LH2 optimization stops at model-coverage preflight for lack of real-fluid coolant surfaces. `--allow-incomplete-physics` permits a screening run whose physics verdict stays `unknown`. LOX/LCH4 and N2O/Ethanol L*/wall-limit constants are flagged `estimated=True`.
8. **Phase 9's own gate is not met.** The code treats SciPy success plus feasibility as a KKT pass; independently computing and reporting KKT stationarity remains open by the plan's own accounting.

### Secondary limitations

- **Stale documentation row.** `ROADMAP_REQUIREMENTS_TO_PARTS.md` §Status lists "R3 requirements layer — not started," but `raosim/requirements.py` landed 2026-08-06 → 2026-08-10 (after that doc's 2026-07-31 update) and is CLI-reachable via `--requirements`. R3's *full* three-layer scope is genuinely partial — layer 1 (architecture enumeration, multi-start) does not exist — but "not started" is wrong.
- **The tool cannot distinguish "no feasible design exists" from "the optimizer failed."** At 5 kN / sea level / Isp ≥ 250 s the requirements run reports `SOLVER_FAILED` with "Positive directional derivative for linesearch." That is honest, but a genuinely infeasible requirement set and a converged-to-a-bad-point optimizer produce the same message. No multi-start exists to help distinguish them.
- **`--eta-cstar` is an input presented alongside "delivered" performance.** Not currently gated as non-predictive.
- **Report/README test counts are historical records** ("1246 passed … July 11, 2026"). The current number is 2174. The README labels this explicitly as a completed-run record rather than a promise, which is the right treatment.
- **Chamber `L*` / contraction-ratio placeholders warn rather than refuse** on a hardware-oriented run.
- **Legacy surface area.** `Rocket_nozzle_sim_phase2.py` (the original prototype) is retained and unwired; `main.py` remains packaged as a legacy entry point. Both are labeled. Module-level dead code is otherwise minimal — every `raosim` module checked is imported by at least one other source file and covered by at least one test.
- **The pump legacy triangle-mesh CAD path self-labels `schematic_only`** and coexists with the B-rep path.

### Expected incompleteness

Absent because development has not reached it, not because it is broken: multipoint; Phase 11 verification layers; the resolved-configuration object; the fidelity ladder (F1–F7); continuation and the σ_min/λ_min diagnostics; the valves/lines/brackets/gimbal mass models; RQ1–RQ4 results. These are specified with acceptance gates and are the current work queue, not defects.

Genuinely blocked rather than merely unbuilt: the igniter model (no source in the corpus), the combustion-response half of a stability screen (SP-8113 supplies damping only), and the L4 datum/tolerance-stack scaffolding (blocked on standards).

---

## 13. Actual technical ambition

The technical capability the project is trying to create: **a design tool that takes engine requirements and returns a fabricable, mass-accounted, multipoint-feasible engine definition in which every number is either measured from the geometry that was exported or explicitly refused.** The research questions (RQ1–RQ4) are then run on top of that tool, inheriting its error bars — a sequencing decision the plan states explicitly and defends: *"tool completeness leads; the research questions are run on the finished tool rather than racing ahead of it."*

Why the existing components are built the way they are:

- The **differentiable core** exists because the coupling — jacket Δp → pump rise, film fraction → wall temperature → c*, wall thickness → conductance vs structure — is what a sequential process resolves by hand and a gradient method resolves at once. Exact total derivatives through the converged state are the engineering claim.
- The **contour work** exists because the physical claim is about the Rao existence boundary, and you cannot study a boundary using a parabola fitted to solutions that stop existing there.
- The **provenance and refusal layers** exist because a design tool whose screens quietly extrapolate produces designs that fail in ways the tool cannot see. The registries, gates, and unavailable-metrics list are how the error bar stays visible through a chain of twelve models.
- The **CAD and mass work** exists because the endpoint is a fabricable article, and a mass that does not come from the exported body is not a mass.

What becomes possible when the chain is complete: a requirement statement in, a configuration-hashed parts package out, with per-discipline signed deltas from independent higher-fidelity checks and throttle-envelope feasibility proven rather than assumed — and then a genuine answer to whether whole-engine optima migrate toward Rao's cliff, or whether separation and thermal limits mask it first.

How much of that ambition is in the repository today: the physics layer, the differentiable core, the optimizer, the output contract, the CAD, and the provenance infrastructure are built. The verification layer, the multipoint layer, the whole-engine mass, and the configuration binding are not. The current architecture supports all four as additions rather than redesigns.

> **One divergence worth naming**
>
> The stated end state is "a general-purpose design tool that can also answer RQ1–RQ4." The single most-developed and most-corrected subsystem in the repository is the Rao variational solver — which the general-purpose tool does not currently use, since the bulk MDO contour is the analytic TOP approximation and the exact path is not promoted. That is not incoherent: the plan's rule 9 says fidelity is switched by the question, and the exact path is reserved for existence-boundary and exact-vs-TOP studies. But it does mean a large share of the deepest work serves the research half rather than the tool half, and the tool half is what the completion plan says leads.

---

## 14. Project scale

| Measure | Value |
|---|---|
| Commits | 72, sole author, 2026-02-22 → 2026-08-24 (29 distinct commit days; 31 of them in June 2026) |
| Tracked Python | **133 720 lines** across ~270 files |
| `raosim` package | 127 files, 95 420 lines |
| `raosim/mdo` | 30 files, 18 745 lines |
| `raosim/jax` | 13 files, 3380 lines |
| `raosim/spray` | 14 files, 6018 lines |
| `raosim/openfoam` | 3 files, 2565 lines |
| Tests | 120 modules, 32 386 lines, **2174 collected** (2148 non-slow + 26 slow) — all passing, verified today |
| Test distribution | jax 527 (460 of which are parametrized sweeps in one file), mdo 349, injector 144, spray 120, rao 115, regen 68, requirements 64 |
| Scripts | 18 files, 3579 lines (diagnostics, scans, comparisons, sizing studies) |
| CLI surface | **328 distinct flags**, plus 4 job verbs and an interactive interview |
| Contour methods | 4 (Bézier/TOP, conical, direct MOC, Rao variational BVP) |
| MDO design variables | 10 active (11 with O/F) |
| Constraint manifest | **31 rows** — 23 differentiable hard, 7 mandatory final gates, 1 report-only |
| Model provenance registry | **61 entries** across 17 subsystems; audit passes |
| Registered-unavailable outputs | 10, each with a stated reason |
| Propellants / materials | 5 / 8 |
| Packaged CEA property tables | 4 pairs, hash-pinned, every-cell held-out gated |
| Benchmark cases / curves | 7 manifests (source SHA-256 pinned) / 8 digitized reference curves |
| Rao domain artifact | 65 knots (13 ε × 5 L%), 40 `converged_valid` / 25 `not_converged`, 101 932 bytes, SHA-256 pinned |
| Documentation | 49 markdown files, ~123 500 words, plus a 7600-word README |
| LaTeX reports | 4 compiled PDFs (reference manual, MDO plan, TVC actuator sizing, report) |
| Literature corpus | 255 PDFs; 224-file Markdown mirror of 190 unique PDFs spanning 12 047 source pages, with per-paper extraction diagnostics |
| Vendored external source | NASA/JHU `Three-Dimensional-Nozzle-Design-Code` (MOC_Grid_BDE + STT2001 + 3D_MOC), under NASA Open Source Agreement |
| Generated build outputs | 512 entries under `builds/` |
| Release state | `lrekit` 0.1.1 wheel + sdist built; `v0.1.0` tag; 4 console entry points; GitHub Actions test + trusted-publishing workflows; PyPI publication pending a license decision |

Line counts measure size, not depth. The three largest files — `run_nozzle.py` (6641), `rao_variational.py` (6179), `injector.py` (4349) — differ sharply in character: the first is a CLI surface, the second is the dense numerical core, the third is hydraulics plus extensive provenance and gating.

---

## 15. External dependencies vs. project work

| Dependency | What it does | What the project supplies |
|---|---|---|
| JAX 0.6.2 | forward/reverse automatic differentiation, jit, float64 arrays | every residual, every physical model, the static-shape discipline the models are written under, the choice of what is implicit vs explicit, the parity oracles |
| Optimistix 0.0.11 / Equinox / Lineax | Newton and Levenberg–Marquardt root-finders, implicit-diff plumbing | the residual systems being solved, the state scaling, the nesting structure (outer (R_t, ṁ), inner T_wg vector), convergence tolerances 100× tighter than optimizer feasibility |
| SciPy | SLSQP, secant/brentq, splines, linear algebra | the NLP formulation — 10 variables, 23 enforced constraints, the ε-constraint sweep — and the exact Jacobian handed to SLSQP, which SciPy would otherwise finite-difference |
| NumPy | array primitives | the oracle implementations of every JAX block, and the entire NumPy-only existence-scan research track |
| CadQuery 2.7 / OpenCascade | B-rep Booleans, sweeps, lofts, STEP I/O | every dimension, every feature, the coaxial five-part architecture, the blade camber math (`pumps.impeller_blade_camber` — physics owns it, CAD sweeps it), the machining gates, the export→re-import→volume validation, the geometry-ID hashing |
| CoolProp 7.2 | real-fluid property lookups | the coolant property interface, applicability screens, and the explicit decision *not* to autodiff it — with the consequence (no cryogenic HTD coverage) registered as unavailable rather than papered over |
| RocketCEA 1.2.3 | chamber equilibrium thermochemistry | the offline sampling scheme, the C¹ Fritsch–Carlson/tensor-Hermite surface construction, the schema/hash identity contract, the every-cell held-out value-and-gradient gate, and the rule that CEA is never called inside a traced loop |
| Matplotlib | rendering | every diagnostic figure's content and the MOC/BDE construction diagrams |
| NASA/JHU `MOC_Grid_BDE` | an independent reference implementation of the Rao B-D-E kernel and topology | the stagnation-normalized Python port, the JAX reimplementation with the exact row policy, the bit-parity gates, and the provenance investigation that demoted the `TT'.out` fixture from promotion authority |

The pattern is consistent across rows: the external tool performs a general-purpose numerical or geometric operation; the project supplies the physics being operated on, the structure of the problem handed to it, and the criteria by which the answer is accepted or refused.

---

## 16. Evidence useful for future letters and applications

*Factual items only; no prose written here.*

### Technical independence

- Sole author of 72 commits over ~6 months, spanning a rewrite from a single interactive script to a 133 000-line package with a 2174-test suite, CI, and a built distribution.
- Wrote the governing planning documents that direct the implementation — `DIFFERENTIABLE_ENGINE_MDO_PLAN.md` (formulation, physics, literature basis, 12-phase tracker with acceptance gates), `COMPLETION_PLAN.md` (7-track deduplicated backlog with dependency graph), `PHYSICS_FIDELITY_ROADMAP.md`, `INTEGRATION_PROGRAM_2026-08-22.md` — including a hard-rules section constraining implementers (no unrolled solver differentiation, no relaxed integers, no `max()` in the differentiable core, no silent Bézier substitution for a failed Rao solve, NumPy oracle plus parity test for every JAX block).
- Identified that three of eight required workstreams (resolved-configuration object; injector/ignition/stability; experiment-to-model return path) were represented in *none* of the four existing plans, and specified them.
- Extended the project from "nozzle contour tool" to whole-engine design tool: chamber, regen cooling, structures, pintle injector, spray, electric feed system, mass ledger, CAD, requirements layer.

### Technical depth

- Implemented Rao's 1958 constrained variational problem as a boundary-value problem with fifteen separately-scaled residual groups, two solver backends at ~1e-10 parity, and export gates keyed to Rao's own Eq. (5) existence criterion from AIAA 99-2584.
- Achieved bit-parity (max |ΔM| ≈ 5e-10 over 120 nodes) between a JAX reimplementation and a NumPy port of NASA/JHU's C++ throat-arc kernel march, with exact d(BD)/dθ_B verified against finite differences to ~1e-8 — requiring reproduction of NASA's exact row-decision, row-width, and C+ pairing policy under static-shape constraints.
- Built an MDF differentiable engine model with two nested implicit solves differentiated by the implicit function theorem, exact reverse-mode constraint Jacobians, and a documented forward-mode failure mode.
- Reproduced Rao 1958's published Nozzle B to 0.0089 R_t contour RMS with direct-integration C_F 1.5769 against Rao's 1.5829.
- Built C¹ shape-preserving property surfaces with an every-cell held-out value-and-gradient gate against RocketCEA, hash-pinned and schema-versioned, for four propellant pairs.
- Built a coaxial five-part machined pintle injector B-rep package with a flow-separation audit, machining gates, STEP round-trip volume verification, and geometry-ID-keyed mass.
- Built and audited a 61-entry model-provenance registry and a 31-row constraint manifest with per-row applicability, availability, scaling, optimizer role, and source identifier.

### Persistence and debugging

- Diagnosed a spurious cosμ/cos(θ±μ) factor in the characteristic compatibility invariants mirrored through four modules, and corrected it across the codebase.
- Diagnosed that the length-constraint formulation double-counted the start station, making two constraints contradictory and driving `kernel_d_fraction` to the throat plane — a drift invisible before an earlier kernel fix.
- Diagnosed a degenerate optimization basin created by carrying x as a free unknown, and eliminated it by reconstructing x from dx/dr = 1/tan(θ+μ).
- Diagnosed that a Mach-monotonicity penalty was strong enough to displace the correct solution branch, on the physical grounds that the control surface crosses streamlines. *(Attributed in code: "identified by ibrahim.")*
- Established by empirical probe that the coking constraint was infeasible because the wall is coolant-enthalpy-limited rather than film-coefficient-limited (measured T_wc 1241 → 1124 K across the entire channel-geometry box), which identified film cooling as the correct lever and made the constraint satisfiable at 7.7 % film.
- Found that `jacfwd` silently drops a `jnp.min` tangent through jitted nested Optimistix root-finds, verified against finite differences, and switched the constraint Jacobian to reverse mode with a warning against reverting.
- Found two dead design variables by Jacobian-column and objective-gradient inspection and removed or fixed them; shipped the re-check procedure in the reviewer's guide.
- Found and fixed an off-by-one/double-counted segment in the cooling march quadrature and a silent clamp hiding the channel-fit constraint.
- Found that a 130 kN run had no feasible point in the box hand-tuned for 13 kN, and replaced the hard-coded architecture with physics-derived scaling (throat from thrust closure, chamber length from L*/CR, channels from circumference at fixed pitch, pintle from blockage band, pump speed from specific speed).
- Traced the upstream git history of NASA's checked-in `TT'.out` reference fixture and established that the generating executable is not present in the local tree or the public repository — then demoted it from promotion authority.

### Research or engineering judgment

- Corrected a load-bearing literature attribution after reading source PDFs rather than inferring from filenames (the "contour must remain an implicit solved state" claim is physically correct but was attributed to the wrong source; Rao 1958 does not require re-solving the contour at every MDO iterate, and the Rao/TOP parabola is a validated approximation to the same optimum).
- Rejected the classical gaseous film-cooling correlation family for a liquid RP-1 film after computing that its velocity ratio (~10⁻³) sits three orders below the correlations' 0.45–33.3 calibration band, and implemented a liquid phase-change model instead — retaining the gaseous VR as a pinned diagnostic so the inapplicability stays documented.
- Rejected reporting side load, cycles to failure, whole-engine dry mass, combustion stability, engine-system Isp, and five other quantities, registering each as unavailable with a specific technical reason rather than supplying a correlation.
- Rejected a monolithic block-sparse Newton for the engine coupling on the grounds that enlarging the nonlinear system to *appear* coupled misrepresents a one-way physical chain.
- Refused to promote the variational contour: `BENCHMARK_VALIDATED_AT_RELEASE = False`, and real B3 probes report `UNKNOWN` rather than a pass, because forward-MOC compatibility fails at 0.118/0.137 against a 0.10 policy limit.
- Made the multipoint job path *raise* rather than return a single-point answer labeled as a throttle envelope.
- Recorded a prior-art caveat against the project's own novelty claim: Rao 1999 Fig. 7 already shows length-limited vacuum optima on the valid boundary, so the claim must be engine-level masking rather than the existence of the effect.
- Distinguished repository policy from literature authority throughout — the 0.10 neighbor-Mach limit, the domain-map residual and spacing thresholds, and the contour discrepancy tolerances are each labeled repository screening choices, not qualification limits.
- Replaced a binned pump-efficiency estimator with a C¹ surrogate and demonstrated the property with a test showing finite differences converge to the analytic gradient at a point where the binned version's FD diverges.

### Initiative

- Built the literature infrastructure the technical claims rest on: 255-PDF corpus, 224-file Markdown mirror of 190 unique PDFs / 12 047 pages with per-paper extraction diagnostics, benchmark manifests with source SHA-256 hashes, and a claim-by-claim source-verification audit.
- Built a job/CLI layer as an additive translator after observing that O/F optimization had been implemented long before it was reachable from a command line — "which meant in practice it was not usable."
- Designed the larger architecture and is implementing it incrementally against dated acceptance gates, rather than treating the currently-completed subset as the whole project.

### Collaboration and mentorship

No human collaborator, mentor, or co-contributor is visible anywhere in the artifacts: all 72 commits, the planning documents, the correction commits, and the acceptance criteria are the author's.

### Tangible outputs

- A Python package (`lrekit` 0.1.1) with wheel and sdist built, four console entry points, CI, and a trusted-publishing release workflow.
- 2174 passing tests across 120 modules; a 61-entry audited model registry; a 31-row constraint manifest.
- Four compiled LaTeX documents including a full mathematical/physical reference manual and the MDO plan.
- Generated engineering artifacts: valid B-rep STEP for nozzle wall, jacket, regen channels with manifolds, a five-part machined pintle injector, and a pump package; watertight STL; machining-gate reports; dimension tables; a pump BOM; a 226 KB structured design summary.
- Research artifacts: a hash-pinned 65-knot Rao convergence-domain map; existence-scan CSVs and heatmaps; NASA reference-comparison station data and metrics; MOC construction diagrams.
- ~123 500 words of technical documentation with dated status heads and acceptance gates.

---

## 17. Factual characterizations for later use

1. Built LREKit, a 133 000-line Python liquid-rocket-engine preliminary-design toolkit that takes an operating point and produces sized nozzle, chamber, regenerative-cooling, pintle-injector, and electric-pump-feed geometry, a mass ledger, and valid B-rep STEP/STL CAD, with 2174 passing tests and a 61-entry audited model-provenance registry.
2. Implemented Rao's 1958 optimum-thrust nozzle problem as a finite-dimensional boundary-value problem — fifteen scaled residual groups over stationarity, characteristic compatibility, mass/length closure, and D-state continuity — and reproduced Rao's own published Nozzle B geometry to 0.0089 R_t contour RMS with a direct-integration thrust coefficient of 1.5769 against his 1.5829.
3. Ported NASA/JHU's `MOC_Grid_BDE` C++ throat-arc kernel to Python and reimplemented it in JAX at bit parity (max |ΔM| ≈ 5e-10 over 120 nodes) with exact θ_B sensitivity verified against finite differences, which required reproducing NASA's exact row-decision and characteristic-pairing policy under static-shape constraints.
4. Built a multidisciplinary-feasible differentiable engine model (`raosim/mdo/`, 30 modules) in which nozzle performance, 1-D+fin regenerative cooling, pintle injector hydraulics, and electric pump feed solve as one system with the cooling-jacket pressure drop genuinely feeding the pump rise; converged implicit states are differentiated through the implicit function theorem, and exact reverse-mode Jacobians drive an SLSQP ε-constraint optimizer over 10 design variables under 23 enforced constraints.
5. Established by empirical probe that the regen wall in this design class is coolant-enthalpy-limited rather than film-coefficient-limited, which identified film cooling as the correct lever for the SP-8087 coking constraint and turned a constraint that had been infeasible throughout the design box into an active, satisfiable one at 7.7 % film flow.
6. Designed a refusal architecture enforced in code rather than documented: ten engine outputs are registered unavailable with specific technical reasons rather than estimated, contour export is blocked when method-of-characteristics closure fails, multipoint jobs raise rather than returning a single-point answer, and cryogenic optimization stops at model-coverage preflight for lack of real-fluid coolant surfaces.
7. Currently implementing the contour-provider and hardware-mass tracks — the typed post-solve Rao validator and a hash-pinned 65-knot convergence-domain map landed 2026-08-24 with promotion explicitly blocked on a forward-MOC compatibility gate that real probes fail at 0.118 and 0.137 against a 0.10 policy limit.
8. The current implementation is a single-design-point tool whose remaining major components — multipoint throttle envelopes, whole-engine dry mass from built geometry, post-optimum verification at higher fidelity, and a configuration-hashed resolved-engine object — are specified with acceptance gates in a 7-track completion plan and are architecturally additions rather than redesigns.
9. Built the literature infrastructure the technical claims rest on: a 255-PDF corpus converted to a 224-file Markdown mirror spanning 12 047 source pages with per-paper extraction diagnostics, seven benchmark case manifests pinned by source-document SHA-256, and a claim-by-claim source verification that corrected a load-bearing attribution and reclassified a film-cooling correlation family as inapplicable to liquid coolants.
10. Verification is strong and physical validation is thin, by design and by the project's own account: NumPy↔JAX parity at ~1e-10, autodiff-versus-finite-difference gates, a bit-parity reference port, and one strict published-geometry benchmark — with no CFD, conjugate heat-transfer, FEA, cold-flow, or hot-fire evidence, and every contour stamped `hardware_qualified = False`.

---

## 18. Canonical project fact sheet

**Project** — LREKit (packaged name) / RaoRocketSim (repository name)
**Project type** — Hybrid: engineering-design software + simulation/modeling codebase + differentiable-optimization research + computational research investigation, with generated CAD as a first-class output.
**Primary objective** — Take liquid-rocket-engine requirements and return a fabricable, mass-accounted, multipoint-feasible engine definition in which every number is either measured from the exported geometry or explicitly refused; then use that tool to study whether whole-engine optima interact with the Rao nozzle existence boundary.
**Current state** — Working single-design-point tool. Two parallel pipelines (sequential `design_nozzle_v2` and differentiable `raosim/mdo/`) sharing one versioned output contract. 2174 tests passing (verified 2026-08-25). Alpha release built, PyPI publication pending a license decision.
**What currently exists** — Four contour methods (Rao/TOP Bézier promoted; conical; direct MOC; Rao variational BVP experimental); chamber/throat geometry with derived shoulder fillet and Hall-based Ru; Bartz + Sieder–Tate + fin regen cooling with a counterflow finite-volume march; SP-125 structural and Coffin–Manson fatigue screens; pintle injector hydraulics with a Son-2017 movable branch and a Lagrangian spray parcel solver; centrifugal pump meanline with electric drive and battery sizing; a configuration-controlled mass ledger with CAD-linked injector mass; B-rep STEP + STL export with re-import validity and machining gates; a coupled differentiable engine solve with exact total derivatives; a 31-row constraint manifest; an SLSQP ε-constraint NLP with Pareto sweeps; a requirements layer mapping SP-125 §2.1; a 61-entry model registry; a 10-entry unavailable-output registry; a NASA/JHU reference port at kernel bit parity; four hash-pinned CEA property surfaces; seven benchmark manifests; a 65-knot Rao convergence-domain map.
**Under implementation** — Track B (contour and property closure): B3 post-solve variational validator blocked on forward-MOC compatibility; B5 domain map partial beyond one fixed slice. Track G (hardware mass and CAD): injector build-once landed and verified; chamber mass from disjoint liner/ribs/jacket CAD partial. Ongoing scope/validity gates from the fidelity roadmap (§3.1, §4.1, §5.1 landed).
**Intended completed system** — Requirements in → discrete architecture enumeration in an outer loop → simultaneous continuous MDO across every operating point in a throttle envelope → post-optimum verification by an independent higher-fidelity layer per discipline reporting signed deltas → a configuration-hashed parts package where every mass came from a measured body → optional fabrication and hot-fire with test data returning to calibrate the model.
**Definition of "done"** — For an arbitrary propellant and thrust class in the declared domain: every reported mass measured from built geometry or `None` + reason; every constraint stating its applicability scope and source; the optimum verified by ≥1 independent higher-fidelity layer per discipline with signed deltas; multipoint throttle envelopes honored rather than refused; the contour provider declaring where it is valid and refusing where Rao says no shock-free solution exists; and no number in any report a screening surrogate without saying so. RQ1–RQ4 then run on that tool.
**Inputs** — **Traditional:** 328 CLI flags / `.args` files / job YAML covering operating point, propellant, contour method, chamber and throat geometry, materials, coolant and channel geometry, injector architecture, feed architecture, electric drive and battery constants, CAD and export options. **MDO:** a `MissionSpec` plus a 10- or 11-variable design vector (`Pc`, `eps`, `dp_f_frac`, `dp_o_frac`, `D_pintle`, `N_rpm`, `channel_width`, `channel_height`, `film_frac`, `t_wall`, [`OF`]).
**Core process** — MDO path: outer Optimistix Newton on (R_t, ṁ) → C¹ chamber property surfaces → typed contour provider → station grid → inner Newton on the stationwise wall-temperature vector (both IFT-differentiated) → injector hydraulics → hydraulic edge Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank → pump/electric feed with separate battery energy/power branches → geometric chamber mass → 31 constraint margins → jacrev exact Jacobian → SLSQP → versioned snapshot → authoritative re-evaluation through `design_nozzle_v2` with pinned thermochemistry → report and CAD.
**Outputs** — `contour.csv`; profile/cooling/separation/structural/injector/spray PNGs; watertight STL; unit-declared neutral B-rep STEP (wall, jacket, regen with manifolds, five-part machined pintle, pump package, engine assembly); pintle dimension table and machining-gate report; pump BOM and reference geometry; a 39-section `summary.json` (~226 KB); MDO constraint-margin tables, Pareto frontiers, and snapshot comparisons; research artifacts (existence-scan CSVs, heatmaps, convergence-domain JSON, NASA comparison metrics).
**Major components** — `design.py` (sequential orchestrator); `rao_variational.py` (variational BVP); `nasa_moc.py` + `moc.py`; `physics.py` + `thermal_design.py` + `regen_*`; `injector.py` + `spray/`; `pumps.py`; `mass_ledger.py`; `*_cad.py`; `raosim/jax/`; `raosim/mdo/`; `requirements.py`; `model_registry.py`; `run_nozzle.py` + `lrekit/`.
**Key methods** — Method of characteristics (axisymmetric, Kliegel–Levine/Sauer transonic start line); Rao variational optimum-thrust formulation with Lagrange multipliers; implicit function theorem differentiation of converged states; JAX autodiff with jit and static shapes; Optimistix Newton and Levenberg–Marquardt; SLSQP ε-constraint multi-objective optimization; Fritsch–Carlson C¹ shape-preserving interpolation; Bartz with the σ property factor; Sieder–Tate with a Shah–London laminar branch; land-fin efficiency; counterflow upwind finite volume; Knuth critical-flow liquid-film phase-change model; Son-2017 movable-pintle two-branch area; Lee-2021 battery epigraph; specific-speed pump meanline with SP-8052/SP-8109 screens; SP-125 eq. 4-31 combined stress and Coffin–Manson strain-life; WAVE/KH breakup and Spalding evaporation; B-rep solid modeling via OpenCascade with round-trip validation.
**Major contributions** — The variational BVP formulation and residual architecture; the NASA kernel port and its JAX bit-parity reimplementation; the MDO layer's design and invariant rule set; four load-bearing formulation corrections (characteristic invariants, length constraint, x-degeneracy, Mach-monotonicity penalty); the coolant-enthalpy-limited diagnosis and the film-cooling lever; the coaxial five-part injector CAD architecture; the provenance, refusal, and gate infrastructure; the requirements layer; the literature corpus and benchmark manifests; and the planning documents specifying the whole program with acceptance gates.
**Strongest capabilities** — Rao 1958 Nozzle B reproduction at 0.0089 R_t RMS and C_F 1.5769 vs 1.5829; NASA kernel bit parity at 5e-10; exact total derivatives through two nested implicit solves matched to re-solved central differences; a feasible KKT-scale optimum at `max_violation` ~1e-11 with a genuinely thermal-limited active constraint; full 13 kN engine package generation with kernel-validated B-rep CAD; a 2174-test suite passing in 34 minutes.
**Major limitations** — Whole-engine dry mass does not exist (partial subtotal only); the exact-Rao contour is not promoted and the MDO uses the TOP approximation; all flow is inviscid and constant-γ; multipoint is refused; no CFD/CHT/FEA/cold-flow/hot-fire validation of any result; cryogenic coolant coverage incomplete; the documented `--engine-mdo-optimize` CLI invocation fails on its own 10 s default burn time (Python API works).
**Unfinished components** — `mdo/multipoint.py`; Phase 11 verification layers; the `ResolvedEngineConfiguration` binding object; whole-engine mass models (valves, lines, brackets, gimbal, pump BOM, chamber/nozzle configuration control); the fidelity ladder (BL displacement loop, variable-γ MOC, equilibrium chemistry, FSS/RSS classifier, PNS layer, Rao adjoint); `mdo/continuation.py` and the σ_min/λ_min diagnostics; igniter and combustion-stability models (blocked); the experiment-to-model return path.
**Validation performed** — 2174 passing tests; NumPy↔JAX parity at ~1e-10; AD vs re-solved central FD < 1e-4; NASA `MOC_Grid_BDE` kernel bit parity 5e-10 and BDE wall vs NASA `wall.out` at 1.8e-4 RMS; Rao 1958 Nozzle B strict benchmark 7/7 metrics; every-cell held-out CEA value-and-gradient gate on four propellant pairs; STEP export→re-import→volume gates; model registry audit 61/61; benchmark manifests pinned by source SHA-256; Schomberg LEA TOP strict geometry with report-only pressure; Vulcain/S1 report-only separation trend.
**Development frontier** — B3 forward-MOC compatibility (0.118/0.137 vs a 0.10 policy limit); B5 domain-map coverage beyond one γ = 1.4 / P_a/P_0 = 0.01 slice with 40/65 knots converged; Track G chamber mass from disjoint CAD.
**Intended endpoint** — A requirements-driven, multipoint, verification-backed engine design tool producing a configuration-hashed fabricable parts package, on top of which RQ1–RQ4 (value of simultaneous optimization; exact gradients vs conventional; interaction with the Rao existence boundary; frozen vs equilibrium migration) are answered.
**Next technical steps** — (1) the `ResolvedEngineConfiguration` object, a precondition for the mass, CAD, and test-return tracks; (2) Track G completion so the optimizer minimizes a real dry mass; (3) `mdo/multipoint.py` and removal of the `job.py` guard; (4) D1 quasi-2-D cooling as a post-optimum signed delta (the model already exists and is unwired); (5) B3's forward-MOC gate; (6) fix the `--burn-time` failure in the optimizer CLI path and add a regression test at the CLI default.
**Tangible artifacts** — `lrekit` 0.1.1 wheel + sdist; `v0.1.0` tag; CI and trusted-publishing workflows; 4 compiled LaTeX documents; 512 generated build outputs including full engine packages with valid B-rep STEP; a hash-pinned 65-knot Rao convergence-domain artifact; existence-scan datasets and heatmaps; NASA comparison station data; ~123 500 words of dated technical documentation; a 224-file / 12 047-page literature mirror with extraction diagnostics.
**Quantitative context** — 72 commits, sole author, Feb–Aug 2026. 133 720 tracked Python lines. 2174 tests in 120 modules. 328 CLI flags. 31 constraint rows (23 hard / 7 final gates / 1 report-only). 10–11 design variables. 61 model-registry entries across 17 subsystems. 10 registered-unavailable outputs. 5 propellants, 8 materials, 4 packaged property tables. 7 benchmark cases, 8 reference curves. 49 documentation files. 255 PDFs in the literature corpus.

---

## 19. Concise descriptions

### One sentence

LREKit is a liquid-rocket-engine preliminary-design toolkit under active development — currently a single-design-point tool that sizes nozzle, chamber, regenerative-cooling, pintle-injector, and electric-pump-feed hardware and exports validated B-rep CAD, with a coupled differentiable (JAX) engine model whose exact total derivatives drive a constrained optimizer, built toward a requirements-driven multipoint system whose every reported number is either measured from the exported geometry or explicitly refused.

### Three sentences

LREKit is intended to become a design tool that takes engine requirements — thrust, specific impulse, mass, envelope, throttle range — and returns a fabricable, mass-accounted, multipoint-feasible engine definition, and then to serve as the platform for studying whether whole-engine optima interact with the Rao nozzle existence boundary.

What is built today is the physics and optimization core of that system: four contour construction methods including a variational Rao boundary-value solver that reproduces Rao's own published geometry to 0.0089 throat radii, a NASA/JHU method-of-characteristics kernel port at bit parity, a coupled differentiable engine model with two nested implicit-function-theorem solves feeding exact Jacobians to an SLSQP optimizer under 31 named constraints, B-rep CAD with kernel round-trip validation and geometry-linked mass, and a provenance layer in which ten engine outputs are registered unavailable with reasons rather than estimated.

What remains between that and the intended system is multipoint operation (currently refused rather than approximated), whole-engine dry mass, post-optimum verification at higher fidelity with signed deltas, a configuration-hashed object binding the solved state to the exported geometry, and the physical validation — CFD, conjugate heat transfer, FEA, cold flow, hot fire — that none of the current results have.

### Technical paragraph

LREKit is a 133 000-line Python toolkit for liquid-rocket-engine preliminary design, developed by a single author over six months. It runs two parallel pipelines against one versioned output contract: a sequential orchestrator that sizes a contour, chamber, regenerative jacket, pintle injector, and electric pump feed and exports B-rep STEP and STL geometry; and a multidisciplinary-feasible differentiable layer in which nozzle performance, a 1-D+fin cooling march, injector hydraulics, and the pump/battery chain solve as one system, with converged implicit states differentiated by the implicit function theorem and exact reverse-mode Jacobians handed to an SLSQP ε-constraint optimizer over ten design variables and 23 enforced constraints. The most developed component is the nozzle work: an axisymmetric method-of-characteristics solver, a stagnation-normalized port of NASA/JHU's `MOC_Grid_BDE` kernel reimplemented in JAX at bit parity, and Rao's 1958 variational optimum-thrust problem posed as a boundary-value problem with fifteen scaled residual groups, which reproduces Rao's published Nozzle B contour to 0.0089 throat radii. Verification is extensive — 2174 passing tests, NumPy↔JAX parity at 1e-10, autodiff-versus-finite-difference gates, held-out property-surface gates, CAD round-trip volume checks — and physical validation is absent, a distinction the codebase enforces by registering ten engine outputs as unavailable with reasons and by stamping every contour `hardware_qualified = False`. Active work is closing the contour-provider and hardware-mass tracks; the intended system adds multipoint throttle envelopes, whole-engine dry mass measured from built geometry, post-optimum verification reporting signed deltas per discipline, and a configuration-hashed object binding the analyzed engine to the exported one.
---

*Audit performed 2026-08-25 against commit f763fe1 · All "verified" results were executed on the audit host (Apple M3 Pro, macOS 15.7.5, Python 3.12.13, JAX 0.6.2, NumPy 2.4.6, SciPy 1.17.1, CadQuery 2.7.0, RocketCEA 1.2.3) Technical source document — not application, portfolio, or recommendation prose.*
