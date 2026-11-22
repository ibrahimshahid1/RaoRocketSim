---
title: LREKit Differentiable Engine-MDO — Reviewer's Guide
subtitle: What every module does, where each model comes from, and how to run it
date: 2026-08-26
audience: whoever is reviewing or extending raosim/mdo/
---

# Reviewer's Guide to the Engine-MDO Layer

This is the orientation document for `raosim/mdo/`. It answers four questions:
**what is here**, **where each model comes from**, **how to run it**, and **what
is still soft**. Companion documents: `DIFFERENTIABLE_ENGINE_MDO_PLAN.md` (the
formulation + phase tracker) and `LITERATURE_AUDIT_2026-07-24.md` (the
claim-by-claim source verification).

---

## 1. The 60-second version

`raosim/mdo/` turns the existing sequential LREKit models into **one coupled,
end-to-end differentiable engine** and drives it with a gradient-based
constrained optimiser. The inner numerical engine is JAX; its converged
implicit states are differentiated by the implicit function theorem rather than
by unrolling a solver, and each inner block keeps a NumPy oracle and parity
test. Host contracts such as snapshots, authoritative postprocessing, and the
opt-in variational contour validator intentionally remain outside that traced
inner loop.

Run it two ways from the existing CLI:

```bash
lrekit                      # interactive: pick workflow, then its parameters
lrekit --engine-mdo         --pc 3e6 --epsilon 8 --film-frac 0.10   # one point
lrekit --engine-mdo-optimize --isp-min 200                          # optimise
```

The traditional nozzle/CAD workflow is still the default.  The MDO bridge now
validates the solved state before creating artifacts, pins its exact chamber
thermochemistry, passes explicit efficiency/feed/film/material conventions into
the host workflow, and augments its JSON reports/CAD metadata with the
authoritative snapshot handoff.

---

## 2. Module map

| File | Role | Key entry points |
|---|---|---|
| `schema.py` | Design variables, bounds, and **every physical constant with its citation** | `DesignVector`, `MissionSpec`, `default_design_space()` |
| `scaling.py` | Affine map to the unit box (conditioning) | `ScaledSpace` |
| `properties.py` | C¹ property surfaces over (P_c, O/F) | `load_chamber_surfaces`, `sample_cea_tables` |
| `grid.py` | Fixed-topology station grid on the analytic Rao/TOP approximation | `build_station_grid`, `rao_wall_angles` |
| `nozzle.py` | Typed analytic TOP provider and provider-failure contract | `ContourProvider`, `resolve_contour_provider` |
| `contour_validation.py` | Opt-in host TOP-versus-variational post-solve adjudication; not an inner-loop provider | `VariationalPostSolveConfig`, `validate_optimized_top_contour` |
| `cooling.py` | Regen jacket + film cooling; the one **implicit** block | `solve_cooling`, `film_effectiveness` |
| `injector.py` | Pintle: orifice/TMR/blockage + Son two-branch area | `injector_readouts` |
| `pump.py` | Electric feed: duty, C¹ efficiency, separate battery energy/power branches | `electric_feed` |
| `engine.py` | **The coupled solve** — all blocks, one gradient | `solve_engine`, `engine_outputs` |
| `nlp.py` | ε-constraint optimiser + Pareto sweep; fail-closed trial points, independent KKT check, multi-start (§4a) | `solve_min_mass`, `pareto_frontier`, `kkt_check` |
| `convergence.py` | Station-refinement (grid-convergence) study, plan §11 criterion (§4a) | `station_refinement_study` |
| `../electric_technology.py` | **One electric-drive technology set for both pipelines** (§4b) | `ElectricTechnologySet`, `MissionSpec.with_electric_technology` |
| `assembly.py` | The original 4-variable walking skeleton (kept as a gate) | `make_engine_fn` |
| `state.py` | Versioned, fixed-shape pure-JAX numerical output pytree | `EngineState`, `solve_engine_state` |
| `snapshot.py` | Versioned host contract shared by MDO and traditional analyses | `EngineAnalysisSnapshot`, `snapshot_from_mdo`, `snapshot_from_traditional`, `compare_snapshots` |
| `postprocess.py` | **Phase 11 bridge** → authoritative contour/electric pumps/report-CAD handoff + parity report | `to_design_input`, `reevaluate`, `summarise` |

**Where to start reading:** `engine.py` docstring → `solve_engine` → then whichever
block interests you. `schema.py` is the reference for every constant.

---

## 2a. Thrust scaling — the solver is NOT 13 kN-specific

`MissionSpec.for_thrust(F)` derives the whole architecture from physics rather
than hard-coded 13 kN values, and `mission.scaled_design_space()` scales the
bounds with it:

* **throat** from the thrust closure, A_t = F/(C_F P_c);
* **chamber length** from the characteristic length, L ≈ L*/CR — SP-125 Table
  4-1 gives L* = 40–50 in for LOX/RP-1, and note this makes chamber length
  essentially *thrust-independent* at fixed L* and CR;
* **channel count** ∝ chamber circumference at a fixed manufacturable pitch;
* **pintle diameter** sized so the blockage factor lands mid-band (BF being the
  literature design knob — Hwang 2022, Freeberg 2019), not an invented ratio;
* **pump speed** from the specific speed ω = N_s(g₀H)^¾/√Q, so bigger engines
  correctly run *slower*.

Sanity check: at 13 kN the derived architecture reproduces the hand-tuned
defaults (194 channels vs 192, D_p 22.6 vs 20 mm). Without this, a 130 kN run
has **no feasible point in the box** and every Pareto row comes back infeasible.

```python
m = MissionSpec.for_thrust(130e3)      # 615 channels, D_p 71.6 mm, 19.3 krpm
```

## 2b. Propellant is a selectable input

`MissionSpec.for_propellant(name, thrust)` drives the chamber gases (γ, T_c, R),
**L\*** (SP-125 Table 4-1), densities, coolant thermophysical properties and the
**coolant wall limit** (SP-8087) from `raosim/mdo/propellants.py`:

| combination | L\* | coolant wall limit | status |
|---|---|---|---|
| LOX/RP-1 | 45 in (SP-125 40–50) | 728 K (SP-8087 850 °F) | verified feasible |
| N2O4/MMH | 32 in (SP-125 30–35) | 589 K (SP-8087 hydrazine family) | verified feasible |
| LOX/LH2 | 35 in (SP-125 30–40) | **none** — no carbon, cannot coke | see caveat |
| LOX/LCH4 | 40 in *(estimate)* | 950 K *(estimate)* | see caveat |
| N2O/Ethanol | 40 in *(estimate)* | 589 K (SP-8087 alcohols) | untested |

Hydrogen carrying **no** coking limit is the physically correct behaviour, not a
missing number — the gas-side material limit governs instead.  Methane and
N2O/ethanol post-date SP-125/SP-8087, so their L\*/wall limits are flagged
`estimated=True` in the table rather than passed off as literature.

```bash
lrekit --engine-mdo --mdo-propellant n2o4/mmh --target-thrust 5000 --pc 2e6
```

**Caveat — cryogenic coverage is incomplete.** Authoritative LOX/LCH4 and
LOX/LH2 optimization now stops at model-coverage preflight because the traced
cooling march lacks the real-fluid coolant surfaces required for the applicable
HTD screen. `--allow-incomplete-physics` permits an explicit screening run, but
its physics verdict remains unknown. The LH2 specific-speed sizing can also ask
for a speed inappropriate for a single stage; tank pressure, NPSH, and pump
stage count still need propellant-specific resolution.

## 3. The design layout (12 or 13 physical variables)

| Variable | Bounds | Why it is a variable |
|---|---|---|
| `Pc` | architecture-labelled recommended window; electric-pump default 1.5–6.0 MPa, explicitly overridable | chamber pressure; hard admissibility comes from property/model domains and live limits |
| `eps` | Rao/TOP table domain (currently 4–50) | expansion ratio |
| `dp_f_frac`, `dp_o_frac` | 0.12–0.45 | injector Δp / P_c (chug rule ≥ 0.2) |
| `D_pintle` | mission-scaled around the resolved reference pintle | pintle post OD: hardware mass vs blockage band (slotted) or Son opening and centre-gap fit (valve) |
| `N_rpm` | mission-scaled around the specific-speed reference | pump speed → specific speed → efficiency |
| `channel_width` | 0.3–1.2 mm | jacket geometry |
| `channel_height` | 0.8–5.0 mm | jacket geometry (AR ≤ 8 enforced) |
| `film_frac` | 0–0.30 | fuel diverted to wall film — **the coking lever** |
| `t_wall` | 0.4–2.0 mm | thin for conductance, thick for structure |
| `t_post` | 0.4 mm – D_ref/4 | pintle post wall: hardware mass vs post pressure screen; for the valve it also sets the Son tip radius |
| `D_cg` | rod diameter – 2 D_ref | Son valve centre gap: transition area vs the bore it must fit inside. **Pinned** for the slotted post |

O/F is the thirteenth variable only for an explicitly variable layout backed by
a validated O/F-dependent sampled property table. Fixed mode keeps a 12-value
physical optimizer vector but every state/snapshot stores a 13-value physical
contract vector containing the real effective O/F; no numerical sentinel is
used. `D_cg` is always in the contract; for the fixed slotted pintle the NLP and
job layer pin it at its reference, so it never becomes a dead free column.

### 3a. The pintle component (`mdo/pintle.py`)

The pintle is an optimizable component with its own variables (`D_pintle`,
`t_post`, `D_cg`), its hardware mass in the partial dry-mass objective, and
constraints chosen by an outer-enumeration architecture
(`MissionSpec.pintle_architecture`, plan rule 4):

| Architecture | What it is | Rows that apply | CAD handoff |
|---|---|---|---|
| `fixed_slotted` (default) | radial fuel through N slots in a fixed post; the five-part coaxial CAD and the water cold-flow article | blockage band, post stress, wall fit, module fit | yes: `t_post` reaches `PintleMechanicalSpec.pintle_wall_thickness` |
| `son_movable` | Son et al. 2017 continuous-gap pintle **valve** metered by a centre rod; throttles by stroke | Son transition, minimum opening at the lowest throttle level, centre-gap fit, post stress, wall fit, module fit | refused (`to_design_input`, `--mdo-export`); numerical design only |

Mass is direct prismatic geometry: a post tube over `pintle_body_length_factor`
× D (3.0, the host layout default) less the slot prisms (NumPy-parity with the
host `injector_mass_ledger` post row), plus the valve's centre rod. Material is
the closeout/jacket alloy, the host's injector structural alloy. The post
screen is `p r_o / t` with the radial stream's injector-inlet pressure against
ambient; on the outside radius it bounds the elastic Lamé bore stress for any
wall. The valve's throttle check holds the radial Δp fraction and scales Pc
with flow (host `throttle_map` convention), so the metered area scales as
f^½; Son 2017 PDF pp. 2 and 10 supply the 0.1 mm uniform-sheet floor. The
minimum opening is an applicability bound for the observed sheets, not a
universal hardware limit, and the throttle check is a single-point surrogate
for Phase 10.

Select it with `--mdo-pintle-architecture son_movable --mdo-pintle-throttle-min
0.4` (CLI) or `dataclasses.replace(mission, pintle_architecture="son_movable",
pintle_throttle_min_fraction=0.4)`. Unified jobs do not offer the valve yet,
because they always run the host re-evaluation it has no handoff for.

**Every active variable is live.** A zero Jacobian column means a variable that cannot
reach any constraint or the objective; two such variables were found and fixed
(see §7). Re-check with the snippet in §8 after adding any variable.

---

## 4. One constraint manifest (36 rows)

`raosim.mdo.constraints.CONSTRAINT_MANIFEST` owns the stable row order,
applicability, availability, required mask, scaling, optimizer role, source
identifier and **basis class**. It currently contains 28 differentiable hard
rows, seven mandatory post-solve gates, and one report-only thrust-closure
equality. Requirement rows, the sampled-property row, the coking/HTD pair and
the architecture-specific pintle rows are active only when applicable.

Every row declares `basis_class` (2026-10-16): `literature` (the limit or its
governing allowable is stated in the cited source), `geometric_definition`,
`model_validity_domain`, `repository_rule` (a repository number; any
`source_id` cites only the criterion's form), `numerical_contract` or
`requirement`. A `literature` row must carry a `source_id`; the eight hard rows
that had none now carry either a page-verified source or one of the other
labels. The knowledge graph reports such rows as `DECLARED_<BASIS>` instead of
`SCIENTIFIC_BASIS_UNKNOWN`.

| Constraint | Meaning | Basis | Source |
|---|---|---|---|
| `isp_epsilon` | I_sp ≥ floor | requirement | the ε-constraint |
| `separation` | no flow separation | literature | Östlund Eq. 28–30, SP-8120 |
| `coking` | T_wc ≤ 728 K (RP-1 liquid-wall) | literature | SP-8087 (850 °F), Sellers 1961 |
| `land_fit` | channels physically fit the circumference (positive land) | geometric definition | — |
| `chug` | min(χ_f, χ_o) ≥ 0.2 preliminary screen | literature | SP-125 injector-design rule of thumb (15–20%, source-PDF p. 137 / printed p. 128); SP-194 supplies qualitative chug context |
| `pintle_transition` | Son valve stays on the tip-controlled branch (`son_movable` only) | literature | Son 2017 Eq. 1, PDF pp. 4, 9 |
| `pintle_post_stress` | post hoop `p r_o/t` ≤ closeout yield / SP-8087 FOS | repository_rule (sourced allowable, screening stress form) | outer-radius hoop screening assumption; SP-8087 §2.1.3 |
| `pintle_wall_fit` | t_post ≤ 0.8 r_o (keeps a bore) | repository rule | host machined-layout cap (`injector_cad.resolve_machined_pintle_layout`) |
| `pintle_min_opening` | Son opening at the lowest throttle level ≥ 0.1 mm (`son_movable` only) | literature | Son 2017 PDF pp. 2, 10 |
| `pintle_center_gap_fit` | D_cg ≤ D − 2 t_post (`son_movable` only) | geometric definition | Son 2017 Fig. 2 geometry |
| `pintle_module_fit` | axial-annulus OD < chamber diameter | literature | Son 2017 PDF p. 9 |
| `pump_suction` | N_ss ≤ 4 | literature | SP-8052 |
| `pump_tip_speed` | U₂ ≤ 400 m/s | literature | SP-8109 |
| `aspect_ratio` | channel h/w ≤ 8 | literature | Wadel 1997, Carlile 1992; Mirzamoghadam lists 8 (initial) and 7 (present criteria), PDF p. 7 |
| `blockage_lo` | BF ≥ 0.30 (`fixed_slotted` only) | literature | TRW advice 0.3 < BF < 0.7 relayed by Freeberg 2019 (DOI 10.2514/6.2019-0152), PDF pp. 5–6, 10 |
| `blockage_hi` | BF ≤ 0.90 (`fixed_slotted` only) | repository rule | no held source states 0.90; held evidence is 0.7 (TRW via Freeberg) and the Ryu optimum near 85 % with thermal damage at 86 % (Hwang et al., PDF pp. 2, 4, 8) |
| `structural_stress` | σ_thermal + \|σ_pressure\| ≤ post-FOS allowable | literature | SP-125 eq. 4-31 convention |
| `wall_temp` | T_wg ≤ 800 K | repository rule | criterion form Mirzamoghadam 1991 (PDF p. 5; Table 2 PDF p. 7, 1050 °F "Depends on Material"); 800 K is not in the paper (900 °F = 755 K is an H2/O2 copper value) |
| `film_capacity` | installed film circuit capacity ≥ 2× design film flow | literature | SP-8087 capacity recommendation plus explicit 60%-of-fuel architecture |
| `property_domain` | stay inside the sampled property table | model validity domain | the sampled CEA table box |
| `chart_domain` | stay inside the Rao/TOP chart extent (ε 4–50, L_f 60–100 %) | model validity domain | extent of SP-125 Fig. 4-14 (PDF p. 100) and SP-8120 Fig. 5(b) (PDF p. 29); the tabulated ordinates are NOT cited (§9 item 13) |
| `wall_monotonic` | divergent wall radius does not reverse | geometric definition | — |
| `chamber_volume` | the solved barrel satisfies the L* chamber-volume construction | geometric definition | SP-125 chamber definition (printed p. 88) |
| `jacket_thin_shell` | the selected shell approximation remains in its declared thin-shell regime | literature | SP-125 p. 336 |
| `nozzle_collapse` | closeout retains the modeled external-pressure/collapse margin | literature | SP-8087 structural load path |
| `contour_provider` | selected contour provider returned an available, feasible result | numerical contract | plan §0.1 rule 1 |
| `liner_buckling` | preliminary rib-supported liner external-pressure screen | repository_rule (sourced failure mode, screening relation) | SP-8087 failure mode; classical long-plate screening assumption |
| `engine_residual`, `cooling_residual`, `solver_status`, `finite` | inner solves closed, finite | numerical contract | plan §0.1 rule 6, §11 |
| `envelope_diameter`, `envelope_length` | active partial-envelope requirement margins | requirement | SP-125 §2.1 item 6 |
| `dry_mass_partial` | active resolved-partial mass requirement margin | requirement | SP-125 §2.1 item 5; explicitly not full engine dry mass |

Reported but deliberately **not** constrained: coolant Mach (≈2 orders of
margin — a constraint would just add a dead column). Engine/cooling residual
closure, root status, finiteness, provider availability, liner buckling, and
applicable coolant HTD coverage are mandatory final gates. An unavailable
required model yields an `unknown` physics verdict; it never becomes a passing
boolean.

## 4a. Solver robustness and optimality evidence (2026-10-16)

The audit `MDO_ISSUES_AND_WORKFLOW_BLOCKERS_2026-10-16.md` §1.1 found that one
bad iterate aborted the NLP, that SciPy success stood in for a KKT check, and
that there was no multi-start or grid-convergence evidence. All four are now in
`nlp.py` / `convergence.py`, each with tests on closed-form analytic engines
(`tests/test_mdo_nlp_robustness.py`, `tests/test_mdo_convergence.py`).

**Fail-closed trial points.** The compiled programs contain two implicit
solves; a non-finite value reaching a Lineax solve raises
`jax.errors.JaxRuntimeError` (Equinox `error_if`). `_GuardedProblem` answers a
primal failure at a trial point with a finite penalty (objective 10×|ref| above
the largest finite objective seen, every enforced scaled row at −10) so SLSQP's
merit line search backtracks; a derivative failure at a finite primal falls
back to central differences of the same compiled primal in the unit box (step
1e-6, one-sided at a face). An iterate where neither the exact derivative nor
the fallback exists (a non-finite gradient whose only probe also fails) stops
that run as a typed failure rather than handing SLSQP a zero-filled gradient,
which it would read as convergence. A failure at the *returned* point, or at
a start point itself, gives `success=False`, `numerical_status="fail"` and a
message; `solve_min_mass` does not raise for numerical failures.
Configuration errors (`ValueError`, `TypeError`, `ModelCoverageError`) still
raise. `NLPResult.evaluation_failures` / `derivative_fallbacks` /
`derivative_failures` count the three cases.
`pareto_frontier` isolates each floor: a failed floor is recorded and the next
floor warm-starts only from the last successful, gate-passing point; the CLI
prints a failure row and keeps sweeping, and `--mdo-export` skips it.

**Independent KKT check** (`nlp.kkt_check`), in the optimizer's own scaled
unit-box space: exact objective gradient and enforced-row Jacobian, active set
= scaled margin ≤ 1e-5 plus bound faces (u ≤ 1e-6 or ≥ 1 − 1e-6; a 1e-9
tolerance missed the Son valve's t_post at u = 1.5e-9 and reported a false
failure), multipliers
by non-negative least squares (so dual feasibility holds by construction),
stationarity `‖∇f − Jᵀλ − μ_lo + μ_hi‖∞ / max(1, ‖∇f‖∞)`, complementarity.
`kkt_status` is `pass` when stationarity ≤ 1e-4 (plan §11) **and** the point is
primal feasible to 1e-5, `fail` otherwise, `unknown` without derivatives or
free variables. The check is independent of SciPy's convergence flag but
reuses the same exact IFT derivatives; it is not a separate finite-difference
cross-check. **It gates the result:** `success` and `workflow_status` pass only
when `kkt_status` is `pass` (or `unknown` because every variable is fixed), so
a run that stalls on the edge of a failing region, which SciPy reports as
converged, is a failure. That makes the CLI exit 4, keeps the point out of a
Pareto warm start, and gives a unified job a `SOLVER_FAILED` verdict with
`statuses["kkt"]`. `optimizer_status` still reports feasibility alone. The
definition is the standard textbook one; no held source states it, and the
tolerances are the plan's repository definition of done.

**Multi-start** (`n_starts`, `start_seed`; CLI `--mdo-multistart N`): start 0
is the reference (or `u0`), the rest scrambled Sobol points in the unit box.
The compiled programs are built once. The returned result is the best start
passing optimizer and numerical status (physics pass, then convergence,
preferred), else the least-violating start; `start_summaries` and `n_basins`
(single-linkage: unit-box max-norm ≤ 1e-3 AND relative objective ≤ 1e-6, a
repository choice) record all starts.

**Station refinement**, two readings of the plan's < 0.5 % criterion.
`convergence.station_refinement_study` (CLI `--mdo-station-study`) holds one
design fixed, re-solves the engine with every segment count doubled
(24 → 47 stations), and reports objective/Isp (relative) and every hard-row
margin (|Δ| / manifest scale): the discretization error of that design.
`convergence.reoptimized_refinement_study` re-solves the whole NLP on the
refined grid (`solve_min_mass(..., topology=...)`, warm-started from the base
optimum) and compares the optima: objective, Isp, KKT status and active set.
Every block accepts a non-default `GridTopology`; no hard-coded station count
was found.

Measured at the 13 kN LOX/RP-1, Isp ≥ 190 s optimum (2026-10-16, after §4b):

| Evidence | Result |
|---|---|
| Single solve | 26.2274 kg, Isp 224.69 s, all five statuses pass, 26 iterations; Isp ≥ 225 s: 26.2309 kg, KKT pass |
| Son valve, 40 % throttle floor | 26.2569 kg, all pass, KKT pass (stationarity 2.0e-6); active: transition, centre-gap fit, coking, chug, chart_domain, chamber_volume + lower eps/t_wall/t_post |
| KKT | **pass**, stationarity 1.6e-7, complementarity 2.7e-12; active: coking, chug, blockage_hi, chart_domain, chamber_volume + lower bounds on eps, t_wall, t_post (rank 7 of 8: chart_domain and the eps bound are the same face, LICQ fails, residual unique) |
| 8-start multi-start (seed 0) | 2 basins: 6 starts reach 26.2274 kg; one reaches 26.5055 kg (+1.06 %, KKT pass; wider/shallower channels, film 0.216 vs 0.184); one fails ("Inequality constraints incompatible" from a far Sobol start). The reference basin is the best found. 4 starts: 1 basin, all KKT pass |
| Station doubling, fixed design (24 → 47) | objective +0.009 %, Isp unchanged (1-D closure), but the active **coking margin moves +14.6 K = 4.9 % of its 300 K scale: NOT met** for that design. ×4 (93 stations): +22.3 K (7.4 %). `wall_temp` moves 15 K; `wall_monotonic` (a min of station differences) is grid-dependent by construction |
| Station doubling, re-optimized | 24 → 47: optimum 26.2274 → 26.1606 kg (**−0.25 %**), Isp 224.69 → 225.30 s, film 0.184 → 0.179; 24 → 93: −0.39 %, Isp 225.62 s. Same active set and KKT pass on every grid: **met**. The optimum is still drifting downward with refinement, so the 24-station grid is slightly conservative |

## 4b. One electric technology for both pipelines (2026-10-16)

About 94 % of the objective is the electric package, and the MDO and
`raosim/pumps.py` used to disagree on every density. `raosim/electric_technology.py`
now holds complete `ElectricTechnologySet`s; `pumps.SCREENING_DEFAULTS` and
`MissionSpec`'s eight electric fields both default to `repository_screening_v1`
(the former `pumps.py` values: battery 3.0 kW/kg and 250 Wh/kg, κ_b 1.2,
η_disch 0.95, motor 2.5 kW/kg shaft at 0.90, inverter 15 kW/kg output at 0.96),
labelled a repository screening assumption with its published brackets — the
inverter density is below and the inverter efficiency above every held value,
and the constant discharge efficiency stands in for the Lee/Kwak burn-time
correlation. `lee2021_ijass_electric_pump` is the page-verified Lee et al. 2021
Table 2 set (PDF p. 11; η_E is Eq. 38 at 600 s). `MissionSpec.electric_technology`
names the set; a mission whose numbers disagree with its declared set is
refused unless the set is `custom`; `with_electric_technology(id)` swaps all
eight fields atomically; the MDO→traditional handoff and resolved inputs carry
the same set; CLI `--electric-technology`. `pump_specific_mass` stays an
MDO-only, flagged screening value (no held source on its basis).

Effect at 13 kN, Isp ≥ 190 s: **35.761 kg → 26.227 kg** partial dry mass (the
review predicted ≈26.3 kg); Isp 224.77 → 224.69 s; design essentially unchanged
(Pc 2.35 MPa, ε 4.0, film 0.184); the battery stays power-limited.

---

## 5. Physics, block by block — and its source

**Nozzle / performance.** Isentropic relations, Vandenkerckhove Γ, C_F with the
ambient term, `c*_delivered = η_c*·c*_ideal` (pinned in one place). SP-125,
SP-8120, Anderson.

**The bulk-MDO contour is the analytic Rao/TOP approximation, not the exact
variational Rao solution.** `grid.py` builds the conventional downstream throat
arc plus quadratic Bézier wall. Its θ_n/θ_e values use a pure-JAX bilinear
interpolator that matches the repository's SciPy linear chart oracle between
knots. It is piecewise differentiable, not C¹, and an explicit chart-domain
constraint prevents clamping from being accepted as extrapolation. The
typed `mdo/nozzle.py` provider seam preserves that analytic construction and
makes provider failure an explicit final gate. The authoritative post-optimum
contour is produced by `design_nozzle_v2`; its variational solver now refuses a
Rao Eq. (5)-invalid or unevaluable contour. The separate
`mdo/contour_validation.py` contract has now landed: when `reevaluate()` receives
an explicit `VariationalPostSolveConfig`, it preflights a B5 convergence domain,
runs the host BVP/forward MOC, requires Eq. (5) and direct full-C-D-E
performance, and then applies caller-supplied TOP-versus-variational discrepancy
tolerances. With no config, the default analytic path does not launch a
variational solve. This is not yet B3 qualification: historical coupled and BDE
probes returned `UNKNOWN` at distinct downstream numerical/topology audits. The
BDE probes failed the former raw neighbor-state-increment hard screen; that
quantity is now diagnostic only. The coupled probe failed a separate
characteristic-crossing audit and does not compute the BDE screen. The exact
implicit Rao/JAX replacement has its own approval checkpoint in
`IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md`.

**Cooling (`cooling.py`) — the only implicit block.**
- gas side: full **Bartz 1957** h_g, coefficient **0.026 verified verbatim**, with
  the σ property factor; turbulent recovery r = Pr^⅓.
- wall: series resistance + land **fin efficiency** tanh(mH)/mH and the
  `(w+2η_f h)/pitch` area augmentation (SP-125 §4; the HARCC mechanism of
  Pizzarelli/Carlile).
- coolant: **Sieder–Tate** 0.027·Re^0.8·Pr^⅓ via the audited primitive.
- march: **counterflow, upwind finite-volume** — counterflow is the
  literature-preferred arrangement for ε ≤ 7 (Mirzamoghadam).
- the stationwise wall-temperature vector is solved by Newton + IFT.

**Film cooling — liquid, not gaseous.** This is the subtlest part of the repo and
worth understanding before changing it. The classical
ε = C(X/VR)^(−0.8)Re^(0.2) correlations (Hatch & Papell TN D-130, Stollery,
Hartnett, Tribus) are fitted to **gaseous** coolants at velocity ratios
0.45–33.3. An RP-1 film is a *liquid*: continuity through any sane slot gives
VR ≈ 10⁻³, three orders below that band. So the block instead implements the
**liquid phase-change model** (Shine & Nidhi §4.3 — Kinney/Graham/Sellers/Emmons;
Huzel & Huang SP-125 Eq. 4-34):

1. per-mass enthalpy `H = c_p,l(T_w−T_co) + ΔH_vap + c_p,v(T_aw−T_w)`
2. hot-gas load `q = h_g(T_aw−T_w)`
3. film-cooled length `A_FCL = η_fc·ṁ_film·H/q`
4. a C¹ decay over that length.

`η_fc` is **derived, not fitted**: Stechman makes it a function of coolant
Reynolds number, and Grisson supplies the mechanism — below **Knuth's critical
flow-per-circumference** `Γ_cr = 1.01e5 μ_v²/μ_ℓ` the film is smooth; above it
waves shear droplets off and mass loss is "2 to 4× the evaporation rate". The
code blends 0.9 → 0.33 across that transition. Our engine runs at Γ/Γ_cr ≈ 9,
i.e. firmly wavy, so the model self-selects ≈0.33.

The gaseous velocity ratio is retained as a **diagnostic**
(`film_slot_validity`) and pinned by a test, so the inapplicability stays
documented.

The current plumbing topology is explicit: the selected film branch bypasses
the regenerative jacket, and the jacket receives the remaining fuel. SP-8087
supports combined regenerative/film systems and recommends film-flow capacity
of twice the estimated requirement; it does **not** require this particular
bypass topology. The bypass and the installed capacity of 60% of total fuel are
repository architecture choices and are recorded as such.

**Injector.** Incompressible orifice, TMR, blockage factor, and the **Son 2017
two-branch minimum area** (tip-opening vs centre-gap) with a consistency
inequality — never a differentiable `min()`. The pintle hardware, post screen
and valve opening live in the separate pintle component (§3a).

**Pump.** Meanline duty, N_s / N_ss / NPSH, a **C¹ η(N_s)** surrogate replacing
the shipped binned estimator (which is a C0 step and cannot be differentiated),
calibrated to the SP-125 60–85 % rocket-pump band, plus the **Lee 2021 battery
branches** (power- and energy-limited masses kept separate). The optimizer uses
a clearly named smooth governing-branch surrogate; the state, snapshot, CLI,
and NLP result also report the exact installed governing-branch mass.

---

## 6. How the coupling actually works

```
outer Newton on (Rt, mdot)  ──►  grid ──►  cooling (inner Newton on T_wg vector)
                                              │
                                     Δp_regen ▼
      pump feed ◄── Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank
                                              ▼
              P_electric, battery branches, exact mass, smooth objective
```

Two nested implicit solves, both differentiated by the IFT. The **hydraulic edge
is genuinely closed** — the jacket Δp feeds the pump rise. The one true two-way
loop (spray → η_c* → ṁ → spray) is **off by default**, because the plan flags it
as the strongest coupling but the weakest physics; `ablation_delta()` measures
exactly how much any result depends on it.

*Honest note:* most of the remaining data flow is a one-way chain, and it is
modelled that way deliberately (the plan warns against enlarging the nonlinear
system just to look coupled).

---

## 7. Two traps worth knowing about

**Dead design variables.** A variable with a zero Jacobian column *and* zero
objective gradient is worse than useless — it makes the NLP rank-deficient.
`film_slot_height` (physically inert for a liquid film) was removed;
`D_pintle` was made live by constraining blockage factor, but with no path to
the objective it still ended every solve at its starting value; it now also
reaches the objective through the pintle post mass. `D_cg` would be dead for a
slotted post, so it is pinned there. Always re-check.

**Forward-mode AD through nested implicit solves.** `jacfwd` silently drops the
tangent of geometry-only `jnp.min` margins through jit'd nested Optimistix
root-finds. The constraint Jacobian therefore uses **`jacrev`** (FD-verified).
Do not "optimise" this back to forward mode without re-running the FD gate.

---

## 8. Running and checking it yourself

```bash
# environment (sandbox/CI): jax 0.6.2, optimistix 0.0.11, equinox 0.13.8
export PYTHONPATH=/path/to/deps:$PWD

# the whole MDO suite
python -m pytest tests/test_mdo_*.py -q

# single design point, with design margins on
lrekit --engine-mdo --pc 3e6 --epsilon 8 --film-frac 0.10 --design-margins

# optimise, then trace the frontier (altitude makes the trade meaningful)
lrekit --engine-mdo-optimize --isp-min 200
lrekit --engine-mdo-optimize --isp-sweep 250,320,6 --engine-mdo-ambient 1000

# multi-start + grid-convergence evidence; another electric technology set
lrekit --engine-mdo-optimize --isp-min 190 --mdo-multistart 8 --mdo-station-study
lrekit --engine-mdo-optimize --isp-min 190 --electric-technology lee2021_ijass_electric_pump
```

Every optimizer line now ends with a `checks:` line: KKT status, stationarity,
complementarity, evaluation failures and derivative fallbacks (and starts/basins
with `--mdo-multistart`), then the active set. `--burn-time` defaults to *absent*:
the traditional pump sizing then uses 10 s, while `--engine-mdo/-optimize` and
`--requirements` keep `MissionSpec.burn_time` (120 s) — previously the CLI's
10 s silently replaced it.

Unified jobs can export the reference STEP: `output_policy` with
`parts="reference_cad"`, `cad_formats=["step"]`, an `output_dir` and
`allow_partial_package=true` (required, because the package is always partial).
The authoritative re-evaluation writes into `<run>/parts/`, and
`raosim.mdo.build` writes `engine_build_manifest.json` there. That module
records path-only files as unverified, so no part is "built" and nothing earns
geometry fidelity; the uniform-wall `thrust_chamber_wall.step` is listed as an
unattributed artifact, not as the liner. The run manifest records the part
manifest with `status: partial` and `hardware_release: false`.

```bash
# post-analyse an MDO point with the existing host Rao variational/MOC solver
# (preliminary numerical analysis only; this path cannot emit manufacturing CAD)
lrekit --engine-mdo --mdo-export --contour-method rao-bvp --cad none
```

`--contour-method rao-bvp` is deliberately rejected in an MDO command unless
`--mdo-export` is present. It configures the existing host post-analysis and
forwards the same backend, resolution, iteration, seed, throat-radius, kernel,
and physics-weight controls as the traditional CLI. It does not replace
`mdo/grid.py`. The proposed pure-JAX implicit Rao/KKT solver remains
architecture-only in `IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md`.

That CLI contour selection is distinct from the typed B3 comparison validator.
The latter is opt-in through `EngineJobSpec.models` (validator identity, B5
domain path, and all four discrepancy thresholds) or directly through
`reevaluate(..., variational_validation_config=...)`. It cannot pass without a
matching B5 declared-domain artifact. The historical first fixed-context artifact is
`raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json`
(canonical SHA-256
`a85de663fffced5559e92760fe933a7edd0ca269fc227aae5908c0eccbf87b19`);
schema v2 rejects it because v1 omitted the kernel angular-spacing and
mass-flow policies. It must not be used as current B5 evidence. The current
replacement is
`raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v2.json`
(canonical SHA-256
`2b2d1f4017f88d230cd6857dbc1e1a859ac5d3f611aa14a60f9e413d2c5949a9`),
with the explicit B3 validation profile
`kernel_dtheta_limit_deg=0.25` and `kernel_massflow_rel_tol=0.02`.

**Check for dead variables after any change:**

```python
from raosim.mdo.nlp import _make_callables, CONSTRAINT_NAMES, DEFAULT_ENFORCED
from raosim.mdo.schema import MissionSpec, default_design_space
import numpy as np, jax.numpy as jnp
m = MissionSpec(); idx = tuple(CONSTRAINT_NAMES.index(n) for n in DEFAULT_ENFORCED)
ss, obj, og, con, cj = _make_callables(m, 190.0, False, idx)
u = np.full(len(default_design_space()), 0.5)
J = np.asarray(cj(jnp.asarray(u))); g = np.asarray(og(jnp.asarray(u)))
for k, s in enumerate(default_design_space()):
    print(s.name, np.linalg.norm(J[:, k]), g[k])   # both zero => DEAD
```

---

## 9. What is still soft (read before trusting a number)

1. **Property surfaces.** The exact `catalog` selection loads packaged,
   hash-pinned, every-cell held-out-gated RocketCEA surfaces for LOX/RP-1,
   LOX/LCH4, LOX/LH2, and N2O4/MMH. Empty-table mode deliberately remains the
   constant γ/T_c model and is flat in O/F; explicit table paths remain
   supported. N2O/Ethanol is refused by `catalog` until its oxidizer inlet
   state/card is selected. The parity re-evaluation pins the exact
   `EngineState` γ/T_c/R/c* values and complete surface fingerprint into
   `design_nozzle_v2`; this prevents accidental live CEA/fallback drift. A
   separate live re-evaluation at the final optimum is still required by the
   authoritative Phase-11 correction/qualification lane.
2. **η_fc transition location.** Woodmansee & Hanratty measured Γ_cr ≈ 3× lower
   than Knuth for water; Shine reports Stechman's model errs −20 % … +13 %.
   Those are the film block's honest error bars.
3. **Pump-efficiency shape.** The 60–85 % band is verified; the peaked *shape* is
   a smooth screening fit, not SP-125 Fig. 6-23 (image-only).
4. **η_c*(TMR).** A screening knob, default off, with an ablation. Do not report
   a coupled result without also reporting `ablation_delta`.
5. **Low-cycle fatigue.** `structural_stress` is the combined static screening
   stress, not a cycle count. The traditional pipeline has a sourced
   Coffin–Manson/Basquin screen, but the MDO lacks a defined duty cycle and
   stabilized elastic-plastic strain-range closure. Consequently
   `life.cycles_to_failure` is explicitly unavailable and cannot be selected as
   an objective or target.
6. **Sea-level Pareto is weak.** Higher ε is overexpanded at sea level, so the
   mass–I_sp trade only becomes interesting with `--engine-mdo-ambient` set for
   altitude/vacuum.
7. **The optimized mass is not whole-engine dry mass.** The default
   differentiable objective contains pumps, motors, inverters, a smooth battery
   governing-branch surrogate, screening liner/land/closeout chamber masses,
   and the screening pintle post (plus the valve's centre rod).
   Exact installed electric and resolved-partial subtotals are reported
   separately. The traditional handoff now measures its five injector parts
   from the exact CadQuery bodies that are exported, with a shared geometry
   identity and STEP round-trip gate; its partial subtotal adds the measured
   pintle body and tip, which include the feed stub, flange and replaceable
   tip that the MDO post tube does not, so that parity delta is expected. The
   rest of the MDO injector mass and chamber/nozzle CAD-linked qualification
   remain unavailable, so the full engine total stays `None` plus an
   availability reason and the objective remains a partial screening subtotal.
8. **Nonzero-film thermal parity is intentionally unavailable.** The
   traditional bridge closes and reports the regen/film mass split, but the
   traditional thermal solver does not apply the MDO wall-film heat-load
   model and neither pipeline has a separate film-injector hardware/state
   model. Film-sensitive thermal and main-pintle fields are therefore `None`
   with a reason; a separate zero-film parity case compares common
   thermal/structural/injector outputs.
9. **Missing pump analysis is not feasibility.** If authoritative electric-pump
   sizing is skipped or fails, the traditional snapshot's whole-engine and
   physics-feasibility fields are unavailable with a reason. The
   chamber/nozzle/injector workflow-readiness gate remains separately reported.
10. **Unsupported output and invalid metadata stay explicit.** State schema,
    mission/design/coupling identity, and exact property surfaces are checked
    before reporting. Nested non-finite payloads cannot be marked available, and
    report/CAD attachment failures are persisted in the authoritative snapshot
    rather than returned only as transient console warnings.
11. **Exact-Rao post-solve promotion is still blocked.** The B5
    $\varepsilon\times L_\%$ schema and B3 typed validator are implemented, but
    the historical 8/12/16/24-ladder H2 map at fixed $\gamma=1.4$ and
    $P_a/P_0=0.01$ contains 40 of 65 `converged_valid` knots and 25
    `not_converged` knots, but its v1 identity omitted the inherited 0.5°/5%
    kernel policies. Schema v2 rejects it fail-closed. The regenerated
    0.25°/2% v2 artifact preserves the same 40/25 classification grid and
    records the complete numerical identity. It still supplies kernel-
    refinement evidence rather than proof of full mesh or full-BVP convergence.
    Historical 24×24 and 32×32 BDE probes passed numerical BVP, Eq. (5),
    full-C-D-E, and every other BDE/MOC audit, yet returned `UNKNOWN` under the
    former hard screen: raw neighbor $\max|\Delta M|$ was 0.11813 and 0.13752
    versus the repository diagnostic limit
    0.10. At the explicitly refined 0.25° kernel spacing these values fall to
    0.06999 and 0.06705. A 12×12 coupled probe does not compute this BDE screen;
    it fails a separate characteristic-crossing audit. Raw state increments
    remain diagnostics. The refined 24×24/32×32 BDE pair now passes B3's typed,
    spacing-aware common-coordinate physical-grid comparison; this one fixed
    reference does not qualify other design contexts. These
    tolerances are repository screening policy, not literature qualification
    limits; see `RAO_B3_B5_HOST_EVIDENCE_2026-08-24.md`.
12. **Fixed-design thermal margins are grid-sensitive.** At one fixed design,
    doubling the stations moves the active coking margin by ~5 % of its scale
    (§4a). Re-solving the NLP on the refined grids moves the optimum only
    −0.25 % (×2) and −0.39 % (×4) with the same active set, so the solved
    problem meets the plan criterion, but the optimum still drifts downward:
    the 24-station grid is slightly conservative. No grid was retuned to pass.
13. **The Rao/TOP angle tables need adjudication.** `nozzle_geometry.py`'s
    θ_n/θ_e tables (used by the MDO contour too) give 31.6°/16.7° at ε = 14,
    L_f = 0.8 where the held SP-125 worked example (PDF p. 107) reads 27.4°/9.8°,
    and their θ_e rises with ε while both held charts fall. Only the chart
    extent is cited (`chart_domain`); the values are unchanged pending an owner
    decision.

---

## 10. Provenance

Every physical constant in `schema.py` carries its source in a comment. The
claim-by-claim verification (with file and line numbers into
`propulsion_texts_for_agents/`) is in `LITERATURE_AUDIT_2026-07-24.md`, including
the corrections made along the way — the separation-criterion relabelling, the
gaseous-vs-liquid film correction, and the hoop-stress recommendation that turned
out to be the wrong criterion.
