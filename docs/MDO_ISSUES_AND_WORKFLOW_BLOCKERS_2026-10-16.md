# MDO solver: open issues and what keeps it out of the part-geometry workflow

Date: 2026-10-16. Working tree on top of `6c04d92` (uncommitted, including other
in-progress work on the electric-drive convention and `python -m raosim`).

Sources: the *LREKit Consolidated Technical Review and Forward Program*
(70 pp., Oct 7; cited as "review §n"), `docs/` (chiefly
`DIFFERENTIABLE_ENGINE_MDO_PLAN.md`, `MDO_GUIDE.md`,
`MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md`, `ROADMAP_REQUIREMENTS_TO_PARTS.md`,
`LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md`), the knowledge graph
(`knowledge/kg.sh`, `knowledge/reports/KNOWLEDGE_GRAPH_AUDIT_2026-10-16.md`,
`knowledge/reports/pdf_issue_disposition_2026-10-15.md`) and the corpus PDFs it
indexes (`propulsion_texts/`, with `propulsion_texts_for_agents/` used only to
locate passages).

Evidence tags: **[run]** re-executed in this session; **[code]** read in the
current code; **[doc]** stated in the repository or knowledge-graph records
and not re-executed here.

## 0. What changed in this session

* **Pintle component** (`raosim/mdo/pintle.py`). The pintle is now an
  optimizable MDO component: an outer-enumerated architecture
  (`fixed_slotted` default, `son_movable` = Son 2017 valve), two new design
  variables (`t_post`, `D_cg`; the contract grew from 11 to 13 values, state
  schema v3), post/rod hardware mass in the partial dry-mass objective, a post
  pressure screen, the Son minimum opening at the lowest declared throttle
  level, centre-gap and module fits, and `t_post` carried to the injector CAD
  inputs. See `MDO_GUIDE.md` §3a.
* **Phantom constraint removed.** The Son transition row was enforced on the
  slotted post, which has no moving tip; it now applies only to the valve and
  carries a Son 2017 `source_id` (review §15.6 listed it among the binding rows
  without a source).
* **`D_pintle` is no longer inert.** Review §15.6 observed it ending "exactly at
  its 20 mm start". With the post mass in the objective the 13 kN optimum now
  moves it to 16.0 mm with the blockage band active (BF = 0.90) and the post
  wall at its 0.4 mm floor **[run]**.
* **NaN in the reverse pass fixed.** When the L\* closure is active the solved
  barrel length can be exactly 0. Coincident stations then made
  `dseg = sqrt(dx² + dr²)` return the correct value but a NaN derivative
  (0 × ∞), which raised in the cooling IFT linear solve and aborted the NLP.
  The committed HEAD raises at the same physical point **[run]**, so the
  defect predates this change; the new variables only steered SLSQP onto it.
  `grid.station_segment_lengths` keeps forward lengths bit-identical and takes
  the zero subgradient.

| 13 kN LOX/RP-1, Isp ≥ 190 s | Before | After |
|---|---|---|
| Partial dry mass (objective) | 35.754 kg | 35.761 kg (+7.4 g pintle post) |
| Isp | 224.77 s | 224.77 s |
| D_pintle / t_post | 20.0 mm (unchanged from start) / — | 16.0 mm / 0.40 mm |
| Verdicts | all pass | all pass |
| Son valve, 40 % throttle floor | — | 35.791 kg, all pass; D = 15.4 mm, D_cg on the bore fit, transition active |

## 1. Issues inside the MDO solver

### 1.1 Numerical robustness

1. **One bad iterate aborts the whole optimization.** Any non-finite value
   reaching a Lineax solve raises an Equinox runtime error rather than being
   reported as an infeasible point. One instance is fixed above; nothing
   guards the class generally **[run]**. The review's 225 s floor failure
   (§15.3, §7 item 27) shows the same message. It did **not** reproduce on this
   machine at HEAD (225 s converged, 35.81 kg), so whether it was this instance
   is unproven.
2. **A frontier sweep is all-or-nothing.** `pareto_frontier` calls each solve
   with no per-point isolation (`raosim/mdo/nlp.py:569`), and the CLI wraps the
   whole sweep in one `try` (§7 item 27) **[code]**.
3. **No independent KKT check.** SciPy success plus feasibility stands in for
   stationarity and complementarity (§7 item 16) **[code]**.
4. **No multi-start**, so an infeasible requirement set and a failed
   nonconvex solve look the same (§7 item 15) **[code]**.
5. **No grid-convergence evidence.** The station grid is 24 stations
   (4 + 6 + 13 + 1); the plan's station-doubling criterion has no test
   (review §15.5) **[code]**.
6. **Liner buckling is only a post-solve gate**, not an SLSQP row, because the
   implicit root is not robust at valid-box corners
   (`raosim/mdo/constraints.py:205`) **[code]**.
7. **The contour chart is piecewise bilinear, not C¹.** The optimum sits on its
   lower knot (ε = 4.0), where derivatives switch **[run]**.
8. **Cost.** Single cold solves took between 22 s and about 15 min in this
   session (same code; different Isp floors and machine load). Sweeps are
   host-only and exceed the CI budget **[run]**, **[doc]**
   (`DIFFERENTIABLE_ENGINE_MDO_PLAN.md` Phase 9 note).

### 1.2 Physics scope

9. **Thrust cannot see the contour.** The closure is 1-D
   `Cf = eta_CF * Cf_ideal` (`raosim/mdo/engine.py:167`, `:319`; §7 item 8)
   **[code]**.
10. **One contour provider, with a known internal shock.** Only the analytic
    TOP/Bézier provider runs in the loop. The MOC remediation found a
    compression coalescence about 20 mm past the inflection in that contour.
    No exact Rao provider runs in the loop, and B3 qualification is open
    **[doc]** (`MDO_GUIDE.md` §5, review §4).
11. **One operating point.** Multipoint jobs are refused
    (`raosim/mdo/job.py:715`; Phase 10 not started). The new valve throttle
    row is an algebraic single-point surrogate, not a throttle mission
    **[code]**.
12. **Cooling is 1-D plus fin.** Quasi-2-D (Phase 4b) has not started. Coolant
    real-fluid surfaces are unwired, so authoritative LOX/LCH4 and LOX/LH2
    runs stop at the HTD coverage gate **[doc]** (`MDO_GUIDE.md` §2b, plan
    Phase 2).
13. **η_c\* is an input.** The spray→c\* loop is a screening knob, off by
    default **[code]**.
14. **The pump is a C¹ efficiency surrogate.** There is no impeller or inducer
    geometry and no blade stress in the loop. The traditional `pumps.py`
    sizing, which the parts come from, fails at 500 N before any geometry
    exists (§11.4, §7 items 4 and 19) **[doc]**.
15. **Feed losses are placeholders.** `line_dp_allowance = 0.0`
    (`raosim/mdo/schema.py:568`) and there is no valve model, so pump duty is
    understated **[code]**.
16. **Film cooling is load-bearing but has no hardware.** The optimum uses
    18 % film. Neither pipeline models a film injector, and nonzero-film
    thermal parity is unavailable **[run]**, **[doc]** (`MDO_GUIDE.md` §9.8).
17. **Structural screens are preliminary.** LCF life is explicitly
    unavailable **[doc]**.
18. **Property coverage gaps.** N2O/ethanol is blocked on the oxidizer inlet
    state, and held-out CEA/CoolProp re-evaluation is open (Phase 2/11)
    **[doc]**.

### 1.3 Objective and mass

19. **The objective is mostly unsourced electric constants.** About 94 % of the
    objective is the electric package, and the MDO and `pumps.py` disagree on
    every density: battery power 1.3 vs 3.0 kW/kg, battery energy ≈180.6 vs
    250 Wh/kg, motor 3.0 vs 2.5 kW/kg, inverter 20 vs 15 kW/kg
    (`raosim/mdo/schema.py:599-600`, `raosim/pumps.py:29-32`). Neither set has
    a recorded basis (§7 item 28) **[code]**. The Lee inverter-denominator
    convention was corrected in the uncommitted tree (knowledge-graph audit).
20. **Only a partial mass is optimized.** The objective adds the chamber
    metal and now the pintle post. It still omits the injector body,
    faceplate and orifice plate, and the flange and bolts, which the hardware
    ledger shows dominate modelled mass (18.1 kg at 13 kN, `ROADMAP` R2). It
    also omits valves, lines and mounts; total dry mass stays `None`
    **[code]**, **[doc]**.
21. **The optimum is set by bounds and coverage edges.** It sits on ε = 4.0
    (chart edge), t_wall 0.4 mm and t_post 0.4 mm (floors), a zero-length
    barrel (L\* row active) and BF = 0.90. Physics trades are not what
    determines it **[run]**.

### 1.4 Provenance

22. **Eight hard rows have no `source_id`:** `land_fit`, `blockage_lo`,
    `blockage_hi`, `pintle_wall_fit`, `wall_temp`, `property_domain`,
    `chart_domain`, `wall_monotonic` **[run]**. The knowledge graph reports the
    constraint manifest as `SCIENTIFIC_BASIS_UNKNOWN`, and 241 current
    unknown-basis constants. `raosim/mdo/injector.py` symbols do not resolve
    in the graph **[run]** (`kg.sh basis`).
23. **Burn-time defaults disagree.** `--burn-time` defaults to 10 s
    (`raosim/run_nozzle.py:2942`) against 120 s in `MissionSpec` (§7 item 7).
    It does not move the 13 kN optimum, because the battery is power-limited
    **[code]**, **[doc]**.

## 2. What keeps the MDO out of the part-geometry workflow

1. **It is not the default path.** `lrekit` runs the traditional sequential
   pipeline; the MDO is opt-in (`--engine-mdo`, `--engine-mdo-optimize`)
   **[code]**.
2. **Parts come from a second pipeline, not from the MDO geometry.**
   `--mdo-export` calls `reevaluate()`, which rebuilds the engine in
   `design_nozzle_v2` from the MDO design vector, and that pipeline writes the
   CAD. The station grid the MDO analysed is not the exported wall, and no
   `ResolvedEngineConfiguration` binds solve, CAD, mass and BOM; only
   `ResolvedEngineInputs` exists (§7 item 5) **[code]**, **[doc]**.
3. **The requirements-driven job path cannot emit parts.** Capabilities are
   `parts = none` and `cad_fidelity = L0_numerical_only`; any `cad_formats`
   request is rejected (`raosim/mdo/job.py:631`); the `pareto` workflow is not
   wired (`raosim/mdo/execute.py:335`, `:394`); mass and envelope requirements
   are refused as unavailable (`raosim/mdo/job.py:927`) **[code]**.
4. **No production assembly orchestrator.** `build_engine_parts` /
   `ENGINE_PART_MANIFEST` is exercised only by `tests/test_mdo_build.py`. Of 27
   parts, 11 reach L2, 4 reach L1 and 12 are numerical only; none reaches L3
   or L4 (review §11.1) **[code]**, **[doc]**.
5. **The trusted contour cannot become CAD.** `--contour-method rao-bvp`
   refuses CAD (`raosim/run_nozzle.py:1806`), and the default TOP contour
   carries the internal compression (item 10) **[code]**.
6. **MDO/traditional parity is unadjudicated.** There are no field-specific
   tolerances or adjudication rule, and the cooling correlations and station
   geometry differ materially (`MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md` §11)
   **[doc]**. The partial-mass comparison now also carries the expected delta
   between the MDO post tube and the production pintle body and tip.
7. **The pintle valve has no handoff.** `son_movable` optimizes numerically,
   but `to_design_input` and `--mdo-export` refuse it. The host's movable
   branch is report-only and needs calibrated Cd, stops, metrology and
   actuator evidence (`docs/MOVABLE_PINTLE_MODEL.md`) **[code]**.
8. **Other thrust classes fail before geometry.** At 500 N the pump path
   fails before any geometry exists; at 130 kN the bolt layout fails while
   sizing reports feasible (§7 items 4, 14) **[doc]**.
9. **No manufacturing definition.** There are no tolerances/GD&T, process
   routes or inspection outputs, which L4 needs (review §11.2) **[doc]**.
10. **Cryogenic coolants are refused.** Authoritative MDO runs for LOX/LCH4
    and LOX/LH2 stop at the coverage preflight unless
    `--allow-incomplete-physics` is given, and that mode cannot return a
    physics verdict **[doc]**.

## 3. Suggested order

1. Make solver failure a typed infeasible point rather than an exception
   (1.1 items 1–2), add the KKT residual, and add multi-start (items 3–4).
2. Bind the analysed and exported geometry in one
   `ResolvedEngineConfiguration`, and drive the CAD from it (2.2).
3. Wire `parts` / `cad_formats` through `execute_engine_job` for the
   L2-capable parts, using `build_engine_parts` as the manifest (2.3–2.4).
4. Source or replace the eight unsourced hard rows and one electric-drive
   constant set (1.3 item 19, 1.4 item 22).
5. Then take on the model scope: multipoint (Phase 10), contour-dependent
   thrust, quasi-2-D cooling, and the valve handoff.

## 4. Repair status (2026-10-16, second pass)

A repair pass used a Sonnet research agent to page-check the held sources and
one Opus executor, then a Sonnet adversarial reviewer and a Sonnet independent
verifier; the coordinator fixed the confirmed review findings. Fast suites:
see the final run recorded in the session. Numbers below are from re-runs of
the final code **[run]**.

| Item | Status | What changed / evidence |
|---|---|---|
| 1.1.1 one bad iterate aborts the NLP | **Repaired** | `nlp._GuardedProblem`: a failed trial point gets a backtracking penalty; a failed derivative falls back to central differences; an iterate with no derivative at all stops that run as a typed failure (`derivative_failures`); a failure at the returned or start point is a typed `success=False` result. Tested on analytic engines with injected Lineax-style failures |
| 1.1.2 sweep all-or-nothing | **Repaired** | `pareto_frontier` isolates each floor; only successful, KKT-confirmed points seed warm starts; CLI prints a `FAILED` row, keeps sweeping, exits 4 |
| 1.1.3 no KKT check | **Repaired** | `nlp.kkt_check` (NNLS multipliers on the scaled active set and bound faces). It now **gates** `success`/`workflow_status`; unified jobs report `statuses["kkt"]`. 13 kN optima: stationarity 1.6e-7 (190 s), 3.7e-7 (225 s), 2.0e-6 (Son valve) |
| 1.1.4 no multi-start | **Repaired** | `n_starts`, seeded scrambled Sobol, compiled once; 8 starts at 13 kN: 2 basins, the reference basin best (+1.06 % second basin); 4 starts: 1 basin |
| 1.1.5 no grid-convergence evidence | **Measured** | Fixed design: coking margin moves 4.9 % of scale (not met). Re-optimized: −0.25 % (×2), −0.39 % (×4), same active set, KKT pass (met). `convergence.station_refinement_study`, `reoptimized_refinement_study` |
| 1.1.6 liner buckling only a final gate | Open | unchanged |
| 1.1.7 chart not C¹ | Open | New finding: the Rao/TOP θn/θe tables disagree with the held SP-125 worked example (PDF p. 107: 27.4°/9.8° vs tables 31.6°/16.7° at ε 14, Lf 0.8); left unchanged pending an owner decision |
| 1.2 physics scope (items 9–18) | Open | out of scope for this pass (new physics or sources) |
| 1.3.19 two electric constant sets | **Repaired** | `raosim/electric_technology.py`: one registry read by `pumps.py` and `MissionSpec`; both default to `repository_screening_v1` (the former traditional values, still unsourced, published brackets recorded). Page-verified `lee2021_ijass_electric_pump` selectable (`--electric-technology`); it warns outside its 600 s burn. 13 kN optimum 35.761 → **26.227 kg** |
| 1.3.20 partial mass only | Open | unchanged (pintle post now included) |
| 1.3.21 optimum on bounds/edges | Open | KKT confirms stationarity, but the active set is still ε chart edge, wall floors, zero barrel, BF 0.90 |
| 1.4.22 unsourced hard rows | **Repaired (labelled)** | every manifest row declares `basis_class`; `blockage_lo` now cites Freeberg 2019 (TRW BF 0.3–0.7, secondhand); `chart_domain` cites the SP-125 Fig. 4-14 / SP-8120 Fig. 5b chart extent as a model-validity domain; `wall_temp` (800 K), `blockage_hi` (0.90), `pintle_wall_fit`, `pintle_post_stress`, `liner_buckling` are labelled `repository_rule` because no held source states the implemented number or relation; geometric and domain rows labelled as such |
| 1.4.23 burn-time default | **Repaired** | `--burn-time` defaults to none: MDO keeps 120 s, traditional keeps 10 s; an explicit value reaches both |
| 2.1 not the default path | Open | unchanged |
| 2.2 analyzed vs exported wall | Open | no `ResolvedEngineConfiguration` yet |
| 2.3 jobs cannot emit parts | **Partly repaired** | `parts="reference_cad"` + `cad_formats=("step",)`: the re-evaluation writes STEP into the run, `build_engine_parts` writes the 27-part manifest, and the run manifest hashes the unattributed STEP. No part is attributed or earns fidelity (path-only evidence is unverified by design), so the package is always partial |
| 2.4 no production orchestrator | Partly | the part manifest builder now runs inside unified jobs |
| 2.5–2.10 | Open | unchanged |

`knowledge/kg.sh rebuild` refreshed the graph's stale line locators; the
knowledge suite now passes (287 tests) and the graph records the 36-row
manifest. Still open: unified jobs cannot yet select an electric technology
set or the Son valve.
