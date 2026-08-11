# Engine-MDO remediation, output contract, and parity review

**Review baseline:** commit `e57c93e` (2026-07-26)  
**Contract versions:** `EngineState` schema 2; `EngineAnalysisSnapshot` 2.0.0  
**Exact implicit Rao status:** architecture proposed, not implemented pending approval

## STATUS (2026-08-10)

Remediation items 1–8, 10, 12, 14–18 are **implemented and verified by
reproduction**; the fixed-O/F sentinel crash, the discarded optimized O/F, the
three divergent constraint lists, both requirement-integrity bypasses, the
fictitious fallback O/F domain, the electric-package-only mass objective, and
the unreachable HTD gate no longer reproduce. Full suite: 1924 passed.

Remaining work, in the order it should be taken:

| Item | State | Gap |
|---|---|---|
| 9 — `ResolvedEngineInputs` | contract landed and load-bearing; builder switch pending | `raosim/mdo/resolved_inputs.py` defines the frozen, versioned (`1.0.0`), content-addressed contract; `resolve_engine_inputs()` builds it; `crosscheck_design_input()` compares 40 shared scalars against the traditional `DesignInput`. Every `reevaluate()` now resolves the contract, crosschecks the handoff, warns by name on any divergence, and records the digest + material selection in snapshot `optimizer_metadata`. Current drift across GRCop-84/NARloy-Z/CuCrZr: **zero**. Remaining: switch `to_design_input()` from *builder* to *consumer* of the contract — now a mechanical, gate-protected step rather than a rewrite on faith. |
| 11 — injector build-once | partial | `injector_mass_ledger_from_built_parts` measures the exported body, but `design.py` still calls the labelled screening proxy and the export test uses monkeypatched fakes. CadQuery 2.7.0 is present in `.venv-jax`, so the real volume/STEP round-trip test is runnable. |
| 13 — chamber mass vs CAD | partial | `regen_cad.py` builds disjoint liner/ribs/jacket regions with a `geometry_id`; `design.py` still exports uniform-wall. Disclosed, not hidden. |
| 19 — `ContourProvider` | not started | `rao_variational` is reachable from the traditional CLI only; no provider interface, no post-solve validation of the MDO optimum. |

An unresolved coolant identity (uncatalogued propellant) now marks both the
coking and HTD rows applicable-but-unavailable, so it reduces to `unknown`
rather than silently dropping the governing wall-side gate.

### Material selection is now part of the MDO

Previously `--material` reached only the traditional pipeline while the
differentiable core kept flat class defaults, so selecting GRCop-84 optimized a
NARloy-Z-class liner. Those defaults were not even one alloy — NARloy-Z
conductivity and density against a CuCrZr-class allowable. The error was
design-changing, not cosmetic:

| liner | `wall_temp_margin` | `structural_stress_margin` |
|---|---|---|
| unattributed default | −126.2 (infeasible) | +134.7 MPa |
| GRCop-84 | **+66.7 (feasible)** | +34.3 MPa |
| NARloy-Z | −116.2 | +14.5 MPa |
| OFHC Copper | −218.9 | **−15.6 MPa** |

`raosim/mdo/material_map.py` is now the single typed mapper. It is **atomic**:
either every field a selection owns resolves from one catalog record, or the
selection is rejected — there is no partial application and no per-field
fallback, because a half-applied material is an alloy that exists in no catalog
and matches no qualification data. Liner and structural closeout are selected
separately (SP-8087 sec. 2.1.3.1) and are never inherited from one another.
`MissionSpec.for_material()` / `.with_materials()` are the entry points, and
`_mission_from_mdo_args` applies them so the CLI flag reaches the traced core.
A mission carrying class defaults reports `liner_material_name = None` and the
snapshot says `unattributed_class_default` rather than naming an alloy the MDO
never traced.

## 1. Bottom line

The MDO work is worth retaining, but its proper role is a differentiable
optimization and screening layer—not a replacement for every traditional
analysis.

Commit `e57c93e` created a useful end-to-end derivative path through nozzle
performance, regenerative/film cooling, a pintle injector, and electric feed.
That enables coupled design trades and exact derivatives through the two
implicit numerical solves. Before this remediation, however, it was not a
drop-in alternative to the traditional pipeline:

- several nominal inputs and performance conventions differed;
- the MDO exposed a smaller, incompatible result object;
- some missing mass branches appeared as zero placeholders;
- optimizer feasibility did not fully depend on root convergence/finiteness;
- the postprocessor did not map all shared inputs and did not run the
  traditional electric-pump sizing;
- most tests compared JAX code with simplified mirrors, not with
  `design_nozzle_v2`; and
- the chart-based analytic TOP wall was described too strongly as the exact Rao
  solution.

Those interface and correctness problems are now repaired. On a representative
LOX/RP-1 point (`Pc=3 MPa`, `epsilon=8`, 10% fuel film), the aligned MDO and
traditional pipelines agree to numerical precision on the common ideal and
delivered performance convention: thrust, mass flow, `Cf`, `c*`, `Isp`, exit
Mach/pressure, O/F, throat size, and both throat-radius ratios.

The new comparison also makes the remaining model discrepancy visible instead
of hiding it. At that same point, representative relative differences are
about 42% in peak gas-side wall temperature, 6.5% in jacket pressure drop, 22%
in the normalized radius profile, and 1.0% in electric power. These are not
regression-test tolerances; they are evidence that the MDO thermal/grid
surrogates and the traditional models are not yet physically interchangeable.
The traditional pump BOM also lacks some core component mass estimates, so the
contract correctly marks traditional pump and complete electric-package mass
unavailable rather than comparing an incomplete subtotal.

The right workflow is therefore:

```text
differentiate and optimize with MDO
             |
             v
capture pure-JAX EngineState
             |
             v
run design_nozzle_v2 + size_electric_pumps on the optimum
             |
             v
build authoritative EngineAnalysisSnapshot
             |
             v
compare every common scalar/profile; bind reports/CAD metadata to that snapshot
```

## 2. What the latest commit introduced

Commit `e57c93e` changed 38 files (`+6477/-63`). Its main additions were:

- the MDO schema, unit-box scaling, propellant table, and optional chamber
  property surfaces;
- a fixed-topology analytic TOP station grid;
- a regenerative/film cooling march with an implicit wall-temperature solve;
- differentiable pintle and electric-feed screening blocks;
- a coupled outer `(Rt, mdot)` solve;
- a 10-variable constrained SLSQP problem and Pareto sweep;
- CLI entry points for point evaluation and optimization;
- an initial post-optimum bridge;
- a literature audit and MDO guide; and
- discipline, integration, derivative, and CLI tests.

That was a substantial and coherent research implementation. Its main weakness
was not the idea of differentiation; it was the absence of a stable public
output boundary and a real cross-pipeline acceptance test.

## 3. Corrections made in this remediation

| Problem | Correction | Result |
|---|---|---|
| LOX/RP-1 defaults drifted from the traditional database | Aligned O/F, `gamma`, `Rgas`, `Tc`, `eta_cstar`, and `eta_CF`; propellant factories now carry the same split | Common ideal/delivered performance closes to numerical precision |
| `eta_CF` was missing from MDO thrust/Isp closure | Added separate ideal and delivered `Cf`; apply `eta_CF` independently of `eta_cstar` | No ambiguous combined efficiency inside `Cf` or `c*` |
| One throat-radius field represented both sides of the throat | Added distinct `Ru/Rt=1.5` and `Rd/Rt=0.382` inputs and outputs | Geometry matches the repository TOP convention and records both values |
| MDO chart interpolation used a different smooth surface than the traditional chart oracle | Replaced it with pure-JAX bilinear interpolation matching SciPy linear interpolation; added chart-domain and monotonic-wall gates | Same tabulated angle convention, with no silent extrapolation |
| The grid ignored active property-surface `gamma` | Pass active `gamma` into the station-grid Mach solve | CEA/property-table runs no longer use stale fallback Mach values |
| CEA-table-backed MDO points were re-evaluated by launching live RocketCEA (or a built-in fallback) | Pin the solved `EngineState` chamber snapshot (`gamma`, `Tc`, `Rgas`, ideal `c*`, and the complete surface fingerprint) into `design_nozzle_v2` | The parity lane now compares downstream models at identical thermochemistry; an independent CEA validation remains a separate correction/qualification lane |
| Separation mixed attached-flow ratio and design reserve | Preserve raw `Pe/psep`, then enforce `Pe/psep - 1.2`; vacuum is a finite explicit pass | Correct semantics and no `inf`/`nan` vacuum gate |
| Film fuel could be counted in both the film and jacket branches | Define a bypass topology: film gets `mdot_fuel*film_frac`; the jacket gets the remainder | Fuel routing is mass-conservative for the declared architecture |
| The traditional pipeline interpreted reduced jacket flow as a broken fuel closure | Add an explicit film-bypass branch and split ledger while retaining total fuel upstream of the common pump and pintle | Nonzero-film optima now pass a real traditional fuel-flow closure; separate film-injector hardware remains unavailable |
| Custom efficiency splits were discarded by constant-gamma/CEA resolution | Add explicit `ThermoSpec.eta_cstar` and `eta_CF` overrides and propagate the solved effective value | Default and custom efficiency conventions survive the real traditional pipeline |
| Feed density/vapor-pressure inputs drifted into backend defaults | Pass complete, explicit constant-liquid feed states; MDO values govern shared properties and the traditional resolver supplies only unmodeled terms | Pump comparisons no longer measure accidental property-backend drift |
| Structural screens compared thermal-only/post-FOS stress with combined/yield stress | Use the same SP-125 combined-stress convention and carry an explicit structural FOS | The MDO allowable maps to traditional yield without applying the FOS twice |
| Film-system reserve was only a note | Add an explicit installed-capacity assumption and `capacity - 2*design_flow` constraint | The SP-8087 capacity recommendation is testable |
| Numerical solver failure could masquerade as a feasible design | Carry Optimistix status, maximum residual, convergence, and finiteness through cooling, engine, NLP, state, and snapshot outputs | Final `success`/`feasible` requires numerical closure |
| Several exact extrema/selectors were awkward for gradients | Use conservative smooth envelopes for multi-stream NLP selectors and the battery objective; retain exact branches/extrema in `EngineState` | Differentiability no longer destroys reporting fidelity |
| CLI pump speed and pintle diameter did not reach their design-vector leaves | Map both CLI controls explicitly | User inputs now change the intended MDO variables |
| Zero film was a flat, poor SLSQP start when coking required film | Move the design-space reference to 0.10 while retaining zero as a valid bound/design | Feasible optimization is robust from the default reference |
| Finite-difference derivative test used a step below useful implicit-solve subtraction scale | Move the unit-box central-difference step to the observed `1e-4` plateau | AD/FD test measures derivative error rather than root-solver noise |
| MDO results lacked the traditional output breadth | Add the two-layer versioned output contract described below | Both pipelines have one auditable consumer interface |
| A solved state could be paired with a different host mission | Embed a fixed-shape numerical input-convention block and validate it during snapshot adaptation | Host reporting rejects mismatched tank, material, efficiency, feed, geometry, or electrical assumptions |
| Mission/schema checks occurred only after the traditional report/CAD run | Validate state schema, design, mission, property-surface identity, and coupling mode before constructing `DesignInput` or creating output directories | A stale or mismatched state cannot generate misleading artifacts before failing |
| The property-surface identity omitted derivative fields and admitted aggregate collisions | Fingerprint every evaluator-defining grid/value/derivative array plus shape, name, and provenance; accept caller-supplied surfaces only when the exact object fingerprint matches | Custom surfaces remain supported without false acceptance or false rejection |
| Smooth objective mass and exact physical mass shared one field name | Retain exact-governing and smooth-objective battery/package totals separately | Snapshot comparisons never compare unlike mass definitions |
| Postprocessing hard-coded/omitted conventions and treated performance like a dictionary | Map all shared inputs, read the actual performance dataclass, call both traditional solvers, and preserve the full results | Post-optimum comparison is end-to-end rather than a scalar sketch |
| Equivalent architectures used pipeline-specific labels that were silently skipped | Compare categorical scalars exactly and canonicalize the shared pintle/feed labels | Topology, method, coolant, injector, and feed convention drift is now visible |
| Traditional report/CAD metadata was written before the authoritative snapshot existed | Add the versioned snapshot path, SHA-256 digest, source, contract version, and optimizer metadata to generated JSON report/CAD sidecars after re-evaluation | Downstream report/CAD consumers have one explicit authoritative analysis handoff |
| Artifact-attachment failures were returned only as transient warnings | Converge the warning-bearing snapshot report and downstream handoff hash, then persist attachment failures in the returned snapshot and JSON bundle | The authoritative contract records failures in its own provenance instead of silently claiming a complete handoff |
| Missing chamber/injector/pump masses could look real | Use numerical `NaN + availability` inside JAX and `None + reason` on the host; reject incomplete traditional pump BOM totals | Unsupported output is explicit data, never a zero or partial total |
| Missing authoritative pump sizing counted as a passed whole-engine branch | Mark whole-engine and physics feasibility unavailable with a reason while retaining the traditional design/readiness gate separately | “Not evaluated” can no longer become “passed” |
| MDO `rao-bvp` export dropped traditional solver controls and ignored the requested upstream throat radius | Forward backend, grid sizes, iteration budget, angle seed, starting-line/wall method, kernel cap, physics weight, and both throat-radius ratios into the existing host solve | The same CLI request is configuration-equivalent in traditional and MDO post-analysis |
| `rao-bvp` was a silent no-op without `--mdo-export`, and no-CAD analysis still claimed STEP metadata | Reject the no-op request, block non-Bezier manufacturing CAD, and emit `authoritative_cad=null` for numerical-only analysis | Preliminary host analysis cannot be mistaken for the MDO core or manufacturing geometry |
| Scheduled ambient pressure was used inconsistently by sizing and reported performance | Resolve one finite, nonnegative design pressure below `Pc` and use it for sizing, contour, performance, gates, and structure | A scheduled case is one coherent operating point |

## 4. Layer 1: pure-JAX `EngineState`

`raosim/mdo/state.py` defines schema version 1 as nested `NamedTuple` pytrees.
Every leaf is a fixed-shape numerical or Boolean JAX array. No strings, Python
dictionaries, paths, report objects, or variable-length topology are present.

The state retains:

- the design vector, schema version, and the full numerical input convention
  needed to reproduce the operating point;
- complete ideal/delivered performance (`Cf`, `c*`, thrust, exhaust velocity,
  and `Isp`);
- total, oxidizer, total-fuel, core-fuel, film, regenerative-jacket, and
  core-total mass flows;
- station coordinates, radius, area ratio, Mach, segment length, throat index,
  masks, throat radii, region lengths, and geometry-validity values;
- wall, coolant, heat-flux, coefficients, recovery temperature, fin
  augmentation, coking, thermal/pressure/combined-stress fields, jacket duty, and the full cooling
  residual;
- injector pressure drops, velocities, areas, momentum ratio, spray angle,
  blockage, branch areas, and stability/transition margins;
- fuel and oxidizer pump duty, efficiency, powers, head, specific-speed
  quantities, cavitation/tip-speed margins, and mass;
- total electrical power and separate battery energy/power branches;
- cell-level, installed, exact-governing, smooth-objective, and electric-feed
  package masses;
- a stable ordered constraint vector; and
- outer/cooling residual vectors, maximum norms, status, convergence, and
  finiteness.

JAX cannot put Python `None` in a numerical pytree without changing the tree
structure. Unsupported numerical fields therefore use `NaN` together with a
fixed-shape Boolean availability vector. The host adapter converts that pair to
`None` plus a stable reason.

The differentiable optimizer minimizes the explicitly named
`electric_feed_package_objective_mass`: the battery governing branch is a
smooth log-sum-exp surrogate.  The state and host snapshot separately report
`electric_feed_package_exact_mass`, formed with the exact maximum of the
installed energy- and power-limited battery branches.  CLI and optimizer
metadata preserve both names; the surrogate is never presented as a physical
“package total.”

The current MDO does **not** have qualified thrust-chamber/nozzle or injector
hardware-mass models. Those state entries are unavailable, and total engine dry
mass is consequently unavailable. The optimized objective is an
electric-feed-package screening mass, not total engine mass.

## 5. Layer 2: host `EngineAnalysisSnapshot`

`raosim/mdo/snapshot.py` defines contract version 1.0.0. Both
`snapshot_from_mdo()` and `snapshot_from_traditional()` produce the same
top-level sections and the same version-1 field manifest inside every section:

- `performance`
- `geometry`
- `thermal`
- `cooling`
- `injector`
- `feed_electrical`
- `masses`
- `constraints_gates`
- `provenance`
- `warnings`
- `artifacts`

Every field is a `SnapshotValue`. An unavailable field must have
`value=None` and an `availability_reason`; an available field cannot have an
availability reason. This invariant is enforced by the constructor.

Profiles use a throat-aware normalized axial coordinate. The chamber/convergent
side maps to `[-1, 0]`, the throat is zero, and the divergent side maps to
`[0, 1]`. `compare_snapshots()` walks every common numeric scalar and every
common normalized profile. Categorical analysis scalars (for example topology,
method, injector architecture, and feed architecture) are compared by exact
equality. It returns a delta, exact-match status, or an explicit
`not_comparable_reason`; it does not silently drop missing physics.

The traditional snapshot preserves:

- the full `ValidatedDesignResult` as its authoritative source result;
- the full electric-pump result as an auxiliary result;
- all generated report/CAD paths;
- traditional design gates and pump gates; and
- optimizer method, convergence, iteration, target, and feasibility metadata.

When `reevaluate(..., output_dir=...)` is used, the report/CAD handoff is
written as `engine_analysis_snapshot_v1.json`. Its primary payload is the
traditional authoritative snapshot. The MDO screening snapshot and the
all-common-field comparison are attached as evidence. CAD consumers obtain
paths from `authoritative_snapshot.cad_artifacts`; report consumers use
`authoritative_snapshot.report_payload()` or the JSON bundle. Generated JSON
design reports and CAD metadata/manifest sidecars are augmented with the
snapshot path and SHA-256 digest, contract version, authoritative source, and
optimizer metadata. Binary geometry is not rewritten.

## 6. Input/convention alignment

The post-optimum conversion now maps shared assumptions before comparing any
outputs.

| Quantity | MDO source | Traditional destination |
|---|---|---|
| propellant / coolant | `MissionSpec.propellant_name` and MDO propellant table | `ThermoSpec` identities and `CoolingSpec.coolant` |
| O/F | `MissionSpec.OF` | `ThermoSpec.mixture_ratio` |
| chamber thermochemistry | solved `EngineState` `gamma`, `Tc`, `R_gas`, ideal `c*`, and property-surface fingerprint | immutable `PinnedChamberState` consumed by `design_nozzle_v2` in the parity lane |
| `eta_cstar`, `eta_CF` | solved effective c-star efficiency and separate Cf efficiency | explicit `ThermoSpec.eta_cstar` / `eta_CF`; `eta_Isp` records their product for legacy consumers |
| ambient pressure | `MissionSpec.Pa` | `MissionAmbientSpec.Pa` |
| material | liner name, `k`, post-FOS allowable/temp, `E`, `alpha`, `nu`, density, structural FOS | `MaterialSpec` yield = allowable × FOS plus the same explicit FOS |
| throat radii | `throat_ru_factor`, `throat_rd_factor` | `ThroatGeometrySpec` |
| chamber geometry | contraction ratio and `L*` | `DesignInput` |
| channels | count, width, height, roughness, wall thickness | cooling/manufacturing specs |
| cooling split | actual MDO regenerative-jacket and wall-film branches | `CoolingSpec.coolant_mass_flow` plus `fuel_film_mass_flow`; split ledger enforces jacket + film = total fuel |
| tank/line pressures | fuel/oxidizer tank pressures and line allowance | `FeedSystemSpec` |
| pump speed/efficiency | design `N_rpm` and solved per-stream efficiencies | `PumpSizingSpec` |
| injector | pintle type/architecture, diameter, slot count, both pressure-drop fractions, both `Cd` values | `InjectorSpec` |
| feed properties | fuel/oxidizer density and vapor pressure; constant-liquid fuel viscosity | explicit `PropellantFeedSpec`; backend supplies only properties absent from the MDO |

Two limitations remain explicit.  First, the traditional workflow records the
regen/film mass split but does not apply the MDO wall-film effectiveness/heat
load or size a separate film injector/orifice and branch-specific pressure
state.  At nonzero film flow the traditional snapshot therefore marks wall
temperatures, heat flux, coolant temperature, and thermal/combined stress
fields unavailable with this reason.  Both snapshots also mark main-pintle
TMR/spray/fuel-area/velocity and dependent geometry fields unavailable: the
internal total-fuel pintle calculation is retained only as an optimizer
screening surrogate and is not presented as a physical core-fuel injector.
Film mass, split closure, common-pump duty, jacket hydraulics,
coolant-pressure profile, and pressure-stress profile remain comparable.  A
zero-film end-to-end case exercises common thermal, structural, and injector
profiles/scalars.

The declared MDO topology has exactly two downstream fuel branches: all
non-film fuel traverses the regenerative jacket and the remainder is film.
`MissionSpec` consequently requires `cooling_fraction == 1`; a lower value
would create an untracked third bypass and is rejected until that branch has
its own state and closure constraint.

Second, the handoff declares the MDO's constant-liquid property assumption; it
does not claim that a near-critical jacket outlet has been validated as
single-phase liquid.  For an actual `EngineState` re-evaluation, the parity lane
uses `pinned_chamber_state`: it passes the exact solved chamber-property values
and full surface fingerprint into the traditional workflow instead of
resampling live CEA or accepting a fallback.  This makes downstream differences
attributable to the compared models, but it is not an independent
thermochemistry validation.  A live/held-out CEA re-evaluation remains a
separate correction and qualification step.

The contour method is an explicit host option.  The default remains the
established Bezier/TOP path.  The existing auditable host BVP route is
`rao_variational_moc` (`rao` is the older experimental prototype).  Both
non-Bezier routes remain preliminary and cannot currently emit manufacturing
CAD or claim the validated-mode gate set; neither is the unapproved exact
implicit JAX replacement described in the architecture checkpoint.

## 7. Post-optimization authoritative workflow

`reevaluate()` now performs the following sequence:

1. obtain or solve the MDO `EngineState`;
2. validate its schema, design, mission, exact property surfaces, and coupling
   convention before any output work;
3. pin its solved chamber thermochemistry and map every remaining shared
   convention into `DesignInput`;
4. call the real `design_nozzle_v2`;
5. rehydrate its two-stream feed ledger;
6. call the real `size_electric_pumps` with aligned speed, efficiencies,
   densities, burn time, battery assumptions, tank pressures, line losses, and
   injector drops;
7. build MDO and traditional snapshots;
8. compare every common scalar and normalized profile;
9. preserve both rich results and all warnings/gates; and
10. write the authoritative snapshot, converge any artifact-attachment warnings,
   and bind generated report/CAD metadata to that handoff when export is
   enabled.

The CLI passes the already solved point result directly for a single design.
For an optimized result it records SLSQP status, feasibility, iterations,
message, Isp floor, maximum violation, and coupling mode. Native IPT remains
deferred; STEP is the authoritative geometry supplied for `step`, `ipt`, or
`both` MDO export requests.

## 8. End-to-end parity tests

The added tests are deliberately not only self-parity tests:

- `tests/test_mdo_state.py` jits and differentiates the complete fixed-shape
  state, verifies profile shapes, checks the ideal/delivered convention, checks
  declared fuel routing, and prevents zero availability sentinels.
- `tests/test_mdo_snapshot.py` runs a real nonzero-film MDO engine solve, real
  `design_nozzle_v2`, and real `size_electric_pumps`; it verifies exact
  cross-pipeline parity for the aligned performance, split, feed-state, and
  core-geometry fields; imposes field-specific discrepancy ceilings on common
  geometry/hydraulic/pump results; proves film-sensitive thermal fields are
  explicitly non-comparable; runs a separate zero-film common-profile case;
  verifies identical field manifests; checks custom efficiencies,
  material/FOS/feed inputs, state/design/coupling mismatch rejection, pressure
  stress semantics, bounded zero-film thermal/structural/cooling/injector
  discrepancies, pinned custom-surface thermochemistry, early mismatch
  rejection, unavailable pump-feasibility semantics, and unavailable mass
  semantics; preserves both rich results; persists attachment failures; and
  reads every emitted JSON report/CAD sidecar back to verify its optimizer
  metadata and digest-bearing authoritative-snapshot link.
- Existing MDO engine/NLP tests now cover chart-oracle parity at interior points
  and all chart corners, explicit out-of-domain violations, finite JIT
  derivatives, active `gamma`,
  Ru/Rd, explicit efficiency split, separation reserve/vacuum behavior, film
  routing/capacity, root status/finiteness, and AD/FD agreement.

Profile/thermal differences are intentionally reported by the parity test; they
are not declared equivalent merely so a test can pass.

## 9. Evidence record from the local propulsion corpus

The source search started in
`propulsion_texts/propulsion_texts_for_agents/paper_index.md`, used the Markdown
mirrors for navigation, and checked the load-bearing statements against the
original PDFs because these conversions are marked `needs_review`.

### Direct source claims

- **Separation reserve:** J. C. Hyde and G. S. Gill, *Liquid Rocket Engine
  Nozzles*, NASA SP-8120 (1976), §3.1.2.1.3, original PDF p. 82 / printed p. 68,
  recommends reducing expansion or using a nonseparating contour when exit wall
  pressure is within about 20% of separation pressure. This is an
  overexpanded/ground-test rule, not a vacuum margin. Mirror:
  `propulsion_texts/propulsion_texts_for_agents/markdown/19770009165.md`;
  original: `propulsion_texts/19770009165.pdf`.
- **TOP chart role:** the same report, §2.1.2.1.2, original PDF p. 29 / printed
  p. 15, Figure 5(b), gives initial/final wall angles versus expansion ratio and
  60–100% length families. Section 3.1.2.1.2, original PDF p. 81 / printed
  p. 67, treats chart extrapolation as a trial estimate that still requires
  efficiency calculation. This supports a chart seed/surrogate, not silent
  extrapolation or a particular high-order interpolation.
- **Throat-radius distinction:** NASA SP-8120, original PDF pp. 24–25, reports
  nearly constant aerodynamic efficiency over a range of upstream radius
  ratios and calls downstream `Rd/Rt` near 0.4 a usual compromise. Jerry
  Seitzman, *Rocket Nozzle Geometries* (2012), slide 15 / original PDF p. 8,
  records the repository-style approximate TOP construction with 1.5 upstream
  and 0.382 downstream. Rao's 1958 example used a different downstream value;
  therefore 0.382 is a repository/TOP convention, not a universal “canonical
  Rao” constant.
- **Combined regenerative and film cooling:** C. W. Van Huff and A. J.
  Fairchild, *Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, NASA
  SP-8087 (1972), original PDF pp. 65–67, discusses combined systems; the design
  criteria on original PDF p. 108 require film-flow capability of twice the
  estimated required flow and allow growth up to 100%. It does not require the
  film branch to bypass regeneration.
- **Battery branches:** H. Kwak, S. Kwon, and C. Choi, “Performance assessment
  of electrically driven pump-fed LOX/kerosene cycle rocket engine” (2018),
  DOI `10.1016/j.ast.2018.02.033`, §3.2.4, original PDF p. 8, defines separate
  energy- and power-limited battery mass expressions and their governing
  maximum; original PDF p. 13 discusses the burn-time transition. Mirror:
  `propulsion_texts/propulsion_texts_for_agents/markdown/fuel_pump_design/1-s2.0-S1270963817320953-main.md`;
  original:
  `propulsion_texts/fuel_pump_design/1-s2.0-S1270963817320953-main.pdf`.
- **Exact Rao validity:** G. V. R. Rao, “Exhaust Nozzle Contour for Optimum
  Thrust” (1958), DOI `10.2514/8.7324`, original PDF pp. 3–4, supplies the
  variational/control-surface, characteristic-compatibility, mass, and length
  construction. G. V. R. Rao, J. Beck, and T. Booth, AIAA 99-2584 (1999),
  original PDF pp. 2–6, separates valid/invalid characteristic regions and
  minimum-length behavior. Its numerical map is case-specific, not a universal
  MDO domain.

### Repository design decisions, not literature claims

The following are engineering choices made for this codebase:

- film bypasses the regenerative jacket;
- installed film capacity is 60% of total fuel;
- bilinear interpolation is used to match the existing chart oracle;
- smooth envelopes are used in selected differentiable constraints/objectives;
- `NaN + availability` is the pure-JAX representation of unsupported fields;
- the throat-aware normalized profile coordinate;
- the versioned snapshot schema and authoritative JSON bundle; and
- the proposed square KKT/IFT exact-Rao architecture, continuation windows,
  padded topology, and multifidelity staging.

The local corpus does not contain a primary IFT/MDF software-architecture
reference. Those choices must be reviewed as repository engineering, not
described as source-validated propulsion physics.

## 10. Exact differentiable Rao decision

Calling the existing `solve_rao_bvp_jax()` directly from `mdo/grid.py` is not
the recommended next change. Its public shell still owns variable topology,
seed construction, host objects, and reliability gating, while only the inner
least-squares work is JAX. Direct use would produce incomplete derivatives and
value-dependent output shapes.

The proposed architecture is documented in
`docs/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md`. In brief, it uses:

- traced `RaoDesignParams`;
- static padded topology and explicit masks;
- a fixed-shape pure-array `RaoState`;
- one square KKT/root system with IFT derivatives;
- a traced transonic start line;
- explicit numerical, physics, and Rao-validity judgments;
- a fixed normalized-wall sampler for cooling; and
- staged adoption: post-optimum exact analysis, existence-aware screening,
  trust-region multifidelity correction, then optional exact-in-loop research.

No exact-Rao replacement has been implemented. That is the requested approval
checkpoint.

## 11. Remaining work before claiming solver equivalence

1. Choose field-specific parity tolerances and an adjudication rule for cases
   where both pipelines are screening models.
2. Reconcile or deliberately select the authoritative thermal/cooling
   correlations and station geometry; current profile differences are material.
3. Add a qualified film-injector/orifice and branch-pressure model if the film
   circuit is to contribute hardware mass or independent pump duty.
4. Add configuration-controlled chamber/nozzle and injector hardware-mass
   models before optimizing or reporting whole-engine dry mass.
5. Complete the traditional pump BOM mass model before comparing total pump or
   electric-package mass.
6. Approve/revise the exact Rao/JAX architecture before replacing the analytic
   TOP grid.
7. Validate optimized candidates with higher-fidelity thermochemistry,
   real-fluid cooling, pump maps, structural/life analysis, and test evidence;
   neither pipeline by itself qualifies flight hardware.
