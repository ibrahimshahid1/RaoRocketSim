# Roadmap: requirements in, parameterized parts out

Target end state: a user supplies **dimensions, or targets to optimize toward**
(a thrust target, an Isp floor, a mass budget, an envelope limit) plus whatever
other engine requirements they care about, and receives **parameterized,
manufacturable parts** — nozzle + throat + chamber assembly with flanges and
correctly sized bolts, injector parts, and electric pump parts.

This document assesses where the repository actually stands against that goal
and what to build next, in order. It supersedes the open-gap list in
`MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md` §12 for prioritisation purposes.

---

## Where the repository actually is

Three of the four pieces of the end state exist:

- **Physics.** Nozzle contour (Rao chart/Bézier and a host variational-MOC
  solver), Bartz gas-side heat transfer, coupled Sieder-Tate regenerative
  cooling with fin and curvature corrections, SP-125 eq. 4-31 combined wall
  stress, Coffin-Manson fatigue screening, pintle injector hydraulics and spray,
  centrifugal pump meanline with inducer and cavitation screens, electric drive
  and battery sizing.
- **CAD.** STEP/STL export for the nozzle wall, regen channels, machined pintle
  bodies, and pump components, with re-import validity gates.
- **Optimization.** A differentiable simultaneous MDO with implicit solves,
  gradients through the coupled engine, an SLSQP interface, and — as of this
  change — a real thrust-chamber mass in the objective space.

The fourth piece is the one that is genuinely missing, and it is the one that
turns the other three into the stated product:

> **There is no requirements layer.** A user cannot say "give me 5 kN at sea
> level, Isp ≥ 250 s, under 30 kg, fits in a 300 mm envelope" and get parts.
> They must already know `Pc`, `ε`, `L*`, contraction ratio, channel count,
> channel width, channel height, wall thickness, pintle diameter, slot count,
> pump RPM, bus voltage — roughly forty coupled decisions. The tool sizes a
> design; it does not yet *take a requirement*.

Everything below is ordered by how directly it closes that gap.

---

## Status

| item | state |
|---|---|
| R0 chamber-length convention | **done** — wetted-area parity 0.8828 → 1.00102, mass delta 12.06 % → 0.044 % |
| R1 jacket sizing + SP-8087 loads | **done** — hoop-sized tapered jacket, thin-shell and nozzle-collapse constraints added |
| R2 flange/faceplate sizing | **advisory shipped** — 18.13 → 5.53 kg available; not yet applied to exported CAD |
| R3 requirements layer | not started — still the product gap |
| R4 parts package | not started |
| R5 physics gaps | not started |

Details of R0–R2 outcomes are in `HARDWARE_MASS_LEDGER.md`.

---

## R0 — Reconcile the two chamber-length conventions ✅ DONE

The mass ledger work exposed this: the MDO's analytic grid and the traditional
`L*`-derived contour describe **different chambers** for the same design.

| | MDO station grid | traditional contour |
|---|---|---|
| chamber start | −229.3 mm | −209.2 mm |
| meridional arc | 428.0 mm | 401.3 mm |
| wetted area | 1877.4 cm² | 1657.4 cm² |

`MissionSpec.chamber_length` is a prescribed cylindrical length;
`DesignInput` derives the chamber from `L_star`. An 11.7 % wetted-area
difference is an 11.7 % difference in **total heat load**, not just in mass, so
this contaminates every cooling comparison between the two paths — it has simply
been invisible until there was a mass number to notice it with.

**Done.** `L*` and contraction ratio are the single shared convention. The MDO
grid now builds barrel → shoulder fillet → straight convergent → upstream throat
arc, identically to `chamber_geometry.chamber_contour`, and solves the barrel
from the volume closure in closed form. `MissionSpec.chamber_length` is gone.
Basis: SP-125 printed p. 88 defines the chamber volume as injector face to
throat plane, so `L*/CR` (barrel-only) necessarily over-runs the barrel.

Outcome: wetted-area ratio 0.8828 → **1.00102**, thrust-chamber mass delta
12.06 % → **0.044 %**, both paths hitting `L*·A_t` to 1e-9. A parity test pins
barrel length and chamber volume to 1e-6 and wetted area to 1 %.
`chamber_volume_margin` was added as a reported constraint for the case where
the fixed sections alone exceed `L*·A_t`.

---

## R1 — Size the structural closeout instead of assuming a thickness ratio ✅ DONE

Currently `t_closeout = 2 × t_wall`, an assumption carried identically into both
paths. Against SP-125 p. 109 (*"the outer shell is subjected only to the hoop
stress induced by the coolant pressure"*, `t = p_co·r_o/σ`) at SP-8087 §2.1.3
factors of safety (yield 1.0–1.32, ultimate 1.3–1.8), the 13 kN baseline needs
**0.6–0.8 mm** of Inconel 718 but **5.0–6.8 mm** of NARloy-Z. The default
single-alloy wall is therefore under-thick by 3–4×, and its closeout mass is a
lower bound.

**Done.**

1. The jacket is sized per station from the solved coolant pressure,
   `t_j = FoS·p_co·r_o/σ_y`, floored at a manufacturing minimum and *tapered* —
   SP-8087 §2.1.3.1: *"The brazed jacket can be tapered for optimum strength and
   weight."* Baseline: 0.500–0.591 mm of Inconel 718 weighing 0.714 kg, against
   2.53 kg for the old copper 2×-`t_wall` assumption. Sizing it against the
   solved jacket pressure means the structure and hydraulics cannot disagree
   about the load.
2. `jacket_thin_shell_margin` enforces SP-125 p. 336's `t/r ≤ 1/15` validity
   limit on the hoop formula, and `nozzle_collapse_margin` implements SP-8087's
   third structural job from NASA SP-8007 §4.2.3 (eqs. 16–21, γ = 0.75/0.90) —
   see `raosim/mdo/structures.py`. It is deliberately separate from
   `separation_margin`: separation asks whether the flow detaches, collapse asks
   whether the shell survives the external pressure while attached.
3. Liner and jacket are separate materials by default (copper-alloy liner,
   Inconel 718 jacket), in both pipelines and in the mass ledger.

**Still open from R1:** SP-8087's *second* job — throat bending and buckling
support — is not screened. SP-125 eq. 4-29 exists in `physics.py` but is not
wired into the MDO constraint set, and the current collapse screen credits only
the jacket, not the land-stiffened liner/jacket sandwich (SP-8007 §4.4), so it
is conservative by an unquantified margin.

---

## R2 — Optimize the interface hardware that currently dominates mass ⚠️ ADVISORY SHIPPED

The ledger shows the auto-sized flange (7.34 kg) and faceplate (10.99 kg) are
**75 % of modelled engine hardware mass** at the 13 kN baseline — larger than the
entire thrust chamber. They are layout defaults from bolt-circle and
edge-distance rules plus a manifold-depth floor, not structural optima. The
flange grows to a 285.5 mm OD around a 177 mm chamber.

**Advisory shipped.** `raosim.interface.size_bolted_interface` selects the
lightest admissible joint from a real ISO 262 coarse-thread series, sizing bolt
count against the separation load with the ISO 724 stress area and ISO 898-1
proof stress. Every candidate is still resolved through the existing
edge-distance, pitch and plate-bending rules.

Root cause found: the bolt hole defaults to `0.06 × chamber diameter`, which
drives the bolt circle, the flange OD *and* the faceplate thickness (`2 × hole`).
The structural faceplate requirement is 5.07 mm against 21.3 mm delivered.
Shrinking the fastener fixes both at once.

| | flange OD | faceplate t | joint mass |
|---|---|---|---|
| layout default | 285.5 mm | 21.3 mm | 18.13 kg |
| M5×0.8 × 14 (default floor) | 234.0 mm | 11.00 mm | **5.53 kg** |
| M3×0.5 × 36 (unbounded) | 212.0 mm | 6.60 mm | 2.48 kg |

**Not yet applied to exported CAD** — it is reported as
`hardware_mass.joint_sizing_opportunity`, because adopting it changes STEP
output and that deserves an explicit decision. Wiring it through `InterfaceSpec`
is the first carry-over into R4.

Also fixed unconditionally: the flange, bolts and injector body are priced in
the jacket/structure alloy rather than the copper liner (SP-8087 §2.1.3.1).

**Still open:** gasket/seal compression and preload scatter, which
`interface.py`'s own docstring flags as missing; and the `2 × hole` faceplate
floor, which governs over the bending screen and is probably over-conservative
for a through-bolted joint but needs a sourced basis before being weakened.

---

## R3 — Build the requirements layer

This is the feature the goal statement describes and the repository does not
have. Proposed shape:

```python
EngineRequirement(
    thrust=5_000.0, thrust_condition="sea_level",   # or vacuum, or a trajectory
    isp_min=250.0,
    dry_mass_max=30.0,
    envelope_diameter_max=0.30, envelope_length_max=0.60,
    propellants=("LOX", "RP-1"),
    burn_time=30.0,
    throttle_range=(0.6, 1.0),
    reusable_cycles=20,
)
```

Three layers, because the problem is genuinely mixed-discrete:

1. **Architecture enumeration (outer, discrete).** Injector type, cooling
   scheme, liner and jacket material family, channel count, pump arrangement,
   motor topology, battery chemistry, number of pump stages. These are already
   correctly treated as static/discrete throughout the MDO — they must not be
   forced into a continuous gradient formulation. Enumerate and screen.
2. **Continuous MDO (inner).** The existing differentiable solve, extended with
   `L*`, contraction ratio, closeout thickness, and **O/F** as variables (gap
   12.8 — needs the CEA property surfaces, which already exist behind
   `MissionSpec.cea_table_path`, plus combustion-stability and property-domain
   constraints).
3. **Requirement mapping.** Thrust becomes an equality constraint rather than a
   post-hoc scaling; Isp, mass, and envelope become inequalities; the objective
   becomes user-selected (min mass, max Isp, min cost proxy) with the others as
   constraints. The `dry_mass_partial` output added in this change is the first
   piece of a real mass constraint.

**Prerequisite:** R0, R1 and the O/F variable. **Also needs:** multi-start, since
a mixed-discrete outer loop over a non-convex inner problem will find local
optima; the plan already calls for this.

---

## R4 — Complete the parts package

What "get parameterized parts" currently misses:

| part | status |
|---|---|
| nozzle + chamber wall | ✅ STEP/STL, revolved from the wall profile |
| regen channels | ✅ |
| chamber flange | ⚠️ dimensions resolved, geometry not exported as a part |
| bolts | ⚠️ sized and counted, no fastener callout or hole pattern in CAD |
| injector faceplate / post / sleeve | ✅ machined STEP |
| pump impeller / inducer / volute | ⚠️ STEP exists; watertightness and manufacturability gaps recorded |
| manifolds, inlet bosses, igniter port | ⚠️ layout-resolved, partially exported |
| valves, lines, brackets, gimbal | ❌ not modelled at all |

**Do:**

1. Export the flange as a real solid with the bolt-hole pattern, and emit a
   fastener callout (thread designation, grade, length, torque) rather than only
   a diameter. This is what turns "sized bolts" into orderable bolts.
2. Close the pump CAD STEP/watertight gaps already on record.
3. Add an assembly-level interference and datum audit across all parts —
   `engine_cad.audit_engine_component_interference` exists, so extend it to the
   full package rather than pairs.
4. Emit a single machine-readable **build package**: STEP per part, a BOM with
   masses (now available), materials, tolerances, and the provenance/gate report
   so a shop can see what is screened vs qualified.

---

## R5 — Finish the physics gaps that block honest comparison

Ordered by how much they currently distort results.

**R5a — Film-cooled thermal parity (gaps 12.1, 12.2).** The MDO applies a
liquid-film heat-load model; the traditional solver carries the film mass ledger
but no film heat-load model, so every film-sensitive thermal quantity is marked
unavailable for nonzero film. Since film cooling is the primary coking lever at
high `Pc`, this means the two paths cannot be compared on exactly the designs
where it matters most. Also needs a separate film-injector hardware branch —
both paths still screen film through the main pintle surrogate.

**R5b — Held-out CEA and CoolProp validation (gap 12.6).** The property surfaces
are currently self-validated. Needs held-out cases with documented error bounds
before any performance number is quoted as predictive.

**R5c — Epigraph variables for the remaining hard extrema (gap 12.11).**
Stationwise `min`/`max` collapses and the battery `jnp.maximum` produce
subgradients, not smooth switching. Replace with epigraph variables. The new mass
term is deliberately free of these — no `where` on a design-dependent predicate,
no station extremum — so it adds no new active-set switching.

**R5d — Multipoint mission (gap 12.9).** Sea level, altitude, and throttled
operation as simultaneous design points. The nozzle-separation constraint and
the SP-8087 nozzle-collapse load in R1 are both fundamentally multipoint
problems being solved at one point today.

**R5e — The fair solver benchmark (gap 12.12).** Simultaneous differentiable MDO
vs finite-difference NLP vs block-coordinate, on identical physics and identical
output definitions, measuring function evaluations, wall-clock, JIT compile time,
KKT residual, multistart robustness, Pareto quality, and degradation after
authoritative re-evaluation. Until this exists, no claim that the differentiable
approach is better is supported. Note this now costs more to run than before: the
chamber mass term makes the objective genuinely multidisciplinary, which is the
regime where a simultaneous formulation should win — so this benchmark is now
more informative than it would have been.

---

## R6 — Longer horizon

- **Exact implicit-JAX Rao solver in the loop.** The staged plan in
  `IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md` stands; do not advance it past
  post-optimum validation until R0–R3 are done, because contour fidelity is not
  the binding error today — chamber-length convention and unsized structure are.
- **Cost and manufacturability objectives.** Machining time, material cost,
  additive build volume. The BOM this change produced is the natural hook.
- **Validated manufacturing CAD from the numerical contour** (gap 12.7). Keep the
  current restriction until contour smoothness, topology and manufacturability
  are qualified.
- **Regenerate the PDF plan from the Markdown** (gap 12.13). The PDF is a
  2026-07-22 snapshot; the implementation moved on 2026-07-26 and again here.
  One source of truth.

---

## Suggested order

```
R0  chamber-length convention          DONE
R1  size the closeout + SP-8087 loads  DONE
R2  optimize flange/faceplate          ADVISORY SHIPPED -- apply to CAD next
R5b held-out property validation       ← can run in parallel
R3  requirements layer + O/F variable  ← the actual product feature, now unblocked
R4  complete parts package             ← what the user receives
R5a film parity + film injector
R5c epigraph variables
R5e fair solver benchmark
R5d multipoint
R6  exact Rao, cost, manufacturing CAD
```

R0 through R2 were the physics and structure work that make the numbers
trustworthy, and they are now largely done: the two pipelines describe one
chamber to 0.04 %, the jacket is sized against a real load instead of a
thickness ratio, and the joint that dominated engine mass has a sized
alternative on the table.

**R3 is now the critical path.** It was previously blocked because a
requirements layer built on a 12 % geometry inconsistency and a 3–4×
under-thick jacket would have emitted parts that were wrong in hard-to-see
ways. That objection no longer applies.

Two carry-overs from R2 belong at the front of R3/R4:

1. **Apply the sized joint to exported geometry.** The advisory saves 12.6 kg on
   the baseline but does not yet drive `InterfaceSpec`, so the STEP files still
   carry the layout-default flange. Wiring it is small and is what turns the
   finding into parts.
2. **Revisit the `2 × hole` faceplate floor.** It governs at 11.0 mm against a
   5.07 mm bending requirement. For a through-bolted joint (bolt + nut) the
   plate needs bearing area, not thread engagement, so the rule is likely
   over-conservative — but it needs its own sourced basis before being
   weakened.
