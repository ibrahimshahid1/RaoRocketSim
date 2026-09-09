---
title: The LREKit hardware program
subtitle: From reference geometry to machinable, testable hardware, and back into the models
version: "0.1"
audience: [ibrahim, AI coding agents]
created: 2026-09-05
companions:
  - docs/PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md      # the grounding for tracks H, I, J
  - docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md              # H1-H14, what must be acquired
  - docs/COMPLETION_PLAN.md                                # tracks A-G, the numerical backlog
  - docs/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md   # P0-P11, becomes track H's backlog
  - docs/audit_work/LREKIT_WORKFLOW_AUDIT_2026-09-04.md    # W01-W06, S0-S7
  - docs/INTEGRATION_PROGRAM_2026-08-22.md                 # §2.1 specifies the configuration object
  - docs/PROJECT_AUDIT_2026-08-25.md                       # §9.7 and §9.9 are the gaps closed here
---

# The LREKit hardware program

> **Implementation review, September 6:** use the [consolidated readiness review](LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md)
> and its D-series backlog as the delivery index. Section 3 of that review corrects
> this draft's impeller P1 status, TE stacking, quantization requirement, machining
> and trimming claims, and envelope interpretation. The H–M material below is
> retained as a workstream draft; those corrected claims must not become code or
> article-release criteria.

> **What this document is.** The plan that takes the repository from *a tool that
> emits reference geometry* to *a toolkit that emits hardware you can cut, measure,
> test, and feed back*. Six tracks, H through M, sequenced, with acceptance gates.
>
> **What it deliberately does not do.** Restate tracks A–G of
> `COMPLETION_PLAN.md`, the P0–P11 backlog of the pump geometry plan, or the
> W01–W06 / S0–S7 findings of the 2026-09-04 workflow audit. It points at them,
> reorders them where the hardware goal changes their priority, and adds what none
> of them covers.

---

## 0. Two definitions, and why the order between them is forced

The whole program reduces to making two words checkable.

**Machineable** = (a) a flow-path solid that represents what the meanline
actually solved, **plus** (b) a process model that says a shop can cut it, out of
a stated material, on a stated machine, to a stated tolerance and finish.

**Testable** = (c) a rotating assembly that is a mechanical design rather than a
hydraulic sketch, **plus** instrumentation provisions designed in before
fabrication, **plus** an identity a measurement can bind to.

The order is not a preference:

- **(a) before (b)** — a tool-reachability check run against a blade that is about
  to be replaced by a twisted one is wasted work, and worse, it would certify a
  geometry that will not exist.
- **fillets last** — OpenCascade fillet operations are fragile on geometry still in
  motion. This is P10's existing position and it stays there.
- **(c) is mostly not a CAD problem.** Bearings, seals, rotordynamics and thrust
  balance are design disciplines whose *criteria* the corpus states and whose
  *data* it does not (see the basis document §11). Track J is therefore
  literature-gated in a way H and I are not.
- **but the configuration object comes before all of it.** Every one of (a), (b),
  (c) produces solved quantities that other parts consume. Building three tracks
  that each invent their own way to pass state is how the repository gets a fourth
  identity scheme. Track K's first item is a prerequisite, not a parallel effort.

---

## 1. Correcting two premises before planning against them

Both corrections came from reading the tree and the corpus this session. Both make
the program smaller than it looked.

### 1.1 The data loop is one missing thing, not four

The claim was that closing the experiment-to-model loop needs four things that do
not exist: a configuration content hash, a measurement schema, a comparison layer,
and a calibration layer. Verified state:

| Piece | Actual state |
|---|---|
| **Configuration content hash** | **Three implementations exist**, with incompatible canonicalisation. `movable_geometry_fingerprint` (`raosim/movable_pintle.py:414`) and `coaxial_geometry_id` (`raosim/injector_coaxial_cad.py:583`) are both schema-versioned, strict (`sort_keys`, `separators=(",",":")`, `allow_nan=False`), and the latter is already load-bearing — `raosim/design.py:2041` and `:2645` **refuse** when the mass ledger's id and the exported STEP's id disagree. `ResolvedEngineInputs.digest()` (`raosim/mdo/resolved_inputs.py:288`) is the weak one: bare `asdict`, no schema field, `default=str`, no `allow_nan=False`, and **no declared quantization**. |
| **Measurement schema** | **Genuinely absent.** The only truly greenfield piece. |
| **Comparison layer** | **Exists for model-vs-model**: `raosim/mdo/snapshot.py:3414`, `:3477`, `:3554` — signed and absolute scalar deltas, profile comparisons, explicit unavailable states. Missing: uncertainty propagation and station matching. |
| **Calibration layer** | **A complete provenance contract already exists for one coefficient.** `raosim/injector.py:1160-1215` requires, all-or-nothing in both directions, that a supplied `cd_vs_opening_fraction` table carry `cd_calibration_source`, `cd_calibration_artifact_sha256` (validated 64-hex), **`cd_geometry_fingerprint_sha256`** — the geometry binding, already implemented — and Re / Δp / T / cavitation-number ranges each validated `0 ≤ min < max`, plus the fluid name. |

So `ResolvedEngineConfiguration` is a **Merkle composition of per-component
fingerprints**, not a new hash function; and `CalibratedCoefficient` is a
**generalisation of a working, tested contract**, not an invention. The
`ResolvedEngineInputs.digest()` weakness is the one thing that must be fixed before
composition, and the fatal part is the missing quantization: two runs that produce
the same design to any meaningful engineering tolerance can produce different
digests, which is disqualifying for binding a measurement.

### 1.2 The feasible envelope is blocked on prose, not on physics

`ModelProvenance` (`raosim/model_registry.py:17`) already carries a `validity`
field on all 61 audited entries. That *is* the envelope datum. But it is prose —
`"single-phase, incompressible liquid; calibrated Cd required"`, `"no demonstrated
performance-validity range"` — and no machine can evaluate it.

The blocker for turning the 500 N crash into a computed refusal is therefore
**61 prose strings**, not missing physics and not missing numbers. That reframes
Track L from a research task into a mechanical, incremental, citation-checked one.

---

## 2. The six tracks

```
K0  configuration object  ──┬──> H  3-D flow path ──> I  process + inspection
                            │                          ↑
                            ├──> J  rotating assembly ─┘   (J is literature-gated)
                            │
                            ├──> K1-K4  measurement, comparison, calibration
                            │
                            └──> L  feasibility envelope ──> M  one entry point
```

`COMPLETION_PLAN.md` tracks A–G continue unchanged alongside. The one edge into
them: **G5 (L4 datum/feature-control scaffolding) is a dependent of Track I**, not
an independent item, because the tolerance-stack input contract needs the process
model to know what tolerances are achievable.

---

## 3. Track H — the 3-D flow path

**Owns**: P2–P11 of `PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md`.
**Grounding**: `PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md` §3, §4.
**Blocking defect**: `impeller_blade_camber` (`raosim/pumps.py:1959`) integrates
one camber line with β linear in radius; `_camber_ribbon_xy`
(`raosim/pump_cad_brep.py:106`) extrudes it as a planar ribbon. Hub and shroud get
identical sections. The meanline already exports hub-side and tip β₁ — it knows the
blade twists and the solid cannot say so.

### H1 — `camber_3d`: per-streamline integration with a declared stacking rule

The construction, per the plan's own C1 correction:

```
dθ_s/du = (dm_s/du) / (r_s(u) · tan β_s(u)),    s ∈ {hub, mid, shroud}
```

integrated on the **common paired parameter u** that `_meridional_channel`
(`raosim/pumps.py:2004`) already emits for hub and shroud. Stack at the **trailing
edge**, θ_hub(1) = θ_shroud(1) = 0, so the exit edge is axial and matches the common
β₂ exit; record `stacking: trailing_edge` in the manifest. Lean is then a
deliberate θ-offset distributed over **local span height**, never global z.

The β distribution along each streamline is the design freedom the current model
gives away by fixing "linear in r". SP-8109 does not prescribe a distribution; it
prescribes what the distribution must achieve (§2.3.1: "The impeller blade angle
distribution and the blade number are adjusted until desired design limits are
achieved", with velocity gradients evaluated for eddies and separation). So the
implementation should carry β(u) as a small parameter set — endpoints fixed by the
meanline, interior shape controlled and defaulted to the current linear case — so
that H1 is bit-compatible with today's output at its default and opens the freedom
without spending it.

**Acceptance.** The trailing-edge clocking difference between hub and shroud,
computed from the solid, equals the analytic integral of the two dθ/du to < 0.1°.
At default β-shape parameters, the mid-span section reproduces the current 2-D
camber to within CAD tolerance. The exported hub β₁ and tip β₁ are both readable
off the built solid.

### H2 — the solid: planar transverse sections, not a ribbon loft

The plan's C2 finding stands and is not to be re-litigated: lofting between two
closed nonplanar hub/shroud ribbons produced **invalid solids** in CadQuery 2.7 on
the saved 13 kN curves, both ruled and smooth. Build instead planar transverse
sections spanning hub→shroud at each u — points from both streamlines plus
thickness — and loft the ordered planar wires; or sew pressure and suction faces
over the (u × span) grids and cap.

**Acceptance.** Every blade re-imports from STEP as exactly one valid solid with a
positive measured volume, at the existing gate. The full impeller assembles with no
self-intersection. Run at both the 13 kN baseline and at least one small-thrust and
one large-thrust case, because a construction that only closes at one scale is not
a construction.

### H3 — ruled vs free-form, decided explicitly

This decision gates I2 and must be made in H, not discovered in I. A blade whose
hub and shroud directrices admit straight-line generators is **flank-millable**;
one that does not must be **point-milled**, at a large cost in time and finish.
SP-8109 does not use the words, but it constrains the same thing from the other
side: `Z₂ ≤ ≈ 28 sin β₂` and cutter L/D ≤ 10 (Al) / 8 (Ti) are statements about
what a cutter can reach between two blades.

Emit a **ruledness residual** — the maximum deviation between the built surface and
the ruled surface through the same directrices — as a reported number, not a
boolean. It becomes an input to I2 and a design lever: a designer who wants
flank-milling can constrain the β distributions until the residual falls below the
tool's tolerance.

### H4 — leading and trailing edges

Elliptical LE profile with the ratio as a parameter; TE thickness and cut-back;
LE sweep in the meridional view. SP-8052's J-2/F-1 0.005 in. low-end LE practice is
already used and should be re-verified against the PDF at the same time.

**Carry the trim.** SP-8109 §2.3.1.2 documents that under-filing or over-filing the
TE moves head and horsepower by up to **±10 % with little or no efficiency change**.
Model the TE profile with an explicit `trim` degree of freedom whose nominal is
zero and whose declared authority is ±10 % on ψ. This is not decoration: it is the
adjustment the test loop in Track K will actually command, and it must exist in the
geometry before a measurement can ask for it.

### H5 — the collector, replacing two placeholders

**Diffuser** (`build_diffuser_ring` currently emits rectangular boxes): vane inlet
angle β₃ matched to the entering flow angle at design — or at 0.8 φ if a throttle
range is declared; vane width `b₃ = (0.9–1.0) · b_t2` with rounded or faired side
walls; throat sized so the diffusion factor `D ≤ 0.6`; throat→discharge cone
**7°–10°**. Vane-island variant when the diffuser carries casing structural load.

**Vane count**: implement the SP-8109 §2.4.1.2.3 wave-reinforcement criterion over
(Z₂, Z_d, πD_v, a, W) — the reinforcement index m must not be integral for the
harmonics of interest — rather than the folk rule about common factors. The two
equations are OCR-scrambled and **must be rendered off the PDF before coding**.

**Volute**: implement **constant moment of momentum** (r·c_u = const) as the
default, with constant-mean-velocity retained as a named, selectable
simplification. Growing centerline radius. Asymmetric cross-section. Tongue at zero
incidence at design flow. Exit conical diffuser at 7°–9° circular / 6° square /
11° two-parallel-wall, with the terminal area coming from the cone, **not** set
equal to the outlet port area.

**Why the volute is not cosmetic.** SP-8109 §2.4.1.3.1 records that the efficiency
difference between the two laws is minor, but that constant-mean-velocity produces
**higher radial hydraulic force on the impeller**. That force sizes the radial
bearing, the shaft, and the critical speed. Track J cannot size a bearing against a
load that is an artifact of a simplification, so H5 is a **prerequisite for J**, not
a parallel nicety.

### H6 — shroud, splitters, fillets

Shrouded and open-face variants as an explicit configuration (it selects the
tip-speed ceiling and the process route in I1). Splitters already exist in the
meanline bookkeeping and the CAD; extend them to the 3-D camber. Fillets **last**.

**Track H gate.** No pump geometry is described as anything beyond reference CAD
until: the built solid reproduces both exported β₁ values; the ruledness residual is
reported; the collector is built from the SP-8109 constructions above; and every
part re-imports as one valid solid at three thrust scales.

---

## 4. Track I — the process and inspection model

**Owns**: generalising `raosim/manufacturing_limits.py` from one source covering
one part into a process layer covering all part families.
**Grounding**: `PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md` §1, §2.

Today: 203 lines, milled regen channels only, from Wadel 1997 (LH2, 89 kN class,
with its own applicability string admitting other coolants and thrust classes are
extrapolation), consumed in exactly one place, `raosim/mdo/job.py:808`.

**The finding that reshapes this track: manufacturing is the *least* literature-
blocked of the three hardware tracks.** SP-8109 §2.3.3 and §3.3.3 constitute a
quantitative, part-specific process model for the impeller, already on disk, and
verified against the scan this session.

### I1 — the process registry, keyed on quantities the solver already computes

A `ProcessCapability` record in the shape of `ModelProvenance` — because the
repository already has a working registry idiom and should not grow a second one:

```
process_id, family, applicable_materials, applicable_configurations,
selection_rule, achievable_tolerance, achievable_finish, min_feature,
min_internal_radius, max_aspect, orientation_constraints,
required_qualification, source, validity, validation_level
```

Populate the impeller row from SP-8109 first, because it is the one that is fully
sourced:

| Selector | Value | Source |
|---|---|---|
| U₂ < 1000 fps, shrouded, dense fuel/oxidiser | cast | §2.3.3 **[verified]** |
| U₂ < 1400 fps | casting preferred | §3.3.3 |
| U₂ > 1400 fps, open face | machined from forging | §3.3.3 |
| U₂ ≤ 2200 fps, shrouded | machined, or diffusion-bonded if Ti | §3.3.3 |
| finish, cast Al ceramic-shell, D ≤ 10 in | 63 µin/in | §2.3.3 **[verified]** |
| finish, cast Al ceramic-shell, D 10–20 in | 125 µin/in | §2.3.3 **[verified]** |
| finish, SLM as-built | ≈ powder diameter; 200–500 µin observed on GRCop-84 | Gradl 2017 |
| overhang, SLM | 45° rule | Gradl 2017 |

**Note the shape**: `achievable_finish` is a *function of the part envelope*, not a
constant. The record must permit that.

**Replace, do not supplement, the tip-speed scalar.**
`PumpSizingSpec.material_tip_speed_limit = 350.0` (`raosim/pumps.py:150`) becomes a
lookup on (material, process, shroud configuration, fluid). The same key selects the
process route, which is why this one change is the spine of the track.

### I2 — geometry interrogation

The checks that need to read the B-rep rather than the parameters:

- **minimum internal radius** vs candidate tool radius;
- **thin-wall regions** below the process minimum;
- **undercut / inaccessible faces** relative to a candidate tool axis;
- **setup count**, from the number of distinct tool axes required;
- **cutter reachability between blades** — implement SP-8109's rule directly:
  `Z₂ ≤ ≈ 28 sin β₂` for shrouded machined impellers, with the underlying
  `cutter L/D ≤ 10 (aluminium) / 8 (titanium)` as the general check, and the
  validity window `D₂/b₂ < 20`, stage Ns > 1000;
- **AM powder-removal accessibility** — every internal void must have a traced path
  to an opening. Gradl 2017 is unambiguous that this is a design constraint: a
  chamber had to be cut open after CT scanning revealed trapped powder.

**Ship the blade-count ceiling first.** It is one function, one gate, one test,
sourced from a monograph the repository already cites for the *minimum* on the same
page range, and today a design can be emitted with a blade count no shop can cut.

### I3 — process outputs that are physics inputs

The point at which a process model stops being a checklist. Two required feedbacks:

1. **Surface finish → hydraulics and heat transfer.** An SLM internal passage is
   200–500 µin as-built and, per Gradl 2017, effectively cannot be finished. The
   repository's regen Δp (Darcy/Blasius) and Sieder–Tate heat transfer assume smooth
   walls. The process must hand a roughness to those models, and the roughness must
   appear in the provenance of the resulting Δp and wall temperature.
2. **Process → allowable stress.** Cast, wrought, and L-PBF properties differ, and
   SP-8109 records that shot peening raises the endurance limit by 40 %. The
   structural screen must take its allowable from the *process-qualified* material
   state, not the generic catalogue entry.

### I4 — required inspection, generated from the design

Every process decision emits its qualification requirement rather than leaving it
to a human:

- **Spin test** — required when U₂ > 1000 fps, at ≥ 1.2 × maximum anticipated
  operating speed, with an explicit temperature correction if run at a different
  temperature (§3.3.3, verified).
- **Residual imbalance** — `oz-in ≈ 4 · W[lb] / N[rpm]` (§3.3.3, verified;
  SI equivalent `U[kg·mm] ≈ 6.35 · m[kg] / N[rpm]` is my conversion).
- **Balance-correction provisions** — location, amount and procedure specified, and
  the correction feature present *in the geometry*.
- **CT scan** for any AM part with internal passages, on the Gradl precedent.
- **Cold-vs-operating fit statement** for every running clearance on a cryogenic
  pump.

### I5 — the fidelity ladder, made truthful

`raosim/mdo/build.py` labels fidelity L0–L5 and the workflow audit's W04 reproduced
the defect: a text file named `fake.step` containing `this is not a STEP file` is
recorded as `built`, `L2_VALID_REFERENCE_CAD`. Track I supplies what a real
assertion needs — a parsed solid, a measured volume, a declared unit, a material, a
process route, and the achieved tolerance and finish. `build.py`'s ceiling may move
from L2 toward L4 only when I1–I4 exist and G5's datum scaffolding is in place;
**L5 remains unreachable by construction** and that does not change.

**Track I gate.** No part is labelled machinable without: a selected process with
its selection rule and source; every geometric interrogation passed or explicitly
`unavailable` with a reason; an achievable tolerance and finish attached; the
resulting roughness fed back into any model that depends on it; and the generated
inspection requirements listed.

---

## 5. Track J — the rotating assembly

**Owns**: the items `pump_cad_brep.py:1545 external_release_blockers` already lists
honestly — bearings, seals, wear rings, thrust balance, torque coupling,
lubrication, shaft retention, gasket/preload/dowel selection, tolerance stack,
thermal growth, rotordynamics, burst/FEA, drawings and GD&T.

**This is the literature-gated track.** SP-8109 states criteria and defers data:
bearing speed limits to its ref. 43, critical-speed method to ref. 28, shaft seals
to ref. 44, and the non-pumping-surface pressure distribution — the axial-thrust
integrand — to refs. 79 / 53 / 54 / 80. None is held locally. `LITERATURE_REQUESTS.md`
items 13 (SP-8048 bearings), 14 (SP-8101 shafts and couplings) and 15 (SP-8121
rotating-shaft seals) are exactly right and remain outstanding.

### What can proceed now, without the missing monographs

- **J1 — Axial-force accounting structure.** Build the force ledger — momentum
  term, front and rear shroud pressure fields, wear-ring diameters, balance-hole
  area — with the shroud pressure-field term explicitly `unavailable` pending the
  reference. Today wear rings start at "equal-diameter neutral"; SP-8109 §2.5.2.1
  makes their **relative diameters the sizing variable**, so the structure is
  buildable even where one integrand is not.
- **J2 — Rotordynamic gate as a reported margin.** The criterion is known and
  quotable: **≈ 20 % margin** between operating speed and the nearest calculated
  whirl critical, and **no sustained mainstage operation between 80 % and 120 %** of
  a bearing-spring-rate critical. Implement the *gate* now, reporting `unavailable`
  because bearing stiffness is unknown — which is far more useful than silence,
  because it names precisely what datum is missing at a speed the architecture
  actually runs at.
- **J3 — The electric-pump-specific statement.** An electric drive removes the
  turbine axial force that SP-8109 assumes the pump balances. Say so explicitly
  rather than inheriting the assumption. Then state the consequence that does *not*
  go away: at 30–60 krpm, SP-8109 §2.5.2.3 says ball bearings cannot sustain normal
  operating unbalanced axial forces, and balance pistons or hydrostatic bearings are
  used instead. That is a *design-space* statement about this architecture, and it
  is derivable today.
- **J4 — Blade and disc structure.** Blade thickness from SP-8109 eq. (22), cyclic
  load at 30 % of steady hydraulic load, fatigue margin on a modified Goodman
  diagram (fig. 23). Carry the monograph's own honesty about the ceiling: stress
  concentrations "are difficult to analyze, and spin tests in air using stress coat
  and strain gages are required to determine the magnitude of these local stresses."
- **J5 — Thermal growth and fits.** Cold-assembly vs operating clearance for every
  running fit, which is both a mechanical result and a drawing dimension.
- **J6 — Torque path.** Spline / curvic / pin / bolt selection with contact force
  sized to prevent fretting — SP-8109 notes fretting "has caused explosions" — plus
  the wrench-flat and torque-access provisions for breakaway and drag-torque
  measurement.

### What genuinely waits

Bearing selection and life; seal package design and leakage; the critical-speed
calculation itself; the axial-thrust pressure integrand. These stay `unavailable`
with a named missing reference. **Do not substitute a plausible number.**

**Track J gate.** No pump is described as testable hardware until every
`external_release_blockers` entry is either closed with a sourced method or open
with a named missing reference. The list shrinks by evidence, never by assertion.

---

## 6. Track K — identity, measurement, comparison, calibration

**Owns**: `PROJECT_AUDIT_2026-08-25.md` §9.7 and §9.9;
`INTEGRATION_PROGRAM_2026-08-22.md` §2.1; workflow-audit stages S2–S5.

### K0 — the configuration object *(prerequisite for everything else)*

1. **One canonicalisation policy.** Adopt the strict idiom that
   `movable_geometry_fingerprint` already uses — explicit named fields, a schema
   identifier, `sort_keys=True`, `separators=(",",":")`, `allow_nan=False` — and add
   the thing none of the three has: a **declared per-quantity quantization**. Without
   it, two runs producing the same design to any meaningful tolerance yield different
   digests, and a measurement cannot bind reliably to either.
2. **Retrofit `ResolvedEngineInputs.digest()`** onto that policy: replace bare
   `asdict` with an explicit versioned payload, drop `default=str`.
3. **Compose, do not invent.** `ResolvedEngineConfiguration` is a Merkle root over
   per-component fingerprints — `coaxial_geometry_id`, `regen_geometry_id`, the new
   pump fingerprints — plus the resolved inputs digest, the model-registry version,
   the property-table hashes, the software commit and dirty-patch identity, and
   solver settings.
4. **Every artifact stamps the root; assembly refuses mismatches.** The refusal
   pattern already exists at `raosim/design.py:2041` and `:2645`; generalise it.

**Acceptance** (from the workflow audit's S2, unchanged): change any physical
dimension, model coefficient, or part transform and the identity changes; change
only the output directory and it does not; cross-configuration artifacts are
refused.

### K1 — the measurement record *(the one genuinely new schema)*

Every field below is justified by a specific experiment in the corpus that would
have been misread without it — see the basis document §9:

| Field | Why, with the evidence |
|---|---|
| value, units | — |
| uncertainty, with its type and coverage factor | Reis 2019 quotes no uncertainty anywhere; a residual without one cannot be adjudicated |
| **station**, with a datum | "measured at 50 mm upstream of the inducer leading edge" |
| **test fluid**, separate from design fluid | the L75 ethanol pump was tested **on water**; without this field the comparison is nonsense |
| instrument identity + calibration certificate | separate HBM modules were used specifically to remove electrical noise |
| acquisition chain and timebase | steady and dynamic channels answer different questions |
| conditions vector (speed, inlet pressure, flow, temperature, ambient, throttle, run number, thermal state) | inlet pressure was actively set by regulator and calibrated orifice — it is a controlled variable, not context |
| pressure convention (absolute vs gauge) | the classic silent error |
| **quality flags with a cause** | "at flow rate 0.6 the head fall can not be measured because of vacuum limitations" — unavailable-for-a-facility-reason, not a missing value |
| raw-data hash | — |
| article + configuration identity | binds to K0 |
| **acceptance convention** | SP-8109 fig. 4 uses **2 % head loss for pumps and 10 % for inducers**, not the commercial 3 % |

Keep this **strictly separate** from `EvidenceRecord.passed`
(`raosim/release_readiness.py:24`), which stays what it is: a reviewed pass/fail
record. A measurement is not a verdict.

### K2 — the comparison layer

Extend `mdo/snapshot.py`'s existing delta machinery rather than replacing it. Add:
operating-window matching; station matching; unit and convention normalisation;
**similarity transformation when the test fluid differs from the design fluid**;
the signed residual convention `measurement − prediction`; relative error only where
the prediction is bounded away from zero; and uncertainty propagated into the
residual.

Refuse, loudly: mismatched configuration, mismatched station, mismatched units,
missing uncertainty.

### K3 — the calibration layer

Generalise `injector.py:1160-1215` into a shared `CalibratedCoefficient` carrying
source, artifact hash, **geometry fingerprint**, validity envelope per independent
variable, fluid, fitted value, uncertainty, and identifiability diagnostics.
Promotion creates a **new coefficient version**; it never mutates a baseline.

**First target, and it is unusually cheap**: `pump_efficiency`
(`raosim/mdo/pump.py:50`). Its three coefficients are already fields on
`MissionSpec` (`raosim/mdo/schema.py:571-573`), it is already differentiable, and
its docstring already concedes it is "a smooth screening surrogate, not a fit to
SP-125 Fig. 6-23 (image-only)". Two paths, both available: digitise SP-8109 fig. 5,
which additionally carries δ = D_t1/D_t2 — a second parameter the surrogate lacks;
and fit to measured H–Q–η from a pump bench.

Hold the line on validity: never fit and validate on the same observations, and
never apply a calibrated coefficient outside its declared envelope.

### K4 — geometry pinning, and `editable_reference_id` finally consumed

`editable_reference_id` is emitted on every pump BOM row (`raosim/pumps.py:602`,
`:628`, `:2887`) and consumed by nothing. Wire it so a user can pin a component and
re-solve around it. Pinning a part is **not** pinning a filename: a pinned part needs
units, a coordinate frame, topology and feature identity, inferred physical
dimensions, provenance, and compatibility checks. A re-solve must preserve the pin
exactly and state which degrees of freedom remain free; an incompatible requested
performance must produce a reasoned infeasibility, never a silent overwrite.

### K5 — the trim loop, which is why any of this pays

SP-8109 documents two bounded post-fabrication adjustments: TE under/over-filing
worth **±10 % on head and horsepower with little or no efficiency change**
(§2.3.1.2), and balance-rib trimming to adjust axial thrust without changing wear-
ring diameters (§2.5.2.1). These are the historical precedent for the entire return
path — these pumps were adjusted against test results by design.

So the first end-to-end demonstration of Track K should be exactly this: measure
H–Q on a built impeller, compute the signed residual against prediction, and emit a
**TE trim recommendation with its authority bounded by the documented ±10 %** —
producing a linked as-built configuration rather than silently rewriting the
nominal design. It is a complete loop, it is small, and its authority is quoted
rather than invented.

---

## 7. Track L — generalisation and the feasible envelope

**Owns**: the 500 N failure, the scale-dependent assumptions, and partial
deliverables. Extends `GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md`.

### L1 — typed applicability on the model registry

Add an optional machine-checkable `applicability` predicate **alongside** the
existing prose `validity` (never replacing it) on `ModelProvenance`. Populate it
entry by entry as citation-checked work. The envelope for a requested design point
is the conjunction over the models that point actually invokes:

- typed predicate satisfied → **PASS**;
- typed predicate violated → **REFUSE**, naming the `model_id`, the bound, and the
  offending value;
- no typed predicate → **UNKNOWN**, reported, never silently passed.

This yields a completion metric that is honest and monotone: *n of 61 models have
machine-checkable validity*. Reuse `validation_level` (`assumption_only` /
`published_correlation_screening_use` / `software_verified`) as the **DEGRADE** axis,
orthogonal to REFUSE.

Seed it with the envelope statements already in the tree and in the corpus:
`WADEL_APPLICABILITY` (milled channels, LH2, 89 kN class); SP-8120's small-nozzle
exclusion (throat Re < 100, and "do not employ an inviscid-core solution corrected
by a boundary-layer perturbation" for small nozzles); SP-8109's Ns range, tip-speed
ceilings, and D₂/b₂ < 20 machinability window; Hansen's 700–1300 specific-speed test
range.

### L2 — refuse, do not crash; and still deliver the rest

The rule the repository already holds — a gap must be visible as `unknown`, never
absorbed — is violated in the *opposite* direction by a fatal crash. Required
behaviour: when the pump is infeasible at 500 N, the nozzle, chamber and injector
designs are still written; the failed CAD gate either cleans up its directory or
stamps it partial with per-part reasons; and NPSH margin is reported or registered
unavailable.

This is workflow-audit finding **W05**, and the pattern to copy already exists:
`raosim/mdo/postprocess.py:1090` catches pump-sizing failures into `pump_error`
while the rest of the result survives. Allocate a run identity and an initial
manifest **before** computation; update stage records atomically; export into a
staging directory and commit only when the output policy is met.

### L3 — the scale-assumption sweep

Enumerate the hard-coded dimensional constants, floors, ceilings and dataclass
defaults that only make sense near 10 kN, and either attach each to a source with an
envelope or mark it a repository heuristic. `material_tip_speed_limit = 350.0` is a
worked example: it is not wrong so much as unattached, and I1 replaces it with a
sourced lookup.

**Track L gate.** A requested design point outside the envelope produces a named
refusal, not a traceback; and every subsystem that *is* in-envelope still produces
its artifacts.

---

## 8. Track M — one entry point

**Owns**: workflow-audit findings W01–W06 and stage S6. Those findings are current
as of 2026-09-04 and are not restated here; what follows is the shape they imply.

Today there are two terminal front doors — the bare-`lrekit` three-way menu in
`run_nozzle.py:1699`/`:3511`, and the `lrekit guided|validate|run|catalog` verbs —
plus a 328-flag traditional runner, several parallel execution routes
(`--engine-mdo`, `--engine-mdo-optimize`, `--complete-package`, `--mdo-export`,
`--cad`), and a Python API. There is no maintained graphical frontend.

### M1 — the capability registry

The same idiom as `MODEL_REGISTRY` and the process registry, a third sibling:
architectures, cooling schemes, injector types, feed systems, output policies, CAD
formats and fidelities, processes — each with source, validity, an
implemented/unimplemented status, and its dependencies. This is what lets the tool
**refuse an unsupported architecture at parse time** instead of silently resolving it
to the default (W03), and it is what a UI enumerates instead of a hand-maintained
menu.

### M2 — one application service

One job document, one resolution path, one execution service, three clients:
terminal, Python API, and a future UI. Legacy invocations become adapters onto the
same contract, with compatibility tests. The verb family the workflow audit
proposed stands:

```
lrekit job new | validate | run | inspect
lrekit component inspect | pin
lrekit build create | verify
lrekit test ingest | inspect | compare
lrekit calibrate fit | validate | promote
lrekit catalog models | components | processes | coefficients
lrekit ui
```

Note that `test`, `calibrate`, and `component pin` are exactly Tracks K1–K4
surfacing, and `build` is Track I surfacing. The command family is not a UI
decision; it is the shape of the tracks made reachable.

### M3 — the UI, as five views over one project

Mission and pinned inputs · coverage and feasibility (Track L's verdict) · solved
components and assembly with per-part status and CAD preview (Tracks H, I, J) ·
fabrication, inspection and evidence package (Track I) · test, comparison and
calibration history (Track K). A user must be able to inspect a component that
succeeded even when another failed — which is L2's partial-deliverable behaviour
made visible.

A single large form containing hundreds of solver flags is not this, and building
it would make promises execution does not honour. **Fix the contract before
polishing the surface** — the workflow audit's central warning, and it stands.

---

## 9. Sequencing

| Stage | Work | Rationale |
|---|---|---|
| **0** | **K0** configuration object | Prerequisite. Everything downstream produces state that needs one home, and three hashing idioms already exist to reconcile. |
| **1** | **I2 blade-count ceiling** · **L1 typed applicability** (first tranche) · **L2 refuse-and-still-deliver** | The cheapest real defects. The blade-count gate is one function from an already-cited source. L2 is plumbing that converts a crash into a refusal. |
| **2** | **H1–H3** 3-D camber, valid solid, ruledness | The enabler. Nothing in I or J is worth doing against a blade about to be replaced. |
| **3** | **H5** collector | Prerequisite for J: the volute law sets the radial load that sizes the bearing. |
| **4** | **I1, I3, I4** process registry, physics feedback, inspection | Now the geometry is stable enough to check, and I3's roughness feedback changes results in the cooling models. |
| **5** | **J1–J6** mechanical, to the limit of held literature | Parallel with I once H5 lands. Blocked items stay `unavailable` with a named reference. |
| **6** | **K1–K3** measurement, comparison, calibration | Can start at stage 2 in parallel — the schema work does not need the geometry. **The measurement plan must influence instrumentation provisions before fabrication**, so it cannot be last. |
| **7** | **H4 trim** + **K5** trim loop | The first end-to-end demonstration, with authority bounded by SP-8109's documented ±10 %. |
| **8** | **H6** fillets · **I5** fidelity ladder · **G5** datum scaffolding | Fillets last by CAD fragility; the ladder last because it certifies the others. |
| **9** | **M1–M3** capability registry, one service, UI | Last by design. The contract must be true before the surface advertises it. |

Two edges deserve a word. **H5 → J** exists because a bearing sized against a
radial load that is an artifact of the constant-mean-velocity simplification is a
bearing sized against nothing. **K1 → fabrication** exists because instrumentation
provisions — SP-8109's wrench flats for breakaway and drag torque are the canonical
example — are geometry, and geometry is decided before metal is cut.

---

## 10. Definition of done

The program is complete when, for a user-supplied thrust class and high-level
performance parameters, the toolkit produces:

1. A **feasibility verdict** naming every model that refused, every model that is
   an extrapolation, and every model whose validity is unknown — and still delivers
   every subsystem that is in-envelope.
2. A **flow-path solid** that reproduces the quantities the meanline solved,
   including both exported blade angles, at three thrust scales.
3. A **process route** per part, with its selection rule, its source, its
   achievable tolerance and finish, and its consequences fed back into the physics
   that depends on them.
4. A **mechanical design** in which every entry of `external_release_blockers` is
   either closed with a sourced method or open with a named missing reference.
5. An **inspection package** generated from the design — spin test, imbalance
   limit, CT scan, fit statements — not written by hand.
6. One **configuration identity** stamped on every artifact, with assembly refusing
   mismatches.
7. A **measurement path**: ingest a measured H–Q curve against that identity,
   compute signed residuals with uncertainty, and emit a bounded trim
   recommendation that produces a linked as-built configuration.
8. **One entry point** through which all of the above is reachable, with every
   offered capability backed by a callable route.

Item 7 is the one that makes it a kit rather than a calculator. Items 3 and 4 are
what make items 1 and 2 mean anything. And none of the eight is a claim about
physical correctness — that is what tracks A–G and the measurement loop are for.
