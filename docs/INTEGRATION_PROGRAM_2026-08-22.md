---
title: Integration program — from design candidates to one fabricable article
subtitle: Mapping the eight-item gap list onto the existing tracks, with corpus evidence and literature requests
version: "0.1"
audience: [ibrahim, AI coding agents]
created: 2026-08-22
companions:
  - docs/COMPLETION_PLAN.md                     # tracks A–G, the current backlog
  - docs/PHYSICS_FIDELITY_ROADMAP.md            # fidelity ladder
  - docs/ROADMAP_REQUIREMENTS_TO_PARTS.md       # R0–R6, requirements→parts
  - docs/LITERATURE_REQUESTS.md                 # items 1–26, all still outstanding
  - docs/LITERATURE_REQUESTS_GENERALIZATION.md  # G1–G14, 13/14 acquired
---

# Integration program

> **What this document is.** The eight-item gap list, reconciled against what is
> actually in the tree and what is actually in `propulsion_texts/`. It says which
> items are already tracked, which are genuinely new, which are blocked on
> literature that is still missing, and what to build in what order.
>
> **Scope note.** This is a *reconciliation and sequencing* document. Where
> `COMPLETION_PLAN.md` already owns an item, this document points at it rather
> than restating it.

---

## 0. Answer to the standing question: is there a CLI option for BL correction?

**No.** `raosim/run_nozzle.py` exposes ~300 flags; none of them is a
boundary-layer, displacement, or viscous-correction switch. The nearest thing is
`--contour-method {bezier,rao-bvp}`, and both branches are inviscid.

The mechanism exists but is one-way:

| step | location | state |
|---|---|---|
| δ\* computed | `raosim/physics.py:32 boundary_layer_displacement` | flat-plate `0.046·s/Re_x^0.2` with a `√(T/T_w)` compressibility factor |
| δ\* consumed | `raosim/design.py:716` | called once, result reported |
| δ\* fed back to the wall | — | **does not happen** |
| throat Reynolds number | — | **not computed anywhere for the nozzle** |

SP-8120 (`propulsion_texts/19770009165.pdf`, corpus md line 1274) states the
procedure the repository is missing verbatim:

> "For accurate performance prediction, boundary-layer displacement thickness is
> computed. Under conditions where the boundary layer is unusually thick relative
> to the nozzle size, the boundary-layer displacement thickness is computed and
> the wall is moved outward by the displacement thickness point by point."

Two guardrails come from the same monograph and should ship *with* the loop, not
after it:

- **Validity floor.** Line 1635: "For throat Re values of less than 100, the
  method of boundary-layer correction to an inviscid core breaks down; i.e., the
  boundary-layer thickness at the throat exceeds the throat radius."
- **Regime exclusion.** §3.1.2.1.5.2 (line 4127): "Methods for flow analysis of
  small nozzles *shall* include viscous effects in the generation of the
  core… **Do not employ an inviscid-core solution corrected by a boundary-layer
  perturbation.**"

So the correct shape of the feature is a flag plus a gate, not a flag alone:

```
--boundary-layer-correction {off,displacement}   # default off
```

with `displacement` refusing to run — `unknown`, not a number — when the
computed throat Reynolds number puts the design in SP-8120's small-nozzle
regime. This is `COMPLETION_PLAN.md` **F1**, and F1's own acceptance rule
("report the correction magnitude, and if it exceeds a threshold the flat-plate
assumption is out of its domain and the result is `unknown`") already anticipates
it. What F1 does not yet carry is the *throat-Reynolds gate*, which is a
separate, cheaper, and independently shippable honesty gate. Add it as **F1a**.

---

## 1. Reconciliation: the eight items against the existing tracks

| # | Item | Already owned by | Genuinely new work |
|---|---|---|---|
| 1 | One solved configuration drives every artifact | G2, G3 (mass↔CAD); `mdo/build.py` part manifest; `release_readiness.configuration_id` | **Yes — the resolved-configuration object itself.** See §2.1 |
| 2 | Thermochemistry and combustion connection | B1 (done, 4 pairs), F2, F4, D3; R3's O/F variable | **Partly — predicted combustion efficiency.** See §2.2 |
| 3 | Nozzle and off-design modeling | F1, F2, F3, F5, F6, B3, B5, C1, C2 | No — fully covered, needs F1a gate |
| 4 | Thermal/structural/life on the built wall | D1, D2, D6, A4 | **Partly — transient load histories.** See §2.3 |
| 5 | Injector, film, ignition, stability | G1 (mass only), R5a (film parity) | **Largely — igniter and stability are unmodelled.** See §2.4 |
| 6 | Operating feed system | `PLAN_VALVES_LINES_BRACKETS_GIMBAL.md`, `PUMP_ADVANCED_GEOMETRY_ROADMAP` | No new *plan* — **hard-blocked on literature.** See §4 |
| 7 | Whole-engine mass, CAD, manufacturing | Track G entire; R2/R4 | No — covered, G5 blocked on standards |
| 8 | Experiment-to-model return path | **nothing** | **Yes, entirely.** See §2.5 |

Three items (1, 5, 8) are not represented anywhere in the four existing plans.
Those are the ones that need new design work rather than scheduling.

---

## 2. The genuinely new workstreams

### 2.1 Item 1 — the resolved configuration object

**What exists.** `raosim/mdo/build.py` already owns the right *contract*:
`ENGINE_PART_MANIFEST` enumerates expected parts, `BuildFidelity` (L0–L5) is
*achieved* rather than requested, and a part that was never modelled appears as
unavailable-with-reason rather than absent or zero. Its own docstring names the
failure being avoided: "Two competing build paths is how the repository ended up
able to report a 'complete' ledger for an engine missing whole subsystems."

**What is missing.** There is no engine-wide configuration identity.
`regen_geometry_id` covers the regen profile; `release_readiness.configuration_id`
is a manifest-level string a human supplies. Nothing binds *the solved state* to
*the exported geometry* to *the mass ledger* to *the BOM*.

**Shape of the fix.** A frozen `ResolvedEngineConfiguration` carrying a content
hash over the solved state, consumed — not re-derived — by thermal analysis,
structures, mass, every CAD builder, BOM, and assembly. Every artifact stamps the
hash. `build_engine_parts` refuses to assemble parts carrying different hashes.

**This is a software-architecture item, not a physics item**, and it is the one
that most directly serves the stated end goal, because it is what makes "the
analyzed wall and the exported wall are the same wall" checkable rather than
asserted. It has no literature dependency. ASME Y14.41-2003 (*Digital Product
Definition Data Practices*, already in the corpus) is the right reference for
what a configuration-controlled digital dataset must carry when this reaches L4.

**Do this first.** It is a precondition for items 4, 7, and 8 being meaningful.

### 2.2 Item 2 — predicted combustion efficiency

`--eta-cstar` is an input, and `design.py:666` warns when `L*`, contraction
ratio, or minimum cylinder length fall back to placeholders (1.0 m, 2.5, 1e-6 m).
Both are honest today, and both prevent the tool from *predicting* what a given
injector/chamber pair will actually deliver.

The corpus supports the framing but not yet a closed model. SP-8089
(`19760023196.pdf`, md line 309):

> "the level of combustion performance is a strong function of the propellant
> spray distributions (mass, mixture-ratio, and drop size). High combustion
> efficiency requires a reasonably uniform overall mixture-ratio distribution,
> initial drop size consistent with the chamber geometry and operating
> conditions, and a uniform mass distribution."

and it names the methodology it defers to — "the JANNAF Performance
Standardization Working Group" models. Line 435 gives an actual mixing-efficiency
correlation for unlike-impinging elements at 60° included angle. But for the
element family this repository actually builds, SP-8089's own table II records
the pintle's status as **"no correlations for level of mixing and spray size"**,
pointing instead to design studies (its refs 10, 38, 39).

**Consequence for scoping.** A general `η_c*` predictor is not derivable from the
corpus as it stands. What *is* derivable is a **calibrated** model: a
configuration-specific `η_c*(mixing, vaporization)` decomposition whose
coefficients are fitted from test data. That makes item 2 depend on item 8, not
the reverse. Until then the honest move is to keep `--eta-cstar` an input and add
a gate: refuse to report delivered `c*` as *predictive* when `eta_cstar` was
supplied rather than derived.

The `L*`/contraction placeholder is separable and should be gated now — a
hardware-oriented job should refuse to run on placeholder chamber geometry rather
than warn.

### 2.3 Item 4 — transient load histories

D1 (quasi-2-D land hot spot) and D2 (Betti conjugate pass) already cover the
steady 1-D → higher-fidelity direction, and they are correctly sited as
post-optimum checks. What no plan covers is that **steady thrust is not the
sizing case**.

SP-125 (`19710019929.pdf`, md line 22694) treats start and shutdown transients as
first-class, and the corpus has enough to *frame* the load case but not to size
it: there is no transient thermal-shock source, no ignition-overpressure source,
and `LITERATURE_REQUESTS_GENERALIZATION.md` already recorded start transients as
"real literature exists… but the tool is steady-state. Not a generalization
blocker."

That judgment was right for generalization and is wrong for fabrication. Once the
end goal is a test article, the start transient is the case that sizes the liner.
Promote it, and request the sources (§5).

### 2.4 Item 5 — injector, film, ignition, stability

Four sub-items with very different readiness:

| sub-item | corpus | code | verdict |
|---|---|---|---|
| Injector hydraulics from real CAD geometry | `pintle_injector/` (62 files) + `PINTLE_LITERATURE_CATALOG.md` | `injector.py`, `injector_cad.py`, G1 landed | **ready to build** |
| Film cooling | Shine 2018, Stechman 1969, the liquid-film review | MDO-only heat-load model; traditional path has ledger but no model (R5a) | **ready to build**; needs a film-ring hardware branch |
| Igniter | **nothing** | **no igniter model exists** — only a port diameter and depth cut through the pintle | **hard-blocked** |
| Combustion stability | SP-8113 (`19750020175`) §2.2.4 acoustic damping analysis, response times at line 1410; SP-194 (`19720026079`) | `chug_margin` static screen only | **partly blocked** — damping side is sourced, combustion-response side is not |

The stability asymmetry matters. SP-8113 gives a calculable damping coefficient
for absorbers and baffles; it does **not** give a combustion response function for
a pintle element. A stability screen built from the damping half alone would look
like a criterion and behave like a guess. Either acquire a response-function
source (ROCCID, §5) or register combustion stability as unavailable with that
reason — the latter is cheap and correct today.

### 2.5 Item 8 — the experiment-to-model return path

Absent from every plan, and the one that closes the loop the whole tool is aimed
at. The corpus has exactly one directly usable seed:

**Perakis, Preis & Haidn, *Wall Heat Flux Evaluation in Regeneratively Cooled
Rocket Thrust Chambers*, J. Thermophysics & Heat Transfer** (`perakis2021.pdf`,
DOI 10.2514/1.T6056). It is a near-perfect template for one channel of the return
path:

- compares gradient methods against an **inverse heat-transfer method**, finding
  the gradient methods show "unacceptable deviations";
- shows that a Nusselt-correlation-based inverse solve "directly influences the
  obtained results and strongly depends on the fuel and geometry" — i.e. it
  identifies the model-form error the repository would otherwise absorb;
- proposes **simultaneous optimization of heat flux and heat-transfer
  coefficient**, which "seems to eliminate the need for additional modeling";
- analyses thermocouple-noise sensitivity (`ΔT_noise`) explicitly;
- and — directly useful for a test article — concludes that **thermocouple
  placement can be optimized before the hardware is designed**.

That last point is the argument for building item 8 *before* the first hot fire,
not after: the instrumentation plan is an output of the model, and a chamber
built without it cannot produce data the model can learn from.

What Perakis does not supply is the general machinery: test-data schemas, sensor
mapping, time alignment, steady-window extraction, uncertainty propagation, and
the calibration/held-out split. Those need V&V/UQ standards, which the corpus
does not have at all (§5).

---

## 3. What the corpus already supports, by workstream

Recorded so it is not re-litigated. "Supported" means a sourced basis exists in
`propulsion_texts/` for the model, within that source's declared scope.

| workstream | supported | thin | absent |
|---|---|---|---|
| 1 config object | ASME Y14.41 (digital dataset), Y14.5 (GD&T) | — | fastener standards (see §4) |
| 2 thermochemistry | RP-1311 I/II, TP-2002-211556, CoolProp, Huber RP-1 surrogate, Arnold hypergolics | pintle-specific mixing | JANNAF/ROCCID methodology |
| 3 nozzle | Rao 1958/1961/1999, SP-8120, Frey & Hagemann, Östlund, Whitfield (scoped), NASA/JHU MOC port | — | — |
| 4 thermal/structural/life | Pizzarelli ×6, Betti 2014, Carlile, Naraghi, Wadel, Miller 1974, Porowski 1985, Hötte 2020, `materials_science/` (30 files) | conjugate validation cases | transient/thermal-shock sources; SP-8040 fracture control |
| 5 injector/film/stability | `pintle_injector/` (62), SP-8089, SP-8113, SP-194, Shine 2018, Stechman 1969 | pintle mixing correlations | igniter (entirely); combustion response function |
| 6 feed system | SP-8109/8052/8107/8125, Parsley & Zhang cycles, SP-8110 turbines, SP-8081 GG | — | **all 6 valve/line/joint SPs; all 4 rotor-support SPs** |
| 7 mass/CAD/mfg | Zandbergen MERs, SP-8087, SP-125, ASME Y14 set | — | fastener family; Waters 1937 flange model |
| 8 test return path | Perakis 2021 (inverse heat flux) | — | **all V&V/UQ methodology** |

---

## 4. Literature status: everything in `LITERATURE_REQUESTS.md` is still missing

Verified this session by NTRS-ID search across all 222 corpus PDFs. **All 25
requested documents are absent.** Not one has been acquired since that document
was written.

| block | items | what it blocks |
|---|---|---|
| Valves, lines, joints, seals | SP-8094 `19740019163`, SP-8097 `19740008129`, SP-8090 `19730013807`, SP-8080 `19730013806`, SP-8123 `19780008146`, SP-8119 `19770009166` | **Item 6 entirely.** `valves`, `lines`, `brackets`, `gimbal` do not exist in the tree; `total_engine_package_mass_kg` can never close |
| Structural | SP-8083 `19720008048`, SP-8019 `19690013956`, SP-8007 `19690013955`, SP-8068 `19710022285`, Waters 1937 (DOI 10.1115/1.4020426), SP-8040 `19700018283` | Track A3 hard enforcement; the flange model; the nozzle-collapse screen's qualification path; item 7 |
| Pump rotor support | SP-8048 `19710018568`, SP-8101 `19730012577`, SP-8121 `19780012205`, SP-8100 `19740014507` | **Item 6's mechanical half.** Bearings are a `0.12 × shaft` placeholder, seals `0.08 × shaft`, no critical-speed check at 30–60 krpm |
| Feed dynamics | SP-8112 `19760010113`, SP-8055 `19710015337`, SP-8030 `19690023005`, SP-8072 `19710023719`, SP-8088 `19740018454` | Item 6's transients; bracket sizing (frequency-driven, not strength-driven); **item 4's transient loads** |
| TVC | SP-8114 `19760010106`, MSFC TB-03 `20240012525` | gimbal |
| Igniter | Tinker `20210000596`, Aerospace 7(8):114 (DOI 10.3390/aerospace7080114) | **Item 5's ignition half, entirely** |
| Injector methodology | ROCCID CR-187109/187110 `19910014917`, `19910014918`; Bazarov & Yang (DOI 10.2514/2.5343) | **Item 2's efficiency model and item 5's stability model** |

**This is the single largest constraint on the program.** Items 5 and 6 — two of
the eight — cannot be built beyond their current placeholders without this block.
Items 2 and 4 are partially constrained by it.

---

## 5. New literature requests

Additions to the existing list, driven by the three new workstreams.

### 5.1 Priority A — V&V and uncertainty methodology (item 8, blocking)

The corpus has **no** verification-and-validation or uncertainty-quantification
methodology at all. Item 8 cannot be specified without it, and building an
ad-hoc calibration pipeline is exactly the "invented authority" the plans forbid.
These are standards and textbooks, not papers:

1. **ASME V&V 20-2009 (R2021)**, *Standard for Verification and Validation in
   Computational Fluid Dynamics and Heat Transfer*. The validation-comparison
   error and validation-uncertainty definitions. This is the one that defines
   what "predicted vs. measured, within tolerance" is allowed to mean.
2. **AIAA G-077-1998**, *Guide for the Verification and Validation of
   Computational Fluid Dynamics Simulations*. Companion framing.
3. **JCGM 100:2008 (GUM)**, *Evaluation of Measurement Data — Guide to the
   Expression of Uncertainty in Measurement*. Free from BIPM. The sensor-side
   uncertainty budget.
4. **Coleman & Steele**, *Experimentation, Validation, and Uncertainty Analysis
   for Engineers*, 4th ed. The practical bridge between 1 and 3 — steady-window
   extraction, propagation, regression uncertainty.

Broad category if you would rather search than acquire specific titles:
*calibration and validation of physics models against instrumented test data,
with parameter estimation and held-out validation*. The Bayesian-calibration
literature (Kennedy & O'Hagan 2001, *Bayesian calibration of computer models*,
JRSS-B 63(3)) is the right formal treatment of the model-discrepancy term, and
would let the return path separate *parameter error* from *model-form error*
rather than absorbing both into fitted coefficients.

### 5.2 Priority B — transients (item 4)

Broad category: *liquid rocket engine start and shutdown transients — thermal
shock, ignition overpressure, and priming loads on regeneratively cooled walls.*

Specific starting points already identified in
`LITERATURE_REQUESTS_GENERALIZATION.md` but never requested:

5. NTRS `20040000363` and `19700024650` (start transient / chill-down).
6. The EUCASS 2017 startup-sequence optimisation paper.
7. **SP-8030** `19690023005` — already on the outstanding list as item 20;
   promote it, because it is the ignition/shutdown transient load source and it
   now blocks item 4 rather than only the gimbal mounts.

### 5.3 Priority C — combustion efficiency and stability (items 2, 5)

8. **ROCCID** CR-187109/CR-187110 (`19910014917`, `19910014918`) — already
   requested as G12′ and still not acquired. It is now blocking *two* items, not
   one. Promote from Tier 3 to Priority 1.
9. Broad category: *JANNAF liquid rocket engine performance prediction
   methodology* — SP-8089 defers to it by name (its refs 1 and 2) and the corpus
   does not contain it. The standardized `ERE`/`η_c*` decomposition lives here.
10. Broad category: *pintle injector combustion efficiency and c\* efficiency
    measurements vs. geometry* — SP-8089 explicitly records that the pintle has
    no mixing/spray-size correlations, so this has to come from the modern pintle
    literature. Several files in `pintle_injector/` may already carry it; that
    folder has a catalog but has not been mined for `η_c*` data specifically.
    **This is a search task inside the existing corpus before it is a request.**

### 5.4 Priority D — igniter (item 5)

Items 24–26 from `LITERATURE_REQUESTS.md`, unchanged and still missing. Broad
category if those specific ones are hard to obtain: *torch / augmented-spark
igniters for small liquid rocket engines — sizing, mass flow split, ignition
energy, and ignition probability.*

---

## 6. Sequenced program

Ordering rule: **an item that makes a later item's evidence checkable comes
first.** That promotes the configuration object above everything, and the return
path above the models it will calibrate.

```
Phase 0 — gates that cost days and unblock honesty (no literature needed)
  0a  F1a: throat Reynolds number + SP-8120 small-nozzle regime gate
  0b  L*/contraction placeholder → refuse for hardware-oriented jobs
  0c  eta_cstar: mark supplied-vs-derived; refuse "predictive" claim when supplied
  0d  combustion stability → register unavailable with the SP-8113 asymmetry reason

Phase 1 — the configuration object            [ITEM 1]  ← critical path
  1a  ResolvedEngineConfiguration + content hash
  1b  every CAD builder, mass ledger, BOM consumes it; none re-derives
  1c  build_engine_parts refuses mixed-hash assembly
  1d  regen exporter takes the solved wall profile directly (closes G2)

Phase 2 — F1 displacement loop                [ITEM 3]  ← now gated by 0a
  2a  iterate δ* into the contour, report the correction magnitude
  2b  the corrected contour is what gets exported, under Phase 1's hash

Phase 3 — the return path                     [ITEM 8]  ← blocked on §5.1
  3a  test-data schema, sensor map, time alignment, steady-window extraction
  3b  inverse heat-flux channel, following Perakis 2021
  3c  calibration / held-out split; residual report per quantity
  3d  instrumentation plan as a model *output* (Perakis's own conclusion)

Phase 4 — what the return path unlocks        [ITEMS 2, 5-partial]
  4a  calibrated eta_cstar decomposition (needs 3c + §5.3)
  4b  injector hydraulics from real CAD geometry (ready now, no blocker)
  4c  film-ring hardware branch + R5a parity (ready now)

Phase 5 — transients                          [ITEM 4]  ← blocked on §5.2
  5a  start/shutdown load histories
  5b  D1/D2 run automatically on the Phase-1 configuration, not by hand

Phase 6 — feed system                         [ITEM 6]  ← hard-blocked on §4
Phase 7 — whole-engine mass/CAD/mfg           [ITEM 7]  ← partly blocked (fasteners)
```

**Phases 0–2 and 4b/4c need no new literature and can start immediately.**
Phase 3 needs §5.1, which is four standards/textbooks rather than a research
hunt. Phases 5–7 are gated on the §4 block that has been outstanding since
`LITERATURE_REQUESTS.md` was written.

---

## 7. What this changes about the defensible claim

The current honest claim — "LREKit can generate coupled engine-design candidates
and reference component geometry that can be carried into cold-flow and subscale
thrust-chamber development" — is right, and Phase 1 is what upgrades it, because
the blocker named in the assessment is precisely a configuration-identity
problem, not a physics problem.

After Phases 0–2 the claim becomes:

> LREKit emits a single configuration-controlled engine definition in which every
> exported artifact, mass, and margin derives from one solved state, with each
> model's validity domain enforced rather than reported.

That is a claim about *integrity*, and it is fully achievable with what is on
disk today. The stronger claim — a complete integrated hot-fire test-article
design — stays gated on §4, and honestly so: an engine definition without valves,
lines, an igniter, and rotor support is not a test article no matter how good the
chamber is.

---

## 8. Immediate next actions

1. **Acquire §5.1** (four V&V/UQ standards). Cheapest unblock in the program; it
   converts item 8 from unspecifiable to buildable.
2. **Re-request the §4 block**, prioritising the six valve/line/joint SPs and
   ROCCID. Nothing has been acquired since the original request.
3. **Mine `pintle_injector/` for `η_c*` data** before requesting more injector
   literature — 62 files with a catalog, not yet searched for this specifically.
4. **Start Phase 0**. Four gates, no dependencies, all in the tree today.
