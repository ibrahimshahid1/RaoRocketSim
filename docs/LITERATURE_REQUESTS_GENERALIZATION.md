# Literature requests — generalization to any thrust class, propellant, and feed system

Companion to `LITERATURE_REQUESTS.md` (items 1–26, which cover the *unmodelled
subsystems*: valves, lines, joints, TVC, igniter, pump rotor support). This file
covers what the **generalization axes** in
`GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md` need, and nothing that is
already on disk.

**Method.** Every entry was checked against the 222 PDFs in `propulsion_texts/`
and its `fuel_pump_design/`, `materials_science/` and `pintle_injector/`
subfolders via `propulsion_texts_for_agents/paper_index.md`. Section 5 lists the
gaps I *expected* to find and did not, so you can see what was ruled out rather
than missed.

**ID provenance.** NTRS IDs and DOIs marked ✅ were resolved during this pass.

---

## Acquisition status — 2026-08-06

**13 of 14 acquired and verified.** Files are in `propulsion_texts/`; page
counts and text layers were checked by extraction, and all three large SP
monographs (SP-8081, SP-8110, SP-8124) have usable text layers on body pages.

| item | file | pages | status |
|---|---|---|---|
| G1 SP-8110 turbines | `19740026132.pdf` | 160 | ✅ verified |
| G2 SP-8081 gas generators | `19730018978.pdf` | 116 | ✅ verified |
| G3 SP-8124 self-cooled chambers | `19780013268.pdf` | 138 | ✅ verified |
| G4 RP-1311 Pt I | `19950013764.pdf` | 64 | ✅ verified |
| G5 RP-1311 Pt II | `19960044559.pdf` | 184 | ✅ verified |
| G6 TP-2002-211556 | `20020085330.pdf` | 295 | ✅ verified |
| G7 RP-1/RP-2 surrogate | `huber2009.pdf` | 6 | ✅ verified |
| G8 CoolProp | `bell2014.pdf` | 11 | ✅ verified |
| G9 hypergolic properties | `Vandenberg_Corrected_Properties.pdf` | 10 | ✅ verified, **citation now resolved** |
| G10 pseudo-boiling / HTD | `nasuti2021.pdf` | 14 | ✅ verified |
| G11 mass/size MERs | `PaperLREenginemassandsizing.pdf` | 15 | ✅ verified |
| G12 swirl-coax stability | `20130014182.pdf` | **1** | ❌ **abstract only — see G12′** |
| G13 LOX/CH4 perf. & stability | `20100034924.pdf` | 55 | ✅ verified |
| G14 throttling review | `20090037061.pdf` | 61 | ✅ verified |

**Rename before the next conversion run** so `paper_index.json` resolves them
consistently with the rest of the corpus: `huber2009.pdf`, `bell2014.pdf` and
`nasuti2021.pdf` already follow the author-year convention;
`PaperLREenginemassandsizing.pdf` → `zandbergen2015.pdf` and
`Vandenberg_Corrected_Properties.pdf` → `arnold_hypergolic_properties.pdf`
would match. NTRS-ID filenames are already correct.

Naming convention for future additions: drop into `propulsion_texts/` (or the
matching subfolder) using the NTRS ID as filename, e.g. `19740026132.pdf`, so
`paper_index.json` picks it up on the next conversion run.

---

## Tier 1 — Directly blocks a generalization axis

### 1a. Feed system and cycle (Axis C)

The repository models exactly one architecture: electric-pump-fed. Adding any
turbine-driven cycle needs a turbine model and a gas-generator model, neither of
which exists in any form. Adding a pressure-fed upper stage needs a non-regen
chamber model.

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| G1 | **NASA SP-8110**, *Liquid Rocket Engine Turbines* (Jan 1974, 158 pp) | ✅ NTRS [19740026132](https://ntrs.nasa.gov/citations/19740026132) | Turbine type selection, admission, velocity-ratio efficiency, blade stress, temperature limits, installation | **Every pump-fed cycle except electric.** There is no turbine model in the repository, so gas-generator and staged-combustion cycles cannot close their power balance at all |
| G2 | **NASA SP-8081**, *Liquid Propellant Gas Generators* (Mar 1972, 110 pp) | ✅ NTRS [19730018978](https://ntrs.nasa.gov/citations/19730018978) | Bipropellant GG sizing, mixture-ratio and temperature control, turbine-inlet conditioning, stability | The gas-generator cycle specifically — the 10–15 MPa `Pc` band that Yang 2004 identifies as its performance optimum |
| G3 | **NASA SP-8124**, *Liquid Rocket Engine Self-Cooled Combustion Chambers* (Sep 1977) | ✅ NTRS [19780013268](https://ntrs.nasa.gov/citations/19780013268) | Five self-cooled types — ablative, radiation-cooled, internally regenerative (Interegen), heat sink, adiabatic wall — and adiabatic-wall temperature control via injector or film-coolant ring | Pressure-fed upper stages, which are usually **not** regeneratively cooled. Both A-3 and A-4 in the SP-125 Alpha validation set are pressurized gas-feed; the repository can only model a regen jacket, so it cannot currently reproduce either |

Pressure-fed itself needs no new source — SP-125 Ch. V (`19710019929.pdf`,
printed pp. 151–175) covers pressurant determination, stored-gas systems and
§5.6 *Selection of the Pressurization System*.

### 1b. Thermochemistry (Axis B) — the largest single hole found

The entire property-surface layer depends on CEA. **No CEA primary reference is
in the corpus** — only papers that cite it. R5b (held-out property validation)
is on the critical path in `ROADMAP_REQUIREMENTS_TO_PARTS.md`, and a validation
report cannot be written against a code with no specification on hand.

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| G4 | Gordon & McBride, **NASA RP-1311 Part I**, *Computer Program for Calculation of Complex Chemical Equilibrium Compositions and Applications: I. Analysis* (1994) | ✅ NTRS [19950013764](https://ntrs.nasa.gov/citations/19950013764) | The assumptions, governing equations and solution method behind every `c*`, γ and `T_c` the tool reports; rocket-performance, shock and detonation applications | R5b held-out validation; any defensible statement about what the property surfaces are approximating |
| G5 | McBride & Gordon, **NASA RP-1311 Part II**, *II. Users Manual and Program Description* (1996) | ✅ NTRS [19960044559](https://ntrs.nasa.gov/citations/19960044559) | Input-file semantics, the exact problem types, output definitions | Reproducibility of `scripts/sample_cea_surface.py` — what was actually asked of CEA has to be recorded, not just the answer |
| G6 | McBride, Zehe & Gordon, **NASA/TP-2002-211556**, *NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species* (Sep 2002) | ✅ NTRS [20020085330](https://ntrs.nasa.gov/citations/20020085330) | Least-squares coefficients for >2000 species, 200–20 000 K: 7-term fit for `Cp°(T)/R` with integration constants for `H°(T)/RT` and `S°(T)/R` | **Generating property surfaces without a RocketCEA install.** This is the data layer, so it also removes the host-dependency that currently gates O/F from becoming a design variable |

### 1c. Real-fluid coolant and propellant properties (Axis B)

`MissionSpec` carries `rho_cool`, `cp_cool`, `k_cool`, `mu_cool` as four
scalars. For "any fuel" that has to become a fluid model, and the two fuels the
repository most needs are exactly the two CoolProp does not have.

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| G7 | Huber, Lemmon, Ott & Bruno, *Preliminary Surrogate Mixture Models for the Thermophysical Properties of Rocket Propellants RP-1 and RP-2*, **Energy & Fuels** 23(6), 2009 | ✅ DOI [10.1021/ef900216z](https://doi.org/10.1021/ef900216z) | Four-component RP-1 surrogate (α-methyldecalin, n-dodecane, 5-methylnonane, heptylcyclohexane) fitted to density, sound speed, viscosity, thermal conductivity and advanced distillation curves. Quoted 95 %-confidence accuracy: density 0.4 %, sound speed 2 %, viscosity 2 %, thermal conductivity 4 %, distillation 0.5 % | **CoolProp has no RP-1.** This is the only route from four constants to a validated fluid, and it comes with its own error bars — which is what R5b needs |
| G8 | Bell, Wronski, Quoilin & Lemort, *Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp*, **Ind. Eng. Chem. Res.** 53(6):2498–2508, 2014 | ✅ DOI [10.1021/ie4033999](https://doi.org/10.1021/ie4033999) | The reference for a library the project already depends on | Provenance completeness — `MODEL_REGISTRY.md` cites CoolProp as a backend with no citation behind it |
| G9 | **S. L. Arnold** (ENSCO, Inc., Vandenberg AFB, CA), *Physical & Thermodynamic Properties of Hypergolic Propellants: A Review and Update*. Approved for public release, distribution unlimited; venue and year are not printed in the document | ✅ author/affiliation confirmed from the acquired PDF; `propulsion_texts/Vandenberg_Corrected_Properties.pdf` | N2O4, MMH and Aerozine-50 properties, **many as explicit temperature-dependent correlations**, derived by (1) cross-comparison of literature values with a data-quality assessment separating measured from estimated, (2) fitted temperature-dependent coefficients validated against independent measurements, (3) corresponding-states / group-contribution estimation for missing parameters | N2O4/MMH and N2O4/N2H4, which CoolProp also lacks. Needed for A-4 in the Alpha set. **Read the scope caveat:** the author states the set is "suitable for environmental modeling applications and more general engineering calculations", and the reference list is dominated by LLNL dispersion-modelling reports — so treat these as validated *bulk fluid* properties, not as a combustion-chamber thermochemistry source, and keep the provenance tag honest about that. Companion of record is **CPIA/M4** *Liquid Propellant Manual* (JHU-APL), which is distribution-controlled |

### 1d. The cooling failure mode not currently screened (Axis B)

The coking constraint (SP-8087, 728 K for RP-1) is a **hydrocarbon** mechanism.
`mdo/propellants.py` correctly sets it to `None` for hydrogen. But nothing
replaces it — and for methane and hydrogen the governing coolant-side failure is
heat-transfer deterioration, which the repository does not screen at all.

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| G10 | Nasuti & Pizzarelli, *Pseudo-boiling and heat transfer deterioration while heating supercritical liquid rocket engine propellants*, **J. Supercritical Fluids** 168:105066, 2021 | ✅ DOI [10.1016/j.supflu.2020.105066](https://doi.org/10.1016/j.supflu.2020.105066) | HTD onset conditions, the pseudo-boiling mechanism, and the role of exit-to-critical pressure ratio and surface roughness in avoiding it | A **methane or hydrogen** cooling constraint. Without it, LOX/LCH4 and LOX/LH2 designs are unconstrained on the coolant side, which will read as "feasible" when it is really "unmodelled". The corpus already has Pizzarelli 2015 (supercritical methane *modelling*, PAPER-0207/0208) but not the deterioration review |

---

## Tier 2 — Needed to validate across the thrust span

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| G11 | Zandbergen, *Simple mass and size estimation relationships of pump fed rocket engines for launch vehicle conceptual design*, **6th EUCASS**, 2015 | ResearchGate 279711349 — EUCASS proceedings, confirm DOI at download | Regressions from >45 historical and current pump-fed engines, 15 kN–8 MN, split by pressure-fed / turbopump-fed and storable / cryogenic / semi-cryogenic; also envelope-diameter relations | **Independent cross-check on `burnout_mass_max` and envelope across thrust classes.** Explicitly *not* a replacement for the geometry-integrated ledger in `HARDWARE_MASS_LEDGER.md` — that approach is better. But there is currently no way to tell whether the ledger is right at 3 MN, and this is the cheapest way to find out |

---

## Tier 3 — Architecture axes beyond the feed system

Lower priority than Tier 1, but each unlocks a discrete choice that is currently
hardcoded.

| # | Document | ID | Supplies | Blocks |
|---|---|---|---|---|
| ~~G12~~ | ~~Hulka & Casiano, *Review of Combustion Stability Characteristics of Swirl Coaxial Element Injectors*, 60th JANNAF Propulsion Meeting, Colorado Springs, Apr–May 2013~~ | ❌ NTRS [20130014182](https://ntrs.nasa.gov/citations/20130014182) hosts **the abstract only** (1 page) — JANNAF papers are distribution-restricted | — | **Superseded by G12′ and G12″ below.** The abstract is still worth keeping: it names the two analysis methodologies the paper evaluates against, which is what led to the replacements |
| **G12′** | **NASA CR-187109 / CR-187110**, Muss & Nguyen, *User's Manual for Rocket Combustor Interactive Design (ROCCID) and Analysis Computer Program*, Vol. I and Vol. II (Appendices A–K), 1991 | ✅ NTRS [19910014917](https://ntrs.nasa.gov/citations/19910014917) (Vol. I), [19910014918](https://ntrs.nasa.gov/citations/19910014918) (Vol. II) | A **standardized NASA methodology** for steady-state combustion performance *and* combustion stability of mixed-element injector patterns: impinging like-doublet, unlike-triplet, showerhead, shear-coaxial and swirl-coaxial elements — plus real propellant properties for oxygen, hydrogen, methane, propane and RP-1 | **This is a better fit than the paper it replaces.** It covers all five injector families in one document rather than one family, and it is the code Hulka's abstract names as its evaluation baseline. It is the natural reference for turning `injector_type` into a real enumerated architecture axis, and its propellant set overlaps the repository's almost exactly |
| **G12″** | Bazarov & Yang, *Liquid-Propellant Rocket Engine Injector Dynamics*, **J. Propulsion and Power** 14(5):797–806, 1998 | ✅ DOI [10.2514/2.5343](https://doi.org/10.2514/2.5343) | Injector dynamic response under chamber and feed-line oscillation — the second methodology Hulka's abstract cites alongside Hewitt | The dynamic half of injector stability. The repository's existing `chug_margin` is a static screen; this is what a feed-coupled criterion would be built on |
| G13 | Hulka & Jones, *Performance and Stability Analyses of Rocket Thrust Chambers with Oxygen/Methane Propellants* | ✅ NTRS [20100034924](https://ntrs.nasa.gov/citations/20100034924) | Combustion performance and stability modelling specifically for LOX/CH4, including the Hewitt-type `d₀/U_j` screening parameter | Methalox stability screening. The repository's methane entry is already flagged `estimated=True`; this is the corresponding combustion-side evidence |
| G14 | Casiano, Hulka & Yang, *Liquid-Propellant Rocket Engine Throttling: A Comprehensive Review*, **J. Propulsion and Power** 26(5):897–923, 2010 | ✅ DOI [10.2514/1.49791](https://doi.org/10.2514/1.49791); NTRS [20090037061](https://ntrs.nasa.gov/citations/20090037061) | Throttling methods, achievable ranges by architecture, and the stability/performance cost of each | `EngineRequirement.throttle_range`. The repository has a movable-pintle model but no basis for saying which *architectures* can meet a requested range |

---

## What I checked and concluded you do **not** need

Recorded so this is not re-litigated.

| topic | why it is covered |
|---|---|
| Combustion instability theory | **SP-8113** *Combustion Stabilization Devices* = `19750020175`, and the Princeton volume **NASA SP-194** *Liquid Propellant Rocket Combustion Instability* = `19720026079`. Both in corpus |
| Injectors, general design criteria | **SP-8089** *Liquid Rocket Engine Injectors* = `19760023196`, in corpus (also duplicated in `pintle_injector/`) |
| Axial-flow pumps | **SP-8125** *Liquid Rocket Engine Axial-Flow Turbopumps* = `19780023221`, in corpus. Axial pumps are already sourced even though not modelled |
| Centrifugal pumps, inducers, turbopump systems | SP-8109 = `19740020848`, SP-8052 = `19710025474`, SP-8107 = `19750012398`. All in corpus |
| Cycle selection and hardware limits | Yang et al. 2004, `fuel_pump_design/thermodynamic-power-cycles-for-pumpfed-liquid-rocket-engines-2004.pdf`. This is the enumeration table for `FeedArchitecture` |
| Pressure-fed vs pump-fed trade | NASA MSFC *Liquid Propulsion: Propellant Feed System Design* = `fuel_pump_design/20100035254.pdf` |
| AM copper alloys and channel-wall nozzles | Gradl 2016–2020 = PAPER-0081…0088; `materials_science/` carries GRCop-42/84, NARloy-Z, CuCrZr and Inconel coverage |
| Nozzle contour and separation | Rao 1958/1999, SP-8120 = `19770009165`, Frey & Hagemann, Hagemann 1998, Schomberg, the dual-bell variational paper. Corpus is strong here |
| Regen cooling correlations and HARCC | Pizzarelli 2011/2013/2014/2015/2020, Betti 2014, Carlile 1992, Naraghi 2004, Wadel 1997, Mirzamoghadam 1991. Corpus is strong here |
| Film cooling | Shine & Nidhi 2018, Stechman 1969, the liquid-film-cooling review. In corpus |
| Low-cycle fatigue / life | Miller 1974, Porowski 1985, Hötte 2020, the 1995 reusable-chamber life paper, `s12666-010-0089-7`. In corpus |
| Start transients, chill-down, priming | Real literature exists (NTRS 20040000363, 19700024650, and the EUCASS 2017 startup-sequence optimisation paper) but the tool is **steady-state**. Not a generalization blocker — revisit only if start/shutdown becomes a requirement, which SP-125 §2.1 notes it often is at the vehicle level |

---

## Still outstanding

Only one request remains open:

| # | Document | ID | Why |
|---|---|---|---|
| **G12′** | NASA CR-187109 / CR-187110, ROCCID Vol. I and Vol. II | ✅ NTRS [19910014917](https://ntrs.nasa.gov/citations/19910014917), [19910014918](https://ntrs.nasa.gov/citations/19910014918) | Replaces the abstract-only G12. Five injector element families and five real propellants in one NASA-standardized methodology |
| **G12″** | Bazarov & Yang, JPP 14(5):797–806, 1998 | ✅ DOI [10.2514/2.5343](https://doi.org/10.2514/2.5343) | Injector dynamic response — optional, only if feed-coupled stability becomes a constraint |

G12′ is Tier 3 (it unlocks the injector-family axis, not the thrust/propellant/
feed axes), so it is not blocking. G12″ is optional.

---

## Where this leaves the plan

With G1–G11, G13 and G14 in hand, **every Tier-1 blocker is now sourced**. The
sequence in `GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md` §6 no longer has a
literature dependency until item 6 (gas-generator and staged-combustion
architectures), and that dependency is now satisfied too.

Revised order, with the sources each step now has:

| # | item | source now on disk |
|---|---|---|
| 1 | Sample CEA property surfaces; make O/F a design variable | G4, G5, **G6** — the 7-term species fits mean this can be done without a RocketCEA install |
| 2 | Derive `Pc`/`ε` bounds from (cycle, propellant, ambient, envelope) | Yang 2004 + SP-8120, both already held |
| 3 | `EngineRequirement` + requirement→constraint mapping | SP-125 §2.1, already held |
| 4 | Add `pressure_fed` as the second `FeedArchitecture` | SP-125 Ch. V + **G3** for the non-regen chamber |
| 5 | Alpha-set acceptance tests (A-1…A-4) | **G9** covers A-4's N2O4/N2H4 bulk properties; LF2/LH2 for A-3 still has no property source beyond SP-125's own operating table |
| 6 | Gas-generator and staged-combustion architectures | **G1** (turbines) + **G2** (gas generators) |
| 7 | Non-regen cooling for pressure-fed upper stages | **G3** |
| 8 | Ox-vs-fuel pump efficiency split; derived tip-speed limit | SP-8109, already held |
| — | Methane/hydrogen coolant-side constraint | **G10** — this was not previously in the plan and should be added; it is the missing counterpart to the RP-1 coking screen |
| — | Cross-check mass and envelope across the thrust span | **G11** |
| — | Throttle-range feasibility by architecture | **G14** |
| — | Methalox combustion performance and stability screening | **G13** |

**One residual gap worth naming:** LF2/LH2 (A-3 in the Alpha set) still has no
property source other than SP-125's own operating-parameter table. Fluorine is
not in CoolProp and is not covered by G7 or G9. Either accept SP-125's tabulated
values as the single validation point, or drop A-3 from the acceptance set and
validate on A-1, A-2 and A-4 — which still spans 100× in thrust, three
propellant classes and both feed architectures.
