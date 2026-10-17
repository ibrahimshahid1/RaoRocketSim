# Literature requests — what the repository needs and why

> **Source-status review, September 6:** use the [reconciled literature audit](audit_work/LREKIT_LITERATURE_AUDIT_2026-09-05.md)
> for current acquisition status and applicability. This historical request list
> includes sources now held outside the agent index. Six NASA identifiers have
> been corrected here; the remaining links are not thereby newly verified.

Every entry below is a **gap I actually hit** while implementing, not a general
reading list. Each says what the source supplies, which module needs it, and
what the repository is doing in its absence. Ordered by how much it blocks.

Most NASA SP-8000 monographs predate DOIs; they are identified by NTRS document
ID, which is a stable permalink of the form
`https://ntrs.nasa.gov/citations/<ID>`. Journal entries carry DOIs.

**Naming convention when you convert them:** please drop them in
`propulsion_texts/` (or the relevant subfolder) using the NTRS ID as filename,
e.g. `19730013807.pdf`, matching the existing corpus so
`propulsion_texts_for_agents/paper_index.json` picks them up on the next
conversion run.

---

## Priority 1 — Blocking valves, lines and joints (the unmodelled subsystems)

These four SP monographs are the design-criteria set for everything downstream
of the pumps. Without them there is no sourced basis for valve sizing, actuator
sizing, line sizing or joint design, and I would be inventing correlations.

| # | Document | NTRS | What it gives us | Blocks |
|---|---|---|---|---|
| 1 | **NASA SP-8094**, *Liquid Rocket Valve Components* (Aug 1973, 150 pp) | [19740019163](https://ntrs.nasa.gov/citations/19740019163) | Poppet/ball/butterfly/gate configurations, seat and seal design, flow coefficients, actuation force build-up, materials. Tables of components on operational vehicles. | Valve geometry, valve mass, `Cv`/`Kv` pressure drop |
| 2 | **NASA SP-8097**, *Liquid Rocket Valve Assemblies* (Nov 1973, 154 pp) | [19740008129](https://ntrs.nasa.gov/citations/19740008129) | Valve *selection* parameters, design integration, assembly and functional test. SP-8094's companion — the pair is the documented design sequence (select assembly type, then design components). | Which valve type a given duty needs |
| 3 | **NASA SP-8090**, *Liquid Rocket Actuators and Operators* (May 1973, 158 pp) | [19730013807](https://ntrs.nasa.gov/citations/19730013807) | Pneumatic/hydraulic/electromechanical actuator sizing, response, stall torque, power. | Valve actuator mass and power; also feeds gimbal actuators |
| 4 | **NASA SP-8080**, *Liquid Rocket Pressure Regulators, Relief Valves, Check Valves, Burst Disks, and Explosive Valves* (Mar 1973, 123 pp) | [19740002611](https://ntrs.nasa.gov/citations/19740002611) | Regulator droop and lockup, relief cracking/reseat, burst-disk sizing. | Tank pressurisation branch, relief sizing |
| 5 | **NASA SP-8123**, *Liquid Rocket Lines, Bellows, Flexible Hoses, and Filters* (Apr 1977) | [19780008146](https://ntrs.nasa.gov/citations/19780008146) | Line sizing and routing, bellows spring rate and cycle life, flex-hose selection, filter sizing. **The gimbal flex-joint source.** | Line mass and Δp; gimballed line loads |
| 6 | **NASA SP-8119**, *Liquid Rocket Disconnects, Couplings, Fittings, Fixed Joints, and Seals* (Sep 1976) | [19770017247](https://ntrs.nasa.gov/citations/19770017247) | Fitting and fixed-joint types, seal selection, leakage criteria. | Joint and seal mass; the seals my flange ledger currently excludes |

**Current state without these:** `valves`, `lines`, `brackets` and `gimbal` do
not exist in the repository at all. The mass ledger lists them under `excludes`,
which is honest but means `total_engine_package_mass_kg` can never close.

---

## Priority 2 — Structural, and directly blocking work already in flight

| # | Document | NTRS / DOI | What it gives us | Blocks |
|---|---|---|---|---|
| 7 | **NASA SP-8083**, *Discontinuity Stresses in Metallic Pressure Vessels* (Aug 1971, 69 pp) | [19720008048](https://ntrs.nasa.gov/citations/19720008048) | Shell-to-flange and shell-to-head discontinuity bending. | **The flange model I just wrote.** `flange_cad` sizes a ring but has no discontinuity stress at the chamber/flange junction — which is exactly where flanged pressure vessels fail |
| 8 | **NASA SP-8019**, *Buckling of Thin-Walled Truncated Cones* (Sep 1968, 32 pp) | [19690013956](https://ntrs.nasa.gov/citations/19690013956) | Conical-shell buckling with its own correlation factors. | `mdo/structures.nozzle_collapse_screen` currently maps the cone to an equivalent cylinder via `r/cos α` and cites SP-8019 as the qualification path it cannot yet take |
| 9 | **NASA SP-8007**, *Buckling of Thin-Walled Circular Cylinders* (rev. Aug 1968) | [19690013955](https://ntrs.nasa.gov/citations/19690013955) | §4.2.3 external-pressure criteria. | Already **used** for the nozzle-collapse screen, fetched from NTRS. Please add it to the corpus so the claim is locally verifiable |
| 10 | **NASA SP-8068**, *Buckling Strength of Structural Plates* (Jun 1971, 50 pp) | [19710022285](https://ntrs.nasa.gov/citations/19710022285) | Plate buckling coefficients. | The rib-supported liner screen in `physics.py` uses a classical long-plate relation with a self-declared "screening only" status |
| 11 | Waters, Wesstrom, Rossheim & Williams, *Formulas for Stresses in Bolted Flanged Connections*, ASME Trans. 59(3), 1937 | [10.1115/1.4020426](https://doi.org/10.1115/1.4020426) | The ring + tapered hub + shell elastic model behind ASME VIII Div.1 Appendix 2. | Turning `flange_cad` from a ring into a real flange: hub taper, flange rotation, gasket seating |
| 12 | **NASA SP-8040**, *Fracture Control of Metallic Pressure Vessels* (May 1970, 65 pp) | [19700018283](https://ntrs.nasa.gov/citations/19700018283) | Flaw-growth-based proof-test logic. | Any claim about reusable-cycle life beyond the existing Coffin-Manson screen |

---

## Priority 3 — Pump completion (impeller / inducer / volute / rotor)

The corpus already has SP-8052 (inducers), SP-8107 (turbopump systems) and
SP-8109 (centrifugal turbopumps). The rotor-support side is missing, and it is
what stops the pump BOM from being a real machine rather than a hydraulic sketch.

| # | Document | NTRS | What it gives us | Blocks |
|---|---|---|---|---|
| 13 | **NASA SP-8048**, *Liquid Rocket Engine Turbopump Bearings* (Mar 1971, 85 pp) | [19710018535](https://ntrs.nasa.gov/citations/19710018535) | DN limits, cryogenic bearing life, cooling, preload. | `pumps.py` bearings are a `0.12 × shaft` mass placeholder tagged `placeholder_screen` |
| 14 | **NASA SP-8101**, *Liquid Rocket Engine Turbopump Shafts and Couplings* (Sep 1972, 130 pp) | [19740006328](https://ntrs.nasa.gov/citations/19740006328) | Critical speed, shaft sizing, coupling selection. | Shaft mass is a plain cylinder; there is no rotordynamic critical-speed check at 30–60 krpm, which is where an electric-pump engine actually lives |
| 15 | **NASA SP-8121**, *Liquid Rocket Engine Turbopump Rotating-Shaft Seals* (Feb 1978) | [19780022641](https://ntrs.nasa.gov/citations/19780022641) | Face/lift-off seal design, leakage, purge. | Seals are a `0.08 × shaft` placeholder; LOX-side sealing is a real design driver |
| 16 | **NASA SP-8100**, *Liquid Rocket Engine Turbopump Gears* (Mar 1974, 117 pp) | [19740014507](https://ntrs.nasa.gov/citations/19740014507) | Gear sizing if a reduction stage is ever added. | Lower priority — only if direct-drive is abandoned |

---

## Priority 4 — Feed system, pressurisation and dynamics

| # | Document | NTRS | What it gives us | Blocks |
|---|---|---|---|---|
| 17 | **NASA SP-8112**, *Pressurization Systems for Liquid Rockets* (Oct 1975) | [19770009165](https://ntrs.nasa.gov/citations/19760010113) | Pressurant mass, heat transfer to ullage, regulator sizing. | Tank-side boundary; `MissionSpec.P_tank_*` is currently a free input with no system behind it |
| 18 | **NASA SP-8088**, *Liquid Rocket Metal Tanks and Tank Components* (May 1974, 165 pp) | [19740018454](https://ntrs.nasa.gov/citations/19740018454) | Tank wall, ends, baffles, expulsion. | Only if the tool ever sizes tanks; SP-125 ch. VIII already covers the basics |
| 19 | **NASA SP-8055**, *Prevention of Coupled Structure-Propulsion Instability (POGO)* (Oct 1970, 51 pp) | [19710015337](https://ntrs.nasa.gov/citations/19710015337) | Feed-line compliance, pump gain, structural coupling. | Line and bracket **stiffness** requirements — brackets are not sized by strength, they are sized by frequency |
| 20 | **NASA SP-8030**, *Transient Loads from Thrust Excitation* (Feb 1969, 28 pp) | [19710014805](https://ntrs.nasa.gov/citations/19710014805) | Ignition/shutdown transient loads. | Gimbal and mount load cases; steady thrust is not the sizing case |
| 21 | **NASA SP-8072**, *Acoustic Loads Generated by the Propulsion System* (Jun 1971, 54 pp) | [19710023719](https://ntrs.nasa.gov/citations/19710023719) | Near-field acoustic environment. | Bracket and line random-vibration fatigue |

---

## Priority 5 — Thrust vector control

There is **no liquid-engine gimbal SP monograph** — SP-8114 is solid-rocket TVC.
This is a genuine hole in the SP series, so the gimbal plan leans on SP-8090
(actuators), SP-8123 (flex joints in gimballed lines) and journal literature.

| # | Document | Link | What it gives us |
|---|---|---|---|
| 22 | **NASA SP-8114**, *Solid Rocket Thrust Vector Control* (Dec 1974, 200 pp) | [19760010106](https://ntrs.nasa.gov/citations/19760010106) | Its actuator, valve and duty-cycle sections transfer; the flexible-joint treatment is directly relevant to gimbal bearings. Take the TVC-system framing, ignore the solid-motor specifics |
| 23 | MSFC **TB-03**, *Derivation of Thrust Vector Control Actuator-Force / Gimbal-Torque Transformation Matrix* (2024) | [20240012525](https://ntrs.nasa.gov/citations/20240012525) | The exact actuator-force ↔ gimbal-torque kinematics for a two-actuator gimbal. Modern, short, directly implementable |

You already have `latex-report/Gimbal_TVC_Actuator_Sizing.tex` and
`scripts/tvc_sizing_study.py` in the tree — if those were built from sources,
tell me which and I will not duplicate the request.

---

## Priority 6 — Igniter

The SP series covers *solid* motor igniters (SP-8051) only. Liquid torch
igniters are journal/thesis territory.

| # | Document | Link | What it gives us |
|---|---|---|---|
| 24 | Tinker, D. C., *Compact Augmented Spark Igniters for Liquid Rocket Engines* (PhD, Vanderbilt) | [20210000596](https://ntrs.nasa.gov/citations/20210000596) | Geometric and mass-flow effects on mixture composition, ignition probability mapping. The most complete single source |
| 25 | *Oxygen–Methane Torch Ignition System*, Aerospace 7(8):114, 2020 | [10.3390/aerospace7080114](https://doi.org/10.3390/aerospace7080114) | Sizing and test data for an O2/CH4 torch |
| 26 | Design and Testing of a GOX/GCH4 Igniter for Small-Scale Rocket Engine Thrust Chambers | search AIAA/ResearchGate — I could not resolve a stable DOI | Small-scale sizing, closest to this repository's thrust class |

**Current state:** the igniter is a *port* — a diameter and a depth cut through
the pintle — with no energy balance, no flow rate, no ignition-probability
criterion. It is geometry with no physics behind it.

---

## What I am explicitly NOT asking for

To keep the list actionable:

- **CEA / CoolProp validation cases** (gap 12.6) — these are data, not papers;
  I can generate held-out cases myself once you say go.
- **Anything on the Rao/MOC contour** — the corpus is already strong there
  (Rao 1958, Rao 1999, SP-8120, the variational papers).
- **Combustion instability** — SP-8113 and the Princeton volume are already in
  the corpus and nothing I am building touches them yet.
- **Materials** — `materials_science/` already has GRCop, NARloy-Z, CuCrZr and
  Inconel coverage sufficient for the mass and structural work.

---

## Suggested order if you want to batch downloads

1. **Items 1–6** unblock the entire valves/lines/joints subsystem, which is the
   largest remaining hole in "requirements in, parts out".
2. **Items 7, 8, 11** unblock finishing the flange and the nozzle-collapse
   screen I have already written — smallest effort, immediate payoff.
3. **Items 13–15** turn the pump BOM into a real machine.
4. Everything else is second-order.

---

## Request status 2026-10-16: SP-8068/8094/8097/8090/8080/8048/8121 not held; Huber 2009 and SP-8087 held

The owner's instruction for this session was *do not download anything*. Holdings
were checked against `knowledge/manifest.json` (keys: NTRS accession, SP number, DOI,
title and author keywords) on 2026-10-16. "Pages searched" lists the original-PDF
pages that were read for this session's questions. A NOT HELD entry is a statement
about the local corpus, not about whether the literature exists.

| Request | Accession / DOI | Status | Pages searched / note | Blocks (unblock table) |
|---|---|---|---|---|
| NASA SP-8068, *Buckling Strength of Structural Plates* | 19710022285 | NOT HELD | — | (g) liner buckling as a hard constraint |
| NASA SP-8094, *Liquid Rocket Valve Components* | 19740019163 | NOT HELD | — | (e) valve losses |
| NASA SP-8097, *Liquid Rocket Valve Assemblies* | 19740008129 | NOT HELD | — | (e) |
| NASA SP-8090, *Liquid Rocket Actuators and Operators* | 19730013807 | NOT HELD | — | (e) |
| NASA SP-8080, *Regulators, Relief/Check Valves, Burst Disks* | 19740002611 | NOT HELD | — | (e) |
| NASA SP-8048, *Turbopump Bearings* | 19710018535 | NOT HELD | — | pump BOM placeholders |
| NASA SP-8121, *Turbopump Rotating-Shaft Seals* | 19780022641 | NOT HELD | — | pump BOM placeholders |
| Low-specific-speed / partial-emission (Barske-type) pump design texts | — | NOT HELD | One related held paper (`fuel_pump_design/s11630-013-0601-6.pdf`, 6 pp) was not reviewed this session | (i) 500 N pump path |
| Huber et al. 2009, RP-1 thermophysical properties | 10.1021/ef900216z | **HELD** — `DOC_HUBER2009`, `propulsion_texts/huber2009.pdf` (6 pp) | Not yet extracted | (d) real-fluid coolant properties |
| NASA SP-8087, *Fluid-Cooled Combustion Chambers* | 19730022965 | **HELD** — `DOC_NTRS_19730022965` (130 pp) | PDF pp. 65–68 (sec. 2.3 film cooling), 107–109 (sec. 3.3, Table XI), 78 (coking limit, earlier session) | (f) film-injector hardware |

### Requests raised 2026-10-16 by source reviews (all NOT HELD)

| Request | Why | Evidence |
|---|---|---|
| G. V. R. Rao (1960), "Approximation of Optimum Thrust Nozzle Contour", *ARS Journal* 30(6) | Origin of the θn/θe parabolic-approximation charts that `raosim/nozzle_geometry.py` cites. Rao 1958 (held, `DOC_DOI_10_2514_8_7324`) has no such charts (Figs. 1–6 checked by caption). | P1 record; gap G3 |
| E. N. Sieder and G. E. Tate (1936), *Ind. Eng. Chem.* 28 | Basis of the code's 0.027 Re^0.8 Pr^(1/3) form; the held SP-125 prints Pr^0.4 with a coolant-specific C1 | Gap G19 |
| Coolant-specific C1 for SP-125 Eq. 4-23 (methane, hydrogen, propane) | SP-125 gives only RP-1 (C1 = 0.0214, PDF p.119) | Gap G19, unblock (d) |
| M. E. Hatch and S. S. Papell (1959), NASA TN D-130 (SP-8087 ref. 21) | Basis of the SP-8087 Table XI gaseous-film equation; defines the v_g/v_f symbols | Unblock (f) |
| SP-8087 ref. 60 (film mixing-efficiency ratio η_m/η_g data) | Table XI step (3) otherwise uses 0.4 "as a conservative approximation" | Unblock (f) |
| D. Escher (1996), Penn State MS thesis, pintle injectors | Primary record of the TRW 0.3 < BF < 0.7 advice that Freeberg relays second-hand | P3 record (blockage_lo/hi) |
| H. Ryu et al. (2017), combustion performance of a pintle injector (Hwang 2022 ref. [17]) | Two-row BF sweep 70→85 % and the BF ≈ 85 % optimum, held only second-hand | P3 record; gap G6 |
| D. Kang et al. (2020), pintle tip thermal analysis (Hwang 2022 ref. [20]) | Primary record of the 86 % BF tip damage cited by Hwang | P3 record; gap G6 |
