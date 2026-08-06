# Literature requests — what the repository needs and why

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
| 4 | **NASA SP-8080**, *Liquid Rocket Pressure Regulators, Relief Valves, Check Valves, Burst Disks, and Explosive Valves* (Mar 1973, 123 pp) | [19730013806](https://ntrs.nasa.gov/citations/19730013806) | Regulator droop and lockup, relief cracking/reseat, burst-disk sizing. | Tank pressurisation branch, relief sizing |
| 5 | **NASA SP-8123**, *Liquid Rocket Lines, Bellows, Flexible Hoses, and Filters* (Apr 1977) | [19780008146](https://ntrs.nasa.gov/citations/19780008146) | Line sizing and routing, bellows spring rate and cycle life, flex-hose selection, filter sizing. **The gimbal flex-joint source.** | Line mass and Δp; gimballed line loads |
| 6 | **NASA SP-8119**, *Liquid Rocket Disconnects, Couplings, Fittings, Fixed Joints, and Seals* (Sep 1976) | [19770009166](https://ntrs.nasa.gov/citations/19770009166) | Fitting and fixed-joint types, seal selection, leakage criteria. | Joint and seal mass; the seals my flange ledger currently excludes |

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
| 13 | **NASA SP-8048**, *Liquid Rocket Engine Turbopump Bearings* (Mar 1971, 85 pp) | [19710018568](https://ntrs.nasa.gov/citations/19710018568) | DN limits, cryogenic bearing life, cooling, preload. | `pumps.py` bearings are a `0.12 × shaft` mass placeholder tagged `placeholder_screen` |
| 14 | **NASA SP-8101**, *Liquid Rocket Engine Turbopump Shafts and Couplings* (Sep 1972, 130 pp) | [19730012577](https://ntrs.nasa.gov/citations/19730012577) | Critical speed, shaft sizing, coupling selection. | Shaft mass is a plain cylinder; there is no rotordynamic critical-speed check at 30–60 krpm, which is where an electric-pump engine actually lives |
| 15 | **NASA SP-8121**, *Liquid Rocket Engine Turbopump Rotating-Shaft Seals* (Feb 1978) | [19780012205](https://ntrs.nasa.gov/citations/19780012205) | Face/lift-off seal design, leakage, purge. | Seals are a `0.08 × shaft` placeholder; LOX-side sealing is a real design driver |
| 16 | **NASA SP-8100**, *Liquid Rocket Engine Turbopump Gears* (Mar 1974, 117 pp) | [19740014507](https://ntrs.nasa.gov/citations/19740014507) | Gear sizing if a reduction stage is ever added. | Lower priority — only if direct-drive is abandoned |

---

## Priority 4 — Feed system, pressurisation and dynamics

| # | Document | NTRS | What it gives us | Blocks |
|---|---|---|---|---|
| 17 | **NASA SP-8112**, *Pressurization Systems for Liquid Rockets* (Oct 1975) | [19770009165](https://ntrs.nasa.gov/citations/19760010113) | Pressurant mass, heat transfer to ullage, regulator sizing. | Tank-side boundary; `MissionSpec.P_tank_*` is currently a free input with no system behind it |
| 18 | **NASA SP-8088**, *Liquid Rocket Metal Tanks and Tank Components* (May 1974, 165 pp) | [19740018454](https://ntrs.nasa.gov/citations/19740018454) | Tank wall, ends, baffles, expulsion. | Only if the tool ever sizes tanks; SP-125 ch. VIII already covers the basics |
| 19 | **NASA SP-8055**, *Prevention of Coupled Structure-Propulsion Instability (POGO)* (Oct 1970, 51 pp) | [19710015337](https://ntrs.nasa.gov/citations/19710015337) | Feed-line compliance, pump gain, structural coupling. | Line and bracket **stiffness** requirements — brackets are not sized by strength, they are sized by frequency |
| 20 | **NASA SP-8030**, *Transient Loads from Thrust Excitation* (Feb 1969, 28 pp) | [19690023005](https://ntrs.nasa.gov/citations/19690023005) | Ignition/shutdown transient loads. | Gimbal and mount load cases; steady thrust is not the sizing case |
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
