# Generalization plan: any thrust class, any propellant, any feed system

**Question this answers.** What has to change for a user to state performance
targets — thrust, Isp, mass, envelope, materials — and receive parts, for an
arbitrary engine rather than the 13 kN LOX/RP-1 electric-pump point the solver
was built around.

Companion to `ROADMAP_REQUIREMENTS_TO_PARTS.md` (which orders the *parts* work)
and `LITERATURE_REQUESTS.md` (which lists sources needed for the *unmodelled
subsystems*). This document covers the requirement layer and the three
generalization axes, and names the sources for each.

Corpus paths are relative to `propulsion_texts/`. Markdown mirrors are in
`propulsion_texts/propulsion_texts_for_agents/markdown/`.

---

## 1. What the user should be allowed to specify

Do not invent a requirement list. SP-125 already fixes it.

> "To fit the engine system properly into a vehicle system, engine systems
> design and development specifications will have to cover the following
> parameters above all:
> (1) Thrust level (2) Performance (specific impulse) (3) Run duration
> (4) Propellant mixture ratio (5) Weight of engine system at burnout
> (6) Envelope (size) (7) Reliability (8) Cost (9) Availability
> (time table-schedule)"

— Huzel & Huang, *Design of Liquid Propellant Rocket Engines*, NASA SP-125,
1971, §2.1 "The Major Rocket Engine Design Parameters", printed p. 31
(`19710019929.pdf`, PDF p. 40).

Note what is **not** on that list: chamber pressure, expansion ratio, `L*`,
contraction ratio, channel geometry, pintle diameter, pump speed. SP-125 treats
those as design outputs. That is exactly the split `raosim/mdo` already
implements — the repository's design vector is on the correct side of the line.
What is missing is the left-hand side.

### Conventions SP-125 attaches to these, which the API must carry

**Thrust is meaningless without its condition.** SP-125 §2.1: *"Thrust levels
for first-stage booster engines, which start at or near sea-level altitude and
stop at a specified higher altitude, are usually quoted for sea-level
conditions... The nominal thrust of engines in stages starting and operating at
or near-vacuum conditions is quoted for that environment."* So
`thrust_condition` is part of the requirement, not metadata.

**Isp must state its basis.** *"It is important to state whether a specified
value of `I_s` refers to the complete engine system, or to the thrust chamber
only."* The repository currently reports thrust-chamber Isp; a whole-engine
requirement would have to include turbine/GG overboard flow once open cycles
exist.

**Duration is two numbers, not one.** *"run-duration times of most large
liquid-propellant rocket engines fall into a relatively narrow band, about 50 to
400 seconds"* — but *"User specifications include a formal demonstration...
requiring accumulated duration times, without breakdown, of many times the
comparatively short rated flight duration (typical: six full duration tests for
PFRT of an ICBM). These specifications, therefore, govern most engine design
considerations."* Flight duration sizes battery/tankage; **cumulative
qualification duration** sizes coking, fatigue and life. The repository has one
`burn_time` doing both jobs.

**Mixture ratio is an output, not an input.** SP-125 §2.1 derives it from a
balance between energy release and molecular weight, then modifies it for two
reasons: *stay time* (finite combustion time vs chamber size and weight) and
*cooling considerations* — *"The temperatures resulting from stoichiometric or
near-stoichiometric mixture ratios... may impose severe demands on the
chamber-wall cooling system. A lower temperature, therefore, may be desired and
obtained by selecting a suitable ratio."* This is precisely the coking trade the
MDO already resolves with film fraction, and it is the reason O/F belongs in the
design vector rather than the mission spec.

**Envelope has a literature definition.** *"definition of a hypothetical
smallest cylinder, cube, or sphere into which the engine would fit conveys a
good feeling of engine size or bulkiness."* A cylinder (d_max, l_max) is the
right constraint form.

**Reliability, cost and schedule are real but not NLP constraints.** SP-125
treats them through part count, use of existing designs, and simplicity — von
Kármán's *"Essential elements have to be designed as simply as possible, even if
this means a reduction in quantitative efficiency"* is quoted in §2.1. These map
to *screens on the architecture enumeration* (part count, whether a component
model is qualified vs screening), not to a differentiable constraint.

### Proposed object

```python
EngineRequirement(
    # SP-125 §2.1 (1)-(2), with the conventions above
    thrust                = 5_000.0,
    thrust_condition      = "sea_level" | "vacuum" | ("altitude", 12_000.0),
    isp_min               = 260.0,
    isp_basis             = "thrust_chamber" | "engine_system",

    # (3) — two numbers
    flight_duration       = 30.0,
    qualification_duration= 6 * 30.0,

    # (4) — a *bound*, not a value; None means "let the optimizer choose"
    of_range              = None,

    # (5)-(6)
    burnout_mass_max      = 30.0,
    envelope_diameter_max = 0.30,
    envelope_length_max   = 0.60,

    # architecture candidates — enumerated, not optimized continuously
    propellants           = ("LOX/RP-1", "LOX/LCH4"),
    liner_materials       = ("GRCop-42", "NARloy-Z"),
    feed_architectures    = ("electric_pump", "gas_generator"),

    # operability
    throttle_range        = (0.6, 1.0),
    restarts              = 1,
    reusable_cycles       = 20,

    objective             = "min_mass" | "max_isp" | "min_part_count",
)
```

Everything not chosen as `objective` becomes a constraint. Thrust stays an
internal equality (the outer Newton already closes it), so it costs the NLP
nothing.

---

## 2. Axis A — any thrust class

### What actually breaks today

`MissionSpec.for_thrust()` is sound: throat from the thrust closure, channel
count from chamber circumference at fixed pitch, pintle diameter from the
mid-band blockage factor, pump speed from specific speed. Those derivations are
thrust-general.

`MissionSpec.scaled_design_space()` is not. It scales `D_pintle` and `N_rpm`
and leaves the rest at absolute bounds — including:

```python
VariableSpec("Pc",  1.5e6, 6.0e6, 3.0e6)
VariableSpec("eps", 3.0,   40.0,  8.0)
```

Its own docstring says these *"bracket the 13 kN LOX/RP-1 baseline"*. They are
carried unchanged into every thrust class and every propellant.

### What sets them, from literature

**The `Pc` ceiling is set by the engine cycle, not by thrust.** Yang et al.,
*Thermodynamic Power Cycles for Pump-Fed Liquid Rocket Engines*, 2004
(`fuel_pump_design/thermodynamic-power-cycles-for-pumpfed-liquid-rocket-engines-2004.pdf`),
§I:

| cycle | chamber-pressure limit | limiting mechanism |
|---|---|---|
| expander (open or closed) | ~10 MPa | *"energy available... is limited by the thrust chamber and nozzle heat transfer"* |
| gas generator | 10–15 MPa optimum | total-engine performance optimum incl. overboard flow (<4 % of total) |
| staged combustion | 20–25 MPa | *"performance... generally begins to become hardware limited"* |
| pressure-fed | far lower | *"generally limited to relatively low chamber pressures because high pressures make the vehicle tanks too heavy"* — NASA MSFC, *Liquid Propulsion: Propellant Feed System Design*, `fuel_pump_design/20100035254.pdf` |

**The `ε` ceiling is set by ambient and envelope, not by thrust.** SP-125 Ch. III
on the Alpha upper stages: *"for the A-3 and A-4 engines a slightly smaller
nozzle expansion area ratio has been specified than for the A-2. While all three
upper stages operate in the vacuum and can use the largest practical expansion
area ratio for best performance, other considerations will influence the ratio
actually chosen."* Those other considerations are envelope and mass — which is
why `ε ≤ 40` should fall out of the envelope requirement rather than being
typed in. At sea level the binding limit is separation, already screened per
SP-8120 (`19770009165.pdf`).

**Action.** Replace the two constants with `bounds_for(cycle, propellant,
ambient, envelope)`. It is a function of the architecture selection, so it
belongs at the boundary between Layer 1 and Layer 2 in §5.

### The acceptance test already exists

SP-125 Chapter III designs a four-stage "Alpha" vehicle and works every engine
through to hardware, with intermediate numbers printed (Tables 3-2 through 3-5,
printed pp. 66–77):

| engine | thrust | propellant | feed system |
|---|---|---|---|
| A-1 | 750 000 lbf (3.34 MN), sea level | LOX/RP-1 | turbopump, gas generator |
| A-2 | 150 000 lbf (667 kN) | LOX/LH2 | turbopump |
| A-3 | 16 000 lbf (71 kN), vacuum | LF2/LH2 | pressurized gas-feed |
| A-4 | 7 500 lbf (33 kN), vacuum | N2O4/N2H4 | pressurized gas-feed, throttleable |

A 100× thrust span, four propellant combinations, both feed architectures, one
internally consistent source, already in the corpus. **This is the regression
suite for "any thrust class" and "any propellant" and nothing needs to be
acquired to build it.** Drive each engine from its requirement row (thrust,
Isp, duration, propellant) and compare against the printed table.

---

## 3. Axis B — any propellant

`raosim/mdo/propellants.py` already carries five combinations with sourced
constants (SP-125 Table 4-1 for `L*`, SP-8087 for coolant wall limits, the
repository's Sutton-derived table for chamber gases) and is honest about
coverage — LH2 has no coking limit because hydrogen has no carbon; methane's
`L*` and wall limit are flagged `estimated=True` because methane post-dates both
monographs.

Three gaps, in order.

**B1 — γ, T_c, R are constants, flat in O/F.** Until the CEA surface is sampled,
changing `--mixture-ratio` moves the mass split and densities but not the
thermochemistry, so SP-125's requirement (4) cannot be optimized and the
cooling-driven O/F trade quoted in §1 cannot be resolved. The wiring and the
sampler (`scripts/sample_cea_surface.py`, `MissionSpec.cea_table_path`) already
exist. **This is data generation, not a literature request** — it is the single
highest-leverage item on this page because it unblocks O/F as a design variable,
which is what makes flame temperature the second coking lever the constraint set
already anticipates.

**B2 — coolant and film properties are constants too.** CoolProp is already a
dependency. Properties should be evaluated over the (T, p) the solve actually
visits rather than read from a table row, and validated against held-out cases
(R5b in the roadmap).

**B3 — LF2/LH2 and N2O4/N2H4 are absent**, and are needed for the Alpha
acceptance set. SP-125 Table 4-1 covers N2O4/hydrazine-base at `L*` 30–35 in.
LF2/LH2 needs a source; SP-125 Ch. III uses it for A-3 and gives the operating
parameters, which may be sufficient for a single validation point.

**Methodology note.** The correct generalization is that a propellant is not a
row of constants but a triple: (chamber thermochemistry surface over (Pc, O/F),
coolant property backend, wall-limit rule). Only the third is genuinely a
lookup. Keeping it as a dataclass of scalars is what forces B1 and B2.

---

## 4. Axis C — any feed system / any pump

This is the largest structural change, because it is an *architecture* axis and
`raosim/mdo/` contains exactly one architecture: pintle injector, regen + film
cooling, electric pump-fed, bell nozzle. Grep for `pressure_fed` across
`raosim/mdo/*.py` returns nothing.

### The enumeration is already published

Yang et al. 2004 §I derives all cycle options from **two** configuration
variables:

1. **Turbine energy source** — auxiliary combustion device (gas generator or
   preburner), or the main chamber directly (tapoff) or indirectly (heat
   transfer through the chamber walls, i.e. expander).
2. **Turbine discharge location** — high-pressure sink (main chamber) = closed
   cycle; low-pressure sink (overboard or nozzle skirt) = open cycle.

Their Fig. 1 tabulates the resulting eight configurations with turbine gas
composition options, propellant limitations, and operational-engine examples.
That figure *is* the enumeration table for `FeedArchitecture`. Add pressure-fed
(from the MSFC feed-system chapter) and electric-pump (already implemented) and
the outer loop is defined.

### Screens that make enumeration cheap

Most candidates are eliminated before any NLP runs:

- **Expander cycles restrict the fuel.** *"The fuel must have a high heat
  capacity and adequate heat-transfer properties, and it must vaporize easily.
  Generally, fuels are limited to hydrogen, methane, or propane."* (Yang §II.)
  So LOX/RP-1 + expander is not a candidate, full stop.
- **Preburner richness is propellant-determined.** For LOX/kerosene at equal
  turbine temperature and pump discharge pressure, Yang's worked iteration
  balances the oxidizer-rich cycle at ~23 MPa vs ~12.3 MPa fuel-rich — *"the
  oxygen-rich cycle provides a chamber pressure that is 87 % higher... provides
  the rationale for the selection of the oxidizer-rich cycle as the preferred
  approach for oxygen and kerosene staged combustion engines."* For LOX/LH2 the
  same analysis inverts and fuel-rich wins.
- **Cooling flow fraction differs by cycle.** *"The percentage of the fuel
  required for most cycles is generally less than 20 %... For high-pressure
  kerosene cycles, significantly more than 20 % cooling flow may be required."*
  This couples directly to the existing `film_frac` / jacket split.
- **Hardware ceilings** (Yang §I.C.4): pump discharge pressure ~50 MPa for
  hydrogen, ~100 MPa for kerosene; allowable impeller tip speed ~700 m/s;
  *"maximum number of stages is generally limited to three to avoid pump
  integration concerns such as rotor vibrations, rotor thrust balance, and total
  power transmitted."*

### Two sourced corrections to the existing pump model

**Oxidizer pumps are less efficient than fuel pumps at the same size and
specific speed.** SP-8109 §2.2.2.1 (`fuel_pump_design/19740020848.pdf`): *"The
pumped fluid influences rocket engine pump performance primarily because
oxidizer pumps require large clearances to avoid the possibility of explosion
that may result from rubbing. Oxidizer pumps therefore are less efficient than
fuel pumps for the same size and specific speed."* `raosim/mdo/pump.py`
currently applies one efficiency surrogate to both streams.

**The tip-speed limit should be derived, not fixed.** `raosim/pumps.py` sets
`material_tip_speed_limit: float = 350.0` with no local citation. Yang 2004 puts
the state-of-the-art allowable at ~700 m/s (*"The allowable tip speed of ~700
m/s allows acceptable stresses to be maintained for the impeller"*). 350 m/s may
be defensible for a small additively-manufactured electric-pump impeller, but it
should come out of (σ_y, ρ, shrouded/unshrouded, FoS) — the repo already carries
`rotor_yield_strength` and `rotor_material_density` — not out of a constant.

**Suction-specific-speed context, for the record.** SP-8109 §2.2.2.2: commercial
pumps without inducers are designed for `S_s ≈ 10 000` (gpm units); rocket engine
pumps *"often designed for suction specific speeds in excess of 40 000"*, at an
efficiency penalty from the enlarged inlet (Wislicenus). The repository's
existing `nss_margin` screen is on the right footing.

### Sources needed for Axis C that are not yet in the corpus

`LITERATURE_REQUESTS.md` already covers bearings (SP-8048), shafts and couplings
(SP-8101), rotating-shaft seals (SP-8121) and gears (SP-8100). Axis C adds:

| what | supplies | blocks |
|---|---|---|
| **NASA SP-8110, *Liquid Rocket Engine Turbines*** | turbine sizing, admission, blade stress, efficiency vs velocity ratio, temperature limits | Every pump-fed cycle other than electric. There is no turbine model in the repository at all, so GG and staged combustion cannot close their power balance |
| **NASA SP-8081, *Liquid Propellant Gas Generators*** | GG combustor sizing, mixture-ratio and temperature control, turbine-inlet conditioning | The gas-generator cycle specifically |
| **NASA SP-8124, *Liquid Propellant Rocket Engine Self-Cooled Combustion Chambers*** (ablative/radiation-cooled) | cooling schemes other than regen | Pressure-fed upper stages, which are usually not regen-cooled — A-3 and A-4 in the Alpha set |

NTRS IDs for these three should be confirmed at download time rather than
trusted from memory; the confirmed IDs for documents already in the corpus are
SP-125 = 19710019929, SP-8087 = 19730022965, SP-8120 = 19770009165,
SP-8109 = 19740020848, SP-8052 = 19710025474, SP-8107 = 19750012398,
axial-flow turbopumps = 19780023221, MSFC feed-system chapter = 20100035254.

Pressure-fed needs no new source: SP-125 Chapter V (*Design of Pressurized-Gas
Propellant-Feed Systems*, printed pp. 151–175) covers pressurant determination,
stored-gas, propellant-evaporation and chemical-reaction systems, and §5.6
*Selection of the Pressurization System*. It is the cheapest architecture to add
because it introduces no new turbomachinery — only a different pressure ledger
and a tank-mass term.

---

## 5. Methodology

Four layers. Only Layer 2 exists today, and the plan's §0.1 rule — discrete
choices enumerated outside the traced core, never forced into a continuous
gradient — is already the correct one. The problem is that the enumeration has
exactly one member and it is hardcoded.

```
Layer 0  Requirement mapping        SP-125 §2.1 nine parameters -> objective + constraints
             |                      thrust: internal equality (already)
             |                      Isp, burnout mass, envelope: inequalities
             |                      reliability/cost: screens on Layer 1, not NLP constraints
             v
Layer 1  Architecture enumeration   feed architecture x injector type x cooling scheme
         (discrete, outer)          x liner/jacket family x channel count x pump stages
             |                      screened by Yang Fig.1 + propellant compatibility
             |                      before any NLP runs
             v
Layer 2  Continuous MDO (inner)     existing solver, extended:
             |                        + O/F, L*, contraction ratio, closeout thickness
             |                        + bounds derived per architecture (§2)
             |                        + multi-start (non-convex under a discrete outer loop)
             v
Layer 3  Authoritative re-eval      postprocess.reevaluate -> design_nozzle_v2 -> STEP
                                    (exists)
```

Three methodological commitments worth stating explicitly, because each is a
place where it would be easy to cheat:

1. **Architecture stays discrete.** No relaxation of cycle choice into a
   continuous blend. The screens in §4 are boolean predicates evaluated on the
   candidate, not penalties in the objective.
2. **Requirements become hard constraints, not penalties.** `nlp.py` already
   follows this (*"Hard constraints only; penalties would be for feasibility
   restoration, not the reported optimum"*). Envelope and burnout mass must
   enter the same way.
3. **Generalization is not validated until it reproduces the Alpha set.** A
   solver that converges at 3 MN does not thereby predict at 3 MN. The four
   printed engines are the only cheap, internally consistent, multi-decade-old
   ground truth available, and they span exactly the three axes in question.

---

## 6. Order of work

| # | item | needs new sources? |
|---|---|---|
| 1 | Sample the CEA property surfaces; make O/F a design variable | no — data generation |
| 2 | Derive `Pc`/`ε` bounds from (cycle, propellant, ambient, envelope) | no — Yang 2004 + SP-8120, both in corpus |
| 3 | `EngineRequirement` + requirement→constraint mapping | no — SP-125 §2.1, in corpus |
| 4 | Add `pressure_fed` as the second `FeedArchitecture` | no — SP-125 Ch. V, in corpus |
| 5 | Alpha-set acceptance tests (A-1…A-4) | LF2/LH2 properties only |
| 6 | Gas-generator and staged-combustion architectures | **yes** — SP-8110 turbines, SP-8081 gas generators |
| 7 | Non-regen cooling for pressure-fed upper stages | **yes** — SP-8124 self-cooled chambers |
| 8 | Ox-vs-fuel pump efficiency split; derived tip-speed limit | no — SP-8109, in corpus |

Items 1–5 require nothing that is not already on disk. That is the sequence to
run before asking for anything new.
