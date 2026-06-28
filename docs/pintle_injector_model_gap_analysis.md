# Pintle injector model gap analysis

This document completes the second half of the pintle-injector literature task:
it evaluates the current RaoRocketSim approach and proposes a literature-rooted
design flow for pintle injector sizing, testing, and simulation.

Companion sources:

- `docs/pintle_injector_literature_inventory.md`
- `docs/injector_pintle_provenance.md`
- `docs/chamber_design_literature.md`
- `docs/thermofluid_literature_provenance.md`
- `docs/regen_wall_model.md`

## Executive diagnosis

RaoRocketSim is already pointed in the right architectural direction: the
pintle injector is not an isolated calculator. It is sized from the same engine
operating point that sets the nozzle, chamber, and cooling system:

```text
user intent -> Pc, thrust or Rt, epsilon, O/F, propellants
            -> Cf, c*, At, total mdot
            -> fuel/oxidizer mdot split
            -> injector dP and feed states
            -> annulus/slot areas, velocities, TMR, spray and gates
            -> chamber/nozzle/cooling/pump feasibility
```

That is the right backbone for a physical design tool. The main problem is not
that the repository lacks pintle math. The problem is that several high-impact
pintle-specific relations are still represented by generic screening surrogates
instead of correlations selected from the pintle literature. This is most
important for spray angle, atomization/SMD, combustion efficiency, stability,
and heat-flux distribution.

## Current approach in the repository

### What is already strong

1. The injector is coupled to the engine operating point.

   `raosim/design.py` and `scripts/run_nozzle.py` size the pintle from chamber
   pressure, throat area or target thrust, expansion ratio, c-star, thrust
   coefficient, and mixture ratio. This matches the correct engine-level
   causality in Huzel and Huang/SP-125, SP-8089, and the TRW/Apollo pintle
   papers: injector flow areas should serve the selected engine point, not
   invent an independent injector thrust.

2. The hydraulic core uses the correct first-order injector relation.

   The current orifice model uses:

   ```text
   mdot = Cd A sqrt(2 rho dP)
   v = mdot / (rho A)
   TMR = (mdot_radial v_radial) / (mdot_axial v_axial)
   ```

   This is the right SP-8089/Rocket Propulsion Elements starting point for
   preliminary injector sizing.

3. The model distinguishes liquid, gas, supercritical, and flashing/two-phase
   conditions.

   `raosim/injector.py` has incompressible liquid and compressible gas branches
   with a choked-flow test. Two-phase/flashing states are rejected instead of
   being silently forced through the wrong equation. This is exactly the kind
   of phase-branch discipline the pintle literature requires, because
   liquid-liquid, gas-liquid, gas-gas, and supercritical pintles do not share
   one universal spray law.

4. The model includes practical design gates.

   Existing gates cover mass-flow closure, injector pressure sufficiency,
   cavitation margin, minimum features, annulus concentricity, manifold
   maldistribution, spray-wall interception, combustion-development length,
   target TMR, thermal margin, chug, chamber acoustic modes, n-tau screening,
   and cold-flow/hot-fire validation status.

5. Regenerative cooling can hand the fuel outlet state to the injector.

   `evaluate_pintle_injector` checks whether direct regen coolant flow equals
   cycle fuel flow before using the jacket outlet temperature and pressure as
   the fuel injector inlet. This is a good physical boundary: bypassed fuel
   needs an explicit split/mixing model.

6. The tool already produces diagnostics.

   The code exports `pintle.json`, figures for hydraulics/spray/atomization/
   thermal/stability/manifold/gates, a throttle map, and preliminary CAD. That
   is the right framework for iterative design.

### What is weak or incomplete

1. Spray half-angle is currently a momentum-vector surrogate.

   The code computes spray direction from axial and radial momentum vectors
   and an optional deflector angle. That is a useful physical sanity check, but
   the local corpus has better pintle-specific correlations: Cheng et al.
   (2017), Son et al. (2015/2017), Freeberg et al. (2019), Zhou and Shen
   (2022), Lee et al. (2020/2021), and the 2022 review by Zhao et al.

   The missing inputs are block factor, local momentum ratio, annular area
   ratio, skip/impingement distance, and sometimes recess or radial-hole
   geometry. The current model computes blockage factor but does not use it to
   choose a correlation.

2. SMD and combustion efficiency are generic screens.

   Current SMD uses a Hinze critical-Weber aerodynamic breakup estimate capped
   at jet hydraulic diameter, then a Reitz-Bracco breakup length and d-squared
   vaporization law. That is acceptable as a clearly labeled screen, but it is
   not yet a pintle injector atomization model. The local papers by Ninish,
   Radhakrishnan, Son, Zhou, Song, and the gas-liquid GO2/kerosene study should
   drive branch-specific correlations.

3. Chamber heat-flux coupling is too simple for pintles.

   Pintle injectors can create wall-directed sheets, recirculation zones,
   local hot streaks, or deliberate fuel film cooling. The current
   face/tip thermal model is a Dittus-Boelter recirculation screen. It does not
   yet use injector recess, spray-wall impingement, flame anchoring, fuel-rich
   wall film fraction, or azimuthal nonuniformity. Ahn et al. (2014), Kang et
   al. (2022), Sakaki et al. (2015-2018), and the GO2/kerosene heat-flux study
   should inform this layer.

4. The chamber L-star is still mostly a user/default geometry input.

   `chamber_contour` solves the geometry exactly for `V = L* At`, which is
   good. But L-star should be tied to the chosen injector/propellant quality.
   SP-125 gives baseline ranges, while the pintle papers show that atomization,
   spray angle, chamber pressure, propellant phase, and throttle level can move
   the required chamber length.

5. Pump/feed closure is only partially explicit.

   The injector checks `feed pressure >= Pc + injector dP`, and regen cooling
   can set a jacket outlet boundary. The missing holistic condition is:

   ```text
   pump/tank outlet pressure
     >= Pc
      + injector dP
      + injector manifold loss
      + line and valve losses
      + cooling jacket loss, if the fuel passes through regen first
      + control margin
   ```

   The turbopump monographs in the corpus should become pump-capacity and NPSH
   gates, not just background references.

6. The interactive CLI does not yet ask the injector design questions.

   `scripts/run_nozzle.py` exposes many pintle flags and a good backend flow,
   but its interactive interview currently prompts nozzle/chamber/regen
   questions, not the pintle-specific design intent. `main.py` has pintle flags
   in batch-v2, but the interactive path still stops at nozzle/chamber output.

7. "Target momentum ratio" is only a gate, not an optimizer.

   If the user asks for a target TMR, the model reports achieved vs target, but
   it does not solve the fuel/oxidizer dP split, stream assignment, pintle
   diameter, slot count, or annulus area to hit the target subject to pump and
   manufacturing constraints.

## Literature-rooted design variables

The CLI should separate user intent, engine-cycle variables, injector
geometry, validation assumptions, and pump/feed constraints.

### User intent variables

Ask these first because they set the design target:

- Target thrust or throat radius.
- Chamber pressure.
- Ambient/design altitude or nozzle pressure-ratio target.
- Propellant pair and mixture ratio.
- Thermochemistry mode: constant gamma, CEA frozen, or CEA equilibrium.
- Desired throttle range and number of throttle design points.
- Desired design mode: screening, cold-flow-calibrated, or hot-fire-correlated.

### Chamber/nozzle variables

These set the domain into which the pintle sprays:

- Expansion ratio or target exit pressure.
- Chamber contraction ratio.
- L-star or "auto L-star from injector quality" mode.
- Minimum cylindrical chamber length.
- Convergent angle and throat radii.
- Chamber material/cooling method and wall temperature limit.

Physical coupling:

- Higher `Pc` raises mass flow at fixed throat area and generally raises heat
  flux and pump discharge pressure.
- Higher `epsilon` changes nozzle thrust coefficient, exit pressure, wall area,
  and separation risk.
- Higher contraction ratio increases chamber radius, usually giving the spray
  more radial room before wall contact, but it changes chamber volume and
  cooling area.
- Larger L-star increases residence time and chamber length, helping
  vaporization/mixing but adding mass, wall area, pressure drop, and cooling
  duty.

### Injector hydraulic variables

Ask or solve:

- Fuel and oxidizer injector dP fraction. Default 0.15-0.25 Pc is reasonable
  for screening, but the user should know this is also a stability and pump
  requirement.
- Stream assignment: fuel-radial/oxidizer-annular or oxidizer-radial/fuel-
  annular.
- Pintle diameter as a chamber-diameter fraction or explicit value.
- Slot/hole count, slot aspect ratio, radial-hole pattern, and slot length/L/D.
- Annular gap and annular area ratio.
- Target TMR and/or target local momentum ratio.
- Deflector angle.
- Skip/impingement distance.
- Injector recess.
- Face OD, face thickness, edge land, minimum feature, and concentricity
  tolerance.

Physical coupling:

- Increasing dP reduces required area and increases injection velocity. That
  usually improves atomization and chamber-length margin, but it increases pump
  pressure and can raise erosion/thermal risk.
- Increasing pintle diameter increases circumference, which can reduce slot
  blockage and web stress, but changes annular gap for the same area and moves
  the spray origin closer to the wall.
- More slots reduce area per slot and can improve circumferential distribution,
  but may violate minimum feature/web requirements and can alter block factor.
- Higher TMR generally opens the spray cone. That can improve transverse mixing
  but can hit the chamber wall too early.
- Larger skip distance changes breakup and spray formation. Lee et al. show it
  must be treated as a spray variable, not just a CAD offset.
- Radial fuel injection may support wall film cooling if intentionally biased,
  but it can also lower core mixture uniformity or create wall heat-flux peaks.

### Pump/feed variables

Ask or compute:

- Pressure-fed or pump-fed architecture.
- Available fuel and oxidizer pump/tank outlet pressure.
- Available pump mass-flow capacity for each stream.
- Pump speed/head curve or at least rated pressure rise and rated flow.
- Inlet tank pressure and temperature.
- NPSH margin, vapor pressure, and allowable cavitation margin.
- Line, valve, filter, and manifold loss budget.
- Whether fuel goes through regen before the injector.
- If regen fuel flow is not the full fuel flow, bypass fraction and mixing
  temperature before injection.

Physical coupling:

- Injector dP is not free; it must come from the pump/tank pressure budget.
- Regen pressure drop subtracts from the pressure available to the fuel
  injector if fuel cools the chamber first.
- Cryogenic inlet temperature affects density, viscosity, vapor pressure, and
  cavitation/NPSH.
- At throttle, pump curves and valve schedules determine whether O/F, TMR, and
  dP fractions stay within useful ranges.

### Validation/test variables

Ask before promoting a design above screening:

- Cold-flow working fluids and density/viscosity/surface-tension matching
  strategy.
- Diagnostics available: high-speed imaging, PIV/PDI/PDPA, mass-flow
  calibration, pressure transducers.
- Test pressure range and backpressure/chamber simulator.
- Hot-fire instrumentation: chamber pressure, high-frequency pressure,
  thermocouples, calorimetry/heat flux, thrust, flow rates.
- Acceptance criteria: c-star efficiency, pressure oscillation amplitude,
  mixture-ratio error, spray angle band, wall heat-flux margin, no tip damage.

SP-8089 is blunt here: pintle spray distribution requires cold-flow testing.
The CLI should keep that warning visible whenever it reports spray angle, SMD,
mixing, or efficiency from correlations.

## Proposed calculation flow

### Step 1: Resolve the engine operating point

Inputs:

- `F_target` or `Rt`
- `Pc`
- `Pa` or altitude/design pressure
- `epsilon` or matched expansion target
- propellant pair and `O/F`
- thermochemistry mode

Calculations:

```text
At = pi Rt^2
Cf = nozzle_thrust_coefficient(gamma, epsilon, Pa/Pc)
F = Cf Pc At
mdot_total = Pc At / cstar_eff
mdot_fuel = mdot_total / (1 + O/F)
mdot_ox = O/F * mdot_fuel
```

This is already implemented.

### Step 2: Resolve chamber geometry

Inputs:

- `L*`, contraction ratio, throat geometry; or an auto policy.

Calculations:

```text
V_chamber = L* At
Ac = contraction_ratio At
Rc = sqrt(Ac/pi)
Lc = exact root solve for chamber volume after shoulder/convergent/throat arcs
```

Already implemented. Missing improvement: auto-suggest L-star from propellant
pair, phase branch, atomization quality, and desired c-star efficiency.

### Step 3: Select pintle branch and target correlations

Branch by stream phase:

- Liquid-liquid.
- Gas-liquid.
- Gas-gas.
- Supercritical.
- Gelled.
- Flashing/two-phase: reject unless a flashing model is explicitly selected.

For each branch, select a correlation set:

- Liquid-liquid: Cheng et al. spray-angle prediction, Zhou and Shen TMR/LMR
  and block-factor behavior, Ninish/Radhakrishnan data for spray/mixing.
- Gas-liquid: Zhou et al. (2022), GO2/kerosene heat-flux/performance paper,
  local momentum ratio paper.
- Movable pintle: Son et al. design procedure plus Casiano/Hulka/Yang
  throttling review.
- Supercritical/LOX-methane: Son, Radhakrishnan, Lucchese, and Vasques/Haidn.

### Step 4: Solve injector hydraulic geometry

Inputs:

- dP fractions or pump pressure budget.
- Cd values or correlation/table by orifice L/D and edge geometry.
- stream assignment.
- target TMR/LMR or target spray angle.
- manufacturing limits.

Calculations:

```text
A_i = mdot_i / G_i
G_liquid = Cd sqrt(2 rho dP)
G_gas = compressible/choked mass flux
annulus gap from A_annulus and Dp
slot width/height/count from A_radial
TMR = radial momentum / axial momentum
LMR = local momentum ratio using block factor / local interaction area
```

Missing improvement: solve a constrained design problem instead of only
computing whatever TMR results from user dP values:

```text
minimize:
  |spray_angle - target|
  + |TMR - target|
  + penalties for wall hit, low We, bad L/D, pump exceedance, min features

variables:
  dP_f, dP_o, Dp, slot_count, slot_width/height, annulus_gap,
  stream assignment, skip distance

constraints:
  pump pressure, mdot closure, O/F closure, min feature/web, cavitation,
  chamber wall clearance, L*/residence, material thermal margin
```

### Step 5: Predict spray, mixing, and chamber interaction

Current:

- Generic vector spray angle.
- Hinze/Reitz/d-squared SMD and vaporization screen.
- Wall-hit distance from a simple cone.

Replace/augment with:

- Spray angle from selected pintle branch/correlation.
- TMR and LMR models including block factor.
- Annular-area and skip-distance correction where applicable.
- Recess-dependent heat-flux and recirculation flags.
- SMD correlations fitted to relevant branch data.
- A chamber-development score that reports:
  - breakup length,
  - vaporization length,
  - mixing length,
  - `Lc / L_required`,
  - expected c-star efficiency band, not a false single exact number.

### Step 6: Close pump/feed/cooling

For each stream:

```text
P_required_injector_manifold = Pc + dP_injector
P_required_pump_out =
    P_required_injector_manifold
  + line_losses
  + valve_losses
  + injector_manifold_losses
  + regen_losses_if_upstream
  + control_margin
```

For pump/tank feasibility:

```text
mdot_required <= mdot_pump_available(P_required)
NPSHa / NPSHr >= margin
P_tank + pump_head - all_losses >= Pc + dP_injector
```

RaoRocketSim already has pieces of this in regen pressure loss, injector
pressure checks, and manifold screens. The improvement is to make one explicit
feed-system closure table that both injector and cooling use.

### Step 7: Stability and throttle map

Current:

- dP fraction chug screen.
- closed-chamber acoustic frequencies.
- n-tau screening with atomization/vaporization lag.
- movable-sleeve throttle schedule preserving dP fraction/TMR by resizing area.

Improve with:

- Feed-system admittance model: tank/pump/line compliance, valve resistance,
  injector resistance, and chamber response.
- Combustion response model calibrated from cold-flow/hot-fire or literature.
- Stability map over throttle levels, not just one operating point.
- Mixture-ratio drift model for fixed geometry and pump/valve schedules.
- Explicit acoustic compatibility with chamber length/diameter and any baffles
  or acoustic cavities.

## Specific code/model upgrades

### 1. Add `PintleCorrelationSpec`

New config fields:

```python
spray_angle_model: "momentum_vector" | "cheng2017" | "zhou_shen_lmr" | "son2015" | "calibrated"
smd_model: "hinze_screen" | "radhakrishnan_lagrangian" | "liquid_liquid_fit" | "gas_liquid_fit"
combustion_efficiency_model: "vaporization_screen" | "austin2005_fit" | "sakaki_fit" | "calibrated"
validity_mode: "screening" | "cold_flow_calibrated" | "hot_fire_calibrated"
```

### 2. Promote local momentum ratio and block factor

Already computed or derivable:

```text
BF = N slot_width / (pi Dp)
TMR = total radial momentum / total axial momentum
LMR = local momentum ratio, branch/correlation-specific
```

Use LMR/BF in spray-angle gates instead of treating blockage as only a
manufacturing check.

### 3. Add injector geometry parameters that matter physically

CLI/API fields to add:

- `--pintle-skip-distance`
- `--pintle-recess`
- `--pintle-annular-area-ratio`
- `--pintle-block-factor-target`
- `--pintle-film-cooling-fraction`
- `--pintle-tip-shape`
- `--pintle-hole-diameter` and `--pintle-hole-count` for multi-hole pintles
- `--pintle-correlation-set`
- `--injector-validation-mode`

Some are already present under different names (`impingement_distance`,
`deflector_angle`, fixed slot dimensions), but the CLI should expose them in
the language used by the literature.

### 4. Add a solver for target user intent

The model should support:

```text
Given:
  target thrust, Pc, propellants, O/F, epsilon, chamber policy,
  pump pressure limits, manufacturing limits,
  desired throttle range and spray/wall policy

Solve:
  Rt if not supplied
  chamber Rc/Lc if auto chamber policy selected
  injector dP split
  pintle diameter
  slot/annulus geometry
  throttle sleeve schedule

Report:
  feasible/infeasible
  limiting constraint
  required pump outlet pressures
  required validation tests
```

### 5. Replace single predicted c-star with a range and evidence level

Current `predicted_cstar_efficiency` looks precise. It should become:

```text
cstar_efficiency_estimate
cstar_efficiency_low
cstar_efficiency_high
evidence_level = "screening correlation" | "cold-flow calibrated" | "hot-fire fit"
calibration_source
```

This better matches the literature: Austin, Sakaki, TRW, GO2/kerosene, and
LOX/methane studies show performance is empirical and branch-dependent.

### 6. Add pump/cooling/injector pressure ledger

One output table should show:

```text
fuel:
  Pc
  injector dP
  injector manifold dP
  regen dP
  line/valve dP
  required pump outlet pressure
  available pump outlet pressure
  margin

oxidizer:
  same, without regen unless oxidizer-cooled
```

This directly answers whether pump output and capacity can support the pintle
geometry the CLI selected.

## More user parameters needed

For a real pintle design workflow, the CLI should ask for the following when
`--injector pintle` is selected.

Minimum required:

- Propellant pair and mixture ratio.
- Target thrust or throat radius.
- Chamber pressure.
- Expansion ratio or design ambient pressure.
- Fuel and oxidizer feed phase and inlet temperature.
- Injector dP fraction or available pump/tank outlet pressure.
- Radial stream assignment or "auto".
- Design mode: screening, cold-flow-calibrated, or hot-fire-calibrated.
- Manufacturing minimum feature and material.

Strongly recommended:

- Desired throttle range.
- Allowed pump outlet pressure for each stream.
- Cooling path: fuel regen, oxidizer regen, ablative, film, or external.
- Fuel bypass fraction if coolant flow differs from total fuel flow.
- Chamber L-star policy: user value, propellant default, or auto from
  injector-quality target.
- Target spray behavior: wall-avoiding, wall-film cooling, or specified spray
  half-angle.
- Target c-star efficiency.
- Maximum allowable chamber wall heat flux or wall temperature.
- Injector recess and skip distance limits.
- Cold-flow validation data, if any.

Advanced:

- Pump head curves and NPSHr.
- Valve schedules for throttle.
- Line lengths/diameters/roughness and filter losses.
- Injector manifold port count/geometry.
- Allowed acoustic pressure amplitude.
- Required life/cycle count.
- Surface roughness/as-built tolerances.

## Geometry interdependencies

### Injector versus chamber

- Chamber radius controls spray-wall clearance. A wider chamber tolerates a
  larger spray angle; a narrow chamber may require lower TMR, smaller radial
  momentum, larger skip distance, or film-cooling intent.
- Chamber length controls vaporization and mixing residence. Better atomization
  can allow shorter L-star; poor atomization needs longer L-star or higher dP.
- Recess shifts recirculation and heat-flux patterns. A recess can protect or
  punish the face depending on spray/flame anchoring.
- Contraction ratio changes chamber diameter and volume, so it indirectly
  changes both spray clearance and cooling area.

### Injector versus nozzle

- Nozzle throat area and chamber pressure set total mass flow, which sets
  injector flow areas.
- Nozzle expansion ratio and contour do not directly set injector geometry, but
  they change thrust coefficient and therefore the throat size needed for a
  target thrust.
- Nonuniform pintle mixing can create hot streaks that affect throat erosion
  and nozzle heat flux. The current model should eventually pass an injector
  nonuniformity factor into the throat/nozzle thermal screen.

### Injector versus cooling

- Fuel-as-coolant outlet temperature changes fuel density, viscosity, vapor
  pressure, and therefore injector area and cavitation margin.
- Regen pressure drop consumes the same pressure budget the fuel injector
  needs.
- Pintle wall-directed spray or fuel film cooling can reduce/redistribute wall
  heat flux, but unbalanced spray can create local hot spots.
- Pintle tip thermal damage must be checked against material and cooling
  passage geometry; Kang et al. (2022) should inform this design gate.

### Injector versus pump/feed system

- Higher injector dP improves decoupling and atomization but demands higher
  pump outlet pressure.
- Pump pressure limits may force larger injector areas, lower velocities,
  lower Weber number, worse atomization, and longer chamber length.
- Pump flow capacity sets maximum thrust at a given Pc/O/F.
- NPSH and vapor pressure constrain inlet temperature and tank pressure,
  especially for LOX, methane, hydrogen, N2O4/NTO, and nitrous.

### Injector versus throttle

- Fixed injector areas make dP scale roughly with `mdot^2`; deep throttle can
  collapse atomization and feed decoupling.
- A movable sleeve can preserve dP fraction and TMR better, but requires stroke
  schedule, minimum open area, leakage/shutoff, actuator limits, and transient
  mixture-ratio checks.
- Pump/valve schedules determine whether the hydraulic schedule is physically
  reachable.

## Recommended implementation order

1. Add the missing CLI questions for `--injector pintle` in
   `scripts/run_nozzle.py` interactive mode and expose the same fields in
   `main.py` batch-v2.
2. Add a feed-pressure ledger that combines pump/tank pressure, regen loss,
   line/valve/manifold loss, and injector dP for each stream.
3. Add correlation selection and validity metadata for spray angle:
   keep the current vector model as `momentum_vector_screen`, then add
   Cheng 2017 and Zhou/Shen LMR.
4. Add block-factor and local-momentum-ratio calculations to the result JSON,
   figures, and gates.
5. Add skip-distance, annular-area-ratio, recess, and film-cooling intent as
   first-class geometry/design variables.
6. Replace single SMD/c-star surrogates with branch-specific estimates and
   uncertainty bands.
7. Add a constrained optimizer that solves injector dP split and geometry to
   match target spray/TMR/efficiency subject to pump and manufacturing limits.
8. Build validation fixtures from the local papers: reproduce a small set of
   reported spray angles, SMD ranges, and performance points as regression
   tests.

## Bottom line

The repository has a good integrated scaffold. It should not be rewritten from
scratch. The next step is to turn its pintle screens into correlation-backed
models with explicit validity domains and to make the CLI ask for pump/feed,
throttle, chamber, and validation intent. Once that is done, the tool can
generate injector geometry that is not merely dimensionally consistent, but
actually constrained by the same literature that governs pintle design,
testing, chamber sizing, nozzle sizing, regenerative cooling, and pump output.
