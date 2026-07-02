# Electric Pump Model Basis

This model is a first-pass sizing screen. It separates quantities the engine
and feed ledger demand from quantities that must be selected from pump, motor,
inverter, and battery hardware.

## Solved From The Engine And Feed Ledger

For each propellant stream, the injector/feed ledger provides mass flow, fluid
density, required downstream pressure, tank/inlet pressure when known, and NPSH
margin when known. The pump model then uses:

```text
Q = mdot / rho
deltaP_pump = P_required_outlet - P_tank_or_inlet
H = deltaP_pump / (rho g0)
P_hyd = deltaP_pump Q
P_shaft = P_hyd / eta_pump
omega = 2 pi rpm / 60
torque = P_shaft / omega
P_electric = P_shaft / (eta_motor eta_inverter)
I_bus = P_electric / V_bus
E_pack = P_electric burn_time / eta_battery
```

So the pump is not sized from thrust alone. Thrust only enters upstream by
setting total propellant flow through Isp or c-star, then the injector/feed
pressure budget sets pressure rise.

## Hydraulic Meanline V1

When tank/inlet pressure is known, the electric-pump path now builds an
explicit centrifugal meanline for each stream. The exported `pump.json` keeps
the old summary fields and adds `hydraulic_meanline` and `performance_curve`.

The meanline keeps the first-pass SP-8109 style geometry anchors:

```text
psi = g H_stage / U2^2
phi = Cm2 / U2
Ns = omega sqrt(Q) / (g H_stage)^0.75
Ds = D2 (g H_stage)^0.25 / sqrt(Q)
```

It then adds the missing hydraulic bridge from geometry to efficiency:

- inlet and outlet velocity triangles, assuming no inlet prewhirl;
- a Stodola-style slip screen from blade count and outlet blade angle;
- Euler head margin from `U2 * Cu2 / g`;
- Reynolds-sensitive passage friction using the feed-ledger viscosity;
- blade-loading, incidence, disk-friction, leakage, and recirculation loss
  buckets;
- auto pump efficiency from `H_stage / (H_stage + loss_head)` unless the user
  supplies `--pump-efficiency-fuel` or `--pump-efficiency-oxidizer`.

The performance curve is a fixed-speed screening map generated around the
design point. It exports `H(Q)`, `eta(Q)`, pressure rise, hydraulic power, and
shaft power for several flow ratios. It is intended for throttle/system-curve
trades and should be replaced by measured or vendor pump maps as soon as real
hardware exists.

## Selected, Solved, Or Screened Assumptions

Some pump parameters can be inferred from the solved duty; others remain
hardware assumptions until real component data exist.

- Pump efficiency is auto-estimated from the stream flow/head duty unless a
  user-supplied efficiency is passed. The automatic meanline result is capped
  conservatively for screening; values near the cap emit a warning and should
  be replaced with a measured/vendor pump curve before hardware decisions.
- Pump rpm is auto-selected from target nondimensional specific speed plus
  impeller diameter and outlet-width bounds unless a user-supplied rpm is
  passed.
- The default electric architecture is one shared DC pack/bus feeding all pump
  drives. Bus voltage is auto-selected from total electric power and current
  limits unless a user-supplied motor or battery voltage is passed. Different
  motor and battery voltages require explicit converter/separate-pack modeling;
  the screening path flags that mismatch instead of silently assuming it.
- Head coefficient, flow coefficient, maximum head per stage, and impeller
  material tip-speed limit are screening assumptions.
- Motor speed range / maximum rpm is still a hardware constraint.
- Motor/inverter efficiency, bus voltage, current limit, torque limit, thermal
  limit, and power density.
- Battery voltage, discharge efficiency, power density, energy density, pulse
  current limit, structural/package margin, and allowable vehicle mass fraction.
- NPSH requirement, vapor pressure/subcooling state, inlet line losses, and
  tank/inlet pressure.

The defaults are screening values, not constants. Replace them with pump
curves, motor maps, inverter limits, and battery pulse/thermal data as soon as
hardware exists.

## Current Status And Next Work

Implemented now:

- feed-ledger coupling from injector demand to per-stream pump pressure rise,
  head, volumetric flow, hydraulic power, and NPSH bookkeeping;
- electric drive, inverter, shared DC bus, battery energy/power/current/heat,
  and preliminary feasibility gates;
- centrifugal impeller geometry from head/flow coefficients, selected or
  auto-solved RPM, specific speed, outlet width, and tip-speed screens;
- inducer diameter, hub ratio, solidity, suction-specific-speed, and NPSH
  screens;
- diffuser/volute selection, velocity triangle, meanline loss buckets, and a
  screening pump curve in `pump.json`;
- explicit pump architecture classification per stream.  The classifier keeps
  the current radial-centrifugal meanline when the duty fits, but flags
  mixed-flow, axial, staged centrifugal, positive-displacement, inducer-assisted,
  and off-the-shelf electric-feed candidates from specific speed, head per
  stage, flow scale, and NPSH state;
- normalized hardware BOM export.  `pump.json` carries `hardware_bom`, and the
  CLI also writes `pump_bom.json`, with rows for inducer, impeller,
  diffuser/volute, shaft/coupling, motor, inverter/controller, battery, casing,
  bearings, seals, inlet/outlet ports, and instrumentation placeholders;
- editable reference geometry export.  Each pump line carries
  `reference_geometry` with meridional stations, impeller disk, blade envelope,
  inducer helix, diffuser vane ring, volute scroll, shaft datum, and ports; the
  CLI writes this as `pump_reference_geometry.json`;
- pump CAD/reference package export.  `--pump-cad auto`/`parts` writes a
  `pump/` folder with `pump_parameters.json`, `pump_dimensions.csv`,
  per-component impeller, inducer, diffuser/volute, motor, inverter, and battery
  package reference solids, and a `pump_reference_assembly` in STL, faceted
  STEP, or both;
- fixed-speed pump-curve versus feed-system curve coupling over throttle/flow
  ratios.  The screen compares the generated pump curve against a quadratic
  injector/line/regen loss curve and a linearly scaled chamber-pressure demand,
  reporting supported throttle range and margin;
- basic thermal and stress ledgers: motor/controller heat, pump loss heat,
  estimated propellant temperature rise, impeller and inducer rotating hoop
  stress, blade-root bending, shaft torsion, casing hoop stress, bearing DN,
  seal face speed, seal heat, and margin gates.

Still explicitly preliminary:

- named pump/motor/battery technology records and measured pump maps are not
  yet attached.  The current defaults are versioned assumptions in
  `PumpSizingSpec`, exported in `pump.json`, and should be replaced by vendor
  curves, motor maps, battery pulse/thermal data, bearing catalogs, seal
  compatibility data, and material allowables before hardware decisions.
- propellant heating is reported as a temperature-rise screen only; it does not
  yet re-solve fluid properties or vapor pressure and feed the changed state
  back into the inducer/NPSH calculation.

## Literature Anchors

- NASA SP-8109, *Liquid Rocket Engine Centrifugal Flow Turbopumps*
  (`propulsion_texts/fuel_pump_design/19740020848.pdf`): pump design begins
  from required flowrate, head rise, and inlet pressure. It motivates the
  head-coefficient relation `psi = gH/U2^2`, the specific speed/specific
  diameter screens, tip-speed/staging warnings, and diffuser/volute selection.
- NASA SP-8052, *Liquid Rocket Engine Turbopump Inducers*
  (`propulsion_texts/fuel_pump_design/19710025474.pdf`): inducer and suction
  performance are driven by inlet pressure, vapor pressure, NPSH, flow, speed,
  inlet diameter, and suction specific speed.
- Huzel and Huang SP-125 and Sutton/Biblarz feed-system practice, summarized in
  `docs/PINTLE_DESIGN_EVALUATION.md`: pump discharge pressure must cover chamber
  pressure, injector drop, cooling-jacket drop, line/valve losses, dynamic
  allowances, and margin.
- Lee et al., *Performance Analysis and Mass Estimation of a Small-Sized Liquid
  Rocket Engine with Electric-Pump Cycle*, International Journal of Aeronautical
  and Space Sciences 22:94-107, 2021
  (`propulsion_texts/fuel_pump_design/s42405-020-00325-z.pdf`): useful as a
  comparison case for electric-pump mass closure. It is not treated as a
  universal source of pump rpm or pump efficiency.
- Spiller, Stabile, and Lentini, *Design and Testing of a Demonstrator Electric
  Pump Feed System for Liquid Propellant Rocket Engines*
  (`propulsion_texts/fuel_pump_design/BF03404670.pdf`): small off-the-shelf pump
  tests are used as a caution that very small pump duty may require custom pump
  design and can have poor efficiency if the pump type is mismatched.

## Dependency Chain

```text
engine duty
  -> mdot_fuel, mdot_oxidizer, Pc, injector/cooling/line losses
  -> Q and deltaP per stream
  -> pump head and hydraulic power
  -> shaft power through supplied or auto-estimated pump efficiency
  -> auto-selected or supplied rpm from specific-speed / geometry constraints
  -> torque through rpm
  -> impeller diameter/width through head and flow coefficients
  -> inducer/NPSH screen through inlet pressure, vapor pressure, rpm, and Q
  -> electric power/heat through motor and inverter efficiency
  -> shared DC bus voltage/current from total power and current limits
  -> battery mass/current/heat through burn time, voltage, energy density,
     power density, and discharge efficiency
  -> architecture classification, normalized hardware BOM, editable reference
     geometry, pump/system throttle margin, thermal ledger, and stress screens
```

## Blade and Channel Geometry Fidelity (2026-07-02, pump CAD plan Phase 2)

The reference geometry consumed by the CAD chain is now solved, not assumed
(running record: `docs/PUMP_CAD_IMPLEMENTATION_PLAN.md` STATUS head):

- **Impeller blade count** comes from the digitized NASA SP-8109 fig. 16
  minimum-blade-number chart (`pumps.sp8109_min_blade_count`; psi vs phi2,
  zero prewhirl, shrouded, delta = 0.65, read-off ~ +/-0.02 in psi), snapped
  to a multiple of the inducer blade count per SP-8052 sec. 3.1.14.  An
  explicit `PumpSizingSpec.blade_count` overrides; the basis string is
  exported with the geometry.
- **Impeller blade camber** is the log-spiral family
  d(theta)/dr = 1/(r tan beta(r)) with beta linear from the solved
  velocity-triangle beta1 to beta2 (`pumps.impeller_blade_camber`); CAD
  sweeps it, physics owns it.
- **Inducer blade angles** follow NASA SP-8052: the inlet tip blade angle
  carries the tip flow angle atan(phi_tip) plus the incidence from the
  alpha/beta design ratio (sec. 3.1.9; 0.35 thin .. 0.50 thick, 0.425
  preferred, the cavitation design variable); the blades are a constant-lead
  helix r tan(beta) = const (sec. 3.1.10) whose lead sets the exported pitch;
  wrap comes from cascade solidity 2.5 (sec. 3.1.15) with the developed-chord
  cos(beta) factor; leading edges take the low end of the J-2/F-1 0.005-0.010
  in. edge practice (sec. 2.1.6).  Cross-checked against the Hong et al. 2012
  rocket turbopump inducer (10.4 deg tip blade angle, 3 blades, solidity 2.6).
- **Meridional channel**: quarter-ellipse hub/shroud curves honoring D1, D2,
  b2 and the inducer hub, with exact eye-annulus and exit areas; the
  discharge/inlet meridional-velocity ratio is screened against SP-8109
  sec. 2.3.1.2 (cm2 = 1 to 1.5 x inlet).
- **Thrust balance hooks**: hub-side wear ring at the eye diameter
  (SP-8109 sec. 3.5.2.1 recommends wear rings over balance ribs), shaft seal
  land with the solved face speed vs the screening limit, and balance holes
  sized by the sec. 3.5.2.1 rule (flow area ~ 4 x seal-clearance area) once a
  wear-ring radial clearance is supplied
  (`PumpSizingSpec.wear_ring_radial_clearance`).
- **Benchmarks pinned as tests** (`tests/test_pumps.py`): Lee et al. 2021
  500 N / 20 bar / 600 s case closes the drive/battery mass chain (motor
  451.2 g, battery 985.6 g, power-limited); solved specific speeds sit inside
  the SP-8109 flight-proven 450-2100 (US units) envelope.

CAD remains labeled reference geometry: blade-to-blade CFD, rotordynamics
beyond the DN screen, bearing/seal selection, motor electromagnetic design,
and measured pump maps stay out of scope
(`qualification_status: reference_geometry_not_hardware_qualified`).
