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
```
