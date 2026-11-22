# Design plan: valves, lines, brackets and gimbal

These four subsystems are the last entries in the mass ledger's `excludes` list,
and they are why `total_engine_package_mass_kg` still reports `unavailable`. An
engine is not a thrust chamber plus an injector plus pumps — it is those plus the
plumbing that connects them, the structure that holds them and the mechanism that
points them.

This document specifies how each will be modelled: the governing relations, the
interfaces, what is buildable from the current corpus, and what is blocked on the
literature requested in `LITERATURE_REQUESTS.md`.

A note on the honesty contract that applies throughout: every component gets a
`MassItem` with a `status` of `geometry_resolved` or `screening_sized`, or it is
emitted with `mass_kg = None` and a reason. None of these subsystems may be
allowed to default to zero, which is the failure mode the whole ledger exists to
prevent.

---

## 1. Valves

### What an engine actually needs

For an electric-pump-fed bipropellant engine the minimum valve set is:

| valve | duty | actuation |
|---|---|---|
| main fuel valve (MFV) | isolate + open on start, close on shutdown | pneumatic or electromechanical |
| main oxidiser valve (MOV) | same, usually sequenced after MFV | same |
| igniter fuel/ox valves | small, fast, often solenoid | solenoid |
| tank pressurisation regulator | maintain NPSH at the pump inlet | self-regulating |
| relief / burst disk | protect the tanks | passive |
| fill/drain and vent | ground ops | manual or solenoid |

### Model

**Flow sizing** is the part that needs no new literature — it is incompressible
orifice flow, which the injector module already does:

```
Cv  = Q sqrt(SG / Δp)          (US customary Cv, the industry currency)
Δp  = (ṁ / (Cd A))² / (2 ρ)     (SI form the repo already uses)
```

The valve's flow area then sets its port diameter, which sets its body size.
The chain is `ṁ, ρ, allowable Δp → A → D_port → body envelope → mass`.

The **allowable Δp** is a real design coupling, not a free parameter: every Pa
spent in the valve is a Pa the pump must add, so valve Δp feeds
`dp_rise_fuel` / `dp_rise_ox` in `mdo/engine.py` exactly as the regen jacket and
injector Δp already do. This is the single most important reason to model valves
at all — right now `mission.line_dp_allowance` is a lumped constant standing in
for the entire valve-plus-line budget.

**Body mass** follows the same principle as everything else in the ledger:
integrate the geometry. A poppet or ball valve body is a pressure vessel with a
bore, so the wall follows the same thin-shell hoop relation used for the jacket,
`t = FoS·p·r/σ`, plus the seat, stem and actuator interface.

**Actuator mass and power** is where SP-8090 is needed. Actuation force is
`F = p·A_seat + F_spring + F_friction`, and stroke follows the port geometry, but
the force-to-mass and force-to-power relations for flight actuators are exactly
what that monograph tabulates. Until it is in the corpus this is `unavailable`,
not a guess.

### Interfaces

```python
ValveSpec(role, kind, mdot, density, allowable_dp_fraction,
          actuation, response_time_max, body_material, ...)
ValveSizing(port_diameter, flow_area, cv, pressure_drop, body_mass,
            actuator_mass, actuator_power, response_time, gates)
```

`ValveSizing.pressure_drop` feeds the pump duty; `body_mass + actuator_mass`
feeds the ledger under a new `feed_system` subsystem.

### Blocked on

SP-8094 (components), SP-8097 (assemblies), SP-8090 (actuators), SP-8080
(regulators/relief). **Buildable now:** flow sizing and the Δp coupling into the
pump duty, which is worth doing first because it removes a lumped constant from
the optimisation.

---

## 2. Lines

### Model

Line sizing is a closed problem given a velocity limit and a Δp budget:

```
A = ṁ / (ρ v)                       velocity limit sets the bore
Δp = f (L/D) ρ v²/2  +  Σ K ρ v²/2   Darcy-Weisbach + minor losses
t  = FoS · p · r / σ                 thin-wall hoop, same relation as the jacket
m  = ρ_metal · 2π (r + t/2) · t · L  Pappus, same as every other shell here
```

Every one of those is already implemented somewhere in the repository —
`thermofluids.py` has Darcy friction, `mass_ledger` has the shell integral. The
work is assembling them into a routed network, not inventing physics.

**Velocity limits** are the one thing needing a source. The convention is roughly
5–10 m/s for pump-inlet (suction) lines and 10–20 m/s for discharge, tightened on
the oxidiser side for LOX-compatibility reasons. SP-8123 is the citable source;
`pumps.py` already has a `liquid_velocity_recommendation` field that currently
carries a screening value.

**Routing** is the genuinely hard part and I do not propose to solve it. The
plan is a *declared* route — the user (or an architecture enumerator) specifies
tank → pump → jacket → injector segment lengths and bend counts, and the model
sizes and prices them. Automatic 3-D routing is a separate problem and is not
needed for a mass ledger.

**Bellows and flex joints** matter only where a line crosses the gimbal plane,
which ties this to §4.

### Blocked on

SP-8123 (lines/bellows/hoses/filters) for velocity limits, bellows spring rate
and cycle life; SP-8119 for fittings and joints. **Buildable now:** the whole
sizing chain with the velocity limit as a documented user input.

---

## 3. Brackets and mounts

### The key insight

Brackets are **not sized by strength**. A bracket that carries a 5 kg pump under
6 g is trivially strong at almost any sensible thickness. Brackets are sized by
**stiffness** — the first natural frequency must sit above the excitation
environment, or the component fatigues in random vibration.

That means the governing relation is a frequency one:

```
f_1 = (1/2π) sqrt(k_eff / m_supported)     target f_1 >= f_min
```

with `f_min` set by the acoustic and vibration environment, and `k_eff` from the
bracket's cross-section. Sizing to a strength margin and declaring victory would
produce brackets that pass every check in this repository and fail on a shaker.

### Model

1. **Mass inventory** — each supported component (pump, motor, inverter,
   battery, valve, line run) contributes a point mass at its centroid, which the
   ledger already knows because it knows the geometry.
2. **Load cases** — quasi-static `g` loads in three axes, plus the thrust
   transient from SP-8030, plus a random-vibration PSD from SP-8072.
3. **Sizing** — for a declared bracket topology (a plate, a strut, a ring),
   solve for the section that meets both the stress margin and the frequency
   floor, take the governing one.
4. **Mass** — geometry integral, as everywhere else.

### Honest limitation

A real bracket is a shaped part found by topology optimisation and verified by
FEA. What this model can defensibly produce is a **parametric bracket family**
(plate, L-bracket, tripod strut) sized to the governing constraint — enough for a
mass budget and a first CAD envelope, explicitly labelled `screening_sized`.

### Blocked on

SP-8072 (acoustic loads) and SP-8030 (thrust transients) for the environment;
SP-8055 (POGO) for the feed-line stiffness coupling. **Buildable now:** the
quasi-static strength path and the mass inventory, with the frequency floor as a
user input.

---

## 4. Gimbal

### Model

Three coupled pieces:

**(a) Gimbal bearing.** Carries the full thrust as a compressive load through a
joint that must rotate ±α. Sizing is Hertzian contact stress and bearing pressure
`p = F/A_projected` against the allowable for the bearing material, plus friction
torque `T_f = μ F r_eff` which sets the actuator's static requirement.

**(b) Actuators.** Two linear actuators at 90° give pitch and yaw. The kinematics
— actuator force to gimbal torque — are exactly the transformation matrix in MSFC
TB-03. The torque budget is

```
T_required = T_friction + T_line + T_inertia + T_aero
```

where `T_line` is the restoring torque of the propellant lines crossing the
gimbal plane (bellows spring rate × deflection), and is often the **dominant**
term for a small engine. This is why the gimbal cannot be modelled independently
of the lines: a stiff line makes a big actuator.

`T_inertia = I_engine · α̈` needs the engine's mass moment of inertia — which the
mass ledger can now supply, because it knows every component's mass and position.
That is a concrete payoff from the ledger work.

**(c) Structure.** The gimbal block transfers thrust into the vehicle. Same
strength-plus-stiffness treatment as brackets.

### Interfaces

```python
GimbalSpec(deflection_max_deg, slew_rate_deg_s, bearing_type,
           actuator_count, line_crossings, ...)
GimbalSizing(bearing_mass, actuator_mass, actuator_force, actuator_power,
             block_mass, friction_torque, line_torque, inertia_torque, gates)
```

### Blocked on

SP-8090 (actuators), SP-8123 (bellows spring rate — the dominant torque term),
MSFC TB-03 (kinematics), SP-8114 (flexible-joint treatment). **Buildable now:**
the inertia torque, since the ledger already knows the mass distribution.

---

## Sequencing

```
Step 1  Valve + line FLOW sizing, wired into the pump duty         no new lit
        -> removes mission.line_dp_allowance as a lumped constant
Step 2  Engine mass-moment-of-inertia from the existing ledger      no new lit
        -> feeds gimbal inertia torque and vehicle-level analysis
Step 3  Line and valve BODY mass (thin-shell hoop + Pappus)         no new lit
Step 4  Bracket quasi-static strength path + mass inventory         no new lit
--- literature checkpoint: items 1-6 and 19-21 ---
Step 5  Valve actuator mass/power (SP-8090)
Step 6  Bellows spring rate and line torque (SP-8123)
Step 7  Gimbal bearing, actuators, kinematics (SP-8090, TB-03)
Step 8  Bracket frequency floor (SP-8072, SP-8030, SP-8055)
Step 9  total_engine_package_mass_kg finally closes
```

Steps 1–4 are worth doing before the literature arrives: they are the parts with
no sourcing gap, and step 1 in particular removes a lumped constant from the
optimisation, which improves results immediately.

## What closing this buys

Once steps 1–9 are done, `total_engine_package_mass_kg` becomes available for
the first time, which means:

- the MDO can optimise against a **real dry mass** rather than a feed-system
  proxy;
- `dry_mass_max` becomes a usable requirement in the R3 requirements layer;
- the engine mass moment of inertia exists, so gimbal and vehicle-level analysis
  become possible;
- the parts package is complete enough that "requirements in, parts out"
  describes something a person could actually build.
