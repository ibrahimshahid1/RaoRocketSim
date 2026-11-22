# Injector/Chamber Interface Screen

This note documents the preliminary interface ledger implemented in
`raosim.interface`.  It is a CAD/readiness and first-pass sizing screen for the
injector body, faceplate/flange, and chamber shell interface.  It is not a
hardware qualification method.

## Scope

The chamber/nozzle wall export remains the hot-gas liner or shell.  The pintle
package remains injector reference geometry.  The interface screen connects the
two by reporting the pressure load, faceplate bending thickness, bolt clamp
load, and bolt-pattern lands.

## Equations

Projected pressure separating load:

```text
F_sep = Pc * A_projected = Pc * pi * Rc^2
```

This is the basic pressure-vessel free-body load on the injector closure.

Thin chamber-wall pressure screen, used only when no sized regenerative wall
profile is available:

```text
sigma_hoop = Pc * Rc / t_wall
```

This is the standard thin-cylinder pressure-vessel relation.  The regenerative
wall model remains the authoritative path for station-wise thermal/structural
liner sizing.

When `run_nozzle.py` has a resolved `RegenWallProfile` from `--size-wall`, the
interface ledger replaces this scalar gate with `composite_regen_wall_hoop` at
the injector-face chamber station.  It does not let the throat/nozzle station
govern an injector/chamber interface check.  That screen keeps the local SP-125
liner channel-roof stress separate from the absolute coolant-pressure hoop
carried by the outer jacket.  Any chamber-over-coolant residual pressure is then
shared by the smeared bonded liner/rib plus jacket section:

```text
t_cu,eq = t_hot + land_fraction * channel_height
N_theta = max(Pc - p_coolant, 0) * r
eps = (N_theta + sum(E_i * t_i * alpha_i * DeltaT_i)) / sum(E_i * t_i)
sigma_i = E_i * (eps - alpha_i * DeltaT_i)
```

The reported copper demand conservatively adds the local liner stress to the
absolute residual/common-strain membrane stress.  The reported jacket demand
adds the absolute coolant-pressure hoop stress to that residual/common-strain
stress, so the jacket pressure load is not counted once as coolant hoop and
again as full chamber-pressure membrane.

Injector faceplate bending screen:

```text
sigma_face ~= 0.75 * Pc * a^2 / t_face^2
t_face >= a * sqrt(0.75 * Pc / sigma_allow)
```

This is the clamped circular-plate, uniform-pressure screen from classical plate
theory/Roark-style formulas.  The unsupported radius `a` is conservatively taken
as the chamber bore radius.  Center deflection is reported when `E` and `nu` are
available, but not gated because allowable seal deflection is gasket-specific.

Bolt separation screen:

```text
F_clamp,total >= K_sep * F_sep
F_bolt = F_clamp,total / N
sigma_bolt = F_bolt / A_tensile
```

`K_sep` defaults to 1.5.  This is a Shigley-style preliminary bolted-joint
separation check.  If an actual bolt diameter is missing, the code infers
`d_bolt = 0.9 * d_hole` and records that assumption.

Bolt pattern checks:

```text
inner_edge >= 1.5 * d_hole
outer_edge >= 1.5 * d_hole
pitch      >= 3.0 * d_hole
```

These are machine-design manufacturability heuristics, not final flange rules.

## Literature Basis

- Huzel and Huang, NASA SP-125: thrust-chamber structural context, regenerative
  liner/jacket design, and the station-wise wall model used elsewhere in the
  repo.
- Sutton and Biblarz, *Rocket Propulsion Elements*: chamber/injector pressure
  and thrust-chamber design context.
- Roark/classical plate theory: clamped circular plate under uniform pressure.
- Shigley-style machine design: bolted-joint preload/separation and common
  bolt-pattern edge-distance/pitch heuristics.

## Explicit Limits

The screen does not model gasket seating, flange rotation, bolt preload scatter,
thread engagement, local boss stress, weld/braze details, thermal gradients
through the injector body, fatigue, creep, proof factors, or nonlinear contact.
Final hardware needs detailed joint design, FEA, inspection planning, and test
evidence.
