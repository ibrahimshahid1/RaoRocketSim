# Son continuous movable-pintle model

## Status and scope

LREKit implements the preliminary hydraulic and static-actuation model for the
continuous radial-gap, center-rod pintle described by Son et al. (2017). The
local source is
[`propulsion_texts/pintle_injector/son2017.pdf`](../propulsion_texts/pintle_injector/son2017.pdf).
The implementation is `son2017_continuous_radial_gap` in
[`raosim/movable_pintle.py`](../raosim/movable_pintle.py), dispatched by
[`raosim/injector.py`](../raosim/injector.py).

This is a software-verified preliminary model. It does not establish a
transferable discharge coefficient, a measured spray distribution, actuator
dynamics, machining readiness, cold-flow acceptance, hot-fire acceptance, or
hardware qualification.

## Metering geometry

Define:

- `D_post`: post outside diameter;
- `t_post`: post lip thickness;
- `D_cg`: fixed center-gap outside diameter;
- `D_pr`: moving pintle-rod diameter;
- `L_open`: mechanical axial opening;
- `theta_pt`: pintle-tip angle;
- `r_f = D_post/2 - t_post`.

Son Eq. 1 gives the minimum post/tip area:

```text
A_tip = pi/sin(theta_pt) *
        [r_f^2 - (r_f - L_open*sin(theta_pt)*cos(theta_pt))^2]
```

The code evaluates the algebraically equivalent stable form:

```text
A_tip = pi * [2*r_f*L_open*cos(theta_pt)
              - L_open^2*sin(theta_pt)*cos(theta_pt)^2]
```

Its zero-angle limit is `A_tip = 2*pi*r_f*L_open`. The fixed center-gap
area and controlling area are

```text
A_cg  = pi/4 * (D_cg^2 - D_pr^2)
A_eff = min(A_tip, A_cg)
L_min = L_open*cos(theta_pt)
```

The solver inverts `mdot = Cd*A_eff*sqrt(2*rho*dP)` on the monotone,
tip-controlled branch. It computes the transition opening from
`A_tip = A_cg`; the physical open stop must remain below that transition.
When no explicit open stop is supplied, the default stop is derived at
`A_tip/A_cg = 0.95`. Travel beyond the transition is rejected because it has
lost minimum-area authority.

The equation implementation is regression-tested against the published Son
transition-opening table for 0, 20, and 40 degree tips. That is an equation
benchmark, not validation of a new injector configuration.

## Discharge-coefficient evidence contract

Son Eq. 3 defines how discharge coefficient is measured; it does not provide a
universal `Cd` curve that can be transferred to different geometry, fluids, or
operating states. The movable branch therefore accepts a piecewise-linear map
of `(L_open/L_max, Cd)`, spanning exactly 0 to 1, only as
configuration-controlled calibration data.

A movable design passes the `movable_cd_calibration` gate only when all of the
following are true:

- a source label, 64-character lowercase artifact SHA-256, and matching
  `raosim.son2017_movable_geometry.v1` fingerprint are present;
- the calibrated fluid identity matches the resolved radial liquid;
- solved Reynolds number is inside the declared calibration range;
- solved pressure drop is inside its declared range;
- resolved liquid temperature is inside its declared range;
- solved cavitation number is inside its declared range; and
- the radial feed is a resolved liquid state.

Omitting the map leaves an explicitly labelled `constant_uncalibrated`
fallback. It is useful for a rough hydraulic screen but fails the movable
calibration gate. Broad placeholder ranges or a fabricated hash do not create
physical evidence.

## Fixed-hardware throttle semantics

The center rod changes only the continuous radial metering area. It does not
change the axial annulus. `throttle_map()` therefore:

1. sizes the full-power hardware once;
2. holds post, rod, center gap, annulus gap, and open stop fixed;
3. solves `L_open` and the calibrated radial `Cd` at each requested point;
4. holds the radial pressure-drop fraction at its full-power value; and
5. solves the pressure-drop command for a separate upstream controller on the
   fixed axial-annulus stream so requested mass flow and O/F close.

The controller role is the stream opposite `--pintle-radial-stream`. Each
point reports physical `L_open/L_max`, radial effective-area fraction, axial
controller `dP/Pc`, delivered mass flow, O/F, TMR, and feasibility separately.
Effective annulus-area command is never relabelled as physical stroke.

The controller solve is bounded by
`--movable-axial-controller-dp-fraction-min/max`. Reachability inside that
hydraulic envelope does not size a valve, controller, drive, transient
response, feed-system compliance, or stability margin.

## Position, leakage, actuator, and stem gates

Position authority uses the declared sum of position tolerance, feedback
resolution, and backlash relative to the solved opening. Closed-stop leakage
area is reported relative to the open effective area. Position metrology,
seat/leakage bounds, and the actuator/material input set each require their own
source and configuration-artifact SHA-256; none is inferred from nominal CAD.

The static absolute-load ledger is

```text
F_pressure = dP * A_unbalanced
F_momentum = mdot * velocity * axial_momentum_fraction
F_inertia  = moving_mass * maximum_acceleration
F_required = safety_factor *
             (F_pressure + F_momentum + preload + friction + F_inertia)
margin_force = actuator_capacity / F_required
stress_stem  = F_required / (pi*D_stem^2/4)
```

Pressure balance is never assumed: the net unbalanced projected area must be
declared explicitly, including an intentional value of zero. Seal/guide
friction, moving mass, acceleration, actuator capacity, stem diameter, and a
temperature-appropriate stem allowable are also caller inputs. This is a
static conservative sum, not a dynamic actuator or structural analysis.

## Mechanical opening is not sheet thickness

`L_open`, the minimum normal gap `L_min`, the internal controlling area
`A_eff`, the external 360-degree geometric opening area, and liquid-sheet
thickness are distinct quantities. The hydraulic report includes the
continuity-equivalent value

```text
delta_eq = A_eff / (2*pi*R_exit)
```

only as a diagnostic. It is explicitly not VOF-resolved or measured sheet
truth and is not admitted as a parcel diameter.

The VOF/measured spray handoff accepts an independent `delta_sheet` only when
it is bound to a method (`vof` or `measured`), source, artifact SHA-256, exact
Son-geometry fingerprint, fluid, and validity ranges for mechanical opening,
radial pressure drop, and radial mass flow. The solved point must match that
geometry and lie within all ranges. Without that contract, hydraulics remain
available but the geometry-aware primary-sheet to
Lagrangian handoff is blocked. The current OpenFOAM external-gap wedge is a
screen and not evidence for the full internal movable passage until the exact
case is run, converged, and configuration-controlled.

## CLI configuration

Select the architecture and continuous radial gap together:

```bash
lrekit --injector pintle \
  --injector-architecture son_continuous_movable \
  --injector-sizing auto \
  --pintle-radial-exit continuous_radial_gap \
  --pintle-radial-stream fuel \
  --pintle-deflector-angle 20 \
  --movable-post-diameter 0.020 \
  --movable-post-thickness 0.001 \
  --movable-center-gap-diameter 0.012 \
  --movable-pintle-rod-diameter 0.008
```

For a physically gated design, add configuration-controlled calibration,
metrology, leakage, and actuator inputs. This example shows the complete flag
shape; its numbers are illustrative and must be replaced by evidence for the
actual hardware and operating envelope:

```bash
CD_SHA256="$(shasum -a 256 test-data/pintle_cd.csv | awk '{print $1}')"
SHEET_SHA256="$(shasum -a 256 test-data/pintle_sheet_vof.json | awk '{print $1}')"
METROLOGY_SHA256="$(shasum -a 256 test-data/pintle_metrology.json | awk '{print $1}')"
LEAKAGE_SHA256="$(shasum -a 256 test-data/pintle_leakage.json | awk '{print $1}')"
ACTUATOR_SHA256="$(shasum -a 256 test-data/pintle_actuator_material.json | awk '{print $1}')"
GEOMETRY_SHA256="$(python -c 'from raosim.movable_pintle import MovablePintleSpec,movable_geometry_fingerprint as f; s=MovablePintleSpec(post_diameter=.020,post_thickness=.001,center_gap_diameter=.012,pintle_rod_diameter=.008); print(f(s,tip_angle_deg=20))')"

lrekit --propellant LOX/RP-1 --pc 7000000 --rt 0.02 \
  --injector pintle --no-electric-pump --injector-cad none \
  --injector-architecture son_continuous_movable \
  --injector-sizing auto \
  --pintle-radial-exit continuous_radial_gap \
  --pintle-radial-stream fuel --pintle-deflector-angle 20 \
  --movable-post-diameter 0.020 --movable-post-thickness 0.001 \
  --movable-center-gap-diameter 0.012 \
  --movable-pintle-rod-diameter 0.008 \
  --movable-cd-map '0:0.62,0.5:0.70,1:0.76' \
  --movable-cd-source 'configuration-controlled cold-flow data' \
  --movable-cd-sha256 "$CD_SHA256" --movable-cd-fluid RP-1 \
  --movable-cd-geometry-sha256 "$GEOMETRY_SHA256" \
  --movable-cd-re-min 10000 --movable-cd-re-max 200000 \
  --movable-cd-dp-min 500000 --movable-cd-dp-max 2000000 \
  --movable-cd-temperature-min 285 --movable-cd-temperature-max 310 \
  --movable-cd-cavitation-min 1.5 --movable-cd-cavitation-max 20 \
  --movable-position-tolerance 0.000001 \
  --movable-position-feedback-resolution 0.000001 \
  --movable-backlash 0.000001 \
  --movable-metrology-source 'configuration-controlled metrology record' \
  --movable-metrology-sha256 "$METROLOGY_SHA256" \
  --movable-closed-leakage-area 0 \
  --movable-leakage-source 'configuration-controlled seat/leak test' \
  --movable-leakage-sha256 "$LEAKAGE_SHA256" \
  --movable-unbalanced-pressure-area 0.000020 \
  --movable-spring-preload-force 5 --movable-seal-friction-force 4 \
  --movable-moving-mass 0.2 --movable-maximum-acceleration 50 \
  --movable-actuator-force-capacity 500 \
  --movable-stem-diameter 0.006 \
  --movable-stem-allowable-stress 200000000 \
  --movable-actuator-source 'configuration-controlled actuator/material record' \
  --movable-actuator-sha256 "$ACTUATOR_SHA256" \
  --movable-sheet-thickness 0.000125 \
  --movable-sheet-thickness-method vof \
  --movable-sheet-thickness-source 'configuration-controlled VOF result' \
  --movable-sheet-thickness-sha256 "$SHEET_SHA256" \
  --movable-sheet-geometry-sha256 "$GEOMETRY_SHA256" \
  --movable-sheet-thickness-fluid RP-1 \
  --movable-sheet-opening-min 0.0001 \
  --movable-sheet-opening-max 0.0010 \
  --movable-sheet-dp-min 500000 --movable-sheet-dp-max 2000000 \
  --movable-sheet-mass-flow-min 0.1 \
  --movable-sheet-mass-flow-max 2.0 \
  --movable-axial-controller-dp-fraction-min 0.0001 \
  --movable-axial-controller-dp-fraction-max 1.0 \
  --throttle-map 0.2,0.6,1.0 --out builds/son_movable_report
```

`--injector-sizing fixed` evaluates a supplied
`--movable-commanded-opening` and annulus gap without resizing. Internal
`movable` sizing holds a previously resolved full-power annulus gap fixed and
solves center-rod travel; the public throttle-map workflow constructs that
state automatically.

## Export and release boundary

With `--injector-cad none`, the movable branch writes a report-only package:

- `pintle_parameters.json`;
- `pintle_dimensions.csv`;
- `pintle_schematic.svg`; and
- `pintle_cross_section.png`.

The report keeps `A_tip`, `A_cg`, `A_eff`, opening, transition/open stop,
calibration provenance, actuator ledger, `delta_eq`, and independent
`delta_sheet` evidence separate. Every STEP/DXF/parts/machined/auto CAD request
for `son_continuous_movable` fails closed. The existing fixed-pintle exporters
do not model a swept moving assembly with closed/open stops, running
clearances, seals, guides, collision checks, and tolerance-stack verification.
They are not silently reused for the movable architecture.

A future swept assembly still requires selected materials, seals, fits,
surface finishes, drawings/GD&T, structural and thermal analysis, inspection,
proof/leak testing, configuration-specific cold flow, optical spray data,
stability assessment, and hot-fire evidence. `hardware_qualified` remains
`false`.
