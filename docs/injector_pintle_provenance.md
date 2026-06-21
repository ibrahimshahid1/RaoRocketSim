# Pintle injector model — provenance and validity

`raosim/injector.py` sizes a **fixed, liquid/liquid, automatically-sized**
pintle injector from the operating point produced by the nozzle + chamber
solver. It is a preliminary screening model, not a qualification tool.

## Governing chain

The injector is never sized from an independent injector thrust input. It
consumes the engine cycle:

```
F, Pc, Pa, eps, O/F  ->  Cf, At, mdot  ->  mdot_f = mdot/(1+O/F), mdot_o = O/F·mdot_f
                     ->  A_i = mdot_i/(Cd_i·sqrt(2·rho_i·dp_i))
                     ->  annulus gap + radial slots  ->  TMR, spray, gates
```

## Equations and sources

| Quantity | Relation | Source |
|----------|----------|--------|
| Orifice flow area | `A = mdot/(Cd·sqrt(2·rho·dp))` | Sutton & Biblarz, *Rocket Propulsion Elements* (RPE), incompressible injector hydraulics |
| Axial annulus | `A_a = (pi/4)(Do^2−Di^2) ≈ pi·D_p·h` | NASA SP-8089, *Liquid Rocket Engine Injectors* |
| Radial slots | `A_r = N·w·h` | NASA SP-8089 |
| Injection velocity | `v = mdot/(rho·A)` | RPE |
| Total momentum ratio | `TMR = (mdot_r·v_r)/(mdot_a·v_a)` | SP-8089 (pintle momentum balance governs spray) |
| Reynolds / Weber / Ohnesorge | `Re=rho·v·D_h/mu`, `We=rho·v²·D_h/sigma`, `Oh=mu/sqrt(rho·sigma·D_h)` | Lefebvre/Heister atomization |
| Spray half-angle (surrogate) | `theta = atan2(M_r·cos δ, M_a + M_r·sin δ)` | first-order momentum-vector estimate; **cold-flow required** (SP-8089) |
| Cavitation number | `K = (P_manifold − Pvap)/dp` | injector cavitation/hydraulic-flip screening |
| Chamber acoustics | `f_L1 = a/(2·Lc)`, `f_T1 = 1.8412·a/(pi·D_c)`, `a=sqrt(γ·R·Tc)` | cylindrical-chamber acoustic modes |
| Manifold maldistribution | annular two-header square-law network | `thermofluids.solve_annular_manifold_network` (NASA SP-8087) |

## Feed (inlet) fluid properties

Liquid `rho, mu, sigma, Pvap` for the hydraulic sizing:

- **CoolProp HEOS** (literature-grade equations of state; Bell et al., *IECR*
  53, 2014) for Oxygen (LOX), Methane (LCH4), Hydrogen (LH2), Ethanol, Water,
  Nitrous Oxide.
- **Constant-property literature table** (screening-grade, no T-dependence) for
  RP-1/Jet-A, MMH, N2O4/NTO, UDMH — from Sutton & Biblarz RPE, NASA SP-8087,
  and the CRC handbook. Values are cited inline in `injector.py`.

## Liquid/liquid restriction (deliberate)

A feed is rejected (`InjectorUnsupportedState`) when its resolved state is
gaseous, supercritical (`T ≥ 0.98·Tcrit`), or within 10% of its vapor pressure
(cavitation/flashing risk). Gaseous / near-critical injection needs a separate
compressible / real-fluid branch which is **not** implemented in this MVP.

## What is explicitly NOT validated here

Spray distribution, droplet SMD, mixing/evaporation length, c\* efficiency,
combustion stability, and face/tip heating are screening surrogates or
informational gates. SP-8089 is explicit that pintle spray distributions
require **cold-flow** testing, and the engine requires **hot-fire** validation.
Both are reported by the `validation_status` gate as REQUIRED and not performed.

## Deferred (later implementation passes)

Named-body STEP CAD, cross-section/spray plots, detailed manifold geometry,
experimental spray-angle/SMD correlations, thermal coupling to the face and
pintle tip, movable-sleeve throttle map, stability/admittance screening, and the
gaseous/near-critical compressible branch.
