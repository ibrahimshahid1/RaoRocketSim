# Pintle injector model — provenance and validity

`raosim/injector.py` sizes a **fixed, liquid/liquid, automatically-sized**
pintle injector from the operating point produced by the nozzle + chamber
solver. It is a preliminary screening model, not a qualification tool.

Both the strict `DesignInput` workflow and `scripts/run_nozzle.py` call the
same `evaluate_pintle_injector` integration boundary. A requested injector is
therefore included in backend reports and design gates rather than being a
CLI-only side calculation. Failed injector gates block integrated design/CAD
output unless the caller explicitly selects the preliminary
`allow_infeasible` override.

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
| Sauter mean diameter (SMD) | `d_32 = We_crit·σ/(ρ_g·v²)`, `We_crit≈13`, capped at d_jet | Hinze critical-Weber aerodynamic breakup (Hinze, AIChE J. 1, 1955) |
| Primary breakup length | `L_b ≈ 15·d_jet` (atomization regime) | Reitz & Bracco prompt-atomization scaling |
| Vaporization length | d²-law `d²(t)=d_32²−K_b·t`, `K_b≈1e-6 m²/s` | classic droplet d²-law |
| Predicted c\* efficiency | mass-weighted vaporized fraction in the chamber length | Priem & Heidmann vaporization-limited combustion (NASA TR R-67, 1960) |

The atomization block (`raosim.injector.spray_atomization`) drives the
combustion-development-length gate and the L\*/injector-quality coupling: when
the limiting stream's `breakup + vaporization` length exceeds the available
cylindrical chamber length the gate **warns** (never hard-fails — these are
order-of-magnitude surrogates) and suggests increasing L\*, Δp, or velocity, or
changing the stream assignment.

For a direct regenerative-fuel path, coolant mass flow must close against the
cycle fuel flow. The calculated jacket outlet temperature and pressure are
used as injector feed conditions only when the flows agree within 1%; an error
of 5% or greater fails the integrated design because a bypass and mixing model
would otherwise be required. Explicit feed pressure below
`Pc + injector delta-P` also fails rather than merely warning.

## Feed (inlet) fluid properties

Liquid `rho, mu, sigma, Pvap` for the hydraulic sizing:

- **CoolProp HEOS** (literature-grade equations of state; Bell et al., *IECR*
  53, 2014) for Oxygen (LOX), Methane (LCH4), Hydrogen (LH2), Ethanol, Water,
  Nitrous Oxide.
- **Constant-property literature table** (screening-grade, no T-dependence) for
  RP-1/Jet-A, MMH, N2O4/NTO, UDMH — from Sutton & Biblarz RPE, NASA SP-8087,
  and the CRC handbook. Values are cited inline in `injector.py`.

## Phase branches

Each stream is sized by the branch its resolved state selects:

- **Liquid** → incompressible orifice `A = mdot/(Cd·sqrt(2·ρ·dp))`.
- **Gas / supercritical** → compressible orifice with an explicit choke test
  against the critical pressure ratio `(2/(γ+1))^(γ/(γ−1))` (Sutton & Biblarz;
  Anderson, *Modern Compressible Flow*). Choked: sonic throat injection,
  `G = Cd·P0·sqrt(γ/(R·T0))·(2/(γ+1))^((γ+1)/(2(γ−1)))`; subsonic otherwise.
  Real-gas γ and specific R come from CoolProp (`Cpmass/Cvmass`, `M`); a gas
  state without γ/R (e.g. the liquid-only literature table, or RP-1 forced to
  gas) is rejected.
- **Two-phase / flashing** (within 10% of the vapor pressure, below `Tcrit`) →
  rejected (`InjectorUnsupportedState`): neither branch applies without a
  flashing model.

The injection-state gate reports each stream's branch and, for gas, the choke
state and pressure ratio.

## Extended models

Beyond hydraulic sizing the result carries (all screening-grade):

- **Manifold distribution** (`manifold_distribution`) — the annular two-header
  square-law network run for *both* the fuel and oxidizer manifolds, with a
  per-manifold maldistribution gate.
- **Face / pintle-tip thermal** (`face_tip_thermal`) — a recirculation
  Dittus-Boelter gas-side coefficient + a propellant-side Dittus-Boelter
  coefficient through a 1-D series gas/wall/coolant circuit, giving a real wall
  temperature and margin against the material limit (no longer info-only).
- **Stability** (`stability_screen`) — feed-system chug (injector
  decoupling-fraction rule, SP-8113/SP-194), the chamber L1/L2/T1/R1 acoustic
  modes, and an n-τ reduced-frequency band (Crocco sensitive time lag).
- **Throttle map** (`throttle_map`) — a movable-sleeve schedule that holds the
  dp-fractions (and hence O/F and TMR) constant as `Pc(f)=Pc·f^exp`; it exposes
  the velocity/Re/We/atomization fall and the min-feature throttle floor.
- **Figures** (`raosim.injector_plots`) — a full diagnostic set emitted on
  every pintle run via `export_all_injector_figures`: cross-section, spray
  envelope, hydraulics (areas/velocities/Re/We), atomization/combustion-
  development, the face/tip thermal stack, stability (modes + chug + n-τ),
  manifold maldistribution, the gate scorecard, and (with `--throttle-map`) the
  throttle sweep.
- **CAD** (`raosim.injector_cad`) — a CadQuery named-body assembly (faceplate,
  hollow pintle body, tip, axial annulus, radial slot network, fuel/oxidizer
  manifolds, igniter interface, regen-coolant outlet, optional movable sleeve);
  STEP is authoritative, per-body STLs for printing.

## What is explicitly NOT validated here

Spray distribution, droplet SMD, mixing/evaporation length, c\* efficiency,
combustion stability, manifold maldistribution, and face/tip heating are
screening surrogates or informational gates. SP-8089 is explicit that pintle
spray distributions require **cold-flow** testing, and the engine requires
**hot-fire** validation. Both are reported by the `validation_status` gate as
REQUIRED and not performed. The CAD is a preliminary schematic, not a
drawing-ready part.
