# LREKit / RaoRocketSim — full feature, data-flow, literature & CAD audit (2026-07-11)

Scope: every implemented feature; the complete data flow from CLI input through every
mathematical/physical model to exported CAD; verification that each relation is rooted in
`propulsion_texts/`; a fresh census of all shipped STEP/STL artifacts with a machinability
assessment for cold-flow/hot-fire hardware. All code claims below were re-verified against the
current working tree today (last commit `ba8fcb6` plus ~6,040 uncommitted inserted lines across
60 files). Companion documents: the historical
[2026-07-02 physics audit](REPO_PHYSICS_AUDIT_2026-07-02.md), the
[2026-07-11 remediation record](PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md), the
[model registry](MODEL_REGISTRY.md) (61 entries; `audit_model_registry(".")` passes live on this
tree), and the new [pump advanced-geometry roadmap](PUMP_ADVANCED_GEOMETRY_ROADMAP_2026-07-11.md).

---

## 1. Repository map and entry points

`raosim/` is the engine (47,110 lines across ~45 modules plus `raosim/jax/` at 3,350 lines);
`lrekit/` is the thin console-script package (`lrekit`/`raosim`/`RaoRocketSim` → `raosim.cli.main` →
`raosim/run_nozzle.py::main`, the 4,558-line packaged CLI; `RaoRocketSimLegacy` → top-level
`main.py`, the older interactive/batch toolbox that drives `raosim.design.design_nozzle_v2`
directly). `scripts/` holds research probes (J4 gate, chart sweeps, topology plots).
`tests/` is the regression suite; `TEST_PROMPT.md` + `examples/cli/test_13kn_sealevel_regen_allstl.args`
is the standing 13 kN LOX/RP-1 Pc 3.0 MPa regression harness. `builds/` holds versioned run
artifacts; `propulsion_texts/` is the source corpus (main folder + `pintle_injector/` +
`fuel_pump_design/` + `materials_science/`).

## 2. Master data flow (packaged CLI, one run)

```
args (@argfiles, complete-package defaults, split injector-pressure model, wall-sizing mode)
  └─ raosim/cea.py::resolve_thermochemistry ──────────────── Propellant(γ, Tc, R_gas, c*, η_c*, η_CF)
       │   builtin table (propellants.py) or RocketCEA FROZEN chamber snapshot;
       │   equilibrium mode rejected; validated mode requires CEA
  └─ ε from Pc/Pa if absent … gas_dynamics.expansion_ratio_from_pressure (isentropic M_e → A/A*)
  └─ Rt from target thrust … design.throat_radius_for_target_thrust: At = F/(Cf_delivered·Pc)
       │   Cf from gas_dynamics.thrust_coefficient (Sutton eq. 3-30 family), Cf_delivered = Cf·η_CF
  └─ contour … design._build_v2_contour
       ├─ throat_geometry.ThroatGeometrySpec (Ru/Rt=1.5, Rd/Rt=0.382, 45° convergent; single
       │   shared contract — chamber and nozzle must carry the identical spec)
       ├─ optional --cd-target: Hall Cd = 1 − ((γ+1)/96)(Rt/Ru)² inverted for Ru/Rt inside
       │   SP-8120 bounds (0.6–1.5); the 1.5–2.0 extension is a labelled repository allowance
       ├─ nozzle_geometry.bell_nozzle_contour  (method dispatch, §3)
       ├─ chamber_geometry.chamber_contour     (L*-exact volume closure, §4)
       └─ full_engine_contour + thrust_chamber_geometry_checks (7 hard geometry gates)
  └─ engine.compute_engine_performance: Me, Pe, Cf_ideal→Cf·η_CF, ṁ = Pc·At/(c*·η_c*), Isp
  └─ physics.boundary_layer_displacement (flat-plate turbulent screen)
  └─ physics.bartz_heat_flux → station-wise h_g, T_aw, q (§5)
  └─ wall sizing: scalar OR --wall-sizing regen → thermal_design.size_wall_profile /
       joint_wall_channel_design (station-wise SP-125 stress-closed t_hot + channel geometry)
  └─ physics.regenerative_cooling_screen / _analysis (coupled Sieder-Tate + 1-D conduction march)
  └─ injector: design/evaluate_pintle_injector → injector.size_pintle_injector (§6)
       ├─ fixed_discrete: annulus + slots/holes
       ├─ son_continuous_movable: Son A_tip/A_cg opening + actuator/evidence ledger
       │   └─ throttle map: fixed annulus hardware + separate upstream axial controller
       ├─ optional spray/c* fixed point (spray_coupling.solve_spray_cstar_fixed_point, opt-in,
       │   requires explicit η_mix, η_comb; blocked with CLI and Python regen)
       └─ injector.feed_system_ledger → per-stream required pump outlet P, Q, H, NPSH
  └─ pump: pumps.size_electric_pumps(feed_system, spec) (§7) → pump.json, BOM, reference geometry
  └─ CAD: export.py (wall/jacket), regen_cad.py, injector_cad.py + injector_coaxial_cad.py,
       pump_cad.py (mesh) + pump_cad_brep.py (B-rep) → per-part STEP+STL + engine_assembly.step
       with interface.py bolt screens
  └─ gates & artifacts: validation.evaluate_design_gates + injector/interface/structural gates,
       model_registry audit, release_readiness evidence manifests (hardware_qualified always
       false), summary.json / pintle.json / pump.json / contour.csv / plot panels / build_log
```

Legacy `main.py` runs the same chain through `design_nozzle_v2` with the strict v2 schema
(ThermoSpec/CoolingSpec/MaterialSpec/InterfaceSpec/ManufacturingSpec…), adds sweeps
(`trade_study.py`), altitude maps, separation checks, wall-pressure plots, and the trajectory
integrator. Failed injector gates block all artifact writes unless `allow_infeasible`.

## 3. Nozzle construction — four contour methods

`nozzle_geometry.bell_nozzle_contour` dispatches on `method`:

**3.1 `bezier` (default, the trusted preliminary path).** Three sections: upstream circular arc
(R_u = 1.5 R_t) from the 45° convergent tangent to the throat; downstream circular arc
(R_d = 0.382 R_t) from the throat to the inflection point N at angle θ_n; quadratic Bézier
(≡ canted parabola) from N to exit E(x = L_n, r = √ε·R_t) with control point P1 at the
θ_n/θ_e tangent intersection. L_n = (L%/100)·(R_e − R_t)/tan 15°. θ_n, θ_e come from the
digitized Rao TOP charts (`_THETA_N_TABLE`/`_THETA_E_TABLE`, ε ∈ [4,50] × L% ∈ [60,100],
bilinear `RegularGridInterpolator`); chart provenance is Rao ARS J. 1960 as reproduced in
Sutton and `nozzle_geometries.pdf` (ε=10/80% → 30.0°/15.5°, chart-verified in the 07-02 audit),
with the γ-insensitivity justification quoted from Rao 1961 (`RaoRecentDevinRockNozConfig.pdf`
p. 1490). Out-of-grid lookups are flagged `rao_chart_extrapolated` and demote the contour to a
non-benchmark diagnostic. In the CLI this path is wrapped in `_TrustedTopSolution` with
reliability `BENCHMARK_VALIDATED` and zeroed residual fields — it is a chart construction, not
a solve.

**3.2 `moc` (direct MOC optimizer, `rao_optimizer.moc_bell_nozzle`).** Experimental inviscid
axisymmetric path: optimizes a monotone cubic-Hermite spline wall audited by an MOC evaluation;
since the remediation, the exported wall is sampled from the optimized spline itself (the
tangent-intersection Bézier point is metadata only) and C_F is throat-area normalized.

**3.3 `rao` / `rao_variational_moc` (research BVP, `rao_variational.py` + `raosim/jax/`).**
The full Rao 1958 variational problem: transonic start line (Kliegel–Levine / Hall,
`transonic_kernel.py`), kernel march ported from the NASA/JHU `MOC_Grid_BDE` C++ (binary
fidelity, incl. the `5/8` integer-division transcription fix and Ru start-line fix), correct
characteristic invariants (C−: d(θ+ν) = +S ds; C+: d(θ−ν) = −S ds, S = sinθ·sinμ/r — Anderson
§11.4, Zucrow & Hoffman Ch. 17), control-surface residual blocks (mass, length, endpoint,
kernel-D stationarity), Optimistix Levenberg–Marquardt on the JAX backend with a constraint
ladder, fixed-end topology seeding (`set_theta_b` secant), BDE-region wall march
(`moc_topology.build_reference_topology`), and — new since 07-02 — direct full C-D-E
control-surface thrust integration with point-D projection/Mach/angle continuity reporting and
full-surface mass closure (`thrust_surface_scope == "full_control_surface_cde"`); partial D-E
results are diagnostic only. Validation track: `outputs_M3.5Perf` oracle suites
(`tests/test_nasa_kernel_march_parity.py`, `test_nasa_port.py`, wall RMS 1.8e-4 vs `wall.out`),
Rao 1958 Nozzle B digitized Table 2/3 as a strict benchmark, explicit rejection of the scarfed
1990 case. CLI: `--contour-method rao-bvp` is an explicit opt-in; failed geometry/reliability
gates block export. Reference solved state at ε=10/L80: θ_B ≈ 25.5°, bell peak ≈ 26.3° near the
throat, exit ≈ 11.2°, seams exactly closed; the chart 30/15.5 pair is documented as the
parabola-fit family, not a residual-solve constraint. J6 sensitivities
(`raosim/jax/sensitivities.py`) provide dCf/d{M,θ,r,kdf,pa,γ} at the converged surface.

**3.4 Throat construction (both sides).** `ThroatGeometrySpec` is the single contract: Ru/Rt
(default 1.5), Rd/Rt (0.382), convergent half-angle (45°), throat at x=0.
`full_engine_contour()` raises if chamber and nozzle specs differ; the chamber/nozzle seam gap
must be ≤ 1e-10 m. The Hall discharge coefficient is computed and *reported*
(`throat_discharge_coefficient_hall`) but deliberately **not** folded into ṁ or thrust sizing —
the closure note in `thrust_closure` explains why (uncalibrated delivered-performance model).
Anchors: 1.5/0.382 arcs verbatim in `nozzle_geometries.pdf`; SP-8120 §2.1.1.1
(`19770009165.pdf`) for the Ru/Rt 0.6–1.5 efficiency-constant range and the one-Rt-upstream
independence quote; Hall 1962/Kliegel–Levine 1969 are the true primaries for the Cd series
(external, honestly labelled — corpus carries them as SP-8120 charts).

## 4. Chamber sizing

`chamber_geometry.chamber_contour` assembles injector face → cylinder (R_c = R_t·√CR) →
shoulder fillet (R_s) → straight 45° convergent cone → upstream throat arc (R_u) → throat. The
cylinder length is **root-solved (bisection, xtol 1e-13) so the exact conical-frustum volume of
the revolved polyline equals V_c = L*·A_t** (measured rel. error ~3e-15; `enclosed_volume` uses
the exact frustum formula per segment). L* is an explicit user input in validated mode; the
default 1.0 m is a labelled placeholder with a warning (L* is a residence-volume proxy —
Huzel & Huang / SP-125 — that must be chosen per propellant/injector). Contraction ratio
default 2.5 (labelled placeholder). The shoulder is auto-derived when unset:
`auto_shoulder_factor` = 0.8 × the closed-form maximum fillet that still leaves a straight
convergent segment — a *geometric closure* justified by SP-8120's one-Rt-upstream throat
independence (see `docs/shoulder_radius_design_basis.md`), not a performance equation.
`injector_to_throat_length` is the true face-to-throat distance (used by spray, residence and
acoustic models — not just the cylinder). Seven hard geometry gates
(`HARD_THRUST_CHAMBER_GEOMETRY_CHECKS`): monotonic x, watertight seam (≤1e-10 m), position
continuity, slope continuity (≤1° at every join), volume closure (≤1e-8), positive cylinder
length, offset-contour (wall-thickness) self-intersection-free. Failing any blocks CAD/export.

## 5. Thermal, regen wall, and structure

Gas side: full Bartz 1957 h_g = (0.026/D_t^0.2)(μ^0.2 c_p/Pr^0.6)(Pc/c*)^0.8(D_t/r_c)^0.1
(A_t/A)^0.9 σ (`physics.py::bartz_heat_transfer_coefficient`; the 0.026 constant is verbatim in
`technical-notes-1957.pdf`), recovery T_aw with r = Pr^(1/3), local M from the isentropic area
ratio each station. Coolant side: Sieder–Tate Nu = 0.027 Re^0.8 Pr^(1/3)(μ_b/μ_w)^0.14
(turbulent branch; Sieder & Tate 1936, SP-125 §4 practice) with the **Shah–London
all-walls-heated rectangular-duct polynomial** for the laminar branch
(`rectangular_duct_laminar_nusselt`, Nu = 8.235(1 − 2.0421α + 3.0853α² − 2.4765α³ + 1.0578α⁴
− 0.1861α⁵) — coefficients match the published H1 solution; local source routed through the
White *Fluid Mechanics* noncircular-duct section per the registry). Fin efficiency for channel
lands, Darcy Δp with f = 64/Re / Blasius / Swamee–Jain, optional curvature correction
(Niino–Kumakawa/Taylor via `pizzarelli2011.pdf`, off by default per SP-8087 caution), gray/
banded radiation (`leccese2018.pdf`), Zuber CHF screen (labelled pool-boiling reference),
RP-1 700 K coking screen (SP-8087), two-ring manifold Newton network. Structure: SP-125
eq. 4-31 liner stress (Δp·r/t + Eαq·t/(2(1−ν)k)), eq. 4-29 inelastic buckling, eq. 4-28 thermal
strain, Coffin–Manson / total-strain-life fatigue — all OCR-verified against `19710019929.pdf`
in the 07-02 audit. Station-wise wall sizing (`--wall-sizing regen`,
`thermal_design.size_wall_profile`) closes t_hot and channel geometry against the stress and
temperature limits simultaneously; the 13 kN Pc 3.0 MPa baseline lands GRCop-84 feasible
(margin 1.05, t_hot 0.60 mm, 194 × 0.5 mm channels, peak wall 946 K) where Pc 7 MPa was
thermally infeasible for every catalog liner — the reason the standard test case runs 3.0 MPa.

## 6. Pintle injector — sizing, gates, feed system

`injector.size_pintle_injector` dispatches by architecture. For
`fixed_discrete`, auto sizing solves annulus plus slot/hole areas from the
cycle split, while fixed sizing evaluates supplied geometry and reports
delivered-vs-required drift. For `son_continuous_movable`, auto sizing solves
the full-power annulus and center-rod opening; fixed sizing evaluates an
explicit opening; and the internal movable mode holds a previously resolved
annulus gap fixed while solving center-rod travel.

Metering: ṁ = C_d·A·√(2ρΔp) per stream (SP-8089/Sutton practice; registry
`injector.incompressible_orifice` routes to `19760023196.pdf`), Δp = χ·Pc with χ default 0.20
(H&H/SP-8089 chug-stability floor), compressible subsonic/choked branch for gas or
supercritical states, two-phase/flashing states rejected (`InjectorUnsupportedState`). Feed
states resolve through CoolProp when available with subcool/vapor-pressure margins; the fuel
stream can be taken from the regen-jacket outlet (the CLI's split-pressure model). **Target-TMR
active solve** (`_target_momentum_dp_fraction`, new since 07-02): bisection on the radial
stream's Δp/Pc within declared bounds so the achieved TMR = ṁ_r v_r/(ṁ_a v_a) hits the request
exactly, with unreachable targets rejected up front with the achievable range. Geometry: central
pintle post (D_p = 0.30·D_chamber packaging default, labelled repository heuristic in the
registry), axial annulus around it, radial rows of rectangular slots or round holes
(exact per-hole area/L-D/pitch/web/blockage propagated to CAD). Spray: half-angle from the
radial/axial momentum resultant with deflector tilt (atan2 form — Heister leading order;
matches Hwang 2022/Son 2017 definitions in the pintle corpus); BF = N·w/(πD_p). Atomization:
Hinze critical-We stable drop, 15·D_h primary-breakup (labelled repository midpoint of the
Reitz–Bracco range), d²-law + Priem–Heidmann vaporization-limited η_vap — each with phase/
pressure applicability gates that *reject* rather than extrapolate (gas, two-phase,
transcritical). Vaporization is reported separately from η_mix/η_comb/η_c* (no silent
promotion); the opt-in spray/c* fixed point closes ṁ = Pc·A_t/(η_c*·c*) only when the two
missing efficiencies are supplied explicitly. Manifold: two-ring + N-branch Newton network as a
maldistribution *screen* (reported, deliberately not auto-charged to the pump budget).
Face/tip thermal: recirculation T_aw = 0.8 T_c + Dittus–Boelter series circuit, margin ≥ 1.2
(Kang 2022-aligned screening constants, labelled). Stability: chug χ floor, chamber acoustic
modes + n–τ sensitive band screen. The fixed-discrete throttle study commands both effective
areas and explicitly supplies no actuator stroke. The separate Son branch uses
`r_f=D_post/2-t_post` and the stable Eq. 1 form
`A_tip=π[2r_f L_open cosθ-L_open² sinθ cos²θ]`, with
`A_cg=π(D_cg²-D_pr²)/4` and `A_eff=min(A_tip,A_cg)`. Its open stop remains below
`A_tip=A_cg`, since travel loses minimum-area authority at that transition. The
implementation reproduces Son's published 0°/20°/40° transition openings.

Son Eq. 3 defines the measured discharge coefficient but supplies no universal curve. The
movable solve therefore interpolates only a configuration-controlled `Cd(L_open/L_max)` artifact
whose `raosim.son2017_movable_geometry.v1` fingerprint matches the resolved post/rod/center-gap/tip/open-stop geometry;
the gate fails unless its source/hash, radial liquid identity, Re, ΔP, temperature, and
cavitation-number envelope all match the solved point. Position tolerance, feedback resolution,
and backlash, closed-stop leakage, an explicit unbalanced pressure area, momentum reaction,
preload/friction/inertia, actuator capacity, and stem stress form separate static gates. Pressure
balance is never inferred.

Movable throttling is fixed-hardware: the center rod meters only the radial stream; post, rod,
center gap, open stop, and axial annulus stay fixed. A separate upstream controller on the axial
stream is solved within declared ΔP/Pc bounds to close delivered mass flow and O/F. The output
reports `L_open/L_max` as physical stroke and the controller pressure drop separately; it does not
rename an effective annulus-area fraction as stroke. The hydraulic reachability result is not a
valve/control/transient validation.

Mechanical `L_open`, minimum gap, internal `A_eff`, and `delta_eq=A_eff/(2πR_exit)` are not
liquid-sheet thickness. A VOF/measured handoff requires an independent thickness artifact with
matching fluid and opening/ΔP/mass-flow domains; otherwise the primary-sheet-to-parcel handoff is
blocked even though hydraulics can be reported. See
`docs/MOVABLE_PINTLE_MODEL.md`. Injector gates also cover mass-flow closure per stream, stiffness,
cavitation K ≥ 1.5, L/D flip screen, manufacturing floors ≥ 0.3 mm, concentricity, spray-wall
impingement with the Apollo gouging note, combustion-length, TMR tracking, tip/face thermal,
acoustics, mandatory cold-flow warning per SP-8089…). `feed_system_ledger` then produces, per
stream: required pump outlet pressure = Pc + injector Δp + declared manifold allowance
(charged once) + regen jacket Δp (if that stream is the coolant) + line/valve losses + control
margin; head H = Δp/(ρg), Q, ideal power, NPSH_available; gates
`feed_pump_pressure/capacity/npsh_{role}`. Literature audit for all of this lives in
`docs/PINTLE_DESIGN_EVALUATION.md` + `propulsion_texts/pintle_injector/PINTLE_LITERATURE_CATALOG.md`
(61-file corpus).

## 7. Electric pump — how every dimension is decided

`pumps.size_electric_pumps(feed_system_ledger, PumpSizingSpec)` per line (fuel, oxidizer):

1. **Duty** from the injector ledger: Q = ṁ/ρ, H = ΔP_required/(ρg), NPSH_a from tank state.
2. **RPM** (`_select_rpm`): solved from the nondimensional specific-speed target
   Ns = ω√Q/(gH_stage)^0.75 = 0.45 (inside SP-8109's flight-proven 450–2100 US band, verified
   verbatim), then clamped by geometry-derived bounds (min/max D2, minimum machinable b2,
   maximum b2/D2) and the motor max-rpm. Stages split H when it exceeds the conservative
   2,500 m/stage cap (a repository cap, deliberately below SP-8109's ~100 kft LH2 figure).
3. **Impeller** (`_impeller_geometry`): U2 = √(gH_stage/ψ) with ψ = 0.55 (SP-8109 eq. (7)
   form); D2 = 2U2/ω; b2 = Q/(πD2·φ·U2) with φ = 0.08; eye D1 from the inlet flow coefficient
   0.12; Ds reported. **Blade count Z from the digitized SP-8109 fig. 16 ψ–φ₂ minimum-blade
   carpet** (`sp8109_min_blade_count`, ±0.02 ψ read-off), snapped to a multiple of the inducer
   count (SP-8052 §3.1.14) — e.g. the 13 kN case solves Z = 12 on both lines. β2 = 25°
   screening backsweep; β1 currently atan(φ) — see the flag in §10.
4. **Velocity triangle + slip** (`_velocity_triangle`): Stodola σ = 1 − π sinβ2/Z clamped
   [0.55, 0.92] (13 kN case: 0.889 = hand value), c_u2 = σU2 − c_m2/tanβ2, Euler head
   U2·c_u2/g ≥ stage head enforced as a margin.
5. **Loss meanline** (`_hydraulic_meanline`): Reynolds-banded passage friction, incidence,
   blade-loading, disk friction, leakage, low-Q/off-Ns recirculation buckets → η_hyd clamped
   [0.20, 0.78] (deliberately conservative; a labelled `centrifugal_meanline_v1` model with
   SP-8109/SP-8052 source IDs, not a claimed pump map).
6. **Inducer** (`_inducer_geometry`, SP-8052 throughout): suction specific speed
   Nss = ω√Q/(g·NPSH)^0.75; tip flow coefficient from eye continuity; blade angle =
   flow angle/(1 − 0.425) per the §2.1.9/3.1.9 incidence-to-blade-angle ratio (band 0.35–0.50);
   constant-lead helix r·tanβ = const (§3.1.10) sets pitch and hub angle; wrap from solidity
   2.5 (§3.1.15) with the developed-chord cosβ factor; hub ratio 0.35 (§2.1.6 "normally 0.2 to
   0.4", verified verbatim); leading-edge thickness at the 0.005-in low end of J-2/F-1 practice.
   Hong 2012 (10.4° tip/3 blades/σ 2.6) is pinned as a cross-source benchmark in `test_pumps.py`.
7. **Coupled eye/channel/free area** (`_solve_annular_eye_and_shaft`,
   `_meridional_channel`): the shaft torsion/manufacturing diameter, bore fit, root wall, hub,
   D1, blade blockage, and phi1 close through
   Q = pi(R1²-Rh²)(1-B1)phi1 omega R1. Quarter-ellipse hub and shroud curves carry the
   *exact net* eye and exit areas honoring D1/D2/b2, screened against SP-8109 §2.3.1.2
   (c_m2/c_m1 must lie in 1–1.5). **Blade camber** (`impeller_blade_camber`): dθ/dr =
   1/(r·tanβ(r)) with β linear in r between the solved angles — the log-spiral family of
   SP-8109 blade-geometry practice. The converged zero-incidence relative-flow angle is the
   exported metal beta1; the old atan(phi2) value is provenance only. Four full blades reach
   the eye and the other fig.-16 discharge blades are downstream splitters; B1 <= 0.20 and
   B2 <= 0.15 are solve/feasibility gates, and b2 uses net circumference.
8. **Diffuser/volute selection** (`_diffuser_volute_geometry`): vaned diffuser above Ns 0.20,
   throat area = max(1.15·Q/c_m2, 0.35·A_exit), volute exit area from a velocity cap.
9. **Thrust balance** (`_thrust_balance_geometry`): hub wear ring at the eye diameter and
   balance holes (flow area ≈ 4× seal-clearance area) per SP-8109 §3.5.2.1–3.5.2.2; shaft seal
   land face-speed screen.
10. **System coupling & maps**: synthetic homologous performance curve, system-curve
    intersection for the commanded throttle band, thermal/stress ledger (tip speed 350 m/s
    screen, DN, hoop), shaft diameter from torque.
11. **Electric drive & battery** (`_drive_sizing`/`_battery_sizing`): P_shaft = ρgQH/η;
    motor/inverter/battery sized by SCREENING_DEFAULTS densities (2,500 W/kg, 15 kW/kg,
    250 Wh/kg, η 0.90/0.96/0.95) — versioned technology assumptions anchored to the Lee 2021 /
    Spiller-class comparison cases in `fuel_pump_design/` (e.g. `s42405-020-00325-z.pdf`,
    `BF03404670.pdf`), exported as assumptions, never claimed as literature constants; shared
    DC bus voltage selection; hardware BOM with the Lee 2021 mass-closure benchmark
    (motor 451.2 g / battery 985.6 g) pinned in tests.

13 kN LOX/RP-1 baseline outputs (`builds/tests/13kn_sl_full/pump.json`): fuel 54,633 rpm,
oxidizer 28,793 rpm, both Ns = 0.45, Z = 12, 3-blade inducers, σ = 0.889.

Those artifact numbers predate the 2026-07-12 free-area closure and must be
regenerated before comparison; current solves may lower an automatically selected
RPM/increase D2 to respect the manufacturing thickness and SP-8109 free-area limits.

## 8. CAD generation chain and the artifact census

**Wall/jacket** (`export.py`): contour cleaned to a simple wire, closed x–r profile revolved
360° in CadQuery (SI → mm at the kernel boundary, explicit), re-imported and checked (scale,
`isValid`, solid count, volume vs the exact revolved-profile volume ≤ 1e-5); STL watertightness
hard-gated. `--require-brep` default; `--no-require-brep` permits only a labelled diagnostic
fallback. **Regen** (`regen_cad.py`): liner + full-count lofted channels/ribs + seals + jacket
as one Boolean compound; cold-flow variant requires manifolds, ports, and a *connected extracted
coolant volume*; sealed-end solids are reference-only. **Pintle** (`injector_cad.py`): named
assembly + machined mode with real Boolean cuts and a `slot_cut_through` gate.
**Fixed-discrete coaxial five-part injector** (`injector_coaxial_cad.py`, new): the TRW/Elverum architecture
from the Rezende/Nardi exploded view — pintle body (axial central bore), replaceable pintle tip
(radial metering ring at the skip-distance line), injector body with *toroidal plenum* and
lateral inlet, orifice distribution plate, faceplate forming the annular metering gap; sealed by
construction, driven entirely by the solved hydraulics, with circuit-connectivity, pairwise
interference, O-ring gland/spigot/thread envelope, and STEP round-trip gates. The
`son_continuous_movable` branch deliberately does not dispatch into these fixed-geometry
exporters. With CAD disabled it writes a JSON/CSV/SVG/PNG report containing Son control areas,
travel/stops, calibration/actuator evidence, and separate equivalent-versus-measured/VOF sheet
thickness. Every movable reference/parts/DXF/STEP/machined/auto request fails closed until a
swept assembly implements closed/open stops, running clearances, seals/guides, and collision and
tolerance checks. **Pump mesh
package** (`pump_cad.py`): schematic primitives, 2π seam welded, `_write_part` hard-fails
non-watertight meshes. **Pump B-rep** (`pump_cad_brep.py`): per §7 geometry — hub/backplate
single revolve through the mechanically closed meridional hub curve, tapered camber-ribbon
full blades plus truncated downstream splitters trimmed by the shroud-curve revolve
(semi-open), constant-lead helical inducer sweep, vaned diffuser ring, hollow volute flow
passage (A(θ) = A_exit·θ/2π circular-section loft) with tangential outlet bore,
inlet bore, rotor/diffuser pockets; audits: `audit_pump_component_interference`,
`audit_pump_clearances`, `audit_volute_flow_passage` (single connected inlet→impeller→diffuser
→scroll→outlet void with positive handoff overlaps), `audit_meanline_geometry_fidelity`
(an identity check; any CAD hub change is now an exporter error), and
`audit_split_casing_manufacturability`. The operating casing is separate rear-body/front-cover
geometry split through the scroll centerplane with a keyhole full-face gasket land,
circumferential/outlet-neck bolt bores, dowels, volume/tool/access/clamp gates, and separate
STEP round trips;
`cold_flow_release_ready` is hard-coded `false` with explicit external blockers.
`--engine-assembly` aggregates wall/jacket/pintle/pump/battery into `engine_assembly.step` with
an `interface.py` bolt screen.

**Fresh census (today).** 138 STEP files under `builds/`: **137 true OpenCASCADE B-reps**
(`MANIFOLD_SOLID_BREP`), the single exception being the known May-era faceted leftover
`builds/v019_20260511_174335/rao_nozzle.step`. Pump STEP is now genuinely B-rep everywhere it
ships (the 07-02 faceted-writer gap is closed and unreachable). STL edge-topology check on the
full 13 kN reference build: every part mesh watertight and manifold with positive signed volume
— wall (172,992 tris), jacket, all four named pintle parts, all pump B-rep parts (impeller
30,968 tris) and mesh-package parts. The two non-conforming meshes are both labelled
non-part visualization surfaces: `regen.stl` (3,804 boundary edges, open by design) and
`pintle/pintle_reference.stl` (11 non-manifold edges, schematic reference).

**Machinability assessment (geometry-level, for cold-flow/hot-fire intent):**
wall/jacket/chamber are bodies of revolution — turnable, with the regen liner channels millable
/ slitting-saw cut before closeout (closeout process itself — electroform/braze — is an
explicit external item). Pintle machined parts are lathe+mill geometry with real drilled/slotted
features and the cut-through gate; the coaxial five-part stack is the classic machinable
architecture (each part a solid of revolution plus drill patterns) with glands and thread
envelopes present, though thread standards/finish/tolerances remain drawing-level externals.
Pump: impeller is 3-axis-millable in principle (tapered extruded main/splitter blades on an
open face), inducer requires 4/5-axis or turn-mill (helical blades), and the diffuser ring is
flat-plate millable. The scroll is exposed in two valid centerplane-split casing halves; the
cover-removal/tool/bolt/dowel/gasket-land topology is now modeled and gated. This is not a
qualified pressure joint: selected gasket, fasteners/threads/dowel fits, flange FEA/fatigue,
shaft retention, bearings/seals, tolerances, thermal growth, proof and flow tests remain.
No drawings, GD&T,
tolerances, materials callouts, or process routing exist for any part — by design, these are
the release-gate externals (`PHYSICAL_RELEASE_GATES.md`), and every package reports
`hardware_qualified: false`.

## 9. Literature-verification status (equation → source)

The 61-entry `model_registry.py` is the authoritative machine-readable map (each entry:
relation, source, local corpus route, validity envelope, verification, validation level;
the live audit passes — every declared local source exists, every repository heuristic is
labelled as such). Spot re-verification done today on top of the 07-02 verbatim checks:

| Relation (code site) | Corpus anchor | Status |
|---|---|---|
| Rao TOP tables, 1.5/0.382 arcs, parabola (nozzle_geometry.py) | `RaoRecentDevinRockNozConfig.pdf`, `nozzle_geometries.pdf`, SP-8120 | verified 07-02 (chart render), unchanged |
| Characteristic invariants ± S ds (moc.py, jax) | Anderson §11.4 / Z&H; oracle `outputs_M3.5Perf` RMS ≤ 2.3e-6 | verified 06-11, tests green |
| Hall Cd inversion (throat) | SP-8120 §2.1.1.1 charts; Hall/K-L external primaries | verified 07-02; bounds now honestly split 0.6–1.5 vs repo extension |
| L*·A_t volume closure (chamber_geometry.py) | H&H / SP-125 L* practice | re-executed: exact frustum integral, bisection to 1e-13 |
| Bartz h_g 0.026 (physics.py:172) | `technical-notes-1957.pdf` verbatim | verified 07-02, code re-read today |
| Shah–London rectangular Nu (physics.py:422) | published H1 polynomial; local route White ch. 6 | coefficients re-checked today — exact |
| Sieder–Tate, Darcy/Swamee–Jain, SP-125 4-28/29/31/32, Zuber, Leccese radiation | SP-125 OCR, `atefi2019.pdf`, `leccese2018.pdf` | verified 07-02 |
| Pintle TMR/BF/spray-angle/SMD chain (injector.py) | Hwang 2022 Eq.1/Son 2017 Eq.4/Hwang Eq.3, Hinze, Reitz–Bracco, Priem–Heidmann via pintle catalog | verified 06-28 eval doc; TMR active solve re-read today |
| Movable `A_tip`, `A_cg`, transition and fixed-hardware schedule (`movable_pintle.py`, `injector.py`) | Son 2017 Eqs.1–4, `pintle_injector/son2017.pdf` | exact geometry implemented; 0°/20°/40° published transitions pinned; `Cd` remains configuration-specific and evidence-gated |
| ψ = gH/U2², Ns 450–2100, Stodola, fig. 16 Z-chart (pumps.py) | SP-8109 (`19740020848.pdf`) | verified 07-02 + fig. 16 digitization notes in-code; σ hand-check 0.889 today |
| Inducer α/β 0.425, constant-lead helix, solidity 2.5, hub 0.2–0.4 (pumps.py) | SP-8052 (`19710025474.pdf`) §§2.1.6–3.1.15 | verified 07-02/plan doc; §2.1.7 sweep & §2.1.8 cant re-extracted today |
| SP-8109 §2.3.1.2 c_m2/c_m1 ∈ 1–1.5 screen (_meridional_channel) | SP-8109 | quote present in-code; section verified in scan |
| Wear rings / balance holes (SP-8109 §3.5.2) | SP-8109 | verified in plan doc |
| Blade-fillet 1.5×t, LE 2:1–3:1 ellipse | SP-8109 fillet-radii section | **extracted verbatim today** (roadmap anchor; not yet implemented in CAD) |
| Electric drive/battery densities | Lee 2021/Spiller-class cases in `fuel_pump_design/` | correctly labelled technology assumptions, benchmark-pinned |

Known naming/provenance flags that remain (all already labelled in code): the Ru/Rt upper
extension to 2.0 is repository allowance, not SP-8120; max head/stage 2,500 m is a repo cap;
D_p = 0.30·D_c is a packaging heuristic; recirculation thermal fractions 0.8/0.2 are screening
constants; the historical NASA `TT'.out` fixture generator provenance is unresolved and treated
as non-authoritative.

## 10. What the tool can and cannot do today

**Can (software-verified, within declared envelopes):** size a complete LOX/RP-1-class engine
from thrust/Pc/propellant to a consistent thrust-chamber contour (chart-anchored Rao/TOP
Bézier), with CEA-frozen or table thermochemistry; close chamber volume to L*·A_t exactly;
compute Bartz/regen station thermals and station-wise SP-125-closed wall+channel sizing with
feasibility gates; size or evaluate a fixed liquid/liquid pintle injector with active TMR
targeting and a feed-system pressure ledger; solve/report the preliminary Son continuous
movable-pintle control area, physical travel, static actuation, strict calibration/sheet-evidence
domains, and a fixed-hardware throttle schedule with a separate axial controller; size
electric pump(s) end-to-end (meanline, inducer/NPSH, losses, maps, thrust balance, motor/
inverter/battery/BOM) off that ledger; emit true B-rep STEP + watertight STL for wall, jacket,
regen compound, machined pintle, five-part coaxial injector, and all pump parts, with
re-import/units/interference/clearance/flow-connectivity gates and an aggregated engine
assembly; run the research Rao BVP with full C-D-E thrust closure, NASA-oracle-validated MOC
kernel, and design sensitivities; produce altitude/separation screens, trade sweeps, a
vertical point-mass trajectory, flow-field visualizations, and versioned, provenance-stamped
build artifacts with a live model-registry audit and evidence-manifest release gates.

**Cannot (explicit non-capabilities):** no executed/validated CFD of any kind (chamber, nozzle, spray, pump
blade-to-blade), no conjugate heat transfer or FEA, no finite-rate/variable-composition nozzle
chemistry (the variable-$c_p$ option remains fixed-composition quasi-1D), no gas/gas or
transcritical injector branches (rejected, not
extrapolated), no universal/transferable movable-pintle `Cd(stroke)` law, no transient actuator
or feed-controller dynamics, no swept movable-pintle CAD assembly, and no cavitation solve beyond
NPSH/Nss margins, no rotordynamics/critical-speed solve, no turbine (electric drive only) and
no axial pump (SP-8125 sits unused in the corpus), no drawings/GD&T/tolerance stacks/process
routing, no thread/seal standard selection, and no path from any passing gate to
`hardware_qualified` — cold-flow/hot-fire readiness is gated on external evidence IDs by
design. The pump B-rep remains packaging-reference geometry whenever
`requires_meanline_resolve` fires (shaft/hub conflict), and the coupled meanline-shaft-hub
mechanical re-solve is on the explicit open-work list.

**Inconsistency flagged during this audit (new):** the impeller CAD camber inlet angle is
sourced from `CentrifugalPumpGeometry.inlet_blade_angle_deg = atan(φ)` (= 4.6° at φ = 0.08)
while the hydraulic meanline's velocity triangle computes the eye-tip relative flow angle
β1 = atan(c_m1/U1) (= 14.4° on the 13 kN case) — the loss model even scores incidence against
the latter. The blade the CAD cuts therefore carries ~10° of built-in incidence relative to the
meanline's own inlet state. Fix (small): export the triangle β1 (and, with the roadmap's 3-D
blading, β1(r) across the eye) into the CAD manifest. Tracked as item 0 in the
[pump advanced-geometry roadmap](PUMP_ADVANCED_GEOMETRY_ROADMAP_2026-07-11.md), which covers
splitter blades, lean, sweep, thickness distribution, hub-vs-shroud loading, fillets, diffuser
vane profiles, and volute generation against the `fuel_pump_design/` corpus.
