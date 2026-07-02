# Repository physics & CAD audit — 2026-07-02

Full-repo verification of the bell nozzle, throat geometry, chamber, regen wall,
pintle injector, engine CAD assembly, and electric-pump CAD against the
`propulsion_texts/` corpus. Every equation checked below was traced to its code
site AND confirmed in the cited source PDF (verbatim quote or OCR of the scanned
page), and the geometric constructions were re-executed numerically in this
session. Where an external (non-corpus) source is the true primary, that is
stated.

**Scope of runs performed:** `lookup_angles`, `bell_nozzle_contour`,
`chamber_contour`, `throat_discharge_coefficient_hall`, the pump meanline
(`_select_rpm`/`_impeller_geometry`/`_velocity_triangle`), `inspect_stl` on
shipped meshes, a census of all 77 STEP files in `builds/`, and the targeted
test suite (injector 80 pass/1 skip; nozzle/chamber/Bartz 49 pass; the only
failures in the sampled set are `ModuleNotFoundError: jax` — an environment
gap in the audit sandbox, not physics).

---

## 1. Bell nozzle (Rao TOP) — VERIFIED

Construction in `raosim/nozzle_geometry.py`: upstream circular arc
`Ru = 1.5·Rt`, downstream arc `Rd = 0.382·Rt` to the inflection point N,
then a quadratic Bézier (≡ parabola) from N to exit E with the control point
at the θ_n/θ_e tangent intersection.

| Item | Code | Source in corpus | Status |
|---|---|---|---|
| Parabola approximation of the optimum contour | Bézier N→E | Rao 1961 (`RaoRecentDevinRockNozConfig.pdf`): "the optimum thrust nozzle contour can be closely approximated by a parabola… fit the maximum wall inclination θm … and θe at the exit"; SP-8120 (`19770009165.pdf`): "a single canted parabola will very closely approximate the variety of optimum contours" | ✅ verbatim |
| 1.5·Rt / 0.382·Rt arcs | `throat_geometry.py` defaults | Seitzman AE6450 `nozzle_geometries.pdf` (from Sutton): "R1/Rt=1.5 … after throat and up to N: R1/Rt=0.382" | ✅ verbatim |
| θ_n/θ_e tables (ε×L% grid) | `_THETA_N_TABLE`/`_THETA_E_TABLE` | Rendered the Sutton/Rao chart (`nozzle_geometries.pdf` p.9, "Approximate Optimal Design Angles") and read it directly: ε=10 → θ_N≈30° (code 30.0), θ_e≈15.5° (code 15.5) at 80%; 60%/ε=4 → 33/17 (code 33.0/17.0); ε=50 endpoints consistent | ✅ chart-verified (±0.5° digitization) |
| Magnitude cross-check | table mid-grid | Rao 1961: wall angles "about 28° to 30°" downstream, exit "about 10° to 14°" | ✅ verbatim |
| γ-insensitivity of contour | tables used at γ≠1.23 | Rao 1961: "the differences in nozzle contours are negligible" at fixed (ε, L) | ✅ verbatim |
| Bell length | `Ln = L%·(Re−Rt)/tan15°` | standard 15°-cone fraction (Sutton/SP-8120) | ✅ numeric: exact |
| Tangency/continuity | arc→Bézier slopes | re-executed: slope at N ≈ tanθ_n, exit slope ≈ tanθ_e (within discretization); `Re = √ε·Rt` exact | ✅ numeric |

MOC paths (`method='moc'`, `rao_variational_moc`) carry their own validation
track against the NASA/JHU `MOC_Grid_BDE` M3.5Perf oracle
(`docs/nasa_tt_prime_provenance.md`, `tests/test_nasa_*`); the shipped default
remains the chart-anchored Bézier.

## 2. Throat geometry — how it is computed and outputted — VERIFIED

Computation chain:

1. `Rt` either given or sized from target thrust: `At = F/(Cf_actual·Pc)`,
   `Cf` from the standard isentropic thrust-coefficient relation
   (`gas_dynamics.py`; Sutton/Anderson forms).
2. `ThroatGeometrySpec` (`throat_geometry.py`) is the single shared contract
   for both sides of the throat: `Ru/Rt` (default 1.5), `Rd/Rt` (0.382),
   convergent half-angle (45°), throat station `x=0`. Chamber and nozzle
   contours must carry the *identical* spec or `full_engine_contour()` raises.
3. Optional closure `--cd-target`: inverts the Hall leading-order transonic
   discharge coefficient `Cd ≈ 1 − ((γ+1)/96)(Rt/Ru)²` for `Ru/Rt`.
   Numerically confirmed: Cd=0.99, γ=1.24 → Ru/Rt=1.5275 — exactly what the
   shipped `builds/samples/latest_13kn_check/summary.json` records
   (`upstream_radius_source: cd_target_hall_sp8120`). Chart cross-check
   matches SP-8120 fig. 3 trend (0.990 @1.5 → 0.935 @0.6).
4. Hard geometry gates before any export (`chamber_geometry.py`): monotonic x,
   watertight seam (≤1e-10 m), position+slope continuity (≤1°), enclosed
   volume = `L*·At` (measured 3e-15 rel. err.), positive cylinder length,
   offset-contour self-intersection-free.

Outputs: `contour.csv` (x,y in m), `summary.json` (`throat_geometry` dict incl.
Cd provenance), `wall.stl`/`wall.step` solids, profile plots.

Literature status: Hall inversion — SP-8120 §2.1.1.1 quotes verified verbatim
("aerodynamic efficiency remained constant for Ru/Rt values from 1.5 down to
0.6"; film-cooling benefit at R/Rt=1.4; throat flow independent "about one
throat radius upstream"). Hall (1962)/Kliegel–Levine (1969) are the true
primaries for the Cd series — external to the corpus, which reproduces them
only as SP-8120 charts; the docs state this honestly.

⚠️ One naming nit: `SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS = (0.6, 2.0)` — the
verbatim SP-8120 range is 0.6–1.5 (efficiency-constant); the 2.0 upper cap is
a reasonable engineering allowance (large radii only cost length/weight) but
is not literally the SP-8120 stated range.

## 3. Chamber & shoulder — VERIFIED

`chamber_contour()` assembles cylinder → shoulder fillet (Rs) → straight cone
(α) → upstream arc (Ru) → throat, root-solving the cylinder length so the
revolved frustum volume encloses `L*·At` exactly (verified 3e-15). Shoulder
auto-sizing is a *geometric closure* (0.8 × max feasible fillet), justified by
the SP-8120 one-Rt-upstream quote (verified) — i.e. the shoulder is a subsonic
manufacturing feature, not a throat-performance one; SP-125 p. 88 gives only a
factor list for the contraction. This split (performance-determinate Ru vs
constrained-choice Rs) is documented in `docs/shoulder_radius_design_basis.md`
and matches the sources.

## 4. Regenerative cooling & wall — VERIFIED (screening scope honestly labeled)

| Equation | Code site | Source | Status |
|---|---|---|---|
| Bartz h_g = (0.026/Dt^0.2)(μ^0.2cp/Pr^0.6)(Pc/c*)^0.8(Dt/rc)^0.1(At/A)^0.9·σ | `physics.py:172` | Bartz 1957 (`technical-notes-1957.pdf`): "The resulting value of C was found to be 0.026" — verified verbatim; σ property-variation form per Bartz/H&H (ω=0.6 recovers 0.68/0.12 exponents) | ✅ |
| Recovery T_aw with r=Pr^(1/3) | `physics.py` | standard turbulent recovery (Bartz/H&H) | ✅ |
| Coolant side Nu = 0.027 Re^0.8 Pr^(1/3)(μb/μw)^0.14 | `physics.py:422` | Sieder & Tate 1936; standard regen practice (SP-125 §4) | ✅ |
| Laminar proxy Nu = 4.36 + fin area; validity flags at Re<10,000 | `physics.py` | uniform-q circular duct (textbook); flagged as proxy | ✅ labeled |
| Curvature Nu ratio = [Re(Dh/2Rc)²]^±0.05, opt-in only | `--curvature-correction` | Niino–Kumakawa/Taylor via `pizzarelli2011.pdf`/`eucass1p171.pdf`; disabled by default per SP-8087 caution | ✅ |
| Δp = Σ f (L/Dh)(ρV²/2); f = 64/Re, Blasius, Swamee–Jain | `physics.py:501` | SP-125 eq. 4-32 (Darcy); Swamee–Jain per Atefi 2019 (`atefi2019.pdf`) | ✅ |
| Liner stress S_c = Δp·r/t + Eαq t/(2(1−ν)k) | `physics.py:1824` | SP-125 eq. 4-31 — OCR of `19710019929.pdf` p.108–109 shows "(Pco − Pg) r / t" + thermal term; code exact; r = coolant-passage scale per eq. 4-27 (correct reading) | ✅ |
| Tube buckling S_c = 4EtEc/(√Et+√Ec)² · t/(√(3(1−ν²))·r) | `physics.py:1951` | SP-125 eq. 4-29 — OCR matches structure exactly; rectangular-channel use labeled approximate, opt-in gate | ✅ |
| Thermal strain S_l = EαΔT (no hidden 1/(1−ν)) | fatigue path | SP-125 eq. 4-28 | ✅ |
| Radiation (gray Leccese κ=2.4/1.9 @10 bar, ∝P; or banded RTE) | `thermofluids.py` | `leccese2018.pdf`; explicitly *not* folded into Bartz silently | ✅ |
| CHF screen (Zuber 0.131) | `thermofluids.py:216` | Zuber AECU-4439; labeled pool-boiling reference, not forced-flow | ✅ labeled |
| Manifold network (2 rings + N branches, Newton) | `thermofluids.py:357` | SP-8087 hydraulics; Kang & Sun 2011; labeled 1-D | ✅ |

RP-1 coking screen (700 K, SP-8087), coolant inlet-temperature defaults
(120 K CH4 / 25 K LH2), heated-area integration over meridional arc length —
all present and as documented in `docs/regen_wall_model.md`, which itself
proved accurate on every point spot-checked.

## 5. Pintle injector — constraints, calculations, provenance — VERIFIED

Sizing (`raosim/injector.py::size_pintle_injector`): per-stream metering
`ṁ = Cd·A·√(2ρΔp)` (Sutton; SP-8089) with dp = χ·Pc (χ default 0.20 per
H&H/SP-8089 stability floor, exposed per stream); auto mode solves A from the
cycle split, fixed mode evaluates supplied geometry; compressible branch for
gas/supercritical states with choke detection; two-phase states rejected.
Master variables verified against the corpus (see
`docs/PINTLE_DESIGN_EVALUATION.md`, cross-checked this session):
TMR = ṁ_r v_r/(ṁ_a v_a) (Hwang 2022 Eq. 1 / Son 2017 Eq. 4);
BF = N·w/(π·D_p) (Hwang Eq. 3); spray half-angle = atan2 of the momentum
resultant with deflector tilt (Heister/Cheng leading order); pintle diameter
anchor 0.30·D_chamber (packaging default, labeled); slots from area with
aspect-ratio + L/Dh floor; SMD via Hinze critical-We, breakup ≈15 d_jet
(Reitz–Bracco), d²-law + Priem–Heidmann vaporization-limited η_c* (general,
not pintle-specific — labeled).

**Complete physical-constraint (gate) inventory, all warn/fail thresholds
explicit in code:** mass-flow closure per stream; stiffness Δp/Pc; cavitation
K=(P_man−P_vap)/Δp ≥1.5; injection-state (choked/subsonic) report; orifice
L/D≥1 hydraulic-flip screen; manufacturing floors (slot width, web, annulus
gap ≥ min_feature 0.3 mm; blockage <1); concentricity tol/gap <0.10;
manifold maldistribution <10% (Newton network, informational screen value NOT
auto-charged to the pump budget — deliberate); spray-wall clearance (fail at
≥90°, warn <5% Lc, "Apollo oxidizer fan" gouging note); combustion-development
length vs chamber length (vaporization surrogate, warn-only per SP-8089);
target-TMR tracking; tip-radius ≤ pintle radius; face OD/edge distance;
face/tip thermal margin ≥1.2 (recirculation T_aw=0.8Tc + Dittus–Boelter series
circuit — right idea per Kang 2022, 0.8/0.2 fractions still screening
constants); chug decoupling χ≥0.20 (fail <0.10); chamber acoustic modes +
n–τ sensitive-band screen; mandatory cold-flow/hot-fire validation warning
(SP-8089). Feed ledger: per-stream required pump-outlet pressure
(Pc + injector Δp + manifold allowance + regen jacket + lines + margin), head
H=Δp/(ρg), Q, ideal power, NPSH — H&H SP-125/Sutton budget, gates
`feed_pump_pressure/capacity/npsh`.

Known, documented gaps (unchanged from the June eval): spray angle lacks Son
2017 saturation/We dependence; tip-thermal not yet coupled to BF/TMR (Kang
2022 quantifies both); movable-pintle throttling map is a schedule, not a
solved Cd(L_open) branch; recirculation fractions are defaults.

## 6. CAD assembly & STEP availability — VERIFIED WITH FINDINGS

How each artifact is made:

- **wall.step / jacket.step** — `export.py::export_step`: contour cleaned to a
  simple wire (chamber stations preserved so the L* volume survives), closed
  x–r profile revolved 360° in CadQuery/OpenCascade (mm at the kernel
  boundary) → true B-rep. Faceted AP214 fallback only if CadQuery is absent;
  `--require-brep` rejects it; `step_representation()` records which one you
  got. STL solids independently gated: watertight edge topology + signed
  volume vs exact revolved-profile volume (≤1e-5) before writing.
- **regen.step (`--regen-brep`)** — `regen_cad.py`: liner + full-count
  patterned ribs + end seals + jacket as ONE material solid (transform-
  patterned ribs, multi-shape fuse, optional plenum/port voids), then
  **re-imported and validated** (solid count, `isValid`, sliver fraction).
- **Pintle** — `injector_cad.py`: CadQuery named assembly (faceplate, hollow
  post, tip, annulus, slot network, manifolds, igniter, regen outlet, optional
  sleeve) + machined mode with real Boolean cuts and a `slot_cut_through` gate
  (cut must breach the pintle wall), inlet bosses fused before pocket cuts so
  the part stays one solid. Dimensions come from the SIZED hydraulic result;
  layout details (manifold offsets, velocity limits 8/20 m/s) are labeled
  preliminary layout rules.
- **Pump** — `pump_cad.py`: primitive triangle meshes (annular cylinders +
  swept blade boxes at fixed 12°/25° twists) parameterized by the meanline
  D2/D1/b2/Z, inducer D/hub/L, diffuser annulus, motor cylinder, battery box.

**STEP census of `builds/` (77 files):** 76 are true OpenCASCADE B-reps
(`MANIFOLD_SOLID_BREP`, OCC 7.8/7.9 headers) — every wall, jacket, and pintle
STEP shipped in June builds included. The single exception is a May artifact
(`builds/v019_20260511_174335/rao_nozzle.step`, pre-rename faceted fallback).

**Findings (the direct answer to "is STEP available for every CAD output"): NO
— two gaps, both in line with the modules' own disclaimers but worth fixing:**

1. **Pump parts ship STL-only by default** (`--pump-cad-format` default
   `stl`; every `pump/pump_parts/*.stl` in the sample builds, zero pump STEP
   anywhere in `builds/`). Requesting `step` produces the hand-rolled
   `_write_faceted_step` output, which (a) is a faceted mesh, not a B-rep, and
   (b) has AP214 conformance risks (bare `FACE` entities, malformed
   `GEOMETRIC_REPRESENTATION_CONTEXT(3)`, no unit context) — some importers
   will reject it. Recommended: route pump parts through the same CadQuery
   revolve/extrude path used everywhere else (impeller disk + hub are pure
   revolves; blades are extruded boxes — trivially B-rep-able).
2. **Shipped pump part meshes are not watertight**: `inspect_stl` on
   `builds/samples/latest_13kn_check` reports boundary edges on impeller (14),
   inducer (6), diffuser (8) — the 2π seam vertices don't weld exactly and
   overlapping primitives are not unioned. `_write_part` records diagnostics
   but does not gate on them (unlike `export_stl`, which hard-fails). The
   inverter/battery boxes, wall.stl and jacket.stl are watertight; `regen.stl`
   is non-watertight *by design* (visualization surfaces, labeled).

Also note `--cad` defaults to `none` (wall STEP is opt-in) and IPT is
correctly a conversion manifest, never a fake native file.

**CLOSURE (2026-07-02, pump CAD plan Phases 0–3 —
`docs/PUMP_CAD_IMPLEMENTATION_PLAN.md` STATUS head is the running record):**

1. **CLOSED.** The faceted pump pseudo-STEP writer is unreachable:
   `--pump-cad-format step/both` errors without CadQuery and otherwise
   routes to `raosim/pump_cad_brep.py` — true B-rep parts (hub/backplate
   revolve through the meridional-channel curves, log-spiral camber blades
   at the solved velocity-triangle angles, constant-lead helical inducer
   blades per SP-8052 §3.1.10, diffuser vane ring at the solved flow angle,
   volute casing + linear-area scroll, wear-ring land + balance holes),
   per-part STEP+STL, named per-role assemblies, and an export → re-import
   → `isValid`/volume gate identical in spirit to the wall path.  The
   default stays `stl` until CI installs CadQuery (the mesh writer remains
   the no-CadQuery fallback).  `--engine-assembly` additionally aggregates
   wall/jacket/pintle/pump/battery STEPs into `engine_assembly.step` with
   an interface.py bolt screen for the pump mount.
2. **CLOSED.** The 2π seam is welded (ring vertices computed once, closing
   segment reuses column 0) and `_write_part` now hard-fails non-watertight
   or non-positive-volume meshes (`--allow-open-pump-mesh` waives); every
   pump artifact in a fresh build passes `inspect_stl`
   (`tests/test_pumps.py` gates all emitted parts).

§7 constants have also moved: Z is no longer a fixed 6 — it comes from the
digitized SP-8109 fig. 16 minimum-blade-number chart snapped to a multiple
of the inducer count (SP-8052 §3.1.14); inducer solidity default is 2.5
(SP-8052 §3.1.15) and the inlet tip blade angle follows the §3.1.9
incidence-to-blade-angle ratio (0.425) on the §3.1.10 constant-lead helix;
Lee 2021 package-mass closure (motor 451.2 g / battery 985.6 g) and the
SP-8109 Ns envelope are pinned as benchmarks in `tests/test_pumps.py`.

## 7. Electric pump model — constants and their anchors

Chain: injector `FeedSystemLedger` (Q, Δp per stream) → `pumps.py` meanline →
`pump_cad.py` reference geometry. Verified numerically this session: ψ back-out
= 0.55 exactly; D2 = 2U2/ω; b2 = Q/(πD2φU2); Ns(nondim) 0.45 ≈ 1230 US-units —
inside SP-8109's flight-proven 450–2100 (verified verbatim: "Current
flight-proven centrifugal flow pumps range from 450 to 2100 in specific
speed"); Stodola slip σ = 1 − π sinβ₂/Z (0.779 both by code and by hand);
Euler head ≥ stage head enforced.

| Constant | Value | Anchor | Assessment |
|---|---|---|---|
| ψ = gH/U2² | 0.55 | SP-8109 eq. (7) verified in text; ">0.5 is high" note appears in SP-8109 | literature form, screening value |
| φ, φ_inlet | 0.08 / 0.12 | SP-8109 flow-coefficient practice (chart) | screening values, labeled |
| Ns target | 0.45 nondim | SP-8109 450–2100 US ✅ | in range |
| Tip-speed limit | 350 m/s | SP-8109 tip-speed/stress cautions (chart-level) | screening, user-overridable |
| Max head/stage | 2500 m | SP-8109 states ≈100,000 ft for high-speed LH2 pumps | repo value is a deliberately conservative cap, NOT the SP-8109 number — fine, but don't cite it as SP-8109 |
| Inducer hub ratio | 0.35 | SP-8052: "normally 0.2 to 0.4" ✅ verbatim | in range |
| Inducer solidity | 1.5 | SP-8052: "solidity of 1 or higher"; 2.0–2.5 for one class ✅ | low-mid of range |
| Nss | ω√Q/(g·NPSH)^0.75 | SP-8052 suction screen ✅ | correct nondim form; the >4.0 caution (~11k US) is conservative for inducer-equipped pumps (SP-8052 inducers reach 20k–40k US) |
| Z=6 impeller / 3 inducer / 8 diffuser | counts | SP-8109 fig. 16 (Z vs ψ), SP-8052 blade-number sections | chart-anchored screening |
| Stodola slip | 1−πsinβ/Z clamp [0.55,0.92] | classical Stodola; SP-8109 covers slip via Wiesner (ref. 56) | standard screen |
| Motor/inverter/battery densities 2500 W/kg, 15 kW/kg, 250 Wh/kg, η 0.90/0.96/0.95 | `SCREENING_DEFAULTS` | Lee 2021 / Spiller 2013 comparison cases only | technology assumptions, versioned & exported — correctly NOT claimed as literature constants |
| CAD package densities 2700/1200/750 kg/m³, box ratios, blade 0.025·D2, 12°/25° twists | `pump_cad.py` | none — explicit "package density placeholder / layout placeholder" | placeholders, honestly labeled |

The honest self-description holds: pump CAD is *labeled reference geometry for
layout and trade review*, "deliberately does not claim production-ready blade
surfaces"; the physics lives in the meanline + gates (tip speed, DN, seal
speed, hoop stress, NPSH, throttle-range vs system curve), not in the mesh.

## 8. Bottom line

- Bell contour, throat construction, chamber closure, Bartz/Sieder–Tate/SP-125
  wall equations, and the pintle hydraulic/TMR/BF/gate framework are
  **empirically rooted and correctly implemented**; every constant checked
  traces to the corpus or is explicitly labeled a screening assumption.
- The engine-level CAD chain (wall/jacket/regen-solid/pintle) is **true B-rep
  STEP with validation gates**, confirmed across all June builds.
- The pump CAD chain is the outlier: schematic primitive meshes, STL default,
  no B-rep STEP path, and shipped part meshes fail the repo's own
  watertightness standard. Constants driving it are correct meanline values;
  the geometry is a placeholder by design.

**Recommended next actions (priority order):**
1. Pump parts → CadQuery B-rep STEP (reuse `injector_cad` patterns), and gate
   `_write_part` STL output on `inspect_stl` watertightness like `export_stl`.
2. Fix or retire the faceted AP214 writer (bare `FACE`/context entities);
   at minimum emit `FACETED_BREP`-conformant faces + unit context, or make
   CadQuery a hard dependency of any STEP request.
3. Weld the 2π seam in `pump_cad`/`_annular_cylinder` (reuse first-column
   vertices) so even the schematic meshes close.
4. Rename `SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS` upper bound provenance (0.6–1.5
   is the quoted range; 2.0 is a repo allowance).
5. Pintle roadmap items already documented: Kang 2022 BF/TMR→tip-thermal
   coupling, Son 2017 spray-angle saturation mode, Cd(L_open) throttling.
