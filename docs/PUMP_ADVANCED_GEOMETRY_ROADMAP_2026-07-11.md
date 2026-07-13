# Pump advanced-geometry roadmap — 2026-07-11

Implementation analysis for the next tier of impeller/inducer/diffuser/volute geometry
features, grounded in `propulsion_texts/fuel_pump_design/` (primary anchors: NASA SP-8109
*Liquid Rocket Engine Centrifugal Flow Turbopumps* = `19740020848.pdf`; NASA SP-8052
*Liquid Rocket Engine Turbopump Inducers* = `19710025474.pdf`; plus the paper set cited per
item). Each item states: current state in code → literature basis (with the local file) →
math → CAD construction against the existing `pump_cad_brep.py` architecture → gates/tests.

## Implemented closure update (2026-07-12)

The original beta1, shaft/hub, blockage, and trapped-casing findings are now
closed in the bounded software sense: the meanline uses the exact net annular
eye; shaft/fit/root-wall constraints are upstream; beta1 is the converged
zero-incidence metal angle; net inlet/discharge blockage participates in the
solve; full inlet blades and downstream splitters are separate; blade thickness
tapers; casing wall includes a pressure screen; and the operating volute is a
rear body plus removable front cover split through the scroll centerplane with
a keyhole gasket/bolt/dowel layout. CAD hub mutation is forbidden and the
fidelity audit passes as an identity check. Full STEP round-trip and split/tool/
clamp/access gates are tested.

Still roadmap work: three-dimensional hub/mid/tip blade surfaces, LE/TE profile
and fillets, diffuser throat/camber/tongue and angular-momentum volute law,
selected joint hardware and flange FEA, shaft torque/axial retention features,
bearings/seals/lubrication, tolerances/thermal growth, rotordynamics, CFD,
cavitation/cold-flow testing, and measured maps. The split topology is not a
hardware-release claim.

Baseline architecture being extended: `pumps.impeller_blade_camber` integrates
dθ/dr = 1/(r·tanβ(r)) with β linear in r (log-spiral family); `pump_cad_brep.build_impeller`
turns that camber into a **planar ribbon (±t/2 normal offset) extruded axially**, trimmed by
the shroud-curve revolve — i.e., today's blade is 2-D (identical section at every z), constant
thickness, unfilleted, no splitters; the diffuser vane is a flat rectangular box at the solved
flow angle; the volute is a circular-section loft with A(θ) = A_exit·θ/2π on a constant-radius
centerline. Everything below is therefore additive, not a rewrite.

---

## 0. Prerequisite fix — camber inlet angle consistency (small, do first)

`CentrifugalPumpGeometry.inlet_blade_angle_deg` is `atan(φ)` (4.6° at φ=0.08) and is what the
CAD manifest exports, while `_velocity_triangle` computes the physical eye-tip relative flow
angle β1 = atan(c_m1/U1) (14.4° on the 13 kN baseline) and the loss model scores incidence
against it. Export the triangle β1 into `_reference_geometry`/the CAD manifest (keep the old
value as `screening_inlet_angle_deg` for provenance) so the cut blade meets the meanline's own
inlet flow. Test: manifest β1 == triangle β1; camber wrap changes accordingly.

## 1. 3-D blade loading (hub-vs-shroud angle variation) — the enabling feature

**Current:** one camber line, one β(r); blade surface is the same at hub and shroud because
the ribbon is extruded in z.

**Literature:** SP-8109 impeller-design practice develops the blade on hub and shroud
streamlines separately (blade-surface velocity-gradient limits are exactly what fig. 16's
minimum-blade-number carpet encodes); Chen, Prueger, Chan & Eastland (Rocketdyne),
*On the Use of a 3-D Navier-Stokes Solver for Rocket Engine Pump Impeller Design*
(`chen1992.pdf`) is the corpus's direct statement of hub-to-shroud loading control as the
design lever for rocket impellers; Lin (`lin1993.pdf`) is the CFD-validation companion;
Bellary & Samad (`bellary2014.pdf`) and Jaiswal et al. (`jaiswal2021.pdf`) demonstrate
β-distribution optimization on centrifugal impellers; Hong et al. (`hong2012.pdf`) carry
separate hub/tip angles through a rocket turbopump design.

**Math:** the inlet angle must vary along the leading edge because U(r) = ωr while c_m1 is
~uniform: tanβ1(r) = c_m1/(ωr) ⇒ β1_hub > β1_tip (computed on the shipped 13 kN fuel line:
β1_tip = 14.4°, β1_hub = 36.2° at the solved hub/tip ratio 0.35 — a 22° twist the current
single-angle blade cannot represent). Define camber per streamline: for streamline s ∈ {hub,
shroud} (later: mid), θ_s(m) integrated along the *meridional* coordinate of that streamline —
generalize `impeller_blade_camber` from dθ/dr to dθ/dm = 1/(r(m)·tanβ_s(m)), sampling r(m)
from the existing `_meridional_channel` hub/shroud curves (they already carry exact endpoint
areas). β_s(m) blends β1(r_s,inlet) → β2 (β2 common at exit; both streamlines end at r2).

**CAD:** replace extrude-and-trim with a **ruled loft between two ribbons**: build the hub
ribbon on the hub curve (points (r(m)cosθ_hub, r(m)sinθ_hub, z_hub(m)) offset ±t/2) and the
shroud ribbon on the shroud curve, then `cq.Solid.makeLoft`/`sweep` with ruled=True between the
two closed sections (or loft 3–5 intermediate sections for smoothness). Keep the existing tip
cylinder intersect and root overlap into the backplate (the current tangency-avoidance
tricks carry over).

**Gates/tests:** wrap angle per streamline exported; loft validity + watertight + re-import
(existing harness); new hydraulic gate: β1(r) matches atan(c_m1/ωr) within tolerance at hub,
mid, tip; incidence loss in `_hydraulic_meanline` evaluated per-streamline.

## 2. Thickness distribution (with leading-edge shaping)

**Current:** constant ±t/2 normal offset (`_camber_ribbon_xy`), square LE/TE; inducer blade is
a swept rectangle (blunt LE).

**Literature:** SP-8109 fillet/finish section (extracted verbatim today): "It is recommended
that the leading-edge cross section be a 2:1 to 3:1 ellipse"; SP-8052 §2.1.6 gives inducer
leading-edge practice (0.005-in class edges, suction-side fairing per ref. 86, centerline
fairing option, "all sharpening" kept smooth — §2.x fairing passages) and trailing-edge
fairing guidance; Shigemitsu et al., *Influence of Blade Outlet Angle and Blade Thickness on
Performance and Internal Flow Conditions of Mini Centrifugal Pump* (`Influence of Blade
Outlet Angle and Blade Thickness….pdf`) quantifies why constant thick blades hurt at this
size class (the repo's 0.4 mm floor already cites it); `energies-11-02588.pdf` (blade
leading-edge shape → cavitation in a centrifugal impeller) is the cavitation-side anchor.

**Math:** thickness law t(m̂) along normalized camber length: elliptical LE over the first
ℓ_LE = k·t_max (2:1–3:1 ellipse ⇒ k ∈ [2,3]) blending to t_max, linear or constant midchord,
faired TE (t_TE ≥ manufacturing floor 0.4 mm unless cast). Optional spanwise taper
t_hub→t_tip (structural: root carries the bending; SP-8052 cant discussion). Keep total
blockage Z·t/(2πr·sinβ) reported at eye and exit — blockage is why SP-8109's blade-number and
b2 practice cares about t.

**CAD:** `_camber_ribbon_xy` already walks the polyline with normals — replace `half = t/2`
by `half_i = t(m̂_i)/2` and close the outline with a semi-elliptical LE arc (sample the
ellipse, not a chamfer) and TE arc. Works unchanged for both the 2-D blade and item 1's
per-streamline ribbons. For the inducer, sweep a faired section (rectangle with elliptical
nose) instead of `rect()`, or post-cut a suction-side wedge per SP-8052's suction-side
fairing description.

**Gates/tests:** LE ellipse ratio in [2,3]; min thickness ≥ floor; eye blockage reported and
screened; mesh watertight; area/volume deltas vs constant-t documented.

## 3. Splitter blades

**Current:** none (full blades only, Z from SP-8109 fig. 16 snapped to a multiple of the
inducer count).

**Literature:** SP-8109 documents the practice directly — the experimental F-1 fuel impeller
"with six full blades and six splitters" (fig. caption, printed p. 32 region of
`19740020848.pdf`) — and, for volutes, splitter vanes/multiple tongues for radial-thrust
reduction (fig. 28 discussion, §"Splitter vanes or multiple tongues…"). Yang et al.
(`yang2019.pdf`) is a rocket-engine turbopump with Z_main + Z_splitter impeller and a vaned
diffuser, and quantifies the rotor-stator-interaction pressure-pulsation consequences
(main-blade vs total blade passing frequencies) — the reason splitter count/clocking matters.
Tani et al. (`tani2008.pdf`) and the Dnipro cavitation-parameter study (`191-Article
Text….pdf`) tie splitter/blade layout to cavitation behavior; `TOMEJ-2-75.pdf` is the generic
parametric context.

**Why:** at the eye, Z=12 full blades at t=0.4 mm on a ~9 mm-radius eye is severe blockage
and cavitation exposure; the classic resolution is Z_full = 6 full + 6 splitters starting
mid-passage — full exit solidity (slip, loading) at half the inlet blockage.

**Math:** splitter = same camber law truncated at r_LE,split = r1 + f·(r2 − r1) (start
f ≈ 0.4–0.6 of the *meridional* length in practice; make f a spec field), circumferentially
offset half a main-blade pitch (π/Z_main). Slip/loading bookkeeping: at the exit both families
are present, so keep Stodola with Z_eff = Z_main + Z_split (both reach r2); fig.-16 minimum-Z
check should be satisfied by Z_eff at the discharge but by Z_main at the inlet — expose both
in `sp8109_min_blade_count` usage. Blockage screens evaluated at eye with Z_main only.

**CAD:** trivial under the current builder — generate a second `impeller_blade_camber` list
sliced at r ≥ r_LE,split, ribbon it (item 2 thickness law, elliptical LE), extrude/loft,
rotate by π/Z_main + k·2π/Z_main. All existing trims and fuses apply.

**Gates/tests:** Z_eff ≥ fig.-16 minimum at (ψ, φ2); eye blockage(Z_main) < exit
blockage(Z_eff) both under limits; slip recomputed with Z_eff and Euler-head margin re-gated;
interference audit between splitter and main-blade solids (must fuse, not graze).

## 4. Blade lean (cant)

**Current:** none — the extrusion direction is the axis, so blade sections stack radially
(zero lean by construction); item 1's loft removes that constraint.

**Literature:** SP-8109 (shroudless-impeller discussion, ~printed p. 9447 text region):
unshrouded impellers see cyclic pressure loading "±30 percent of the steady-state" and the
"loading may be minimized by use of radial-element blades" — i.e., for open/semi-open
impellers, *zero lean is the load-relieving choice*, which makes lean an explicitly *justified
deviation*, not a default. SP-8052 §2.1.8 (extracted verbatim today): "Canting of the blade is
done for mechanical reasons only. At high blade loadings, the blade is canted forward to
partially counterbalance hydrodynamic and centrifugal bending forces."

**Math:** lean angle λ(m): tangential displacement of the section stack,
Δθ(m, z) = z·tanλ(m)/r(m) added to θ_s between hub and shroud (positive = pressure-side lean
into rotation = "forward cant" in SP-8052's sense). First implementation: constant λ, bounded
|λ| ≤ ~15° with the SP-8052 mechanical rationale recorded in the manifest.

**CAD:** free once item 1 lands — lean is just a θ-shift between the hub and shroud ribbons
before lofting. Keep λ = 0 the default (radial elements, per SP-8109's semi-open guidance).

**Gates/tests:** λ recorded with justification string; root bending index (∝ blade height ×
loading × tanλ) reported; loft validity.

## 5. Blade sweep (leading-edge shaping in the meridional plane)

**Current:** none — impeller LE is the radial line at r1; inducer LE is square.

**Literature:** SP-8052 §2.1.7 *Blade Sweep* (extracted verbatim today): "Sweeping back and
rounding off the radial contour of the leading edge has resulted in increases of 10 to 25
percent in suction specific speed (refs. 35 and 36). Structurally, the sweepback removes the
corner flap and redistributes the blade load… The blade wrap is reduced, but the reduction can
be allowed for… by a slight increase in axial length. On shrouded inducers, the leading edge
is usually swept forward to avoid sharp corners and to provide fillets where the blade meets
the shroud." `energies-11-02588.pdf` covers LE-shape → cavitation for the impeller;
`Mishra_2015…pdf` (LOX booster inducer CFD) is the corpus's inducer-CFD cross-check.

**Math:** define the LE as a curve in the meridional plane: m_LE(span) with the tip cut back
by Δm_tip (backsweep) — e.g., linear or circular-arc LE from hub (unmoved) to tip (moved
downstream Δm_tip ≈ 0.1–0.3 of tip chord), then round per item 2's ellipse. Wrap deficit
compensated by extending axial length (SP-8052's own instruction) — add the check.

**CAD:** inducer: cut the swept helical blade with a ruled "sweep surface" solid (a revolve of
the LE meridional curve ± generous tangential extent) — one robust Boolean, same pattern as
the existing shroud-cutter trim. Impeller: start the hub ribbon at m=0 and the shroud ribbon
at m=Δm_tip (item 1 loft handles the rest).

**Gates/tests:** Nss screen note (expected direction, not a claim); wrap/axial-length
compensation check; watertight/valid after the extra Boolean.

## 6. Fillets at hub and shroud junctions

**Current:** none anywhere (blade-root corners are sharp; the audit's machinability review and
SP-8109 both flag this).

**Literature:** SP-8109 *Fillet Radii and Surface Finish* (extracted verbatim today): "The
fillet radii at the blade-to-hub, blade-to-shroud, and blade-to-backplate junctions should be
equal to 1.5 times the blade thickness. This ratio will reduce the stress-concentration factor
in the area to a value approximating 1," plus the 125→63 µin rms finish and shot-peen
guidance. SP-8052's forward-swept shrouded-inducer note (§2.1.7) exists precisely "to provide
fillets where the blade meets the shroud."

**Math:** r_fillet = 1.5·t (local t if item 2 lands; at a minimum 1.5·t_root). That is the
whole law — the value is structural, not hydraulic.

**CAD:** two options, in order of robustness: (a) OCC edge fillets — after fusing blades to
the hub revolve, classify edges by adjacency (blade face ↔ hub/backplate face) and apply
`fillet(r)` with a retry ladder (r, r/2, r/4, report achieved) because OCC fillets fail on
near-tangent spiral edges; (b) if (a) is unreliable, add a swept fillet bead: sweep a
quarter-round profile along the blade-root camber curve on each side and fuse (deterministic,
watertight by construction). Record `fillet_radius_achieved_m` per junction; never silently
skip.

**Gates/tests:** achieved r ≥ 1.5·t or an explicit degradation note; volume increase sanity;
re-import validity; mesh watertightness (fillets are the classic tessellation breaker — keep
the existing hard STL gate).

## 7. Diffuser vane CAD (real profiles instead of flat plates)

**Current:** `build_diffuser_ring` places flat rectangular boxes at the solved absolute flow
angle between two washer plates; vane thickness is a pitch ratio; throat area exists only as
the meanline number (`_diffuser_volute_geometry.throat_area`), never enforced in metal.

**Literature:** SP-8109 vaned-diffuser practice ("A vaned diffuser provides volute
flow-matching… reduced volute velocity… Vaned diffusers are also used to obtain maximum pump
efficiency", §2.3.x; diffuser-vane/volute matching cautions around the tongue); Yang 2019
(`yang2019.pdf`) for vane-count/RSI clocking constraints (Z_d chosen against Z_main+Z_split
blade-passing harmonics); Hong 2012 carries a designed vaned diffuser in a rocket turbopump.

**Math:** vane camber = log spiral r(θ) = r3·exp(θ·tanα) (constant flow angle is exactly the
log-spiral streamline of a free vortex with radial through-flow — the same family the impeller
camber already uses): inlet at r3 = r2 + semi-vaneless gap (keep the existing
rotor-stator clearance, but expose r3/r2 as a spec value ~1.05–1.10; Yang 2019's RSI results
are the rationale), camber from α3 (solved absolute flow angle + small incidence) to α4
(chosen for the volute's inlet swirl), Z_d vanes with LE ellipse and wedge-diffuser passage.
Enforce the meanline throat: passage throat width a_th between adjacent vane suction/pressure
surfaces solves Z_d·a_th·b3 = throat_area (iterate α or vane thickness until matched).
Vane number rule: gcd(Z_d, Z_main) = 1 preferred (no common harmonics), Z_d ≠ Z_main ± 1
caution — report the blade-passing frequency table per Yang 2019.

**CAD:** same two side plates; vane solid = ribbon between log-spiral suction/pressure curves
(item 2 thickness machinery reused verbatim in the (r,θ) plane), extruded b3, intersected
with the annular band — a drop-in replacement for the box. Emit `diffuser_throat_area_cad`
measured on the actual solid (section the passage at the geometric throat) alongside the
meanline value.

**Gates/tests:** CAD throat area within tol of meanline throat; gcd rule reported; angles
match the solved triangle within incidence allowance; interference/fuse validity; the existing
casing pocket keeps clearing the new ring (clearance audit already exists).

## 8. Volute generation (correct area law, spiral centerline, tongue, exit diffuser)

**Current:** `build_volute_casing` lofts circular sections with A(θ) = A_exit·θ/2π centered on
a *constant-radius* circle, subtracts scroll/outlet/inlet/pockets from a material envelope,
and `audit_volute_flow_passage` proves single-connected void topology. No tongue geometry, no
angular-momentum sizing, no exit-diffuser cone; the casing is one piece (not manufacturable
subtractively — flagged in today's audit).

**Literature:** SP-8109 §2.4.1.3 VOLUTE (printed p. 47, rendered and read verbatim today):
"The object of volute design is to provide a distribution of cross-sectional area with respect
to wrap angle that will yield a constant impeller discharge static pressure at the design
point of the pump; the radial load on the shaft and the impeller vibrations are thereby
minimized." §2.4.1.3.1: "Two methods are in use for sizing the volute cross-sectional area:
constant moment of momentum, and constant mean velocity" — constant moment of momentum
("fluid tangential velocity … inversely proportional to the radius") was used on Titan I/II
housings and the J-2S Mark 29 fuel pump "which experienced very light radial bearing loads";
constant mean velocity (area ∝ wrap angle) "was developed as a simplification," and "the
unsymmetrical pressure (with its associated radial hydraulic forces upon the impeller) …
has been found to be higher in designs based on the constant-mean-velocity method." The
conical exit diffuser "will operate efficiently when the included angle for circular cross
sections is between 7° and 9°; for square cross sections, 6°; and for two parallel walls,
11°." Tongue: "The inlet angle of the volute tongue is designed for zero incidence angle at
the design flow … Stable flow is achieved by fairing one diffuser vane into the volute tongue
or by leaving a large clearance between the vane discharge and the tongue." Elsewhere:
splitter vanes / multiple tongues / double-outlet volutes reduce radial thrust over a wide
flow range (fig. 28 discussion). Cross-checks in the corpus: the Firatoglu
stages/outlet-width study (`Investigation-of-the-Effect-of-the-Stages-Number…pdf`), Li 2024
LH2 integrated inducer+impeller pump (`Li_2024…pdf`), `20180003628.pdf` (Pinera patent
packaging variant). **Note: the repo's current A(θ) = A_exit·θ/2π is precisely SP-8109's
"constant mean velocity" simplification — legitimate, but the document itself attributes
higher radial loads to it; the upgrade below implements the preferred method.**

**Math (first-principles upgrade, still meanline):** conservation of angular momentum in the
scroll: r·c_u = r2·c_u2 = const. The section at angle θ must pass Q(θ) = Q·θ/2π:
Q(θ) = ∫_A (r2·c_u2 / r) dA over the section. For a circular section of radius a(θ) centered
at r_c(θ): ∫dA/r = 2π(r_c − √(r_c² − a²)), so a(θ) solves
r2·c_u2·2π(r_c − √(r_c² − a²)) = Q·θ/2π — closed-form invertible for a given r_c; the
centerline then *spirals*, r_c(θ) = r4 + a(θ) + wall standoff (sections grow outward, not
into the diffuser). Keep both SP-8109 §2.4.1.3.1 methods selectable,
`volute_law ∈ {"constant_moment_of_momentum", "constant_mean_velocity"}`, defaulting to
constant moment of momentum (the document's radial-load-preferred method; the current linear
schedule remains as the labelled simplification). Tongue: per §2.4.1.3 verbatim, tongue inlet
angle set for **zero incidence at design flow** — angle = atan(c_m/(r2·c_u2/r_t)) at the
tongue radius, tip radius ≥ manufacturing floor, and either fair one diffuser vane into the
tongue or keep a large vane-discharge clearance (the two stability options the section
gives); report off-design tongue incidence as the radial-load screen. Exit: replace the plain
cylinder bore with a conical diffuser from A(2π) to the outlet-port area at a **7–9° included
angle for circular sections** (6° square, 11° parallel-walls — §2.4.1.3 verbatim), length
solved from the areas and angle.

**CAD:** same loft machinery with a(θ) and r_c(θ) arrays; tongue = blend the first section
into the exit-diffuser wall with a ruled patch + tip-radius cylinder (one extra Boolean);
conical outlet = `makeCone` fluid + boss. Add a **split line**: build the casing as two
solids (scroll body + front cover through the plane of the diffuser pocket, bolt circle via
the existing `interface.py` pressure-bolt screen) so the internal void is actually reachable
by machining — this also resolves today's machinability finding; keep the one-piece export as
a reference option for casting.

**Gates/tests:** numerically integrate ∫(r2·c_u2/r)dA over each lofted section and assert
Q(θ) linearity within tol (this is the honest version of the current linear-area assumption);
tongue tip radius ≥ floor; diffuser included angle in [7°, 11°]; two-solid split passes the
existing interference/clearance/re-import audits and the connected-void audit still passes
with the tongue in place; radial-load screen: report the SP-8109 asymmetry caveat at
off-design Q.

## 9. Related items unlocked by the above (brief)

Shrouded-impeller option (SP-8109 selection basis; fig. 16 carpet is *for* shrouded wheels —
add front-shroud revolve fused over item 1's lofted blades, wear ring moving to the shroud
OD, fillets per item 6 at the shroud junction, and note SP-8052's forward-sweep-for-fillets
practice); inducer suction-side fairing (SP-8052 §2.1.6/ref. 86) on top of item 2's section
machinery; diffuser/volute splitter vanes and double-outlet volute for radial thrust
(SP-8109 fig. 28) reusing item 7's vane builder; blade-surface finish/shot-peen callouts
(SP-8109 verbatim: cast 125 µin → hand-finish 63 µin in high-stress areas) belong in the
manufacturing report, not the solid.

## 10. Recommended order and effort

0 (β1 export fix — hours) → 1 (per-streamline camber + ruled loft — the enabler, ~1–2
sessions incl. tests) → 2 (thickness/LE law — 1 session, shared machinery) → 3 (splitters —
short, high payoff for the Z=12 eye-blockage problem) → 7 (log-spiral diffuser vanes +
throat enforcement) → 8 (volute law + tongue + split casing — closes the machinability
finding) → 6 (fillets — after geometry stabilizes, OCC-fragile) → 5 (sweep) → 4 (lean stays
default-zero with SP-8109's radial-element rationale) → 9 (shrouded option last; it flips
wear-ring/fillet topology). Every item keeps the existing gate pattern: solved-manifest
inputs only (no CAD-invented dimensions), export → re-import → isValid/volume, watertight
STL, interference/clearance audits, `requires_meanline_resolve` semantics untouched, and
`cold_flow_release_ready`/`hardware_qualified` remain external-evidence-gated.
