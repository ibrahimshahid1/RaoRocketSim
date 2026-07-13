# Pump geometry implementation plan — 2026-07-13

Updates `docs/PUMP_ADVANCED_GEOMETRY_ROADMAP_2026-07-11.md` after (a) the discrepancy
review of that roadmap and (b) a code+literature re-verification performed 2026-07-13.
Primary anchors remain `propulsion_texts/fuel_pump_design/`: NASA SP-8109 *Liquid Rocket
Engine Centrifugal Flow Turbopumps* (`19740020848.pdf`) and NASA SP-8052 *Liquid Rocket
Engine Turbopump Inducers* (`19710025474.pdf`). All verbatim quotes below were re-read
from the PDF text layers on 2026-07-13; discrepancy-review numeric claims were re-checked
against the current working tree (see §1 caveat).

---

## 1. Verified starting state (uncommitted working tree on `ba8fcb6`)

The 2026-07-12 working-tree drop already closed a large part of both the roadmap's item 0
and the discrepancy review's items 1–5. Verified directly in code and by a live probe
(`size_electric_pumps` on a fuel line with the baseline's exact Q = 1.9688e-3 m³/s,
H = 494.2 m — rpm reproduces 56 252.9 exactly):

| Finding (roadmap / discrepancy review) | Status in working tree |
|---|---|
| β1 export is `atan(φ)` = 4.574° | **Closed.** `_velocity_triangle` exports `inlet_relative_flow_angle_deg`, `inlet_blade_metal_angle_deg`, `inlet_incidence_deg` (zero at design by declared assumption); old value kept as `legacy_screening_inlet_angle_deg` (`pumps.py`, `CentrifugalPumpGeometry`). |
| Eye/channel solve is the true prerequisite | **Closed.** `_solve_annular_eye_and_shaft` (`coupled_annular_eye_shaft_bisection_v1`) + net-annular-eye triangle (`centrifugal_meanline_annular_eye_v2`) + `pump_annular_eye_continuity_<role>` residual gate. Probe: D1 20.807 mm (vs 14.918 mm in the stale 07-08 build), β1_tip = 6.84° on the enlarged eye. |
| CAD enlarges hub, invalidating exported streamline | **Closed by policy.** Shaft/fit/root-wall constraints are upstream in the solve; CAD hub mutation forbidden; `audit_meanline_geometry_fidelity` is an identity check. |
| `cm2/cm1 = 0.585` guaranteed by default coefficients | **Closed.** Probe: cm1 7.354, cm2 7.510 → cm2/cm1 = 1.021. |
| Incidence loss is an angle-range bucket | **Closed (semantics).** Loss now scored on `inlet_incidence_deg` = metal − flow. Note it is zero at design by construction; the loss term only activates off-design or if a deliberate non-zero metal angle is ever set. |
| Blade t = 1.275 mm from `0.04·D2` | **Stale.** `blade_thickness_ratio` default is now 0.012 → floor-limited t = 0.4 mm, separate `inlet_blade_thickness` = 0.2 mm, linear LE→exit taper in `_camber_ribbon_xy`. |
| 12 full blades, 116–146 % eye blockage | **Stale.** Meanline splits `inlet_blade_count` + `splitter_blade_count` (probe: 4 + 8), and blockage is screened per station with separate limits: inlet 0.103 vs limit 0.20, exit 0.113 vs limit 0.15 — exactly the "each station needs its own limit" correction, already in. |
| Splitters only proposed | **Partially closed.** Meanline bookkeeping + CAD splitter solids exist (`build_impeller` slices the camber at `splitter_start_radius_fraction` = 0.55). Residual gaps in §3-P2. |
| One-piece volute not machinable | **Closed.** `build_split_volute_casing` + `audit_split_casing_manufacturability` (rear body + front cover, gasket/bolt/dowel layout). |
| Diffuser vane = flat box; throat never enforced in metal | **Open.** `build_diffuser_ring` still places rectangular boxes; `_diffuser_volute_geometry` throat is still `1.15·Q/cm2` on a radial-area basis. |
| Volute law = `A_exit·θ/2π`, no tongue, no exit cone | **Open.** `build_volute_casing` still lofts the labelled constant-mean-velocity simplification on a constant-radius centerline; scroll terminal area = outlet-port area. |
| 3-D blade (hub/shroud twist), LE/TE profile, sweep, lean, fillets, shroud | **Open.** `impeller_blade_camber` is still dθ/dr with β linear in r; one planar ribbon extruded axially. `blade_envelope` exports a single inlet angle (no hub-side β1 yet). |

**Caveat:** probe numbers above come from a synthetic feed line with matched Q/H (density
810 kg/m³, guessed viscosity/NPSH); rpm matching confirms the geometry chain, but the
authoritative baseline numbers must be re-derived by re-running
`examples/cli/test_13kn_sealevel_regen_allstl.args` on the host (sandbox 45 s limit) —
this is P0 below. The discrepancy review's specific figures (16.286°/39.853°, 116–146 %
blockage, 1.386 mm vs 12.52 mm throat, 46.5 mm² vs 542.7 mm² terminal area) were computed
against the pre-drop code; the *directions* all remain valid for the still-open items and
are re-derived per item below.

---

## 2. Corrections adopted from the discrepancy review

These override the corresponding parts of the 2026-07-11 roadmap.

**C1 — Stacking convention before any 3-D blade work.** Per-streamline camber uses the
channel's common paired parameter u ∈ [0,1] (same u indexes hub and shroud points of
`_meridional_channel`, which already emits paired `hub_curve`/`shroud_curve` samples):

    dθ_s/du = (dm_s/du) / (r_s(u)·tanβ_s(u)),   s ∈ {hub, shroud, later mid}

Independent hub/shroud integration produces tens of degrees of trailing-edge clocking
difference (review estimate ~60–68°), so the surface is only defined once a stacking rule
is declared: stack at the **trailing edge** (θ_hub(1) = θ_shroud(1) = 0), which makes the
exit edge axial, matches the common-β2 exit, and gives "zero lean" a concrete meaning.
Record `stacking: trailing_edge` in the manifest. Lean (C4) is then a deliberate
θ-offset distributed over *local span height* (distance along the LE/section between hub
and shroud points at equal u), never global z.

**C2 — No closed-nonplanar-ribbon loft.** The 2026-07-11 CAD proposal (loft between two
closed hub/shroud ribbons) produced invalid solids in CadQuery 2.7 tests on the saved
13 kN curves (both ruled and smooth): the closed outlines are nonplanar. Adopt instead:
(a) **planar transverse sections lofted along the chord** — at each u build the planar
quad/aerofoil section spanning hub→shroud (points from both streamlines plus thickness),
loft the ordered planar wires; or (b) **surface sewing** — build pressure and suction
faces as `interpPlate`/ruled surfaces over the (u × span) point grids, cap LE/TE/root/tip,
sew into a solid. Prototype (a) first; fall back to (b). Thickness offsets are applied
along the local blade-surface normal in 3-D — `_camber_ribbon_xy` (2-D polyline normal)
is retired for the impeller once this lands, kept for the legacy 2-D path.

**C3 — Thickness semantics: three quantities, three gates.** Report and gate separately:
`core_thickness` (mid-chord structural t, floor 0.4 mm per the existing manufacturing
basis), `le_nose` (impeller: 2:1–3:1 ellipse ⇒ nose radius t/(2k), k ∈ [2,3]; inducer:
SP-8052 practice below — near-sharp, t/100 nose or 0.13–0.25 mm edge), `te_thickness`
(≥ floor unless cast). A blanket "every sampled thickness ≥ 0.4 mm" gate would forbid the
elliptical nose whose width → 0 at the tip; do not add one. Add the **pre-CAD pitch
gate** at eye, splitter-LE, and exit stations:

    t/sinβ + clearance < 2πr/Z(station)

with Z(station) = mains at the eye, mains+splitters at exit — this is the existing
blockage-fraction screen restated per blade passage; keep both forms.

**C4 — Splitter semantics.** Keep `inlet_blade_count` (mains), `splitter_blade_count`,
and an explicit `effective_exit_blade_count` in the manifest. Move the splitter start
from a *radius* fraction to the **common meridional parameter u = f_split** (with the
quarter-ellipse channel these differ materially near the eye bend). Splitters and mains
must **not** fuse blade-to-blade: audit zero blade↔blade intersection volume, positive
blade↔hub fuse, and minimum passage clearance ≥ pitch-gate clearance. Fig.-16 minimum-Z
and Stodola with Z_eff are *engineering extrapolations* for partial blades — label them
(`blade_number_basis: sp8109_fig16_extrapolated_splitters`), keep fig.-16 satisfied by
Z_eff at discharge and the eye pitch gate by mains only. Per-station blockage limits (in
code) replace the review-rejected "eye < exit" ordering.

**C5 — Diffuser throat projection.** The meanline `throat_area` is currently a *radial*
area (Q/cm basis). The vane-passage throat is measured normal to the flow: with passage
normal width a_n and vane span b3,

    Z_d · a_n · b3 = A_radial · sinα₃

(the review's baseline evaluation: a_n ≈ 1.386 mm, vs 12.52 mm if the sinα projection is
dropped — an 9× error). A log spiral is exact only for constant α; for α3→α4 camber,
integrate dθ/dr = 1/(r·tanα(r)) numerically (same quadrature as
`impeller_blade_camber`). Size the throat with SP-8109's stated method: velocity at the
throat mean radius from **conservation of momentum** from the impeller discharge (fig. 25
ratio), then A_throat = Q/(velocity·(1 − vane blockage)). Emit
`diffuser_throat_area_cad` sectioned on the actual solid and gate |CAD − meanline| ≤ tol.
Fix the passage audit: today's connected-void audit treats the whole diffuser pocket as
fluid (measured ~148 mm² open radial area passing against a 301.5 mm² meanline target on
the flat-vane build); after the vane rebuild the audit must measure the *vaned* fluid
volume and its throat section.

**C6 — Vane count rule.** Replace the roadmap's gcd/±1 rule (not in the corpus) with
SP-8109's verbatim practice: number of diffuser vanes "usually, the prime number nearest
to the number of impeller blades", vane width 90–100 % of impeller tip width, and
tongue/diffuser-vane separation "radius ratio greater than 1.05 or the tongue should
virtually touch the diffuser" (§3.4.1.2/3.4.1.3 region, `19740020848.pdf`). A prime Z_d
near Z_eff avoids common blade-passing harmonics, which is what the gcd rule was after.
Report the main-blade and total blade-passing frequencies in the manifest; deeper RSI
clocking claims are dropped until an RSI source is added to the corpus (§4).

**C7 — Volute integral, dataflow, and exit cone.** Constant moment of momentum with
K = r·c_u held at the **volute inlet state**: K = r4·c_u4 (diffuser discharge) when a
vaned diffuser is fitted, r2·c_u2 only in the vaneless/volute-only branch. Use the
numerically stable form

    ∫_A dA/r = 2πa²/(r_c + √(r_c² − a²)),  circular section, centerline radius r_c

and for the outward-growing scroll r_c = R0 + a the closed form a(θ) = δ + √(2·R0·δ),
δ = Qθ/(4π²K). Export the missing manifest fields: r4, c_u4, α4, Q_design, K, and
volute_law. Keep both SP-8109 §2.4.1.3.1 laws selectable
(`constant_moment_of_momentum` default, `constant_mean_velocity` as the labelled
simplification) — the design-criteria section is prescriptive: "The constant-moment-of-
momentum method adjusted for friction loss will produce minimum design-point impeller
radial loads" (§3.4.1.3.1). Terminal scroll area must **not** silently equal the
outlet-port area: insert the conical exit diffuser from A(2π) to the port area at
**7°–9° included angle for circular sections** (6° square, 11° two-parallel-walls — the
11° figure is *only* the parallel-wall case; the circular gate is 7–9°, not 7–11°).
Review estimate on the old baseline: A(2π) ≈ 46.5 mm² vs 542.7 mm² port ⇒ cone length
~118–152 mm — expect a packaging re-solve (shaft span, casing envelope, interface
clearances) when this lands; treat that as part of the item, not a surprise.

**C8 — Tongue.** Inlet angle for zero incidence at design flow: angle = atan(c_m,t/c_u,t)
with c_u,t = K/r_t at the tongue radius; tip radius ≥ manufacturing floor; stability by
either fairing one diffuser vane into the tongue or a large vane-discharge clearance
(both are SP-8109's options, plus the r ratio > 1.05 rule in C6). Report off-design
tongue incidence as the radial-load screen.

---

## 3. Implementation sequence

Every item keeps the repo gate pattern: solved-manifest inputs only, export → re-import →
`isValid`/volume, watertight STL, interference/clearance audits,
`requires_meanline_resolve` semantics untouched, `cold_flow_release_ready`/
`hardware_qualified` remain external-evidence-gated.

**P0 — Commit and host-verify the current tree (hours).** The eye-solve/splitter/split-
volute work is uncommitted on `ba8fcb6`. Run the full host suite + the 13 kN benchmark
args, snapshot the authoritative new baseline numbers (β1_tip on the solved annular eye,
D1, blockages, splitter counts), update `TEST_PROMPT.md` reference snapshot, commit.
Nothing below starts against an unpinned baseline.

**P1 — Residual meanline gaps (small).**
- Rename/alias the provenance field to `legacy_atan_phi2_deg` (review's naming; keep the
  current name as a deprecated alias for one release).
- Export the **hub-side** annular β1 alongside the tip value (both from the solved eye:
  tanβ1(r) = c_m1_net/(ωr) at r_hub and r_tip) so the twist the 3-D blade must represent
  is a manifest fact before any CAD consumes it.
- Splitter start → common-u fraction (C4 field change only; CAD still slices by index).
- Add the explicit per-station pitch gate (C3) beside the blockage fractions.
- Tests: manifest hub/tip β1 match atan(c_m1/ωr) at the solved radii; pitch gate trips on
  a constructed over-bladed case.

**P2 — Common-u camber surface with declared stacking (the enabler).**
`impeller_blade_camber_3d(channel, beta1_hub, beta1_tip, beta2, stacking="trailing_edge")`
integrating C1 on both streamlines; β_s(u) blends β1(r_s) → β2 (common exit). Emit wrap
angle per streamline and the TE-stacked θ grids. Pure-math module + tests first (no CAD):
wrap monotonicity, TE clocking = 0 by construction, hub/tip β checks at u=0.

**P3 — Blade solid from planar sections (C2).** Loft ordered planar transverse sections
(option a), fallback surface-sew (option b). Constant thickness first (0.4 mm core, blunt
ends) to isolate loft validity; keep tip-cylinder intersect + root overlap tricks.
Gates: isValid, watertight STL, re-import, volume vs 2-D blade documented, blade↔hub fuse
positive.

**P4 — Thickness and edge profiles on the section generator (C3).** t(û) law: elliptical
LE (k ∈ [2,3]) → core → faired TE for the impeller; SP-8052 near-sharp nose for the
inducer (sweep a faired section instead of `rect()`, or post-cut the suction-side wedge —
§2.1.6's suction-side fairing). Spanwise taper t_hub→t_tip optional field, default off.
Gates: k in range, core ≥ floor, TE ≥ floor, nose radius reported (not floor-gated),
blockage/pitch gates re-evaluated with core t.

**P5 — Splitter CAD refinement (C4).** Slice the 3-D camber at u ≥ f_split, half-pitch
clock, same section generator. Gates: no blade↔blade fuse, passage clearance, fig.-16
Z_eff label, eye pitch gate mains-only.

**P6 — Diffuser vanes + throat enforcement (C5, C6).** Numeric α(r) camber vanes between
the side plates (reuse P4 section machinery in the (r,θ) plane), prime-count rule,
semi-vaneless gap r3/r2 ∈ [1.05, 1.10] as a spec field, throat iteration on α3 incidence
or vane thickness until Z_d·a_n·b3 matches the projected meanline throat. Rebuild the
passage audit to measure vaned fluid volume + CAD throat. Gates: |A_throat,CAD −
A_throat,meanline| ≤ tol; angles within incidence allowance; casing pocket clearance
audit still passes.

**P7 — Volute law, tongue, exit cone, dataflow (C7, C8).** Meanline: both laws +
K = r4·c_u4 + new manifest fields + cone sizing (7–9° circular) + tongue angle/incidence
screen. CAD: a(θ), r_c(θ) loft; tongue blend patch + tip cylinder; conical outlet;
split-casing audits re-run. Gates: numerically integrate ∫(K/r)dA on each lofted section
and assert Q(θ) linearity within tol; cone angle in [7°, 9°] (circular); terminal-area ≠
port-area assertion retired in favor of the cone; connected-void audit passes with
tongue; packaging/interface re-solve documented.

**P8 — Sweep (SP-8052 §2.1.7).** Meridional LE curve, tip cut back Δm_tip ≈ 0.1–0.3 tip
chord; wrap deficit compensated by axial-length increase (the section's own instruction —
add the check). Inducer via the revolve-cutter Boolean; impeller via section start offset
in u. Note expected Nss direction (+10–25 % per refs. 35/36 quoted therein), no claim.

**P9 — Lean (SP-8052 §2.1.8; default zero).** θ-offset over local span height (C1),
|λ| ≤ 15°, manifest justification string mandatory ("canting … for mechanical reasons
only; at high blade loadings … canted forward"); SP-8109's radial-element rationale for
open/semi-open impellers recorded when λ = 0. Root bending index reported.

**P10 — Fillets (SP-8109, geometry-frozen last).** r = 1.5·t at blade-to-hub/-shroud/
-backplate ("reduce the stress-concentration factor … to a value approximating 1"). OCC
edge fillets with retry ladder r, r/2, r/4 → swept quarter-round bead fallback;
`fillet_radius_achieved_m` per junction, never silent. 125→63 µin finish + shot-peen are
manufacturing-report callouts, not solids.

**P11 — Shrouded option.** Front-shroud revolve over the lofted blades, wear ring to
shroud OD, forward LE sweep for shroud fillets (SP-8052 §2.1.7), fig. 16 applies natively
(carpet is for shrouded wheels). Flips wear-ring/fillet topology — deliberately last.

Order rationale vs the review's list: its steps 1–3 are P0/P1 (mostly landed), 4–5 are
P2/P3, 6 is P4→P5, 7 is P6, 8 is P7, 9 is P8–P11 with fillets after geometry stabilizes
(OCC-fragile) — identical sequence, restated against the tree that now exists.

---

## 4. Literature verification ledger (2026-07-13)

Verbatim-confirmed from local text layers this session:

- **SP-8109** (`19740020848.pdf`): fillets — "The fillet radii at the blade-to-hub,
  blade-to-shroud, and blade-to-backplate junctions should be equal to 1.5 times the
  blade thickness … It is recommended that the leading-edge cross section be a 2:1 to
  3:1 ellipse"; finish 125 → 63 µin rms + shot peen. Volute cone — "for circular cross
  sections, 7° to 9°; for square cross sections, 6°; and for two parallel walls, 11°"
  (ref. 71 therein). Volute sizing — two methods (constant moment of momentum / constant
  mean velocity; Titan I/II housings, J-2S Mark 29 fuel pump), design criteria: "The
  constant-moment-of-momentum method adjusted for friction loss will produce minimum
  design-point impeller radial loads" (§3.4.1.3.1). Splitters — "experimental F-1 fuel
  impeller with six full blades and six splitters". Diffuser — throat area "the most
  important parameter for determining a match with the impeller discharge flow" (fig. 25
  velocity-ratio method, conservation of momentum from the impeller discharge radius);
  vane count "usually, the prime number nearest to the number of impeller blades"; vane
  width 90–100 % of tip width; tongue/vane radius ratio > 1.05 or virtually touching;
  off-design radial-load control by multiple tongues / vaned diffusers / double-outlet
  volutes (fig. 28).
- **SP-8052** (`19710025474.pdf`): §2.1.6 — inducer LE "knife-sharp" limit, practical
  nose radius t/100, large inducers 0.005–0.010 in edge; §2.1.7 — sweep "increases of 10
  to 25 percent in suction specific speed (refs. 35 and 36)", wrap reduction compensated
  by axial length, forward sweep on shrouded inducers for fillets; §2.1.8 — "Canting of
  the blade is done for mechanical reasons only. At high blade loadings, the blade is
  canted forward…".

Claim narrowing accepted from the review (citations adjusted accordingly): Chen
(`chen1992.pdf`) + Lin (`lin1993.pdf`) + Bellary (`bellary2014.pdf`) + Jaiswal
(`jaiswal2021.pdf`) support spanwise loading/stacking as the design lever; **Hong**
(`hong2012.pdf`) does *not* supply separate hub/tip impeller angles or a diffuser design;
**Mishra** (`Mishra_2015_…pdf`) does not study sweep (drop as sweep anchor; keep as
inducer-CFD cross-check); **energies-11-02588** supports LE-shape cavitation sensitivity
but not ellipse-optimality (the ellipse rule stands on SP-8109 alone).

**Missing from corpus — flagged:** `yang2019.pdf` and `tani2008.pdf` are cited by the
2026-07-11 roadmap but are **not present** in `propulsion_texts/fuel_pump_design/`.
RSI/clocking-specific claims are therefore dropped (C6) until the papers are added.
In-corpus substitutes used instead: the Dnipro LPRE inducer-centrifugal study
(`191-Article Text-322-1-10-20240202.pdf`, main + 6 splitter blades on an LPRE pump) and
SP-8109's F-1 6+6 caption for splitter practice; SP-8109's prime-number rule for vane
count.

Electric-pump context set (system-level anchors for drive/rpm/throttling assumptions the
geometry work must not violate; all local): Kwak & Kwon electric-pump vs GG cycle
(`1-s2.0-S1270963817320953-main.pdf`), Casalino/Masseni/Pastrone electric-pump hybrid
upper stage (`aerospace-06-00036.pdf`), deep-throttling electric pump control
(`hu2021.pdf`), electric-pump feed demonstrator (`BF03404670.pdf`), robust e-pump control
(`electronics-12-03527.pdf`), e-pump engine system design (`Zhou_2022_…pdf`,
`Liu_2021_…pdf`, `s41598-025-18499-5.pdf`), NASA LH2 electric-drive pump component rig
(`19730015803.pdf`). Closest-scale physical fuel pump: L75 centrifugal fuel pump
design/test (`reis2019.pdf`) — two-blade inducer + **single volute without diffuser**,
which validates keeping the repo's low-Ns `selection="volute"` (vaneless) branch as a
first-class path through P6/P7 rather than forcing vanes at this scale.

---

## 5. Out of scope (unchanged)

Joint hardware FEA, shaft torque/axial retention, bearings/seals/lubrication,
tolerances/thermal growth, rotordynamics, CFD, cavitation/cold-flow testing, measured
maps. None of the above items change release-gate semantics; split topology and every
new solid remain software-bounded claims.
