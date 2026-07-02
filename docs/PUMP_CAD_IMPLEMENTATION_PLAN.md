# Pump CAD implementation plan — 2026-07-02

> **STATUS 2026-07-02 (latest) — Phases 2 (items 1–4) and 3 landed; plan
> complete except the deferred default-flip and optional NURBS.**
> (2.2) SP-8052 inducer blade angles: α/β = 0.425 transcribed from §3.1.9
> (band 0.35 thin – 0.50 thick), inlet tip blade angle =
> atan(φ_tip)/(1−0.425) with φ_tip from eye continuity; constant-lead helix
> r·tanβ = const (§3.1.10) now sets the exported pitch (lead) and the hub
> blade angle; wrap from solidity 2.5 (§3.1.15, was 1.5) with the
> developed-chord cos β factor; leading edge takes the 0.005 in. low end of
> J-2/F-1 practice (§2.1.6). Hong 2012 (10.4° tip / 3 blades / σ 2.6)
> pinned as a cross-source consistency benchmark.
> (2.3) Z from the digitized SP-8109 fig. 16 ψ–Z carpet
> (`pumps.sp8109_min_blade_count`, ±0.02 ψ read-off; zero prewhirl,
> shrouded, δ=0.65), snapped to a multiple of the inducer count (SP-8052
> §3.1.14); `PumpSizingSpec.blade_count=None` is now the auto default →
> ψ=0.55/φ₂=0.08 solves Z=12 (chart min 10) with the basis string exported
> in `blade_count_source`; 0.4 mm thickness floor kept with
> mini-centrifugal-pump-paper provenance.
> (2.1) Quarter-ellipse meridional hub/shroud channel
> (`pumps._meridional_channel`) honoring D1/D2/b2 + inducer hub with EXACT
> eye-annulus and exit areas; SP-8109 §2.3.1.2 cm2/cm1 = 1–1.5 screen
> exported (`cm_ratio_status`); the B-rep impeller now revolves the hub
> curve (one solid with the backplate) and trims blades to the shroud-curve
> revolve.
> (2.4) Thrust-balance hooks (`pumps._thrust_balance_geometry`): wear rings
> selected per SP-8109 §3.5.2.1 (recommended over balance ribs), hub ring
> at the eye diameter (equal-diameter neutral start, editable trim), shaft
> seal land with solved face speed vs the screening limit, balance holes
> sized by the §3.5.2.1 rule (flow area ≈ 4× seal-clearance area) once
> `PumpSizingSpec.wear_ring_radial_clearance` is supplied; CAD adds the
> wear-ring land and drills the holes when sized. (2.5 NURBS) skipped —
> prisms haven't proven limiting.
> **Phase 3:** `raosim/engine_cad.py` + `--engine-assembly` aggregates the
> gated wall/jacket/pintle/pump/battery STEPs into `engine_assembly.step`
> (re-import gated, layout placements documented, not routed feed lines)
> with an `interface.py::resolve_bolted_interface_geometry` screen for the
> pump mounting flange. Benchmarks as tests: Lee 2021 500 N/20 bar/600 s
> package closure — motor 451.2 g and battery 985.6 g (power-limited)
> reproduced through our drive/battery chain with the paper's constants
> (0.875 kW/kg @87%, 60 kW/kg @85%, 325 Wh/kg / 0.650 kW/kg @92.5%,
> 20% margin, values transcribed from the PDF text); SP-8109 Ns 450–2100 US
> envelope on solved pumps. Provenance rows added to
> `pump_dimensions.csv`: θ_c camber wrap, β_t1/β_h1/α_i/φ_1/t_le, p_ind as
> lead, cm2/cm1, D_wr_hub/d_bh/U_seal. Docs:
> `electric_pump_model_basis.md` blade/channel fidelity section + audit §6
> closure notes (README left to the user's in-flight revamp).
> **Deferred:** `--pump-cad-format` default stays `stl` until CI installs
> CadQuery (flip in the same release as the CI gate, per the risk note);
> wall/pintle children of `engine_assembly` exercised via CLI only.
> Tests: `tests/test_pumps.py` 28, `tests/test_pump_cad_brep.py` 7
> (`.venv-jax`), test_injector 78, both pump CLI regressions — all green.

> **STATUS 2026-07-02 (later) — Phase 1 landed** (`raosim/pump_cad_brep.py`,
> CadQuery-gated, `.venv-jax` only): true B-rep shaft / inducer (helical
> swept blades) / impeller (log-spiral camber blades per
> `pumps.impeller_blade_camber`, trimmed by the shroud envelope, semi-open) /
> diffuser ring (side plates + vanes at the SOLVED absolute flow angle, now
> exported via `diffuser_vane_ring.vane_angle_deg`) / volute casing (scroll
> loft with linear A(θ), tangential exit port at the outlet-port equivalent
> diameter) / motor+inverter+battery placeholders; named `fuel_pump` /
> `oxidizer_pump` assemblies; every STEP re-imported and gated
> (isValid/volume), diagnostics into `summary.json.pump_package`.
> `--pump-cad-format step|both` = B-rep (parser error without CadQuery);
> default still `stl` (flip waits for CI per plan). Tests:
> `tests/test_pump_cad_brep.py` (6, skipif-gated) + camber closed-form in
> test_pumps. OCC lessons baked into code comments: n-ary fuse only with
> volumetric overlaps (no coincident tangent faces), never `clean()` after
> thin-blade fuses, `intersect` silently empties on fuse results (use
> complement cuts), no cylindrical trims tangent to swept blade faces.
> **Physics finding for Phase 2:** the solved minimum shaft (Ø6 mm screen)
> does not fit the SP-8052 inducer hub (0.35·D_ind ≈ Ø2.7 mm) or the
> impeller hub boss at this pump size — CAD skips the bore and notes
> "integral shaft/rotor assumed"; hub sizing vs min-shaft consistency needs
> a real rule. Next: Phase 2 item 2 (SP-8052 inducer blade angles from φ +
> incidence, transcribe i/β from §3.1) → item 3 (ψ–Z chart) → item 1
> (meridional hub/shroud curves).

> **STATUS 2026-07-02 — Phase 0 landed.** 2π seam welded in
> `_annular_cylinder` (ring vertices precomputed once; closing segment
> reuses column 0), which closed the audit's 14/6/8 boundary edges — every
> pump part STL now passes the repo mesh gate, and `_write_part` fails
> non-watertight or non-positive-volume meshes unless
> `--allow-open-pump-mesh`. Faceted pseudo-STEP removed for pumps:
> `--pump-cad-format step/both` is now a parser error and
> `export_pump_package` raises (`_PUMP_STEP_UNAVAILABLE`).
> `pump_reference_geometry()` consumes `PumpReferenceGeometry` only —
> impeller disk D2/D1/b2, blade-envelope thickness, helix pitch·wrap axial
> length, meridional-station axial widths and diffuser radii, shaft datum —
> with the fallback re-derivations deleted except the honest `not_sized`
> path. Also fixed a dataclass-vs-`to_dict` key mismatch (`motor_mass` vs
> `motor_mass_kg`, battery `mass`/`voltage`) that silently zeroed
> drive/battery packages when CAD was fed a pump.json round-trip. Tests:
> new `tests/test_pumps.py` (mesh gates, STEP refusal, single-source
> dimension checks, not-sized path, CLI rejection) + both pump CLI
> regressions green. Next: Phase 1 impeller+inducer B-rep in
> `raosim/pump_cad_brep.py`.

Successor to the pump findings in `docs/REPO_PHYSICS_AUDIT_2026-07-02.md` §6–8.
Goal: bring pump CAD to parity with the wall/jacket/pintle chain — true B-rep
STEP, validation gates, dimensions traceable to the meanline and the corpus —
without pretending to be blade-to-blade design.

**Governing principle — single source of truth.** `pumps.py` already exports
`PumpReferenceGeometry` per stream (meridional stations, impeller disk
D2/D1/b2, blade envelope Z/β1/β2/thickness, inducer helix D/hub/pitch/wrap,
diffuser vane ring, volute scroll, shaft datum, ports, casing radius/wall).
CAD must CONSUME this object. Today `pump_cad.py` re-derives dimensions with
its own fallbacks (`d1=0.45·d2`, `b2=0.04·d2`, blade thickness
`max(0.025·d2, b2)`, ad-hoc lengths) — a divergence channel to eliminate.
Physics stays in `pumps.py`; `pump_cad*` only turns solved numbers into solids.

---

## Phase 0 — make what exists correct (small; do first)

1. **Gate part meshes on watertightness.** `_write_part` already runs
   `inspect_stl` but ignores the verdict. Fail (with an
   `--allow-open-pump-mesh` escape) exactly like `export_stl` does for the
   wall. Audit baseline: shipped impeller/inducer/diffuser STLs have 14/6/8
   boundary edges.
2. **Weld the 2π seam.** In `_annular_cylinder`/`_radial_blade_boxes`,
   precompute the ring vertex arrays once and index them (last segment reuses
   column 0) so float drift can't split vertices. This alone should close the
   boxes/cylinders; overlapping-primitive unions wait for Phase 1 (booleans
   need a kernel).
3. **Stop emitting the hand-rolled faceted STEP.** `--pump-cad-format step`
   currently routes to `_write_faceted_step` (bare `FACE` entities, malformed
   `GEOMETRIC_REPRESENTATION_CONTEXT`, no unit context — import-hostile).
   Until Phase 1 lands: requesting `step` without CadQuery → hard error
   consistent with `--require-brep` semantics; never silently write the
   pseudo-STEP for pumps.
4. **Point `pump_reference_geometry()` at `PumpReferenceGeometry`.** Delete
   the fallback re-derivations except for the honest `not_sized` path
   (missing tank pressure).
5. Tests: extend `tests/test_pumps.py` with mesh-diagnostic assertions
   (watertight, volume>0) for every emitted part.

Exit criteria: every pump artifact in a fresh build passes the repo's own STL
gates; no faceted pseudo-STEP can be produced.

## Phase 1 — B-rep parts + named assembly (CadQuery parity with the pintle path)

New `raosim/pump_cad_brep.py`, mirroring `injector_cad.py` conventions
(`cadquery_available()`, named `cq.Assembly`, SI-m → mm at the kernel
boundary, +Z shaft axis) and `regen_cad.py` validation (export → re-import →
`isValid`/solid-count/volume gates). All shapes are revolves, extrusions,
helical sweeps, and booleans — nothing exotic:

- **Shaft** — revolve from `shaft_datum` (d_shaft, span).
- **Inducer** — hub revolve (hub_ratio·D to eye); blades = rectangular section
  swept along the parametric helix already defined by `inducer_helix`
  (pitch, wrap angle, blade count), patterned + multi-fused like the regen
  ribs; thickness from `blade_envelope`/spec floors.
- **Impeller** — hub/backplate revolve through the meridional stations
  (eye D1 → exit D2 with outlet width b2); blades = log-spiral camber
  `θ(r) = ∫ dr/(r·tanβ(r))` with β linear from β1(r1) to β2(r2) (the solved
  velocity-triangle angles; standard circular-arc/log-spiral pump blade
  layout per SP-8109 blade-geometry practice), extruded across the channel,
  trimmed by hub/shroud, unioned; bore for shaft. Camber math goes in
  `pumps.py` (physics owns it), CAD just sweeps it. Optional shroud disk
  (SP-8109 shrouded/unshrouded selection; the corpus design example
  `fuel_pump_design/hong2012.pdf` is a shrouded 7-blade rocket impeller).
- **Diffuser ring** — annulus + vane prisms at the vane angle implied by the
  throat area and cm2/cu2 (count from `diffuser_vane_ring`).
- **Volute** — casing revolve (casing_r, casing_t) + collecting scroll swept
  with linear area schedule `A(θ) = A_exit·θ/2π` (constant-angular-momentum
  first pass; SP-8109 collecting-volute practice), exit port at the
  equivalent diameter from `ports.outlet`.
- **Motor / inverter / battery** — cylinder + boxes as today, B-rep, still
  labeled package placeholders.
- **Assembly** — `fuel_pump` / `oxidizer_pump` assemblies positioned along
  the shaft by the meridional stations, plus `shared_battery_pack`; per-part
  STEP + STL, one assembly STEP each, diagnostics into `pump.json`
  (`step_representation`, re-import validity) like the wall path.

CLI: `--pump-cad-format step` now means true B-rep (error without CadQuery
unless STL also requested); default flips `stl` → `step` once the re-import
gate is green in CI. Keep the Phase-0 mesh writer as the no-CadQuery fallback
for STL only.

Tests: `tests/test_pump_cad_brep.py` with the existing
`pytest.mark.skipif(not cadquery_available())` pattern
(`test_regen_cad.py:18`); assert named bodies, re-import validity, and that
key CAD dimensions equal the meanline values (D2, b2, D_ind, pitch) to 1e-9.

## Phase 2 — meanline-faithful geometry (the physics upgrades)

Each item = a `pumps.py` model change + a CAD consumer + a corpus citation:

1. **Meridional channel.** Replace the 6-station envelope with hub/shroud
   curves (arc/Bézier) honoring D1, D2, b2 and a smooth area progression —
   SP-8109 impeller design section (`fuel_pump_design/19740020848.pdf`;
   inlet-to-discharge diameter ratio and meridional-velocity practice, cf.
   the "discharge meridional velocity 1 to 1.5× inlet" rule already in the
   text).
2. **Inducer blade angles.** Inlet tip blade angle from the inlet flow
   coefficient plus an incidence margin — SP-8052 ties the inlet tip blade
   angle to φ and makes the incidence-to-blade-angle ratio the cavitation
   design variable, with leading-edge wedge/thickness rules
   (`fuel_pump_design/19710025474.pdf`, §2.1.1/2.1.14; transcribe the
   recommended i/β value from §3.1 at implementation — do not invent it).
   Benchmark row: Hong et al. 2012 rocket turbopump — 3-blade inducer, inlet
   tip blade angle 10.4°, outlet mean 17°, tip solidity 2.6; 7-blade
   shrouded impeller, inlet mean blade angle 19° (`hong2012.pdf`, verified).
3. **Blade count/thickness.** Z from the SP-8109 fig. 16 ψ–Z chart rather
   than a fixed 6 (digitize like the Rao θ tables were); small-pump blade
   thickness/outlet-angle sensitivity from the mini-centrifugal-pump studies
   in the corpus (`Influence of Blade Outlet Angle and Blade Thickness….pdf`,
   IOP impeller papers) + the existing 0.4 mm manufacturing floor.
4. **Wear rings, seal lands, balance holes.** Geometry hooks tied to the
   loss/thrust models that already exist (leakage and disk-friction buckets,
   seal face-speed screen) — SP-8109 wear-ring/axial-thrust sections.
5. **Optional NURBS blade surfaces** (ruled/lofted B-spline) if prisms prove
   limiting — representation per the corpus CAD notes (`CAD_04.pdf`,
   `L-05_BSplines_NURBS.pdf`).

## Phase 3 — integration, benchmarks, docs

- **Engine-level assembly:** one optional `engine_assembly.step` placing
  wall + jacket + machined pintle + both pump packages (pump inlets at the
  feed-line stubs, shared battery), reusing `interface.py` bolt screens for
  the pump mounting flange.
- **Benchmarks as tests** (same pattern as the Rao chart benchmark):
  Lee 2021 electric-pump case for package mass/size closure
  (`s42405-020-00325-z.pdf`), Hong 2012 geometry ratios, SP-8109 Ns–Ds
  envelope sanity.
- **Provenance:** extend `pump_dimensions.csv` with camber/thickness/wrap
  rows + source strings; keep `qualification_status:
  reference_geometry_not_hardware_qualified` until vendor/test data exist.
- **Docs:** update `electric_pump_model_basis.md`, README pump section, and
  the audit doc's §6 findings as they close.

## Out of scope (unchanged, keep the labels)

Blade-to-blade CFD, rotordynamics/critical speed beyond the DN screen,
bearing/seal selection, motor electromagnetic design, measured pump maps.
The existing notes in `pump_cad.py`/`pumps.py` already say this; preserve them.

## Risks / notes

- Helical sweeps at large wrap angles can self-intersect — validate per-blade
  before fuse; regen_cad's pattern-then-multi-fuse approach controls Boolean
  cost for Z blades.
- CadQuery is heavy; keep it optional at import (already the repo pattern)
  and CI-mark the B-rep tests.
- Default-flip of `--pump-cad-format` is a user-visible change; do it in the
  same release as the re-import gate so "step" can never mean the old
  pseudo-STEP again.

**Recommended order:** Phase 0 in one sitting → Phase 1 impeller+inducer
first (they're the credibility parts), volute second, assembly third →
Phase 2 items 2→3→1 (blade angles are the highest-value fidelity gain) →
Phase 3.
