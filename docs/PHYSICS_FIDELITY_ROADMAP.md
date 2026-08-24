# Physics fidelity roadmap

**Status head (2026-08-22):** items **§3.1, §4.1, and §5.1 are implemented** as
scope/validity gates, and §4.3's existing liner-buckling screen is now integrated
as a mandatory MDO final gate (not as a qualification calculation); everything
else remains analysis and plan only. See the landed notes in §3–§5. Each item
states what the code does *today* (verified against the tree, not recalled),
what the literature in `propulsion_texts/` actually supports, and the smallest
change that raises fidelity without inventing authority the model does not have.

The governing rule from the remediation work applies throughout: **a fidelity
gap must be visible as `unknown`, never absorbed into a passing number.** Every
item below therefore ships its gate before, or with, its model.

---

## 1. Constant-property nozzle flow; CEA does not drive variable γ through MOC

### Today

`raosim/moc.py` threads `gamma: float` as a **scalar** through every unit
process — `_make_point`, `solve_interior_point`, `solve_axis_point`,
`solve_wall_point` all take one γ. `prandtl_meyer(M, gamma)` therefore assumes a
calorically perfect gas along every characteristic. The CEA surfaces added in
`raosim/mdo/properties.py` give γ(P_c, O/F) at the *chamber*, and the MDO uses
them, but the value entering the characteristic net is still frozen at a single
number.

The consequence is not small at high area ratio: γ falls as the gas expands and
cools, so a constant-γ net systematically misplaces the Prandtl–Meyer turning
and therefore the wall angles.

### What the literature supports

Rao, Beck & Booth, *Nozzle Optimization for Space-Based Vehicles*
(AIAA 99-2584, `rao1999.md`) is the direct authority. Rao himself extended the
variational method beyond perfect gas: the abstract states a solution *"is
obtained for the case of equilibrium and frozen chemistry"*, and line 106 notes
that where *"the flow cannot be treated as a perfect gas, it is possible to
consider equilibrium or frozen chemistry and derive a relation similar to that
above."*

This matters for scoping: **frozen and equilibrium are two different models**,
and Rao treats them as such. Frozen chemistry with variable c_p is a much
smaller change than equilibrium, which requires a composition solve at every
node.

### Plan

Three stages, each independently useful:

1. **γ(s) along the expansion, frozen composition.** Replace the scalar with a
   `ThermoProvider` interface exposing `gamma(M)` / `gamma(T, p)` backed by the
   existing sampled surfaces. Prandtl–Meyer becomes a quadrature
   ∫ √(M²−1)/(1+½(γ(M)−1)M²) dM/M rather than the closed form. This is the
   honest first step because `raosim/frozen_flow.py` already exists.
2. **Equilibrium option** behind the same interface, sampled offline into a
   table exactly as the chamber surfaces are (schema, content hash, identity
   checks). Never call CEA inside the MOC loop.
3. **Report both** and record which was used in `model_identities`. Rao 1999
   presents equilibrium and frozen as bracketing cases; the snapshot should be
   able to say which bracket a contour came from.

**Gate first:** add a `thermo_model` identity to the contour provider and make
a constant-γ contour report `gamma_model="calorically_perfect"`. A parity
comparison between a constant-γ and a variable-γ contour is then a measurable
number rather than an assumption.

---

## 2. Inviscid MOC with boundary-layer growth applied only as a separate screen

### Today

`raosim/physics.py:32 boundary_layer_displacement` is explicitly labelled
`"turbulent_flat_plate_displacement_screening"`. It estimates δ\* and an
effective area ratio *after* the inviscid contour is fixed. The wall the MOC
designed and the wall the gas actually sees are therefore different surfaces,
and nothing iterates between them.

### What the literature supports

`19970004933.md` — *Investigation of Low-Reynolds-Number Rocket Nozzle Design
Using PNS-Based Optimization Procedure* — is the right template. Parabolized
Navier–Stokes is the standard intermediate rung: viscous, space-marching,
far cheaper than full Navier–Stokes, and it captures exactly the displacement
effect a flat-plate screen approximates. `wang2006.md` (*Multidimensional
Unstructured-Grid Liquid Rocket-Engine Nozzle Performance and Heat Transfer
Analysis*) is the upper rung and explicitly makes the point that performance and
heat transfer *"are often analyzed separately"* — the same split this repository
has.

### Plan

1. **Close the loop before raising the model.** Even with the existing flat-plate
   screen, iterate: design inviscid contour → compute δ\*(x) → subtract from the
   wall → re-run MOC on the corrected wall → repeat to a tolerance. This is
   cheap, uses only what exists, and converts a one-way screen into a
   displacement-corrected contour. Report the correction magnitude; if it
   exceeds a threshold the flat-plate assumption is out of its domain and the
   result is `unknown`.
2. **Replace the screen with a compressible integral BL** (Whitfield/Sasman–
   Cresci-class) using the local edge conditions from the MOC net rather than a
   flat-plate approximation. This is the smallest change that removes the
   pressure-gradient blindness.
3. **PNS as a validation layer**, following `19970004933`, run at the candidate
   optimum only — never in the inner loop. Same architecture as the Rao
   variational validation layer (item 5).

---

## 3. Empirical separation onset; no shock/BL interaction or side-load solver

### Today

`raosim/separation.py` offers four algebraic correlations — Summerfield,
Kalt–Badal, Schmucker, Schilling — selected by `method=`. They return an onset
*pressure ratio*. There is no shock structure, no shock/boundary-layer
interaction, and no side-load model at all.

### What the literature supports

Frey & Hagemann, *Restricted Shock Separation in Rocket Nozzles*
(J. Propulsion & Power 16(3), 2000, `frey2000.md`) is decisive here, and it
shows why the current model is not merely low-fidelity but **structurally
incomplete**. There are two distinct separation topologies — free shock
separation (FSS) and restricted shock separation (RSS) — and the paper states
RSS *"also occurs in full-scale, thrust-optimized rocket nozzles"* (line 33).
Thrust-optimized parabolic/Rao contours are precisely what this repository
generates.

That is the finding: **a single onset pressure ratio cannot represent RSS at
all.** RSS involves reattachment — line 416 notes the flow *"will be forced to
reattach to the wall"* — and it is the reattachment, not the onset, that drives
the damaging lateral loads. Side loads are largest during the FSS↔RSS
transition, which an onset correlation cannot even detect.

`schomberg2016.md`/`schomberg2018.md` and `fulltext01.md` carry supporting
separation and side-load material.

### Plan

1. **Say what the model cannot see.** Immediately: mark the separation
   constraint's applicability as FSS-only, and register a `side_load` metric as
   *unavailable with reason* (per §3 of the metric registry) rather than
   leaving it unmentioned. A thrust-optimized contour with no RSS check must not
   report an unqualified separation pass.
2. **Add an FSS/RSS topology classifier** from Frey & Hagemann's criteria —
   this is a classification, not a CFD solve, and it is tractable from the
   existing wall-pressure profile plus the internal shock position.
3. **Side loads require the shock solver.** Do not attempt a correlation. The
   honest path is: RSS classification → axisymmetric shock/BL interaction →
   only then a lateral-load estimate. Until then `side_load` stays unavailable,
   which is a correct statement about the repository, not a gap to paper over.

### Landed (§3.1)

`check_separation()` now returns its own scope, so an unqualified
"not separated" cannot be read as a complete verdict:

```
separation_pattern_scope            free_shock_separation_only
restricted_shock_separation         not_modelled
restricted_shock_separation_reason  <Frey & Hagemann, with the mechanism>
side_load_n                         None
side_load_unavailable_reason        <same>
scope_source_id                     Frey-Hagemann-2000-JPP-16-3-...
```

`aerodynamics.side_load` is registered in the metric registry as **unavailable
with reason**, so it appears in `lrekit catalog unavailable` and is refused as
both a target and an objective. Numerics are unchanged — this is scope
metadata only.

---

## 4. Low-order thermal and structural models

### Today

Bartz 1957 gas-side, Sieder–Tate coolant-side, a series 1-D circuit with 2-D/3-D
conduction options, SP-125 eq-4-31 combined stress at a station, and
Coffin–Manson life. Individually sourced; collectively still a decoupled
sequence — the wall temperature that sizes the structure is not the temperature
the structural deflection would produce.

### What the literature supports

Betti et al., *Coupled Heat Transfer Analysis in Regeneratively Cooled Thrust
Chambers* (`betti2014.md`) is exactly the coupling target: the chamber-side, the
wall conduction and the coolant march solved together rather than in sequence.
Wadel, *Comparison of High Aspect Ratio Cooling Channel Designs*
(`wadel1997.md`) is the authority for the HARCC regime the optimizer already
selects — the repository's aspect-ratio constraint bounds land in exactly the
range Wadel studied, so this is the source for whether those bounds are
defensible.

### Plan

1. **Wadel first — it is a bounds question, not a model question.** Check the
   implemented aspect-ratio limit against Wadel's tested range and record the
   applicability. Cheap, and it either validates a live constraint or exposes
   it as extrapolation.
2. **Betti-style conjugate coupling** as a post-solve validation of the
   decoupled march, reporting the delta. The decoupled march stays the inner
   loop; the coupled solve is the check.
3. **Structural:** the inner-shell external-coolant-pressure screen already
   exists as `raosim.physics.rib_supported_liner_buckling_profile`; it is a
   rib-supported long-plate screen and is distinct from SP-125 eq. 4-29's
   longitudinal inelastic-buckling case. The MDO now mirrors it in
   `raosim.mdo.structures.rib_supported_liner_buckling_screen` and carries
   `liner_buckling` as a required, available `final_gate` in
   `CONSTRAINT_MANIFEST`: a negative margin fails the post-solve physics
   verdict. It is deliberately not a hard SLSQP row yet. Hard inner-loop
   enforcement is deferred until the implicit root is robust at valid box
   corners and NASA SP-8068 authority is available to qualify the classical
   plate coefficient; until then this remains a source-labelled preliminary
   screen, not a qualification calculation.

### Landed (§4.1) — and it found two real gaps

Wadel's milled-channel fabrication list (line 204 of the transcription) is now
recorded in `raosim/manufacturing_limits.py` with provenance and an explicit
applicability note (Wadel is a **liquid-hydrogen**, 89 kN-class study, so
applying it to a 13 kN RP-1 engine is an extrapolation in both coolant and
scale).

Checking the code against it:

| Wadel milled limit | code | verdict |
|---|---|---|
| aspect ratio < 8 | `channel_aspect_ratio_max = 8.0` | **exact match** |
| channel height < 5.1 mm | box `[0.80, 5.00] mm` | within |
| channel width > 0.51 mm | box `[0.30, 1.20] mm` | **lower bound below the milling minimum** |
| land width > 0.51 mm | constraint is `land_min >= 0` | **no manufacturing floor at all** |

Two findings. The optimizer may select a 0.30 mm channel that cannot be milled,
and the land-width constraint only requires lands to be *non-negative* — a
0.01 mm land passes. At the 13 kN reference point the land happens to come out
at 0.54 mm, above Wadel's 0.51 mm, but nothing enforces that.

These are **reported, not enforced**: milling practice is not a physical
constraint, and silently narrowing the search box would change every optimum
without the user asking. `resolve_engine_job` now emits a
`manufacturability:` note naming the quantity, the limit and the source, so a
box corner that cannot be milled is visible before it is explored.

The aspect-ratio constraint was also mis-categorized as `thermal` with
`source_id=None`. It is now `manufacturing`, sourced to Wadel 1997 plus Carlile
1992 and Mirzamoghadam 1991. Wadel is explicit that 8 is a *fabrication*
ceiling — its own designs reached AR 40 analytically and were cut back "from 40
down to the limit of 8" only when milling constraints were applied, and Carlile
tested 8 and 15 experimentally. So AR ≤ 8 is well-corroborated as milling
practice, and is **not** a claim that AR > 8 is unphysical.

---

## 5. The exact Rao path is auditable but not release-promoted

### Today

`raosim/rao_variational.py` implements the variational/MOC solution, the
corrected BDE topology audit, and a typed `RaoRadialIncrementValidity` object.
The solver evaluates Rao 1999 Eq. (5) segment-by-segment, stores the predicted
dR/R result, and derives `rao_region`, reliability, and `shock_free` from it.
Contour export refuses both invalid and unevaluable validity results. Eq. (6)
is retained only as a labelled legacy boundary diagnostic. The path remains
non-default because the other two promotion gates have not cleared.

### What the literature supports

Rao 1999 supplies the missing piece: the method has a **valid computation
region**, and below a minimum length *"a solution that is shock-free is not
possible by these methods"* (line 28). The paper defines the boundary between
valid and invalid regions and gives a construction for the invalid region using
an isentropic compression at a discontinuity point.

### Plan

Promotion is not one gate but three, and Rao 1999 defines the first:

1. **DONE (2026-08-22): implement the valid/invalid boundary.** Eq. (5), not a
   geometric-radius-difference proxy or Eq. (6), now determines validity; an
   invalid or unevaluable result cannot be exported as a contour.
2. **Convergence domain mapping** across the design box (this is what
   `rao_existence_scan.py` is for), so the provider can declare where it works.
3. **Then** add the variational implementation to the selectable
   `ContourProvider` contract (the analytic TOP provider/failure gate is already
   landed), initially as post-solve validation with a geometry/performance
   discrepancy tolerance.

---

## 6. Partial differentiability: no complete start-line → wall map

### Today

The inner MDO uses the fixed-topology Bézier contour, which is differentiable.
The variational path is not wired into the gradient. Total design derivatives
therefore do not flow through contour shape.

### Plan

This is a sequencing problem, not a physics problem, and the answer is the
implicit-function theorem — already used twice in this codebase for the engine
and cooling roots. The variational BVP is a constrained stationarity condition;
its sensitivities can come from an adjoint rather than from differentiating the
inner solve. Order: (a) fixed output topology, (b) warm starts, (c) validated
implicit sensitivities against finite differences, (d) only then admit it as an
inner-loop provider. A sampled surrogate generated from the JAX BVP is the
practical intermediate.

---

## 7. Preliminary CAD: no drawings, GD&T, tolerance stack, fastener standards

### Today

The build manifest (`raosim/mdo/build.py`) now enumerates 27 expected parts and
caps achieved fidelity at **L2** (valid reference CAD). `L4`
(manufacturing reference) and `L5` (qualified hardware) are defined and
unreachable — `L5` deliberately never awarded by a solver.

### Plan

This is the one area where the correct action is mostly **not to build models**.
GD&T, a signed tolerance stack, and selected fastener standards are engineering
*decisions and approvals*, not solver outputs. What the repository can honestly
add:

1. **Datum and feature-control scaffolding** at L4 — machine-readable datum
   frames and fits, marked as proposals requiring sign-off.
2. **A tolerance-stack input contract** — the geometry the stack would consume,
   so an external analysis is reproducible. The stack itself remains a human
   deliverable.
3. **Fastener standard selection** — currently `fasteners.interface_bolts` is
   sized but has no standard and no solid bodies; that is already recorded as a
   manifest limitation. Selecting a standard is a decision to expose, not to
   automate.

L5 stays unreachable by construction.

---

## Suggested order

Cheapest-first, and gates before models throughout:

1. Separation applicability + `side_load` registered unavailable (§3.1) — hours
2. Wadel aspect-ratio bounds check (§4.1) — hours
3. Rao valid/invalid boundary (§5.1) — **DONE 2026-08-22**
4. BL displacement iteration loop using the existing screen (§2.1) — days
5. Variable-γ frozen MOC (§1.1) — the largest genuine fidelity gain
6. FSS/RSS classifier (§3.2)
7. Betti-style conjugate validation pass (§4.2)
8. Equilibrium chemistry option (§1.2), PNS validation layer (§2.3)
9. Rao adjoint sensitivities (§6)

Items 1–2 are gates that make current results honest and cost almost nothing.
Item 5 is where the physics actually improves.
