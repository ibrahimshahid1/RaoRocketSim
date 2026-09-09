---
title: Pump physics and process basis
subtitle: What the corpus actually supports for promoting pump geometry from reference to hardware
version: "0.1"
audience: [ibrahim, AI coding agents]
created: 2026-09-05
companions:
  - docs/LREKIT_HARDWARE_PROGRAM_2026-09-05.md      # the program this grounds
  - docs/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md  # P0-P11, the geometry backlog
  - docs/electric_pump_model_basis.md               # the existing meanline basis
  - docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md       # what is still missing
---

# Pump physics and process basis

> **Applicability review, September 6:** read the [pump audit](audit_work/LREKIT_PUMP_AUDIT_2026-09-05.md)
> before implementing the recommendations below. Historical fabrication and trim
> guidance is not a universal shop capability or response law. Existing limited
> pump process screens and rough-wall cooling branches are present; the audit
> identifies their actual gaps and the nonzero-roughness unit mismatch.
> The [delivery plan](LREKIT_READINESS_AND_DELIVERY_PLAN_2026-09-05.md) owns sequencing and acceptance.

> **What this document is.** A grounded statement of the physical relations, design
> limits, and *manufacturing process rules* that the pump geometry must honor
> before it can be called machinable or testable — read out of
> `propulsion_texts/` rather than from memory, with the verification status of
> every number recorded.
>
> **What it is not.** A textbook. It covers the concepts that constrain a
> geometric degree of freedom or a process decision, because those are the ones
> that become CAD parameters, design gates, and inspection requirements.

---

## 0. Verification status, stated plainly

The primary source is **NASA SP-8109, *Liquid Rocket Engine Centrifugal Flow
Turbopumps*, R. B. Furst (Rocketdyne), December 1973**, held locally at
`propulsion_texts/fuel_pump_design/19740020848.pdf`. It is a scanned document;
the corpus mirror at
`propulsion_texts/propulsion_texts_for_agents/markdown/fuel_pump_design/19740020848.md`
is an OCR conversion flagged `needs_review`.

Page mapping for this document: **printed page + 12 = PDF page**.

Three verification levels appear below and are marked on every entry:

| Mark | Meaning |
|---|---|
| **[V]** | Read directly off a rendered image of the source PDF page during this session. Quotable. |
| **[T]** | Transcribed from the OCR text layer only. Prose is reliable at this level; treat numerals as provisional until rendered. |
| **[F]** | Comes from a **figure or table** in a scanned document. **Not usable until digitized off the image.** OCR of these is scrambled and must not be trusted. |

Two pages were rendered and verified in full this session: **PDF 50 (printed 38,
§2.3.3 Fabrication)** and **PDF 84 (printed 72, §3.3.3 Fabrication)**. Both
confirm their OCR faithfully, which raises confidence in [T] prose elsewhere but
does not promote it.

---

## 1. The one variable that ties hydraulics to metal: impeller tip speed

This is the organizing insight of the whole document, so it comes first.

`U₂ = √(g₀ H / ψ)` is already computed in the meanline
(`raosim/pumps.py:1108`, `:1239`). SP-8109 makes **U₂ the selector for the
fabrication process, the material, the shroud configuration, the required
qualification test, and the achievable surface finish.** It is not one screen
among many; it is the spine.

### 1.1 Process selection by tip speed  **[V]** — §2.3.3, printed p. 38

> "Impellers are fabricated by casting, machining, and diffusion bonding. The
> highest-tip-speed impellers, used for pumping liquid hydrogen, are open-face
> impellers machined from a forged titanium alloy (ref. 62). Shrouded impellers
> for dense fuels or oxidizer service are cast, since cast material properties
> are adequate for the required tip speeds of less than 1000 fps. Open-face
> impellers for oxidizer service may be cast or machined."

And from §3.3.3, printed p. 71–72 **[T→V]**:

> "Casting is the preferred method of fabrication for impellers with tip speeds
> below 1400 fps… High-tip-speed impellers (>1400 fps) with open faces should be
> machined from forgings. Shrouded high-tip-speed impellers for tip speeds to
> 2200 fps may be machined or, if fabricated from titanium, may be diffusion
> bonded."

| Tip speed | Configuration | Process | Note |
|---|---|---|---|
| < 1000 fps (305 m/s) | shrouded, dense fuel / oxidizer | **cast** | cast properties adequate |
| < 1400 fps (427 m/s) | general | **casting preferred** | cheaper; permits hydrodynamic optimization; excellent shroud-to-blade strength and cleanliness |
| > 1400 fps (427 m/s) | open face | **machined from forgings** | |
| to 2200 fps (671 m/s) | shrouded | **machined, or diffusion bonded if titanium** | |
| to 2500 fps (762 m/s) | open face, Ti-5Al-2.5Sn, smooth central hole | machined from forging | demonstrated in LH2, ref. 45 **[T]** |
| to 2870 fps (875 m/s) | shrouded, diffusion-bonded Ti-5Al-2.5Sn, β₂ = 37° from tangential | diffusion bonded | **spun**, room temperature, ref. 60 **[T]** |

Material tip-speed ceilings **[T]** — §2.3.2.1, printed p. 37:
current limit for **shrouded cast impellers pumping liquid hydrogen is 1400 fps
for Inconel 718 and for vacuum-melt, vacuum-cast aluminum.**

**Discrepancy against the tree.** `PumpSizingSpec.material_tip_speed_limit`
(`raosim/pumps.py:150`) is a **single scalar defaulting to 350.0 m/s**, screened
at 80 % of itself (`raosim/pumps.py:1334`). SP-8109 makes the limit a function of
**(material, fabrication method, shrouded/open-face, pumped fluid)** — four
arguments, not zero. A 350 m/s scalar simultaneously over-restricts a machined
open-face titanium rotor (762 m/s demonstrated) and under-restricts nothing in
particular, because it is not attached to a process at all.

### 1.2 Qualification test keyed to the same variable  **[V]** — §3.3.3, printed p. 72

> "It is recommended that, as part of the manufacturing procedure, impellers
> expected to operate at tip speeds above 1000 fps be spun to a speed at least 20
> percent above the maximum anticipated operating speed; this procedure will
> result in local yielding without failure. The speed should be corrected for
> temperature influence on material properties if the overspeed run is conducted
> at a temperature different from that for normal operation."

So the *same* U₂ that picks the process also decides whether a spin test is
required, and at what speed, with an explicit temperature correction. This is a
required-inspection record generated from a solved quantity — exactly the shape
the process layer needs.

---

## 2. Manufacturing limits that are already in the corpus

The claim that "no manufacturing-process model goes anywhere near the pump" is
true of the **code**. It is **not** true of the **literature**: SP-8109 §2.3.3 and
§3.3.3 together are a compact, quantitative, part-specific process model, and
they are already on disk.

### 2.1 Cutter reachability, verbatim  **[V]** — §3.3.3, printed p. 72

> "Shrouded machined impellers should be limited to stage specific speeds over
> 1000 and tip diameter-to-width ratios less than 20. The maximum impeller blade
> number that can be machined is approximately 28 sin β₂. **The cutter
> length-to-diameter ratio that establishes the blade spacing should not exceed
> 10 for aluminum and 8 for titanium parts.**"

and the causal statement **[V]** — §2.3.3, printed p. 38:

> "The maximum blade number for machined blades in shrouded impellers is limited
> to approximately 28 sin β₂ **by required cutting tool clearance and limits on
> tool length-to-diameter ratio.** The existence of the shroud imposes obvious
> restrictions on blade shapes that can be machined. Open-face impellers,
> however, may be machined easily with no serious limits imposed upon the blade
> shape or blade number by machine-tool limitations."

This is the cutter-reachability constraint in closed form, and it has three parts
that must be implemented together:

1. **Z₂ ≤ ≈ 28 sin β₂** — machinability ceiling on blade count, *shrouded only*.
2. **cutter L/D ≤ 10 (aluminum), ≤ 8 (titanium)** — the underlying reason, and
   the check that generalizes to other materials once their values are sourced.
3. **D₂/b₂ < 20** and **stage Ns > 1000** — the validity window for calling a
   shrouded impeller machinable at all.

**Discrepancy against the tree.** `raosim/pumps.py:1179 sp8109_min_blade_count`
implements SP-8109's **hydrodynamic minimum** blade count from the fig. 16
velocity-gradient carpet. There is **no corresponding machinability maximum**
anywhere in `raosim/pumps.py` or `raosim/pump_cad_brep.py` (searched: zero hits
for `28 sin`, one unrelated hit for `machinable`). A design can therefore be
emitted today with a blade count that no shop can cut, and the bound that would
have caught it is in a monograph the repository already cites for the opposite
bound on the same page range.

This is the cheapest high-value item in the entire program: one function, one
gate, one test, sourced from a document already in the tree.

### 2.2 Achievable surface finish by process  **[V]** — §2.3.3, printed p. 38

> "Aluminum impellers cast in ceramic-shell core molds can be made with a 63 µin.
> per inch surface finish for diameters up to 10 in. and a 125 µin. per inch
> finish for diameters of 10 to 20 in. Similar finishes have been obtained with
> investment castings of Inconel 718 and stainless steels."

Note the finish is **size-dependent**, which means the process capability record
cannot be a per-process constant — it is a function of the part envelope.

### 2.3 Balance, imbalance, and machining discipline  **[V]** — §3.3.3, printed p. 72

> "Residual imbalance, oz-in. ≈ 4 × rotor weight in lb / speed in rpm"

*(My conversion, not the source's: U [kg·mm] ≈ 6.35 · m [kg] / N [rpm].
1 oz·in = 0.72008 kg·mm.)*

> "Cutting tools should be changed to produce weight symmetry as cutters wear.
> Cutters should be changed every 60°, 90°, or 180°, for example, rather than
> after a certain amount of cutter wear, so that the dimensional changes
> resulting from cutter wear will be minimized."

> "Location, amount, and procedure for removing or adding balance correction
> material should be specified… careful attention be given to fixturing and arbor
> problems along with consideration of **room-temperature fits relative to
> operating fits on cryogenic pumps**."

> "Special design provisions (e.g., double registers, conical registers, and
> dowel pins) or fixtures could be necessary. In general, it is recommended that
> the disassembly of the rotating assembly after balancing be kept to a minimum."

Three things follow for the toolkit. A balance-correction feature is
**geometry the CAD must contain**, not a shop note. The cold-vs-operating fit is a
**thermal-growth calculation that must produce a drawing dimension**. And
"minimize disassembly after balancing" is an **assembly-sequence constraint** that
the engine assembly model can express.

### 2.4 Configuration control that SP-8109 asks for explicitly  **[V]** — §3.3.3, printed p. 72

> "The direction of rotation should be clearly specified on design layouts. The
> detail and assembly drawings should be checked against this control document.
> In addition, direction should be verified at assembly."

A monograph from 1973 is asking for exactly the identity discipline that
`docs/INTEGRATION_PROGRAM_2026-08-22.md` §2.1 specifies and the tree does not yet
have: one control document, downstream artifacts checked against it, and a
physical verification at assembly.

### 2.5 Additive manufacturing — Gradl, NASA MSFC  **[T]**

`propulsion_texts/gradl2017.pdf` (SLM GRCop-84 combustion chambers) gives design
rules that apply directly to an additively-manufactured impeller or volute:

- **Surface finish ≈ powder diameter.** "SLM processing leaves a surface finish
  roughly equivalent to the diameter of the powder used. Typically a powder cut
  in the −325 mesh (<40 micron) range is used… the as built surface would have
  the rough equivalent of a 63 microinch machined surface."
- **Internal passages cannot be finished.** "it is very difficult and more likely
  impossible to machine the interior cooling channels of liners and other SLM
  parts. The rough surface will increase turbulent flow during engine operation.
  This will increase the pressure drop but also increase the heat transfer to the
  coolant."
- **Observed GRCop-84 finishes**: one chamber at 400 µin; development work
  ranging **200–500 µin**.
- **45-degree overhang rule** applied as the design constraint, with exceptions
  only for small features.
- **Powder removal is a first-class design constraint**, not a post-process
  detail: CT scanning revealed substantial trapped powder; ports had to be
  drilled and tapped; ultimately a part had to be cut open. Recommendations
  include leaving one set of channels open for access and designing features
  specifically to allow powder removal.
- Common build failures: gross and micro-cracking from residual stress, and
  material lifts from the re-coater blade catching, often with short powder
  feeds. Fixes were **thinning material, adding support features, and changing
  wall and overhang angles** — i.e. the process changed the geometry.

**The load-bearing consequence.** AM does not merely add a checklist. It changes
model inputs: an as-built 200–500 µin internal passage is not the smooth wall the
repository's Darcy/Blasius pressure-drop and Sieder–Tate heat-transfer models
assume. A process model that only says yes/no has missed the point; it must
**hand a roughness back to the physics**.

---

## 3. Meanline relations the geometry must honor

All **[T]** unless marked. Section numbers are SP-8109's.

### 3.1 Coefficients and their ranges — §2.3.1.2, printed p. 28

- **Head coefficient ψ**: "Centrifugal-pump head coefficients vary from
  approximately 0.35 to more than 0.70. The lower head coefficients are obtained
  with a small number of blades (3 to 5); the higher head coefficients require
  many blades (20 to 60)."
- **Flow coefficients**: "Both the impeller inlet and discharge flow coefficients
  may vary from as low as 0.05 to 0.30."
- **Meridional velocity ratio**: "The discharge meridional component of velocity
  c_m2 may vary from **1 to 1.5 times** the impeller inlet velocity. The impeller
  inlet velocity is considered to be the velocity at the exit of a coupled
  inducer or at the equivalent exit of an integral inducer."
  *(Already implemented as `cm_ratio_status`; the station definition in the second
  sentence is the part worth re-checking in code — it is the inducer exit, not the
  eye.)*
- **Selection logic**: φ₁ is selected to satisfy required suction performance;
  φ₂ is determined by the blade angle **as limited by stress** and the desired ψ.
  "Higher head coefficients are generally used for low-specific-speed
  applications… With high-specific-speed pumps, low head coefficients are accepted
  in order to achieve the blade length along the shroud needed for good efficiency
  over a wide flow range."
- ψ = f(φ₂, β₂, Z₂), eq. (12); S_s = f(φ₁, β₁, ν, t₁), eq. (13), where ν is the
  inlet hub-to-tip diameter ratio and t₁ the inlet blade thickness.

### 3.2 Blade number and angle — §2.3.1.3, printed p. 29

- Discharge blade angles in flown rocket pumps have ranged **22.5° to 90°**;
  blade numbers **6 to 48** (table I).
- "From the fabrication standpoint, the minimum number of blades is desired."
- "Backswept impellers (blade angles less than 90°) with low head coefficients
  tend to have wider and more stable operating range than radial-bladed
  impellers. The range of high efficiency for pumps with low-head-coefficient
  impellers is, however, smaller."
- The minimum blade number satisfying **impeller diffusion / suction-surface
  velocity-gradient limits** is fig. 16 **[F]**, for shrouded impellers with
  δ = 0.65. *(Already digitized in `pumps.sp8109_min_blade_count` at ±0.02 ψ.)*
- **Splitters**: Anisimov (ref. 51) proposed partial-length blades between
  full-length blades to improve flow range by reducing boundary-layer thickness;
  rocket-industry test results agree. "The state-of-the-art practice is to use a
  small number of inlet blades (8 or less) with additional blades at the discharge
  as required to meet the design head coefficient." *(This is exactly the
  `inlet_blade_count` + `splitter_blade_count` split already implemented.)*

### 3.3 Blade thickness from stress — §3.3.2.4, printed p. 70  **[T]**

For preliminary analysis, for a liquid of specific gravity near 1, eq. (22):

```
t ≈ 0.087 · b_t2 · U_t2 / (sin β_t2 · σ_allowable)
```

with t blade thickness, b_t2 impeller tip width, U_t2 tip speed, β_t2 tip blade
angle. **[T] — the constant 0.087 and the equation form should be rendered before
being coded.**

And the loading rule: "Calculation of blade stresses should include a **cyclic
pressure loading 30 percent of the steady-state hydraulic loading**."

Hub-to-tip radius ratio is explicitly a **compromise between blade tip speed and
blade root stress**: large ratio → larger tip diameter and fluid loads but smaller
root stresses; small ratio → smaller tip diameter and loads but larger root
stresses.

**Discrepancy against the tree.** Blade thickness in `raosim/pumps.py` is a
geometric ratio (`blade_thickness_ratio` default 0.012) with a 0.4 mm floor whose
provenance is a mini-centrifugal-pump paper. SP-8109's is **structural** — it
falls out of tip speed, tip width, blade angle and allowable stress. The current
value is not wrong as a floor; it is unattached to the load. Fatigue margin is
assessed on a **modified Goodman diagram** (fig. 23, printed p. 38 **[V]**, seen
on the rendered page) with σ_alt against σ_mean, endurance limit, yield and
tensile strength — that is the criterion the structural screen should target.

### 3.4 Suction performance — §2.2.1.2, printed p. 11  **[T]**

- Corrected suction specific speed, eq. (9): `S'_s = S_s / (1 − ν²)^½`, where
  ν = D_h/D_t1 — "the suction specific speed of a hypothetical inducer with zero
  inlet hub diameter that operates with the same inlet axial velocity, inlet tip
  speed, rotational speed, and minimum required NPSH as the test inducer."
- "rocket engine pumps have been operated with suction specific speed
  capabilities ranging from **15 000 to over 40 000**" (US units) when pumping
  propellants with vapor pressures similar to cold water.
- **The measurement convention, and it is not the commercial one**: fig. 4's data
  points are at "**pump data … for 2-percent head loss, inducer data for
  10-percent head loss**." The industrial convention is 3 %. A measurement
  ingested against the wrong convention is a silent error of exactly the class the
  measurement schema exists to prevent.

### 3.5 Efficiency — §2.2.2, printed p. 14–15  **[T]**

Why rocket pumps sit below commercial pumps at the same Ns, stated causally:

1. they run at much higher speed, so for a given Q and Ns they are **physically
   smaller** than commercial pumps;
2. high speed and reactive propellants require **larger operating clearances**;
3. high suction specific speed requires **increased inlet diameter**, which
   reduces efficiency at a given Ns (fig. 5 **[F]**).

Design factors: open-face vs fully shrouded, vaned diffuser vs volute, staging,
and head coefficient (fig. 3 **[F]**).

**Note for the tree.** `raosim/mdo/pump.py:50 pump_efficiency` is a log-Gaussian
in Ns with three coefficients on `MissionSpec`
(`raosim/mdo/schema.py:571-573`). Its own docstring already says it is "a smooth
screening surrogate, not a fit to SP-125 Fig. 6-23 (image-only)". SP-8109 fig. 5
is the *other* image that would let this be a real fit, and it carries δ = D_t1/D_t2
as a second parameter the current surrogate does not have. Digitizing fig. 5 is a
bounded, self-contained task that would convert a declared guess into a sourced
surrogate — and it is the natural first calibration target.

### 3.6 Off-design stability — §2.2.3, printed p. 22  **[T]**

- "the pump with the steepest negative slope of the head-vs-flow curve is most
  stable in a given rocket engine system and therefore will operate over the
  widest flow range."
- Hansen (ref. 50): widest negative-slope flow range with a **small** blade number
  and a moderate head coefficient (ψ ≈ 0.5). A 7-blade impeller with
  rising-head-to-shutoff reached ψ = 0.523 at BEP with a vaned diffuser and
  volute; with a **vaneless** diffuser and volute the blade number had to drop to
  4 for rising-head-to-shutoff, giving ψ = 0.388 — at a substantial efficiency
  cost. β₂ = 20.75° in both. Specific-speed range 700–1300.
- Contrarily, experimental F-1 oxidizer pumps doubled discharge blade number 6 →
  12 and raised BEP head coefficient 0.42 → 0.49 while **maintaining or improving**
  the negative-slope flow range, with a slight efficiency gain.
- "For pumps with a given head coefficient, an impeller with the largest blade
  number will result in the steepest slope."
- Smaller hub diameter → superior flow range at comparable inlet flow coefficient
  and tip blade angle. Light hydrodynamic loading in the inlet region also
  improves flow range.
- LH2 specifically: internal heating at reduced flow can cause loss of pumping
  ability from backflow of internally generated gaseous hydrogen; increased inlet
  pressure delays it. Pump bypass is limited to what yields **20 % by volume gas
  flowrate at the pump inlet** (refs. 39, 40).

### 3.7 A documented post-manufacture trim — §2.3.1.2, printed p. 28  **[T]**

> "The impeller head coefficient may be increased or decreased by underfiling or
> overfiling the impeller trailing edge as shown in figure 15. **The pump head and
> horsepower can be varied by as much as 10 percent in this manner with little or
> no change in efficiency.**"

Under-filing (material removed from the pressure side at the TE) **increases** the
head coefficient; over-filing **decreases** it.

This matters far beyond blade detail. It is documented evidence that these pumps
were **adjusted after fabrication, against test results, by a bounded ±10 % trim
that does not cost efficiency**. The "measure the article, then adjust" loop is not
a modern software conceit being retrofitted onto 1970s hardware practice — it *is*
1970s hardware practice, and SP-8109 quantifies the authority the trim has. A
second instance appears in §2.5.2.1: on the J-2 oxidizer pump, "control of the
pressure gradient in that region **by trimming the ribs** permits adjusting the
axial thrust of the impeller without changing the diameters of the impeller wear
rings."

Both trims should be first-class objects in the toolkit: a declared adjustment
range, a measured input, and a resulting as-built configuration linked to the
nominal one.

---

## 4. The collector: diffuser and volute

The repository builds **rectangular boxes** for diffuser vanes and a
constant-mean-velocity volute on a constant-radius centerline with terminal area
equal to the outlet-port area. SP-8109 §2.4.1 says what the real construction is.

### 4.1 Vaned diffuser — §2.4.1.2.2, printed p. 45–46  **[T]**

- **Vane inlet angle β₃ matches the entering fluid flow angle at design flow.**
  For a throttled engine, β₃ may instead be matched to the inlet flow angle at a
  flow coefficient as low as **80 % of the nominal** value for maximum thrust.
- **Vane width**, eq. (19): `b₃ = (0.9 to 1.0) · b_t2`, "and the side walls are
  rounded or faired. These practices minimize flow separation even under
  conditions of axial misalignment with the impeller."
- **Diffusion factor**, eq. (18): for a single stage of diffusion,
  `D = (p_s3 − p_s,min) / (p_t3 − p_s,min) ≤ 0.6`.
- **Throat-to-discharge relation is set by a cone angle, normally 7° to 10°**
  (ref. 71). Same requirement for vane-island diffusers, which are used when the
  diffuser must carry casing structural forces.

### 4.2 Blade/vane count interaction — §2.4.1.2.3, printed p. 46  **[T]**

This is the real form of "the vane count must not share a factor with the blade
count", and SP-8109 gives the criterion rather than the folklore:

> "An additional objective when using a vaned diffuser with a volute is to avoid
> the possibility of wave reinforcement of the pressure waves that result from the
> interaction of the impeller blade wakes with the diffuser vanes. The impeller
> discharge blade number Z₂, diffuser vane number Z_d, and volute flow-path length
> πD_v are important design parameters. Superposition of pressure waves can result
> in large amplitude oscillations in discharge pressure."

Reinforcement of the j-th harmonic occurs whenever the reinforcement index m is an
**integer**, eqs. (20) and (21), with j the harmonic order, D_v the average
distance from pump centre to volute passage centre, a the speed of sound in the
liquid, and W the average relative velocity in the volute passage. **[T] — the two
equations are OCR-scrambled and must be rendered before implementation**, but the
*criterion* — m must not be an integer for the harmonics of interest — is
unambiguous, and it is a computable gate over (Z₂, Z_d, D_v, a, W).

### 4.3 Volute area law — §2.4.1.3.1, printed p. 47  **[T]**

Two methods, and SP-8109 is explicit about the trade:

> "**Constant moment of momentum.** The fluid tangential velocity is assumed to be
> inversely proportional to the radius… The constant-moment-of-momentum method was
> applied to the design of Titan I and Titan II pump housings; corrected for
> friction losses (ref. 74), the method was used in the design of the J-2S Mark 29
> fuel pump, which experienced very light radial bearing loads.
>
> **Constant mean velocity.** The velocity is assumed to be constant, and therefore
> the cross-sectional area is increased proportionally as the central volute wrap
> angle increases. **The method was developed as a simplification in volute
> design.**
>
> Although there are only minor differences in pump efficiency between the two
> methods (ref. 64), the unsymmetrical pressure (with its associated **radial
> hydraulic forces upon the impeller**) around the volute passage has been found to
> be **higher** in designs based on the constant-mean-velocity method (ref. 63)."

**This settles the volute question, and it settles it in a way that matters more
than shape fidelity.** The repository's `A_exit · θ/2π` law is the named
simplification. Its documented penalty is not efficiency — it is **higher radial
load on the impeller**, which is the load that sizes the radial bearing, the shaft,
and the critical speed. The volute simplification therefore propagates directly
into the mechanical design the repository does not yet have. Fixing the volute law
before sizing bearings is not aesthetic sequencing; it is the difference between
sizing a bearing against a real load and sizing it against an artifact of a
simplification.

Supporting geometry, same section:

- **Objective**: "provide a distribution of cross-sectional area with respect to
  wrap angle that will yield a constant impeller discharge static pressure at the
  design point… the radial load on the shaft and the impeller vibrations are
  thereby minimized."
- **Asymmetric cross-section preferred** — "it produces a single vortex that is
  stable and that improves the efficiency of the conical diffuser at the exit."
- **Exit conical diffuser included angle**: circular cross-sections **7°–9°**;
  square **6°**; two parallel walls **11°**.
- **Tongue**: "The inlet angle of the volute tongue is designed for **zero
  incidence angle at the design flow** in order to reduce the losses associated
  with the volute, to minimize local pressure differences, and to reduce the
  amplitude of the pressure oscillations in the pump discharge."
- **Diffuser-to-tongue transition**: if a vaned diffuser precedes the volute, the
  transition "must be designed to avoid an interaction that would lead to an
  unstable (e.g., bi-stable) pump head-versus-flow characteristic. Stable flow is
  achieved by **fairing one diffuser vane into the volute tongue** or by leaving a
  **large clearance** between the vane discharge and the tongue."
- **Off-design radial load** (§2.4.1.3.2): reduced by splitter vanes, multiple
  tongues, double-outlet volutes, or vaned diffusers. "The greater the number of
  symmetrically located splitter vanes, the better is the balance of radial
  thrust."

---

## 5. Axial thrust: forces, balance devices, and where the residual goes

§2.5, printed p. 57 onward. All **[T]**.

### 5.1 Where the forces come from

- Pressure gradients on the **smooth non-pumping impeller hub and shroud
  surfaces**, and on the **open faces** of open impellers. Gradients on the
  non-pumping surfaces "caused by viscous forces may be calculated by the
  procedures presented in reference 79"; open-face pressures by refs. 53, 54, 80.
  **SP-8109 defers the actual calculation to references not held locally** — see
  the literature request list.
- In a turbine-driven pump, turbine axial forces are balanced by the pump axial
  thrust system. **An electric-pump architecture removes that term**, which is a
  simplification the toolkit should state rather than inherit silently.

### 5.2 Balance devices, in SP-8109's own order of preference

1. **Impeller wear rings (impeller seals)** — front shroud and hub. The area
   inboard of the hub wear ring is held slightly above impeller inlet pressure by
   routing wear-ring leakage back to the inlet through **balance holes** in the
   impeller or external passages. "The relative diameters of the two wear rings
   are sized to produce the required balance force."
2. **Impeller balance ribs** — blades on the back of the impeller hub forming a
   low- or zero-flow impeller. Holes through the impeller vent the inner region
   and provide coolant flow to prevent cavitation from rib pumping work. "usually
   more development work is required to obtain a configuration that is as
   effective as wear rings." §3.5.2.1 recommends **wear rings over balance ribs**
   because "wear rings are not subject to force changes caused by cavitation."
3. **Balance pistons and hydrostatic bearings** — "When pumps operate at very high
   speeds, ball bearings are not capable of sustaining the normal operating
   unbalanced axial forces." Series-flow balance piston integral with the impeller,
   or separate parallel-flow hydrostatic bearing (ref. 45). Both self-compensating,
   seeking an operating clearance; "designed to operate with a sufficient effective
   spring rate to **avoid axial resonances** of the rotating assembly."

**The electric-pump reading of item 3.** An electric-pump engine lives at
30–60 krpm. SP-8109 says that is precisely the regime where ball bearings stop
being able to take the residual axial load. So the axial-thrust chain is not
optional detail for this architecture — it decides whether a rolling-element
bearing is admissible at all.

Field experience worth carrying as design constraints: F-1 fuel turbopump
lead-plated wear rings "rolled up" on rubbing and wore the impeller back disk;
fixed by improving the lead bond and **enlarging clearances**. J-2S balance-piston
rub rings of fibreglass-reinforced Teflon delaminated and opened the orifices;
J-2 axial fuel pump carbon rub rings cracked; **both solved with lead-filled
porous bronze**.

---

## 6. Rotordynamics: the criterion, stated

§2.2.1.1, printed p. 8–9. **[T]**

Two design philosophies:

- **Below the first rigid-body whirl critical.** Requires high bearing spring
  rates; roller bearings often at both ends plus ball bearings for axial thrust.
  "Nearly all dense-fluid turbopumps operate below the first critical speed."
- **Above the first and second whirl criticals, but below any shaft-bending
  mode.** Requires lower bearing/support spring rates and minimal internal
  bearing looseness; **only preloaded ball bearings**, often duplex for radial
  capacity. Risk: subsynchronous whirl instability, and excessive bearing dynamic
  loads at partial speed unless damped (ref. 31).

**The criterion, common to both:**

> "In both design approaches, a **margin of approximately 20 percent** is allowed
> between the shaft operating speed and the nearest calculated whirl critical
> speed."

and the operating rule:

> "sustained operation during rocket engine mainstage at speeds between **80 and
> 120 percent** of a shaft critical speed set by bearing spring rate is avoided.
> Limited operation during development tests designed to evaluate amplitudes at
> critical speed is allowable."

Passing through resonance on start and shutdown transients is acceptable for many
designs. Amplitude at resonance is set by damping and by the energy input, which is
"usually set by rotor imbalance" — closing the loop back to §3.3.3's imbalance
formula. Practical mitigations: light, carefully balanced hardware; short axial
dimensions; flow passages shaped to yield optimum bearing spans.

**What is missing to compute this.** SP-8109 states the margin but defers the
critical-speed calculation to ref. 28, and bearing speed limits to ref. 43, and
shaft seal types and speed limits to ref. 44. None of the three is held locally.
That is why `LITERATURE_REQUESTS.md` items **13 (SP-8048 bearings)**, **14
(SP-8101 shafts and couplings)** and **15 (SP-8121 rotating-shaft seals)** are the
correct asks and remain genuinely blocking — the margin criterion is known; the
stiffness and load data needed to evaluate it are not.

---

## 7. Clearances, shrouding, and materials

§3.3.2.6 and §3.3.4, printed p. 71–73. **[T]**

- "It is recommended that **shrouded impellers with nonmetallic wear rings** be
  selected whenever tip speed limits are not high enough to prohibit this design.
  The use of shrouded impellers permits large axial and radial clearances except
  for the wear ring. The larger housing deflections tolerable with shrouded
  impellers can result in **minimum pump weight**." *(Note the direction: shrouding
  buys mass, not costs it, by relaxing the deflection budget.)*
- Open-face impellers required above 2200 fps.
- For **LF₂**, wear-ring radial clearance must preclude rubbing entirely. For less
  reactive propellants light rubbing is allowable if side loads are small, no
  particles are generated, and no heat-driven chemical reaction can occur.
- **Inert wearing materials — Teflon, Kel-F, silver — recommended for oxidizer
  pumps.**
- Unshrouded tip clearances minimized within practical mechanical limits, with
  distortion and thrust excursions accounted for; tip-clearance-ratio effect on
  efficiency in fig. 21 **[F]**.
- Materials: select for **ductility** in high-stress applications, to relieve stress
  concentrations by local yielding. Evaluate modulus, density, ductility, fatigue
  strength, **damping**, propellant compatibility and thermal contraction
  coefficient. "**Strength-to-weight ratio should not be the dominant criterion.**"
  Below 5 % ductility, notch effects must be considered explicitly.
- Fatigue/production notes: casting cracks from residual stress eliminated by
  proper technique (chills) and heat treatment; **endurance limits increased 40 %
  by shot peening** to induce surface compressive stress.
- Stress prediction status, stated candidly: "The present state of the art of
  impeller structural design permits the prediction of the minimum required
  material thicknesses for most of the shroud, blades, and disk. However, regions
  where stress concentrations exist are difficult to analyze, and **spin tests in
  air using stress coat and strain gages are required** to determine the magnitude
  of these local stresses."

That last sentence is the honest ceiling on any analysis-only claim about the
impeller, and it should be quoted in the release-readiness record rather than
paraphrased.

---

## 8. Torque path and instrumentation designed into the part

§3.3.2.5, printed p. 70. **[T]**

- "Torque should be transmitted into regions of the impeller where the combined
  loads can best be withstood without failure. Contact forces of the
  torque-transmitting devices (**splines, curvics, pins, or bolts**) should be
  large enough to **prevent fretting, which has caused explosions**; lubricants
  that are compatible with the propellant can be used to minimize fretting."
- Shaft torque capacity must cover off-design operation and include allowance for
  errors in estimating component efficiency and efficiency shift with flow
  coefficient.
- And an instrumentation requirement expressed as **geometry**:

> "It is recommended that **wrench flats or wrench-type surfaces and torque-wrench
> access or turbine-drive access be provided** to facilitate breakaway-torque and
> drag-torque measurements. Provide instructions in the assembly procedure with
> limits based upon measurement data from new, damaged, and used but undamaged
> comparison units."

This is the single clearest statement in the corpus that **measurement provisions
are part design, decided before fabrication** — and that acceptance limits come
from a *population* of comparison units, not a single prediction.

---

## 9. How these pumps are actually tested

From Reis et al., *Design and Testing of Centrifugal Fuel Pump for Liquid
Propellant Rocket Engine L75*, Instituto de Aeronáutica e Espaço, 2019
(`propulsion_texts/fuel_pump_design/reis2019.pdf`). **[T]**

The L75 ethanol pump: two-stage centrifugal, radial shrouded impeller, upstream
open axial **two-bladed** inducer, **floating ring seals front and rear** of the
impeller sized to reduce both leakage and axial force, radially symmetric inlet,
**circular-section single volute without a diffuser**. Design method was
one-dimensional theory plus polynomial fits to historical empirical data
(Ovsyannikov), with inducer and impeller designed together through a cavitation-
coefficient method.

The BTBH test bench, and what it implies for a measurement schema:

| Test-bench fact | Schema consequence |
|---|---|
| "a full-scale ethanol pump demonstrator using **water** as the [working fluid]" | The record must carry **test fluid separately from design fluid**, and the comparison layer must apply an explicit similarity transformation. A datapoint without its test fluid is uninterpretable. |
| Closed water circuit: two pressurized tanks, filter, flowmeters, regulation valves | Facility configuration is a **condition**, not metadata. |
| "pressure and temperature transducers, pressure dynamics sensors, accelerometers and displacement sensors" | Channels are typed and heterogeneous; steady and dynamic instruments answer different questions. |
| Inlet pressure controlled by compressed air via regulator; "low level inlet pressures (around 0.35 bar) can be achieved by pneumatic schemes and calibrated orifice" | The **independent variable is actively controlled**, and its setting method belongs in the record. |
| "The inlet pressure fluctuation was measured at **50 mm upstream of the inducer leading edge**" | **Station is a required field with a datum**, not a label. |
| "an external acquisition system (based on HBM modules)… in order to eliminate electric noise on the data signal" | Acquisition chain identity affects data quality and must be recorded. |
| "At flow rate 0.6 the head fall can not be measured because of **vacuum limitations**" | A datapoint can be **unavailable for a facility reason**. That is a quality flag with a cause, not a missing value — the same `unavailable`-with-reason discipline the repository already applies to models. |
| "the Head values measured are **below the prediction curve**, mainly for lower flow values" | The interesting output is a **signed residual as a function of the operating point**, not a pass/fail. |
| Vibration in **g_RMS** plotted over the cavitation curves | Some measured quantities have **no model counterpart at all** and must still be storable. |

---

## 10. Discrepancy ledger against the working tree

Findings only; nothing here has been changed.

| # | Repository | Literature | Severity |
|---|---|---|---|
| D1 | No machinability ceiling on blade count anywhere in `raosim/pumps.py` / `raosim/pump_cad_brep.py` | SP-8109 §2.3.3 / §3.3.3 **[V]**: `Z₂ ≤ ≈ 28 sin β₂` for shrouded machined impellers, from cutter clearance and tool L/D ≤ 10 (Al) / 8 (Ti) | **High** — an unmakeable design can be emitted today, and the bound is already on disk |
| D2 | `material_tip_speed_limit = 350.0` m/s, a single scalar (`pumps.py:150`), screened at 0.8× (`:1334`) | SP-8109 makes the limit a function of (material, process, shrouded/open, fluid): 1000 / 1400 / 2200 / 2500 / 2870 fps by configuration **[V/T]** | **High** — the scalar is the same variable that should select the process route |
| D3 | Volute law `A_exit·θ/2π`, constant radius centerline, terminal area = port area | SP-8109 §2.4.1.3.1 **[T]**: constant-mean-velocity is the *named simplification*; documented penalty is **higher radial hydraulic force on the impeller**, which sizes bearing/shaft/critical speed | **High** — propagates into mechanical design, not just shape |
| D4 | Diffuser vanes are rectangular boxes (`build_diffuser_ring`); throat on a radial-area basis | SP-8109 §2.4.1.2.2 **[T]**: β₃ matched to flow angle (or to 0.8 φ if throttled), `b₃=(0.9–1.0)b_t2`, faired side walls, `D ≤ 0.6`, throat→discharge cone 7°–10° | **High** |
| D5 | No Z₂/Z_d wave-reinforcement check | SP-8109 §2.4.1.2.3 eqs. (20)/(21) **[T]**: reinforcement index m must not be integral over (Z₂, Z_d, πD_v, a, W) | Medium |
| D6 | Blade thickness = geometric ratio 0.012 with a 0.4 mm floor | SP-8109 eq. (22) **[T]**: `t ≈ 0.087 b_t2 U_t2 / (sin β_t2 · σ_allow)`, plus a **cyclic load 30 % of steady hydraulic load** and a modified-Goodman margin (fig. 23) | Medium — current value is a floor, not a load-derived thickness |
| D7 | No spin-test requirement generated | SP-8109 §3.3.3 **[V]**: above 1000 fps, spin to ≥ 1.2 × max operating speed, temperature-corrected | Medium — a required-inspection record derivable from a solved quantity |
| D8 | No residual-imbalance requirement | SP-8109 §3.3.3 **[V]**: `oz-in ≈ 4 W[lb] / N[rpm]` | Medium |
| D9 | No surface-finish attribute on any part; regen Δp and heat transfer assume smooth walls | SP-8109 **[V]**: cast Al 63 µin ≤ 10 in dia, 125 µin for 10–20 in. Gradl 2017 **[T]**: SLM as-built 200–500 µin, interior passages unfinishable | **High** — process determines a *physics input*, not just a yes/no |
| D10 | `cm2/cm1` screened 1–1.5 | Correct per §2.3.1.2 **[T]**, but the source defines cm1 at the **inducer exit** (or equivalent exit of an integral inducer), not the eye. Worth re-checking the implemented station. | Low, but silent if wrong |
| D11 | `pump_efficiency` log-Gaussian in Ns only (`mdo/pump.py:50`) | SP-8109 fig. 5 **[F]** carries efficiency against Ns **and** δ = D_t1/D_t2, and states the causal reasons for the rocket-vs-commercial offset | Medium — the missing second parameter is the suction/efficiency trade the tool is supposed to expose |
| D12 | No axial-thrust force accounting; wear rings placed at "equal-diameter neutral start" | SP-8109 §2.5 **[T]**: relative wear-ring diameters are the *sizing variable*; balance-hole area rule; and above a speed threshold ball bearings cannot take the residual at all | **High** for an electric pump at 30–60 krpm |

---

## 11. What the corpus supports, and what it does not

### Supported today, with a citation and a number

Meanline coefficient ranges and their selection logic; blade number minimum
(fig. 16, digitized) and machinable maximum; blade thickness from stress; the
modified-Goodman fatigue criterion; diffuser vane angle, width, diffusion factor
and cone angle; the wave-reinforcement criterion; both volute area laws and the
radial-load reason to prefer one; volute tongue incidence and exit-cone angles;
axial-thrust device selection and the wear-ring/balance-hole rule; the 20 %
critical-speed margin and the 80–120 % avoidance band; tip-speed-keyed process
selection; cutter L/D limits; cast and SLM surface finishes; spin-test and
imbalance requirements; clearance and wearing-material practice; the torque-path
and anti-fretting rules; the wrench-flat instrumentation provision; the 2 %/10 %
head-loss measurement conventions; and a documented ±10 % post-test head trim.

### Not supported — and what would close it

SP-8109's reference list was rendered and read this session (printed p. 102, PDF
p. 114), which resolved most of these deferrals to named documents. Full citations
and acquisition order are in `LITERATURE_REQUESTS_HARDWARE_LOOP.md`; the summary:

| Gap | What is needed | Note |
|---|---|---|
| Bearing load ratings, DN limits, cryogenic life | **NASA SP-8048** *Turbopump Bearings* (Mar 1971) | SP-8109 ref. 43, confirmed verbatim. `LITERATURE_REQUESTS.md` item 13. |
| Shaft sizing, couplings, critical-speed calculation method | **NASA SP-8101** *Turbopump Shafts and Couplings* | SP-8109 defers the method to its ref. 28. Item 14. |
| Rotating shaft seal design, leakage, purge, speed limits | **NASA SP-8121** *Turbopump Rotating-Shaft Seals* | SP-8109 ref. 44, listed as "to be published" in 1973. Item 15. |
| **Pressure distribution on non-pumping impeller surfaces — the axial-thrust integrand** | **Due, H. F. Jr., *An Empirical Method for Calculating Radial Pressure Distribution on Rotating Discs*, J. Eng. Power, Trans. ASME Ser. A, vol. 88, 1966, pp. 188–196** (ref. 79); Young & Due, PWA FR-1761 (ref. 80) for open faces | **Now a named citation.** The balance calculation cannot close without it. |
| Balance pistons and hydrostatic bearings | **Watters & Luehr, NASA CR-72630**, Aerojet-General, Mar. 1970 (ref. 81) | NASA CR — likely free. Required for the regime where ball bearings cannot take the residual axial load. |
| Volute area law with friction correction | **Brown & Bradshaw, NACA Report 936, 1949** (ref. 74) | **NACA report — free.** The uncorrected constant-moment-of-momentum law is implementable today from §2.4.1.3.1 alone. |
| Off-design radial load data | **Knapp, Trans. ASME vol. 63, 1941, pp. 251–260** (ref. 75) | the source of fig. 29. |
| Quasi-3-D impeller flow analysis | **Katsanis, NASA TN D-2546 (1964)**; **Katsanis & McNally, NASA TN D-5044 (1969)** (refs. 53, 54) | **Both free on NTRS.** Also the basis for choosing a blade-angle distribution. |
| Low-thrust pump design and testing | ***Rotating and Positive-Displacement Pumps for Low-Thrust Rocket Engines*, NASA CR-72965 / CR-72966**, Rocketdyne (ref. 55) | Directly addresses where the centrifugal meanline stops being right as thrust falls — the Track L question. |
| Impeller disc stress method | **Unobtainable.** SP-8109 refs. 82 and 83 (Aerojet-General, 1963) are **Confidential** (AD 350716, AD 340146) | A substitute rotating-disc stress reference is required, plus the *Aerospace Structural Metals Handbook*, AFML-TR-68-115 (ref. 78) for allowables. |
| Diffuser area-distribution evaluation | SP-8109 ref. 70 (a computer program) | Likely unobtainable; the diffusion-factor limit D ≤ 0.6 is the usable substitute. |
| Slip factor, if ever wanted explicitly | **Wiesner, *A Review of Slip Factors for Centrifugal Impellers*, J. Eng. Power, Trans. ASME Ser. A** (ref. 56) | SP-8109's own ψ = f(φ₂, β₂, Z₂) formulation avoids needing one; this is the fallback. |
| LOX materials compatibility | **MSFC-Spec-106A**, *Testing Compatibility of Materials for Liquid Oxygen Systems*, Sept. 1966 (ref. 84) | Converts the "Teflon, Kel-F, silver" recommendation into a test criterion. |
| Pump acceptance-test procedure and uncertainty | **ISO 9906** or **ANSI/HI 14.6**; **ASME PTC 8.2** | Nothing in the corpus defines a *test standard*. SP-8109 gives conventions (2 %/10 % head loss) but not a procedure. **New request.** |
| Measurement uncertainty framework | **ISO/IEC Guide 98-3 (GUM)** or **ASME PTC 19.1** | Required to make the uncertainty field of a measurement record mean something. **New request.** |
| Flow measurement | **ISO 5167** (orifice/venturi/nozzle) | Reis 2019 uses flowmeters and a calibrated orifice with no stated uncertainty. **New request.** |
| Achievable tolerance by process; fits | **ISO 2768**, **ISO 286** | ASME Y14.5 defines the *language* of tolerance, not the *achievable values*. **New request.** |
| Bolted-joint preload | **VDI 2230** or equivalent; a selected fastener family | Already flagged in `COMPLETION_PLAN.md` G5 as "partial — request fastener selection/standards". |
| Ruled-surface / flank-milling geometry for impeller blades | A turbomachinery blade-geometry reference | Needed to decide whether the generated blade is 5-axis flank-millable or must be point-milled. **New request.** |

---

## 12. The single most useful thing this reading changed

Before: the pump work looked like three parallel problems — 3-D geometry,
manufacturing, and mechanical design — with manufacturing blocked on literature we
do not have.

After: **manufacturing is the least blocked of the three.** SP-8109 §2.3.3 and
§3.3.3 are a quantitative process model for exactly this part, already on disk,
already verified against the scan. The **mechanical** side is where SP-8109
consistently states the *criterion* and defers the *data* — but rendering its
reference list turned almost all of those deferrals into named documents, several of
them free NACA/NASA reports. So the mechanical track is not blocked on *searching*;
it is blocked on one afternoon of downloading, plus three NASA monographs that have
been on the request list since before this session.

The one genuine dead end is worth stating separately: SP-8109's impeller
mechanical-design references are **Confidential**, so its structural method cannot be
recovered from its own bibliography. That is not a gap to be closed by looking
harder — it needs a declared substitute, and the monograph's own position is that
spin testing rather than analysis is what establishes local stress at concentrations
anyway.

And the three problems are not parallel. Tip speed selects the process; the volute
law sets the radial load that sizes the bearing; the wear-ring diameters that
balance axial thrust are also the clearance that sets volumetric efficiency, which
feeds back into the head the meanline had to produce. The couplings run in both
directions, which is an argument for building the configuration object *first*, so
that each of these solved quantities has one place to live.
