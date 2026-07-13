# Pintle Injector: Literature-Grounded Evaluation of the Current Approach

This document (1) states the governing equations for pintle design **as verified
against the corpus**, (2) audits `raosim/injector.py` line-by-line against them,
(3) maps the chamber/nozzle/feed-system coupling, and (4) specifies the
additional CLI inputs and parameter propagation needed. Every correlation is
attributed; where a coefficient is hardware-specific or could not be transcribed
cleanly from a scanned source, that is stated rather than asserted. Companion
file: `propulsion_texts/pintle_injector/PINTLE_LITERATURE_CATALOG.md`.

**Bottom line.** `injector.py` sizes from the solved nozzle/chamber operating
point and uses the correct master variables (TMR and blockage factor). It now
has two deliberately distinct architectures: fixed slots/holes, and a Son et
al. (2017) continuous movable radial gap. The latter implements the published
tip-area/center-gap geometry, a physical opening solve, fixed-hardware throttle
semantics, and strict evidence gates for configuration-specific discharge
coefficient, actuator authority, and any liquid-sheet handoff. It is
software-verified preliminary design, not cold-flow or hot-fire validation.
The exact model contract and CLI example are in
[`MOVABLE_PINTLE_MODEL.md`](MOVABLE_PINTLE_MODEL.md).

---

## 1. Verified governing equations

### 1.1 Hydraulic sizing (well-established, matches code)

Per-stream incompressible orifice flow — Sutton & Biblarz *Rocket Propulsion
Elements*; NASA SP-8089 (Gill & Nurick 1976):

```
mdot = Cd · A · sqrt(2 · rho · ΔP)            (single-phase metering)
v    = mdot / (rho · A)
```

Dimensionless groups (standard; Ohnesorge identity holds):

```
Re = rho·v·D_h/mu      We = rho·v²·D_h/sigma      Oh = mu/sqrt(rho·sigma·D_h) = sqrt(We)/Re
```

**Master geometric variables** (the two knobs every pintle paper sweeps):

```
TMR = (mdot_radial · v_radial) / (mdot_axial · v_axial)      Total Momentum Ratio
BF  = Σ(orifice width) / (π · D_pintle)                       Blockage Factor
```

- TMR form verified against **Hwang 2022, Eq. 1** and **Son 2017 (JPP), Eq. 4** (ratio of radial to axial stream momentum). `size_pintle_injector()` implements exactly this.
- BF form verified against **Hwang 2022, Eq. 3** (`BF = N·w/(π·D_p)`). The fixed-discrete geometry implements it as a fraction; Hwang reports ×100%.
- Annular-stream momentum identity (**Hwang Eq. 2**): `mdot_a = A_a·Cd,a·sqrt(2·ρ_a·Δp_a) = ρ_a·A_a·u_F` — consistent with the code's per-stream solve.

**Empirical anchors for the master variables (use as design targets):**

- **TMR ≈ 0.7** maximizes velocity / c\* efficiency (Austin & Heister; restated in Hwang 2022 and Son 2017). TR-class TRW engines and the LMDE cluster here.
- **BF ≈ 85%** maximizes c\* efficiency (Hwang 2022, citing Ryu) — but see the **thermal counter-pressure** in §1.4: high BF runs the tip hot.
- **Injector stiffness ΔP/Pc ≈ 0.15–0.25**, with ~0.20 the common floor for combustion-stability margin (Huzel & Huang SP-125; SP-8089). The code's default `dp_fraction = 0.2` sits correctly on this rule.

### 1.2 Continuous movable-pintle control area and discharge coefficient

For the Son et al. (2017) center-rod architecture, let
`r_f = D_post/2 - t_post`, mechanical opening be `L_open`, and pintle-tip angle
be `theta_pt`. The implemented form of Son Eq. 1 is

```
A_tip = pi/sin(theta_pt) *
        [r_f^2 - (r_f - L_open*sin(theta_pt)*cos(theta_pt))^2]
      = pi * [2*r_f*L_open*cos(theta_pt)
              - L_open^2*sin(theta_pt)*cos(theta_pt)^2]
```

The expanded expression is numerically stable at zero tip angle, where its
limit is `2*pi*r_f*L_open`. The fixed center-gap cap is

```
A_cg  = pi/4 * (D_cg^2 - D_pr^2)
A_eff = min(A_tip, A_cg)
```

The open stop is required to remain below the transition `A_tip = A_cg`.
Past that transition, pintle travel no longer controls the minimum area; the
solver rejects the condition instead of presenting it as additional throttle
authority. `L_min = L_open*cos(theta_pt)` is used for the thin-gap hydraulic
diameter screen.

**Son 2015** and **Son 2017** show `Cd` for the pintle annulus/center-gap varies
with the pintle **opening distance** (Cd decreased ~linearly up to L_open ≈ 0.75 mm,
then rose as the minimum area transitioned from the tip gap to the fixed center
gap). Reported values are configuration-dependent and do not define a
transferable universal `Cd(L_open)`. Fixed-discrete sizing still accepts an
explicit per-stream `Cd` (0.7 default). The movable branch interpolates a
user-supplied `Cd(L_open/L_max)` map, but it passes the physical gate only when
the map is bound to a source artifact, SHA-256, and exact Son-geometry
fingerprint and the solved fluid,
Reynolds number, pressure drop, temperature, and cavitation number all lie
inside that artifact's declared domain. A constant fallback is retained only
as a labelled, failing preliminary screen.

### 1.3 Spray cone angle and SMD

**Leading-order (what the code does).** With the radial and axial streams treated
as momentum vectors, the spray half-angle is the resultant direction:

```
tan(θ_spray) ≈ radial_momentum / axial_momentum   →   θ_spray ≈ arctan(TMR)
```

This is the accepted first-order relation (the TMR framework of Heister and of
Cheng et al.) and is what `size_pintle_injector()` computes (with an added
deflector term). At TMR = 0.7 it gives θ ≈ 35°, consistent with observed
pintle cones.

**Empirical refinements (in the corpus, not yet in the code):**

- **Son 2015** (J. Thermal Science 24-1): spray half-angle has an **exponentially decreasing** correlation in **(momentum ratio / Weber number)**, i.e. `θ_half ≈ A·exp(−B·M/We)`. The fitted A, B are specific to that test article (gas-centered, water/air cold flow).
- **Son 2017** (JPP 33-4): a **logistic** spray-angle correlation and a **power-law SMD** correlation in the aerodynamic Weber number and a characteristic parameter K:
  ```
  We = ρ_gas·(V_gas − V_liq)²·L_open / σ          (Eq. 5, aerodynamic, on opening distance)
  D32 ∝ 10³·L_open·ξ⁻¹·exp(4.0 − q·We^0.1)  [µm]   (Eq. 9 form; q, ξ hardware-specific)
  ```
  Critically, the spray angle **saturates** between an upper and a lower limit set by the **pintle tip angle** — pure `arctan(TMR)` has no such limit. This is the main physics the code's surrogate misses.
- **Ninish 2018**, **Zhou 2022**, **Freeberg 2019**: independent cold-flow cone-angle / breakup datasets for cross-checking and bounding.

**Caveat that matters for the tool:** these are fits to specific hardware and
propellant pairs. The defensible upgrade is to keep the dimensionless framework
(TMR, We, BF, L_open/tann) and expose the empirical correlations as *optional,
validity-flagged* modes — not to hardcode one lab's coefficients as truth.

### 1.4 Atomization, vaporization, and the thermal coupling

**Atomization screen (code is literature-rooted):** SMD from the Hinze
critical-Weber maximum-stable-drop limit (`d32 = We_crit·σ/(ρ_g·v²)`, We_crit ≈
12–13, Hinze AIChE J. 1955), primary-breakup length ≈ 15·d_jet (Reitz & Bracco),
d²-law vaporization with a separately reported vaporized-liquid fraction (Priem & Heidmann,
NASA TR R-67, 1960). These are sound general references; they are **not**
pintle-specific, so they complement rather than replace Son's SMD law.

The legacy correlation screen does not reinterpret the movable mechanical gap
as a droplet scale. For the continuous movable branch, a typed spray handoff
requires an independently measured or VOF-resolved sheet thickness with a
matching fluid/opening/pressure-drop/mass-flow evidence domain. Without it,
the hydraulic result remains usable but the primary-sheet-to-parcel bridge is
blocked.

**Pintle-tip thermal — the highest-value coupling (Kang 2022):**

- The pintle tip sits in a **high-temperature recirculation zone at the center of the pintle** — this directly **validates the code's recirculation assumption** (`T_aw ≈ 0.8·Tc`) in `face_tip_thermal`.
- Tip damage occurred ~0.7 s into a hot fire on the prototype; mitigations were an orifice-shape change + an insert/cooling device (up to **21.4%** tip-cooling improvement).
- **Quantified couplings the code does not yet capture:**
  - Reducing **BF 86% → 58% lowered pintle-tip temperature 17.3%** (more, narrower jets cool the tip better). So BF trades c\* efficiency (wants high BF, §1.1) against tip survival (wants lower BF) — a genuine optimization, not a single optimum.
  - **TMR↑ raises both c\* efficiency and tip heat flux** — the performance knob is also a thermal-load knob.
- **Ahn 2014:** longer injector **recess → higher chamber-wall heat flux**, with Pc a strong driver; recess also shifts mixing, performance, and stability. → recess length is a coupled injector↔cooling-channel variable.

**Implication:** the tip-thermal screen should become a real constraint that reads
BF, TMR, and the fuel-vs-oxidizer-centered choice (a fuel-centered pintle films
the tip with fuel; an oxidizer-centered tip faces hot oxidizer-rich gas), with
Kang 2022 as the validation anchor.

### 1.5 Combustion-chamber coupling

- **Chamber size from vaporization distance (Son 2017, Eqs. 15–16):** with `x` the spray **vaporization distance** (Fig. 12) and `α` the spray half-angle,
  ```
  D_c = 2·x·sinα + D_pt          (chamber diameter)
  L_c = x·cosα + L_open + r_post  (chamber length)
  ```
  i.e. the chamber is sized so the spray finishes vaporizing inside it (Son's worked baseline: α ≈ 52.6°). The code expresses the diameter side geometrically as the spray-cone wall-impingement distance; it does **not** yet use the vaporization-distance length relation — worth adding so chamber **length** is driven by spray completion, consistent with the existing `L_comb` margin.
- **Contraction ratio / per-element flow ↔ stability (NASA TN, 19680021046):** chamber pressure, flow-per-element, and contraction ratio jointly set acoustic-mode instability — relevant when choosing slot count and BF.
- **L\* / residence time** must exceed the spray combustion length `L_comb` (the code already forms `margin = chamber_length / L_comb`); SP-8089 and SP-8087 give the chamber-sizing ground rules.

### 1.6 Nozzle coupling

The injector is sized from `F, Pc, Pa, ε, O/F → Cf, At, mdot`. Those come from the
repo's existing nozzle/throat solver (Rao/MOC, SP-8120). The only injector-facing
nozzle quantities are **At** (throat area) and **Cf** (thrust coefficient) which
set total **mdot** = (Pc·At)/c\* and thence the per-stream flows. This coupling is
already correct in the code; no change needed beyond making the handoff explicit.

### 1.7 Feed-system / pump coupling

Standard feed-pressure budget (Huzel & Huang SP-125; Sutton & Biblarz):

```
P_pump,discharge = Pc + ΔP_injector + ΔP_cooling_jacket + ΔP_lines/valves + ΔP_dynamic
ΔP_injector      = χ · Pc          (χ = dp_fraction ≈ 0.2)
Pump head:   H = ΔP_pump / (ρ · g)
Pump flow:   Q = mdot / ρ
Pump power:  P = mdot · ΔP_pump / (ρ · η_pump)
Suction margin:  NPSH_available ≥ NPSH_required   (inducer sizing — NASA SP-8052)
```

Turbopump system/component sizing: NASA SP-8107 (systems), SP-8052 (inducers),
SP-8109 (centrifugal), SP-8125 (axial). `feed_system_ledger()` now rolls the
injector drop, declared manifold allowance, applicable regen-jacket drop,
line/valve losses, and control margin into per-stream discharge pressure,
head, flow, ideal power, and NPSH fields. Configuration-specific loss and pump
map validation remain external.

---

## 2. Audit of `raosim/injector.py`

| Quantity (code site) | Basis | Verdict |
|----------------------|-------|---------|
| `mdot = Cd·A·√(2ρΔP)` per stream (`_stream_mass_flux`) | Sutton; SP-8089 | **Correct.** Standard metering. |
| `TMR = ṁ_r v_r / (ṁ_a v_a)` | Hwang Eq.1 / Son Eq.4 | **Correct, exact match.** |
| `BF = N·w/(π·D_p)` | Hwang Eq.3 | **Correct, exact match.** |
| `θ_spray = atan2(radial, axial)` | Heister/Cheng TMR framework | **Defensible leading-order.** Missing Son's tip-angle saturation + We dependence. |
| Fixed-discrete per-stream `Cd` inputs (0.7 default) | mid-range preliminary input | **Explicit but uncalibrated by default.** Cold-flow calibration remains configuration-specific. |
| Son Eq. 1 `A_tip`, fixed `A_cg`, and `A_eff=min(A_tip,A_cg)` (`movable_pintle.py`) | Son 2017 Eqs. 1–2 geometry | **Implemented and benchmarked** at the published 0°, 20°, and 40° transition openings. The open stop is below the center-gap transition. |
| Movable `Cd(L_open/L_max)` interpolation | Son 2017 Eq. 3 measurement definition | **Fail-closed evidence contract.** Requires source/hash, an exact Son-geometry fingerprint, plus matching fluid, Re, ΔP, temperature, and cavitation domains; no universal curve is claimed. |
| `ΔP/Pc = 0.2` default | Huzel & Huang; SP-8089 | **Correct** stability floor; configurable by stream. |
| SMD = Hinze We_crit limit (`spray_atomization`) | Hinze 1955 | **Sound general screen;** not pintle-specific and not applied as movable-sheet truth. |
| Breakup ≈ 15·d_jet (`spray_atomization`) | Reitz & Bracco | **Sound screening midpoint.** |
| d²-law vaporized fraction (`spray_atomization`) | NASA TR R-67 1960 | **Sound screen;** kept separate from mixing, combustion, and total c\* efficiency. |
| Tip thermal: recirc `T_aw=0.8Tc` + Dittus-Boelter series (`face_tip_thermal`) | screening; cf. SP-125 wall solve | **Right idea, validated qualitatively by Kang 2022.** Recirc fractions (0.8, 0.2) unvalidated; BF→T_tip and TMR→q couplings absent. |
| Wall impingement `x = (R_c − r0)/tan θ` | geometry; cf. Son Eq.15 | **Correct geometry.** |
| Feed pressure → pump | Huzel & Huang; NASA SP-810x | **Implemented** — `feed_system_ledger()` builds the per-stream pump-outlet budget + head/flow/power, with `feed_pump_pressure/capacity/npsh` gates. |
| Throttling / movable pintle | Son 2017 control-area geometry | **Implemented, preliminary.** The center rod meters only the radial stream; the axial annulus remains fixed and a separately reported upstream controller closes its flow. |
| Gas/liquid & supercritical | compressible branch present for resolved gas/supercritical states | **Partly implemented.** Needs corpus validation, two-phase exclusion clarity, and real-fluid/supercritical calibration (Jin 2022, Zhou 2023, Hwang 2022). |

**Net:** the continuous movable-pintle hydraulic/kinematic branch is no longer
missing. Its geometry and throttle bookkeeping are implemented, while
hardware-specific `Cd`, position/leakage, actuation loads, and sheet thickness
must be supplied as separate source/hash-bound configuration-controlled
evidence records. Remaining model gaps
include empirical spray-angle saturation, the quantified BF/TMR tip-thermal
couplings, transient actuator/feed dynamics, and physical cold-flow/hot-fire
validation. Gas/supercritical injection remains outside the liquid-sheet
model's applicability.

---

## 3. Required CLI inputs and parameter propagation

### 3.1 What the interview already has (from the nozzle/chamber solve)
`F, Pc, Pa, ε, O/F → Cf, At, mdot`, propellant identities, and chamber radius/length.
These fully determine total and per-stream mass flow — no new inputs needed for sizing the *flow*.

### 3.2 New inputs the pintle stage should collect

**Tier 1 — minimal (defaults from literature if skipped):**

1. **Centered stream**: fuel-centered vs oxidizer-centered (drives tip thermal & which stream is radial). Default fuel-centered for tip protection (Kang 2022).
2. **Target TMR** (default 0.7; Austin/Heister) and **target BF** (default trade between 85% c\* and tip-thermal; see Tier-2).
3. **Injector stiffness** ΔP/Pc per stream (default 0.20; range 0.15–0.25).
4. **Discharge coefficients** per stream for fixed discrete geometry. For the
   movable branch, supply a configuration-specific `Cd(L/L_max)` artifact and
   its fluid/Re/ΔP/temperature/cavitation envelope; a literature-wide numeric
   range is not a substitute for that calibration.

**Tier 2 — geometry / manufacturing:**

5. **Slot count N** and **slot aspect ratio** (already in code) — couples to BF and to the contraction-ratio/stability rule (NASA TN 19680021046).
6. **Pintle tip angle θ_pt** and **post/deflector angle** — set the spray-angle saturation limits (Son 2017) and the geometric cone.
7. **Recess / opening distance** (and, for throttling, post/rod/center-gap
   geometry, closed/open stops, position tolerance, feedback resolution,
   backlash, and leakage bound) — couples to `Cd(L_open)` and chamber heat flux
   (Ahn 2014).
8. **Manufacturing floors** (min feature, min web) — already present.

**Tier 3 — operating envelope (enables the new branches):**

9. **Throttle ratio** (e.g. 4:1) → physical radial opening schedule plus the
   required separate upstream controller schedule for the fixed axial annulus.
   The code does not resize both passages and call the result stroke.
10. **Inlet phase/temperature/pressure per stream** → selects liquid/liquid vs gas/liquid vs supercritical branch (Jin 2022; Zhou 2023; Hwang 2022). Partly present (`PropellantFeedSpec.phase`).

**Tier 4 — feed-system close-out:**

11. **Cooling-jacket ΔP** (from the regen module) and **line/valve loss allowance** → required pump discharge pressure, head, capacity, power, and NPSH check (Huzel & Huang; SP-8107/8052).

### 3.3 Propagation map (holistic coupling)

```
NOZZLE  (At, Cf) ─┐
CHAMBER (Pc,O/F) ─┼─► mdot, mdot_f, mdot_o ─► per-stream A, ΔP, v
PROPELLANTS ──────┘                              │
                                                 ├─► TMR, BF, θ_spray, We, SMD
TIP/THERMAL ◄── BF, TMR, centered-stream ────────┤        │
   (Kang 2022)                                    │        ▼
CHAMBER Ø ◄── spray vaporization distance x ──────┤   L_comb vs L*  (chamber length check)
   (Son Eq.15; Ahn recess→q→cooling channels)     │
FEED/PUMP ◄── ΔP_inj + ΔP_jacket + ΔP_lines ──────┴─► P_discharge, H, Q, NPSH
   (Huzel&Huang; SP-8107/8052)
```

The single most important addition is making **BF and TMR drive the tip-thermal
constraint and the c\*-efficiency estimate simultaneously**, so the CLI can present
the real trade (performance vs tip survival) instead of two independent numbers.

---

## 4. Prioritized recommendations

1. **Close the feed→pump budget** (Tier-4 inputs + §1.7 equations). ✅ **DONE** — implemented as `FeedSystemSpec`/`FeedLineSpec` (inputs) → `feed_system_ledger()` → `FeedSystemLedger` (per-stream required pump-outlet pressure, head, capacity, NPSH) with three new gates, wired through `evaluate_pintle_injector`, `to_dict`/`pintle.json`, and `main.py` CLI flags; 11 unit tests added. Design note: the maldistribution-network manifold loss (~20 bar here) is reported as an informational `manifold_screen_loss` but **not** auto-charged — the charged manifold allowance defaults to 0 so an unvalidated screen can't silently dominate the pump requirement.
2. **Make Cd and the recirculation thermal fractions inputs, not constants.**
   ✅ **Movable Cd DONE as an evidence-gated map** — not as a universal Son
   curve. The Kang-aligned thermal fractions remain screening constants.
3. **Couple tip-thermal to BF/TMR** per Kang 2022 (BF 86→58% ⇒ −17.3% T_tip; TMR↑ ⇒ q↑) and gate on it; tie recess to chamber heat flux per Ahn 2014.
4. **Add the empirical spray-angle saturation** (Son 2017 logistic, tip-angle limits) as an optional validity-flagged mode beside the `arctan(TMR)` default; cross-check against Ninish 2018 / Zhou 2022.
5. **Implement the movable-pintle (throttling) branch.** ✅ **DONE at the
   preliminary hydraulic/kinematic level** — exact Son tip/center-gap control
   area, hard stops, configuration-specific `Cd(L/L_max)`, static force/stem
   ledgers, position/leakage gates, and fixed-hardware throttle scheduling with
   a separate axial controller. A swept moving CAD assembly, transient control
   model, and physical validation remain open.
6. **Validate and harden** the gas/liquid and supercritical branches (Jin 2022; Zhou 2023; Hwang 2022; Cavalieri thesis) after the liquid/liquid path is validated.

**Validation targets** (do not invent data): Son 2015/2017 and Ninish 2018 for
cold-flow spray angle/SMD; Hwang 2022 for TMR/BF vs c\*; Kang 2022 for tip
thermal; Austin 2002/2005 and Sakaki 2015–2018 for hot-fire c\* and stability.

---

*All correlations attributed to source. Hardware-specific fitted coefficients
(Son Eqs. 8–9, etc.) are to be transcribed directly from the cited paper at
implementation time and applied only within their stated validity envelope.
Scanned monographs (SP-8089) require OCR for verbatim equation capture; their
relations here are cross-confirmed through the peer-reviewed papers that cite them.*
