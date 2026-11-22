---
title: Literature Audit of the Differentiable Engine-MDO Layer
subtitle: Every design decision, equation, and constant vs. the propulsion_texts corpus
date: 2026-07-24
scope: raosim/mdo/* (Phases 1–9 + 7b film cooling) and the separation fix
method: text-extracted corpus verification (propulsion_texts_for_agents/markdown), not filename inference
---

# Literature Audit — Differentiable Engine MDO

## 0. Purpose and method

This document re-verifies **every** physical/mathematical choice made in the
`raosim/mdo/` workstream against the local corpus
(`propulsion_texts/propulsion_texts_for_agents/markdown/`), citing file and
line/page. Each item gets one of three verdicts:

- **VERIFIED** — the form and constants match a primary source verbatim (or to
  its stated tolerance).
- **SURROGATE** — a deliberately simplified, C¹, screening model that is
  *consistent with* the literature's mechanism and band but is not a fitted
  correlation. Every surrogate is labelled as such in the code and has a
  documented upgrade path.
- **GAP** — a place where the literature imposes something we do not yet model;
  a recommended fix is given.

The headline: the physics core is **well grounded**; three components are
explicitly **surrogates** (pump-efficiency shape, η_c*(TMR) coupling, film
effectiveness); and there are **four modelling gaps** worth closing, the most
important being a channel aspect-ratio limit.

---

## 1. Nozzle, performance, thermochemistry

| Item | Source (corpus) | Verdict |
|---|---|---|
| Isentropic area–Mach, p/p_c relations | Anderson *Modern Compressible Flow*; SP-125 (`19710019929`) | VERIFIED |
| Vandenkerckhove Γ(γ), c*_ideal = √(RT_c)/Γ | SP-125 | VERIFIED |
| C_F, ambient thrust-coefficient term (p_e−p_a)ε/p_c | SP-125; SP-8120 (`19770009165`) | VERIFIED |
| c* convention c*_del = η_c*·c*_ideal (no double count) | plan invariant #2; `engine.py:130` audit | VERIFIED |
| I_sp = C_F·c*_del/g0 | SP-125 | VERIFIED |
| Frozen/constant-γ property model (Phase-2 CEA surfaces pending) | Gordon & McBride RP-1311 (method ref) | SURROGATE (constant γ/T_c until CEA table) |

The property model is the biggest open *fidelity* item, not a correctness
error — it is the Phase-2 deliverable. Until then γ, T_c, c*_ideal are constants
(`constant_chamber_surfaces`), which is why O/F cannot yet be a meaningful design
variable (§5).

---

## 2. Flow separation (the pre-existing fix)

| Item | Source | Verdict |
|---|---|---|
| Summerfield p_sep ≈ 0.4·p_a | Östlund `fulltext01` Eq. 28 (p.51) | VERIFIED |
| Schilling p_sep/p_a = k₁(p_c/p_a)^k₂ | Östlund Eq. 29 | VERIFIED |
| Kalt–Badal k₁=2/3, k₂=−0.2 | Östlund p.52 (line 2797) | VERIFIED |
| Schmucker (1.88·M−1)^(−0.64), local-Mach march | Östlund Eq. 30 (line 2810) | VERIFIED |
| SP-8120 "within 20 %" design-margin verdict | SP-8120 | VERIFIED |

The 2026-07-22 correction (Schmucker/Kalt–Badal had been cross-labelled) is
confirmed correct against the source equations.

---

## 3. Regenerative cooling

| Item | Source | Verdict |
|---|---|---|
| Bartz h_g coefficient **0.026** | Bartz 1957 (`technical-notes-1957` line 94: "value of C was found to be 0.026") | VERIFIED (verbatim) |
| Bartz σ property factor, ω=0.6 → 0.68/0.12 exponents | Bartz 1957 | VERIFIED |
| Turbulent recovery T_aw, r=Pr^(1/3) | SP-8087; standard BL theory | VERIFIED |
| Series gas/wall/coolant resistance framing | SP-8087 (`19730022965`) | VERIFIED |
| Land/rib **fin efficiency** tanh(mH)/mH | SP-125 §4; `physics.fin_efficiency` | VERIFIED |
| Coolant-area augmentation (w+2η_f·h)/pitch | `physics.py:879` "literature-standard fin (land) area correction" | VERIFIED |
| Sieder–Tate h_c = 0.027·Re^0.8·Pr^(1/3)·(μ_b/μ_w)^0.14 | Sieder & Tate 1936; `physics.py:459` | VERIFIED |
| Darcy Δp (64/Re, Blasius/Swamee–Jain) | White *Fluid Mechanics* §6 | VERIFIED |
| HARCC aspect-ratio raises conductance | Pizzarelli 2011 (`pizzarelli2011` HARCC, lines 165–170); Carlile 1992 | VERIFIED (mechanism) |
| **Counterflow** coolant (enters at nozzle exit) | **Mirzamoghadam 1991 line 137**: "for an area ratio of seven and below, the counterflow coolant arrangement produced lower pressure drop" | VERIFIED (validates our choice for ε≤7) |
| **RP-1 coking limit 728 K** liquid-wall | SP-8087 line 4390 ("do not operate with liquid-wall temperatures above 850°F (728 K) for RP-1"); corroborated by Mirzamoghadam line 145 (copper compatible to 755 K) | VERIFIED |

Cooling is the most thoroughly grounded block. The one new confirmation this
audit adds is that **counterflow is the literature-preferred arrangement for our
expansion-ratio regime** (Mirzamoghadam), which we had chosen on general grounds.

---

## 4. Injector (pintle)

| Item | Source | Verdict |
|---|---|---|
| Incompressible orifice G=C_d√(2ρΔp), v=G/ρ | `injector.py:1725` liquid branch; Sutton | VERIFIED |
| TMR = (ṁ_r v_r)/(ṁ_a v_a), radial=fuel/axial=ox | Freeberg 2019 (line 149); Sakaki 2015; Hwang `s42405`; repo convention | VERIFIED |
| Spray θ = atan2(m_r cosδ, m_a+m_r sinδ); δ=0 ⇒ arctan(TMR) | `injector.py:3201`; Freeberg 2019 line 158 ("θ = tan⁻¹(TMR)") | VERIFIED (leading order; see §6) |
| Blockage factor BF = N·w/(πD_p) | Freeberg 2019; Hwang `s42405` (TMR+BF master knobs) | VERIFIED |
| **Son 2017 two-branch min area** A_tip vs A_cg | `son2017` (Amin near tip, Acg center gap); `movable_pintle.py` | VERIFIED |
| Son Eq.(1) tip area π[2r_f L cosθ − L²sinθcos²θ] | Son 2017 (DOI 10.2514/1.B36301) | VERIFIED |
| Chug screen min(χ_f,χ_o) ≥ 0.2 | SP-8113/SP-194 (`19720026079`); `injector.stability_screen` | VERIFIED |

---

## 5. Pump / electric feed

| Item | Source | Verdict |
|---|---|---|
| Meanline Euler head H=(U₂c_θ2−U₁c_θ1)/g0 with slip | SP-8109 (`19740020848`); `pumps._velocity_triangle` | VERIFIED |
| Specific speed N_s=ω√Q/(g0H)^¾; N_ss (suction) | SP-8109 / SP-8052 (`19710025474`); `pumps.py:2410` | VERIFIED |
| NPSH = (p_in−p_v)/(ρg0)(+V²/2g0) | SP-8052 | VERIFIED (static term; velocity term deferred) |
| Suction screen N_ss ≤ 4 | SP-8052; `pumps.py:2457` | VERIFIED |
| Rocket-pump efficiency band **60–85 %**, rises with capacity/N_s | SP-125 line 21674 ("60 to 85 percent"); SP-8109 low-N_s penalty (line 802/1765) | VERIFIED (band) |
| **C¹ η(N_s) log-Gaussian** shape | — (calibrated to the SP-125 band) | **SURROGATE** (band verified; the peaked *shape* is a smooth screening fit, not SP-125 Fig. 6-23) |
| Battery epigraph m_b ≥ max(energy, power) | Lee 2021 (`s42405-020-00325-z`) | VERIFIED |
| Motor/inverter/pump specific-power masses | Lee 2021 | VERIFIED (screening constants) |

---

## 6. Engine coupling and the η_c*(TMR) edge

| Item | Source | Verdict |
|---|---|---|
| MDF architecture; IFT total derivatives (no unroll) | Martins & Lambe 2013; Martins & Ning 2021; Blondel 2022 | VERIFIED (method) |
| Hydraulic edge Δp_pump = P_c+Δp_inj+Δp_regen+Δp_line−P_tank | SP-8109/SP-125 feed ledger | VERIFIED |
| **η_c*(TMR)** spray→c* coupling (default OFF) | Sakaki 2016/2017 (c*-eff varies with TMR); Hwang `s42405` | **SURROGATE** (trend real, *direction config-dependent*; correlation is a screening knob; frozen by default + ablation — exactly the plan's RQ1 protocol) |
| arctan(TMR) spray angle | Freeberg 2019 line 158–159: Escher found the bare arctan "a poor approximation", proposed θ ∝ TMR^½ | **SURROGATE** (leading-order kinematic; documented ladder to Escher/Son-2015) |

---

## 7. Film cooling (Phase 7b) — the item the user flagged

> **RESOLVED 2026-07-24 — now VERIFIED, not a surrogate.** The user supplied
> **Hatch & Papell, NASA TN D-130 (1959)** (`19890068390`) and **Shine & Nidhi
> (2018)** (`shine2018`), both now extracted into the corpus. The film block was
> re-implemented on the published correlation:
>
> **ε = C·(X/VR)^(−0.8)·Re_c^(0.2)**, X = x/D, VR = v_c/v_g
>
> with published coefficients C = **3.09** (Stollery & El-Ehwany, the
> conservative choice and our default), 3.39 (Hartnett), 4.62 (Tribus & Klein) —
> Shine & Nidhi Eqs. (1)–(5); the coefficient spread encodes the mixing
> assumption ("the more the mixing, the lower the coefficient"). Hatch & Papell
> report **±5 % for ε ∈ [0.2, 1.0]** over velocity ratio 0.45–33.3, and the
> reference scales are calibrated so the model is exercised inside that band.
>
> The **c\* penalty is now calibrated to experiment too**: Coulbert (via Shine
> §4.5) finds the loss is *proportional to coolant flow* (our linear form), and
> Morrell measured 4 % Isp loss at 5 % water film (ratio 0.8), 4 % at 15 % alcohol
> (0.27), 2 % at 15 % ammonia (0.13) — our 0.5 sits inside that measured band.
>
> Two model errors were found and fixed in the process: the reference scales
> initially saturated ε at 1.0 (no gradient), and the hard on/off coverage mask
> produced a **non-physical wall-temperature cliff** at the first unfilmed
> station — removed, since the film's decay should be physical rather than a
> mask.

### 7.1 Correction — the gaseous correlation does not apply to a liquid film

Promoting the film-injector slot to a **design variable** (Hatch & Papell's
`S`, tested 0.063–0.50 in) exposed a deeper modelling error. With the slot as
real geometry, the correlation's velocity ratio follows from mass continuity
(TN D-130's own sizing relation `S = Ẇ_c/(ρ_c V_c π D)`), and it comes out at

    VR ≈ 1e−4 … 3e−3   across the entire tested slot-height range

against a **fitted band of VR = 0.45–33.3**. The reason is physical, not
numerical: Hatch & Papell, Stollery, Hartnett and Tribus all used **gaseous**
coolants (air, helium). An RP-1 film is a *liquid* at ρ ≈ 810 kg/m³, so for the
same mass flow it moves ~800× slower than a gas. Reaching even VR = 0.45 would
need a 0.003 mm slot. **Using the gaseous family for a liquid RP-1 film is
extrapolation by three orders of magnitude, not validation.**

The block therefore now implements the **liquid** model the same review
prescribes (Shine & Nidhi §4.3: Kinney, Graham, Sellers and Emmons all "equated
the convective energy transfer on the surface of the liquid film from the hot gas
stream to the energy utilized for the phase change of the liquid coolant";
Huzel & Huang SP-125 Eq. 4-34 and Shine Eq. 12 for the film-cooled length):

1. per-mass enthalpy absorption `H = c_p,l(T_w − T_co) + ΔH_vap + c_p,v(T_aw − T_w)`;
2. hot-gas load `q = h_g (T_aw − T_w)`;
3. protected wetted area (film-cooled length) `A_FCL = η_fc·ṁ_film·H/q`;
4. a C¹ decay over that length (differentiable stand-in for the cut-off).

`η_fc` — the fraction that cools rather than being entrained — is the single
empirical constant, and is exactly Mirzamoghadam's Aerojet "entrainment
fraction". The gaseous velocity ratio is retained as an explicit **diagnostic**
(`film_slot_validity`) and pinned by a regression test, so the inapplicability is
documented rather than forgotten. Result: T_wc falls smoothly and monotonically
with film fraction; the optimiser converges to ≈8 % film.

### 7.2 η_fc calibrated — the model is now fully derived

The last screening constant (`η_fc = 0.5`) was closed out with two further
sources the user supplied, both now in the corpus:

- **Stechman, Oberstone & Howell (1969)**, *Design Criteria for Film Cooling for
  Small Liquid-Propellant Rocket Engines*, AIAA J. Spacecraft 6(2) —
  10–1000 lbf engines, i.e. our scale. Their Eq. (2) is the liquid-cooled length
  (sensible + latent terms — the same structure implemented here), and they state
  that because of liquid-film instability "the coolant flow was corrected by an
  **efficiency factor that is a function of coolant Reynolds number**",
  Re_L = W_L/(π D μ_ℓ) (their Fig. 2, spanning Re 1000–4000).
- **Grisson (1991)**, *Liquid Film Cooling in Rocket Engines*, AEDC-TR-91-1 —
  supplies the mechanism and the numbers: below **Knuth's critical
  flow-per-circumference** `Γ_cr = 1.01e5·μ_v²/μ_ℓ` the film is smooth; above it
  large waves shear droplets from their crests and "the mass loss rate ... is
  **2 to 4 times the normal evaporation rate**" — i.e. η_fc falls to ≈1/2–1/4.
  Grisson's conclusions also validate this model's structure ("a simple
  one-dimensional model gives satisfactory comparison with existing data for
  liquid film lengths in rocket engines").

η_fc is therefore no longer a fitted constant: it is a **C¹ transition between
the two literature limits** (0.9 smooth → 0.33 wavy) keyed on Γ/Γ_cr. For the
13 kN RP-1 case the film runs at **Γ/Γ_cr ≈ 9 — firmly in the wavy, entraining
regime** — so the model self-selects η_fc ≈ 0.33 rather than the optimistic 0.5
assumed before. The optimum accordingly became more conservative (≈11 % film vs
8 %), which is the physically correct direction.

### 7.3 Dead-variable audit — slot height removed, D_pintle made live

A direct check of the constraint-Jacobian columns (prompted by the question
"does slot height actually influence anything?") found **two design variables
with identically zero columns *and* zero objective gradient** — i.e. the NLP was
optimising in a rank-deficient space (rank 8 of 9 with 10 variables):

| variable | finding | resolution |
|---|---|---|
| `film_slot_height` | ε, T_wc, the residual and every gradient were *bit-identical* across a 25× slot range | **removed from the design vector** — and that is physically right: Stechman *derives* S from continuity, his liquid-cooled-length Eq. (2) contains no S, and Knuth's criterion is written on flow-per-circumference Γ = Ẇ/(πD), also S-independent. S is a sizing/validity quantity for a liquid film (it matters for a *gaseous* film, where it enters Hatch & Papell directly). It stays a `MissionSpec` constant + the `film_slot_validity` diagnostic. |
| `D_pintle` | reached only `blockage_factor` and `tip_opening`, neither of which was a constraint or the objective | **made live by constraining BF** — the blockage factor is, with TMR, one of "the two master geometric knobs" (Hwang et al. IJASS 23, 2022; Freeberg 2019 makes spray angle a function of BF and TMR), and Ryu et al. measured +3.85 % c*-efficiency sweeping BF 70 → 85 %. A two-sided band (0.30–0.90) is now enforced. |

After the fix: **9 variables, 11 constraints, rank 9, zero dead columns.** The
optimiser now moves `D_pintle` (20 → 25.9 mm at the min-mass point) instead of
leaving it wherever the initial guess put it.

*Remaining uncertainty (documented, not hidden):* Woodmansee & Hanratty measured
Γ_cr ≈ 3× lower than Knuth's correlation for water, so the transition **location**
carries ~3× uncertainty; and Shine reports Stechman's model errs −20 % … +13 %
depending on chamber configuration. Those are the honest error bars on the film
block — the structure and both limits are now literature-sourced.

---

## 8. Optimisation method

| Item | Source | Verdict |
|---|---|---|
| ε-constraint (min mass s.t. I_sp≥floor) | Martins & Ning 2021 §; plan §8 | VERIFIED |
| Unit-box variable scaling | Martins & Ning ch.4 | VERIFIED |
| Exact JAX Jacobians → SLSQP; KS for max-type | Griewank; Martins & Hwang 2013; plan §8 | VERIFIED |
| Reverse-mode constraint Jacobian (forward-mode drops a min-margin tangent through nested implicit solves) | engineering finding, FD-verified | VERIFIED (correct + documented) |

---

## 9. Film cooling — the validated correlation (research result)

**Answer: the corpus already has a validated liquid-film-cooling design model —
SP-125 (Huzel & Huang) Equation 4-34, Figure 4-34** (`19710019929` lines
13290–13460), which is itself the Aerojet-class entrainment model that
Mirzamoghadam 1991 (`mirzamoghadam1991`, AIAA-91-1982) generalises.

The validated model has three physically-grounded pieces:

1. **Enthalpy-absorption capacity of the film** (SP-125, text-extracted):
   `H = Cp_lc·(T_wg − T_co) + ΔH_vc + Cp_vc·(T_aw − T_wg)` — the film absorbs
   sensible liquid heating to the wall temperature, latent heat of vaporisation,
   and vapor superheat to the recovery temperature.
2. **A heat balance** giving the required film mass flux per unit wall area:
   `G_c = h_g·(T_aw − T_wg) / (η_fc · H)`, where η_fc is the *film-cooling
   efficiency* — the fraction of coolant that cools rather than being lost to the
   core (SP-125 line 13449).
3. **An entrainment / velocity-ratio effectiveness** for the axial decay: SP-125
   gives an exponential form with coefficients `a = 2V_d/V_m`, `b = (V_g/V_d)−1`
   (velocity ratios of the film, core, and boundary-layer-edge streams);
   Mirzamoghadam defines the same physics as an *entrainment fraction that
   depends on the coolant/freestream velocity ratio* (lines 107–111), calibrated
   at Aerojet, with barrier cooling modelled by reducing that entrainment.

**What this buys us:** replacing `η_film = η_max(1−e^(−film_frac/f_ref))` with the
SP-125 heat-balance + velocity-ratio effectiveness makes film cooling a
*first-principles, literature-grounded* discipline: the c* penalty becomes the
physically-correct "coolant lost to core" term (η_fc), and the effectiveness
follows from the enthalpy balance and the velocity ratio rather than two fitted
constants. It stays C¹ and differentiable (all the pieces are smooth in the film
flow and the velocity ratio).

**What I still need from you to make it fully validated (rather than
corpus-reconstructed):** the SP-125 pages 124–126 OCR is partially garbled
(Fig. 4-34 and the exponential's a/b arrangement), so I can either

- (a) reconstruct the standard published Huzel–Huang Eq. 4-34 (it is a
  well-known result) and implement it now, cross-checking magnitudes against the
  Mirzamoghadam narrative and the subscale test data already in the corpus
  (`leccese2018`, `kang2011`, `perakis2021` — subscale film-cooled rocket
  chamber wall-temperature measurements); **or**
- (b) validate to a *published, coefficiented* correlation if you can provide one
  of: **Hatch & Papell, NASA TN D-130 (1959)** (the canonical gaseous tangential
  film-cooling correlation SP-8087 cites); **Shine & Nidhi, "Review on film
  cooling of liquid rocket engines," Propulsion and Power Research / Prog. Aero.
  Sci. (2018)** (a consolidated set of modern correlations with coefficients);
  or **Grisson, NASA liquid-film-cooling correlation**. None of these three are
  in the corpus.

My recommendation: do **(a) now** (SP-125 Eq. 4-34 is enough to move from
surrogate → grounded model, validated qualitatively against the corpus subscale
data), and treat **(b)** as the calibration refinement if you supply one of those
three papers.

---

## 10. Gaps found — status (re-verified in code 2026-07-24)

| # | Gap | Status |
|---|---|---|
| 1 | Channel aspect-ratio limit | **CLOSED** |
| 2 | `t_wall` structurally constrained | **CLOSED** — as a *thermal-stress* constraint; the hoop-stress fix originally recommended here was wrong (see below) |
| 3 | Design margins (streaking / maldistribution / 2× film) | **CLOSED** — `--design-margins` |
| 4 | Coolant Mach limit | **CLOSED as a diagnostic** (constraint deliberately declined: ~2 orders of margin ⇒ dead column) |

> **All four closed 2026-07-24.** Implementation notes below; closing gap 2 also
> surfaced a *fifth*, previously unnoticed gap — with `t_wall` free, a thicker
> wall lowers T_wc (helping coking) while **raising** T_wg, and nothing bounded
> the gas-side wall temperature. A `wall_temp` constraint (Mirzamoghadam's
> "allowable design gas-side wall temperature", 800 K copper-alloy class) was
> added, so the optimiser cannot trade the liner away. The design vector is now
> **10 variables / 12 constraints, rank 10, zero dead columns**.

**1 — Channel aspect ratio: CLOSED.** `MissionSpec.channel_aspect_ratio_max = 8.0`
with an `aspect_ratio_margin` constraint in `engine.py`, enforced in the NLP
(`CONSTRAINT_NAMES`). It was *active* at the optimum (AR = 8.00) before the film
model was recalibrated, and sits interior (AR ≈ 5.9) now.

**2 — `t_wall`: OPEN, and the recommendation above was incorrect.** `t_wall`
remains a `MissionSpec` constant, not a design variable. But the proposed fix —
a hoop-stress constraint `σ = p·r/t` — is the wrong criterion for a channel-wall
liner. Mirzamoghadam is explicit that the wall is sized on the **pressure
differential across it**, and for channel-wall construction the liner spans the
channel width as a plate, not the chamber radius as a shell. Evaluated properly
at the current optimum:

- coolant-to-chamber Δp = 0.34 MPa, channel w = 0.85 mm, t = 0.80 mm
- liner plate bending `σ = Δp·w²/(2t²)` = **0.2 MPa** — utterly negligible
- wall ΔT ≈ 19 K ⇒ **thermal stress ≈ 27 MPa** — two orders larger

So the pressure criterion would never bind; the genuine `t_wall` driver is the
**thermal gradient and low-cycle fatigue**, which the plan already routes to the
§8 constraint layer (Coffin–Manson, NASA CR-134627, Porowski 1985). Correct fix:
add `t_wall` as a variable against an **LCF/thermal-stress** constraint, not a
hoop-stress one. Priority accordingly lower than first stated.

**3 — Design margins: OPEN, now highest priority.** Nothing in `raosim/mdo/`
implements them. This is the one gap written in SP-8087 *shall* language
("film-cooling flow **shall** be capable of providing twice the estimated
required quantity", line 6082) plus Mirzamoghadam's hot-channel practice (+10 %
heat flux for injector streaking, −10 % channel flow for maldistribution, lines
168–170). It is also the cheapest to implement — three documented multipliers
defaulting to 1.0 — and it is what would make the reported optimum defensible as
a *design point* rather than a nominal-conditions answer.

**4 — Coolant Mach: OPEN but confirmed non-binding.** Measured at the optimum:
v_cool = 2.6 m/s against a liquid-RP-1 sound speed ≈ 1300 m/s ⇒ **M_cool =
0.002, i.e. 173× below** Mirzamoghadam's 0.35 limit (line 145). A constraint
would be dead weight (another zero-ish Jacobian column); a cheap assertion or a
reported diagnostic is the proportionate response.

---

## 11. Net assessment

The differentiable MDO layer is **physically sound and, where it matters,
literature-verified**: Bartz 0.026 verbatim, the four separation criteria against
Östlund, the Son two-branch pintle area, the SP-8109 pump meanline, the SP-8087
728 K coking limit (corroborated by Mirzamoghadam's 755 K), and — newly confirmed
here — counterflow cooling as the literature-preferred arrangement for ε ≤ 7.

Two components remain honest, clearly-labelled **surrogates** (the pump-efficiency
*shape* and the η_c*(TMR) coupling, the latter default-off with an ablation). The
film block is no longer among them: it is now a liquid phase-change model with
both limits literature-sourced (§7.2), and the O/F lever is wired to CEA surfaces
(§1).

**Updated near-term priority** (superseding the original ordering, which is now
partly done):

1. **Design margins** (§10.3) — the only remaining gap in SP-8087 *shall*
   language, and the cheapest to add.
2. **`t_wall` against an LCF/thermal-stress constraint** (§10.2) — noting the
   pressure criterion originally proposed does not bind.
3. **Host CEA sampling run** — the surfaces are wired and gated, but a real
   `scripts/sample_cea_surface.py` table is still needed to replace the constant
   fallback.

The coolant-Mach guard (§10.4) is measured at 173× margin and is not worth a
constraint.
