# Differentiable Engine MDO Plan — Verification and Feasibility Evaluation

**Date:** 2026-07-22
**Subject:** `docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md` (v0.1, mirrored in `latex-report/Differentiable_Engine_MDO_Plan.pdf`)
**Method:** Every load-bearing claim was checked against (a) the AI-readable corpus in `propulsion_texts/propulsion_texts_for_agents/markdown/` (224 files, page-traceable via `<!-- PDF page N -->` markers) and (b) the actual code at commit `f2a73ba` (+ working tree). Corpus quotes below carry file + PDF page; code claims carry file + line. Per the corpus's own `AGENTS.md`, the original PDFs remain authoritative where the Markdown mirror flags `needs_review`; items resting on prior PDF-level verification are marked as such.

**Verdict up front.** The current repository is in line with the literature it claims to implement, with **one substantive deviation found** (flow-separation criteria, §A.2.1) and a handful of minor gaps. The proposal itself is technically sound, unusually honest about its weak points, and accurately describes the codebase it plans to modify — all 23 claimed entry points exist, including the exact line number `bvp.py:69`. The plan is feasible in the staged form it prescribes and is worth doing, subject to the amendments in §F.

---

## A. Verification of the current repo against the corpus

### A.1 Verified clean

| Model in repo | Code location | Corpus anchor | Result |
|---|---|---|---|
| Area–Mach, isentropic ratios, Prandtl–Meyer, Mach angle, M* | `raosim/jax/primitives.py:31–91`, `raosim/gas_dynamics.py` | Anderson, *Modern Compressible Flow* (`5f36b7c4….md`); `prmeyer.md`; Rao 1999 Eq. 3 for M* | Match. Standard closed forms, term-for-term. |
| Thrust coefficient C_F (momentum + pressure term) | `raosim/jax/primitives.py:150` | SP-125 / Huzel & Huang (`19710019929.md`); Anderson | Match — identical to the plan's §6.1 expression. |
| c* = √(γRT_c)/(γ·(2/(γ+1))^((γ+1)/(2(γ−1)))) | `raosim/gas_dynamics.py:256` | SP-125 | Match — algebraically identical to √(RT_c)/Γ(γ) with the Vandenkerckhove Γ in the plan. |
| **c\* convention (plan invariant #2)** | `raosim/engine.py:59,130`: `c_star_effective = c_star·eta_cstar`; `m_dot = Pc·At/c_star_eff` | — | Already pinned exactly as the plan demands; `design.py:627` explicitly refuses to double-count a screen into the mdot closure. |
| MOC characteristic compatibility (C+: d(θ−ν)=−S ds; C−: d(θ+ν)=+S ds; S = sinθ·sinμ/r) | `raosim/jax/residuals.py:61–105`, mirrored in `rao_residuals.py`, `moc.py`, `nasa_moc.py` | Anderson §11.4 (axisymmetric section at `5f36b7c4….md` line 15795); Zucrow & Hoffman V2 | Match. Note: the mirror preserves Anderson's *prose* but lost the displayed axisymmetric equations (extraction limitation) — the sign convention was oracle-validated in June 2026 against the NASA `MOC_GridCalc_BDE` output (corrected C− on kernel RRCs RMS 2.3e-6; corrected C+ on the derived LRC RMS 8.8e-8), which is stronger evidence than a text match. |
| Rao topology (B–D–E, control surface on a left characteristic, λ₂/λ₃ multipliers, mass + length constraints) | `raosim/rao_variational.py`, `raosim/moc_topology.py`, `raosim/jax/assembly.py` | Rao 1958 (`rao1958.md` p.2–3): multipliers λ₂, λ₃ (nomenclature + Eq. context, lines 45, 194); compatibility "implicit in the solution of Equations [12,13]" (p.3, line 234) | Match. |
| Existence scan closures (smooth / position / PM-fan at D) | `raosim/rao_existence_scan.py` (module docstring, three closures) | Rao 1999 (`rao1999.md` p.3): valid-region boundary where the boundary function → 0 and dR along DE ceases to be positive; invalid region reached by *designed discontinuity* (DEF nozzles) | Match in structure — the scan's smooth-vs-fan distinction mirrors Rao 1999's smooth-vs-D′-discontinuity construction. |
| Bartz h_g (0.026 coefficient, exponents 0.2/0.6/0.8/0.1/0.9, σ with ω=0.6) | `raosim/jax/thermal.py:36–55`, `raosim/physics.py:147–220` | Bartz 1957 (`technical-notes-1957.md` p.2): "The resulting value of C was found to be 0.026" | Match, verbatim constant. (SI form Pc/c*; the plan's p_c·g₀/c* is the imperial-units form — same equation.) |
| Recovery temperature, r = Pr^(1/3) turbulent | `raosim/jax/thermal.py:58` | Standard turbulent recovery; consistent with SP-8087's Bartz-based methodology (`19730022965.md` line 1673 cites Bartz as ref. 20) | Match. |
| Sieder–Tate coolant side, Nu = 0.027 Re^0.8 Pr^(1/3) (μ/μ_w)^0.14, + Shah–London rectangular laminar branch, Re<2300 switch | `raosim/jax/thermal.py:69`, `raosim/physics.py:452–491` | Classic Sieder–Tate constants; laminar-branch provenance in `docs/thermofluid_literature_provenance.md` | Match. |
| Fin (land) efficiency η = tanh(mH)/(mH), m = √(2h_c/(k·t)) | `raosim/physics.py:491–507` | Standard fin theory; the mechanism Pizzarelli resolves in full | Match with the plan's §6.2. |
| Pizzarelli 2011 claims used by the plan | — | `pizzarelli2011.md` p.1,3: "one-dimensional governing equations for coolant mass conservation and momentum balance, and … two-dimensional governing equation for coolant [energy]"; "aspect ratio as high as 8 in the throat" | Plan's characterization is verbatim-accurate. |
| Carlile 1992 claims | — | `carlile1992.md` p.1,3: same-Δp hot-wall temperature ~50% lower for high-AR; AR 8 conventional / 15 platelet manufacturable | Plan's characterization accurate. |
| Orifice equation, TMR = ṁ_r·U_r/(ṁ_a·U_a), BF = N·w/(πD_p), TMR solved by bisection on radial dP/Pc | `raosim/injector.py` (TMR at ~:2708, active bisection :2748) | Son 2017 Eq. 4 (`pintle_injector/son2017.md`); Hwang Eq. 1/3 per `docs/PINTLE_DESIGN_EVALUATION.md` | Match (previously verified at PDF level; unchanged). |
| Movable-pintle minimum area: Son Eq. (1) tip area, center-gap area, explicit `transition_opening` | `raosim/movable_pintle.py:267–380` | Son 2017 (`son2017.md` p.2): "the transition of the minimum area inside [the pintle nozzle]… changed to the center gap area … with increasing pintle tip angle" | Match — the code already carries the **two-branch structure with an explicit transition**, exactly what plan §6.3 requires (no `min()`). |
| Chug screen: min(χ_f,χ_o) ≥ 0.2 pass / 0.1–0.2 marginal / <0.1 chug-prone; default dp fractions 0.2 | `raosim/injector.py:234–235, 2058–2078` | SP-125 (`19710019929.md` p.137): "rule-of-thumb design value for injector pressure drop varies from 15 to 20 percent of the chamber-nozzle stagnation pressure"; SP-194 (`19720026079.md`) chug phenomenology | Match — the plan's Δp/P_c ≥ 0.2 is the conservative end of the classical 15–20% band. |
| Chamber acoustic modes (L: a/2L; T1: 1.8412·a/πD; R1: 3.8317·a/πD) | `raosim/injector.py:2061–2074` | SP-194 Bessel roots | Match. |
| Feed pressure ledger (P_c + injector ΔP + manifold allowance + regen-if-coolant + line/valve + margin; NPSH; capacity) | `raosim/injector.py:636+` (`FeedLineLedger`) | SP-125; SP-8109 | Match — superset of plan §6.4's ledger. Manifold *screen* loss is reported info-only, not auto-charged (deliberate, documented decision — keep it that way in `mdo/`). |
| Pump meanline (Euler head, ψ = gH/U₂², N_s, N_ss, NPSH, Stodola slip) | `raosim/pumps.py` | SP-8109 (`fuel_pump_design/19740020848.md` p.19): "Current flight-proven centrifugal flow pumps range from 450 to 2100 in specific speed" | Match (full re-derivation done in the 2026-07-02 audit; unchanged). |
| Inducer screens (hub/tip, solidity ~2.0–2.5, suction specific speed) | `raosim/pumps.py` | SP-8052 (`fuel_pump_design/19710025474.md`): solidity 2.0–2.50, S_s optimization | Match. |
| Battery sizing: mass = max(E/(η·ρ_E), P/ρ_P)·margin, separate `mass_energy_limited` / `mass_power_limited` fields, limiting-term gate | `raosim/pumps.py:983–1019, 3202–3208` | Lee 2021 (`fuel_pump_design/s42405-020-00325-z.md` p.2): power density and energy density as separate sizing drivers | Match — the plan's epigraph reformulation maps 1:1 onto fields that already exist. |
| Strain-life screen (Basquin + Coffin–Manson, bisection in log 2N_f), SP-125 buckling, hoop stress | `raosim/physics.py:1994–2216` | CR-134627 (`materials_science/19740017910.md`) NARloy-Z data; Porowski 1985 (`porowski1985.md`); SP-8087 §life (`19730022965.md` lines 5094, 5172: design life = 4× service cycles) | Match in form; honestly labeled a screen. Note SP-8087's factor-of-4 LCF philosophy should be carried explicitly as the MDO life-constraint margin. |
| Schmucker/Kalt-Badal/Summerfield separation criteria | `raosim/separation.py` | Östlund 2002 thesis (`fulltext01.md` p.52) | **Deviation — see §A.2.1.** |

### A.2 Deviations and gaps found

**A.2.1 Flow-separation criteria are cross-labeled and quantitatively off (the one substantive finding). — FIXED 2026-07-22, same day.**
The corpus (Östlund 2002, `fulltext01.md` PDF p.52) gives:

- **Schilling** (Eq. 29): **p_i/p_a = k₁·(p_c/p_a)^k₂**, with k₁ = 0.582, k₂ = −0.195 (contoured) / 0.541, −0.136 (conical); **Kalt–Badal** (1965) refit Schilling's form with **k₁ = 2/3, k₂ = −0.2**, i.e. p_sep/p_a = (2/3)·(p_c/p_a)^(−0.2) (equivalently p_sep/p_c = (2/3)·(p_a/p_c)^1.2);
- **Schmucker** (Eq. 30): **p_i/p_a = (1.88·M_i − 1)^(−0.64)**, "still widely used"; NASA's 1976 recommendation adds a 20% margin (consistent with SP-8120, `19770009165.md` line 4062).

The repo implemented (`raosim/separation.py`, pre-fix): "Kalt-Badal: p_sep/p_a = 1/(1.88·M − 1)" — Schmucker's functional form with exponent −1 instead of −0.64 — and "Schmucker: p_sep/P_c = (P_a/P_c)^0.8/M" — a pressure-ratio form matching neither namesake. At M_e = 3, P_a/P_c = 0.034: repo-"Schmucker" p_sep/P_c ≈ 0.0223 vs true Schmucker ≈ 0.0127 (**1.75×** its namesake) vs true Kalt–Badal ≈ 0.0115. `jax/thermal.py:121` propagated the same form into the differentiable screen.
**Why it matters for the MDO:** separation is a constraint *at every operating point* (plan §8, Phase 10), and gradient optimizers drive designs onto active constraints.
**Resolution (landed 2026-07-22):** `separation.py` rewritten with Schmucker per Eq. 30 (local-Mach march retained, correctly applying only to the local-Mach criterion), Kalt–Badal per the Schilling k₁=2/3/k₂=−0.2 form (signature now takes P_c/P_a), Schilling itself added as a fourth method, Summerfield 0.4·p_a retained, and the SP-8120 "within 20%" rule reported as `design_margin_required/ok` (default 1.2) without changing the physical separated/attached semantics. `jax/thermal.py:schmucker_separation_margin` corrected in lock-step; `jax/design_opt.py` `sep_margin_min` default 1.0 → 1.2. Tests re-baselined and extended (`test_v2_features.py` pins all four criteria against the Östlund forms + cross-family magnitude clustering + vacuum behavior; `test_jax_thermal_design_opt.py` pins NumPy↔JAX parity and re-baselines the "tight" threshold 2.0 → 4.0 since corrected margins roughly double). Affected suites green (v2 20, design-opt 11, frozen-flow 43, design_v2+injector 140/1 pre-existing cadquery-env fail). **Host follow-up:** the 13 kN baseline snapshot's separation-margin fields will drift on the next full run — expected, re-baseline the snapshot.

**A.2.2 Equilibrium chemistry is honestly absent.** `cea.py:68` raises `NotImplementedError` for `cea_equilibrium` because a single chamber γ cannot represent shifting-composition expansion. This is correct engineering honesty, but it means **RQ4 has an unstated dependency** (§D.3).

**A.2.3 Mass ledger is scattered, not consolidated.** Component masses exist (liner/jacket per station in `run_nozzle.py:3871`, pump `mass_estimate_kg`, motor/inverter/battery in `pumps.py`), but there is no single §3-style package rollup with explicit zero-placeholders for excluded items. Phase 0's "mass ledger — DONE" is therefore ~90% true; `mdo/mass.py` (already planned) closes it. Injector/pintle hardware mass should be added while doing so.

**A.2.4 Corpus-mirror caveats.** The mirror is excellent for retrieval but 213/224 files are `needs_review`; displayed equations are lost in exactly the places that matter most (Anderson's axisymmetric compatibility relations; parts of SP-8089, which is an OCR'd scan with garbled table columns). The repo's practice of validating against numerical oracles (NASA MOC output, CoolProp, RocketCEA) rather than extracted text is the right defense; keep it.

---

## B. Audit of the proposal's own claims

**B.1 Source identities (plan §2.2).** All verified against the mirror: SP-8087 = `19730022965` ("Liquid Rocket Engine Fluid-Cooled Combustion Chambers"), SP-8109 = `19740020848`, SP-8052 = `19710025474`, SP-194 = `19720026079`, SP-8120 = `19770009165`, Bartz = `technical-notes-1957`, Rao 1958/1999, Pizzarelli 2011, Carlile 1992, Son 2017/2015, Lee 2021, CR-134627, Porowski 1985 — all present. One update: the plan says SP-8089 has "no text layer"; the agents mirror now carries an OCR conversion (flagged `needs_review`, inconsistent tables) — usable for locating content, not for extracting correlations without PDF checks. The plan's Rao 1958 note ("text begins p.2") matches the mirror.

**B.2 The Rao 1958 → Rao 1999 correction (plan §2.1) is right, and verbatim-supported.** Rao 1958 p.3: the compatibility condition "is *implicit in the solution* of Equations [12, 13]"; nothing in Rao 1958 requires re-solving the contour inside an engine MDO. Rao 1999 defines the valid/invalid regions ("the region of valid computation of the control surface where a shock-free solution is attainable… Below this minimum length… the invalid region", p.1–3) and extends the boundary to equilibrium and frozen chemistry — the correct citation for the cliff and the direct prior art for RQ4. Two sharpenings the plan should absorb:

1. **The boundary mechanism in Rao 1999 is a positivity condition, not literally a caustic:** the "boundary function" (Eq. 6) reaching zero means the radial increment dR along DE ceases to be positive and construction fails. The plan's §7 "caustic/envelope" language should either cite this operational criterion or defend the envelope interpretation separately.
2. **Rao 1999 already answers part of RQ3:** its Fig. 7 result — the upper limit of vacuum C_F vs length "appears to coincide with the boundary of the valid region," i.e. **length-constrained vacuum optima sit on the cliff** — and its DEF construction shows the invalid region is *usable* via a designed discontinuity, with quantified thrust loss. RQ3's novelty is therefore not "does the optimum approach the boundary" (known: yes, for length-limited vacuum nozzles) but "do *engine-level* couplings (mass, cooling, separation, battery) mask or expose it" — which is a genuine, sharper question. Frame it that way.

**B.3 Phase → code map (plan §12.2).** Every cited symbol exists at the cited location: `design_nozzle_v2` (design.py:483), `make_differentiable_solution` (bvp.py:69 — line number exact), `solve_rao_bvp_jax` (api.py:135), `rao_sensitivities`, `constrained_nozzle_design` (design_opt.py:45), `size_cooling_channels` (thermal_design.py:106), `joint_wall_channel_design` (:478), `size_electric_pumps` (pumps.py:3431), `_estimate_pump_efficiency` (:814), `cea_propellant`/`resolve_thermochemistry`, `couple_atomization_to_performance`, `FeedSystemLedger`/`StabilityScreen`/`ManifoldDistribution`, `bartz_hg`/`recovery_temperature`/`sieder_tate_hc`/`throat_wall_temperature`, `area_mach_relation`/`thrust_coefficient`, `least_squares_solve`, `cf_de_jax`, `rao_existence_scan.py`, `separation.py`/`altitude_performance.py`/`trajectory.py`, `spray_coupling.py`. `raosim/mdo/` correctly does not exist yet. Status column is accurate: Phase 0 effectively done (13 kN LOX/RP-1 baseline closes; see A.2.3), Phase 3 correctly WIP (J6 v1 sensitivities landed; J3b open), everything else TODO.

**B.4 Claims verified true about the repo's weak spots.** The plan's two most important self-criticisms check out in code: (i) `_estimate_pump_efficiency` is a step-binned lookup (Q thresholds 2e-5/1e-4/5e-4/2e-3 m³/s → η 0.30/0.40/0.52/0.62/0.70, hard head penalties at 2500/6000 m) — genuinely C⁰-discontinuous and unusable in the differentiable core, as claimed. (ii) The η_c* loop is a one-way correlation fixed point: `handoff.py` marks every handoff cycle-ineligible by evidence and `spray_coupling.py` consumes only the correlation screen — the plan's "strongest edge, weakest physics" warning and mandatory ablation are exactly right.

**B.5 Minor doc nits.** §6.2 cites "Colebrook/Churchill" for friction; the code implements 64/Re + Swamee–Jain (rough) / Blasius (smooth) — which is *better* for the differentiable core (explicit, smooth) — update the text, not the code. §6.4's oxidizer-line note ("omits Δp_regen unless explicitly cooled") matches the implemented ledger. The battery epigraph inequality set matches Lee 2021's two-driver structure verbatim.

---

## C. How each component actually gets modified (concrete map)

The plan's own build order is correct; this section states, per component, what exists today (verified), what changes, and the effort class (S ≤ ~1 session; M = a few; L = sustained).

**C.1 Nozzle (`mdo/nozzle.py`).** Exists: analytic gas dynamics fully differentiable (`jax/primitives.py`); TOP/Bézier chart contour; the full Rao BVP with JAX backend as default (J4 gate passed at 7.5e-4, full-pin continuity); `rao_sensitivities` (J6 v1) giving dCf/d{node states, p_a, γ}. Changes: (a) wrap the analytic path as residuals R(y,x) — S; (b) implicit path calls `solve_rao_bvp_jax` host-only, with the frozen-kernel caveat: **wall-coordinate design totals wait on J3b** (differentiable kernel/BDE march) — correctly kept off the skeleton's critical path — L when it comes; (c) fix separation criteria first (A.2.1) — S. Rule 1 (no silent Bézier fallback) needs a hard `infeasible` marker in the adapter — S.

**C.2 Cooling (`mdo/cooling.py`).** Exists: Bartz/recovery/Sieder–Tate/fin already in JAX at the throat station (`jax/thermal.py`); the full station-marching NumPy analysis (`physics.py:regenerative_cooling_analysis`) with Darcy Δp, curvature correction, structural/LCF screens as oracle; **`regenerative_cooling_2d` already exists in NumPy** — a ready-made oracle for 4b that the plan under-credits. Changes: 4a = port the march to a fixed-topology `lax.scan` over stations (shapes static, as the plan specifies) with stationwise T_wg/T_wc/Δp closures — M. 4b = differentiable per-station 2-D cross-section solve, staged after the skeleton — L. Parity oracle: existing NumPy profiles; gate vs Pizzarelli HARCC trends as planned.

**C.3 Injector/pintle (`mdo/injector.py`).** Exists: all algebra (orifice, TMR, BF, spray angle, SMD correlations); the TMR bisection (`injector.py:2748`); `movable_pintle.py` with Son Eq. (1), center-gap area, `transition_opening`, Cd-curve contract. Changes: (a) port algebra to jnp — S (trivially smooth); (b) replace TMR bisection with a `custom_root` inner solve or closed-form elimination — S; (c) movable-pintle two-branch: reuse `son_minimum_tip_area` and `center_gap_area` as the two smooth subproblems with the consistency inequality, `transition_opening` as the a-posteriori check — S/M, the physics structure already matches the plan's rule 8; (d) keep the manifold screen info-only (A.1). Stability stand-in Δp/P_c ≥ 0.2 is already the code default.

**C.4 Pumps/electric (`mdo/pump.py`).** Exists: meanline (Euler/ψ/Stodola/N_s/N_ss/NPSH), electric drive sizing (motor/inverter power densities, bus selection), battery two-branch masses. Changes: (a) **replace the binned efficiency with a C¹ model — the one genuinely new modeling artifact in the skeleton** — M, and it needs a declared data source: fit η(N_s, D_s or φ) to SP-8109's flight-proven envelope (Ns 450–2100) + the corpus's electric-pump papers (Lee 2021, Kwak 2018 `1-s2.0-S1270963817320953-main.md`, reis2019); guard against branch-hopping with the plan's operating-point inequality; (b) battery epigraph variable + two inequalities — S (fields already separated); (c) bus voltage stays enumerated (discrete rule 4) — the current `_standard_bus_at_or_above` ladder moves to the outer loop — S.

**C.5 Properties (`mdo/properties.py`).** Exists: RocketCEA frozen chamber properties (`cea.py`), CoolProp coolant states, `frozen_flow.py` thermally-perfect expansion (deliberately not fed to MOC). Changes: offline tabulation over (P_c, O/F) and (T, p) + C¹ shape-preserving fit + domain box constraints — M; **equilibrium sampling for RQ4 is available from RocketCEA but the constant-γ solver cannot consume it** (A.2.2) — the Pareto half of RQ4 works through property surfaces; the *existence-boundary* half needs variable-property characteristics (§D.3).

**C.6 Assembly/solve/derivatives/NLP (`mdo/assembly.py, solve.py, derivatives.py, nlp.py`).** Exists: the residual-stacking pattern (`jax/assembly.py`), Optimistix LM + IFT (`bvp.py`), penalty-BFGS seed (`design_opt.py`, admitted for initialization only, matching §8's rule). New: monolithic damped Newton with block-sparse direct linear step, homotopy budget for the state solve, forward-mode constraint Jacobian, SciPy `trust-constr` handoff — M/L. One technical note: `make_differentiable_solution` differentiates the *least-squares* solution; that equals the IFT on R=0 only at (near-)zero-residual roots. For the coupled engine state use a **square root-find** (`optx.root_find` / Newton with `custom_root` semantics) so the implicit derivative is of R=0 itself, and keep the 100× inner-tolerance rule (already the plan's rule 6).

**C.7 Existence/continuation (`mdo/continuation.py`, Paper 2).** Exists: `rao_existence_scan.py` with the three D-closures, `theta_b_solve`, the fold diagnosis machinery from the June J-series (the 12.4 mass-check fix, fixed-end existence roots). New: pseudo-arclength continuation + the three diagnostics (σ_min(R_u), the Rao 1999 boundary function as the *physical* diagnostic — use Eq. 6's positivity form, §B.2 — and λ_min(H_red) with the mesh-doubling gate) — M, NumPy, sandbox-runnable, independent of the MDO spine. This is well-de-risked.

---

## D. Feasibility assessment

**D.1 Why this is more feasible here than it would be almost anywhere else.** The three genuinely hard prerequisites of differentiable engine MDO are already retired in this repo: (i) a differentiable implicit-solve seam exists and passed its gate (J4 at 7.5e-4 with full D-state continuity; 790/0 suite); (ii) the parity-oracle discipline the plan's rule 6 demands is not aspirational — it is how the repo already works (march-parity vs NASA output at 1e-6–1e-10; JAX-vs-NumPy parities at 1e-10–1e-15); (iii) the sequential baseline (Phase 0) closes end-to-end at 13 kN with feed, thermal, structural, and electric gates. The plan is largely "port audited NumPy into residual blocks," and that reading of the codebase is accurate.

**D.2 The honest long poles (all correctly staged by the plan).**

1. **J3b** (differentiable kernel/BDE march): gates only wall-coordinate sensitivities on the implicit path. The skeleton, Paper-1 Pareto, and Paper-2 continuation do not wait on it. Biggest single engineering item; schedule it after the skeleton closes.
2. **4b quasi-2-D cooling**: expensive per-station solve; 4a carries the aspect-ratio trend through fin efficiency meanwhile (Pizzarelli-consistent to first order). The NumPy `regenerative_cooling_2d` oracle shortens this.
3. **C¹ pump efficiency**: needs a real fit to a declared envelope, not just smoothing the bins (§C.4).
4. **Host/sandbox split**: JAX BVP solves are host-only (5–15 min); CI must gate on parity + analytic-path tests only. Already the working pattern.
5. **Multipoint (Phase 10)** multiplies state size by the number of operating points and stresses the separation screen + off-BEP pump surface simultaneously — fine for direct factorization at this scale, but do it last, as planned.

**D.3 Dependencies the plan under-states.**

1. **Separation-criterion fix before any separation-active optimization** (A.2.1). Small; do it in Phase 1's window.
2. **RQ4 is two different studies.** (a) Frozen-vs-equilibrium *Pareto migration* through property surfaces — cheap, well-supported (RocketCEA equilibrium sampling offline). (b) Equilibrium *existence-boundary migration* — requires variable-property/equilibrium characteristic relations in the kernel/DE system (Rao 1999 derived the equilibrium form of its Eq. 1/6 to do exactly this). The current MOC is calorically perfect throughout, and `frozen_flow.py` is deliberately kept out of the invariants. Scope (b) explicitly or defer it; do not let it hide inside "property surfaces" (§9's table implies the surfaces suffice — for the boundary study they do not).
3. **Phase 0 mass ledger consolidation** (A.2.3) — fold into `mdo/mass.py`, add injector hardware mass.
4. **SP-8087's ×4 LCF factor** as the explicit life-constraint margin (A.1, last row).

**D.4 Formulation soundness (method literature).** MDF with converged states per iterate, IFT/adjoint duality, and the forward-vs-adjoint sizing argument (n_x ≈ 12–20, hundreds of constraints ⇒ forward/direct; adjoint only for scalar objectives) are textbook-correct (Martins & Lambe 2013; Martins & Hwang 2013; Gray 2019 — method sources, outside the corpus). KS aggregation with verification, epigraph battery, two-branch pintle, discrete outer enumeration, ε-constraint Pareto, Keller pseudo-arclength continuation, and the mesh-convergence gate on H_red are all standard, defensible choices. The ten implementer invariants in §0.1 are the strongest part of the document — they encode exactly the failure modes (silent fallback, double-counted η, unrolled solvers, fake continuous integers, hidden `clip()`) that produce plausible-but-wrong differentiable models.

---

## E. Is it worth doing?

**Yes — with the RQ3/RQ4 framing sharpened.** By research question:

- **RQ1 (value of coupling)** — real and publishable for the electric-pump class: Lee 2021 sizes the e-pump feed system *sequentially*; nobody in the corpus closes injector–cooling–pump–battery–nozzle simultaneously with exact coupled derivatives. The result is only as strong as the η_c* correlation, and the plan already mandates the fixed-η ablation and bounded claims. The repo's own honesty artifacts (cycle-ineligible handoff) make this credible.
- **RQ2 (exact gradients vs FD/derivative-free)** — solid methods contribution; the honest framing (exactness + KKT quality, not dimensionality) preempts the standard review objection at this n_x.
- **RQ3 (optima vs the cliff)** — novel *at the engine level only* (§B.2): Rao 1999 already shows length-limited vacuum optima sit on the boundary. The sharpened question — which constraint (separation, thermal, mass, battery) becomes active first, and whether the cliff is ever the binding one for a real engine — is worth answering and requires exactly the coupled model being built. The three-diagnostic separation (solvability vs validity vs optimality) is a genuinely good methodological contribution; the June existence-scan work (fold behavior, fixed-end roots) is direct groundwork.
- **RQ4 (real-gas migration)** — the Pareto half is cheap and interesting; the boundary half is a major extension (D.3.2) with Rao 1999 as the template. Stage it; a frozen-vs-equilibrium *Pareto* paper section does not need the variable-property MOC.

**Cost check.** The walking skeleton (phases 1, 2, 3-analytic, 4a, 5, 6-with-C¹-efficiency, 7, 8, 9) is bounded, mostly ports of audited code behind stable interfaces, and produces the first end-to-end gradient and one Pareto point — after which every deepening (4b, J3b, implicit contour, multipoint, equilibrium) is independently schedulable. The two-track structure (Paper 2 mostly NumPy/sandbox) de-risks in parallel. This is the correct shape for the effort, and the repo is unusually well-positioned to execute it.

---

## F. Recommended amendments (actionable, ordered)

1. **[DONE 2026-07-22] Fix `separation.py` + `jax/thermal.py` criteria** per Östlund Eq. 30 / Kalt–Badal k₁=2/3, k₂=−0.2 (A.2.1); SP-8120's 20% margin added as `design_margin` reporting + the design-opt constraint default; affected tests re-baselined and extended. Remaining: re-baseline the 13 kN host snapshot's separation fields.
2. In plan §7, state the Rao 1999 boundary operationally (boundary function/dR positivity, Eq. 6) alongside the caustic language; cite Fig. 7's boundary-coincides-with-max-performance result in RQ3's motivation (B.2).
3. Split RQ4 into 4-P (Pareto, property surfaces — in scope) and 4-B (boundary, variable-property characteristics — explicitly scoped or deferred) (D.3.2).
4. In §12.2 row 4b, credit `physics.py:regenerative_cooling_2d` as the existing NumPy oracle.
5. Specify the pump-efficiency data source and envelope for the C¹ fit (SP-8109 Ns 450–2100 + Lee/Kwak/Reis corpus points), and add the fit's residuals to the Phase-6 gate (C.4).
6. In `mdo/solve.py`, use square root-find IFT (not least-squares implicit diff) for the coupled state (C.6).
7. Consolidate the §3 mass ledger in `mdo/mass.py` with explicit zero-placeholders; add injector hardware mass (A.2.3).
8. Carry SP-8087's ×4 cyclic-life factor as the explicit LCF margin (A.1).
9. Update §6.2's friction citation to Swamee–Jain/Blasius as implemented (B.5).
10. Note in §2.2 that SP-8089 now has an OCR mirror (usable for search, PDF-verify before extracting correlations).

---

## Sources

**Corpus** (`propulsion_texts/propulsion_texts_for_agents/markdown/`): `rao1958.md` p.2–3; `rao1999.md` p.1–3; `technical-notes-1957.md` p.2; `19710019929.md` (SP-125/Huzel & Huang) p.137; `19730022965.md` (SP-8087) incl. lines 1673, 5094, 5172; `19770009165.md` (SP-8120) line 4062; `19720026079.md` (SP-194); `fulltext01.md` (Östlund 2002 thesis) p.52; `pizzarelli2011.md` p.1,3; `carlile1992.md` p.1,3; `pintle_injector/son2017.md` p.2; `pintle_injector/s11630-015-0753-7.md` (Son 2015) p.1; `fuel_pump_design/19740020848.md` (SP-8109) p.19; `fuel_pump_design/19710025474.md` (SP-8052); `fuel_pump_design/s42405-020-00325-z.md` (Lee 2021) p.2; `materials_science/19740017910.md` (CR-134627); `porowski1985.md`; `5f36b7c4….md` (Anderson, MCF) §11.4 region; `prmeyer.md`. Method sources cited by the plan (Martins & Lambe 2013, Martins & Hwang 2013, Gray 2019, Blondel 2022, Keller 1977) are outside the corpus and were assessed on method-soundness grounds, not re-verified from source.

**Code** (commit `f2a73ba` + working tree): file:line references as given inline; prior PDF-level verifications referenced from `docs/REPO_PHYSICS_AUDIT_2026-07-02.md` and `docs/PHYSICAL_INTEGRITY_REMEDIATION_2026-07-11.md`.
