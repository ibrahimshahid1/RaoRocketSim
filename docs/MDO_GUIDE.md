---
title: LREKit Differentiable Engine-MDO — Reviewer's Guide
subtitle: What every module does, where each model comes from, and how to run it
date: 2026-07-24
audience: whoever is reviewing or extending raosim/mdo/
---

# Reviewer's Guide to the Engine-MDO Layer

This is the orientation document for `raosim/mdo/`. It answers four questions:
**what is here**, **where each model comes from**, **how to run it**, and **what
is still soft**. Companion documents: `DIFFERENTIABLE_ENGINE_MDO_PLAN.md` (the
formulation + phase tracker) and `LITERATURE_AUDIT_2026-07-24.md` (the
claim-by-claim source verification).

---

## 1. The 60-second version

`raosim/mdo/` turns the existing sequential LREKit models into **one coupled,
end-to-end differentiable engine** and drives it with a gradient-based
constrained optimiser. Everything is JAX; every converged state is
differentiated by the implicit function theorem (never by unrolling a solver);
every block keeps a NumPy oracle and a parity test.

Run it two ways from the existing CLI:

```bash
lrekit                      # interactive: pick workflow, then its parameters
lrekit --engine-mdo         --pc 3e6 --epsilon 8 --film-frac 0.10   # one point
lrekit --engine-mdo-optimize --isp-min 200                          # optimise
```

The traditional nozzle/CAD workflow is untouched and is still the default.

---

## 2. Module map

| File | Role | Key entry points |
|---|---|---|
| `schema.py` | Design variables, bounds, and **every physical constant with its citation** | `DesignVector`, `MissionSpec`, `default_design_space()` |
| `scaling.py` | Affine map to the unit box (conditioning) | `ScaledSpace` |
| `properties.py` | C¹ property surfaces over (P_c, O/F) | `load_chamber_surfaces`, `sample_cea_tables` |
| `grid.py` | Fixed-topology station grid on the **real Rao/TOP contour** | `build_station_grid`, `rao_wall_angles` |
| `cooling.py` | Regen jacket + film cooling; the one **implicit** block | `solve_cooling`, `film_effectiveness` |
| `injector.py` | Pintle: orifice/TMR/blockage + Son two-branch area | `injector_readouts` |
| `pump.py` | Electric feed: duty, C¹ efficiency, battery epigraph | `electric_feed` |
| `engine.py` | **The coupled solve** — all blocks, one gradient | `solve_engine`, `engine_outputs` |
| `nlp.py` | ε-constraint optimiser + Pareto sweep | `solve_min_mass`, `pareto_frontier` |
| `assembly.py` | The original 4-variable walking skeleton (kept as a gate) | `make_engine_fn` |
| `postprocess.py` | **Phase 11 bridge** → authoritative contour/CAD/reports + discrepancy report | `to_design_input`, `reevaluate`, `summarise` |

**Where to start reading:** `engine.py` docstring → `solve_engine` → then whichever
block interests you. `schema.py` is the reference for every constant.

---

## 2a. Thrust scaling — the solver is NOT 13 kN-specific

`MissionSpec.for_thrust(F)` derives the whole architecture from physics rather
than hard-coded 13 kN values, and `mission.scaled_design_space()` scales the
bounds with it:

* **throat** from the thrust closure, A_t = F/(C_F P_c);
* **chamber length** from the characteristic length, L ≈ L*/CR — SP-125 Table
  4-1 gives L* = 40–50 in for LOX/RP-1, and note this makes chamber length
  essentially *thrust-independent* at fixed L* and CR;
* **channel count** ∝ chamber circumference at a fixed manufacturable pitch;
* **pintle diameter** sized so the blockage factor lands mid-band (BF being the
  literature design knob — Hwang 2022, Freeberg 2019), not an invented ratio;
* **pump speed** from the specific speed ω = N_s(g₀H)^¾/√Q, so bigger engines
  correctly run *slower*.

Sanity check: at 13 kN the derived architecture reproduces the hand-tuned
defaults (194 channels vs 192, D_p 22.6 vs 20 mm). Without this, a 130 kN run
has **no feasible point in the box** and every Pareto row comes back infeasible.

```python
m = MissionSpec.for_thrust(130e3)      # 615 channels, D_p 71.6 mm, 19.3 krpm
```

## 2b. Propellant is a selectable input

`MissionSpec.for_propellant(name, thrust)` drives the chamber gases (γ, T_c, R),
**L\*** (SP-125 Table 4-1), densities, coolant thermophysical properties and the
**coolant wall limit** (SP-8087) from `raosim/mdo/propellants.py`:

| combination | L\* | coolant wall limit | status |
|---|---|---|---|
| LOX/RP-1 | 45 in (SP-125 40–50) | 728 K (SP-8087 850 °F) | verified feasible |
| N2O4/MMH | 32 in (SP-125 30–35) | 589 K (SP-8087 hydrazine family) | verified feasible |
| LOX/LH2 | 35 in (SP-125 30–40) | **none** — no carbon, cannot coke | see caveat |
| LOX/LCH4 | 40 in *(estimate)* | 950 K *(estimate)* | see caveat |
| N2O/Ethanol | 40 in *(estimate)* | 589 K (SP-8087 alcohols) | untested |

Hydrogen carrying **no** coking limit is the physically correct behaviour, not a
missing number — the gas-side material limit governs instead.  Methane and
N2O/ethanol post-date SP-125/SP-8087, so their L\*/wall limits are flagged
`estimated=True` in the table rather than passed off as literature.

```bash
lrekit --engine-mdo --mdo-propellant n2o4/mmh --target-thrust 5000 --pc 2e6
```

**Caveat — cryogenic feed systems are not yet propellant-specific.**  LOX/LCH4
currently comes back *infeasible* (pump suction, chug, wall temp) and the LH2
specific-speed sizing asks for ~183 krpm, which is unphysical for a single
stage — real LH2 pumps are multistage.  Both point at the same missing piece:
tank pressures, NPSH and pump **stage count** still carry storable-propellant
defaults.  Verified working today: **LOX/RP-1 and N2O4/MMH**.

## 3. The design vector (10 variables)

| Variable | Bounds | Why it is a variable |
|---|---|---|
| `Pc` | 1.5–6.0 MPa | chamber pressure |
| `eps` | 3–40 | expansion ratio |
| `dp_f_frac`, `dp_o_frac` | 0.12–0.45 | injector Δp / P_c (chug rule ≥ 0.2) |
| `D_pintle` | 10–40 mm | sets blockage factor (constrained band) |
| `N_rpm` | 15–60 krpm | pump speed → specific speed → efficiency |
| `channel_width` | 0.3–1.2 mm | jacket geometry |
| `channel_height` | 0.8–5.0 mm | jacket geometry (AR ≤ 8 enforced) |
| `film_frac` | 0–0.30 | fuel diverted to wall film — **the coking lever** |
| `t_wall` | 0.4–2.0 mm | thin for conductance, thick for structure |

**Every variable is live.** A zero Jacobian column means a variable that cannot
reach any constraint or the objective; two such variables were found and fixed
(see §7). Re-check with the snippet in §8 after adding any variable.

---

## 4. Constraints (12, all enforced by default)

| Constraint | Meaning | Source |
|---|---|---|
| `isp_epsilon` | I_sp ≥ floor | the ε-constraint |
| `separation` | no flow separation | Östlund Eq. 28–30, SP-8120 |
| `coking` | T_wc ≤ 728 K (RP-1 liquid-wall) | SP-8087 (850 °F), Sellers 1961 |
| `land_fit` | channels physically fit the circumference | geometry |
| `chug` | min(χ_f, χ_o) ≥ 0.2 | SP-8113 / SP-194 |
| `pintle_transition` | stay on the tip-controlled branch | Son 2017 |
| `pump_suction` | N_ss ≤ 4 | SP-8052 |
| `pump_tip_speed` | U₂ ≤ 400 m/s | SP-8109 |
| `aspect_ratio` | channel h/w ≤ 8 | Pizzarelli 2011, Carlile 1992, Mirzamoghadam |
| `blockage_lo/hi` | BF in 0.30–0.90 | Hwang 2022, Freeberg 2019, Ryu et al. |
| `thermal_stress` | σ_th ≤ σ_allow | SP-8087; CR-134627 / Porowski basis |
| `wall_temp` | T_wg ≤ 800 K | Mirzamoghadam allowable gas-side wall temp |

Reported but deliberately **not** constrained: coolant Mach (≈2 orders of
margin — a constraint would just add a dead column).

---

## 5. Physics, block by block — and its source

**Nozzle / performance.** Isentropic relations, Vandenkerckhove Γ, C_F with the
ambient term, `c*_delivered = η_c*·c*_ideal` (pinned in one place). SP-125,
SP-8120, Anderson.

**The contour is the real Rao/TOP parabola.** `grid.py` builds the classical
thrust-optimised contour — throat downstream arc to the initial wall angle θ_n,
then the quadratic Bézier to the exit angle θ_e — with θ_n/θ_e read from the
Rao/NASA charts through the **C¹ tensor-Hermite surfaces** (the same machinery
as the CEA properties), so the whole contour is differentiable in ε and L%.
Verified against `nozzle_geometry.bell_nozzle_contour`: angles match the chart
to 0.00°, exit radius and bell length to 4 decimal places.

**Cooling (`cooling.py`) — the only implicit block.**
- gas side: full **Bartz 1957** h_g, coefficient **0.026 verified verbatim**, with
  the σ property factor; turbulent recovery r = Pr^⅓.
- wall: series resistance + land **fin efficiency** tanh(mH)/mH and the
  `(w+2η_f h)/pitch` area augmentation (SP-125 §4; the HARCC mechanism of
  Pizzarelli/Carlile).
- coolant: **Sieder–Tate** 0.027·Re^0.8·Pr^⅓ via the audited primitive.
- march: **counterflow, upwind finite-volume** — counterflow is the
  literature-preferred arrangement for ε ≤ 7 (Mirzamoghadam).
- the stationwise wall-temperature vector is solved by Newton + IFT.

**Film cooling — liquid, not gaseous.** This is the subtlest part of the repo and
worth understanding before changing it. The classical
ε = C(X/VR)^(−0.8)Re^(0.2) correlations (Hatch & Papell TN D-130, Stollery,
Hartnett, Tribus) are fitted to **gaseous** coolants at velocity ratios
0.45–33.3. An RP-1 film is a *liquid*: continuity through any sane slot gives
VR ≈ 10⁻³, three orders below that band. So the block instead implements the
**liquid phase-change model** (Shine & Nidhi §4.3 — Kinney/Graham/Sellers/Emmons;
Huzel & Huang SP-125 Eq. 4-34):

1. per-mass enthalpy `H = c_p,l(T_w−T_co) + ΔH_vap + c_p,v(T_aw−T_w)`
2. hot-gas load `q = h_g(T_aw−T_w)`
3. film-cooled length `A_FCL = η_fc·ṁ_film·H/q`
4. a C¹ decay over that length.

`η_fc` is **derived, not fitted**: Stechman makes it a function of coolant
Reynolds number, and Grisson supplies the mechanism — below **Knuth's critical
flow-per-circumference** `Γ_cr = 1.01e5 μ_v²/μ_ℓ` the film is smooth; above it
waves shear droplets off and mass loss is "2 to 4× the evaporation rate". The
code blends 0.9 → 0.33 across that transition. Our engine runs at Γ/Γ_cr ≈ 9,
i.e. firmly wavy, so the model self-selects ≈0.33.

The gaseous velocity ratio is retained as a **diagnostic**
(`film_slot_validity`) and pinned by a test, so the inapplicability stays
documented.

**Injector.** Incompressible orifice, TMR, blockage factor, and the **Son 2017
two-branch minimum area** (tip-opening vs centre-gap) with a consistency
inequality — never a differentiable `min()`.

**Pump.** Meanline duty, N_s / N_ss / NPSH, a **C¹ η(N_s)** surrogate replacing
the shipped binned estimator (which is a C0 step and cannot be differentiated),
calibrated to the SP-125 60–85 % rocket-pump band, plus the **Lee 2021 battery
epigraph** (power- and energy-limited masses kept separate).

---

## 6. How the coupling actually works

```
outer Newton on (Rt, mdot)  ──►  grid ──►  cooling (inner Newton on T_wg vector)
                                              │
                                     Δp_regen ▼
      pump feed ◄── Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank
                                              ▼
                        P_electric, battery, masses, mass ledger
```

Two nested implicit solves, both differentiated by the IFT. The **hydraulic edge
is genuinely closed** — the jacket Δp feeds the pump rise. The one true two-way
loop (spray → η_c* → ṁ → spray) is **off by default**, because the plan flags it
as the strongest coupling but the weakest physics; `ablation_delta()` measures
exactly how much any result depends on it.

*Honest note:* most of the remaining data flow is a one-way chain, and it is
modelled that way deliberately (the plan warns against enlarging the nonlinear
system just to look coupled).

---

## 7. Two traps worth knowing about

**Dead design variables.** A variable with a zero Jacobian column *and* zero
objective gradient is worse than useless — it makes the NLP rank-deficient.
`film_slot_height` (physically inert for a liquid film) was removed;
`D_pintle` was made live by constraining blockage factor. Always re-check.

**Forward-mode AD through nested implicit solves.** `jacfwd` silently drops the
tangent of geometry-only `jnp.min` margins through jit'd nested Optimistix
root-finds. The constraint Jacobian therefore uses **`jacrev`** (FD-verified).
Do not "optimise" this back to forward mode without re-running the FD gate.

---

## 8. Running and checking it yourself

```bash
# environment (sandbox/CI): jax 0.6.2, optimistix 0.0.11, equinox 0.13.8
export PYTHONPATH=/path/to/deps:$PWD

# the whole MDO suite (~59 tests)
python -m pytest tests/test_mdo_*.py -q

# single design point, with design margins on
lrekit --engine-mdo --pc 3e6 --epsilon 8 --film-frac 0.10 --design-margins

# optimise, then trace the frontier (altitude makes the trade meaningful)
lrekit --engine-mdo-optimize --isp-min 200
lrekit --engine-mdo-optimize --isp-sweep 250,320,6 --engine-mdo-ambient 1000
```

**Check for dead variables after any change:**

```python
from raosim.mdo.nlp import _make_callables, CONSTRAINT_NAMES, DEFAULT_ENFORCED
from raosim.mdo.schema import MissionSpec, default_design_space
import numpy as np, jax.numpy as jnp
m = MissionSpec(); idx = tuple(CONSTRAINT_NAMES.index(n) for n in DEFAULT_ENFORCED)
ss, obj, og, con, cj = _make_callables(m, 190.0, False, idx)
u = np.full(len(default_design_space()), 0.5)
J = np.asarray(cj(jnp.asarray(u))); g = np.asarray(og(jnp.asarray(u)))
for k, s in enumerate(default_design_space()):
    print(s.name, np.linalg.norm(J[:, k]), g[k])   # both zero => DEAD
```

---

## 9. What is still soft (read before trusting a number)

1. **Property surfaces.** Without a CEA table the code uses *constant* γ/T_c —
   correct at the stated O/F, flat in O/F. Run `scripts/sample_cea_surface.py`
   on a host with RocketCEA to make O/F a real lever (it becomes a second coking
   lever via flame temperature). The wiring and gates already exist.
2. **η_fc transition location.** Woodmansee & Hanratty measured Γ_cr ≈ 3× lower
   than Knuth for water; Shine reports Stechman's model errs −20 % … +13 %.
   Those are the film block's honest error bars.
3. **Pump-efficiency shape.** The 60–85 % band is verified; the peaked *shape* is
   a smooth screening fit, not SP-125 Fig. 6-23 (image-only).
4. **η_c*(TMR).** A screening knob, default off, with an ablation. Do not report
   a coupled result without also reporting `ablation_delta`.
5. **Low-cycle fatigue.** `thermal_stress` is the constrained-expansion stress,
   not a cycle count. The Coffin–Manson / CR-134627 / Porowski LCF screens are
   the §8 constraint-layer deepening.
6. **Sea-level Pareto is weak.** Higher ε is overexpanded at sea level, so the
   mass–I_sp trade only becomes interesting with `--engine-mdo-ambient` set for
   altitude/vacuum.

---

## 10. Provenance

Every physical constant in `schema.py` carries its source in a comment. The
claim-by-claim verification (with file and line numbers into
`propulsion_texts_for_agents/`) is in `LITERATURE_AUDIT_2026-07-24.md`, including
the corrections made along the way — the separation-criterion relabelling, the
gaseous-vs-liquid film correction, and the hoop-stress recommendation that turned
out to be the wrong criterion.
