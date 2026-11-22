# Making the chamber "shoulder radius" solver-determined — design basis & proposal

**Scope.** The CLI now defaults to `--shoulder-sizing auto`, deriving the
chamber shoulder fillet from geometric closure. A scalar
`--shoulder-radius-factor` remains available for explicit override, with the
legacy `0.25·Rt` value used only when scalar mode is requested without a value.
This memo records what physically sets that radius, which variables control it,
and which parts are geometric closure versus future physical optimization. All
claims are grounded in the `propulsion_texts` corpus; a few standard transonic
results are cited to the primary literature the corpus only reproduces as
charts.

---

## 0. Key finding first: there are *two* radii, and they are governed differently

The single most important result of the corpus search is that the convergent
side of the throat involves **two distinct radii of curvature**, and the code
already separates them:

| Radius | Code variable | Default | Where it lives | What governs it |
|---|---|---|---|---|
| **Throat‑approach (upstream) radius `Ru`** | `upstream_radius_ratio` (`throat_geometry.py:16`) | `1.5·Rt` | The arc *immediately* upstream of the throat (within ~1 `Rt`) | **Performance‑determinate**: transonic flow, discharge coefficient, throat heat flux |
| **Chamber contraction entrance fillet `Rs`** (the "shoulder") | `shoulder_radius_factor` (`chamber_geometry.py:72`) | auto geometric closure (`0.25·Rt` only in legacy scalar mode) | The fillet where the cylindrical chamber turns into the convergent cone, far upstream of the throat | **Subsonic / manufacturing trade**: contraction pressure loss, corner recirculation, local heat flux, length, weight |

NASA SP‑8120 is explicit that the throat flowfield "is independent of the
nozzle geometry downstream of the throat and **for a distance of about one
throat radius upstream of the throat**" (SP‑8120 §2.1.1, `19770009165.pdf`).
The shoulder fillet sits well upstream of that one‑`Rt` zone, so **it does not
set throat performance** — it is a subsonic contraction feature. This is why
the literature gives a clean equation for `Ru` but only a *factor list* for
`Rs`.

Consequence for the request: "solve the shoulder radius" splits into two
different problems. Both are worth implementing; only one (`Ru`) has a
closed‑form performance criterion.

---

## 1. How the contour is built today (controlling variables)

`chamber_contour()` (`raosim/chamber_geometry.py`) assembles, from the injector
face to the throat:

```
cylinder (radius Rc) → shoulder arc (Rs) → straight convergent cone (half-angle α) → upstream throat arc (Ru) → throat (Rt)
```

with
- `Rc = sqrt(contraction_ratio)·Rt`            (chamber radius)
- `Rs = shoulder_radius_factor·Rt`             (the shoulder, auto-derived by
  default or explicitly overridden)
- `Ru = upstream_radius_ratio·Rt` (=1.5·Rt)    (`throat_geometry.py`)
- `α  = convergent_half_angle_deg` (=45°)
- cylinder length `Lc` is **already solved** by root‑finding the conical‑frustum
  volume to enclose `L*·At` (`chamber_geometry.py:163‑174`).

So `Rs` is the *only* convergent‑side length scale that is neither solved nor
physically anchored. Its feasible range is geometrically coupled to the others
(verified by running `chamber_contour` over a sweep):

- a straight convergent segment must exist:
  `Rc − Rs(1−cos α) > Rt + Ru(1−cos α)` ⇒ for `Rt=23.4 mm, CR=2.5, α=45°, Ru=1.5`,
  the cap is `Rs ≲ 0.45·Rt` (factor `0.5` already fails);
- dropping `α` to 30° opens the cap past `Rs = 1.0·Rt` and restores a real
  ~10–14 mm convergent cone.

This coupling is the seed of a determinate closure (see §4).

---

## 2. What governs the throat‑approach radius `Ru` (the determinate one)

NASA **SP‑8120, "Liquid Rocket Engine Nozzles," §2.1.1.1 "Upstream Wall"**
(`19770009165.pdf`) is the controlling reference:

> "Constant‑radius arcs are used for the shape of the throat‑approach wall. A
> small radius is desirable both for **minimum overall length** and for
> **minimum wall area exposed to the high heat fluxes** … However, as the radius
> decreases, the difficulty in obtaining an accurate solution of the transonic
> flowfield increases. … the nozzle aerodynamic efficiency remained constant for
> **Ru/Rt values from 1.5 down to 0.6**. A nozzle inlet of relatively **large
> radius (R/Rt = 1.4)** … has been shown to be effective in boundary‑layer
> **film cooling** through the nozzle throat."

So `Ru` is set by a four‑way trade — **discharge coefficient (performance),
throat heat flux/cooling, transonic solvability, weight/length** — and SP‑8120
states the radii "are selected on the basis of a rough trade of performance and
fabrication considerations against cooling difficulty and weight."

**Discharge coefficient (the quantitative driver).** The effective throat area
`Āt/At` (= discharge coefficient `Cd` for inviscid flow) falls as `Ru/Rt`
decreases (SP‑8120 fig. 3). The closed form is the transonic perturbation series
of **Hall (1962)**, corrected by **Kliegel & Levine (1969), "Transonic Flow in
Small Throat Radius of Curvature Nozzles"** (SP‑8120 ref. 5; also `20030067852.pdf`
ref. 13). Leading term (with `Rc ≡ Ru` the wall radius of curvature at the throat):

```
Cd ≈ 1 − ((γ + 1)/96) · (Rt/Ru)²          (Hall leading order)
```

Kliegel–Levine reformulate the series in `1/(1 + Ru/Rt)` so it stays physical at
small `Ru/Rt`. **Cross‑check against SP‑8120 fig. 3**, γ = 1.24 (LOX/RP‑1):
`Ru/Rt = 1.5 → Cd ≈ 0.990`; `Ru/Rt = 0.6 → Cd ≈ 0.94` — matches the chart's
~0.99→~0.94 trend. Viscous/low‑Re correction is a second displacement term
(SP‑8120 fig. 8; Cuffel, Back & Massier 1969). Units: dimensionless; `γ` is the
chamber‑product ratio of specific heats already resolved in `raosim/propellants.py`.

**Throat heat flux.** Small `Ru` shrinks the area near Mach 1 (less integrated
heat load) but raises wall convex curvature, which augments the Bartz
gas‑side coefficient — the repo already exposes a curvature screen
(`--curvature-correction`, Niino‑Kumakawa/Taylor) that is the right hook for
this constraint.

## 3. What governs the chamber shoulder `Rs` (the subsonic one)

SP‑125 (Huzel & Huang, `19710019929.pdf`, p. 88) does **not** give a shoulder‑
radius number. It instead lists the factors that optimize the whole convergent
contraction, and these are exactly the drivers for `Rs`:

> "(1) Combustion performance … (2) Chamber gas‑flow pressure drop (3) Chamber
> wall cooling requirements (4) Combustion stability (5) Weight (6) Space
> envelope (7) Ease of manufacturing." (SP‑125 p. 88)

Physics notes from the corpus that make these tractable:

- The contraction is a **favorable** (accelerating) pressure gradient, so
  classic boundary‑layer separation is *not* the limiter (unlike a diffuser —
  White, *Fluid Mechanics*, `propulsion_texts/Fluid Mechanics 7th Ed.pdf`). The
  real risks of too‑sharp a shoulder are a **corner recirculation pocket**
  (local hot spot / `c*` loss) and a **local heat‑flux peak** from wall
  curvature.
- Gas‑flow pressure drop in the contraction grows with chamber Mach number,
  which is fixed by the contraction ratio (`Mc` from `Ac/At`, γ). SP‑125
  ties low chamber Mach (larger `εc`) to lower stagnation‑pressure loss
  (SP‑125 ch. 1 "gas‑flow processes," p. 88).

Net: `Rs` is a **constrained‑optimization** quantity (smooth enough to avoid a
corner pocket and a heat‑flux spike, short/light enough for weight/envelope,
manufacturable), not a closed‑form one.

---

## 4. Proposal and implementation — make each radius solver‑determined

### 4A. Throat‑approach radius `Ru` (recommended; physically closed)

**Free variable:** `Ru/Rt`.
**Objective / closure (pick one):**
- fix a target discharge coefficient `Cd*` and invert
  `Cd(Ru/Rt, γ)` (Hall/Kliegel–Levine) for `Ru/Rt`; **or**
- minimize throat integrated heat load subject to `Cd ≥ Cd*`.

**Constraints (all corpus‑grounded):**
- transonic solvability / validated range: `0.6 ≤ Ru/Rt ≤ ~2` (SP‑8120 §2.1.1.1);
- throat gas‑side heat flux (Bartz + curvature) ≤ coolant capability — already
  computable in the thermal path;
- optional film‑cooling preference → bias toward `Ru/Rt ≈ 1.4` (SP‑8120).

**Implemented first closure:** `--cd-target` derives `Ru/Rt` from the Hall
leading-order inviscid discharge-coefficient relation and enforces the
SP‑8120 `0.6 ≤ Ru/Rt ≤ 2` range. The higher-fidelity heat-load optimization
can still reuse existing `γ`, throat `Re` (from `ṁ, μ, Rt`), and cooling-margin
outputs in a later validated mode.

### 4B. Chamber shoulder `Rs` (what was literally asked)

Two tiers, from cheapest to most physical:

**Tier 1 — geometric closure (zero new physics, removes the free input).**
Make the convergent a tangent‑arc spline and pick `Rs` by one extra geometric
condition instead of a user number, e.g.:
- *largest smooth fillet*: set `Rs` to the feasibility cap that just preserves a
  non‑negative straight segment (a function of `Rt, Rc(εc), α, Ru`); or
- *target convergent length / blend*: choose `Rs` so the convergent has a
  specified length `Lconv*` or so the cylinder→cone→throat arcs are curvature
  (`G2`) continuous.
Inputs: only the values the contour already has (`Rt, contraction_ratio, α, Ru`)
plus one of {`Lconv*`, "max‑fillet", "G2"}. This alone makes `Rs` *derived*.

**Tier 2 — subsonic optimization (physical, matches SP‑125 p. 88).**
**Free variables:** `Rs` (optionally co‑solve `α`).
**Objective:** minimize convergent stagnation‑pressure loss `Δp0,conv` (or keep
`≤` a budget).
**Constraints:** no corner recirculation (cap the wall‑turn rate / minimum
fillet), local contraction heat flux ≤ coolant capability, convergent
length/weight budget, manufacturing minimum radius, `G1` tangency (already
enforced).
**New parameters needed:** `--convergent-dp-budget` (or `--max-convergent-loss`),
`--min-fillet-radius` (manufacturing), and a length/weight weight; chamber Mach
`Mc` is derived from `contraction_ratio` + γ (no new input).
**Already available:** `contraction_ratio`, `convergent_half_angle`, the thermal
model, the geometric feasibility check in `chamber_geometry.py`.

### Recommended path
The first implementation now covers **4A's `Cd*` closure** and **4B Tier 1**.
The remaining upgrade is **4B Tier 2**: a validated subsonic contraction-loss /
local heat-flux optimization behind an explicit flag, mirroring how
`--size-wall` upgrades the wall from a scalar to a screened profile.

---

## 5. Parameter summary (what the solver needs)

To solve **`Ru`**: `γ` (have), throat `Re` (derive from `ṁ, Rt, μ`), `Cd_target`
(new) **or** heat‑vs‑performance weight, coolant heat‑flux capability (have via
thermal model), bounds `0.6 ≤ Ru/Rt ≤ 2` (literature).

To solve **`Rs`**: `contraction_ratio` (have), `α` (have/co‑solve), chamber Mach
`Mc(εc, γ)` (derive), one closure of {convergent length target, max‑fillet, G2}
(Tier 1) **or** `convergent_Δp_budget` + `min_fillet_radius` + local heat‑flux
limit (Tier 2).

---

## References

Corpus (`propulsion_texts/`):
- `19770009165.pdf` — **NASA SP‑8120, *Liquid Rocket Engine Nozzles*** — §2.1.1.1
  Upstream Wall (throat‑approach radius trade; `Ru/Rt` 0.6–1.5; film cooling at
  1.4); fig. 3 (Cd vs `Ru/Rt`); fig. 8 (Cd vs throat `Re`); throat flow
  independent beyond ~1 `Rt` upstream.
- `19710019929.pdf` — **NASA SP‑125, Huzel & Huang, *Design of Liquid‑Propellant
  Rocket Engines*** — p. 88 convergent‑contraction optimization factors and
  contraction‑ratio practice (2–5 low‑thrust); ch. 1 gas‑flow pressure loss.
- `Fluid Mechanics, 7th Ed. (White).pdf` — favorable‑gradient contraction vs
  diffuser separation; minor‑loss treatment of contractions.
- `20030067852.pdf` (Kliegel–Levine ref. 13), `rao1958.pdf`,
  `5f36…14be.pdf` (Anderson, *Modern Compressible Flow*; Sauer transonic) —
  transonic throat / sonic‑line context.

External (corpus reproduces only as charts):
- Hall, I.M. (1962), *Transonic flow in two‑dimensional and axially‑symmetric
  nozzles*, Q. J. Mech. Appl. Math.
- Kliegel, J.R. & Levine, J.N. (1969), *Transonic Flow in Small Throat Radius of
  Curvature Nozzles*, AIAA Journal 7(7) — discharge‑coefficient series used here.
- Cuffel, Back & Massier (1969), *Transonic flowfield in a supersonic nozzle with
  small throat radius of curvature*, AIAA Journal — experimental Cd/heat‑flux.
