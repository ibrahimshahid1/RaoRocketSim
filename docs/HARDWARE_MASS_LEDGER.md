# Engine hardware mass ledger

Closes gaps **12.3** (chamber/nozzle hardware mass), **12.4** (injector hardware
mass) and **12.5** (pump BOM mass) from the MDO remediation review.

## The problem this solves

Before this change the optimizer minimised a quantity it labelled *engine
package mass* that contained pumps, motors, inverters and battery — and nothing
else. Chamber, nozzle and injector mass were zero placeholders. Two consequences:

- the reported mass-vs-Isp Pareto front was a **feed-system** front, not an
  engine front, so every trade that spends chamber structure to buy feed-system
  mass was invisible to the optimizer;
- `EngineState.masses` carried three permanently-`NaN` slots and the shared host
  snapshot reported `thrust_chamber_mass_kg`, `injector_mass_kg` and
  `total_engine_package_mass_kg` as unavailable on both paths, so the MDO and
  traditional pipelines could never be compared on mass at all.

## Governing design decision

**Mass is integrated from the same resolved geometry the CAD exporters build.**
Not from a mass-scaling correlation.

| ledger scope | geometry source | what the CAD does with it |
|---|---|---|
| thrust chamber | `raosim.regen_profile.RegenWallProfile` | `raosim.regen_cad` revolves it |
| bolted interface | `raosim.interface.resolve_bolted_interface_geometry` | sets the flange and bolt circle |
| injector | `raosim.injector_cad.resolve_machined_pintle_layout` | `build_machined_pintle_bodies` cuts the solids |
| electric feed | `raosim.pumps` reference geometry | `raosim.pump_cad` / `pump_cad_brep` |

A reported mass and an exported STEP part therefore cannot silently disagree —
the property the "requirements in, parameterized parts out" workflow needs.

## Physical basis

Every entry is a solid-of-revolution or prismatic volume times a catalog
density. That is the relation NASA SP-125 uses for preliminary shell mass:

```
W_c = 2 π a l_c t_c ρ            (SP-125 eq. 8-32, printed p. 339)
```

*Design of Liquid Propellant Rocket Engines*, Huzel & Huang, NASA SP-125, 1971
(`propulsion_texts/19710019929.pdf`, ch. VIII). Here `a` is the **nominal
(mid-surface)** radius, so a shell of thickness `t` standing off a gas-side
radius `r` contributes `2π(r + t/2)·t` per unit meridional arc — Pappus's
centroid theorem. Per station:

```
A_liner    = 2π (r + t_w/2) · t_w
A_land     = π (r_o² − r_i²) · b/(b + w)          r_i = r + t_w,  r_o = r_i + h
A_closeout = 2π (r_o + t_j/2) · t_j
```

The land is written as an **area fraction** of the channel annulus rather than
`N·b·h` because the fraction is invariant to helical stretch (`b` and `w` are
both widths normal to the coolant path), is unconditionally non-negative, and
stays smooth in the design variables — all three matter inside an SLSQP loop.
The tests pin the two forms to agree within 1 % on a curved wall and exactly on
the MDO's analytic grid.

SP-125 also shows that preliminary estimates normally carry an explicit
non-ideal allowance: the pressurant-vessel estimate

```
W_v = π d² ρ_m (pd/4s) + 3π d ρ_m (0.5 pd/4s)     (SP-125 eq. 5-16, p. 173)
```

adds a 3-inch weld-land band at half wall thickness on top of two hemispherical
membranes. `thrust_chamber_mass_ledger(..., joint_allowance=…)` exposes the same
idea, wired to `ManufacturingSpec.weld_allowance` / `braze_allowance`. It
**defaults to 1.0** — no allowance is invented on the user's behalf, and the
ledger emits a warning saying so.

Structural context is NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled
Combustion Chambers*, 1973 (`propulsion_texts/19730022965.pdf`). §2.1.3 names
the three jobs of the metal this ledger integrates — hoop support about the
combustion chamber, support at the throat against bending and buckling, and hoop
support about the expansion nozzle against collapse from hoop compression when
the nozzle runs overexpanded at sea level — and quotes the design factors of
safety in use: **yield 1.0–1.32, ultimate 1.3–1.8**.

## Two bugs found and fixed along the way

**Quadrature.** The pre-existing private helper `raosim.thermal_design._wall_mass`
summed `hypot(gradient(x), gradient(y))`, which gives each end node a full
segment and over-counts the meridian by one grid interval — the repository's own
`regen_profile._nodal_weights_from_segments` docstring already describes exactly
this failure. It also used the bore radius rather than the mid-surface radius,
under-counting each shell by `t/2`. Both are corrected here; a test pins the
nodal weights to sum to the true arc length.

**Missing land metal.** `_wall_mass` integrated only liner and jacket. At the
13 kN baseline the channel lands are **2.07 kg of 6.31 kg** — 33 % of the
thrust chamber — so omitting them was not a rounding error.

## Honesty rules

- Missing geometry or density produces an item with `mass_kg = None` and an
  `unavailable_reason`. **Never `0.0`.**
- `MassLedger.complete` is `False` if any item is unavailable, and
  `total_mass` is then `None`. A partial rollup exists as `resolved_mass` and is
  always flagged `resolved_mass_is_partial`.
- Subsystem rollups in the snapshot withhold the whole subsystem if any of its
  rows is unknown, so a lower bound can never be read as a total.
- `status` separates `geometry_resolved` (integrated from a resolved CAD layout)
  from `screening_sized` (a documented first-order shape assumption — currently
  only the ISO 4014/4032 hex bolt-head and nut envelopes, and the pump
  diffuser-vane and port stubs).

## What is now available

| field | MDO path | traditional path |
|---|---|---|
| `thrust_chamber_liner_mass_kg` | ✅ station-grid integral | ✅ `RegenWallProfile` integral |
| `thrust_chamber_land_mass_kg` | ✅ | ✅ |
| `thrust_chamber_closeout_mass_kg` | ✅ | ✅ |
| `thrust_chamber_mass_kg` | ✅ | ✅ |
| `injector_mass_kg` | ❌ needs the host machined layout | ✅ faceplate + post + sleeve |
| chamber flange + bolts | ❌ | ✅ |
| `pump_mass_kg` | ✅ | ✅ core BOM now complete |
| `total_engine_package_mass_kg` | ❌ | ❌ — see below |

`total_engine_package_mass_kg` is **deliberately still unavailable**. Valves,
lines, manifolds, gimbal and mounts are not modelled, and a thrust chamber plus
an injector plus a feed system is not an engine dry mass. Claiming otherwise
would repeat the exact failure this work exists to fix.

## Verification

1. **Numerical identity.** The JAX station-grid integral reproduces an
   independent NumPy re-derivation to 12 significant figures for all three
   branches.
2. **Formulation cross-check.** The land area-fraction form and the discrete
   `N·b·h` rib form agree exactly on the MDO grid and to 0.26 % on the curved
   host contour (they differ only because `RegenWallProfile` measures the rib
   pitch on the *normal*-offset mid-surface while the annulus uses radial
   offsets).
3. **Gradient.** `jax.grad` of the total is finite in all four traced inputs,
   matches central finite differences to 1e-5 relative on `t_wall`, and has the
   right signs: `∂m/∂t_wall > 0`, `∂m/∂h > 0`, `∂m/∂Rt > 0`, `∂m/∂w < 0`.
4. **Cross-pipeline.** At the 13 kN baseline the traditional chamber mass is
   5.548 kg against the MDO's 6.309 kg — a 12.1 % gap. The wetted-area ratio
   between the two contours is 0.8828 and the per-branch mass ratios are 0.8832
   (liner) and 0.8852 (closeout). **The mass gap is entirely a geometry-
   convention gap, not a mass-model gap** — see the finding below. A parity test
   pins the mass ratio to the wetted-area ratio so a future divergence in the
   mass models themselves cannot hide inside it.

## Findings this made visible — and what was then done about them

These were all invisible while chamber and injector mass were zeros. All three
have since been acted on; the "resolved" notes below record the outcome.

### 1. The two pipelines do not agree on chamber length

The MDO's analytic grid runs the chamber from `x = −229.3 mm`; the traditional
`L*`-derived chamber starts at `−209.2 mm`. Same throat, same exit, same radii —
different chamber length convention, worth 12 % of thrust-chamber mass and
11.7 % of wetted area (and therefore of total heat load). `MissionSpec.chamber_length`
was a prescribed cylindrical length taken as `L*/CR`; `DesignInput` derives its
chamber from `L_star`.

**RESOLVED (R0).** SP-125 is explicit that the chamber volume spans injector
face to throat plane (printed p. 88: *"it has been arbitrarily defined that the
combustion chamber volume includes the space between injector face I-I and the
nozzle throat plane II-II"*), so the convergent section carries part of `L*·A_t`
and `L*/CR` necessarily makes the barrel too long. The MDO grid now uses the
same four-section construction as `chamber_geometry.chamber_contour` — barrel →
shoulder fillet → straight convergent → upstream throat arc — with the barrel
solved from the volume closure. No root solve is needed: the fixed sections'
volume does not depend on the barrel, and a constant-radius barrel's revolved
volume is exactly `π Rc² Lc`, so `Lc = (L*·A_t − V_fixed)/(π Rc²)` in closed
form. `chamber_length` is no longer a `MissionSpec` field.

| | before | after |
|---|---|---|
| barrel length, MDO vs traditional | 136.2 vs 15.6174 mm | 15.6174 vs 15.6174 mm |
| chamber volume / `L*·A_t` | not tracked | 1.000000000 both paths |
| wetted-area ratio trad/MDO | 0.8828 | **1.00102** |
| thrust-chamber mass delta | 12.06 % | **0.044 %** |

The residual 0.1 % is the MDO's 24-station grid chord-cutting the fillet and
throat arcs against the traditional contour's several hundred points. A parity
test now pins barrel length and chamber volume to 1e-6 and wetted area to 1 %,
so a convention drift fails loudly instead of hiding in heat load again.
`chamber_volume_margin` is a reported constraint: a chamber whose fixed sections
already exceed `L*·A_t` is infeasible and says so, rather than being clamped to
a positive barrel.

### 2. The default closeout is under-thick for a copper jacket

SP-125 (p. 109): *"the outer shell is subjected only to the hoop stress induced
by the coolant pressure"*, so `t = p_co · r_o / σ_allow`. At the 13 kN baseline
(jacket pressure 5.19 MPa, outer radius 91.0 mm), at SP-8087's factors of safety:

| jacket alloy | S_y | t required (yield 1.32) | t required (ult. 1.8) | vs. assumed 1.6 mm |
|---|---|---|---|---|
| Inconel 718 | 1035 MPa | 0.60 mm | 0.82 mm | **2.0–2.7× conservative** |
| Stainless 316L | 290 MPa | 2.15 mm | 2.93 mm | 1.3–1.8× **thin** |
| NARloy-Z | 125 MPa | 4.99 mm | 6.81 mm | 3.1–4.3× **thin** |

The default mission used one `rho_wall` (9130 kg/m³, NARloy-Z class) for both
liner and closeout, so the default closeout mass was a **lower bound**. This is
precisely why SP-8087 §2.1.3.1 records that *"hardenable materials often are
used for jacket designs, where, after brazing, the strength can be increased
considerably by agehardening"* — the jacket is normally a different, stronger
alloy than the liner.

**RESOLVED (R1).** The jacket is now **sized, not assumed**, in both pipelines.
Per station,

```
t_j(x) = FoS · p_coolant(x) · r_outer(x) / σ_yield,jacket
```

floored at a manufacturing minimum, using the coolant pressure the cooling march
already solves — so the structure and the hydraulics cannot disagree about the
load. The thickness is *tapered*, which SP-8087 §2.1.3.1 records as normal
practice: *"The brazed jacket can be tapered for optimum strength and weight."*
The factor of safety is SP-8087 §2.1.3's conservative yield value, 1.32. The
liner and jacket are separate materials, defaulting to a copper-alloy liner
inside an Inconel 718 jacket.

At the baseline this gives a 0.500–0.591 mm tapered jacket weighing **0.714 kg**,
against **2.53 kg** for the old copper 2×-`t_wall` assumption — lighter *and*
structurally justified, where the old one was neither.

Two new constraints came with it:

- **`jacket_thin_shell_margin`** — SP-125 (printed p. 336) limits the membrane
  treatment to `t/r ≤ ~1/15`. A jacket thick enough to violate it is reporting
  that the alloy or the jacket pressure is wrong, so it is an admissibility
  constraint rather than a warning. Baseline margin +0.0518.
- **`nozzle_collapse_margin`** — SP-8087 §2.1.3's third structural job, which
  the repository did not screen at all: *"hoop support about the expansion
  nozzle to resist collapse from hoop compression … during operation at sea
  level, where jet separation occurs during start and shutdown and the nozzle
  runs overexpanded."* SP-8120 §2.2 records the consequence: *"A typical failure
  of this kind is the collapse of the nozzle from overexpansion during ground
  testing."* Implemented in `raosim/mdo/structures.py` from NASA SP-8007 (rev.
  Aug 1968) §4.2.3 eqs. (16)–(21) with its recommended correlation factors
  γ = 0.75 and γ = 0.90. This is **not** the separation constraint: separation
  asks whether the flow detaches, collapse asks whether the shell survives the
  external pressure while attached and overexpanded. Baseline margin +1.81.

One subtlety worth recording, because getting it backwards rejects good designs:
SP-8007 eq. (19) is the long-cylinder **floor** that a finite shell sits above,
not a cap on it — figure 4's eq. (16) line *falls* with the Batdorf parameter
until it meets the oval-mode asymptote. The allowable is therefore the **larger**
of the two branches. Taking the smaller applies the infinite-length asymptote to
a short nozzle; on this baseline that error alone reported a −0.84 collapse
margin for a shell whose true margin is +1.81.

### 3. Auto-sized interface hardware dominates engine mass

At the 13 kN baseline the ledger was 24.5 kg:

| subsystem | mass |
|---|---|
| thrust chamber (liner + lands + closeout) | 5.55 kg |
| chamber interface (flange ring 7.34 + 12 bolts 0.04) | 7.87 kg |
| injector (faceplate 10.99 + post 0.07 + sleeve 0.03) | 11.09 kg |

The flange auto-sized to a 285.5 mm OD on a 177 mm chamber and the faceplate
floored at 21.3 mm thick. **Those two layout defaults were 75 % of the engine's
modelled hardware mass** — larger than the entire thrust chamber — and came from
bolt-circle and edge-distance spacing rules, not from a load path.

**PARTLY RESOLVED (R2).** Tracing the chain shows one root cause. The bolt hole
defaults to `0.06 × chamber diameter` = 10.6 mm; the bolt circle to
`chamber_OD + 6 × hole`; the flange OD to `bcd + hole + 2 × edge_req` = 285.5 mm;
and the faceplate to `2 × hole` = 21.3 mm. The *structural* faceplate
requirement from the clamped-plate screen `σ ≈ 0.75 Pc a²/t²` is only 5.07 mm.
**Shrinking the fastener fixes the flange diameter and the faceplate thickness
at once.**

`raosim.interface.size_bolted_interface` now selects the lightest admissible
joint from a real ISO 262 coarse-thread series, sizing the bolt count against
the separation load `F = k·Pc·π r²` using the ISO 724 stress area and the
ISO 898-1 proof stress. Every candidate is still resolved through the existing
edge-distance, pitch and plate-bending rules — this narrows the layout to the
lightest admissible one, it does not bypass a screen.

| min bolt | selection | flange OD | faceplate t | joint mass |
|---|---|---|---|---|
| layout default | — | 285.5 mm | 21.3 mm | 18.13 kg |
| M5 (default) | M5×0.8 × 14 | 234.0 mm | 11.00 mm | **5.53 kg** |
| unbounded | M3×0.5 × 36 | 212.0 mm | 6.60 mm | 2.48 kg |

Mass falls monotonically with fastener size, so the unbounded search runs to the
smallest thread that carries the load. That is real but poor hardware — 36 small
fasteners are hard to torque consistently and easy to gall — so the floor is an
explicit `min_bolt_diameter` parameter with a conventional M5 default rather
than a hidden penalty term.

This is reported as `hardware_mass.joint_sizing_opportunity` and marked
**advisory**: adopting it changes exported CAD, so the resolved geometry is left
alone until the user passes the selection through `InterfaceSpec`. Two things
did change unconditionally, because they were simply wrong: the flange, bolts
and injector body are now priced in the jacket/structure alloy instead of the
copper liner (SP-8087 §2.1.3.1), which is why the ledger reads 20.74 kg rather
than 24.51 kg.

What remains: the faceplate is still 11.0 mm against a 5.07 mm bending
requirement, because `2 × hole` — a bearing/engagement heuristic — governs
instead of the plate screen. For a through-bolted joint that rule is
over-conservative, but weakening an existing screen needs its own basis, so it
is reported rather than changed.

### 4. The legacy wall-mass helper was wrong in three ways

`thermal_design._wall_mass` feeds the channel auto-sizer's `min_mass` objective.
It summed `hypot(gradient(x), gradient(y))` (over-counting the meridian by one
grid interval), used the bore radius instead of the mid-surface radius
(under-counting each shell by `t/2`), and omitted the channel lands entirely.
The land omission was the serious one and was *systematically* biased: narrower
channels leave wider ribs, so ignoring land metal made fine channels look free
when they are not. All three are fixed, and the land term is now passed the
channel count and width so the objective ranks layouts by the right quantity.

## API

```python
from raosim.mass_ledger import (
    thrust_chamber_mass_ledger,   # RegenWallProfile -> liner/lands/closeout
    flange_bolt_mass_ledger,      # InterfaceGeometryResolution -> ring + fasteners
    injector_mass_ledger,         # machined pintle layout -> faceplate/post/sleeve
    combine_ledgers,
)
from raosim.mdo.mass import chamber_mass   # differentiable mirror
```

The traditional pipeline emits the combined ledger as
`ValidatedDesignResult.report_sections["hardware_mass"]`, and the snapshot layer
maps it into `masses.*`.
