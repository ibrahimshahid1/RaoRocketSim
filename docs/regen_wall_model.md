# Regenerative wall sizing and CAD model

This note records what RaoRocketSim now computes, how the user inputs enter
the wall design, and which parts are literature equations versus repository
engineering assumptions.

## Literature inventory

The local `propulsion_texts/` directory was inventoried for regenerative-wall
sizing, channel geometry, pressure stress, thermal resistance, and helical
passages.

The governing local source is:

- `propulsion_texts/19710019929.pdf` — Huzel and Huang, *Design of Liquid
  Propellant Rocket Engines*, NASA SP-125. The relevant material is printed
  pages 104–110 (PDF pages 113–119): wall conduction and overall heat-transfer
  relations, coolant-passage design, tubular-wall longitudinal inelastic
  buckling (equations 4-28/4-29), coaxial-shell liner stress (equation 4-31),
  and Darcy pressure loss (equation 4-32).

The expanded local corpus also supplies:

- `19730022965.pdf` — NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled
  Combustion Chambers*: variable passage/wall tailoring, thermal margin,
  coolant velocity, RP-1 coking, manifolds, and design cautions.
- `pizzarelli2011.pdf` and `eucass1p171.pdf` — roughness, entrance, and
  Niino-Kumakawa/Taylor curvature corrections for rocket cooling channels.
- `atefi2019.pdf` — variable channel-height/width optimization, fin
  resistance, and Swamee-Jain rough-channel pressure loss.
- `carlile1992.pdf`, `wadel1997.pdf`, `pizzarelli2013.pdf`, and
  `pizzarelli2014.pdf` — experimental/numerical high-aspect-ratio channel
  behavior and the onset of diminishing benefit from thermal stratification.
- `leccese2018.pdf` — the non-negligible gas-radiation contribution omitted
  by a convective-only Bartz screen.
- `pizzarelli2015.pdf` — near-critical methane heat-transfer deterioration
  and the competing effects of pressure and roughness.

Adjacent local sources were checked but are not used as regenerative-wall
allowables:

- `fulltext01.pdf` and the nozzle-geometry/Rao papers concern contour,
  separation, or method-of-characteristics behavior.
- `CAD_04.pdf` and `L-05_BSplines_NURBS.pdf` concern CAD curve/surface
  representation, not rocket-wall sizing.
- the general fluid-mechanics text supports standard Darcy/hydraulic-diameter
  practice but is not used for material allowables.

Primary online record for the local SP-125 scan:

- [NASA NTRS 19710019929](https://ntrs.nasa.gov/citations/19710019929)
- [NASA/TM-2005-213582, GRCop-84](https://ntrs.nasa.gov/api/citations/20050123582/downloads/20050123582.pdf)
  is the primary source used to identify GRCop-84 as a high-temperature,
  high-heat-flux copper alloy. The catalog still uses representative
  single-point properties rather than a qualified temperature-dependent
  material card.
- The scalable B-rep implementation follows the official OpenCascade
  [`BRepBuilderAPI_Transform`](https://dev.opencascade.org/doc/refman/html/class_b_rep_builder_a_p_i___transform.html)
  and [`BRepAlgoAPI_Cut`](https://dev.opencascade.org/doc/refman/html/class_b_rep_algo_a_p_i___cut.html)
  APIs: patterned ribs share transformed source geometry, Boolean operations
  run as kernel-level multi-shape operations, and the STEP is re-imported to
  verify one valid solid.

## Implemented equations

The station-wise thermal circuit is

$$
\frac{1}{H(x)} =
\frac{1}{h_g(x)} +
\frac{t_\mathrm{hot}(x)}{k_w} +
\frac{1}{h_c(x)}.
$$

`h_g` uses the repository's Bartz implementation. `h_c` uses turbulent
Sieder-Tate with local rectangular hydraulic diameter and coolant viscosity,
plus a `Nu=4.36` circular-duct, uniform-heat-flux laminar proxy and fin area.
The laminar rectangular-duct aspect ratio, heated-wall configuration, and
entrance region are not yet resolved. The optional curvature multiplier is the
Niino-Kumakawa/Taylor relation
`Nu_curved/Nu_straight = [Re (D_h/(2R_c))^2]^(+/-0.05)`. Automatic
liquid-coolant sizing leaves it disabled because SP-8087 recommends that
liquid curvature enhancement not be credited without experimental
calibration. `--curvature-correction` opts into the screen.

Heat is integrated over the actual meridional wall area,

$$
\dot Q = \int q''(s)\,2\pi r(s)\,\mathrm{d}s,
$$

using trapezoidal control lengths. Axial spacing `dx` is not used as a
substitute for wall arc length at the convergent section or throat.

The CLI exposes `--coolant-inlet-temperature`. Fluid-specific defaults are
120 K for methane, 25 K for LH2, and 300 K for the remaining screening
coolants; an explicit value overrides the default. The same inlet temperature
is propagated through direct analysis, channel auto-sizing, and variable-wall
sizing.

The coolant pressure loss is integrated locally:

$$
\Delta p = \sum_i f_i \frac{\Delta L_i}{D_{h,i}}
                 \frac{\rho_i V_i^2}{2}.
$$

`f` uses `64/Re` in laminar flow, smooth-wall Blasius when roughness is zero,
and Swamee-Jain when `channel_roughness > 0`. Transition detail and local
fitting/manifold losses are reported as unresolved rather than hidden inside
the distributed friction term.

For RP-1/kerosene, the solver reports a conservative coolant-side wall
temperature margin against the 700 K lower edge of the coking range in
SP-8087. This is warning-only unless `--gate-coolant-chemistry` is selected.
The Bartz output is convective heat flux; gas radiation is explicitly
reported as not included rather than silently folded into a safety factor.

For helical channels, `ΔL` is the 3-D centerline length. SP-125 supplies the
Darcy relation; the particular helix parameterization
`theta(x) = theta0 + 2*pi*turns*(x-x0)/L` is a RaoRocketSim geometry
assumption shared by analysis and visualization.

`channel_width` is defined transverse to coolant flow. For a helical path,
the same width occupies a larger circumferential band in a constant-axial
section by the local stretch `dl/ds`. The thermal fin pitch, channel-fit
gate, pressure-drop model, and B-rep all apply this same transformation.

Absolute jacket pressure is recovered from a user-supplied coolant outlet
pressure. If omitted, the explicit screening boundary is

$$
p_{c,\mathrm{out}} = P_c + \Delta p_\mathrm{injector}.
$$

The inner-liner stress is evaluated at every station using SP-125 equation
4-31:

$$
\sigma_c(x) =
\frac{|p_\mathrm{cool}(x)-p_g(x)|r_\mathrm{channel}(x)}
     {t_\mathrm{hot}(x)}
+
\frac{E\alpha q''(x)t_\mathrm{hot}(x)}
     {2(1-\nu)k_w}.
$$

For the local liner channel-roof hoop term, `r_channel` is the coolant-passage
scale (`channel_width/2` for the milled rectangular channels in the variable
profile path), not the nozzle/chamber shell radius.  The shell radius is still
used where the model is actually a shell or jacket hoop screen.

The outer closeout jacket uses the preliminary thin-shell hoop screen

$$
t_\mathrm{jacket}(x) =
\max\left[
\frac{p_\mathrm{cool}(x)R_\mathrm{jacket}(x)\,\mathrm{FoS}_j}
     {S_{y,j}},
t_{\mathrm{jacket,min}}
\right].
$$

This is a screening interpretation of SP-125's statement that the outer
shell carries coolant-pressure hoop load. It is not a buckling, joint, or
nonlinear shell analysis.

SP-125 equation 4-29 is also implemented:

$$
S_c =
\frac{4E_tE_c}{(\sqrt{E_t}+\sqrt{E_c})^2}
\frac{t}{\sqrt{3(1-\nu^2)}\,r_\mathrm{tube}}.
$$

The source defines `r` as the coolant-tube radius and requires tensile and
compressive tangent moduli at wall temperature. For a milled rectangular
channel, RaoRocketSim reports an explicitly approximate equivalent-tube
screen using `r_tube = h/2` and a user-visible `Et/E = Ec/E` assumption.
It does not gate feasibility by default. `--gate-sp125-tube-buckling` opts
into that approximation.

Coolant-over-gas pressure is separately screened as circumferential liner
compression between ribs. The current long-plate buckling relation and its
knockdown are an engineering screening assumption, not SP-125 equation 4-29
and not a replacement for imperfection-sensitive shell/panel FEA.

Coffin-Manson-Basquin inversion is implemented only when a complete sourced
strain-life coefficient set is supplied. The nominal thermal strain scale
follows SP-125 equation 4-28 directly: `S_l = E alpha DeltaT`, hence
`Delta epsilon = alpha DeltaT`. No hidden `1/(1-nu)` multiplier is applied.
The result remains a screen because a real life prediction requires the
stabilized local elastic-plastic hysteresis range.

## How the inputs alter thickness

| Input | Current effect on wall/channel sizing |
|---|---|
| `Rt` | Sets throat circumference, heated area, cycle mass flow, channel fit, and stress radius. |
| `epsilon` | Sets exit radius and downstream stress/area; a large exit can govern pressure stress. |
| `length-pct` | Changes heated and hydraulic path length, coolant heat pickup, and pressure drop. |
| `gamma` | Changes gas state, Mach/pressure distribution, performance, and Bartz inputs. |
| `pa/p0` | Primarily affects contour/performance/separation; it does not directly prescribe wall thickness. |
| `Pc` | Strongly changes gas heating, total propellant flow, absolute coolant pressure, and liner/jacket pressure load. |
| coolant | Selects density, viscosity, conductivity, heat capacity, and their screening temperature trends. |
| `O/F` | In the simplified cycle, fuel coolant is `mdot_total/(1+O/F)` times cooling fraction. Lower `O/F` gives more fuel coolant; real thermochemistry must also be rerun. |
| thermal margin / max wall T | Sets the permitted peak gas-side wall temperature. |
| pressure-drop budget | Rejects channel/helix designs with excessive hydraulic loss. |
| channel height/width/count | Changes flow area, velocity, hydraulic diameter, film coefficient, land width, and pressure drop. |
| helix turns | Increases passage length and pressure drop; it does not create free extra heat capacity. |
| `auto-size` | Selects channel count/width while holding wall thickness fixed. |
| `size-wall` | Co-sizes channel count/width/depth and a uniform seed liner, then refines `t_hot(x)`, `h(x)`, and `t_jacket(x)`. |

Without `--size-wall`, `--wall-thickness` is only a uniform reference input.
It is useful for geometry previews and fixed-thickness sensitivity studies,
but it is not reported as an analyzed wall design. The CLI summary labels the
distinction as `uniform_reference_input_not_sized` versus
`station_wise_thermostructural_sizing`.

The variable profile does not impose a generic “thick chamber and throat,
thin bell” rule. SP-8087 section 3.1.1.3 recommends thin walls where heat
loads are highest, tapering through the expansion nozzle for
heat-flux-limited coolants, and increasing thickness where a larger
subcritical-coolant safety margin is required. Those are competing local
requirements. Accordingly, `size_wall_profile` solves pressure, thermal
gradient, wall-temperature, buckling, manufacturing/degradation floor, and
coolant-pressure jacket bounds at each station. `--t-hot-min` exposes the
process-specific liner floor; its default is a preliminary placeholder, not
a universal literature limit.

## Geometry and CAD status

Wall offsets are measured along the local contour normal, not by adding a
constant radius. A `RegenWallProfile` carries:

- hot-gas liner thickness `t_hot(x)`;
- channel width and height;
- land width;
- closeout-jacket thickness `t_jacket(x)`;
- channel count and helix turns.

The runner exports:

- `wall.stl` / `wall.step`: the closed liner-base solid;
- `jacket.stl` / `jacket.step` after variable wall sizing: the separately
  closed closeout-jacket solid;
- `regen.stl`: liner/channel/jacket visualization surfaces.
- with `--regen-brep`, `regen.step`: one re-import-validated OpenCascade
  material solid containing the liner, full channel-count ribs, end seals,
  and jacket. The passages are the gaps between patterned positive ribs, so
  the kernel performs one multi-shape material fuse instead of one channel
  subtraction per passage.
- with `--regen-manifolds`, the same one-solid STEP additionally contains
  two annular plenum voids and area-sized radial port voids. Total port area
  defaults to total channel flow area. The report records plenum/channel area
  ratios and port/plenum intersection volumes.

With CadQuery/OpenCascade installed, STEP is a revolved B-rep that CAD
systems can import as a solid body. Without it, the AP214 file is explicitly
reported as `faceted_brep`. `--require-brep` rejects that fallback.

Binary solid STL export is independently gated before writing: angular-seam
vertices are shared exactly, both normal-offset end rings are closed, the
optional flange continues the inner bore to its upstream cap, every mesh edge
must have two oppositely wound incident triangles, and the signed triangle
volume must match the exact revolved piecewise-linear profile volume after
the regular-polygon faceting correction. The CLI records boundary-edge count,
nonmanifold-edge count, and enclosed volume in `summary.json`.

`regen.step` is a neutral B-rep, not an Inventor feature tree. A STEP file can
contain the final patterned topology but cannot preserve a native Inventor
pattern feature. Inventor can import it as a real solid body and save it as
IPT. Native editable feature history would require a separate Inventor,
FreeCAD, or other parametric-CAD document generator.

With `--hydraulic-network` (automatically enabled by
`--regen-manifolds`), the analytical model now solves a nonlinear graph with
every channel branch, two annular header rings, each discrete port, ring
friction, and entry/exit minor losses. It reports source-to-sink pressure
drop and branch-flow maldistribution. The graph is one-dimensional: 3-D
turning, separation, port jets, and calibrated local K values still require
manifold CFD or test data.

## Qualification limits

- Material strengths/conductivity remain representative screening values,
  not heat-certified temperature-dependent allowables.
- NARloy-Z fatigue and cyclic tangent data are derived from NASA CR-134627;
  GRCop-84 uses the direct strain-life and cyclic stress-strain regressions
  in NASA 20060005216. These are active preliminary screens, not
  thermomechanical hardware qualification.
- Jacket buckling, creep, plasticity, weld/braze efficiency, stress
  concentration, calibrated 3-D manifold flow, forced-flow cryogenic CHF,
  supercritical heat-transfer deterioration, soot radiation, and conjugate
  CFD/FEA qualification remain outside this model.
