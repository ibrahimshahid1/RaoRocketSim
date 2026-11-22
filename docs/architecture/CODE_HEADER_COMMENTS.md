# Source-file header comments, organized by CLI data-flow stage

Verbatim module docstrings and leading comment blocks of every code file, grouped by the pipeline stage that owns the file
(stage taxonomy: [DATAFLOW.md](DATAFLOW.md)). Extracted 2026-11-05 by Sonnet agents; nothing is summarized or trimmed. A file with no
header prose is marked *Inferred role*. Use [CODE_MAP.md](CODE_MAP.md) for the one-line-per-file table.


## S01 - CLI entry points, argument parsing and workflow dispatch

*8 files.*

### `lrekit/__init__.py`

*18 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
LREKit public package facade.

The original solver modules currently live under :mod:`raosim` for
compatibility. New user-facing entry points should prefer :mod:`lrekit`.
~~~~

### `lrekit/__main__.py`

*9 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
Allow `python -m lrekit` to launch the main CLI.
~~~~

### `lrekit/cli.py`

*25 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
Console entry points for LREKit.

The job verbs (``run``/``validate``/``catalog``) are handled first; everything
else falls through to the existing CLI unchanged.  That ordering is the
compatibility translator: no existing invocation changes meaning, and the
unified workflow becomes reachable without a flag-day migration.
~~~~

### `raosim/__init__.py`
*5 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
raosim – Rao bell nozzle design & rocket engine performance toolbox
~~~~

### `raosim/__main__.py`
*9 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
Allow `python -m raosim` to launch the CLI.
~~~~

### `raosim/cli.py`
*24 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
Console entry points for LREKit and the legacy RaoRocketSim alias.
~~~~

### `raosim/run_nozzle.py`

*6981 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
run_nozzle.py — the main runner: solve the contour with the
differentiable / MOC backend, then (optionally) add the regen cooling
coils, run the cooling analysis, and export everything.

Pipeline
--------
1. Construct the trusted deterministic Rao/TOP chart Bézier contour by
   default.  ``--contour-method rao-bvp`` opts into the experimental Rao
   variational / MOC boundary-value problem (``solve_rao_bvp``):
   ``--backend jax`` is the differentiable Optimistix-LM path and
   ``--backend numpy`` is the SciPy finite-difference oracle.
2. Build the chamber from ``L*`` and contraction ratio using the same
   throat geometry as the nozzle.
3. ``--regen`` adds the N regenerative cooling channels (axial, or
   helical "coils" with ``--helix-turns``) wrapping the solved wall.
4. ``--thermal`` runs the coupled Bartz + Sieder-Tate cooling analysis
   and paints the channels with the gas-side wall temperature.

Exports (under ``--out``): ``contour.csv`` (hot-gas wall polyline),
``profile.png`` (2-D), a closed ``wall.stl``, and optionally ``wall.step``
(``--cad step``).  With ``--regen`` it also writes the channel-visualization
``regen.stl`` + ``regen_3d.png``, plus ``summary.json``.

Run examples
------------
    # INTERACTIVE: run bare (or with -i) and it asks for each dimension
    # (epsilon = expansion ratio, throat radius, length %, ...) showing
    # the [default]; press Enter to accept:
    python scripts/run_nozzle.py
    python scripts/run_nozzle.py -i

    # experimental BVP contour, host (full quality):
    PYTHONPATH=. python scripts/run_nozzle.py --contour-method rao-bvp --backend jax \
        --rt 0.02 --epsilon 10 --length-pct 80 --gamma 1.24 \
        --out builds/run1

    # contour + 80 axial regen channels + thermal colouring:
    PYTHONPATH=. python scripts/run_nozzle.py --backend jax --regen \
        --channels 80 --thermal --pc 7e6 --out builds/run_regen

    # contour + 24 helical coils, 3 turns:
    PYTHONPATH=. python scripts/run_nozzle.py --regen --channels 24 \
        --channel-width 0.0018 --helix-turns 3 --out builds/run_coil

    # AUTO-SIZE the channels from the cooling requirement (coolant flow
    # derived from the cycle); you give the limits, not the geometry:
    PYTHONPATH=. python scripts/run_nozzle.py --regen --auto-size \
        --rt 0.08 --pc 7e6 --mixture-ratio 2.6 --channel-height 0.004 \
        --margin-target 1.2 --dp-budget 300 --out builds/run_sized

    # seed diagnostic (no LM solve). Invalid seed geometry is reported and
    # deliberately not exported:
    PYTHONPATH=. python scripts/run_nozzle.py --max-nfev 0 --regen \
        --out builds/run_smoke

Note: the full ``--contour-method rao-bvp --backend jax`` solve
(max_nfev ~4000 + weight ladder)
runs for minutes — it is a host job. ``--max-nfev 0`` evaluates the seed
only and is useful for diagnostics; hard geometry-gate failures block export.
~~~~

### `scripts/run_nozzle.py`

*34 lines · stage S01 CLI entry points, argument parsing and workflow dispatch*

~~~~text
Compatibility wrapper for the packaged RaoRocketSim runner.

The maintained implementation lives in :mod:`raosim.run_nozzle` so editable
installs, console scripts, ``python -m raosim``, and this historical script path
all execute the same CLI.
~~~~


## S02 - Inputs, units, validation and job contracts

*8 files.*

### `lrekit/guided.py`

*300 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
The guided workflow: describe the engine, not the solver.

The wizard asks only what SP-125 sec. 2.1 calls the defining engine
requirements -- thrust and the condition it is quoted at, duration, propellant,
mixture-ratio intent, mass and envelope limits -- plus the architecture and
output choices a user genuinely owns.

Mission-target mode leaves design variables to the optimizer. Partial and
complete-design modes explicitly collect the values the user elects to pin.

Prompting is injected rather than calling ``input`` directly, so the whole flow
is testable with scripted answers and cannot hang a non-interactive run.
~~~~

### `lrekit/job_cli.py`

*416 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
Job-oriented verbs for the LREKit CLI.

These make the unified workflow reachable outside the Python API.  That
matters on its own terms: O/F optimization was implemented long before it was
reachable from a command line, which meant in practice it was not usable.

Verbs:

    lrekit guided                interview, preview coverage, save and/or run
    lrekit run JOB.yaml          resolve, solve, judge; exit code is the verdict
    lrekit validate JOB.yaml     resolve and report without solving
    lrekit catalog WHAT          list variables, targets, objectives,
                                 constraints, propellants or materials

Anything not recognized here falls through to the existing CLI unchanged, so
this is additive rather than a replacement.
~~~~

### `raosim/input_validation.py`
*236 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
Shared host-side validation for engine-analysis inputs.

The traditional, direct-MDO, optimization, and requirement workflows all
consume the same basic physical quantities.  Keeping their elementary domain
checks here prevents workflow dispatch order from changing whether bad input
is rejected cleanly or reaches a logarithm, square root, or division first.

These checks are mathematical/input-contract checks, not literature-derived
design limits.  Model validity domains and engineering recommendations remain
separate constraints with their own provenance.
~~~~

### `raosim/manufacturing_limits.py`
*203 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
Manufacturing limits, kept distinct from physical validity limits.

A milled channel that cannot be cut is not a physics failure -- it is a
geometry no shop will produce.  Mixing the two is how a "feasible" optimum
becomes unbuildable, so these limits live here with their own provenance and
their own applicability rather than inside the physics constraints.

Source
------
Wadel, M. F., *Comparison of High Aspect Ratio Cooling Channel Designs for a
Rocket Combustion Chamber* (NASA TM-107276 / AIAA-97-????, 1997), the
"fabrication constraints for milled channels" list:

    Aspect ratios <8; coolant channel heights <0.51 cm (0.20 in.); coolant
    channel widths >0.051 cm (0.02 in.); coolant channel landwidths
    >0.051 cm (0.02 in.); no sharp changes in coolant channel width or height

Applicability, stated plainly: Wadel's study is an analytical investigation of
a **liquid-hydrogen-cooled** chamber, and the companion validation hardware
(ref. 2 therein) is an 89 kN (20 000 lbf) class combustor.  Applying these
numbers to a 13 kN RP-1 engine is an extrapolation in both coolant and scale.
They are quoted here as *milling practice*, which travels better than a
coolant-specific thermal result, but the extrapolation is recorded rather than
hidden.

Note what Wadel does **not** say: that AR>8 is unphysical.  The paper's own
designs reached AR 40 analytically and were cut back "from 40 down to the limit
of 8" only when fabrication constraints were imposed, and Carlile (1992) tested
AR 8 and 15 experimentally.  Eight is therefore a manufacturing bound, not a
model-validity bound, and this module labels it as such.
~~~~

### `raosim/mdo/capabilities.py`
*69 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
Executable job capabilities shared by validation and user-facing catalogs.

An accepted selection names an implemented execution path. Unsupported choices
must acquire an implementation here before a client can advertise them.
~~~~

### `raosim/mdo/job.py`
*1209 lines · stage S02 Inputs, units, validation and job contracts*

~~~~text
One versioned engine-job input that every front end compiles into.

The guided wizard, ordinary CLI flags, and advanced JSON/YAML all produce an
:class:`EngineJobSpec`; :func:`resolve_engine_job` turns it into a
:class:`ResolvedEngineJob`.  Nothing downstream should branch on which front
end was used, because that is how ``--propellant`` and ``--mdo-propellant``
came to resolve the same concept differently.

``ResolvedEngineJob`` **contains** a
:class:`~raosim.mdo.resolved_inputs.ResolvedEngineInputs` rather than sitting
beside one.  A job is workflow intent (analyze / optimize / pareto, variable
policies, targets, output policy) layered on top of the physical inputs; if the
two were peers they would become a third competing definition of the same
conventions, which is precisely the failure this contract exists to end.

Design intent is always explicit.  A variable states its mode rather than being
inferred from whether a value happens to be present, and O/F intent is an enum
rather than an overloaded ``None`` -- the same lesson as the fixed-O/F
sentinel.
~~~~

### `raosim/mdo/metrics.py`

*382 lines · stage S02 Inputs, units, validation and job contracts (stage from module_stage_map)*

~~~~text
The registry of selectable objectives and targets.

"Any user-selected output target" cannot safely mean any arbitrary state
field.  A quantity is selectable only if something can actually be said about
it: where it lives in the snapshot contract, whether it is differentiable,
whether a counterpart exists in the traditional pipeline, and which models must
be available before a verdict about it means anything.

Two structural rules keep this from becoming a third competing definition:

* **Constraint metrics are derived from** :data:`~raosim.mdo.constraints
  .CONSTRAINT_MANIFEST`, never re-listed.  A constraint is a metric plus a
  relation, so the manifest stays the single authority on names, scales,
  applicability and optimizer role.
* **Every reportable metric is validated against**
  :data:`~raosim.mdo.snapshot.SNAPSHOT_FIELD_MANIFEST`.  A registry entry
  pointing at a snapshot field that does not exist is worse than no entry, so
  the module refuses to build one.

Quantities the repository cannot yet model are registered as *unavailable with
a reason* rather than omitted.  Omission reads as "not thought about"; an
explicit unavailable row lets a catalog command explain why full engine dry
mass, cost, or reliability is not on offer.
~~~~

### `raosim/units.py`
*224 lines · stage S02 Inputs, units, validation and job contracts (from module_stage_map)*
~~~~text
Engineering-notation input parsing with a canonical SI store.

Users write ``13 kN``, ``30 bar``, ``0.8 mm``.  The repository stores SI.  The
one rule that makes that safe is: **never guess a unit**.  A bare number where
a unit is expected is rejected, because the failure mode being prevented is not
a parse error -- it is a 1000x design error that runs to completion and
produces a plausible-looking answer.  That is the same class of bug as the
pintle STEP metres-as-millimetres defect and the material label that never
reached the traced core.

Each parse is dimension-checked: asking for a pressure and receiving ``13 kN``
is an error, not a silent reinterpretation.
~~~~


## S03 - Propellant identity and chamber thermochemistry

*3 files.*

### `raosim/cea.py`
*466 lines · stage S03 Propellant identity and chamber thermochemistry*

~~~~text
cea.py - Optional RocketCEA/NASA CEA thermochemistry integration.

RocketCEA is intentionally optional. When it is not installed, callers can
fall back to the built-in propellant table while preserving an explicit warning
that the values are demo-grade constants rather than CEA-derived properties.
~~~~

### `raosim/frozen_flow.py`
*1005 lines · stage S03 Propellant identity and chamber thermochemistry*

~~~~text
Thermally-perfect, frozen-composition quasi-one-dimensional nozzle flow.

The existing characteristic solvers use the constant-``gamma`` Prandtl-Meyer,
critical-Mach, compatibility, mass-flux, and thrust relations.  This module is
therefore deliberately separate: it does **not** pass a station-dependent
``gamma`` into MOC or Rao invariants.

For a fixed ideal-gas composition, ``R`` is constant while ``cp(T)`` may vary.
Here "frozen" refers to chemical composition. Thermal properties use one
temperature; no separate vibrational temperature or relaxation equation is
solved. Supplied ``cp(T)`` must describe that fixed composition, excluding the
equilibrium heat-capacity contribution from changing species mass fractions
(Anderson, Modern Compressible Flow, 3rd ed., PDF pp. 675-678, Eqs. 17.21-17.25).
The implementation uses an explicit piecewise-linear ``cp`` table and exact
segment integrals:

``h(T2)-h(T1) = integral(cp dT)``

``s(T2,p2)-s(T1,p1) = integral(cp/T dT) - R ln(p2/p1)``

An isentropic station follows from total enthalpy and entropy.  The throat is
the sonic root ``u^2 = gamma(T) R T``; mass conservation then gives
``A/A* = G*/G``.  Exit thrust is reconstructed directly from momentum and
pressure, with ``c* = p0/G*``.

All quantities are SI.  Extrapolation outside the declared property table is
forbidden, equilibrium composition changes are excluded, and a property table
must identify its composition freeze basis, generation state/tool/database,
and upstream source-artifact digest.  Every result reports closure residuals
and a deterministic input fingerprint.
~~~~

### `raosim/propellants.py`
*213 lines · stage S03 Propellant identity and chamber thermochemistry*

~~~~text
propellants.py – Propellant combustion-product database.

Each entry stores the thermodynamic properties of the *exhaust gas*
(combustion products) at the nominal O/F ratio, not the raw propellants.

Fields
------
gamma    : ratio of specific heats of combustion products  (γ)
Mw       : mean molecular weight of exhaust  [kg/mol]
Tc       : adiabatic flame / chamber stagnation temperature  [K]
eta_cstar: characteristic-velocity (combustion) efficiency  η_c*
eta_CF   : thrust-coefficient (nozzle) efficiency  η_CF
eta_Isp  : overall specific-impulse efficiency  η_Isp = η_c*·η_CF
OF       : nominal oxidiser-to-fuel mass ratio
source   : provenance string for the thermochemical constants

Efficiency convention
---------------------
Combustion efficiency and nozzle efficiency act on physically distinct
quantities and must not be lumped together (see Sutton & Biblarz,
*Rocket Propulsion Elements*, §3 / §5):

    η_c*  lowers the delivered c*  → mass flow Pc·A_t/c* rises (more
          propellant is needed to hold the same chamber pressure when the
          combustion is incomplete);
    η_CF  lowers the delivered thrust coefficient (divergence, friction,
          two-phase, kinetic losses);
    η_Isp = η_c*·η_CF  is the net specific-impulse multiplier.

Back-compatibility: callers that supply only ``eta_Isp`` (the historical
single lumped multiplier) get ``eta_CF = eta_Isp`` and ``eta_cstar = 1.0``,
which reproduces the previous "all loss on Cf, ideal mass flow" behaviour
bit-for-bit.  Supply ``eta_cstar``/``eta_CF`` explicitly for the honest
split, in which case ``eta_Isp`` is overridden with their product.

Derived quantities (computed at import time)
--------------------------------------------
R_gas   : specific gas constant  R_universal / Mw   [J/(kg·K)]
c_star  : ideal characteristic velocity  [m/s]
~~~~


## S04 - Throat sizing, ideal/delivered performance and off-design screens

*5 files.*

### `raosim/atmosphere.py`
*79 lines · stage S04 Throat sizing, ideal/delivered performance and off-design screens*

~~~~text
atmosphere.py – Simplified ISA (International Standard Atmosphere) model.

Provides temperature, pressure, and density as a function of geometric
altitude up to ~86 km.  Uses the standard 7-layer piecewise-linear
temperature model for the first three key layers.
~~~~

### `raosim/engine.py`
*240 lines · stage S04 Throat sizing, ideal/delivered performance and off-design screens*

~~~~text
engine.py – Rocket engine performance computed from nozzle + propellant.

Combines gas-dynamics relations with propellant thermochemistry to produce
thrust, Isp, mass-flow rate, and related performance metrics.
~~~~

### `raosim/gas_dynamics.py`
*270 lines · stage S04 Throat sizing, ideal/delivered performance and off-design screens*

~~~~text
gas_dynamics.py – Quasi-1-D isentropic compressible-flow relations.

All functions assume calorically perfect ideal gas with constant γ.
References: Anderson, *Modern Compressible Flow*; the provided Rao-nozzle
formulation document (isentropic relations, area-Mach, Prandtl-Meyer).
~~~~

### `raosim/separation.py`
*311 lines · stage S04 Throat sizing, ideal/delivered performance and off-design screens (from module_stage_map)*
~~~~text
separation.py – Flow separation prediction for overexpanded nozzles.

Implements four empirical separation criteria for full flow separation (FSS),
with forms and constants taken from the criteria survey in Östlund (2002)
(`propulsion_texts/fulltext01.pdf`, PDF p. 51–52; md mirror
`propulsion_texts_for_agents/markdown/fulltext01.md`) and the design practice
of NASA SP-8120:

  • Summerfield (1940s test base, conical nozzles) — Östlund Eq. 28:
        p_sep / Pa ≈ 0.4
  • Schilling (1962, conical + truncated-ideal test base) — Östlund Eq. 29:
        p_sep / Pa = k1 · (Pc/Pa)^k2
        k1 = 0.582, k2 = −0.195   (contoured nozzles)
        k1 = 0.541, k2 = −0.136   (conical nozzles)
  • Kalt–Badal (1965) — Schilling's form refit, Östlund p. 52:
        p_sep / Pa = (2/3) · (Pc/Pa)^(−0.2)
  • Schmucker (1970s, fully turbulent BL) — Östlund Eq. 30, "still widely
    used"; local-Mach form, evaluated at each wall station:
        p_sep / Pa = (1.88 · M_i − 1)^(−0.64)

Design margin: NASA SP-8120 (`19770009165.pdf`) recommends reducing the
expansion ratio when the exit wall pressure is within 20 percent of the
separation pressure; ``check_separation`` therefore reports a design-margin
verdict at ``Pe ≥ design_margin · p_sep`` (default 1.2) alongside the raw
separated/attached prediction.

History note (2026-07-22): before this date the module carried the Schmucker
and Kalt–Badal *labels* on interchanged/altered forms ("Schmucker" =
(Pa/Pc)^0.8 / M, "Kalt–Badal" = 1/(1.88·M − 1)).  Both were off their
literature namesakes (~1.75× at Me = 3); see
docs/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md §A.2.1.  The JAX mirror
(`raosim/jax/thermal.py:schmucker_separation_margin`) is kept in lock-step —
change both together (parity test: tests/test_jax_thermal_design_opt.py).

References
----------
- J. Östlund, "Flow Processes in Rocket Engine Nozzles with Focus on Flow
  Separation and Side-Loads", KTH thesis (2002).  [in propulsion_texts]
- NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976).  [in propulsion_texts]
- R. Stark, "Flow Separation in Rocket Nozzles – An Overview" (2009).
  [secondary; not in the local corpus]
~~~~

### `raosim/wall_pressure.py`
*173 lines · stage S04 Throat sizing, ideal/delivered performance and off-design screens (from module_stage_map)*
~~~~text
wall_pressure.py – Estimate wall pressure distribution p(x) along the nozzle.

Uses the local area ratio A(x) = π·y(x)² to find M(x) via the area-Mach
relation, then applies the isentropic pressure relation.

Includes a monotonicity check per NASA SP-8120: wall pressure should
decrease continuously toward the exit.  A positive dp/dx downstream of the
throat signals separation risk.
~~~~


## S05 - Default Bezier/TOP nozzle contour, throat and chamber geometry

*5 files.*

### `raosim/chamber_geometry.py`
*481 lines · stage S05 Default Bezier/TOP nozzle contour, throat and chamber geometry*

~~~~text
Combustion-chamber, convergent, and full thrust-chamber geometry.
~~~~

### `raosim/nozzle_geometry.py`
*403 lines · stage S05 Default Bezier/TOP nozzle contour, throat and chamber geometry*

~~~~text
nozzle_geometry.py – Rao Thrust-Optimized Parabolic (TOP) bell nozzle contour.

Generates the three-section contour:
  1. Upstream circular arc  (convergent side, R_u = 1.5 Rₜ)
  2. Downstream circular arc (supersonic side, R_d = 0.382 Rₜ)
  3. Quadratic Bézier bell  (canted parabola from inflection N to exit E)

References:
  - NASA SP-125, Figure 4-14 / Eq.4-7 / worked example4-2 (PDF pp100,107)
  - NASA SP-8120, Figure5 and section2.1.2.1.2 (PDF p29), conceptual cross-check
  - The originating Rao1960 chart paper is not held; no inspection claimed.
~~~~

### `raosim/rao_top_chart.py`
*16 lines · stage S05 Default Bezier/TOP nozzle contour, throat and chamber geometry*

~~~~text
Rao/TOP angles measured from NASA SP-125 Figure 4-14 (PDF p. 100).

These are reviewer digitizations, not source-printed numerical tables.  The
single printed worked point on PDF p. 107 is retained exactly.  Historical raw
measurements and their scripts live in the provenance artifacts named below.
~~~~

### `raosim/throat_geometry.py`
*134 lines · stage S05 Default Bezier/TOP nozzle contour, throat and chamber geometry (from module_stage_map)*
~~~~text
Shared geometric contract for the chamber/nozzle throat region.
~~~~

### `raosim/validation.py`
*399 lines · stage S05 Default Bezier/TOP nozzle contour, throat and chamber geometry (from module_stage_map)*
~~~~text
validation.py - Design-status labels and engineering gate checks.

These checks are intentionally conservative. Passing them means the generated
nozzle is suitable for downstream design review, not qualified for hardware.
~~~~


## S06 - Opt-in Rao variational / method-of-characteristics BVP contour

*10 files.*

### `raosim/moc.py`
*890 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
moc.py – Axisymmetric Method of Characteristics unit processes and the
forward flow-field analysis entry point.

Layer 1 of the Rao nozzle design.  This module owns the three Anderson-form
unit processes (interior / axis / wall point) and the transonic start-line
helpers; the bounded characteristic net itself is marched by
:func:`raosim.moc_bounded.march_bounded_net`, which
:func:`solve_flowfield` drives.

History (see ``docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md``):
the original :func:`march_coupled_net` marched zig-zag "fronts" in which
the computed axis and wall points were never parents of any later point,
so the net was only the shrinking Cauchy triangle of its starting line —
it never reached the exit and its exit plane was a two-point
interpolation.  It is retained, deprecated, for the audit trail only.

Three primitive solvers:
  1. solve_interior_point: two adjacent parents → intersection
  2. solve_axis_point:     symmetry BC (θ=0, r=0)
  3. solve_wall_point:     wall tangency BC (θ = wall angle)

Axisymmetric compatibility equations, with nodes ordered downstream:
  Along C+ (slope θ+μ): d(θ − ν) = −S ds
  Along C- (slope θ−μ): d(θ + ν) = +S ds
  S = sin(θ) sin(μ) / r
  (S=0 for planar flow)
  (Anderson MCF 3rd ed. eqs 11.52/11.53; oracle-validated 2026-06-11
  against the NASA M3.5Perf net, see tests/test_characteristic_pairing.py)

References:
  - Anderson, Modern Compressible Flow, 3rd ed., Ch. 11
  - Rice, T., NASA/JHU 3-D nozzle design code manual (20030067852)
  - NASA SP-8120, Liquid Rocket Engine Nozzles

Starting lines:
  - :func:`solve_flowfield` starts from the axis-to-wall Kliegel–Levine
    data line of :func:`raosim.nasa_moc._make_throat_initial_line`
    (default ``'kliegel_levine'``; ``'nasa_visible_kliegel_levine'`` keeps
    the NASA C++ expression for oracle parity).
  - :func:`approximate_starting_line` samples the DOWNSTREAM THROAT ARC
    (the wall T→N), not a data line.  It is kept for the transonic-kernel
    tests and the deprecated march; do not use it as MOC initial data.
~~~~

### `raosim/moc_bounded.py`
*763 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
moc_bounded.py – Bounded axisymmetric MOC net for a given nozzle wall.

Forward *analysis* march of the characteristic net between the axis and a
prescribed wall (throat arc + bell), built characteristic-by-characteristic
on the NASA/JHU topology so that boundary reflections propagate back into
the interior.  This replaces the shrinking front-marching scheme of
``raosim.moc.march_coupled_net`` (see
``docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md``), whose axis and
wall points were never parents of any later point.

Topology (Rice/NASA manual 20030067852 PDF 7-9, 17; NASA C++
``MOC_GridCalc_BDE.cpp`` ``CalcRRCsAlongArc`` 1030-1116,
``CalcInteriorMeshPoints`` 2466-2534, ``CalcAxialMeshPoint`` 2262-2306;
Anderson MCF 3rd ed. §11.5):

* the net is a stack of right-running characteristics (RRC, C−, slope
  ``tan(θ−μ)``), each stored **wall-first** (index 0 = wall point,
  index −1 = axis point);
* RRC ``j`` is built wall → axis: the wall point from the C+ leaving
  ``[1][j−1]`` (tangency BC ``θ = θ_wall``), interior ``[i][j]`` from the C+
  of ``[i+1][j−1]`` (``[i][j−1]`` after a special-point insertion) and the
  C− of ``[i−1][j]`` — so the wall point **is** the C− parent of the first
  interior point and the previous axis point is a C+ parent — and the
  axis point closes the row by symmetry;
* the point count per RRC is ``iLast[j] = iLast[j−1] + specialFlag`` —
  it never shrinks except by negative-radius truncation at the axis.

On the throat arc the NASA arc unit processes are reused verbatim
(``nasa_moc._calc_arc_wall_point_with_special``), so rows up to the arc
end are identical to :func:`raosim.nasa_moc.build_kernel`.  Beyond the
arc a general wall point (same dθ-form algebra, wall geometry from the
wall object) is used; wall turning larger than ``dtheta_limit`` inserts a
NASA-style special point (``_calc_special_wall_point_at``).  Invalid cells
are never geometrically patched: a cell whose intersection lands upstream
of its parents, or a subsonic/non-convergent state, triggers one
refinement (special half-step) and otherwise stops the march with an
explicit reason (Young 2012 §3.3 eq. 3.10 rejects crossed solutions;
Rice manual PDF 21 prescribes refining DTHETAB).

The exit plane is obtained the way NASA's ``CropNozzleToLength``
(C++ 1341-1429) does it: each RRC is linearly interpolated to ``x = x_E``
and the resulting table is integrated across the plane.
~~~~

### `raosim/moc_topology.py`
*305 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
Explicit Rao construction topology (REWRITE_PLAN §11.7, Phase 12.6).

This module formalises the NASA/Rao contour construction as one explicit
object.  The field set follows the REWRITE_PLAN §11.7 spec exactly:

    TT_prime        initial data line (transonic start line)
    B               end-of-expansion wall point (throat arc exit)
    BF              final RRC of the kernel, wall -> axis
    D               selected point on BF (control-surface attachment)
    BD              wall-to-D portion of BF (the mass-flow curve)
    DE              control surface (left-running characteristic D -> E)
    E               nozzle exit point
    streamline_BE   wall contour from B to E -- THE bell wall
    theta_B         converged end-of-arc wall angle
    mass_BD / mass_DE   closure pair (equal at convergence)

``streamline_BE`` is NASA's ``CalcWallContour`` streamline traced from B
through the BFE mesh to E (Rice, *Three-Dimensional Nozzle Design Code*,
JHU/APL 2003, §3.4 — ported as :func:`raosim.nasa_moc.calc_bde_region`).
The full nozzle wall is the throat-arc wall (theta = 0 ... theta_B) +
``streamline_BE``; :meth:`RaoTopology.full_wall` assembles it.  Once the
solver default flips to the characteristic formulation, this field — not
``_construct_wall_from_ce`` — is the wall (see the DIRECTION block in
JAX_DIFFERENTIABLE_PLAN.md; the legacy constructor is deleted at that
point, not before).

Nodes are :class:`raosim.moc.CharPoint` (theta, M, nu, mu, compat±) so
plots and diagnostics can consume the topology directly.  The lighter
:class:`raosim.nasa_moc.RaoTopology` (FlowNode-based, produced by the
construction routines) remains the internal carrier; :func:`build_topology`
lifts it into the full-form object.  When ``raosim/jax`` absorbs the core
(DIRECTION item 1), this module is the intended public boundary.
~~~~

### `raosim/nasa_moc.py`
*3589 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
NASA/JHU MOC_Grid_BDE port for Rao optimum-thrust topology.

This module is a stagnation-normalised Python port of the routines in
``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp``
that produce the Rao B-D-E topology:

* ``build_kernel`` -- TT' starting line + RRCs along the throat
  expansion arc up to wall angle ``theta_B`` (mirrors
  ``CalcInitialThroatLine`` + ``CalcRRCsAlongArc``).
* ``calc_massflow_along_rrc`` / ``calc_mdot_bd`` -- axisymmetric
  annular trapezoidal mass-flow integration (``CalcMassFlowAndThrustAlongMesh``
  / ``CalcMdotBD``).
* ``nasa_deriv`` / ``nasa_runge_kutta`` -- C++ ``Deriv`` + ``RungeKutta``
  derivative system in radius, used for forward integration of the
  left-running characteristic from point D.
* ``find_point_e`` -- ``FindPointE`` analogue: integrate the C+
  characteristic from D, accumulate annular mass flow, and secant on
  the final radial step so the cumulative mass equals ``mass_BD``.
* ``calc_lrc_de`` -- ``CalcLRCDE`` analogue: outer secant on the axial
  D-location, picking D so the Rao stationarity residual
  ``theta_E - theta_calc(p_E, rho_E, M_E)`` vanishes.
* ``set_theta_b`` -- ``SetThetaB`` analogue: outer secant on the
  initial wall expansion angle.

Units are stagnation-normalised throughout: lengths in metres, M
dimensionless, ``rho/rho0`` and ``V/a0`` for the mass-flux integrand,
``p/p0`` for the thrust integrand. Gas-dimensional factors (``GRAV``,
``144``, ``GASCON/molWt``) cancel from the residual definitions so we
drop them.
~~~~

### `raosim/rao_existence_scan.py`
*2926 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
Solver-independent D-attachment existence scans for Rao B-D-E topology.

This module deliberately stays out of the JAX/Optimistix/least-squares stack.
It samples candidate ``(theta_B, kernel_d_fraction)`` pairs, builds the
NASA-style kernel with the existing pure-Python/NumPy MOC port, starts the
downstream DE characteristic from point D, and records exit residual fields.

The three supported D-attachment closures are:

``smooth``
    Position, flow angle, and Mach number are inherited continuously from the
    kernel point D.
``position``
    Position is inherited from D, while the post-D flow angle and Mach number
    are selected by an explicit local grid scan.  The stored residuals are the
    minimum-norm result over that grid.
``fan``
    Position is inherited from D, while the post-D state is connected to the
    pre-D state by a centered Prandtl-Meyer fan,
    ``nu_post - nu_pre = theta_post - theta_pre``.  The stored residuals are
    the minimum-norm result over the requested fan-turn grid.

The B5 public artifact is not one of those internal ``theta_B x kdf`` slices.
``scan_convergence_domain`` sweeps the external ``epsilon x length_pct`` design
box, uses each internal scan only as a root seed, then records the refined
``theta_B`` and ``kdf`` as evidence outputs.  Its typed, hash-pinned domain
classifies residual convergence, kernel refinement, and Rao Eq. (5) validity
separately.  Numerical failure remains unknown rather than being labelled
physical non-existence, and lookups never extrapolate beyond the declared box.
~~~~

### `raosim/rao_optimizer.py`
*433 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
rao_optimizer.py – direct wall-shape optimization on the forward MOC net.

Layer 2: finds the wall contour that maximizes thrust, subject to
geometric and regularity constraints, using the forward bounded MOC
analysis of each candidate wall (:func:`raosim.moc.solve_flowfield` →
:func:`raosim.moc_bounded.march_bounded_net`).

Uses SplineWall candidates (monotone cubic Hermite through control radii).
θ_n is a design variable seeded from NASA SP-8120 lookup tables.
No scipy dependency (SLSQP when available, Nelder–Mead otherwise).

STATUS (2026-09-21): the forward net is now a genuine bounded
characteristic net (NASA/JHU RRC topology, oracle-matched on the M3.5
perfect nozzle: exit M 3.5009 ± 0.0017, stream thrust within 0.02 % of
NASA's ``summary.out``).  A candidate wall whose compression waves
coalesce (an internal shock — typical of parabolic/Bézier bells, see
Hagemann et al. 1998, Frey & Hagemann 2000) stops the march with
``stop_reason == "rrc_coalescence"``; such a candidate is penalised
by how far the wall was reached rather than scored on a fabricated
exit plane.  This optimizer is not reachable from any CLI/MDO path and
its optimum has not been benchmarked against Rao's contour — treat it
as experimental (``experimental_moc_geometry``).

References:
  - Rao, G.V.R., "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA SP-8120, "Liquid Rocket Engine Nozzles," 1976

Starting line: ``starting_line_method`` is passed to
:func:`raosim.moc.solve_flowfield` (default ``'kliegel_levine'`` — the
axis-to-wall Kliegel–Levine data line; ``'nasa_visible_kliegel_levine'``
keeps the NASA C++ expression).  The old ``'area_ratio'`` /
``'sauer_modified'`` options are refused by ``solve_flowfield`` because
neither is a transonic data line.
~~~~

### `raosim/rao_residuals.py`
*160 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
Residual primitives for the Rao variational/MOC coupling.
~~~~

### `raosim/rao_variational.py`
*7083 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour*

~~~~text
rao_variational.py – experimental Rao-style variational nozzle prototype.

This module is an exploratory implementation of the Rao control-surface idea.
It follows the Rao/NASA functional form for thrust, mass flow, and length on
the supersonic control surface, including the design ambient-pressure term.
The public API remains tagged as experimental until published benchmarks pass.

The implementation currently:
  1. Defines a discretized control surface CE in the supersonic region.
  2. Evaluates the Rao thrust, mass-flow, and length integrands.
  3. Uses a constrained direct-method optimizer to maximize thrust subject to
     fixed mass flow and fixed nozzle length.
  4. Attempts a CE-driven MOC wall construction.

Known limitations:
  - The nonlinear residual solve is a finite-dimensional direct method; the
    strict Rao-1958 Nozzle-B benchmark validates its published shape/integral
    envelope, not every internal characteristic state.
  - A result is promoted only when its per-run BVP and MOC topology gates pass;
    otherwise the returned wall remains a labelled geometric approximation.

References:
  - G. V. R. Rao, "Exhaust Nozzle Contour for Optimum Thrust," 1958
  - NASA TM (1990), Rao method re-derivation with explicit functionals
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
~~~~

### `raosim/transonic_kernel.py`
*303 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour (from module_stage_map)*
~~~~text
Kliegel-Levine 1969 transonic kernel for the MOC starting line.

Kliegel, J.R. and Levine, J.N., *Transonic Flow in Small Throat Radius of
Curvature Nozzles*, AIAA J. 7(7) 1375-1378, 1969.  Open access:
<https://arc.aiaa.org/doi/pdf/10.2514/3.5355>.  Also reproduced in
Zucrow & Hoffman, *Gas Dynamics* Vol. 2 Ch. 16, and Östlund 2002 §3.

The closed-form 3rd-order toroidal-coordinate expansion is valid down
to Rc/Rt ~ 0.5, well below the leading-order Hall expansion's ~2.0
floor.  This module's coefficients are a direct port of NASA/JHU
``MOC_GridCalc_BDE.cpp::KLThroat`` (lines 3103-3178) with two typo
corrections in the AXI ``v[3]`` polynomial (NASA line 3133) and the
analogous typo in the TWOD ``v[3]`` polynomial (NASA line 3157).

Typo corrections
================
NASA AXI line 3133 reads::

    z*((556*G*G + 1737*G + 3069)*y*y*y*y*y/1728
       * (388*G*G + 1161*G + 1181)*y*y/576
       + (304*G*G + 831*G + 1242)*y/864) + ...

There are three problems with the middle sub-term:

* The ``*`` between the y^5 term and the y^2 term must be ``+``: with
  ``*``, the dimensional power becomes y^7 which is inconsistent with
  a third-order radial-expansion coefficient and breaks the series
  recurrence.
* The constant ``1181`` is a typo for ``1881`` (cross-check Kliegel-
  Levine 1969 Table 1; the same coefficient ``1881`` appears in the
  ``u[3]`` polynomial at NASA line 3127 as ``(388*G*G + 1161*G + 1881)``,
  confirming that v[3] should match).
* The exponent ``y*y`` (= y^2) is a typo for ``y*y*y`` (= y^3): the
  2D branch's analogous term at line 3157 uses ``y*y*y``, and the
  y^5 → y^3 → y power progression matches the descending parity
  expected of a third-order coefficient.

NASA AXI line 3129 reads::

    z*((52*G*G + 51*G + 327)*y*y*y*y/34
       - (52*G*G + 75*G + 279)*y*y/192
       + (92*G*G + 180*G + 639)/1152)

The divisor ``/34`` on the y^4 z-coupling term in ``u[3]`` is off by
~11× — every other divisor in the same polynomial is /24, /144, /192,
/1152, /82944 (powers of 2 × 3).  The correct divisor is ``/384``
(2 × /192, the natural recurrence factor for the y^4 term immediately
above the /192 y^2 term).  Cross-check Kliegel-Levine 1969 Table 1.
This implementation uses the corrected ``/384.0``.

The 2D analogue (NASA line 3157) has the same ``*`` → ``+`` typo in
the middle of the z-bracket; the constant ``1665`` is correct.

This module uses the corrected forms.  Do **not** "fix" the comments
back to literal NASA source — the original lines are mathematically
wrong; see ``test_transonic_kernel.py`` for the validation gate.
~~~~

### `raosim/wall_model.py`
*173 lines · stage S06 Opt-in Rao variational / method-of-characteristics BVP contour (from module_stage_map)*
~~~~text
wall_model.py – Spline-based wall geometry for MOC nozzle design.

Provides a SplineWall class that the MOC solver queries for:
  - r(x):            wall radius at axial position x
  - theta(x):        wall angle (atan(dr/dx)) at x
  - intersect_char(): where an incoming characteristic hits the wall
~~~~


## S07 - Regenerative wall sizing, cooling analysis and wall structural screens

*6 files.*

### `raosim/coolants.py`
*24 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens*

~~~~text
Canonical coolant identities shared by every thermofluid path.
~~~~

### `raosim/materials.py`
*374 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens*

~~~~text
materials.py — thrust-chamber / nozzle construction materials catalog.

A regeneratively cooled wall is a *two-fluid heat exchanger that also
carries pressure and thermal stress*, so the metal it is built from sets
the whole problem, not just one number.  The wall properties enter every
physical concept the design tool screens:

* **Channel sizing (thermal).**  The gas-side wall temperature is driven
  by the conduction term ``t_w / k_w`` in the series resistance circuit
  (NASA SP-125, Huzel & Huang, eqs. 4-19 and 4-22; printed p. 104):

      q = (k/t)(T_wg - T_wc),   H = 1 / (1/h_gc + t/k + 1/h_c)

  A high-conductivity copper liner runs a *cool* gas-side wall at a given
  flux; a low-k superalloy runs a much hotter one.  This is why the
  channel auto-sizer (:func:`raosim.thermal_design.size_cooling_channels`)
  takes ``wall_k`` — and why it must come from the chosen material, not a
  disconnected flag.

* **Service temperature limit.**  SP-125 (printed p. 105) states that for
  the metals "commonly used in thrust chamber walls, such as stainless
  steel, nickel, and Inconel, the limiting hot-gas-side wall temperature
  is around 1500deg-1800deg F" (1089-1255 K).  Copper alloys conduct far
  better but soften far earlier, so their service limit is set by creep /
  strength, not melting.  The cooling *margin* is (limit / peak wall T),
  so the limit is a per-material number.

* **Structural / thermal stress.**  SP-125's coaxial-shell wall design
  (eq. 4-31, printed p. 109) sizes the inner liner against the combined
  pressure-differential and thermal stress

      S_c = (p_co - p_g) R / t  +  E a q t / (2 (1 - v) k)

  which needs the elastic modulus ``E``, thermal-expansion coefficient
  ``a``, Poisson ratio ``v`` and conductivity ``k`` of the wall — see
  :func:`raosim.physics.coaxial_shell_wall_stress`.  The longitudinal
  thermal stress S_l = E a dT (eq. 4-28) and the outer-jacket hoop stress
  use the same properties.

Provenance of the numbers
-------------------------
The *relations* above are all grounded in the local literature
(``propulsion_texts/19710019929.pdf`` = NASA SP-125).  SP-125 also names
the classic wall materials (stainless steel, nickel, Inconel; sample
calculation 4-4 picks Inconel X for an RP-1-cooled tube) and gives the
1500deg-1800deg F class limit, so the superalloy/steel entries are grounded
directly.  SP-125 predates the modern copper combustion-chamber alloys
(NARloy-Z, GRCop-84, CuCrZr), so their property values are representative
handbook / NASA-Glenn values from each alloy's own literature and are
flagged as such in ``source``; they are screening-grade, not a heat of
certified material.  All values are single representative points near the
service temperature — temperature-dependent k(T), S_y(T) tables are a
documented future extension (see JAX_DIFFERENTIABLE_PLAN.md).

Units are SI throughout: conductivity W/(m.K), strength/modulus Pa,
temperature K, density kg/m^3, expansion 1/K, heat flux W/m^2.
~~~~

### `raosim/physics.py`
*2357 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens*

~~~~text
physics.py - Screening physics for validated preliminary nozzle design.

The models in this module are intentionally conservative early-design checks.
They are suitable for design review triage, not hardware qualification.
~~~~

### `raosim/regen_profile.py`

*430 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens*

~~~~text
regen_profile.py — station-wise regenerative wall + channel geometry.

A real regenerative wall is not one thickness.  Following NASA SP-125
(Huzel & Huang, coaxial-shell / tube-and-jacket thrust-chamber design,
``propulsion_texts/19710019929.pdf`` printed pp. 108-110), the wall has
several distinct dimensions the thermal and structural design size
*separately*, and they vary along the contour:

* ``t_hot(x)``    — hot-gas liner (inner-shell) thickness
* ``channel_height(x)`` — coolant channel depth ``h``
* ``channel_width(x)``  — coolant channel width ``w``
* ``land_width(x)`` — rib / land width ``b`` between channels
* ``t_jacket(x)``  — outer closeout (jacket) thickness

SP-125 also notes the passage area should *vary along the chamber to hold
an appropriate coolant velocity* and that the **throat is the critical
region** (sample calc 4-4).  ``RegenWallProfile`` carries these as
per-station arrays over the contour and derives what the cooling and
pressure-drop models need.

Helical channels
----------------
When the channels are wound helically (``helix_turns > 0``) the coolant
travels farther than the axial run, which raises the Darcy-Weisbach
pressure drop ``Δp = f (L/D_h)(ρV²/2g)`` (SP-125 eq. 4-32) in proportion
to the path length ``L``.  The helix here matches
:mod:`raosim.regen_geometry`: a channel's angle is
``θ(x) = θ0 + 2π·turns·(x−x0)/L_axial`` and its centerline sits at
``r_mid = r_inner + t_hot + h/2``, so the per-station path element is

    dl = sqrt(ds_meridian² + (r_mid · dθ)²),   dθ = 2π·turns·dx/L_axial

— exactly the arc the STL coils trace, so the modeled Δp is consistent
with the exported geometry (previously the helix changed only the STL).
Channel width and land width are defined transverse to this path; their
constant-axial circumferential footprint is multiplied by ``dl/ds``.

All lengths SI [m].
~~~~

### `raosim/thermal_design.py`
*1460 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens (from module_stage_map)*
~~~~text
thermal_design.py — cooling-coupled contour selection.

The screening workflow in :mod:`raosim.design` fixes the contour first
(chart Bézier) and evaluates heat flux / cooling AFTER, as pass/fail
gates.  This module inverts that: it computes the **cooling physics
before the contour is fixed** and lets it shape the geometry, so the
returned contour is one the thermal design can actually survive.

Two physical levers, co-designed against the cooling margin (peak
gas-side wall temperature vs the channel-wall limit):

* **Throat curvature** ``Rd_factor`` — opening the throat radius of
  curvature lowers the Bartz peak flux through the ``(D*/r_c)^0.1``
  term.  This is a *weak* lever (~20 % flux relief from 0.382→3.0 Rt):
  geometry alone cannot cool a throat, which is exactly why engines
  rely on active cooling.  It is applied first because it costs nothing
  hydraulically.
* **Channel mass flux** ``G`` — shrinking the channel cross-section at
  fixed coolant flow raises the coolant velocity and the Sieder-Tate
  ``h_c ∝ G^0.8``.  This is the *strong* lever and carries the rest.

The result reports the selected contour, the converged cooling state,
the binding lever, and the full search history.  This is the NumPy
feedback precursor to the rigorous differentiable constrained design
(the J-series gradient loop): same coupling direction (physics →
geometry), heuristic search instead of exact gradients.
~~~~

### `raosim/thermofluids.py`
*569 lines · stage S07 Regenerative wall sizing, cooling analysis and wall structural screens (from module_stage_map)*
~~~~text
Higher-fidelity coolant-network, real-fluid, radiation, and boiling tools.

These models deliberately expose their validity and data source.  They extend
the axisymmetric preliminary regen solver; they do not turn it into a
qualification CFD/CHT analysis.
~~~~


## S08 - Pintle injector sizing and optional spray/c* fixed point

*3 files.*

### `raosim/injector.py`
*4355 lines · stage S08 Pintle injector sizing and optional spray/c* fixed point*

~~~~text
injector.py - Parameterized pintle-injector hydraulic sizing.

The injector is sized from the operating point the nozzle + chamber solver
already produces, never from an independent injector thrust input:

    F, Pc, Pa, eps, O/F  ->  Cf, At, mdot  ->  mdot_f, mdot_o
                          ->  A_f, A_o, dp_f, dp_o  ->  pintle geometry + spray

Core incompressible orifice hydraulics (Sutton & Biblarz, *Rocket Propulsion
Elements*; NASA SP-8089 *Liquid Rocket Engine Injectors*):

    mdot = Cd * A * sqrt(2 * rho * dp)            (per stream)
    v    = mdot / (rho * A)                        (mean passage velocity)
    TMR  = (mdot_r * v_r) / (mdot_a * v_a)         (radial / axial momentum)
    Re   = rho * v * D_h / mu
    We   = rho * v**2 * D_h / sigma
    Oh   = mu / sqrt(rho * sigma * D_h)            ( = sqrt(We) / Re )

Pintle passage geometry:
    axial annulus      A_a = (pi/4) (Do^2 - Di^2)  ~ pi * D_p * h
    radial rect slots  A_r = N_slots * w * h_slot

The hydraulic implementation supports automatically-sized or fixed rectangular
slots/round holes plus the Son et al. (2017) continuous movable radial gap.
The Son branch holds the axial annulus fixed, resolves physical centre-rod
travel below the centre-gap transition, and synthesizes the separate upstream
annulus-controller schedule required for throttling.  Configuration-specific
Cd calibration and VOF/measured sheet evidence are provenance- and
operating-domain-bound.  Single-phase gas streams use an explicit compressible
orifice branch; the movable radial-sheet branch itself requires liquid.
Two-phase states and unsupported radial exit geometries are rejected.

NASA SP-8089 identifies momentum balance, annulus/slot geometry, and the
deflector angle as the governing pintle variables, while warning that spray
distributions generally require cold-flow testing — so every spray/atomization
number below is a clearly-labeled screening surrogate, not a validated result.
~~~~

### `raosim/movable_pintle.py`
*686 lines · stage S08 Pintle injector sizing and optional spray/c* fixed point*

~~~~text
Literature-routed geometry and static actuation for a movable pintle.

This module implements the *continuous radial gap* geometry in Son et al.,
``Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines``
(JPP 33(4), 2017, DOI 10.2514/1.B36301).  It is intentionally separate from
the repository's fixed radial slots and holes.

The controlling tip/post area is Son Eq. (1), evaluated in its numerically
stable expanded form.  It is capped by the fixed centre-gap annulus.  Once the
tip area reaches that cap, pintle travel no longer controls the minimum flow
area; the paper reports an abrupt flow/spray transition there.  Design solves
therefore stay below a declared fraction of the transition area.

No universal discharge-coefficient-versus-stroke or actuator-force law is
invented.  A user calibration may be interpolated versus opening fraction only
when its artifact hash, exact Son-geometry fingerprint, fluid, and operating
domain match; otherwise the caller's constant Cd is retained and explicitly
labelled uncalibrated.  Static actuation uses only user-declared unbalanced
pressure area, spring, seal friction, moving mass/acceleration, stem area, and
capacity. Dynamic response, seal life, thermal growth, and qualification are
outside this preliminary model.
~~~~

### `raosim/spray_coupling.py`
*320 lines · stage S08 Pintle injector sizing and optional spray/c* fixed point (from module_stage_map)*
~~~~text
Opt-in fixed-point coupling between injector vaporization and cycle mass flow.

The default design path remains independent of the screening atomization
correlation.  When a user supplies explicit mixing and chemical-completion
efficiencies, this module can close the otherwise one-way loop

``eta_cstar -> mdot -> injector geometry -> eta_vaporization -> eta_cstar``.

It does not infer mixing or combustion efficiency and it refuses to use an
inapplicable gas/transcritical droplet result.
~~~~


## S09 - Feed-pressure ledger to electric pump, motor, inverter and battery sizing

*2 files.*

### `raosim/electric_technology.py`
*496 lines · stage S09 Feed-pressure ledger to electric pump, motor, inverter and battery sizing*

~~~~text
raosim.electric_technology — one electric-drive technology source for both
pipelines.

Before this module the traditional sizing (``raosim.pumps.SCREENING_DEFAULTS``)
and the differentiable MDO (``raosim.mdo.schema.MissionSpec``) carried two
different, unsourced electric-drive sets — battery power density 3.0 vs
1.3 kW/kg, battery energy density 250 vs ≈180.6 Wh/kg, motor 2.5 vs 3.0 kW/kg,
inverter 15 vs 20 kW/kg (MDO issues audit 2026-10-16, §1.3 item 19).  About
94 % of the MDO objective is that electric package, so an MDO optimum and the
parts sized from it described two different technologies.

Each :class:`ElectricTechnologySet` here is one complete, internally
consistent set, and both pipelines read the same set:

* ``raosim.pumps.SCREENING_DEFAULTS`` takes its electric entries from
  :data:`DEFAULT_ELECTRIC_TECHNOLOGY`;
* ``MissionSpec.electric_technology`` names a set, its numeric electric fields
  default to that set, and ``MissionSpec`` refuses values that disagree with
  the declared set unless the set is :data:`CUSTOM_ELECTRIC_TECHNOLOGY_ID`;
* the MDO→traditional handoff (``raosim.mdo.postprocess._pump_spec``) passes
  the mission's set, so the re-evaluated pump package uses the numbers the
  optimizer used.

Power-density conventions (all sets, Lee et al. 2021, DOI
10.1007/s42405-020-00325-z, PDF p. 7 Eqs. (26)-(29) and PDF p. 8 Eqs.
(30)-(32), checked on the page):

* motor density on motor SHAFT OUTPUT power, ``m_m = P_m,out / δ_m``;
* inverter density on inverter OUTPUT power, which equals the motor
  electrical input: ``m_inv = P_m,out / (δ_inv η_m)``;
* battery power branch on battery output = inverter input,
  ``m_b,P = κ_b P_inv,out / (δ_P,b η_inv)``;
* battery energy branch ``m_b,E = κ_b P_inv,out t_b / (δ_E,b η_E η_inv)``;
  ``m_b = max(m_b,P, m_b,E)``, the structural margin κ_b applying to both.

Two sets are registered:

``repository_screening_v1`` (default)
    The values the traditional ``raosim.pumps`` path has always used.  They
    are a repository screening assumption with no single source; the
    published brackets below place each against the held literature.  Two of
    them lie outside every held value (inverter density below, inverter
    efficiency above) and the constant discharge efficiency stands in for a
    burn-time correlation.  Recorded, not hidden.
``lee2021_ijass_electric_pump``
    Lee et al. 2021, Table 2 (PDF p. 11): every battery, motor and inverter
    value read on the page.  Table 2's pump efficiencies (66 %/61 %) and 10 %
    pump margin are not part of a technology set and are not applied.  Its
    battery energy efficiency is Eq. (38) evaluated at the paper's 600 s burn
    (PDF p. 10); it is not a constant at other burn times, and a mission with
    a different burn time raises :class:`ElectricTechnologyValidityWarning`.

The MDO-only ``MissionSpec.pump_specific_mass`` is deliberately NOT part of a
technology set: no held source states a pump mass per pump shaft power on the
basis the MDO uses (Kwak 2018 and Pavlov Rachov 2013 give pump-only densities
of 15-22 kW/kg; Casalino 2019 and Solda & Lentini 2008 give combined
drive+pump masses per electric power), and the traditional path sizes pump
mass from hardware geometry instead.  See ``MissionSpec`` for its flag.
~~~~

### `raosim/pumps.py`
*3976 lines · stage S09 Feed-pressure ledger to electric pump, motor, inverter and battery sizing*

~~~~text
Electric pump-feed sizing layered on the injector feed ledger.

The injector module already owns the feed-pressure balance.  This module
starts from that ``FeedSystemLedger`` and adds first-pass electric drive,
battery, centrifugal pump, inducer, diffuser/volute, and feasibility screens.
All geometry correlations here are preliminary sizing rules anchored to the
standard rocket turbopump design variables (head coefficient, suction specific
speed, specific speed/diameter), not hardware qualification.
~~~~


## S10 - Injector/chamber bolted interface and hardware mass ledger

*3 files.*

### `raosim/interface.py`
*1529 lines · stage S10 Injector/chamber bolted interface and hardware mass ledger*

~~~~text
Injector-to-chamber interface screening utilities.

This module keeps the injector body / chamber joint separate from the pintle
metering geometry.  It is a preliminary design ledger, not a bolted-joint
qualification model.  The implemented checks follow standard first-pass
mechanics:

* pressure separating force: F = Pc * A_projected
* chamber shell hoop screen: sigma = Pc * r / t (thin pressure-vessel form)
* injector faceplate bending screen: clamped circular plate under uniform
  pressure, sigma ~= 0.75 * Pc * a^2 / t^2 (Roark/classical plate theory)
* bolt clamp screen: total clamp load >= separation_factor * F, with per-bolt
  stress over an approximate tensile area (Shigley-style bolted-joint screen)
* edge distance / pitch screens: common preliminary machine-design heuristics
  (about 1.5 hole diameters to a free edge, about 3 diameters pitch)

NASA SP-125 section 9.3 (printed pp. 362-363) independently emphasizes that
flange bolts must be spaced closely enough to distribute load around the gasket
circumference.  It does not prescribe this module's minimum bolt count: the
three-bolt explicit minimum and even-count automatic policy are CAD/layout
invariants for a non-degenerate circumferential pattern.

These are deliberately conservative checks to keep the CLI honest about what it
knows.  Final hardware still needs gasket/seal design, preload scatter, thread
engagement, flange flexibility, thermal gradients, fatigue, and FEA/test data.
~~~~

### `raosim/mass_ledger.py`
*1507 lines · stage S10 Injector/chamber bolted interface and hardware mass ledger*

~~~~text
mass_ledger.py — configuration-controlled engine hardware mass ledger.

Why this module exists
----------------------
Until now the repository could size a thrust chamber, a pintle injector and an
electric feed system, but it could not say what any of that hardware *weighed*.
``raosim.mdo.state.MassState`` carried three permanently-``NaN`` slots
(``thrust_chamber_mass``, ``injector_mass``, ``total_dry_mass``) and the shared
host snapshot reported them as ``unavailable``.  The optimizer therefore
minimised an *electric-feed* package mass while calling it engine mass.

The governing modelling decision is that CAD-linked mass must come from the
same resolved geometry and, where possible, the same in-memory bodies handed to
export.  A profile/layout integral alone is a screening result until an exported
report carries the matching deterministic geometry ID.  Production injector
mass has a stricter built-part path that measures each OpenCascade body once;
the retained legacy closed forms are explicitly labelled ``screening_proxy``.

Physical basis
--------------
Every entry is a solid-of-revolution or prismatic volume times a catalog
density.  The shell relation here is a geometric Pappus calculation.  NASA
SP-125 uses the same surface-times-thickness relation for preliminary *tankage*
mass, e.g. the cylindrical tank shell

    W_c = 2 π a l_c t_c ρ                                   (SP-125 eq. 8-32)

(*Design of Liquid Propellant Rocket Engines*, Huzel & Huang, NASA SP-125,
1967; ``propulsion_texts/19710019929.pdf``, ch. VIII, printed p. 339).  That
equation corroborates the geometry; it is not a thrust-chamber mass model.  Here
``a`` is the *nominal* (mid-surface) radius, so a thin shell of thickness ``t``
standing off a gas-side radius ``r`` contributes ``2π (r + t/2) t`` per unit
meridional arc — Pappus's centroid theorem, which is what this module applies.
The pre-existing private helper ``raosim.thermal_design._wall_mass`` used the
gas-side radius rather than the mid-surface radius and summed
``hypot(gradient(x), gradient(y))``, which over-counts the path by one grid
interval (see the note in :func:`raosim.regen_profile._nodal_weights_from_segments`).
This module uses the correct centroid radius and the correct trapezoidal nodal
weights.

SP-125 gives one component-specific example of a non-ideal allowance in its
pressurant-storage-vessel estimate:

    W_v = π d² ρ_m (p d / 4 s) + 3 π d ρ_m (0.5 p d / 4 s)   (SP-125 eq. 5-16)

adds a 3-inch weld-land band at half wall thickness on top of two hemispherical
membranes (SP-125 ch. V, printed p. 173).  That equation does **not** prescribe a
thrust-chamber allowance.  This module therefore exposes only a generic,
user-controlled ``joint_allowance`` multiplier, defaulting to 1.0; callers must
substantiate any non-unit value from their actual weld/braze joint definition.

Structural context for the thicknesses being integrated is NASA SP-8087,
*Liquid Rocket Engine Fluid-Cooled Combustion Chambers* (1972; archived by
NTRS as 19730022965;
``propulsion_texts/19730022965.pdf``).  Its §2.1.3 identifies the three
structural jobs the metal in this ledger is doing — hoop support about the
combustion chamber, support at the throat against bending and buckling, and hoop
support about the expansion nozzle against collapse under overexpanded sea-level
operation — and §2.1.3 states the design factors of safety in use: yield 1.0 to
1.32, ultimate 1.3 to 1.8 (PDF pp. 38-39).  This module does not *size* against
those; it integrates thicknesses that the thermal/structural screens
(:mod:`raosim.thermal_design`, SP-125 eq. 4-31 via
:func:`raosim.physics.coaxial_shell_wall_stress`) already produced, and records
the closeout sizing basis in its provenance so the load path is auditable.

Honesty rules
-------------
* A component whose geometry or density is missing is emitted as an item with
  ``mass_kg = None`` and an ``unavailable_reason``.  It is never emitted as
  ``0.0``.
* :attr:`MassLedger.complete` is ``False`` whenever any item is unavailable, and
  :attr:`MassLedger.total_mass` is then ``None``.  A partial rollup is available
  separately as :attr:`MassLedger.resolved_mass` and is always labelled as such.
* ``status`` distinguishes ``geometry_resolved`` (integrated from a resolved CAD
  layout) from ``screening_sized`` (a documented first-order shape assumption,
  e.g. hex bolt head/nut envelopes).
~~~~

### `raosim/regen_volumes.py`

*317 lines · stage S10 Injector/chamber bolted interface and hardware mass ledger*

~~~~text
Shared numerical volume kernel for regenerative thrust-chamber metal.

The kernel deliberately does one job: integrate the three disjoint material
regions represented by :class:`raosim.regen_profile.RegenWallProfile`:

* the hot-gas liner;
* the channel lands/ribs between coolant passages; and
* the structural closeout/jacket.

NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, Van
Huff and Fairchild (1972), section 2.1.1.1, PDF page 24 (NASA SP-8087,
NTRS 19730022965), describes channel-wall coolant passages as integral with
the thin high-conductivity liner.  Section 2.1.3.1 separately describes the
outer reinforcement/jacket.  That literature establishes that all three
regions are real hardware; it does not prescribe this quadrature.

The equations below are geometric calculations (Pappus/shell volumes), not a
thrust-chamber mass correlation.  NASA SP-125 equation 8-32 uses the same
surface-times-thickness form for a cylindrical tank shell, which corroborates
the geometry but is not chamber-specific authority.
~~~~


## S11 - CAD, mesh and geometry deliverables

*9 files.*

### `raosim/engine_cad.py`
*314 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
engine_cad.py - engine-level STEP assembly (pump CAD plan Phase 3).

Aggregates already-exported and topology-gated STEP artifacts (regen wall,
jacket, machined pintle, pump packages, shared battery) into one named
``engine`` assembly for layout and trade review.  Nothing is re-derived
here: every child is imported from its own gated export, placed by a
documented layout transform, and the saved assembly is re-imported and
gated like the wall path.  The pump mounting flange is screened with the
existing bolted-interface resolver (:mod:`raosim.interface`), not designed.

Axes: engine axis +Z with the injector face plane at Z = 0 and the chamber
downstream at Z > 0 (the pintle STEP convention).  The wall/jacket STEPs
are revolved about +X by :mod:`raosim.export` and are rotated into +Z here.
Pump shaft axes stay parallel to the engine axis, packages placed behind
the injector face; the placements are layout placeholders, not routed
feed lines.
~~~~

### `raosim/export.py`
*875 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
export.py – CSV and STL export for nozzle contours.

STL supports two modes:
  - Inner surface only  (wall_thickness=None)  – for CAD/CAM reference
  - Solid wall mesh     (wall_thickness > 0)   – a topology/volume-validated
    triangle solid with outer wall, closed end rings, and optional flange.
    Manufacturing qualification remains outside this exporter.
~~~~

### `raosim/injector_cad.py`
*1521 lines · stage S11 CAD and geometry export*

~~~~text
injector_cad.py - named-body pintle STEP/STL export.

Builds a CadQuery assembly of the sized pintle with the named bodies the design
brief calls for (faceplate, hollow pintle body, tip, axial annulus, radial slot
network, fuel/oxidizer manifolds, optional movable sleeve, igniter interface,
regen-coolant outlet).  STEP is authoritative (it carries the named assembly);
per-body STLs are emitted for printing.  Geometry is the SIZED geometry from
:mod:`raosim.injector`; it is a preliminary CAD schematic, not a drawing-ready
part.

Axes: axial = +Z (injector face at Z=0, chamber downstream Z>0, manifolds /
faceplate behind at Z<0); radial in XY.
~~~~

### `raosim/injector_coaxial_cad.py`
*1209 lines · stage S11 CAD and geometry export*

~~~~text
injector_coaxial_cad.py — coaxial (TRW/Nardi) pintle injector CAD.

The machined-pintle export in :mod:`raosim.injector_cad` modelled the injector
as two concentric ring-manifolds cut into a faceplate plus radial transfer
drills.  That scheme cannot realise a pintle's coaxial flow topology: the inner
(radial) propellant lives in the *central bore* while the outer (axial)
propellant lives in the *surrounding annulus*, so an outer-radius ring manifold
can neither feed the central bore nor stay sealed from it.

This module builds the physically-coherent architecture from Rezende (Nardi),
*Experiments with the Pintle Injector*, and the TRW/Elverum patent lineage — the
five solids of revolution of the classic exploded view:

* **pintle body**  — inner propellant enters axially at the top and flows down
  the central bore;
* **pintle tip** (replaceable) — the ring of radial metering exits (round jets,
  the paper's preferred multi-jet, or rectangular slots) at the skip-distance
  line, one pintle-diameter into the chamber;
* **injector body** — outer propellant enters *laterally* into a *toroidal
  plenum* wrapping the pintle body;
* **orifice plate** — a ring of holes distributing the outer propellant evenly
  down into the annular collector;
* **faceplate** — the continuous **metering gap** between the pintle body and
  the faceplate that meters the axial sheet onto the pintle surface.

Sealed by construction: the inner propellant is fully enclosed by the pintle
body/tip walls, the outer propellant lives in the surrounding plenum/annulus,
and the two only meet in the chamber at the tip.  The layout is driven entirely
by the hydraulic sizing already resolved in
:func:`raosim.injector_cad.resolve_machined_pintle_layout`, so it generalises to
any solved pintle injector.

Convention (shared with ``injector_cad``): axis +Z, injector face at Z=0,
chamber downstream at Z>0, feed manifolds at Z<0.  Geometry is built in SI
metres and scaled to millimetres at the STEP boundary (repo convention;
see :func:`raosim.injector_cad._to_mm_step_solid`).
~~~~

### `raosim/injector_export.py`
*837 lines · stage S11 CAD and geometry export*

~~~~text
injector_export.py - pintle reference-geometry resolution + output package.

Turns a solved :class:`raosim.injector.InjectorDesignResult` into the CLI
deliverable folder::

    pintle/
      pintle_parameters.json     # full labeled geometry + operating point
      pintle_dimensions.csv      # flat dimension table (symbol, value, unit)
      pintle_schematic.svg       # mandatory labeled 2-D schematic (vector)
      pintle_cross_section.png   # mandatory labeled 2-D schematic (raster)
      pintle_reference.step       # optional, --injector-cad reference
      pintle_parts/               # optional, --injector-cad parts
        pintle_rod.step  pintle_tip.step  annular_sleeve.step  injector_face.step
      pintle_body.step            # optional, --injector-cad machined
      pintle_tip.step
      injector_body.step
      orifice_plate.step
      faceplate.step
      injector_assembly_machined.step
      injector_manufacturing_report.json

Fixed discrete exits retain the repository's existing slot/hole report.  A
Son et al. (*Design Procedure of a Movable Pintle Injector*, J. Propulsion &
Power 33-4, 2017) continuous-gap result instead reports its resolved internal
metering geometry, hard stops, calibration, actuator ledger, and independent
sheet-thickness evidence.  Those reports deliberately do not dispatch into the
fixed-geometry CAD exporters: a swept moving assembly is not implemented.
~~~~

### `raosim/pump_cad.py`
*938 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
Reference CAD package for electric-pump feed hardware.

The pump sizing model solves a meanline, not a blade-to-blade design.  This
module therefore exports labeled reference geometry for layout, interfaces, and
trade review.  It deliberately does not claim production-ready blade surfaces.

Literature basis:
* NASA SP-8109, *Liquid Rocket Engine Centrifugal Flow Turbopumps*:
  centrifugal pump elements, specific speed/specific diameter, head and flow
  coefficients, impeller blade count/angles, diffuser/volute selection.
* NASA SP-8052, *Liquid Rocket Engine Turbopump Inducers*: inducer inlet eye,
  hub ratio, blade count, solidity, suction-specific-speed and NPSH screening.
* Lee et al. 2021 and Spiller et al. 2013: electric-pump package mass/energy
  closure and small-pump efficiency caution.
~~~~

### `raosim/pump_cad_brep.py`
*1552 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
pump_cad_brep.py - true B-rep CAD for the electric-pump package.

Builds named CadQuery assemblies for the sized fuel/oxidizer pumps
(shaft, inducer with helical swept blades, impeller with log-spiral camber
blades, diffuser vane ring, volute casing + collecting scroll, and the
motor/inverter/battery package placeholders).  Every dimension is consumed
from the meanline manifest (:func:`raosim.pump_cad.pump_reference_geometry`,
itself fed by ``pumps.PumpReferenceGeometry``); blade camber math lives in
:func:`raosim.pumps.impeller_blade_camber` (physics owns it, CAD sweeps it).

Conventions mirror :mod:`raosim.injector_cad` (+Z shaft axis, SI metres in,
millimetres at the CadQuery/OpenCascade kernel boundary, named
``cq.Assembly``) and :mod:`raosim.regen_cad` validation (export -> re-import
-> ``isValid``/solid-count/volume gates).  This is reference geometry for
layout and trade review, not blade-to-blade design: use pump CFD,
rotordynamics, seals/bearings, and cold-flow tests before hardware release.

Axes: +Z is the shaft axis; the impeller exit plane sits at Z = 0 (the
meridional station x = 0), flow approaches from -Z; XY is the radial plane.
~~~~

### `raosim/regen_cad.py`

*1185 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
OpenCascade regenerative-wall manufacturing geometry.

This module builds one material body containing:

* the hot-gas liner;
* the ribs/lands between passages;
* the outer closeout jacket;
* lofted rectangular channel voids following axial or helical paths;
* annular inlet/outlet plenums; and
* radial inlet/outlet ports.

All public geometry inputs use SI metres.  CadQuery/OpenCascade uses
millimetres internally, so the conversion is explicit at the kernel boundary.
The result is exported as a true STEP B-rep and re-imported for validation.
~~~~

### `raosim/regen_geometry.py`

*356 lines · stage S11 CAD, mesh and geometry deliverables*

~~~~text
regen_geometry.py — 3-D nozzle geometry with regenerative cooling channels.

Builds a 3-D surface of revolution of the nozzle wall plus the N
regenerative cooling channels (the "coils") wrapping it, consistent
with the cooling-analysis geometry: each channel is an annular-segment
cross-section of width ``w`` (arc) × height ``h`` (radial), seated on
the inner wall at radius ``r_inner + t_wall``, with the channel count
``N`` and pitch ``2πr/N`` from the :class:`CoolingSpec`.  Channels run
axially (``helix_turns=0``, matching the 1-D/2-D/3-D cooling model's
straight-channel assumption) or helically (``helix_turns>0``, the
spiral "coil" look of tube-wall regen).

Outputs: a binary STL of the wall + channels for CAD/print, and a
Matplotlib 3-D render.  Pure NumPy + Matplotlib (no JAX).
~~~~


## S12 - Figures, MOC construction diagrams and flow animations

*4 files.*

### `raosim/flow_viz.py`
*894 lines · stage S12 Figures, MOC construction diagrams and flow animations*

~~~~text
flow_viz.py — physically-interpretable steady flow-field visualisation.

The MOC solve computes the resolved supersonic flow state (Mach, flow
angle, hence pressure/temperature) at every characteristic-net node.
This module turns that scattered data into a field while explicitly
checking that the solved nodes cover the nozzle from axis to wall:

* smooth filled **Mach**, **pressure**, **flow-angle**, and **temperature**
  contours, interpolated from the MOC net and masked to the nozzle interior;
* the **characteristic lines** (the Mach waves) overlaid — the visible
  physics of MOC, including the throat expansion fan;
* **streamlines** integrated through the flow-angle field (the gas
  paths);
all mirrored about the axis for a full nozzle-slice view.

Desktop use: pass ``show=True`` for a pop-up window (matplotlib's
interactive backend — MacOSX/TkAgg/Qt), or ``save_path=...`` to write a
PNG.  Pure NumPy + SciPy + Matplotlib.
~~~~

### `raosim/injector_plots.py`
*742 lines · stage S12 Figures, MOC construction diagrams and flow animations*

~~~~text
injector_plots.py - pintle injector diagnostic figures.

A full diagnostic set drawn from the sized :class:`InjectorDesignResult`:
cross-section, spray envelope, hydraulics, atomization/combustion development,
face/tip thermal stack, stability, manifold maldistribution, the gate
scorecard, and (optionally) the throttle map.  Kept separate from the
pure-numeric solver, matching the repo's plotting/solver split.
~~~~

### `raosim/moc_diagrams.py`
*1069 lines · stage S12 Figures, MOC construction diagrams and flow animations*

~~~~text
moc_diagrams.py — construction diagrams for the Rao/MOC nozzle build.

Where ``flow_viz`` renders the *solved field* (Mach/p/theta/T contours,
streamlines), this module renders the *construction* itself — the geometry
and characteristic topology the solver actually builds:

* :func:`plot_kernel_expansion_fan` — the throat **kernel**: the TT'
  starting line and the right-running characteristics (RRCs) fanning
  through the expansion arc up to the wall point B (the throat expansion
  fan of the method of characteristics).
* :func:`plot_rao_topology` — the annotated **Rao B-D-E topology**: the
  throat-arc wall (0→B), the last kernel RRC BD (B→D), the control
  surface / left-running characteristic DE (D→E), and the wall
  streamline (B→E), with the mass-flux balance ``mdot_BD == mdot_DE``
  that fixes the optimum contour (Rao 1958).
* :func:`plot_bde_mesh` — the **BDE characteristic net**: the interior
  mesh NASA's ``CalcBDERegion``/``CalcRemainingMesh`` marches from the
  control surface down to the axis, drawn as its two characteristic
  families.  A near-axis zoom makes visible the singular axis
  convergence (where rows are legitimately negative-r truncated) — the
  region the topology audit deliberately does *not* treat as a folded
  cell.
* :func:`plot_bde_diagnostics` — solved BDE-node Mach, pressure, and flow
  angle contours together with explicit C+/C- compatibility residuals,
  wall-to-axis row mass conservation, and the local same-family crossing
  audit.  This is the numerical companion to the topology-only mesh plot.
* :func:`plot_bde_integrity` — a compact four-panel numerical audit of
  Mach-line geometry/compatibility, oriented cell areas, the axis state,
  field smoothness, and several interpolated axial mass-flow cuts.

These figures read the in-memory ``bde_artifacts`` the ``bde`` wall method
stashes on ``solution.construction_diagnostics`` (kernel, NASA topology,
BDE region).  Each follows the ``flow_viz`` conventions: pass
``save_path=...`` to write a PNG (Agg backend) or ``show=True`` for an
interactive window; the Matplotlib ``Figure`` is returned.
~~~~

### `raosim/plotting.py`
*1338 lines · stage S12 Figures, MOC construction diagrams and flow animations*

~~~~text
plotting.py – 2-D and 3-D nozzle visualisation.
~~~~


## S13 - Differentiable whole-engine MDO, optimizer and requirement-driven design

*25 files.*

### `raosim/mdo/__init__.py`
*142 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo — multidisciplinary-feasible (MDF) differentiable engine MDO layer.

Implements docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md (§10 architecture, §12 build
order).  This package is a *pure-numerical* layer: no CAD, no file I/O in the
differentiated path, no host callbacks inside jitted code (plan rule 10 and
Phase-1 gate).  CAD/reporting stays downstream in the existing LREKit workflow.

Build state (see the plan's §11 tracker):

* Phase 1 — ``schema.py`` + ``scaling.py``: design-variable pytrees, bounds,
  affine scaling, jit/jacfwd-clean evaluate.                      [this commit]
* Phase 2 — ``properties.py``: C¹ shape-preserving property surfaces with an
  offline CEA sampler (``scripts/sample_cea_surface.py``).        [this commit]
* Phases 3–7 — ``assembly.py``: the walking-skeleton coupled residual
  R(y, x) = 0 (analytic nozzle + throat thermal + injector/pump/battery
  algebra) solved with a square Newton root-find and differentiated via the
  implicit function theorem.
* Phase 4a — ``grid.py`` + ``cooling.py``: fixed-topology station grid and the
  1-D+fin regen residual (Bartz/Sieder-Tate/land-fin, counterflow upwind march)
  solved for the stationwise wall temperatures; exposes the channel-fit and
  SP-8087 RP-1 coking (T_wc ≤ 728 K) constraints.
* Phase 5 — ``injector.py``: closed-form pintle block (orifice/TMR/blockage/
  spray) with the Son-2017 movable-pintle two-branch minimum area and the chug
  screen.
* Phase 6 — ``pump.py``: pump/electric-feed block with a C¹ efficiency-vs-
  specific-speed surrogate (replacing the binned ``pumps.py`` estimator),
  meanline duty/NPSH/tip-speed screens, and the Lee-2021 battery epigraph.
* Phase 7 — ``engine.py``: the coupled whole-engine solve ``solve_engine`` —
  all four blocks in one differentiable evaluation with the cooling Δp → pump
  hydraulic edge closed and an optional spray→η_c* fixed point (default frozen +
  ablation).  Surfaced via the ``--engine-mdo`` CLI flag.
* Phase 8/9 — ``nlp.py``: the ε-constraint hard-constrained NLP ``solve_min_mass``
  / ``pareto_frontier`` — min smooth electric-feed objective mass s.t. Isp ≥
  floor and every enforced margin ≥ 0, over the 10-variable unit-box design
  vector, exact JAX Jacobians → SLSQP.  Results also retain the exact installed
  electric-package mass as a distinct reporting quantity.
  Surfaced via ``--engine-mdo-optimize``.
* Output contracts — ``state.py`` provides a fixed-shape pure-JAX
  ``EngineState``; ``snapshot.py`` and ``postprocess.py`` map MDO and
  traditional results into one versioned host contract, run the authoritative
  nozzle/electric-pump re-evaluation, and compare common scalars/profiles.

Hard rules inherited from the plan (§0.1): discrete choices are enumerated
outside the traced core; converged implicit states are differentiated by the
IFT rather than by unrolling solver iterations; non-smooth reporting branches
remain explicit while conservative smooth envelopes are used where the NLP
requires derivatives; authoritative re-evaluation stays host-side.
~~~~

### `raosim/mdo/assembly.py`
*289 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.assembly — the walking-skeleton coupled engine residual (plan §12.5).

One residual signature (plan §12.1 rule 2): everything is R(y, x) = 0 stacked
into a single vector, solved by a square Newton root-find, and differentiated
via the implicit function theorem (Optimistix implicit diff — evaluation
report amendment #6: root-find, not least-squares implicit diff).

Skeleton state vector (scaled, fixed shape):

    y = [ Rt_hat, mdot_hat, Twg_hat ]

with residuals

    R1  thrust closure     F_target − CF(ε, γ, Pc, Pa) · Pc · π Rt²      = 0
    R2  c* mass-flow       mdot − Pc·At / (η_c* · c*_ideal(Pc, O/F))     = 0
    R3  throat thermal     T_wg − [T_aw − q(T_wg)/h_g(T_wg)]             = 0

R1/R2 are *naturally explicit* today (plan §4.1 warns against enlarging the
system to appear coupled) — they are carried as implicit states deliberately,
as integration scaffolding: the feedback edges that make them genuinely
implicit (regen Δp → feed → mdot; T_fuel_out → injector → η_c*; §5 loops)
land behind this interface in Phases 4a/5 without changing the solve/IFT
plumbing.  The parity test exploits the current triangularity: the solved
states must match the NumPy closed forms to solver tolerance.

Everything downstream of the state (injector velocities, pump duty, electric
power, battery/motor masses, the §3 mass ledger with explicit placeholders)
is EXPLICIT algebra in ``readouts`` — no ``max()`` in any differentiated
constraint path (plan rule 5): the battery power- and energy-limited masses
are exposed separately for the epigraph treatment at the NLP layer;
``package_mass_report`` (which does take the max) is a *reporting* scalar.
~~~~

### `raosim/mdo/bounds.py`
*258 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
mdo/bounds.py — design-variable bounds derived from the architecture.

Why this module exists
----------------------
``MissionSpec.scaled_design_space`` used to hard-code

    VariableSpec("Pc",  1.5e6, 6.0e6, 3.0e6)
    VariableSpec("eps", 3.0,   40.0,  8.0)

with a docstring admitting they "bracket the 13 kN LOX/RP-1 baseline".  Those
two boxes were then applied unchanged to LOX/LH2, N2O4/MMH and every thrust
class, which is exactly the defect
``docs/GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md`` §2 is about.  Neither
limit is actually set by thrust or by kerosene:

* **Chamber-pressure guidance depends on the engine cycle.**  Parsley and
  Zhang (2004) describe different limiting or optimizing mechanisms for the
  common pump-fed cycles.  Their approximate optima and onset-of-limitation
  values are not universal hard-validity endpoints.
* **Expansion ratio is set by the model's own validity domain.**  The analytic
  TOP wall comes from the Rao/NASA chart, which is tabulated over a specific
  box; outside it ``rao_chart_domain_violation`` already declares the design
  inadmissible.

One source of truth
-------------------
Both bounds are taken from the *same* place as the constraint that screens
them.  ``eps`` is bounded by the chart grid that ``chart_domain_margin`` uses;
``OF`` (in ``MissionSpec.of_design_space``) is bounded by the CEA grid that
``property_domain_margin`` uses.  A bound and a constraint that disagree about
the same limit is how the two pipelines drifted apart before R0.

Sources
-------
Parsley, R. C. and Zhang, B., *Thermodynamic Power Cycles for Pump-Fed Liquid
Rocket Engines*, 2004, Chapter 18, DOI
10.2514/5.9781600866760.0621.0648 —
``propulsion_texts/fuel_pump_design/thermodynamic-power-cycles-for-pumpfed-liquid-rocket-engines-2004.pdf``.

Huzel & Huang, *Design of Liquid Propellant Rocket Engines*, NASA SP-125, 1967,
ch. V and ch. III — ``propulsion_texts/19710019929.pdf``.

NASA MSFC, *Liquid Propulsion: Propellant Feed System Design*, 2010 —
``propulsion_texts/fuel_pump_design/20100035254.pdf``.
~~~~

### `raosim/mdo/constraints.py`
*593 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
Authoritative constraint and feasibility contract for the engine MDO.

The discipline solve, NLP, numerical :mod:`raosim.mdo.state`, and host
snapshot all consume this ordered manifest.  A constraint therefore cannot be
added to reporting without making an explicit decision about whether the
optimizer enforces it and whether missing model coverage is a failure or an
unknown.

The manifest is software policy around equations implemented in the discipline
modules; it does not invent new physical limits.  ``source_id`` points back to
the source already documented beside each underlying equation.  The two new
policy-sensitive rows have the following exact source map (the original PDFs,
not the OCR, were checked):

* ``NASA-SP-125-1967-sec-4.2-PDF137-printed128`` — Dieter K. Huzel and
  David H. Huang, *Design of Liquid Propellant Rocket Engines* (1967), NASA
  SP-125, sec. 4.2, "Injection Pressure Drop," source-PDF p. 137 /
  printed p. 128.  The stated 15--20 percent range supports the numerical 0.20 design
  screen.  It is not presented here as a universal chug-stability boundary.
* ``NASA-SP-194-1972-sec-6.2.3.1-PDF293-294`` — David T. Harrje (editor) and
  Frederick H. Reardon (associate editor), *Liquid Propellant Rocket
  Combustion Instability* (1972), NASA SP-194, sec. 6.2.3.1, "Injector
  Impedance," source-PDF pp. 293--294.  It supports the qualitative coupling
  between injector pressure drop and low-frequency stability and explicitly
  warns that increasing only one stream's drop can be destabilizing.
* ``Nasuti-Pizzarelli-2021-sec-2.2-Eq9-PDF6`` — Francesco Nasuti and Marco
  Pizzarelli, *Pseudo-boiling and heat transfer deterioration while heating
  supercritical liquid rocket engine propellants* (2021), sec. 2.2, Eq. (9),
  source-PDF p. 6, DOI 10.1016/j.supflu.2020.105066.  Its ``(beta/cp)_b`` input
  is a real-fluid bulk property and cannot be reconstructed from the cooling
  march's constant properties.
* ``NASA-SP-8087-1972-sec-3.1.1.5.4-PDF78`` — *Liquid Rocket Engine
  Fluid-Cooled Combustion Chambers* (NASA SP-8087, April 1972), sec. 3.1.1.5.4,
  "Wall Temperatures," source-PDF p. 78.  It directly gives the 728 K RP-1
  liquid-wall coking limit already implemented by the cooling discipline.
* ``Son-2017-JPP-33-4-DOI-10.2514-1.B36301`` — Son, Radhakrishnan, Koo, Kwon
  and Kim,
  *Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines*,
  JPP 33(4) (2017), local PDF ``propulsion_texts/pintle_injector/son2017.pdf``
  (SHA-256 ``bc10dd3b…a2b1a7``).  Eq. (1) and the transition discussion
  (PDF p. 4: the controllable opening range lies below the transition) support
  ``pintle_transition``; PDF p. 2 (no uniform sheet within 0.1 mm) and PDF
  p. 10 (a schedule below 0.1 mm rejected as infeasible) support
  ``pintle_min_opening``; PDF p. 9 lists both "minimum area ... lower than the
  center gap area" and "chamber diameter should be larger than the injector
  module diameter" (``pintle_module_fit``) among its feasibility checks.  The
  0.95 transition fraction and the 0.1 mm floor applied to a new geometry are
  repository applicability choices, not universal limits.

The pintle rows are architecture-specific (plan rule 4).  ``pintle_slotted``
rows apply only to the fixed slotted post and ``pintle_movable`` rows only to
the Son continuous-gap valve; a row that does not apply is returned inert by
the engine and marked not applicable here.

Basis of every row (``basis_class``; MDO issues audit 2026-10-16, §1.4 item
22).  A ``source_id`` is a citation route; ``basis_class`` says what kind of
claim the implemented limit is, so a row with no literature behind its number
is labelled rather than left as an unknown basis:

* ``literature`` — the limit (or its governing allowable) is stated in the
  cited source;
* ``geometric_definition`` — a packing/ordering identity of the modelled
  geometry (non-negative land, non-decreasing divergent wall, ...);
* ``model_validity_domain`` — the domain over which an implemented model or
  table is defined (sampled CEA box, Rao/TOP chart extent);
* ``repository_rule`` — a repository choice; any ``source_id`` cites only the
  criterion's form, not the implemented number;
* ``numerical_contract`` — a software gate on the solve itself;
* ``requirement`` — a user requirement row.

Rows corrected or newly attributed in the 2026-10-16 pass (original PDFs,
page-verified):

* ``blockage_lo`` (BF ≥ 0.30): Freeberg 2019, ``propulsion_texts/
  pintle_injector/freeberg2019.pdf``, PDF pp. 5-6 relays TRW advice of
  0.3 < BF < 0.7 (second-hand, via Escher), PDF p. 10 (conclusions) repeats
  the 0.3-0.7 band and PDF p. 2 Eq. (1) defines BF.
* ``blockage_hi`` (BF ≤ 0.90): no held source states 0.90.  The held bounds
  are 0.7 (the same TRW advice) and the Ryu optimum near BF 85 % that Hwang
  et al. report second-hand together with thermal damage of an 86 % tip
  (``pintle_injector/s42405-022-00489-w.pdf`` PDF pp. 2, 4, 8).  Repository
  rule; the value is unchanged pending an owner decision.
* ``wall_temp`` (800 K): the criterion form — an allowable gas-side wall
  temperature set from the material's yield-strength sensitivity — is
  Mirzamoghadam 1991 (PDF p. 5; Table 2, PDF p. 7, lists the 1050 °F
  hot-channel value as "Depends on Material").  800 K itself is not in the
  paper (its 900 °F = 755 K is an H2/O2 copper compatibility value, PDF
  p. 4); repository rule.
* ``chart_domain``: ε ∈ [4, 50], L_f ∈ [60, 100] % match the extent of the
  held Rao/TOP angle charts (SP-125 Fig. 4-14, PDF p. 100; SP-8120 Fig. 5(b),
  PDF p. 29), graphs only.  The extent is cited, not the tabulated
  ordinates (see the adjudication note in ``raosim.nozzle_geometry``).
* ``land_fit``, ``wall_monotonic``: geometric definitions;
  ``pintle_wall_fit``: the host CAD layout cap 0.8 r_o
  (``injector_cad.resolve_machined_pintle_layout``), a repository rule;
  ``property_domain``: the sampled CEA table box.
~~~~

### `raosim/mdo/convergence.py`
*396 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.convergence — station-refinement (grid-convergence) evidence.

The coupled solve runs on a fixed-topology station grid
(:class:`raosim.mdo.grid.GridTopology`, default 4 + 6 + 13 + 1 = 24 stations).
The plan's numerical definition of done (``docs/DIFFERENTIABLE_ENGINE_MDO_PLAN
.md`` §11) asks that "doubling cooling/nozzle stations changes objective and
active margins by < 0.5 %"; the review (§15.5) found no test of it.  This
module performs that study at a given design point:

1. solve the engine at the base topology and at the topology with every
   segment count multiplied by ``factor`` (default 2: 8 + 12 + 26 + 1 = 47);
2. report the objective, Isp and every hard-row margin at both;
3. normalize the changes — relative for the objective and Isp, and for the
   margins the absolute change divided by the manifest row ``scale`` (the
   same conditioning reference the optimizer uses), because an active margin
   sits at zero and has no meaningful relative change;
4. state whether the < 0.5 % criterion holds for the objective and for every
   active row (scaled base margin ≤ 1e-5, the optimizer feasibility
   tolerance), with both evaluations numerically valid.

:func:`station_refinement_study` holds the design point fixed.  That
measures the discretization error of one design, but the plan's criterion
concerns the solved problem, so :func:`reoptimized_refinement_study`
re-solves the whole NLP on each grid and compares the optima: their
objectives, Isp and active sets.  The two readings differ (at the 13 kN
optimum the fixed design's coking margin moves ~5 % of scale, while the
re-optimized objective moves ~0.26 %), so both are reported.  Nothing is
tuned or clipped; the studies are evidence, not gates.  The < 0.5 %
threshold and the doubling procedure are the plan's repository definition of
done; no held source prescribes them.
~~~~

### `raosim/mdo/coolant_htd.py`
*289 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
mdo/coolant_htd.py — supercritical heat-transfer-deterioration screen.

The gap this closes
-------------------
The repository's only coolant-side wall constraint is the 728 K RP-1
**coking** limit (NASA SP-8087, April 1972, sec. 3.1.1.5.4, source-PDF p. 78).
Coking is a hydrocarbon phenomenon, and
``raosim.mdo.propellants`` correctly reports ``coolant_wall_limit_K = None`` for
hydrogen — but nothing replaces it.  For methane and hydrogen a supercritical
real-fluid wall-side screen is therefore required; reporting only a carbon-
deposition/coking screen would call an unevaluated mechanism feasible.

Nasuti & Pizzarelli, *Pseudo-boiling and heat transfer deterioration while
heating supercritical liquid rocket engine propellants*, J. Supercritical
Fluids 168:105066 (2021), sec. 2.2, source-PDF p. 6,
DOI 10.1016/j.supflu.2020.105066, ``propulsion_texts/nasuti2021.pdf``, states
the problem exactly:

    "Heating of liquid propellants used as the coolant in rocket engines may
    lead to undesired phenomena such as pseudo-boiling or heat transfer
    deterioration under specific conditions.  This can be an issue for
    propellants characterized at the same time by relatively low critical
    pressure and temperature.  Light hydrocarbons, as for instance methane,
    belong to this family."

The criterion
-------------
Their Eq. (9) gives the onset of deterioration as

.. math::

    \frac{q_w}{G\,f_w}\left(\frac{\beta}{c_p}\right)_b > K ,
    \qquad K = 0.187

with :math:`q_w` the wall heat flux, :math:`G` the coolant mass flux,
:math:`f_w` the friction factor defined there as
:math:`f_w = 4\tau_w / (\tfrac12 \rho_b u_b^2)` (i.e. the Darcy friction
factor), and :math:`(\beta/c_p)_b` the isobaric thermal expansion coefficient
over the specific heat, both evaluated at **bulk** conditions.  The threshold
value is quoted as "essentially similar to that proposed in [20] and equal to
that proposed in [21] where the threshold value is also estimated to be
K = 0.187".

The group is dimensionless: ``[W/m2] / [kg/m2/s] = J/kg``, and
``beta/cp`` carries ``[1/K] / [J/(kg K)] = kg/J``.

Physical reading, in the authors' words: deterioration "occurs for large heat
flux q_w and low mass flux G", and "for a given flow in a tube, heat transfer
deterioration occurs in the tube sections where (β/c_p)_b is sufficiently
large" — that group "presents a maximum which nearly occurs" at the Widom
(pseudo-critical) line.  **That peak is the mechanism.**

Why this is not yet a live constraint
-------------------------------------
:math:`(\beta/c_p)_b` is a real-fluid property that peaks sharply near the
pseudo-critical temperature.  The MDO cooling march currently carries the
coolant as four constants (``MissionSpec.rho_cool``, ``cp_cool``, ``mu_cool``,
``k_cool``) and has no :math:`\beta` at all.  Evaluating Eq. (9) with constant
properties would return a smooth, flat number that misses precisely the peak
the criterion is detecting — a wrong model, not a coarse one.

So this module implements the criterion honestly and refuses to fake its
inputs: :func:`htd_margin` requires ``beta_over_cp`` to be supplied, and
:func:`htd_availability` reports the screen as unavailable, with the reason,
whenever it cannot be.  Closing it needs a **CoolProp-sampled coolant property
surface** over (T, p) — structurally the same object as the CEA chamber
surfaces in :mod:`raosim.mdo.properties`, and buildable from the same two
sources now in the corpus: Bell et al. 2014 (CoolProp) for the cryogens, and
Huber et al. 2009 (``huber2009.pdf``) for RP-1, which CoolProp does not carry.

Conservatism
------------
The authors note their models "are based on the case of smooth and uniformly
heated tubes which are more prone to deteriorate than the actual rough,
asymmetrically heated channels", and that deterioration "is mitigated in case
of pressure much higher than critical and by wall roughness".  So a margin
computed here errs toward predicting deterioration — the safe direction for a
screen, and worth stating in any report that quotes it.

Buoyancy is neglected, following the paper: "as the typical flow conditions in
rocket engine cooling channels are relevant to relatively high mass flux, of
the order of thousands of kg/s/m2, the resulting turbulent flow can be
considered unaffected by any buoyancy effect."
~~~~

### `raosim/mdo/cooling.py`
*545 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.cooling — Phase 4a: 1-D + fin regenerative-cooling residual block.

Physics (plan §6.2, all corpus-anchored, mirroring the audited NumPy oracle in
``raosim.physics``):

* gas side — full Bartz (1957) h_g with the σ property-variation factor
  (``jax.thermal.bartz_hg`` / ``bartz_sigma``, exponents 0.68/0.12 at ω=0.6),
  station Mach/area from ``mdo.grid``; turbulent recovery T_aw with r = Pr^(1/3)
  (``jax.thermal.recovery_temperature``);
* series thermal circuit per station through the hot wall (SP-8087 regen
  framing: coolant, Δp, wall temperature and geometry as one coupled
  allocation), with the rib/land **fin efficiency** η_f = tanh(mH)/(mH),
  m = √(2 h_c/(k_w t_land)), carrying the aspect-ratio (HARCC) trend
  (Huzel & Huang SP-125 §4; Pizzarelli 2011 / Carlile 1992 resolve the same
  first-order mechanism in full 2-D — that is the 4b deepening, not 4a);
* coolant side — **Sieder–Tate** h_c via the audited ``jax.thermal.sieder_tate_hc``
  primitive (0.027·Re^0.8·Pr^(1/3)·(μ_b/μ_w)^0.14), evaluated with μ_wall = μ_bulk
  at this screening fidelity so the property ratio is unity (Sieder & Tate 1936;
  ``physics.sieder_tate_coefficient``);
* hydraulics — Darcy Δp with 64/Re laminar and Blasius/Swamee–Jain turbulent
  (explicit, smooth — deliberately not the implicit Colebrook; White §6);
* energy march — ṁ_cool·cp·dT_c = q″·dA over each wall segment, **counterflow**
  (coolant enters at the nozzle-exit manifold, exits at the injector end),
  discretised as a first-order **upwind finite-volume** march in the coolant
  flow direction (each of the n−1 wall segments is crossed exactly once, the
  circuit for a crossing evaluated at its upwind node).  The march is explicit
  algebra given the stationwise wall temperatures — nothing is unrolled through
  solver iterations (plan rule 3): the T_wg array lives in the state vector and
  is driven to R = 0 by the outer Newton/IFT (``solve_cooling``).

Fixed-geometry consequence: with static channel count/section (N, w, h) and
constant screening coolant properties, the coolant mass flux G, Re, Dh, velocity
and hence h_c and the Darcy factor are *uniform* along the jacket; only the fin
augmentation (through the land width, which follows the local radius), h_g and
T_aw vary station-to-station.  A width-tapered (constant-aspect) channel and
variable properties are the 4b/Phase-2 deepenings.

Channel-fit constraint (plan §9 — never ``clip()`` invalid geometry): the land
between channels, t_land = 2πr_cool/N − w, must stay positive.  ``cooling_march``
still floors it for numerical safety in η_f, but ALSO returns the *true* minimum
land ``land_min`` so the NLP layer can carry ``land_min ≥ land_floor`` as an
explicit inequality instead of silently absorbing an infeasible packing.

Coolant coking constraint (SP-8087 recommended practice): RP-1 cokes on the
liquid side of the heated wall above ~850°F (728 K) liquid-wall temperature
(onset 800–900°F, 700–756 K; Sellers, ARS J. 31(5), 1961).  The block therefore
computes the coolant-side wall temperature T_wc = T_wg − q″·t_wall/k_wall and
returns the stationwise ``coking_margin`` = T_coke_limit − T_wc (≥0 feasible),
a differentiable inequality for the NLP.  With the *screening* (non-optimised)
channel geometry this margin is strongly negative near the throat — the finding
that the fixed-geometry jacket is thermally infeasible at P_c = 3 MPa; the
optimiser is meant to size N/w/h/t_wall (or lower P_c, add film cooling) to
recover it, which is precisely why the constraint must be visible rather than
absorbed.

Residual block:  R_i = T_wg,i − [T_aw,i − q″_i / h_g,i]   (i = 1..n),
with q″_i from the series circuit at the *marched* local coolant temperature.

NumPy twin: ``tests/test_mdo_cooling.py`` carries a plain-Python re-implementation
(same discretisation, same primitives) proving ~1e-9 parity — the §12.1 rule-1
oracle for this block.  Cross-validation against the richer
``physics.regenerative_cooling_analysis`` (curvature correction, variable
properties, 2-D wall options) is a host-side follow-up and the 4b ladder.
~~~~

### `raosim/mdo/engine.py`
*829 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.engine — Phase 7: the coupled whole-engine solve.

Integrates the four discipline blocks (nozzle performance, regen cooling,
pintle injector, electric pump feed) into ONE differentiable evaluation, so a
single gradient of any objective/constraint flows from the design vector x
through every block and both implicit-function-theorem solves (plan §12.3, gate
"all residual blocks converge without manual sequential iteration").

Coupling honesty (plan §4.1: "do not enlarge the nonlinear system merely to
appear coupled").  At the current physics fidelity the data flow is:

    outer state (Rt, mdot)  ──►  grid ──►  cooling  ─┐  (inner IFT: T_wg vector)
                                          injector   ├─►  Δp_regen, Δp_inj
                                                     ▼
             pump feed  ◄───  Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank
                                                     ▼
                          P_electric, battery, drive masses, §3 ledger

So the §5 **hydraulic edge is now genuinely closed** — the cooling jacket Δp and
the injector Δp feed the pump rise (previously ``regen_dp_allowance = 0``
placeholders).  The wall-temperature vector is a real inner implicit solve.
Everything else is the explicit chain the physics actually is: the heat load and
feed pressures do not change the throat area or chamber mass flow, so (Rt, mdot)
stay a small outer root — kept implicit as the seam where the ONE genuine
two-way feedback lives.

The **combustion/mass-flow loop** (spray → η_c* → mdot → spray, plan §5) is the
only genuine 2-way coupling.  It is *off by default* (``couple_eta_cstar=False``
⇒ frozen ``mission.eta_cstar``): the plan flags it as the strongest edge but the
weakest physics and asks for the ablation.  With it on, R2's η_c* depends on the
injector TMR(mdot), so the outer Newton solves a real fixed point and the
``ablation_delta`` (coupled − frozen) is directly measurable (RQ1).

Nothing here touches CAD or the reporting workflow (plan rule 10).
~~~~

### `raosim/mdo/envelope.py`
*204 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
mdo/envelope.py — differentiable thrust-chamber envelope.

Why this module exists
----------------------
NASA SP-125 §2.1 (printed p. 31, ``propulsion_texts/19710019929.pdf``) lists
**envelope (size)** as one of the nine parameters that *"engine systems design
and development specifications will have to cover... above all"*, and gives the
working definition:

    "In those cases where only approximate values are required for comparison
    or for overall estimates, the term 'envelope' is preferred.  For instance,
    definition of a hypothetical smallest cylinder, cube, or sphere into which
    the engine would fit conveys a good feeling of engine size or bulkiness."

A cylinder is the natural choice for an axisymmetric thrust chamber, so this
module returns the smallest enclosing cylinder of the **thrust-chamber outer
mould line**: a diameter and a length.

SP-125 also records why this is a design driver rather than a nicety:
*"engine size directly affects engine weight... The vehicle structure, which
becomes heavier, especially with upper stages.  Engine size directly affects the
size and thus weight of the aft end and/or interstage structure."*

What this is NOT
----------------
Both returned quantities are **lower bounds on the installed engine envelope**,
and callers must not present them as the engine envelope.

* *Diameter* is built from the same radial stack the mass ledger integrates
  (``r + t_wall + channel_height + t_jacket``), so it is the outside of the
  cooling jacket.  It excludes the **bolted-interface flange**, which is
  resolved host-side in :mod:`raosim.interface` and which
  ``docs/HARDWARE_MASS_LEDGER.md`` records as the *largest* diameter on the
  13 kN baseline — a 285.5 mm flange around a 177 mm chamber.  A flange that
  large governs the envelope outright.
* *Length* runs from the injector face (station 0 of the grid, which is the
  SP-125 chamber-volume datum) to the nozzle exit plane.  It excludes injector
  body depth, the dome, the pump package, valves, lines and the gimbal.

The requirement layer therefore records an envelope requirement as *partially*
screened; see :class:`raosim.requirements.RequirementCoverage`.  The
authoritative envelope comes from the host re-evaluation, which has the flange.

Smoothness
----------
The radial maximum uses the same conservative smooth upper envelope as
:mod:`raosim.mdo.engine` (``logsumexp(k·v)/k``, never below the true maximum).
Erring large is the conservative direction for an envelope constraint, and it
keeps the NLP free of the active-set kink a hard ``max`` over stations would
introduce (plan §0.1; see also gap 12.11 on remaining hard extrema).
~~~~

### `raosim/mdo/grid.py`
*468 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.grid — fixed-topology chamber/nozzle station grid (plan §10, §6.2).

A differentiable analytic stand-in for the thrust-chamber inner contour,
parameterized by the solved throat radius and the design expansion ratio, so
JAX array shapes never change (plan §6.2: "fixed-topology normalized axial
grid") while the geometry itself moves with the state during the Newton solve.

Fidelity note (plan §2.1 "two contour fidelities"): this is the *bulk-sweep*
geometry.  It exists to give the 4a cooling march station areas, radii, and
Mach numbers; the Rao/TOP contour remains the performance/export geometry, and
the implicit variational wall stays on the existence-boundary path.  Station
count and the subsonic/supersonic split are STATIC (Python ints) — only
coordinates move.

Chamber convention (2026-07-31)
-------------------------------
The upstream half now uses the **same four-section construction** as
:func:`raosim.chamber_geometry.chamber_contour`: cylindrical barrel → shoulder
fillet → straight convergent cone → upstream throat arc, with the barrel length
solved so that the injector-face-to-throat volume equals ``L* · A_t``.

This replaces the earlier ``chamber_length = L*/CR`` barrel plus a cosine
blend.  That approximation assumed the *barrel alone* held the whole chamber
volume, but NASA SP-125 is explicit that it does not (Huzel & Huang, NASA
SP-125, 1967; ``propulsion_texts/19710019929.pdf``, ch. IV, printed p. 88):

    "In design practice, it has been arbitrarily defined that the combustion
    chamber volume includes the space between injector face I-I and the nozzle
    throat plane II-II."

Because the convergent section carries part of ``L*·A_t``, ``L*/CR`` makes the
barrel too long.  At the 13 kN baseline it over-stated the chamber by 20.1 mm
and the wetted area by 11.7 % — an 11.7 % error in total heat load, not only in
mass, and it made every MDO-vs-traditional cooling comparison meaningless.
SP-125's own approximate chamber (eq. 4-6, printed p. 89) is barrel + straight
cone; the shoulder fillet and upstream throat arc used here are the
repository's existing refinement of that, and matching them is what makes the
two pipelines describe one chamber.

Region layout (n = n_chamber + n_conv + n_div + 1 stations, throat shared):

    [0 .. n_chamber)                cylindrical barrel, A/At = contraction_ratio
    [n_chamber .. n_chamber+n_conv] shoulder fillet + convergent cone + upstream
                                    throat arc, resampled by arc length
    (throat at index n_chamber + n_conv)
    (throat .. n-1]                 throat downstream arc + Rao/TOP quadratic
                                    Bezier to the exit
~~~~

### `raosim/mdo/injector.py`
*180 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.mdo.injector — Phase 5: differentiable pintle-injector block.

Explicit algebra (no implicit state, so no IFT needed — just closed-form jnp
that jacfwd/jacrev flow through), mirroring the audited NumPy oracles in
``raosim.injector`` and ``raosim.movable_pintle``.  Physics (plan §6.3, all
corpus-anchored):

* orifice metering, incompressible (LOX and RP-1 both liquid at injection —
  ``injector._stream_mass_flux`` liquid branch): G = C_d·√(2 ρ Δp),
  A = ṁ/G, v_inj = G/ρ = C_d·√(2 Δp/ρ);
* total momentum ratio TMR = (ṁ_r v_r)/(ṁ_a v_a) with the radial stream = fuel
  through N slots and the axial stream = oxidiser through the annulus (repo
  convention ``PintleGeometrySpec.radial_stream="fuel"``; universally the pintle
  master knob — Son 2015/2017, Hwang 2022 s42405, Sakaki 2015 "TMR usually ≈1");
* spray half-angle from the radial/axial momentum vectors with the deflector
  tilt, θ = atan2(m_r cosδ, m_a + m_r sinδ) (``injector.py`` L3199-3201); at
  δ=0 this is the **leading-order kinematic** θ ≈ arctan(TMR).  *Fidelity note*:
  Escher found the bare arctan a poor fit and proposed θ ∝ TMR^½; Son 2015
  (s11630-015-0753-7) gives an exponential θ(TMR, We) correlation — those are
  the higher-fidelity swaps (like the 4a→4b thermal ladder), deferred;
* blockage factor BF = N·w_slot/(π D_pintle) (Freeberg 2019, Hwang 2022);
* **movable-pintle minimum area as TWO branches** (plan rule 8 / §6.3, Son 2017):
  the tip-opening-limited area A_tip = π[2 r_f L cosθ − L² sinθ cos²θ]
  (``movable_pintle.son_minimum_tip_area``, Son Eq. 1; r_f = D_p/2 − t_post) and
  the fixed center-gap-limited area A_cg = (π/4)(D_cg² − D_pr²)
  (``movable_pintle.center_gap_area``).  We size the opening L that meters the
  required radial area (closed-form inverse of Eq. 1) and expose the
  *consistency inequality* ``transition_margin`` = frac·A_cg − A_tip ≥ 0 — never
  a differentiable ``min()``; which branch governs is an outer-enumeration
  architecture choice;
* injector-drop screen: min(χ_f, χ_o) ≥ 0.2 as a stationless conservative
  design inequality.  The number is the upper endpoint of the 15--20% rule of
  thumb in Huzel & Huang, *Design of Liquid Propellant Rocket Engines*
  (1967), NASA SP-125, sec. 4.2, source-PDF p.137 / printed p.128.  Harrje &
  Reardon (eds.), *Liquid Propellant Rocket Combustion Instability* (1972),
  NASA SP-194, sec. 6.2.3.1, source-PDF pp.293--294 supports the qualitative
  stability coupling but warns that increasing one stream's drop alone can be
  destabilizing.  This row is therefore not a universal chug boundary or a
  substitute for coupled stability analysis.

Parity oracle: tests/test_mdo_injector.py checks these against the NumPy
``movable_pintle`` functions and the ``injector`` orifice/TMR forms to ~1e-9.
~~~~

### `raosim/mdo/mass.py`
*259 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
mdo/mass.py — differentiable thrust-chamber hardware mass.

This is the JAX mirror of :mod:`raosim.mass_ledger`.  Both integrate the same
solid of revolution over the same wall description; this one does it in pure
``jnp`` on the fixed MDO station grid so ``d(mass)/d(design)`` exists and the
optimizer can trade chamber structure against feed-system mass instead of
minimising only the electric package and calling the result engine mass.

Physical basis
--------------
The equations below are geometric shell-volume calculations (Pappus's centroid
theorem), not a thrust-chamber mass correlation.  NASA SP-125 (Huzel & Huang,
*Design of Liquid Propellant Rocket Engines*, 1967;
``propulsion_texts/19710019929.pdf``, Ch. VIII, “Design of Propellant Tanks,”
§8.1.2.3, source-PDF p. 348 / printed p. 339) uses the same surface-times-
thickness relation for a cylindrical **tank** section:

    W_c = 2 π a l_c t_c ρ                                  (eq. 8-32, p. 339)

with ``a`` the *nominal* (mid-surface) radius.  That tank equation corroborates
the geometry but does not prescribe thrust-chamber thickness or mass.  Applying
the geometric relation per station over the meridional arc gives, for a
regeneratively cooled wall,

    A_liner    = 2 π (r + t_w/2) t_w
    A_land     = π (r_o² − r_i²) · b/(b + w)
    A_closeout = 2 π (r_o + t_j/2) t_j

with ``r_i = r + t_w``, ``r_o = r_i + h``.  The land term is written as an area
*fraction* of the channel annulus rather than ``N·b·h`` because the fraction is
invariant to helical stretch, is unconditionally non-negative, and stays smooth
in the design variables — all three matter inside an SLSQP loop.

What the closeout is and is not
-------------------------------
NASA SP-8087 (*Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, 1973;
``propulsion_texts/19730022965.pdf``) §2.1.3 states the three structural jobs of
chamber reinforcement — hoop support about the combustion chamber, support at
the throat against bending and buckling, and hoop support about the expansion
nozzle against collapse from hoop compression when the nozzle runs overexpanded
at sea level — and quotes the design factors of safety in use: yield 1.0 to
1.32, ultimate 1.3 to 1.8.  **This module does not size the closeout against
those loads.**  It integrates a closeout whose thickness is set by
``MissionSpec.closeout_thickness_ratio``, and the mass it returns is therefore
conditional on that assumption.  Making the jacket thickness a solved structural
variable is tracked as future work; until then the ratio is reported in the
state's provenance so a reader can see the assumption rather than infer it.

Differentiable invalid-domain behavior
--------------------------------------
On physically valid geometry the equations are smooth in ``(Rt, eps, t_wall,
channel_width, channel_height)``.  An infeasible optimizer probe can make a
computed land width negative.  The shared JAX volume kernel uses the absolute
land magnitude only as a nonnegative continuation outside that domain, while
the independent ``land_min`` constraint rejects the point.  This introduces a
kink only at the invalid/valid boundary and prevents negative material from
artificially lowering the objective.
~~~~

### `raosim/mdo/material_map.py`

*223 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
The single typed mapper from a catalog material to traced MDO properties.

A regeneratively cooled wall is two different alloys doing two different jobs
-- NASA SP-8087 sec. 2.1.3.1: *"Hardenable materials often are used for jacket
designs, where, after brazing, the strength can be increased considerably by
agehardening"* -- so the liner and the structural closeout are selected
separately and mapped separately here.

The contract this module exists to enforce is **atomicity**.  Before it, a
host-side material label reached the traditional pipeline through
``MaterialSpec.from_catalog`` while the differentiable core kept flat scalar
defaults, so ``--material GRCop-84`` optimized a NARloy-Z-class liner: a 12
percent conductivity error, a 21 percent modulus error, and a 200 K error in
the allowable gas-side wall temperature.  That is the same failure class as the
O/F sentinel -- one concept with two definitions that silently disagree.

Therefore either **every** field a selection owns is resolved from one catalog
record, or the selection is rejected.  There is deliberately no partial
application and no per-field fallback to a class default: a half-applied
material is a mongrel alloy that exists in no catalog and matches no
qualification data.
~~~~

### `raosim/mdo/nlp.py`

*1456 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.nlp — Phase 8/9: the ε-constraint hard-constrained NLP + Pareto sweep.

Objective/constraint policy (plan §8):

    min  m_objective      s.t.  I_sp ≥ I_sp,min          (the ε-constraint)
                                every *enforced* discipline margin ≥ 0

Both implemented objective identities use a smooth battery governing-branch
surrogate so gradients remain continuous. ``min_dry_mass_partial`` (the
default) adds liner, channel lands, and closeout to the electric-feed branch;
it is explicitly not full engine dry mass. Exact electric and resolved-partial
subtotals are reported separately.

sweeping ``I_sp,min`` traces the mass–performance frontier.  The engine is
always thrust-closed internally (the outer Newton), so thrust is NOT a free NLP
constraint — the design variables are the twelve hardware values of
``DesignVector`` (``D_cg`` pinned for a slotted pintle) and every
evaluation returns a converged, differentiable engine (``mdo.engine``).

Derivatives (plan §4.3, §12.2 row 8): a *low-dimensional, constraint-rich*
problem (n_x = 11–13, 28 hard rows), so total derivatives are exact through the two
IFT solves and handed to SLSQP with no finite-difference step noise.  The
constraint Jacobian is assembled by **reverse mode** (``jax.jacrev``) and the
scalar objective by ``jax.grad``: forward mode (``jacfwd``) through the jit'd
nested Optimistix root-finds drops the tangent of the geometry-only min-
aggregated margins (a forward-through-implicit-solve quirk), and at 28×13 reverse
mode is no costlier than forward here anyway.  Optimisation runs in the unit box
(``ScaledSpace``) so O(1e6) pressures and O(0.1) fractions are conditioned
together (Martins & Ning 2021 ch. 4).

**Coking is now enforced** (all of ``CONSTRAINT_NAMES``).  An earlier finding
showed the RP-1 liquid-wall coking limit (SP-8087, 728 K) was violated
everywhere in the box with pure regen and fixed channels — the wall is
coolant-enthalpy-limited, not film-coefficient-limited, so channel geometry
alone cannot satisfy it.  The fix (this commit) is **film cooling**: the design
vector now carries ``channel_width``, ``channel_height`` and a fuel ``film_frac``
that reduces the gas-side driving temperature over the chamber→throat region at
a delivered-c* penalty (SP-8087/SP-125 combined regen+film; §6.2b).  With that
lever coking is satisfiable, so the frontier is genuinely **thermal-limited**:
raising Isp wants less film (less c* penalty) but that pushes the wall toward
the coking limit — a real trade the optimiser now resolves.  ``enforced=`` can
still drop it for diagnostics.

Hard constraints only; penalties would be for feasibility restoration, not the
reported optimum (plan §8, §11 Phase-9 gate).

Robustness and verification (MDO issues audit 2026-10-16, §1.1 items 1-4)
--------------------------------------------------------------------------
* **A bad trial point is a rejected step, not an aborted solve.**  The
  compiled objective/constraint programs contain two implicit solves; a
  non-finite input reaching a Lineax linear solve raises an Equinox runtime
  error (``jax.errors.JaxRuntimeError``).  :class:`_GuardedProblem` turns a
  primal failure at a trial point into a finite *penalty* (objective above
  every finite value seen, every enforced row strongly violated), so SLSQP's
  merit line search backtracks.  A derivative failure at a point whose primal
  is finite falls back to central differences of the same compiled primal in
  the unit box.  The penalty is never a reported optimum: the returned point
  is re-evaluated from scratch, and a failure there yields ``success=False``
  with ``numerical_status="fail"``.  Configuration errors still raise.
* **Independent KKT check** (:func:`kkt_check`): stationarity, dual
  feasibility and complementarity recomputed at the returned point from the
  exact gradient and Jacobian, with multipliers from non-negative least
  squares, against the plan §11 stationarity tolerance (1e-4).  The SciPy
  ``success`` flag is no longer the only evidence of a converged optimum.
* **Multi-start** (``n_starts``): the reference start plus scrambled Sobol
  points in the unit box, compiled once and reused, with a basin count so an
  infeasible requirement set and a failed nonconvex solve can be told apart.
~~~~

### `raosim/mdo/nozzle.py`

*202 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
Typed contour-provider seam for the differentiable engine MDO.

The current production provider is deliberately only the fixed-topology
analytic TOP/Bézier construction already implemented in :mod:`raosim.mdo.grid`.
This module does not add another contour model; it gives that construction a
stable provider boundary and an explicit outcome contract so a future provider
can report failure without being silently replaced by the analytic contour.

Provider failures still carry a finite, fixed-shape ``StationGrid`` supplied by
the provider.  That grid is a shape carrier which lets the coupled evaluation
finish and report the candidate.  ``available`` and ``infeasible`` decide
whether the geometry may be accepted; consumers must never infer availability
from finite coordinates or from NaN sentinels.
~~~~

### `raosim/mdo/objectives.py`

*63 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
Explicit objective identities for the differentiable engine optimizer.

An objective name is a reporting contract, not just a scalar expression.  In
particular, electric feed hardware is not an engine dry mass, and the currently
resolved thrust-chamber-plus-feed subtotal is still not a complete engine.
~~~~

### `raosim/mdo/pintle.py`

*241 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.pintle — the pintle element as an optimizable MDO component.

``mdo/injector.py`` sizes the injector's *flow*: metering areas, momentum
ratio, blockage factor and the Son (2017) two-branch minimum area.  Before this
block the pintle had no hardware in the optimizer: its diameter reached the
objective through no path, so the optimum left it at its starting value, and
the movable-pintle transition row was enforced on a slotted post that has no
moving tip.  This block gives the pintle the same standing as the thrust
chamber: owned design variables, a differentiable hardware mass in the partial
dry-mass objective, and constraints that belong to the architecture actually
selected.

Architecture is an outer-enumeration choice (plan rule 4), selected by
``MissionSpec.pintle_architecture`` and never relaxed into a continuous value:

* ``"fixed_slotted"`` — the radial stream leaves N slots cut through a fixed
  post.  This is the architecture the five-part coaxial CAD exports and the
  first physical article (a water cold-flow pintle) uses.  The blockage-factor
  band applies; the Son transition and minimum-opening rows do not.
* ``"son_movable"`` — the pintle valve of Son et al., *Design Procedure of a
  Movable Pintle Injector for Liquid Rocket Engines*, JPP 33(4), 2017,
  DOI 10.2514/1.B36301: a continuous radial gap metered by the travel of a
  centre rod inside the post.  Its metering opening, transition and throttle
  range apply; the slot blockage factor does not.  The host exporter refuses
  CAD for this architecture, so its optimum is a numerical design only.

Design variables owned here: ``D_pintle`` (post outside diameter, already a
variable), ``t_post`` (post wall) and ``D_cg`` (centre-gap diameter, active only
for ``son_movable``; the NLP pins it for the slotted post).

Physics and its basis:

* **Hardware mass** (direct prismatic screening geometry, the basis the
  thrust-chamber Pappus/shell integration uses).  The post is an annular tube
  of outside diameter D and wall t over ``pintle_body_length_factor * D``; the
  host machined layout uses the same 3 D default body length
  (``raosim.injector_cad.resolve_machined_pintle_layout``, a repository
  heuristic).  The slotted post subtracts N slot prisms w * h * t, exactly the
  "pintle post" row of the host ``raosim.mass_ledger.injector_mass_ledger``
  (its NumPy parity oracle).  The movable valve adds its centre rod,
  pi/4 * D_pr^2 over the same length; no host oracle exists for the rod.  The
  material is the host's injector structural alloy, which the MDO already
  carries as its closeout/jacket material.  Faceplate, injector body, orifice
  plate, manifolds and seals stay outside this block, so the injector total
  remains unavailable.
* **Post pressure screen.**  sigma = p * r_o / t, the thin pressure-vessel
  (Barlow) hoop form ``raosim.interface`` already uses.  Evaluated with the
  OUTSIDE radius it bounds the elastic Lame bore stress
  p (r_o^2 + r_i^2)/(r_o^2 - r_i^2) from above for every wall thickness, so
  the screen stays conservative where a thin-shell t/r limit would not hold.
  The load is the radial stream's full injector-inlet pressure against
  ambient (an unpressurised chamber at start or shutdown), which bounds the
  steady differential chi_f * Pc.  Allowable = closeout yield / closeout
  factor of safety (SP-8087 sec. 2.1.3 yield band, already on MissionSpec).
  Slot ligaments, external pressure from the surrounding oxidizer annulus and
  thermal stress are not screened.
* **Wall fit.**  t <= 0.8 * D/2 mirrors the host layout's own wall cap
  (``injector_cad.resolve_machined_pintle_layout``), keeping a bore for the
  radial stream.
* **Valve opening** (``son_movable``).  Son 2017 reports no uniform liquid
  sheet within 0.1 mm of opening (PDF p. 2) and rejects a throttle schedule
  whose opening falls below 0.1 mm as infeasible because of nonuniform
  injection (PDF p. 10).  The opening at the lowest declared throttle level is
  therefore held above ``pintle_min_opening``.  Throttling follows the host
  ``throttle_map`` convention: the radial pressure-drop fraction is held and
  Pc scales with mass flow (Son 2017 PDF p. 10, "proportional reduction in
  the chamber pressures and mass flow rates"), so the metered radial area
  scales as f**0.5.  This is an algebraic single-point surrogate for the
  throttle requirement, not the multipoint mission of plan Phase 10.
* **Centre-gap fit** (``son_movable``).  The centre gap sits inside the post
  bore (Son 2017 Fig. 2), so D_cg <= D - 2 t.  The rod must stay inside the
  centre gap; the transition row (A_tip <= frac * A_cg, Son PDF pp. 4 and 9)
  already enforces A_cg > 0.
* **Module fit.**  Son 2017 PDF p. 9: "the calculated chamber diameter should
  be larger than the injector module diameter".  The module diameter is the
  outside diameter of the axial annulus around the post,
  sqrt(D^2 + 4 A_axial / pi).

Not-applicable rows return a positive inert value so the fixed-shape
constraint vector is unchanged; ``mdo.constraints`` marks them not applicable.
~~~~

### `raosim/mdo/propellants.py`

*373 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.propellants — propellant combinations as a first-class MDO input.

The MDO layer was originally written around a single LOX/RP-1 engine: the
coking limit, the coolant thermophysical properties, L* and the chamber gas
constants were all RP-1 values baked into ``MissionSpec``.  This module makes
the propellant a **selectable input**, so the same optimiser designs a
LOX/LH2 upper stage or an N2O4/MMH thruster without editing code.

Sources
-------
* **Chamber gases** (γ, M_w, T_c) come from the repository's existing audited
  table, ``raosim.propellants`` (Sutton Table 5-5 values) — not re-derived here.
* **L\*** from **SP-125 (Huzel & Huang) Table 4-1**, "Recommended Combustion
  Chamber Characteristic Length for Various Propellant Combinations":
  LOX/RP-1 40–50 in, LOX/LH2 30–40 in (LH2 injection) or 22–28 in (GH2),
  N2O4/hydrazine-base 30–35 in, LOX/ammonia 30–40 in, H2O2/RP-1 60–70 in.
* **Coolant wall limits** from **SP-8087**: "do not operate with liquid-wall
  temperatures above 850°F (728 K) for RP-1, or above 600°F (589 K) for
  furfuryl alcohol.  Aerozine-50 should be used at wall temperatures below
  600°F (589 K)" — RP-1's coking onset is quoted as 800–900°F (700–756 K).

Honesty about coverage
----------------------
Coking is a **hydrocarbon** phenomenon.  Hydrogen has no carbon and therefore
no coking limit at all: for LOX/LH2 the limit is ``None`` and the gas-side
wall-temperature constraint governs instead — which is the physically correct
behaviour, not a missing number.  Methane post-dates SP-125/SP-8087, so its
L\* and wall limit are flagged ``estimated=True``; treat those two numbers as
engineering estimates, not literature values.
~~~~

### `raosim/mdo/properties.py`

*1683 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.properties — C¹ shape-preserving property surfaces (Phase 2).

Plan §9: CEA/CoolProp are never autodiffed.  Chamber thermochemistry is
sampled OFFLINE on a (Pc, O/F) grid (``scripts/sample_cea_surface.py`` on a
host with RocketCEA), stored as plain arrays, and evaluated here through a
C¹ monotone tensor-product Hermite interpolant:

* 1-D node slopes by the Fritsch–Carlson (1980) monotone limiter — the same
  scheme as SciPy's ``PchipInterpolator`` (parity is test-pinned in
  tests/test_mdo_properties.py against SciPy where available);
* 2-D bicubic Hermite patches from the limited partials with zero cross
  (twist) terms — C¹ across cell edges, shape-preserving along grid lines;
* evaluation is pure ``jnp`` (jit/grad-safe; the cell index is a
  ``stop_gradient`` integer — the interpolant is C¹ so the derivative is
  continuous across cells).

Domain policy (plan rule 7): the optimizer must be *constrained* to the
tabulated box.  Evaluation clamps queries to the box only to stay finite;
``domain_violation`` exposes the (positive-when-outside) distances so the NLP
layer can enforce membership as explicit constraints instead of silently
extrapolating.

Final designs are re-evaluated on the authoritative backends (Phase 11) —
these surfaces are a differentiable *stand-in*, not the truth source.
~~~~

### `raosim/mdo/pump.py`

*174 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.pump — Phase 6: differentiable pump + electric-feed block.

Closed-form jnp (no implicit state → plain jacfwd/jacrev), mirroring the
``raosim.pumps`` duty/specific-speed/NPSH conventions but with the binned
efficiency estimator replaced by a C¹ surrogate.  Physics (plan §6.4):

* duty: Q = ṁ/ρ, required head H = Δp_pump/(ρ g0), Δp_pump the §6.4 ledger rise
  (P_c + Δp_inj + Δp_regen + Δp_line − P_tank), ω = 2π N/60;
* dimensionless specific speed Ns = ω√Q/(g0 H)^¾ and suction specific speed
  Nss = ω√Q/(g0·NPSH)^¾ with NPSH = (p_in − p_v)/(ρ g0) (``pumps.py`` L2410
  form; SP-8052 suction screen);
* **C¹ efficiency** η(Ns) — a smooth log-Gaussian peaked at Ns_opt, replacing
  ``pumps._estimate_pump_efficiency`` (a C0-discontinuous step in Q that cannot
  be differentiated — plan §6.4 note).  Calibrated to the SP-125 rocket-pump
  band (Huzel & Huang: overall efficiency 60–85 %, rising with capacity/Ns; ~10 %
  below industrial) and the SP-8109 low-Ns penalty.  It is a *smooth screening
  surrogate*, not a fit to SP-125 Fig. 6-23 (image-only);
* electrical chain: P_hyd = ṁ Δp/ρ, P_shaft = P_hyd/η_pump,
  P_elec = P_shaft/(η_motor η_inv);
* tip-speed sizing U₂ = √(g0 H/ψ) with a stress-limited screen U₂ ≤ U₂,max
  (SP-8109), and the suction screen Nss ≤ Nss,max (SP-8052) — both exposed as
  margins, never clamped;
* **battery as an epigraph** (Lee 2021): the power- and energy-limited masses
  are returned *separately* (m_b ≥ Σ P_e Δt/(η_disch ρ_E) and m_b ≥ P_e/ρ_P);
  the NLP takes m_battery ≥ both — no ``max()`` in the differentiable core
  (plan rule 5).

Parity oracle: tests/test_mdo_pump.py checks Q/H/Ns/Nss/power against the
``pumps.py`` forms where they coincide, and pins that the new η(Ns) is C¹
(continuous value AND derivative) across the old bin edges where the estimator
jumps.
~~~~

### `raosim/mdo/scaling.py`

*60 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.scaling — affine variable/residual scaling (Phase 1).

Gradient-based NLP on raw SI units mixes O(1e6) pressures with O(0.1)
fractions; the standard cure (Martins & Ning 2021, ch. 4 practice) is an
affine map to the unit box and residual scales chosen so a "well-solved"
residual is O(1).  Everything here is closed-form and jit/grad-safe.

``ScaledSpace`` is the single owner of the map:

    z = (x - lo) / (hi - lo)          in [0, 1]
    x = lo + z * (hi - lo)

State/residual scaling for the skeleton solve lives with the assembly
(each residual is nondimensionalized by its natural reference — thrust by
``F_target``, mass flow by the reference mdot, temperature by ``Tc``), which
keeps the §11 gate metric ("normalized constraint violation", "max scaled
residual") meaningful across disciplines.
~~~~

### `raosim/mdo/schema.py`

*1402 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
raosim.mdo.schema — design variables, bounds, and mission definition (Phase 1).

The MDF formulation (plan §4.1) separates:

* ``MissionSpec`` — fixed requirements and architecture constants for one
  operating point: target thrust, ambient pressure, burn time, tank states,
  efficiency/density constants for the electric feed.  Frozen dataclass of
  floats; a static argument to jitted functions.
* ``DesignVector`` — the continuous design variables ``x_h`` (skeleton subset
  of the plan's full list): ``Pc``, ``eps``, ``dp_f_frac``, ``dp_o_frac``.
  Registered as a JAX pytree so ``jit`` / ``jacfwd`` / ``jacrev`` flow through
  it with no host callbacks (the Phase-1 gate, tests/test_mdo_schema.py).
* ``VariableSpec`` / ``default_design_space`` — per-variable bounds used by
  ``scaling.ScaledSpace`` to map to the unit box.

Discrete architecture choices (channel counts, bus voltage class, movable vs
fixed pintle) are *not* variables here — plan rule 4: outer enumeration only.

Constant defaults below are repository screening assumptions. The electric
drive and battery values come from ONE registered technology set shared with
``raosim.pumps`` (:mod:`raosim.electric_technology`; default
``repository_screening_v1``, the former ``raosim.pumps`` values, with their
published brackets recorded there).  ``MissionSpec.electric_technology``
names the set and ``with_electric_technology`` selects another, e.g. the
page-verified Lee et al. 2021 Table 2 set.  ``pump_specific_mass`` stays an
MDO-only screening assumption (see its field comment).  Each placeholder
remains explicit in the §3 mass ledger until its intended duty is reviewed.
~~~~

### `raosim/mdo/state.py`

*1062 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
Versioned, pure-JAX numerical output contract for the engine MDO.

``EngineResult`` is the discipline-internal object used while evaluating the
model.  It intentionally contains Python dictionaries and rich dataclasses,
which are convenient inside a traced function but are not a stable public
output contract.  This module converts that result into nested ``NamedTuple``
pytrees containing only fixed-shape numerical leaves.

Contract rules
--------------
* Shapes depend only on :class:`~raosim.mdo.grid.GridTopology`, never on values.
* Ideal and delivered performance are both retained; efficiencies are not
  folded into ambiguous fields.
* Every profile used by cooling is retained on the same station grid.
* Battery energy and power branches remain separate.
* An unsupported quantity is ``NaN`` with a false availability entry.  The
  host-side snapshot converts that pair to ``None`` plus a reason.  A physical
  zero is never used as an availability sentinel.
* Human-readable names are module-level tuples, not JAX leaves.

Use an explicit :class:`~raosim.mdo.schema.DesignLayout` and
``DesignVector.from_active_array`` inside a jitted closure.  Passing a
``MissionSpec`` as a traced argument is neither required nor supported.
~~~~

### `raosim/mdo/structures.py`

*288 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design (stage from module_stage_map)*

~~~~text
mdo/structures.py — differentiable structural screens for the chamber shell.

NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled Combustion Chambers* (1973;
``propulsion_texts/19730022965.pdf``) §2.1.3 names exactly three structural jobs
for chamber reinforcement:

    "The three areas of structural concern are the hoop support about the
    combustion chamber, the support at the nozzle throat to resist bending and
    buckling, and hoop support about the expansion nozzle to resist collapse
    from hoop compression.  The last condition occurs during operation at sea
    level, where jet separation occurs during start and shutdown and the nozzle
    runs overexpanded during steady-state operation."

Until now the repository screened none of them at the MDO level:

1. **Chamber hoop support** — now covered by the SP-125 p.109 outer-shell hoop
   sizing in :func:`raosim.mdo.mass.closeout_thickness`.
2. **Nozzle collapse under overexpansion** — this module.
3. **Throat bending/buckling support** — this module (reported; see the caveat
   in :func:`throat_support_margin`).

The hot-gas liner has a separate, local failure mode: coolant-over-gas
pressure puts the liner in circumferential compression between adjacent ribs.
The repository's traditional pipeline already screens that rib bay as a
simply-supported long plate.  :func:`rib_supported_liner_buckling_screen` is
the pure-JAX mirror of that screening relation; it is deliberately not labeled
as SP-125 equation 4-29, which is the distinct longitudinal inelastic-buckling
case.

SP-8120, *Liquid Rocket Engine Nozzles* (1977;
``propulsion_texts/19770009165.pdf``) §2.2 records the consequence of missing
(2): *"Nozzle structural failures often are due to inaccurate predictions of the
gas expansion produced by the nozzle.  A typical failure of this kind is the
collapse of the nozzle from overexpansion during ground testing or from high
side loads generated by severe accelerations of the exhaust gas resulting from
unsteady, asymmetric separation of the expanding gases from the nozzle wall."*
SP-8120 cites NASA SP-8007 (*Buckling of Thin-Walled Circular Cylinders*) and
SP-8019 (*Buckling of Thin-Walled Truncated Cones*) as the governing buckling
monographs; those are not in the local corpus, so SP-8007's criteria are
reproduced here from the NASA Technical Reports Server copy of the August 1968
revision (NTRS 19690013955), §4.2.3.

Relationship to the separation constraint
-----------------------------------------
This is *not* a duplicate of ``separation_margin``.  Separation asks whether the
flow detaches; collapse asks whether the shell survives the external pressure it
sees while it is attached and overexpanded.  A nozzle can pass the separation
screen and still be crushed — SP-8087 is explicit that the overexpanded
steady-state condition is the design case, not just the start/shutdown transient.
~~~~

### `raosim/requirements.py`

*831 lines · stage S13 Differentiable whole-engine MDO, optimizer and requirement-driven design*

~~~~text
raosim.requirements — Layer 0: what the user asks for, not what they design.

The requirement set is not invented here.  Dieter K. Huzel and David H. Huang,
*Design of Liquid Propellant Rocket Engines* (1967), NASA SP-125, §2.1,
*The Major Rocket Engine Design Parameters* (printed p. 31,
``propulsion_texts/19710019929.pdf``, source-PDF p. 40) fixes it:

    "To fit the engine system properly into a vehicle system, engine systems
    design and development specifications will have to cover the following
    parameters above all:
    (1) Thrust level  (2) Performance (specific impulse)  (3) Run duration
    (4) Propellant mixture ratio  (5) Weight of engine system at burnout
    (6) Envelope (size)  (7) Reliability  (8) Cost
    (9) Availability (time table-schedule)"

Note what is *absent* from that list: chamber pressure, expansion ratio, ``L*``,
contraction ratio, channel geometry, pintle diameter, pump speed.  SP-125 treats
those as design outputs, which is exactly the split :mod:`raosim.mdo` already
implements.  This module supplies the missing left-hand side.

This module is **host-side**.  It never runs inside a traced computation; it
produces a :class:`~raosim.mdo.schema.MissionSpec` and a constraint selection,
and the differentiable layer takes it from there (plan rule 10).

The central discipline
----------------------
A requirement that cannot be screened must say so.  Three statuses:

``ENFORCED``
    A hard NLP constraint on the actual requirement quantity.
``PARTIALLY_ENFORCED``
    A hard NLP constraint on a **lower bound** of the requirement quantity.
    Satisfaction is necessary but not sufficient — the real quantity may still
    violate.  Reported with the reason and with what is missing.
``UNSUPPORTED``
    No screen exists.  The requirement is carried into the report and ignored
    by the optimiser.  It is never silently dropped and never reported as met.

This mirrors the output-contract rule the repository already follows: an
unavailable quantity stays unavailable rather than becoming a convenient zero
(``docs/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md`` §1).
~~~~


## S14 - Authoritative re-evaluation, provenance, release gates and run records

*10 files.*

### `raosim/design.py`
*2855 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
design.py - High-level design-gated nozzle workflow.
~~~~

### `raosim/mdo/build.py`
*363 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
The expected-part manifest and one build/artifact orchestrator.

Two competing build paths is how the repository ended up able to report a
"complete" ledger for an engine missing whole subsystems.  This module owns the
opposite contract: an engine has a **known, enumerated** part list, and every
entry is either built and measured, or unavailable with a stable reason.  A
part that was never modelled is present in the manifest as unavailable, never
absent and never zero.

The fidelity ladder is deliberately *achieved*, not requested.  A caller states
a target; the build reports what it actually reached and why it stopped there.
``L5`` is never awarded automatically -- physical qualification is not a
property any solver can confer.
~~~~

### `raosim/mdo/contour_validation.py`
*1598 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
Host-side validation of an optimized TOP contour against Rao's BVP.

This module deliberately does not implement a second inner-loop
``ContourProvider``.  The differentiable optimizer continues to use the fixed
topology TOP/Bezier wall.  After that optimizer has selected a candidate, this
module can solve the independent Rao variational/MOC boundary-value problem at
the *solved* state and compare the two contours under an explicit adjudication
policy.

The contract is fail closed:

* a convergence-domain artifact must cover the exact solve context;
* the variational solve must satisfy its numerical and MOC closure gates;
* an independently refined BDE solve must agree at common physical coordinates;
* Rao 1999 Eq. (5) must be evaluable and valid;
* performance must come from the direct complete C-D-E surface integral; and
* every geometry/performance discrepancy must be finite.

Missing evidence and failed numerical solves are ``unknown``.  A demonstrated
invalid Rao region or an exceeded discrepancy tolerance is ``fail``.  No error
path substitutes the analytic TOP wall for the variational result.
~~~~

### `raosim/mdo/execute.py`
*899 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
One entry point that runs an engine job end to end.

The point of this module is that a user should not have to know whether an
answer came from the differentiable core or the traditional pipeline.  A job
goes in; a verdict, a snapshot, and the resolved inputs that produced them come
out.  Internally the flow is always the same:

    resolve job -> solve (evaluate or optimize) -> EngineState v2
                -> resolved inputs -> authoritative traditional parity pass
                -> snapshot -> targets -> verdict

A verdict is deliberately four-valued.  "The optimizer did not converge" is not
evidence that a requirement is impossible, and "a governing model is
unavailable" is not evidence that a design is sound; collapsing either into a
boolean is how an unknown becomes a pass.
~~~~

### `raosim/mdo/postprocess.py`

*1475 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records (stage from module_stage_map)*

~~~~text
Post-optimum bridge from the differentiable MDO to authoritative outputs.

This module is intentionally outside the differentiated path.  It maps an MDO
design onto the traditional ``DesignInput`` contract, runs
``design_nozzle_v2``, optionally sizes the electric pump package from the
resulting feed ledger, and compares versioned host-side snapshots.

The bridge keeps three rules explicit:

* shared inputs and conventions are mapped, not silently defaulted;
* unavailable physics (notably film-injector hardware and complete engine mass)
  remains unavailable rather than becoming a fake zero;
* the full authoritative result and generated artifacts remain attached to the
  returned re-evaluation object.
~~~~

### `raosim/mdo/resolved_inputs.py`

*741 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records (stage from module_stage_map)*

~~~~text
One host-side definition of every input and convention both pipelines use.

``ResolvedEngineInputs`` is remediation item 9.  It is deliberately written to
be *contained by* a future ``ResolvedEngineJob`` rather than to sit beside one:
a job adds workflow intent (analyze / optimize / pareto), variable policies,
targets, and an output policy on top of these physical inputs.  Building the
two as peers would create a third competing definition of the same
conventions, which is the failure this contract exists to end.

The invariant is that neither snapshot adapter may reconstruct an input from a
default.  Every field here is resolved once, on the host, in SI, and both the
differentiable core and the traditional pipeline read the same object.  A
missing value is therefore an explicit ``None`` with a stated reason, never a
silent fallback that lets the two paths diverge.

Scope note: this is the *input* contract.  ``EngineState`` v2 remains the
numerical state and ``EngineAnalysisSnapshot`` v2 remains the output contract.
~~~~

### `raosim/mdo/run_record.py`

*146 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records (stage from module_stage_map)*

~~~~text
Isolated, atomic report publication with an explicit completeness manifest.

Report completeness and design feasibility are separate: a complete report can
describe a failed solve. No file in this package is a hardware release claim.
~~~~

### `raosim/mdo/snapshot.py`

*3719 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records (stage from module_stage_map)*

~~~~text
Versioned host-side output contract for engine analyses.

The differentiable MDO core intentionally returns JAX-native numerical
objects, while the traditional design workflow returns a rich
``ValidatedDesignResult`` containing reports and artifacts.  This module is
the non-differentiated boundary between those two worlds:

* :func:`snapshot_from_mdo` converts an ``EngineResult`` into a stable,
  JSON-ready analysis contract.
* :func:`snapshot_from_traditional` converts a ``ValidatedDesignResult`` (and
  optional electric-pump sizing) into the same contract.
* :func:`compare_snapshots` compares every common scalar and every common
  normalized axial profile.

Missing physics is data, not zero.  Every unavailable value is represented by
``SnapshotValue(value=None, availability_reason="...")``.  In particular, the
current MDO chamber and injector mass placeholders are never exported as real
zero-mass hardware.

This is deliberately a host-side module.  It may use NumPy, rebuild a station
grid for reporting, preserve Python result objects, and carry CAD/report paths;
none of it is part of a jitted or differentiated function.
~~~~

### `raosim/model_registry.py`
*787 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
Machine-readable provenance and validity metadata for design models.

The registry deliberately distinguishes a published equation from a repository
heuristic.  A citation is not, by itself, evidence that a particular use of an
equation is validated for the current geometry, propellant, scale, or operating
state.  Consumers can therefore expose the same status in JSON reports and
refuse to promote screening assumptions into hardware authority.
~~~~

### `raosim/release_readiness.py`

*478 lines · stage S14 Authoritative re-evaluation, provenance, release gates and run records*

~~~~text
Evidence-backed release-readiness gates for physical engine hardware.

Numerical checks and valid CAD topology are necessary, but neither one proves
that an engine component is safe to manufacture or test.  This module keeps
those claims separate by requiring traceable external evidence for every
physical release gate.  It deliberately never sets ``hardware_qualified``;
formal qualification remains the responsibility of the applicable engineering
authority and test organisation.
~~~~


## X01 - Differentiable JAX core (rao-bvp LM backend + MDO primitives/thermal)

*8 files.*

### `raosim/jax/__init__.py`
*28 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax — differentiable (JAX) Rao nozzle core.

This subpackage is a *separate solver backend* living inside the same repo.
It re-expresses the throat-to-contour numerics of ``raosim.rao_variational``
in JAX so the BVP residual is end-to-end autodifferentiable.  This backend is
the default for ``solve_rao_bvp``; the NumPy/SciPy path remains available via
``solver_backend="numpy"`` as the finite-difference regression oracle.

Run tag for outputs/builds produced by this backend: ``"jax"``.

IMPORTANT: float64 is required for bit-parity with the NumPy primitives.  JAX
defaults to float32, so x64 is enabled here at import time, before any array is
created.  Import ``raosim.jax`` (or anything under it) before constructing JAX
arrays elsewhere if you need x64 globally.
~~~~

### `raosim/jax/api.py`
*164 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.api — public entry points for the JAX backend.

Two layers, one boundary (JAX_DIFFERENTIABLE_PLAN.md §2): the NumPy shell
in ``rao_variational.solve_rao_bvp`` keeps owning seeding, kernel
construction, reliability gating, diagnostics, and output assembly.  This
module owns only the *inner least-squares solve* — Optimistix
Levenberg–Marquardt with the exact autodiff Jacobian of the assembled
residual (``raosim.jax.assembly``), replacing scipy's finite-difference
``least_squares``.

Bound handling: scipy's trust-region-reflective supports box bounds
natively; Optimistix LM does not.  We reparametrise through a smooth
bijection ``u = lo + (hi - lo) * sigmoid(z)`` so every iterate is strictly
inside the box (this matters: ``kernel_d_fraction``'s 0.7 cap is the
Option-2 valid-region workaround and must hold during iteration, not just
at the end).  The seed is nudged a relative ``SEED_NUDGE`` into the
interior so no ``z`` starts at ±inf; convergence is judged on the
*residual*, which is reparametrisation-invariant.

``solve_rao_bvp_jax(config)`` is sugar for
``solve_rao_bvp(replace(config, solver_backend="jax"))`` and returns the
identical ``RaoSolution`` structure (same reliability ladder, same
diagnostics), so export/plotting/CLI consumers are unchanged.
~~~~

### `raosim/jax/assembly.py`
*644 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.assembly — the full assembled Rao BVP residual in JAX (J2/J3b).

This is the JAX re-expression of the NumPy residual assembly:

    _unpack_bvp -> _rao_bvp_residual_groups -> RaoResidualGroups.flat()

i.e. exactly what ``scipy.optimize.least_squares`` minimises in
``rao_variational.solve_rao_bvp`` (rao_variational.py:3559).  The J2 gate
(JAX_DIFFERENTIABLE_PLAN.md §7) is 1e-8 parity of :func:`make_residual`'s
output against ``_scaled_rao_bvp_residual`` on identical packed unknown
vectors — proving the port changes numerics, not physics.

Faithfulness notes (each mirrors a specific NumPy code path):

* CE flow nodes clamp ``M`` at **1.001** (``_control_surface_flow_nodes``);
  kernel-BD / wall-interp nodes clamp at **1.000001**
  (``nasa_moc.bd_segment_at_fraction``, ``_interp_wall_at_fraction``).
  The two clamps are deliberately NOT unified here.
* ``x_ce`` is reconstructed from the left-Mach ODE ``dx = dr/tan(θ+μ)``
  with the NumPy fallback step ``1e-9`` when ``|tan| < 1e-12``
  (``_integrate_ce_x_from_left_mach``; M floor **1.000001** there).
* The kernel BD is a *static* polyline during the solve (it is built once
  by the NumPy seed path); only ``kernel_d_fraction`` is an unknown.  The
  B→D mass flux is therefore evaluated as a masked partial-segment sum,
  smooth in ``kernel_d_fraction`` — this replaces
  ``calc_mdot_bd``/``bd_segment_at_fraction`` without changing values.
* Mass-flux midpoint clamps: CE-side **1.001**
  (``rao_variational.curve_mass_flux``), BD-side **1.000001**
  (``nasa_moc.curve_mass_flux``); both floor midpoint r at 1e-12 and skip
  segments with ds <= 1e-12.
* Group order and scaling are byte-for-byte ``RaoResidualGroups.flat()``.

Static configuration (node counts, active blocks, targets, chart angles,
the kernel BD arrays) is baked into a closure by :func:`make_residual`;
the returned function is pure ``fn(u) -> r`` and jit/jacfwd/jacrev-safe.

Unsupported (raise at build time, not silently wrong): residual blocks
``stationarity``/``transversality`` (finite-difference reference blocks,
not in the default stack) and ``angle_boundary_mode != "free"``.
~~~~

### `raosim/jax/moc_kernel.py`
*828 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.moc_kernel — differentiable throat-arc kernel march (J3b).

Plan reference: JAX_DIFFERENTIABLE_PLAN.md §4.2 / §5.  Ports the
*Anderson-form* axisymmetric unit processes from ``raosim.moc`` —
``solve_interior_point`` / ``solve_axis_point`` / ``solve_wall_point``
with the CORRECTED characteristic pairing (2026-06-11; Anderson MCF
§11.4, Zucrow & Hoffman Vol. 2 Ch. 17, oracle-proven in
``tests/test_characteristic_pairing.py``)::

    along C− (slope θ−μ):  d(θ+ν) = +S ds
    along C+ (slope θ+μ):  d(θ−ν) = −S ds,      S = sinθ sinμ / r

This module reaches BIT-PARITY with the NumPy oracle ``build_kernel``
(M3.5Perf-validated) at the BD level — it ports NASA's *exact* row
policy and dθ-form unit processes (below), not merely the same physics.
The Anderson-form processes (``interior_point`` / ``axis_point`` /
``wall_point_arc``) are kept as tested reference ports of the corrected
``raosim.moc`` unit processes, but the MARCH uses the dθ-form ports
(``*_nasa``) because BD-oracle agreement requires NASA's near-axis
coefficient treatment (the Anderson midpoint-source chain converges to
a *different* continuum near the axis: axis Mach +9%, resolution-
divergent).

March structure (the J3b θ_B-differentiability design)
------------------------------------------------------
The throat expansion fan keeps emitting new C+ characteristics at the
wall corner, so the kernel mesh genuinely GROWS toward BD.  NASA limits
the wall step to ``dtheta_limit`` by MIXING two row types
(``_calc_arc_wall_point_with_special``):

* RAW rows (~5%: marched-row indices 2/9/21/39/65 at the reference) —
  the C+ leaving prev[1] TERMINATES on the throat arc and becomes the
  new wall point (every C+ in the fan ends on the wall; this is
  physics, not bookkeeping).  Width is unchanged; the interior C+
  pairing shifts by one (``ii = i + 1``).
* SPECIAL rows (~95%) — when the raw step would exceed ``dtheta_limit``
  (or overshoot θ_B) a prescribed-angle wall point is inserted at
  ``min(θ_B, θ_prev + dtheta_limit/2)``; the row GROWS by one and the
  pairing is same-index (``ii = i``).

Skipping the raw rows (an all-special march) starves the wall-adjacent
strip — measured ν_w − θ_w → −0.67° at B where the oracle's is +0.68°
(axisymmetric wall expansion must run AHEAD of planar PM).  Two other
static-shape designs failed earlier: a static-width forward march
blows up (no step limit; wall steps 0.56°→2.3°→5.4°→NaN); a regridded
inverse march is weakly unstable (axis Mach runaway).

The landed march carries NASA's policy in static PADDED shapes
(``W = n0 + n_rows``): the raw/special decision, valid row width
(``axis_idx``), and pairing offset are TRACED; padded slots hold the
row's axis state (benign finite sentinels so masked lanes never
produce NaNs — the §5 where-grad trap).  Marching freezes once the
wall reaches θ_B; the scan carry at the end IS the BD row.

θ_B-differentiability: the pre-clamp wall grid (raw landings + fixed
``dtheta_limit`` special steps) is θ_B-INDEPENDENT, so
``d(BD)/d(theta_B)`` flows only through the final clamped row — exact
and smooth within a row-count window; the window boundary (θ_B
crossing a grid step) is handled by per-rung re-assembly exactly like
``kernel_bd`` re-seeding today.  ``n_rows`` is a static upper bound on
the row count (frozen rows no-op).

The transonic start line is taken as INPUT ARRAYS (wall-first), e.g.
from the NumPy seed kernel's TT' row: the start line is
θ_B-independent, so ``d/d(theta_B)`` is exact with a frozen start line.
(Differentiable Kliegel–Levine for d/d(Rt, γ) totals is the J3b-2 /
args-lifting increment, tracked in the plan.)

Unit-process parity contract: the Anderson-form ports at ``n_corr=10``
reproduce the NumPy ``raosim.moc`` fixed point (early break off) to
float64 round-off; the dθ-form ``*_nasa`` ports match the NumPy NASA
processes at their 1e-10 fixed points — both pinned in
``tests/test_jax_moc_kernel.py``.
~~~~

### `raosim/jax/primitives.py`
*178 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.primitives — gas-dynamics leaf primitives in JAX.

These mirror, term for term, the calorically-perfect closed forms in
``raosim.gas_dynamics`` (which are themselves cross-checked against
``propulsion_texts/prmeyer.pdf`` and ``rao1999.pdf``).  Every Rao/MOC residual
differentiates through this module, so the J1 gate is bit-parity with the NumPy
versions (1e-10 on closed forms; round-trip consistency on the two iterative
inverses).

Sonic-safety: ``mach_angle`` and ``prandtl_meyer`` are singular/branch-sensitive
at M -> 1.  Arguments to ``sqrt``/``asin`` are guarded so values and first
derivatives stay finite at the throat.  See JAX_DIFFERENTIABLE_PLAN.md §4.1.

x64 is enabled by ``raosim.jax``'s package __init__; importing this module pulls
that in.
~~~~

### `raosim/jax/sensitivities.py`
*252 lines · stage X01 Differentiable JAX core*
~~~~text
J6 v1 — exact gradient API over the solved Rao BVP (plan §6).

Scope (v1, honest about what is differentiated):

* **Node-level tolerance fields** — exact reverse-mode gradients of the
  control-surface thrust coefficient with respect to every solved
  unknown: ``dCf_dM / dCf_dtheta / dCf_dr`` per CE/DE node, plus the
  scalar tail (``dCf_dkdf`` — D sliding along the frozen kernel BD —
  and the multipliers, which must come out ~0 at a converged optimum).
  The per-node ``dCf_dr`` is the plan's "manufacturing-tolerance map"
  evaluated on the control surface: how much Cf moves if the surface
  deviates at node i *without re-solving* — exactly the right object
  for a tolerance interpretation.
* **Explicit design partials at fixed u*** — ``dCf_dpa`` and
  ``dCf_dgamma`` through the functional's explicit dependence.  These
  are *partial*, not total, derivatives: they exclude the implicit
  shift of the solution u*(p).  ``dCf_dpa`` is analytically
  ``-(r_E^2 - r_D^2)/Rt^2`` (the trapezoidal pressure term telescopes),
  which the test suite pins as the J6 known-sign/value gate.
* **Jacobian diagnostics** — exact ``jacfwd`` Jacobian of the residual
  at u*, with its extreme singular values (conditioning of the
  converged system).

Deferred to v2 (documented, not silently missing):

* Total design derivatives ``d(outputs)/d(Rt, epsilon, length_pct,
  gamma)`` via the implicit function theorem through
  ``optimistix.least_squares`` — needs the assembly's solve constants
  lifted into a traced ``args`` pytree, and (for anything reaching the
  kernel: Rt, Rd, theta_B) the J3b differentiable march, since the
  kernel BD is a frozen NumPy artifact today.
* ``dCf_dwall`` on the *bell wall* nodes — needs the BDE region march
  differentiable (same J3b family).
* ``hessian_thrust`` — the §11 soft-mode study; ``jax.hessian`` of
  :func:`cf_from_u` slots in once wanted.

Cf convention: :func:`cf_de_jax` is a line-for-line ``jax.numpy`` port
of :func:`raosim.nasa_moc.surface_thrust_coefficient` (momentum +
pressure flux through the DE control surface, normalised by
``pi * Rt^2 * p0``), so the parity test can hold to ~1e-12 in x64.
~~~~

### `raosim/jax/thermal.py`
*154 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.thermal — differentiable nozzle thermal/performance physics.

JAX (jnp) ports of the design-driving physics so they enter a
gradient-based constrained design loop (raosim.jax.design_opt):

* full Bartz (1957) gas-side h_g + σ  (mirror of physics.bartz_*)
* Sieder-Tate coolant-side h_c          (mirror of physics.sieder_tate_*)
* the series gas/wall/coolant circuit -> throat wall temperature
* Schmucker separation margin            (mirror of separation.schmucker_*)
* ambient thrust coefficient             (primitives.thrust_coefficient)

The constraints evaluate at closed-form stations — the throat (M=1,
A_t/A=1) for cooling and the exit (isentropic Me from ε) for
separation — so no contour march is needed in the differentiated path.
Parity with the NumPy originals is the J-style acceptance gate
(tests/test_jax_thermal.py); every function is jit/grad-safe.
~~~~

### `raosim/jax/theta_b_solve.py`
*243 lines · stage X01 Differentiable JAX core*
~~~~text
raosim.jax.theta_b_solve — θ_B as a solved unknown inside the BVP (J3b-2).

Replaces the seed secant's frozen kernel with the differentiable march:
the unknown vector gains one trailing component θ_B, and the residual
recomputes the kernel BD *in-graph* via :func:`raosim.jax.moc_kernel.
march_kernel` each evaluation — mass-closure target, D-state pins, and
the BD-flux scale all read the live BD(θ_B).

Wiring strategy (zero changes to the J2-parity-gated assembly):
``assembly.make_residual`` reads the BD polyline through
``jnp.asarray(sp.bd_*)``, and ``StaticParams`` is a plain NamedTuple —
so a ``_replace`` with TRACED arrays flows through the existing
masked formulations untouched.  The march's padded BD (tail = exact
axis-state repeats) is safe by construction: padding segments have
ds = 0, which ``bd_flux_to_fraction`` / ``_polyline_mass_flux`` already
mask (``ds > 1e-12``), and the arc-length cumsum plateaus so
``kernel_d_fraction`` keeps parametrising the valid arc only.

Correctness gate (pinned in tests/test_jax_theta_b_solve.py): at the
seed θ_B the live-BD residual equals the static-``kernel_bd`` residual
to ~1e-8 — automatic, because the march reproduces the NumPy seed
kernel's BD at bit parity (tests/test_jax_moc_kernel.py).

θ_B window: the march's pre-clamp wall grid (raw landings + fixed
dθ-limit special steps) is θ_B-independent, so d(residual)/dθ_B is
exact and smooth while θ_B stays within its current grid interval;
crossing a dθ-limit-dependent grid boundary changes the clamp-row index (a small
discontinuity).  Solver bounds default to ± half a special step around
the seed; per-ladder-rung re-assembly re-centres the window exactly
like ``kernel_bd`` re-seeding today.
~~~~


## X02 - Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export (export-openfoam-spray verb; parcel solver is API-only, imported by spray_coupling for handoff types)

*17 files.*

### `raosim/openfoam/__init__.py`
*54 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Deterministic external-CFD case contracts.

The package writes version-pinned OpenFOAM inputs but never imports or runs
OpenFOAM.  Generated cases therefore remain software-unverified until the
explicit mesh, execution, convergence, and validation gates are supplied.
~~~~

### `raosim/openfoam/case.py`
*1642 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Version-pinned OpenFOAM VOF case generation for movable-pintle studies.

This module deliberately separates two stages in Radhakrishnan et al. (2018):

* Section 2.3 used VOF with a *water-only* mass-flow inlet to obtain liquid
  sheet thickness.  The surrounding phase was initially quiescent air.
* Section 2.4 used the 3.03 g/s annular-air stream as the Eulerian carrier for
  the later Lagrangian droplet calculation.

The first template implemented here is a reduced external-gap wedge.  It maps
the mechanical opening to a radial water boundary and is useful for checking
interface transport and external sheet evolution.  It does not contain the
paper's internal center-gap turning passage, so it cannot claim to predict the
sheet at formation.  That limitation is machine-readable in every manifest.

OpenFOAM is an external solver.  Builders in this module are pure and
deterministic; :func:`write_openfoam_case` only writes text files and never
spawns a process.  The target is OpenFOAM Foundation v13 at patch/tag
``20260624`` using ``foamRun`` with ``solver incompressibleVoF``.
~~~~

### `raosim/openfoam/handoff.py`
*869 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Fail-closed contract for an OpenFOAM VOF-to-parcel handoff.

This module deliberately does not import OpenFOAM bindings, read a case, or run
subprocesses.  It records the evidence produced by a separate VOF extraction
workflow and determines whether that evidence is strong enough to construct the
radial-sheet source used by :mod:`raosim.spray`.

All dimensional values are SI.  In particular,
``full_sheet_thickness_mean_m`` is the *full* liquid-sheet thickness (the
movable-pintle opening convention), never a half-thickness or hydraulic
diameter.  A handoff is usable only when every derived and caller-supplied
required gate passes.
~~~~

### `raosim/spray/__init__.py`
*124 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Deterministic, physics-explicit Lagrangian spray modelling.

The package is intentionally opt-in.  Importing it does not replace the legacy
injector correlation screen, and parcel results are not eligible for cycle
coupling until their conservation and declared literature-validation gates
pass.
~~~~

### `raosim/spray/benchmarks.py`
*735 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Provenance-first spray benchmark fixtures and readiness gates.

This module deliberately keeps experimental validation targets separate from
author CFD outputs.  In particular, Radhakrishnan et al. (2021) Tables 7 and 8
are VOF/Lagrangian simulation results.  They are useful literature-reproduction
fixtures, but they are never promoted to experimental evidence here.
~~~~

### `raosim/spray/breakup.py`
*703 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Versioned WAVE/KH-RT breakup primitives for Lagrangian spray parcels.

The Kelvin-Helmholtz equations follow the Reitz WAVE formulation used by
Radhakrishnan et al. (2018, 2021).  Three coefficient sets are deliberately
kept separate:

``reitz_1987``
    The coefficients implemented by the OpenFOAM Foundation ReitzKHRT model:
    0.4 in the Taylor-number factor and 0.865 in the Weber-number factor.

``radhakrishnan_2018``
    The coefficients printed in Radhakrishnan et al. (2018), Eq. (5):
    0.4 and 0.87 respectively.

``radhakrishnan_2021``
    The coefficients printed in Radhakrishnan, Lee, and Koo (2021), Eq. (3):
    0.45 and 0.87 respectively.

The optional Rayleigh-Taylor branch reproduces the equations and timer logic
of the OpenFOAM Foundation ReitzKHRT implementation.  It is disabled by
default because Radhakrishnan et al. use the WAVE/KH branch, not KH-RT, in
their modelling-2 calculation.

This module represents breakup by reducing the diameter of the droplets in a
parcel and increasing parcel multiplicity so represented liquid mass remains
constant.  Velocity is unchanged by the breakup primitive, hence represented
parcel momentum is also an identity.  Any aerodynamic momentum exchange is a
separate one-way carrier source demand and is not a globally closed carrier
momentum balance here.
~~~~

### `raosim/spray/carrier.py`
*354 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Explicit steady carrier fields for parcel tracking.

The rectilinear field stores cylindrical velocity components on an axisymmetric
``(axial, radial)`` grid and performs bilinear interpolation without hidden
extrapolation.  Cartesian parcel coordinates use column 0 as the engine axis;
columns 1 and 2 span the radial plane.
~~~~

### `raosim/spray/dispersion.py`
*161 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Seeded discrete-random-walk turbulent parcel dispersion.

Each parcel interacts with one isotropic turbulent eddy for
``tau_e = C_L k / epsilon``.  At eddy expiry a new velocity fluctuation is
sampled with component standard deviation ``sqrt(2 k / 3)``.  The model owns a
local generator created by :func:`numpy.random.default_rng`; it never touches
NumPy's process-global random state.
~~~~

### `raosim/spray/domain.py`
*215 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Axisymmetric parcel-tracking domain and exact segment boundary events.

The chamber/nozzle wall is represented by a piecewise-linear radius
``r_wall(x)``.  Parcel trajectories remain Cartesian with ``x`` as the engine
axis.  Boundary events are found along each straight integration segment so a
parcel that crosses both a wall and the outlet during one numerical step is
assigned to the first physical boundary it reaches.
~~~~

### `raosim/spray/drag.py`
*141 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
One-way spherical-parcel drag using the Schiller--Naumann correlation.
~~~~

### `raosim/spray/evaporation.py`
*354 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Explicit Spalding evaporation primitives for Lagrangian spray parcels.

Radhakrishnan, Lee, and Koo (2021), Eq. (16), give the positive
evaporation-rate magnitude

``m_dot = k_c A_d rho_g ln(1 + B_m)``.

The parcel derivative implemented here is negative.  The mass-transfer
coefficient is closed only through a caller-selected, named Sherwood model;
diffusivity, densities, viscosity, relative speed, and Spalding number are all
required explicitly.  No propellant or thermodynamic defaults are inferred.

For ``k_c = Sh*D/d``, freezing ``Sh`` over a step integrates the corresponding
``d^2`` equation exactly.  Re-evaluating ``Sh`` on subsequent calls gives a
first-order march when Reynolds-dependent Sherwood closures are selected.
~~~~

### `raosim/spray/handoff.py`
*513 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Typed, gate-derived handoff from parcel results to cycle coupling.

The legacy fixed point accepts the injector correlation screen.  This module
defines the stronger contract a future Lagrangian fixed point must consume.
It distinguishes liquid parcels from gas carrier streams and refuses to turn
a finite vaporization fraction into cycle authority by itself.

The builder is useful now for complete provenance reports, but the presently
implemented one-way/no-energy solver and public benchmark fixtures make every
such handoff cycle-ineligible.  That is an evidence result, not a hard-coded
caller-selectable flag.
~~~~

### `raosim/spray/ledger.py`
*223 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Mass and momentum accounting for represented Lagrangian droplets.

The parcel solver currently accepts a prescribed one-way carrier field.  It
therefore closes the *parcel* momentum equation and reports the equal/opposite
carrier impulse as a source demand, but it must not claim globally coupled
momentum conservation until an Eulerian carrier actually consumes that source.
~~~~

### `raosim/spray/primary.py`
*994 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Geometry-dispatched, deterministic primary parcel source initialization.

This module distinguishes the movable-pintle *radial liquid sheet* used by the
Radhakrishnan Lagrangian/WAVE workflow from the injector geometries currently
implemented in :mod:`raosim.injector`.  Only that radial-sheet geometry is
declared eligible for the literature-calibrated primary path.  Axial annuli,
rectangular slots, and round holes can be converted into geometric source blobs
for downstream drag/secondary-breakup studies, but this module deliberately
does not invent LISA, slot-sheet, or round-jet primary-breakup correlations for
them.

Coordinates are Cartesian SI: ``x`` is the engine/chamber axis and ``y,z`` span
the radial plane.  For radial-sheet tip angle ``alpha`` the convention is

``alpha = 0 deg``: purely radial, perpendicular to the axial carrier gas;
``u_x = U sin(alpha)`` and ``u_r = U cos(alpha)``.
~~~~

### `raosim/spray/solver.py`
*944 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Deterministic one-way Lagrangian parcel march.

This module connects the geometry source, prescribed Eulerian carrier field,
WAVE/KH breakup, turbulent dispersion, Spalding evaporation, trajectory
events, droplet statistics, and conservation ledgers.  It is intentionally a
mid-tier engineering model rather than a CFD solver:

* the carrier is prescribed and does not consume parcel source terms;
* droplet temperature and carrier energy are not solved;
* primary-model and literature-readiness gates remain visible; and
* consequently a successful numerical march is not automatically eligible to
  alter the engine cycle.

All dimensional inputs are SI.  No propellant transport property is inferred.
~~~~

### `raosim/spray/statistics.py`
*215 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Weighted droplet-cloud statistics for Lagrangian spray parcels.

Parcel ``weight`` is the physical droplet multiplicity represented by each
computational parcel.  Sauter mean diameter therefore uses number weighting,

``d32 = sum(weight*d**3) / sum(weight*d**2)``,

while Rosin-Rammler fitting uses liquid-mass weighting.  Radhakrishnan et al.
(2021, Eq. following Table 7) write ``exp(-(d/d_bar)**n)``; that expression is
the mass *survival* fraction, not a conventional increasing CDF.  The naming
below keeps that distinction explicit.
~~~~

### `raosim/spray/types.py`
*342 lines · stage X02 (module_stage_map: Opt-in Lagrangian spray mid-tier and OpenFOAM VOF case export)*

~~~~text
Core, dependency-light contracts for deterministic parcel-spray models.

The spray package uses SI units throughout.  These types intentionally contain
no guessed thermophysical properties: every property needed by a selected
physical model must either be supplied here or provided by a separate property
backend.  ``frozen=True`` is reinforced by copying NumPy inputs into read-only
arrays so callers cannot mutate a solved state through an array alias.
~~~~


## X03 - Legacy toolbox CLI (RaoRocketSimLegacy -> main.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)

*8 files.*

### `docs/archive/Rocket_nozzle_sim_phase2.py`

*375 lines · stage OFFLINE (per module_stage_map; not a numbered stage, a legacy standalone prototype kept in docs/)*

~~~~text
rao_bell_toolbox.py – interactive early-design tool
• Solves ε from Pc/Pa (or vice-versa)
• Optimises θₙ, θₑ for axial thrust (Rao 1961)
• Generates and plots a reduced-length Rao bell
   (60 %, 80 % or 90 % of full length)
PyPI deps: numpy, matplotlib, requests
~~~~

### `main.py`

*1280 lines · stage X03 Legacy toolbox CLI*

~~~~text
main.py – legacy CLI for the LREKit/Rao bell nozzle design toolbox.

Usage:
    lrekit                                # interactive mode
    lrekit --help                         # show all flags
    lrekit --Rt 25 --Pc 60 \\
        --propellant LOX/LCH4 --epsilon 12 \\
        --output nozzle.csv               # batch mode (no prompts)
    lrekit --sweep epsilon 4 50 20        # parameter sweep
~~~~

### `raosim/altitude_performance.py`
*163 lines · stage X03 Legacy toolbox CLI (RaoRocketSimLegacy -> main.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)*

~~~~text
altitude_performance.py – Engine performance vs altitude map.

Sweeps altitude from sea level to vacuum, shows how thrust, Isp, and Cf
change as ambient pressure drops.  Marks the altitude at which flow
separation starts/stops.
~~~~

### `raosim/benchmarks.py`
*1159 lines · stage X03 Legacy toolbox CLI (RaoRocketSimLegacy -> main.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)*

~~~~text
benchmarks.py - Literature-backed diagnostic benchmark runner.

The benchmark suite is intentionally evidence-first: it records what published
cases say, compares the current solver against those references, and separates
strict pass/fail checks from diagnostic xfail/report-only physics gaps.
~~~~

### `raosim/build_log.py`
*139 lines · stage X03 Legacy toolbox CLI (RaoRocketSimLegacy -> main.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)*

~~~~text
build_log.py – Versioned build output with metadata logging.

Every run of main.py stores its artefacts (CSV, STL, sweep data) in a
monotonically increasing versioned directory under ``builds/``.

Directory naming scheme::

    builds/v001_20260302_121700/
    builds/v002_20260302_130015/
~~~~

### `raosim/conical.py`
*187 lines · stage X03 Legacy toolbox CLI (RaoRocketSimLegacy -> main.py: sweeps, benchmarks, versioned builds/vNNN + metadata.txt)*

~~~~text
conical.py – Conical nozzle contour and divergence correction.

The conical nozzle is the simplest diverging nozzle design and serves as
the baseline reference for bell nozzle length (the "80% bell" convention
means the bell is 80% of the length of a 15° half-angle cone with the
same area ratio).

Divergence correction (Rao 1961):
    η_div = (1 + cos α) / 2

For α = 15°, η_div ≈ 0.9830 → the 1D Cf decrement is ~1.7%.

References:
  - G. V. R. Rao, "Recent Developments in Rocket Nozzle Configurations,"
    ARS Journal (1961)
  - NASA SP-8120, "Liquid Rocket Engine Nozzles" (1976)
~~~~

### `raosim/nozzle_comparison.py`
*274 lines · stage X03 Legacy toolbox CLI*

~~~~text
nozzle_comparison.py – Compare different nozzle contour methods.

Provides utilities to:
  - Compute 2D divergence loss from exit-plane flow profiles
  - Calculate nozzle efficiency η_n = Cf_net / Cf_ideal
  - Generate side-by-side comparison tables and overlay plots

References:
  - NASA report on "Perfect Bell Nozzle Parametric Optimization Curves" (1983)
  - G. V. R. Rao, "Recent Developments in Rocket Nozzle Configurations" (1961)
~~~~

### `raosim/trade_study.py`
*165 lines · stage X03 Legacy toolbox CLI (from module_stage_map)*
~~~~text
trade_study.py – Parameter sweep and trade-study plotting.

Sweep one design variable while holding others constant.  Returns results
as a list of dicts (easily convertible to a DataFrame) and produces
multi-panel comparison plots.
~~~~


## X04 - Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling

*22 files.*

### `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCDlg.cpp`

*256 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
// 3D_MOCDlg.cpp : implementation file
//
~~~~

(only a short MFC wizard file-name comment; no prose banner.)

Inferred role (inferred from the code, not from a header): MFC dialog for the 3D MOC design tool: collects inputs, launches the C3D_MOCGrid solve thread, reports progress.

### `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCGrid.cpp`

*2789 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			C3D_MOCGrid.cpp
*
*	Start Date:			5/14/02
*
*	Last Modified:		5/14/02
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG800XXX (NASA GRC Nozzle Work)
*
*	Comments:			This is the 3D_MOC class. It uses the reference plane
*						method to do the solution.  The best write up I have 
*						found on this is
*						WC Armstrong, "A Method of Characteristics Computer
*						Program for Three-Dimensional Supersonic Internal 
*						Flows," AEDC-TR-78-68.			
*
*						Inputs: Nozzle geometry
*						Outputs: 3d nozzle data in TECPLOT format
* 
*	Last Update:		Initial Start	
*						10.10.02 Implemented an equally spaced initial plane
*							Changed _pt from 3 dimensions to 1	
*							Added _nPTS global to track total number of reference plane points			 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/3D_MOCGrid.hpp`

*152 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			3D_MOCGrid.hpp
*
*	Start Date:			5/14/02
*
*	Last Modified:		5/14/02
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG800XXX (NASA GRC Nozzle Work)
*
*	Comments:			This is the 3D_MOC class			
* 
*	Last Update:		Initial Start					 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/3D_MOC/newt.cpp`

*82 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
(no header comment)
~~~~

Inferred role (inferred from the code, not from a header): Numerical Recipes Newton-Raphson with line search (NR::newt), calling funcvDeriv from 3D_MOCGrid.cpp for the analytic Jacobian; used to locate body points.

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/Chart.cpp`

*649 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
// Chart.cpp : implementation file
// Originator : Kris Jearakul  tuktun@hotmail.com
//				http://krisj.iwarp.com
// Comment	  : Use with your own risk !!
~~~~

(only a short MFC wizard file-name comment; no prose banner.)

Inferred role (inferred from the code, not from a header): Third-party MFC 2D chart control (CChart, by Kris Jearakul) used for the in-GUI plots; not nozzle physics.

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp`

*3754 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			MOC_Grid.cpp
*
*	Start Date:			7/27/01
*
*	Last Modified:		3/11/02
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG888BXX (NASA GRC PDE Work)
*
*	Comments:			This class will calculate the MOC grid for a nozzle				
* 
*	Last Update:		Added Cone option						 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.h`

*201 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			MOC_GridCalc_BDE.h
*
*	Start Date:			7/27/01
*
*	Last Modified:		7/27/01
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG888BXX (NASA GRC PDE Work)
*
*	Comments:			This class will calculate the MOC grid for a nozzle				
* 
*	Last Update:		Initial StartUp							 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE_IO.cpp`

*760 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			MOC_GridCalc_BDE_IO.cpp
*
*	Start Date:			3/8/02
*
*	Last Modified:		2/23/03
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG888BXX (NASA GRC PDE Work)
*						FG1CAXXX (URETIS)
*
*	Comments:			This is the part of the class that outputs everything				
* 
*	Last Update:		2.23.03	Added a TDK rao.dat data file output							 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridDlg.cpp`

*731 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
// MOC_GridDlg.cpp : implementation file
//
~~~~

(only a short MFC wizard file-name comment; no prose banner.)

Inferred role (inferred from the code, not from a header): MFC dialog for the 2D MOC nozzle design GUI: collects inputs, runs MOC_GridCalc_BDE, plots, and writes a $$.bat to launch STT2001 (see readme.txt, lines 694-730).

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/dummyStruct.h`

*37 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/***************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-115
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			MOC_Grid.cpp
*
*	Start Date:			8/3/01
*
*	Last Modified:		8/3/01
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			FG888BXX (NASA GRC PDE Work)
*
*	Comments:			This file contains a dummy structure (dS) that
*						contains arrays of double and int.  It can be used
*						to return more that one variable in a function			
* 
*	Last Update:		Initial StartUp							 
*						
*
****************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/engineering_constants.hpp`

*144 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
/**********************************************************************************************
*
*					The Johns Hopkins University Applied Physics Laboratory
*					        Research and Technology Development Center
*						    Aeronautical Science and Technology Group
*							                  MS 10A-118
*							             Johns Hopkins Road
*								           Laurel, MD 20723
*								             443-778-3863
*								          443-778-5850 (FAX)
*
*
*	Program:			engineering_Constants.hpp	
*
*	Start Date:			03/02/00
*
*	Last Modified:		03/02/00
*
*	Author:				Tharen Rice   x3863
*
*	Sponsor:			IRAD X83RX5XX
*
*	Comments:			This program will include various engineering constants
*						that are used regularly.  Most are taken from the CRC
*						books
*
*
***********************************************************************************************/
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/readme.txt`

*2 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
In order for this program to be able to run the Streamline Tracing Tool directly from its GUI, the MOC_GridDlg.cpp file contains code to generate a batch file "$$.bat" on lines 694-730. The relative references to the STT2001 folder (line 706) may have to be changed to work on your machine.
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/README.md`

*20 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
# Three-Dimensional-Nozzle-Design-Code

This package contains 3 programs written for the Microsoft Visual C++ compiler:

1) 2D Method of Characteristics, MOC_GRID_BDE
	- Computes a planar or axisymmetric two-dimensional nozzle contour based on user inputs using method of characteristics
	
2) Streamline Tracing Tool, STT2001
	- Performs streamline tracing on a supplied nozzle solution grid (usually generated by MOC_GRID_BDE)

3) 3D Method of Characteristics Design Tool, 3D_MOC
	- Computes three-dimensional flowfield properties for a given 3D nozzle contour. This could be a modified version of a nozzle contour generated by MOC_GRID_BDE.

	

Each program is contained in a separate folder containing source code and sample outputs. These programs are designed to use the Microsoft Foundation Class (MFC) libraries to generate the GUI windows for user input.

Detailed descriptions of the mathematical basis and operation of each program are provided in the attached report: 

**[Rice, T., "2D and 3D Method of Characteristic Tools for Complex Nozzle Development," JHU/APL Report RTDC-TPS-481. 2003.](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20030067852.pdf)**
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Chart3d.h`

*65 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
// Chart3d.h: interface for the CChart3d class.
// Originator : Kris Jearakul  tuktun@hotmail.com
//				http://krisj.iwarp.com
// Comment	: Use with your own risk !!
//
//////////////////////////////////////////////////////////////////////
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Matrix.cpp`

*532 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
(no header comment)
~~~~

Inferred role (inferred from the code, not from a header): Matrix class implementation: constructors, sub-matrix extraction, arithmetic and solve helpers used by the Streamline Tracing Tool.

### `Three-Dimensional-Nozzle-Design-Code-master/STT2001/STT2001Dlg.cpp`

*3117 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
// STT2001Dlg.cpp : implementation file
//
~~~~

(only a short MFC wizard file-name comment; no prose banner.)

Inferred role (inferred from the code, not from a header): MFC dialog and implementation of the Streamline Tracing Tool: reads a supplied MOC solution grid, traces streamlines, writes outputs (large file, ~3100 lines; reads .inp files).

### `Three-Dimensional-Nozzle-Design-Code-master/STT2001/Vector.hpp`

*69 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
(no header comment)
~~~~

Inferred role (inferred from the code, not from a header): Vector class (dense double array, 2-D/3-D constructors, friend of Matrix) used for streamline-tracing linear algebra.

### `Three-Dimensional-Nozzle-Design-Code-master/STT2001/readme.txt`

*3 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
Streamline Tracing Tool

Execution of this program will bring up a dialog window where inputs may be specified manually. Alternatively, a *.inp file may be selected using "File>>Open" to run the program with previously entered inputs. Similarly, a *.inp file may be saved for later re-use using "File>>Save As".
~~~~

### `Three-Dimensional-Nozzle-Design-Code-master/license.txt`

*241 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
NASA OPEN SOURCE AGREEMENT VERSION 1.3

THIS OPEN SOURCE AGREEMENT ("AGREEMENT") DEFINES THE RIGHTS OF USE,
REPRODUCTION, DISTRIBUTION, MODIFICATION AND REDISTRIBUTION OF CERTAIN
COMPUTER SOFTWARE ORIGINALLY RELEASED BY THE UNITED STATES GOVERNMENT AS
REPRESENTED BY THE GOVERNMENT AGENCY LISTED BELOW ("GOVERNMENT AGENCY").
THE UNITED STATES GOVERNMENT, AS REPRESENTED BY GOVERNMENT AGENCY, IS AN
INTENDED THIRD-PARTY BENEFICIARY OF ALL SUBSEQUENT DISTRIBUTIONS OR
REDISTRIBUTIONS OF THE SUBJECT SOFTWARE.  ANYONE WHO USES, REPRODUCES,
DISTRIBUTES, MODIFIES OR REDISTRIBUTES THE SUBJECT SOFTWARE, AS DEFINED HEREIN,
OR ANY PART THEREOF, IS, BY THAT ACTION, ACCEPTING IN FULL THE RESPONSIBILITIES
AND OBLIGATIONS CONTAINED IN THIS AGREEMENT.

Government Agency: National Aeronautics and Space Administration (NASA)
Government Agency Original Software Designation: LEW-20180
Government Agency Original Software Title: Three-Dimensional Nozzle Design Code
User Registration Requested.  Please Visit https://github.com/nasa/Three-Dimensional-Nozzle-Design-Code
Government Agency Point of Contact for Original Software: heath.reising@nasa.gov


  1. DEFINITIONS

    A. "Contributor" means Government Agency, as the developer of the Original
       Software, and any entity that makes a Modification.
    B. "Covered Patents" mean patent claims licensable by a Contributor that
       are necessarily infringed by the use or sale of its Modification alone
       or when combined with the Subject Software.
    C. "Display" means the showing of a copy of the Subject Software, either
       directly or by means of an image, or any other device.
    D. "Distribution" means conveyance or transfer of the Subject Software,
       regardless of means, to another.
    E. "Larger Work" means computer software that combines Subject Software, or
       portions thereof, with software separate from the Subject Software that
       is not governed by the terms of this Agreement.
    F. "Modification" means any alteration of, including addition to or
       deletion from, the substance or structure of either the Original
       Software or Subject Software, and includes derivative works, as that
       term is defined in the Copyright Statute, 17 USC 101.  However, the act
       of including Subject Software as part of a Larger Work does not in and
       of itself constitute a Modification.
~~~~

(license excerpt: first 40 lines of a 241-line NASA Open Source Agreement 1.3; remainder is standard license terms.)

### `raosim/legacy_io.py`
*415 lines · stage X04 Vendored NASA/JHU MOC_Grid_BDE reference and comparison tooling*

~~~~text
Parsers for NASA/JHU MOC_Grid_BDE reference outputs.

The legacy code writes a mix of whitespace tables and Tecplot multi-zone
``.plt`` files.  These parsers intentionally preserve the raw numeric data
while giving the known files stable, normalized column names so tests and
comparison scripts can use the NASA/JHU outputs as an oracle.
~~~~

### `scripts/compare_nasa_reference.py`

*904 lines · stage X04 Vendored NASA/JHU reference and comparison tooling*

~~~~text
Compare current Python Rao/MOC outputs against NASA/JHU reference files.

This is a visibility tool, not a pass/fail gate.  It reports the current
baseline mismatch so algorithm ports can tighten one file family at a time.
Unavailable metrics are reported explicitly instead of silently dropped.
~~~~

### `scripts/nasa_wall_probe.py`

*72 lines · stage X04 Vendored NASA/JHU reference and comparison tooling*

~~~~text
Phase 12.5 probe: BDE wall constructor vs NASA wall.out (M3.5Perf).

Perfect-nozzle branch: D = axis end of BD, mdot_BD = full kernel mass,
DE = uniform-state characteristic from D (C++ FindPointE PERFECT-AXI
closed form).  Then calc_bde_region builds the wall; compare against
the i=0 trace of outputs_M3.5Perf/wall.out.
~~~~


## X05 - Standalone research diagnostics and offline generators (J4/J5 gates, theta_B probes, CEA surface sampler, TVC study, contour/regen generators)

*15 files.*

### `scripts/compare_jax_scan_stationarity.py`

*178 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Compare one Rao BVP solution against independent scan stationarity metrics.
~~~~

### `scripts/d_attachment_diagnostic.py`

*160 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
D-attachment diagnostic for the 3e-3 stationarity floor (J4).

1. Dumps nodes 0..4 of the converged DE: x, r, M, theta, nu, mu, C+/-,
   chord slope vs characteristic slope, stationarity contribution.
2. Variants on the start condition:
     A: current theta_D pin (BD interpolation)            [baseline]
     B: drop the theta pin entirely (r pin only)
     C: C+ invariant pin  (theta0+nu0) - (theta_D+nu_D) = 0
Run: PYTHONPATH=. python scripts/d_attachment_diagnostic.py
~~~~

### `scripts/generate_contour.py`

*177 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Parametric Rao nozzle contour generator (converged BVP + BDE wall).

Runs the current solver configuration (characteristic formulation, JAX
backend, constraint-weight ladder, full D-state continuity) for the
design point you specify, builds the wall via NASA's BDE-region march,
and writes:

    <out>/contour.csv      wall polyline (x, r) in metres
    <out>/contour_3d.stl   binary STL surface of revolution
    <out>/contour.png      2-D profile + residual summary
    <out>/contour_3d.png   3-D rendering (matplotlib)

Usage examples
--------------
    PYTHONPATH=. python scripts/generate_contour.py
    PYTHONPATH=. python scripts/generate_contour.py \
        --rt 0.025 --epsilon 8 --length-pct 85 --gamma 1.23 --out builds/my_nozzle

Reliability: the solve must pass the 2e-3 residual gate or the script
exits nonzero (use --allow-unconverged to override).  Outputs are
preliminary design geometry — NOT hardware-qualified (no CFD, thermal,
structural, or hot-fire validation; see RaoSolution.warnings).
~~~~

### `scripts/generate_regen_nozzle.py`

*92 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Generate a 3-D regen-cooled nozzle (wall + cooling channels).

Builds the nozzle wall as a surface of revolution and the N
regenerative cooling channels wrapping it (axial, or helical "coils"),
exports a binary STL for CAD/print and a 3-D PNG.  Optionally colours
the channels by the gas-side wall temperature from the coupled cooling
analysis (the throat hot spot painted onto the 3-D model).

Examples
--------
    # 80 axial channels, copper, coloured by wall temperature
    PYTHONPATH=. python scripts/generate_regen_nozzle.py --channels 80 --thermal

    # 24 helical coils, 3 turns
    PYTHONPATH=. python scripts/generate_regen_nozzle.py \
        --channels 24 --channel-width 0.0018 --helix-turns 3 --out builds/coil
~~~~

### `scripts/j4_gate_wall_probe.py`

*121 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
J4 gate replication + solved-CE BDE wall shape, in a single solve.

Replicates the corrected J4 reference config
(characteristic formulation, JAX backend, constraint-weight ladder,
full D-state continuity) with ``wall_method="bde"`` so the same solve
also yields the BDE wall.  Prints the gate verdict and the wall-slope
profile, and writes a JSON checkpoint + the wall polyline.

Purpose (Phase 12.4 follow-through, re-grounded post-correction): the
kernel-march mass-integral fix changed the seed — ``set_theta_b`` now
converges the fixed-(L, ε) topology exactly (θ_B ≈ 25.57° for the
ε=10/L80 reference) instead of fail-bracketing at the old ~24.2° cap —
so both the gate value and the solved-CE wall shape need re-measuring.
The bell criterion is the smooth stationary-DE existence root
(test_smooth_existence_root_regression): slope peaks ≈ θ_B = 25.57°
(+ ~0.7° arc/BFE discretisation overshoot) just after the throat arc
and decreases monotonically to ≈ 11.12° at the exit.  (The chart's
30° / 15.5° are Rao-1960 *parabola-fit* values — expected deltas, not
targets; see the J5 de-circularization.)  The historic defect was a
35.6° flare at 60% length driven by the stale characteristic pairing
and the relaxed D-state diagnostic branch.

The config relies on package defaults, which since 2026-06-11 ARE the
J4-gate configuration (characteristic formulation, JAX backend, weight
ladder, full D-state pins); only deliberate non-defaults are explicit.

Run:  PYTHONPATH=. python scripts/j4_gate_wall_probe.py
~~~~

### `scripts/j4_report.py`

*109 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
One-shot J0-J4 status report on the reference case (eps=10, L80, gamma=1.4).

Usage:
    python scripts/j4_report.py            # kernel + topology + JAX solve (~20-30 s)
    python scripts/j4_report.py --scipy    # also run the scipy backend (slow, minutes)

What to look for:
  [1] kernel march: rrcs == 58 on the NASA M3.5Perf geometry (oracle match)
  [2] Rao-case kernel: rrcs >> 1 (pre-fix failure mode was exactly 1)
  [3] topology: mass_BD == mass_DE (nonzero), E.r pinned to Re, DE nodes >= 12
  [4] JAX solve: max_scaled < 0.8, physics blocks (stationarity/moc_c+/-) < 0.2
      -- residual should concentrate in length / wall_endpoint (the open
      variational tension documented in JAX_DIFFERENTIABLE_PLAN.md STATUS).
~~~~

### `scripts/j5_chart_sweep.py`

*89 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
J5 record run: the Rao chart sweep under the JAX defaults.

DE-CIRCULARIZED (2026-06-12): the solver columns are now genuine
solver outputs — theta_N is the kernel arc-end angle theta_B the BVP
closed on (per-row ``theta_n_source`` records the provenance) and
theta_E is the solved CE exit flow angle.  The pre-J5 record run
(2026-06-11) measured a chart echo (err_n circularly ~0) and the
chart-N → exit straight chord (the entire old "solved theta_E"
signature — ~21.0° @L70 / ~18.5° @L80 / ~16.6° @L90,
epsilon-independent — is reproduced to ~0.1 deg by that chord's pure
geometry; its "low-ε theta_E gap" was chart-vs-chord, not physics).

What this record now documents: the systematic deltas between the
EXACT variational solution and Rao's 1960 ARS J. parabola-fit charts
(gamma=1.23; contours gamma-insensitive per Rao 1961 p. 1490).
Expected reference-point deltas: theta_B 25.57 vs chart 30 deg,
theta_E 11.12 vs chart 15.5 deg (the solver-independent smooth
existence root).  The ``*_gate`` fields are kept for the historical
record; post-J5 they are findings, not pass/fail criteria — the
plan-target xfail is definitional (see
tests/test_rao_chart_benchmark.py).

This is the *record* run for the J5 milestone: the suite's sweep at
the J4-gate physics weight (1.0); prints the per-case table and dumps
JSON for the repo record (used to pin the delta-baseline regression
test — see the full-grid test's TODO).

Run:  PYTHONPATH=. python scripts/j5_chart_sweep.py
~~~~

### `scripts/jax_convergence_experiment.py`

*122 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Experiment driver for the J4 gate: JAX LM strategies on the Phase-6 case.

Usage:  python scripts/jax_convergence_experiment.py [strategy]
  strategy in {barrier, homotopy, sigmoid}  (default: homotopy)

Prints per-rung max_scaled so stalls are visible immediately.
~~~~

### `scripts/plot_reference_topology.py`

*54 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Render the Phase-12.6 reference topology (fixed-end closure) to PNG.

Builds the full-form RaoTopology for the reference design point
(epsilon=10, L80, Rd=0.382Rt, Ru=1.5Rt) and writes:

    builds/reference_topology.png    plot_topology overlay
    builds/reference_wall.csv        full wall (throat arc + streamline_BE)

This is the geometrically-sane characteristic closure (NOT the Rao
optimum — see moc_topology.build_reference_topology's docstring).

Run:  PYTHONPATH=. MPLBACKEND=Agg python scripts/plot_reference_topology.py
~~~~

### `scripts/rao_existence_scan.py`

*401 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Run a NumPy-only Rao D-attachment existence scan.

Example:
    PYTHONPATH=. python scripts/rao_existence_scan.py \
      --Rt 0.020 --epsilon 10 --length-pct 80 \
      --theta-b-center-deg 24 --theta-b-span-deg 12 --theta-b-count 31 \
      --kdf-min 0.05 --kdf-max 0.85 --kdf-count 31 \
      --models smooth,position,fan --plot

Declared B5 domain example (fixed-context D-attachment and kernel evidence
only; choose an independently justified external grid and resolution policy):

    PYTHONPATH=. python scripts/rao_existence_scan.py \
      --theta-b-count 9 --kdf-count 9 --models smooth \
      --kernel-dtheta-limit-deg 0.25 \
      --kernel-massflow-rel-tol 0.02 \
      --resolution-convergence 6,8,12 \
      --domain-theta-seed-policy rao_chart_centered \
      --domain-chart-theta-span-deg 40 \
      --domain-chart-theta-count 21 \
      --domain-epsilon-values 4,6,8,10 \
      --domain-length-pct-values 60,70,80,90,100 \
      --domain-max-epsilon-spacing 2 \
      --domain-max-length-pct-spacing 10
~~~~

### `scripts/sample_cea_surface.py`

*201 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Offline CEA property-table sampler (host-only; needs RocketCEA).

Phase-2 workflow (docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md §12.2 row 2):
sample frozen chamber properties on a (Pc, O/F) grid, save as .npz, load in
the differentiable layer via ``raosim.mdo.properties.load_chamber_surfaces``.

Example:
    python scripts/sample_cea_surface.py --oxidizer LOX --fuel RP-1 \
        --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 13 \
        --of-min 1.6 --of-max 3.2 --n-of 13 \
        --out builds/cea_tables/lox_rp1_frozen.npz

Validation: ``--holdout`` samples seeded points strictly inside fit-grid cells
and compares both values and gradients to live RocketCEA.  It exits nonzero if
any field-specific tolerance is exceeded.  ``--validate-only`` applies the same
gate to an existing, hash-verified table without resampling its fit grid.
~~~~

### `scripts/seed_diagnostic.py`

*175 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
seed_diagnostic.py  —  Seed-vs-optimizer falsification test for the Rao
coupled-wall convergence stall (max_scaled ~ 8, gate 2e-3).

WHY THIS EXISTS
---------------
The JAX plan's priority-1 thesis is that the stall is caused by scipy's
*finite-difference Jacobian noise* near the sonic / Mach-line singularities, so
exact autodiff (jacrev) Jacobians will fix it.  The competing hypothesis (the
repo's own Phase-6 xfail text, docs/legacy_code_audit.md, and the JHU/APL
report's statement that the initial-line shape drives convergence) is that the
stall is the SEED/KERNEL: the kernel BD cannot carry the throat target mass, so
the residual has NO root reachable from this seed and no optimizer can descend.

These make OPPOSITE, MEASURABLE predictions at the stalled point x*:

  SEED hypothesis      : x* is a (near-)stationary point of 0.5||r||^2 with
                         ||r|| ~ 8.  ||J^T r|| ~ 0 even with an ACCURATE
                         Jacobian; scipy stops on gtol; one block (mass /
                         moc_cminus) holds the floor; a more accurate (3-point)
                         Jacobian does NOT move the floor.  => exact JAX
                         Jacobians cannot fix convergence.

  OPTIMIZER hypothesis : x* is NOT stationary.  ||J^T r|| is large, a
                         Gauss-Newton step predicts a big cost drop, scipy
                         stopped on ftol/xtol, and a 3-point Jacobian escapes
                         the floor.  => exact Jacobians may help.

We test the REAL NumPy residual (no JAX re-port), so there is no port-mismatch
confound.  No JAX required — runs in the existing numpy/scipy env.

RUN
---
    python scripts/seed_diagnostic.py
    # ~1-3 min: it runs the real ref-case solve, then probes x*.

Paste the final "SUMMARY (paste this back)" block to me and I'll fold it in.
~~~~

### `scripts/theta_b_freeze_sweep.py`

*115 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Full-continuity sweep around the smooth stationary-DE root.

Sweeps ``theta_b_freeze_deg`` with FULL D-state continuity
(pin_d_theta + pin_d_mach) at the reference point (eps=10, L80, gamma
1.4).  The corrected characteristic formulation and independent RK
existence scan predict:

  * the full-pin residual closes near theta_B = 25.5659 deg and
    kdf = 0.15216;
  * at the best theta_B the gate (2e-3) closes WITH full continuity;
  * the BDE wall peaks ~theta_B right after the throat arc (mid-bell
    flare gone) and exits near 11.1 deg.

Run:  PYTHONPATH=. python scripts/theta_b_freeze_sweep.py
      (optionally: --band 24.5 26.5 --step 0.5)
~~~~

### `scripts/theta_b_picard_probe.py`

*156 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Historical theta_B Picard diagnostic, retained for reproducibility.

Full D-state continuity (``pin_d_theta=True`` + ``pin_d_mach=True``) with
the kernel march now clearing the historic ~24.2 deg cap.  Outer loop:

    solve BVP (characteristic, full D pins, ladder, jax backend)
      -> refresh theta_B: secant on the kernel-BD Rao stationarity at the
         solved (kdf, log_C)   [cheap kernel rebuilds only]
      -> reseed at the refreshed theta_B and re-solve
    until |d theta_B| < 0.05 deg.

The values below were measured before the characteristic-pairing fix
and must not be interpreted as evidence against smooth D continuity:

Measured in the 2026-06-11 session (n_control=24, ladder
(1, 10, 30, 100)):

    iter0  theta_B=21.87  ->  max_scaled 5.67e-2, kdf 0.0882, refresh -> 28.10
    iter1  theta_B=28.10  ->  max_scaled 5.70e-2, kdf 0.0865, refresh -> 28.17

The corrected formulation instead has a smooth full-continuity root near
theta_B=25.5659 deg and kdf=0.15216.  This script remains useful only for
comparing the old stationarity-refresh strategy with direct theta_B
freezing; use ``theta_b_freeze_sweep.py`` for the current experiment.

Note: ``solve_rao_bvp``'s seed path runs ``set_theta_b`` (its own inner
secant on *length*), so ``thetaN_guess_deg`` only initialises that secant;
the kernel each solve actually freezes sits near the fixed-end topology's
theta_B (~25.5 deg for the reference).  The outer refresh here therefore
probes the seed basin, not a hard theta_B override.

Run:  PYTHONPATH=. python scripts/theta_b_picard_probe.py
~~~~

### `scripts/tvc_sizing_study.py`

*443 lines · stage X05 Standalone research diagnostics and offline generators*

~~~~text
Gimbal-TVC actuator sizing study over LREKit-generated engine contours.

Consumes the ``contour.csv`` files written by ``python -m raosim`` (one per
thrust point), builds the 316L pressure-vessel shell, and evaluates the
per-axis gimbal torque budget and the actuator feasibility screen.

The three screens applied to every candidate actuator are, in order of how
often they are the binding one in practice:

  1. TORQUE     tau_avail = N * eta * tau_motor  >=  tau_required
  2. SPEED      omega_motor = N * omega_gimbal   <=  omega_motor_max
  3. INERTIA    N^2 * J_rotor                    <=  k * J_load
                (k = 10 open-loop stepper, 30 closed-loop stepper, 100 servo;
                 Oriental Motor / PMD published load-to-rotor inertia limits)

Screens 1 and 2 pull the gear ratio N in opposite directions, so they define a
feasible band in N.  Screen 3 caps the band from above independently.  Where the
band is empty the actuator cannot do the job at any reduction -- which is the
correct way to answer "can I use two NEMA 17s?", because the naive comparison
of motor holding torque against required gimbal torque ignores that gearing is
free to fix that.

Literature basis for the requirement side
-----------------------------------------
* Bandwidth 4 Hz with 20-25 deg phase lag at 1 Hz -- the SSME TVC requirement
  envelope, Cowan & Weir, "Design and Test of Electromechanical Actuators for
  Thrust Vector Control", NASA MSFC, NTRS 19940025147.
* Gimbal deflection +/-5 to +/-10 deg is the conventional design band.
* Reflected rotor inertia dominates TVC actuator sizing: "the amount of inertia
  created by the rest of the system is essentially negligible when compared to
  the inertia created by the cyclic action of each rotor inertia" -- ibid.
* Gimbal bearing friction torque is small relative to inertia and flex-line
  terms for rolling-element gimbals -- Neugebauer et al., "Bearing Development
  for a Rocket Engine Gimbal" (VINCI), 38th AMS / ESMATS 2006.
* 316L rho = 8000 kg/m^3, S_y = 290 MPa -- raosim/materials.py catalog entry.

Usage
-----
    python scripts/tvc_sizing_study.py --contours /tmp/tvc --json out.json
~~~~


## X06 - Knowledge graph, source corpus and packaged data assets (non-CLI)

*63 files.*

### `knowledge/AGENTS.md`
*354 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
# LREKit scientific knowledge system — agent manual

This directory is LREKit's **local scientific memory and engineering provenance
layer**. Read this file before any task that changes or evaluates physics,
equations, correlations, validity criteria, empirical coefficients, structures,
injectors, sprays, pumps, optimisation constraints, CAD dimensions derived from
physics, manufacturing rules, requirements, measurement comparison, calibration,
validation, or qualification claims.

It is an **index and reasoning aid**, not a replacement for reading the source.
The graph never outranks the PDF it points to.

---

## 0. Startup protocol (mandatory for scientifically consequential work)

```bash
knowledge/kg.sh status                       # 1. corpus synchronized? stores populated?
knowledge/kg.sh context --task "<your task>" # 2. retrieve_for_task(): the context package
knowledge/kg.sh model  "<concept>"           # 3. what LREKit already implements, and on what basis
knowledge/kg.sh equation "<concept>"         # 4. extracted, page-anchored equations
knowledge/kg.sh claims "<subject>"           # 5. claims, incl. validity statements and assumptions
knowledge/kg.sh validity MODEL_...           # 6. validity envelopes (prose + any source-stated bounds)
knowledge/kg.sh basis raosim/<file>.py:<Symbol>   # 7. the scientific basis of existing code
knowledge/kg.sh sources <EQ_|CLM_|VAL_ id>   # 8. provenance + source context + PDF page to escalate to
knowledge/kg.sh compare "<topic>"            # 9. competing formulations across documents
```

Then: **open the PDF page** for every load-bearing value, equation or limit
(`sources` prints the path and page). Distinguish literature-derived science from
implementation choices (every model and constant carries a `basis_class`). Only
then implement. This applies to auditing/repairing existing code too — **existing
code is not scientific authority merely because it exists**.

If `status` reports `synchronized: false`, run the sync (§3) before trusting
retrieval; a PDF that is on disk but not converted/ingested is invisible to the
system and its absence from results means nothing.

Purely mechanical tasks (formatting, renames, packaging) do not need retrieval.

~~~~

### `knowledge/README.md`
*164 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
# LREKit local scientific memory

This repository-local system connects source PDFs and passages to equations,
claims, model-registry entries, code, tests, requirements, and provenance.
It runs offline using JSON stores, SQLite FTS5, and a NumPy/SciPy LSA vector index.
The retrieval API is a Python module; there is no HTTP service.

## Start here

The wrapper tries `.venv`, `.venv-jax`, then `.venv-kg` (NumPy and SciPy are required). Set
`KG_PYTHON=/path/to/python` to use another environment. PDF conversion additionally
uses `.venv-kg` with PyMuPDF, pymupdf4llm, and optional Tesseract OCR.
The PDF corpus is local and gitignored; a fresh checkout does not include it.

```bash
knowledge/kg.sh status
knowledge/kg.sh ask "What supports the Rao variational nozzle method?"
knowledge/kg.sh --json context --task "Review coolant heat-transfer validity"
knowledge/kg.sh equation "Rao"
knowledge/kg.sh basis raosim/physics.py
knowledge/kg.sh sources EQ_ID
knowledge/kg.sh artifacts "faceplate"
knowledge/kg.sh artifact CAD_ID
```

Use the concrete equation ID returned by a query in place of `EQ_ID`.
`--json` belongs before the subcommand. For Python callers:

```bash
PYTHONPATH=knowledge python -c 'from kg.api import retrieve_for_task; print(retrieve_for_task("Rao nozzle"))'
```

Read [AGENTS.md](AGENTS.md) for the authority and provenance contract and
[the implementation audit](reports/IMPLEMENTATION_AUDIT_2026-09-20.md) for what
Claude completed, what was repaired, and the remaining implementation scope.
[The generated report](reports/latest_report.md) contains current store counts.

## Refresh without losing completed reviews

```bash
~~~~

### `knowledge/kg.sh`
*15 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Prefer a repository environment with the KG dependencies.
~~~~

### `knowledge/kg/__init__.py`
*17 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
LREKit local scientific memory and engineering provenance layer.

Public entry points live in :mod:`kg.api` (programmatic) and :mod:`kg.cli`
(``kg`` command).  Storage layout is documented in ``knowledge/AGENTS.md``.
~~~~

### `knowledge/kg/__main__.py`
*3 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
(no header comment)

Inferred role: (inferred) module entry shim for `python -m kg`; imports `main` from `kg.cli` and exits with its return code.
~~~~

### `knowledge/kg/api.py`
*791 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Stable programmatic interface for LREKit development agents.

The storage layout may change; these function names and return shapes are
the contract.  Every returned object keeps its ``id`` so the caller can drill
down (``get_equation``, ``get_source_context``) instead of loading the corpus.

States are explicit.  Empty results are reported with a ``state`` such as
``NO_RELEVANT_SOURCE_FOUND`` rather than silently returning ``[]``.
~~~~

### `knowledge/kg/candidate_links.py`
*33 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Identity checks for reviewed source objects linked to raw parser candidates.
~~~~

### `knowledge/kg/chunker.py`
*241 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Structure-aware chunking of generated corpus Markdown.

Chunks follow the document's own structure - page markers, headings,
paragraphs, tables, fenced blocks - rather than a fixed token window, so an
equation stays with the sentence that introduces it and a table is never cut
in half.  Every chunk keeps the page span, heading path, document order and
character offsets needed to escalate back to the PDF.
~~~~

### `knowledge/kg/cli.py`
*246 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
``kg`` - command-line access to LREKit's scientific memory.

Run from the repository root:

    python knowledge/kg/cli.py <command> [args]      (any repo venv; needs numpy)
    knowledge/kg                                     (wrapper script, same thing)

Commands
    search  "query"                 hybrid lexical+semantic chunk search
    docs    "query"                 documents ranked for a query
    doc     DOC_ID                  document record + linked objects
    entity  "name" | ID             find graph nodes / resolve a symbol
    equation "concept" | EQ_ID      find or show equations
    claims  "subject" [--pred X]    find claims
    model   "concept" | MODEL_ID    find or show registry models
    validity MODEL_ID|"concept"     validity conditions for a model
    sources  ID                     provenance + source context + PDF escalation for any object
    implementation ID               software that implements a scientific object
    basis   raosim/x.py[:Symbol]    scientific basis of a software component
    path    "a" "b"                 shortest graph path between two nodes
    compare "query"                 per-document passages + recorded conflicts
    context --task "..."            retrieve_for_task() context package (JSON)
    ask     "question"              alias of context, with a compact rendering
    status                          corpus/ingestion/validation state
    sync [--convert]                check PDF<->Markdown sync (convert with --convert; needs .venv-kg)
    rebuild                         re-run manifest -> chunks -> index -> graph (no LLM extraction)
    validate                        run quality-control checks and write a report
~~~~

### `knowledge/kg/completion.py`
*438 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Deterministic completeness, equation-fidelity and model-audit data products.

Each product is computed only from stores, journals and the packet ledger. The
denominators are kept separate on purpose: indexed text, parser candidates,
stored source assertions, independent verification and mathematical
completeness are different evidence states. Nothing here declares a document
mathematically complete; that state stays ``NOT_ESTABLISHED``.
~~~~

### `knowledge/kg/constant_basis.py`
*46 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Keep legacy survey labels separate from independently reviewed source basis.
~~~~

### `knowledge/kg/coverage.py`
*160 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Lightweight evidence coverage, without pretending to measure completeness.

Counts describe stored, current source assertions. Neither a populated index nor
an independently confirmed equation establishes that all source math was found.
No PDF parsing, source writes, or index rebuilds occur here.
~~~~

### `knowledge/kg/enrichment.py`
*440 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Docling derivatives are inspectable candidates, never automatic source support.

Keep the reviewed Markdown/chunks and scientific stores intact. A formula model
can hallucinate notation; byte hashes and bounding boxes prove traceability, not
correctness. Reviewed equations still enter through bank_extraction.py.
~~~~

### `knowledge/kg/execution_reach.py`
*142 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Bounded static call-site evidence; never a runtime trace or physics validation.

Explicit model-to-symbol mappings below are project-maintained implementation
routes. Unmapped models and unresolved object/dynamic calls remain UNKNOWN.
~~~~

### `knowledge/kg/freshness.py`
*77 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Preserve historical extraction records while invalidating changed evidence.
~~~~

### `knowledge/kg/graph.py`
*563 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Graph assembly, traversal and export.

The graph is *assembled* from the JSON stores (it is a view, never the
primary record) so that a node always traces back to the store that owns it.
Every edge carries a ``provenance`` field; edges without provenance are
reported by validation, not silently kept.

Exports: ``knowledge/graph/graph.json`` and ``knowledge/graph/graph.graphml``.
~~~~

### `knowledge/kg/ground.py`
*90 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Deterministic grounding of LLM-extracted scientific objects against source text.

An extracted equation or claim enters the knowledge stores only if its
``quote`` (a short verbatim anchor the extractor must supply) can be located
in the ingested chunk text of the stated document on the stated page (±2
pages, because page markers sit at PDF-page granularity and layout rebuilds
can shift a line). Matching preserves word order and requires a contiguous
anchor after whitespace/markup normalization. Token overlap is not proof
that a source states a claim.

Outcomes:
    GROUNDED               anchor found on the stated page
    GROUNDED_PAGE_SHIFTED  anchor found within ±2 pages (page corrected)
    UNGROUNDED             anchor not found -> object is NOT stored, only reported
~~~~

### `knowledge/kg/harvest_artifacts.py`
*341 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Observe existing design and injector CAD reports without creating test evidence.

Only two producer formats present in this checkout are recognized. A serialized
input/layout is a *knowledge snapshot*, not a producer configuration identity.
An existing STEP and its hash prove current bytes only. Stored CAD inspections
and manufacturing gates remain report assertions; no kernel or physical check
is performed here. ManufacturingProcess records describe reference plans, never
executed manufacturing, and no Measurement/Experiment/Calibration is inferred.
~~~~

### `knowledge/kg/harvest_constraints.py`
*192 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Harvest the MDO constraint contract into Requirement / Constraint / Gate records.

Source of truth: ``raosim.mdo.constraints.CONSTRAINT_MANIFEST`` - the repo's
authoritative ordered manifest.  Its ``optimizer_role`` decides the
restriction kind; the knowledge system never re-labels a row:

    optimizer_role="hard"        -> optimizer_constraint   (enforced inside the solve)
    optimizer_role="final_gate"  -> postsolve_gate         (mandatory check after solving)
    optimizer_role="report_only" -> diagnostic
    kind="availability_gate"     -> applicability_gate     (decides whether a prediction may be made)
    category="requirement"       -> also tagged requirement (what the requested engine must satisfy)

``source_id`` strings are mapped to Documents ONLY when a report number in the
string matches a Document's title/identifier; unmatched strings are preserved as
``source_id_text`` with ``source_state="UNRESOLVED_SOURCE_KEY"``.  ``None`` becomes
``SCIENTIFIC_BASIS_UNKNOWN`` - the row is software policy with no cited basis.

The manifest module is imported with the JAX-free fallback: if the import fails
(e.g. an environment without jax), the harvester reads the ``ConstraintSpec(...)``
literals from the source with ``ast`` (positional AND keyword arguments, string
concatenation included; nothing is executed) instead and says so.

Each row's ``basis_class`` (literature, geometric_definition,
model_validity_domain, repository_rule, numerical_contract, requirement) is
carried as ``declared_basis_class``.  A row without a ``source_id`` whose
declared basis is not literature is labelled ``DECLARED_<BASIS>`` instead of
``SCIENTIFIC_BASIS_UNKNOWN``: the repository states what kind of limit it is,
which is a label, not source support.
~~~~

### `knowledge/kg/harvest_extraction.py`
*1243 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Store grounded literature extractions (equations, claims, validity, constants).

Input: the structured results of the extraction workflow.  Every object is
passed through :class:`kg.ground.DocGrounder`; objects whose verbatim anchor
cannot be found in the indexed source text are NOT stored - they are listed in
the returned report under ``ungrounded`` so the loss is visible.

Model links: per-source citation assessments and fingerprinted implementation
reviews are retained independently. Only an equivalent structured review may
create IMPLEMENTS or claim SUPPORTS; comparison formulations remain CITES.
Historical single-source records stay readable. A secondary assessment cannot
erase a prior source's state, and a citation never establishes physical validity.
~~~~

### `knowledge/kg/harvest_parameters.py`
*400 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Observe electric-feed defaults from Python AST, without importing models.

These are separate repository technology assumptions, not literature constants.
Exact definitions, static references and history are retained; disagreement does
not choose a preferred default set or create scientific source support.
~~~~

### `knowledge/kg/harvest_registry.py`
*230 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Harvest ``raosim.model_registry`` into Model records and software links.

The registry is the repository's own statement of what each implemented model
is, where its source is, and how far it has been verified.  Its authority is
therefore ``PROJECT_DERIVES`` - it is the implementation-state authority, not
the scientific authority.  The scientific authority remains the cited PDF,
which is resolved here to a ``Document`` id from ``knowledge/manifest.json``
by exact path match (never by name similarity).
~~~~

### `knowledge/kg/harvest_repo.py`
*382 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Deterministic harvest of repository-side knowledge.

* ``docs/*.md`` -> RepositoryDocument records with dates, STATUS heads and
  explicit precedence language (SUPERSEDES / CORRECTS / UPDATES / DEPRECATES /
  BLOCKS), each relation carrying the verbatim line that asserts it.
  Repository documents are implementation-state authority, never scientific
  authority; that distinction is stored on every record.
* ``tests/*.py`` -> Test records classified from each exact AST function's
  name/body. Module citations never become function evidence. Classification
  is heuristic, explicitly scoped, and never experimental validation.
* ``raosim/ lrekit/ scripts/`` -> Capability records for explicit
  NotImplementedError / unavailable / blocked paths, with file:line.
* ``docs/LITERATURE_REQUESTS*.md`` -> BLOCKED_ON_LITERATURE capability records.
* every ``propulsion_texts/...pdf`` string in code/docs/tests -> CITES edges
  (path must exist on disk; unresolved paths are reported).
~~~~

### `knowledge/kg/harvest_survey.py`
*245 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Turn the agent survey of LREKit source into verified SoftwareComponent records.

Grounding rule: an agent's report is a *lead*, not a fact.  Each reported
component is verified against the working tree before it is stored:

* the file must exist;
* the qualname's terminal symbol must be defined in that file (``def``/``class``),
  and the reported line is corrected to the actual definition line;
* each hard-coded constant's ``line`` must exist and its ``value`` must appear
  in the file within a few lines of the reported line, else it is stored with
  ``verified=false`` and excluded from the graph;
* each ``model_registry_ids`` entry must be a real registry id;
* each cited corpus path must exist on disk.

Unverifiable items are kept in the ingestion report as ``UNGROUNDED`` so the
loss is visible, never silently dropped.
~~~~

### `knowledge/kg/implementation_freshness.py`
*273 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Refresh exact Python AST locators without certifying implementation science.

This reads source; it never imports the inspected modules or infers runtime
reach. Existing model/equation associations are retained, not invented. Missing
or ambiguous definitions remain inspectable as inactive records. A structural
definition change requires a new implementation-versus-literature comparison;
model source-support verdicts are owned by the source review workflow.
~~~~

### `knowledge/kg/index.py`
*292 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Hybrid retrieval index: FTS5 (lexical, BM25) + LSA (latent semantic, dense).

Why not a neural embedding model?  The knowledge system must run offline in any
of the repository's virtual environments with nothing beyond NumPy/SciPy.  A
TF-IDF -> truncated-SVD (LSA) space over the whole corpus gives useful
semantic neighbourhoods for a single technical domain and is fully
inspectable: every dimension is a linear combination of vocabulary terms
stored on disk.  The lexical channel keeps exact symbols (``c*``, ``T_wg``)
findable; the dense channel catches paraphrase.  Results are fused with
reciprocal-rank fusion.

Artifacts (``knowledge/vector_index/``):
    chunks.sqlite        chunk text + FTS5 (see store.py)
    vocab.json           term -> column index
    idf.npy              inverse document frequency per term
    components.npy       V x K projection (float16)
    chunk_vectors.npy    N x K unit-normalised chunk vectors (float16)
    chunk_ids.json       row -> chunk_id
    index_meta.json      build parameters, corpus fingerprint
~~~~

### `knowledge/kg/ingest.py`
*311 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Document manifest construction and incremental chunk ingestion.

Pipeline (idempotent):
    conversion manifest (propulsion_texts_for_agents/_conversion_manifest.json)
      + paper_index.json (prior automated bibliographic pass)
      + PDF metadata recorded at conversion time
    -> knowledge/manifest.json           one Document record per unique PDF payload
    -> knowledge/vector_index/chunks.sqlite   chunks for every canonical, readable document
    -> ingestion_state.json              per-document state, honest failure states

Duplicate PDF payloads (identical sha256 under different filenames) become ONE
Document with ``alternate_paths``; they never get independent scientific
authority or duplicate chunks.
~~~~

### `knowledge/kg/locks.py`
*133 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Advisory inter-process locks for authoritative stores and heavyweight jobs.

Several agents may operate on one checkout. JSON stores are read, modified and
replaced as whole files, so two concurrent writers can silently lose each
other's records. ``writer()`` serializes those read-modify-write sections
across processes and across threads of one process; it is reentrant only for
the thread that holds it, so nested helpers do not deadlock.

``heavy_job()`` is non-blocking: at most one Docling inference process may run
on this machine at a time, and a second request is refused instead of queued.
Locks are advisory ``flock`` locks; they disappear with the holding process,
so an interrupted job cannot leave a stale lock behind.
~~~~

### `knowledge/kg/packet_queue.py`
*330 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Persistent, atomic state machine for bounded source packets.

Every packet follows::

    QUEUED -> DOCLING_RUNNING -> RAW_CAPTURED -> CANDIDATES_PREPARED
      -> OPUS_SOURCE_REVIEW -> VERIFIED | REJECTED | BLOCKED
      -> GRAPH_INTEGRATED -> TESTED -> FINAL_AUDIT_ELIGIBLE

Transitions are role-gated: mechanical workers (S1-S3) cannot record a source
review, and graph integration requires the exact reviewed verdict bytes. Raw or
candidate states can never jump to GRAPH_INTEGRATED. Rejected and blocked
packets keep their history and never count as verified. Object-level outcomes
come from the reviewed verdict files, not from the packet label.
~~~~

### `knowledge/kg/pipeline.py`
*131 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
End-to-end (re)build of the deterministic layers of the knowledge system.

    .venv/bin/python knowledge/tools/kg_pipeline.py [--force-chunks] [--skip-sync]

Steps (all idempotent):
  1. corpus sync check          (PDF <-> Markdown; conversion needs .venv-kg, see sync_corpus.py)
  2. manifest                   knowledge/manifest.json
  3. chunks                     knowledge/vector_index/chunks.sqlite (only changed docs)
  4. vector index               knowledge/vector_index/*.npy
  5. quantities                 knowledge/quantities.json
  6. model registry harvest     knowledge/models.json, validity.json
  7. repository harvest         repo_documents.json, tests.json, capabilities.json, CITES links
  8. graph                      knowledge/graph/graph.json (+ .graphml)
  9. validation                 knowledge/reports/latest_validation.json
 10. ingestion report           knowledge/reports/latest_ingestion.json

LLM-extracted stores (equations/claims/entities from literature, survey-derived
software components) are NOT regenerated here; they are produced by the
extraction workflows and merged by kg.harvest_extraction / kg.harvest_survey.
This pipeline preserves them and re-links them.
~~~~

### `knowledge/kg/quantities.py`
*326 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Canonical LREKit physical quantities, symbols, and units.

This is a *normalisation* table, not scientific authority.  It records how the
same physical quantity is written across the propulsion literature and across
LREKit source code so retrieval can find it, and it records dimensionality so
two quantities are never merged merely because they share a glyph.

Every entry carries ``provenance="SYSTEM_INFERS"`` semantics: the canonical
name and the alias list are this system's terminology decision.  A quantity's
*value*, *validity range*, or *correlation* is never stored here - those live
in equations/claims with document-level provenance.

Ambiguity is preserved.  ``resolve_symbol("k")`` returns several candidates;
callers must disambiguate with subsystem/document context or report
``AMBIGUOUS_SYMBOL``.
~~~~

### `knowledge/kg/record_identity.py`
*167 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Preserve duplicate survey facets without manufacturing scientific support.

Only an exact identity can share a record. Historical list unions are explicitly
separate from active model/test/constant associations. Ambiguous ID collisions
raise with the original records attached; no guessed ID/link remapping occurs.
~~~~

### `knowledge/kg/report.py`
*97 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Human-readable ingestion / validation report (Markdown) from the JSON reports.
~~~~

### `knowledge/kg/schema.py`
*245 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
LREKit knowledge-system ontology.

Versioned deliberately: consumers record ``ONTOLOGY_VERSION`` alongside any
derived artifact so a later schema change is detectable rather than silent.

Design notes
------------
* Node and edge types exist because they enable a retrieval an LREKit agent
  actually needs, not to maximise graph size.
* The status vocabularies are the repository's own engineering vocabulary.
  ``UNKNOWN`` and ``UNAVAILABLE`` are first-class results, never placeholders
  to be filled in later with a plausible number.
~~~~

### `knowledge/kg/standards.py`
*90 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Standards as typed sources: held documents or explicit BLOCKED_ON_STANDARD.

The registry lists standards that repository requests or the consolidated
review require tolerance, inspection, fatigue-test or uncertainty rules to
cite. A standard is ``HELD`` only when an exact manifest document with a
current fingerprint is mapped to it; otherwise it stays ``BLOCKED_ON_STANDARD``.
Editions are recorded only where the held document identifies one; requested
but unheld standards keep ``edition: UNSPECIFIED`` rather than a guessed year.
Holding a PDF is not compliance, and no rule-level link is inferred here.
~~~~

### `knowledge/kg/store.py`
*258 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Inspectable on-disk stores for the knowledge system.

Curated knowledge objects (documents, equations, claims, models, ...) are
plain JSON files under ``knowledge/`` so a human or agent can open them.
Chunks are numerous (tens of thousands) and live in SQLite with an FTS5
index under ``knowledge/vector_index/``.

Every JSON store is a list of records with a stable ``id`` field; writes are
atomic (temp file + rename) and sorted by id so diffs stay reviewable.
~~~~

### `knowledge/kg/validate.py`
*262 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Quality control for the knowledge stores.

Every check is a function returning ``(issues, stats)``; ``run()`` writes
``knowledge/reports/validation_<date>.json`` and ``latest_validation.json``.
Checks are honest about what they can and cannot verify.
~~~~

### `knowledge/kg/variable_links.py`
*24 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Rebuild conservative quantity aliases without rewriting source mathematics.
~~~~

### `knowledge/kg/verification_freshness.py`
*110 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Bind independent source verification to the scientific content inspected.

Source identity and assertion content are separate fingerprints. Graph links,
parser bookkeeping, timestamps and symbol-resolution indexes are not source
transcription content. Legacy verdicts remain inspectable, but without a content
fingerprint they cannot establish that every current field was inspected.
~~~~

### `knowledge/kg/work_journal.py`
*99 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Append-only work journal: which role and model performed each task.

Every line records the acting role, the model identity as known to that role,
input/output fingerprints and the acceptance state. The journal is evidence of
who did what; it never upgrades a scientific object by itself. A Sonnet worker
role cannot record an authoritative scientific verdict (``ACCEPTED_SCIENCE``).
~~~~

### `knowledge/kg/work_queue.py`
*114 lines · stage X06 Knowledge graph, source corpus and packaged data assets (non-CLI) (assigned X06; module not in module_stage_map)*

~~~~text
Deterministic, resumable review queue. This module never starts an LLM.
~~~~

### `knowledge/tools/__init__.py`
*1 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Local command-line utilities for the scientific memory.
~~~~

### `knowledge/tools/agentic_packet.py`
*571 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Deterministic role operations for one source packet.

    s1-run       PACKET  Docling the packet pages (one process, cache reuse)  DOCLING_RUNNING -> RAW_CAPTURED
    s2-prepare   PACKET  renders + candidates + stored objects + checks       RAW_CAPTURED -> CANDIDATES_PREPARED
    s3-integrate PACKET  bank the exact reviewed files, rebuild graph view    VERIFIED|REJECTED -> GRAPH_INTEGRATED
    s3-test      PACKET  knowledge tests, replay and integrity checks         GRAPH_INTEGRATED -> TESTED

These commands contain no scientific judgment. Candidate bundles list parser
output and stored records side by side without deciding equivalence; banking
uses only files the reviewer (role F) signed and the queue fingerprinted.
~~~~

### `knowledge/tools/bank_extraction.py`
*173 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Bank workflow extraction results from a journal into the knowledge stores.

    .venv/bin/python knowledge/tools/bank_extraction.py <journal.jsonl> [--grouped]

Reads every ``result`` line of a Workflow journal, grounds each object against
the indexed source text (kg.ground) and upserts equations / claims / validity /
entities / provenance.  Safe to run repeatedly: ids are content-derived, so
re-banking the same journal is a no-op.  Writes
``knowledge/reports/latest_extraction.json`` and re-assembles the graph.
~~~~

### `knowledge/tools/build_audit_inventory.py`
*90 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Reproducible implementation/source inventory and provisional evidence-debt queue.

This inventories every stored component; it does not certify each component's
science. Missing runtime leverage, source reviews and hardware evidence remain
explicit. Priorities are review scheduling assessments, never physical gates.
~~~~

### `knowledge/tools/build_completion_reports.py`
*18 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Write CORPUS_COMPLETENESS, EQUATION_FIDELITY_AUDIT and MODEL_SOURCE_AUDIT reports.

    .venv-jax/bin/python knowledge/tools/build_completion_reports.py

Deterministic views over stores, journals and the packet ledger; read-only
with respect to scientific stores.
~~~~

### `knowledge/tools/docling_batch.py`
*182 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Checkpointed local Docling continuation. Each bounded job banks before the next.

The default only writes the full-corpus plan. --run executes a limited number of
jobs in isolated workers, with wall-clock timeouts; it never starts paid agents.
~~~~

### `knowledge/tools/docling_enrich.py`
*221 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Run one bounded local Docling formula/table enrichment and bank derivatives.

No cloud inference; remote services are disabled. Model-weight downloads may
require network on the first run. Existing reviewed derivatives are preserved.
~~~~

### `knowledge/tools/docling_formula_regions.py`
*111 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Re-enrich visually located PDF formula regions using Docling's local stage.

Use when automatic layout split a fraction or missed the equation. Region JSON
contains original page numbers and PDF-point BOTTOMLEFT boxes, never expected
formula text. The raw output remains a candidate requiring original-page review.
~~~~

### `knowledge/tools/fresh_checkout_check.py`
*87 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Prove that a commit of the current knowledge changes yields a working checkout.

    .venv-jax/bin/python knowledge/tools/fresh_checkout_check.py [--output report.json]

Exports ``HEAD`` with ``git archive`` into a temporary directory (no refs,
index or objects are written), overlays exactly the working-tree files a commit
would add or change (tracked modifications plus untracked, non-ignored files),
and runs the corpus-free knowledge suite there. It also runs the same suite on
the bare ``HEAD`` export, so the before/after difference is explicit. The
listed files are the ones the owner must ``git add``.
~~~~

### `knowledge/tools/kg_pipeline.py`
*10 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Compatibility entry point; the CLI and this script use the same pipeline.
~~~~

### `knowledge/tools/packet_queue.py`
*103 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Command-line access to the persistent source-packet state machine.

    .venv-jax/bin/python knowledge/tools/packet_queue.py summary
    .venv-jax/bin/python knowledge/tools/packet_queue.py list --state QUEUED
    .venv-jax/bin/python knowledge/tools/packet_queue.py show PACKET_ID
    .venv-jax/bin/python knowledge/tools/packet_queue.py next QUEUED DOCLING_RUNNING --role S1 --model MODEL
    .venv-jax/bin/python knowledge/tools/packet_queue.py transition PACKET_ID RAW_CAPTURED --role S1         --model MODEL --run-id ENR_... --note "..."

Every write is atomic and journaled with the acting role and model. This tool
never starts agents, never runs Docling and never edits scientific stores.
~~~~

### `knowledge/tools/pdf_to_markdown.py`
*427 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Faithful PDF -> agent-readable Markdown conversion for the LREKit propulsion corpus.

Design contract
---------------
* The source PDF stays authoritative.  This module produces a *derivative*
  Markdown rendering and records the ``source PDF -> generated Markdown``
  relationship, the source checksum, the converter identity, and honest
  extraction diagnostics.
* Nothing is invented.  When a page yields no reliable text the page marker is
  still emitted and the failure is recorded; no plausible text is synthesised.
* Output format matches the pre-existing ``propulsion_texts_for_agents`` corpus
  (``<!-- PDF page N -->`` markers, optional ``> [!WARNING]`` callout header)
  so old and new conversions are consumed identically.

Statuses recorded per document: ``good``, ``needs_review``,
``partial_conversion``, ``pdf_conversion_failed``.
~~~~

### `knowledge/tools/session_completion_report.py`
*171 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Write the session completion report from the packet queue, integration logs and journal.

    .venv-jax/bin/python knowledge/tools/session_completion_report.py         --session knowledge/workflows/results/agentic-2026-10-16         --fresh-checkout <fresh_checkout.json> --out knowledge/reports/KNOWLEDGE_GRAPH_COMPLETION_2026-10-16.md

Every count in the generated tables is recomputed from those records, so the
report cannot drift from what was actually integrated; the narrative sections
are passed in by the reviewer (``--narrative``) and kept verbatim.
~~~~

### `knowledge/tools/session_replay_check.py`
*198 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Replay every reviewed file of a session's finished packets and prove the stores do not change.

    .venv-jax/bin/python knowledge/tools/session_replay_check.py --since 2026-10-16         --output knowledge/reports/session_replay_2026-10-16.json

For each FINAL_AUDIT_ELIGIBLE packet with history on or after ``--since``, every
reviewed file is checked against the SHA-256 bound at VERIFIED and re-applied
through the same idempotent paths S3 uses:
- extraction results are re-banked;
- parser reviews and duplicate retirements are re-recorded;
- verdicts are re-applied.
The scientific stores are compared before and after with timestamps ignored.
A non-empty ``stores_changed_ignoring_timestamps`` or ``changed_reviewed_files``
means the replay is not a no-op. ``success`` also requires every step to resolve;
missing source/reviewed files, failed banking and refused new verdicts remain
failures even when they leave the stores unchanged. Exact historical assertions
are retained history, never a new current verification. The run exits 1 unless
both requirements hold.
~~~~

### `knowledge/tools/sync_corpus.py`
*266 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Synchronise ``propulsion_texts/*.pdf`` with ``propulsion_texts_for_agents/markdown``.

Idempotent.  Run it repeatedly: unchanged sources are not reconverted, and a
run with nothing to do rewrites nothing.

Mapping evidence, strongest first:
1. source PDF sha256 recorded in the conversion manifest,
2. explicit ``source_pdf`` path recorded in the manifest,
3. the generated Markdown's own ``lrekit-kg-conversion`` provenance block,
4. filename, used only as a last resort and reported as weak evidence.
~~~~

### `knowledge/tools/text_layout.py`
*109 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Structure-preserving fallback renderer for PDFs the Markdown parser cannot read.

Some corpus PDFs (notably scanned pages carrying an invisible text layer, and
some older AIAA/ASME scans) yield an empty result from ``pymupdf4llm`` even
though PyMuPDF's own text extractor reads them fine.  Rather than silently
emitting an empty document, the converter falls back to this renderer, which
rebuilds reading order and a conservative amount of Markdown structure from the
raw span geometry.

Nothing here reconstructs mathematics.  Ambiguous layout is emitted as-is.
~~~~

### `knowledge/tools/trace_flow.py`
*105 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Trace one source object end to end through the knowledge system.

    .venv-jax/bin/python knowledge/tools/trace_flow.py EQ_... --query "son movable pintle minimum area" [--out trace.json]

Reads only: manifest, enrichments (Docling runs and candidates), the object's
store record and provenance, the assembled graph, model reviews that cite the
object, and ``kg.api.retrieve_for_task``. Each hop is reported with its state so
a missing link is visible (``MISSING``) rather than silently skipped.
~~~~

### `knowledge/tools/triage_pdfs.py`
*110 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Checkpointed native-text sampling; isolates malformed PDFs with timeouts.

Run with the PDF environment, e.g. .venv-kg/bin/python knowledge/tools/triage_pdfs.py.
This is a conversion triage sample, not a complete OCR census or accuracy score.
No OCR, model inference, corpus edits, or network requests are performed.
~~~~

### `knowledge/tools/validate_audit_replay_2026_10_16.py`
*162 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Replay the completed 2026-10-16 audit journals and verify preserved content.

Run with the repository's scientific Python environment. This deliberately banks
the twelve named journals and rebuilds derived indexes. It checks the saved White
relation inventory independently of replay stability: stable missing data must
not pass. This is a bounded audit fixture, not a corpus completeness certificate.
~~~~

### `knowledge/tools/work_journal.py`
*54 lines · stage X06 Knowledge graph, source corpus and packaged data assets (not in module_stage_map; chosen)*
~~~~text
Append to (or show) the agentic work journal.

    .venv-jax/bin/python knowledge/tools/work_journal.py append --role S2 --model MODEL         --task "structure packet P" --state CANDIDATES_PREPARED --packet P         --input path --output path --note "..."
    .venv-jax/bin/python knowledge/tools/work_journal.py tail --n 20

Mechanical worker roles (S1-S3) cannot record authoritative scientific states.
~~~~

### `raosim/data/tools/acquire_sources.py`
*165 lines · stage X06 (module_stage_map: Knowledge graph, source corpus and packaged data assets)*

~~~~text
Acquire research sources without importing or changing LREKit physics.

Catalog entries describe candidate uses, not a completed scientific review.
Receipts record acquisition and byte identity, never training eligibility.
~~~~

### `raosim/data/tools/verify_collection.py`
*186 lines · stage X06 (module_stage_map: Knowledge graph, source corpus and packaged data assets)*

~~~~text
Check source integrity and registry coverage; never certify model validity.
~~~~


## X07 - Test harness and regression arg files

*174 files.*

### `.github/workflows/publish.yml`

*98 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
(no header comment)
~~~~

Inferred role: (inferred) GitHub Actions workflow 'Publish Python package' that builds the distribution on v* tags or manual dispatch and publishes it to TestPyPI and PyPI (jobs build, publish-to-testpypi, publish-to-pypi).

### `.github/workflows/test.yml`

*95 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
(no header comment)
~~~~

Inferred role: (inferred) GitHub Actions workflow 'Test' that runs the knowledge-graph, non-slow pytest and slow-nozzle jobs on pull requests, pushes to main and codex/** branches, and manual dispatch.

### `examples/cli/lox_rp1_13kn_copper_316l_full_engine.args`

*73 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
Full-engine CLI regression sample.

Run from the repository root with:
  lrekit @examples/cli/lox_rp1_13kn_copper_316l_full_engine.args

Override the output directory by appending another --out after the @file.
~~~~

### `examples/cli/test_13kn_movable_pintle.args`

*72 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
Variant check: Son (2017) continuous-gap MOVABLE pintle on the 13 kN
sea-level LOX/RP-1 benchmark operating point.

Companion to examples/cli/test_13kn_sealevel_regen_allstl.args — the movable
architecture (--injector-architecture son_continuous_movable) replaces the
fixed discrete radial exits, so it cannot share the machined coaxial CAD
benchmark run.  Operating point, chamber, and feed values are identical to
the main benchmark file (provenance documented there).  Without a measured
Cd map / metrology / actuator artifacts the movable model runs in its
labelled constant-Cd screening mode and the release gates stay
evidence-blocked — that is the expected result of this check, not a FAIL.

  PYTHONPATH=. python -m raosim @examples/cli/test_13kn_movable_pintle.args
~~~~

### `examples/cli/test_13kn_rao_bvp_flowfield.args`

*44 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
Variant check: full Rao variational MOC (rao-bvp) solve with flow-field
render and construction diagrams, on the 13 kN sea-level benchmark
operating point.

Companion to examples/cli/test_13kn_sealevel_regen_allstl.args — that file
owns the authoritative Bezier/TOP-chart export benchmark; --flowfield (and
the kernel_fan / rao_topology / bde_mesh construction diagrams plus the
characteristic-crossing CSV) require --contour-method rao-bvp and live
here.  Expect minutes of JAX BVP runtime; non-convergence exits nonzero
(no --allow-unconverged).

  PYTHONPATH=. python -m raosim @examples/cli/test_13kn_rao_bvp_flowfield.args
~~~~

### `examples/cli/test_13kn_sealevel_regen_allstl.args`

*156 lines · stage X07 Test harness and regression arg files (assigned: not in module_stage_map)*

~~~~text
Regression-test engine: 13 kN sea-level LOX/RP-1, non-scalar (station-wise)
regen wall sizing, machined pintle STEP, and pump CAD package.

Run from the repository root with:
  PYTHONPATH=. python -m raosim @examples/cli/test_13kn_sealevel_regen_allstl.args
(or `lrekit @examples/cli/test_13kn_sealevel_regen_allstl.args` when installed)

This file is the full benchmark package. Any later command-line flag
overrides the value here; append `--max-nfev 0 --out builds/tests/13kn_sl_fast`
only when you intentionally want a quick seed-geometry smoke run.

Parameter provenance (see TEST_PROMPT.md and docs/REPO_PHYSICS_AUDIT_2026-07-02.md):
  - LOX/RP-1 gamma/Mw/Tc/O-F from the built-in table anchored to Sutton &
    Biblarz RPE 9th ed. Table 5-5 (raosim/propellants.py source note).
  - Pc = 3.0 MPa: verified feasible for the station-wise GRCop-84 regen
    sizing at this thrust class; Bartz throat heat flux scales ~Pc^0.8
    (Bartz 1957, propulsion_texts/technical-notes-1957.pdf), and 7 MPa was
    probed thermally infeasible for every catalog liner at 13 kN.
  - Sea-level matched expansion Pe = Pa = 101325 Pa (optimum CF when
    Pe = Pa, Sutton RPE; repo relation expansion_ratio_from_pressure):
    epsilon = 4.524, pa/p0 = 101325/3.0e6 = 0.033775.
  - GRCop-84 liner + Inconel 718 closeout per NASA GRCop chamber work
    (propulsion_texts/gradl2016..2020.pdf); catalog T_max 1000 K, S_y 186 MPa.
  - SP-125 station-wise liner stress screens (eqs 4-28/29/31/32, audited
    2026-07-02).
  - Upstream throat radius Ru = 1.5 Rt explicitly (--ru-factor 1.5): the
    SP-8120 throat-approach radius-ratio upper bound (0.6-1.5, verbatim;
    SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS in raosim/throat_geometry.py) and
    the standard Rao/Sutton bell convention Ru = 1.5 Rt
    (propulsion_texts/nozzle_geometries.pdf, audited 2026-07-02).  The
    implied Hall inviscid Cd at gamma = 1.24 is 0.98963.  The file
    previously requested --cd-target 0.99, which inverts to Ru/Rt = 1.5275
    — outside the strict SP-8120 range now enforced — and aborted at
    argument validation (evaluation 2026-07-13, FINDING 1).
~~~~

### `knowledge/tests/test_kg_agentic_controls.py`
*195 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Concurrency, packet-state and role controls for multi-agent knowledge work.

Corpus-free: every test relocates the knowledge root into a temporary tree.
~~~~

### `knowledge/tests/test_kg_agentic_packet_ops.py`
*241 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Role operations on one packet: candidate preparation and reviewed-file integration.

Corpus-free. Rendering and graph assembly are stubbed; the tested contract is
that S2 bundles stay provisional and S3 integrates only the exact reviewed bytes.
~~~~

### `knowledge/tests/test_kg_artifact_integration.py`
*46 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Artifact retrieval preserves scope and never revives historical binding.
~~~~

### `knowledge/tests/test_kg_artifacts.py`
*228 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Artifact retrieval may observe simulated output without creating physical evidence.
~~~~

### `knowledge/tests/test_kg_audit_fixes.py`
*325 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Regression tests for the 2026-10-16 independent audit (Opus A findings A-01..A-11, N-04).

Corpus-free: stores, graph and code live in a temporary tree; grounding is stubbed
as in test_kg_review_provenance.py.
~~~~

### `knowledge/tests/test_kg_bank_gate.py`
*266 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
S3 must retain bank-report failures and block the packet's integration.
~~~~

### `knowledge/tests/test_kg_candidate_links.py`
*161 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
K8: source objects preserve raw parser identity without promoting parser authority.
~~~~

### `knowledge/tests/test_kg_citation_support.py`
*226 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
A citation is a route to inspect, never automatic scientific support.
~~~~

### `knowledge/tests/test_kg_claim_quantity_tokens.py`
*23 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
English connectors must not create physical-quantity associations.
~~~~

### `knowledge/tests/test_kg_completeness.py`
*206 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Corpus-free regressions for honest extraction coverage and ingestion states.
~~~~

### `knowledge/tests/test_kg_constant_basis.py`
*29 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
(no header comment)
~~~~

Inferred role: test that numerical solver-tuning constants are not classified as physical calibration and legacy basis labels are scoped (kg.constant_basis.reconcile).

### `knowledge/tests/test_kg_constraint_citations.py`
*63 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
A resolved report key cannot validate an engineering threshold.
~~~~

### `knowledge/tests/test_kg_coverage_constants.py`
*58 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Coverage gaps describe current constants while preserving stored history.
~~~~

### `knowledge/tests/test_kg_derivation_identity.py`
*98 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Long endpoint IDs cannot collapse distinct derivation facts or their history.
~~~~

### `knowledge/tests/test_kg_derivation_links.py`
*110 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Only explicit relations between uniquely current source records are banked.
~~~~

### `knowledge/tests/test_kg_docling_attempts.py`
*246 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Immutable Docling attempt and bounded scheduling tests; no model inference.
~~~~

### `knowledge/tests/test_kg_enrichment.py`
*177 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Parser candidates must not acquire scientific authority or stale links.
~~~~

### `knowledge/tests/test_kg_execution_reach.py`
*68 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Static call evidence must remain separate from runtime and validation claims.
~~~~

### `knowledge/tests/test_kg_extraction.py`
*385 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Isolated extraction/recovery regression tests; no literature or agents needed.
~~~~

### `knowledge/tests/test_kg_implementation_freshness.py`
*131 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Exact AST locator regressions; no inspected code is imported or executed.
~~~~

### `knowledge/tests/test_kg_minicorpus.py`
*181 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Run the real KG rebuild/retrieval pipeline without local literature or JAX.

PDFs and derivative text below are authored synthetic fixtures. Only the external
project-science catalogs are replaced; sync, grounding, stores, chunking, vector
indexing, repository harvest, graph, validation and cache reset remain real.
~~~~

### `knowledge/tests/test_kg_multisource_reviews.py`
*354 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Multi-source review retention, link fidelity and declared-link bank integrity.

Synthetic sources and code isolate graph/retrieval behavior from the local corpus.
~~~~

### `knowledge/tests/test_kg_parameters.py`
*210 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Repository defaults stay distinct, inspectable and free of source support.
~~~~

### `knowledge/tests/test_kg_parser_review_history.py`
*140 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Parser-QA assertions retain history and replay without changing the latest verdict.
~~~~

### `knowledge/tests/test_kg_pdf_conversion.py`
*204 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Corpus-free checks of page-level OCR eligibility and honest diagnostics.
~~~~

### `knowledge/tests/test_kg_pipeline.py`
*224 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Corpus-free regression tests for preservation, sync and incremental rebuilds.
~~~~

### `knowledge/tests/test_kg_record_identity.py`
*122 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Duplicate survey facets stay inspectable; conflicts cannot overwrite stores.
~~~~

### `knowledge/tests/test_kg_registry.py`
*72 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Registry refresh must not erase independently harvested source evidence.
~~~~

### `knowledge/tests/test_kg_relationship_states.py`
*68 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Retired relation assertions remain history even when endpoints stay current.
~~~~

### `knowledge/tests/test_kg_repository_tracking.py`
*76 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
A checkout must contain every knowledge module, test, tool and reviewed journal.

Regression for 2026-10-16: a blanket ``knowledge/`` ignore rule left tracked
modules importing untracked ones (``kg.execution_reach``), so a clean export
of HEAD failed at import and the corpus-free CI job could not run.
~~~~

### `knowledge/tests/test_kg_retrieval.py`
*313 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Source-free retrieval regressions; all index/graph/store writes are temporary.
~~~~

### `knowledge/tests/test_kg_review_provenance.py`
*221 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Reviewer identity, self-verification refusal and fingerprinted model reviews.

Corpus-free: grounding is stubbed exactly as in test_kg_extraction.py.
~~~~

### `knowledge/tests/test_kg_session_replay.py`
*210 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Session replay distinguishes unchanged stores from fully resolved operations.
~~~~

### `knowledge/tests/test_kg_system.py`
*232 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Tests for the LREKit knowledge system.

Run:  .venv/bin/python -m pytest knowledge/tests -q

Two layers:
* deterministic-unit tests (chunker, grounding, symbol resolution, sync scan, stores)
* acceptance queries from the master instructions (§46): each must return a
  provenance-backed, id-addressable context package - or an explicit state
  (never a silent empty list).
~~~~

### `knowledge/tests/test_kg_test_classification.py`
*101 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
Function evidence must not inherit a module's unrelated literature claim.
~~~~

### `knowledge/tests/test_kg_variable_units.py`
*110 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
(no header comment)
~~~~

Inferred role: test that kg.quantities.resolve_symbol and kg.variable_links keep overloaded symbols (g, gamma, specific heats, speeds) distinct by units and meaning.

### `knowledge/tests/test_kg_verification_freshness.py`
*250 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen)*
~~~~text
K12: independent source verdicts are scoped to the assertion inspected.
~~~~

### `tests/test_advanced_thermofluids.py`
*135 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_sourced_cyclic_tangent_modulus_softens_with_stress, test_spectral_band_radiation_is_positive_and_optically_bounded, test_full_annular_network_conserves_flow_and_resolves_every_branch, test_real_fluid_backend_never_silently_falls_back, test_real_fluid_aliases_use_the_canonical_backend, test_zuber_chf_increases_with_latent_heat, test_regen_integrates_radiation_and_full_hydraulic_network
~~~~

### `tests/test_bartz_heat_flux.py`
*168 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Full Bartz (1957) gas-side heat-transfer correlation.

Pins the exact formula (σ factor, the pressure/area/curvature exponents,
the recovery temperature, the SI gas-property estimates) and the
physical behaviour (textbook throat heat-flux magnitude, peak near the
throat, σ and Mach trends).  Refs: Bartz, *Jet Propulsion* 27(1) 1957;
Huzel & Huang, NASA SP-125 §4.
~~~~

### `tests/test_benchmarks.py`
*152 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_benchmark_manifests_and_curves_are_valid, test_lea_top_bezier_benchmark_passes_explicit_geometry, test_benchmark_report_contains_source_status_and_metrics, test_vulcain_s1_wall_pressure_trend_is_report_only, test_benchmark_cli_writes_reports, test_scarfed_case_is_explicitly_unsupported_by_axisymmetric_moc, test_current_bvp_passes_strict_rao1958_nozzle_b_tables_2_and_3
~~~~

### `tests/test_cea_catalog.py`
*321 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Installed, hash-pinned CEA catalog resources and explicit sentinel wiring.
~~~~

### `tests/test_cea_pinned.py`
*88 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Contract tests for parity-aligned pinned chamber thermochemistry.
~~~~

### `tests/test_cea_surface_validation.py`
*217 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Phase-2 held-out CEA value/gradient gate without a RocketCEA dependency.
~~~~

### `tests/test_chamber_geometry.py`
*226 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_max_feasible_shoulder_factor_matches_analytic_cap, test_auto_shoulder_factor_is_fraction_of_cap_and_builds, test_auto_shoulder_factor_rejects_bad_inputs, test_auto_shoulder_factor_opens_up_at_shallower_convergent_angle, test_hall_throat_discharge_coefficient_and_inverse, test_shared_throat_spec_removes_chamber_nozzle_discontinuity, test_sampled_chamber_volume_matches_lstar, test_volume_is_exact_polyline_frustum_not_trapezoidal_area ...
~~~~

### `tests/test_characteristic_pairing.py`
*184 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Characteristic invariant pairing — the 2026-06-11 correction.

Root cause (identified by ibrahim): the rao_residuals/moc/jax stack
enforced d(θ+ν) along the C+ (θ+μ) family and d(θ−ν) along C− — the
families' relations mirrored — and the axisymmetric source carried a
spurious cosμ/cos(θ±μ) factor.  Correct relations (Anderson, *Modern
Compressible Flow* §11.4; Zucrow & Hoffman Vol. 2 Ch. 17), nodes
ordered downstream, S = sinθ sinμ / r:

    C− (slope θ−μ):  d(θ + ν) = +S ds
    C+ (slope θ+μ):  d(θ − ν) = −S ds

These tests pin the correction against two independent oracles:
the M3.5Perf kernel march (RRC rows are true C− characteristics,
oracle-matched to NASA output files) and the verbatim NASA
``Deriv``/``RungeKutta`` port (LRC integration with the axisymmetric
terms in closed form).
~~~~

### `tests/test_cli_module_exit.py`
*19 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
python -m raosim must preserve the CLI's process status.
~~~~

### `tests/test_compare_nasa_reference.py`
*100 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_compare_report_declares_fixture_overlay_provenance, test_compare_report_marks_sauer_kernel_complete_without_reference_promotion
~~~~

### `tests/test_comparison.py`
*94 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.nozzle_comparison

Validates the comparison utilities: divergence loss, efficiency,
and comparison table generation.
~~~~

### `tests/test_conical.py`
*95 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.conical

Validates conical nozzle divergence factor and contour geometry.
~~~~

### `tests/test_design_v2.py`
*1308 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining _write_frozen_lox_rp1_table, _prelim_input, _install_fake_rocketcea, test_preliminary_v2_constant_gamma_runs, test_v2_explicit_efficiency_split_survives_constant_gamma, test_v2_cea_preserves_legacy_or_explicit_efficiency_convention, test_v2_frozen_variable_cp_bezier_uses_profile_and_fails_validation_gate, test_v2_frozen_variable_cp_closes_target_thrust_and_matched_pressure ...
~~~~

### `tests/test_design_workflow.py`
*203 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_bezier_contour_has_reliability_metadata, test_rao_wrapper_keeps_requested_exit_radius_or_warns, test_design_gates_record_bad_cad_thickness, test_step_and_solid_stl_exports, test_solid_stl_flange_closes_bore_and_wall_offset, test_station_wise_normal_wall_offset_and_step_export, test_require_brep_rejects_faceted_fallback, test_design_nozzle_api_writes_report_and_step ...
~~~~

### `tests/test_electric_technology.py`
*218 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
One electric-drive technology set for both pipelines (audit §1.3 item 19).

The traditional ``raosim.pumps`` defaults, the MDO ``MissionSpec`` electric
fields and the MDO→traditional handoff must all read one registered set, and a
mission may never carry numbers its declared set does not hold.
~~~~

### `tests/test_engine.py`
*58 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.engine
~~~~

### `tests/test_fatigue.py`
*156 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Low-cycle fatigue: Coffin-Manson N_f model + its gating policy.

The nominal strain driver is α·ΔT, obtained directly from SP-125 equation
4-28's thermal stress E·α·ΔT; N_f inverts the standard strain-life relation.
The local NASA material reports now supply a Coffin-Manson fit for NARloy-Z
and direct total-strain/life regressions for GRCop-84.  These are sourced
screening gates, not a claim of hardware qualification.
~~~~

### `tests/test_fidelity_scope_gates.py`
*162 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Two scope gates: separation topology, and milling practice.

Neither adds physics. Both convert a silent assumption into a stated one, which
is the cheaper half of the fidelity roadmap and the half that stops a result
being read as more than it is.
~~~~

### `tests/test_flange_cad.py`
*127 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.flange_cad — chamber flange solid and fastener callout.
~~~~

### `tests/test_flow_viz.py`
*175 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Steady flow-field visualisation (raosim.flow_viz): the MOC net is
interpolated into Mach/pressure/flow-angle/temperature fields with
characteristics and streamlines.  Headless (Agg) smoke coverage.
~~~~

### `tests/test_frozen_flow.py`
*728 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Thermally-perfect frozen-composition quasi-1-D flow contracts.
~~~~

### `tests/test_gas_dynamics.py`
*105 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.gas_dynamics

Validated against the worked example in the Rao nozzle formulation document:
  ε = 10, γ = 1.4  →  Me ≈ 3.9226, pe/p0 ≈ 0.007306, Te/T0 ≈ 0.2453
                       ν(Me) ≈ 64.75°,  Cf_vac ≈ 1.647
                       c*(γ=1.4, R=287, T=3000) ≈ 1355 m/s
~~~~

### `tests/test_injector.py`
*1468 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.injector — pintle hydraulic sizing and gates.
~~~~

### `tests/test_injector_export.py`
*935 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.injector_export - pintle reference geometry + output package.
~~~~

### `tests/test_input_validation.py`
*60 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Workflow-independent engine input validation.
~~~~

### `tests/test_interface.py`
*338 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining _gate, test_resolve_bolted_interface_geometry_matches_flange_and_face, test_bolt_pattern_rejects_degenerate_counts_but_allows_three_explicit, test_invalid_joint_separation_factor_is_rejected, test_interface_screen_rejects_nonfinite_or_nonpositive_contract_values, test_interface_screen_rejects_degenerate_explicit_bolt_count, test_cad_contract_rejects_a_mutated_degenerate_resolution, test_bolt_pattern_boundary_roundoff_does_not_fail ...
~~~~

### `tests/test_jax_assembly_parity.py`
*225 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J2 gate (JAX_DIFFERENTIABLE_PLAN.md §7): assembled-residual parity.

``raosim.jax.assembly.make_residual`` must reproduce the NumPy
``_scaled_rao_bvp_residual`` (which folds in ``_unpack_bvp``,
``_rao_bvp_residual_groups``, ``_coupled_wall_residuals``, and
``RaoResidualGroups.flat()``) to 1e-8 on identical packed unknown
vectors.  States are the *real* Phase-6 seed (kernel-seeded CE + Bezier
wall) for the reference case, plus random in-bounds perturbations.

Passing this proves the JAX port changed numerics, not physics — every
downstream convergence result is then attributable to the optimizer
(exact Jacobian LM), not to a different residual.

Also covers J3 wiring on the real system: jacfwd/jacrev of the assembled
residual are finite and match finite differences.
~~~~

### `tests/test_jax_bvp.py`
*180 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J3 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the differentiable solve infrastructure.

Demonstrates, on a *real* Rao residual built from the J2-validated leaves:
  1. Optimistix Levenberg-Marquardt + exact autodiff Jacobian converges the
     algebraic Rao stationarity system (REWRITE_PLAN.md §2.B closed-form optimum
     condition) to machine precision.
  2. The exact jacrev Jacobian matches a finite-difference Jacobian.
  3. Gradients through the converged solution (implicit function theorem) are
     finite and FD-consistent — the J6 sensitivity mechanism.
  4. The §2.D mass-closure integral matches an independent NumPy implementation
     and behaves sensibly.
  5. pack/unpack round-trips.

This is the solve *infrastructure*.  The full coupled-wall grouped residual and
the Phase-6 ``max_scaled < 2e-3`` gate (J4) require the MOC marching grid
(J3b) and are intentionally out of scope here.

Skips cleanly if JAX/Optimistix are absent.
~~~~

### `tests/test_jax_convergence.py`
*320 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J4 (JAX_DIFFERENTIABLE_PLAN.md §7): convergence of the differentiable backend
on the Phase-6 reference case (epsilon=10, length_pct=80, gamma=1.4,
PHYSICS_WEIGHT=1.0, couple_wall=True, kernel_d_fraction_max=0.7).

Status after the J4 spike (June 2026) — the plan's §10 diagnosis branch fired,
and productively.  Exact-Jacobian LM removed the FD-noise confound and the
stall decomposed into three layers, two of which are now fixed:

    max_scaled ~ 8      scipy FD Jacobian + degenerate kernel   (pre-spike)
    max_scaled ~ 2.8-3.5  exact-Jacobian LM, same degenerate kernel
    max_scaled ~ 2.5    after the KLThroat integer-division fix
                        (kernel march now runs; BD is a real RRC —
                        tests/test_nasa_kernel_march_parity.py)
    max_scaled ~ 0.5-0.7  after the upstream-radius (Ru) fix + sane
                        theta_B seed: real kernel, real BD anchor

The remaining gap to the 2e-3 gate is *seed topology*, not optimisation:
``calc_lrc_de`` / ``set_theta_b`` still collapse to a degenerate D~E
topology on marched kernels (mass_BD ~ 1e-8, DE of 1-2 nodes), so the CE
seed stays the legacy linear ramp far from the Rao basin.  That is
REWRITE_PLAN Phase-12 work (NASA find_point_e / CalcLRCDE), tracked by the
explicit convergence diagnostics below.  Homotopy over PHYSICS_WEIGHT, LM restarts, and both bound
treatments (sigmoid reparametrisation, vanishing barrier) all land on the
same stall point — evidence the obstruction is structural, exactly what
§10 predicted exact Jacobians would reveal.

The tests pin the measured gains and remaining gates as regression floors.
~~~~

### `tests/test_jax_moc_kernel.py`
*275 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J3b — differentiable throat-arc kernel march (raosim.jax.moc_kernel).

Landed state (2026-06-12, second increment):

* the march reproduces the NumPy NASA oracle kernel (``build_kernel``)
  at BIT-PARITY — same node counts, max |ΔM| ≤ 5e-10 over all BD
  nodes at the reference config — by porting NASA's exact row policy:
  per row a RAW wall point (the C+ from prev[1] TERMINATING on the
  arc) unless its step exceeds ``dtheta_limit``, in which case the
  SPECIAL prescribed-angle point at θ_prev + dθ/2 is inserted and the
  row grows by one; raw rows keep width and shift the C+ pairing.
  The decision/width/pairing are traced; shapes stay static (padded).
* ``d(BD)/d(theta_B)`` is exact: the pre-clamp grid is
  θ_B-independent, so the sensitivity flows only through the final
  clamped row — jacfwd matches central FD to ~1e-8 relative.
* the NASA dθ-form unit-process ports match the NumPy originals at
  their 1e-10 fixed points (parity ≤ 1e-8 pinned below; measured
  1e-12..1e-16 on oracle row inputs).

Design history (all measured, recorded in the module docstring of
``raosim/jax/moc_kernel.py``): a static-width forward march explodes
(no step limit); a regridded inverse march is weakly unstable (axis
Mach runaway); an all-special prescribed-grid march with Anderson-form
processes is stable but converges to the WRONG continuum (wall chain
under-accumulates ν: ν_w−θ_w → −0.67° at B vs the oracle's physical
+0.68° — C+ termination on the wall is required physics, not grid
adaptivity).
~~~~

### `tests/test_jax_primitives.py`
*115 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J1 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the JAX primitives must reproduce the
literature-checked NumPy closed forms in ``raosim.gas_dynamics``.

- Closed forms: bit-parity to 1e-10 across M and gamma sweeps.
- Iterative inverses (mach_from_prandtl_meyer, mach_from_area_ratio): round-trip
  consistency to 1e-10 and cross-agreement with the NumPy Newton output to 1e-8
  (the two solvers use different stopping rules, so exact bit-parity is not the
  right contract — round-trip is).
- Sonic safety: values finite and derivatives finite at M -> 1.

Skips cleanly if JAX is not installed, so the existing suite is unaffected.
~~~~

### `tests/test_jax_residual_parity.py`
*167 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J2 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the JAX residual leaves reproduce the
NumPy residual functions on a fixed CE/wall state.

Covers the pure node-wise leaves the BVP differentiates through:
  - rao_residuals: C+/C- axisym, left-Mach, wall tangency, C+ child position
  - rao_variational CE blocks: axisym C+/C- groups, algebraic stationarity,
    differential (fd) stationarity, left-Mach /ds, smoothness regularization

The full grouped/weighted assembly (mass/length/stationarity integrals) lands in
J3 — it is coupled to the marching construction, not a pure leaf — so it is out of
scope here by design.

Skips cleanly if JAX is absent.
~~~~

### `tests/test_jax_sensitivities.py`
*172 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J6 v1 gates (plan §7: "smoke + a known-sign sensitivity").

Solve-free tests (fast): Cf parity numpy<->jax on a real fixed-end DE,
the analytic dCf/dpa known-sign/value identity, and an FD cross-check
of the node gradient at a non-converged state.  The end-to-end
``rao_sensitivities`` smoke runs a full JAX solve and is @slow
(host-budget; the sandbox per-call cap cannot fit it).
~~~~

### `tests/test_jax_thermal_design_opt.py`
*223 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Constrained differentiable nozzle design (raosim.jax.thermal +
raosim.jax.design_opt).

* the JAX thermal physics matches the NumPy originals (Bartz σ /
  Sieder-Tate / Schmucker) at the float-parity level;
* gradients are exact (jacfwd vs central FD);
* the constrained optimiser (Optimistix BFGS + penalty) maximises Cf
  and lets the constraints SHAPE the design: loose -> unconstrained Cf
  optimum; tight separation -> ε capped at the limit; tight cooling ->
  throat opened (and infeasibility reported when geometry alone can't
  satisfy it, the physical finding from raosim.thermal_design).
~~~~

### `tests/test_jax_theta_b_solve.py`
*148 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
J3b-2 — θ_B as a solved unknown inside the BVP (raosim.jax.theta_b_solve).

The wiring gate: at the seed θ_B the live-BD residual must equal the
static-``kernel_bd`` residual (the march reproduces the seed kernel's
BD at bit parity, so any deviation here is plumbing, not physics).
Plus: d(residual)/dθ_B is finite and FD-exact, and a coarse
end-to-end LM solve over [u, θ_B] runs and stays in bounds.
~~~~

### `tests/test_job_execution_integrity.py`
*334 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Regression checks at the public job, CLI, verdict and report boundaries.
~~~~

### `tests/test_job_reference_parts.py`
*184 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Unified jobs can export reference CAD with an honest part manifest.

``parts='reference_cad'`` routes the authoritative re-evaluation's STEP into
the run directory and records the engine part manifest from
``raosim.mdo.build``.  That module records path-only artifacts as unverified,
so the package is always partial and no part may claim geometry fidelity.
~~~~

### `tests/test_joint_design.py`
*152 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Joint wall + channel optimizer (raosim.thermal_design).

Co-sizes the hot-wall thickness AND the channels against the coupled
thermal limit (peak wall T ≤ material T_max / margin) and the SP-125
eq. 4-31 combined structural stress, reporting the thickness feasibility
band  max(t_pressure, t_mfg) ≤ t_hot ≤ t_thermal.
~~~~

### `tests/test_legacy_io.py`
*135 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
(no header comment)
Inferred role: [inferred] test module defining test_parse_nasa_wall_out_sample, test_parse_nasa_kernel_and_rao_outputs, test_parse_nasa_summary_initial_data, test_every_m35perf_reference_file_parses, test_parse_nasa_tecplot_zone_files, test_parse_nasa_iteration_and_kernel_tables
~~~~

### `tests/test_lrekit_guided.py`
*115 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
The guided interview: describe the engine, not the solver.
~~~~

### `tests/test_lrekit_job_cli.py`
*174 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
The job verbs, and the compatibility guarantee around them.

O/F optimization was implemented long before it was reachable from a command
line, which meant in practice it was not usable. These gates keep the unified
workflow reachable, and keep the verdict visible in the process exit code.
~~~~

### `tests/test_mass_ledger.py`
*796 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Tests for raosim.mass_ledger and its differentiable mirror raosim.mdo.mass.

The point of these tests is that hardware mass is *integrated geometry*, not a
correlation: every assertion below is checkable by hand from the shapes the CAD
exporters build.  Where a quantity cannot be resolved, the ledger must say so
with a reason rather than emit a zero.
~~~~

### `tests/test_materials.py`
*150 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
Materials catalog + SP-125 eq. 4-31 wall-stress relation.

The catalog (``raosim.materials``) feeds one coherent set of wall
properties into channel sizing (k -> wall temperature) and the structural
screen (E, a, v -> thermal stress), grounded in NASA SP-125
(``propulsion_texts/19710019929.pdf``).
~~~~

### `tests/test_mdo_bounds.py`
*191 lines · stage X07 Test harness and regression arg files (not in module_stage_map; chosen as cross-cutting test harness)*

~~~~text
tests/test_mdo_bounds.py — design bounds derived from the architecture.

The defect these replace: ``Pc in [1.5, 6.0] MPa`` and ``eps in [3, 40]`` were
hard-coded from the 13 kN LOX/RP-1 baseline and then applied unchanged to every
propellant, ambient and thrust class.

Two properties are pinned here.

**One source of truth.**  A hard bound and the constraint that screens the same
limit must come from the same data.  ``eps`` is bounded by the Rao chart grid
that ``chart_domain_margin`` uses; extend the chart and the design space widens
with it, automatically.

**Sourced or labelled.**  Every chamber-pressure search window either cites
Parsley and Zhang (2004) / SP-125, or is flagged ``literature=False``.  A
sourced performance optimum is still not mislabeled as a hard-validity domain.
~~~~

### `tests/test_mdo_build.py`
*164 lines · stage X07 test-harness*

~~~~text
The expected-part manifest and the achieved-fidelity contract.

A partial assembly must never be labelled a complete engine, and a part that
was never modelled must be present-and-unavailable rather than absent.
~~~~

### `tests/test_mdo_cea_surfaces.py`
*89 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_cea_surfaces.py — Phase-2 property-surface wiring.

The C¹ (P_c, O/F) surface machinery itself is gated in ``test_mdo_properties.py``.
This file gates the *engine wiring*: when a saved CEA table is supplied via
``MissionSpec.cea_table_path``, the engine consumes it and **O/F becomes a real
physical lever** (γ, T_c and hence c* vary with mixture ratio, so the flame
temperature moves both performance and the coking margin — the second coking
lever identified in the 2026-07-24 literature audit).  Without a table the
constant fallback is used and its provenance says so.

A *synthetic but physically shaped* table is used here (T_c peaking near
O/F ≈ 2.6 for LOX/RP-1); generating a real table needs RocketCEA on the host
(`scripts/sample_cea_surface.py`), which is the documented Phase-2 gate.
~~~~

### `tests/test_mdo_constraints.py`
*192 lines · stage X07 test-harness*

~~~~text
Single-manifest and tri-state feasibility contract gates.
~~~~

### `tests/test_mdo_contour_validation.py`
*833 lines · stage X07 test-harness*

(no header comment)

Inferred role: (inferred) tests for raosim.mdo.contour_validation: typed post-solve validation of the optimized TOP contour against the variational Rao contour, tolerance policy, unknown/failure verdicts, fail-closed B3/B5 gates, discrepancy flagging.

### `tests/test_mdo_convergence.py`
*122 lines · stage X07 test-harness*

~~~~text
Station-refinement study (plan §11 grid criterion, review §15.5).

The study re-solves one fixed design at doubled station counts; it must run on
every block with a non-default topology and return finite, self-consistent
numbers.  Whether the < 0.5 % criterion holds is a measured result reported by
the study, not something this test asserts.
~~~~

### `tests/test_mdo_coolant_htd.py`
*207 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_coolant_htd.py — supercritical heat-transfer-deterioration screen.

Source: Nasuti & Pizzarelli, J. Supercritical Fluids 168:105066 (2021),
``propulsion_texts/nasuti2021.pdf``, Eq. (9) and its K = 0.187 threshold.

The point of these tests is as much about what the screen *refuses* to do as
what it computes.  Evaluating Eq. (9) with the constant coolant properties the
cooling march currently carries would produce a flat number that misses the
(beta/cp) peak the criterion exists to detect — so the screen must come back
unavailable, loudly, rather than pass.
~~~~

### `tests/test_mdo_cooling.py`
*388 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_cooling.py — Phase-4a acceptance gate for ``raosim.mdo.cooling``.

Two comparisons per the §12.1 rule-4 discipline:

1. **NumPy parity oracle.** A plain-Python re-implementation of the same clean
   upwind finite-volume coolant march, built on the *audited* NumPy primitives
   (``physics.bartz_heat_transfer_coefficient``, ``physics.fin_efficiency``, the
   Sieder & Tate 1936 turbulent correlation at unit property ratio), must match
   the JAX ``cooling_march`` to ~1e-9.  Because the oracle uses ``physics.*`` and
   the block uses the ``jax.thermal.*`` mirrors, this also cross-validates the
   mirror claim.
2. **AD-vs-FD gate** (in test_mdo_cooling_gate.py) — kept separate so this file
   stays a pure parity check.

Also pins the two defects fixed while completing the block: the upwind march
crosses each of the n-1 wall segments exactly once (no double-counted segment),
and the channel-fit land width is *exposed* (``land_min``), never clipped away.
~~~~

### `tests/test_mdo_cooling_gate.py`
*78 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_cooling_gate.py — Phase-4a AD-vs-FD acceptance gate.

Total derivatives of converged cooling outputs (throat wall temperature, jacket
Δp, coolant exit temperature) with respect to the differentiable design inputs
(P_c, ṁ_cool) must come from the implicit differentiation of the Newton root
(``solve_cooling``) and agree with re-solved central differences to ~1e-4 in the
smooth regime (plan §11 Phase-4a gate, §12.1 rule 4).  This is the property that
makes the block usable inside a gradient-based MDF optimizer: exact derivatives
through the converged state, free of finite-difference step-size noise.
~~~~

### `tests/test_mdo_custom_propellant.py`
*252 lines · stage X07 test-harness*

~~~~text
Runtime registration of a custom propellant, and the wall-side mechanism.

The defect this closes is not only "custom propellants are unreachable". It is
that the coolant-side mechanism was inferred from a **coolant name string**:
``"LNG"``, ``"ethane"`` and ``"subcooled methane"`` all miss the built-in
supercritical list and would have been handed the hydrocarbon coking limit,
silently, for fluids whose governing physics is deterioration.
~~~~

### `tests/test_mdo_engine.py`
*242 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_engine.py — Phase-7 coupled whole-engine gate.

Pins the integration milestone: all four blocks solve as one differentiable
evaluation, the §5 hydraulic edge (cooling Δp → pump rise) is genuinely closed,
the η_c* combustion loop is a real (optional) fixed point with a measurable
ablation, and a single gradient flows from x through every block and both IFT
solves and matches finite differences.
~~~~

### `tests/test_mdo_envelope.py`
*178 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_envelope.py — SP-125 §2.1 item 6 envelope screen.

The envelope is the "smallest cylinder ... into which the engine would fit"
(SP-125 printed p. 31).  Three properties matter and are pinned here:

* it is **conservative** — the smooth radial maximum never understates the
  diameter, because an envelope screen that under-reports would pass designs
  that do not fit;
* its conservatism is **bounded and computable**, not unknown;
* it describes the **same jacket** the mass ledger integrates, so envelope and
  mass cannot disagree about one piece of hardware.
~~~~

### `tests/test_mdo_execute.py`
*407 lines · stage X07 test-harness*

~~~~text
One entry point, four-valued verdicts.

A user should not have to know whether an answer came from the differentiable
core or the traditional pipeline. These gates pin the parts of that promise
that are easy to erode: the verdict must be able to say "unknown", a failed
solve must not be reported as a violated requirement, and a violation must name
itself.
~~~~

### `tests/test_mdo_injector.py`
*155 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_injector.py — Phase-5 parity + branch-structure gate for
``raosim.mdo.injector``.

Parity oracle: the Son movable-pintle geometry is checked against the *audited*
``raosim.movable_pintle`` functions themselves (center_gap_area,
son_minimum_tip_area, opening_for_tip_area), and the orifice/TMR/blockage
algebra against the ``raosim.injector`` conventions, to ~1e-9.  Also pins the
plan-rule-8 structure: the movable minimum area is exposed as TWO branches with
a consistency inequality, never collapsed with a differentiable ``min()``.
~~~~

### `tests/test_mdo_injector_gate.py`
*72 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_injector_gate.py — Phase-5 AD-vs-FD acceptance gate.

The pintle block is closed-form (no implicit state), so total derivatives are
plain forward/reverse AD; they must still match re-evaluated central differences
to ~1e-4 and be jit-safe, so the block drops straight into the constraint
Jacobian of the MDF optimizer (plan §12.1 rule 4).
~~~~

### `tests/test_mdo_job.py`
*381 lines · stage X07 test-harness*

~~~~text
The versioned engine-job contract every front end compiles into.

The gates here encode the two structural decisions: design intent is always
explicit (never inferred from whether a value is present), and a job
*contains* the physical input contract rather than restating it.
~~~~

### `tests/test_mdo_material_map.py`
*166 lines · stage X07 test-harness*

~~~~text
The single typed catalog->traced-property mapper, and its atomicity.

Regression context: a host-side material label reached the traditional
pipeline through ``MaterialSpec.from_catalog`` while the differentiable core
kept flat class defaults, so ``--material GRCop-84`` optimized a
NARloy-Z-class liner.  The defaults were not even one alloy -- NARloy-Z
conductivity and density against a CuCrZr-class allowable.
~~~~

### `tests/test_mdo_metrics.py`
*124 lines · stage X07 test-harness*

~~~~text
The selectable-output registry, and its two structural rules.

Constraint metrics are *derived* from the constraint manifest rather than
re-listed, and every reportable metric is validated against the snapshot
contract. Both rules exist so the registry cannot become a third competing
definition of quantities the repository already names elsewhere.
~~~~

### `tests/test_mdo_nlp.py`
*88 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_nlp.py — Phase 8/9 gate for the ε-constraint NLP.

Checks the machinery the plan's Phase-9 gate needs: exact constraint Jacobians
(the Phase-8 deliverable — jacfwd through the two IFT solves matches finite
differences), a single min-mass solve reaching a feasible KKT point over the
enforced constraint set, and that coking is *reported* (not silently dropped).

The full multi-point Pareto sweep is host-only (compile + SLSQP over several
points exceeds the sandbox per-call budget); it is exercised via the CLI.  These
tests keep a single compile by fixing one Isp floor.
~~~~

### `tests/test_mdo_nlp_robustness.py`
*568 lines · stage X07 test-harness*

~~~~text
Fail-closed NLP evaluation, independent KKT check and multi-start.

These tests replace the coupled engine with closed-form analytic engines so the
optimizer contract is tested in isolation and in seconds (the pattern of
``tests/test_mdo_variable_policies.py``).  A failing region is produced with
``equinox.error_if``: under ``jax.jit`` it raises the same
``jax.errors.JaxRuntimeError`` that Lineax raises when a linear solve inside
the engine sees non-finite input, and un-jitted it raises
``equinox.EquinoxRuntimeError`` — the two runtime failure classes the NLP must
contain (MDO issues audit 2026-10-16, §1.1 items 1-4).
~~~~

### `tests/test_mdo_nozzle.py`
*199 lines · stage X07 test-harness*

~~~~text
Gates for the typed MDO contour-provider adapter.
~~~~

### `tests/test_mdo_objectives.py`
*37 lines · stage X07 test-harness*

~~~~text
Mass-objective names must match the hardware subtotal they select.
~~~~

### `tests/test_mdo_of_variable.py`
*260 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_of_variable.py — O/F as a design variable (plan item 1).

SP-125 §2.1 lists mixture ratio among the nine requirement parameters, but its
own derivation makes it an *output*: the optimum balances energy release
against molecular weight and is then moved off that optimum for cooling —
"The temperatures resulting from stoichiometric or near-stoichiometric mixture
ratios ... may impose severe demands on the chamber-wall cooling system.  A
lower temperature, therefore, may be desired and obtained by selecting a
suitable ratio."  That is the trade this optimiser resolves, so O/F belongs on
the design vector.

The property surfaces are what make it real.  With constant gamma/T_c/R the
thermochemistry is FLAT in O/F, so moving O/F would change the propellant mass
split without changing combustion — a wrong model, not a coarse one.  These
tests pin the two-regime behaviour:

* no CEA table  -> O/F is **absent** from the design space, and the solver uses
  the mission's fixed value;
* CEA table     -> O/F is the eleventh variable, bounded by the sampled domain,
  and it is **alive** (non-zero constraint/objective sensitivity).

The synthetic surface below is not a physics claim — it exists only to give the
wiring something with real (Pc, O/F) curvature to differentiate through.  The
production table comes from ``scripts/sample_cea_surface.py``.
~~~~

### `tests/test_mdo_pintle_component.py`
*307 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_pintle_component.py — the pintle as an optimizable MDO component.

Gates (plan §12.1 rule 4): NumPy parity against the host post-mass ledger and
the host Son inverse; AD-vs-central-difference on every owned variable;
architecture applicability (plan rules 4 and 8); live, non-dead design
variables in the NLP; and the post-wall handoff to the injector CAD inputs.
~~~~

### `tests/test_mdo_propellants.py`
*113 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_propellants.py — propellant as a first-class MDO input.

Pins that the MDO is a *design tool*, not a single-engine script: the chamber
gases, L*, densities, coolant properties and the coolant wall limit all follow
the selected propellant combination, and hydrogen correctly carries **no**
coking limit (no carbon) so the gas-side material limit governs instead.
~~~~

### `tests/test_mdo_properties.py`
*206 lines · stage X07 test-harness*

~~~~text
Phase-2 machinery gates: the C¹ monotone interpolant reproduces values and
gradients, preserves shape, matches SciPy's PCHIP, and flags its domain.
~~~~

### `tests/test_mdo_pump.py`
*164 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_pump.py — Phase-6 parity + C¹-efficiency gate for
``raosim.mdo.pump``.

Two things are checked:

1. **Duty/power parity** — Q, head, ω, Ns, Nss and the electrical chain match
   the closed-form ``raosim.pumps`` conventions (``pumps.py`` L2410 Nss form,
   the ω=2πN/60 / Ns=ω√Q/(g0H)^¾ definitions) to ~1e-10.
2. **The new η(Ns) is C¹** where the shipped ``pumps._estimate_pump_efficiency``
   is not: the binned estimator jumps at its Q bin edges, the surrogate is
   smooth (finite ``jax.grad`` everywhere, small adjacent-sample deltas) and
   sits in the SP-125 60–85 % band, peaked at Ns_opt.
~~~~

### `tests/test_mdo_pump_gate.py`
*84 lines · stage X07 test-harness*

~~~~text
tests/test_mdo_pump_gate.py — Phase-6 AD-vs-FD acceptance gate.

Total derivatives of pump/feed outputs (specific speed, efficiency, electric
power, tip speed, battery mass) w.r.t. the differentiable inputs (Δp rise, pump
speed, mass flow) match re-evaluated central differences to ~1e-4 and are
jit-safe — so the block, and in particular the *smooth* η(Ns) that replaced the
binned estimator, drops into the MDF constraint Jacobian (plan §6.4, §12.1).
~~~~

### `tests/test_mdo_resolved_inputs.py`
*328 lines · stage X07 test-harness*

~~~~text
The single host-side input contract (remediation item 9).

These gates encode the invariant the contract exists for: one resolved
definition of every convention, consumed by both pipelines, with no field
reconstructible from a default and no quantity re-derived in a second place.
~~~~

### `tests/test_mdo_roughness_contract.py`
*204 lines · stage X07 test-harness*

~~~~text
D26: hydraulic k_s is a length; legacy mission roughness remains k_s/D_h.

Exercise the actual cooling marches and handoff, rather than only checking
field names. Correlations and their existing transition differences are not
changed by this unit-contract remediation.
~~~~

### `tests/test_mdo_schema.py`
*172 lines · stage X07 test-harness*

~~~~text
Phase-1 gate (plan §11 row 1): schema pytrees + scaling are jit / jacfwd /
jacrev clean with no host callbacks.
~~~~

### `tests/test_mdo_skeleton.py`
*184 lines · stage X07 test-harness*

~~~~text
Walking-skeleton gates (plan §12.5): the coupled residual converges, the
solved state matches the NumPy closed-form oracle (the system is currently
triangular by design), and end-to-end AD through the CONVERGED state matches
re-solved central differences — the first mass–Isp gradient of the MDO layer.
~~~~

### `tests/test_mdo_snapshot.py`
*1261 lines · stage X07 test-harness*

~~~~text
End-to-end tests for the versioned MDO/traditional output bridge.
~~~~

### `tests/test_mdo_state.py`
*250 lines · stage X07 test-harness*

~~~~text
Contract tests for the fixed-shape pure-JAX ``EngineState``.
~~~~

### `tests/test_mdo_structures.py`
*159 lines · stage X07 test-harness*

~~~~text
Differentiable structural-screen parity and constraint wiring.
~~~~

### `tests/test_mdo_variable_policies.py`
*223 lines · stage X07 test-harness*

~~~~text
Pins and requested bounds must govern the evaluated optimization problem.
~~~~

### `tests/test_moc.py`
*393 lines · stage X07 test-harness*

~~~~text
Tests for the MOC unit processes and the forward flow-field entry point.

Validates primitives, the bounded-net march driven by ``solve_flowfield``
(topology gates live in tests/test_moc_bounded_net.py), and the direct
MOC optimizer pipeline.  The deprecated front march ``march_coupled_net``
is pinned only for its documented (defective) behaviour.
~~~~

### `tests/test_moc_bounded_net.py`
*399 lines · stage X07 test-harness*

~~~~text
Bounded characteristic net (raosim.moc_bounded) — topology and oracle gates.

These are the gates from docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md
Phase 0/2.  They would all FAIL on the deprecated front march
(``raosim.moc.march_coupled_net``): its axis/wall points were never
parents, the net shrank by one point per row and died at ~30 % of the
nozzle, and its exit plane was a two-point interpolation.

Oracle B uses the NASA/JHU M3.5 perfect-nozzle run shipped in
``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/outputs_M3.5Perf``:
the wall from ``wall.out`` (throat arc Rd/R* = 1 up to theta_B = 15.2196 deg,
then the BFE contour), the NASA-visible Kliegel–Levine start line, and the
exit-plane table printed at the end of ``summary.out`` (stream thrust
2518.05 lbf, mass flow 35.9981 lbm/s, 0.7537 % initial-vs-exit mass
difference, uniform M = 3.5).  Note ``summary.out``'s "2-D Gross Thrust
1971.99 lbf" is the THROAT-plane figure, not the exit.
~~~~

### `tests/test_moc_diagrams.py`
*98 lines · stage X07 test-harness*

~~~~text
MOC / Rao construction diagrams (raosim.moc_diagrams): the throat kernel
expansion fan, the Rao B-D-E topology, and the BDE characteristic net,
rendered from the in-memory artifacts the ``bde`` wall method stashes on
the solution.  Headless (Agg) smoke coverage.
~~~~

### `tests/test_moc_topology.py`
*152 lines · stage X07 test-harness*

~~~~text
Phase 12.6: full-form RaoTopology (REWRITE_PLAN §11.7).

Builds the explicit topology for the Rao reference design point
(epsilon=10, length_pct=80, Rd=0.382Rt, Ru=1.5Rt) via the now-converging
fixed-end construction (post-Phase-12.4 kernel march) and pins:

* the §11.7 field set, with streamline_BE as THE bell wall;
* seam/closure metrics (B/D/E attachments, mass pair);
* the assembled full wall (throat arc + streamline_BE): monotone in x,
  exit on the commanded station, TOP bell slope profile.
~~~~

### `tests/test_model_registry.py`
*234 lines · stage X07 test-harness*

~~~~text
Provenance registry contracts for physical equations and heuristics.
~~~~

### `tests/test_movable_pintle.py`
*208 lines · stage X07 test-harness*

~~~~text
Son-geometry and static-actuation contracts for movable pintles.
~~~~

### `tests/test_nasa_kernel.py`
*475 lines · stage X07 test-harness*

~~~~text
Phase 12.4 — NASA-style kernel build (CalcRRCsAlongArc port) tests.

The kernel build in :func:`raosim.nasa_moc.build_kernel` is now a port
of the NASA marching architecture (``CalcInitialThroatLine`` +
``CalcRRCsAlongArc``).  For Rd/Rt geometries where the throat plane
contains a substantial subsonic region (e.g. Rd/Rt = 0.382, this
codebase's default), the row march cannot proceed past TT' because
``solve_interior_point``/``solve_wall_point`` require everywhere-
supersonic parents.  The fallback path emits an arc-following BD
curve and is what most current tests exercise.

Tests pin:

* The mass-flow sanity check — wall-side mass on every successfully
  marched RRC equals TT''s wall-side mass to within the configured
  ``mdot_tol``.
* BD endpoints — point B at the wall corner (theta = theta_B, on the
  arc), axis end at r = 0.
* Total RRC mass equals the sum of trapezoidal annular contributions.
* Legacy NASA kernel files parse without error (``TT'BF_Kernel.out``,
  ``BFE_Kernel.out``) — already covered by ``test_legacy_io.py`` but
  re-asserted here for the Phase-12.4 contract.
* End-to-end: switching to the new marching kernel does not regress
  the BVP residual on the reference ε=10/length_pct=80 case.
~~~~

### `tests/test_nasa_kernel_march_parity.py`
*255 lines · stage X07 test-harness*

~~~~text
NASA kernel-march oracle parity (the kernel half of the J6 gate, landed early).

These tests pin the two root-cause fixes that unblocked the RRC march:

1. ``_visible_source_kl_throat`` honours the C++ *integer-division*
   semantics in ``z*(y*y - 5/8)`` (== ``z*y*y``; 5/8 is 0 in C++).  The
   explicit compatibility mode reproduces ``outputs_M3.5Perf`` with the term
   dropped.  This is output compatibility, not proof of the unavailable
   historical executable's provenance.  (The theory-correct
   ``y^2 - 5/8`` lives in ``raosim.transonic_kernel.kliegel_levine``.)
2. ``build_kernel`` evaluates the transonic start line with the
   *upstream* throat radius (C++ ``CalcInitialThroatLine(rUp, ...)``)
   via the new ``Ru=`` parameter, while the marching arc keeps the
   downstream radius.

Regression overlay: the checked-in NASA/JHU M3.5Perf outputs
(``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/outputs_M3.5Perf``):
``TT'.out`` (start line), ``TT'BF_Kernel.out`` (full kernel grid),
``LastKernel.out`` (BD, j=57).

The overlay generator's exact executable/source variant remains unresolved;
these tests verify the explicit source-visible compatibility mode and do not
authorize a ``NASA_REFERENCE_MATCHED`` reliability promotion.

Historical note: before these fixes the march never advanced past TT'
(``rrcs == 1`` for every tested configuration) and silently fell back to
a degenerate arc+sonic-line BD — the root cause of the former Phase 6
``max_scaled ~ 8`` expected failure (see JAX_DIFFERENTIABLE_PLAN.md §10
and tests/test_jax_convergence.py).
~~~~

### `tests/test_nasa_moc_topology.py`
*65 lines · stage X07 test-harness*

(no header comment)

Inferred role: (inferred) tests for raosim.nasa_moc calc_mdot_bd and calc_lrc_de: BD mass flux and closure of the D-to-E left-Mach segments in the NASA-style Rao topology.

### `tests/test_nasa_port.py`
*202 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 12.5 (REWRITE_PLAN §11.6): regression tests against NASA ground truth.

The NASA/JHU ``outputs_M3.5Perf`` grids are the canonical reference
implementation.  ``run_python_port_for_case`` reproduces the perfect-nozzle
workflow end to end with the Python ports — kernel march (KLThroat with
binary-faithful semantics), perfect-branch D/E placement (D = axis end of
BD, uniform-state DE per the C++ FindPointE PERFECT-AXI closed form), and
the BDE-region wall (``calc_bde_region``/``CalcWallContour``).

Measured agreement at first light: wall r(x) RMS = 1.8e-4 (max 4.8e-4)
against ``wall.out`` — already inside the plan's eventual 1e-3 target, so
the gate asserts 1e-3 directly rather than the loose initial 1e-2.
~~~~

### `tests/test_nozzle_geometry.py`
*121 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Tests for raosim.nozzle_geometry

Verifies geometric correctness: continuity, dimensions, monotonicity.
~~~~

### `tests/test_openfoam_export.py`
*295 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Determinism, provenance, geometry, and fail-closed OpenFOAM exports.
~~~~

### `tests/test_openfoam_handoff.py`
*288 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining _digest, _study, _handoff, test_complete_handoff_preserves_full_thickness_and_derives_angle, test_contract_and_nested_inputs_are_immutable_and_fingerprint_is_deterministic, test_mass_flux_failure_is_derived_and_blocks_conversion, test_boundary_and_sheet_kinematic_momentum_closure_are_independent, test_required_evidence_gates_fail_closed
~~~~

### `tests/test_phase6_coupled_wall.py`
*462 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 6 — Coupled wall in the BVP tests.

The wall (x, r, M, theta) is added to the BVP unknown vector via
``RaoSolverConfig.couple_wall=True``.  Four new residual blocks fire:

* ``wall_endpoint``  — wall[0] = N, wall[-1] = E
* ``wall_tangency``  — dr/dx = tan(theta) on each segment
* ``cplus_ce_to_wall`` — axisymmetric C+ compatibility from CE[i] to
  wall[i] (linear pairing)
* ``wall_intersection`` — geometric: wall[i] lies on the C+ line from
  CE[i] (the lightweight ``residual_cplus_child_position`` form)

The tests here pin:

* Default (``couple_wall=False``) is identical to the pre-Phase-6
  behaviour — the new residual blocks are empty arrays.
* When enabled, the BVP solves a longer unknown vector and the wall is
  available on the solution.
* Wall residuals are finite (not NaN).
* Wall endpoint residuals shrink toward zero after a converged solve.
* ``PHYSICS_WEIGHT`` ramp: confirm the ladder ``[0.05, 0.1, 0.25]`` does
  not blow up mass closure beyond 5e-2 (Phase 7 promotion gate).
~~~~

### `tests/test_plotting.py`
*374 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 13 smoke tests (REWRITE_PLAN §12 / test plan row 13).

Headless (Agg) smoke coverage: every plot function must produce a figure
from representative inputs without a solver run.  plot_topology gets a
real Phase-12.6 topology (cheap: one fixed-end construction); the
solution-based plots get a minimal duck-typed solution.
~~~~

### `tests/test_pump_cad_brep.py`
*295 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
True B-rep pump CAD (raosim.pump_cad_brep): named assemblies, re-import
validity, and meanline dimension fidelity.  CadQuery-gated.
~~~~

### `tests/test_pumps.py`
*808 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Pump CAD package gates (docs/PUMP_CAD_IMPLEMENTATION_PLAN.md Phase 0).

Every emitted pump part mesh must pass the repo's own STL gates
(watertight, positive volume), CAD dimensions must be consumed from the
meanline's ``PumpReferenceGeometry`` rather than re-derived, and the
removed faceted pseudo-STEP must be impossible to produce.
~~~~

### `tests/test_rao_chart_benchmark.py`
*423 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 7 — Rao TOP chart benchmark sweep.

Run :func:`raosim.rao_variational.solve_rao_bvp` with
``angle_boundary_mode='free'`` across the published (eps, length_pct)
chart grid (NASA SP-8120 / Rao TOP) and compare the BVP's *reported*
theta_N / theta_E against the chart tables in
:mod:`raosim.nozzle_geometry` (``_THETA_N_TABLE`` / ``_THETA_E_TABLE``).

J5 DE-CIRCULARIZATION (2026-06-12) — what the columns mean now:

* ``solver_theta_n`` is the kernel arc-end angle theta_B the BVP
  closed on (seed secant fixed-end closure) and ``solver_theta_e`` is
  the solved CE exit flow angle.  Both are genuine solver outputs.
  Pre-J5 they were a chart echo (``_design_angles_rad`` lookup —
  err_n was circularly ~0) and the chart-N → exit straight chord
  (with ``evaluate_moc=False`` the export wall *is* that chord;
  pure geometry reproduces the old "solved theta_E" record table to
  ~0.1 deg) — neither contained solver information.
* The chart is Rao's 1960 ARS J. *parabola fit* (gamma=1.23;
  contours gamma-insensitive per Rao 1961 p. 1490).  The exact
  variational solution is not expected to match it to plan-target
  precision: at eps=10/L80 the smooth stationary-DE root sits at
  theta_B = 25.57 deg / theta_E = 11.12 deg vs chart 30 / 15.5 deg
  (test_smooth_existence_root_regression).  Deltas are the
  benchmark's documented FINDING; the right ground-truth question is
  "what does the exact variational solution say across the grid".

Gates, accordingly:

* the live full-grid test asserts sweep completion + physical-band
  sanity and RECORDS the deltas (numeric delta baselines get pinned
  from the de-circularized host record run of
  ``scripts/j5_chart_sweep.py``);
* the historical plan-target gate (RMS < 1.5 / max < 3 deg agreement
  with the chart) is tested as an explicit *negative applicability*
  regression: the exact solution must not be certified against a tolerance
  defined for the different parabola-fit model.

Marked ``@pytest.mark.slow`` -- the full sub-grid runs ~3 min.  CI
runs this nightly / on release candidates.
~~~~

### `tests/test_rao_convergence_domain.py`
*679 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining _context, _resolution, _sample, _domain, _root, test_supported_lookup_requires_all_valid_bracketing_samples, test_invalid_exact_sample_is_refused_with_typed_exception, test_mixed_or_spatially_coarse_cell_is_explicitly_under_resolved
~~~~

### `tests/test_rao_existence_scan.py`
*220 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_kernel_march_controls_are_threaded_to_both_build_paths, test_prandtl_meyer_fan_relation_is_explicit, test_tiny_existence_scan_returns_three_closure_fields, test_position_model_minimizes_over_post_state_grid, test_stationarity_mode_uses_absolute_sigma_e, test_root_refinement_closes_geometry_from_scan_seed, test_existence_scan_cli_runs_without_pythonpath
~~~~

### `tests/test_rao_optimizer.py`
*63 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining _run_opt, test_optimization_converged_and_constraints, test_reproducibility, test_nelder_mead_fallback
~~~~

### `tests/test_rao_residuals.py`
*77 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_residual_Cplus_planar_invariant, test_residual_Cminus_planar_invariant, test_residual_axisym_matches_interior_march, test_left_mach_geometry_zero_on_matching_segment, test_wall_tangency_zero_on_matching_segment
~~~~

### `tests/test_rao_resolution_convergence.py`
*217 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining _solution, test_nonnested_linear_fields_pass_at_common_physical_coordinates, test_common_coordinate_mach_exceedance_is_under_resolved, test_common_coordinate_comparison_fails_closed_without_matching_context, test_common_coordinate_arguments_must_match_solution_provenance, test_common_coordinate_comparison_requires_physical_bde_artifacts, test_common_coordinate_policy_is_strictly_typed, test_rao_moc_heuristic_bundle_is_typed_and_self_classifying
~~~~

### `tests/test_rao_stationarity.py`
*250 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 3 tests: algebraic Rao stationarity, differential consistency, and
left-Mach-line geometry residual.

References:
  - Rao 1958, "Exhaust Nozzle Contour for Optimum Thrust", Jet Propulsion
  - Rao-Beck-Booth 1999, AIAA 99-2584 (propulsion_texts/rao1999.pdf)
  - Östlund 2002, KTH thesis (propulsion_texts/fulltext01.pdf, §3)
~~~~

### `tests/test_rao_valid_region.py`
*432 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Tests for Rao 1999's radial-increment validity gate and diagnostic plots.

Rao-Beck-Booth 1999 (AIAA 99-2584), PDF page 4, obtains Eq. (5) by
eliminating the velocity increment between Eqs. (2) and (3):

    dR/R = {dθ[tan(θ-α)-tan(α)] - dα[tan(θ-α)+tan(α)]}
            / {sin(α)tan(α)sin(θ)/sin(θ+α)}.

The boundary is dR=0 and DE construction requires positive dR.  Eq. (6)'s
``b`` function remains a legacy diagnostic; its sign alone is not the gate.
~~~~

### `tests/test_rao_variational.py`
*174 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Tests for raosim.rao_variational

Validates the Rao calculus-of-variations optimizer:
  - Stationarity conditions
  - Transversality
  - Contour quality
  - API compatibility
~~~~

### `tests/test_rao_variational_moc.py`
*1066 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_bde_physical_gate_explicitly_excludes_raw_state_increments, test_rao_bvp_solution_is_auditable_and_not_hardware_qualified, test_rao_variational_moc_contour_exposes_raw_and_export_layers, test_rao_variational_wrapper_forwards_complete_host_config, test_nozzle_geometry_forwards_host_rao_controls, test_zero_length_throat_arc_collapses_to_one_station, test_moc_wall_throat_bridge_is_tangent_and_exit_preserving, test_nozzle_geometry_dispatches_rao_variational_moc
~~~~

### `tests/test_regen_cad.py`
*138 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Cooling-aware B-rep (raosim.regen_cad): the liner + ribs + jacket as one
STEP solid with REAL channel voids (not a surface mesh).  CadQuery-gated.
~~~~

### `tests/test_regen_cooling.py`
*558 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Regenerative cooling: Sieder-Tate coolant side + coupled 1-D wall
conduction (replaces the old ad-hoc ``h_cool = 1000 + 2e8·area`` film
model).  Pins the Sieder-Tate correlation, the series thermal circuit,
and the physically-correct design trends.

Refs: Sieder & Tate, *Ind. Eng. Chem.* 28, 1936; Huzel & Huang, NASA
SP-125 §4.
~~~~

### `tests/test_regen_geometry.py`
*129 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
3-D regen nozzle geometry (raosim.regen_geometry): wall surface of
revolution + the N cooling channels (axial or helical), STL/PNG export,
and optional wall-temperature colouring tied to the cooling analysis.
~~~~

### `tests/test_regen_profile.py`
*221 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
RegenWallProfile (station-wise wall/channel geometry) + helix-length Δp.

The profile carries the SP-125 four-thickness wall (t_hot, channel w/h,
land, jacket) as per-station arrays; helical channels lengthen the coolant
path, raising the Darcy-Weisbach Δp (SP-125 eq. 4-32) in proportion —
previously the helix changed only the exported STL.
~~~~

### `tests/test_regen_volumes.py`
*105 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Shared regenerative metal-volume kernel contracts.
~~~~

### `tests/test_release_readiness.py`
*165 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining _passing, test_default_engine_release_is_explicitly_blocked, test_component_requirements_do_not_include_other_components, test_complete_traceable_evidence_passes_software_gate_only, test_failed_or_untraceable_evidence_cannot_pass, test_duplicate_records_are_rejected, test_unknown_target_rejected, test_manifest_template_is_versioned_and_blocked
~~~~

### `tests/test_requirements.py`
*471 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
tests/test_requirements.py — Layer 0 gate: requirements in, honest coverage out.

Two things are under test, and the second matters more than the first.

1. **Mapping.** An :class:`~raosim.requirements.EngineRequirement` stated in
   NASA SP-125 §2.1 terms produces the right ``MissionSpec`` and the right
   constraint selection.

2. **Honesty.** A requirement that is only partially screened, or not screened
   at all, is *reported as such* and can never be laundered into a claim that
   the requirement is met.  These are the tests that would fail if someone
   later "simplified" ``requirements_met`` into a plain boolean.

The SP-125 conventions pinned here are quoted in ``raosim/requirements.py``;
the source is ``propulsion_texts/19710019929.pdf`` §2.1, printed p. 31.
~~~~

### `tests/test_run_nozzle.py`
*1203 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Smoke test for the main runner (scripts/run_nozzle.py): the
solve -> contour -> wall -> regen pipeline produces all the expected
artifacts.  Uses ``--max-nfev 0`` (seed contour only) so it runs fast
without the host-scale LM solve.
~~~~

### `tests/test_spray_benchmarks.py`
*262 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Provenance and readiness contracts for the Radhakrishnan spray fixtures.
~~~~

### `tests/test_spray_breakup_evaporation.py`
*464 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_wave_coefficient_variants_are_named_and_not_blended, test_reitz_dimensionless_groups_use_parent_radius_exactly, test_wave_coefficient_variants_produce_distinct_wavelengths, test_vof_calibration_uses_half_full_sheet_thickness_and_round_trips, test_backward_euler_diameter_update_is_exact_discrete_formula, test_backward_euler_relaxation_converges_with_timestep_refinement, test_kh_breakup_updates_multiplicity_and_conserves_mass_and_momentum, test_kh_breakup_is_inactive_below_radius_weber_threshold
~~~~

### `tests/test_spray_carrier_drag.py`
*313 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Deterministic contracts, carrier interpolation, drag, and dispersion tests.
~~~~

### `tests/test_spray_coupling.py`
*182 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_fixed_point_closes_mass_flow_and_efficiency, test_coupling_requires_explicit_mixing_and_combustion_efficiency, test_lagrangian_source_rejects_eta_duck_type_and_failed_handoff, test_coupling_rejects_inapplicable_atomization, test_coupling_does_not_renormalize_over_an_inapplicable_stream, test_coupling_rejects_liquid_stream_below_atomization_regime, test_coupling_enforces_minimum_cycle_efficiency
~~~~

### `tests/test_spray_domain.py`
*72 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_cylinder_contains_and_radius_do_not_extrapolate, test_first_crossing_finds_curved_radial_wall_intersection, test_first_crossing_resolves_wall_before_outlet, test_first_crossing_resolves_outlet_before_later_wall, test_piecewise_wall_knot_is_used_to_bracket_impact, test_invalid_domain_and_outside_segment_start_are_rejected
~~~~

### `tests/test_spray_handoff.py`
*179 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Typed liquid-parcel/gas-carrier cycle-handoff tests.
~~~~

### `tests/test_spray_ledger.py`
*80 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Conservation ledgers distinguish parcel closure from carrier closure.
~~~~

### `tests/test_spray_primary.py`
*467 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Geometry dispatch and conservation tests for primary parcel sources.
~~~~

### `tests/test_spray_solver.py`
*278 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
End-to-end deterministic parcel-march and conservation tests.
~~~~

### `tests/test_spray_statistics.py`
*86 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Deterministic, multiplicity-aware spray statistics.
~~~~

### `tests/test_thermal_design.py`
*180 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Cooling-coupled contour selection (raosim.thermal_design): the cooling
physics is computed BEFORE the contour is fixed and shapes the geometry
(throat curvature) + channels, vs the post-hoc screening in design.py.
~~~~

### `tests/test_trajectory.py`
*46 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
(no header comment)
Inferred role: (inferred) test module defining test_actual_delta_v_is_burnout_velocity_change, test_coast_does_not_get_counted_as_propulsive_loss
~~~~

### `tests/test_transonic_kernel.py`
*253 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Phase 9 — Kliegel-Levine transonic kernel tests.

The KL series is a direct port of NASA/JHU ``MOC_GridCalc_BDE.cpp::
KLThroat`` (lines 3103-3178) with two typo corrections documented in
:mod:`raosim.transonic_kernel`.

These tests pin:

* The bit-comparable wall Mach at the throat plane against
  ``MOC_Grid_BDE/outputs_M3.5Perf/wall.out`` row 0 (M = 1.17779 for
  Rt = Rd = 1 inch, γ = 1.4, axisymmetric).
* The 1-D sonic limit (Rc/Rt → ∞ ⇒ U → 1, V → 0, M → 1, θ → 0).
* Throat-plane monotonicity along the arc from axis to wall.
* End-to-end: KL as the default ``starting_line_method`` does not
  regress the BVP ``max_scaled`` residual (absolute pin; the former
  ``area_ratio`` baseline was a silent flat-line fallback).

The packaged Cuffel-Back-Massier 1969 narrative anchors provide an additional
non-xfailed literature check without pretending that untranscribed plot data
are pointwise digital measurements.
~~~~

### `tests/test_units.py`
*102 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Engineering-notation parsing with a canonical SI store.

The rule under test is that a unit is never guessed. The failure being
prevented is not a parse error but a silent 1000x design error -- the same
class as the pintle STEP metres-as-millimetres defect.
~~~~

### `tests/test_v2_features.py`
*209 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Tests for the new v2 features: wall_pressure, separation, trade_study,
altitude_performance, chamber_geometry.
~~~~

### `tests/test_wall_physics.py`
*149 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
SP-125 longitudinal inelastic buckling (eq 4-28/4-29) + the distinct
external-pressure rib-supported-liner screen, and the wall-arc-length heat
load.

eq 4-29 (``S_c = 4 E_t E_c t / [(√E_t+√E_c)² √(3(1−ν²)) r]``) reduces to the
classic elastic cylinder axial-buckling stress ``E t / (r √(3(1−ν²)))`` when
``E_t = E_c = E``; the tangent moduli supply the inelastic knockdown.  The
rib screen is a SEPARATE plate-buckling check under the ``|Δp|R/t``
compression — not eq 4-29.
~~~~

### `tests/test_wall_profile.py`
*134 lines · stage X07 Test harness and regression arg files (assigned; not in module_stage_map)*

~~~~text
Variable hot-wall + jacket profile sizing (``size_wall_profile``) and
its use in the regen geometry — the manufacturable variable wall the
verdict flagged as missing over a single uniform ``t_hot``.

Grounded in SP-125 (``propulsion_texts/19710019929.pdf``): the inner-shell
eq. 4-31 combined stress sets the hot-wall band station by station, and the
OUTER shell carries the coolant-pressure hoop (jacket sizing).
~~~~


## X08 - Library code not reachable from the CLI (test-only or orphan modules)

*8 files.*

### `raosim/flange_cad.py`
*354 lines · stage OFFLINE (module_stage_map; chamber flange CAD, best-fit S11 CAD, mesh and geometry deliverables / S10 interface)*

~~~~text
flange_cad.py — chamber flange solid, bolt pattern and fastener callout.

The gap this closes
-------------------
Every other part of the bolted chamber/injector joint existed as geometry: the
injector faceplate is exported with its bolt-hole pattern, o-ring groove and
inlet bosses (``injector_cad._build_machined_faceplate``), and the chamber wall
is exported from its meridian (``export.export_step`` / ``regen_cad``).  The
**chamber-side flange was not modelled at all** — no ring, no holes.  The
assembly therefore had a faceplate with twelve bolt holes and nothing to bolt it
to, and ``raosim.mass_ledger.flange_bolt_mass_ledger`` was pricing a part that
no exporter wrote.

This module builds that ring, and emits an *orderable* fastener callout rather
than only a diameter, because "correctly sized bolts" means a thread
designation, a property class, a grip length and a torque — not a number in
metres.

Preload and torque
------------------
The recommended preload follows standard bolted-joint practice: a target preload
of ``k_preload`` times the proof load, with ``k_preload = 0.75`` for
reusable/non-permanent joints, and the short-form torque relation

    T = K * F_i * d

with a nut factor ``K`` (0.20 dry steel, 0.15 lubricated) — Shigley,
*Mechanical Engineering Design*.  These are machine-design conventions, not
propulsion-corpus values, so the callout is labelled ``screening_sized`` and
carries its assumptions explicitly.  ``K`` in particular is empirical and
scatters by 20-30 % in practice; a flight joint needs measured torque-tension
data or angle/stretch control, which this does not replace.

The flange itself is **not** a qualified joint design.  It has no gasket
seating-stress calculation, no flange rotation, no preload scatter, no thermal
gradient and no fatigue analysis.  It is the geometry that matches the sizing
screens already implemented in :mod:`raosim.interface`.
~~~~

### `raosim/jax/bvp.py`
*102 lines · stage OFFLINE offline/not wired into current CLI (stage per module_stage_map)*
~~~~text
raosim.jax.bvp — Optimistix Levenberg-Marquardt least-squares solve (J3).

Replaces ``scipy.optimize.least_squares`` (rao_variational.py:2965/3559) for the
JAX backend.  The LM step uses the *exact* autodiff Jacobian (Optimistix calls
``jax.jacfwd`` internally), and the converged solution ``u*(params)`` is
differentiable via the implicit function theorem (Optimistix's default) — no
backprop through the iterations.  That implicit diff is what powers the J6
gradient API (sensitivity fields).

This module is solver-agnostic: callers pass a residual ``fn(u, args) -> array``.
J3b assembles the full grouped coupled-wall residual on top; J4 then checks the
``max_scaled < 2e-3`` Phase-6 gate on the ε=10/L80/γ=1.4 reference case.

Inner secant loops (θ_B, point D) will likewise be wrapped as
``optimistix.root_find`` so their gradients also come from the implicit function
theorem (REWRITE_PLAN.md §2.F / JAX_DIFFERENTIABLE_PLAN.md §4.5).
~~~~

### `raosim/jax/design_opt.py`
*460 lines · stage OFFLINE offline/not wired into current CLI (stage per module_stage_map)*
~~~~text
raosim.jax.design_opt — constrained differentiable nozzle design.

Maximize the ambient thrust coefficient over a small contour design
vector subject to the physics constraints, with EXACT JAX gradients
(Optimistix BFGS on a quadratic-penalty objective):

    maximize   Cf(ε; Pc, Pa)
    over       ε (expansion ratio), Rd_factor (throat curvature)
    s.t.       separation_margin(ε) ≥ sep_margin_min        (Schmucker)
               T_wg,throat(Rd_factor) ≤ T_wall_limit        (Bartz + Sieder-Tate)

This is the rigorous form of the coupling the screening workflow
lacks: the physics SHAPES the geometry (ε set by the thrust/separation
trade, Rd_factor by cooling) instead of being checked after a fixed
Bézier.  It is the differentiable successor to the NumPy feedback loop
in :mod:`raosim.thermal_design`.

The gas properties (cp, Pr, μ) and the coolant-side h_c are solve
constants (independent of ε / Rd_factor) precomputed on the host; only
the inner objective is differentiated.  Design variables are kept in
their boxes by a sigmoid reparametrisation, and the penalty weight is
ramped for accuracy — the reported margins are the unweighted physics.
~~~~

### `raosim/jax/pack.py`
*62 lines · stage OFFLINE offline/not wired into current CLI (stage per module_stage_map)*
~~~~text
raosim.jax.pack — flat unknown-vector <-> structured CE state (J3).

Mirrors the role of ``_pack_bvp`` / ``_unpack_bvp`` (rao_variational.py:1787/1844)
for the JAX backend.  Packing is a pure, static-shape transform so ``jit`` and
``jacrev`` apply.

The CE unknown layout (this phase) is::

    u = [ x(n) | r(n) | M(n) | theta(n) | log_C(1) ]

Wall + characteristic-net unknowns (the full Phase-6 coupled vector,
REWRITE_PLAN.md §2.F) extend this in J3b once the MOC march is ported; the
layout is deliberately block-contiguous so appending wall blocks is additive.
~~~~

### `raosim/jax/reference.py`
*38 lines · stage OFFLINE offline/not wired into current CLI (stage per module_stage_map)*
~~~~text
raosim.jax.reference — canonical M3.5Perf reference constants.

Per JAX_DIFFERENTIABLE_PLAN.md decision (4): static array shapes for the JAX
march are sized from the NASA/JHU ``outputs_M3.5Perf`` reference case rather than
made configurable.  Values are taken from
``docs/rice_jhu_moc_topology_map.md`` (parsed from ``summary.out``).

The march runs as a ``lax.scan`` over a padded grid with a validity mask;
``MAX_KERNEL_ROWS`` must comfortably exceed the reference ``lastKernelJ`` so no
real row is ever truncated.  Padded (masked) rows contribute zero residual.
~~~~

### `raosim/jax/residuals.py`
*261 lines · stage OFFLINE offline/not wired into current CLI (stage per module_stage_map)*
~~~~text
raosim.jax.residuals — vectorized Rao/MOC residual leaves (J2).

Faithful JAX ports of the pure, node-wise residual functions that the BVP
differentiates through:

  - ``raosim.rao_residuals``: C+/C- axisymmetric compatibility, left-Mach
    geometry, wall tangency, C+ child position.
  - ``raosim.rao_variational``: the CE-array blocks
    ``_ce_axisymmetric_compatibility_residual_groups``,
    ``_rao_algebraic_stationarity_residuals``,
    ``_rao_left_mach_geometry_residuals``, ``_ce_smoothness_regularization``,
    and the per-node/per-segment algebraic + differential Rao stationarity.

These are vmap'd over node pairs instead of Python ``for`` loops, but compute
exactly the same numbers (J2 gate: 1e-10 parity on a fixed CE state — see
``tests/test_jax_residual_parity.py``).  Scaling (``/radians(1°)``, ``/ds``)
matches the NumPy callers so the assembled residual magnitudes — and the J4
``max_scaled < 2e-3`` gate — are directly comparable.

The full grouped/weighted assembly (``_rao_bvp_residual_groups``) plus the
mass/length/stationarity integrals land in **J3** once the CE/kernel geometry is
ported; those blocks are coupled to the marching construction, not pure leaves.

State convention: a CE/wall polyline is passed as four 1-D arrays
``(x, r, M, theta)``.  Mach is clamped to >= 1.001 to mirror
``_control_surface_flow_nodes`` (which builds ``FlowNode(M=max(M, 1.001))``).
~~~~

### `raosim/thermal_correlations.py`
*100 lines · stage OFFLINE (module_stage_map: not on the CLI data flow; best-fit stage S07 Regenerative wall sizing, cooling analysis and wall structural screens)*
~~~~text
Source-scoped thermal correlation choices and applicability reports.

The numeric bounds below are Bartz's preliminary-estimation guidance, not a
physical validation envelope. Original PDF identities and pages are recorded in
knowledge/reports/THERMAL_VERIFICATION_IMPLEMENTATION_2026-10-17.md.
~~~~

### `raosim/trajectory.py`
*111 lines · stage OFFLINE (module_stage_map: not on the CLI data flow; optional vertical-ascent trajectory utility, no better-fitting stage)*
~~~~text
trajectory.py – Optional vertical-ascent trajectory integrator.

Simple 1-D point-mass simulation with gravity, drag, and atmospheric density.
Kept as a separate module so the nozzle design tools can be used independently.
~~~~

