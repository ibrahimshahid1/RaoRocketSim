# Physical-integrity remediation record — 2026-07-11

This record maps the findings from the 2026-07-02 audit and the subsequent
full-repository review to their current implementation, including both closed
software defects and deliberately unresolved blockers. It records code
consistency, physical applicability, CAD topology, and provenance. It is
**not** a claim of CFD validation, production drawing release, test readiness,
or hardware qualification.

The governing distinction is:

- an internal design gate checks whether the implemented preliminary model was
  applied consistently and inside its declared domain;
- a CAD gate checks units, topology, interfaces, flow connectivity, nominal
  clearance, interference, and export/re-import behavior;
- a release-evidence gate checks whether independent analyses, drawings,
  inspections, and tests for the exact configuration were supplied;
- `hardware_qualified` remains `false` regardless of the first three results.

The machine-readable model inventory is
[`raosim/model_registry.py`](../raosim/model_registry.py), summarized in
[`MODEL_REGISTRY.md`](MODEL_REGISTRY.md). External evidence is governed by
[`raosim/release_readiness.py`](../raosim/release_readiness.py) and
[`PHYSICAL_RELEASE_GATES.md`](PHYSICAL_RELEASE_GATES.md).

## Remediation map

| Area | Audit issue | Current behavior | Remaining boundary |
|---|---|---|---|
| Trusted design path | Research BVP behavior could be mistaken for the normal design path. | The deterministic Rao/TOP Bézier is the CLI default and trusted preliminary baseline. `--contour-method rao-bvp` is an explicit research opt-in; failed BVP geometry/reliability gates block export. | The Bézier remains a preliminary constant-$\gamma$ design method, not a CFD-validated contour. |
| Thermochemistry and thrust closure | Multiple property/efficiency interpretations could drive inconsistent mass flow. | The constant-$\gamma$ path retains one authoritative combustion-product state. A separate thermally-perfect, fixed-composition quasi-1D path now integrates bounded $c_p(T)$ data, closes sonic/energy/entropy/area/mass residuals, drives target-thrust or matched-pressure sizing, and supplies wall-pressure/separation stations. Its schema binds freeze basis, chamber state, O/F, generator/database, and source hashes. CEA equilibrium remains rejected, and Hall $C_d$, Bartz, boundary-layer, MOC, and Rao authority cannot be inherited by substituting a local $\gamma$. | The new path is Bézier-only and software-verified against manufactured/constant-$c_p$ oracles. Pinned external property/performance benchmarks, property-grid refinement, variable-property thermal/viscous adapters, equilibrium/finite-rate chemistry, and physical validation remain gates. |
| Throat and chamber | The SP-8120 upstream-radius citation, chamber length meaning, and fallback assumptions were ambiguous. | The literature-backed $0.6\le R_u/R_t\le1.5$ range is distinct from the explicit repository diagnostic extension. Chamber volume closes to $L^*A_t$, and reported length is the true injector-plane-to-throat distance. Validated mode requires explicit chamber inputs instead of treating a geometric fallback as combustion evidence. | $L^*$ is still a residence-volume proxy and must be validated for the propellant, injector, stability, cooling, and duty cycle. |
| Direct MOC optimizer | The optimizer evaluated a spline but exported a reconstructed quadratic Bézier. | The exported divergent wall is sampled from the optimized monotone cubic-Hermite spline. The tangent-intersection Bézier point is metadata only. $C_F$ uses throat-area normalization. | The optimizer remains an experimental inviscid axisymmetric path. |
| Rao BVP control surface | Only the partial $D$-$E$ surface was available to the public thrust gate, and the point-$D$ seam was weakly diagnosed. | The solver reconstructs $C$-$D$ from the kernel, joins $D$-$E$, directly integrates full $C$-$D$-$E$ thrust, and reports point-$D$ projection/Mach/angle continuity plus full-surface mass closure. Partial-$D$-$E$ results are diagnostic only. | Arbitrary designs still require independent flowfield validation. |
| Rao literature/reference evidence | Literature cases were expected failures and the NASA source/fixture roles were conflated. | Rao 1958 Nozzle B uses digitized Table 2 geometry and Table 3 $C_F$ as a strict benchmark. The scarfed 1990 case is explicitly unsupported by an axisymmetric solver. Cuffel–Back–Massier anchors are packaged and source-labelled. No literature/reference test uses `pytest.xfail`. | The visible NASA/JHU source is SHA-pinned and software-verified only for the M3.5 reference workflow. General `CropNozzleToLength` export is unported, and the historical `TT'.out` generator provenance remains unresolved and non-authoritative. |
| Injector hydraulics | Slot and round-hole geometry, target TMR, manifold allowance, and throttle schedules could disagree with the reported hydraulics or CAD; no physical movable-pintle branch existed. | Each fixed-discrete stream uses its actual metering geometry. The separate `son_continuous_movable` branch implements Son Eq. 1 $A_{tip}$, $A_{cg}=\pi(D_{cg}^2-D_{pr}^2)/4$, the tip-controlled open-stop boundary, implicit $C_d(L/L_{max})$ opening solve, and static position/leakage/actuator/stem ledgers. Its throttle map holds all hardware and the axial annulus fixed, meters only the radial stream with center-rod travel, and solves/reports the required separate upstream axial controller. | Movable $C_d$ must come from a configuration-specific artifact whose Son-geometry fingerprint, fluid, Re, $\Delta P$, temperature, and cavitation domain match the solve. Position metrology, seat/leakage bounds, and actuator/material inputs require separate source/hash-bound evidence. Maldistribution, transient control/feed dynamics, and all hardware behavior still require cold-flow/hot-fire evidence. |
| Spray and $c^*$ | A correlation-based vaporization estimate was treated as if it were total combustion efficiency, including outside liquid-droplet applicability. | Phase/pressure gates reject gas, two-phase, transcritical, and otherwise inapplicable droplet claims. Vaporization, mixing, chemical-completion, and total $\eta_{c^*}$ are separate. The legacy opt-in fixed point is explicitly labeled. Its direct-fuel regenerative outer loop now re-solves coolant flow, wall state, jacket loss, injector feed, final structure, and feed/pump duty at the current mass flow; independent coolant/bypass topology remains blocked. A deterministic one-way Lagrangian mid-tier now provides geometry dispatch, prescribed carrier fields, WAVE/KH-RT, drag/dispersion, Eq.16 evaporation, trajectories, SMD/RR statistics, conservation ledgers, benchmark fixtures, and a typed fail-closed handoff. A deterministic OpenFOAM Foundation v13/20260624 water-only VOF wedge exporter and a separately gated VOF-to-parcel evidence contract are now present. | The first VOF mesh is an external-gap screen: it prescribes the mechanical opening and does not resolve the internal center-gap turn/post/tip geometry. OpenFOAM has not been executed here, and no convergence, real-fluid/energy/two-way carrier closure, reacting spray, chamber/nozzle CFD, or physical validation is claimed. Current annulus/slot/hole injector geometry still lacks a validated primary model. |
| Regenerative cooling | Rectangular channels used a circular-pipe laminar Nusselt proxy. | Actual rectangular-channel calls use the Shah–London all-walls-uniform-heat-flux polynomial as a function of aspect ratio; Sieder–Tate remains the turbulent branch. | Entrance/developing flow, unequal wall heating, curvature credit, maldistribution, CHT, and temperature-dependent structural response need higher-fidelity analysis and test calibration. |
| CAD units and exports | SI model dimensions, millimetre CAD-kernel inputs, and neutral-file scale could be conflated; requested CAD could degrade silently. | Public sizing remains SI. CAD-kernel conversion is explicit, neutral artifacts carry unit metadata/sidecars, and STEP outputs are re-imported and checked for scale, validity, expected solid count, and volume. Requested B-rep is a hard default; `--no-require-brep` permits only a labelled wall diagnostic fallback. | Neutral STEP has no native Inventor feature history and is not a released drawing. |
| Injector/chamber CAD | Hydraulic passages and mechanical interfaces were not demonstrably one consistent, assemblable model. | The fixed-discrete coaxial five-part injector includes metering passages, manifolds, retention/flange patterns, spigot/socket/thread envelopes, O-ring glands, nominal clearances, circuit-connectivity checks, pairwise interference checks, and STEP round-trip gates. Chamber interfaces share resolved bolt/seal geometry. The Son movable branch exports only JSON/CSV/SVG/PNG evidence reports; every CAD request fails closed instead of reusing fixed geometry. | The movable architecture still needs a swept assembly with closed/open stops, running clearances, seals/guides, collision and tolerance-stack checks. For both architectures, selected standards, surface finish, gland squeeze, preload, fits, assembly access, cleaning, proof/leak tests, and cold/hot-flow results remain external. |
| Regen CAD | A sealed reference solid could be confused with a cold-flow article. | Full-channel B-rep topology is checked. Cold-flow release requires manifolds, ports, and a connected extracted coolant volume; sealed ends remain reference-only. | Channel roughness, closeout process, NDE, hydroproof, flow distribution, and cyclic life require manufacturing and test evidence. |
| Pump CAD | Pump STEP could be faceted/open, the full-disk meanline ignored the shaft/hub annulus and blade blockage, beta1 disagreed with CAD, blade-root stress was report-only, and the hollow volute was one trapped part. | The pump now converges shaft torsion/diameter, fit/root-wall hubs, the net annular eye, inlet/exit free area, achieved phi1/phi2, velocity-triangle beta1, four full inlet blades plus downstream splitters, pressure-sized casing wall, and pump power. The blade-root stress requirement feeds its thickness back through exit blockage and RPM/diameter; an impossible fixed-RPM or maximum-diameter closure returns an explicit infeasible gate. CAD consumes those values without hub mutation. STEP exports separate rear-body/front-cover volute halves with an exposed scroll centerplane, keyhole gasket land, body/outlet-neck bolt holes, dowels, pressure-clamp/tool-access gates, interference/clearance checks, and re-import. Legacy STL is schematic-only. | This closes software geometry consistency and a bounded machining/assembly topology, not hardware qualification. Selected gasket/bolts/threads/dowels, flange FEA, shaft retention, bearings/seals/lubrication, tolerances/thermal growth, rotordynamics, cavitation, proof/cold-flow tests, and measured head/efficiency/NPSH maps remain required; `cold_flow_release_ready` stays false. |
| Separation and trajectory | Separation state names and burn/coast bookkeeping admitted ambiguous interpretations. | Separation outputs distinguish attached flow from predicted separation. The trajectory integrator preserves burn/coast state and final propellant accounting consistently. | Side loads, shock/boundary-layer interaction, guidance, winds, staging, slosh, and 6-DOF dynamics are outside scope. |
| Release claims | Internal numerical success could be read as physical release authority. | Engine, injector, regen, and pump evidence manifests are configuration-bound and require result, artifact reference, SHA-256 where locally verifiable, reviewer, and review date. Missing evidence is a hard blocker when requested. `evidence_complete` is separate from, and never changes, `hardware_qualified=false`. | Only the responsible engineering and test authority can approve a configuration for machining, pressure test, cold flow, hot fire, or flight. |

## Verification coverage

Automated tests now cover the failure modes above, including:

- trusted-default/explicit-BVP selection, chart-domain and throat-radius
  applicability, target-thrust closure, chamber seam/volume/length, and CEA
  mode rejection;
- full $C$-$D$-$E$ thrust, point-$D$ continuity, NASA kernel/source parity,
  Rao 1958 geometry/$C_F$, explicit scarfed-case rejection, and direct-MOC
  spline export;
- injector phase and geometry applicability, exact round-hole propagation,
  TMR pressure-drop closure, Son movable control-area transition benchmarks,
  fixed-hardware travel/controller throttle semantics, strict calibration and
  sheet-evidence domains, actuator/leakage gates, movable CAD fail-closed
  behavior, spray fixed-point convergence/failure behavior, and feed-pressure
  bookkeeping;
- Shah–London rectangular laminar trends and thermal/hydraulic gate behavior;
- SI-to-mm CAD boundaries, STEP re-import/scale/solid counts, injector circuit
  separation and clearances, regen connectivity, pump fluid connectivity and
  meanline-fidelity status, and assembly interfaces;
- model-registry completeness and configuration-bound release-evidence schema,
  digest, review, and failure behavior.

The repository CI separates the normal suite from marked slow literature/JAX
solves. A passing software suite verifies implementation behavior only; it
does not satisfy any external evidence item below.

## External blockers that software cannot close

No CAD artifact generated by this repository is ready for machining, cold
flow, hot fire, or flight until the exact configuration has, as applicable:

1. independently reviewed Euler/RANS and reacting-flow CFD, conjugate heat
   transfer, structural/joint/fatigue/buckling FEA, and rotordynamics;
2. process-, temperature-, direction-, and lot-appropriate material allowables;
3. released drawings, GD&T, fits, surface finish, selected seals, threads,
   fasteners/preload, BOM, and manufacturing/assembly route;
4. manufacturability and inspection review, cleaning compatibility, NDE, and
   configuration-controlled as-built measurements;
5. proof-pressure and leak tests, injector and regen cold-flow distribution,
   pump performance/NPSH/cavitation maps, and optical spray data;
6. ignition/transient and combustion-stability assessment plus hot-fire
   pressure, $c^*$, heat-flux, durability, and shutdown evidence.

These requirements are represented by explicit evidence IDs in
[`PHYSICAL_RELEASE_GATES.md`](PHYSICAL_RELEASE_GATES.md). They are intentionally
not replaceable by `--allow-unconverged`, a valid STEP, or a passing unit test.

## Remaining research and implementation work

The following remain explicit work rather than hidden capability claims:

- generalizing the pinned NASA/JHU M3.5 source-port workflow while preserving
  the unresolved historical-fixture provenance boundary;
- pinned external validation and property-grid refinement for the implemented
  frozen-composition variable-$c_p$ quasi-1D flow, plus variable-property
  thermal/viscous coupling and any future equilibrium/characteristic model;
- a validated Eulerian liquid-sheet primary-breakup handoff and reacting-spray
  CFD (the deterministic one-way parcel mid-tier is implemented);
- swept movable-pintle CAD with stops, running clearances, seals/guides,
  collision/tolerance checks, and independent manufacturability review;
- total differentiability from design inputs through final manufacturable wall
  coordinates;
- selected/qualified pump joint hardware, shaft retention, bearings/seals,
  rotordynamics, cavitation evidence, and measured maps beyond the implemented
  coupled meanline/free-area/split-casing solve.

Until those capabilities are independently validated, the deterministic
Rao/TOP Bézier and the low-order component models remain preliminary design
tools with the validity and release gates described above.
