# Decision report: is RaoRocketSim's CAD arbitrary, and should the owner hand-CAD templates?

Scope: read-only review of the repo and the local corpus. Claims are cited as path:line. Two other labels are used:
- **[BG]** means background knowledge that was not verified in the corpus.
- **Derived** means I worked the result out algebraically from the cited code rather than measuring it on a run.

## 1. Bottom line

**The geometry is not arbitrary wherever a solver acts on it.** These features trace to SP-125 (Huzel & Huang), SP-8109, SP-8052 and Nardi 2015:
- the Rao contour
- liner, channel and jacket thickness
- the pintle metering features
- the impeller and inducer flow path

**The "hardware shell" is heuristic or arbitrary:** flanges, bolts, seal glands, ports, manifolds, the pump stator and mechanical stack, and the assembly. Instrumentation, fittings, alignment features, mounts, valves and lines are **absent**.

Arbitrariness is not the worst problem. The current gates pass **functional defects**. Each is confirmed from code; the full evidence is in §2:

1. **Injector exits are half blocked.** The pintle body's spigot fills z ∈ [z_holes − tip_engage, z_holes] over r ∈ [Ri, Ri + 0.55(Rp − Ri)] (`raosim/injector_coaxial_cad.py:386-391`, `:155`). The exits are centred at z_holes and cut through the tip only (`:416-458`). So the lower half of every exit is walled off from the bore, in both slot and hole styles.
   - Derived: the built part has exactly 50 % of the designed exit area.
   - At fixed ṁ, the inner-propellant Δp rises about 4×, so the momentum ratio and the spray model no longer describe the built part.
2. **The pump scroll intrudes inside the impeller tip.** Its section radius is a_exit·√(θ/2π), centred on casing_r = 0.68·D2 (`raosim/pump_cad_brep.py:465-489`, `raosim/pumps.py:2259`). Nothing constrains casing_r − a ≥ r2.
   - Derived from the area floors at `pumps.py:2579-2581` (A_exit ≥ 1.8·0.35·2π·r2·b2): intrusion is **guaranteed whenever b2/D2 > ~0.051**, which is the normal range. It is not an edge case.
3. **The diffuser vanes do not span the diffuser.** chord = Δr / max(sin α, 0.25) (`pump_cad_brep.py:427`), so the radial span is 4·sin α of the annulus whenever α < 14.5°.
   - Derived: 35 % at the 5° floor (`:423`) and 42 % at 6°.
4. **The chamber flange bore is sized from the liner OD, not the jacketed OD.**
   - After regen sizing, `args.wall_thickness = t_hot` (`raosim/run_nozzle.py:5273`).
   - That value is passed to the interface (`:6377`).
   - The interface computes `chamber_od = 2r + 2·wall` (`raosim/interface.py:479`).
5. **The regen chamber body (`regen.step`) has no flange, and the only flanged body is the unanalysed uniform-wall `wall.step`.** `regen_cad.py` contains no flange or bolt code, and `export.py:326-343` is the only flange path. This ties directly to RW-S11-01: the analysed wall is not the exported wall.

A hand-made template would have caught none of these. Cross-feature invariants and functional checks on the built solids will catch all five.

**The owner's instinct is right about what is missing:** a designer's intent for the mechanical hardware. **The literal mechanism is wrong.** A GUI CAD template that the software resizes fails for four reasons:
- it breaks on parts whose topology changes with sizing;
- STEP cannot carry the driving parameters;
- geometric scaling breaks discrete standards and pressure-driven thicknesses;
- it creates a second source of truth for a solo developer.

**Verdict: adopt-hybrid.**
- **What the owner authors:** a **design-intent sheet**, meaning a dimensioned sketch plus a dimension table. Every dimension is labelled as a standard, as `= k·driver` with the reasoning, or as a manufacturing value.
- **Which parts get one:** only the heuristic-dominated, fixed-topology families. These are joints and flanges, glands, ports and bosses, manifold heads, the injector mechanical stack, the pump casing joint, the bearing cartridge and shaft stack, mounts, and the test-stand adapter.
- **What the agent does:** encodes each sheet as CadQuery code with a provenance record for every **load-bearing** dimension.
- **Role of the owner's own CAD:** if any is produced, it is a visual reference only, not the runtime generator.
- **Bought parts:** selected from vendor STEP files, never generated.

**Order of work:**
- **P0 (~3 weeks agent time):** add layout invariants and solid-based functional gates, fix the five defects, and add a scoped provenance ledger.
- **P1:** a standards and feature library, an engine datum skeleton, instrumentation features, a regeneration-envelope sweep, and a cheap 3D-printed form, fit and flow article.
- **P2:** the owner's design-intent families, catalog parts, and literature golden cases.

## 2. Per-component verdict table

Share key: S = solved, R = rule (literature), St = standard, M = manufacturing limit, H = heuristic, A = arbitrary.

| Component | How geometry is produced today | Share (qualitative) | Most consequential arbitrary or defective dimensions | Resembles usable hardware? | Already accounted for in repo plans? |
|---|---|---|---|---|---|
| Chamber and nozzle contour | Lc is root-solved to L*·At (`chamber_geometry.py`); Rao TOP/Bézier contour (`nozzle_geometry.py`); throat arcs 1.5Rt and 0.382Rt (`raosim/throat_geometry.py:22-23`); revolved in `export.py` | Mostly S/R | Rs fill fraction 0.8 has no anchor (`docs/subsystems/S05-nozzle-and-chamber-geometry/shoulder_radius_design_basis.md`). Defaults are L* 1.0 m, CR 2.5 and α 45° (`raosim/run_nozzle.py:2913-2937`) | **Y**, as a contour | Yes. RW-S05-07 (`docs/program/REMAINING_WORK_AND_GOALS.md:157`) proposes refusing these placeholders for hardware jobs |
| Regen liner, channels and jacket | SP-125 Eq 4-31 and jacket hoop sizing in `thermal_design.py`; normal-offset lofts in `regen_cad.py` | S for thickness; H for search-grid bounds and floors | The 0.5 mm floors are flagged "preliminary placeholder" (`docs/subsystems/S07-.../regen_wall_model.md:242`). Channel corners are sharp, with no cutter radius. A 50 µm bond overlap is baked into the solid (`regen_cad.py:446, :571`). The Wadel limits (`manufacturing_limits.py:86ff`) are checked only in MDO (`mdo/job.py:1175-1185`), never in the CAD exporters | **Partly.** The channel block is plausible. The default `release_mode="reference"` seals both ends by design (`end_seal_fraction`, `regen_cad.py:570`); the `cold_flow` mode refuses to run without manifolds (`:596-598`) | Partly. The floors are flagged and RW-S11-01 records that the analysed wall is not the exported wall. The missing manufacturing gate on the CAD side is not recorded |
| Regen manifolds and coolant ports | The plenum is the channel zone with ribs removed over 0.06·L. Ports are bare cylinders (`regen_cad.py:575-576, 691-701`) | H/A | 4 ports per manifold. Ports end at floor + 0.35·h (`:701`). No boss, fitting or thread. No SP-8087 manifold sizing | **N.** It cannot be plumbed | No |
| Chamber–injector bolted joint | `resolve_bolted_interface_geometry` (`interface.py:409`) drives both sides (`run_nozzle.py:6373-6395`). The flange exists only on the uniform-wall `wall.step` (`export.py:326-343`). The ISO sizer `size_bolted_interface` (`interface.py:765`) is advisory | The arithmetic is consistent; the dimensions are H | Bolt hole = max(6 mm, 0.06·D) (`interface.py:43, :506`). BCD = OD + 6·hole (`:786-787`). By the repo's own account these make the flange and faceplate about ¾ of modelled hardware mass (`design.py:2336-2348`, `applied_to_exported_geometry=False` at `:2373`). The bore comes from the liner OD (defect 4). `regen_cad.py` has no flange | **N / partly.** The mating relation is real, but the parts are oversized and the analysed regen body has nothing to bolt to | Sizer wiring: RW-S10-06 (`REMAINING_WORK_AND_GOALS.md:233`) and RW-X08-01 (`:345`). **Symptoms recorded:** RW-S11-03 (`:241`, wall flange and jacket collide, 3424 mm³) and RW-S10-02 (`:229`, "two inner-edge definitions"). **Root cause not recorded:** t_hot is used as the wall at `run_nozzle.py:5273`, giving liner OD at `interface.py:479` |
| Injector metering core (coaxial pintle) | `resolve_coaxial_layout`: gap, land L/Dh, radial exits, and skip distance = Dp (Nardi 2015, `.../pintle_injector/nardi2015.md:214`) | S/R | 24 slots and L/Dh = 2 are defaults. The skip distance ignores `impingement_distance`. **Defect 1:** half of each exit is blocked. The cap above the exit row is a fixed 0.5 mm (`injector_coaxial_cad.py:433`); `hole_d` is case-specific | **N as built.** It delivers 50 % of the designed area | No. ROADMAP R4 marks the injector "✅ machined STEP" (`docs/program/ROADMAP_REQUIREMENTS_TO_PARTS.md:220`) |
| Injector mechanical envelope | Hard-coded mm floors and multiples of min_tool (`injector_coaxial_cad.py:90-136`) | Mostly H; M only where the min_tool branch of a `max()` is active | **Glands:** at the default min_tool of 0.3 mm (`injector_cad.py:359`), the literal floors bind, so every gland is 1.5 × 0.75 mm (`:530-531`). That suits a ~1 mm-section O-ring but is not keyed to a named one (see §3). **Retention:** 4 bolts of Ø2 mm (`injector_coaxial_cad.py:131-134`). **Inlets:** bare bores marked `"layout_only"` (`injector.py:193, 197`). **Clearances:** 0.1 mm pilot running clearance (`injector_coaxial_cad.py:93`). **Nose:** a hemisphere that ignores `deflector_angle` (0 hits in the coaxial module), whereas the legacy builder uses a cone (`injector_cad.py:120-134`) | **Partly.** The architecture is right (Nardi 5-layer stack, `nardi2015.md:398-422`); the details are wrong | Partly. `injector_pintle_provenance.md:129-132` lists fitting standards, O-ring crush, preload, GD&T and cold-flow as still required |
| Pump impeller and inducer | `pumps.py` meanline feeds `pump_cad_brep.py`; SP-8109 blade count; SP-8052 inducer | Mostly S/R | β2 = 25° is hard-coded with no spec override (`pumps.py:1264, :1296`), though it sits inside the conventional band [BG]. The backplate equals the blade thickness. No fillets, although SP-8109 calls for them (`19740020848.md:5463ff`). Blades are planar ribbons | **Partly.** The hydraulics are plausible; the rotor is not manufacturable | Yes. RW-S09-08 (`:218`) and the PUMP_GEOMETRY plan (`docs/program/PUMP_GEOMETRY_IMPLEMENTATION_PLAN_2026-07-13.md` §1) |
| Pump stator (diffuser, volute, casing) | Heuristic ratios on D2 and area floors | Mostly H/A; several dimensions contradict SP-8109 | **Defects 2 and 3.** `casing_r = 0.68·D2` appears four times (`pumps.py:1828, 2259, 2645, 2750`). The throat floor is 0.35·A_imp,exit and the exit area is 1.8·throat (`:2579-2581`). Vane count is 8 (`:167`), but SP-8109 says the prime nearest Z2 (`19740020848.md:5805-5807`). Width is 1.15·b2 (`pumps.py:2580`), but SP-8109 says b3 ≈ b2 (`:3933`). The pocket is cylindrical, but the casing should follow the impeller contour (`:3620`). There is no tongue and no exit cone (`:4012ff`). The A(θ) law is honestly labelled as constant-mean-velocity, not SP-8109 (`pump_cad_brep.py:455-457`) | **N** | Partly. C6, the tongue, the exit cone and the vane shape are planned (`PUMP_GEOMETRY plan :117-121, :204`), and RW-S09-08 covers the D1-D4 collector items. **Not recorded:** scroll intrusion, vane-span clamp, pocket shape |
| Pump mechanical and electrical stack | Placeholders | A, honestly labelled | The shaft is a bare Ø6 mm floor (`pumps.py:192`). The span is 1.8·D2 (`:2267`). The motor is a cylinder whose size comes from mass ÷ 2700 kg/m³, with L = 1.2·D (`pump_cad.py:28, :455`). No bearings or seals | **N** | Yes. RW-S09-09 (`:219`) covers bearings and seals as mass placeholders, with no retention or rotordynamics |
| Pump split-casing joint | `_split_casing_joint_layout` (`pumps.py:1834`) | St bolts; H for land, thickness and dowels | Flange thickness 1.5·d is split across two halves, so thread engagement is thin. The gasket land has no groove. Dowel positions are A. The B-rep summary fails whenever a split volute is built (RW-S09-02, `:212`, P0) | Partly | Partly. RW-S09-02 is recorded |
| Engine assembly | `engine_cad.py` places re-imported parts by bounding box with 10 mm clearance (`:154`), then runs a solid interference audit that fails closed (`:244-248`) | A for placement, by design | Every relative position is arbitrary. No flange, mounts or lines | **N** (the docstring says placeholder) | Yes: RW-S11-03 and RW-S11-04 (`:241-242`) and ROADMAP R4 |
| Instrumentation, igniter, fittings, alignment, mounts, test-stand adapter | Absent, apart from an injector igniter tube | — | No Pc tap, TC bosses, coolant P/T taps, dowels or pilots on the chamber joint, lifting features, or stand interface | **N.** This is the largest gap for anything "testable" | No plan item for the features. L6 (`:379`) notes that no instrumentation standard is held. The igniter is L0 with no source in the corpus (RW-S08-06, `:205`; L8) |
| Not modelled at all | — | — | Nozzle extension and attach ring, coolant crossover, throat insert, thrust-structure mount, motor internals, bearing and seal internals | n/a | Partly (RW-S09-09, RW-S10-05) |
| Valves, lines, brackets, gimbal | No CAD | — | — | n/a | Yes. `docs/program/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md` and RW-S10-05 (`:232`), blocked on literature L2 |

**Root-cause pass.** The five defects have no single shared line of code. They do share a pattern: each dimension is set by a local floor or ratio, and nothing constrains it against neighbouring features.
- The spigot top is placed exactly on the exit plane (`z1 + tip_engage = z_holes`).
- The scroll's inner edge is never compared with r2 or r4.
- The chord clamp never checks the radial span it produces.
- The flange bore is never compared with the jacket.

Underneath that, no gate measures function on the built solids. That is why §7 starts with cross-feature invariants (P0.0).

## 3. Alignment with literature and prior software

**Where the repo matches traditional practice:**
- **Chamber and nozzle (SP-125, which is Huzel & Huang, *Design of Liquid Propellant Rocket Engines*, NASA SP-125):**
  - The Rao TOP chart and the throat arcs follow the SP-125 Fig 4-14 convention.
  - L* closure and CR/α come from the SP-125 §4.3 ranges.
  - Liner stress uses SP-125 Eq 4-31, and the jacket uses the SP-125 hoop formula.
  - Milled-channel limits follow Wadel 1997 (`manufacturing_limits.py:86ff`).
- **Pump:** the SP-8109 chain (D2, b2, eye, blade count, camber) and the SP-8052 inducer (hub ratio, 3 blades, solidity, α/β).
- **Pintle:** the Nardi 2015 five-layer architecture and the skip distance.

**Where the repo sits relative to prior software:**
- The chain "1-D sizing → meridional/camber distributions → solid" is the CFturbo/BladeGen pattern [BG].
- "Every feature is a function of solver outputs plus encoded rules" is the Noyron pattern [BG; press sources only]. The repo is already a rules-as-code system. Compared with Noyron it lacks a dense manufacturing and hardware rule base, cross-feature constraints, and feedback from tests.

**Where it diverges.** Each item below is a rule in the local corpus that the code does not use:
- **SP-8109** (`19740020848.md`):
  - prime vane count (`:5805`)
  - b3 ≈ b2 (`:3933`)
  - casing that follows the impeller contour, especially for an open-face impeller (`:3620`)
  - blade fillets (`:5463ff`)
  - inlet seal clearance (`:3046ff`)
  - asymmetric volute section, conical exit diffuser and tongue (`:4012ff`)
- **SP-125 §9.3 flange-joint method** (`.../markdown/19710019929 (1).md:41311ff`; gasket factor m at `:41526`, flange factor n at `:41531`):
  - The method is bolt load from gasket seating and operating conditions, plus flange-thickness equations.
  - It is written for feed-line flanges. It applies by analogy to the head joint.
  - `interface.py` uses ratio heuristics plus a Roark plate screen instead.
- **SP-8087** (fluid-cooled chambers): manifold practice, i.e. variable- versus constant-area torus and flow splitters. The regen plenum does not apply it.
- **SP-8089** (injectors): orifice L/d of at least about 4 for jet-direction control. Pintle exit L/d is roughly (Rp−Ri)/d, which is about 3 for the reviewed case. Treat this as a check to add, not yet a defect.
- **SP-125 §9.2 and §9.6:** tube support spacing; gimbal ±4–10°.

**Out-of-corpus standards the hardware shell needs:**
- AS568 / ISO 3601 O-rings with the Parker gland tables
- ISO 273 clearance holes and ISO 4762 counterbores
- ISO 286 fits
- one port family (SAE J1926 / AS5202 boss, AN/JIC, or NPT)
- VDI 2230 or NASA-STD-5020 for bolted joints (see `docs/literature/LITERATURE_REQUESTS.md`; L15 at `REMAINING_WORK_AND_GOALS.md:388`)

**O-ring scale note** (corrects the draft):
- AS568 does include 0.040, 0.050 and 0.060 in cross-sections, but only as dash -001 to -003 at inside diameters of a few mm or less.
- At a seal diameter above about 3 mm, the smallest AS568 cross-section is 0.070 in (1.78 mm) [BG].
- The 1.5 × 0.75 mm gland therefore corresponds to a ~1.0 mm-section metric O-ring. Those exist as vendor catalog items, not as AS568 sizes [BG].
- The rule must therefore be scale-aware:
  - above a threshold, key the gland to an AS568 / ISO 3601 dash size;
  - below it, key the gland to a named metric catalog ring, or switch to a crush gasket or metal seal;
  - the ledger records which branch applied.

**Prior tools: what to copy and what not to.**
- **CFturbo / BladeGen [BG]:**
  - Copy a defaults table that shows the correlation behind each distribution.
  - Copy recorded user overrides, with provenance.
  - Copy the volute module's feature set: cross-section families, tongue and exit diffuser.
- **ESP / OpenCSM / CAPS [BG]:**
  - Copy attribute propagation, i.e. semantic face names such as `seal_face`, `bolt_hole[i]` and `coolant_inlet` that downstream checks can query.
  - Copy topology-stable parameterisation with a published validity range.
  - Copy "regeneration failure = infeasible point".
  - Do **not** migrate: it is the same OCCT kernel, aimed at aircraft, with no gain over CadQuery here.
- **KBE / HLCt (La Rocca 2012):** a stable parameter interface plus an explicit validity envelope. That is the contract each builder should publish.
- **Skeleton / master-model design:** the repo has one real skeleton object, `InterfaceGeometryResolution` (`interface.py:291`). Generalise it.
- **RPA / TDK, and RocketPy / OpenRocket [BG]:** geometry stops at contour export, or the tool is vehicle-level. Nothing to copy.
- **Literature outside the corpus:**
  - The 1992 AIAA edition of Huzel & Huang is not held. The 1967/71 SP-125, which is held, carries the flange, manifold and gland practice the draft relies on.
  - Sutton & Biblarz is general background and contains no parametric hardware rules.
- **PicoGK / nTop / implicit geometry:** not now. It loses the B-rep, the STEP output and tolerancing, and the channel counts here do not need it. **Revisit if** the process becomes LPBF with conformal manifolds, lattice infill or integral cooling features that B-rep lofts cannot express.

## 4. The owner's proposal: hand-CAD generic templates sized by the software

**What the hypothesis gets right.** Sizing alone does not produce hardware. Today the shell dimensions are min_tool multiples or mm literals (`injector_coaxial_cad.py:90-136`), or ratios on one driving diameter (`interface.py:479-506`, `pumps.py:2259`). That is exactly what a designer's reference layout settles: where the seal sits, what the port looks like, and how parts locate and retain each other. Defects 1–4 are precisely the cross-feature relations a designer checks by eye.

**What it gets wrong, if taken literally.**
1. **Topology changes with sizing.** Channel count N, blade count Z, splitter count, vane count, bolt count, exit count and element count are all solver outputs. GUI templates break on integer changes and on topological naming [BG]. In code these are loops, and the repo already handles them that way.
2. **Freeform surfaces are solver output.** The Rao/MOC contour, channel h(x), camber lines and the volute area law cannot live in a template. A template could only wrap a code-generated spline, which is what the code already does.
3. **Geometric similarity does not hold.** Wall thickness follows P·r/σ, not size. O-ring sections, threads, bolts, fittings and tool radii are discrete. A "sized" template needs one driving rule per dimension. Once those rules are written down, the rules *are* the template, and they belong in code.
4. **Toolchain.**
   - STEP (AP214/AP242) carries no regenerable parameters.
   - No desktop CAD application is installed in `/Applications`: no FreeCAD, Fusion, SolidWorks, Inventor or Onshape app. The only CAD-adjacent app is BambuStudio.
   - Browser CAD such as Onshape cannot be detected, so the owner may still be fluent in one. See Q1.
   - The repo's only native-CAD reference is an outbound Inventor `.ipt` conversion manifest (`export.py:856-870`).

**Toolchain options, evaluated:**

| Option | Verdict | Why |
|---|---|---|
| SolidWorks design tables / API; Inventor iLogic | **Reject** | Windows-only and licensed. Not headless or CI-capable on the owner's Mac. Cannot sit in a batch or MDO loop |
| Onshape configurations / FeatureScript via REST | **Authoring only** | Fine for sketching intent. As a runtime dependency it brings network access, accounts, rate limits and latency. Free-plan documents are public, which is an IP issue |
| FreeCAD 1.x spreadsheet + headless `freecadcmd` | **Possible, not recommended** | Free, local and the same OCCT kernel. But it adds a second kernel wrapper, topological-naming fragility when counts change, and a second source of truth |
| Template as CadQuery code (owner specifies intent, agent encodes) | **Adopt** | CadQuery 2.7.0 is already the kernel in `.venv-jax`. Diffable and testable; loops handle topology; builders already emit sidecars and fail closed. build123d is not installed in the repo venv, so do not add a second CAD library to the repo |
| Agent-side text-to-cad tools (build123d CAD skill, CAD Viewer, step.parts, engineering-drawing, dfm, dfam-check) | **Adopt as tooling, not as a repo dependency** | Uses that need no repo change: view exported STEP files in CAD Viewer for owner review; draw quick build123d mock-ups of a proposed design-intent sheet in scratch space; fetch fastener, bearing and motor STEP from step.parts; produce drawing sheets; run process checks on the exported STL/STEP. Whether engineering-drawing accepts an external STEP is to be verified before relying on it |
| Owner hand-CAD at one sizing point | **Visual reference only** | STEP has no feature names, so automated conformance against it would need bespoke feature extraction. Conformance is tested against the owner's dimension table instead (P2.1) |
| Vendor STEP for bought parts | **Adopt, scoped** | Select by sizing and place by skeleton datums, never resize. Vendor STEPs have no named datums, so interface datums are entered once per part |
| 3D printing (BambuStudio installed; printer type unverified) | **Adopt for form, fit and flow-path checks** | The quickest physical answer to "does it resemble usable hardware": assembly order, O-ring fit, bolt access, and visibly blocked exits under water flow. It is **not** a Cd or performance calibration, and FDM will not hold features below about 0.5 mm. Resin is better |

**Engineering criteria for the code-template route:**
- **Validity envelope.** Each builder declares its parameter ranges and refuses out-of-range calls, following the existing fail-closed pattern used for the movable pintle and chart extrapolation.
- **Regeneration robustness.** A host-run LHS-plus-corners sweep checks for a single valid solid, the functional gates, and minimum wall and ligament. OCCT fuzzy-boolean failures (`regen_cad.py` `fuzzy_tolerance_mm`) are triaged as either kernel or design failures.
- **Testing.** Named-dimension conformance against the owner's dimension table, measured on builder-tagged features. No mesh or volume diffs.
- **Version control.** Code and rule tables diff cleanly; native CAD files do not.
- **MDO.**
  - Today MDO does not rebuild CAD in the loop. `job.py:1175-1185` runs only design-space and channel checks, and `parts="reference_cad"` is a post-solve export (`:639-655`).
  - "CAD regeneration failure = infeasible point" is therefore a contract to build for the sweep and for any future in-loop use. It is not existing behaviour.
  - No differentiability is needed.
- **Maintenance.** One source of truth.

**Recommendation: ADOPT-HYBRID.** The challenge was that owner intake is the weak link. The fix is to change the owner deliverable to a design-intent sheet (sketch plus dimension table) instead of a CAD file; the agent can mock it up in scratch build123d for review.
- **Owner design-intent sheets, encoded as code, for:**
  - the chamber head flange, seal and pilot register
  - regen manifold heads with port bosses
  - instrumentation bosses
  - the igniter port
  - the injector mechanical stack
  - the pump split joint, bearing cartridge, shaft stack and motor interface
  - mounts and the test-stand adapter
- **Keep as pure solver code; fix the defects and add rule checks:** contour, channels, liner and jacket, metering core, impeller, inducer, diffuser and volute flow surfaces.
- **Catalog STEP:** fasteners and bearings first (step.parts). Valves get an envelope plus port datums until a vendor is chosen.

## 5. What would NOT work

1. **GUI templates for solver-shaped parts** (nozzle, channels, blades, volute, injector element patterns). Their topology and freeform shape come from the solvers.
2. **Uniform or affine scaling of a reference STEP.** Standards are discrete, and thickness follows pressure, not size. A scaled O-ring, thread or fitting is not a real part.
3. **STEP as a parameter carrier** ("CAD it, export STEP, the software resizes it"). STEP has no driving parameters.
4. **Windows-only CAD APIs** in the pipeline.
5. **Onshape or FreeCAD as a runtime generator next to CadQuery.** Two sources of truth drift.
6. **Expecting templates to fix the defects.**
   - Solid checks do exist today: `isValid` and volume closure (`tests/test_pump_cad_brep.py`), and solid–solid interference in the assembly (`engine_cad.py:244-248`).
   - The gates that *should* catch these defects do not look at the built solids. Injector `audit_flow_connectivity` rebuilds idealised void primitives (`injector_coaxial_cad.py:721-730`). The area identity is arithmetic. Pump interference is recorded but not raised (RW-S11-04).
   - Only cross-feature invariants and open-area / swept-clearance gates on the built solids catch them.
7. **Treating the provenance ledger as hardware qualification.** It makes arbitrariness measurable. It does not move a part past `L3_REFERENCE_ASSEMBLY` (`mdo/build.py:36-44`). Tolerances and drawings (RW-S11-07), cold-flow tests and FEA are still required.
8. **A full-census ledger with a hard ratchet.** Classifying every intermediate scalar in a layout dict is expensive. The HEURISTIC count can be gamed by relabelling or splitting dimensions, and the baseline churns with every legitimate feature. Limit it to load-bearing dimensions and keep the ratchet advisory (P0.5).
9. **Migrating to ESP/CAPS or PicoGK now.** High cost, and no capability the repo lacks at this stage. The trigger for revisiting is in §3.
10. **Generating valve or motor internals before the literature exists.** SP-8094/8097/8090 and any motor-design text are not held.
11. **Treating printed parts as calibration.** A printed tip validates flow-path topology and fit only. Cd and spray data must come from the pintle pilot hardware (RW-S08-01).

## 6. What is already done or accounted for (do not redo)

- **Code-first B-rep everywhere,** with STEP re-import, validity and volume gates, and assembly interference audits:
  - chamber `export.py`
  - regen `regen_cad.py`
  - 5-part coaxial injector `injector_coaxial_cad.py`
  - pump `pump_cad_brep.py`
  - assembly `engine_cad.py`
  - honest `hardware_qualified: False` flags
- **Contour and wall driven by solvers and rules** (S05 docs; SP-125 Eq 4-31 in `thermal_design.py`).
- **Shared chamber/injector interface object:** `InterfaceGeometryResolution` (`interface.py:291`), the resolver (`:409`) and `validate_bolted_interface_geometry` (`:335`).
- **ISO bolt sizer and fastener callout exist:** `size_bolted_interface` (`interface.py:765`), and `flange_cad.py`, which is orphaned and only test-referenced. Wiring them is scheduled: RW-S10-06 (`:233`), RW-X08-01 (`:345`), ROADMAP R4.
- **Pump items already open:**
  - C6 prime vane count, b3 = 0.9–1.0·b2, tongue separation (`PUMP_GEOMETRY plan :117-121`)
  - r3/r2 ∈ [1.05, 1.10] (`:204`)
  - 3-D blade and D1-D4 collector items (RW-S09-08)
  - rotating assembly (RW-S09-09)
  - split-volute summary failure (RW-S09-02, P0)
- **Chamber items already open:**
  - analysed wall vs exported wall (RW-S11-01, P0)
  - flange/jacket collision and assembly shift (RW-S11-03)
  - two inner-edge definitions (RW-S10-02)
  - generic L*/CR defaults (RW-S05-07)
  - whole-engine dry mass (RW-S10-03)
- **Weak gates already recorded** (RW-S11-04): coaxial separation audited on idealised cylinders; pump interference not raised.
- **Process and inspection** (RW-S11-06, Track I: process registry plus B-rep interrogation) and **tolerances and drawings** (RW-S11-07). These are the repo-native homes for the DfM/DfAM and drawing work in §7. Do not create parallel structures.
- **Expected-part graph by article scope** (RW-S11-05): component articles must not demand an igniter or gimbal.
- **Valves, lines, brackets and gimbal** are planned (`PLAN_VALVES_LINES_BRACKETS_GIMBAL.md`, RW-S10-05).
- **Pintle pilot data campaign** (RW-S08-01): the owner's first experimental-feedback step.
- **Infrastructure to reuse:**
  - `BuildFidelity` L0–L5 (`mdo/build.py:36-44`)
  - `raosim/benchmarks.py` and `raosim/benchmark_data/cases`
  - `manufacturing_limits.py`
  - the ledger seed `_dim(...)` (`pump_cad.py:68`)
- **Literature already requested:** SP-8119, SP-8123, SP-8048/8101/8121, the Parker handbook, L15 tolerancing (`docs/literature/LITERATURE_REQUESTS.md`; `REMAINING_WORK_AND_GOALS.md:388`).
- **New findings, not recorded anywhere:**
  - injector exit half-blockage
  - `deflector_angle`-blind coaxial nose
  - scroll intrusion inside r2, and its systematic b2/D2 condition
  - vane-span clamp
  - cylindrical pocket on an open-face impeller
  - root cause of the bore defect (t_hot at `run_nozzle.py:5273`)
  - no flange on the analysed `regen.step`
  - glands not keyed to a named O-ring
  - all instrumentation and fitting features
  - cross-feature invariants and functional solid gates
  - per-dimension provenance
  - ROADMAP R4's ✅ for the injector STEP and regen channels (`ROADMAP_REQUIREMENTS_TO_PARTS.md:217, :220`) overstates what those parts can do

## 7. Recommended plan

### P0: make what exists functional and measurable (~3 weeks agent time)

**P0.0 Cross-feature layout invariants.** The cheapest catch, runnable in the sandbox without CadQuery.
- **Deliverable:** fail-closed asserts inside the layout and resolver functions, plus fast tests:
  - **Injector:** spigot top ≤ z_holes − h_exit/2 − min_ligament; cap ≥ max(min_feature, 0.5·h_exit).
  - **Pump:** casing_r − a(θ) ≥ r4 + clearance for every θ (≥ 1.05·r2 when vaneless); vane radial span chord·sin α ≥ 0.95·(r4 − r3).
  - **Chamber:** the interface chamber_od ≥ 2(Rc + t_hot + h + t_jacket) on the regen path.
- **Acceptance:** each assert fails on current `main` with a planted fixture, and passes after P0.1–P0.3.
- **Effort:** 0.5–1 day.

**P0.1 Injector exit window.**
- **Deliverable:** end the spigot at least `h_exit/2 + min_ligament` below `z_holes`, or cut the exits through the spigot as well (`injector_coaxial_cad.py:386-391`). Size the cap per P0.0. Optionally build the nose as a cone from `deflector_angle`, as in `injector_cad.py:120-134`.
- **Acceptance:** the P0.4 open-area gate returns at least 0.98·A_design at r = Ri and r = Rp, in both styles.
- **Effort:** 1 day.
- **Owner:** confirm the tip intent: one exit row or Nardi's two; cone or hemisphere.

**P0.2 Pump stator corrections.**
- **Deliverable:**
  - Derive the scroll base circle from r4 (or r3) and let the centroid radius grow with A(θ). Replace the four `0.68·D2` copies with one function.
  - Prime vane count (C6).
  - b3 = (0.9–1.0)·b2.
  - Wedge or cambered vanes that span the full annulus.
  - A casing pocket that follows the shroud, with a declared tip clearance.
  - Blade fillets.
  - Close RW-S09-02 in the same pass.
- **Acceptance:** P0.0 invariants; swept rotor ∩ stator = ∅ (P0.4); one rule-check test per SP-8109 ratio.
- **Effort:** 4–5 days, overlapping the existing plan.

**P0.3 Chamber joint**, split by its dependency on Q2.
- **P0.3a (no owner input needed):**
  - Pass the jacketed OD into `resolve_bolted_interface_geometry` on the regen path.
  - Call `check_channel_geometry` inside the CAD export.
  - Add a run-level probe test that reads the regen exporter's jacket OD and the resolved bore from one canonical run.
  - Effort: 1 day.
- **P0.3b (after Q2):**
  - **Integral flange:** fuse the flange, holes and seal land into the regen body.
  - **Bolt-on flange:** build a separate ring with a bore of jacket OD plus clearance and a seal to the jacket.
  - Either way, retire the uniform `wall.step` flange path for regen runs, which closes the RW-S11-03 collision and RW-S11-01 for the flange.
  - Adopt `size_bolted_interface` (RW-S10-06).
  - Acceptance: hole axes coincide within 1e-6 m on both solids; report the flange mass delta.
  - Effort: 2 days.

**P0.4 Functional solid gates.**
- **Deliverable:** `raosim/cad_gates.py`. Use **point-classification sampling** (`Solid.isInside` on a grid across each exit section at several radial stations) rather than booleans, because OCCT booleans on 0.5 mm caps are fragile:
  - `open_area_through(...)`, built first as the proof, using the current spigot as a planted-defect fixture that must FAIL
  - `swept_rotor_clearance(...)`, using a revolved rotor envelope intersected with the stator solids
  - `hole_pattern_coincidence(...)`
  - `min_ligament(...)`
  - Wire them into the reports and fail closed.
- **Tests:** host-only, in `.venv-jax`. The fast layout twin is P0.0.
- **Effort:** 3–4 days.

**P0.5 Scoped provenance ledger.**
- **Deliverable:** `raosim/cad_provenance.py`:
  - `DimClass` with the values SOLVED / RULE / STANDARD / MFG / HEURISTIC / PLACEHOLDER
  - `DimRecord` with key, value, class, source_id, locator, rule_expr, valid_range and the **active branch of any `max()` floor**
  - records written to the sidecar `.cad.json`
- **Scope:** **load-bearing dimensions only** (seal, wall, ligament, flow area, bolt, fit, clearance). Start with the injector stack and the chamber joint, then the pump.
- **Class rules:**
  - Fix the class boundaries with worked examples before populating.
  - A floor counts as MFG only when the min_tool branch is active and cites the tool-diameter input. When a mm literal binds, it is HEURISTIC.
- **Tests:**
  - RULE and STANDARD records resolve to a corpus file or a `docs/literature/standards_register.md` entry.
  - RULE records carry an executable check (prime vane count, 0.9 ≤ b3/b2 ≤ 1.0, L/d ≥ 4).
  - No part reaches `L4_MANUFACTURING_REFERENCE` while a load-bearing record is HEURISTIC or PLACEHOLDER.
  - The HEURISTIC-count ratchet is **advisory** (reported in `summary.json`) for the first release.
  - Prefer rule-check tests over the census.
- **Effort:** 2 days scaffold, then 0.5–1 day per component. Migrate `pump_cad.py:68 _dim`.
- **Owner:** approves each HEURISTIC rationale, or routes the dimension to a design-intent sheet.

**P0.6 Visual review pack.**
- **Deliverable:** a meridional section and a key cross-section PNG per part, with solver-driven dimensions labelled, in each build directory, in the style of `moc_diagrams.py`. Add CAD Viewer links to the STEP files for interactive review.
- **Owner:** a 10-minute checklist per release: seal location, ligaments, flow-path continuity, bolt access.
- **Effort:** 1 day.

### P1: standards, skeleton, testability (~4 weeks)

**P1.1 Standards and feature library.**
- **Deliverable:** `raosim/standards/` tables, keyed from vendor catalogs with the source recorded, not copied from the ISO/SAE text:
  - AS568 / ISO 3601 with Parker static glands, plus a named metric small-section catalog ring and a scale-applicability rule (§3)
  - ISO 273 holes and ISO 4762 counterbores
  - ISO 286 fits
  - ISO 8734 dowels
  - one port family, chosen by Q4
- **Deliverable:** `raosim/cad_features.py` with gland, bolt hole, pilot, dowel pair, port boss and instrument boss features, each emitting STANDARD records.
- **Acceptance:** every gland and port is built through the library; squeeze and gland-fill checks pass.
- **Effort:** 5–6 days.
- **Owner:** chooses the standards; confirms that Parker ORD 5700 is acceptable as a source [BG: free vendor download].

**P1.2 Engine datum skeleton.**
- **Deliverable:** generalise `InterfaceGeometryResolution` into `EngineSkeleton` (axis, face and flange planes, port positions and types, pump mount planes). `engine_cad.py` places parts by datum, which closes RW-S11-03.
- **Effort:** 2–3 days.

**P1.3 Testability features.**
- **Deliverable:**
  - Pc tap
  - TC wells near the injector face (SP-8087 practice)
  - coolant P/T bosses
  - injector manifold P taps and a dynamic-pressure port
  - pump inlet/outlet taps
  - igniter boss
- **Acceptance:** each feature is reachable per P0.4 and present in the ledger.
- **Effort:** 2–3 days.
- **Owner:** Q3.

**P1.4 Validity-envelope sweep.**
- **Deliverable:** `scripts/cad_envelope_sweep.py`, a host-run LHS plus corners across every builder and gate. Classify each failure as kernel (boolean or fuzzy tolerance) or design. Record each builder's envelope in its manifest; out-of-envelope calls refuse.
- **Acceptance:** at least 95 % pass inside the declared envelope.
- **Effort:** 3–4 days plus host compute.

**P1.5 Printed form, fit and flow article.**
- **Deliverable:** after P0.1, print the injector stack at 1:1, in resin if available. Check assembly order, O-ring seating, bolt access, and a water-flow visual of the exit row. Optionally print a transparent or sectioned pump scroll after P0.2.
- **Acceptance:** an owner checklist. Results are logged as form, fit and flow-path evidence only.
- **Effort:** agent 1–2 days for export and process checks (`dfam-check` on the STL); owner print time.

**P1.6 Interim process checks.**
- **Deliverable:** run `dfm` on the machined parts (internal corners, tool reach) and `dfam-check` on the LPBF candidates (overhang, minimum wall, powder removal) against the exports, and log the findings. Fold the result into the RW-S11-06 process registry; do not build a parallel one.
- **Effort:** 1–2 days.

### P2: owner design-intent families and bought parts (ongoing)

**P2.1 Design-intent families.**
- **Owner deliverable:** a sketch plus a dimension table (CSV or markdown) at one declared sizing point. Each dimension is labelled as a standard, as `= k·driver` with the reasoning, or as an MFG value. A STEP is optional, as a visual reference.
- **Agent deliverable:** a rule table, a CadQuery builder and ledger records. Store the sheet in `raosim/benchmark_data/cases/`. Add a conformance test that measures builder-tagged features against the table. Regeneration across the design space is covered by P1.4, not by this golden case.
- **Order, following RW-S08-01 (pintle pilot as the first experimental step), unless Q3 says otherwise:**
  1. injector mechanical stack
  2. chamber head flange and manifold heads
  3. pump joint, cartridge, shaft stack and motor interface
  4. mounts and the stand adapter
- **Effort per family:** owner 2–6 h; agent 1–2 days.

**P2.2 Literature golden cases.**
- Nardi 2015 pintle proportions.
- The SP-125 §9.3 flange sample, with the m/n gasket method implemented as a RULE in `interface.py`.
- SP-8109 J-2 pump ratios.
- Effort: 2–3 days.

**P2.3 Catalog parts.**
- **Deliverable:** `raosim/catalog/`:
  - fasteners and bearings from step.parts first
  - motors by power and speed where step.parts covers them
  - valves and fittings as an envelope plus port-datum stubs until a vendor is chosen
- Datums are entered manually per vendor part. Placement goes through `EngineSkeleton`.
- **Effort:** 5–8 days.
- **Owner:** approves the vendors.

**P2.4 Literature.** Acquire SP-8119, SP-8123, SP-8048/8101/8121, the Parker handbook, and VDI 2230 or NASA-STD-5020, in the order the families need them.

## 8. Open questions for the owner

1. **Which CAD tool do you actually design in, and on which OS?** No desktop CAD app is installed here. A browser tool such as Onshape is possible. The answer only changes the intake format for P2.1.
2. **What manufacturing process is intended for each part?** For example: LPBF GRCop chamber with integral manifolds and flange; machined injector; machined or cast pump. This is a **prerequisite for P0.3b**, and it sets the MFG rule set and the P1.6 checks.
3. **What is the first physical test article and test?** Pintle pilot cold-flow per RW-S08-01, pump water test, or hot-fire. The answer sets the P2.1 order and the P1.3 instrumentation.
4. **What fitting and port standard does your stand use?** AN/JIC, SAE J1926/AS5202, NPT or compression. This sets the P1.1 port family.
5. **Over what design space must the CAD regenerate** (thrust and Pc ranges, propellant pairs)? This sets the P1.4 bounds.
6. **Will you accept bought parts as fixed vendor STEPs selected by sizing?** If not, valve and motor generators wait on literature that is not held.
7. **Which printer and process do you have** (BambuStudio suggests Bambu FDM)? Is resin printing available? This sets whether P1.5 can resolve the sub-millimetre exits.

## 9. Verification notes (corrections after red-team review)

- **Cap location:** kept at `injector_coaxial_cad.py:433`. The red-team's `:427` was wrong: grep finds the `# thin cap ring` union at `:433`. I removed the "~0.9 mm hole" figure because `hole_d` is case-specific.
- **"About 50 % blocked":** upgraded from plausible to derived. The spigot occupies all z < z_holes over r ∈ [Ri, Ri + 0.55(Rp − Ri)], and `tip_engage` ≥ 3 mm > h/2, so exactly the lower half of each exit is blocked in both styles.
- **Flange bore vs jacket OD:**
  - The red-team said this was unproven for the regen path. Re-checked and upheld: `run_nozzle.py:5273` replaces the seed with `t_hot` after sizing, `:6377` passes it, and `interface.py:479` sets OD = 2r + 2·wall.
  - Reclassified as "symptom already recorded" (RW-S11-03 collision, RW-S10-02 inner-edge definitions) with the root cause newly identified.
- **Paths:** `run_nozzle.py` → `raosim/run_nozzle.py` (`scripts/run_nozzle.py` is a 34-line shim). Plan documents now resolve under `docs/program/`, `docs/subsystems/S0x-.../` and `docs/literature/`.
- **Line references:** `throat_geometry.py:19-20` → `:22-23`. `manufacturing_limits.py:76-86` → `:86ff`. `design.py ~2370` → `:2373`. `pumps.py` area floors and vane width → `:2579-2581`, `:2580`. `tip_thread_major` → `injector_coaxial_cad.py:155`. Regen defaults cited at `regen_cad.py:570-576`.
- **Gates:** the claim "gates do not check built solids" is narrowed. `isValid`, volume and the assembly interference audit are solid checks. What is missing is open-area, swept-clearance and flow-path checks on the built solids; the injector connectivity audit checks idealised voids (`:721-730`).
- **Glands:**
  - The red-team said 1.5 × 0.75 mm is only a floor. Re-checked: at the default min_tool of 0.3 mm (`injector_cad.py:359`), the literal floors bind, so 1.5 × 0.75 mm *is* the realised gland.
  - Corrected my own claim that this is "shallower than any AS568". AS568 -001 to -003 (0.040–0.060 in) exist at tiny IDs. The gland fits a ~1 mm metric ring and is not keyed to a named one. Added the scale-applicability rule.
- **β2 = 25°:** "hard-coded" upheld. There is no spec override at `pumps.py:1264-1296`. Added that the value lies inside the conventional band [BG].
- **Toolchain:** corrected. BambuStudio is installed; text-to-cad agent tools are available; browser CAD cannot be detected. Added the corresponding rows and P1.5 / P1.6.
- **Sonnet-only measurements:** the figures 9.69 vs 12.05 mm, "45 % vane span" and "39 `_dim` rows" are replaced by derived conditions (b2/D2 > 0.051; span = 4·sin α) or dropped.
- **Red-team omissions rejected:**
  - "Huzel & Huang missing": NASA SP-125 *is* Huzel & Huang and is cited throughout.
  - "SP-8087 = injector criteria": SP-8087 covers fluid-cooled chambers and SP-8089 covers injectors, as the draft already used them.
- **Recommendations revised:**
  - Ledger scoped to load-bearing dimensions, with an advisory ratchet and active-branch recording.
  - P0.4 effort raised and switched to point sampling, with a layout twin (P0.0).
  - P0.3 split so that Q2 gates only P0.3b.
  - Hybrid intake changed to a design-intent sheet plus table.
  - Catalog effort raised and scoped to step.parts first.
  - Implicit-geometry revisit trigger recorded.
  - "Regeneration failure = infeasible" stated as a contract to build, not existing behaviour.
