# Pump audit: reference geometry to inspectable hardware

Audit date: 2026-09-05. Scope: current working tree at HEAD `f763fe1`, including existing local changes. This report changes no solver, CAD builder, or pre-existing plan. Code line references describe this audited tree and can move after edits.

## Finding

The pump branch is a coupled preliminary sizing and reference-CAD system with useful hydraulic, structural, and assembly screens. It is not yet a released mechanical pump design or a process-qualified set of parts. The central missing geometry is still the spanwise blade surface. Nevertheless, three premises in the supplied assessment need correction: P1 has not exported impeller hub/tip inlet angles; some pump manufacturing and mechanical screens already exist; and unknown NPSH is already explicit in pump feasibility records.

The current output envelope must be described by configuration, model applicability, process, and evidence status. The reported 800 N to approximately 100 kN sweep is an observed software-success interval for one set of inputs, not a validated operating envelope and not evidence that all intermediate points or all propellants are covered.

## 1. What is present and what remains

| Area | Current implementation evidence | Remaining requirement |
|---|---|---|
| Coupled annular eye | `pumps.py:1690` onward and `:2004`: shaft/hub fit, net annular continuity, inlet blockage, exit blockage, iteration; tests cover convergence and structural thickness feedback | Structured failure outcome; impeller hub/tip and interior-span inlet velocity triangles; explicit prewhirl/incidence assumptions; geometry and regime applicability |
| Impeller camber | `pumps.py:1959` integrates one polar camber line with angle linear in radius | Common meridional-parameter/span surface, stacking rule, consistent rotation convention, angle and continuity verification |
| Impeller solid | `pump_cad_brep.py:106`, `:215` and `:226`: planar camber ribbon extruded axially, clipped by meridional bodies | Valid 3-D pressure/suction faces or planar-section construction, edges, thickness distribution, measured clearances, shrouded/open variants |
| Inlet angles | Impeller `PumpVelocityTriangle` at `pumps.py:423`, `:1423` has one inlet metal angle evaluated using the eye tip speed. `InducerGeometry` at `:354` has hub/tip angles, exported at `:2324` | Complete P1 before P2. The existing inducer hub/tip angles cannot serve as impeller hub/tip angles |
| Splitters | Separate inlet/main and splitter counts, CAD partial ribbons | Define start using common meridional parameter, carry 3-D section law, verify blade-to-blade separation and local passage area |
| Diffuser | `build_diffuser_ring`, `pump_cad_brep.py:399` onward, rotates rectangular solids at the selected vane angle | Cambered vanes, local incidence, correctly defined normal-flow throat, operating-range and interaction screens |
| Volute | `pump_cad_brep.py:450`: explicit constant-mean-velocity area schedule on a constant-radius centerline; actual fluid-volume subtraction and split casing exist | Selectable sourced collection law, passage/exit transition, tongue, collector loads; matching meanline/CAD flow-area definitions |
| Process screens | `pumps.py:1810` onward: split-casing clamp/land/bolt layout and nominal scroll-tool diameter check; `pump_cad_brep.py:899` onward audits the split | Material/process/machine/tool/holder/fixture capability, full reachable toolpaths, radius/finish/tolerance capability, feature inspection and process records |
| Mechanical screens | Rotating-stress, shaft-torsion, blade-root, casing-pressure, bearing-DN and seal-speed screens; wear-ring/balance-hole hooks | Selected bearings/seals, supported shaft layout, leakage/thrust balance, fits and thermal growth, coupling/retention, lubrication, detailed joint design, dynamic and life analyses |
| CAD audits | STEP re-import, volume/validity; interference, nominal clearances, split access, flow-passage and selected meanline identity checks | More complete geometry-to-model comparison and an explicit one-solid requirement for each monolithic part; transaction/partial-package state |
| Release state | `pump_cad_brep.py:1542` returns `cold_flow_release_ready=False` and `hardware_qualified=False`, with external blockers | Evidence-backed promotion for a particular hardware configuration and specified test, not promotion merely because a STEP can be re-imported |

`git log -1 -- raosim/pump_cad_brep.py` reports commit `625f1b3`, dated 2026-07-22. The assertion that this file has not changed since July 13 is stale. The central 2-D-blade finding remains correct.

The STEP export gate at `pump_cad_brep.py:1437` checks validity and positive volume. It does **not** explicitly require `solid_count == 1`. A stronger future single-part acceptance gate should not be described as already implemented at this export boundary.

## 2. Bounded reproduction

Verification executed with `.venv-jax/bin/python`:

```text
python -m pytest -q tests/test_pumps.py
38 passed in 1.80s
```

The following minimal CLI probe was run at thrusts 500 N and 130000 N, with separate output directories. It intentionally omits regen and CAD to isolate pump sizing and its numerical CAD-layout inputs. It is not the full user sweep or the 13 kN all-feature benchmark.

```text
.venv-jax/bin/python -m raosim --no-banner
  --propellant LOX/RP-1 --oxidizer LOX --fuel RP-1
  --pc 3000000 --target-thrust THRUST --epsilon 6 --max-nfev 0
  --injector pintle --electric-pump
  --fuel-tank-pressure 500000 --oxidizer-tank-pressure 600000
  --burn-time 30 --pump-cad none --cad none --out OUTPUT
```

| Probe | Observed result | Interpretation |
|---|---|---|
| 500 N | Exit 1; `ValueError: coupled shaft/hub envelope leaves no credible impeller-eye annulus`, `pumps.py:2047`; no `summary.json`. `pintle.json`, contour CSV and images survive | The reported fatal pump failure is reproduced. The whole result record is lost, although some preceding artifacts survive |
| 130000 N, no CAD | Exit 0; `summary.json` and `pump.json` written; aggregate pump `feasibility.feasible` is `true` | Sizing completion currently overstates the downstream geometry viability |
| Fuel bolt-layout helper from the 130000 N `pump.json` | Requested 16 body bolts; `bolt_layout_fits_clear_arc=true`; `_split_joint_hole_layout` places 16 | This particular helper passes |
| Oxidizer bolt-layout helper from the same file | Requested 32 body bolts; upstream `bolt_layout_fits_clear_arc=false`; `_split_joint_hole_layout` raises `split casing has insufficient clear body-bolt sectors` | The high-end CAD failure can be predicted before building any expensive solid |
| NPSH in the 130000 N result | Both `npsh_margin_pa` values are `null`; both `inducer_npsh_*` gates are informational and explicitly say the margin is unknown | Unknown reporting exists; a pump requirement/performance model and stronger release handling remain absent |

Temporary reproduction artifacts were written under `/tmp/lrekit-pump-audit-20260905-500` and `/tmp/lrekit-pump-audit-20260905-130000`, with sibling `.log` files. The recipe above is the durable reproduction specification.

No full CAD sweep was repeated. The reported 19 STEP / 17 STL success counts and nine surviving fuel STEP files at 130 kN remain user-provided observations. Inspection of the export control flow supports the partial-package mechanism, but this audit does not relabel those counts as independently reproduced.

### Immediate software defects

1. `run_nozzle.py:5912` calls pump sizing outside an exception-to-result boundary. The failure occurs before `pump.json`, pump feasibility, or final summary serialization. Introduce a component outcome that captures the failed model/constraint/input context and continue producing independently valid component results. Preserve a non-success run state and a documented exit code.
2. The split joint already returns `bolt_layout_fits_clear_arc` (`pumps.py:1899` and `:1932`). `_assess_feasibility` around `:3326` checks clamp stress and tool fit but does not gate the layout-fit boolean. The 130 kN reproduction is therefore `feasible=true` even when its own upstream layout knows it cannot fit. Add the missing constraint and share one layout evaluator between sizing and CAD.
3. `export_pump_brep_package` writes each role in sequence (`pump_cad_brep.py:1455` onward). A later role failure leaves earlier-role files. Units and other package records are written near the end. `run_nozzle.py:6013` catches the CAD exception, replaces the package record with `{status,error}`, writes `summary.json`, and returns; it neither removes the partial CAD directory nor gives that directory a complete inventory/status record. Use staging plus atomic promotion, or a manifest written before generation and updated per artifact. Never delete a user's pre-existing output directory as cleanup.
4. A numerical preflight should distinguish physical/model infeasibility, unsupported configuration, unavailable input/evidence, and CAD implementation failure. A bolt-layout implementation limit is not a universal thrust limit. NPSH unknown should be visible and should prevent an evidence-based test-release claim when NPSH is required by that test.

## 3. Corrections required before adopting the new hardware plans

These findings apply to the existing untracked `LREKIT_HARDWARE_PROGRAM_2026-09-05.md` and `PUMP_PHYSICS_AND_PROCESS_BASIS_2026-09-05.md`. Those documents were read but not modified.

### P1 and the 3-D camber acceptance criteria

The program at lines 123-127 says the meanline already exports impeller hub/tip inlet angles. It does not; the paired fields belong to the inducer. Complete P1 using the solved impeller annulus and declared velocity-profile/prewhirl assumptions before building a surface intended to match those values.

The proposed common-parameter equation is a useful geometric construction when the blade angle is defined from the tangential direction. The following is a mathematical derivation, not a literature performance claim:

```text
s_s(u) = (dm_s/du) / (r_s(u) tan(beta_s(u)))
theta_s(u) = integral[0,u] s_s(v) dv - integral[0,1] s_s(v) dv
theta_s(1) = 0
theta_hub(0) - theta_shroud(0)
    = -integral[0,1] (s_hub(v) - s_shroud(v)) dv
```

Thus TE stacking makes the hub/shroud trailing-edge clocking difference **zero**, not equal to the difference of their integrated wraps. The latter appears with the stated sign at the leading edge. The program's H1 acceptance at lines 153-154 contradicts its TE boundary condition at line 139. Match TE clocking, independent wrap integrals, local inlet/exit tangent angles, rotation handedness, and surface regularity as separate checks.

TE stacking does not itself enforce a common exit metal angle. Exit tangent angle and stack clocking are independent boundary conditions. An axial TE also relies on the meridional endpoint geometry sharing the exit radius. That happens in the current channel; it is not guaranteed for every later meridional family.

A new integration using meridional arc length `dm` cannot generally be bit-compatible with the old radial integration `dr` on an axial-to-radial passage. Use the radial planar limit as a compatibility test and explicitly record differences on the curved channel. Do not constrain the corrected 3-D geometry to reproduce a wrong 2-D projection at arbitrary mid-span.

A transverse-section construction also needs an explicit coplanarity check. Offsetting hub and shroud endpoints along different 3-D surface normals does not automatically give four coplanar points. Prototype the parameterization and sew/loft strategy before calling the geometry construction settled.

### Ruledness is not a machining certificate

H3 at lines 174-187 treats straight generators as sufficient for flank milling and non-ruled geometry as obligatorily point milled. That is too strong. A ruledness residual can be an input to a process route, but it does not establish cutter-envelope fit, reachable axes, holder/fixture clearance, gouge-free motion, or achieved tolerance. The offset pressure/suction surfaces also need evaluation; checking only the camber surface is insufficient.

Primary CAM documentation explicitly treats surface developability/parameterization, tool-axis behavior, gouge checking, and holder collisions as distinct concerns: Autodesk, *Restrictions of swarf milling* and *Strategy tab (Swarf)*, FeatureCAM 2024, accessed 2026-09-05. [Surface restrictions](https://help.autodesk.com/cloudhelp/2024/ENU/FCAM/files/GUID-D8B72BD5-55EF-4389-A3CA-9AE5C9696C11.htm); [gouge and holder checks](https://help.autodesk.com/cloudhelp/2024/ENU/FCAM/files/GUID-963C77CD-6561-43E6-A2F0-58333D4F51AF.htm). These establish the required software/process distinction, not capability for a particular shop or pump.

### Trailing-edge trimming is not universal calibration authority

H4 at lines 195-200 converts historical trimming observations into a universal nominal `trim` parameter with ±10% authority on head coefficient. The source does not establish such a transfer function, guaranteed range, or symmetric response for every impeller. Keep the geometry parameter, but its response must be measured or otherwise validated for the selected family and conditions; any coefficient fit must retain uncertainty, applicability, and independent validation.

Source verified directly from the original PDF image: R. B. Furst, *Liquid Rocket Engine Centrifugal Flow Turbopumps*, NASA SP-8109, December 1973, §2.3.1.2, PDF page 41 / printed page 29, Figure 15, document ID 19740020848. The passage describes increasing/decreasing head through under/over-filing and changes as large as 10%; it is not a general response law. Local source: `propulsion_texts/fuel_pump_design/19740020848.pdf`.

### Historical process recommendations need scoped applicability

SP-8109 §3.3.3 is useful existing literature, but the basis document's description of its criteria as a complete closed-form reachability test or the limits for calling a shrouded part machinable “at all” is not defensible. Implement historical recommendation screens with their original configuration, material, tool assumptions and unit convention. Follow with feature/toolpath checks against a selected current process. Do not apply a shrouded-impeller rule globally to the current open-face rotor.

The repository's pump specific speed is nondimensional (`pumps.py:1238`). A source threshold expressed under another convention must never be compared directly with that field. Resolve the source definition and verify its original PDF before coding a numeric conversion or threshold.

Source image verified: Furst, NASA SP-8109 (1973), §3.3.3 “Fabrication,” PDF page 84 / printed page 72, document ID 19740020848. This page contains the shrouded-impeller machining recommendations, process-specific cutter proportions, assembly/balance guidance, and historical overspeed-test recommendation. A historical demonstrated rotor speed is not a design allowable or proof of compatibility for another fluid/material/process combination. A generated test requirement needs the selected design's mechanical and test evidence, not tip speed alone.

The corpus conversion of SP-8109 is flagged `needs_review`; this audit read `propulsion_texts_for_agents/AGENTS.md`, the index and extraction report first. Only the two original PDF pages cited above were independently image-verified for this pump audit. Other numerical recommendations in the new basis remain subject to its own verification status and must not become constants merely by being copied into a backlog.

### Existing screens and roughness support must not be erased from the assessment

The basis at lines 120-121 repeats that no process model reaches the pump. The limited split-casing nominal tool-fit and pressure/joint screens contradict that absolute claim. The correct gap is comprehensive process and inspection capability, especially for the impeller.

Its claim that regen pressure-drop models assume only smooth walls is also stale. `run_nozzle.py:2752` accepts absolute channel roughness; `physics.py:1005` divides by hydraulic diameter, and `mdo/cooling.py:227` provides a rough-wall friction branch. The missing capability is selected-process/as-built-metrology to roughness/model-input provenance, together with an explicit roughness definition. A surface-finish statistic is not automatically an equivalent sand-grain roughness.

There is a concrete integration concern to resolve: MDO `schema.py:415` labels `channel_roughness` relative and `mdo/cooling.py:413` consumes it as relative, while `mdo/postprocess.py:301` passes the same resolved value to the scalar cooling input, whose definition is metres. Nonzero values require a dimensional-contract audit before any process model is connected.

### Validity prose and hashes are reusable foundations, not completed loops

The hardware program's lines 87-96 say a computed envelope is blocked only by prose. Translating known constraints is useful software work, but “no demonstrated performance-validity range” contains no hidden numerical envelope to transcribe. Unknown requirements remain unknown until supported by a model or evidence. Hardware/process domains need both implementation and new evidence.

The assertion at lines 82-85 that missing hash quantization disqualifies measurement binding should also be withdrawn. An exact configuration hash should identify the exact authoritative resolved definition. A separate tolerance-equivalence or similarity key may assist comparisons; rounding must not silently bind a measurement to a different geometry or erase a physically significant change. A manufacturing serial/revision/as-built record is additionally needed to identify the physical article actually tested.

## 4. Recommended pump development sequence and acceptance

| Order | Deliverable | Acceptance that changes capability |
|---|---|---|
| A, immediately | Typed component outcomes, early run manifest, bolt-layout preflight, atomic or explicitly partial package | 500 N writes an honest failed pump result while preserving valid upstream results; the 130 kN known layout failure is reported before CAD; interrupted/failed export leaves a self-describing directory and inventory |
| B, foundational | Resolved configuration/component identity, model revision and units contract | Each geometry, prediction, gate, process plan, BOM row and result refers to the same resolved configuration; exact content identity and equivalence are distinct |
| C, P1 then P2/P3 | Impeller spanwise inlet data and 3-D blade surface/solid | Local metal angles and integral wraps agree with the geometry; TE stacking is verified independently; monolithic parts re-import as one valid solid; representative regimes and geometric limit cases pass |
| D, P4-P7 | Section/edge law, splitters, diffuser throat, collector and outlet geometry | Measured CAD passage/throat areas use the same stations and normal directions as the model; changed passages trigger hydraulic/load re-evaluation; source applicability and approximations are explicit |
| E, in parallel from B | Measurement and prediction-comparison pipeline | Geometry/serial-bound values, units, uncertainty and conditions; explicit station mapping and signed residuals; correlated uncertainty supported where needed; no automatic parameter mutation |
| F, starts during C/D | Process and mechanical architecture | A shop/process/tool envelope and a bearing/seal/shaft/joint arrangement constrain the developing geometry; later full reachability analysis is applied to stable surfaces; unresolved items remain explicit |
| G, after stable topology | Detail geometry, final fillets, process plan, inspection/drawings, mechanical verification | Final geometry is rechecked after fillets; tolerance, finish, assembly, dynamic and pressure-load evidence supports the specified test article and operating/test conditions |
| H, controlled feedback | Calibration proposals and design revision | Fit only declared identifiable coefficients; retain source measurement IDs, applicability, uncertainty, parameter covariance and held-out validation; issue a new model/configuration revision instead of silently changing predictions for an existing record |

Process and mechanical architecture should start alongside the 3-D flow-path work. Full cutter/holder reachability on final blade geometry can wait until that geometry stabilizes. Waiting to choose open/shrouded construction, bearing locations or assembly access until after all hydraulic CAD is complete risks invalidating the geometry. Fillet operations should remain late, followed by renewed passage, stress-feature and interference checks.

The corresponding product promise is: high-level requirements produce a resolved design, component outcomes and evidence-linked artifacts within a declared capability domain. Unsupported inputs should still produce a useful explanation and unaffected subsystem outputs. “Any thrust class” is an input/API ambition; it is not a legitimate universal hardware-validity claim.
