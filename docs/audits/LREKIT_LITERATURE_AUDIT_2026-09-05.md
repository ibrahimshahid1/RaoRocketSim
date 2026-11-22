# LREkit literature reconciliation and acquisition issues

Audit date: 2026-09-05. Scope: the local corpus and the source requests for reference geometry, manufacturing, assembly, testing, and model feedback. This is a source/acquisition audit, not a completed verification of every equation in these references. Existing request ledgers are historical inputs; their links and acquisition claims must not be inherited without checking.

## Evidence and inventory

The corpus instructions in `propulsion_texts/propulsion_texts_for_agents/AGENTS.md` were read. They require the original PDF to control numerical and physical claims and require title, author/year, section, PDF page, and identifier in derived claims. The index and Markdown are retrieval aids. In particular, the extraction report flags 213 of 224 records `needs_review`; only 11 are `good`. A `good` conversion still does not validate an implemented model.

Recounting the filesystem and `paper_index.json` gives:

| Inventory unit | Count | Interpretation |
|---|---:|---|
| Loose PDF paths below `propulsion_texts/` | 255 | Includes duplicate copies and unrelated papers; not 255 independent references |
| Indexed records | 224 | 218 loose-file paths plus six ZIP-member paths |
| Loose PDFs absent from index | 37 | A search of the index alone misses acquired literature |
| ZIP-member index records | 6 | `archive.zip::member.pdf` is not a missing loose-file path |

Presence classifications below mean **present/indexed**, **present/unindexed**, **abstract only**, or **not located**. “Not located” means no matching owned source was found in the index, loose-file inventory, or the relevant retrieved corpus text; references cited inside other papers do not count as acquired copies. No documents were downloaded or purchased in this pass. Web checks establish publisher/NTRS identity and advertised scope, not that a full source has been read.

## Sources already held: use them before requesting duplicates

| Source family | Local evidence | Acquisition status and remaining work |
|---|---|---|
| NASA SP-8052, *Liquid Rocket Engine Turbopump Inducers*; SP-8109, *Liquid Rocket Engine Centrifugal Flow Turbopumps*; SP-8107, *Turbopump Systems for Liquid Rocket Engines*; SP-8125, *Liquid Rocket Engine Axial-Flow Turbopumps* | `fuel_pump_design/19710025474.pdf`, `19740020848.pdf`, `19750012398.pdf`, `19780023221.pdf`; also duplicated elsewhere | Present/indexed. Extract source-specific hydraulic, mechanical, fabrication, and test requirements; retain applicability by architecture and fluid. SP-8125 is axial-flow material, so transfer to the centrifugal implementation needs an explicit justification. [NASA confirms SP-8125, April 1978](https://ntrs.nasa.gov/citations/19780023221). |
| Reis et al. (2019), *Design and Testing of Centrifugal Fuel Pump for Liquid Propellant Rocket Engine L75*, DOI 10.2514/6.2019-4432 | `fuel_pump_design/reis2019.pdf` and duplicate | Present/indexed, `needs_review`. A candidate independent pump benchmark. Build a station/geometry/conditions/data-quality record from the original, rather than importing curves without test context. |
| Wadel (1997), *Comparison of High Aspect Ratio Cooling Channel Designs for a Rocket Combustion Chamber* | `wadel1997.pdf` | Present/indexed. A channel study does not establish pump cutter access or a universal shop capability. Preserve the application restrictions already identified in the physics audit. |
| Gradl series, Kerstens et al. (2021), and GRCop material studies | `gradl2016.pdf` through `gradl2020.pdf`, `kerstens2021.pdf`, `materials_science/` | Present/indexed. Valuable manufacturing and testing case studies; they do not replace a selected alloy/process/machine qualification or inspection plan for every part. |
| Huzel and Huang, NASA SP-125; NASA SP-8087 fluid-cooled chambers; SP-8089 injectors; SP-8120 nozzles; SP-8113 combustion stabilization devices | `19710019929.pdf`, `19730022965.pdf`, `19760023196.pdf`, `19770009165.pdf`, `19750020175.pdf` | Present/indexed. Do not describe the library as containing no test practice. The gap is a selected, implemented, traceable procedure with applicable acceptance criteria and uncertainty. [SP-8120 is July 1976](https://ntrs.nasa.gov/citations/19770009165); [SP-8113 is November 1974](https://ntrs.nasa.gov/citations/19750020175). |
| Perakis et al. (2021), *Wall Heat Flux Evaluation in Regeneratively Cooled Rocket Thrust Chambers*, DOI 10.2514/1.T6056 | `perakis2021.pdf` and duplicate | Present/indexed, `needs_review`. The synthetic CFD benchmark identified in the physics audit must be labelled simulation-derived; it is not independent hardware validation. |
| G1–G11, G13–G14 from `LITERATURE_REQUESTS_GENERALIZATION.md` | Turbines `19740026132.pdf`; gas generators `19730018978.pdf`; self-cooled chambers `19780013268.pdf`; CEA `19950013764.pdf`, `19960044559.pdf`, `20020085330.pdf`; `huber2009.pdf`, `bell2014.pdf`, `Vandenberg_Corrected_Properties.pdf`, `nasuti2021.pdf`, `PaperLREenginemassandsizing.pdf`, `20100034924.pdf`, `20090037061.pdf` | All 13 present/unindexed. CEA being absent is a stale statement in the body of the old ledger. Acquisition is complete; equation/source review and indexing are separate tasks. |
| G12, Hulka and Casiano swirl-coaxial stability review | `20130014182.pdf` | Abstract only: original file has one page. The proposed ROCCID replacements `19910014917`/`19910014918` were not located locally. |
| ASME Y14.5-2018, Y14.24-2012, Y14.41-2003 | Long descriptive filenames at corpus root | Present/unindexed, including another copy of Y14.5. Inspect title/edition and adopted drawing convention before use. Owning Y14.41 does not demonstrate that an exported STEP file carries compliant product/manufacturing information. |

## Correct the identifiers before acquisition

The following identities were checked against primary NASA catalog metadata. All are **not located locally**. The old request is identified to preserve correction provenance. On September 6 the six identifiers were repaired in `LITERATURE_REQUESTS.md` and, where repeated, `LITERATURE_REQUESTS_HARDWARE_LOOP.md`; acquisition and full-source review remain open.

| Existing request | Correct source and primary record | Previous incorrect identifier in the request ledgers |
|---|---|---|
| 13 / hardware-loop §1 | Butner and Keller (1971), *Liquid Rocket Engine Turbopump Bearings*, NASA SP-8048 — [19710018535](https://ntrs.nasa.gov/citations/19710018535) | `19710018568` |
| 14 / hardware-loop §1 | NASA (1972), *Liquid Rocket Engine Turbopump Shafts and Couplings*, SP-8101 — [19740006328](https://ntrs.nasa.gov/citations/19740006328) | `19730012577` |
| 15 / hardware-loop §1 | Burcham and Keller (1978), *Liquid Rocket Engine Turbopump Rotating-Shaft Seals*, SP-8121 — [19780022641](https://ntrs.nasa.gov/citations/19780022641) | `19780012205` |
| 6 | NASA (1976), *Liquid Rocket Disconnects, Couplings, Fittings, Fixed Joints, and Seals*, SP-8119 — [19770017247](https://ntrs.nasa.gov/citations/19770017247) | `19770009166` |
| 4 | NASA (1973), *Liquid Rocket Pressure Regulators, Relief Valves, Check Valves, Burst Disks, and Explosive Valves*, SP-8080 — [19740002611](https://ntrs.nasa.gov/citations/19740002611) | `19730013806` |
| 20 | NASA (1969), *Transient Loads from Thrust Excitation*, SP-8030 — [19710014805](https://ntrs.nasa.gov/citations/19710014805) | `19690023005` |

The visible label for request 17 also disagrees with its URL: `19770009165` is the locally held nozzle monograph, not proof that the pressurization-system source is acquired. The remaining old IDs have not all been independently checked in this bounded pass; they must not be relabelled “verified.”

## Acquisition and implementation issues

These are proposed repository issues, not posted external issues. Cross-references use the existing H/I/J/K/L/M track names in `LREKIT_HARDWARE_PROGRAM_2026-09-05.md`; the issue acceptance criteria below stand independently of that draft's technical claims.

### LIT-01 — Reconcile the source registry before importing more coefficients

**Links:** all tracks; H12/H13; existing requests 1–26 and G1–G14.

Import the 37 unindexed paths, preserve archive-member identity, merge duplicate-source records by content hash without deleting files, fix the six verified identifiers above, and classify the G12 abstract separately from a full paper. Each equation/curve record needs original PDF hash, author/year, report/DOI, section, PDF and printed page, units, convention, applicable operating conditions, extraction method, reviewer, and whether the data are measured, digitized, simulated, or assumed. Digitizing an existing SP-8109 figure is a validation/data task, not an acquisition. Record digitization uncertainty and distinguish a historical comparison table from held-out test evidence.

**Done when:** a source manifest reconciles 255 loose paths and the archive records, and the model registry links to reviewed source records rather than free-text citations alone. This work can start immediately.

### LIT-02 — Support the 3-D hydraulic shape and collector with explicit scope

**Links:** H1–H6; original pump P2–P9; H1/H2/H8/H9 literature requests.

The following are **not located locally**:

| Source, exact identity | What to extract and what it does not prove |
|---|---|
| Katsanis (1964), *Use of Arbitrary Quasi-Orthogonals for Calculating Flow Distribution in the Meridional Plane of a Turbomachine*, NASA TN D-2546 — [19650002705](https://ntrs.nasa.gov/citations/19650002705) | Throughflow method, boundary conditions, and assumptions for a future hydraulic check. A 3-D CAD representation does not require adopting this particular solver first. |
| Katsanis and McNally (1969), *FORTRAN Program for Calculating Velocities and Streamlines on a Blade-to-Blade Stream Surface of a Tandem Blade Turbomachine*, NASA TN D-5044 — [19690012449](https://ntrs.nasa.gov/citations/19690012449) | Blade-to-blade flow analysis; check its assumptions before applying it to the chosen impeller. |
| Zangeneh (1991), *A Compressible Three-Dimensional Design Method for Radial and Mixed Flow Turbomachinery Blades*, DOI [10.1002/fld.1650130505](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650130505) | Prescribed-loading/inverse-design method. Publisher abstract reports an inviscid Euler comparison on a radial-inflow turbine; it is not pump test validation. Optional after geometry fidelity, not a prerequisite for fixing the planar blade. |
| Brown and Bradshaw (1949), *Design and Performance of Family of Diffusing Scrolls with Mixed-Flow Impeller and Vaneless Diffuser*, NACA Report 936 — [19930092001](https://ntrs.nasa.gov/citations/19930092001) | Scroll method including friction. NASA describes compressor-component tests; do not treat its measured results as direct LOX/RP-1 pump evidence. |

**Done when:** the selected geometry laws have source-backed conventions and testable invariants, and any hydraulic correlation has a declared domain and validation plan. No retrieved title alone establishes a universally correct blade loading, diffuser throat, or volute law.

### LIT-03 — Acquire the missing mechanical methods and selected-part data

**Links:** J; existing requests 13–15; H1/H1b/H1c/H7/H10/H11.

Acquire SP-8048, SP-8101, and SP-8121 using the corrected records. Add the following **not located** sources:

| Source | Role |
|---|---|
| Luehr and Watters (1970), *Development of Steady-State and Dynamic Performance Prediction Methods for Turbopump Self-Compensating Thrust Balance Systems*, NASA CR-72630 — [19700014095](https://ntrs.nasa.gov/citations/19700014095) | Evaluate a thrust-balance option if the computed load ledger calls for it. Rotational speed alone does not establish that a balance piston or hydrostatic bearing is mandatory. |
| Due (1966), *An Empirical Method for Calculating Radial Pressure Distribution on Rotating Disks*, ASME paper 65-WA/FE-10; journal record [19660051053](https://ntrs.nasa.gov/citations/19660051053), conference record [19660036439](https://ntrs.nasa.gov/citations/19660036439) | Candidate cavity-pressure basis. NTRS verifies identity but the journal record has no downloadable full text; full-source acquisition remains open. |
| Barrett (1990), *Fastener Design Manual*, NASA RP-1228 — [19900009424](https://ntrs.nasa.gov/citations/19900009424) | Public starting reference for joint/fastener analysis. Still requires an adopted fastener family, actual material/process data, gasket behavior, and relevant preload evidence. |
| Rosales, Shoffstall and Stoltzfus (2007), *Guide for Oxygen Compatibility Assessments on Oxygen Components and Systems*, NASA/TM-2007-213740 — [20070016582](https://ntrs.nasa.gov/citations/20070016582) | Component/system compatibility methodology. Historical wearing-material recommendations are not a complete compatibility assessment for a selected pump and process. |

The missing deliverable is also a **vendor/application data package**: identified bearings, seals, coupling, retention, motor/drive, temperature-dependent material allowables, and their load/speed/environment restrictions. The [SKF super-precision catalog](https://www.skf.com/binaries/pub12/Images/0901d19680495562-Super-precision-bearings-catalogue---13383_2-EN_tcm_12-129877.pdf) is a primary example of selection data, but describes machine-tool products; it does not qualify a bearing for propellant lubrication or oxygen exposure. Supplier evidence must match the selected application.

Do not mark the 1963 impeller reports permanently “unobtainable” merely because SP-8109's historical bibliography marks them confidential. Their present release/access status was not verified here. Record that status as unresolved and select an accessible structural-analysis basis plus material/process allowables; analysis and test evidence remain separate requirements.

**Done when:** every rotating-assembly and joint check has a sourced method, selected-part inputs, explicit unknowns, and linked structural/rotordynamic validation evidence. Acquiring the three SP monographs alone does not complete mechanical design.

### LIT-04 — Turn manufacturing literature into feature and shop constraints

**Links:** I1–I4, H3; H6/H8.

Chu, Wu and Lei (2012; online 2010), *Tool Path Planning for 5-Axis Flank Milling of Ruled Surfaces Considering CNC Linear Interpolation*, DOI [10.1007/s10845-010-0386-3](https://link.springer.com/article/10.1007/s10845-010-0386-3), is **not located**. Its publisher abstract explicitly considers tool-motion/interpolation error and a milling/CMM comparison. It supplies a concrete successor to the vague H8 request: geometrical rulings alone are insufficient to establish achievable flank-milling accuracy.

[OPEN MIND's Multiblade documentation](https://www.openmind-tech.com/en-gb/cam/5-axis-milling/impeller-blisk/) is a verified primary vendor reference for distinct hub, point, flank, edge, and fillet operations and collision checking. This is process-workflow evidence, not a promise that the repository's blade is machinable. Obtain the chosen shop's tool/holder geometry, machine kinematics, fixtures and stock assumptions, achievable finish/tolerance evidence, inspection access, and a representative CAM collision/material-removal review.

Do not use an ISO fit or general-tolerance table as a measured process capability. Also, [ISO 2768-2:1989 is withdrawn and replaced by ISO 22081:2021](https://www.iso.org/standard/7749.html); an adopted edition and convention must be explicit. The held ASME drawing standards already supply language, while shop capability and part-specific dimensions/PMI remain work.

**Done when:** each critical feature maps to a chosen process, tool/fixture access, inspection method and evidence; unsupported features return `unknown` or `infeasible`. Historical SP-8109 and Gradl guidance is retained with its original process/material context instead of promoted to universal cutoffs.

### LIT-05 — Establish measurement, uncertainty, comparison, and validation conventions

**Links:** K0–K5, I4/I5, L1; H3–H5/H14.

The following are **not located locally** and have primary publisher records:

| Source | Required use and applicability |
|---|---|
| JCGM (2008), *Evaluation of Measurement Data — Guide to the Expression of Uncertainty in Measurement*, JCGM 100 — [BIPM official DOI](https://www.bipm.org/en/doi/10.59161/jcgm100-2008e) | Measurement uncertainty convention. Freely available; do not classify GUM as requiring paid/institutional access. Select the applicable supplements from the [official GUM catalog](https://www.bipm.org/en/web/guest/publications/guides), including correlated outputs and nonlinear propagation where needed. |
| NASA (2024), *Standard for Models and Simulations*, NASA-STD-7009B — [NASA standards record](https://standards.nasa.gov/standard/nasa/nasa-std-7009) | Model credibility, lifecycle and declared acceptance criteria. A framework to adopt, not a certification automatically obtained by citing it. |
| ASME, *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*, V&V 20-2009 (R2021) — [official edition](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer/2009) | Comparison of specified simulation variables with data at specified validation points and their uncertainties. It does not establish accuracy throughout an arbitrary extrapolated thrust envelope. |
| ISO 9906:2012, *Rotodynamic Pumps — Hydraulic Performance Acceptance Tests — Grades 1, 2 and 3* — [official scope](https://www.iso.org/standard/41202.html) | Select a cold-flow acceptance convention. Its published scope is liquids behaving as clean, cold water; cryogenic test transfer and cavitation conventions require separate justification. |
| ISO 5167-1:2022, *Measurement of Fluid Flow by Means of Pressure Differential Devices… — Part 1* — [official scope](https://www.iso.org/standard/79179.html) | DP-meter installation/reduction and uncertainty, with relevant device-specific parts. Its scope excludes pulsating flow and requires single-phase, subsonic flow; it cannot alone validate startup or two-phase channels. |

**Done when:** measurements bind to an as-built article and configuration revision; retain value, units, station/frame, instrument/calibration, conditions, reduction version, uncertainty convention and covariance where applicable. Comparisons report signed residuals and combined uncertainty. Calibration produces a new versioned coefficient set with training/holdout separation and provenance, never retrospectively turns its training points into independent validation. These schema and provenance contracts can be implemented while literature is acquired; missing uncertainty may be recorded explicitly without being treated as zero.

### LIT-06 — Complete engine integration and transient-test coverage

**Links:** J, L, M; existing requests 1–6/17/20/24–26; H2b/H14.

SP-8080 and SP-8119 use the corrected IDs above. NASA SP-8123, *Liquid Rocket Lines, Bellows, Flexible Hoses, and Filters* (1977), is **not located**; its record [19780008146](https://ntrs.nasa.gov/citations/19780008146) is verified. These complement, rather than replace, the valve/actuator, pressure-vessel, igniter, mount, and instrumentation sources still needed by the complete engine BOM.

Breisacher and Ajmani (2008), *LOX/Methane Main Engine Igniter Tests and Modeling*, NASA/TM-2008-215421 / AIAA 2008-4757, [20090001308](https://ntrs.nasa.gov/citations/20090001308), and Tinker (2021), *Compact Augmented Spark Igniters for Liquid Rocket Engines*, [20210000596](https://ntrs.nasa.gov/citations/20210000596), are **not located**. They are concrete acquisition candidates for ignition-model/test traceability, with methane/oxygen applicability that must not silently become RP-1 evidence. Use SP-8030 for thrust-transient structural context, and procure selected valve/drive response and instrumentation bandwidth evidence for the transient implementation.

Macgregor and Csomor (1974), *Rotating and Positive-Displacement Pumps for Low-Thrust Rocket Engines*, Vol. 1, NASA CR-72965, [19740015230](https://ntrs.nasa.gov/citations/19740015230), and Csomor (1974), Vol. 2, NASA CR-72966, [19740015231](https://ntrs.nasa.gov/citations/19740015231), are **not located**. NASA identifies liquid-fluorine/Freon-12 work. These are useful architecture-selection references, not proof that a particular alternate pump solves the repository's 500 N LOX/RP-1 case, nor a measurement of its working envelope.

**Done when:** applicability and test coverage are reported per component and operating state through the same CLI/UI result contract. Literature gaps, unavailable boundary conditions, computational failures, out-of-domain extrapolation, and absent hardware evidence remain distinct statuses.

## Sequence and acceptance policy

1. Close LIT-01 and encode source/applicability fields immediately; retain successful subsystem artifacts and explicit unknowns independently of acquisition.
2. Run LIT-02 and LIT-05 in parallel with the 3-D blade and configuration/measurement implementation.
3. Start LIT-03 now because selected-part data determine the assembly; run LIT-04 against the intended blade/process before claiming machinability.
4. Close the component and integrated transient-test coverage in LIT-06 before any generated package is labelled ready for its named test.

The proposed literature gate is: **identity checked → full source held → relevant pages visually verified → applicability declared → implementation verified → independent validation evidence linked**. File presence, an equation citation, STEP export, machining feasibility, and hardware-test readiness are different achievements. No literature-only milestone establishes that arbitrary thrust classes or all component geometries are validated.
