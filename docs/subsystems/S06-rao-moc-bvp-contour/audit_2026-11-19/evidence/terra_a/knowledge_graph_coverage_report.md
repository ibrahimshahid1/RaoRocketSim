# Terra A: MOC knowledge-graph coverage and source requirements

Date: 2026-11-19 (America/New_York). Role: research and evidence routing, no production-code edits. Root/Sol A owns acceptance, implementation specifications, and the LaTeX iteration log.

## Scope and method

Read `knowledge/AGENTS.md`, then ran `knowledge/kg.sh --json status`; queried `retrieve_for_task`, `find_models`, `find_equations`, `find_claims`, `find_validity_conditions`, `find_scientific_basis`, `compare_sources`, and `get_claim_sources`. Saved each query in this directory. Inventoried source-specific records directly to avoid treating retrieval ranking or a zero-result generic search as proof of absence. Escalated load-bearing source claims to original PDF text plus newly rendered pages. Existing scientific assertions are evidence-routing records, not code correctness certificates.

The corpus reports synchronized=true: 296 PDFs on disk, 262 canonical unique payloads, 257 chunked documents; no missing/stale mirrored conversions. One orphan derivative is explicitly reported. Corpus-wide extraction completeness remains NOT_ESTABLISHED: only 30 documents have source objects; 48/61 models have no current source review. Those global counts do not by themselves assess the MOC code.

## Coverage assessment

`coverage_inventory.json` is the machine-readable inventory. Raw stored counts are distinguished from current source verification.

| Source | Equation / claim coverage | Assessment |
|---|---|---|
| DOC_DOI_10_2514_8_7324, Rao 1958, `propulsion_texts/rao1958.pdf` | 9 equations, 5 claims; source-derived validity plus registry validity | Strong source routing for control-surface integrals, stationarity, exit condition, and compatibility; incomplete coverage of detailed marching and boundary closures. Stored historical CONFIRMED objects are `LEGACY_CONTENT_NOT_BOUND`, so renewed visual checks in this report, not historical labels alone, support the requirements below. |
| DOC_NTRS_20030067852, NASA/JHU report, `propulsion_texts/20030067852.pdf` | No extracted equations or claims; one PROJECT_DERIVES registry validity | Document is retrievable, but the detailed unit processes/start-line/source support have not been banked as current reviewed equations. |
| DOC_5F36B7C4DED79BB3E90754D0F81682F7A68014BE, Anderson 3rd ed. | Current extracted thermally-perfect/quasi-1D equations exist; axisymmetric MOC equations not extracted | Do not treat thermochemistry source review as verification of Ch.11 MOC. The held PDF contains the actual Ch.11.8 derivation for targeted follow-up. |
| DOC_YOUNG_THESIS | No extracted equations/claims | Retrievable chunks exist; no source-reviewed recurrence/mesh rejection data in graph. |
| DOC_FULLTEXT01, Östlund thesis | No extracted equations/claims | Retrievable chunks exist; no current graph verification of KL coefficients or nozzle-shock details. |
| DOC_RAO1999 | No extracted equations/claims | Alternative stationarity expressions may be useful; not independently verified in this task. |
| DOC_NTRS_19690063911, small-throat transonic study | No extracted equations/claims | It cites Kliegel/Levine, but is not the original KL coefficient paper. |
| Standalone Kliegel & Levine 1969, DOI 10.2514/3.5355 | No canonical standalone manifest source located | Critical limitation: existing code's claimed Table-1 polynomial corrections cannot be verified from a graph citation or the NASA manual's high-level description. |

`MODEL_NOZZLE_MOC` has source-review state NOT_CHECKED; `validation_level=reference_matched_subset` is registry-reported. Its `promotion_status=optimizer_coupled` is explicitly a legacy keyword heuristic, while execution reach is UNKNOWN. It does not establish that the MDO inner loop uses MOC. All 25 software components returned by the two basis queries are `implementation_scientific_comparison_state=NOT_RECHECKED`. For `moc.py`, 5 basis results are NOT_CHECKED and 2 SCIENTIFIC_BASIS_UNKNOWN. For `nasa_moc.py`, 13 are NOT_CHECKED, 3 OK through reviewed Rao equations, and 2 PARTIALLY_SUPPORTED. The latter cannot certify every line of a numerical procedure.

## Original-source requirements and traceability

Source page numbers below are one-based PDF pages, with printed pages where different. Rendered pages in this directory were examined in this task; NASA pp7–9,16,20–21 were read as original-PDF text, with pp17–19 additionally rendered and inspected.

| ID | Requirement / interpretation | Original-source evidence | Relevant code / graph route | Finding category |
|---|---|---|---|---|
| TA-R01 | Full nozzle mass and thrust are fluxes over complete C-to-E control surface; DE alone is only part unless CD is separately accounted for. | Rao1958 PDF p2/printed378, Eqs.[1]–[3], Fig.1; original page visually verified. | `nasa_moc.full_control_surface_thrust`; EQ...801D1C3E, EQ...F1B6F455, EQ...7F5A3DE9 | Literature requirement; implementation review by Sol A needed. |
| TA-R02 | DE is a left characteristic, phi=theta+alpha; stationarity invariants and exit transversality apply along/at their specified locations. They are necessary conditions, not a blanket sufficiency theorem. | Rao1958 PDF p3/printed379, Eqs.[11]–[15]; visually verified. | `rao_variational.py`, `nasa_moc.calc_lrc_de`, `rao_residuals.py`; EQ...082529F7, ...BEF48217, ...1BE77D02, ...3E537078, ...B52B8DD0 | Correct criterion; output matching alone insufficient. |
| TA-R03 | Characteristic compatibility cannot be waived simply because shape or integrated thrust matches. M/theta/phi are continuous across the C-D-E interface in the displayed construction. | Rao1958 PDF p3/printed379, prose before Eq.[6] and after Eq.[15]. CLM...E79A2E24. | MOC compatibility and D-state-continuity gates. | Literature supports retaining these failed gates. |
| TA-R04 | Match mass flux through the selected BD characteristic segment and DE; integrate along their actual paths, not a uniform-plane stand-in. | Rao1958 PDF p4/printed380, Eq.[19] and adjacent prose; visually verified. Graph quote has OCR-damaged point labels, although stored interpretation is correct. | `calc_mdot_bd`, `calc_lrc_de`, `_mass_closure_diagnostics`; CLM...FCFB9073. | Literature requirement; preserve path orientation/normal conventions. |
| TA-R05 | Downstream contour changes cannot alter the prescribed upstream kernel; the BDE region is completed and the B-to-E streamline is then selected. | Rao1958 PDF p3 before Eq.[6], p4 bottom, p5 Fig.4; NASA/JHU PDF pp9–10,17 Fig.17. | `build_kernel`, `calc_bde_region`, wall tracing. | Intentional separation of kernel and BDE is valid, not inherently an architectural defect. |
| TA-R06 | Initial data must be known and supersonic; wall point enforces theta=wall angle and centerline acts as axis of symmetry. | NASA/JHU PDF p7 §3.1, p9 §3.1. | `_make_throat_initial_line`, axis/wall unit processes. | Source-supported boundary requirement. Exact axis regularity formula needs Anderson/other source escalation. |
| TA-R07 | NASA tool's 2% initial-vs-last-kernel mass check terminates that implementation. It is a documented numerical policy, not physical uncertainty or universal accuracy specification. | NASA/JHU PDF p16 §3.2.6.1. | `build_kernel` mass check, bounded-net mass tolerance. | Valid source-port convention; must retain distinction from stronger convergence evidence. |
| TA-R08 | Starting-point count, DTHETAB, and BDE RRC count control different portions of the discretization. Nominal DTHETAB is .25–.5 degrees, and smaller can refine at a runtime cost. | NASA/JHU PDF pp17–18 §3.2.8, Figs.17–18; visually verified. | `n_kernel`, `dtheta_limit`, BDE resolution policies. | More than one resolution knob is needed for a meaningful refinement study. |
| TA-R09 | The NASA Mach<=1.5 starting-line cap and chosen line shape are explicitly arbitrary numerical conventions. They must not be called physical validity bounds or silently used to justify theory corrections. | NASA/JHU PDF p19 §3.3; visually verified. | `_make_throat_initial_line`, source-visible KL mode. | Important policy vs theory distinction. |
| TA-R10 | More initial points need not monotonically improve the net: RRCs can approach/cross. Small downstream throat radius reduces points representing expansion and can amplify downstream error. | NASA/JHU PDF pp20–21 §3.4, Fig.23–24. | Refinement experiments and topology diagnostics; default Rd/Rt=.382 differs from manual's nominal1.0. | Source-grounded reason to diagnose with actual mesh, not raise point count alone. |
| TA-R11 | The displayed Rao examples are inviscid, isentropic, constant-gamma; boundary-layer correction is discussed separately. | Rao1958 PDF p3 example gamma=1.23 and p6/printed382 first paragraph. | `MODEL_NOZZLE_MOC`, Rao BVP restrictions. | An intentional scope limitation, not automatically an incorrect implementation. Variable-gamma/shock-fitting extensions require their own formulations. |
| TA-R12 | A standalone KL coefficient source is needed before claiming NASA polynomial differences are confirmed typos. | NASA/JHU p19 describes method but contains no polynomial table; KL paper absent from canonical manifest. | `transonic_kernel.py` module docstring and `_kl_axi_coeffs` corrections. | Uncertain, requiring primary-source evidence. No numerical change recommended solely from this gap. |

The common `EQ...`/`CLM...` shorthand in the table expands to the `EQ_DOI_10_2514_8_7324_` / `CLM_DOI_10_2514_8_7324_` prefix. Full identifiers/forms are in `rao_equation_inventory.json`.

## Proposed bounded work and non-claims

1. Preserve source-compatible and corrected/numerical KL branches as explicit separate modes. Never tune coefficients to the historical fixture or claim both branches independently verified from it.
2. Audit failed consistency residuals by edge family, location, cell resolution, and D attachment; source equations support these checks even when output-level benchmark passes.
3. Run dependency-aware common-coordinate resolution studies varying start line, kernel turning limit, DE/BDE samples as applicable. Record gates as numerical policy and compare actual errors without changing tolerances merely to obtain PASS.
4. Correct stale fixtures and provenance/reporting only when independently demonstrated; do not change a theory coefficient based on a stale fixture alone.
5. Bank the source requirements above into durable review artifacts before a graph rebuild. No graph stores were edited by Terra A in this task, so historical review states remain unchanged.
6. Mark the following absent/uncertain, rather than invent them: primary KL coefficient confirmation, exact M/Mstar convention support for every branch, source-reviewed axis limit, independent second full-flow oracle, general BVP convergence envelope, variable-gamma MOC, shock fitting, viscous validation.

No tests were executed by Terra A. The results here are literature/KG evidence, not numerical acceptance, benchmark completion, or physical qualification.

## Targeted follow-up: D attachment and meaning of convergence

Original Rao1958 PDF p3/printed379, immediately before Eq.[6], specifies interior continuity of M and theta and continuity of phi along CDE. PDF p4/printed380 after Eq.[19] selects point D on the kernel characteristic BD, obtains its M/theta and radial coordinate, and builds DE from it. The same symbol D is a shared geometric/state point. This supports an actual boundary attachment and continuity check, not merely a soft penalty that can trade seam error against another residual block. The source does not supply the repository's absolute seam tolerances or residual stopping thresholds; those remain numerical acceptance policy.

The graph has two source-derived Rao Assumption objects: the prescribed initial-expansion throat contour with its endpoint B part of the solution, and prescribed inlet conditions while maximizing the control-surface thrust expression. The inviscid/isentropic scope appears separately in source-derived claims and validity. No requirement records are linked to MODEL_NOZZLE_MOC or MODEL_NOZZLE_RAO_VARIATIONAL_BVP by model ID. Existing model/test labels therefore do not prove successful D attachment or nonlinear convergence. In particular, stored registry `validation_level=strict_literature_benchmark_passed_not_cfd_validated` means a published-output comparison, while code-level optimizer termination and local physical consistency remain separate conditions. No graph assumption discovered authorizes promoting a budget-exhausted optimizer merely because a weighted residual is below the user's tolerance.

## Targeted follow-up: axis source limit

Newly rendered and visually inspected: Anderson 3rd ed., held PDF `propulsion_texts/5f36b7c4ded79bb3e90754d0f81682f7a68014be.pdf`, PDF p422/printed406 §11.8, Eqs.(11.52)–(11.53); PDF p421/printed405 gives characteristic slopes Eq.(11.48). These are original source equations, not currently extracted graph records.

For characteristics oriented downstream with physical arc length s, dr/ds=sin(theta-mu) on C-minus and sin(theta+mu) on C-plus. Trigonometric substitution in Anderson's dr/r compatibility equations gives:

- C-minus: d(theta+nu)/ds = +sin(theta)sin(mu)/r.
- C-plus: d(theta-nu)/ds = -sin(theta)sin(mu)/r.

This confirms the corrected source form used in `rao_residuals.py`. For an explicitly assumed smooth symmetric flow, theta(x,r)=k(x)r+O(r^3), with finite local mu, the axis limit is k(x)sin(mu_axis). This finite limit is a derivation from the source equations plus the stated smoothness assumption; it is not an explicitly printed Anderson axis-point closure. Setting the source to zero exactly at r=0 is not generally this limit. However, present residual callers evaluate the source at segment-midpoint radius, and the axis unit process likewise uses a midpoint or NASA-specific parent extrapolation. No active numerical bug is established simply by the private helper's r<=1e-12 guard. The numerical choices require a targeted regular-axis/refinement comparison before changing them.

## Targeted follow-up: KL1969 external primary-source access

A bounded live search attempted to retrieve the original coefficient paper on 2026-11-19. Original publication identity is supported by the held NASA/JHU PDF p42 reference13: J.R. Kliegel and J.N. Levine, AIAA Journal 7(7), July1969, pp1375–1378; DOI10.2514/3.5355. NASA-hosted bibliography https://ntrs.nasa.gov/api/citations/19910013882/downloads/19910013882.pdf also lists that article; it is a bibliography and does not supply the original coefficient table.

Primary publisher access attempts:

- https://arc.aiaa.org/doi/10.2514/3.5355 — web fetch returned 403 Forbidden.
- https://arc.aiaa.org/doi/pdf/10.2514/3.5355 — web fetch returned 403 Forbidden.
- https://doi.org/10.2514/3.5355 — web fetch returned cache miss.
- https://arc.aiaa.org/doi/epdf/10.2514/3.5355 — web tool reported inaccessible.
- https://aiaa.onlinelibrary.wiley.com/doi/10.2514/3.5355 and `/doi/pdf/10.2514/3.5355` — web tool reported inaccessible.

Targeted searches for the exact title, DOI, author/page combination, and a NASA-hosted manuscript found citations and other papers, but no accessible original full text. Search absence is bounded to those queries, not a proof the article is unavailable anywhere. Secondary summaries and other-author papers were not used to validate coefficients. The proposed Table1 corrections remain unverified against the primary paper; no coefficient change is justified by this access attempt.
