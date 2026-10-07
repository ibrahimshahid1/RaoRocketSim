# LREKit knowledge system — ingestion report

Generated 2026-09-20T22:58:13.163341+00:00

## Corpus

- PDFs on disk: 280; conversion rows: 286; synchronized: 286; missing Markdown: 0; stale: 0; orphan Markdown: 2; duplicate payload groups: 27
- Unique documents: 252; alternate copies: 4
- Conversion status: {'needs_review': 234, 'good': 18}
- Low-confidence conversions: 14

## Chunks and index

- {'ingested': 0, 'unchanged': 248, 'skipped_no_markdown': 0, 'skipped_failed_conversion': 0, 'n_chunks_total': 19565, 'removed': 0, 'skipped_alternate_copy': 4, 'generated_at': '2026-09-20T22:58:11.033785+00:00'}
- {'skipped': 'no chunk changes'}

## Knowledge stores

- models: 61
- validity: 95
- equations: 37
- claims: 179
- entities: 74
- software_links: 1479
- tests: 1061
- capabilities: 98
- requirements: 31
- repo_documents: 62
- relationships: 0
- provenance: 1161

## Model registry ↔ source support (extraction pass)

- {'PARTIALLY_SUPPORTED': 8, 'NOT_FOUND_IN_SOURCE': 1, 'SUPPORTED': 2, 'NOT_CHECKED': 50}
  - `altitude.attached_thrust` → PARTIALLY_SUPPORTED: Hagemann et al. 1998 (Advanced Rocket Nozzles) is a qualitative review with no numbered equations. It does NOT state a quasi-one-dimensional attached-flow thrust-coefficient or nozzle momentum-balance relation anywhere; 
  - `cad.kernel_topology` → NOT_FOUND_IN_SOURCE: CAD_04.md is a 173-slide lecture deck on B-Splines and NURBS (basis functions, divided differences, Cox-de Boor recursion, knot insertion/removal, rational curves and surfaces). The registry attributes to it 'OpenCascade
  - `chamber.shoulder_fill_fraction` → PARTIALLY_SUPPORTED: SP-125 (Huzel & Huang) contains NO chamber-shoulder fillet radius, no symbol Rs or Rs,max, and no 0.8 factor anywhere in the thrust-chamber geometry text (grep of 'shoulder' hits only fatigue/seal contexts; 'fillet' only
  - `flow.boundary_layer_displacement` → PARTIALLY_SUPPORTED: The document does contain the relation the registry attributes to it: Sec. 7.4 'Turbulent Flow' (PDF pp. 839-842) derives a turbulent flat-plate displacement-thickness estimate, Eq. (7.46), from Prandtl's one-seventh-pow
  - `injector.compressible_orifice` → PARTIALLY_SUPPORTED: The document (White, Fluid Mechanics 7th ed., Ch. 9 sections 9.4 and 9.6) does contain the relation the registry describes: an isentropic subsonic mass-flow function (Eq. 9.47, valid when the flow is not choked) and a ch
  - `injector.d2_evaporation` → PARTIALLY_SUPPORTED: The registry attributes the d-squared law d^2(t) = d0^2 - K t to this document via 'evaporation-model context and Eq. 16'. The document's Eq. 16 is a Spalding-type mass-transfer rate, m˙_d = k_c A_d ρ ln(1 + B_m), not a 
  - `injector.incompressible_orifice` → PARTIALLY_SUPPORTED: NASA SP-8089 (Liquid Rocket Engine Injectors, 1976) never writes the orifice mass-flux relation G = Cd*sqrt(2*rho*dP) or any algebraic orifice-flow equation; a genuine search for discharge coefficient, sqrt/square root, 
  - `injector.momentum_resultant_angle` → PARTIALLY_SUPPORTED: The registry relation theta = atan2(M_radial*cos(delta), M_axial + M_radial*sin(delta)) does not appear anywhere in SP-8089. Grepped 'momentum', 'resultant', 'vector', 'direction', 'axial', 'radial', 'deflect', 'tan', 'c
  - `injector.n_tau_stability` → PARTIALLY_SUPPORTED: NASA SP-8089 (this document) contains no n-tau equation, no interaction-index or sensitive-time-lag values, and no sensitive-frequency formula. Its combustion-stability content (sec. 2.1.1.3, pp. 42-45; design criteria s

## Extraction grounding

- grounded: 324, page-shifted: 0, ungrounded (dropped): 0
- equations: 37 (confirmed 0, refuted 0), claims: 179, validity statements from sources: 34

## Validation

- issues: 3 — {'markdown_without_manifest_row': 2, 'registry_citation_not_confirmed_in_local_source': 1}
- stats: {"equation_variables": {"ambiguous": 12, "unresolved": 88, "total": 140}, "constants_by_basis": {"IMPLEMENTATION_CHOICE": 351, "UNKNOWN_BASIS": 223, "LITERATURE_DERIVED": 189, "PROJECT_DERIVED": 41, "REPOSITORY_HEURISTIC": 175, "EMPIRICALLY_CALIBRATED": 2}, "components_without_scientific_link": 214, "graph": {"nodes": 4514, "edges": 5685, "edges_without_provenance": 0, "top_degree": [["MODEL_NOZZLE_RAO_VARIATIONAL_BVP", 145], ["DOC_NTRS_19710019929", 131], ["DOC_NTRS_19760023196", 111], ["MODEL_NOZZLE_MOC", 97], ["DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI", 83]]}, "duplicate_chunk_texts": 15}

## Explicitly unavailable / blocked

- {'BLOCKED_ON_LITERATURE': 35, 'UNSUPPORTED': 13, 'NOT_QUALIFIED': 9, 'NOT_IMPLEMENTED': 9, 'UNAVAILABLE': 32}

## Hard-coded constants by basis

- {'IMPLEMENTATION_CHOICE': 351, 'UNKNOWN_BASIS': 223, 'LITERATURE_DERIVED': 189, 'PROJECT_DERIVED': 41, 'REPOSITORY_HEURISTIC': 175, 'EMPIRICALLY_CALIBRATED': 2}

UNKNOWN_BASIS constants (223), first 30:
  - raosim/atmosphere.py:50 `20–47 km lapse` = 0.001
  - raosim/atmosphere.py:55 `tail temperature above 47 km` = 270.65
  - raosim/benchmarks.py:972 `residual_tol (default)` = 5e-3
  - raosim/benchmarks.py:314 `rao residual_tol default` = 5e-3
  - raosim/benchmarks.py:376 `exit Mach tolerance` = 0.25 * ideal_me
  - raosim/cea.py:111 `eta_Isp default` = 0.95
  - raosim/design.py:1904 `boundary-layer epsilon loss gate` = 0.08
  - raosim/design.py:230 `channel_entry_loss_coefficient` = 0.5
  - raosim/design.py:231 `channel_exit_loss_coefficient` = 1.0
  - raosim/design.py:229 `port_loss_coefficient default` = 1.5
  - raosim/design.py:259 `conductivity default` = 16.0
  - raosim/design.py:259 `default conductivity` = 16.0
  - raosim/design.py:261 `default max_heat_flux` = 25e6
  - raosim/design.py:260 `default max_temperature` = 1250.0
  - raosim/design.py:258 `default yield_strength ('Inconel 718')` = 900e6
  - raosim/design.py:261 `max_heat_flux default` = 25e6
  - raosim/design.py:260 `max_temperature default` = 1250.0
  - raosim/design.py:258 `yield_strength default` = 900e6
  - raosim/flange_cad.py:58 `_PRELOAD_FRACTION_PERMANENT` = 0.90
  - raosim/injector.py:2059 `Dittus-Boelter coefficients` = 0.023, 0.8, 0.4
  - raosim/injector.py:2013 `recirculation_temp_fraction (default)` = 0.8
  - raosim/injector.py:2014 `recirculation_velocity_fraction (default)` = 0.2
  - raosim/injector.py:258 `evaporation_constant (default)` = 1.0e-6
  - raosim/injector.py:236 `fuel_cd (default)` = 0.7
  - raosim/injector.py:234 `fuel_dp_fraction (default)` = 0.2
  - raosim/injector.py:237 `oxidizer_cd (default)` = 0.7
  - raosim/injector.py:235 `oxidizer_dp_fraction (default)` = 0.2
  - raosim/injector.py:3735 `cavitation number pass/warn thresholds` = 1.5 / 1.0
  - raosim/injector.py:4161 `concentricity pass/warn` = 0.10 / 0.25
  - raosim/injector.py:3800 `L/D warn threshold` = 1.0
