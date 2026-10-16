# LREKit knowledge system — ingestion report

Generated 2026-10-16T18:20:23.849702+00:00

## Corpus

- PDFs on disk: 286; conversion rows: 286; synchronized: 286; missing Markdown: 0; stale: 0; orphan Markdown: 1; classified reference-note mirrors: 1; duplicate payload groups: 27
- Unique documents: 252; alternate copies: 4
- Conversion status: {'needs_review': 234, 'good': 18}
- Low-confidence conversions: 16

## Chunks and index

- {'ingested': 0, 'unchanged': 247, 'skipped_no_markdown': 0, 'skipped_failed_conversion': 0, 'skipped_empty_source': 1, 'skipped_empty_extraction': 0, 'n_chunks_total': 19565, 'removed': 0, 'skipped_alternate_copy': 4, 'generated_at': '2026-10-16T18:20:17.735125+00:00'}
- {'skipped': 'no chunk changes'}

## Knowledge stores

- models: 61
- validity: 97
- equations: 93
- claims: 188
- entities: 94
- software_links: 1792
- tests: 1068
- capabilities: 100
- requirements: 36
- repo_documents: 64
- relationships: 39
- provenance: 1709
- artifacts: 306
- enrichments: 91

## Scientific coverage

- Canonical publications: 248; with current extracted objects: 9; without: 239
- Independent source-object verification: {'CONFIRMED': 65, 'NOT_CHECKED': 216}
- A confirmed source transcription does not validate an implementation or hardware.

- Equation/concept completeness: NOT_ESTABLISHED; current model reviews with source fingerprints: 6/61.
- Explicitly incomplete equation transcriptions: 0 (quarantined from implementation edges).
- Docling processing coverage: {'canonical_documents': 248, 'source_pages': 11375, 'processed_pages': 2, 'remaining_pages': 11373, 'empty_sources': 1}; parser candidates require original-page review.

## Engineering evidence

- Active report snapshots by type: {'ArtifactManifest': 17, 'CADArtifact': 70, 'EngineConfiguration': 17, 'ManufacturingProcess': 15}
- CAD references with observed bytes: 64
- Configurations are serialized input/layout snapshots. CAD hashes describe observed current bytes; generation-time configuration binding remains unknown.
- Manufacturing records are reference plans. No measurement, experiment, calibration, or executed manufacturing is inferred from simulation outputs.

## Model registry ↔ source support (extraction pass)

- {'PARTIALLY_SUPPORTED': 10, 'NOT_FOUND_IN_SOURCE': 1, 'SUPPORTED': 3, 'NOT_CHECKED': 47}
  - `altitude.attached_thrust` → PARTIALLY_SUPPORTED: Hagemann et al. 1998 (Advanced Rocket Nozzles) is a qualitative review with no numbered equations. It does NOT state a quasi-one-dimensional attached-flow thrust-coefficient or nozzle momentum-balance relation anywhere;
  - `cad.kernel_topology` → NOT_FOUND_IN_SOURCE: CAD_04.md is a 173-slide lecture deck on B-Splines and NURBS (basis functions, divided differences, Cox-de Boor recursion, knot insertion/removal, rational curves and surfaces). The registry attributes to it 'OpenCascade
  - `chamber.shoulder_fill_fraction` → PARTIALLY_SUPPORTED: SP-125 (Huzel & Huang) contains NO chamber-shoulder fillet radius, no symbol Rs or Rs,max, and no 0.8 factor anywhere in the thrust-chamber geometry text (grep of 'shoulder' hits only fatigue/seal contexts; 'fillet' only
  - `flow.boundary_layer_displacement` → PARTIALLY_SUPPORTED: Original White PDF pp840-842 recovers Eq7.42 delta/x approx0.16 Re_x^(-1/7) and Eq7.46 delta_star approxdelta/8, implying0.02 x Re_x^(-1/7). This is a smooth flat-plate derivation using a one-seventh-power profile and co
  - `injector.compressible_orifice` → PARTIALLY_SUPPORTED: Original White PDFpp1092,1101 supplies ideal-gas sonic pressure ratio and choked/local mass-flow functions, matching the perfect-gas functions in injector._stream_mass_flux before its supplied Cd multiplier. White does n
  - `injector.d2_evaporation` → PARTIALLY_SUPPORTED: The registry attributes the d-squared law d^2(t) = d0^2 - K t to this document via 'evaporation-model context and Eq. 16'. The document's Eq. 16 is a Spalding-type mass-transfer rate, m˙_d = k_c A_d ρ ln(1 + B_m), not a
  - `injector.incompressible_orifice` → PARTIALLY_SUPPORTED: NASA SP-8089 (Liquid Rocket Engine Injectors, 1976) never writes the orifice mass-flux relation G = Cd*sqrt(2*rho*dP) or any algebraic orifice-flow equation; a genuine search for discharge coefficient, sqrt/square root,
  - `injector.momentum_resultant_angle` → PARTIALLY_SUPPORTED: The registry relation theta = atan2(M_radial*cos(delta), M_axial + M_radial*sin(delta)) does not appear anywhere in SP-8089. Grepped 'momentum', 'resultant', 'vector', 'direction', 'axial', 'radial', 'deflect', 'tan', 'c
  - `injector.n_tau_stability` → PARTIALLY_SUPPORTED: NASA SP-8089 (this document) contains no n-tau equation, no interaction-index or sensitive-time-lag values, and no sensitive-frequency formula. Its combustion-stability content (sec. 2.1.1.3, pp. 42-45; design criteria s
  - `performance.frozen_variable_cp_quasi1d` → PARTIALLY_SUPPORTED: Anderson originalPDFpp671-672,675-678,682 supports conserved energy/continuity and thermally-perfect fixed-composition sound speed. Chemically frozen composition permits variable speciescp(T); bothchemical AND vibrationa
  - `pump.electric_drive` → PARTIALLY_SUPPORTED: Lee PDF pp6-8 Eqs20,26-32 support the power and component-density relations. The Eq29 inverter-density convention mismatch was corrected on 2026-10-14 in NumPy and both MDO consumers: inverter mass now uses output power

## Extraction grounding

- grounded: 411, page-shifted: 0, ungrounded (dropped): 1
- equations: 93 (confirmed 58, refuted 0), claims: 188 (confirmed 7), validity statements from sources: 36

Dropped (anchor not found):
  - DOC_5F36B7C4DED79BB3E90754D0F81682F7A68014BE p671 [equation] Anderson Eq. (17.15) nozzle enthalpy-to-kinetic-energy relation

## Validation

- issues: 13 — {'empty_source_pdf': 1, 'canonical_document_without_chunks': 1, 'markdown_without_manifest_row': 1, 'registry_citation_not_confirmed_in_local_source': 1, 'declared_cad_bytes_unavailable': 6, 'source_extraction_incomplete': 1, 'source_verification_incomplete': 1, 'unknown_basis_software_constants': 1}
- stats: {"equation_variables": {"ambiguous": 6, "unresolved": 233, "total": 440}, "constants_by_basis": {"UNKNOWN_BASIS": 274, "IMPLEMENTATION_CHOICE": 351, "LITERATURE_DERIVED": 181, "PROJECT_DERIVED": 41, "REPOSITORY_HEURISTIC": 168}, "components_without_scientific_link": 200, "engineering_artifacts": {"active_by_type": {"ArtifactManifest": 17, "CADArtifact": 70, "EngineConfiguration": 17, "ManufacturingProcess": 15}, "inactive_snapshots": 0}, "graph": {"nodes": 4870, "edges": 6267, "edges_without_provenance": 0, "top_degree": [["MODEL_NOZZLE_RAO_VARIATIONAL_BVP", 146], ["DOC_NTRS_19760023196", 111], ["DOC_FLUID_MECHANICS_7TH_ED_MCGRAW_HILL_SERIES_IN_MECHANICAL_ENGI", 110], ["MODEL_NOZZLE_MOC", 97], ["MODEL_PUMP_ELECTRIC_DRIVE", 81]]}, "duplicate_chunk_texts": 15, "coverage": {"equation_completeness": "NOT_ESTABLISHED", "concept_completeness": "NOT_ESTABLISHED", "scope_note": "Coverage of stored source objects against manifest identities; live PDF/Markdown bytes are not rehashed here. No all-equation denominator or physical validation.", "documents": {"records": 252, "canonical": 248, "alternate_copies": 4, "with_source_objects": 9, "without_source_objects": 239, "with_equations": 9, "wi

## Explicitly unavailable / blocked

- {'BLOCKED_ON_LITERATURE': 35, 'UNSUPPORTED': 14, 'NOT_QUALIFIED': 9, 'NOT_IMPLEMENTED': 9, 'UNAVAILABLE': 33}

## Hard-coded constants by basis

- {'UNKNOWN_BASIS': 257, 'IMPLEMENTATION_CHOICE': 351, 'LITERATURE_DERIVED': 181, 'PROJECT_DERIVED': 41, 'REPOSITORY_HEURISTIC': 168}

UNKNOWN_BASIS constants (257), first 30:
  - raosim/pumps.py:47 `auto_current_target` = 250.0
  - raosim/mdo/schema.py:627 `eta_motor` = 0.9
  - raosim/pumps.py:53 `seal_face_speed_limit` = 35.0
  - raosim/electric_technology.py:298 `battery_structural_margin` = 1.2
  - raosim/electric_technology.py:296 `battery_power_density` = 3000.0
  - raosim/pumps.py:44 `battery_structural_margin` = 1.2
  - raosim/pumps.py:38 `inverter_power_density` = 15000.0
  - raosim/mdo/schema.py:640 `pump_specific_mass` = 2500.0
  - raosim/mdo/schema.py:628 `eta_inverter` = 0.96
  - raosim/pumps.py:51 `structural_fos` = 1.5
  - raosim/pumps.py:39 `battery_energy_density` = 900000.0
  - raosim/mdo/schema.py:641 `battery_energy_density` = 900000.0
  - raosim/electric_technology.py:299 `battery_discharge_efficiency` = 0.95
  - raosim/mdo/schema.py:610 `pump_speed_rpm` = 30000.0
  - raosim/mdo/schema.py:607 `pump_eta_peak` = 0.82
  - raosim/electric_technology.py:332 `inverter_efficiency` = 0.85
  - raosim/mdo/schema.py:608 `pump_ns_opt` = 0.55
  - raosim/electric_technology.py:297 `battery_energy_density` = 900000.0
  - raosim/pumps.py:41 `battery_discharge_efficiency` = 0.95
  - raosim/mdo/schema.py:612 `pump_tip_speed_max` = 400.0
  - raosim/electric_technology.py:303 `inverter_efficiency` = 0.96
  - raosim/electric_technology.py:302 `inverter_power_density` = 15000.0
  - raosim/mdo/schema.py:643 `battery_structural_margin` = 1.2
  - raosim/electric_technology.py:300 `motor_power_density` = 2500.0
  - raosim/pumps.py:36 `inverter_efficiency` = 0.96
  - raosim/electric_technology.py:301 `motor_efficiency` = 0.9
  - raosim/mdo/schema.py:642 `battery_power_density` = 3000.0
  - raosim/mdo/schema.py:609 `pump_eta_ns_width` = 1.5
  - raosim/mdo/schema.py:631 `inverter_power_density` = 15000.0
  - raosim/pumps.py:37 `motor_power_density` = 2500.0
