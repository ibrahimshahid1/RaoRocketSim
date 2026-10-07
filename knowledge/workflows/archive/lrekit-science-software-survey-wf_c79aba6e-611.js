export const meta = {
  name: 'lrekit-science-software-survey',
  description: 'Map LREKit subsystems to their physics, equations, constants, gates, tests and literature sources',
  phases: [
    { title: 'Subsystems', detail: 'one agent per LREKit subsystem: code -> equations, constants, gates, sources' },
    { title: 'Contracts', detail: 'docs precedence, tests/benchmarks, MDO constraints vs gates, unavailable capabilities' },
  ],
}

const REPO = '/Users/ibrahimshahid/Documents/Github/RaoRocketSim'

const COMPONENT_SCHEMA = {
  type: 'object',
  properties: {
    subsystem: { type: 'string' },
    components: {
      type: 'array',
      description: 'Scientifically consequential functions/classes. Only ones you actually read.',
      items: {
        type: 'object',
        properties: {
          qualname: { type: 'string', description: 'module.path:FunctionOrClass' },
          file: { type: 'string', description: 'repo-relative path' },
          line: { type: 'integer', description: '1-indexed def line' },
          role: { type: 'string', description: 'one sentence: what physical quantity it computes' },
          relations: { type: 'array', items: { type: 'string' }, description: 'the actual equations/relations implemented, in the notation used in the code' },
          inputs: { type: 'array', items: { type: 'string' } },
          outputs: { type: 'array', items: { type: 'string' } },
          model_registry_ids: { type: 'array', items: { type: 'string' }, description: 'raosim.model_registry ids this implements, if any' },
          cited_sources: { type: 'array', items: { type: 'string' }, description: 'literature cited in docstrings/comments, verbatim' },
          hardcoded_constants: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                value: { type: 'string' },
                units: { type: 'string' },
                line: { type: 'integer' },
                stated_basis: { type: 'string', description: 'verbatim comment/docstring justification, or empty string if none exists' },
                basis_class: { type: 'string', enum: ['LITERATURE_DERIVED', 'PROJECT_DERIVED', 'IMPLEMENTATION_CHOICE', 'EMPIRICALLY_CALIBRATED', 'REPOSITORY_HEURISTIC', 'UNKNOWN_BASIS'] },
              },
              required: ['name', 'value', 'line', 'basis_class'],
            },
          },
          validity_statements: { type: 'array', items: { type: 'string' }, description: 'verbatim validity/applicability text found in code or its registry entry' },
          refusals: {
            type: 'array',
            description: 'conditions under which this raises/returns unavailable/unknown',
            items: {
              type: 'object',
              properties: {
                condition: { type: 'string' },
                behavior: { type: 'string', enum: ['raise', 'return_unavailable', 'return_unknown', 'return_nan', 'clamp', 'warn_only'] },
                line: { type: 'integer' },
                message: { type: 'string' },
              },
              required: ['condition', 'behavior'],
            },
          },
          variant_of: { type: 'string', description: 'if this is one fidelity/backend variant (numpy vs jax, traditional vs mdo, TOP vs variational), name the sibling. Empty otherwise.' },
          tests: { type: 'array', items: { type: 'string' }, description: 'tests/*.py::test_name that exercise it' },
        },
        required: ['qualname', 'file', 'line', 'role'],
      },
    },
    quantities: {
      type: 'array',
      description: 'physical quantities this subsystem produces/consumes, with the symbol/variable names actually used in the code',
      items: {
        type: 'object',
        properties: {
          canonical: { type: 'string' },
          code_symbols: { type: 'array', items: { type: 'string' } },
          units: { type: 'string' },
        },
        required: ['canonical', 'code_symbols'],
      },
    },
    notes: { type: 'string', description: 'anything surprising, contradictory, or unverifiable. Say UNKNOWN where you could not confirm.' },
  },
  required: ['subsystem', 'components'],
}

const RULES = `
You are surveying the LREKit / RaoRocketSim repository at ${REPO}.
Work ONLY from files you actually read. Use Read/Grep/Glob/Bash.

HARD RULES:
- Never invent a file path, line number, function name, citation, constant, or validity range.
  Every 'file' + 'line' you report must be one you saw. If unsure of the line, read the file again.
- 'cited_sources' must be VERBATIM text from a docstring or comment. Do not add your own knowledge
  of what paper an equation comes from. If a Bartz-looking correlation has no citation in the code,
  cited_sources is empty and basis_class is UNKNOWN_BASIS.
- 'stated_basis' is verbatim repository text or the empty string. Never write your own justification there.
- Distinguish a hard-coded number that the code justifies from one it does not.
- A refusal is a real code path (raise / return None / return 'unavailable' / NaN / explicit gate), not a
  hypothetical. Report the actual message text where short.
- Prefer FEWER, verified entries over many plausible ones.

Useful context: raosim/model_registry.py holds a 61-entry provenance registry keyed by model_id with
fields source/local_source/equation_ref/validity/status/validation_level. Cross-reference it, and note
where code implements a registry entry and where a scientifically consequential model has NO registry entry.
`

const SUBSYSTEMS = [
  { key: 'nozzle-rao', prompt: `Rao / TOP / variational nozzle contour generation and the existence/validity machinery. Read raosim/rao_variational.py, raosim/rao_residuals.py, raosim/rao_optimizer.py, raosim/rao_existence_scan.py, raosim/nozzle_geometry.py, raosim/throat_geometry.py, raosim/conical.py, raosim/nozzle_comparison.py. Pay particular attention to: the difference between the Bezier/TOP approximation path and the variational (stationarity/transversality) path, what makes a Rao solution "physically valid" vs merely converged, and the D-attachment / domain machinery.` },
  { key: 'nozzle-moc', prompt: `Method of characteristics and transonic start line. Read raosim/moc.py, raosim/nasa_moc.py, raosim/moc_topology.py, raosim/transonic_kernel.py, raosim/moc_diagrams.py. Capture the characteristic relations actually coded, the topology audit criteria (including any recently reclassified crossing/axis-truncation rules), and the NumPy-vs-port parity story.` },
  { key: 'nozzle-jax', prompt: `JAX/differentiable nozzle kernels and their NumPy oracles. Read everything under raosim/jax/. For each JAX model, identify its NumPy reference implementation (REFERENCE_ORACLE_FOR) and the parity test that compares them. Note implicit-differentiation / custom_vjp usage and what is assumed for it to be valid.` },
  { key: 'thermal-cooling', prompt: `Regenerative cooling, heat transfer, wall model. Read raosim/thermal_design.py, raosim/wall_model.py, raosim/regen_geometry.py, raosim/regen_profile.py, raosim/coolants.py, raosim/thermofluids.py. Capture: gas-side (Bartz-type) heat flux, coolant-side convection, fin efficiency, CHF, radiation, wall temperature closure, coolant pressure drop. For every correlation, record whether the code states its validity range and whether it cites a source.` },
  { key: 'thermal-mdo', prompt: `The MDO/differentiable cooling path and how it differs from the traditional path. Read raosim/mdo/cooling.py plus whatever it imports, and compare against raosim/thermal_design.py. Explicitly enumerate where the two implementations differ physically (not just numerically): different correlations, different closures, different assumptions, different validity gates. This comparison is the deliverable.` },
  { key: 'structures-materials', quiet: true, prompt: `Chamber/liner structures, buckling, fatigue, materials. Read raosim/materials.py, raosim/manufacturing_limits.py and the structural parts of raosim/thermal_design.py / raosim/chamber_geometry.py. Capture liner stress relations, buckling and fatigue screens, factor-of-safety policy, and the material property catalog: for each material property, is there a cited source or is it an uncited number?` },
  { key: 'injector-pintle', prompt: `Injector and pintle hydraulics. Read raosim/injector.py, raosim/movable_pintle.py, raosim/spray_coupling.py. Capture discharge-coefficient handling (literature value vs user-calibrated vs default), TMR/momentum-ratio relations, skip distance / blockage geometry, stability screens, throttling schedule, and every gate that makes the injector model refuse.` },
  { key: 'spray', prompt: `Spray / atomization / evaporation modelling. Read raosim/spray/ (all files) and raosim/spray_coupling.py. There are ~16 model_registry entries under subsystem 'spray'; map each to code. Be explicit about which parts are one-way couplings, report-only, or optional-unvalidated.` },
  { key: 'pumps-feed', prompt: `Pump meanline, inducer, NPSH, electric drive, battery, feed system. Read raosim/pumps.py, raosim/pump_cad.py, raosim/pump_cad_brep.py. Capture head/flow coefficients, specific speed, NPSH criteria, blade angle constraints, efficiency models, motor/battery sizing. Distinguish hydraulic design outputs from reference geometry.` },
  { key: 'performance-gasdyn', prompt: `Thermochemistry, quasi-1D gas dynamics, nozzle performance, altitude/separation. Read raosim/cea.py, raosim/gas_dynamics.py, raosim/physics.py, raosim/frozen_flow.py, raosim/altitude_performance.py, raosim/separation.py, raosim/wall_pressure.py, raosim/atmosphere.py, raosim/propellants.py. Capture c*, CF, Isp, eta_cstar / eta_CF split, frozen-vs-equilibrium handling, separation criteria.` },
  { key: 'cad-manufacturing', prompt: `CAD/B-rep generation and manufacturability. Read raosim/engine_cad.py, raosim/injector_cad.py, raosim/injector_coaxial_cad.py, raosim/regen_cad.py, raosim/flange_cad.py, raosim/export.py, raosim/manufacturing_limits.py. The deliverable: what does the code actually verify about a generated solid (topology? watertightness? tool access? tolerance?) and what does it NOT verify. Record every place a CAD artifact is bound (or not bound) to a resolved engine configuration.` },
  { key: 'mdo-core', prompt: `The MDO problem formulation. Read raosim/mdo/ (nlp.py, job.py, resolved_inputs.py, capabilities.py, contour_validation.py and siblings). The deliverable: an exact list of (a) quantities enforced as optimizer constraints inside the solve, (b) quantities checked only as postsolve gates, (c) quantities that are report-only diagnostics, (d) capabilities that are explicitly UNAVAILABLE/NOT_IMPLEMENTED. Quote the code that establishes each classification.` },
  { key: 'requirements-gates', prompt: `Requirements, validation, release readiness. Read raosim/requirements.py, raosim/validation.py, raosim/release_readiness.py, raosim/model_registry.py, raosim/benchmarks.py, raosim/build_log.py. Capture the requirement objects, the gate taxonomy, the pass/fail/unknown/unavailable vocabulary the codebase actually uses, and the audit rules in audit_model_registry.` },
  { key: 'engine-integration', quiet: true, prompt: `End-to-end engine design chain and configuration identity. Read raosim/engine.py, raosim/design.py, raosim/interface.py, raosim/mass_ledger.py, raosim/trade_study.py, raosim/trajectory.py, raosim/input_validation.py, raosim/units.py, lrekit/cli.py, lrekit/job_cli.py, raosim/cli.py. Capture the configuration/fingerprint machinery, the sequential design path, and how artifacts are bound to configurations.` },
]

phase('Subsystems')
const surveys = await parallel(SUBSYSTEMS.map(s => () =>
  agent(`${RULES}\n\nYOUR SUBSYSTEM: ${s.key}\n\n${s.prompt}\n\nReturn the structured survey.`,
    { label: `survey:${s.key}`, phase: 'Subsystems', schema: COMPONENT_SCHEMA })
))

const CONTRACT_SCHEMA = {
  type: 'object',
  properties: {
    topic: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          kind: { type: 'string' },
          title: { type: 'string' },
          detail: { type: 'string' },
          evidence: { type: 'array', items: { type: 'string' }, description: 'repo-relative file:line or file:heading you actually read' },
          quote: { type: 'string', description: 'short verbatim excerpt supporting the finding' },
          status: { type: 'string', description: 'e.g. LANDED, PLANNED_NOT_IMPLEMENTED, SUPERSEDED, BLOCKED_ON_LITERATURE, UNKNOWN' },
          relations: { type: 'array', items: { type: 'string' }, description: 'e.g. "SUPERSEDES docs/X.md", "GATED_BY moc_closure"' },
        },
        required: ['id', 'kind', 'title', 'detail', 'evidence'],
      },
    },
    unresolved: { type: 'array', items: { type: 'string' }, description: 'disagreements you could NOT resolve; preserve them rather than picking one' },
  },
  required: ['topic', 'findings'],
}

phase('Contracts')
const CONTRACTS = [
  { key: 'docs-precedence', prompt: `Build the document precedence and staleness map for ${REPO}/docs/ plus README.md. There are ~65 documents accumulated over time. For EACH document record: its date (from filename or content), its type (plan / audit / implementation review / literature request / provenance registry / roadmap), its STATUS head if it has one, and its relations to other documents (SUPERSEDES, CORRECTS, UPDATES, AUDITS, IMPLEMENTS_PLAN_ITEM, DEPRECATES, BLOCKS). Detect supersession/staleness language explicitly. Where two documents disagree about whether something is implemented, DO NOT pick a winner - record the disagreement in 'unresolved' with both citations. Note especially which document is the current delivery index. The newest documents are dated 2026-09-05 and 2026-09-09.` },
  { key: 'tests-benchmarks', prompt: `Inventory verification artifacts under ${REPO}/tests/ (~125 files) and ${REPO}/raosim/benchmark_data/ and raosim/benchmarks.py. Classify EVERY meaningful test file into exactly one of: unit test, numerical regression, NumPy<->JAX parity, autodiff<->finite-difference, published benchmark reproduction (name the published case and source), CAD round-trip, model-vs-model comparison, experimental validation, snapshot/contract test. Critically: identify which tests are PUBLISHED BENCHMARK REPRODUCTIONS with a named literature source and expected value/tolerance - those are the highest-value knowledge objects. Do not label a software regression as experimental validation.` },
  { key: 'unavailable-capabilities', prompt: `Enumerate every explicitly UNAVAILABLE / NOT_IMPLEMENTED / BLOCKED / REFUSED capability in LREKit. Grep the whole repo (raosim/, lrekit/, scripts/) for raise NotImplementedError, 'unavailable', 'not implemented', 'unsupported', 'refuse', 'BLOCKED_ON', 'PLANNED', capability gates, and the capabilities module raosim/mdo/capabilities.py. For each: what is unavailable, what code path enforces it, and what the repository says is required to unblock it (literature? standard? test data?). Also read docs/LITERATURE_REQUESTS.md, docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md, docs/LITERATURE_REQUESTS_GENERALIZATION.md and record each outstanding literature request as a finding with status BLOCKED_ON_LITERATURE.` },
  { key: 'literature-links', prompt: `Harvest EVERY literature citation that appears in LREKit source code and docs, and map it to a file in ${REPO}/propulsion_texts/ where possible. Grep raosim/ and docs/ for 'propulsion_texts/', 'NASA SP-', 'NASA CR-', 'NASA TN', 'doi', 'et al', bracketed reference markers, and author-year patterns in docstrings. For each citation record: the citing file:line, the verbatim citation text, and the resolved corpus path if one exists (check the path actually exists on disk). Mark UNRESOLVED where a citation names a source that is NOT in propulsion_texts/. Also read docs/MODEL_REGISTRY.md, docs/REMEDIATION_LITERATURE_TRACEABILITY.md, docs/thermofluid_literature_provenance.md, docs/injector_pintle_provenance.md, docs/nasa_tt_prime_provenance.md, docs/chamber_design_literature.md, docs/pintle_injector_literature_inventory.md and propulsion_texts/pintle_injector/PINTLE_LITERATURE_CATALOG.md.` },
]

const contracts = await parallel(CONTRACTS.map(c => () =>
  agent(`${RULES}\n\nTOPIC: ${c.key}\n\n${c.prompt}`,
    { label: `contract:${c.key}`, phase: 'Contracts', schema: CONTRACT_SCHEMA })
))

return {
  surveys: surveys.filter(Boolean),
  contracts: contracts.filter(Boolean),
  survey_failures: SUBSYSTEMS.filter((_, i) => !surveys[i]).map(s => s.key),
  contract_failures: CONTRACTS.filter((_, i) => !contracts[i]).map(c => c.key),
}
