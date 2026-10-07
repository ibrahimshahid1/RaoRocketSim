# LREKit SCIENTIFIC KNOWLEDGE SYSTEM — MASTER SYSTEM INSTRUCTIONS

You are operating inside the **LREKit / RaoRocketSim repository**.

Your task is to build, maintain, query, and use a **persistent, provenance-aware scientific and engineering knowledge system specifically for LREKit**.

This is not a generic scientific knowledge graph project.

This system exists to support the development, auditing, verification, extension, and scientific grounding of **LREKit**, a liquid-rocket-engine preliminary design, simulation, optimization, CAD, hardware-development, and research codebase.

The knowledge system must behave as **local scientific memory for LREKit development agents**.

Its purpose is to let an agent move reliably from:

task
→ affected LREKit subsystem
→ relevant concepts
→ equations / methods / models
→ assumptions and validity limits
→ claims and competing formulations
→ original literature
→ existing LREKit implementation
→ tests / benchmarks / gates
→ implementation

The system is not merely an embedding database and is not merely a PDF summarizer.

It must combine:

1. a document and corpus manifest,
2. semantic/vector retrieval,
3. a structured LREKit propulsion knowledge graph,
4. explicit claims,
5. equations as first-class objects,
6. model and method records,
7. source-level provenance,
8. applicability and validity information,
9. mappings between literature and LREKit software components,
10. mappings between models and their tests/benchmarks/gates,
11. task-aware retrieval for coding agents,
12. incremental PDF-to-Markdown ingestion,
13. persistent documentation explaining how future LREKit agents should use the system.

---

# 1. UNDERSTAND WHAT LREKIT IS BEFORE BUILDING ANYTHING

Do not treat LREKit as merely a Rao nozzle generator.

Inspect the repository before designing the knowledge system.

LREKit may include or interact with, among other things:

* liquid-rocket-engine preliminary design,
* requirements and mission inputs,
* thermochemistry,
* quasi-1-D gas dynamics,
* nozzle performance,
* Rao/TOP nozzle geometry,
* axisymmetric method of characteristics,
* variational nozzle optimization,
* regenerative cooling,
* heat transfer,
* coolant pressure losses,
* chamber and wall structures,
* buckling and structural screens,
* injector hydraulics,
* pintle injectors,
* movable injectors,
* spray models,
* feed systems,
* centrifugal pump meanline design,
* inducers,
* cavitation and NPSH,
* electric motors,
* batteries,
* mass estimation,
* engine configuration identity,
* optimization,
* differentiable physics,
* implicit differentiation,
* JAX implementations,
* NumPy reference/oracle implementations,
* multipoint operation,
* model-validity envelopes,
* requirements coverage,
* CAD/B-rep generation,
* STEP/STL output,
* manufacturability,
* machining/process constraints,
* inspection,
* experimental measurements,
* model comparison,
* calibration,
* validation,
* benchmark reproduction,
* release/readiness gates,
* refusal and unavailable-state behavior.

The repository contains both a **traditional/sequential engine-design path** and a **coupled differentiable MDO path**.

Do not assume that two implementations of a supposedly similar quantity are scientifically or numerically identical.

Represent the relationship between implementations explicitly.

Examples:

TraditionalCoolingModel
COMPARED_WITH
MDOCoolingModel

NumPyModel
REFERENCE_ORACLE_FOR
JAXModel

RaoVariationalSolver
COMPARED_WITH
TOPApproximation

RocketCEA
PROVIDES_REFERENCE_DATA_FOR
PropertySurface

External dependencies such as JAX, SciPy, Optimistix, CadQuery/OpenCascade, CoolProp, and RocketCEA perform numerical, differentiation, geometry, property, or thermochemistry services.

Do **not** attribute LREKit's engineering models, residual formulations, validity gates, geometry architecture, or acceptance criteria to these dependencies merely because those libraries execute part of the computation.

---

# 2. FIRST INSPECT THE REPOSITORY

Before implementing the knowledge system:

1. recursively inspect the repository,
2. identify the current package structure,
3. identify the literature directories,
4. identify agent-readable Markdown literature,
5. identify project documentation,
6. identify model registries,
7. identify requirements registries,
8. identify validation and benchmark artifacts,
9. identify generated knowledge-system artifacts if they already exist,
10. identify any existing ingestion scripts or retrieval infrastructure.

Locate and read human-authored metadata and project documentation before interpreting the source corpus.

Relevant material may include:

* `README.md`
* `AGENTS.md`
* `knowledge/AGENTS.md`
* `docs/`
* completion plans
* integration plans
* hardware plans
* workflow audits
* project audits
* physics-fidelity roadmaps
* literature-request documents
* model provenance registries
* benchmark manifests
* validation documents
* source indexes
* paper indexes
* literature catalogs
* metadata files
* `for_agents/`
* `agents/`
* similar repository-specific directories discovered during inspection.

Do not ignore these documents.

They contain project intent, terminology, known limitations, implementation state, sequencing, supersession notices, acceptance gates, and relationships between the literature and the code.

However:

**Project documentation is not automatically scientific authority.**

A planning document may describe:

* what the project intends to implement,
* what is currently implemented,
* a repository engineering decision,
* a scientific claim backed by literature,
* an interpretation of literature,
* a provisional assumption,
* or a stale implementation state.

Distinguish these.

---

# 3. HANDLE DOCUMENT PRECEDENCE AND STALENESS

LREKit has accumulated audits, plans, implementation reviews, and correction documents over time.

Do not assume that every Markdown file is simultaneously current.

Detect language such as:

* supersedes,
* corrected by,
* implementation review,
* current delivery index,
* stale,
* landed,
* deferred,
* blocked,
* no longer true,
* use X instead,
* retained only as historical/workstream context.

Represent document relationships such as:

SUPERSEDES
CORRECTS
UPDATES
IMPLEMENTS_PLAN_ITEM
AUDITS
VALIDATES
BLOCKS
DEPRECATES

When two repository documents disagree about implementation state:

1. inspect their dates,
2. inspect explicit supersession statements,
3. inspect current code,
4. inspect current tests and generated artifacts where relevant,
5. preserve the disagreement if it cannot be resolved.

Do not silently choose whichever document was retrieved first.

Scientific authority and implementation-state authority are different things.

A 1970 NASA monograph may remain the authority for a physical model while a September 2026 audit is the authority for whether that model is currently implemented.

---

# 4. BUILD THE DOCUMENT MANIFEST FIRST

Before full ingestion, build or update a document manifest.

Every source document should contain, where available:

* `document_id`
* filename
* canonical title
* authors
* organization
* publication year/date
* DOI
* NASA/NTRS accession number
* report number
* standard identifier
* document type
* propulsion topics
* source PDF path
* generated Markdown path
* associated human-authored metadata
* page count
* checksum/hash
* PDF parsing status
* Markdown conversion status
* extraction status
* extraction version
* ingestion timestamp
* source authority/type
* known limitations
* duplicate/alternate-copy relationships.

Do not confidently invent bibliographic metadata.

When metadata cannot be verified, preserve it as unknown or low-confidence.

Detect duplicate PDFs and alternate copies by hashes and bibliographic identity rather than assigning them independent scientific authority merely because filenames differ.

---

# 5. THE PDF IS THE AUTHORITATIVE SCIENTIFIC SOURCE

Generated Markdown exists to make the literature efficiently readable by agents.

The hierarchy is:

original PDF
→ generated agent-readable Markdown
→ extracted chunks/objects
→ knowledge graph / vector index
→ task context

The Markdown derivative is **not** an independent scientific publication.

Always preserve the relationship:

GeneratedMarkdown
DERIVED_FROM
SourcePDF

Important scientific decisions must remain traceable to the PDF.

If Markdown extraction is ambiguous, malformed, incomplete, or appears inconsistent with the original, inspect the PDF.

Never prefer a parser artifact over the source document.

---

# 6. MANDATORY PDF → MARKDOWN SYNCHRONIZATION

The repository's propulsion literature can grow over time.

The directory:

`propulsion_texts_for_agents/`

is the agent-readable Markdown representation of the LREKit propulsion literature corpus.

Before building, rebuilding, substantially querying, or relying on the scientific knowledge system for a scientifically consequential implementation task, check whether the corpus is synchronized.

Recursively inspect the relevant source-PDF locations.

Determine whether every appropriate source PDF has a corresponding generated Markdown representation in:

`propulsion_texts_for_agents/`

Use the strongest available mapping evidence:

* manifest association,
* PDF checksum,
* source identifier,
* DOI,
* NTRS accession,
* source path,
* conversion metadata,
* explicit source-PDF field,
* filename only when stronger evidence is unavailable.

Maintain an explicit:

`source PDF -> generated Markdown`

mapping.

Do not rely solely on filename similarity.

## If a new PDF has no Markdown representation

You MUST:

1. identify it as a newly discovered source,
2. convert/extract it into Markdown,
3. preserve document structure as faithfully as practical,
4. place the generated Markdown in `propulsion_texts_for_agents/`,
5. add the generated Markdown to the repository,
6. register the PDF↔Markdown relationship,
7. ingest the document,
8. update all affected scientific-memory artifacts.

Preserve, when recoverable:

* title,
* authors,
* publication metadata,
* page boundaries,
* headings,
* sections,
* subsections,
* paragraphs,
* equations,
* equation numbers,
* symbols,
* tables,
* table captions,
* figures and captions,
* footnotes when scientifically important,
* references,
* source-page information.

Record machine-readable conversion metadata where practical:

* source PDF path,
* source PDF hash,
* generated Markdown path,
* converter/parser name,
* converter/parser version,
* conversion timestamp,
* conversion status.

## Never overwrite human-authored Markdown

Generated conversions and human-authored metadata are different classes of object.

Do not overwrite:

* README files,
* agent instructions,
* literature indexes,
* manually written summaries,
* planning documents,
* audits,
* metadata,
* literature catalogs,
* project notes,

simply because a filename resembles a PDF basename.

Generated Markdown must be explicitly distinguishable from human-authored Markdown.

---

# 7. REFRESH THE KNOWLEDGE SYSTEM AFTER CORPUS CHANGES

Whenever PDFs are added or modified, synchronize the knowledge system.

At minimum:

1. update the manifest,
2. register the source PDF,
3. register/update its Markdown derivative,
4. semantically chunk the new content,
5. update vector retrieval,
6. extract entities,
7. extract physical quantities,
8. extract variables,
9. extract equations,
10. extract claims,
11. extract assumptions,
12. extract applicability conditions,
13. extract validity limits,
14. extract methods and algorithms,
15. extract experimental or benchmark information,
16. generate provenance records,
17. resolve entities against the existing graph,
18. update affected relationships,
19. update conflicting/equivalent formulation records,
20. update document/extraction state,
21. update the graph,
22. run validation,
23. regenerate the ingestion report,
24. update `knowledge/AGENTS.md` if capabilities or corpus state changed.

The resulting system must reflect the **complete currently ingested LREKit corpus**.

Do not leave a newly converted paper sitting in the repository while absent from retrieval or graph traversal.

---

# 8. PREFER INCREMENTAL RECONSTRUCTION

"Recreate the knowledge graph" does **not** mean destroying every artifact whenever one paper changes.

Prefer:

new/modified PDF
→ conversion
→ changed-document ingestion
→ scientific-object extraction
→ entity reconciliation
→ affected cross-document relationship recomputation
→ vector update
→ graph update
→ validation

Unchanged sources should not be repeatedly reparsed without reason.

However, incremental ingestion must produce the same logically integrated knowledge state expected from the complete corpus.

Do not merely append isolated nodes for the new paper.

A new paper may alter:

* aliases,
* entity normalization,
* equation equivalence,
* conflicting formulations,
* validity envelopes,
* literature support,
* model relationships,
* citations,
* provenance links,
* cross-document concepts.

Update existing records when necessary.

---

# 9. HANDLE MODIFIED SOURCES CORRECTLY

If a previously ingested PDF changes:

1. detect the checksum change,
2. determine whether generated Markdown is stale,
3. regenerate it when required,
4. version or supersede the previous extraction state,
5. invalidate claims/equations that depended exclusively on obsolete source text,
6. re-extract affected scientific objects,
7. recompute relevant entity resolution,
8. recompute relevant graph relationships,
9. update vector retrieval,
10. validate provenance.

Never retain a claim that points to source text that no longer exists.

---

# 10. FAILURE MUST REMAIN VISIBLE

If PDF conversion fails:

do not invent the missing content.

Record states such as:

`PDF_CONVERSION_FAILED`

`PARTIAL_CONVERSION`

`SOURCE_NOT_INGESTED`

`LOW_EXTRACTION_CONFIDENCE`

Include:

* affected source,
* failing stage,
* diagnostic information,
* whether partial text exists,
* confidence,
* recommended escalation.

Preserve the PDF.

The same philosophy applies throughout the LREKit knowledge system:

**missing information stays missing.**

Do not turn absence into a plausible engineering estimate merely to complete the graph.

---

# 11. LREKIT-SPECIFIC ONTOLOGY

Design the ontology around LREKit rather than around arbitrary scientific domains.

Core object types should include, where appropriate:

## Source and provenance objects

Document
GeneratedDocument
Section
Figure
Table
Citation
SourceReference
Claim
ProvenanceRecord

## Scientific objects

Concept
PhysicalQuantity
Variable
Equation
Correlation
Model
Method
Algorithm
Procedure
Assumption
BoundaryCondition
InitialCondition
Constraint
Requirement
ValidityEnvelope
Recommendation
EmpiricalConstant
MaterialProperty
FailureMode

## LREKit software objects

SoftwareComponent
Module
Function
Class
CLICommand
JobSchema
ModelRegistryEntry
ConstraintRegistryEntry
Capability
Test
Benchmark
ValidationGate
ArtifactManifest

## Engine-system objects

RocketEngine
EngineConfiguration
OperatingPoint
Mission
PropellantPair
Oxidizer
Fuel

CombustionChamber
ChamberLiner
Jacket
Rib
CoolingChannel

Nozzle
NozzleContour
Throat
ExitPlane
ExpansionSection

Injector
Pintle
Orifice
Annulus
Spray
SprayParcel

FeedSystem
Pump
Inducer
Impeller
Diffuser
Volute
Collector
Shaft
Bearing
Seal
Valve
Line

Motor
Battery

CADAssembly
CADPart
Geometry
ManufacturingProcess
InspectionFeature
Datum
Tolerance

Measurement
Experiment
TestArticle
Calibration
CalibratedCoefficient
ComparisonRecord

Do not create every possible type merely to maximize graph complexity.

Create types because they enable scientifically useful retrieval.

---

# 12. IMPORTANT LREKIT PHYSICAL QUANTITIES

Recognize and normalize quantities including, where relevant:

* thrust
* specific impulse
* mass flow
* oxidizer mass flow
* fuel mass flow
* mixture ratio
* chamber pressure
* ambient pressure
* exit pressure
* stagnation pressure
* chamber temperature
* wall temperature
* coolant temperature
* heat flux
* heat-transfer coefficient
* pressure drop
* expansion ratio
* contraction ratio
* throat area
* throat radius
* exit area
* exit radius
* Mach number
* speed of sound
* gamma / specific-heat ratio
* characteristic velocity
* thrust coefficient
* discharge coefficient
* Reynolds number
* Prandtl number
* Nusselt number
* viscosity
* density
* conductivity
* stress
* strain
* wall thickness
* buckling margin
* factor of safety
* fatigue life
* head
* pump pressure rise
* flow coefficient
* head coefficient
* specific speed
* rotational speed
* tip speed
* NPSH
* cavitation number
* efficiency
* shaft power
* motor power
* battery energy
* component mass
* engine dry mass.

Preserve units.

Preserve dimensional meaning.

Never merge two quantities because their symbols match.

---

# 13. ENTITY AND SYMBOL NORMALIZATION

Scientific notation in propulsion literature is not globally consistent.

Store:

* canonical name,
* aliases,
* symbols,
* alternate symbols,
* abbreviations,
* units/dimensionality when applicable,
* source-specific definitions,
* subsystem context,
* external identifiers where appropriate.

Examples:

specific impulse
`Isp`
`I_sp`

chamber pressure
`Pc`
`p_c`
`P_c`

mass flow
`ṁ`
`mdot`

expansion ratio
`ε`
`A_e/A_t`

specific-heat ratio
`γ`
`k`

Do not assume that every `k`, `R`, `T`, `P`, `M`, `A`, or `C_d` means the same thing.

Symbol identity must depend on:

* document,
* section,
* equation,
* definition,
* dimensions,
* context.

When uncertain, preserve multiple entities and mark:

`POSSIBLY_EQUIVALENT`

with confidence rather than performing a destructive merge.

---

# 14. EQUATIONS ARE FIRST-CLASS OBJECTS

Equations are among the most important objects in the LREKit knowledge system.

For every important equation attempt to preserve:

* `equation_id`
* source document
* source page
* equation number/label
* exact or normalized symbolic form
* nearby explanatory text
* variables
* variable definitions
* units/dimensions
* inputs
* outputs
* assumptions
* approximations
* boundary conditions
* validity range
* required material/property data
* dependent equations
* method/procedure using it
* associated model
* associated LREKit implementation if one exists
* tests/benchmarks if one exists
* extraction confidence
* interpretation confidence.

Possible relationships:

Equation
INPUT_VARIABLE
ChamberPressure

Equation
OUTPUT_VARIABLE
ThroatMassFlux

Equation
DERIVED_FROM
GoverningRelation

Equation
APPLICABLE_WHEN
IdealIsentropicFlow

Equation
IMPLEMENTED_BY
`raosim/...`

Equation
VALIDATED_BY
Benchmark

Equation
FOUND_IN
SourceDocument

Do not silently "correct" an equation because another source gives a different form.

Store competing formulations and determine whether they are:

* algebraically equivalent,
* notation variants,
* approximations,
* different closure models,
* applicable under different assumptions,
* or genuinely contradictory.

---

# 15. CLAIMS ARE FIRST-CLASS OBJECTS

Do not turn arbitrary sentences directly into graph edges.

Represent meaningful scientific assertions as claims.

A Claim should contain:

* `claim_id`
* subject
* predicate
* object/value
* qualifiers
* claim type
* source
* page
* section
* supporting text reference
* assumptions
* applicability
* confidence
* extraction version.

Useful claim types include:

* definition
* theoretical
* empirical
* methodological
* comparative
* assumption
* constraint
* recommendation
* validity statement
* design criterion
* derived result
* implementation statement
* benchmark result.

Conflicting claims must coexist.

Do not collapse them into a synthetic "truth."

Represent:

ClaimA
CONTRADICTS
ClaimB

or:

ClaimA
COMPETING_FORMULATION_WITH
ClaimB

with each claim retaining its own provenance.

---

# 16. LREKIT-SPECIFIC RELATIONSHIPS

Useful graph relationships may include:

DEFINES
DESCRIBES
USES
DERIVES
DEPENDS_ON
AFFECTS
CONSTRAINS
ASSUMES
MEASURES
PREDICTS
ESTIMATES
VALIDATES
INVALIDATES
CONTRADICTS
SUPPORTS
EXTENDS
IMPLEMENTS
COMPARED_WITH
PART_OF
INPUT_TO
OUTPUT_OF
PARAMETER_OF
DERIVED_FROM
CITES
EQUIVALENT_TO
POSSIBLY_EQUIVALENT
APPROXIMATES
APPLICABLE_WHEN
HAS_VALIDITY_ENVELOPE
REFUSES_WHEN
GATED_BY
BENCHMARKED_AGAINST
REFERENCE_ORACLE_FOR
PARITY_CHECKED_AGAINST
PRODUCES_ARTIFACT
BOUND_TO_CONFIGURATION
CALIBRATED_FROM
MEASURED_AT
TESTED_WITH
SUPERSEDES
CORRECTS
PLANNED_BY
IMPLEMENTED_IN
AVAILABLE_IN
UNAVAILABLE_IN

Do not force every relationship into this list.

Version the ontology/schema.

---

# 17. MODEL VARIANTS MUST NOT BE FLATTENED

LREKit often contains several implementations or fidelity levels for the same broad physical phenomenon.

Do not merge these into one generic node.

Examples may include:

* TOP/Rao approximation versus variational Rao solution,
* MOC variants,
* NumPy oracle versus JAX implementation,
* traditional cooling versus differentiable cooling,
* simple screens versus higher-fidelity verification,
* frozen versus equilibrium chemistry,
* low-order boundary-layer screens versus future PNS validation,
* reference geometry versus process-qualified geometry.

Record:

* model name,
* implementation,
* fidelity,
* assumptions,
* differentiability,
* validity,
* provenance,
* current promotion status,
* benchmark status,
* whether it is used in optimization,
* whether it is postsolve only,
* whether it is report-only,
* whether it is unavailable.

A higher-fidelity model is not automatically the production model.

A planned model is not an implemented model.

A report-only screen is not an optimizer constraint.

A numerical parity test is not physical validation.

---

# 18. DISTINGUISH SCIENCE FROM IMPLEMENTATION CHOICES

Every important implementation decision should be categorized where useful as:

`LITERATURE_DERIVED`

`PROJECT_DERIVED`

`IMPLEMENTATION_CHOICE`

`EMPIRICALLY_CALIBRATED`

`REPOSITORY_HEURISTIC`

`UNKNOWN_BASIS`

For example:

A governing heat-transfer correlation may be `LITERATURE_DERIVED`.

The decision to cache property tables may be `IMPLEMENTATION_CHOICE`.

A model coefficient fitted from a particular cold-flow test may be `EMPIRICALLY_CALIBRATED`.

A hard-coded value whose scientific basis cannot be found must not be magically upgraded to a literature-backed limit.

Represent it honestly.

---

# 19. NO INVENTED AUTHORITY

This is a critical LREKit rule.

If an engineering correlation, validity range, material property, design criterion, standard, coefficient, or model required for implementation is not supported by the available corpus or another explicitly authorized source:

**do not invent it from memory.**

Instead:

* record the missing basis,
* expose the model/feature as unavailable where appropriate,
* create or update a literature request,
* identify what evidence is needed.

Never convert:

"I know a common aerospace rule of thumb"

into:

"LREKit's validated engineering criterion."

A model gap must remain visible.

---

# 20. `UNKNOWN` IS A VALID ENGINEERING RESULT

LREKit intentionally uses refusal, unavailable states, and explicit gaps.

The knowledge system must understand and preserve distinctions among:

* PASS
* FAIL
* UNKNOWN
* UNAVAILABLE
* NOT_IMPLEMENTED
* OUTSIDE_VALIDITY
* BLOCKED_ON_LITERATURE
* BLOCKED_ON_STANDARD
* NOT_QUALIFIED
* REPORT_ONLY
* POSTSOLVE_ONLY
* CONVERGED_BUT_INVALID
* NUMERICALLY_FAILED.

Do not convert an unavailable quantity into a plausible estimate simply to make downstream logic convenient.

Do not confuse:

software success

with:

scientific validity

or:

hardware qualification.

Likewise:

a generated report package may be complete while its scientific verdict is FAIL or UNKNOWN.

A solver may converge while the physical model is outside its validity envelope.

A CAD file may exist while its manufacturability is unverified.

A numerical model may pass parity tests while lacking experimental validation.

Represent these independently.

---

# 21. REPRESENT THE REFUSAL ARCHITECTURE

Refusal behavior is part of LREKit's engineering semantics.

Capture conditions such as:

model
REFUSES_WHEN
outside_validity_envelope

export
GATED_BY
MOC_closure

artifact
REFUSES_WHEN
configuration_identity_mismatch

multipoint_request
UNAVAILABLE_IN
single_point_implementation

capability
BLOCKED_ON_LITERATURE
required_source

Do not treat a refusal as merely a software exception.

Where the refusal encodes scientific honesty, it is part of the model contract.

---

# 22. REPRESENT VALIDITY ENVELOPES EXPLICITLY

Any model or correlation should retain its applicability information.

Where possible store machine-readable fields for:

* independent variables,
* lower bounds,
* upper bounds,
* propellants,
* materials,
* phase assumptions,
* geometry assumptions,
* flow regime,
* pressure range,
* temperature range,
* Reynolds-number range,
* Mach-number range,
* mixture-ratio range,
* geometry ratios,
* model fidelity,
* required evidence.

Free-form prose validity statements may be preserved, but machine-evaluable validity should be extracted when scientifically justified.

Never infer a numerical validity boundary merely because it would be convenient.

---

# 23. REQUIREMENTS, CONSTRAINTS, AND GATES ARE DISTINCT

Do not flatten all restrictions into one "constraint" type.

Distinguish:

## Requirement

What the requested engine or system must satisfy.

## Optimizer constraint

A quantity actively enforced within an optimization solve.

## Postsolve gate

A required check evaluated after solving.

## Diagnostic

A reported quantity that does not itself determine success.

## Applicability gate

Determines whether a model is allowed to make the prediction.

## Evidence/qualification gate

Determines whether an engineering claim has sufficient supporting evidence.

## Manufacturing gate

Determines whether an artifact/process meets declared manufacturing criteria.

Represent the relationship between them.

A required quantity being checked only postsolve is not equivalent to being enforced by the optimizer.

---

# 24. REPRESENT SOFTWARE ↔ LITERATURE TRACEABILITY

The knowledge graph must connect scientific sources to the actual LREKit implementation.

For scientifically consequential functions/classes/modules record, where practical:

SoftwareComponent
IMPLEMENTS
Equation

SoftwareComponent
IMPLEMENTS
Model

SoftwareComponent
USES
Correlation

SoftwareComponent
GATED_BY
ValidityCriterion

SoftwareComponent
BENCHMARKED_AGAINST
ReferenceCase

Model
SUPPORTED_BY
SourceDocument

Model
APPROXIMATES
HigherFidelityModel

This lets an agent ask:

"What literature supports this function?"

"What implementation uses this equation?"

"Which model does this constraint come from?"

"Which tests validate this implementation?"

"Which source defines this validity range?"

---

# 25. BENCHMARKS AND TESTS ARE KNOWLEDGE OBJECTS

Represent important verification artifacts.

A Test or Benchmark may contain:

* identifier
* subsystem
* source reference
* expected quantity
* expected value/range
* tolerance
* units
* implementation tested
* backend
* configuration
* result
* date/version
* artifact hash
* scientific interpretation.

Distinguish:

unit test
numerical regression
NumPy↔JAX parity
autodiff↔finite-difference comparison
published benchmark reproduction
CAD round-trip test
model-vs-model comparison
experimental validation
qualification testing.

Do not infer physical validation from software regression coverage.

---

# 26. REPRESENT CONFIGURATION AND ARTIFACT IDENTITY

LREKit increasingly depends on binding outputs, geometry, models, and measurements to the exact engine configuration that generated them.

The knowledge system should understand entities such as:

ResolvedEngineConfiguration
ConfigurationFingerprint
ArtifactIdentity
GeometryFingerprint
ModelVersion
CoefficientVersion
MeasurementRecord

and relationships such as:

Artifact
BOUND_TO_CONFIGURATION
EngineConfiguration

Measurement
MEASURED_FROM
TestArticle

TestArticle
INSTANCE_OF
EngineConfiguration

Calibration
APPLIES_TO
GeometryFingerprint

Do not associate measurements, CAD, mass, or calibrated coefficients with a different configuration merely because names appear similar.

---

# 27. MANUFACTURABILITY AND TESTABILITY ARE NOT JUST CAD

Do not reduce hardware-development knowledge to geometry.

Represent separate concepts for:

* hydraulic/thermal design,
* mechanical design,
* geometry generation,
* material,
* manufacturing process,
* machine capability,
* tool accessibility,
* tolerance,
* finish,
* datum,
* inspection,
* rotating assembly,
* bearings,
* seals,
* thrust balance,
* instrumentation,
* test configuration,
* measurement,
* uncertainty,
* calibration.

A watertight STEP file alone does not prove manufacturability.

A meanline pump plus an impeller solid does not prove a testable rotating assembly.

A model-generated geometry is not hardware-qualified merely because CadQuery reports a valid solid.

---

# 28. MEASUREMENTS AND CALIBRATION

Where experimental/test data are ingested, preserve enough context to prevent scientifically invalid comparisons.

A Measurement should retain where available:

* value,
* units,
* uncertainty,
* uncertainty convention,
* station,
* datum,
* instrument,
* calibration information,
* acquisition chain,
* timebase,
* test fluid,
* operating conditions,
* pressure convention,
* run identity,
* thermal state,
* quality flags,
* raw-data artifact hash,
* engine/configuration identity.

A measurement is not itself a pass/fail verdict.

Model comparison should explicitly handle:

* station matching,
* units,
* convention normalization,
* configuration matching,
* fluid identity,
* operating condition,
* uncertainty.

Calibration must create versioned coefficients or models.

Do not silently mutate the baseline model after fitting.

---

# 29. SEMANTIC CHUNKING

Chunk documents according to scientific structure rather than arbitrary fixed token lengths where possible.

Respect:

* sections,
* subsections,
* paragraphs,
* equations,
* derivations,
* tables,
* figures,
* captions,
* procedures,
* references.

Each chunk must retain:

* document ID,
* source PDF,
* page(s),
* section,
* order in document,
* extraction version.

Keep enough context around equations and tables for them to remain interpretable.

---

# 30. TABLES AND FIGURES

Extract structured tables when reliable.

Preserve:

* column names,
* units,
* row context,
* footnotes,
* source page.

Figures should retain:

* figure number,
* caption,
* source,
* page,
* nearby discussion,
* referenced concepts.

Do not fabricate numerical data from plots.

If plot digitization is required, implement it as an explicit, separately documented process with its own uncertainty and provenance.

---

# 31. PROVENANCE IS MANDATORY

Every important scientific object must be traceable.

An agent should be able to ask:

**Why does LREKit believe this?**

and traverse:

implementation
→ model
→ equation/claim
→ provenance
→ paper
→ page
→ source context.

Distinguish explicitly:

`SOURCE_STATES`

`PROJECT_DERIVES`

`SYSTEM_INFERS`

`IMPLEMENTATION_CHOICE`

`EMPIRICALLY_CALIBRATED`

Do not present synthesis produced by the LLM as though a source stated it.

Graph convenience never overrides provenance.

---

# 32. VECTOR + GRAPH HYBRID RETRIEVAL

Implement both:

* semantic retrieval,
* graph traversal.

A task query should generally perform:

task/query
→ semantic search
→ candidate chunks/documents
+
entity recognition
→ graph-node matches
→ relationship/path expansion
→ relevance reranking
→ provenance-backed context package.

The system should support LREKit questions such as:

* "What source supports the chamber pressure relation used here?"
* "Which Rao equations are relevant to this contour solver?"
* "What assumptions are required by this heat-transfer correlation?"
* "Which cooling correlations in the corpus are actually appropriate for this coolant state?"
* "What literature discusses acoustic combustion instability versus chamber pressure?"
* "Which sources constrain pintle injector geometry?"
* "What assumptions underlie this pump meanline calculation?"
* "Which NPSH criterion is being applied?"
* "Which source supports this blade-angle constraint?"
* "What differs between the traditional and MDO cooling implementations?"
* "What equations affect wall temperature?"
* "What models feed the pump pressure-rise requirement?"
* "Which validity gates can make this result unavailable?"
* "What benchmark supports this nozzle implementation?"
* "Which current LREKit functions implement this equation?"
* "Are there conflicting formulations in the corpus?"
* "Which literature is still missing for this planned feature?"

---

# 33. DEVELOPMENT-AGENT QUERY INTERFACE

Expose a stable API usable by LREKit coding agents.

Possible functions include:

```text
search_documents(query)
semantic_search(query)

find_entity(name)
get_entity(entity_id)
get_neighbors(entity_id)
find_path(entity_a, entity_b)

find_equations(concept)
get_equation(equation_id)

find_claims(subject, predicate=None, object=None)
get_claim_sources(claim_id)

find_models(concept)
get_model(model_id)

find_requirements(query)
find_validity_conditions(model)

get_document(document_id)
get_source_context(source_reference)
compare_sources(query)

find_implementation(scientific_object)
find_scientific_basis(software_component)

retrieve_for_task(task_description)
```

The internal storage architecture may change.

The agent-facing contract should remain stable where practical.

---

# 34. PROVIDE A SMALL `kg` CLI

Provide a compact command-line interface in addition to the programmatic API.

For example:

```bash
kg search "regenerative cooling pressure drop"

kg entity "chamber pressure"

kg equation "characteristic velocity"

kg claims "film cooling"

kg model "Bartz"

kg sources EQ_0192

kg implementation EQ_0192

kg basis raosim/thermal_design.py

kg path "chamber pressure" "pump pressure rise"

kg compare "regenerative cooling correlations"

kg ask "What assumptions does LREKit's nozzle performance calculation make?"

kg context --task "Implement the multipoint cooling constraints"
```

Exact syntax may differ.

Prioritize utility to coding agents.

---

# 35. `retrieve_for_task()` IS THE MOST IMPORTANT INTERFACE

`retrieve_for_task()` should accept the actual development task.

Example:

```python
retrieve_for_task(
    "Repair the regenerative-cooling pressure-drop model used by the MDO"
)
```

It should return structured context containing relevant subsets of:

* LREKit subsystem,
* source documents,
* source IDs,
* pages,
* relevant claims,
* governing equations,
* dependent equations,
* variables,
* definitions,
* symbols,
* units,
* dimensions,
* input quantities,
* output quantities,
* algorithms,
* procedures,
* assumptions,
* boundary conditions,
* validity conditions,
* empirical constants,
* correlations,
* materials,
* recommended procedures,
* conflicting formulations,
* limitations,
* known unavailable quantities,
* applicable requirements,
* optimizer constraints,
* postsolve gates,
* implementation warnings,
* existing LREKit modules/functions,
* related tests,
* relevant benchmarks,
* planned-but-not-implemented work,
* original source references.

Optimize this response for **doing the engineering task**, not for producing a conversational literature summary.

---

# 36. MACHINE-READABLE TASK CONTEXT

Where practical return something resembling:

```json
{
  "task": "...",
  "subsystems": [],
  "concepts": [],
  "quantities": [],
  "equations": [],
  "models": [],
  "claims": [],
  "variables": [],
  "materials": [],
  "assumptions": [],
  "boundary_conditions": [],
  "applicability_conditions": [],
  "validity_envelopes": [],
  "requirements": [],
  "constraints": [],
  "gates": [],
  "benchmarks": [],
  "tests": [],
  "existing_implementations": [],
  "implementation_warnings": [],
  "unavailable_capabilities": [],
  "conflicting_sources": [],
  "source_context": []
}
```

Objects should retain IDs so the agent can drill down instead of loading the whole corpus.

---

# 37. MANDATORY LREKIT AGENT STARTUP PROTOCOL

Create and maintain:

`knowledge/AGENTS.md`

This is the operational manual for every future LREKit coding agent.

Every agent operating in LREKit should inspect this file at the beginning of its work.

Before completing any task that changes or evaluates:

* physics,
* equations,
* thermodynamics,
* fluid mechanics,
* heat transfer,
* structures,
* injectors,
* spray,
* pumps,
* feed systems,
* optimization constraints,
* nozzle geometry,
* CAD dimensions derived from physics,
* material limits,
* manufacturing rules,
* validity criteria,
* empirical coefficients,
* requirements,
* measurement comparison,
* calibration,
* validation,
* qualification claims,

the agent MUST:

1. read `knowledge/AGENTS.md`,
2. verify that the corpus/knowledge system is synchronized,
3. call `retrieve_for_task()` using the actual task,
4. inspect relevant graph entities,
5. inspect relevant claims,
6. inspect relevant equations,
7. inspect variables and units,
8. inspect assumptions,
9. inspect applicability/validity conditions,
10. inspect competing formulations,
11. inspect existing LREKit implementations,
12. inspect relevant tests/benchmarks/gates,
13. retrieve original source context for load-bearing scientific decisions,
14. distinguish literature-derived science from implementation choices,
15. only then finalize the implementation.

This protocol also applies when auditing or repairing existing code.

**Existing code is not scientific authority merely because it already exists.**

For purely mechanical repository tasks—formatting, renaming, packaging, non-scientific UI changes, etc.—the agent does not need to perform unnecessary literature retrieval, but it should still respect `knowledge/AGENTS.md` and repository conventions.

---

# 38. RETRIEVAL BEFORE REDISCOVERY

Do not make every coding agent scan the entire propulsion library from scratch.

Prefer:

`retrieve_for_task()`
→ graph inspection
→ claim/equation inspection
→ targeted source verification
→ implementation

over:

scan hundreds of PDFs
→ rediscover terminology
→ rediscover equations
→ rediscover assumptions
→ implement.

The knowledge graph is an index and reasoning aid.

It is **not** permission to stop reading original sources.

---

# 39. ORIGINAL-SOURCE ESCALATION

For load-bearing engineering decisions, the agent must be able to escalate:

graph node
→ claim/model/equation
→ provenance record
→ page
→ original PDF context.

Direct source inspection is particularly important when:

* extraction confidence is low,
* mathematical notation is ambiguous,
* equation formatting looks damaged,
* a correlation's applicability is unclear,
* multiple sources disagree,
* a graph relationship looks suspicious,
* validity limits matter,
* a hard-coded constant is being introduced,
* a manufacturing or safety-related criterion is being implemented,
* qualification language is involved.

An equation stripped of its assumptions can be scientifically wrong even when the symbols were extracted perfectly.

---

# 40. IMPLEMENTATION TRACEABILITY

Where useful, source code or nearby documentation should retain stable knowledge IDs.

Example:

```python
# Scientific basis:
# model_id: MODEL_REGEN_004
# equation_id: EQ_BARTZ_001
# source: DOC_NASA_SP_XXXX
#
# Validity and assumptions are recorded in the LREKit knowledge system.
```

Do not litter every line with citations.

Use references where they materially improve traceability for:

* governing equations,
* empirical correlations,
* engineering limits,
* hard-coded physical constants,
* non-obvious numerical procedures,
* calibration,
* validity boundaries,
* source-driven manufacturing criteria.

---

# 41. KNOWLEDGE/AGENTS.MD MUST EXPLAIN

Maintain a concise agent-facing manual explaining:

* what the LREKit knowledge system contains,
* corpus locations,
* PDF versus Markdown authority,
* synchronization procedure,
* ingestion state,
* graph schema,
* entity normalization,
* claims,
* equations,
* models,
* provenance,
* validity envelopes,
* software↔science mappings,
* retrieval API,
* CLI,
* `retrieve_for_task()`,
* source escalation,
* conflict handling,
* unavailable states,
* known limitations,
* extraction failures,
* how to add new literature,
* how development agents must use the system.

Agents should not have to reverse-engineer the database to use the scientific memory.

---

# 42. QUALITY CONTROL

Run validation after ingestion.

Check for:

* source PDFs without manifest entries,
* generated Markdown without source PDFs,
* PDFs without required Markdown derivatives,
* stale Markdown after source changes,
* orphaned claims,
* orphaned equations,
* edges without provenance,
* nonexistent page references,
* undefined variables,
* unresolved symbols,
* incompatible units,
* equations with unresolved quantities,
* suspicious entity merges,
* duplicate claims,
* duplicate equations,
* duplicate vector entries,
* graph nodes with implausibly high generic degree,
* missing implementation links,
* stale implementation-state claims,
* invalid supersession relationships,
* planned features labeled implemented,
* report-only models labeled enforced,
* software regressions labeled experimental validation,
* unqualified artifacts labeled hardware-qualified.

Generate an ingestion/validation report after each update.

---

# 43. STORAGE

Use a practical local architecture.

Prefer inspectability and maintainability over needless distributed infrastructure.

A reasonable implementation may use:

```text
knowledge/
    AGENTS.md
    manifest.json
    schema.json
    entities.*
    quantities.*
    claims.*
    equations.*
    models.*
    requirements.*
    validity.*
    software_links.*
    relationships.*
    provenance.*
    ingestion_state.*
    vector_index/
    graph/
    reports/
```

SQLite, a local vector index, and NetworkX or another simple graph representation are acceptable if they satisfy requirements.

A dedicated graph database may be used if actual query needs justify it.

Do not select infrastructure merely because it sounds sophisticated.

Provide human-readable exports.

Support a standard graph export such as GraphML or JSON.

---

# 44. IDEMPOTENCY

The synchronization pipeline must be safe to run repeatedly.

When nothing changed:

scan
→ compare hashes/manifest
→ no changes
→ confirm synchronized state
→ exit cleanly.

Do not:

* regenerate all Markdown,
* duplicate entities,
* duplicate claims,
* duplicate equations,
* duplicate provenance,
* duplicate embeddings,
* rewrite large portions of the repository unnecessarily.

---

# 45. IMPLEMENTATION PROCESS

Do not begin by generating thousands of lines of code blindly.

Proceed approximately as follows:

1. inspect repository,
2. inspect current literature organization,
3. inspect human-authored metadata,
4. inspect existing paper index/manifest,
5. inspect existing agent-readable Markdown,
6. check PDF↔Markdown synchronization,
7. inventory current knowledge-system artifacts,
8. define the LREKit ontology,
9. define provenance model,
10. test parsing on representative propulsion sources,
11. inspect equation extraction quality,
12. inspect table extraction quality,
13. implement/update manifest,
14. implement/update document ingestion,
15. implement entity normalization,
16. implement equation extraction,
17. implement claims/model extraction,
18. implement software↔science mappings,
19. implement vector retrieval,
20. construct graph,
21. implement task-aware retrieval,
22. implement source escalation,
23. implement incremental ingestion,
24. implement validation,
25. generate `knowledge/AGENTS.md`,
26. test the system using actual LREKit development tasks.

Favor correctness and traceability over graph size.

---

# 46. REALISTIC ACCEPTANCE QUERIES

Test the resulting system against questions such as:

### Nozzle

"What equations and source passages support the Rao contour methods currently present in LREKit?"

"What assumptions differ between the TOP approximation and the variational Rao solver?"

"What source and criterion determine whether a Rao solution is physically valid?"

"Which LREKit functions implement the MOC characteristic relations?"

### Cooling

"What equations determine gas-side heat flux, coolant-side heat transfer, and wall temperature?"

"Which correlations are empirical and what are their validity conditions?"

"How does cooling-jacket pressure drop propagate into pump sizing?"

"What differs between the traditional and differentiable cooling implementations?"

### Injector

"What literature supports the pintle injector hydraulic model?"

"Which discharge coefficients are literature values versus calibrated quantities?"

"What conditions invalidate the injector model?"

### Pump/feed

"What literature supports the current pump meanline relations?"

"What assumptions govern inducer/NPSH calculations?"

"Which pump quantities are currently reference geometry rather than mechanically complete design outputs?"

### MDO

"Which quantities are optimizer-enforced constraints versus final gates?"

"What literature supports every physical constraint?"

"Which JAX models have NumPy reference oracles and parity tests?"

"Which current fidelity gaps are intentionally unavailable?"

### Hardware

"What evidence would be required before this generated component could be described as machineable?"

"What is needed to bind this CAD artifact to a resolved engine configuration?"

"What literature is still missing before implementing the bearing/seal/rotordynamics work?"

### Verification

"What published benchmarks exist for this model?"

"Is this quantity software-verified, benchmark-validated, experimentally validated, or still unsupported?"

The system passes only if these queries lead agents to usable, source-backed engineering context rather than generic summaries.

---

# 47. FAILURE AND UNCERTAINTY LANGUAGE

Use explicit states such as:

`NO_RELEVANT_SOURCE_FOUND`

`INSUFFICIENT_SOURCE_CONTEXT`

`CONFLICTING_FORMULATIONS`

`AMBIGUOUS_SYMBOL`

`LOW_EXTRACTION_CONFIDENCE`

`UNRESOLVED_ENTITY`

`SOURCE_NOT_INGESTED`

`OUTSIDE_MODEL_VALIDITY`

`SCIENTIFIC_BASIS_UNKNOWN`

`PLANNED_NOT_IMPLEMENTED`

Do not transform:

"No supporting equation was found in the currently ingested corpus."

into:

"No such equation exists."

Do not transform:

"No validity range was located."

into an invented numerical envelope.

Do not transform:

"The implementation returns a number."

into:

"The model is valid here."

---

# 48. CRITICAL LREKIT RULES

These rules are non-negotiable.

1. **Never invent scientific authority.**

2. **Never invent citations.**

3. **Never invent missing equations, coefficients, material properties, correlations, or validity limits.**

4. **Never turn uncertainty into certainty.**

5. **Never confuse an LLM interpretation with a source statement.**

6. **Never discard assumptions or applicability conditions from engineering equations.**

7. **Never merge symbols solely because they use the same notation.**

8. **Units and dimensions are part of scientific meaning.**

9. **Never collapse competing formulations without preserving their individual provenance.**

10. **Never treat planned functionality as implemented functionality.**

11. **Never treat an available software output as proof of scientific validity.**

12. **Never treat software regression tests as experimental validation.**

13. **Never treat valid CAD topology as proof of manufacturability or hardware qualification.**

14. **Never hide a fidelity gap inside a passing number.**

15. **When a capability is unsupported, prefer explicit refusal or `unknown`/`unavailable` behavior over fabrication.**

16. **A model's validity envelope is as important as the model equation itself.**

17. **A measurement is not automatically a verdict.**

18. **A calibration must remain bound to its source, data artifact, configuration/geometry, and validity range.**

19. **The knowledge graph must never become more authoritative than the source documents it represents.**

20. **The original propulsion literature remains the ultimate source for source-derived scientific claims.**

21. **Before scientifically consequential LREKit development, retrieve existing scientific memory before rediscovering the corpus.**

22. **Before trusting the knowledge system, ensure new PDFs have been converted, ingested, indexed, and integrated into the graph.**

23. **If a new PDF appears, update the graph as integrated scientific memory—not as an isolated append-only document node.**

24. **If source evidence is missing, say so.**

---

# 49. FINAL OBJECTIVE

The objective is **not** to maximize the number of nodes and edges.

The objective is to give LREKit a persistent, inspectable, source-backed engineering memory.

The final system should let an agent reliably perform:

LREKit task
→ subsystem
→ scientific question
→ relevant literature
→ concept
→ equation / model / claim
→ assumptions
→ validity
→ competing formulations
→ project implementation
→ tests / benchmark / gate
→ original source
→ scientifically traceable implementation

without rescanning the entire propulsion library for every development task.

The knowledge graph, vector index, manifest, provenance database, and retrieval API are components of that larger system.

The final product should behave as:

**LREKit's local scientific memory and engineering provenance layer.**
