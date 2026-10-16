"""LREKit knowledge-system ontology.

Versioned deliberately: consumers record ``ONTOLOGY_VERSION`` alongside any
derived artifact so a later schema change is detectable rather than silent.

Design notes
------------
* Node and edge types exist because they enable a retrieval an LREKit agent
  actually needs, not to maximise graph size.
* The status vocabularies are the repository's own engineering vocabulary.
  ``UNKNOWN`` and ``UNAVAILABLE`` are first-class results, never placeholders
  to be filled in later with a plausible number.
"""

from __future__ import annotations

ONTOLOGY_VERSION = "1.1.0"
EXTRACTION_VERSION = "1.0.1"

# --------------------------------------------------------------------------
# Node types
# --------------------------------------------------------------------------

SOURCE_NODES = (
    "Document",            # a source PDF (authoritative)
    "GeneratedDocument",   # its Markdown derivative
    "Section",
    "Chunk",
    "Figure",
    "Table",
    "Citation",
    "SourceReference",     # document + page + verbatim locator
    "ProvenanceRecord",
    "ExtractionCandidate", # parser output awaiting source-page/semantic review
)

SCIENCE_NODES = (
    "Concept",
    "PhysicalQuantity",
    "Variable",
    "Equation",
    "Correlation",
    "Model",
    "Method",
    "Algorithm",
    "Procedure",
    "Assumption",
    "BoundaryCondition",
    "Constraint",
    "Requirement",
    "ValidityEnvelope",
    "Recommendation",
    "EmpiricalConstant",
    "MaterialProperty",
    "FailureMode",
    "Claim",
)

SOFTWARE_NODES = (
    "SoftwareComponent",
    "Module",
    "Function",
    "Class",
    "CLICommand",
    "ModelRegistryEntry",
    "Capability",
    "Test",
    "Benchmark",
    "ValidationGate",
    "ArtifactManifest",
    "RepositoryDocument",   # docs/*.md - implementation-state authority, NOT scientific authority
)

SYSTEM_NODES = (
    "Subsystem",
    "EngineConfiguration",
    "OperatingPoint",
    "PropellantPair",
    "CADArtifact",
    "ManufacturingProcess",
    "Measurement",
    "Experiment",
    "Calibration",
    "ComparisonRecord",
)

NODE_TYPES = SOURCE_NODES + SCIENCE_NODES + SOFTWARE_NODES + SYSTEM_NODES

# --------------------------------------------------------------------------
# Edge types
# --------------------------------------------------------------------------

EDGE_TYPES = (
    # source / provenance
    "DERIVED_FROM", "FOUND_IN", "CITES", "HAS_PROVENANCE", "SUPPORTED_BY",
    # science
    "DEFINES", "DESCRIBES", "USES", "DERIVES", "DEPENDS_ON", "AFFECTS",
    "CONSTRAINS", "ASSUMES", "MEASURES", "PREDICTS", "ESTIMATES",
    "INPUT_VARIABLE", "OUTPUT_VARIABLE", "INPUT_TO", "OUTPUT_OF", "PARAMETER_OF",
    "EQUIVALENT_TO", "POSSIBLY_EQUIVALENT", "APPROXIMATES",
    "APPLICABLE_WHEN", "HAS_VALIDITY_ENVELOPE",
    "CONTRADICTS", "COMPETING_FORMULATION_WITH", "SUPPORTS", "EXTENDS",
    # software <-> science
    "IMPLEMENTS", "IMPLEMENTED_IN", "COMPARED_WITH", "REFERENCE_ORACLE_FOR",
    "PARITY_CHECKED_AGAINST", "BENCHMARKED_AGAINST", "VALIDATES", "INVALIDATES",
    "TESTED_WITH", "GATED_BY", "REFUSES_WHEN", "PRODUCES_ARTIFACT",
    "AVAILABLE_IN", "UNAVAILABLE_IN", "BLOCKED_ON_LITERATURE",
    "PART_OF", "BOUND_TO_CONFIGURATION",
    # document state
    "SUPERSEDES", "CORRECTS", "UPDATES", "AUDITS", "IMPLEMENTS_PLAN_ITEM",
    "DEPRECATES", "BLOCKS", "PLANNED_BY",
    # measurement / calibration
    "MEASURED_FROM", "CALIBRATED_FROM", "APPLIES_TO", "INSTANCE_OF",
)

# --------------------------------------------------------------------------
# Provenance kinds - who is asserting this?
# --------------------------------------------------------------------------

PROVENANCE_KINDS = (
    "SOURCE_STATES",           # the cited document says this, on this page
    "PROJECT_DERIVES",         # a repository document/registry derives it
    "SYSTEM_INFERS",           # this knowledge system inferred it (lowest authority)
    "IMPLEMENTATION_CHOICE",   # a coding decision, not a physical claim
    "EMPIRICALLY_CALIBRATED",  # fitted to named data
    "CODE_STATES",             # LREKit source code asserts it in a comment/docstring
)

BASIS_CLASSES = (
    "LITERATURE_DERIVED",
    "PROJECT_DERIVED",
    "IMPLEMENTATION_CHOICE",
    "EMPIRICALLY_CALIBRATED",
    "REPOSITORY_HEURISTIC",
    "UNKNOWN_BASIS",
)

# --------------------------------------------------------------------------
# Result / state vocabularies
# --------------------------------------------------------------------------

ENGINEERING_STATES = (
    "PASS", "FAIL", "UNKNOWN", "UNAVAILABLE", "NOT_IMPLEMENTED",
    "OUTSIDE_VALIDITY", "BLOCKED_ON_LITERATURE", "BLOCKED_ON_STANDARD",
    "NOT_QUALIFIED", "REPORT_ONLY", "POSTSOLVE_ONLY",
    "CONVERGED_BUT_INVALID", "NUMERICALLY_FAILED",
)

RETRIEVAL_STATES = (
    "NO_RELEVANT_SOURCE_FOUND",
    "INSUFFICIENT_SOURCE_CONTEXT",
    "CONFLICTING_FORMULATIONS",
    "AMBIGUOUS_SYMBOL",
    "LOW_EXTRACTION_CONFIDENCE",
    "UNRESOLVED_ENTITY",
    "SOURCE_NOT_INGESTED",
    "OUTSIDE_MODEL_VALIDITY",
    "SCIENTIFIC_BASIS_UNKNOWN",
    "PLANNED_NOT_IMPLEMENTED",
)

CONVERSION_STATES = (
    "good", "needs_review", "partial_conversion",
    "pdf_conversion_failed", "source_not_ingested",
)

# Restriction kinds are deliberately NOT flattened into one "constraint" type.
RESTRICTION_KINDS = (
    "requirement",          # what the requested engine must satisfy
    "optimizer_constraint", # actively enforced inside the optimisation solve
    "postsolve_gate",       # required check evaluated after solving
    "diagnostic",           # reported, does not itself determine success
    "applicability_gate",   # decides whether a model may predict at all
    "evidence_gate",        # decides whether a claim has sufficient evidence
    "manufacturing_gate",   # decides whether an artifact meets process criteria
)

VERIFICATION_KINDS = (
    "unit_test",
    "numerical_regression",
    "numpy_jax_parity",
    "autodiff_finite_difference",
    "published_benchmark_reproduction",
    "cad_round_trip",
    "model_vs_model_comparison",
    "experimental_validation",
    "qualification_test",
    "snapshot_contract",
)

# A model's promotion status inside LREKit.  Higher fidelity does NOT imply production.
MODEL_PROMOTION = (
    "optimizer_coupled",   # participates in the optimisation
    "postsolve_only",
    "report_only",
    "screening",
    "reference_oracle",
    "planned_not_implemented",
    "unavailable",
)

SUBSYSTEMS = (
    "requirements", "thermochemistry", "performance", "gas_dynamics",
    "nozzle_contour", "moc", "throat", "chamber", "cooling", "heat_transfer",
    "structures", "materials", "injector", "spray", "feed_system", "pump",
    "electric_drive", "mass", "trajectory", "optimization", "mdo",
    "cad", "manufacturing", "verification", "measurement", "configuration",
)


def node_id(prefix: str, key: str) -> str:
    """Stable, human-readable identifiers (``EQ_...``, ``DOC_...``)."""
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").upper()
    return f"{prefix}_{slug}"[:120]


SCHEMA = {
    "ontology_version": ONTOLOGY_VERSION,
    "extraction_version": EXTRACTION_VERSION,
    "node_types": list(NODE_TYPES),
    "edge_types": list(EDGE_TYPES),
    "provenance_kinds": list(PROVENANCE_KINDS),
    "basis_classes": list(BASIS_CLASSES),
    "engineering_states": list(ENGINEERING_STATES),
    "retrieval_states": list(RETRIEVAL_STATES),
    "conversion_states": list(CONVERSION_STATES),
    "restriction_kinds": list(RESTRICTION_KINDS),
    "verification_kinds": list(VERIFICATION_KINDS),
    "model_promotion": list(MODEL_PROMOTION),
    "subsystems": list(SUBSYSTEMS),
    "id_prefixes": {
        "Document": "DOC", "GeneratedDocument": "MD", "Chunk": "CH",
        "Equation": "EQ", "Claim": "CLM", "Model": "MODEL",
        "PhysicalQuantity": "QTY", "Variable": "VAR", "Concept": "CON",
        "Assumption": "ASM", "ValidityEnvelope": "VAL", "Requirement": "REQ",
        "Constraint": "CST", "ValidationGate": "GATE", "Test": "TEST",
        "Benchmark": "BENCH", "SoftwareComponent": "SW", "Capability": "CAP",
        "RepositoryDocument": "RDOC", "ProvenanceRecord": "PROV",
        "ExtractionCandidate": "CAND",
        "EmpiricalConstant": "CONST", "Subsystem": "SUB",
    },
}
