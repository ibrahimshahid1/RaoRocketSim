"""Deterministic external-CFD case contracts.

The package writes version-pinned OpenFOAM inputs but never imports or runs
OpenFOAM.  Generated cases therefore remain software-unverified until the
explicit mesh, execution, convergence, and validation gates are supplied.
"""

from .case import (
    MovablePintleVOFGeometry,
    OpenFOAMCasePackage,
    OpenFOAMCaseWriteResult,
    OpenFOAMExportError,
    OpenFOAMFluidProperties,
    OpenFOAMWedgeControls,
    build_radhakrishnan2018_sheet_vof_case,
    cli_main,
    write_openfoam_case,
)
from .handoff import (
    CarrierAxisymmetricFieldEvidence,
    VOFArtifactProvenance,
    VOFAveragingWindow,
    VOFConvergenceEvidence,
    VOFConvergenceStudy,
    VOFHandoffGate,
    VOFHandoffValidationError,
    VOFLiquidFluxBalance,
    VOFSheetExtractionDefinition,
    VOFSheetStatistics,
    VOFToLagrangianHandoff,
)

__all__ = [
    "MovablePintleVOFGeometry",
    "CarrierAxisymmetricFieldEvidence",
    "OpenFOAMCasePackage",
    "OpenFOAMCaseWriteResult",
    "OpenFOAMExportError",
    "OpenFOAMFluidProperties",
    "OpenFOAMWedgeControls",
    "VOFArtifactProvenance",
    "VOFAveragingWindow",
    "VOFConvergenceEvidence",
    "VOFConvergenceStudy",
    "VOFHandoffGate",
    "VOFHandoffValidationError",
    "VOFLiquidFluxBalance",
    "VOFSheetExtractionDefinition",
    "VOFSheetStatistics",
    "VOFToLagrangianHandoff",
    "build_radhakrishnan2018_sheet_vof_case",
    "cli_main",
    "write_openfoam_case",
]
