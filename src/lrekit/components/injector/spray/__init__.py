"""Deterministic, physics-explicit Lagrangian spray modelling.

The package is intentionally opt-in.  Importing it does not replace the legacy
injector correlation screen, and parcel results are not eligible for cycle
coupling until their conservation and declared literature-validation gates
pass.
"""

from .benchmarks import (
    SprayBenchmarkDataset,
    SprayBenchmarkReadinessReport,
    SpraySMDComparison,
    benchmark_readiness_report,
    compare_smd_to_benchmark,
    list_spray_benchmark_cases,
    load_spray_benchmark,
)
from .breakup import (
    RayleighTaylorConfig,
    WaveBreakupConfig,
    advance_breakup,
    calibrate_wave_constants_from_vof,
    compute_wave_metrics,
)
from .carrier import (
    AxisymmetricRectilinearCarrierField,
    CarrierField,
    UniformCarrierField,
    carrier_field_fingerprint,
)
from .domain import AxisymmetricDomain, BoundaryCrossing
from .evaporation import (
    advance_evaporation,
    evaporation_rate_2021,
    spalding_mass_number,
)
from .handoff import (
    GasCarrierStream,
    NumericalConvergenceEvidence,
    SprayCycleHandoff,
    SprayStreamAccounting,
    build_cycle_handoff,
)
from .ledger import ConservationLedger
from .primary import (
    AxialAnnularSheetGeometry,
    PlanarSlotJetGeometry,
    PrimaryParcelInitialization,
    RadialSheetGeometry,
    RoundHoleJetGeometry,
    initialize_primary_parcels,
    radial_sheet_geometry_from_injector_result,
)
from .solver import (
    EvaporationModelConfig,
    SamplingPlaneCloud,
    SprayMarchConfig,
    SprayMarchResult,
    march_parcels,
)
from .statistics import (
    RosinRammlerFit,
    SprayStatistics,
    fit_rosin_rammler,
    mass_weighted_percentile,
    rosin_rammler_survival,
    sample_rosin_rammler,
    sauter_mean_diameter,
    summarize_spray,
)
from .types import LiquidProperties, ParcelCloud, SpraySolverSpec

__all__ = [
    "AxisymmetricDomain",
    "AxisymmetricRectilinearCarrierField",
    "AxialAnnularSheetGeometry",
    "BoundaryCrossing",
    "CarrierField",
    "ConservationLedger",
    "EvaporationModelConfig",
    "GasCarrierStream",
    "LiquidProperties",
    "NumericalConvergenceEvidence",
    "ParcelCloud",
    "PlanarSlotJetGeometry",
    "PrimaryParcelInitialization",
    "RadialSheetGeometry",
    "RayleighTaylorConfig",
    "RosinRammlerFit",
    "RoundHoleJetGeometry",
    "SamplingPlaneCloud",
    "SprayBenchmarkDataset",
    "SprayBenchmarkReadinessReport",
    "SpraySMDComparison",
    "SprayCycleHandoff",
    "SprayMarchConfig",
    "SprayMarchResult",
    "SpraySolverSpec",
    "SprayStatistics",
    "SprayStreamAccounting",
    "UniformCarrierField",
    "WaveBreakupConfig",
    "advance_breakup",
    "advance_evaporation",
    "benchmark_readiness_report",
    "build_cycle_handoff",
    "carrier_field_fingerprint",
    "calibrate_wave_constants_from_vof",
    "compute_wave_metrics",
    "compare_smd_to_benchmark",
    "evaporation_rate_2021",
    "fit_rosin_rammler",
    "initialize_primary_parcels",
    "list_spray_benchmark_cases",
    "load_spray_benchmark",
    "mass_weighted_percentile",
    "march_parcels",
    "rosin_rammler_survival",
    "sample_rosin_rammler",
    "sauter_mean_diameter",
    "radial_sheet_geometry_from_injector_result",
    "spalding_mass_number",
    "summarize_spray",
]
