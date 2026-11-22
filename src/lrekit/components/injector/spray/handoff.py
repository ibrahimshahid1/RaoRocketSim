"""Typed, gate-derived handoff from parcel results to cycle coupling.

The legacy fixed point accepts the injector correlation screen.  This module
defines the stronger contract a future Lagrangian fixed point must consume.
It distinguishes liquid parcels from gas carrier streams and refuses to turn
a finite vaporization fraction into cycle authority by itself.

The builder is useful now for complete provenance reports, but the presently
implemented one-way/no-energy solver and public benchmark fixtures make every
such handoff cycle-ineligible.  That is an evidence result, not a hard-coded
caller-selectable flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .benchmarks import (
    SprayBenchmarkDataset,
    benchmark_readiness_report,
)
from .primary import PrimaryParcelInitialization
from .solver import SprayGate, SprayMarchResult


@dataclass(frozen=True)
class GasCarrierStream:
    """Traceable Eulerian carrier-stream continuity evidence for one role."""

    role: str
    fluid_name: str
    mass_flow_rate: float
    composition_mass_fraction: Mapping[str, float]
    continuity_relative_residual: float
    continuity_tolerance: float
    operating_point_id: str
    field_fingerprint: str
    continuity_source: str

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if not role:
            raise ValueError("gas carrier role must be nonblank")
        fluid_name = str(self.fluid_name).strip()
        if not fluid_name:
            raise ValueError("gas carrier fluid_name must be nonblank")
        mass_flow = float(self.mass_flow_rate)
        if not math.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("gas carrier mass_flow_rate must be finite and > 0")
        residual = float(self.continuity_relative_residual)
        tolerance = float(self.continuity_tolerance)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("carrier continuity residual must be finite and >= 0")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("carrier continuity tolerance must be finite and > 0")
        composition = {
            str(species).strip(): float(fraction)
            for species, fraction in self.composition_mass_fraction.items()
        }
        if not composition or any(not species for species in composition):
            raise ValueError("carrier composition must name at least one species")
        if any(not math.isfinite(value) or value < 0.0 for value in composition.values()):
            raise ValueError("carrier mass fractions must be finite and >= 0")
        if not math.isclose(sum(composition.values()), 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("carrier mass fractions must sum to one")
        for name in ("operating_point_id", "field_fingerprint", "continuity_source"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"gas carrier {name} must be nonblank")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "fluid_name", fluid_name)
        object.__setattr__(self, "mass_flow_rate", mass_flow)
        object.__setattr__(self, "continuity_relative_residual", residual)
        object.__setattr__(self, "continuity_tolerance", tolerance)
        object.__setattr__(
            self, "composition_mass_fraction", MappingProxyType(composition)
        )

    @property
    def continuity_closed(self) -> bool:
        return self.continuity_relative_residual <= self.continuity_tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fluid_name": self.fluid_name,
            "mass_flow_rate_kg_s": self.mass_flow_rate,
            "composition_mass_fraction": dict(self.composition_mass_fraction),
            "continuity_relative_residual": self.continuity_relative_residual,
            "continuity_tolerance": self.continuity_tolerance,
            "continuity_closed": self.continuity_closed,
            "operating_point_id": self.operating_point_id,
            "field_fingerprint": self.field_fingerprint,
            "continuity_source": self.continuity_source,
        }


@dataclass(frozen=True)
class NumericalConvergenceEvidence:
    """Results of explicit time-step and parcel-count refinement studies."""

    time_step_relative_change: float
    parcel_count_relative_change: float
    acceptance_tolerance: float
    run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "time_step_relative_change",
            "parcel_count_relative_change",
            "acceptance_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        if self.acceptance_tolerance == 0.0:
            raise ValueError("acceptance_tolerance must be > 0")
        run_ids = tuple(str(value).strip() for value in self.run_ids)
        if len(run_ids) < 3 or any(not value for value in run_ids):
            raise ValueError(
                "convergence evidence requires at least three named refinement runs"
            )
        object.__setattr__(self, "run_ids", run_ids)

    @property
    def passed(self) -> bool:
        return (
            self.time_step_relative_change <= self.acceptance_tolerance
            and self.parcel_count_relative_change <= self.acceptance_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_step_relative_change": self.time_step_relative_change,
            "parcel_count_relative_change": self.parcel_count_relative_change,
            "acceptance_tolerance": self.acceptance_tolerance,
            "run_ids": list(self.run_ids),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SprayStreamAccounting:
    role: str
    fluid_name: str
    representation: Literal["liquid_parcels", "gas_carrier"]
    mass_flow_rate: float
    expected_mass_flow_rate: float
    mass_flow_relative_error: float
    accounted: bool
    injected_liquid_mass: float | None
    vaporized_liquid_mass: float | None
    eta_vaporization: float | None
    smd: float | None
    geometry_model_id: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fluid_name": self.fluid_name,
            "representation": self.representation,
            "mass_flow_rate_kg_s": self.mass_flow_rate,
            "expected_mass_flow_rate_kg_s": self.expected_mass_flow_rate,
            "mass_flow_relative_error": self.mass_flow_relative_error,
            "accounted": self.accounted,
            "injected_liquid_mass_kg": self.injected_liquid_mass,
            "vaporized_liquid_mass_kg": self.vaporized_liquid_mass,
            "eta_vaporization": self.eta_vaporization,
            "smd_m": self.smd,
            "geometry_model_id": self.geometry_model_id,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class BenchmarkEvidence:
    case_id: str
    source_sha256: str
    validation_role: str
    strict_end_to_end_smd_validation_ready: bool
    fluid_system_match: bool
    tables_7_8_are_experimental: bool | None
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_sha256": self.source_sha256,
            "validation_role": self.validation_role,
            "strict_end_to_end_smd_validation_ready": (
                self.strict_end_to_end_smd_validation_ready
            ),
            "fluid_system_match": self.fluid_system_match,
            "tables_7_8_are_experimental": self.tables_7_8_are_experimental,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class SprayCycleHandoff:
    model_id: str
    model_version: str
    operating_point_id: str
    smd_sampling_plane: float
    streams: tuple[SprayStreamAccounting, ...]
    eta_vaporization: float
    aggregation_basis: str
    conservation: Mapping[str, Any]
    benchmark_evidence: tuple[BenchmarkEvidence, ...]
    carrier_provenance: tuple[Mapping[str, Any], ...]
    solver_metadata: Mapping[str, Any]
    convergence_evidence: Mapping[str, Any] | None
    required_gates: tuple[SprayGate, ...]
    fingerprint: str

    @property
    def all_streams_accounted(self) -> bool:
        return bool(self.streams) and all(stream.accounted for stream in self.streams)

    @property
    def coupling_eligible(self) -> bool:
        return self.all_streams_accounted and all(
            gate.passed for gate in self.required_gates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "operating_point_id": self.operating_point_id,
            "smd_sampling_plane_m": self.smd_sampling_plane,
            "fingerprint": self.fingerprint,
            "all_streams_accounted": self.all_streams_accounted,
            "coupling_eligible": self.coupling_eligible,
            "eta_vaporization": self.eta_vaporization,
            "aggregation_basis": self.aggregation_basis,
            "streams": [stream.to_dict() for stream in self.streams],
            "conservation": dict(self.conservation),
            "benchmark_evidence": [item.to_dict() for item in self.benchmark_evidence],
            "carrier_provenance": [dict(item) for item in self.carrier_provenance],
            "solver_metadata": dict(self.solver_metadata),
            "convergence_evidence": (
                None if self.convergence_evidence is None
                else dict(self.convergence_evidence)
            ),
            "required_gates": [gate.to_dict() for gate in self.required_gates],
        }


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), float.fromhex("0x1p-1022"))


def build_cycle_handoff(
    result: SprayMarchResult,
    *,
    liquid_sources: Sequence[PrimaryParcelInitialization],
    gas_carriers: Sequence[GasCarrierStream],
    expected_mass_flow_by_role: Mapping[str, float],
    mass_flow_tolerance: float,
    operating_point_id: str,
    smd_sampling_plane: float,
    benchmarks: Sequence[SprayBenchmarkDataset],
    convergence_evidence: NumericalConvergenceEvidence | None,
    regenerative_cooling: bool,
) -> SprayCycleHandoff:
    """Build a provenance-complete, non-forgeable-by-eta cycle handoff."""

    if not isinstance(result, SprayMarchResult):
        raise TypeError("result must be a SprayMarchResult, not an eta-like object")
    expected = {str(role): float(value) for role, value in expected_mass_flow_by_role.items()}
    if not expected or any(
        not role or not math.isfinite(value) or value <= 0.0
        for role, value in expected.items()
    ):
        raise ValueError("expected mass flows must name positive finite streams")
    tolerance = float(mass_flow_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("mass_flow_tolerance must be finite and > 0")
    operating_point = str(operating_point_id).strip()
    if not operating_point:
        raise ValueError("operating_point_id must be nonblank")
    if smd_sampling_plane not in result.sampling_planes:
        raise ValueError(
            "smd_sampling_plane must exactly identify a recorded solver plane"
        )

    liquids = tuple(liquid_sources)
    carriers = tuple(gas_carriers)
    role_collisions = {item.role for item in liquids} & {item.role for item in carriers}
    if role_collisions:
        raise ValueError(
            "roles cannot be both liquid parcels and gas carriers: "
            + ", ".join(sorted(role_collisions))
        )
    supplied_roles = {item.role for item in liquids} | {item.role for item in carriers}
    if supplied_roles != set(expected):
        missing = sorted(set(expected) - supplied_roles)
        extra = sorted(supplied_roles - set(expected))
        raise ValueError(f"stream-role mismatch; missing={missing}, extra={extra}")

    streams: list[SprayStreamAccounting] = []
    for source in liquids:
        ledger = result.conservation.per_role.get(source.role)
        if ledger is None:
            raise ValueError(f"parcel result has no conservation ledger for {source.role!r}")
        mass_error = _relative_error(source.mass_flow_rate, expected[source.role])
        blockers: list[str] = []
        if mass_error > tolerance:
            blockers.append("liquid source mass flow does not match the cycle iterate")
        if not source.primary_path_eligible:
            blockers.append("geometry-specific primary path is not applicable")
        sample_statistics = result.sampling_planes[smd_sampling_plane].statistics(source.role)
        if sample_statistics is None:
            blockers.append("no liquid parcel crossed the declared SMD sampling plane")
        accounted = (
            mass_error <= tolerance
            and ledger.mass_closed(result.conservation.mass_tolerance)
            and ledger.parcel_momentum_closed(result.conservation.momentum_tolerance)
        )
        streams.append(SprayStreamAccounting(
            role=source.role,
            fluid_name=source.liquid.name,
            representation="liquid_parcels",
            mass_flow_rate=source.mass_flow_rate,
            expected_mass_flow_rate=expected[source.role],
            mass_flow_relative_error=mass_error,
            accounted=accounted,
            injected_liquid_mass=ledger.injected_mass,
            vaporized_liquid_mass=ledger.vaporized_mass,
            eta_vaporization=ledger.vaporized_mass / ledger.injected_mass,
            smd=None if sample_statistics is None else sample_statistics.sauter_mean_diameter,
            geometry_model_id=source.model.model_id,
            blockers=tuple(blockers),
        ))

    carrier_payloads: list[Mapping[str, Any]] = []
    for carrier in carriers:
        mass_error = _relative_error(carrier.mass_flow_rate, expected[carrier.role])
        blockers = []
        if mass_error > tolerance:
            blockers.append("carrier mass flow does not match the cycle iterate")
        if not carrier.continuity_closed:
            blockers.append("carrier continuity residual exceeds its declared tolerance")
        if carrier.operating_point_id != operating_point:
            blockers.append("carrier field was generated for a different operating point")
        solved_fingerprint = result.solver_metadata.get(
            "carrier_field_fingerprint_sha256"
        )
        if solved_fingerprint is None:
            blockers.append("solver could not fingerprint the marched carrier field")
        elif carrier.field_fingerprint != solved_fingerprint:
            blockers.append("carrier descriptor does not match the marched field fingerprint")
        accounted = not blockers
        streams.append(SprayStreamAccounting(
            role=carrier.role,
            fluid_name=carrier.fluid_name,
            representation="gas_carrier",
            mass_flow_rate=carrier.mass_flow_rate,
            expected_mass_flow_rate=expected[carrier.role],
            mass_flow_relative_error=mass_error,
            accounted=accounted,
            injected_liquid_mass=None,
            vaporized_liquid_mass=None,
            eta_vaporization=None,
            smd=None,
            geometry_model_id="prescribed_eulerian_carrier_field",
            blockers=tuple(blockers),
        ))
        carrier_payloads.append(MappingProxyType(carrier.to_dict()))

    target_fluids = {
        source.liquid.name.strip().lower() for source in liquids
    } | {carrier.fluid_name.strip().lower() for carrier in carriers}
    benchmark_items = []
    for dataset in benchmarks:
        report = benchmark_readiness_report(dataset)
        fluid_system = dataset.manifest.get("fluid_system", {})
        fixture_fluids = {
            str(fluid_system.get("radial_stream", "")).strip().lower(),
            str(fluid_system.get("axial_stream", "")).strip().lower(),
        }
        fixture_fluids.discard("")
        fluid_match = bool(fixture_fluids) and fixture_fluids == target_fluids
        blockers = report.blockers
        if not fluid_match:
            blockers += (
                "Benchmark fluid system does not match the current liquid/carrier fluids.",
            )
        benchmark_items.append(BenchmarkEvidence(
            case_id=dataset.case_id,
            source_sha256=dataset.source_sha256,
            validation_role=report.validation_role,
            strict_end_to_end_smd_validation_ready=(
                report.strict_end_to_end_smd_validation_ready
            ),
            fluid_system_match=fluid_match,
            tables_7_8_are_experimental=report.tables_7_8_are_experimental,
            blockers=blockers,
        ))
    benchmark_items = tuple(benchmark_items)
    benchmark_ready = bool(benchmark_items) and any(
        item.strict_end_to_end_smd_validation_ready and item.fluid_system_match
        for item in benchmark_items
    )
    all_streams = bool(streams) and all(stream.accounted for stream in streams)
    primary_ready = all(
        source.primary_path_eligible for source in liquids
    )
    convergence_ready = convergence_evidence is not None and convergence_evidence.passed

    gates = (
        SprayGate(
            "all_injector_streams_accounted", "pass" if all_streams else "fail",
            "every expected stream has a phase-correct representation and matching flow"
            if all_streams else "one or more expected streams fail representation/flow closure",
        ),
        SprayGate(
            "geometry_specific_primary_model", "pass" if primary_ready else "fail",
            "every liquid geometry has an applicable primary model"
            if primary_ready else "one or more liquid geometries are secondary-only blobs",
        ),
        SprayGate(
            "phase_and_critical_pressure_applicability", "fail",
            "LiquidProperties does not yet carry saturation/critical-state evidence",
        ),
        SprayGate(
            "time_step_and_parcel_count_convergence",
            "pass" if convergence_ready else "fail",
            "named refinement evidence satisfies its tolerance"
            if convergence_ready else "independent time-step and parcel-count refinement is absent/failing",
        ),
        SprayGate(
            "parcel_mass_conservation",
            "pass" if result.conservation.mass_closed else "fail",
            "represented parcel reservoirs close" if result.conservation.mass_closed
            else "represented parcel reservoirs do not close",
        ),
        SprayGate(
            "parcel_momentum_conservation",
            "pass" if result.conservation.parcel_momentum_closed else "fail",
            "parcel momentum/source demand closes" if result.conservation.parcel_momentum_closed
            else "parcel momentum/source demand does not close",
        ),
        SprayGate(
            "carrier_momentum_and_energy_closure", "fail",
            "the prescribed carrier is one-way and no droplet/carrier energy equation is solved",
        ),
        SprayGate(
            "strict_target_benchmark", "pass" if benchmark_ready else "fail",
            "strict target-case end-to-end SMD evidence is present"
            if benchmark_ready else "available fixtures do not provide strict target-case E2E SMD evidence",
        ),
        SprayGate(
            "non_regenerative_scope", "fail" if regenerative_cooling else "pass",
            "regen/cooling/pump duty is outside this inner parcel handoff"
            if regenerative_cooling else "handoff is explicitly limited to a non-regenerative cycle",
        ),
    )

    convergence_payload = (
        None if convergence_evidence is None
        else MappingProxyType(convergence_evidence.to_dict())
    )
    canonical = {
        "model_id": "radhakrishnan_bridge_one_way_parcel_handoff",
        "model_version": "1",
        "operating_point_id": operating_point,
        "smd_sampling_plane_m": smd_sampling_plane,
        "streams": [stream.to_dict() for stream in streams],
        "eta_vaporization": result.eta_vaporization,
        "conservation": result.conservation.to_dict(),
        "benchmarks": [item.to_dict() for item in benchmark_items],
        "carrier": [dict(item) for item in carrier_payloads],
        "solver_metadata": dict(result.solver_metadata),
        "convergence": None if convergence_payload is None else dict(convergence_payload),
        "gates": [gate.to_dict() for gate in gates],
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SprayCycleHandoff(
        model_id=canonical["model_id"],
        model_version=canonical["model_version"],
        operating_point_id=operating_point,
        smd_sampling_plane=float(smd_sampling_plane),
        streams=tuple(streams),
        eta_vaporization=result.eta_vaporization,
        aggregation_basis="vaporized_liquid_mass_over_injected_liquid_mass",
        conservation=MappingProxyType(result.conservation.to_dict()),
        benchmark_evidence=benchmark_items,
        carrier_provenance=tuple(carrier_payloads),
        solver_metadata=MappingProxyType(dict(result.solver_metadata)),
        convergence_evidence=convergence_payload,
        required_gates=gates,
        fingerprint=fingerprint,
    )


__all__ = [
    "BenchmarkEvidence",
    "GasCarrierStream",
    "NumericalConvergenceEvidence",
    "SprayCycleHandoff",
    "SprayStreamAccounting",
    "build_cycle_handoff",
]
