"""Typed liquid-parcel/gas-carrier cycle-handoff tests."""

import json

import numpy as np
import pytest

from raosim.spray.benchmarks import load_spray_benchmark
from raosim.spray.carrier import UniformCarrierField
from raosim.spray.domain import AxisymmetricDomain
from raosim.spray.handoff import (
    GasCarrierStream,
    NumericalConvergenceEvidence,
    build_cycle_handoff,
)
from raosim.spray.primary import RadialSheetGeometry, initialize_primary_parcels
from raosim.spray.solver import SprayMarchConfig, march_parcels
from raosim.spray.types import LiquidProperties, SpraySolverSpec


def _case():
    water = LiquidProperties(
        "water", 997.0, 8.9e-4, 0.072, 298.0, 2.0e5
    )
    source = initialize_primary_parcels(
        RadialSheetGeometry(1.0e-3, 1.0e-4, 0.0, 90.0),
        role="water",
        liquid=water,
        mass_flow_rate=0.02,
        injection_velocity=10.0,
        injection_duration=1.0e-3,
        parcel_count=4,
    )
    result = march_parcels(
        [source],
        carrier=UniformCarrierField(
            velocity=np.array([10.0, 0.0, 0.0]),
            density=1.2,
            dynamic_viscosity=1.8e-5,
            temperature=300.0,
            pressure=2.0e5,
        ),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=0.1, radius=0.02
        ),
        solver_spec=SpraySolverSpec(1.0e-5, 2.0e-4, 4, 0.15, seed=13),
        march_config=SprayMarchConfig(
            body_acceleration=(0.0, 0.0, 0.0),
            sampling_planes=(1.0e-3,),
            history_stride=2,
            mass_tolerance=1.0e-12,
            momentum_tolerance=1.0e-12,
        ),
    )
    carrier = GasCarrierStream(
        role="air",
        fluid_name="air",
        mass_flow_rate=0.01,
        composition_mass_fraction={"N2": 0.767, "O2": 0.233},
        continuity_relative_residual=1.0e-8,
        continuity_tolerance=1.0e-6,
        operating_point_id="water-air-1",
        field_fingerprint=result.solver_metadata[
            "carrier_field_fingerprint_sha256"
        ],
        continuity_source="manufactured continuity identity",
    )
    return source, result, carrier


def test_handoff_accounts_gas_as_carrier_without_fake_eta_or_smd():
    source, result, carrier = _case()
    table_case = load_spray_benchmark(
        "radhakrishnan2021_variable_area_lox_gch4"
    )
    handoff = build_cycle_handoff(
        result,
        liquid_sources=[source],
        gas_carriers=[carrier],
        expected_mass_flow_by_role={"water": 0.02, "air": 0.01},
        mass_flow_tolerance=1.0e-8,
        operating_point_id="water-air-1",
        smd_sampling_plane=1.0e-3,
        benchmarks=[table_case],
        convergence_evidence=NumericalConvergenceEvidence(
            0.005, 0.007, 0.01, ("base", "dt_half", "parcels_double")
        ),
        regenerative_cooling=False,
    )
    streams = {item.role: item for item in handoff.streams}
    assert handoff.all_streams_accounted
    assert streams["air"].representation == "gas_carrier"
    assert streams["air"].eta_vaporization is None
    assert streams["air"].smd is None
    assert streams["water"].representation == "liquid_parcels"
    assert streams["water"].smd == pytest.approx(1.0e-4)
    # Numerical accounting is not physical/cycle readiness.
    assert handoff.coupling_eligible is False
    failed = {gate.name for gate in handoff.required_gates if gate.status == "fail"}
    assert "carrier_momentum_and_energy_closure" in failed
    assert "phase_and_critical_pressure_applicability" in failed
    assert "strict_target_benchmark" in failed

    payload = handoff.to_dict()
    json.dumps(payload)
    assert len(payload["fingerprint"]) == 64
    evidence = payload["benchmark_evidence"][0]
    assert len(evidence["source_sha256"]) == 64
    assert evidence["tables_7_8_are_experimental"] is False
    assert evidence["fluid_system_match"] is False


def test_handoff_fingerprint_is_deterministic_and_operating_mismatch_blocks_stream():
    source, result, carrier = _case()
    kwargs = dict(
        result=result,
        liquid_sources=[source],
        gas_carriers=[carrier],
        expected_mass_flow_by_role={"water": 0.02, "air": 0.01},
        mass_flow_tolerance=1.0e-8,
        operating_point_id="water-air-1",
        smd_sampling_plane=1.0e-3,
        benchmarks=[],
        convergence_evidence=None,
        regenerative_cooling=False,
    )
    first = build_cycle_handoff(**kwargs)
    second = build_cycle_handoff(**kwargs)
    assert first.fingerprint == second.fingerprint

    wrong_carrier = GasCarrierStream(
        role="air",
        fluid_name="air",
        mass_flow_rate=0.01,
        composition_mass_fraction={"air": 1.0},
        continuity_relative_residual=0.0,
        continuity_tolerance=1.0e-6,
        operating_point_id="stale-point",
        field_fingerprint="sha256:stale",
        continuity_source="synthetic",
    )
    stale = build_cycle_handoff(**{**kwargs, "gas_carriers": [wrong_carrier]})
    assert stale.all_streams_accounted is False
    gas = next(item for item in stale.streams if item.role == "air")
    assert "different operating point" in gas.blockers[0]


def test_handoff_rejects_duck_typed_eta_and_role_or_sampling_mismatch():
    source, result, carrier = _case()
    common = dict(
        liquid_sources=[source],
        gas_carriers=[carrier],
        expected_mass_flow_by_role={"water": 0.02, "air": 0.01},
        mass_flow_tolerance=1.0e-8,
        operating_point_id="water-air-1",
        smd_sampling_plane=1.0e-3,
        benchmarks=[],
        convergence_evidence=None,
        regenerative_cooling=False,
    )
    with pytest.raises(TypeError, match="SprayMarchResult"):
        build_cycle_handoff(type("EtaOnly", (), {"eta_vaporization": 0.8})(), **common)
    with pytest.raises(ValueError, match="recorded solver plane"):
        build_cycle_handoff(result, **{**common, "smd_sampling_plane": 2.0e-3})
    with pytest.raises(ValueError, match="stream-role mismatch"):
        build_cycle_handoff(
            result,
            **{**common, "expected_mass_flow_by_role": {"water": 0.02}},
        )


def test_gas_carrier_composition_and_convergence_evidence_are_strict():
    with pytest.raises(ValueError, match="sum to one"):
        GasCarrierStream(
            "air", "air", 0.1, {"N2": 0.5, "O2": 0.4}, 0.0, 1e-6,
            "point", "hash", "source",
        )
    with pytest.raises(ValueError, match="at least three"):
        NumericalConvergenceEvidence(0.0, 0.0, 0.01, ("base", "refined"))
