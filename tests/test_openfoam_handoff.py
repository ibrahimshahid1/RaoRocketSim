from dataclasses import FrozenInstanceError, replace
import math

import pytest

from raosim.openfoam.handoff import (
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
from raosim.spray.primary import RadialSheetGeometry


def _digest(character: str) -> str:
    return character * 64


def _study(kind: str, index: int, *, change: float = 0.01) -> VOFConvergenceStudy:
    return VOFConvergenceStudy(
        kind=kind,
        baseline_run_fingerprint_sha256=_digest(f"{index:x}"),
        refined_run_fingerprint_sha256=_digest(f"{index + 4:x}"),
        monitored_metric="sheet thickness, velocity, and liquid flux vector",
        relative_change=change,
        acceptance_tolerance=0.02,
        refinement_ratio=2.0,
    )


def _handoff(**changes) -> VOFToLagrangianHandoff:
    mass_flow = 0.2
    velocity = (3.0, 4.0)
    values = dict(
        provenance=VOFArtifactProvenance(
            case_id="water-air-opening-0p2mm",
            case_fingerprint_sha256=_digest("a"),
            input_fingerprint_sha256=_digest("b"),
            solver_name="foamRun/incompressibleVoF",
            solver_version="OpenFOAM-13 patch 20260624",
            solver_version_fingerprint_sha256=_digest("c"),
            extraction_code_fingerprint_sha256=_digest("d"),
        ),
        averaging_window=VOFAveragingWindow(
            start_time_s=0.02,
            end_time_s=0.04,
            sample_count=101,
            observed_flow_through_times=8.0,
            required_flow_through_times=5.0,
        ),
        extraction=VOFSheetExtractionDefinition(
            alpha_liquid_threshold=0.5,
            algorithm_id="connected-alpha-isocontour-thickness-v1",
            algorithm_version_fingerprint_sha256=_digest("e"),
            connected_component_policy="inlet-connected liquid only",
            interpolation_rule="linear face-to-isocontour interpolation",
        ),
        sheet=VOFSheetStatistics(
            exit_radius_m=0.004,
            axial_location_m=0.003,
            full_sheet_thickness_mean_m=0.0002,
            full_sheet_thickness_standard_deviation_m=5.0e-6,
            axial_velocity_mean_m_s=velocity[0],
            radial_velocity_mean_m_s=velocity[1],
            axial_velocity_standard_deviation_m_s=0.03,
            radial_velocity_standard_deviation_m_s=0.04,
            maximum_thickness_coefficient_of_variation=0.05,
            maximum_velocity_coefficient_of_variation=0.05,
        ),
        liquid_flux=VOFLiquidFluxBalance(
            liquid_name="water",
            liquid_density_kg_m3=998.2,
            inlet_mass_flow_rate_kg_s=mass_flow,
            extracted_mass_flow_rate_kg_s=mass_flow,
            inlet_momentum_flux_n=(mass_flow * velocity[0], mass_flow * velocity[1]),
            extracted_momentum_flux_n=(mass_flow * velocity[0], mass_flow * velocity[1]),
            mass_closure_relative_tolerance=0.01,
            momentum_closure_relative_tolerance=0.01,
            kinematic_momentum_relative_tolerance=0.01,
        ),
        carrier_field=CarrierAxisymmetricFieldEvidence(
            operating_point_id="water-air-opening-0p2mm",
            field_fingerprint_sha256=_digest("f"),
            state_fingerprint_sha256=_digest("1"),
            fluid_name="air",
            axial_bounds_m=(0.0, 0.08),
            radial_bounds_m=(0.0, 0.03),
            grid_shape_axial_radial=(201, 101),
            density_kg_m3=1.177,
            dynamic_viscosity_pa_s=1.846e-5,
            temperature_k=300.0,
            pressure_pa=101325.0,
            specific_heat_j_kg_k=1006.0,
            thermal_conductivity_w_m_k=0.0263,
            mean_axial_velocity_m_s=20.0,
            mean_radial_velocity_m_s=0.0,
            turbulent_kinetic_energy_m2_s2=0.1,
            turbulent_dissipation_rate_m2_s3=1.0,
        ),
        convergence=VOFConvergenceEvidence(
            mesh=_study("mesh", 1),
            time_step=_study("time_step", 2),
            domain=_study("domain", 3),
            averaging=_study("averaging", 4),
        ),
        declared_gates=(
            VOFHandoffGate(
                name="published_case_reproduction",
                passed=True,
                detail="target observable is inside the declared acceptance band",
                evidence_fingerprint_sha256=_digest("9"),
            ),
        ),
    )
    values.update(changes)
    return VOFToLagrangianHandoff(**values)


def test_complete_handoff_preserves_full_thickness_and_derives_angle():
    handoff = _handoff()

    assert handoff.ready_for_lagrangian
    assert not handoff.failed_required_gates
    geometry = handoff.to_radial_sheet_geometry()

    assert isinstance(geometry, RadialSheetGeometry)
    assert geometry.exit_radius == pytest.approx(0.004)
    assert geometry.sheet_thickness == pytest.approx(0.0002)
    assert geometry.opening_distance == pytest.approx(0.0002)
    assert geometry.tip_angle_deg == pytest.approx(math.degrees(math.atan2(3.0, 4.0)))


def test_contract_and_nested_inputs_are_immutable_and_fingerprint_is_deterministic():
    first = _handoff()
    second = _handoff()

    assert first.contract_fingerprint_sha256 == second.contract_fingerprint_sha256
    assert len(first.contract_fingerprint_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        first.sheet.exit_radius_m = 0.01
    with pytest.raises(FrozenInstanceError):
        first.declared_gates = ()


def test_mass_flux_failure_is_derived_and_blocks_conversion():
    flux = replace(
        _handoff().liquid_flux,
        extracted_mass_flow_rate_kg_s=0.18,
        extracted_momentum_flux_n=(0.54, 0.72),
    )
    handoff = _handoff(liquid_flux=flux)

    failed = {gate.name for gate in handoff.failed_required_gates}
    assert "liquid_mass_flux_closure" in failed
    with pytest.raises(VOFHandoffValidationError, match="liquid_mass_flux_closure"):
        handoff.to_radial_sheet_geometry()


def test_boundary_and_sheet_kinematic_momentum_closure_are_independent():
    boundary_bad = replace(
        _handoff().liquid_flux,
        inlet_momentum_flux_n=(0.6, 0.8),
        extracted_momentum_flux_n=(0.3, 0.4),
    )
    first = _handoff(liquid_flux=boundary_bad)
    first_failed = {gate.name for gate in first.failed_required_gates}
    assert "liquid_momentum_flux_closure" in first_failed
    assert "sheet_kinematic_momentum_closure" in first_failed

    # Boundary flux closes with itself, but is inconsistent with mdot * mean U.
    kinematic_bad = replace(
        _handoff().liquid_flux,
        inlet_momentum_flux_n=(0.3, 0.4),
        extracted_momentum_flux_n=(0.3, 0.4),
    )
    second = _handoff(liquid_flux=kinematic_bad)
    second_failed = {gate.name for gate in second.failed_required_gates}
    assert "liquid_momentum_flux_closure" not in second_failed
    assert "sheet_kinematic_momentum_closure" in second_failed


@pytest.mark.parametrize(
    ("replacement", "failed_gate"),
    [
        (
            lambda h: replace(
                h,
                averaging_window=replace(
                    h.averaging_window, observed_flow_through_times=2.0
                ),
            ),
            "averaging_window_coverage",
        ),
        (
            lambda h: replace(
                h,
                sheet=replace(
                    h.sheet,
                    full_sheet_thickness_standard_deviation_m=2.0e-5,
                ),
            ),
            "sheet_thickness_variation",
        ),
        (
            lambda h: replace(
                h,
                carrier_field=replace(
                    h.carrier_field, radial_bounds_m=(0.005, 0.03)
                ),
            ),
            "carrier_field_domain_coverage",
        ),
        (
            lambda h: replace(
                h,
                convergence=replace(
                    h.convergence,
                    mesh=replace(h.convergence.mesh, relative_change=0.03),
                ),
            ),
            "mesh_convergence",
        ),
    ],
)
def test_required_evidence_gates_fail_closed(replacement, failed_gate):
    handoff = replacement(_handoff())
    assert not handoff.ready_for_lagrangian
    assert failed_gate in {gate.name for gate in handoff.failed_required_gates}


def test_failed_declared_gate_blocks_but_failed_advisory_gate_does_not():
    required = VOFHandoffGate(
        name="experimental_sheet_validation",
        passed=False,
        detail="no matched image-derived thickness yet",
        evidence_fingerprint_sha256=_digest("8"),
    )
    blocked = _handoff(declared_gates=(required,))
    assert not blocked.ready_for_lagrangian

    advisory = replace(required, required=False)
    eligible = _handoff(declared_gates=(advisory,))
    assert eligible.ready_for_lagrangian


@pytest.mark.parametrize(
    "builder",
    [
        lambda: VOFArtifactProvenance(
            "case", "not-a-digest", _digest("a"), "solver", "13", _digest("b"), _digest("c")
        ),
        lambda: VOFSheetExtractionDefinition(
            1.0, "algorithm", _digest("a"), "connected", "linear"
        ),
        lambda: VOFSheetStatistics(
            0.004, 0.0, math.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.1
        ),
        lambda: CarrierAxisymmetricFieldEvidence(
            "op", _digest("a"), _digest("b"), "air", (0.0, 1.0), (0.0, 1.0),
            (10, 10), 1.0, 1e-5, 300.0, 1e5, 1000.0, 0.03, 1.0, 0.0, 1.0, 0.0
        ),
    ],
)
def test_malformed_or_nonfinite_evidence_is_rejected_at_construction(builder):
    with pytest.raises(VOFHandoffValidationError):
        builder()


def test_audit_payload_labels_si_values_and_reports_all_four_refinements():
    handoff = _handoff()
    payload = handoff.to_dict()

    assert payload["schema"] == "raosim.vof_to_lagrangian_handoff.v1"
    assert payload["sheet"]["full_sheet_thickness_mean_m"] == pytest.approx(2e-4)
    assert [study["kind"] for study in payload["convergence"]] == [
        "mesh",
        "time_step",
        "domain",
        "averaging",
    ]
    assert payload["ready_for_lagrangian"] is True
