"""Geometry dispatch and conservation tests for primary parcel sources."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import numpy as np
import pytest

from raosim.spray.primary import (
    AxialAnnularSheetGeometry,
    PlanarSlotJetGeometry,
    RadialSheetGeometry,
    RoundHoleJetGeometry,
    initialize_primary_parcels,
    radial_sheet_geometry_from_injector_result,
)
from raosim.spray.types import LiquidProperties, SprayValidationError


@pytest.fixture
def water():
    return LiquidProperties(
        name="water",
        density=997.0,
        dynamic_viscosity=8.9e-4,
        surface_tension=0.072,
        temperature=298.15,
        pressure=2.0e5,
    )


def _initialize(geometry, water, **overrides):
    values = {
        "role": "oxidizer",
        "liquid": water,
        "mass_flow_rate": 2.0,
        "injection_velocity": 10.0,
        "injection_duration": 0.01,
        "parcel_count": 8,
    }
    values.update(overrides)
    return initialize_primary_parcels(geometry, **values)


def _represented_momentum(result):
    mass = result.cloud.represented_liquid_mass(result.liquid.density)
    return np.array([
        math.fsum((mass * result.cloud.velocity[:, component]).tolist())
        for component in range(3)
    ])


def _continuous_movable_result(
    *,
    opening=0.200e-3,
    thickness=0.125e-3,
    source="Radhakrishnan 2021 configuration-controlled VOF extraction",
    digest="a" * 64,
    method=None,
):
    """Minimal structural twin of InjectorDesignResult for adapter tests."""

    geometry_digest = "c" * 64
    detail = {
        "injection_form": "movable_pintle_radial_sheet",
        "opening_distance": opening,
        "post_diameter": 8.0e-3,
        "tip_angle_deg": 0.0,
        "sheet_thickness": thickness,
        "sheet_thickness_source": source,
        "sheet_thickness_artifact_sha256": digest,
        "sheet_thickness_geometry_fingerprint_sha256": geometry_digest,
        "resolved_geometry_fingerprint_sha256": geometry_digest,
        "sheet_thickness_fluid_name": "LOX",
        "sheet_thickness_opening_range": (0.15e-3, 0.25e-3),
        "sheet_thickness_pressure_drop_range": (0.5e6, 1.5e6),
        "sheet_thickness_mass_flow_range": (0.05, 0.15),
        # This screen is deliberately different and must never be selected.
        "equivalent_exit_sheet_thickness": 0.181e-3,
        "equivalent_exit_sheet_thickness_basis": (
            "continuity_equivalent_Amin_over_2piR_not_VOF_sheet_truth"
        ),
    }
    if method is not None:
        detail["sheet_thickness_method"] = method
    actuation = SimpleNamespace(
        opening_distance=opening,
        sheet_thickness=thickness,
        sheet_thickness_source=source,
        sheet_thickness_artifact_sha256=digest,
        sheet_thickness_geometry_fingerprint_sha256=geometry_digest,
        resolved_geometry_fingerprint_sha256=geometry_digest,
        sheet_thickness_fluid_name="LOX",
        sheet_thickness_opening_range=(0.15e-3, 0.25e-3),
        sheet_thickness_pressure_drop_range=(0.5e6, 1.5e6),
        sheet_thickness_mass_flow_range=(0.05, 0.15),
    )
    return SimpleNamespace(
        architecture="son_continuous_movable",
        feasible=True,
        slots=SimpleNamespace(
            geometry="continuous_radial_gap",
            role="oxidizer",
            dp=1.0e6,
            mdot=0.1,
            detail=detail,
        ),
        actuation=actuation,
        feed={
            "oxidizer": SimpleNamespace(
                name="LOX", liquid_ok=True, phase="liquid"
            )
        },
        gates=(
            SimpleNamespace(
                name="movable_sheet_thickness_handoff", status="pass"
            ),
        ),
    )


def test_radial_sheet_matches_movable_pintle_angle_and_full_thickness(water):
    geometry = RadialSheetGeometry(
        exit_radius=0.01,
        sheet_thickness=4.0e-4,
        axial_location=0.02,
        tip_angle_deg=30.0,
    )
    result = _initialize(geometry, water)

    assert geometry.opening_distance == pytest.approx(4.0e-4)
    assert result.cloud.diameter == pytest.approx(np.full(8, 4.0e-4))
    assert result.cloud.position[:, 0] == pytest.approx(0.02)
    assert np.linalg.norm(result.cloud.position[:, 1:], axis=1) == pytest.approx(
        np.full(8, 0.01)
    )
    assert result.cloud.velocity[:, 0] == pytest.approx(np.full(8, 5.0))
    assert np.linalg.norm(result.cloud.velocity[:, 1:], axis=1) == pytest.approx(
        np.full(8, 10.0 * math.cos(math.radians(30.0)))
    )


def test_injector_adapter_keeps_radhakrishnan_opening_and_sheet_distinct(water):
    geometry = radial_sheet_geometry_from_injector_result(
        _continuous_movable_result(), axial_location=0.012
    )
    result = _initialize(geometry, water)

    # Radhakrishnan, Lee & Koo (2021), Table 5: the mechanical opening and
    # VOF-resolved full sheet thickness are separate observables.
    assert geometry.mechanical_opening_distance == pytest.approx(0.200e-3)
    assert geometry.opening_distance == pytest.approx(0.200e-3)
    assert geometry.opening_distance_basis == "explicit_mechanical_pintle_opening"
    assert geometry.sheet_thickness == pytest.approx(0.125e-3)
    assert geometry.sheet_thickness_method == "vof"
    assert geometry.sheet_thickness_source.startswith("Radhakrishnan 2021")
    assert geometry.sheet_thickness_artifact_sha256 == "a" * 64
    assert result.cloud.diameter == pytest.approx(np.full(8, 0.125e-3))
    assert result.cloud.diameter[0] != pytest.approx(geometry.opening_distance)


def test_injector_adapter_accepts_explicit_measured_thickness_method():
    result = _continuous_movable_result(
        method="optical measurement",
        source="configuration-controlled cold-flow optical measurement",
        digest="B" * 64,
    )
    geometry = radial_sheet_geometry_from_injector_result(result)

    assert geometry.sheet_thickness_method == "measured"
    assert geometry.sheet_thickness_artifact_sha256 == "b" * 64


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda result: setattr(result, "architecture", "fixed_discrete"),
            "son_continuous_movable",
        ),
        (
            lambda result: setattr(result.slots, "geometry", "slots"),
            "continuous_radial_gap",
        ),
        (
            lambda result: setattr(result, "feasible", False),
            "feasible injector",
        ),
        (
            lambda result: setattr(result, "actuation", None),
            "actuation ledger",
        ),
        (
            lambda result: setattr(result, "gates", ()),
            "handoff gate",
        ),
        (
            lambda result: setattr(
                result.actuation, "sheet_thickness_artifact_sha256", "bad"
            ),
            "disagree",
        ),
        (
            lambda result: result.slots.detail.__setitem__(
                "sheet_thickness", 0.200e-3
            ),
            "disagree",
        ),
        (
            lambda result: setattr(result.slots, "dp", 2.0e6),
            "outside the evidence validity range",
        ),
        (
            lambda result: setattr(result.feed["oxidizer"], "name", "water"),
            "fluid does not match",
        ),
        (
            lambda result: result.slots.detail.__setitem__(
                "sheet_thickness_opening_range", (0.1e-3, 0.3e-3)
            ),
            "disagree",
        ),
        (
            lambda result: setattr(
                result.actuation,
                "sheet_thickness_geometry_fingerprint_sha256",
                "d" * 64,
            ),
            "disagree",
        ),
    ],
)
def test_injector_adapter_rejects_wrong_architecture_or_incomplete_ledger(
    mutate, match
):
    result = _continuous_movable_result()
    mutate(result)
    with pytest.raises(SprayValidationError, match=match):
        radial_sheet_geometry_from_injector_result(result)


@pytest.mark.parametrize(
    "method, source, digest, match",
    [
        (
            "correlation",
            "configuration-controlled empirical correlation",
            "a" * 64,
            "must be VOF-resolved or measured",
        ),
        (
            None,
            "continuity-equivalent A_min / (2 pi R) screen only",
            "a" * 64,
            "not a liquid-sheet handoff",
        ),
        (
            None,
            "configuration-controlled thickness artifact",
            "a" * 64,
            "missing an explicit method",
        ),
        (
            "vof",
            "configuration-controlled VOF extraction",
            "not-a-digest",
            "64 hexadecimal",
        ),
    ],
)
def test_injector_adapter_rejects_screen_or_ambiguous_thickness_evidence(
    method, source, digest, match
):
    result = _continuous_movable_result(
        method=method, source=source, digest=digest
    )
    with pytest.raises(SprayValidationError, match=match):
        radial_sheet_geometry_from_injector_result(result)


def test_radial_sheet_mass_and_momentum_close_to_machine_precision(water):
    geometry = RadialSheetGeometry(0.01, 2.0e-4, -0.03, 30.0)
    result = _initialize(geometry, water, parcel_count=64)
    expected_mass = 2.0 * 0.01
    expected_momentum = np.array([
        expected_mass * 10.0 * math.sin(math.radians(30.0)),
        0.0,
        0.0,
    ])

    assert result.injected_mass == pytest.approx(expected_mass, rel=0.0, abs=1e-16)
    assert result.represented_mass == pytest.approx(expected_mass, rel=0.0, abs=1e-16)
    assert abs(result.relative_mass_residual) <= 2.0e-16
    assert result.injected_momentum == pytest.approx(
        expected_momentum, rel=1e-14, abs=1e-16
    )
    assert _represented_momentum(result) == pytest.approx(result.injected_momentum)


def test_zero_degree_radial_sheet_is_perpendicular_to_axial_gas(water):
    result = _initialize(RadialSheetGeometry(0.01, 2e-4, 0.0, 0.0), water)
    assert result.cloud.velocity[:, 0] == pytest.approx(0.0)
    assert np.linalg.norm(result.cloud.velocity[:, 1:], axis=1) == pytest.approx(10.0)
    assert result.injected_momentum == pytest.approx(np.zeros(3), abs=1e-16)


def test_radial_sheet_initialization_is_deterministic_without_rng(water):
    geometry = RadialSheetGeometry(0.01, 2e-4, 0.0, 15.0)
    first = _initialize(geometry, water, parcel_count=20)
    second = _initialize(geometry, water, parcel_count=20)
    assert np.array_equal(first.cloud.position, second.cloud.position)
    assert np.array_equal(first.cloud.velocity, second.cloud.velocity)
    assert np.array_equal(
        first.cloud.statistical_weight, second.cloud.statistical_weight
    )
    assert np.array_equal(first.injected_momentum, second.injected_momentum)


def test_radial_sheet_is_the_only_primary_path_eligible_form(water):
    radial = _initialize(RadialSheetGeometry(0.01, 2e-4, 0.0, 0.0), water)
    assert radial.model.applicability_status == "literature_calibrated_wave_primary"
    assert radial.model.primary_path_eligible is True
    assert radial.primary_path_eligible is True
    assert {gate.name: gate.status for gate in radial.gates} == {
        "primary_source_geometry": "pass",
        "primary_model_applicability": "pass",
        "primary_cycle_coupling": "pass",
    }
    assert "radhakrishnan2021.pdf" in radial.model.local_source


@pytest.mark.parametrize(
    "geometry",
    [
        AxialAnnularSheetGeometry(0.01, 0.0104, 0.0),
        PlanarSlotJetGeometry(4, 0.01, 4e-4, 8e-4, 1e-3, 0.0),
        RoundHoleJetGeometry(4, 0.01, 5e-4, 1e-3, 0.0),
    ],
)
def test_current_injector_forms_are_secondary_only_and_block_coupling(
    geometry, water
):
    result = _initialize(geometry, water)
    assert result.model.applicability_status == "secondary_only_unvalidated_primary"
    assert result.model.primary_path_eligible is False
    assert result.primary_path_eligible is False
    gates = {gate.name: gate.status for gate in result.gates}
    assert gates["primary_source_geometry"] == "pass"
    assert gates["primary_model_applicability"] == "warn"
    assert gates["primary_cycle_coupling"] == "fail"
    assert "not_primary_breakup" in result.model.initialization_diameter_basis


def test_axial_annulus_uses_full_gap_and_downstream_axial_velocity(water):
    geometry = AxialAnnularSheetGeometry(
        inner_radius=0.01, outer_radius=0.0106, axial_location=-0.02
    )
    result = _initialize(geometry, water)
    assert result.cloud.diameter == pytest.approx(np.full(8, 6.0e-4))
    assert result.cloud.velocity == pytest.approx(
        np.tile([10.0, 0.0, 0.0], (8, 1))
    )
    assert np.linalg.norm(result.cloud.position[:, 1:], axis=1) == pytest.approx(
        geometry.mean_radius
    )
    assert result.injected_momentum == pytest.approx([0.2, 0.0, 0.0])


def test_planar_slots_use_hydraulic_blob_scale_and_equal_opening_population(water):
    geometry = PlanarSlotJetGeometry(
        slot_count=4,
        exit_radius=0.01,
        slot_width=4.0e-4,
        slot_height=8.0e-4,
        slot_length=1.2e-3,
        axial_location=0.0,
        cant_angle_deg=20.0,
    )
    result = _initialize(geometry, water, parcel_count=12)
    expected_dh = 2.0 * 4e-4 * 8e-4 / (4e-4 + 8e-4)
    assert result.cloud.diameter == pytest.approx(np.full(12, expected_dh))
    # Three parcels occupy each of four deterministic slot centers.
    assert np.unique(result.cloud.position[:, 1:], axis=0).shape[0] == 4
    assert result.injected_momentum[0] == pytest.approx(
        0.02 * 10.0 * math.sin(math.radians(20.0))
    )
    assert result.injected_momentum[1:] == pytest.approx(0.0, abs=1e-16)


def test_round_holes_use_full_orifice_diameter_and_discrete_symmetry(water):
    geometry = RoundHoleJetGeometry(
        hole_count=8,
        exit_radius=0.012,
        hole_diameter=5.0e-4,
        hole_length=1.5e-3,
        axial_location=0.003,
    )
    result = _initialize(geometry, water, parcel_count=16)
    assert result.cloud.diameter == pytest.approx(np.full(16, 5.0e-4))
    assert np.unique(result.cloud.position[:, 1:], axis=0).shape[0] == 8
    assert result.injected_momentum == pytest.approx(np.zeros(3), abs=1e-16)


@pytest.mark.parametrize(
    "factory, match",
    [
        (lambda: RadialSheetGeometry(0.0, 1e-4, 0.0, 0.0), "exit_radius"),
        (lambda: RadialSheetGeometry(1e-4, 2e-4, 0.0, 0.0), "thickness"),
        (lambda: RadialSheetGeometry(0.01, 1e-4, 0.0, -1.0), "tip_angle"),
        (lambda: AxialAnnularSheetGeometry(0.01, 0.009, 0.0), "outer_radius"),
        (
            lambda: PlanarSlotJetGeometry(0, 0.01, 1e-4, 1e-4, 1e-3, 0.0),
            "slot_count",
        ),
        (
            lambda: RoundHoleJetGeometry(4, 0.01, -1e-4, 1e-3, 0.0),
            "hole_diameter",
        ),
    ],
)
def test_geometry_validation_is_strict(factory, match):
    with pytest.raises(SprayValidationError, match=match):
        factory()


def test_source_request_rejects_bad_counts_and_inputs(water):
    with pytest.raises(SprayValidationError, match="must be even"):
        _initialize(
            RadialSheetGeometry(0.01, 1e-4, 0.0, 0.0),
            water,
            parcel_count=7,
        )
    with pytest.raises(SprayValidationError, match="divisible"):
        _initialize(
            RoundHoleJetGeometry(6, 0.01, 1e-4, 1e-3, 0.0),
            water,
            parcel_count=8,
        )
    with pytest.raises(SprayValidationError, match="mass_flow_rate"):
        _initialize(
            RadialSheetGeometry(0.01, 1e-4, 0.0, 0.0),
            water,
            mass_flow_rate=0.0,
        )
    with pytest.raises(SprayValidationError, match="unsupported primary geometry"):
        initialize_primary_parcels(
            object(), role="fuel", liquid=water, mass_flow_rate=1.0,
            injection_velocity=1.0, injection_duration=1.0, parcel_count=4,
        )


def test_result_and_geometry_arrays_are_immutable_and_serializable(water):
    geometry = RadialSheetGeometry(0.01, 1e-4, 0.0, 0.0)
    result = _initialize(geometry, water)
    with pytest.raises(FrozenInstanceError):
        geometry.exit_radius = 0.02
    with pytest.raises(ValueError):
        result.cloud.position[0, 0] = 10.0
    with pytest.raises(ValueError):
        result.injected_momentum[0] = 10.0
    report = result.to_dict()
    assert report["parcel_count"] == 8
    assert report["model"]["injection_form"] == "movable_pintle_radial_sheet"
    assert report["primary_path_eligible"] is True
