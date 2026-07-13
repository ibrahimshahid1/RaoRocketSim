"""Son-geometry and static-actuation contracts for movable pintles."""

from __future__ import annotations

import math

import pytest

from raosim.movable_pintle import (
    MovablePintleSpec,
    center_gap_area,
    discharge_coefficient_at_opening,
    minimum_opening_distance,
    movable_geometry_fingerprint,
    resolve_maximum_opening,
    solve_opening_for_mass_flow,
    son_minimum_tip_area,
    static_actuator_ledger,
    transition_opening,
)


def _son_spec(**kwargs):
    values = dict(
        post_diameter=8.0e-3,
        post_thickness=0.5e-3,
        center_gap_diameter=4.55e-3,
        pintle_rod_diameter=3.0e-3,
        transition_area_fraction=0.95,
    )
    values.update(kwargs)
    return MovablePintleSpec(**values)


@pytest.mark.parametrize(
    ("angle", "published_transition_mm"),
    ((0.0, 0.418), (20.0, 0.454), (40.0, 0.568)),
)
def test_son_transition_reproduces_published_table_2(angle, published_transition_mm):
    solved = transition_opening(_son_spec(), tip_angle_deg=angle)
    assert solved * 1.0e3 == pytest.approx(published_transition_mm, abs=0.0015)


def test_son_expanded_equation_matches_printed_equation_and_zero_angle_limit():
    opening = 0.3e-3
    angle = 20.0
    radius = 4.0e-3
    thickness = 0.5e-3
    theta = math.radians(angle)
    effective = radius - thickness
    printed = math.pi / math.sin(theta) * (
        effective**2
        - (effective - opening * math.sin(theta) * math.cos(theta)) ** 2
    )
    expanded = son_minimum_tip_area(
        opening,
        post_diameter=2.0 * radius,
        post_thickness=thickness,
        tip_angle_deg=angle,
    )
    zero = son_minimum_tip_area(
        opening,
        post_diameter=2.0 * radius,
        post_thickness=thickness,
        tip_angle_deg=0.0,
    )
    assert expanded == pytest.approx(printed, rel=2e-15)
    assert zero == pytest.approx(2.0 * math.pi * effective * opening)
    assert minimum_opening_distance(opening, angle) == pytest.approx(
        opening * math.cos(theta)
    )


def test_center_gap_is_annulus_around_pintle_rod():
    expected = math.pi / 4.0 * ((4.55e-3) ** 2 - (3.0e-3) ** 2)
    assert center_gap_area(4.55e-3, 3.0e-3) == pytest.approx(expected)
    with pytest.raises(ValueError, match="must exceed"):
        center_gap_area(3.0e-3, 4.0e-3)


def test_auto_open_stop_stays_below_center_gap_transition():
    spec = _son_spec(transition_area_fraction=0.9)
    maximum, transition, fraction = resolve_maximum_opening(
        spec, tip_angle_deg=20.0
    )
    assert 0.0 < maximum < transition
    assert fraction == pytest.approx(0.9, rel=2e-12)
    bad = _son_spec(maximum_opening=transition)
    with pytest.raises(ValueError, match="reaches/exceeds"):
        resolve_maximum_opening(bad, tip_angle_deg=20.0)


def test_cd_curve_is_interpolated_and_universal_fallback_is_labelled():
    calibrated = _son_spec(
        maximum_opening=0.4e-3,
        cd_vs_opening_fraction=((0.0, 0.60), (0.5, 0.70), (1.0, 0.80)),
        cd_calibration_source="configuration-controlled water cold-flow map",
        cd_calibration_artifact_sha256="b" * 64,
        cd_geometry_fingerprint_sha256="c" * 64,
    )
    cd, model, source = discharge_coefficient_at_opening(
        calibrated,
        opening_distance=0.3e-3,
        maximum_opening=0.4e-3,
        fallback_cd=0.5,
    )
    assert cd == pytest.approx(0.75)
    assert model == "linear_calibrated_cd_vs_opening_fraction"
    assert "cold-flow" in source

    cd, model, source = discharge_coefficient_at_opening(
        _son_spec(),
        opening_distance=0.1e-3,
        maximum_opening=0.3e-3,
        fallback_cd=0.71,
    )
    assert cd == pytest.approx(0.71)
    assert model == "constant_uncalibrated"
    assert source is None


def test_implicit_cd_opening_solve_closes_mass_flow_and_capacity_fails_closed():
    spec = _son_spec(
        maximum_opening=0.4e-3,
        cd_vs_opening_fraction=((0.0, 0.60), (1.0, 0.80)),
        cd_calibration_source="unit-test Cd curve",
        cd_calibration_artifact_sha256="b" * 64,
        cd_geometry_fingerprint_sha256="c" * 64,
    )
    flux_scale = 20_000.0
    solved = solve_opening_for_mass_flow(
        spec,
        tip_angle_deg=20.0,
        required_mass_flow=0.09,
        fallback_cd=0.7,
        mass_flux_for_cd=lambda cd: flux_scale * cd,
    )
    opening, area, cd = solved[:3]
    assert 0.0 < opening < spec.maximum_opening
    assert flux_scale * cd * area == pytest.approx(0.09, rel=1e-10)

    with pytest.raises(ValueError, match="open-stop capacity"):
        solve_opening_for_mass_flow(
            spec,
            tip_angle_deg=20.0,
            required_mass_flow=100.0,
            fallback_cd=0.7,
            mass_flux_for_cd=lambda cd: flux_scale * cd,
        )


def test_static_actuator_ledger_never_infers_missing_pressure_balance():
    incomplete = static_actuator_ledger(
        _son_spec(),
        pressure_drop=1.0e6,
        delivered_mass_flow=0.1,
        injection_velocity=20.0,
    )
    assert incomplete["pressure_force"] is None
    assert incomplete["required_actuator_force"] is None

    spec = _son_spec(
        unbalanced_pressure_area=2.0e-5,
        spring_preload_force=5.0,
        seal_friction_force=4.0,
        moving_mass=0.25,
        maximum_acceleration=20.0,
        actuator_force_capacity=100.0,
        force_safety_factor=1.5,
        stem_diameter=4.0e-3,
        stem_allowable_stress=200.0e6,
    )
    ledger = static_actuator_ledger(
        spec,
        pressure_drop=1.0e6,
        delivered_mass_flow=0.1,
        injection_velocity=20.0,
    )
    expected_required = 1.5 * (20.0 + 2.0 + 5.0 + 4.0 + 5.0)
    assert ledger["required_actuator_force"] == pytest.approx(expected_required)
    assert ledger["actuator_force_margin"] == pytest.approx(
        100.0 / expected_required
    )
    assert ledger["stem_stress_margin"] > 1.0


def test_mechanical_opening_and_vof_sheet_thickness_are_separate_fields():
    spec = _son_spec(
        commanded_opening=0.2e-3,
        sheet_thickness=0.125e-3,
        sheet_thickness_method="vof",
        sheet_thickness_source="Radhakrishnan 2021 VOF validation case",
        sheet_thickness_artifact_sha256="a" * 64,
    )
    assert spec.commanded_opening != spec.sheet_thickness


def test_geometry_fingerprint_is_deterministic_and_configuration_specific():
    base = _son_spec()
    first = movable_geometry_fingerprint(base, tip_angle_deg=20.0)
    second = movable_geometry_fingerprint(base, tip_angle_deg=20.0)
    changed = movable_geometry_fingerprint(
        _son_spec(post_thickness=0.51e-3), tip_angle_deg=20.0
    )

    assert first == second
    assert len(first) == 64
    assert changed != first
