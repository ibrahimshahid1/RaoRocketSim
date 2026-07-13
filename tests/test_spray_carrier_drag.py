"""Deterministic contracts, carrier interpolation, drag, and dispersion tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from raosim.spray.carrier import (
    AxisymmetricRectilinearCarrierField,
    UniformCarrierField,
    carrier_field_fingerprint,
)
from raosim.spray.dispersion import DiscreteRandomWalk, DispersionState
from raosim.spray.drag import (
    drag_acceleration,
    particle_reynolds_number,
    schiller_naumann_drag_coefficient,
    stokes_relaxation_time,
)
from raosim.spray.types import (
    CarrierSample,
    LiquidProperties,
    ParcelCloud,
    SpraySolverSpec,
    SprayValidationError,
)


def _sample(count=1, *, velocity=(10.0, 0.0, 0.0), k=0.0, epsilon=0.0):
    return CarrierSample(
        velocity=np.broadcast_to(np.asarray(velocity, dtype=float), (count, 3)),
        density=np.full(count, 1.2),
        dynamic_viscosity=np.full(count, 1.8e-5),
        temperature=np.full(count, 500.0),
        pressure=np.full(count, 2.0e5),
        turbulent_kinetic_energy=np.full(count, k),
        turbulent_dissipation_rate=np.full(count, epsilon),
    )


def test_immutable_property_solver_and_cloud_contracts():
    liquid = LiquidProperties(
        name="water",
        density=997.0,
        dynamic_viscosity=8.9e-4,
        surface_tension=0.072,
        temperature=298.0,
        pressure=2.0e5,
    )
    assert liquid.latent_heat is None
    with pytest.raises(FrozenInstanceError):
        liquid.density = 1000.0

    spec = SpraySolverSpec(
        time_step=1.0e-5,
        maximum_time=2.1e-5,
        parcels_per_liquid_stream=4,
        eddy_lifetime_constant=0.15,
        seed=17,
    )
    assert spec.maximum_steps == 3
    assert spec.rng_metadata == {"seed": 17, "bit_generator": "PCG64"}

    source_position = np.zeros((2, 3))
    cloud = ParcelCloud(
        position=source_position,
        velocity=np.ones((2, 3)),
        diameter=np.array([1.0e-4, 2.0e-4]),
        temperature=np.array([298.0, 299.0]),
        statistical_weight=np.array([10.0, 20.0]),
        roles=("fuel", "oxidizer"),
    )
    source_position[0, 0] = 99.0
    assert cloud.position[0, 0] == 0.0
    assert cloud.position.flags.writeable is False
    assert cloud.active.tolist() == [True, True]
    expected = np.array([10.0, 20.0]) * 997.0 * math.pi / 6.0 * cloud.diameter**3
    assert cloud.represented_liquid_mass(997.0) == pytest.approx(expected)

    vaporized = ParcelCloud(
        position=np.zeros((1, 3)),
        velocity=np.zeros((1, 3)),
        diameter=np.zeros(1),
        temperature=np.full(1, 298.0),
        statistical_weight=np.ones(1),
        roles=("fuel",),
        active=np.zeros(1, dtype=bool),
    )
    assert vaporized.represented_liquid_mass(997.0)[0] == 0.0
    with pytest.raises(SprayValidationError, match="active parcels"):
        ParcelCloud(
            position=np.zeros((1, 3)),
            velocity=np.zeros((1, 3)),
            diameter=np.zeros(1),
            temperature=np.full(1, 298.0),
            statistical_weight=np.ones(1),
            roles=("fuel",),
        )


@pytest.mark.parametrize(
    "factory, match",
    [
        (
            lambda: LiquidProperties("", 1.0, 1.0, 1.0, 1.0, 1.0),
            "name",
        ),
        (
            lambda: LiquidProperties("x", 0.0, 1.0, 1.0, 1.0, 1.0),
            "density",
        ),
        (
            lambda: SpraySolverSpec(0.0, 1.0, 1, 0.1),
            "time_step",
        ),
        (
            lambda: CarrierSample(
                velocity=[0.0, 0.0, 0.0], density=1.0,
                dynamic_viscosity=1.0e-5, temperature=300.0, pressure=1.0e5,
                turbulent_kinetic_energy=1.0,
                turbulent_dissipation_rate=0.0,
            ),
            "requires positive dissipation",
        ),
    ],
)
def test_contract_validation_errors(factory, match):
    with pytest.raises(SprayValidationError, match=match):
        factory()


def test_uniform_carrier_broadcasts_without_aliasing():
    field = UniformCarrierField(
        velocity=np.array([12.0, -1.0, 0.5]),
        density=2.0,
        dynamic_viscosity=2.0e-5,
        temperature=700.0,
        pressure=4.0e5,
    )
    state = field.sample(np.zeros((5, 3)))
    assert state.velocity.shape == (5, 3)
    assert state.velocity == pytest.approx(np.tile([12.0, -1.0, 0.5], (5, 1)))
    assert state.density == pytest.approx(np.full(5, 2.0))
    assert state.velocity.flags.writeable is False
    assert len(carrier_field_fingerprint(field)) == 64
    changed = UniformCarrierField(
        velocity=np.array([12.0, -1.0, 0.6]),
        density=2.0,
        dynamic_viscosity=2.0e-5,
        temperature=700.0,
        pressure=4.0e5,
    )
    assert carrier_field_fingerprint(changed) != carrier_field_fingerprint(field)


def _manufactured_axisymmetric_field(policy="error"):
    x = np.array([0.0, 0.4, 1.0])
    r = np.array([0.0, 0.2, 0.5])
    X, R = np.meshgrid(x, r, indexing="ij")
    velocity = np.stack((2.0 + 3.0 * X - 4.0 * R, 5.0 * R, np.zeros_like(X)), axis=-1)
    return AxisymmetricRectilinearCarrierField(
        axial_coordinates=x,
        radial_coordinates=r,
        velocity_cylindrical=velocity,
        density=1.0 + 2.0 * X + 0.5 * R,
        dynamic_viscosity=1.0e-5 * (1.0 + X + R),
        temperature=300.0 + 20.0 * X + 10.0 * R,
        pressure=1.0e5 + 2.0e4 * X + 1.0e4 * R,
        turbulent_kinetic_energy=np.zeros_like(X),
        turbulent_dissipation_rate=np.zeros_like(X),
        out_of_domain_policy=policy,
    )


def test_axisymmetric_bilinear_interpolation_is_exact_for_linear_fields():
    field = _manufactured_axisymmetric_field()
    # Positive y makes the cylindrical radial basis equal Cartesian +y.
    points = np.array([[0.1, 0.1, 0.0], [0.7, 0.35, 0.0], [1.0, 0.5, 0.0]])
    state = field.sample(points)
    x = points[:, 0]
    r = points[:, 1]
    assert state.velocity[:, 0] == pytest.approx(2.0 + 3.0 * x - 4.0 * r)
    assert state.velocity[:, 1] == pytest.approx(5.0 * r)
    assert state.velocity[:, 2] == pytest.approx(0.0)
    assert state.density == pytest.approx(1.0 + 2.0 * x + 0.5 * r)
    assert state.temperature == pytest.approx(300.0 + 20.0 * x + 10.0 * r)


def test_axisymmetric_velocity_rotates_from_radial_to_cartesian_basis():
    field = _manufactured_axisymmetric_field()
    point = np.array([0.4, 0.0, 0.2])
    state = field.sample(point)
    assert state.velocity[1] == pytest.approx(0.0)
    assert state.velocity[2] == pytest.approx(1.0)


def test_axisymmetric_field_has_explicit_out_of_domain_policy():
    point = np.array([1.2, 0.8, 0.0])
    with pytest.raises(SprayValidationError, match="outside tabulated domain"):
        _manufactured_axisymmetric_field("error").sample(point)
    clipped = _manufactured_axisymmetric_field("clip").sample(point)
    boundary = _manufactured_axisymmetric_field("error").sample([1.0, 0.5, 0.0])
    assert clipped.velocity == pytest.approx(boundary.velocity)
    assert clipped.density == pytest.approx(boundary.density)


def test_axisymmetric_field_rejects_bad_grid_and_multivalued_axis_velocity():
    x = np.array([0.0, 1.0])
    r = np.array([0.0, 1.0])
    shape = (2, 2)
    velocity = np.zeros(shape + (3,))
    velocity[:, 0, 1] = 1.0
    kwargs = dict(
        axial_coordinates=x,
        radial_coordinates=r,
        velocity_cylindrical=velocity,
        density=np.ones(shape),
        dynamic_viscosity=np.ones(shape),
        temperature=np.ones(shape),
        pressure=np.ones(shape),
        turbulent_kinetic_energy=np.zeros(shape),
        turbulent_dissipation_rate=np.zeros(shape),
    )
    with pytest.raises(SprayValidationError, match="symmetry axis"):
        AxisymmetricRectilinearCarrierField(**kwargs)
    kwargs["velocity_cylindrical"] = np.zeros(shape + (3,))
    kwargs["axial_coordinates"] = np.array([0.0, 0.0])
    with pytest.raises(SprayValidationError, match="strictly increasing"):
        AxisymmetricRectilinearCarrierField(**kwargs)


def test_schiller_naumann_stokes_and_high_reynolds_limits():
    re = 1.0e-8
    expected = 24.0 / re * (1.0 + 0.15 * re**0.687)
    assert schiller_naumann_drag_coefficient(re) == pytest.approx(expected)
    assert math.isinf(schiller_naumann_drag_coefficient(0.0))
    assert schiller_naumann_drag_coefficient(1001.0) == pytest.approx(0.44)
    with pytest.raises(SprayValidationError, match=">= 0"):
        schiller_naumann_drag_coefficient(-1.0)


def test_drag_acceleration_recovers_corrected_stokes_limit_and_zero_slip():
    carrier = _sample(1, velocity=(1.0e-3, 0.0, 0.0))
    diameter = np.array([1.0e-7])
    rho_l = np.array([1000.0])
    parcel_velocity = np.zeros((1, 3))
    acceleration = drag_acceleration(parcel_velocity, diameter, rho_l, carrier)
    re = particle_reynolds_number(1.2, 1.8e-5, diameter[0], 1.0e-3)
    tau = stokes_relaxation_time(1000.0, diameter[0], 1.8e-5)
    expected_x = 1.0e-3 / tau * (1.0 + 0.15 * re**0.687)
    assert acceleration[0, 0] == pytest.approx(expected_x)
    assert acceleration[0, 1:] == pytest.approx(0.0)

    zero = drag_acceleration(carrier.velocity, diameter, rho_l, carrier)
    assert zero == pytest.approx(np.zeros((1, 3)))


def test_discrete_random_walk_is_seeded_and_preserves_eddies_until_expiry():
    carrier = _sample(4, k=1.5, epsilon=0.75)
    a = DiscreteRandomWalk(seed=42, eddy_lifetime_constant=0.2)
    b = DiscreteRandomWalk(seed=42, eddy_lifetime_constant=0.2)
    state = DispersionState.quiescent(4)
    first_a = a.advance(carrier, state, 0.01)
    first_b = b.advance(carrier, state, 0.01)
    assert first_a.velocity_fluctuation == pytest.approx(first_b.velocity_fluctuation)
    assert first_a.remaining_lifetime == pytest.approx(np.full(4, 0.4))
    assert a.rng_metadata == {"seed": 42, "bit_generator": "PCG64"}

    held = a.advance(carrier, first_a, 0.1)
    assert held.velocity_fluctuation == pytest.approx(first_a.velocity_fluctuation)
    assert held.remaining_lifetime == pytest.approx(np.full(4, 0.3))
    renewed = a.advance(carrier, held, 0.31)
    assert not np.array_equal(
        renewed.velocity_fluctuation, held.velocity_fluctuation
    )


def test_zero_turbulence_path_is_rng_free_and_exactly_quiescent():
    laminar = _sample(3, k=0.0, epsilon=0.0)
    turbulent = _sample(3, k=0.6, epsilon=0.3)
    state = DispersionState.quiescent(3)

    after_laminar = DiscreteRandomWalk(seed=9, eddy_lifetime_constant=0.15)
    fresh = DiscreteRandomWalk(seed=9, eddy_lifetime_constant=0.15)
    quiet = after_laminar.advance(laminar, state, 0.01)
    assert quiet.velocity_fluctuation == pytest.approx(np.zeros((3, 3)))
    assert quiet.remaining_lifetime == pytest.approx(np.zeros(3))

    draw_after_laminar = after_laminar.advance(turbulent, quiet, 0.01)
    direct_draw = fresh.advance(turbulent, state, 0.01)
    assert draw_after_laminar.velocity_fluctuation == pytest.approx(
        direct_draw.velocity_fluctuation
    )


def test_dispersion_validation_and_effective_velocity():
    with pytest.raises(SprayValidationError, match="seed"):
        DiscreteRandomWalk(seed=-1, eddy_lifetime_constant=0.1)
    with pytest.raises(SprayValidationError, match="eddy_lifetime_constant"):
        DiscreteRandomWalk(seed=1, eddy_lifetime_constant=0.0)

    carrier = _sample(2, velocity=(3.0, 0.0, 0.0))
    state = DispersionState(
        velocity_fluctuation=np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
        remaining_lifetime=np.ones(2),
    )
    effective = DiscreteRandomWalk.effective_carrier_velocity(carrier, state)
    assert effective == pytest.approx(
        np.array([[4.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    )
