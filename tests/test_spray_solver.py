"""End-to-end deterministic parcel-march and conservation tests."""

import numpy as np
import pytest

from raosim.spray.breakup import WaveBreakupConfig
from raosim.spray.carrier import UniformCarrierField
from raosim.spray.domain import AxisymmetricDomain
from raosim.spray.primary import (
    AxialAnnularSheetGeometry,
    RadialSheetGeometry,
    initialize_primary_parcels,
)
from raosim.spray.solver import (
    EvaporationModelConfig,
    SprayMarchConfig,
    march_parcels,
)
from raosim.spray.types import LiquidProperties, SpraySolverSpec


WATER = LiquidProperties(
    name="water",
    density=997.0,
    dynamic_viscosity=8.9e-4,
    surface_tension=0.072,
    temperature=298.0,
    pressure=2.0e5,
)


def _axial_source(*, gap=1.0e-4, speed=10.0, count=4):
    return initialize_primary_parcels(
        AxialAnnularSheetGeometry(
            inner_radius=1.0e-3,
            outer_radius=1.0e-3 + gap,
            axial_location=0.0,
        ),
        role="oxidizer",
        liquid=WATER,
        mass_flow_rate=0.02,
        injection_velocity=speed,
        injection_duration=1.0e-3,
        parcel_count=count,
    )


def _spec(*, dt=1.0e-4, maximum=2.0e-3, seed=7):
    return SpraySolverSpec(
        time_step=dt,
        maximum_time=maximum,
        parcels_per_liquid_stream=4,
        eddy_lifetime_constant=0.15,
        seed=seed,
    )


def _march_config(*, planes=(), stride=1):
    return SprayMarchConfig(
        body_acceleration=(0.0, 0.0, 0.0),
        sampling_planes=tuple(planes),
        history_stride=stride,
        mass_tolerance=1.0e-12,
        momentum_tolerance=1.0e-12,
        strict_conservation=True,
    )


def _carrier(*, velocity=(10.0, 0.0, 0.0), k=0.0, epsilon=0.0):
    return UniformCarrierField(
        velocity=np.asarray(velocity),
        density=1.2,
        dynamic_viscosity=1.8e-5,
        temperature=500.0,
        pressure=2.0e5,
        turbulent_kinetic_energy=k,
        turbulent_dissipation_rate=epsilon,
    )


def test_ballistic_outlet_march_closes_mass_momentum_and_samples_smd():
    source = _axial_source()
    result = march_parcels(
        [source],
        carrier=_carrier(),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=1.0e-2, radius=1.0e-2
        ),
        solver_spec=_spec(),
        march_config=_march_config(planes=(5.0e-3,), stride=2),
    )
    assert result.terminal_reason == ("outlet",) * 4
    assert result.conservation.mass_closed
    assert result.conservation.parcel_momentum_closed
    assert not result.conservation.globally_momentum_closed
    assert result.eta_vaporization == 0.0
    sample = result.sampling_planes[5.0e-3]
    assert sample.count == 4
    assert sample.statistics("oxidizer").sauter_mean_diameter == pytest.approx(
        source.cloud.diameter[0]
    )
    assert result.statistics(role="oxidizer", reservoir="outlet") is not None
    assert result.to_dict()["all_streams_accounted"] is False
    assert result.coupling_eligible is False


def test_radial_sheet_hits_wall_at_segment_boundary_and_is_conserved():
    source = initialize_primary_parcels(
        RadialSheetGeometry(
            exit_radius=1.0e-3,
            sheet_thickness=1.0e-4,
            axial_location=0.0,
            tip_angle_deg=0.0,
        ),
        role="water",
        liquid=WATER,
        mass_flow_rate=0.01,
        injection_velocity=10.0,
        injection_duration=1.0e-3,
        parcel_count=4,
    )
    result = march_parcels(
        [source],
        carrier=_carrier(velocity=(0.0, 0.0, 0.0)),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=1.0e-2, radius=2.0e-3
        ),
        solver_spec=_spec(dt=1.0e-5, maximum=5.0e-4),
        march_config=_march_config(),
    )
    assert result.terminal_reason == ("wall",) * 4
    assert result.conservation.mass_closed
    assert result.conservation.parcel_momentum_closed
    assert np.hypot(
        result.final_cloud.position[:, 1], result.final_cloud.position[:, 2]
    ) == pytest.approx(np.full(4, 2.0e-3))


def test_source_on_wall_directed_outward_has_exact_zero_residence_event():
    source = initialize_primary_parcels(
        RadialSheetGeometry(1.0e-3, 1.0e-4, 0.0, 0.0),
        role="water",
        liquid=WATER,
        mass_flow_rate=0.01,
        injection_velocity=10.0,
        injection_duration=1.0e-3,
        parcel_count=4,
    )
    result = march_parcels(
        [source],
        carrier=_carrier(velocity=(0.0, 0.0, 0.0)),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=0.1, radius=1.0e-3
        ),
        solver_spec=_spec(dt=1.0e-4, maximum=1.0e-3),
        march_config=_march_config(),
    )
    assert result.terminal_reason == ("wall",) * 4
    assert result.final_cloud.age == pytest.approx(np.zeros(4))
    ledger = result.conservation.per_role["water"]
    assert ledger.drag_impulse_on_parcels == pytest.approx((0.0, 0.0, 0.0))
    assert result.conservation.mass_closed
    assert result.conservation.parcel_momentum_closed


def test_wave_breakup_changes_diameter_and_multiplicity_without_mass_loss():
    source = initialize_primary_parcels(
        RadialSheetGeometry(1.0e-3, 2.0e-4, 0.0, 90.0),
        role="water",
        liquid=WATER,
        mass_flow_rate=0.01,
        injection_velocity=10.0,
        injection_duration=1.0e-3,
        parcel_count=4,
    )
    result = march_parcels(
        [source],
        carrier=_carrier(velocity=(200.0, 0.0, 0.0)),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=0.1, radius=0.02
        ),
        solver_spec=_spec(dt=1.0e-5, maximum=5.0e-5),
        march_config=_march_config(),
        breakup_by_role={
            "water": WaveBreakupConfig(
                b0=0.61, b1=10.0, coefficient_variant="reitz_1987"
            )
        },
    )
    assert np.all(result.final_cloud.diameter < source.cloud.diameter)
    assert np.all(
        result.final_cloud.statistical_weight > source.cloud.statistical_weight
    )
    assert result.breakup_event_counts["water"]["kelvin_helmholtz"] > 0
    assert result.conservation.mass_closed
    assert result.conservation.parcel_momentum_closed


def test_evaporation_uses_reservoir_ledger_and_allows_zero_terminal_diameter():
    source = _axial_source(gap=5.0e-5, speed=1.0)
    result = march_parcels(
        [source],
        carrier=_carrier(velocity=(1.0, 0.0, 0.0)),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=1.0, radius=0.02
        ),
        solver_spec=_spec(dt=1.0e-4, maximum=1.0e-3),
        march_config=_march_config(),
        evaporation_by_role={
            "oxidizer": EvaporationModelConfig(
                mass_diffusivity=1.0e-3,
                spalding_mass_number=1.0,
                sherwood_closure="quiescent_sphere",
            )
        },
    )
    assert result.terminal_reason == ("vaporized",) * 4
    assert result.final_cloud.diameter == pytest.approx(np.zeros(4))
    assert result.eta_vaporization == pytest.approx(1.0)
    assert result.eta_vaporization_by_role["oxidizer"] == pytest.approx(1.0)
    assert result.conservation.mass_closed
    assert result.conservation.parcel_momentum_closed
    energy_gate = next(
        gate for gate in result.gates if gate.name == "droplet_and_carrier_energy"
    )
    assert energy_gate.status == "fail"


def test_turbulent_dispersion_repeats_with_seed_and_changes_with_new_seed():
    source = _axial_source(speed=10.0)
    kwargs = dict(
        initializations=[source],
        carrier=_carrier(velocity=(10.0, 0.0, 0.0), k=2.0, epsilon=20.0),
        domain=AxisymmetricDomain.cylinder(
            axial_start=0.0, axial_end=1.0, radius=0.1
        ),
        march_config=_march_config(),
    )
    first = march_parcels(
        **kwargs, solver_spec=_spec(dt=1.0e-5, maximum=2.0e-5, seed=44)
    )
    repeated = march_parcels(
        **kwargs, solver_spec=_spec(dt=1.0e-5, maximum=2.0e-5, seed=44)
    )
    changed = march_parcels(
        **kwargs, solver_spec=_spec(dt=1.0e-5, maximum=2.0e-5, seed=45)
    )
    assert np.array_equal(first.final_cloud.position, repeated.final_cloud.position)
    assert np.array_equal(first.final_cloud.velocity, repeated.final_cloud.velocity)
    assert not np.array_equal(first.final_cloud.velocity, changed.final_cloud.velocity)
    assert first.solver_metadata["bit_generator"] == "PCG64"


def test_model_roles_sampling_planes_and_numerical_controls_are_strict():
    source = _axial_source()
    with pytest.raises(ValueError, match="unknown liquid roles"):
        march_parcels(
            [source],
            carrier=_carrier(),
            domain=AxisymmetricDomain.cylinder(
                axial_start=0.0, axial_end=0.1, radius=0.02
            ),
            solver_spec=_spec(),
            march_config=_march_config(),
            breakup_by_role={
                "fuel": WaveBreakupConfig(0.61, 10.0, "reitz_1987")
            },
        )
    with pytest.raises(ValueError, match="sampling plane"):
        march_parcels(
            [source],
            carrier=_carrier(),
            domain=AxisymmetricDomain.cylinder(
                axial_start=0.0, axial_end=0.1, radius=0.02
            ),
            solver_spec=_spec(),
            march_config=_march_config(planes=(0.2,)),
        )
