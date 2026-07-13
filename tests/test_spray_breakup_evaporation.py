import math

import pytest

from raosim.spray.breakup import (
    BreakupParcelState,
    RayleighTaylorConfig,
    WAVE_COEFFICIENT_VARIANTS,
    WaveBreakupConfig,
    advance_breakup,
    calibrate_wave_constants_from_vof,
    compute_wave_metrics,
    gas_weber_number,
    kh_breakup_time,
    kh_stable_diameter,
    liquid_reynolds_number,
    liquid_weber_number,
    ohnesorge_number,
    relax_diameter_backward_euler,
    taylor_number,
)
from raosim.spray.evaporation import (
    EvaporationParcelState,
    advance_evaporation,
    diameter_squared_evaporation_rate,
    evaporation_rate_2021,
    gas_reynolds_number,
    schmidt_number,
    sherwood_number,
    spalding_mass_number,
)


WATER_AIR = {
    "liquid_density": 997.0,
    "liquid_dynamic_viscosity": 8.9e-4,
    "surface_tension": 0.072,
    "carrier_density": 1.2,
}


def test_wave_coefficient_variants_are_named_and_not_blended():
    canonical = WAVE_COEFFICIENT_VARIANTS["reitz_1987"]
    paper_2018 = WAVE_COEFFICIENT_VARIANTS["radhakrishnan_2018"]
    paper = WAVE_COEFFICIENT_VARIANTS["radhakrishnan_2021"]
    assert canonical.taylor_coefficient == 0.4
    assert canonical.weber_denominator_coefficient == 0.865
    assert paper_2018.taylor_coefficient == 0.4
    assert paper_2018.weber_denominator_coefficient == 0.87
    assert paper.taylor_coefficient == 0.45
    assert paper.weber_denominator_coefficient == 0.87


def test_reitz_dimensionless_groups_use_parent_radius_exactly():
    diameter = 2.0e-4
    speed = 100.0
    we_g = gas_weber_number(
        diameter=diameter,
        carrier_density=1.2,
        relative_speed=speed,
        surface_tension=0.07,
    )
    we_l = liquid_weber_number(
        diameter=diameter,
        liquid_density=1000.0,
        relative_speed=speed,
        surface_tension=0.07,
    )
    re_l = liquid_reynolds_number(
        diameter=diameter,
        liquid_density=1000.0,
        liquid_dynamic_viscosity=1.0e-3,
        relative_speed=speed,
    )
    oh = ohnesorge_number(
        radius=diameter / 2.0,
        liquid_density=1000.0,
        liquid_dynamic_viscosity=1.0e-3,
        surface_tension=0.07,
    )
    assert we_g == pytest.approx(1.2 * speed**2 * diameter / (2.0 * 0.07))
    assert we_l == pytest.approx(1000.0 * speed**2 * diameter / (2.0 * 0.07))
    assert re_l == pytest.approx(1000.0 * speed * (diameter / 2.0) / 1.0e-3)
    assert oh == pytest.approx(math.sqrt(we_l) / re_l)
    assert taylor_number(ohnesorge=oh, gas_weber=we_g) == pytest.approx(
        oh * math.sqrt(we_g)
    )


def test_wave_coefficient_variants_produce_distinct_wavelengths():
    common = dict(
        diameter=1.52e-4,
        liquid_density=1140.0,
        liquid_dynamic_viscosity=1.95e-4,
        surface_tension=0.01451,
        carrier_density=8.0,
        relative_speed=90.0,
        b0=3.97,
        b1=8.58,
    )
    canonical = compute_wave_metrics(
        **common, coefficient_variant="reitz_1987"
    )
    paper_2018 = compute_wave_metrics(
        **common, coefficient_variant="radhakrishnan_2018"
    )
    paper = compute_wave_metrics(
        **common, coefficient_variant="radhakrishnan_2021"
    )
    assert canonical.wavelength != pytest.approx(
        paper_2018.wavelength, rel=1.0e-8
    )
    assert canonical.wavelength != pytest.approx(paper.wavelength, rel=1.0e-8)
    assert canonical.coefficient_variant == "reitz_1987"
    assert paper_2018.coefficient_variant == "radhakrishnan_2018"
    assert paper.coefficient_variant == "radhakrishnan_2021"


def test_vof_calibration_uses_half_full_sheet_thickness_and_round_trips():
    calibration = calibrate_wave_constants_from_vof(
        full_sheet_thickness=1.52e-4,
        breakup_length=3.677e-3,
        liquid_velocity=25.0,
        liquid_density=1140.0,
        liquid_dynamic_viscosity=1.95e-4,
        surface_tension=0.01451,
        carrier_density=8.0,
        relative_speed=90.0,
        coefficient_variant="radhakrishnan_2021",
    )
    assert calibration.half_sheet_thickness == pytest.approx(0.76e-4)
    assert calibration.b0 == pytest.approx(
        calibration.half_sheet_thickness / calibration.wavelength
    )
    assert calibration.breakup_time == pytest.approx(
        calibration.breakup_length / calibration.liquid_velocity
    )
    reconstructed_tau = kh_breakup_time(
        radius=calibration.half_sheet_thickness,
        wavelength=calibration.wavelength,
        growth_rate=calibration.growth_rate,
        b1=calibration.b1,
    )
    reconstructed_stable_radius = 0.5 * kh_stable_diameter(
        wavelength=calibration.wavelength, b0=calibration.b0
    )
    assert reconstructed_tau == pytest.approx(calibration.breakup_time)
    assert reconstructed_stable_radius == pytest.approx(
        calibration.half_sheet_thickness
    )


def test_backward_euler_diameter_update_is_exact_discrete_formula():
    value = relax_diameter_backward_euler(
        diameter=2.0e-4,
        stable_diameter=5.0e-5,
        breakup_time=2.0e-3,
        dt=5.0e-4,
    )
    fraction = 5.0e-4 / 2.0e-3
    assert value == pytest.approx(
        (2.0e-4 + fraction * 5.0e-5) / (1.0 + fraction)
    )


def test_backward_euler_relaxation_converges_with_timestep_refinement():
    initial = 2.0e-4
    stable = 4.0e-5
    tau = 2.5e-3
    duration = 3.0e-3
    exact = stable + (initial - stable) * math.exp(-duration / tau)

    def march(steps: int) -> float:
        value = initial
        dt = duration / steps
        for _ in range(steps):
            value = relax_diameter_backward_euler(
                diameter=value,
                stable_diameter=stable,
                breakup_time=tau,
                dt=dt,
            )
        return value

    assert abs(march(80) - exact) < abs(march(20) - exact)


def test_kh_breakup_updates_multiplicity_and_conserves_mass_and_momentum():
    state = BreakupParcelState(
        diameter=2.0e-4,
        multiplicity=100.0,
        velocity=(13.0, -2.0, 0.5),
    )
    result = advance_breakup(
        state,
        dt=1.0e-4,
        **WATER_AIR,
        relative_speed=200.0,
        config=WaveBreakupConfig(
            b0=0.61, b1=10.0, coefficient_variant="reitz_1987"
        ),
    )
    assert result.event == "kelvin_helmholtz"
    assert result.state.diameter < state.diameter
    assert result.state.multiplicity > state.multiplicity
    ledger = result.conservation
    assert ledger.represented_mass_after == pytest.approx(
        ledger.represented_mass_before, rel=2.0e-15
    )
    assert ledger.mass_residual == pytest.approx(0.0, abs=1.0e-20)
    assert ledger.momentum_residual == pytest.approx((0.0, 0.0, 0.0), abs=1e-18)
    assert "not_global_carrier_closure" in ledger.carrier_coupling_status


def test_kh_breakup_is_inactive_below_radius_weber_threshold():
    state = BreakupParcelState(
        diameter=2.0e-4,
        multiplicity=10.0,
        velocity=(0.0, 0.0, 0.0),
    )
    result = advance_breakup(
        state,
        dt=1.0e-4,
        **WATER_AIR,
        relative_speed=20.0,
        config=WaveBreakupConfig(
            b0=0.61, b1=10.0, coefficient_variant="reitz_1987"
        ),
    )
    assert result.wave.gas_weber < 6.0
    assert result.event == "none"
    assert result.state == state


def test_optional_rt_branch_is_off_by_default_and_requires_explicit_forcing():
    state = BreakupParcelState(1.0e-3, 2.0, (1.0, 0.0, 0.0))
    default_result = advance_breakup(
        state,
        dt=2.0e-4,
        **WATER_AIR,
        relative_speed=1.0,
        config=WaveBreakupConfig(0.61, 10.0, "reitz_1987"),
    )
    assert default_result.rayleigh_taylor is None
    with pytest.raises(ValueError, match="effective_acceleration"):
        advance_breakup(
            state,
            dt=2.0e-4,
            **WATER_AIR,
            relative_speed=1.0,
            config=WaveBreakupConfig(
                0.61,
                10.0,
                "reitz_1987",
                rayleigh_taylor=RayleighTaylorConfig(enabled=True),
            ),
        )


def test_optional_rt_timer_branch_conserves_represented_mass():
    state = BreakupParcelState(1.0e-3, 2.0, (3.0, 1.0, -0.5))
    result = advance_breakup(
        state,
        dt=2.0e-4,
        **WATER_AIR,
        relative_speed=1.0,
        effective_acceleration=1.0e4,
        config=WaveBreakupConfig(
            0.61,
            10.0,
            "reitz_1987",
            rayleigh_taylor=RayleighTaylorConfig(enabled=True),
        ),
    )
    assert result.event == "rayleigh_taylor"
    assert result.rayleigh_taylor is not None
    assert result.rayleigh_taylor.wavelength < state.diameter
    assert result.state.diameter == pytest.approx(
        (state.diameter**2 * result.rayleigh_taylor.wavelength) ** (1.0 / 3.0)
    )
    assert result.state.rt_timer == -math.inf
    assert result.conservation.mass_residual == pytest.approx(0.0, abs=1e-18)


def test_spalding_mass_number_and_zero_driving_force():
    assert spalding_mass_number(
        surface_vapor_mass_fraction=0.2, bulk_vapor_mass_fraction=0.05
    ) == pytest.approx(0.15 / 0.8)
    assert spalding_mass_number(
        surface_vapor_mass_fraction=0.2, bulk_vapor_mass_fraction=0.2
    ) == 0.0


def _evaporation_rate(**overrides):
    values = dict(
        diameter=1.0e-3,
        multiplicity=100.0,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
        relative_speed=20.0,
        spalding_mass_number_value=0.2,
        closure="ranz_marshall_1952",
    )
    values.update(overrides)
    return evaporation_rate_2021(**values)


def test_evaporation_eq16_has_negative_mass_derivative_and_exact_coefficient():
    rate = _evaporation_rate()
    expected = (
        -rate.mass_transfer_coefficient
        * math.pi
        * (1.0e-3) ** 2
        * 1.2
        * math.log1p(0.2)
    )
    assert rate.mass_derivative_per_droplet == pytest.approx(expected)
    assert rate.represented_mass_derivative == pytest.approx(100.0 * expected)
    assert rate.mass_derivative_per_droplet < 0.0


def test_evaporation_zero_spalding_force_is_exactly_stationary():
    state = EvaporationParcelState(1.0e-3, 100.0, (5.0, 0.0, 0.0))
    result = advance_evaporation(
        state,
        dt=0.1,
        liquid_density=800.0,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
        relative_speed=20.0,
        spalding_mass_number_value=0.0,
        closure="ranz_marshall_1952",
    )
    assert result.rate_at_step_start.mass_derivative_per_droplet == -0.0
    assert result.diameter_squared_loss_rate == 0.0
    assert result.state == state
    assert result.conservation.vapor_mass_source_demand == 0.0


def test_named_sherwood_closures_and_dimensionless_groups():
    re = gas_reynolds_number(
        diameter=1.0e-3,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        relative_speed=20.0,
    )
    sc = schmidt_number(
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
    )
    assert re == pytest.approx(1333.3333333333333)
    assert sc == pytest.approx(0.75)
    assert sherwood_number(
        gas_reynolds=re, schmidt=sc, closure="quiescent_sphere"
    ) == 2.0
    assert sherwood_number(
        gas_reynolds=re, schmidt=sc, closure="ranz_marshall_1952"
    ) == pytest.approx(2.0 + 0.6 * math.sqrt(re) * sc ** (1.0 / 3.0))


def test_evaporation_is_monotonic_in_spalding_force_and_forced_convection():
    weak = _evaporation_rate(spalding_mass_number_value=0.05)
    strong = _evaporation_rate(spalding_mass_number_value=0.5)
    stagnant = _evaporation_rate(relative_speed=0.0)
    forced = _evaporation_rate(relative_speed=40.0)
    assert abs(strong.mass_derivative_per_droplet) > abs(
        weak.mass_derivative_per_droplet
    )
    assert abs(forced.mass_derivative_per_droplet) > abs(
        stagnant.mass_derivative_per_droplet
    )


def test_quiescent_sphere_has_analytic_d_squared_solution():
    state = EvaporationParcelState(1.0e-3, 5.0, (0.0, 0.0, 0.0))
    common = dict(
        liquid_density=800.0,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
        relative_speed=0.0,
        spalding_mass_number_value=0.2,
        closure="quiescent_sphere",
    )
    rate = diameter_squared_evaporation_rate(
        sherwood=2.0,
        mass_diffusivity=common["mass_diffusivity"],
        carrier_density=common["carrier_density"],
        liquid_density=common["liquid_density"],
        spalding_mass_number_value=common["spalding_mass_number_value"],
    )
    duration = 1.0
    one = advance_evaporation(state, dt=duration, **common)
    value = state
    for _ in range(100):
        value = advance_evaporation(value, dt=duration / 100.0, **common).state
    expected = math.sqrt(state.diameter**2 - rate * duration)
    assert one.state.diameter == pytest.approx(expected)
    assert value.diameter == pytest.approx(expected, rel=2e-14)


def test_reynolds_dependent_evaporation_converges_with_timestep_refinement():
    initial = EvaporationParcelState(1.0e-3, 5.0, (0.0, 0.0, 0.0))
    common = dict(
        liquid_density=800.0,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
        relative_speed=20.0,
        spalding_mass_number_value=0.2,
        closure="ranz_marshall_1952",
    )

    def march(steps: int) -> float:
        state = initial
        for _ in range(steps):
            state = advance_evaporation(
                state, dt=1.0 / steps, **common
            ).state
        return state.diameter

    reference = march(4000)
    assert abs(march(80) - reference) < abs(march(20) - reference)


def test_evaporation_conserves_liquid_plus_vapor_mass_and_source_momentum():
    state = EvaporationParcelState(1.0e-3, 100.0, (10.0, 2.0, -1.0))
    result = advance_evaporation(
        state,
        dt=0.1,
        liquid_density=800.0,
        carrier_density=1.2,
        carrier_dynamic_viscosity=1.8e-5,
        mass_diffusivity=2.0e-5,
        relative_speed=20.0,
        spalding_mass_number_value=0.2,
        closure="ranz_marshall_1952",
    )
    ledger = result.conservation
    assert result.state.diameter < state.diameter
    assert ledger.vapor_mass_source_demand > 0.0
    assert ledger.mass_residual == pytest.approx(0.0, abs=1e-20)
    assert ledger.momentum_residual == pytest.approx((0.0, 0.0, 0.0), abs=1e-18)
    assert ledger.carrier_momentum_source_demand == pytest.approx(
        tuple(ledger.vapor_mass_source_demand * v for v in state.velocity)
    )
    assert "not_global_carrier_closure" in ledger.carrier_coupling_status


@pytest.mark.parametrize(
    "call",
    [
        lambda: _evaporation_rate(mass_diffusivity=0.0),
        lambda: _evaporation_rate(carrier_density=0.0),
        lambda: _evaporation_rate(spalding_mass_number_value=-0.1),
        lambda: _evaporation_rate(closure="unnamed_correlation"),
    ],
)
def test_evaporation_rejects_missing_physics_disguised_as_invalid_defaults(call):
    with pytest.raises(ValueError):
        call()
