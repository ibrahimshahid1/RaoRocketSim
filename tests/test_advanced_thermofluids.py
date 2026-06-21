from __future__ import annotations

import numpy as np
import pytest

from raosim.design import CoolingSpec, MaterialSpec
from raosim.materials import cyclic_tangent_modulus, get_material
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
from raosim.propellants import custom_propellant
from raosim.thermofluids import (
    real_fluid_coolant_state,
    solve_annular_manifold_network,
    spectral_band_radiative_heat_flux,
    zuber_critical_heat_flux,
)


def test_sourced_cyclic_tangent_modulus_softens_with_stress():
    material = get_material("grcop-84")
    low = cyclic_tangent_modulus(material, 20e6, 673.15)
    high = cyclic_tangent_modulus(material, 150e6, 673.15)
    assert low["available"] and high["available"]
    assert 0.0 < high["tangent_modulus"] < low["tangent_modulus"]
    assert low["tangent_modulus"] <= material.elastic_modulus


def test_spectral_band_radiation_is_positive_and_optically_bounded():
    thin = spectral_band_radiative_heat_flux(
        3000.0, 700.0, 0.01,
        [{"name": "h2o", "weight": 1.0, "absorption_coefficient": 2.0}],
    )
    thick = spectral_band_radiative_heat_flux(
        3000.0, 700.0, 1.0,
        [{"name": "h2o", "weight": 1.0, "absorption_coefficient": 2.0}],
    )
    black_body_limit = 5.670374419e-8 * (3000.0**4 - 700.0**4)
    assert 0.0 < thin["q"] < thick["q"] < black_body_limit


def test_full_annular_network_conserves_flow_and_resolves_every_branch():
    result = solve_annular_manifold_network(
        channel_count=32,
        ports_per_manifold=2,
        total_mass_flow=4.0,
        density=420.0,
        channel_pressure_drop=3.0e6,
        channel_total_area=1.2e-3,
        manifold_radius=0.06,
        plenum_area_ratio=0.25,
    )
    assert result["converged"]
    assert len(result["channel_mass_flows"]) == 32
    assert np.sum(result["channel_mass_flows"]) == pytest.approx(4.0, rel=1e-7)
    assert result["total_pressure_drop"] > 3.0e6
    assert result["minimum_channel_flow_ratio"] > 0.0


def test_real_fluid_backend_never_silently_falls_back():
    state = real_fluid_coolant_state(
        "methane", np.array([120.0]), np.array([8.0e6]), backend="auto"
    )
    if state["available"]:
        assert state["backend"] == "CoolProp_HEOS"
        assert state["rho"][0] > 0.0
    else:
        assert state["status"] == "coolprop_not_installed"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("ch4", "methane"), ("lch4", "methane"),
     ("lh2", "hydrogen"), ("h2", "hydrogen")],
)
def test_real_fluid_aliases_use_the_canonical_backend(alias, canonical):
    alias_state = real_fluid_coolant_state(
        alias, np.array([120.0]), np.array([8.0e6]), backend="auto"
    )
    canonical_state = real_fluid_coolant_state(
        canonical, np.array([120.0]), np.array([8.0e6]), backend="auto"
    )
    assert alias_state["status"] == canonical_state["status"]
    assert alias_state.get("fluid") == canonical_state.get("fluid")


def test_zuber_chf_increases_with_latent_heat():
    low = zuber_critical_heat_flux(
        latent_heat=1e5, vapor_density=5.0, liquid_density=400.0,
        surface_tension=0.01,
    )
    high = zuber_critical_heat_flux(
        latent_heat=2e5, vapor_density=5.0, liquid_density=400.0,
        surface_tension=0.01,
    )
    assert high == pytest.approx(2.0 * low)


def test_regen_integrates_radiation_and_full_hydraulic_network():
    contour = bell_nozzle_contour(
        Rt=0.04, epsilon=8.0, gamma=1.24, length_pct=80.0
    )
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    heat = bartz_heat_flux(contour, 3.0e6, prop)
    base = dict(
        method="regenerative", coolant="methane", channel_count=60,
        channel_width=8e-4, channel_height=2.5e-3,
        coolant_mass_flow=4.0, coolant_inlet_temperature=120.0,
        coolant_property_backend="constant",
        max_wall_temperature=1000.0,
    )
    convective = regenerative_cooling_analysis(
        heat, contour, CoolingSpec(**base), MaterialSpec.from_catalog("grcop-84"),
        8e-4, prop, 3.0e6, curvature_correction=False,
    )
    coupled = regenerative_cooling_analysis(
        heat,
        contour,
        CoolingSpec(
            **base,
            hydraulic_network=True,
            ports_per_manifold=2,
            plenum_area_ratio=0.5,
            radiation_model="leccese_gray",
            radiation_propellant_family="methane",
        ),
        MaterialSpec.from_catalog("grcop-84"),
        8e-4,
        prop,
        3.0e6,
        curvature_correction=False,
    )
    assert coupled["hydraulic_network_status"] == "full_channel_graph_converged"
    assert coupled["coolant_pressure_drop"] > coupled["channel_friction_pressure_drop"]
    assert np.max(coupled["radiative_heat_flux"]) > 0.0
    assert coupled["total_heat_load"] > convective["total_heat_load"]
