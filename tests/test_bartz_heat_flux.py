"""
Full Bartz (1957) gas-side heat-transfer correlation.

Pins the exact formula (σ factor, the pressure/area/curvature exponents,
the recovery temperature, the SI gas-property estimates) and the
physical behaviour (textbook throat heat-flux magnitude, peak near the
throat, σ and Mach trends).  Refs: Bartz, *Jet Propulsion* 27(1) 1957;
Huzel & Huang, NASA SP-125 §4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import (
    bartz_heat_flux,
    bartz_heat_transfer_coefficient,
    bartz_sigma,
    combustion_gas_viscosity,
    gas_transport_properties,
    prandtl_number_estimate,
)
from raosim.propellants import custom_propellant


GAMMA = 1.24


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=GAMMA, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=0.020, epsilon=10.0, gamma=GAMMA,
                               length_pct=80.0)


# ---------------------------------------------------------------------
#  Gas-property estimates (the documented Bartz defaults).
# ---------------------------------------------------------------------


def test_prandtl_eucken_estimate():
    assert prandtl_number_estimate(GAMMA) == pytest.approx(
        4.0 * GAMMA / (9.0 * GAMMA - 5.0))
    # bounded in (0, 1] for physical gases
    for g in (1.14, 1.2, 1.3, 1.4):
        assert 0.6 < prandtl_number_estimate(g) <= 1.0


def test_cp_from_gamma_and_R(prop):
    cp, Pr, mu = gas_transport_properties(prop)
    assert cp == pytest.approx(GAMMA * prop.R_gas / (GAMMA - 1.0))
    assert Pr == pytest.approx(prandtl_number_estimate(GAMMA))
    assert mu > 0.0


def test_huzel_huang_viscosity_si_magnitude():
    # Combustion gas at 3500 K, ~22 g/mol -> ~7.5e-5 Pa.s
    mu = combustion_gas_viscosity(3500.0, 22.0)
    assert mu == pytest.approx(7.49e-5, rel=0.02)
    # T^0.6 scaling
    assert (combustion_gas_viscosity(7000.0, 22.0)
            / combustion_gas_viscosity(3500.0, 22.0)) == pytest.approx(
        2.0 ** 0.6, rel=1e-6)


def test_property_overrides_are_used(prop):
    cp, Pr, mu = gas_transport_properties(prop, cp=2000.0, Pr=0.7, mu=1e-4)
    assert (cp, Pr, mu) == (2000.0, 0.7, 1e-4)


# ---------------------------------------------------------------------
#  The σ correction factor (exact formula).
# ---------------------------------------------------------------------


def test_sigma_exact_formula_omega_0p6():
    M, Tw, Tc = 1.0, 900.0, 3500.0
    f = 1.0 + 0.5 * (GAMMA - 1.0) * M * M
    base = 0.5 * (Tw / Tc) * f + 0.5
    expected = 1.0 / (base ** (0.8 - 0.6 / 5.0) * f ** (0.6 / 5.0))
    assert float(bartz_sigma(M, GAMMA, Tw, Tc)) == pytest.approx(expected)
    # ω=0.6 gives the classic 0.68 / 0.12 exponents
    assert 0.8 - 0.6 / 5.0 == pytest.approx(0.68)
    assert 0.6 / 5.0 == pytest.approx(0.12)


def test_sigma_decreases_downstream():
    # Higher Mach (further from throat) -> smaller σ.
    s = bartz_sigma(np.array([1.0, 2.0, 3.9]), GAMMA, 900.0, 3500.0)
    assert s[0] > s[1] > s[2]
    assert float(s[0]) == pytest.approx(1.3306, abs=1e-3)


# ---------------------------------------------------------------------
#  The per-station h_g (exponents) and the full distribution.
# ---------------------------------------------------------------------


def test_hg_pressure_exponent_is_0p8(prop):
    cp, Pr, mu = gas_transport_properties(prop)
    kw = dict(Dt=0.04, c_star=prop.c_star, cp=cp, Pr=Pr, mu=mu,
              gamma=GAMMA, Tc=prop.Tc, wall_temperature=900.0,
              throat_curvature_radius=0.00764)
    h1 = float(bartz_heat_transfer_coefficient(1.0, 1.0, Pc=7e6, **kw))
    h2 = float(bartz_heat_transfer_coefficient(1.0, 1.0, Pc=14e6, **kw))
    assert h2 / h1 == pytest.approx(2.0 ** 0.8, rel=1e-9)


def test_hg_area_ratio_exponent_is_0p9(prop):
    cp, Pr, mu = gas_transport_properties(prop)
    kw = dict(Dt=0.04, Pc=7e6, c_star=prop.c_star, cp=cp, Pr=Pr, mu=mu,
              gamma=GAMMA, Tc=prop.Tc, wall_temperature=900.0,
              throat_curvature_radius=0.00764)
    # At the same Mach, h_g ∝ (At/A)^0.9.
    h_full = float(bartz_heat_transfer_coefficient(3.0, 1.0, **kw))
    h_half = float(bartz_heat_transfer_coefficient(3.0, 0.5, **kw))
    assert h_half / h_full == pytest.approx(0.5 ** 0.9, rel=1e-9)


def test_throat_heat_flux_is_textbook_magnitude(contour, prop):
    """A 7 MPa engine should see O(10-100) MW/m² at the throat."""
    hf = bartz_heat_flux(contour, Pc=7.0e6, prop=prop,
                         wall_temperature=900.0)
    assert hf["model"] == "bartz_1957"
    assert 10e6 < hf["throat_q"] < 160e6
    assert 5e3 < hf["throat_h_g"] < 60e3       # W/(m²·K)
    # Peak heat flux sits at/near the throat (within a throat radius).
    assert abs(hf["x_q_max"]) < float(contour["Rt"])


def test_recovery_temperature_uses_cube_root_pr(contour, prop):
    hf = bartz_heat_flux(contour, Pc=7.0e6, prop=prop)
    _, Pr, _ = gas_transport_properties(prop)
    assert hf["recovery_factor"] == pytest.approx(Pr ** (1.0 / 3.0))
    # T_aw below Tc (recovery < 1) and above the wall temperature.
    assert 900.0 < hf["adiabatic_wall_temperature"] < prop.Tc


def test_mach_branch_subsonic_then_supersonic(contour, prop):
    hf = bartz_heat_flux(contour, Pc=7.0e6, prop=prop)
    x = np.asarray(contour["x"])
    y = np.asarray(contour["y"])
    Rt = float(contour["Rt"])
    M = np.asarray(hf["mach"])
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    assert M[throat_idx] == pytest.approx(1.0, abs=0.05)
    assert M[0] < 1.0                           # convergent inlet subsonic
    assert M[-1] > 1.5                           # exit supersonic
    # exit Mach near the isentropic area-ratio value for eps=10
    from raosim.gas_dynamics import mach_from_area_ratio
    assert M[-1] == pytest.approx(
        mach_from_area_ratio(10.0, GAMMA, supersonic=True), rel=0.05)


def test_backward_compatible_keys_present(contour, prop):
    """The cooling/structural screens read these keys."""
    hf = bartz_heat_flux(contour, Pc=7.0e6, prop=prop)
    for k in ("x", "q", "q_max", "x_q_max", "throat_q",
              "adiabatic_wall_temperature", "model"):
        assert k in hf
    assert np.asarray(hf["q"]).shape == np.asarray(contour["x"]).shape
