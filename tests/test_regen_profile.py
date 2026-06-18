"""RegenWallProfile (station-wise wall/channel geometry) + helix-length Δp.

The profile carries the SP-125 four-thickness wall (t_hot, channel w/h,
land, jacket) as per-station arrays; helical channels lengthen the coolant
path, raising the Darcy-Weisbach Δp (SP-125 eq. 4-32) in proportion —
previously the helix changed only the exported STL.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
from raosim.propellants import custom_propellant
from raosim.regen_profile import RegenWallProfile, helix_passage_lengths


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=0.05, epsilon=10.0, gamma=1.24, length_pct=80.0)


@pytest.fixture(scope="module")
def heat(contour, prop):
    return bartz_heat_flux(contour, 7.0e6, prop, wall_temperature=900.0)


def _spec(**over):
    from types import SimpleNamespace
    d = dict(method="regenerative", coolant="rp1", channel_count=120,
             channel_width=0.0012, channel_height=0.003, coolant_mass_flow=12.0,
             coolant_cp=None, coolant_inlet_temperature=300.0,
             max_wall_temperature=1000.0, coolant_density=None,
             coolant_viscosity=None, coolant_conductivity=None)
    d.update(over)
    return SimpleNamespace(**d)


def _mat():
    from types import SimpleNamespace
    return SimpleNamespace(conductivity=285.0)


# --------------------------------------------------------------------- #
#  RegenWallProfile construction
# --------------------------------------------------------------------- #

def test_uniform_profile_broadcasts_scalars(contour):
    p = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                 channel_height=0.003, t_hot=0.001)
    n = len(contour["x"])
    for arr in (p.t_hot, p.channel_width, p.channel_height, p.land_width, p.t_jacket):
        assert arr.shape == (n,)
    assert p.is_uniform
    # jacket defaults to the liner thickness; land to the geometric rib
    assert np.allclose(p.t_jacket, p.t_hot)
    assert np.all(p.land_width >= 0.0)


def test_tapered_profile_varies_along_contour(contour):
    p = RegenWallProfile.tapered(
        contour, channel_count=120,
        throat={"t_hot": 0.0008, "channel_width": 0.0010, "channel_height": 0.0025},
        exit={"t_hot": 0.0015, "channel_width": 0.0020, "channel_height": 0.0040})
    assert not p.is_uniform
    # t_hot, w, h all span their throat→exit range
    assert p.t_hot.min() == pytest.approx(0.0008, abs=1e-6)
    assert p.t_hot.max() == pytest.approx(0.0015, abs=1e-6)
    assert p.channel_height.max() > p.channel_height.min()
    # the coolant velocity therefore varies along the passage
    V = p.coolant_velocity(12.0, 800.0)
    assert float(np.ptp(V)) > 0.0


def test_taper_defaults_do_not_apply_exit_geometry_upstream(contour):
    p = RegenWallProfile.tapered(
        contour, channel_count=120,
        throat={"t_hot": 0.0008, "channel_width": 0.0010, "channel_height": 0.0025},
        exit={"t_hot": 0.0015, "channel_width": 0.0020, "channel_height": 0.0040},
    )
    ti = int(np.argmin(contour["y"]))
    assert np.allclose(p.t_hot[:ti + 1], 0.0008)
    assert np.allclose(p.channel_width[:ti + 1], 0.0010)
    assert np.allclose(p.channel_height[:ti + 1], 0.0025)


def test_taper_accepts_partial_optional_land_mapping(contour):
    """Supplying an exit land alone uses a geometric throat anchor."""
    p = RegenWallProfile.tapered(
        contour, channel_count=120,
        throat={"t_hot": 0.0008, "channel_width": 0.0010, "channel_height": 0.0025},
        exit={"t_hot": 0.0015, "channel_width": 0.0020,
              "channel_height": 0.0040, "land_width": 0.0015},
    )
    assert np.all(np.isfinite(p.land_width))
    assert p.land_width[-1] == pytest.approx(0.0015)


def test_channels_fit_detects_overflow(contour):
    ok = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                  channel_height=0.003, t_hot=0.001)
    assert ok.channels_fit()["fits"] is True
    too_many = RegenWallProfile.uniform(contour, channel_count=600,
                                        channel_width=0.0012, channel_height=0.003,
                                        t_hot=0.001)
    fit = too_many.channels_fit()
    assert fit["fits"] is False
    assert fit["min_clearance_mm"] < 0.0


# --------------------------------------------------------------------- #
#  Helix passage length (the geometry that drives helix-length Δp)
# --------------------------------------------------------------------- #

def test_helix_passage_length_closed_form_on_a_cylinder():
    """On a straight cylinder (r const), the helix length factor is the
    exact closed form sqrt(1 + (2π·turns·r/L)²)."""
    L, R, turns = 0.4, 0.05, 2.0
    x = np.linspace(0.0, L, 200)
    r = np.full_like(x, R)
    dl, ds = helix_passage_lengths(x, r, helix_turns=turns)  # t_wall=h=0 -> r_mid=R
    factor = float(np.sum(dl) / np.sum(ds))
    expected = math.sqrt(1.0 + (2.0 * math.pi * turns * R / L) ** 2)
    assert factor == pytest.approx(expected, rel=1e-6)


def test_helix_factor_grows_with_turns(contour):
    p0 = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                  channel_height=0.003, t_hot=0.001, helix_turns=0.0)
    p1 = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                  channel_height=0.003, t_hot=0.001, helix_turns=1.0)
    p3 = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                  channel_height=0.003, t_hot=0.001, helix_turns=3.0)
    assert p0.passage_length_factor() == pytest.approx(1.0, abs=1e-9)
    assert p3.passage_length_factor() > p1.passage_length_factor() > 1.0
    assert p3.passage_length() > p3.meridional_length()


# --------------------------------------------------------------------- #
#  Helix-length Δp in the coupled cooling solve
# --------------------------------------------------------------------- #

def test_helix_raises_pressure_drop_proportionally(contour, heat, prop):
    axial = regenerative_cooling_analysis(heat, contour, _spec(), _mat(),
                                          0.001, prop, 7.0e6, helix_turns=0.0)
    helix = regenerative_cooling_analysis(heat, contour, _spec(), _mat(),
                                          0.001, prop, 7.0e6, helix_turns=2.0)
    assert axial["passage_length_factor"] == pytest.approx(1.0, abs=1e-9)
    assert helix["passage_length_factor"] > 1.5
    # Δp rises with the longer helical path (Darcy-Weisbach, SP-125 eq. 4-32).
    # Per-station friction (μ_b varies as the coolant heats along the path)
    # makes the rise an f-weighted version of the geometric path factor, so
    # it tracks the factor closely rather than matching it exactly.
    ratio = helix["coolant_pressure_drop"] / axial["coolant_pressure_drop"]
    assert ratio > 1.0
    assert ratio == pytest.approx(helix["pressure_drop_path_factor"], rel=1e-12)


def test_helix_does_not_change_wall_or_coolant_temperature(contour, heat, prop):
    """Energy conservation: routing the coolant helically lengthens the
    path (more Δp) but the total wall heat — and so the coolant ΔT and the
    gas-side wall temperature — is unchanged."""
    axial = regenerative_cooling_analysis(heat, contour, _spec(), _mat(),
                                          0.001, prop, 7.0e6, helix_turns=0.0)
    helix = regenerative_cooling_analysis(heat, contour, _spec(), _mat(),
                                          0.001, prop, 7.0e6, helix_turns=2.0)
    assert helix["peak_gas_side_wall_temperature"] == pytest.approx(
        axial["peak_gas_side_wall_temperature"], rel=1e-9)
    assert helix["coolant_outlet_temperature"] == pytest.approx(
        axial["coolant_outlet_temperature"], rel=1e-9)


def test_profile_cooling_matches_equivalent_scalar_call(contour, heat, prop):
    """A uniform profile run through .cooling() reproduces the scalar
    regenerative_cooling_analysis exactly (back-compat), helix included."""
    p = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                 channel_height=0.003, t_hot=0.001, helix_turns=2.0)
    via_profile = p.cooling(heat, contour, _mat(), _spec(), prop, 7.0e6)
    via_scalar = regenerative_cooling_analysis(heat, contour, _spec(), _mat(),
                                               0.001, prop, 7.0e6, helix_turns=2.0)
    assert via_profile["coolant_pressure_drop"] == pytest.approx(
        via_scalar["coolant_pressure_drop"], rel=1e-9)
    assert via_profile["peak_gas_side_wall_temperature"] == pytest.approx(
        via_scalar["peak_gas_side_wall_temperature"], rel=1e-9)


def test_station_wise_thickness_changes_wall_temperature(contour, heat, prop):
    """A thicker hot wall conducts worse (q t/k), so a tapered liner that
    is thicker at the throat runs a hotter throat than a thin uniform one —
    the thermal solve genuinely consumes t_hot(x)."""
    thin = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                    channel_height=0.003, t_hot=0.0008)
    thick = RegenWallProfile.uniform(contour, channel_count=120, channel_width=0.0012,
                                     channel_height=0.003, t_hot=0.0020)
    rt_thin = thin.cooling(heat, contour, _mat(), _spec(), prop, 7.0e6)
    rt_thick = thick.cooling(heat, contour, _mat(), _spec(), prop, 7.0e6)
    assert (rt_thick["peak_gas_side_wall_temperature"]
            > rt_thin["peak_gas_side_wall_temperature"])
