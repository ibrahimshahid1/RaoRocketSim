"""
Cooling-coupled contour selection (raosim.thermal_design): the cooling
physics is computed BEFORE the contour is fixed and shapes the geometry
(throat curvature) + channels, vs the post-hoc screening in design.py.
"""
from __future__ import annotations

import math

import pytest

from raosim.design import CoolingSpec, MaterialSpec
from raosim.propellants import custom_propellant
from raosim.thermal_design import cooling_coupled_contour


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)


def _marginal_cooling():
    # Channels that fit the 20 mm throat (60 × 0.8 mm = 48 of 126 mm)
    # but with tight coolant flow + wall limit, so the target is NOT met
    # on the throat lever alone and the channel lever must engage.
    return CoolingSpec(
        method="regenerative", coolant="rp1", channel_count=60,
        channel_width=0.0008, channel_height=0.0020,
        coolant_mass_flow=2.5, coolant_inlet_temperature=300.0,
        max_wall_temperature=900.0,
    )


def _copper():
    return MaterialSpec(conductivity=350.0, max_temperature=1300.0)


def test_coupling_meets_reachable_target(prop):
    res = cooling_coupled_contour(
        0.020, 10.0, 7.0e6, prop, _marginal_cooling(), _copper(),
        cooling_margin_target=1.2,
    )
    assert res["feasible"] is True
    assert res["cooling_margin"] >= 1.2
    # The cooling state IS computed (coupled solve present).
    assert res["cooling"]["model"] == "sieder_tate_1d_regen"
    # The selected contour carries the chosen throat curvature.
    contour = res["contour"]
    assert contour["Rd"] == pytest.approx(res["Rd_factor"] * 0.020, rel=1e-6)


def test_channel_lever_is_binding_and_at_least_as_strong(prop):
    """The channel mass-flux lever is the dominant (binding) one: it
    engages after the throat sweep and delivers at least as much cooling
    margin as opening the throat.  (With the Level-1 fin + Dean model
    the throat lever also moves the Dean enhancement, so it is less
    dramatically weak than the bare (D*/r_c)^0.1 term alone — but the
    channels still carry the design and bind.)"""
    res = cooling_coupled_contour(
        0.020, 10.0, 7.0e6, prop, _marginal_cooling(), _copper(),
        cooling_margin_target=1.2,
    )
    hist = res["history"]
    throat_phase = [h for h in hist if h["channel_scale"] == pytest.approx(1.0)]
    margin_gain_throat = (throat_phase[-1]["cooling_margin"]
                          - throat_phase[0]["cooling_margin"])
    channel_phase = [h for h in hist if h["channel_scale"] < 0.999]
    assert channel_phase, "channel lever should have engaged"
    margin_gain_channel = (channel_phase[-1]["cooling_margin"]
                           - throat_phase[-1]["cooling_margin"])
    assert margin_gain_channel >= margin_gain_throat
    assert res["binding_lever"] == "channel_mass_flux"


def test_opening_throat_lowers_peak_wall_temperature(prop):
    """The Bartz (D*/r_c)^0.1 term dominates the throat sweep: a gentler
    throat lowers the peak flux and the peak wall temperature (the Dean
    enhancement weakens too, but net cooling)."""
    res = cooling_coupled_contour(
        0.020, 10.0, 7.0e6, prop, _marginal_cooling(), _copper(),
        cooling_margin_target=5.0,        # unreachable -> full sweep
        n_throat_steps=6,
    )
    throat_phase = [h for h in res["history"]
                    if h["channel_scale"] == pytest.approx(1.0)]
    peaks = [h["peak_wall_T"] for h in throat_phase]
    assert peaks[-1] < peaks[0]           # opening throat cools
    assert (peaks[0] - peaks[-1]) < 400.0  # but it is not the main lever


def test_infeasible_target_reports_binding_limit(prop):
    res = cooling_coupled_contour(
        0.020, 10.0, 7.0e6, prop, _marginal_cooling(), _copper(),
        cooling_margin_target=5.0,        # not reachable within bounds
    )
    assert res["feasible"] is False
    assert res["binding_lever"] in ("throat_curvature", "channel_mass_flux")
    # Still returns the best (lowest peak wall T) contour it found.
    assert res["peak_wall_temperature"] == pytest.approx(
        min(h["peak_wall_T"] for h in res["history"]), rel=1e-6)


def test_generous_cooling_meets_target_immediately(prop):
    # Channels already strong: target met at the first (un-opened) throat.
    strong = CoolingSpec(
        method="regenerative", coolant="rp1", channel_count=300,
        channel_width=0.0005, channel_height=0.0010,
        coolant_mass_flow=14.0, coolant_inlet_temperature=300.0,
        max_wall_temperature=1200.0,
    )
    res = cooling_coupled_contour(
        0.020, 10.0, 7.0e6, prop, strong, _copper(),
        cooling_margin_target=1.1,
    )
    assert res["feasible"] is True
    assert res["binding_lever"] == "throat_curvature"
    assert res["channel_scale"] == pytest.approx(1.0)
