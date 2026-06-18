"""
Cooling-coupled contour selection (raosim.thermal_design): the cooling
physics is computed BEFORE the contour is fixed and shapes the geometry
(throat curvature) + channels, vs the post-hoc screening in design.py.
"""
from __future__ import annotations

import math

import pytest

from raosim.design import CoolingSpec, MaterialSpec
from raosim.nozzle_geometry import bell_nozzle_contour
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
        coolant_mass_flow=3.5, coolant_inlet_temperature=300.0,
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


# ---------------------------------------------------------------------
#  Channel auto-sizing: solve N/w from the cooling requirement.
# ---------------------------------------------------------------------


def test_coolant_flow_from_cycle(prop):
    import math
    from raosim.thermal_design import coolant_flow_from_cycle
    Pc, Rt, MR = 7.0e6, 0.020, 2.6
    mc, mt = coolant_flow_from_cycle(Pc, Rt, prop.c_star, MR)
    At = math.pi * Rt * Rt
    assert mt == pytest.approx(Pc * At / prop.c_star, rel=1e-9)   # c* identity
    assert mc == pytest.approx(mt / (1.0 + MR), rel=1e-9)         # fuel split
    # half-cooling fraction halves the coolant flow
    mc2, _ = coolant_flow_from_cycle(Pc, Rt, prop.c_star, MR, cooling_fraction=0.5)
    assert mc2 == pytest.approx(0.5 * mc, rel=1e-9)
    richer_fuel_flow, _ = coolant_flow_from_cycle(Pc, Rt, prop.c_star, 2.0)
    leaner_fuel_flow, _ = coolant_flow_from_cycle(Pc, Rt, prop.c_star, 3.0)
    assert richer_fuel_flow > leaner_fuel_flow


def test_sizing_finds_feasible_design_for_large_engine(prop):
    from raosim.thermal_design import size_cooling_channels
    c = bell_nozzle_contour(Rt=0.080, epsilon=10.0, gamma=1.24, length_pct=80.0)
    r = size_cooling_channels(c, prop, 7.0e6, margin_target=1.2,
                              dp_budget_bar=300.0, wall_temp_limit=1100.0,
                              mixture_ratio=2.6, channel_height=0.004,
                              w_max=0.004)
    assert r["feasible"] is True
    assert r["channel_count"] is not None and r["channel_count"] > 0
    assert r["channel_width"] is not None
    # The sized design actually meets the requirement.
    assert r["margin"] >= 1.2
    assert r["pressure_drop_bar"] <= 300.0
    # Coolant flow came from the cycle, not an input.
    assert r["mdot_cool"] == pytest.approx(r["mdot_total"] / 3.6, rel=1e-6)


def test_sizing_reports_infeasible_with_diagnosis(prop):
    """A small engine cooled by its own ~1.4 kg/s fuel can't hit margin
    1.2 at any geometry — the sizer says so (honest physics)."""
    from raosim.thermal_design import size_cooling_channels
    c = bell_nozzle_contour(Rt=0.020, epsilon=10.0, gamma=1.24, length_pct=80.0)
    r = size_cooling_channels(c, prop, 7.0e6, margin_target=1.2,
                              dp_budget_bar=300.0, wall_temp_limit=1100.0,
                              mixture_ratio=2.6)
    assert r["feasible"] is False
    assert "unmet" in r["diagnosis"] or "budget" in r["diagnosis"]
    # It still reports the best-effort closest design.
    assert r["channel_count"] is not None


def test_sizing_objective_changes_the_design(prop):
    from raosim.thermal_design import size_cooling_channels
    c = bell_nozzle_contour(Rt=0.080, epsilon=10.0, gamma=1.24, length_pct=80.0)
    kw = dict(margin_target=1.2, dp_budget_bar=300.0, wall_temp_limit=1100.0,
              mixture_ratio=2.6, channel_height=0.004, w_max=0.004)
    min_dp = size_cooling_channels(c, prop, 7.0e6, objective="min_dp", **kw)
    max_m = size_cooling_channels(c, prop, 7.0e6, objective="max_margin", **kw)
    assert min_dp["feasible"] and max_m["feasible"]
    # min_dp gives the lowest pressure drop; max_margin the highest margin.
    assert min_dp["pressure_drop_bar"] <= max_m["pressure_drop_bar"] + 1e-6
    assert max_m["margin"] >= min_dp["margin"] - 1e-6
