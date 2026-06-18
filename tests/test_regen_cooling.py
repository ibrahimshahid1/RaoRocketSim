"""
Regenerative cooling: Sieder-Tate coolant side + coupled 1-D wall
conduction (replaces the old ad-hoc ``h_cool = 1000 + 2e8·area`` film
model).  Pins the Sieder-Tate correlation, the series thermal circuit,
and the physically-correct design trends.

Refs: Sieder & Tate, *Ind. Eng. Chem.* 28, 1936; Huzel & Huang, NASA
SP-125 §4.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import (
    bartz_heat_flux,
    coolant_viscosity,
    hydraulic_diameter,
    regenerative_cooling_analysis,
    regenerative_cooling_screen,
    resolve_coolant_properties,
    sieder_tate_coefficient,
)
from raosim.propellants import custom_propellant


GAMMA = 1.24


class _Cool:
    # Geometrically valid at a 20 mm throat: 100 × 0.6 mm channels need
    # 60 mm of the 126 mm circumference (0.66 mm lands), so the fin
    # model is well-posed.
    method = "regenerative"
    coolant = "rp1"
    channel_count = 100
    channel_width = 0.0006
    channel_height = 0.0025
    coolant_mass_flow = 12.0
    coolant_cp = 2010.0
    coolant_inlet_temperature = 300.0
    max_wall_temperature = 1200.0
    coolant_density = None
    coolant_viscosity = None
    coolant_conductivity = None


class _Mat:
    conductivity = 350.0          # copper
    max_temperature = 1300.0


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=GAMMA, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=0.020, epsilon=10.0, gamma=GAMMA,
                               length_pct=80.0)


@pytest.fixture(scope="module")
def heat(contour, prop):
    return bartz_heat_flux(contour, 7.0e6, prop, wall_temperature=900.0)


# ---------------------------------------------------------------------
#  Sieder-Tate correlation (exact formula).
# ---------------------------------------------------------------------


def test_hydraulic_diameter_rectangular():
    assert hydraulic_diameter(0.001, 0.001) == pytest.approx(0.001)  # square
    assert hydraulic_diameter(0.002, 0.001) == pytest.approx(
        2.0 * 0.002 * 0.001 / 0.003)


def test_sieder_tate_exact_nusselt():
    props = {"k": 0.12, "cp": 2010.0}
    G, D_h = 20000.0, 0.001
    mu_b, mu_w = 1.6e-3, 1.0e-3
    h_c = float(sieder_tate_coefficient(G, D_h, props,
                                        mu_bulk=mu_b, mu_wall=mu_w))
    Re = G * D_h / mu_b
    Pr = mu_b * props["cp"] / props["k"]
    Nu = 0.027 * Re ** 0.8 * Pr ** (1.0 / 3.0) * (mu_b / mu_w) ** 0.14
    assert h_c == pytest.approx(Nu * props["k"] / D_h, rel=1e-9)


def test_sieder_tate_reynolds_and_velocity_scaling():
    props = {"k": 0.12, "cp": 2010.0}
    kw = dict(D_h=0.001, props=props, mu_bulk=1.6e-3, mu_wall=1.6e-3)
    h1 = float(sieder_tate_coefficient(10000.0, **kw))
    h2 = float(sieder_tate_coefficient(20000.0, **kw))
    # h_c ∝ Re^0.8 ∝ G^0.8 at fixed D_h/properties
    assert h2 / h1 == pytest.approx(2.0 ** 0.8, rel=1e-9)


def test_viscosity_ratio_term_enhances_when_wall_hotter():
    props = {"k": 0.12, "cp": 2010.0}
    kw = dict(mass_flux=20000.0, D_h=0.001, props=props, mu_bulk=1.6e-3)
    # Heated wall (lower μ_w for a liquid) -> (μ_b/μ_w)^0.14 > 1.
    hot = float(sieder_tate_coefficient(mu_wall=1.0e-3, **kw))
    iso = float(sieder_tate_coefficient(mu_wall=1.6e-3, **kw))
    assert hot > iso
    assert hot / iso == pytest.approx((1.6e-3 / 1.0e-3) ** 0.14, rel=1e-9)


def test_andrade_viscosity_drops_with_temperature():
    props = resolve_coolant_properties(_Cool())
    assert coolant_viscosity(props, 600.0) < coolant_viscosity(props, 300.0)
    # explicit μ override disables the T-model
    class C(_Cool):
        coolant_viscosity = 1e-3
    p2 = resolve_coolant_properties(C())
    assert coolant_viscosity(p2, 600.0) == pytest.approx(1e-3)


def test_coolant_property_lookup_and_override():
    props = resolve_coolant_properties(_Cool())
    assert props["cp"] == pytest.approx(2010.0)   # RP-1 table
    assert props["k"] == pytest.approx(0.12)

    class C(_Cool):
        coolant = "water"
    assert resolve_coolant_properties(C())["k"] == pytest.approx(0.61)

    class D(_Cool):
        coolant_conductivity = 0.5
    assert resolve_coolant_properties(D())["k"] == pytest.approx(0.5)


# ---------------------------------------------------------------------
#  Coupled 1-D solve (the series thermal circuit + correct trends).
# ---------------------------------------------------------------------


def test_series_thermal_circuit_consistency(contour, prop, heat):
    res = regenerative_cooling_analysis(heat, contour, _Cool(), _Mat(),
                                        0.001, prop, 7.0e6)
    assert res["model"] == "sieder_tate_1d_regen"
    q = np.asarray(res["q"])
    T_wg = np.asarray(res["gas_side_wall_temperature"])
    T_wc = np.asarray(res["coolant_side_wall_temperature"])
    T_c = np.asarray(res["coolant_temperature"])
    h_c = np.asarray(res["h_c"])
    # Wall hotter on the gas side than the coolant side than the bulk.
    assert np.all(T_wg >= T_wc - 1e-6)
    assert np.all(T_wc >= T_c - 1e-6)
    # Coolant-side film balance q = h_c (T_wc − T_c) holds per station.
    np.testing.assert_allclose(q, h_c * (T_wc - T_c), rtol=1e-6, atol=1.0)
    # Wall-conduction balance q = (k/t)(T_wg − T_wc).
    k_over_t = _Mat.conductivity / 0.001
    np.testing.assert_allclose(q, k_over_t * (T_wg - T_wc), rtol=1e-6, atol=1.0)


def test_more_coolant_flow_lowers_wall_temperature(contour, prop, heat):
    def wall(mdot):
        class C(_Cool):
            coolant_mass_flow = mdot
        return regenerative_cooling_analysis(
            heat, contour, C(), _Mat(), 0.001, prop, 7.0e6
        )["peak_gas_side_wall_temperature"]
    assert wall(20.0) < wall(8.0)


def test_smaller_channels_lower_wall_temperature(contour, prop, heat):
    """At fixed total flow, smaller channels = faster coolant = higher
    h_c = cooler wall (the trend the old ad-hoc film model got wrong)."""
    def wall(w, h):
        class C(_Cool):
            channel_width = w
            channel_height = h
        return regenerative_cooling_analysis(
            heat, contour, C(), _Mat(), 0.001, prop, 7.0e6
        )["peak_gas_side_wall_temperature"]
    assert wall(0.0004, 0.0008) < wall(0.0010, 0.0020)


def test_coupling_reduces_heat_flux_vs_cold_wall(contour, prop, heat):
    """q = h_g·(T_aw − T_wg): a COLDER reference wall gives a higher
    flux than the coupled solution, whose wall sits well above it.
    (Using a 500 K reference; the coupled peak wall ≈ 860 K with this
    finned design, so the direction is unambiguous.)"""
    res = regenerative_cooling_analysis(heat, contour, _Cool(), _Mat(),
                                        0.001, prop, 7.0e6)
    cold = bartz_heat_flux(contour, 7.0e6, prop, wall_temperature=500.0)
    assert res["peak_gas_side_wall_temperature"] > 500.0
    assert float(np.max(res["q"])) < float(cold["q_max"])


def test_coolant_heats_along_channel(contour, prop, heat):
    res = regenerative_cooling_analysis(heat, contour, _Cool(), _Mat(),
                                        0.001, prop, 7.0e6)
    assert res["coolant_outlet_temperature"] > _Cool.coolant_inlet_temperature
    assert res["coolant_temperature_rise"] > 0.0
    # Energy balance: rise ≈ total_heat / (mdot·cp).
    expected = res["total_heat_load"] / (
        _Cool.coolant_mass_flow * _Cool.coolant_cp)
    assert res["coolant_temperature_rise"] == pytest.approx(expected, rel=0.1)


def test_screen_dispatches_to_coupled_solve_when_prop_given(contour, prop, heat):
    coupled = regenerative_cooling_screen(heat, contour, _Cool(), _Mat(),
                                          0.001, prop, 7.0e6)
    assert coupled["model"] == "sieder_tate_1d_regen"
    # legacy 2-arg form still works (peak-flux estimate, no NaNs)
    legacy = regenerative_cooling_screen(heat, contour, _Cool(), _Mat(),
                                         0.001)
    assert legacy["model"] == "sieder_tate_peak_flux_estimate"
    assert math.isfinite(legacy["estimated_wall_temperature"])


def test_non_regenerative_returns_adiabatic(contour, heat):
    class C(_Cool):
        method = "none"
    out = regenerative_cooling_screen(heat, contour, C(), _Mat(), 0.001)
    assert out["method"] == "none"
    assert out["cooling_margin"] == 0.0


# ---------------------------------------------------------------------
#  Level 1: fin efficiency + Dean curvature + Darcy-Weisbach.
# ---------------------------------------------------------------------


def test_fin_efficiency_limits():
    from raosim.physics import fin_efficiency
    # Short, thick, high-k fin -> η_f → 1.
    assert float(fin_efficiency(20e3, 350.0, 0.001, 0.0003)) > 0.95
    # Tall, thin fin -> η_f well below 1 (tip runs hot).
    assert float(fin_efficiency(80e3, 350.0, 0.0002, 0.004)) < 0.5


def test_curvature_factor_sign_convention():
    from raosim.physics import curvature_correction_factor
    concave = float(curvature_correction_factor(3e4, 0.001, +0.02))
    convex = float(curvature_correction_factor(3e4, 0.001, -0.02))
    assert concave > 1.0           # enhancement on the concave (throat) wall
    assert convex < 1.0            # degradation on the convex wall
    assert concave * convex == pytest.approx(1.0, rel=1e-9)


def test_darcy_friction_factor_blasius():
    from raosim.physics import darcy_friction_factor
    Re = 3e4
    assert float(darcy_friction_factor(Re)) == pytest.approx(
        0.316 * Re ** -0.25, rel=1e-9)
    # Laminar branch below 2300.
    assert float(darcy_friction_factor(1000.0)) == pytest.approx(64.0 / 1000.0)


def test_level1_corrections_improve_cooling(contour, prop, heat):
    """Fin (land) area + Dean curvature lower the wall temperature vs the
    bare 1-D model (the conservatism the bare model carried)."""
    bare = regenerative_cooling_analysis(
        heat, contour, _Cool(), _Mat(), 0.001, prop, 7.0e6,
        fin_correction=False, curvature_correction=False)
    finned = regenerative_cooling_analysis(
        heat, contour, _Cool(), _Mat(), 0.001, prop, 7.0e6)
    assert finned["fidelity"] == "1d_finned"
    assert (finned["peak_gas_side_wall_temperature"]
            < bare["peak_gas_side_wall_temperature"])
    # Fin enhancement > 1 (tall channels add coolant side-wall area).
    import numpy as _np
    assert float(_np.max(finned["fin_area_factor"])) > 1.0


def test_pressure_drop_is_reported_and_scales(contour, prop, heat):
    res = regenerative_cooling_analysis(heat, contour, _Cool(), _Mat(),
                                        0.001, prop, 7.0e6)
    assert res["coolant_pressure_drop"] > 0.0
    assert res["channel_velocity"] > 0.0
    # Doubling the flow raises Δp (≈ V² ∝ mdot²).
    class C2(_Cool):
        coolant_mass_flow = 2.0 * _Cool.coolant_mass_flow
    res2 = regenerative_cooling_analysis(heat, contour, C2(), _Mat(),
                                         0.001, prop, 7.0e6)
    assert res2["coolant_pressure_drop"] > res["coolant_pressure_drop"]


def test_absolute_coolant_pressure_march_sets_local_liner_load(contour, prop, heat):
    outlet = 8.5e6
    res = regenerative_cooling_analysis(
        heat, contour, _Cool(), _Mat(), 0.001, prop, 7.0e6,
        coolant_outlet_pressure=outlet,
    )
    order = np.asarray(res["coolant_flow_order"], dtype=int)
    p_flow = np.asarray(res["coolant_pressure"])[order]
    # Pressure decreases monotonically from jacket inlet to outlet.
    assert np.all(np.diff(p_flow) <= 1e-6)
    assert res["coolant_inlet_pressure"] == pytest.approx(
        outlet + res["coolant_pressure_drop"], rel=1e-12)
    assert res["coolant_outlet_pressure"] == pytest.approx(outlet)
    np.testing.assert_allclose(
        res["liner_pressure_differential"],
        np.asarray(res["coolant_pressure"]) - np.asarray(res["gas_pressure"]),
    )
    assert res["coolant_pressure_boundary_source"] == (
        "user_supplied_coolant_outlet_pressure"
    )


def test_infeasible_channel_geometry_warns(contour, prop, heat):
    """Channels that exceed the throat circumference are flagged."""
    class TooMany(_Cool):
        channel_count = 400          # 400 × 0.6 mm = 240 mm > 126 mm
    res = regenerative_cooling_analysis(heat, contour, TooMany(), _Mat(),
                                        0.001, prop, 7.0e6)
    assert any("do not fit" in w for w in res["warnings"])


# ---------------------------------------------------------------------
#  Level 2: quasi-2-D wall cross-section (the land hot spot).
# ---------------------------------------------------------------------


def _wide_land_cooling(N=20, w=0.0020):
    # Few wide channels -> wide lands -> a real circumferential hot spot.
    return type("C", (), dict(
        method="regenerative", coolant="rp1", channel_count=N,
        channel_width=w, channel_height=0.0025, coolant_mass_flow=8.0,
        coolant_cp=2010.0, coolant_inlet_temperature=300.0,
        max_wall_temperature=1200.0, coolant_density=None,
        coolant_viscosity=None, coolant_conductivity=None))()


def test_2d_resolves_land_hot_spot(contour, prop, heat):
    from raosim.physics import regenerative_cooling_2d

    class Inc:
        conductivity = 16.0           # Inconel: low k -> visible gradient
    res = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), Inc(),
                                  0.0015, prop, 7.0e6, n_s=20, n_xi=20)
    assert res["model"] == "regen_2d_cross_section"
    ti = int(np.argmin(np.abs(np.asarray(contour["y"]) - 0.02)))
    # The land midpoint runs hotter than the channel centre.
    assert res["T_wg_land"][ti] > res["T_wg_channel"][ti]
    assert res["throat_circumferential_spread"] > 50.0     # measured ~236 K


def test_2d_peak_exceeds_1d_average(contour, prop, heat):
    """The 1-D series circuit averages the circumferential direction;
    the 2-D land hot spot is the real (higher) peak wall temperature."""
    from raosim.physics import regenerative_cooling_2d

    class Inc:
        conductivity = 16.0
    d1 = regenerative_cooling_analysis(heat, contour, _wide_land_cooling(),
                                       Inc(), 0.0015, prop, 7.0e6)
    d2 = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), Inc(),
                                 0.0015, prop, 7.0e6, n_s=20, n_xi=20)
    assert (d2["peak_gas_side_wall_temperature"]
            > d1["peak_gas_side_wall_temperature"])


def test_2d_low_k_has_larger_spread_than_high_k(contour, prop, heat):
    """Copper (high k) evens the wall out; Inconel (low k) does not —
    which is why engines use copper at the throat."""
    from raosim.physics import regenerative_cooling_2d

    class Cu:
        conductivity = 350.0

    class Inc:
        conductivity = 16.0
    cu = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), Cu(),
                                 0.0015, prop, 7.0e6, n_s=20, n_xi=20)
    inc = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), Inc(),
                                  0.0015, prop, 7.0e6, n_s=20, n_xi=20)
    assert (inc["throat_circumferential_spread"]
            > cu["throat_circumferential_spread"])


def test_2d_narrow_channels_nearly_isothermal(contour, prop, heat):
    """Closely-spaced narrow channels (sub-mm lands) -> small spread."""
    from raosim.physics import regenerative_cooling_2d

    class Inc:
        conductivity = 16.0
    res = regenerative_cooling_2d(heat, contour, _Cool(), Inc(),
                                  0.0015, prop, 7.0e6, n_s=16, n_xi=18)
    ti = int(np.argmin(np.abs(np.asarray(contour["y"]) - 0.02)))
    assert abs(res["circumferential_spread"][ti]) < 30.0


# ---------------------------------------------------------------------
#  Level 3: 3-D conjugate conduction (axial-conduction relief).
# ---------------------------------------------------------------------


def test_3d_solve_runs_and_is_finite(contour, prop, heat):
    from raosim.physics import regenerative_cooling_3d

    class Cu:
        conductivity = 350.0
    res = regenerative_cooling_3d(heat, contour, _wide_land_cooling(), Cu(),
                                  0.0015, prop, 7.0e6, n_x=30, n_s=8, n_xi=10)
    assert res["model"] == "regen_3d_conjugate"
    assert res["grid"]["cells"] > 0
    assert np.isfinite(res["T_wg_land_3d"]).all()
    assert res["coolant_outlet_temperature"] > _wide_land_cooling().coolant_inlet_temperature


def test_3d_axial_conduction_relieves_peak_vs_2d(contour, prop, heat):
    """Axial conduction (the Level-3 physics) spreads the throat hot
    spot to cooler neighbours, so the 3-D peak is below the per-station
    2-D peak."""
    from raosim.physics import regenerative_cooling_2d, regenerative_cooling_3d

    class Cu:
        conductivity = 350.0
    d2 = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), Cu(),
                                 0.0015, prop, 7.0e6, n_s=10, n_xi=12)
    d3 = regenerative_cooling_3d(heat, contour, _wide_land_cooling(), Cu(),
                                 0.0015, prop, 7.0e6, n_x=30, n_s=10, n_xi=12)
    assert (d3["peak_gas_side_wall_temperature"]
            < d2["peak_gas_side_wall_temperature"])


def test_3d_axial_relief_larger_for_high_conductivity(contour, prop, heat):
    """High-k walls conduct axially more, so the axial relief (2-D peak −
    3-D peak) is larger for copper than for Inconel."""
    from raosim.physics import regenerative_cooling_2d, regenerative_cooling_3d

    class Cu:
        conductivity = 350.0

    class Inc:
        conductivity = 16.0

    def relief(mat):
        d2 = regenerative_cooling_2d(heat, contour, _wide_land_cooling(), mat,
                                     0.0015, prop, 7.0e6, n_s=10, n_xi=12)
        d3 = regenerative_cooling_3d(heat, contour, _wide_land_cooling(), mat,
                                     0.0015, prop, 7.0e6, n_x=30, n_s=10, n_xi=12)
        return (d2["peak_gas_side_wall_temperature"]
                - d3["peak_gas_side_wall_temperature"])

    assert relief(Cu()) > relief(Inc())


def test_fidelity_dispatcher(contour, prop, heat):
    from raosim.physics import regenerative_cooling

    class Cu:
        conductivity = 350.0
    one = regenerative_cooling(heat, contour, _Cool(), Cu(), 0.001, prop,
                               7.0e6, fidelity="1d")
    two = regenerative_cooling(heat, contour, _wide_land_cooling(), Cu(),
                               0.0015, prop, 7.0e6, fidelity="2d",
                               n_s=10, n_xi=12)
    three = regenerative_cooling(heat, contour, _wide_land_cooling(), Cu(),
                                 0.0015, prop, 7.0e6, fidelity="3d",
                                 n_x=24, n_s=8, n_xi=10)
    assert one["fidelity"] == "1d_finned"
    assert two["fidelity"] == "2d_cross_section"
    assert three["fidelity"] == "3d_conjugate_conduction"
    with pytest.raises(ValueError, match="fidelity"):
        regenerative_cooling(heat, contour, _Cool(), Cu(), 0.001, prop,
                             7.0e6, fidelity="4d")
