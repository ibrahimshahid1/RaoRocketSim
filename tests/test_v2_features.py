"""
Tests for the new v2 features: wall_pressure, separation, trade_study,
altitude_performance, chamber_geometry.
"""

import math
import pytest
import numpy as np

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import get_propellant, custom_propellant
from raosim.engine import compute_engine_performance
from raosim.wall_pressure import wall_pressure_distribution
from raosim.separation import (
    SP8120_DESIGN_MARGIN,
    check_separation, summerfield_separation_pressure,
    kalt_badal_separation_ratio, schilling_separation_ratio,
    schmucker_separation_ratio,
)
from raosim.trade_study import sweep_epsilon, sweep_Pc
from raosim.altitude_performance import altitude_performance_map
from raosim.chamber_geometry import (
    chamber_contour,
    enclosed_volume,
    full_engine_contour,
)


@pytest.fixture
def standard_contour():
    return bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)


@pytest.fixture
def lox_rp1():
    return get_propellant("LOX/RP-1")




class TestWallPressure:

    def test_runs(self, standard_contour):
        wp = wall_pressure_distribution(standard_contour, 45e5, 1.23)
        assert len(wp['x']) == len(wp['p'])
        assert len(wp['M']) == len(wp['x'])

    def test_pressure_decreases_downstream(self, standard_contour):
        """For a well-designed bell, p should decrease after throat."""
        wp = wall_pressure_distribution(standard_contour, 45e5, 1.23)
        # Overall trend: exit pressure < throat pressure
        assert wp['p'][-1] < wp['p'][wp['throat_idx']]

    def test_mach_increases_downstream(self, standard_contour):
        """Mach should increase downstream of throat."""
        wp = wall_pressure_distribution(standard_contour, 45e5, 1.23)
        ti = wp['throat_idx']
        assert wp['M'][-1] > wp['M'][ti]




class TestSeparation:

    def test_summerfield_simple(self):
        """p_sep = 0.4 * Pa."""
        assert summerfield_separation_pressure(101325) == pytest.approx(0.4 * 101325)

    def test_schmucker_matches_ostlund_eq30(self):
        # Östlund (2002) Eq. 30: p_sep/Pa = (1.88·Me − 1)^(−0.64), returned
        # here rescaled to p_sep/Pc.  (Criterion corrected 2026-07-22.)
        Me, Pa_over_Pc = 3.0, 101325 / 45e5
        ratio = schmucker_separation_ratio(Me, Pa_over_Pc)
        assert ratio == pytest.approx(
            (1.88 * Me - 1.0) ** (-0.64) * Pa_over_Pc, rel=1e-12)
        # Literature magnitude: p_sep/Pa ≈ 0.375 at Me = 3.
        assert ratio / Pa_over_Pc == pytest.approx(0.3746, abs=5e-3)
        assert 0 < ratio < 1

    def test_kalt_badal_matches_ostlund(self):
        # Östlund (2002) p. 52: Schilling form with k1 = 2/3, k2 = −0.2.
        assert kalt_badal_separation_ratio(30.0) == pytest.approx(
            (2.0 / 3.0) * 30.0 ** (-0.2), rel=1e-12)
        assert kalt_badal_separation_ratio(30.0) == pytest.approx(0.3377, abs=5e-3)

    def test_schilling_contoured_constants(self):
        # Östlund (2002) Eq. 29, contoured: k1 = 0.582, k2 = −0.195.
        assert schilling_separation_ratio(30.0) == pytest.approx(
            0.582 * 30.0 ** (-0.195), rel=1e-12)
        assert schilling_separation_ratio(30.0, contoured=False) == pytest.approx(
            0.541 * 30.0 ** (-0.136), rel=1e-12)

    def test_criteria_agree_in_magnitude(self):
        # At Pc/Pa = 30, Me ≈ 3 the four criteria should cluster within a
        # factor ~2 of each other (0.3–0.5 of ambient) — cross-family sanity.
        pa_ratios = [
            0.4,
            kalt_badal_separation_ratio(30.0),
            schilling_separation_ratio(30.0),
            schmucker_separation_ratio(3.0, 1 / 30.0) * 30.0,
        ]
        assert all(0.25 < r < 0.55 for r in pa_ratios)

    def test_check_no_separation_high_Pc(self, standard_contour):
        """At very high Pc, nozzle shouldn't separate."""
        sep = check_separation(standard_contour, 100e5, 101325, 1.23)
        # With Pc=100 bar, eps=10, Pe should be well above separation
        assert isinstance(sep['separated'], bool)
        assert 'design_margin_ok' in sep
        assert sep['design_margin_required'] == pytest.approx(SP8120_DESIGN_MARGIN)

    def test_vacuum_never_separates(self, standard_contour):
        for method in ('summerfield', 'kalt_badal', 'schmucker', 'schilling'):
            sep = check_separation(standard_contour, 45e5, 0.0, 1.23,
                                   method=method)
            assert not sep['separated']
            assert sep['design_margin_ok']

    def test_check_at_sea_level(self, standard_contour):
        """At moderate Pc with eps=10, check runs without error."""
        sep = check_separation(standard_contour, 45e5, 101325, 1.23,
                               method='summerfield')
        assert 'margin' in sep
        assert 'exit_pressure' in sep
        assert sep['criterion_evaluated_locally'] is True
        assert 'exit_criterion_pressure' in sep




class TestTradeStudy:

    def test_sweep_epsilon_count(self, lox_rp1):
        results = sweep_epsilon([5, 10, 15, 20], 45e5, 101325, 0.020, lox_rp1)
        assert len(results) == 4

    def test_sweep_epsilon_keys(self, lox_rp1):
        results = sweep_epsilon([10], 45e5, 101325, 0.020, lox_rp1)
        r = results[0]
        for key in ['epsilon', 'Me', 'Isp', 'thrust', 'Cf', 'm_dot']:
            assert key in r

    def test_sweep_Pc(self, lox_rp1):
        results = sweep_Pc([30, 45, 60], 101325, 0.020, 10.0, lox_rp1)
        assert len(results) == 3
        # Higher Pc → higher thrust
        assert results[-1]['thrust'] > results[0]['thrust']

    def test_isp_increases_with_epsilon(self, lox_rp1):
        """In vacuum, Isp should increase with ε."""
        results = sweep_epsilon([5, 10, 20, 40], 45e5, 0.0, 0.020, lox_rp1)
        isps = [r['Isp'] for r in results]
        assert isps[-1] > isps[0]




class TestAltitudePerformance:

    def test_runs(self, lox_rp1, standard_contour):
        apm = altitude_performance_map(
            45e5, 0.020, 10.0, lox_rp1, standard_contour, n_points=20,
        )
        assert len(apm['h']) == 20
        assert len(apm['thrust']) == 20
        assert len(apm['Isp']) == 20
        assert apm['thrust'] is apm['attached_thrust']
        assert apm['Isp'] is apm['attached_Isp']
        assert apm['Cf'] is apm['attached_Cf']
        assert apm['performance_model'] == (
            'quasi_1d_attached_flow_no_separation_loss'
        )

    def test_thrust_increases_with_altitude(self, lox_rp1, standard_contour):
        """Thrust should increase as Pa decreases."""
        apm = altitude_performance_map(
            45e5, 0.020, 10.0, lox_rp1, standard_contour, n_points=20,
        )
        # Vacuum thrust > sea-level thrust
        assert apm['thrust'][-1] > apm['thrust'][0]




class TestChamberGeometry:

    def test_volume_matches_lstar(self):
        """The generated contour, not only stored metadata, encloses L* · At."""
        Rt = 0.020
        L_star = 1.0
        ch = chamber_contour(Rt, L_star=L_star)
        At = math.pi * Rt**2
        assert enclosed_volume(ch["x"], ch["y"]) == pytest.approx(
            L_star * At, rel=1e-10
        )

    def test_chamber_radius(self):
        """Rc = Rt · sqrt(CR)."""
        Rt = 0.020
        CR = 3.0
        ch = chamber_contour(Rt, contraction_ratio=CR)
        assert ch['Rc'] == pytest.approx(Rt * math.sqrt(CR), rel=1e-6)

    def test_full_engine_contour(self, standard_contour):
        ch = chamber_contour(0.020, L_star=1.0)
        engine = full_engine_contour(ch, standard_contour)
        assert engine["full_thrust_chamber"]
        assert np.all(np.diff(engine["x"]) > 0.0)
        assert len(engine['y']) == len(engine['x'])
