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
    check_separation, summerfield_separation_pressure,
    kalt_badal_separation_ratio, schmucker_separation_ratio,
)
from raosim.trade_study import sweep_epsilon, sweep_Pc
from raosim.altitude_performance import altitude_performance_map
from raosim.chamber_geometry import chamber_contour, full_engine_contour


@pytest.fixture
def standard_contour():
    return bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)


@pytest.fixture
def lox_rp1():
    return get_propellant("LOX/RP-1")


# ── Wall pressure ─────────────────────────────────────────────────

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


# ── Separation ───────────────────────────────────────────────────

class TestSeparation:

    def test_summerfield_simple(self):
        """p_sep = 0.4 * Pa."""
        assert summerfield_separation_pressure(101325) == pytest.approx(0.4 * 101325)

    def test_schmucker_returns_value(self):
        ratio = schmucker_separation_ratio(3.0, 101325 / 45e5)
        assert 0 < ratio < 1

    def test_check_no_separation_high_Pc(self, standard_contour):
        """At very high Pc, nozzle shouldn't separate."""
        sep = check_separation(standard_contour, 100e5, 101325, 1.23)
        # With Pc=100 bar, eps=10, Pe should be well above separation
        assert isinstance(sep['separated'], bool)

    def test_check_at_sea_level(self, standard_contour):
        """At moderate Pc with eps=10, check runs without error."""
        sep = check_separation(standard_contour, 45e5, 101325, 1.23,
                               method='summerfield')
        assert 'margin' in sep
        assert 'exit_pressure' in sep


# ── Trade study ──────────────────────────────────────────────────

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


# ── Altitude performance ─────────────────────────────────────────

class TestAltitudePerformance:

    def test_runs(self, lox_rp1, standard_contour):
        apm = altitude_performance_map(
            45e5, 0.020, 10.0, lox_rp1, standard_contour, n_points=20,
        )
        assert len(apm['h']) == 20
        assert len(apm['thrust']) == 20
        assert len(apm['Isp']) == 20

    def test_thrust_increases_with_altitude(self, lox_rp1, standard_contour):
        """Thrust should increase as Pa decreases."""
        apm = altitude_performance_map(
            45e5, 0.020, 10.0, lox_rp1, standard_contour, n_points=20,
        )
        # Vacuum thrust > sea-level thrust
        assert apm['thrust'][-1] > apm['thrust'][0]


# ── Chamber geometry ─────────────────────────────────────────────

class TestChamberGeometry:

    def test_volume_matches_lstar(self):
        """V_chamber ≈ L* · At."""
        Rt = 0.020
        L_star = 1.0
        ch = chamber_contour(Rt, L_star=L_star)
        At = math.pi * Rt**2
        assert ch['V_chamber'] == pytest.approx(L_star * At, rel=1e-6)

    def test_chamber_radius(self):
        """Rc = Rt · sqrt(CR)."""
        Rt = 0.020
        CR = 3.0
        ch = chamber_contour(Rt, contraction_ratio=CR)
        assert ch['Rc'] == pytest.approx(Rt * math.sqrt(CR), rel=1e-6)

    def test_full_engine_contour(self, standard_contour):
        ch = chamber_contour(0.020, L_star=1.0)
        engine = full_engine_contour(ch, standard_contour)
        assert len(engine['x']) == len(ch['x']) + len(standard_contour['x'])
        assert len(engine['y']) == len(engine['x'])
