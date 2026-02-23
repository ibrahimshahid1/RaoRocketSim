"""
Tests for raosim.nozzle_geometry

Verifies geometric correctness: continuity, dimensions, monotonicity.
"""

import math
import pytest
import numpy as np
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles


class TestLookupAngles:
    def test_known_point(self):
        """At ε=10, L%=80: θ_n ≈ 30°, θ_e ≈ 15.5° (from table)."""
        tn, te = lookup_angles(10.0, 80.0)
        assert 28.0 <= tn <= 32.0
        assert 14.0 <= te <= 17.0

    def test_interpolation_runs(self):
        """Interpolation should work for intermediate values."""
        tn, te = lookup_angles(7.0, 75.0)
        assert 20.0 < tn < 45.0
        assert 5.0 < te < 25.0


class TestBellNozzleContour:

    @pytest.fixture
    def contour_default(self):
        """Default contour: Rt=20mm, ε=10, 80% bell."""
        return bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)

    def test_exit_radius(self, contour_default):
        """Exit point y should equal √ε · Rt."""
        c = contour_default
        Re_expected = math.sqrt(10.0) * 0.020
        assert c['y'][-1] == pytest.approx(Re_expected, rel=1e-3)

    def test_bell_length(self, contour_default):
        """Nozzle length should be 80% of 15° cone length."""
        c = contour_default
        Re = math.sqrt(10.0) * 0.020
        L15 = (Re - 0.020) / math.tan(math.radians(15.0))
        expected_Ln = 0.80 * L15
        assert c['Ln'] == pytest.approx(expected_Ln, rel=1e-6)

    def test_throat_radius(self, contour_default):
        """At x ≈ 0, y should be close to Rt."""
        x = contour_default['x']
        y = contour_default['y']
        # Find the point nearest x=0
        idx = np.argmin(np.abs(x))
        assert y[idx] == pytest.approx(0.020, rel=5e-2)

    def test_c0_continuity_arc_to_bell(self, contour_default):
        """The last downstream-arc point should match the first bell point."""
        c = contour_default
        x_thr, y_thr = c['x_throat'], c['y_throat']
        x_bell, y_bell = c['x_bell'], c['y_bell']
        assert x_thr[-1] == pytest.approx(x_bell[0], abs=1e-10)
        assert y_thr[-1] == pytest.approx(y_bell[0], abs=1e-10)

    def test_g1_continuity_slope(self, contour_default):
        """Slope at the arc→bell junction should be tan(θ_n)."""
        c = contour_default
        theta_n_rad = math.radians(c['theta_n'])
        expected_slope = math.tan(theta_n_rad)

        # Compute slope from the last two downstream-arc points
        x_thr, y_thr = c['x_throat'], c['y_throat']
        dx_arc = x_thr[-1] - x_thr[-2]
        dy_arc = y_thr[-1] - y_thr[-2]
        slope_arc = dy_arc / dx_arc if abs(dx_arc) > 1e-15 else float('inf')

        # Compute slope from the first two bell points
        x_bell, y_bell = c['x_bell'], c['y_bell']
        dx_bell = x_bell[1] - x_bell[0]
        dy_bell = y_bell[1] - y_bell[0]
        slope_bell = dy_bell / dx_bell if abs(dx_bell) > 1e-15 else float('inf')

        assert slope_arc == pytest.approx(expected_slope, rel=5e-2)
        assert slope_bell == pytest.approx(expected_slope, rel=5e-2)

    def test_bell_y_monotonically_increasing(self, contour_default):
        """y along the bell section should increase monotonically."""
        y_bell = contour_default['y_bell']
        diffs = np.diff(y_bell)
        assert np.all(diffs >= -1e-12), "Bell section y is not monotonically increasing"

    def test_various_expansion_ratios(self):
        """Contour should generate for various ε without errors."""
        for eps in [4.0, 10.0, 25.0, 50.0]:
            c = bell_nozzle_contour(Rt=0.020, epsilon=eps, length_pct=80.0)
            Re_expected = math.sqrt(eps) * 0.020
            assert c['y'][-1] == pytest.approx(Re_expected, rel=1e-3)

    def test_various_length_fractions(self):
        """Contour should generate for various L% without errors."""
        for lpct in [60.0, 70.0, 80.0, 90.0]:
            c = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=lpct)
            assert c['length_pct'] == lpct
