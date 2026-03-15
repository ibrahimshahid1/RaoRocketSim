"""
Tests for raosim.conical

Validates conical nozzle divergence factor and contour geometry.
"""

import math
import pytest
import numpy as np
from raosim.conical import (
    conical_divergence_factor,
    conical_nozzle_length,
    conical_nozzle_contour,
)


class TestConicalDivergenceFactor:
    def test_zero_angle(self):
        """η_div(0°) = 1.0 (no divergence loss)."""
        assert conical_divergence_factor(0.0) == pytest.approx(1.0, abs=1e-12)

    def test_15_degrees(self):
        """η_div(15°) ≈ 0.9830 (Rao 1961)."""
        eta = conical_divergence_factor(math.radians(15.0))
        assert eta == pytest.approx(0.9830, rel=1e-3)

    def test_30_degrees(self):
        """η_div(30°) ≈ 0.933."""
        eta = conical_divergence_factor(math.radians(30.0))
        assert eta == pytest.approx(0.933, rel=1e-2)

    def test_90_degrees(self):
        """η_div(90°) = 0.5."""
        eta = conical_divergence_factor(math.radians(90.0))
        assert eta == pytest.approx(0.5, abs=1e-10)

    def test_monotone_decreasing(self):
        """η_div should decrease with increasing angle."""
        angles = [math.radians(a) for a in range(0, 91, 5)]
        etas = [conical_divergence_factor(a) for a in angles]
        for i in range(1, len(etas)):
            assert etas[i] <= etas[i - 1]


class TestConicalNozzleLength:
    def test_15deg_eps10(self):
        """L(15°, ε=10) should match the bell nozzle reference."""
        Rt = 0.020
        Re = math.sqrt(10.0) * Rt
        L_expected = (Re - Rt) / math.tan(math.radians(15.0))
        L = conical_nozzle_length(Rt, 10.0, 15.0)
        assert L == pytest.approx(L_expected, rel=1e-6)


class TestConicalNozzleContour:
    @pytest.fixture
    def contour(self):
        return conical_nozzle_contour(Rt=0.020, epsilon=10.0)

    def test_exit_radius(self, contour):
        """Exit y should be √ε · Rt."""
        Re_expected = math.sqrt(10.0) * 0.020
        assert contour['y'][-1] == pytest.approx(Re_expected, rel=1e-2)

    def test_api_compatible(self, contour):
        """Output dict should have the same keys used by bell contour."""
        required_keys = ['x', 'y', 'theta_n', 'theta_e', 'Ln', 'Re',
                         'Rt', 'Ru', 'Rd', 'epsilon', 'x_conv', 'y_conv',
                         'x_throat', 'y_throat', 'x_bell', 'y_bell']
        for key in required_keys:
            assert key in contour, f"Missing key: {key}"

    def test_eta_div_present(self, contour):
        """Divergence factor should be included."""
        assert 'eta_div' in contour
        assert 0.9 < contour['eta_div'] < 1.0

    def test_divergent_section_straight(self, contour):
        """The divergent section should be a straight line (conical)."""
        x = contour['x_bell']
        y = contour['y_bell']
        # Fit a line and check R² ≈ 1
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 1)
            y_fit = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            assert r_sq > 0.999, f"Divergent section is not straight (R² = {r_sq})"

    def test_various_angles(self):
        """Should generate for different half-angles without errors."""
        for angle in [10.0, 15.0, 20.0, 30.0]:
            c = conical_nozzle_contour(0.020, 10.0, half_angle_deg=angle)
            assert c['half_angle_deg'] == angle
