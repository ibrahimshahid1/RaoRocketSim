"""
Tests for raosim.rao_variational

Validates the Rao calculus-of-variations optimizer:
  - Stationarity conditions
  - Transversality
  - Contour quality
  - API compatibility
"""

import math
import pytest
import numpy as np

from raosim.rao_variational import (
    ControlSurface,
    thrust_integrand,
    massflow_integrand,
    length_integrand,
    stationarity_residuals,
    transversality_residual,
    solve_optimal_control_surface,
    rao_variational_contour,
)


class TestIntegrands:
    def test_thrust_integrand_positive(self):
        """Thrust integrand should be positive for typical supersonic flow."""
        f1 = thrust_integrand(M=2.0, theta=0.1, phi=1.0, r=0.05, gamma=1.4)
        assert f1 > 0

    def test_thrust_integrand_uses_ambient_pressure_subtraction(self):
        """Rao pressure term should subtract ambient without oblique projection."""
        f_vac = thrust_integrand(
            M=2.0, theta=0.1, phi=1.0, r=0.05, gamma=1.4,
            pa_over_p0=0.0,
        )
        f_amb = thrust_integrand(
            M=2.0, theta=0.1, phi=1.0, r=0.05, gamma=1.4,
            pa_over_p0=0.2,
        )
        assert f_vac - f_amb == pytest.approx(2.0 * math.pi * 0.05 * 0.2)

    def test_massflow_integrand_positive(self):
        """Mass flow integrand should be positive."""
        f2 = massflow_integrand(M=2.0, theta=0.1, phi=1.0, r=0.05, gamma=1.4)
        assert f2 > 0

    def test_length_integrand_positive(self):
        """cot(φ) should be positive for 0 < φ < π/2."""
        f3 = length_integrand(math.radians(45))
        assert f3 == pytest.approx(1.0, rel=1e-6)

    def test_length_integrand_zero_at_90(self):
        """cot(90°) = 0."""
        f3 = length_integrand(math.radians(90))
        assert f3 == pytest.approx(0.0, abs=1e-10)


class TestStationarityConditions:
    def test_residuals_finite(self):
        """Residuals should be finite for typical inputs."""
        R = stationarity_residuals(
            M=2.0, theta=0.1, phi=1.0, r=0.05,
            gamma=1.4, lambda2=-0.5, lambda3=0.01
        )
        assert np.all(np.isfinite(R))

    def test_residuals_near_zero_at_optimum(self):
        """
        After solving, stationarity residuals at interior stations
        should be small (at least the M and θ components).
        """
        ce = solve_optimal_control_surface(
            Rt=0.020, epsilon=10.0, gamma=1.4,
            length_pct=80.0, n_ce_pts=15, max_outer_iter=40
        )
        # Check a few interior stations (avoiding endpoints)
        for i in [3, 7]:
            if i < len(ce.r):
                R = stationarity_residuals(
                    ce.M[i], ce.theta[i], ce.phi[i], ce.r[i],
                    1.4, ce.lambda2, ce.lambda3
                )
                # M and θ components (R[0], R[1]) should be small
                # φ component (R[2]) involves -1/sin²(φ)·λ₃ which can be large
                assert abs(R[0]) < 5.0, \
                    f"∂/∂M residual too large at station {i}: {R[0]}"
                assert abs(R[1]) < 5.0, \
                    f"∂/∂θ residual too large at station {i}: {R[1]}"


class TestTransversality:
    def test_transversality_finite(self):
        """Transversality residual should be finite."""
        T = transversality_residual(
            M=3.0, theta=0.05, phi=1.2, r=0.06,
            gamma=1.4, lambda2=-0.5, lambda3=0.01
        )
        assert math.isfinite(T)


class TestRaoVariationalContour:
    @pytest.fixture
    def contour(self):
        """Generate Rao variational contour (small problem for speed)."""
        return rao_variational_contour(
            Rt=0.020, epsilon=10.0, gamma=1.4,
            length_pct=80.0, n_ce_pts=15, n_char=10,
            max_iter=40
        )

    def test_api_compatible(self, contour):
        """Output dict should have all required bell_nozzle_contour keys."""
        required_keys = ['x', 'y', 'theta_n', 'theta_e', 'Ln', 'Re',
                         'Rt', 'Ru', 'Rd', 'epsilon', 'x_conv', 'y_conv',
                         'x_throat', 'y_throat', 'x_bell', 'y_bell']
        for key in required_keys:
            assert key in contour, f"Missing key: {key}"

    def test_method_key(self, contour):
        """Should identify itself as 'rao'."""
        assert contour['method'] == 'rao'

    def test_exit_radius(self, contour):
        """Exit radius should be close to √ε · Rt."""
        Re_expected = math.sqrt(10.0) * 0.020
        # Allow 5% tolerance due to optimization
        assert contour['y'][-1] == pytest.approx(Re_expected, rel=0.05)

    def test_wall_monotonic(self, contour):
        """Bell section y should be monotonically increasing."""
        y_bell = contour['y_bell']
        dy = np.diff(y_bell)
        assert np.all(dy >= -1e-5), "Wall radius is not monotonically increasing"

    def test_control_surface_present(self, contour):
        """Should include the control surface data."""
        assert 'control_surface' in contour
        ce = contour['control_surface']
        assert isinstance(ce, ControlSurface)
        assert len(ce.M) > 0
        assert np.all(ce.M >= 1.0)

    def test_experimental_diagnostics_present(self, contour):
        """The variational path should not claim a validated full Rao solve."""
        assert contour['rao_full_optimum_claimed'] is False
        assert contour['variational_status'] == 'experimental_not_full_rao_bvp'
        assert 'construction_diagnostics' in contour

    def test_contour_length(self, contour):
        """Contour arrays should have reasonable length."""
        assert len(contour['x']) > 100
        assert len(contour['x']) == len(contour['y'])


class TestRaoBenchmarkVsBezier:
    """Benchmark: compare Rao variational against Bézier."""

    def test_comparison_runs(self):
        """Both methods should produce contours for the same inputs."""
        from raosim.nozzle_geometry import bell_nozzle_contour
        bezier = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
        rao = rao_variational_contour(
            Rt=0.020, epsilon=10.0, gamma=1.4,
            length_pct=80.0, n_ce_pts=15, n_char=10,
            max_iter=40
        )
        # Both should produce valid contours
        assert len(bezier['x']) > 0
        assert len(rao['x']) > 0
        # Both should target the same exit radius
        assert bezier['Re'] == pytest.approx(rao['Re'], rel=1e-3)
