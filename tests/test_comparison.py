"""
Tests for raosim.nozzle_comparison

Validates the comparison utilities: divergence loss, efficiency,
and comparison table generation.
"""

import math
import pytest
import numpy as np

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.conical import conical_nozzle_contour
from raosim.nozzle_comparison import (
    divergence_loss_2d,
    nozzle_efficiency,
    compare_contours,
    print_comparison_table,
)


class TestDivergenceLoss:
    def test_bell_has_smaller_exit_angle(self):
        """Bell nozzle exit wall slope is much shallower than at start."""
        bell = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
        cone = conical_nozzle_contour(Rt=0.020, epsilon=10.0)

        # Bell θ_e should be much less than θ_n (wall turns back toward axis)
        assert bell['theta_e'] < bell['theta_n']

        # Both divergence factors should be near 1 (high efficiency)
        eta_bell = divergence_loss_2d(bell, 1.4)
        eta_cone = divergence_loss_2d(cone, 1.4)
        assert eta_bell > 0.95
        assert eta_cone > 0.95

    def test_conical_matches_formula(self):
        """Conical η_div should be close to (1+cos α)/2."""
        from raosim.conical import conical_divergence_factor
        cone = conical_nozzle_contour(Rt=0.020, epsilon=10.0,
                                       half_angle_deg=15.0)
        eta_2d = divergence_loss_2d(cone, 1.4)
        eta_formula = conical_divergence_factor(math.radians(15.0))
        # Should be in the same ballpark (within 3%)
        assert eta_2d == pytest.approx(eta_formula, rel=0.03)

    def test_returns_valid_range(self):
        """η_div should be in (0, 1]."""
        c = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
        eta = divergence_loss_2d(c, 1.4)
        assert 0.0 < eta <= 1.0


class TestNozzleEfficiency:
    def test_perfect_nozzle(self):
        """η_n = 1 when Cf_actual = Cf_ideal."""
        assert nozzle_efficiency(1.5, 1.5) == pytest.approx(1.0)

    def test_imperfect_nozzle(self):
        """η_n < 1 when Cf_actual < Cf_ideal."""
        eta = nozzle_efficiency(1.45, 1.5)
        assert eta < 1.0
        assert eta > 0.0

    def test_zero_ideal(self):
        """Edge case: zero ideal Cf should return 0."""
        assert nozzle_efficiency(1.0, 0.0) == 0.0


class TestCompareContours:
    def test_comparison_runs(self):
        """compare_contours should produce results for multiple contours."""
        bell = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
        cone = conical_nozzle_contour(Rt=0.020, epsilon=10.0)

        contours = {'Bell': bell, 'Conical': cone}
        results = compare_contours(contours, Pc=4.5e6, Pa=101325.0, gamma=1.4)

        assert len(results) == 2
        for r in results:
            assert 'name' in r
            assert 'Cf_ideal' in r
            assert 'eta_div' in r
            assert 'Cf_corrected' in r

    def test_table_format(self):
        """print_comparison_table should produce a readable string."""
        bell = bell_nozzle_contour(Rt=0.020, epsilon=10.0, length_pct=80.0)
        contours = {'Bell': bell}
        results = compare_contours(contours, Pc=4.5e6, Pa=101325.0, gamma=1.4)
        table = print_comparison_table(results)
        assert isinstance(table, str)
        assert 'Bell' in table
        assert 'η_div' in table
