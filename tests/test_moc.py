"""
Tests for rewritten MOC solver with coupled wall marching.

Validates primitives, coupled marching, and full pipeline.
"""

import math
import pytest
import numpy as np

from raosim.gas_dynamics import (
    prandtl_meyer,
    mach_angle,
    mach_from_prandtl_meyer,
)
from raosim.moc import (
    CharPoint, CharRow,
    _make_point,
    solve_interior_point,
    solve_axis_point,
    solve_wall_point,
    approximate_starting_line,
    march_coupled_net,
    sample_exit_plane,
    compute_exit_thrust,
)
from raosim.wall_model import SplineWall


class TestGasDynamicsUtilities:
    def test_mach_angle_m2(self):
        assert math.degrees(mach_angle(2.0)) == pytest.approx(30.0, abs=0.01)

    def test_mach_angle_m1(self):
        assert math.degrees(mach_angle(1.0)) == pytest.approx(90.0, abs=0.01)

    @pytest.mark.parametrize("M", [1.5, 2.0, 3.0, 4.0, 5.0])
    def test_pm_roundtrip(self, M):
        nu = prandtl_meyer(M, 1.4)
        assert mach_from_prandtl_meyer(nu, 1.4) == pytest.approx(M, rel=1e-6)

    def test_pm_zero(self):
        assert mach_from_prandtl_meyer(0.0, 1.4) == 1.0


class TestInteriorPoint:
    def test_planar_invariants(self):
        gamma = 1.4
        p1 = _make_point(0.01, 0.02, math.radians(5), 1.5, gamma)
        p2 = _make_point(0.01, 0.015, math.radians(3), 1.8, gamma)
        p3 = solve_interior_point(p1, p2, gamma, axisymmetric=False)
        assert p3.compat_minus == pytest.approx(p1.compat_minus, abs=1e-4)
        assert p3.compat_plus == pytest.approx(p2.compat_plus, abs=1e-4)

    def test_downstream(self):
        gamma = 1.4
        p1 = _make_point(0.005, 0.025, math.radians(8), 1.3, gamma)
        p2 = _make_point(0.005, 0.020, math.radians(2), 1.5, gamma)
        p3 = solve_interior_point(p1, p2, gamma)
        assert p3.x > min(p1.x, p2.x)
        assert p3.M >= 1.0


class TestAxisPoint:
    def test_symmetry_bc(self):
        gamma = 1.4
        pa = _make_point(0.01, 0.005, math.radians(2), 1.5, gamma)
        pax = solve_axis_point(pa, gamma)
        assert pax.theta == pytest.approx(0.0, abs=1e-10)
        assert pax.r == pytest.approx(0.0, abs=1e-10)
        assert pax.M > 1.0


class TestSplineWall:
    def test_endpoint_values(self):
        wall = SplineWall.from_controls(
            np.array([0.03, 0.04, 0.05]),
            Nx=0.005, Ny=0.022, Ex=0.10, Ey=0.06,
            theta_n=math.radians(30),
        )
        assert wall.r(wall.x_start) == pytest.approx(0.022, abs=1e-6)
        assert wall.r(wall.x_end) == pytest.approx(0.06, abs=1e-6)

    def test_start_slope(self):
        theta_n = math.radians(30)
        wall = SplineWall.from_controls(
            np.array([0.03, 0.04, 0.05]),
            Nx=0.005, Ny=0.022, Ex=0.10, Ey=0.06,
            theta_n=theta_n,
        )
        assert wall.dr_dx(wall.x_start) == pytest.approx(math.tan(theta_n), abs=0.01)

    def test_intersection(self):
        wall = SplineWall.from_controls(
            np.array([0.03, 0.04, 0.05]),
            Nx=0.005, Ny=0.022, Ex=0.10, Ey=0.06,
            theta_n=math.radians(30),
        )
        x0, r0 = 0.01, 0.015
        slope = 1.0
        x_hit, r_hit = wall.intersect_char(x0, r0, slope)
        assert x_hit > x0
        assert abs(r_hit - wall.r(x_hit)) < 1e-6


class TestStartingLine:
    def test_all_supersonic(self):
        pts = approximate_starting_line(0.02, 0.382*0.02, math.radians(30), 1.4, 20)
        assert all(p.M > 1.0 for p in pts)

    def test_theta_progression(self):
        pts = approximate_starting_line(0.02, 0.382*0.02, math.radians(25), 1.4, 20)
        assert pts[0].theta < pts[-1].theta


class TestCoupledMarching:
    def test_row_has_wall_and_axis(self):
        """Each row (except row 0) should have axis and wall points."""
        Rt, Rd = 0.02, 0.382*0.02
        theta_n = math.radians(30)
        Re = math.sqrt(10.0) * Rt
        Ln = (Re - Rt) / math.tan(math.radians(15)) * 0.8

        Ny = Rt + Rd * (1.0 - math.cos(theta_n))
        Nx = Rd * math.sin(theta_n)

        wall = SplineWall.from_controls(
            np.linspace(Ny, Re, 5)[1:-1],
            Nx, Ny, Ln, Re, theta_n,
        )
        pts = approximate_starting_line(Rt, Rd, theta_n, 1.4, 10)
        rows = march_coupled_net(pts, wall, 1.4, max_rows=5)

        for i, row in enumerate(rows[1:], 1):
            assert row.axis is not None, f"Row {i} missing axis"
            assert row.wall is not None, f"Row {i} missing wall"

    def test_wall_angle_matches(self):
        """Wall point θ should equal wall.theta(x)."""
        Rt, Rd = 0.02, 0.382*0.02
        theta_n = math.radians(30)
        Re = math.sqrt(10.0) * Rt
        Ln = (Re - Rt) / math.tan(math.radians(15)) * 0.8

        Ny = Rt + Rd * (1.0 - math.cos(theta_n))
        Nx = Rd * math.sin(theta_n)

        wall = SplineWall.from_controls(
            np.linspace(Ny, Re, 5)[1:-1],
            Nx, Ny, Ln, Re, theta_n,
        )
        pts = approximate_starting_line(Rt, Rd, theta_n, 1.4, 10)
        rows = march_coupled_net(pts, wall, 1.4, max_rows=5)

        for row in rows[1:]:
            if row.wall is not None:
                expected_theta = wall.theta(row.wall.x)
                assert row.wall.theta == pytest.approx(expected_theta, abs=0.05)


class TestFullPipeline:
    def test_exit_radius(self):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=10.0, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        Re = math.sqrt(10.0) * 0.02
        assert c['y'][-1] == pytest.approx(Re, rel=0.05)

    def test_method_key(self):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=10.0, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        assert c['method'] == 'moc'

    def test_bell_monotonic(self):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=10.0, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        dy = np.diff(c['y_bell'])
        assert np.all(dy >= -1e-6)

    def test_benchmark_vs_bezier(self):
        """Benchmark only — no hard-coded thresholds."""
        from raosim.rao_optimizer import moc_bell_nozzle
        try:
            from raosim.nozzle_geometry import bell_nozzle_contour
        except ImportError:
            pytest.skip('scipy unavailable')

        bezier = bell_nozzle_contour(Rt=0.02, epsilon=10.0, length_pct=80.0)
        moc = moc_bell_nozzle(Rt=0.02, epsilon=10.0, gamma=1.4,
                              length_pct=80.0, n_control=3, n_char=10,
                              max_iter=50)
        x_b, y_b = bezier['x_bell'], bezier['y_bell']
        x_m, y_m = moc['x_bell'], moc['y_bell']
        x_c = np.linspace(max(x_b[0], x_m[0]), min(x_b[-1], x_m[-1]), 50)
        dev = np.max(np.abs(np.interp(x_c, x_b, y_b) - np.interp(x_c, x_m, y_m)))
        print(f"\n  [BENCHMARK] deviation: {dev*1000:.3f} mm "
              f"({100*dev/bezier['Re']:.1f}% Re)")
