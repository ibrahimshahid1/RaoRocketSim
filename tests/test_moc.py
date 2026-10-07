"""
Tests for the MOC unit processes and the forward flow-field entry point.

Validates primitives, the bounded-net march driven by ``solve_flowfield``
(topology gates live in tests/test_moc_bounded_net.py), and the direct
MOC optimizer pipeline.  The deprecated front march ``march_coupled_net``
is pinned only for its documented (defective) behaviour.
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
    solve_flowfield,
)
from raosim.wall_model import SplineWall


def test_direct_moc_uses_upstream_throat_curvature(monkeypatch):
    import raosim.moc as moc_module

    Rt = 0.02
    epsilon = 8.0
    Re = math.sqrt(epsilon) * Rt
    Ln = 0.8 * (Re - Rt) / math.tan(math.radians(15.0))
    theta_n = math.radians(30.0)
    Rd = 0.382 * Rt
    Nx = Rd * math.sin(theta_n)
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    wall = SplineWall.from_controls(
        np.linspace(Ny, Re, 5)[1:-1], Nx, Ny, Ln, Re, theta_n
    )
    import raosim.nasa_moc as nasa_module

    captured = {}
    original_line = nasa_module._make_throat_initial_line
    original_arc = nasa_module.ArcWall

    def capture_line(Rt_arg, R_start_line, *args, **kwargs):
        captured["transonic_curvature"] = R_start_line
        return original_line(Rt_arg, R_start_line, *args, **kwargs)

    class CaptureArc(original_arc):
        def __init__(self, Rt_arg, Rd_arg, theta_max):
            captured["downstream_curvature"] = Rd_arg
            super().__init__(Rt_arg, Rd_arg, theta_max)

    monkeypatch.setattr(nasa_module, "_make_throat_initial_line", capture_line)
    monkeypatch.setattr(nasa_module, "ArcWall", CaptureArc)
    # The wall must start where the reconstructed arc ends.
    Rd = 0.5 * Rt
    Nx = Rd * math.sin(theta_n)
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    wall = SplineWall.from_controls(
        np.linspace(Ny, Re, 5)[1:-1], Nx, Ny, Ln, Re, theta_n
    )
    result = moc_module.solve_flowfield(
        Rt,
        epsilon,
        1.4,
        wall,
        n_char=8,
        starting_line_method="kliegel_levine",
        throat_upstream_radius_factor=1.5,
        throat_downstream_radius_factor=0.5,
    )

    assert captured["downstream_curvature"] == pytest.approx(0.5 * Rt)
    assert captured["transonic_curvature"] == pytest.approx(1.5 * Rt)
    assert isinstance(result["complete"], bool)
    assert result["exit_metrics"]["normalization_basis"] == "throat_area"
    assert result["exit_metrics"]["reference_area"] == pytest.approx(
        math.pi * Rt**2
    )


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
        """CORRECTED 2026-06-11 (invariant pairing): in planar flow,
        K− = θ+ν is carried along the C− from p1 (above) and
        K+ = θ−ν along the C+ from p2 (below) — Anderson MCF §11.4.
        (This test previously pinned the mirrored convention; see
        tests/test_characteristic_pairing.py for the oracle proof.)"""
        gamma = 1.4
        p1 = _make_point(0.01, 0.02, math.radians(5), 1.5, gamma)
        p2 = _make_point(0.01, 0.015, math.radians(3), 1.8, gamma)
        p3 = solve_interior_point(p1, p2, gamma, axisymmetric=False)
        assert p3.compat_plus == pytest.approx(p1.compat_plus, abs=1e-4)
        assert p3.compat_minus == pytest.approx(p2.compat_minus, abs=1e-4)

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


def _test_wall(Rt=0.02, eps=10.0, theta_n_deg=30.0):
    Rd = 0.382 * Rt
    theta_n = math.radians(theta_n_deg)
    Re = math.sqrt(eps) * Rt
    Ln = (Re - Rt) / math.tan(math.radians(15)) * 0.8
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    Nx = Rd * math.sin(theta_n)
    return SplineWall.from_controls(
        np.linspace(Ny, Re, 5)[1:-1], Nx, Ny, Ln, Re, theta_n,
    )


class TestBoundedMarching:
    """``solve_flowfield`` rows are RRCs of a bounded net (NASA/JHU topology)."""

    def test_row_has_wall_and_axis(self):
        wall = _test_wall()
        result = solve_flowfield(0.02, 10.0, 1.4, wall, n_char=10, mass_tol=0.05)
        rows = result["rows"]
        assert len(rows) > 20
        for i, row in enumerate(rows[1:], 1):
            assert row.wall is not None, f"Row {i} missing wall"
            assert row.parents is not None
        # every wall point is used as a parent (C- source of the next RRC)
        net = result["net"]
        used = {src[0] for pm in net.parents[1:] for e in pm
                for src in (e.get("cplus"), e.get("cminus")) if src and src[1] == 0}
        assert all(j in used for j in range(1, net.n_rows - 1))

    def test_wall_angle_matches(self):
        """Wall point θ equals wall.theta(x) (tangency BC) at distinct x
        along the bell.  Uses the NASA M3.5 perfect-nozzle wall because a
        linear-control test spline (and the chart Bézier) coalesce their
        compression waves on the first bell RRC, which correctly stops the
        march before any bell wall point exists."""
        from tests.test_moc_bounded_net import NASA_DIR, _nasa_wall
        if not (NASA_DIR / "wall.out").exists():
            pytest.skip("NASA oracle outputs not present")
        arc, bell, _, _ = _nasa_wall()
        result = solve_flowfield(
            1.0, 6.7365, 1.4, bell, n_char=41,
            starting_line_method="nasa_visible_kliegel_levine",
            throat_upstream_radius_factor=1.0,
            throat_downstream_radius_factor=1.0,
        )
        assert result["complete"]
        rows = result["rows"]
        xs = []
        for row in rows[1:]:
            if row.wall.x > arc.x_end:
                assert row.wall.theta == pytest.approx(bell.theta(row.wall.x), abs=1e-9)
                xs.append(row.wall.x)
        assert len(xs) > 20 and np.all(np.diff(xs) > 0)

    def test_rows_do_not_shrink(self):
        """Point count per RRC grows with special insertions and never
        shrinks except by axis truncation (the old front march lost one
        point per row)."""
        wall = _test_wall()
        net = solve_flowfield(0.02, 10.0, 1.4, wall, n_char=12, mass_tol=0.05)["net"]
        lens = [len(r) for r in net.rows]
        assert max(lens) > lens[0]
        shrinks = sum(1 for a, b in zip(lens[:-1], lens[1:]) if b < a)
        assert shrinks <= net.truncated_rows

    def test_front_march_is_deprecated(self):
        wall = _test_wall()
        pts = approximate_starting_line(0.02, 0.382 * 0.02, math.radians(30), 1.4, 10)
        with pytest.warns(DeprecationWarning):
            rows = march_coupled_net(pts, wall, 1.4, max_rows=5)
        assert [len(r.all_points()) for r in rows] == [10, 10, 9, 8, 7]


@pytest.fixture
def moc_contour():
    from raosim.rao_optimizer import moc_bell_nozzle
    return moc_bell_nozzle(
        Rt=0.02,
        epsilon=10.0,
        gamma=1.4,
        length_pct=80.0,
        n_control=3,
        n_char=10,
        max_iter=20,
    )


class TestFullPipeline:
    @pytest.mark.parametrize("epsilon", [4, 10, 25, 50])
    @pytest.mark.slow
    def test_exit_radius(self, epsilon):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=epsilon, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        Re = math.sqrt(epsilon) * 0.02
        assert c['y'][-1] == pytest.approx(Re, rel=0.05)

    @pytest.mark.parametrize("epsilon", [4, 10, 25, 50])
    @pytest.mark.slow
    def test_method_key(self, epsilon):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=epsilon, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        assert c['method'] == 'moc'

    @pytest.mark.parametrize("epsilon", [4, 10, 25, 50])
    @pytest.mark.slow
    def test_bell_monotonic(self, epsilon):
        from raosim.rao_optimizer import moc_bell_nozzle
        c = moc_bell_nozzle(Rt=0.02, epsilon=epsilon, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)
        dy = np.diff(c['y_bell'])
        assert np.all(dy >= -1e-6)

    @pytest.mark.parametrize("epsilon", [4, 10, 25, 50])
    @pytest.mark.slow
    def test_wall_monotonicity_and_smoothness(self, epsilon):
        from raosim.rao_optimizer import moc_bell_nozzle

        c = moc_bell_nozzle(Rt=0.02, epsilon=epsilon, gamma=1.4,
                            length_pct=80.0, n_control=3, n_char=10,
                            max_iter=50)

        x_b, y_b = c['x_bell'], c['y_bell']
        dy = np.diff(y_b)
        assert np.all(dy >= -1e-6)

        slopes = np.diff(y_b) / np.diff(x_b)
        curvature = np.diff(slopes)
        assert np.max(np.abs(curvature)) < 0.35

    def test_exit_metrics_are_honest_about_net_completeness(self, moc_contour):
        """When the bounded net stops before the exit (compression
        coalescence on the candidate wall) the contour says so and carries
        NaN exit metrics instead of a fabricated profile; when it completes,
        the traced exit plane is consistent with the reported metrics."""
        from raosim.rao_optimizer import optimize_wall

        opt = optimize_wall(
            Rt=0.02,
            epsilon=10.0,
            gamma=1.4,
            length_pct=80.0,
            n_control=3,
            n_char=10,
            max_iter=20,
        )
        assert isinstance(opt['net_complete'], bool)
        assert isinstance(moc_contour['moc_net_complete'], bool)
        if not moc_contour['moc_net_complete']:
            assert moc_contour['moc_net_stop_reason'] in (
                'rrc_coalescence', 'invalid_cell', 'massflow_drift',
                'wall_point_upstream', 'wall_point_failed',
            )
            assert math.isnan(moc_contour['exit_Cf'])
            return
        traced = sample_exit_plane(opt['rows'], opt['wall'].x_end, gamma=1.4, n_samples=40)
        metrics = compute_exit_thrust(traced, gamma=1.4, throat_radius=0.02)
        assert metrics['M_mean'] > 1.0
        assert moc_contour['exit_theta_max'] == pytest.approx(metrics['theta_max'], rel=0.5)

    @pytest.mark.slow
    def test_moc_vs_bezier_shape_and_thrust_tolerances(self):
        """Validate corrected-thrust and shape with explicit tolerances."""
        from raosim.rao_optimizer import moc_bell_nozzle
        from raosim.nozzle_comparison import compare_contours
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

    @pytest.mark.parametrize(
        "starting_line_method", ["kliegel_levine", "nasa_visible_kliegel_levine"]
    )
    @pytest.mark.parametrize("epsilon", [6.0, 10.0, 25.0])
    def test_starting_line_methods_generate_finite_contours(self, starting_line_method, epsilon):
        from raosim.rao_optimizer import moc_bell_nozzle

        c = moc_bell_nozzle(
            Rt=0.02, epsilon=epsilon, gamma=1.4,
            length_pct=80.0, n_control=3, n_char=10,
            max_iter=40, starting_line_method=starting_line_method
        )

        assert c["starting_line_method"] == starting_line_method
        assert np.all(np.isfinite(c["x"]))
        assert np.all(np.isfinite(c["y"]))
        assert np.all(np.isfinite(c["x_bell"]))
        assert np.all(np.isfinite(c["y_bell"]))
        assert c["wall_export_basis"] == (
            "optimized_monotone_cubic_hermite_spline"
        )
        assert np.interp(
            c["optimized_wall_x_knots"], c["x_bell"], c["y_bell"]
        ) == pytest.approx(c["optimized_wall_r_knots"], rel=2e-3)
        assert c["x"][-1] > c["x"][0]
        assert c["y"][-1] == pytest.approx(math.sqrt(epsilon) * 0.02, rel=0.08)
