"""Phase 13 smoke tests (REWRITE_PLAN §12 / test plan row 13).

Headless (Agg) smoke coverage: every plot function must produce a figure
from representative inputs without a solver run.  plot_topology gets a
real Phase-12.6 topology (cheap: one fixed-end construction); the
solution-based plots get a minimal duck-typed solution.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")  # noqa: E402  (before pyplot import via raosim.plotting)

import numpy as np
import pytest

from raosim.moc import CharPoint, FlowNode
from raosim.gas_dynamics import mach_angle, prandtl_meyer
from raosim.plotting import (plot_characteristic_net, plot_flowfield_mach,
                             plot_topology)


def _cp(x, r, theta, M, gamma=1.4):
    nu = prandtl_meyer(max(M, 1.000001), gamma)
    return CharPoint(x=x, r=r, theta=theta, M=max(M, 1.000001), nu=nu,
                     mu=mach_angle(max(M, 1.000001)),
                     compat_plus=theta + nu, compat_minus=theta - nu)


class _Row:
    def __init__(self, pts):
        self._pts = pts

    def all_points(self):
        return self._pts


class _FakeSolution:
    """Duck-typed minimal RaoSolution for plotting smoke tests."""

    def __init__(self):
        x = np.linspace(0.0, 0.12, 30)
        r = 0.02 + 0.04 * np.sqrt(x / 0.12)
        self.wall_raw = np.column_stack([x, r])
        self.wall_export = self.wall_raw.copy()
        self.kernel_points = [_cp(0.0, ri, 0.0, 1.2)
                              for ri in np.linspace(0.0, 0.02, 6)]
        self.characteristic_net = [
            _Row([_cp(xi, ri, 0.1, 2.0)
                  for xi, ri in zip(np.linspace(0.01 * j, 0.05 + 0.01 * j, 8),
                                    np.linspace(0.0, 0.03, 8))])
            for j in range(4)
        ]

        class _CE:
            x = np.linspace(0.01, 0.12, 10)
            r = np.linspace(0.005, 0.063, 10)

        self.control_surface = _CE()


def test_plot_characteristic_net_smoke():
    fig = plot_characteristic_net(_FakeSolution(), geometry="raw")
    assert fig is not None
    assert fig.axes, "no axes produced"


def test_plot_characteristic_net_export_mode():
    fig = plot_characteristic_net(_FakeSolution(), geometry="export")
    assert fig is not None


def test_plot_flowfield_mach_smoke():
    fig = plot_flowfield_mach(_FakeSolution())
    assert fig is not None


def test_plot_topology_smoke_with_real_construction():
    from raosim.moc_topology import build_topology
    from raosim.nasa_moc import calc_bde_region, set_theta_b
    from raosim.rao_variational import _target_length

    Rt = 0.020
    Ln = _target_length(Rt, 10.0, 80.0)
    nasa_topo, kernel = set_theta_b(
        Rt, 10.0, 80.0, 1.4, 0.01,
        theta_b_init_deg=21.87, n_kernel=24, n_de_points=24,
        starting_line_method="kliegel_levine", L_target=Ln,
        Ru=1.5 * Rt, end_condition="fixed_end", max_iter=30,
    )
    topo = build_topology(kernel, nasa_topo, calc_bde_region(kernel, nasa_topo))
    fig = plot_topology(topo)
    assert fig is not None
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert "streamline BE (bell wall)" in labels
    assert "BD (mass-flow curve)" in labels


def test_plot_topology_duck_typed_minimal():
    class _Mini:
        B = _cp(0.003, 0.0207, math.radians(25.5), 2.16)
        D = _cp(0.05, 0.012, math.radians(18.0), 3.4)
        E = _cp(0.129, 0.0632, math.radians(8.0), 3.9)
        theta_B = math.radians(25.5)

    fig = plot_topology(_Mini())
    assert fig is not None


# ---------------------------------------------------------------------
#  Phase 13 batch (§12.1 plots #1, #4, #5, #6, #7, #10)
# ---------------------------------------------------------------------


def test_plot_nozzle_geometry_smoke():
    from raosim.plotting import plot_nozzle_geometry

    fig = plot_nozzle_geometry(_FakeSolution())
    assert fig is not None


def test_plot_flowfield_pressure_and_theta_smoke():
    from raosim.plotting import plot_flowfield_pressure, plot_flowfield_theta

    sol = _FakeSolution()
    assert plot_flowfield_pressure(sol, 1.4) is not None
    assert plot_flowfield_theta(sol) is not None


def test_plot_wall_distributions_smoke():
    from raosim.plotting import plot_wall_distributions

    fig = plot_wall_distributions(_FakeSolution(), 1.4)
    assert fig is not None
    assert len(fig.axes) == 3


def test_plot_wall_distributions_requires_net():
    from raosim.plotting import plot_wall_distributions

    sol = _FakeSolution()
    sol.characteristic_net = []
    with pytest.raises(ValueError, match="characteristic_net"):
        plot_wall_distributions(sol, 1.4)


def test_plot_exit_plane_smoke():
    from raosim.plotting import plot_exit_plane

    fig = plot_exit_plane(_FakeSolution(), 1.4, x_band=0.5)
    assert fig is not None
    assert len(fig.axes) == 3


def test_plot_nasa_overlay_smoke():
    from pathlib import Path

    from raosim.plotting import plot_nasa_overlay

    nasa_dir = (Path(__file__).resolve().parent.parent
                / "Three-Dimensional-Nozzle-Design-Code-master"
                / "MOC_Grid_BDE" / "outputs_M3.5Perf")
    if not nasa_dir.exists():
        pytest.skip("NASA M3.5Perf reference outputs not present")
    fig = plot_nasa_overlay(_FakeSolution(), nasa_dir, Rt=0.02)
    assert fig is not None
