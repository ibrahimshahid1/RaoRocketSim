"""
MOC / Rao construction diagrams (raosim.moc_diagrams): the throat kernel
expansion fan, the Rao B-D-E topology, and the BDE characteristic net,
rendered from the in-memory artifacts the ``bde`` wall method stashes on
the solution.  Headless (Agg) smoke coverage.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest

from raosim.moc_diagrams import (
    plot_bde_mesh,
    plot_kernel_expansion_fan,
    plot_rao_topology,
)
from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp


@pytest.fixture(scope="module")
def bde_solution():
    # Seed-only (fast) but MOC-evaluated via the bde wall method, which is
    # what stashes ``bde_artifacts`` on the solution.
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=16, n_kernel=16,
        max_nfev=0, evaluate_moc=True, wall_method="bde")
    return solve_rao_bvp(cfg)


def test_bde_artifacts_are_stashed(bde_solution):
    art = bde_solution.construction_diagnostics.get("bde_artifacts")
    assert isinstance(art, dict)
    assert art["kernel"] is not None
    assert art["nasa_topology"] is not None
    assert art["bde_region"] is not None


def test_plot_kernel_expansion_fan_renders(bde_solution, tmp_path):
    out = tmp_path / "kernel_fan.png"
    fig = plot_kernel_expansion_fan(bde_solution, save_path=str(out))
    assert out.exists()
    assert "expansion fan" in fig.axes[0].get_title()


def test_plot_rao_topology_renders(bde_solution, tmp_path):
    out = tmp_path / "rao_topology.png"
    fig = plot_rao_topology(bde_solution, save_path=str(out))
    assert out.exists()
    assert "B-D-E" in fig.axes[0].get_title()


def test_plot_bde_mesh_renders(bde_solution, tmp_path):
    out = tmp_path / "bde_mesh.png"
    fig = plot_bde_mesh(bde_solution, save_path=str(out))
    assert out.exists()
    # full net + near-axis zoom panels.
    assert len(fig.axes) >= 2


def test_diagrams_need_bde_artifacts():
    """Without the bde artifacts the diagrams say so instead of crashing."""
    class _Empty:
        construction_diagnostics: dict = {}
    with pytest.raises(ValueError, match="BDE construction artifacts"):
        plot_bde_mesh(_Empty())
