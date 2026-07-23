"""
Steady flow-field visualisation (raosim.flow_viz): the MOC net is
interpolated into Mach/pressure/flow-angle/temperature fields with
characteristics and streamlines.  Headless (Agg) smoke coverage.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from types import SimpleNamespace

from raosim.flow_viz import _axisymmetric_exit_mean, _field_nodes, plot_flowfield
from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp


@pytest.fixture(scope="module")
def solution():
    # Seed-only solve (fast) but with the MOC net evaluated.
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.24, pa_over_p0=0.01,
        length_pct=80.0, n_control=16, n_kernel=16,
        max_nfev=0, evaluate_moc=True, wall_method="coupled")
    return solve_rao_bvp(cfg)


def test_field_nodes_present(solution):
    x, r, M, theta = _field_nodes(solution)
    assert x.size > 20
    assert x.shape == r.shape == M.shape == theta.shape


def test_plot_flowfield_renders(solution, tmp_path):
    out = tmp_path / "field.png"
    fig = plot_flowfield(solution, gamma=1.24, Tc=3500.0,
                         save_path=str(out), allow_partial=True)
    assert out.exists()
    # Mach, pressure, flow-angle, and temperature panels (+ colorbars).
    assert len(fig.axes) >= 4
    assert "Partial MOC construction field" in fig._suptitle.get_text()


def test_flowfield_is_physical(solution):
    """After filtering, the displayed Mach stays in the physical band
    for ε=10 (no runaway-node spikes)."""
    fig = plot_flowfield(
        solution, gamma=1.24, exit_mach=2.78, allow_partial=True,
    )
    assert "design exit M≈2.78" in fig._suptitle.get_text()


def test_partial_field_is_rejected_by_default(solution, tmp_path):
    stale = tmp_path / "stale-field.png"
    stale.write_bytes(b"old misleading plot")
    with pytest.raises(ValueError, match="partial MOC coverage"):
        plot_flowfield(solution, gamma=1.24, save_path=str(stale))
    assert not stale.exists()


def _complete_bde_solution():
    def node(x, r, M=2.0, theta=0.05):
        return SimpleNamespace(x=float(x), r=float(r), M=float(M),
                               theta=float(theta))

    wall = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 2.0]])
    # Synthetic wall-to-axis stations stand in for the complete kernel and
    # make the coverage expectation unambiguous.
    rrcs = []
    for x in np.linspace(0.2, 2.3, 9):
        rw = float(np.interp(x, wall[:, 0], wall[:, 1]))
        rrcs.append([
            node(x, frac * rw, 1.2 + 0.7 * x + 0.1 * frac,
                 0.08 * frac)
            for frac in np.linspace(1.0, 0.0, 9)
        ])
    grid_rows = (
        (node(0.4, 1.2), node(0.8, 0.9)),
        (node(1.2, 1.6), node(1.5, 1.1)),
    )
    # Index 0 is the raw exterior prefix.  It must not enter the plotted
    # field; index 1 onward is the valid DE-to-axis continuation.
    full_rows = (
        (node(0.1, 9.0, 99.0), node(0.8, 0.9, 2.1),
         node(2.2, 0.2, 3.7)),
        (node(0.2, 9.0, 99.0), node(1.5, 1.1, 2.5),
         node(2.25, 0.3, 3.8)),
    )
    bde = SimpleNamespace(
        iD=1,
        grid_rows=grid_rows,
        full_grid_rows=full_rows,
        wall_contour=(node(0.4, 1.2), node(1.2, 1.6), node(2.0, 2.0)),
        wall_contour_complete=True,
        complete_remaining_mesh=False,
    )
    kernel = SimpleNamespace(rrcs=rrcs)
    return SimpleNamespace(
        wall_raw=wall,
        wall_export=wall,
        thrust_coefficient=1.5,
        characteristic_net=[],
        kernel_points=[],
        construction_diagnostics={
            "bde_artifacts": {"kernel": kernel, "bde_region": bde},
            "net_report": {"bde_physical_mesh_complete": True},
        },
    )


def test_bde_field_uses_all_regions_and_passes_coverage(tmp_path):
    solution = _complete_bde_solution()
    x, _, M, _ = _field_nodes(solution)
    assert x.size > 50
    assert np.max(M) < 99.0
    assert np.any(np.isclose(M, 3.8))

    out = tmp_path / "complete-bde-field.png"
    fig = plot_flowfield(
        solution, gamma=1.24, save_path=str(out),
        show_characteristics=False, show_streamlines=False,
    )
    assert out.exists()
    assert fig.flowfield_coverage["full_field"] is True
    assert fig.flowfield_coverage["exit_radial_fraction"] >= 0.98
    assert "Steady supersonic MOC field" in fig._suptitle.get_text()


def test_axisymmetric_exit_mean_uses_last_populated_cut():
    radius = np.array([0.0, 0.5, 1.0])
    Rg = np.column_stack([radius, radius, radius])
    Mg = np.array([
        [2.0, 3.0, np.nan],
        [2.0, 3.0, np.nan],
        [2.0, 3.0, np.nan],
    ])

    assert _axisymmetric_exit_mean(Rg, Mg) == pytest.approx(3.0)


def test_flowfield_needs_a_net():
    """Without an evaluated net the function says so."""
    class _Empty:
        characteristic_net = []
        kernel_points = []
        wall_raw = np.array([[0.0, 0.02], [0.1, 0.06]])
    with pytest.raises(ValueError, match="characteristic net"):
        plot_flowfield(_Empty(), gamma=1.24)


# ---------------------------------------------------------------------
#  Animations (build the FuncAnimation + step the update fn; no GIF I/O
#  so they stay fast).
# ---------------------------------------------------------------------


def test_animate_moc_march_builds(solution):
    from matplotlib.animation import FuncAnimation
    from raosim.flow_viz import animate_moc_march
    anim = animate_moc_march(solution, gamma=1.24)
    assert isinstance(anim, FuncAnimation)
    anim._func(0); anim._func(3)            # reveal rows without error


def test_animate_particles_builds(solution):
    from matplotlib.animation import FuncAnimation
    from raosim.flow_viz import animate_particles
    anim = animate_particles(
        solution, gamma=1.24, n_frames=20, n_particles=120,
        allow_partial=True,
    )
    assert isinstance(anim, FuncAnimation)
    for k in range(5):                       # advect a few steps
        anim._func(k)
