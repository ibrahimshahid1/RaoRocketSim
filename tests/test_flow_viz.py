"""
Steady flow-field visualisation (raosim.flow_viz): the MOC net is
interpolated into smooth Mach/temperature fields with characteristics
and streamlines.  Headless (Agg) smoke coverage.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from raosim.flow_viz import _field_nodes, plot_flowfield
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
                         save_path=str(out))
    assert out.exists()
    # Mach + temperature panels (+ their colorbars).
    assert len(fig.axes) >= 2


def test_flowfield_is_physical(solution):
    """After filtering, the displayed Mach stays in the physical band
    for ε=10 (no runaway-node spikes)."""
    fig = plot_flowfield(solution, gamma=1.24)
    ax = fig.axes[0]
    title = ax.get_title()
    assert "exit M" in title
    # exit Mach for ε=10, γ=1.24 is ~3.9–4.2; never the 100 a raw net
    # node can carry.
    import re
    m = float(re.search(r"exit M.?([0-9.]+)", title).group(1))
    assert 3.0 < m < 5.0


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
    anim = animate_particles(solution, gamma=1.24, n_frames=20, n_particles=120)
    assert isinstance(anim, FuncAnimation)
    for k in range(5):                       # advect a few steps
        anim._func(k)
