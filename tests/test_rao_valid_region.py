"""Tests for the Rao smooth-flow validity inequality and diagnostic plots.

The inequality follows Rao 1958 / Rao-Beck-Booth 1999 (AIAA 99-2584):

    b(s) = 1 - (dα/dθ) [tan(θ-α) + tan(α)] / [tan(θ-α) - tan(α)] ≥ 0

with α = arcsin(1/M).  When ``b`` becomes negative the smooth-flow
Rao-optimum construction is inapplicable.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from raosim.moc import FlowNode
from raosim.plotting import plot_characteristic_net, plot_flowfield_mach
from raosim.rao_variational import (
    ContourReliability,
    RaoSolverConfig,
    rao_valid_region,
    solve_rao_bvp,
)


def _make_smooth_ce_nodes() -> list[FlowNode]:
    """A monotone, gentle expansion -- well inside the valid region."""
    Ms = np.linspace(2.0, 3.5, 8)
    thetas = np.linspace(math.radians(28.0), math.radians(8.0), 8)
    rs = np.linspace(0.025, 0.063, 8)
    xs = np.linspace(0.0, 0.10, 8)
    return [FlowNode(x=float(x), r=float(r), M=float(M), theta=float(t))
            for x, r, M, t in zip(xs, rs, Ms, thetas)]


def _make_short_nozzle_ce_nodes() -> list[FlowNode]:
    """Large M jump across small theta turn -- violates the Rao inequality.

    With theta near 2*alpha the denominator (tan(theta-alpha) - tan(alpha))
    shrinks while (dalpha/dtheta) is large, driving b strongly negative.
    This is the discontinuous-exit regime Rao 1958 explicitly excludes.
    """
    nodes = [
        FlowNode(x=0.000, r=0.0250, M=2.0, theta=math.radians(60.0)),
        FlowNode(x=0.005, r=0.0270, M=4.0, theta=math.radians(55.0)),
        FlowNode(x=0.010, r=0.0300, M=5.0, theta=math.radians(40.0)),
    ]
    return nodes


def test_valid_region_returns_inf_for_too_few_nodes():
    min_b, vals = rao_valid_region([])
    assert math.isinf(min_b)
    assert vals == []


def test_valid_region_quiet_on_smooth_expansion():
    nodes = _make_smooth_ce_nodes()
    min_b, vals = rao_valid_region(nodes)
    assert vals, "expected per-segment b values"
    assert min_b > 0.0, f"smooth Rao expansion should satisfy b > 0, got {min_b}"


def test_valid_region_fires_on_pathological_case():
    nodes = _make_short_nozzle_ce_nodes()
    min_b, vals = rao_valid_region(nodes)
    assert vals
    assert min_b < 0.0, (
        "low-Mach sharp-turn expansion should violate the Rao smooth-flow "
        f"inequality; got min_b={min_b}"
    )


def test_solve_rao_bvp_populates_rao_region_diagnostics():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=0,
        evaluate_moc=False,
    )
    solution = solve_rao_bvp(cfg)
    diag = solution.construction_diagnostics
    assert "rao_region" in diag
    assert "boundary_min" in diag
    assert "requires_discontinuous_exit_flow_model" in diag
    assert diag["rao_region"] in {
        "valid_shock_free_region",
        "invalid_short_nozzle_region",
    }


def test_invalid_region_downgrades_reliability():
    """If b<0 along CE, the reliability tier must not claim residual-solved."""
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
        max_nfev=0,
        evaluate_moc=False,
    )
    solution = solve_rao_bvp(cfg)

    if solution.construction_diagnostics["rao_region"] == "invalid_short_nozzle_region":
        assert solution.reliability == ContourReliability.GEOMETRIC_APPROXIMATION
        assert any("smooth-flow" in w.lower() or "discontinuous" in w.lower()
                   for w in solution.warnings)


def test_phase5_invalid_region_test_vector():
    """
    Phase 5 reliability cliff: pass an explicit pathological CE polyline
    through ``rao_valid_region`` and assert the inequality fires.  The
    plan §2.E identifies very short / over-expanded designs as the
    canonical invalid region; this test pins the gate against the
    synthetic short-nozzle CE polyline from ``_make_short_nozzle_ce_nodes``.
    """
    nodes = _make_short_nozzle_ce_nodes()
    min_b, b_values = rao_valid_region(nodes)

    assert min_b < 0.0, (
        "Pathological short-nozzle CE should violate the Rao validity "
        f"inequality, but min boundary value was {min_b:.3g}."
    )
    assert any(b < 0.0 for b in b_values), (
        "At least one BD segment should report a negative boundary value."
    )


# ---------------------------------------------------------------------
# Plot smoke tests (raw-vs-export trap, §12.3 of REWRITE_PLAN.md)
# ---------------------------------------------------------------------

def _fake_solution_for_plot():
    """Tiny solution-like namespace with two-row characteristic net."""
    class _FakeRow:
        def __init__(self, pts):
            self._pts = pts

        def all_points(self):
            return self._pts

    pts0 = [
        SimpleNamespace(x=0.0, r=0.020, M=1.0, theta=math.radians(0.0)),
        SimpleNamespace(x=0.005, r=0.022, M=1.2, theta=math.radians(8.0)),
    ]
    pts1 = [
        SimpleNamespace(x=0.01, r=0.0, M=1.5, theta=0.0),
        SimpleNamespace(x=0.02, r=0.030, M=1.8, theta=math.radians(15.0)),
    ]
    raw_wall = np.array([[0.005, 0.022], [0.020, 0.030], [0.040, 0.060]],
                        dtype=float)
    export_wall = np.array([[0.005, 0.020], [0.020, 0.040], [0.040, 0.063]],
                           dtype=float)
    ce = SimpleNamespace(
        x=np.array([0.005, 0.020, 0.040]),
        r=np.array([0.022, 0.030, 0.060]),
    )
    kernel = [
        SimpleNamespace(x=0.0, r=0.020 + 0.001 * i,
                        M=1.0 + 0.05 * i, theta=math.radians(2.0 * i))
        for i in range(4)
    ]
    return SimpleNamespace(
        wall_raw=raw_wall,
        wall_export=export_wall,
        control_surface=ce,
        kernel_points=kernel,
        characteristic_net=[_FakeRow(pts0), _FakeRow(pts1)],
    )


def test_plot_characteristic_net_smoke():
    fig = plot_characteristic_net(_fake_solution_for_plot(), show=False)
    assert fig is not None


def test_plot_characteristic_net_geometry_switch_changes_data():
    sol = _fake_solution_for_plot()
    fig_raw = plot_characteristic_net(sol, geometry="raw", show=False)
    fig_exp = plot_characteristic_net(sol, geometry="export", show=False)
    raw_lines = fig_raw.axes[0].lines[0].get_ydata()
    exp_lines = fig_exp.axes[0].lines[0].get_ydata()
    # Raw and export wall arrays differ by construction in the fake fixture.
    assert not np.allclose(raw_lines, exp_lines)


def test_plot_characteristic_net_rejects_bad_geometry():
    with pytest.raises(ValueError):
        plot_characteristic_net(_fake_solution_for_plot(), geometry="oops",
                                show=False)


def test_plot_flowfield_mach_smoke():
    fig = plot_flowfield_mach(_fake_solution_for_plot(), show=False)
    assert fig is not None
