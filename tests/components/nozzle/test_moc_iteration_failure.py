import math
from types import SimpleNamespace

import pytest

from lrekit.components.nozzle.solvers.nasa import (
    ArcWall,
    MOCNode,
    _calc_arc_wall_point_raw,
    _calc_bde_back_point,
    _finite_iteration_change,
    _make_throat_initial_line,
)


@pytest.fixture(scope="module")
def nasa_arc_parents():
    arc = ArcWall(1.0, 1.0, math.radians(15.2196))
    parents = list(reversed(_make_throat_initial_line(
        1.0, 1.0, arc.theta_max, 1.4, 101, "nasa_visible_kliegel_levine",
    )))
    return arc, parents


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_arc_wall_rejects_unconverged_iterates(nasa_arc_parents, budget):
    arc, parents = nasa_arc_parents
    assert _calc_arc_wall_point_raw(parents, arc, 1.4, max_iter=budget) is None
    assert _calc_arc_wall_point_raw(parents, arc, 1.4) is not None


def test_backward_bde_rejects_unconverged_iterates_but_accepts_zero_angle():
    parent = MOCNode(0.0, 0.8, 2.0, 0.0, 1.4)
    downstream = MOCNode(0.2, 1.0, 2.0, 0.0, 1.4)
    for budget in (0, 1):
        assert _calc_bde_back_point([parent], [None, downstream], 0, 1.4, max_iter=budget) is None
    result = _calc_bde_back_point([parent], [None, downstream], 0, 1.4, max_iter=2)
    assert result is not None
    assert result.M == 2.0
    assert result.theta == 0.0
    assert result.x == pytest.approx(0.27320508075688776)
    assert result.r == pytest.approx(0.9577350269189626)


@pytest.mark.parametrize(("new", "old", "expected"), [
    (0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (1.0, 0.0, float("inf")),
    (float("nan"), 1.0, float("inf")), (1.0, float("inf"), float("inf")),
])
def test_iteration_change_treats_nonfinite_state_as_failure(new, old, expected):
    assert _finite_iteration_change(new, old) == expected


def test_local_processes_reject_nonfinite_parent_state(nasa_arc_parents):
    arc, parents = nasa_arc_parents
    invalid = SimpleNamespace(x=float("inf"), r=1.0, M=2.0, theta=0.0)
    assert _calc_arc_wall_point_raw([parents[0], invalid], arc, 1.4) is None
    finite = MOCNode(0.2, 1.0, 2.0, 0.0, 1.4)
    assert _calc_bde_back_point([invalid], [None, finite], 0, 1.4) is None
