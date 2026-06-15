"""Phase 12.6: full-form RaoTopology (REWRITE_PLAN §11.7).

Builds the explicit topology for the Rao reference design point
(epsilon=10, length_pct=80, Rd=0.382Rt, Ru=1.5Rt) via the now-converging
fixed-end construction (post-Phase-12.4 kernel march) and pins:

* the §11.7 field set, with streamline_BE as THE bell wall;
* seam/closure metrics (B/D/E attachments, mass pair);
* the assembled full wall (throat arc + streamline_BE): monotone in x,
  exit on the commanded station, TOP bell slope profile.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.moc import CharPoint
from raosim.moc_topology import RaoTopology, build_reference_topology
from raosim.rao_variational import _target_length

GAMMA = 1.4
RT = 0.020
EPSILON = 10.0
LENGTH_PCT = 80.0


@pytest.fixture(scope="module")
def reference_topology():
    Ln = _target_length(RT, EPSILON, LENGTH_PCT)
    topo = build_reference_topology(
        RT, EPSILON, LENGTH_PCT, GAMMA, 0.01,
        n_kernel=24, n_de_points=24,
    )
    return topo, Ln


def test_field_set_and_types(reference_topology):
    topo, _ = reference_topology
    assert isinstance(topo, RaoTopology)
    for name in ("TT_prime", "BF", "BD", "DE", "streamline_BE"):
        seq = getattr(topo, name)
        assert len(seq) >= 2, name
        assert all(isinstance(p, CharPoint) for p in seq), name
    for name in ("B", "D", "E"):
        assert isinstance(getattr(topo, name), CharPoint), name
    # B is the wall end of BF; BF spans wall -> axis.
    assert topo.B.x == topo.BF[0].x
    assert topo.BF[-1].r == pytest.approx(0.0, abs=1e-9)
    # theta_B in the post-12.4 band for this design point.
    assert 24.0 < math.degrees(topo.theta_B) < 27.0
    assert topo.diagnostics["kernel_reached_wall"]
    assert topo.diagnostics["bfe_wall_contour_complete"]


def test_closure_seams(reference_topology):
    topo, _ = reference_topology
    rep = topo.closure_report()
    Rt = RT
    # mass pair exact by construction (fixed-end walks D until it is).
    assert rep["mass_rel_mismatch"] < 1e-6
    # attachments: BD runs B -> D; DE runs D -> E; wall runs B -> E.
    assert rep["BD_starts_at_B"] < 1e-9 * Rt + 1e-12
    assert rep["BD_ends_at_D"] < 1e-6 * Rt
    assert rep["DE_starts_at_D"] < 1e-6 * Rt
    assert rep["DE_ends_at_E"] < 1e-6 * Rt
    assert rep["wall_starts_at_B"] < 5e-3 * Rt
    assert rep["wall_ends_at_E"] < 5e-3 * Rt


def test_full_wall_is_bell_on_commanded_station(reference_topology):
    topo, Ln = reference_topology
    Re = math.sqrt(EPSILON) * RT
    wall = topo.full_wall()
    assert wall.ndim == 2 and wall.shape[1] == 2
    assert np.all(np.diff(wall[:, 0]) > -1e-12)
    assert wall[-1, 0] == pytest.approx(Ln, rel=1e-3)
    assert wall[-1, 1] == pytest.approx(Re, rel=1e-3)
    ang = np.degrees(np.arctan2(np.diff(wall[:, 1]), np.diff(wall[:, 0])))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(wall[:, 0]),
                                                  np.diff(wall[:, 1])))])
    i_pk = int(np.argmax(ang))
    # TOP bell: peak ~ theta_B just past the throat arc, monotone after.
    assert s[i_pk] / s[-1] < 0.10
    assert ang.max() == pytest.approx(math.degrees(topo.theta_B), abs=1.5)
    assert np.all(np.diff(ang[i_pk:]) <= 0.25)
    assert 6.0 <= ang[-1] <= 14.0


def test_bd_is_prefix_of_bf_plus_interpolated_d(reference_topology):
    topo, _ = reference_topology
    # Every BD node except the last is a BF node (same objects by value).
    for p, q in zip(topo.BD[:-1], topo.BF):
        assert p.x == q.x and p.r == q.r
    # ... and the last is D itself.
    assert topo.BD[-1].x == pytest.approx(topo.D.x, abs=1e-12)
    assert topo.BD[-1].r == pytest.approx(topo.D.r, abs=1e-12)
    # D interior on BF (not collapsed onto B or the axis end).
    assert 0.0 < topo.d_fraction < 1.0


def test_solve_rao_bvp_exports_solved_topology():
    """§11.7→12.7 wiring: the BDE wall path lifts the SOLVED state into
    the full-form RaoTopology and attaches it to the solution
    (sol.topology_solved + the closure floats in diagnostics)."""
    import math

    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=8, n_kernel=24,
        max_nfev=0, evaluate_moc=True, wall_method="bde",
    )
    sol = solve_rao_bvp(cfg)
    topo = sol.topology_solved
    assert topo is not None
    # Geometric seams close exactly (mass mismatch reflects the
    # deliberately unsolved max_nfev=0 state and is NOT asserted here).
    rep = topo.closure_report()
    for seam in ("BD_starts_at_B", "DE_starts_at_D", "DE_ends_at_E",
                 "wall_starts_at_B", "wall_ends_at_E"):
        assert abs(rep[seam]) < 1e-12, seam
    # The solved topology's corner angle is the kernel's theta_B (the
    # J5-reported sol.theta_N).
    assert topo.theta_B == sol.theta_N
    # Wall ends on the commanded exit station.
    w = topo.full_wall()
    Re = math.sqrt(cfg.epsilon) * cfg.Rt
    assert w[-1, 1] == abs(w[-1, 1])
    assert abs(w[-1, 1] - Re) < 5e-4
    assert "topology_closure" in sol.construction_diagnostics
