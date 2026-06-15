"""Characteristic invariant pairing — the 2026-06-11 correction.

Root cause (identified by ibrahim): the rao_residuals/moc/jax stack
enforced d(θ+ν) along the C+ (θ+μ) family and d(θ−ν) along C− — the
families' relations mirrored — and the axisymmetric source carried a
spurious cosμ/cos(θ±μ) factor.  Correct relations (Anderson, *Modern
Compressible Flow* §11.4; Zucrow & Hoffman Vol. 2 Ch. 17), nodes
ordered downstream, S = sinθ sinμ / r:

    C− (slope θ−μ):  d(θ + ν) = +S ds
    C+ (slope θ+μ):  d(θ − ν) = −S ds

These tests pin the correction against two independent oracles:
the M3.5Perf kernel march (RRC rows are true C− characteristics,
oracle-matched to NASA output files) and the verbatim NASA
``Deriv``/``RungeKutta`` port (LRC integration with the axisymmetric
terms in closed form).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.gas_dynamics import prandtl_meyer
from raosim.moc import FlowNode, solve_interior_point, _make_point
from raosim.nasa_moc import build_kernel, nasa_runge_kutta
from raosim.rao_residuals import (residual_Cminus_axisym,
                                  residual_Cplus_axisym)

GAMMA = 1.4


@pytest.fixture(scope="module")
def m35_kernel():
    return build_kernel(1.0, 1.0, math.radians(15.2196), GAMMA, 101,
                        starting_line_method="nasa_visible_kliegel_levine")


def _flow(p):
    return FlowNode(x=p.x, r=p.r, M=p.M, theta=p.theta)


def _rrc_segments(kernel, rows=(20, 40, -1), r_min=0.05):
    for j in rows:
        row = kernel.rrcs[j]
        for p0, p1 in zip(row[:-1], row[1:]):
            if p0.r < r_min or p1.r < r_min:
                continue
            yield _flow(p0), _flow(p1)


def test_cminus_relation_vanishes_on_oracle_rrc(m35_kernel):
    """d(θ+ν) = +S ds must hold on true C− (kernel RRC) segments."""
    good, bad = [], []
    for p0, p1 in _rrc_segments(m35_kernel):
        good.append(residual_Cminus_axisym(p0, p1, GAMMA))
        bad.append(residual_Cplus_axisym(p0, p1, GAMMA))
    good = np.abs(good)
    bad = np.abs(bad)
    rms_good = float(np.sqrt((good ** 2).mean()))
    rms_bad = float(np.sqrt((bad ** 2).mean()))
    # Observed at this resolution: 2.3e-6 vs 5.5e-3.
    assert rms_good < 5e-5, f"C− relation off on true C−: rms={rms_good:.3e}"
    assert rms_bad > 50 * rms_good, (
        "the C+ relation should NOT hold along C− — if it does, the "
        f"pairing regressed (good={rms_good:.3e}, bad={rms_bad:.3e})"
    )


def test_cplus_relation_vanishes_on_nasa_deriv_lrc(m35_kernel):
    """d(θ−ν) = −S ds must hold on a NASA-``Deriv``-integrated LRC."""
    p = m35_kernel.bd[len(m35_kernel.bd) // 2]
    M, x, th, r = p.M, p.x, p.theta, p.r
    pts = [(x, r, M, th)]
    for _ in range(40):
        out = nasa_runge_kutta(0.02, r, x, M, th, GAMMA)
        if out is None:
            break
        M, x, th, r = out
        pts.append((x, r, M, th))
    assert len(pts) > 20

    res = []
    for (x0, r0, M0, t0), (x1, r1, M1, t1) in zip(pts[:-1], pts[1:]):
        res.append(residual_Cplus_axisym(
            FlowNode(x=x0, r=r0, M=M0, theta=t0),
            FlowNode(x=x1, r=r1, M=M1, theta=t1), GAMMA))
    rms = float(np.sqrt((np.abs(res) ** 2).mean()))
    # Observed: 8.8e-8 (RK4 truncation scale).
    assert rms < 1e-6, f"C+ relation off on a true LRC: rms={rms:.3e}"


def test_planar_invariants_are_correct_families():
    """axisymmetric=False must check K+ = θ−ν (C+) and K− = θ+ν (C−)."""
    p0 = FlowNode(x=0.0, r=1.0, M=2.0, theta=math.radians(10.0))
    # Along planar C+, θ−ν is constant: change both consistently.
    nu0 = prandtl_meyer(2.0, GAMMA)
    nu1 = nu0 + math.radians(3.0)
    from raosim.gas_dynamics import mach_from_prandtl_meyer
    M1 = mach_from_prandtl_meyer(nu1, GAMMA)
    p1 = FlowNode(x=0.1, r=1.1, M=M1,
                  theta=p0.theta + math.radians(3.0))  # dθ = dν
    assert residual_Cplus_axisym(p0, p1, GAMMA,
                                 axisymmetric=False) == pytest.approx(
        0.0, abs=1e-9)
    assert abs(residual_Cminus_axisym(p0, p1, GAMMA,
                                      axisymmetric=False)) > 1e-3


def test_interior_unit_process_reconstructs_oracle_node(m35_kernel):
    """solve_interior_point must land on the oracle mesh node.

    Parents per the NASA connectivity: the C− parent is the previous
    node along the same RRC (wall side); the C+ parent is the paired
    node on the previous RRC.  The corrected unit process must
    reproduce the marched node; the swapped-invariant version misses
    badly in θ/M.
    """
    rows = m35_kernel.rrcs
    j = 40
    i = len(rows[j]) // 2
    # C- parent: previous node along RRC j (wall side, ordered wall->axis).
    pm = rows[j][i - 1]
    # C+ parent: node on RRC j-1 paired with i (NASA ii = i+1 when no
    # special insertion at this j; tolerate either by picking the
    # geometric parent that the C+ through node (i,j) actually passes).
    target = rows[j][i]
    candidates = rows[j - 1][max(i - 1, 0):i + 2]
    pp = min(candidates,
             key=lambda q: abs((target.r - q.r)
                               - math.tan(0.5 * (q.theta + target.theta)
                                          + 0.5 * (math.asin(1 / q.M)
                                                   + math.asin(1 / target.M)))
                               * (target.x - q.x)))

    p_minus = _make_point(pm.x, pm.r, pm.theta, pm.M, GAMMA)
    p_plus = _make_point(pp.x, pp.r, pp.theta, pp.M, GAMMA)
    sol = solve_interior_point(p_minus, p_plus, GAMMA)

    # Tolerances reflect the different unit-process discretisations
    # (NASA dθ-form vs invariant predictor-corrector), not equality.
    assert sol.M == pytest.approx(target.M, rel=2e-3)
    assert sol.theta == pytest.approx(target.theta, abs=math.radians(0.2))
    assert sol.x == pytest.approx(target.x, rel=5e-3, abs=1e-4)
    assert sol.r == pytest.approx(target.r, rel=5e-3, abs=1e-4)


def test_smooth_existence_root_regression():
    """The corrected-formulation smooth root at the reference point.

    Solver-independent construction (no LM): NASA fixed-end closure
    (set_theta_b secant on length; calc_lrc_de walks D so DE — the
    NASA-``Deriv`` stationary-DE integration — pins r_E and carries
    exactly the BD mass).  ibrahim's direct RK existence scan found
    the same root, closing all three targets to ~1e-8:

        theta_B = 25.5659 deg   kdf = 0.15216
        D: M = 3.40145, theta = 18.5182 deg
        E: M = 3.47655, theta = 11.1193 deg

    This regression pins it; full D-state continuity is therefore
    satisfiable (smooth attachment) and the Guderley-jump hypothesis
    stays shelved unless the corrected full-pin BVP refuses to close.
    """
    from raosim.moc_topology import build_reference_topology

    topo = build_reference_topology(0.020, 10.0, 80.0, GAMMA, 0.01,
                                    n_kernel=24, n_de_points=24)
    assert math.degrees(topo.theta_B) == pytest.approx(25.566, abs=0.05)
    assert topo.d_fraction == pytest.approx(0.1522, abs=0.003)
    assert topo.D.M == pytest.approx(3.4015, abs=0.01)
    assert math.degrees(topo.D.theta) == pytest.approx(18.518, abs=0.1)
    assert topo.E.M == pytest.approx(3.4766, abs=0.01)
    assert math.degrees(topo.E.theta) == pytest.approx(11.119, abs=0.1)

    # The smooth DE must satisfy the CORRECTED C+ compatibility.
    res = [residual_Cplus_axisym(
        FlowNode(x=a.x, r=a.r, M=a.M, theta=a.theta),
        FlowNode(x=b.x, r=b.r, M=b.M, theta=b.theta), GAMMA)
        for a, b in zip(topo.DE[:-1], topo.DE[1:])]
    rms = float(np.sqrt((np.abs(res) ** 2).mean()))
    assert rms < 2e-3, f"smooth DE violates corrected C+ compat: {rms:.3e}"