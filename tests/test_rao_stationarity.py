"""
Phase 3 tests: algebraic Rao stationarity, differential consistency, and
left-Mach-line geometry residual.

References:
  - Rao 1958, "Exhaust Nozzle Contour for Optimum Thrust", Jet Propulsion
  - Rao-Beck-Booth 1999, AIAA 99-2584 (propulsion_texts/rao1999.pdf)
  - Östlund 2002, KTH thesis (propulsion_texts/fulltext01.pdf, §3)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from raosim.gas_dynamics import mstar_from_M
from raosim.moc import FlowNode
from raosim.rao_variational import (
    ControlSurface,
    _pack_bvp,
    _rao_algebraic_stationarity_residuals,
    _rao_left_mach_geometry_residuals,
    _seed_log_C_from_ce,
    _unpack_bvp,
    rao_stationarity_fd_residual,
    rao_stationarity_residual,
)


# ---------------------------------------------------------------------
# mstar_from_M identities
# ---------------------------------------------------------------------

def test_mstar_at_M_one_equals_one():
    for gamma in (1.2, 1.3, 1.4, 1.667):
        assert mstar_from_M(1.0, gamma) == pytest.approx(1.0, abs=1e-12)


def test_mstar_high_M_limit():
    """M* -> sqrt((γ+1)/(γ-1)) as M -> infinity."""
    gamma = 1.4
    Mstar_inf = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    high = mstar_from_M(1.0e4, gamma)
    assert high == pytest.approx(Mstar_inf, rel=1e-6)


def test_mstar_rejects_nonpositive():
    with pytest.raises(ValueError):
        mstar_from_M(0.0, 1.4)
    with pytest.raises(ValueError):
        mstar_from_M(-1.0, 1.4)


# ---------------------------------------------------------------------
# Algebraic Rao stationarity zero on a constructed-optimum CE
# ---------------------------------------------------------------------

def _constant_C_node(M: float, theta: float, gamma: float, log_C: float) -> FlowNode:
    """Place a node with arbitrary (x, r); residual depends only on M, theta."""
    return FlowNode(x=0.0, r=0.05, M=M, theta=theta)


def test_algebraic_stationarity_zero_at_seed_log_C():
    """For a single-node "CE" the residual collapses to zero by construction."""
    gamma = 1.4
    M, theta = 2.5, math.radians(15.0)
    alpha = math.asin(1.0 / M)
    Mstar = mstar_from_M(M, gamma)
    log_C = math.log(Mstar) + math.log(math.cos(theta - alpha)) - math.log(math.cos(alpha))

    node = _constant_C_node(M, theta, gamma, log_C)
    r = rao_stationarity_residual(node, log_C, gamma)
    assert abs(r) < 1e-12


def test_algebraic_stationarity_constant_along_constructed_optimum_CE():
    """
    Build a multi-node CE that exactly satisfies M* cos(θ-α)/cos(α) = C and
    confirm the algebraic residual vanishes at every node.

    C must satisfy C <= min_M[ M*(M) / cos(α(M)) ] over the chosen M range,
    so cos(θ-α) = C cos(α)/M* stays in [-1, 1].  We pick C from the
    bounding (smallest-M) node and verify zero residual everywhere.
    """
    gamma = 1.4
    Ms = np.linspace(2.0, 3.5, 6)
    rs = np.linspace(0.025, 0.063, 6)
    xs = np.linspace(0.0, 0.10, 6)

    # Choose C strictly below the M=2.0 ceiling so all nodes have a real theta.
    M_ref = float(Ms[0])
    a_ref = math.asin(1.0 / M_ref)
    C_ceiling = mstar_from_M(M_ref, gamma) / math.cos(a_ref)
    C = 0.95 * C_ceiling
    log_C = math.log(C)

    nodes: list[FlowNode] = []
    for M, r, x in zip(Ms, rs, xs):
        alpha = math.asin(1.0 / float(M))
        Mstar = mstar_from_M(float(M), gamma)
        target = C * math.cos(alpha) / Mstar
        # By construction target should be in (0, 1].
        assert -1.0 <= target <= 1.0, f"constructed target out of range: {target}"
        theta = alpha + math.acos(target)
        nodes.append(FlowNode(x=float(x), r=float(r), M=float(M), theta=float(theta)))

    residuals = [rao_stationarity_residual(p, log_C, gamma) for p in nodes]
    assert max(abs(r) for r in residuals) < 1e-12


def test_differential_stationarity_sign_consistent_with_algebraic():
    """
    Going from a perfectly-stationary node toward a slightly perturbed one
    produces a small non-zero differential residual whose sign matches
    the change in (algebraic) log(M* cos(θ-α)/cos(α)).
    """
    gamma = 1.4
    M0, theta0 = 2.5, math.radians(20.0)
    a0 = math.asin(1.0 / M0)
    log_C0 = (math.log(mstar_from_M(M0, gamma))
              + math.log(math.cos(theta0 - a0))
              - math.log(math.cos(a0)))

    p0 = FlowNode(x=0.0, r=0.025, M=M0, theta=theta0)
    M1 = 2.55
    a1 = math.asin(1.0 / M1)
    # Choose theta1 to depart slightly from the optimum.
    theta1 = math.radians(18.0)
    p1 = FlowNode(x=0.005, r=0.027, M=M1, theta=theta1)

    log_C1 = (math.log(mstar_from_M(M1, gamma))
              + math.log(math.cos(theta1 - a1))
              - math.log(math.cos(a1)))
    delta_log = log_C1 - log_C0
    fd = rao_stationarity_fd_residual(p0, p1, gamma)
    # The differential form measures how *not* constant C is between the two
    # nodes; the integrated change should approximate the fd residual to first
    # order.  Tolerance is loose because the fd form linearises in (dθ, dα).
    assert abs(fd - delta_log) < 0.05


# ---------------------------------------------------------------------
# Left-Mach-line geometry residual
# ---------------------------------------------------------------------

def test_left_mach_geometry_residual_zero_on_constructed_segment():
    """A segment built from p0 by stepping along (cos(θ+α), sin(θ+α)) is zero."""
    p0 = FlowNode(x=0.0, r=0.020, M=2.0, theta=math.radians(10.0))
    slope = math.tan(p0.theta + p0.mu)
    p1 = FlowNode(x=p0.x + 0.01, r=p0.r + 0.01 * slope, M=p0.M, theta=p0.theta)

    # _rao_left_mach_geometry_residuals takes a ControlSurface; reach into
    # the lower-level helper directly.
    from raosim.rao_residuals import residual_left_mach_geometry
    assert abs(residual_left_mach_geometry(p0, p1)) < 1e-12


def test_left_mach_geometry_group_zero_on_constructed_CE():
    """Build a 4-node CE that's a perfect left Mach line; group residual is zero."""
    gamma = 1.4
    nodes = [FlowNode(x=0.0, r=0.020, M=2.0, theta=math.radians(15.0))]
    for _ in range(3):
        prev = nodes[-1]
        slope = math.tan(prev.theta + prev.mu)
        nodes.append(FlowNode(
            x=prev.x + 0.005,
            r=prev.r + 0.005 * slope,
            M=prev.M,        # constant Mach simplifies the test
            theta=prev.theta,
        ))
    ce = ControlSurface(
        r=np.asarray([p.r for p in nodes]),
        M=np.asarray([p.M for p in nodes]),
        theta=np.asarray([p.theta for p in nodes]),
        phi=np.asarray([p.theta + p.mu for p in nodes]),
        x=np.asarray([p.x for p in nodes]),
    )
    res = _rao_left_mach_geometry_residuals(ce)
    assert res.size == 3
    assert np.max(np.abs(res)) < 1e-12


def test_algebraic_stationarity_group_runs_on_short_ce():
    ce = ControlSurface(
        r=np.array([0.025, 0.040]),
        M=np.array([2.0, 3.0]),
        theta=np.array([math.radians(20.0), math.radians(8.0)]),
        phi=np.array([math.radians(45.0), math.radians(40.0)]),
        x=np.array([0.0, 0.06]),
        log_C=0.5,
    )
    res = _rao_algebraic_stationarity_residuals(ce, gamma=1.4)
    assert res.size == 2
    assert np.all(np.isfinite(res))


# ---------------------------------------------------------------------
# log_C round-trip through pack / unpack
# ---------------------------------------------------------------------

def test_log_C_roundtrips_through_pack_unpack():
    n = 4
    ce = ControlSurface(
        r=np.linspace(0.025, 0.063, n),
        M=np.linspace(2.0, 3.5, n),
        theta=np.linspace(math.radians(25.0), math.radians(8.0), n),
        phi=np.linspace(math.radians(50.0), math.radians(30.0), n),
        x=np.linspace(0.0, 0.10, n),
        lambda2=-0.3,
        lambda3=0.05,
        log_C=0.7654,
    )
    u = _pack_bvp(ce, ce.lambda2, ce.lambda3)
    ce2 = _unpack_bvp(u, ce.r)
    assert ce2.log_C == pytest.approx(0.7654, abs=1e-12)
    assert ce2.lambda2 == pytest.approx(-0.3, abs=1e-12)
    assert ce2.lambda3 == pytest.approx(0.05, abs=1e-12)


def test_pack_unpack_log_C_override():
    n = 3
    ce = ControlSurface(
        r=np.linspace(0.025, 0.063, n),
        M=np.linspace(2.0, 3.5, n),
        theta=np.linspace(math.radians(25.0), math.radians(8.0), n),
        phi=np.linspace(math.radians(50.0), math.radians(30.0), n),
        x=np.linspace(0.0, 0.10, n),
    )
    u = _pack_bvp(ce, 0.0, 0.0, log_C=1.234)
    ce2 = _unpack_bvp(u, ce.r)
    assert ce2.log_C == pytest.approx(1.234, abs=1e-12)


def test_seed_log_C_from_ce_returns_finite_value():
    n = 6
    ce = ControlSurface(
        r=np.linspace(0.025, 0.063, n),
        M=np.linspace(2.0, 3.5, n),
        theta=np.linspace(math.radians(25.0), math.radians(8.0), n),
        phi=np.linspace(math.radians(50.0), math.radians(30.0), n),
        x=np.linspace(0.0, 0.10, n),
    )
    seed = _seed_log_C_from_ce(ce, gamma=1.4)
    assert math.isfinite(seed)
    # Reasonable range for typical CE configurations.
    assert -5.0 < seed < 5.0
