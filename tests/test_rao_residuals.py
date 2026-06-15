import math

import pytest

from raosim.gas_dynamics import prandtl_meyer
from raosim.moc import FlowNode, _make_point, solve_interior_point
from raosim.rao_residuals import (
    residual_Cminus_axisym,
    residual_Cplus_axisym,
    residual_left_mach_geometry,
    residual_wall_tangency,
)


def test_residual_Cplus_planar_invariant():
    """CORRECTED 2026-06-11: K+ = θ − ν is the planar C+ invariant
    (Anderson MCF §11.4; oracle proof in test_characteristic_pairing).
    Accelerating along C+ (ν up) turns the flow UP by the same amount.
    (This test previously pinned the mirrored θ+ν convention.)"""
    gamma = 1.4
    m0 = 2.0
    m1 = 2.5
    theta0 = 0.10
    theta1 = theta0 - prandtl_meyer(m0, gamma) + prandtl_meyer(m1, gamma)

    p0 = FlowNode(x=0.0, r=1.0, M=m0, theta=theta0)
    p1 = FlowNode(x=1.0, r=1.4, M=m1, theta=theta1)

    assert residual_Cplus_axisym(p0, p1, gamma, axisymmetric=False) == pytest.approx(0.0, abs=1e-10)
    # ... and the wrong-family relation must NOT vanish on this segment.
    assert abs(residual_Cminus_axisym(p0, p1, gamma, axisymmetric=False)) > 1e-3


def test_residual_Cminus_planar_invariant():
    """CORRECTED 2026-06-11: K− = θ + ν is the planar C− invariant."""
    gamma = 1.4
    m0 = 2.0
    m1 = 2.5
    theta0 = 0.10
    theta1 = theta0 + prandtl_meyer(m0, gamma) - prandtl_meyer(m1, gamma)

    p0 = FlowNode(x=0.0, r=1.0, M=m0, theta=theta0)
    p1 = FlowNode(x=1.0, r=0.6, M=m1, theta=theta1)

    assert residual_Cminus_axisym(p0, p1, gamma, axisymmetric=False) == pytest.approx(0.0, abs=1e-10)
    assert abs(residual_Cplus_axisym(p0, p1, gamma, axisymmetric=False)) > 1e-3


def test_residual_axisym_matches_interior_march():
    gamma = 1.4
    p_plus = _make_point(x=0.0, r=0.020, theta=math.radians(4.0), M=1.45, gamma=gamma)
    p_minus = _make_point(x=0.002, r=0.031, theta=math.radians(8.0), M=1.80, gamma=gamma)

    child = solve_interior_point(p_minus, p_plus, gamma, axisymmetric=True, tol=1e-10, max_iter=30)

    assert abs(residual_Cplus_axisym(p_plus.to_flow_node(), child.to_flow_node(), gamma)) < 1e-6
    assert abs(residual_Cminus_axisym(p_minus.to_flow_node(), child.to_flow_node(), gamma)) < 1e-6


def test_left_mach_geometry_zero_on_matching_segment():
    p0 = FlowNode(x=0.0, r=0.02, M=2.0, theta=math.radians(6.0))
    p1 = FlowNode(
        x=0.01,
        r=0.02 + 0.01 * math.tan(p0.theta + p0.mu),
        M=2.0,
        theta=math.radians(6.0),
    )

    assert residual_left_mach_geometry(p0, p1) == pytest.approx(0.0, abs=1e-12)


def test_wall_tangency_zero_on_matching_segment():
    theta = math.radians(5.0)
    w0 = FlowNode(x=0.0, r=0.02, M=2.0, theta=theta)
    w1 = FlowNode(x=0.02, r=0.02 + 0.02 * math.tan(theta), M=2.1, theta=theta)

    assert residual_wall_tangency(w0, w1) == pytest.approx(0.0, abs=1e-12)
