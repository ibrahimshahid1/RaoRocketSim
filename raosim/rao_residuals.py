"""Residual primitives for the Rao variational/MOC coupling."""

from __future__ import annotations

import math

import numpy as np

from raosim.gas_dynamics import prandtl_meyer
from raosim.moc import FlowNode


def _source_qplus(theta: float, mu: float, r: float) -> float:
    cos_tp = math.cos(theta + mu)
    if abs(cos_tp) <= 1e-12 or r <= 1e-12:
        return 0.0
    return math.sin(theta) * math.sin(mu) * math.cos(mu) / (r * cos_tp)


def _source_qminus(theta: float, mu: float, r: float) -> float:
    cos_tm = math.cos(theta - mu)
    if abs(cos_tm) <= 1e-12 or r <= 1e-12:
        return 0.0
    return -math.sin(theta) * math.sin(mu) * math.cos(mu) / (r * cos_tm)


def residual_Cplus_axisym(
    p0: FlowNode,
    p1: FlowNode,
    gamma: float,
    *,
    axisymmetric: bool = True,
) -> float:
    """
    Axisymmetric C+ compatibility residual from ``p0`` to ``p1``.

    Along C+: d(theta + nu) = Q+ ds.  With ``axisymmetric=False`` this
    reduces to the planar invariant check d(theta + nu) = 0.
    """
    nu0 = prandtl_meyer(max(float(p0.M), 1.001), gamma)
    nu1 = prandtl_meyer(max(float(p1.M), 1.001), gamma)
    lhs = (p1.theta + nu1) - (p0.theta + nu0)
    if not axisymmetric:
        return lhs

    ds = math.hypot(p1.x - p0.x, p1.r - p0.r)
    theta = 0.5 * (p0.theta + p1.theta)
    mu = 0.5 * (p0.mu + p1.mu)
    r = max(0.5 * (p0.r + p1.r), 1e-12)
    return lhs - _source_qplus(theta, mu, r) * ds


def residual_Cminus_axisym(
    p0: FlowNode,
    p1: FlowNode,
    gamma: float,
    *,
    axisymmetric: bool = True,
) -> float:
    """
    Axisymmetric C- compatibility residual from ``p0`` to ``p1``.

    Along C-: d(theta - nu) = Q- ds.  With ``axisymmetric=False`` this
    reduces to the planar invariant check d(theta - nu) = 0.
    """
    nu0 = prandtl_meyer(max(float(p0.M), 1.001), gamma)
    nu1 = prandtl_meyer(max(float(p1.M), 1.001), gamma)
    lhs = (p1.theta - nu1) - (p0.theta - nu0)
    if not axisymmetric:
        return lhs

    ds = math.hypot(p1.x - p0.x, p1.r - p0.r)
    theta = 0.5 * (p0.theta + p1.theta)
    mu = 0.5 * (p0.mu + p1.mu)
    r = max(0.5 * (p0.r + p1.r), 1e-12)
    return lhs - _source_qminus(theta, mu, r) * ds


def residual_left_mach_geometry(p0: FlowNode, p1: FlowNode) -> float:
    """Geometric residual for a left-running Mach line segment."""
    dx = p1.x - p0.x
    dr = p1.r - p0.r
    theta = 0.5 * (p0.theta + p1.theta)
    mu = 0.5 * (p0.mu + p1.mu)
    return dr - dx * math.tan(theta + mu)


def residual_wall_tangency(w0: FlowNode, w1: FlowNode) -> float:
    """Wall segment tangency residual using the average local flow angle."""
    dx = w1.x - w0.x
    dr = w1.r - w0.r
    theta = 0.5 * (w0.theta + w1.theta)
    return dr - dx * math.tan(theta)


def residual_cplus_child_position(
    parent: FlowNode,
    child: FlowNode,
) -> float:
    """
    Geometric residual: does ``child`` lie on the C+ line through ``parent``?

    Lightweight alternative to :func:`residual_intersection` for the
    Phase 6 coupled wall — wall points are streamlines fed by a single C+
    from the CE, so we only need the one geometric equation
    ``dr = tan(theta_avg + mu_avg) * dx`` rather than the two-equation
    C+/C- intersection check.
    """
    dx = child.x - parent.x
    dr = child.r - parent.r
    theta_avg = 0.5 * (parent.theta + child.theta)
    mu_avg = 0.5 * (parent.mu + child.mu)
    return dr - math.tan(theta_avg + mu_avg) * dx


def residual_intersection(
    p_plus: FlowNode,
    p_minus: FlowNode,
    child: FlowNode,
    x_scale: float,
    r_scale: float,
) -> np.ndarray:
    """
    Geometric intersection residual for a child C+/C- point.

    ``p_plus`` carries the C+ line, ``p_minus`` carries the C- line.  The
    returned two-vector is scaled in radius units so it can be stacked in
    least-squares residuals.
    """
    _ = x_scale
    scale = max(abs(r_scale), 1e-12)
    slope_p = math.tan(0.5 * (p_plus.theta + child.theta)
                       + 0.5 * (p_plus.mu + child.mu))
    slope_m = math.tan(0.5 * (p_minus.theta + child.theta)
                       - 0.5 * (p_minus.mu + child.mu))
    cplus = (child.r - p_plus.r) - slope_p * (child.x - p_plus.x)
    cminus = (child.r - p_minus.r) - slope_m * (child.x - p_minus.x)
    return np.array([cplus / scale, cminus / scale], dtype=float)
