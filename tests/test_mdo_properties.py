"""Phase-2 machinery gates: the C¹ monotone interpolant reproduces values and
gradients, preserves shape, matches SciPy's PCHIP, and flags its domain."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from raosim.mdo.properties import (  # noqa: E402
    ChamberSurfaces,
    Pchip1D,
    PropertySurface2D,
    constant_chamber_surfaces,
    fritsch_carlson_slopes,
)


# --------------------------------------------------------------------------- #
# 1-D
# --------------------------------------------------------------------------- #
def test_fc_slopes_match_scipy_pchip():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    x = np.array([0.0, 0.7, 1.1, 2.4, 3.0, 4.5])
    y = np.array([1.0, 1.4, 1.5, 2.9, 3.0, 3.05])
    d = fritsch_carlson_slopes(x, y)
    ref = scipy_interp.PchipInterpolator(x, y).derivative()(x)
    np.testing.assert_allclose(d, ref, rtol=1e-12, atol=1e-12)


def test_pchip1d_values_match_scipy_between_nodes():
    scipy_interp = pytest.importorskip("scipy.interpolate")
    x = np.linspace(0.0, 2.0, 9)
    y = np.tanh(x) + 0.1 * x
    p = Pchip1D.build(x, y)
    ref = scipy_interp.PchipInterpolator(x, y)
    t = np.linspace(0.0, 2.0, 101)
    np.testing.assert_allclose(np.array([float(p(v)) for v in t]), ref(t),
                               rtol=1e-12, atol=1e-12)


def test_pchip1d_preserves_monotonicity():
    x = np.linspace(0.0, 1.0, 7)
    y = np.array([0.0, 0.02, 0.5, 0.51, 0.9, 0.95, 1.0])  # monotone, kinky
    p = Pchip1D.build(x, y)
    t = jnp.linspace(0.0, 1.0, 400)
    vals = jax.vmap(p)(t)
    assert np.all(np.diff(np.asarray(vals)) >= -1e-12)


# --------------------------------------------------------------------------- #
# 2-D surface
# --------------------------------------------------------------------------- #
def _analytic(x, y):
    # smooth, monotone in both directions on the box
    return 1.2 + 0.15 * np.log(x / 1e6 + 1.0) + 0.08 * np.tanh(y - 2.0)


@pytest.fixture(scope="module")
def surface():
    xg = np.linspace(1.0e6, 6.0e6, 11)
    yg = np.linspace(1.5, 3.5, 9)
    Z = _analytic(xg[:, None], yg[None, :])
    return PropertySurface2D.build(xg, yg, Z, name="test")


def test_surface_exact_at_nodes(surface):
    xg = np.asarray(surface.xg)
    yg = np.asarray(surface.yg)
    for i in (0, 3, 10):
        for j in (0, 4, 8):
            assert float(surface(xg[i], yg[j])) == pytest.approx(
                _analytic(xg[i], yg[j]), rel=1e-13)


def test_surface_accuracy_between_nodes(surface):
    rng = np.random.default_rng(7)
    xs = rng.uniform(1.1e6, 5.9e6, 40)
    ys = rng.uniform(1.55, 3.45, 40)
    vals = np.array([float(surface(x, y)) for x, y in zip(xs, ys)])
    ref = _analytic(xs, ys)
    assert np.max(np.abs(vals - ref) / np.abs(ref)) < 2e-4


def test_surface_gradients_match_central_difference(surface):
    g = jax.grad(lambda x, y: surface(x, y), argnums=(0, 1))
    x0, y0 = 3.3e6, 2.7
    gx, gy = (float(v) for v in g(jnp.asarray(x0), jnp.asarray(y0)))
    hx, hy = 1e3, 1e-4
    fdx = (float(surface(x0 + hx, y0)) - float(surface(x0 - hx, y0))) / (2 * hx)
    fdy = (float(surface(x0, y0 + hy)) - float(surface(x0, y0 - hy))) / (2 * hy)
    assert gx == pytest.approx(fdx, rel=1e-6)
    assert gy == pytest.approx(fdy, rel=1e-6)


def test_surface_gradient_continuous_across_cell_edge(surface):
    # C¹ check: derivative from the left and right of an interior grid line
    # agree (small FD straddling the node).
    x_edge = float(np.asarray(surface.xg)[5])
    y0 = 2.4
    h = 0.5  # Pa — tiny straddle across the edge
    g = jax.grad(lambda x: surface(x, jnp.asarray(y0)))
    gl = float(g(jnp.asarray(x_edge - h)))
    gr = float(g(jnp.asarray(x_edge + h)))
    assert gl == pytest.approx(gr, rel=1e-6, abs=1e-16)


def test_surface_monotone_along_grid_lines(surface):
    xs = jnp.linspace(surface.xg[0], surface.xg[-1], 200)
    vals = jax.vmap(lambda x: surface(x, jnp.asarray(2.0)))(xs)
    assert np.all(np.diff(np.asarray(vals)) >= -1e-12)


def test_domain_violation_signs(surface):
    inside = np.asarray(surface.domain_violation(3.0e6, 2.0))
    assert np.all(inside <= 0.0)
    outside = np.asarray(surface.domain_violation(9.0e6, 0.5))
    assert outside[1] > 0.0 and outside[2] > 0.0


# --------------------------------------------------------------------------- #
# Chamber bundle
# --------------------------------------------------------------------------- #
def test_constant_surfaces_reproduce_constants_and_cstar():
    from raosim.gas_dynamics import characteristic_velocity

    cs = constant_chamber_surfaces(gamma=1.24, Tc=3550.0, R_gas=346.0)
    assert float(cs.gamma(3.0e6, 2.3)) == pytest.approx(1.24, rel=1e-12)
    assert float(cs.Tc(2.0e6, 3.0)) == pytest.approx(3550.0, rel=1e-12)
    cstar = float(cs.c_star_ideal(3.0e6, 2.3))
    assert cstar == pytest.approx(
        characteristic_velocity(1.24, 346.0, 3550.0), rel=1e-12)
    assert isinstance(cs, ChamberSurfaces)
