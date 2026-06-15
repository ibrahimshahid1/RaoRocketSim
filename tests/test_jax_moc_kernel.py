"""
J3b — differentiable throat-arc kernel march (raosim.jax.moc_kernel).

Landed state (2026-06-12, second increment):

* the march reproduces the NumPy NASA oracle kernel (``build_kernel``)
  at BIT-PARITY — same node counts, max |ΔM| ≤ 5e-10 over all BD
  nodes at the reference config — by porting NASA's exact row policy:
  per row a RAW wall point (the C+ from prev[1] TERMINATING on the
  arc) unless its step exceeds ``dtheta_limit``, in which case the
  SPECIAL prescribed-angle point at θ_prev + dθ/2 is inserted and the
  row grows by one; raw rows keep width and shift the C+ pairing.
  The decision/width/pairing are traced; shapes stay static (padded).
* ``d(BD)/d(theta_B)`` is exact: the pre-clamp grid is
  θ_B-independent, so the sensitivity flows only through the final
  clamped row — jacfwd matches central FD to ~1e-8 relative.
* the NASA dθ-form unit-process ports match the NumPy originals at
  their 1e-10 fixed points (parity ≤ 1e-8 pinned below; measured
  1e-12..1e-16 on oracle row inputs).

Design history (all measured, recorded in the module docstring of
``raosim/jax/moc_kernel.py``): a static-width forward march explodes
(no step limit); a regridded inverse march is weakly unstable (axis
Mach runaway); an all-special prescribed-grid march with Anderson-form
processes is stable but converges to the WRONG continuum (wall chain
under-accumulates ν: ν_w−θ_w → −0.67° at B vs the oracle's physical
+0.68° — C+ termination on the wall is required physics, not grid
adaptivity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from raosim.moc import (  # noqa: E402
    _make_point,
    solve_axis_point,
    solve_interior_point,
    solve_wall_point,
)
from raosim.nasa_moc import (  # noqa: E402
    ArcWall,
    _calc_axial_mesh_point,
    _calc_interior_mesh_point,
    _calc_special_wall_point,
    _char_point_from_node,
    build_kernel,
)
from raosim.jax.moc_kernel import (  # noqa: E402
    KernelRow,
    axis_point,
    axis_point_nasa,
    interior_point,
    interior_point_nasa,
    march_kernel,
    special_wall_point_nasa,
    wall_point_arc,
)

GAMMA = 1.4
RT, RD = 1.0, 0.382
THETA_B = math.radians(25.5659)   # the smooth existence root angle
N_KERNEL = 24
N_ROWS = 110                      # static bound ≥ oracle's 101 rows


def _pt(x, r, th_deg, M):
    return _make_point(x, r, math.radians(th_deg), M, GAMMA)


def _d(p):
    return dict(x=jnp.float64(p.x), r=jnp.float64(p.r),
                theta=jnp.float64(p.theta), nu=jnp.float64(p.nu),
                mu=jnp.float64(p.mu), M=jnp.float64(p.M))


def _row_of(char_points):
    return KernelRow(
        x=jnp.array([p.x for p in char_points]),
        r=jnp.array([p.r for p in char_points]),
        theta=jnp.array([p.theta for p in char_points]),
        nu=jnp.array([p.nu for p in char_points]),
        mu=jnp.array([p.mu for p in char_points]),
        M=jnp.array([p.M for p in char_points]),
    )


@pytest.fixture(scope="module")
def oracle_kernel():
    return build_kernel(RT, RD, THETA_B, GAMMA, N_KERNEL, Ru=1.5,
                        starting_line_method="kliegel_levine")


@pytest.fixture(scope="module")
def oracle_rows(oracle_kernel):
    return [[_char_point_from_node(n, GAMMA) for n in row]
            for row in oracle_kernel.rrcs]


@pytest.fixture(scope="module")
def start_line(oracle_rows):
    return _row_of(oracle_rows[0])


@pytest.fixture(scope="module")
def march(start_line):
    return march_kernel(start_line, THETA_B, RT, RD, GAMMA,
                        n_rows=N_ROWS)


# ---------------------------------------------------------------------
#  Anderson-form unit-process parity (the corrected-pairing reference
#  ports; float64 round-off vs raosim.moc with the early break off).
# ---------------------------------------------------------------------


def test_interior_point_parity():
    cases = [
        (_pt(0.05, 0.95, 4.0, 1.25), _pt(0.04, 0.80, 2.0, 1.35)),
        (_pt(0.30, 0.70, 12.0, 1.9), _pt(0.28, 0.55, 8.0, 2.1)),
        (_pt(0.90, 0.30, 6.0, 3.1), _pt(0.85, 0.18, 3.0, 3.4)),
    ]
    for pm, pp in cases:
        ref = solve_interior_point(pm, pp, GAMMA, tol=0.0, max_iter=10)
        got = interior_point(_d(pm), _d(pp), GAMMA, n_corr=10)
        for f in ("x", "r", "theta", "M"):
            assert abs(float(got[f]) - getattr(ref, f)) < 1e-12, f


def test_axis_point_parity():
    for pa in (_pt(0.4, 0.06, 1.5, 2.4), _pt(1.1, 0.04, 0.4, 3.6)):
        ref = solve_axis_point(pa, GAMMA, max_iter=10)
        got = axis_point(_d(pa), GAMMA, n_corr=10)
        for f in ("x", "M"):
            assert abs(float(got[f]) - getattr(ref, f)) < 1e-12, f
        assert float(got["r"]) == 0.0
        assert float(got["theta"]) == 0.0


def test_wall_point_arc_parity():
    arc = ArcWall(RT, RD, math.radians(44.0))
    for pi in (_pt(0.02, 0.97, 3.0, 1.15), _pt(0.08, 0.93, 10.0, 1.45)):
        ref = solve_wall_point(pi, arc, GAMMA, tol=0.0, max_iter=10)
        got = wall_point_arc(_d(pi), RT, RD, GAMMA, n_corr=10)
        for f in ("x", "r", "theta", "M"):
            assert abs(float(got[f]) - getattr(ref, f)) < 1e-12, f


# ---------------------------------------------------------------------
#  NASA dθ-form unit-process parity on real oracle rows (these are the
#  processes the march uses; both sides converge a 1e-10 fixed point).
# ---------------------------------------------------------------------


def test_nasa_form_parity_on_oracle_rows(oracle_kernel, oracle_rows):
    arc = ArcWall(RT, RD, THETA_B)
    for j in (0, 30, 70):
        prev = oracle_rows[j]
        nxt = oracle_rows[j + 1]
        prow = _row_of(prev)

        alpha = prev[0].theta + math.radians(0.25)
        ref = _calc_special_wall_point(prev, arc, GAMMA, alpha)
        got = special_wall_point_nasa(prow, jnp.float64(alpha),
                                      RT, RD, GAMMA)
        assert abs(float(got["M"]) - ref.M) < 1e-8

        for i in (1, 5, len(prev) - 2):
            ref_i, neg = _calc_interior_mesh_point(
                prev, nxt[:i], i, True, GAMMA)
            assert ref_i is not None and not neg
            got_i = interior_point_nasa(
                _d(prev[i]), _d(prev[i - 1]), _d(nxt[i - 1]), GAMMA)
            for f in ("x", "r", "M"):
                assert abs(float(got_i[f]) - getattr(ref_i, f)) < 1e-8, (
                    j, i, f)

        ref_a = _calc_axial_mesh_point(nxt[:-1], GAMMA)
        got_a = axis_point_nasa(_d(nxt[-2]), GAMMA)
        assert abs(float(got_a["M"]) - ref_a.M) < 1e-8
        assert abs(float(got_a["x"]) - ref_a.x) < 1e-8


# ---------------------------------------------------------------------
#  March: bit-parity with the oracle kernel + structure checks.
# ---------------------------------------------------------------------


def test_march_reaches_wall_and_matches_oracle_node_count(
        march, oracle_kernel):
    assert bool(march.reached_wall)
    assert int(march.bd_axis_idx) + 1 == len(oracle_kernel.rrcs[-1])


def test_march_bd_bit_parity_with_oracle(march, oracle_kernel):
    """The J3b acceptance gate: the differentiable BD IS the oracle BD
    (measured max |ΔM| 4.8e-10 / |Δx| 3.7e-10 at the reference)."""
    bn = oracle_kernel.rrcs[-1]
    v = int(march.bd_axis_idx) + 1
    assert v == len(bn)
    for field, attr, tol in (("M", "M", 1e-8), ("x", "x", 1e-8),
                             ("r", "r", 1e-8), ("theta", "theta", 1e-9)):
        got = np.asarray(getattr(march.bd, field))[:v]
        ref = np.array([getattr(p, attr) for p in bn])
        assert np.max(np.abs(got - ref)) < tol, field


def test_march_finite_and_on_arc(march):
    for f in march.rows:
        assert np.isfinite(np.asarray(f)).all()
    th = np.asarray(march.wall_theta)
    assert np.all(np.diff(th) >= -1e-15)          # monotone wall trace
    assert th[-1] == pytest.approx(THETA_B, abs=1e-12)
    np.testing.assert_allclose(np.asarray(march.wall_x),
                               RD * np.sin(th), atol=1e-10)
    np.testing.assert_allclose(np.asarray(march.wall_r),
                               (RT + RD) - RD * np.cos(th), atol=1e-10)


def test_march_bd_satisfies_corrected_cminus_invariant(march):
    """d(θ+ν) = +S ds along BD (corrected pairing, midpoint rule) —
    the same residual form the pairing suite pins on oracle rows
    (oracle BD measures RMS ≈ 4e-6 at this resolution)."""
    v = int(march.bd_axis_idx) + 1
    x = np.asarray(march.bd.x)[:v]
    r = np.asarray(march.bd.r)[:v]
    th = np.asarray(march.bd.theta)[:v]
    nu = np.asarray(march.bd.nu)[:v]
    mu = np.asarray(march.bd.mu)[:v]
    ds = np.hypot(np.diff(x), np.diff(r))
    rmid = np.maximum(0.5 * (r[:-1] + r[1:]), 1e-12)
    S = (np.sin(0.5 * (th[:-1] + th[1:]))
         * np.sin(0.5 * (mu[:-1] + mu[1:])) / rmid)
    res = np.diff(th + nu) - S * ds
    assert np.sqrt(np.mean(res ** 2)) < 1e-4


# ---------------------------------------------------------------------
#  Differentiability: exact, smooth d/d(theta_B) through the clamped
#  final row (the pre-clamp grid is theta_B-independent).
# ---------------------------------------------------------------------


def test_dbd_dthetaB_matches_finite_differences(start_line):
    def f(tb):
        m = march_kernel(start_line, tb, RT, RD, GAMMA, n_rows=N_ROWS)
        i = m.bd_axis_idx
        return jnp.stack([m.bd.M[0], m.bd.x[i], m.bd.M[i],
                          m.bd.theta[0]])

    g = jax.jacfwd(f)(jnp.float64(THETA_B))
    d = 1e-6
    fd = (f(THETA_B + d) - f(THETA_B - d)) / (2 * d)
    g, fd = np.asarray(g), np.asarray(fd)
    assert np.isfinite(g).all()
    # d(theta_B)/d(theta_B) = 1 exactly through the clamped B row.
    assert g[3] == pytest.approx(1.0, abs=1e-9)
    rel = np.abs(g - fd) / np.maximum(np.abs(fd), 1e-9)
    assert rel.max() < 1e-6, (g, fd)
