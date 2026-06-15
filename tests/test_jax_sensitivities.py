"""J6 v1 gates (plan §7: "smoke + a known-sign sensitivity").

Solve-free tests (fast): Cf parity numpy<->jax on a real fixed-end DE,
the analytic dCf/dpa known-sign/value identity, and an FD cross-check
of the node gradient at a non-converged state.  The end-to-end
``rao_sensitivities`` smoke runs a full JAX solve and is @slow
(host-budget; the sandbox per-call cap cannot fit it).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optimistix")

import jax.numpy as jnp  # noqa: E402

from raosim.jax.sensitivities import cf_de_jax, cf_from_u  # noqa: E402
from raosim.moc_topology import build_reference_topology  # noqa: E402
from raosim.nasa_moc import surface_thrust_coefficient  # noqa: E402
from raosim.moc import FlowNode  # noqa: E402

GAMMA = 1.4
RT = 0.020


@pytest.fixture(scope="module")
def de_nodes():
    """DE of the reference fixed-end topology (real, no BVP solve)."""
    topo = build_reference_topology(RT, 10.0, 80.0, GAMMA, 0.01,
                                    n_kernel=24, n_de_points=24)
    return topo.DE


def test_cf_jax_matches_numpy_surface_thrust_coefficient(de_nodes):
    """Line-for-line port parity in x64 (~1e-12)."""
    pa = 0.01
    flow = [FlowNode(x=p.x, r=p.r, M=p.M, theta=p.theta) for p in de_nodes]
    cf_np = surface_thrust_coefficient(flow, GAMMA, RT, pa)

    x = jnp.asarray([p.x for p in de_nodes])
    r = jnp.asarray([p.r for p in de_nodes])
    M = jnp.asarray([p.M for p in de_nodes])
    th = jnp.asarray([p.theta for p in de_nodes])
    cf_j = float(cf_de_jax(x, r, M, th, GAMMA, RT, pa))

    assert cf_j == pytest.approx(cf_np, rel=1e-10), (
        f"jax {cf_j!r} vs numpy {cf_np!r}"
    )
    # And the value is a physically sane control-surface Cf.
    assert 0.5 < cf_j < 2.5


def test_dcf_dpa_known_sign_and_analytic_value(de_nodes):
    """J6 known-sign gate: dCf/dpa = -(r_E^2 - r_D^2)/Rt^2 exactly.

    The trapezoidal pressure term sum(2*pi*rbar*(p - pa)*dr)/(pi*Rt^2)
    telescopes in its pa coefficient: sum((r0+r1)(r1-r0)) = r_E^2-r_D^2.
    Raising ambient pressure must lower thrust — the sign is the
    physical gate, the value pins the implementation.
    """
    x = jnp.asarray([p.x for p in de_nodes])
    r = jnp.asarray([p.r for p in de_nodes])
    M = jnp.asarray([p.M for p in de_nodes])
    th = jnp.asarray([p.theta for p in de_nodes])

    g = float(jax.grad(
        lambda pa: cf_de_jax(x, r, M, th, GAMMA, RT, pa))(0.01))
    rD = float(r[0])
    rE = float(r[-1])
    expected = -(rE * rE - rD * rD) / (RT * RT)
    assert g < 0.0
    assert g == pytest.approx(expected, rel=1e-10)


def test_dcf_du_matches_finite_differences():
    """Reverse-mode node gradient vs central differences (no solve).

    Uses a synthetic-but-plausible packed state in the legacy layout
    [M(n), theta(n), r(n), l2, l3, logC, kdf] with a tiny straight BD
    so ``bd_point_at_fraction`` has a real polyline to differentiate
    through.
    """
    import raosim.rao_variational as rv
    from raosim.jax import assembly

    n = 10
    cfg = rv.RaoSolverConfig(
        Rt=RT, epsilon=10.0, gamma=GAMMA, pa_over_p0=0.01,
        length_pct=80.0, n_control=n, n_kernel=8,
        max_nfev=0, evaluate_moc=False, couple_wall=False,
        kernel_bd=tuple(
            FlowNode(x=0.001 + 0.004 * t, r=0.020 - 0.018 * t,
                     M=1.2 + 2.0 * t, theta=0.35 * (1.0 - t))
            for t in np.linspace(0.0, 1.0, 12)
        ),
    )
    sp = assembly.params_from_config(cfg)

    rng = np.random.default_rng(42)
    M = np.linspace(2.6, 3.6, n) + 0.01 * rng.standard_normal(n)
    th = np.linspace(0.30, 0.12, n)
    r = np.linspace(0.012, 0.063, n)
    u = np.concatenate([M, th, r, [0.1, -0.2, 0.05, 0.4]])
    uj = jnp.asarray(u)

    f = lambda uu: cf_from_u(uu, sp, n, RT, GAMMA, 0.01)  # noqa: E731
    g = np.asarray(jax.grad(f)(uj), dtype=float)

    # Central differences on a deterministic component sample.
    idx = [0, n - 1, n + 2, 2 * n + 1, 3 * n - 1, 3 * n + 3]
    h = 1e-7
    for i in idx:
        up = u.copy()
        um = u.copy()
        up[i] += h
        um[i] -= h
        fd = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(um)))) / (2 * h)
        assert g[i] == pytest.approx(fd, rel=5e-5, abs=1e-8), (
            f"component {i}: ad={g[i]:.6e} fd={fd:.6e}"
        )


def test_plot_sensitivity_field_smoke():
    import matplotlib
    matplotlib.use("Agg")
    from raosim.plotting import plot_sensitivity_field

    x = np.linspace(0.01, 0.12, 20)
    r = np.linspace(0.01, 0.063, 20)
    v = np.sin(np.linspace(0, 3, 20))
    fig = plot_sensitivity_field(x, r, v)
    assert fig is not None
    fig2 = plot_sensitivity_field(x, r, v, signed=True,
                                  wall=np.column_stack([x, r + 0.002]))
    assert fig2 is not None


@pytest.mark.slow
def test_rao_sensitivities_end_to_end_smoke():
    """Full J6 v1 smoke on the J4-gate configuration (host budget).

    Gates: fields finite and correctly shaped; the repacked u* must
    reproduce the solve's residual scale; multipliers' Cf-gradients
    exist; conditioning is finite; the explicit pa partial keeps its
    sign at the converged surface.
    """
    import raosim.rao_variational as rv
    from raosim.jax.api import rao_sensitivities

    rv.PHYSICS_WEIGHT = 1.0
    cfg = rv.RaoSolverConfig(
        Rt=RT, epsilon=10.0, gamma=GAMMA, pa_over_p0=0.01,
        length_pct=80.0, n_control=24, n_kernel=24, n_wall=12,
        max_nfev=4000, residual_tol=2e-3, evaluate_moc=False,
        couple_wall=False, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=21.87,
    )
    sens = rao_sensitivities(cfg)
    n = sens.diagnostics["n_control"]
    assert n == 24
    assert sens.max_scaled <= 5e-3  # repacked u* reproduces the solve
    assert np.all(np.isfinite(sens.dCf_du))
    assert sens.dCf_dM.shape == (n,)
    assert sens.dCf_dr.shape == (n,)
    assert np.any(np.abs(sens.dCf_dr) > 0.0)
    assert sens.dCf_dpa_explicit < 0.0
    assert math.isfinite(sens.condition_number)
    assert sens.jacobian.shape[1] == 3 * n + 4
