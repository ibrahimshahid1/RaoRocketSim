from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from lrekit import rao_variational as rv
from tests.experiments.nozzle.audit_moc_reference_profile import profile_config


@pytest.fixture(scope="module")
def seed():
    config = replace(profile_config(), n_control=8, n_kernel=8, exact_exit_attachment=True)
    ce, bd, topology, kernel = rv._initial_ce_from_kernel(config)
    config = replace(config, kernel_bd=tuple(bd))
    full = rv._pack_bvp(ce, -0.5, 0.01)
    return config, ce, topology, kernel, full


def test_exact_exit_restores_exact_endpoints_and_preserves_public_layout(seed):
    config, ce, _, _, template = seed
    free = rv._exact_d_free_indices(config, template.size)
    full = rv._expand_exact_d_unknowns(template[free], template, config)
    domain = rv._require_exact_exit_domain(full, config)
    control, _ = rv._unpack_bvp(full, ce.r, kernel_bd=config.kernel_bd, gamma=config.gamma)
    assert domain["valid"]
    assert template.size - free.size == 7
    assert full.size == template.size
    assert control.r[-1] == math.sqrt(config.epsilon) * config.Rt
    assert abs(control.x[-1] - domain["target_x_m"]) <= domain["roundoff_guard_m"]
    assert config.residual_tol == 0.005


@pytest.mark.parametrize("overrides", [
    {"exact_d_attachment": False}, {"wall_method": "legacy"},
    {"couple_wall": True}, {"residual_blocks": ("mass", "length")},
])
def test_exact_exit_refuses_unsupported_models(overrides):
    with pytest.raises(ValueError, match="exact_exit_attachment"):
        replace(profile_config(), exact_exit_attachment=True, **overrides)


def test_exact_exit_reduction_refuses_disabled_domain_guidance(seed):
    config, _, _, _, template = seed
    config.residual_blocks = ("mass", "length")
    with pytest.raises(ValueError, match="ce_geometry"):
        rv._exact_d_free_indices(config, template.size)
    config.residual_blocks = None


@pytest.mark.parametrize("kind", ["subsonic", "reversed_prefix", "last_dr", "last_dx", "theta_bound", "nonfinite"])
def test_final_boundary_domain_explicitly_refuses_invalid_states(seed, kind):
    config, _, _, _, template = seed
    full = rv._expand_exact_d_unknowns(template[rv._exact_d_free_indices(config, template.size)], template, config)
    n = config.n_control
    if kind == "subsonic":
        full[n - 1] = 0.99
    elif kind == "reversed_prefix":
        full[2 * n + 2] = full[2 * n + 1] - 0.01
    elif kind == "last_dr":
        full[3 * n - 2] = math.sqrt(config.epsilon) * config.Rt + 0.1
    elif kind == "last_dx":
        full[n:2 * n - 1] = -0.1
    elif kind == "theta_bound":
        full[2 * n - 1] = math.radians(66.0)
    else:
        full[n - 1] = float("nan")
    with pytest.raises(rv.RaoBoundaryDomainError) as raised:
        rv._require_exact_exit_domain(full, config)
    assert not raised.value.diagnostics["valid"]
    if kind == "subsonic":
        assert "supersonic_state" in raised.value.diagnostics["failed_checks"]


def test_exact_exit_host_jax_values_and_full_reduced_jacobian_match(seed):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lrekit.numerics.jax import assembly

    config, ce, _, _, template = seed
    indices = rv._exact_d_free_indices(config, template.size)
    reduced = template[indices]
    sp = assembly.params_from_config(config)
    residual = assembly.make_residual(sp)
    expand = lambda vector: assembly.expand_exact_d_unknowns(vector, template, sp, jnp.asarray(indices))
    host_expand = lambda vector: rv._expand_exact_d_unknowns(vector, template, config)
    host = lambda vector: rv._scaled_rao_bvp_residual(host_expand(vector), ce.r, config)
    differentiable = lambda vector: residual(expand(vector))
    np.testing.assert_allclose(expand(reduced), host_expand(reduced), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(differentiable(reduced), host(reduced), atol=1e-8, rtol=1e-8)
    assert assembly.block_slices(sp)["penalties"].stop == len(host(reduced))
    step = 1e-6
    for function, host_function in ((expand, host_expand), (differentiable, host)):
        jacobian = np.asarray(jax.jacfwd(function)(jnp.asarray(reduced)))
        differences = np.column_stack([
            (host_function(reduced + step * axis) - host_function(reduced - step * axis)) / (2 * step)
            for axis in np.eye(len(reduced))
        ])
        assert np.all(np.isfinite(jacobian))
        np.testing.assert_allclose(jacobian, differences, rtol=2e-4, atol=2e-5)
    transform = np.asarray(jax.jacfwd(expand)(jnp.asarray(reduced)))
    for full_index in (3 * config.n_control + 3, config.n_control - 1, 2 * config.n_control + 2):
        column = int(np.flatnonzero(indices == full_index)[0])
        assert abs(transform[2 * config.n_control - 1, column]) > 1e-8


def test_degenerate_trial_extension_has_finite_jax_value_and_derivative(seed):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from lrekit.numerics.jax import assembly

    config, _, _, _, _ = seed
    sp = assembly.params_from_config(config)
    r = jnp.asarray([sp.Re - 0.5, sp.Re, sp.Re])
    theta = jnp.asarray([0.1, 0.1, 0.1])
    mach = jnp.asarray([2.0, 2.1, 2.2])
    prefix = jnp.asarray([sp.L_target - 0.5, sp.L_target])
    function = lambda end: jnp.concatenate((
        jnp.atleast_1d(assembly._exact_exit_values(r.at[-2].set(end[1]), theta, mach, prefix.at[-1].set(end[0]), sp)[0]),
        assembly._exact_exit_values(r.at[-2].set(end[1]), theta, mach, prefix.at[-1].set(end[0]), sp)[1],
    ))
    end = jnp.asarray([sp.L_target, sp.Re])
    assert np.all(np.isfinite(function(end)))
    assert np.all(np.isfinite(jax.jacfwd(function)(end)))
    _, domain, _ = rv._exact_exit_boundary_data(
        np.asarray(r), np.asarray(theta), np.asarray(mach), np.asarray(prefix),
        L=sp.L_target, Re=sp.Re, Rt=sp.Rt,
    )
    assert not domain["valid"]


def test_numpy_optimizer_success_cannot_promote_invalid_final_parameterization(seed, monkeypatch):
    config, ce, topology, kernel, _ = seed
    monkeypatch.setattr(rv, "_initial_ce_from_kernel", lambda _: (ce, config.kernel_bd, topology, kernel))

    def invalid_result(function, initial, **kwargs):
        free = rv._exact_d_free_indices(config, 3 * config.n_control + 4)
        invalid = initial.copy()
        invalid[np.flatnonzero(free == 3 * config.n_control - 2)[0]] = math.sqrt(config.epsilon) * config.Rt + 0.01
        assert np.all(np.isfinite(function(invalid)))
        return SimpleNamespace(x=invalid, success=True, cost=0.0, message="success", nfev=1)

    monkeypatch.setattr(rv, "least_squares", invalid_result)
    with pytest.raises(rv.RaoBoundaryDomainError):
        rv.solve_rao_bvp(config)


def test_standalone_jax_optimizer_success_cannot_bypass_final_domain(seed, monkeypatch):
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from lrekit.numerics.jax import api

    config, _, _, _, template = seed
    config = replace(config, jax_constraint_weight_ladder=(1.0,))
    free = rv._exact_d_free_indices(config, template.size)
    bad_column = int(np.flatnonzero(free == 3 * config.n_control - 2)[0])
    lower = np.full(template.size, -100.0)
    upper = np.full(template.size, 100.0)
    lower[2 * config.n_control:3 * config.n_control] = 0.0
    upper[2 * config.n_control:3 * config.n_control] = 1.05 * math.sqrt(config.epsilon) * config.Rt
    lower[-1], upper[-1] = 0.0, 1.0
    fraction = 1.01 / 1.05

    def invalid_result(function, solver, initial, **kwargs):
        invalid = initial.at[bad_column].set(jnp.log(fraction / (1.0 - fraction)))
        return SimpleNamespace(value=invalid, result=api.optx.RESULTS.successful, stats={"num_steps": 1})

    monkeypatch.setattr(api.optx, "least_squares", invalid_result)
    with pytest.raises(rv.RaoBoundaryDomainError):
        api.least_squares_jax(config, template, lower, upper, max_steps=1)
