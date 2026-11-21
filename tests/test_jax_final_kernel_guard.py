from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
from raosim.jax import theta_b_solve as live
from raosim.nasa_moc import RaoKernelError, build_kernel
from raosim.rao_variational import RaoSolverConfig


@pytest.fixture(scope="module")
def nasa_reference():
    kernel = build_kernel(
        1.0, 1.0, math.radians(15.2196), 1.4, n_kernel=101,
        starting_line_method="nasa_visible_kliegel_levine",
    )
    assert kernel.reached_wall
    config = RaoSolverConfig(
        Rt=1.0, epsilon=6.7, gamma=1.4, n_control=8, n_kernel=101,
        throat_downstream_radius_factor=1.0, throat_upstream_radius_factor=1.0,
        kernel_bd=tuple(point.to_flow_node() for point in kernel.bd),
        jax_constraint_weight_ladder=(1.0,),
    )
    return config, kernel


def test_actual_truncated_final_march_is_refused(nasa_reference):
    config, kernel = nasa_reference
    with pytest.raises(RaoKernelError, match="final live JAX kernel did not reach"):
        live._require_final_live_kernel(config, kernel, kernel.theta_B, 1, math.radians(0.5))


def test_actual_complete_reference_final_march_is_accepted(nasa_reference):
    config, kernel = nasa_reference
    capacity = live._n_rows_bound(kernel.theta_B + math.radians(0.25), math.radians(0.5))
    marched = live._require_final_live_kernel(config, kernel, kernel.theta_B, capacity, math.radians(0.5))
    assert bool(np.asarray(marched.reached_wall))


def test_successful_optimizer_cannot_bypass_final_march_completeness(nasa_reference, monkeypatch):
    import jax.numpy as jnp

    config, kernel = nasa_reference
    monkeypatch.setattr(live, "make_residual_theta_b", lambda *args, **kwargs: (lambda vector: jnp.zeros(200), 1))
    monkeypatch.setattr(live.optx, "least_squares", lambda function, solver, initial, **kwargs: SimpleNamespace(
        value=initial, result=live.optx.RESULTS.successful, stats={"num_steps": 1},
    ))
    size = 3 * config.n_control + 4
    with pytest.raises(RaoKernelError, match="optimizer status cannot validate"):
        live.least_squares_jax_theta_b(
            config, kernel, np.zeros(size), -np.ones(size), np.ones(size), max_steps=1,
        )


def test_direct_live_api_refuses_exact_d_mode(nasa_reference):
    config, kernel = nasa_reference
    with pytest.raises(NotImplementedError, match="exact_d_attachment with live theta_B"):
        live.make_residual_theta_b(replace(config, exact_d_attachment=True), kernel, theta_b_max=kernel.theta_B + 0.01)
