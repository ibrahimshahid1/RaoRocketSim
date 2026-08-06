"""Contract tests for the fixed-shape pure-JAX ``EngineState``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import raosim.jax  # noqa: F401
import jax
import jax.numpy as jnp

from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.engine import chamber_surfaces_for
from raosim.mdo.state import (
    ENGINE_CONSTRAINT_NAMES,
    ENGINE_STATE_SCHEMA_VERSION,
    MASS_FIELD_NAMES,
    solve_engine_state,
    surface_signature,
)


def _design() -> DesignVector:
    return DesignVector(
        Pc=jnp.asarray(3.0e6),
        eps=jnp.asarray(8.0),
        dp_f_frac=jnp.asarray(0.2),
        dp_o_frac=jnp.asarray(0.2),
        D_pintle=jnp.asarray(0.020),
        N_rpm=jnp.asarray(30000.0),
        channel_width=jnp.asarray(5.0e-4),
        channel_height=jnp.asarray(1.5e-3),
        film_frac=jnp.asarray(0.10),
        t_wall=jnp.asarray(8.0e-4),
    )


@pytest.fixture(scope="module")
def state():
    return solve_engine_state(_design(), MissionSpec())


def test_state_is_a_numeric_fixed_shape_pytree(state):
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves
    assert int(state.schema_version) == ENGINE_STATE_SCHEMA_VERSION
    assert all(
        np.issubdtype(np.asarray(leaf).dtype, np.number)
        or np.issubdtype(np.asarray(leaf).dtype, np.bool_)
        for leaf in leaves
    )
    n = state.geometry.x.shape[0]
    assert state.geometry.r.shape == (n,)
    assert state.geometry.area_ratio.shape == (n,)
    assert state.geometry.mach.shape == (n,)
    assert state.geometry.dseg.shape == (n - 1,)
    assert state.thermal.T_wg.shape == (n,)
    assert state.thermal.T_wc.shape == (n,)
    assert state.thermal.sigma_pressure_profile.shape == (n,)
    assert state.thermal.sigma_combined.shape == (n,)
    assert state.thermal.coolant_pressure.shape == (n,)
    assert state.thermal.gas_pressure.shape == (n,)
    assert state.thermal.liner_pressure_differential.shape == (n,)
    assert state.thermal.residual.shape == (n,)
    assert state.constraints.values.shape == (len(ENGINE_CONSTRAINT_NAMES),)
    assert state.masses.values.shape == (len(MASS_FIELD_NAMES),)


def test_complete_ideal_and_delivered_performance_convention(state):
    p = state.performance
    assert float(p.cstar_delivered) == pytest.approx(
        float(p.eta_cstar * p.cstar_ideal), rel=1e-12)
    assert float(p.Cf_delivered) == pytest.approx(
        float(p.eta_CF * p.Cf_ideal), rel=1e-12)
    assert float(p.Isp_delivered) == pytest.approx(
        float(p.Cf_delivered * p.cstar_delivered / 9.80665), rel=1e-12)
    assert float(p.thrust_delivered) == pytest.approx(
        float(p.Cf_delivered * p.Pc * p.At), rel=1e-12)
    assert float(p.Pe_over_Pc) == pytest.approx(float(p.Pe / p.Pc), rel=1e-12)


def test_fuel_routes_are_mass_conservative_for_defined_topology(state):
    p = state.performance
    assert float(p.mdot_fuel_total) == pytest.approx(
        float(p.mdot_fuel_core + p.mdot_film), rel=1e-12)
    assert float(p.mdot_total) == pytest.approx(
        float(p.mdot_oxidizer + p.mdot_fuel_total), rel=1e-12)
    assert float(p.mdot_core_total) == pytest.approx(
        float(p.mdot_oxidizer + p.mdot_fuel_core), rel=1e-12)
    assert float(p.mdot_regen_jacket) == pytest.approx(
        float(p.mdot_fuel_core), rel=1e-12)


def test_mass_branches_and_unavailable_values_are_not_zero_placeholders(state):
    idx = {name: i for i, name in enumerate(MASS_FIELD_NAMES)}
    for name in (
        "battery_energy_cell_mass",
        "battery_power_cell_mass",
        "battery_energy_installed_mass",
        "battery_power_installed_mass",
        "battery_governing_installed_mass",
        "electric_feed_package_exact_mass",
        "electric_feed_package_objective_mass",
        # Thrust-chamber structure is integrated on the station grid
        # (raosim.mdo.mass, SP-125 eq. 8-32), so it is real hardware mass.
        "thrust_chamber_liner_mass",
        "thrust_chamber_land_mass",
        "thrust_chamber_closeout_mass",
        "thrust_chamber_mass",
    ):
        assert bool(state.masses.availability[idx[name]])
        assert np.isfinite(float(state.masses.values[idx[name]]))

    # The chamber rollup is exactly its three load-path branches.
    assert float(state.masses.values[idx["thrust_chamber_mass"]]) == (
        pytest.approx(
            sum(
                float(state.masses.values[idx[name]])
                for name in (
                    "thrust_chamber_liner_mass",
                    "thrust_chamber_land_mass",
                    "thrust_chamber_closeout_mass",
                )
            ),
            rel=1e-12,
        )
    )
    assert float(state.masses.values[idx["thrust_chamber_mass"]]) > 0.0

    # Injector hardware needs the host-side machined layout, and a total dry
    # mass needs plumbing/mounts, so both stay unavailable -- NaN, not zero.
    for name in ("injector_mass", "total_dry_mass"):
        i = idx[name]
        assert not bool(state.masses.availability[i])
        assert np.isnan(float(state.masses.values[i]))

    exact = float(
        state.masses.values[idx["electric_feed_package_exact_mass"]]
    )
    objective = float(
        state.masses.values[idx["electric_feed_package_objective_mass"]]
    )
    expected = sum(
        float(state.masses.values[idx[name]])
        for name in (
            "pump_mass",
            "motor_mass",
            "inverter_mass",
            "battery_governing_installed_mass",
        )
    )
    assert exact == pytest.approx(expected, rel=1e-12)
    assert objective >= exact


def test_state_carries_the_numerical_input_convention(state):
    c = state.input_conventions
    assert c.mission_fingerprint.shape == (8,)
    assert c.surface_signature.shape == (8,)
    assert float(c.thrust) == pytest.approx(13_000.0)
    assert not bool(c.couple_eta_cstar)
    assert float(c.OF) == pytest.approx(2.27)
    assert float(c.eta_cstar_nominal) == pytest.approx(0.975)
    assert float(c.eta_CF) == pytest.approx(0.985)
    assert float(c.throat_ru_factor) == pytest.approx(1.5)
    assert float(c.throat_rd_factor) == pytest.approx(0.382)
    assert float(c.pump_speed_rpm) == pytest.approx(30_000.0)
    assert float(c.liner_structural_fos) == pytest.approx(1.5)
    assert int(c.channel_count) == 192


def test_surface_identity_covers_derivative_fields_and_provenance():
    surfaces = chamber_surfaces_for(MissionSpec())
    changed_gamma = replace(
        surfaces.gamma,
        Zx=surfaces.gamma.Zx.at[0, 0].add(1.0e-12),
    )
    changed_derivative = replace(surfaces, gamma=changed_gamma)
    changed_provenance = replace(
        surfaces, provenance=surfaces.provenance + ":different-source"
    )

    reference = np.asarray(surface_signature(surfaces), dtype=np.uint32)
    assert not np.array_equal(
        reference,
        np.asarray(surface_signature(changed_derivative), dtype=np.uint32),
    )
    assert not np.array_equal(
        reference,
        np.asarray(surface_signature(changed_provenance), dtype=np.uint32),
    )


def test_full_state_is_jittable_and_differentiable():
    mission = MissionSpec()
    x0 = _design().to_array()
    compiled = jax.jit(
        lambda a: solve_engine_state(DesignVector.from_array(a), mission))
    compiled_state = compiled(x0)
    value = compiled_state.performance.Isp_delivered
    reference_state = solve_engine_state(_design(), mission)
    assert float(value) == pytest.approx(
        float(reference_state.performance.Isp_delivered), rel=1e-10
    )
    assert np.array_equal(
        np.asarray(
            compiled_state.input_conventions.surface_signature,
            dtype=np.uint32,
        ),
        np.asarray(
            reference_state.input_conventions.surface_signature,
            dtype=np.uint32,
        ),
    )
    assert compiled_state.geometry.x.shape == compiled_state.thermal.T_wc.shape
    assert compiled_state.residuals.outer.shape == (2,)

    grad_eps = jax.grad(
        lambda eps: solve_engine_state(
            DesignVector(
                Pc=x0[0],
                eps=eps,
                dp_f_frac=x0[2],
                dp_o_frac=x0[3],
                D_pintle=x0[4],
                N_rpm=x0[5],
                channel_width=x0[6],
                channel_height=x0[7],
                film_frac=x0[8],
                t_wall=x0[9],
            ),
            mission,
        ).performance.Isp_delivered
    )(x0[1])
    assert np.isfinite(float(grad_eps))
    assert abs(float(grad_eps)) > 0.0
