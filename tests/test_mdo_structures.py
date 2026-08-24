"""Differentiable structural-screen parity and constraint wiring."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import raosim.jax  # noqa: F401 -- enable float64 before importing JAX
import jax
import jax.numpy as jnp

from raosim.mdo.constraints import CONSTRAINT_MANIFEST
from raosim.mdo.engine import solve_engine
from raosim.mdo.grid import build_station_grid
from raosim.mdo.nlp import DEFAULT_ENFORCED
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.structures import rib_supported_liner_buckling_screen
from raosim.physics import rib_supported_liner_buckling_profile


def test_rib_supported_liner_screen_matches_numpy_oracle():
    """The JAX port changes no physics from the traditional screen."""

    pressure = np.array([0.8e6, -1.5e6, 2.2e6, 3.0e6])
    radius = np.array([0.045, 0.052, 0.068, 0.081])
    thickness = np.array([0.65e-3, 0.72e-3, 0.80e-3, 0.91e-3])
    span = np.array([0.45e-3, 0.52e-3, 0.60e-3, 0.70e-3])
    kwargs = dict(
        pressure_differential=pressure,
        inner_radius=radius,
        wall_thickness=thickness,
        unsupported_span=span,
        elastic_modulus=110.0e9,
        poisson_ratio=0.33,
        plate_coefficient=4.0,
        knockdown=0.65,
    )
    expected = rib_supported_liner_buckling_profile(**kwargs)
    actual = rib_supported_liner_buckling_screen(**kwargs)

    np.testing.assert_allclose(
        np.asarray(actual.compressive_stress_profile),
        expected["compressive_stress_profile"],
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual.critical_stress_profile),
        expected["critical_stress_profile"],
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual.margin_profile),
        expected["margin_profile"],
        rtol=2.0e-15,
    )
    assert float(actual.margin) == pytest.approx(expected["margin"], rel=2.0e-15)


def test_rib_supported_liner_screen_is_jittable_and_differentiable():
    def minimum_ratio(thickness):
        return rib_supported_liner_buckling_screen(
            pressure_differential=jnp.asarray([1.0e6, 2.0e6]),
            inner_radius=jnp.asarray([0.05, 0.07]),
            wall_thickness=thickness,
            unsupported_span=jnp.asarray(0.6e-3),
            elastic_modulus=jnp.asarray(110.0e9),
            poisson_ratio=jnp.asarray(0.33),
        ).margin

    value, derivative = jax.jit(jax.value_and_grad(minimum_ratio))(
        jnp.asarray(0.8e-3)
    )
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
    assert float(derivative) > 0.0


def test_liner_buckling_is_a_sourced_mandatory_final_gate():
    spec = next(row for row in CONSTRAINT_MANIFEST if row.name == "liner_buckling")
    assert spec.engine_key == "liner_buckling_margin"
    assert spec.optimizer_role == "final_gate"
    assert spec.category == "structures"
    assert "SP-8087-sec-2.1.1.4" in (spec.source_id or "")
    assert "classical-long-plate" in (spec.source_id or "")
    assert "SP-125" not in (spec.source_id or "")
    assert spec.name not in DEFAULT_ENFORCED

    from raosim.mdo.constraints import (
        ENGINE_CONSTRAINT_SPECS,
        constraint_metadata,
    )
    from raosim.mdo.properties import constant_chamber_surfaces

    mission = MissionSpec()
    surfaces = constant_chamber_surfaces(
        gamma=mission.gamma, Tc=mission.Tc, R_gas=mission.R_gas
    )
    applicable, available, required, _ = constraint_metadata(mission, surfaces)
    index = {row.name: i for i, row in enumerate(ENGINE_CONSTRAINT_SPECS)}
    i = index[spec.name]
    assert applicable[i] and available[i] and required[i]

    design = DesignVector(
        Pc=jnp.asarray(3.0e6),
        eps=jnp.asarray(8.0),
        dp_f_frac=jnp.asarray(0.2),
        dp_o_frac=jnp.asarray(0.2),
    )
    result = solve_engine(design, mission)
    grid = build_station_grid(
        result.Rt, design.eps, mission, gamma=jnp.asarray(mission.gamma)
    )
    oracle = rib_supported_liner_buckling_screen(
        pressure_differential=result.cooling.liner_pressure_differential,
        inner_radius=grid.r,
        wall_thickness=design.t_wall,
        unsupported_span=design.channel_width,
        elastic_modulus=mission.liner_E,
        poisson_ratio=mission.liner_poisson,
    )
    expected_margin = jnp.min(
        1.0
        - oracle.compressive_stress_profile
        / jnp.maximum(oracle.critical_stress_profile, 1.0e-9)
    )
    assert float(result.constraints[spec.engine_key]) == pytest.approx(
        float(expected_margin), rel=1.0e-12
    )


def test_liner_buckling_final_gate_rejects_an_nlp_candidate(monkeypatch):
    import raosim.mdo.nlp as nlp_module

    # Bypass only the external optimizer iterations.  The real post-solve
    # engine evaluation and manifest-driven verdict reduction still run at the
    # reference point.
    def accept_initial_point(_fun, u0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray(u0), success=True, nit=0,
            message="forced host optimizer completion",
        )

    monkeypatch.setattr(nlp_module, "minimize", accept_initial_point)
    mission = replace(MissionSpec(), liner_E=1.0e7)
    result = nlp_module.solve_min_mass(mission, 0.0)

    final_gates = {
        row.name
        for row in CONSTRAINT_MANIFEST
        if row.optimizer_role == "final_gate"
    }
    assert final_gates <= set(result.constraints)
    assert final_gates.isdisjoint(result.enforced)
    assert result.constraints["liner_buckling"] < 0.0
    assert result.physics_status == "fail"
    assert not result.feasible
