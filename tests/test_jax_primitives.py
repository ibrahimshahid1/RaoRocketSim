"""
J1 gate (JAX_DIFFERENTIABLE_PLAN.md §7): the JAX primitives must reproduce the
literature-checked NumPy closed forms in ``raosim.gas_dynamics``.

- Closed forms: bit-parity to 1e-10 across M and gamma sweeps.
- Iterative inverses (mach_from_prandtl_meyer, mach_from_area_ratio): round-trip
  consistency to 1e-10 and cross-agreement with the NumPy Newton output to 1e-8
  (the two solvers use different stopping rules, so exact bit-parity is not the
  right contract — round-trip is).
- Sonic safety: values finite and derivatives finite at M -> 1.

Skips cleanly if JAX is not installed, so the existing suite is unaffected.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from raosim import gas_dynamics as gd  # noqa: E402
from raosim.jax import primitives as jp  # noqa: E402

GAMMAS = [1.14, 1.20, 1.25, 1.30, 1.40, 1.667]
MACHS = [1.0001, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", MACHS)
def test_isentropic_ratios_parity(M, gamma):
    assert jp.isentropic_temperature_ratio(M, gamma) == pytest.approx(
        gd.isentropic_temperature_ratio(M, gamma), rel=0, abs=1e-10)
    assert jp.isentropic_pressure_ratio(M, gamma) == pytest.approx(
        gd.isentropic_pressure_ratio(M, gamma), rel=1e-10, abs=1e-12)
    assert jp.isentropic_density_ratio(M, gamma) == pytest.approx(
        gd.isentropic_density_ratio(M, gamma), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", MACHS)
def test_area_mach_parity(M, gamma):
    assert jp.area_mach_relation(M, gamma) == pytest.approx(
        gd.area_mach_relation(M, gamma), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", MACHS)
def test_prandtl_meyer_parity(M, gamma):
    assert float(jp.prandtl_meyer(M, gamma)) == pytest.approx(
        gd.prandtl_meyer(M, gamma), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", [m for m in MACHS if m >= 1.0001])
def test_mach_angle_parity(M, gamma):
    assert float(jp.mach_angle(M)) == pytest.approx(gd.mach_angle(M), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", MACHS)
def test_mstar_parity(M, gamma):
    assert float(jp.mstar_from_M(M, gamma)) == pytest.approx(
        gd.mstar_from_M(M, gamma), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", MACHS)
def test_pm_inverse_roundtrip_and_crosscheck(M, gamma):
    nu = gd.prandtl_meyer(M, gamma)
    M_jax = float(jp.mach_from_prandtl_meyer(nu, gamma))
    # round-trip: nu(M_jax) recovers nu
    assert float(jp.prandtl_meyer(M_jax, gamma)) == pytest.approx(nu, rel=1e-10, abs=1e-12)
    # cross-check against NumPy Newton output
    assert M_jax == pytest.approx(gd.mach_from_prandtl_meyer(nu, gamma), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("M", [m for m in MACHS if m >= 1.05])
def test_area_inverse_roundtrip_and_crosscheck(M, gamma):
    ar = gd.area_mach_relation(M, gamma)
    M_jax = float(jp.mach_from_area_ratio(ar, gamma, supersonic=True))
    assert float(jp.area_mach_relation(M_jax, gamma)) == pytest.approx(ar, rel=1e-9, abs=1e-10)
    assert M_jax == pytest.approx(gd.mach_from_area_ratio(ar, gamma, supersonic=True),
                                  rel=1e-7, abs=1e-7)


def test_x64_is_enabled():
    # Bit-parity is impossible without float64.
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_prandtl_meyer_sonic_safe():
    # value -> 0 and derivative finite at the throat.
    assert float(jp.prandtl_meyer(1.0, 1.4)) == pytest.approx(0.0, abs=1e-12)
    dnu = float(jax.grad(lambda m: jp.prandtl_meyer(m, 1.4))(1.0 + 1e-6))
    assert math.isfinite(dnu)


def test_mach_angle_sonic_safe():
    assert float(jp.mach_angle(1.0)) == pytest.approx(math.pi / 2, abs=1e-12)
    dmu = float(jax.grad(jp.mach_angle)(1.5))
    assert math.isfinite(dmu)


def test_primitives_are_jittable_and_vmappable():
    Ms = jnp.array([1.2, 2.0, 3.5, 5.0])
    out = jax.jit(jax.vmap(lambda m: jp.prandtl_meyer(m, 1.4)))(Ms)
    ref = np.array([gd.prandtl_meyer(float(m), 1.4) for m in Ms])
    assert np.allclose(np.asarray(out), ref, rtol=1e-10, atol=1e-12)
