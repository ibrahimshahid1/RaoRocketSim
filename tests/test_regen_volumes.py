"""Shared regenerative metal-volume kernel contracts."""

import math

import numpy as np
import pytest

from raosim.regen_profile import RegenWallProfile
from raosim.regen_volumes import (
    integrate_regen_volumes,
    integrate_regen_volumes_jax,
    regen_geometry_id,
)


def _straight_profile(*, land_width=0.001):
    return RegenWallProfile(
        x=np.array([0.0, 0.2]),
        r_inner=np.array([0.05, 0.05]),
        t_hot=np.array([0.001, 0.001]),
        channel_width=np.array([0.001, 0.001]),
        channel_height=np.array([0.002, 0.002]),
        land_width=np.array([land_width, land_width]),
        t_jacket=np.array([0.0015, 0.0015]),
        channel_count=80,
        helix_turns=0.0,
        Rt=0.05,
    )


def test_shared_kernel_includes_liner_lands_and_closeout():
    profile = _straight_profile()
    volumes = integrate_regen_volumes(profile)
    length = 0.2
    r, tw, h, tj = 0.05, 0.001, 0.002, 0.0015
    expected_liner = 2.0 * math.pi * (r + 0.5 * tw) * tw * length
    expected_lands = (
        math.pi * ((r + tw + h) ** 2 - (r + tw) ** 2) * 0.5 * length
    )
    expected_closeout = (
        2.0 * math.pi * (r + tw + h + 0.5 * tj) * tj * length
    )
    assert volumes.liner == pytest.approx(expected_liner)
    assert volumes.lands == pytest.approx(expected_lands)
    assert volumes.closeout == pytest.approx(expected_closeout)
    assert volumes.total == pytest.approx(
        expected_liner + expected_lands + expected_closeout
    )


def test_geometry_id_binds_every_regenerative_region():
    base = _straight_profile()
    same = _straight_profile()
    changed_land = _straight_profile(land_width=0.0011)
    assert regen_geometry_id(base) == regen_geometry_id(same)
    assert regen_geometry_id(base) != regen_geometry_id(changed_land)


def test_invalid_regenerative_geometry_is_rejected_not_clamped():
    bad = _straight_profile()
    bad.channel_height[0] = -0.001
    with pytest.raises(ValueError, match="positive"):
        integrate_regen_volumes(bad)


def test_jax_kernel_matches_numpy_kernel_on_valid_geometry():
    import raosim.jax  # noqa: F401  -- enable float64 before creating arrays
    jnp = pytest.importorskip("jax.numpy")
    profile = _straight_profile()
    expected = integrate_regen_volumes(profile)
    dseg = np.hypot(np.diff(profile.x), np.diff(profile.r_inner))
    actual = integrate_regen_volumes_jax(
        r_inner=jnp.asarray(profile.r_inner),
        dseg=jnp.asarray(dseg),
        t_hot=jnp.asarray(profile.t_hot),
        channel_width=jnp.asarray(profile.channel_width),
        channel_height=jnp.asarray(profile.channel_height),
        land_width=jnp.asarray(profile.land_width),
        t_jacket=jnp.asarray(profile.t_jacket),
    )

    assert bool(actual.geometry_valid)
    assert float(actual.liner) == pytest.approx(expected.liner, rel=1e-12)
    assert float(actual.lands) == pytest.approx(expected.lands, rel=1e-12)
    assert float(actual.closeout) == pytest.approx(expected.closeout, rel=1e-12)


def test_jax_invalid_probe_never_acquires_negative_land_mass():
    import raosim.jax  # noqa: F401  -- enable float64 before creating arrays
    jnp = pytest.importorskip("jax.numpy")
    profile = _straight_profile(land_width=-0.001)
    dseg = np.hypot(np.diff(profile.x), np.diff(profile.r_inner))
    actual = integrate_regen_volumes_jax(
        r_inner=jnp.asarray(profile.r_inner),
        dseg=jnp.asarray(dseg),
        t_hot=jnp.asarray(profile.t_hot),
        channel_width=jnp.asarray(profile.channel_width),
        channel_height=jnp.asarray(profile.channel_height),
        land_width=jnp.asarray(profile.land_width),
        t_jacket=jnp.asarray(profile.t_jacket),
    )

    assert not bool(actual.geometry_valid)
    assert float(actual.lands) >= 0.0
    assert float(actual.total) >= float(actual.liner + actual.closeout)
