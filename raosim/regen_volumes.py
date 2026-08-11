"""Shared numerical volume kernel for regenerative thrust-chamber metal.

The kernel deliberately does one job: integrate the three disjoint material
regions represented by :class:`raosim.regen_profile.RegenWallProfile`:

* the hot-gas liner;
* the channel lands/ribs between coolant passages; and
* the structural closeout/jacket.

NASA SP-8087, *Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, Van
Huff and Fairchild (1972), section 2.1.1.1, PDF page 24 (NASA SP-8087,
NTRS 19730022965), describes channel-wall coolant passages as integral with
the thin high-conductivity liner.  Section 2.1.3.1 separately describes the
outer reinforcement/jacket.  That literature establishes that all three
regions are real hardware; it does not prescribe this quadrature.

The equations below are geometric calculations (Pappus/shell volumes), not a
thrust-chamber mass correlation.  NASA SP-125 equation 8-32 uses the same
surface-times-thickness form for a cylindrical tank shell, which corroborates
the geometry but is not chamber-specific authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "RegenVolumes",
    "DifferentiableRegenVolumes",
    "integrate_regen_volumes",
    "integrate_regen_volumes_jax",
    "regen_geometry_id",
]


@dataclass(frozen=True)
class RegenVolumes:
    """Integrated material volumes in cubic metres."""

    liner: float
    lands: float
    closeout: float

    @property
    def total(self) -> float:
        return float(self.liner + self.lands + self.closeout)

    def to_dict(self) -> dict[str, float]:
        return {
            "liner_volume_m3": float(self.liner),
            "land_volume_m3": float(self.lands),
            "closeout_volume_m3": float(self.closeout),
            "total_volume_m3": self.total,
        }


class DifferentiableRegenVolumes(NamedTuple):
    """Pure-array form returned by :func:`integrate_regen_volumes_jax`.

    ``geometry_valid`` is deliberately separate from the volumes.  A traced
    optimizer probe cannot raise a Python exception, so an invalid negative
    land width is continued as its absolute magnitude for the objective while
    the live land-fit constraint rejects the point.  This prevents invalid
    geometry from acquiring either negative or falsely authoritative mass.
    """

    liner: Any
    lands: Any
    closeout: Any
    total: Any
    land_area_fraction: Any
    geometry_valid: Any


def _station_array(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def integrate_regen_volumes(
    profile: Any | None = None,
    *,
    x: Any | None = None,
    r_inner: Any | None = None,
    t_hot: Any | None = None,
    channel_width: Any | None = None,
    channel_height: Any | None = None,
    land_width: Any | None = None,
    t_jacket: Any | None = None,
    joint_allowance: float = 1.0,
) -> RegenVolumes:
    """Integrate liner, land and closeout volumes on a meridional grid.

    A ``RegenWallProfile`` may be passed directly.  The explicit keyword form
    exists for NumPy design searches so they use exactly the same equations as
    the hardware mass ledger.  ``joint_allowance`` is an explicit multiplier;
    the default adds no unsourced manufacturing allowance.
    """

    if profile is not None:
        if any(
            value is not None
            for value in (
                x,
                r_inner,
                t_hot,
                channel_width,
                channel_height,
                land_width,
                t_jacket,
            )
        ):
            raise ValueError("pass either profile or explicit station arrays, not both")
        x = getattr(profile, "x")
        r_inner = getattr(profile, "r_inner")
        t_hot = getattr(profile, "t_hot")
        channel_width = getattr(profile, "channel_width")
        channel_height = getattr(profile, "channel_height")
        land_width = getattr(profile, "land_width")
        t_jacket = getattr(profile, "t_jacket")

    required = {
        "x": x,
        "r_inner": r_inner,
        "t_hot": t_hot,
        "channel_width": channel_width,
        "channel_height": channel_height,
        "land_width": land_width,
        "t_jacket": t_jacket,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing regenerative geometry: " + ", ".join(missing))

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size < 2 or not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be a finite one-dimensional grid with at least two stations")
    n = int(x_arr.size)
    r = _station_array(r_inner, n, "r_inner")
    tw = _station_array(t_hot, n, "t_hot")
    width = _station_array(channel_width, n, "channel_width")
    height = _station_array(channel_height, n, "channel_height")
    land = _station_array(land_width, n, "land_width")
    jacket = _station_array(t_jacket, n, "t_jacket")

    if np.any(np.diff(x_arr) <= 0.0):
        raise ValueError("x must be strictly increasing")
    if np.any(r <= 0.0) or np.any(tw <= 0.0) or np.any(height <= 0.0):
        raise ValueError("r_inner, t_hot and channel_height must be positive")
    if np.any(width <= 0.0) or np.any(land < 0.0) or np.any(jacket <= 0.0):
        raise ValueError(
            "channel_width and t_jacket must be positive; land_width must be nonnegative"
        )
    pitch = width + land
    if np.any(pitch <= 0.0):
        raise ValueError("channel_width + land_width must be positive")
    try:
        allowance = float(joint_allowance)
    except (TypeError, ValueError) as exc:
        raise ValueError("joint_allowance must be a positive finite multiplier") from exc
    if not math.isfinite(allowance) or allowance <= 0.0:
        raise ValueError("joint_allowance must be a positive finite multiplier")

    from raosim.regen_profile import _nodal_weights_from_segments

    ds = _nodal_weights_from_segments(np.hypot(np.diff(x_arr), np.diff(r)))
    r_channel_inner = r + tw
    r_channel_outer = r_channel_inner + height
    land_fraction = land / pitch

    liner_area = 2.0 * np.pi * (r + 0.5 * tw) * tw
    land_area = (
        np.pi * (r_channel_outer**2 - r_channel_inner**2) * land_fraction
    )
    closeout_area = (
        2.0
        * np.pi
        * (r_channel_outer + 0.5 * jacket)
        * jacket
    )
    volumes = RegenVolumes(
        liner=float(np.sum(liner_area * ds)) * allowance,
        lands=float(np.sum(land_area * ds)) * allowance,
        closeout=float(np.sum(closeout_area * ds)) * allowance,
    )
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (volumes.liner, volumes.lands, volumes.closeout)
    ):
        raise ValueError("regenerative volume integration produced invalid geometry")
    return volumes


def integrate_regen_volumes_jax(
    *,
    r_inner: Any,
    dseg: Any,
    t_hot: Any,
    channel_width: Any,
    channel_height: Any,
    land_width: Any,
    t_jacket: Any,
) -> DifferentiableRegenVolumes:
    """Equation-identical differentiable mirror of the NumPy volume kernel.

    The caller supplies meridional segment lengths because the MDO station
    grid already computes them.  On the valid domain, the liner, land, and
    closeout equations are exactly those in :func:`integrate_regen_volumes`.

    JAX transformations cannot raise on a traced invalid land width.  For such
    probes only, ``abs(land_width)`` supplies a nonnegative continuation for
    the mass objective and ``geometry_valid=False`` records that the result is
    not physical.  The engine's independent ``land_min`` constraint remains
    authoritative and must be nonnegative at an accepted design.
    """

    import raosim.jax  # noqa: F401  -- repository-wide float64 policy
    import jax.numpy as jnp

    r = jnp.asarray(r_inner, dtype=jnp.float64)
    seg = jnp.asarray(dseg, dtype=jnp.float64)
    tw = jnp.asarray(t_hot, dtype=jnp.float64)
    width = jnp.asarray(channel_width, dtype=jnp.float64)
    height = jnp.asarray(channel_height, dtype=jnp.float64)
    land_raw = jnp.asarray(land_width, dtype=jnp.float64)
    jacket = jnp.asarray(t_jacket, dtype=jnp.float64)

    interior = 0.5 * (seg[:-1] + seg[1:])
    ds = jnp.concatenate((0.5 * seg[:1], interior, 0.5 * seg[-1:]))

    # Exact on the valid domain.  The absolute-value continuation makes an
    # infeasible probe more massive as it moves farther outside the land-fit
    # domain; it never lets negative material reduce the objective.
    land = jnp.abs(land_raw)
    pitch = width + land
    land_fraction = land / jnp.maximum(pitch, 1.0e-12)
    r_channel_inner = r + tw
    r_channel_outer = r_channel_inner + height

    liner_area = 2.0 * jnp.pi * (r + 0.5 * tw) * tw
    land_area = (
        jnp.pi
        * (r_channel_outer**2 - r_channel_inner**2)
        * land_fraction
    )
    closeout_area = (
        2.0 * jnp.pi * (r_channel_outer + 0.5 * jacket) * jacket
    )
    liner = jnp.sum(liner_area * ds)
    lands = jnp.sum(land_area * ds)
    closeout = jnp.sum(closeout_area * ds)
    total = liner + lands + closeout
    valid = (
        jnp.all(jnp.isfinite(r))
        & jnp.all(jnp.isfinite(seg))
        & jnp.all(jnp.isfinite(tw))
        & jnp.all(jnp.isfinite(width))
        & jnp.all(jnp.isfinite(height))
        & jnp.all(jnp.isfinite(land_raw))
        & jnp.all(jnp.isfinite(jacket))
        & jnp.all(r > 0.0)
        & jnp.all(seg > 0.0)
        & jnp.all(tw > 0.0)
        & jnp.all(width > 0.0)
        & jnp.all(height > 0.0)
        & jnp.all(land_raw >= 0.0)
        & jnp.all(jacket > 0.0)
    )
    return DifferentiableRegenVolumes(
        liner=liner,
        lands=lands,
        closeout=closeout,
        total=total,
        land_area_fraction=land_fraction,
        geometry_valid=valid,
    )


def regen_geometry_id(profile: Any) -> str:
    """Return a deterministic ID for the complete regenerative geometry.

    The identifier binds ledgers and CAD reports to the same station arrays.  It
    does not imply that a CAD body was successfully built or qualified.
    """

    payload = {
        "schema": "resolved_regen_hardware_geometry@1",
        "channel_count": int(getattr(profile, "channel_count")),
        "helix_turns": float(getattr(profile, "helix_turns", 0.0)),
        "x": np.asarray(getattr(profile, "x"), dtype=float).tolist(),
        "r_inner": np.asarray(getattr(profile, "r_inner"), dtype=float).tolist(),
        "t_hot": np.asarray(getattr(profile, "t_hot"), dtype=float).tolist(),
        "channel_width": np.asarray(
            getattr(profile, "channel_width"), dtype=float
        ).tolist(),
        "channel_height": np.asarray(
            getattr(profile, "channel_height"), dtype=float
        ).tolist(),
        "land_width": np.asarray(getattr(profile, "land_width"), dtype=float).tolist(),
        "t_jacket": np.asarray(getattr(profile, "t_jacket"), dtype=float).tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "regen:" + hashlib.sha256(encoded).hexdigest()
