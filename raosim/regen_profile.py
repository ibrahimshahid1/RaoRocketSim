"""regen_profile.py — station-wise regenerative wall + channel geometry.

A real regenerative wall is not one thickness.  Following NASA SP-125
(Huzel & Huang, coaxial-shell / tube-and-jacket thrust-chamber design,
``propulsion_texts/19710019929.pdf`` printed pp. 108-110), the wall has
several distinct dimensions the thermal and structural design size
*separately*, and they vary along the contour:

* ``t_hot(x)``    — hot-gas liner (inner-shell) thickness
* ``channel_height(x)`` — coolant channel depth ``h``
* ``channel_width(x)``  — coolant channel width ``w``
* ``land_width(x)`` — rib / land width ``b`` between channels
* ``t_jacket(x)``  — outer closeout (jacket) thickness

SP-125 also notes the passage area should *vary along the chamber to hold
an appropriate coolant velocity* and that the **throat is the critical
region** (sample calc 4-4).  ``RegenWallProfile`` carries these as
per-station arrays over the contour and derives what the cooling and
pressure-drop models need.

Helical channels
----------------
When the channels are wound helically (``helix_turns > 0``) the coolant
travels farther than the axial run, which raises the Darcy-Weisbach
pressure drop ``Δp = f (L/D_h)(ρV²/2g)`` (SP-125 eq. 4-32) in proportion
to the path length ``L``.  The helix here matches
:mod:`raosim.regen_geometry`: a channel's angle is
``θ(x) = θ0 + 2π·turns·(x−x0)/L_axial`` and its centerline sits at
``r_mid = r_inner + t_hot + h/2``, so the per-station path element is

    dl = sqrt(ds_meridian² + (r_mid · dθ)²),   dθ = 2π·turns·dx/L_axial

— exactly the arc the STL coils trace, so the modeled Δp is consistent
with the exported geometry (previously the helix changed only the STL).

All lengths SI [m].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def normal_offset_contour(x, r, distance) -> tuple[np.ndarray, np.ndarray]:
    """Offset an axisymmetric meridional contour along its outward normal.

    ``distance`` may be scalar or station-wise.  This is the shared geometry
    primitive for liner OD, channel floor/mid/ceiling, and CAD wall export;
    adding thickness only to the radius is not a uniform wall on a sloped
    convergent or bell.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    d = np.asarray(distance, dtype=float)
    if x.ndim != 1 or r.shape != x.shape:
        raise ValueError("x and r must be equal-length one-dimensional arrays")
    if d.ndim == 0:
        d = np.full_like(x, float(d))
    elif d.shape != x.shape:
        raise ValueError(f"offset distance array must have shape {x.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("offset distance must be finite")

    dx = np.gradient(x)
    dr = np.gradient(r)
    mag = np.maximum(np.hypot(dx, dr), 1e-15)
    nx = -dr / mag
    nr = dx / mag
    # Keep the radial component pointing away from the axis even if an input
    # contour happens to be stored in descending x order.
    flip = nr < 0.0
    nx = np.where(flip, -nx, nx)
    nr = np.where(flip, -nr, nr)
    return x + d * nx, r + d * nr


def helix_passage_lengths(
    x, r_inner, *, helix_turns: float = 0.0,
    t_wall=0.0, channel_height=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-station coolant path-length elements ``(dl, ds_meridian)``.

    ``ds_meridian`` is the wall arc length ``hypot(dx, dr)``; ``dl`` adds
    the helical circumferential travel of the channel centerline
    (radius ``r_inner + t_wall + h/2``) in quadrature.  With
    ``helix_turns = 0`` the two are identical (axial channels), so callers
    are unchanged.  Consistent with :mod:`raosim.regen_geometry`
    (``θ = θ0 + 2π·turns·(x−x0)/L``).
    """
    x = np.asarray(x, dtype=float)
    ri = np.asarray(r_inner, dtype=float)
    mid_offset = np.asarray(t_wall, dtype=float) + 0.5 * np.asarray(
        channel_height, dtype=float
    )
    x_mid, r_mid = normal_offset_contour(x, ri, mid_offset)
    ds_merid = np.hypot(np.gradient(x_mid), np.gradient(r_mid))
    if not helix_turns:
        return ds_merid, ds_merid
    L_axial = max(float(np.max(x) - np.min(x)), 1e-12)
    dtheta = (2.0 * np.pi * float(helix_turns) / L_axial) * np.abs(np.gradient(x))
    dl = np.hypot(ds_merid, r_mid * dtheta)
    return dl, ds_merid


def _as_array(value, n: int) -> np.ndarray:
    """Broadcast a scalar or sequence to a length-``n`` float array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape[0] != n:
        raise ValueError(f"profile array length {arr.shape[0]} != {n} stations")
    return arr.astype(float)


@dataclass
class RegenWallProfile:
    """Station-wise hot-wall + channel + jacket geometry along the contour.

    All geometry fields are per-station arrays (length = number of contour
    stations); :meth:`uniform` / :meth:`tapered` build them from scalars.
    """

    x: np.ndarray              # axial stations [m]
    r_inner: np.ndarray        # gas-side wall radius (the contour) [m]
    t_hot: np.ndarray          # hot-gas liner thickness [m]
    channel_width: np.ndarray  # w [m]
    channel_height: np.ndarray # h [m]
    land_width: np.ndarray     # b (rib) [m]
    t_jacket: np.ndarray       # outer closeout thickness [m]
    channel_count: int         # N
    helix_turns: float = 0.0
    Rt: float | None = None    # throat radius [m] (for the throat station)

    # ---- constructors ---------------------------------------------------
    @classmethod
    def uniform(
        cls, contour: dict, *, channel_count: int, channel_width: float,
        channel_height: float, t_hot: float, land_width: float | None = None,
        t_jacket: float | None = None, helix_turns: float = 0.0,
    ) -> "RegenWallProfile":
        """Constant wall/channel geometry along the contour (the current
        scalar design).  ``land_width`` defaults to the geometric rib
        (pitch − w) and ``t_jacket`` to the liner thickness."""
        x = np.asarray(contour["x"], dtype=float)
        r = np.asarray(contour["y"], dtype=float)
        n = len(x)
        w = _as_array(channel_width, n)
        if land_width is None:
            pitch = 2.0 * np.pi * np.maximum(r, 1e-9) / max(int(channel_count), 1)
            land = np.maximum(pitch - w, 0.0)
        else:
            land = _as_array(land_width, n)
        return cls(
            x=x, r_inner=r, t_hot=_as_array(t_hot, n), channel_width=w,
            channel_height=_as_array(channel_height, n), land_width=land,
            t_jacket=_as_array(t_jacket if t_jacket is not None else t_hot, n),
            channel_count=int(channel_count), helix_turns=float(helix_turns),
            Rt=float(contour.get("Rt", np.min(r))),
        )

    @classmethod
    def tapered(
        cls, contour: dict, *, channel_count: int,
        throat: dict, exit: dict, chamber: dict | None = None,
        helix_turns: float = 0.0,
    ) -> "RegenWallProfile":
        """Piecewise-linear chamber→throat→exit geometry.

        ``throat`` and ``exit`` contain ``t_hot``, ``channel_width`` and
        ``channel_height`` (optionally ``land_width`` / ``t_jacket``).
        Downstream stations interpolate throat→exit.  Upstream stations
        remain at the throat values unless a distinct ``chamber`` mapping
        is supplied, in which case they interpolate chamber→throat.

        This split is intentional: using absolute distance from the throat
        would incorrectly apply the divergent exit taper to the convergent
        chamber side.
        """
        x = np.asarray(contour["x"], dtype=float)
        r = np.asarray(contour["y"], dtype=float)
        ti = int(np.argmin(r))                 # throat station
        x_t = float(x[ti])
        x_lo = float(np.min(x))
        x_hi = float(np.max(x))
        f_up = np.clip((x_t - x) / max(x_t - x_lo, 1e-12), 0.0, 1.0)
        f_down = np.clip((x - x_t) / max(x_hi - x_t, 1e-12), 0.0, 1.0)

        def value(mapping, key, fallback):
            return float(mapping.get(key, fallback)) if mapping is not None else float(fallback)

        def piecewise(key, default_t=None, default_e=None):
            t_val = float(throat.get(key, default_t))
            e_val = value(exit, key, default_e if default_e is not None else t_val)
            c_val = value(chamber, key, t_val)
            return np.where(
                x < x_t,
                t_val + (c_val - t_val) * f_up,
                t_val + (e_val - t_val) * f_down,
            )

        t_hot = piecewise("t_hot")
        w = piecewise("channel_width")
        h = piecewise("channel_height")
        if ("land_width" in throat or "land_width" in exit
                or (chamber is not None and "land_width" in chamber)):
            throat_pitch = (
                2.0 * np.pi * max(float(r[ti]), 1e-9)
                / max(int(channel_count), 1)
            )
            # A partial mapping is valid.  If only the chamber or exit
            # specifies a land, anchor the throat at its geometric pitch
            # instead of attempting float(None).
            land_t = throat.get(
                "land_width",
                max(throat_pitch - float(throat["channel_width"]), 0.0),
            )
            land = piecewise("land_width", default_t=land_t,
                             default_e=exit.get("land_width", land_t))
        else:
            pitch = 2.0 * np.pi * np.maximum(r, 1e-9) / max(int(channel_count), 1)
            land = np.maximum(pitch - w, 0.0)
        if ("t_jacket" in throat or "t_jacket" in exit
                or (chamber is not None and "t_jacket" in chamber)):
            jacket_t = throat.get("t_jacket", throat["t_hot"])
            tj = piecewise("t_jacket", default_t=jacket_t,
                           default_e=exit.get("t_jacket", jacket_t))
        else:
            tj = t_hot.copy()
        return cls(
            x=x, r_inner=r, t_hot=t_hot, channel_width=w, channel_height=h,
            land_width=land, t_jacket=tj, channel_count=int(channel_count),
            helix_turns=float(helix_turns), Rt=float(contour.get("Rt", np.min(r))),
        )

    # ---- derived geometry ----------------------------------------------
    @property
    def n_stations(self) -> int:
        return len(self.x)

    @property
    def is_uniform(self) -> bool:
        """True when every geometry array is constant along the contour."""
        return all(
            float(np.ptp(a)) <= 1e-15 for a in
            (self.t_hot, self.channel_width, self.channel_height))

    @property
    def channel_mid_radius(self) -> np.ndarray:
        """Coolant channel centerline radius r_mid = r_inner + t_hot + h/2."""
        return self.r_inner + self.t_hot + 0.5 * self.channel_height

    @property
    def flow_area(self) -> np.ndarray:
        """Per-channel cross-section area w·h [m²]."""
        return np.maximum(self.channel_width * self.channel_height, 1e-12)

    @property
    def hydraulic_diameter(self) -> np.ndarray:
        """Per-station rectangular hydraulic diameter 2wh/(w+h) [m]."""
        w, h = self.channel_width, self.channel_height
        return 2.0 * w * h / np.maximum(w + h, 1e-12)

    def axial_length(self) -> float:
        return float(np.max(self.x) - np.min(self.x))

    def coolant_velocity(self, coolant_mass_flow: float, density: float) -> np.ndarray:
        """Per-station bulk coolant velocity V(x) = (ṁ/N)/(ρ·A) [m/s].

        Reveals the SP-125 design intent: shrink the passage area where
        the wall needs more cooling to hold the velocity up."""
        mdot_ch = float(coolant_mass_flow) / max(self.channel_count, 1)
        return mdot_ch / (max(float(density), 1e-9) * self.flow_area)

    def passage_lengths(self) -> tuple[np.ndarray, np.ndarray]:
        """``(dl, ds_meridian)`` per station — see :func:`helix_passage_lengths`."""
        return helix_passage_lengths(
            self.x, self.r_inner, helix_turns=self.helix_turns,
            t_wall=self.t_hot, channel_height=self.channel_height)

    def passage_length(self) -> float:
        """Total per-channel coolant path length [m] (helical if wound)."""
        dl, _ = self.passage_lengths()
        return float(np.sum(dl))

    def meridional_length(self) -> float:
        """Total axial/meridional wall length [m] (the helix=0 path)."""
        _, ds = self.passage_lengths()
        return float(np.sum(ds))

    def passage_length_factor(self) -> float:
        """Helical path length ÷ meridional length (≥ 1; 1 when axial).

        The factor by which the helix multiplies the friction pressure
        drop (Δp ∝ L, SP-125 eq. 4-32)."""
        dl, ds = self.passage_lengths()
        return float(np.sum(dl) / max(np.sum(ds), 1e-12))

    def channels_fit(self) -> dict:
        """Do the N channels + lands fit the liner outer circumference at
        every station?  Channels sit on the liner OD (r_inner + t_hot)."""
        circ = 2.0 * np.pi * (self.r_inner + self.t_hot)
        needed = self.channel_count * (self.channel_width + np.maximum(self.land_width, 0.0))
        clearance = circ - needed
        i = int(np.argmin(clearance))
        return {
            "fits": bool(np.min(clearance) >= 0.0),
            "tightest_station": i,
            "tightest_radius": float(self.r_inner[i]),
            "min_clearance_mm": float(np.min(clearance) * 1e3),
        }

    # ---- bridge to the cooling solver ----------------------------------
    def cooling(self, heat_flux: dict, contour: dict, material: Any,
                cooling_spec: Any, prop: Any, Pc: float, **kwargs) -> dict:
        """Run the coupled 1-D regen analysis on this profile (station-wise
        t/w/h + helical Δp).  ``cooling_spec`` supplies the coolant
        (mass flow, coolant name, limits); the geometry comes from
        ``self``.  Thin wrapper around
        :func:`raosim.physics.regenerative_cooling_analysis`."""
        from raosim.physics import regenerative_cooling_analysis
        return regenerative_cooling_analysis(
            heat_flux, contour, cooling_spec, material, None, prop, Pc,
            wall_profile=self, **kwargs)
