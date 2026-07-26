"""
raosim.mdo.grid — fixed-topology chamber/nozzle station grid (plan §10, §6.2).

A differentiable analytic stand-in for the thrust-chamber inner contour,
parameterized by the solved throat radius and the design expansion ratio, so
JAX array shapes never change (plan §6.2: "fixed-topology normalized axial
grid") while the geometry itself moves with the state during the Newton solve.

Fidelity note (plan §2.1 "two contour fidelities"): this is the *bulk-sweep*
geometry — a smooth barrel + cosine-blend converging section + a
monotone-slope divergent bell surrogate whose length matches the classical
``length_pct``-of-15°-cone convention (SP-8120 / `nozzle_geometry.py`).  It
exists to give the 4a cooling march station areas, radii, and Mach numbers;
the Rao/TOP contour remains the performance/export geometry, and the implicit
variational wall stays on the existence-boundary path.  Station count and the
subsonic/supersonic split are STATIC (Python ints) — only coordinates move.

Region layout (n = n_chamber + n_conv + n_div + 1 stations, throat shared):

    [0 .. n_chamber)                cylindrical barrel, A/At = contraction_ratio
    [n_chamber .. n_chamber+n_conv] cosine blend CR -> 1  (C¹ at both ends)
    (throat at index n_chamber + n_conv)
    (throat .. n-1]                 bell surrogate r(u) = Rt + (Re-Rt)·u(2-u)
                                    (steep at the throat, tangent -> 0 at exit)
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- x64
import jax.numpy as jnp

import numpy as _np

from raosim.jax.primitives import mach_from_area_ratio
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Differentiable Rao/TOP wall-angle charts                                     #
# --------------------------------------------------------------------------- #
# The Rao thrust-optimised-parabolic contour is defined by the initial and exit
# wall angles (θ_n, θ_e) read off the classical Rao/NASA charts as functions of
# (ε, L%).  ``raosim.nozzle_geometry`` holds those tables and interpolates them
# with SciPy — correct but not differentiable.  Here the same tables are wrapped
# in the C¹ tensor-Hermite surfaces built for the CEA properties, so θ_n and θ_e
# (and hence the whole contour) carry exact derivatives w.r.t. ε and L%.
def _build_angle_surfaces():
    from raosim.nozzle_geometry import (
        _EPSILON_VALS, _LPCT_VALS, _THETA_N_TABLE, _THETA_E_TABLE,
    )
    from raosim.mdo.properties import PropertySurface2D
    eps_g = _np.asarray(_EPSILON_VALS, dtype=float)
    lp_g = _np.asarray(_LPCT_VALS, dtype=float)
    return (
        PropertySurface2D.build(eps_g, lp_g,
                                _np.asarray(_THETA_N_TABLE, float), name="theta_n"),
        PropertySurface2D.build(eps_g, lp_g,
                                _np.asarray(_THETA_E_TABLE, float), name="theta_e"),
    )


_THETA_N_SURF, _THETA_E_SURF = _build_angle_surfaces()


def rao_wall_angles(eps: Array, length_pct: Array) -> tuple[Array, Array]:
    """Differentiable (θ_n, θ_e) in **radians** from the Rao/NASA charts."""
    tn = _THETA_N_SURF(eps, length_pct)
    te = _THETA_E_SURF(eps, length_pct)
    return jnp.deg2rad(tn), jnp.deg2rad(te)


@dataclass(frozen=True)
class GridTopology:
    """Static station counts (Python ints — never traced)."""

    n_chamber: int = 4
    n_conv: int = 6
    n_div: int = 13

    @property
    def n(self) -> int:
        return self.n_chamber + self.n_conv + self.n_div + 1

    @property
    def throat_index(self) -> int:
        return self.n_chamber + self.n_conv


@dataclass(frozen=True)
class StationGrid:
    """Per-station geometry + inviscid state (all shapes static)."""

    x: Array            # (n,) axial coordinate [m], 0 at throat
    r: Array            # (n,) hot-gas wall radius [m]
    area_ratio: Array   # (n,) A/At >= 1
    mach: Array         # (n,) subsonic upstream, 1 at throat, supersonic down
    dseg: Array         # (n-1,) wall arc length between stations [m]
    throat_index: int   # static


def build_station_grid(Rt: Array, eps: Array, mission: MissionSpec,
                       topo: GridTopology = GridTopology()) -> StationGrid:
    """Analytic grid from the (possibly still-being-solved) Rt and design ε.

    Pure jnp in (Rt, eps) — safe to rebuild inside the residual at every
    Newton iterate (geometry follows the state, plan §12.3).
    """
    Rt = jnp.asarray(Rt, dtype=jnp.float64)
    eps = jnp.asarray(eps, dtype=jnp.float64)
    CR = mission.contraction_ratio
    r_c = Rt * jnp.sqrt(CR)
    Re = Rt * jnp.sqrt(eps)

    # --- axial extents ------------------------------------------------------ #
    L_conv = (r_c - Rt) / jnp.tan(jnp.deg2rad(mission.converging_half_angle_deg))
    L_ch = mission.chamber_length
    # Classical bell-length convention: percent of the 15° conical length.
    L_div = (mission.length_pct / 100.0) * (Re - Rt) / jnp.tan(jnp.deg2rad(15.0))

    # --- chamber barrel ----------------------------------------------------- #
    u_ch = jnp.linspace(0.0, 1.0, topo.n_chamber, endpoint=False)
    x_ch = -L_conv - L_ch * (1.0 - u_ch)
    r_ch = jnp.full((topo.n_chamber,), 1.0) * r_c

    # --- converging cosine blend (C¹: dr/dx = 0 at barrel and throat) ------- #
    u_cv = jnp.linspace(0.0, 1.0, topo.n_conv + 1)[:-1]   # exclude throat here
    x_cv = -L_conv * (1.0 - u_cv)
    r_cv = Rt + (r_c - Rt) * (0.5 + 0.5 * jnp.cos(jnp.pi * u_cv))

    # --- divergent: the actual Rao/TOP contour ------------------------------ #
    # Throat downstream arc of radius R_d turning from the throat (vertical
    # tangent) to the initial wall angle θ_n, then the thrust-optimised
    # **quadratic Bézier** to the exit at θ_e — the classical Rao parabolic
    # approximation (the same construction as
    # ``nozzle_geometry.bell_nozzle_contour``, but differentiable).
    theta_n, theta_e = rao_wall_angles(eps, mission.length_pct)
    Rd = mission.throat_rd_factor * Rt

    n_arc = max(topo.n_div // 3, 2)
    n_bez = topo.n_div + 1 - n_arc

    # throat arc: centre at (0, Rt+Rd), swept from −90° to θ_n−90°
    t_arc = jnp.linspace(0.0, 1.0, n_arc + 1)[:-1] * theta_n
    x_arc = Rd * jnp.sin(t_arc)
    r_arc = (Rt + Rd) - Rd * jnp.cos(t_arc)

    # Bézier from N (arc end, slope tanθ_n) to E (exit, slope tanθ_e)
    Nx = Rd * jnp.sin(theta_n)
    Ny = (Rt + Rd) - Rd * jnp.cos(theta_n)
    Ex = L_div
    Ey = Re
    m1 = jnp.tan(theta_n)
    m2 = jnp.tan(theta_e)
    # control point = intersection of the two tangent lines
    Qx = (Ey - m2 * Ex - Ny + m1 * Nx) / (m1 - m2)
    Qy = Ny + m1 * (Qx - Nx)
    tb = jnp.linspace(0.0, 1.0, n_bez)
    om = 1.0 - tb
    x_bez = om * om * Nx + 2.0 * om * tb * Qx + tb * tb * Ex
    r_bez = om * om * Ny + 2.0 * om * tb * Qy + tb * tb * Ey

    x_dv = jnp.concatenate([x_arc, x_bez])
    r_dv = jnp.concatenate([r_arc, r_bez])

    x = jnp.concatenate([x_ch, x_cv, x_dv])
    r = jnp.concatenate([r_ch, r_cv, r_dv])
    area_ratio = jnp.maximum((r / Rt) ** 2, 1.0)

    # --- inviscid Mach (static branch split at the throat index) ------------ #
    gamma = mission.gamma  # grid Mach uses the mission constant; the residual
    # re-evaluates gas properties itself, so a Phase-2 surface γ enters there.
    ar_sub = area_ratio[: topo.throat_index]
    ar_sup = area_ratio[topo.throat_index:]
    M_sub = mach_from_area_ratio(ar_sub, gamma, supersonic=False)
    M_sup = mach_from_area_ratio(ar_sup, gamma, supersonic=True)
    mach = jnp.concatenate([M_sub, M_sup])

    dseg = jnp.sqrt(jnp.diff(x) ** 2 + jnp.diff(r) ** 2)
    return StationGrid(x=x, r=r, area_ratio=area_ratio, mach=mach,
                       dseg=dseg, throat_index=topo.throat_index)
