"""
raosim.mdo.grid — fixed-topology chamber/nozzle station grid (plan §10, §6.2).

A differentiable analytic stand-in for the thrust-chamber inner contour,
parameterized by the solved throat radius and the design expansion ratio, so
JAX array shapes never change (plan §6.2: "fixed-topology normalized axial
grid") while the geometry itself moves with the state during the Newton solve.

Fidelity note (plan §2.1 "two contour fidelities"): this is the *bulk-sweep*
geometry.  It exists to give the 4a cooling march station areas, radii, and
Mach numbers; the Rao/TOP contour remains the performance/export geometry, and
the implicit variational wall stays on the existence-boundary path.  Station
count and the subsonic/supersonic split are STATIC (Python ints) — only
coordinates move.

Chamber convention (2026-07-31)
-------------------------------
The upstream half now uses the **same four-section construction** as
:func:`raosim.chamber_geometry.chamber_contour`: cylindrical barrel → shoulder
fillet → straight convergent cone → upstream throat arc, with the barrel length
solved so that the injector-face-to-throat volume equals ``L* · A_t``.

This replaces the earlier ``chamber_length = L*/CR`` barrel plus a cosine
blend.  That approximation assumed the *barrel alone* held the whole chamber
volume, but NASA SP-125 is explicit that it does not (Huzel & Huang, NASA
SP-125, 1971; ``propulsion_texts/19710019929.pdf``, ch. IV, printed p. 88):

    "In design practice, it has been arbitrarily defined that the combustion
    chamber volume includes the space between injector face I-I and the nozzle
    throat plane II-II."

Because the convergent section carries part of ``L*·A_t``, ``L*/CR`` makes the
barrel too long.  At the 13 kN baseline it over-stated the chamber by 20.1 mm
and the wetted area by 11.7 % — an 11.7 % error in total heat load, not only in
mass, and it made every MDO-vs-traditional cooling comparison meaningless.
SP-125's own approximate chamber (eq. 4-6, printed p. 89) is barrel + straight
cone; the shoulder fillet and upstream throat arc used here are the
repository's existing refinement of that, and matching them is what makes the
two pipelines describe one chamber.

Region layout (n = n_chamber + n_conv + n_div + 1 stations, throat shared):

    [0 .. n_chamber)                cylindrical barrel, A/At = contraction_ratio
    [n_chamber .. n_chamber+n_conv] shoulder fillet + convergent cone + upstream
                                    throat arc, resampled by arc length
    (throat at index n_chamber + n_conv)
    (throat .. n-1]                 throat downstream arc + Rao/TOP quadratic
                                    Bezier to the exit
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- x64
import jax
import jax.numpy as jnp

from raosim.jax.primitives import mach_from_area_ratio
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Differentiable Rao/TOP wall-angle charts                                     #
# --------------------------------------------------------------------------- #
# The Rao thrust-optimised-parabolic contour is defined by the initial and exit
# wall angles (θ_n, θ_e) read off the classical Rao/NASA charts as functions of
# (ε, L%).  ``raosim.nozzle_geometry`` holds those tables and interpolates them
# with SciPy — correct but not differentiable.  Use the *same piecewise-linear*
# interpolation in pure JAX so MDO geometry matches the repository's trusted
# ``lookup_angles`` oracle exactly between knots.  It is piecewise
# differentiable (not C¹) and chart-domain constraints below keep clipping
# from being mistaken for admissible extrapolation.
def _load_chart_tables():
    from raosim.nozzle_geometry import (
        _EPSILON_VALS, _LPCT_VALS, _THETA_N_TABLE, _THETA_E_TABLE,
    )
    eps_g = jnp.asarray(_EPSILON_VALS, dtype=jnp.float64)
    lp_g = jnp.asarray(_LPCT_VALS, dtype=jnp.float64)
    return (
        eps_g, lp_g, jnp.asarray(_THETA_N_TABLE, dtype=jnp.float64),
        jnp.asarray(_THETA_E_TABLE, dtype=jnp.float64),
    )


_EPS_GRID, _LPCT_GRID, _THETA_N_TABLE, _THETA_E_TABLE = _load_chart_tables()


def _bilinear_chart(table: Array, eps: Array, length_pct: Array) -> Array:
    """JAX counterpart of SciPy ``RegularGridInterpolator(method='linear')``.

    Queries are clipped only to preserve finite trial points; the explicit
    domain constraint makes every clipped point infeasible in the NLP.
    """
    ec = jnp.clip(eps, _EPS_GRID[0], _EPS_GRID[-1])
    lc = jnp.clip(length_pct, _LPCT_GRID[0], _LPCT_GRID[-1])
    i = jnp.clip(jnp.searchsorted(_EPS_GRID, ec, side="right") - 1,
                 0, _EPS_GRID.size - 2)
    j = jnp.clip(jnp.searchsorted(_LPCT_GRID, lc, side="right") - 1,
                 0, _LPCT_GRID.size - 2)
    i = jax.lax.stop_gradient(i)
    j = jax.lax.stop_gradient(j)
    u = (ec - _EPS_GRID[i]) / (_EPS_GRID[i + 1] - _EPS_GRID[i])
    v = (lc - _LPCT_GRID[j]) / (_LPCT_GRID[j + 1] - _LPCT_GRID[j])
    return ((1.0 - u) * (1.0 - v) * table[i, j]
            + u * (1.0 - v) * table[i + 1, j]
            + (1.0 - u) * v * table[i, j + 1]
            + u * v * table[i + 1, j + 1])


def rao_wall_angles(eps: Array, length_pct: Array) -> tuple[Array, Array]:
    """Differentiable (θ_n, θ_e) in **radians** from the Rao/NASA charts."""
    tn = _bilinear_chart(_THETA_N_TABLE, eps, length_pct)
    te = _bilinear_chart(_THETA_E_TABLE, eps, length_pct)
    return jnp.deg2rad(tn), jnp.deg2rad(te)


def rao_chart_domain_violation(eps: Array, length_pct: Array) -> Array:
    """Positive distances outside the tabulated Rao/TOP chart box.

    The angle surfaces clamp only so an infeasible trial point remains finite
    for an NLP iteration.  This companion is the admissibility constraint that
    prevents a clipped/extrapolated chart angle from being accepted as a
    design.
    """
    return jnp.stack([
        _EPS_GRID[0] - eps, eps - _EPS_GRID[-1],
        _LPCT_GRID[0] - length_pct, length_pct - _LPCT_GRID[-1],
    ])


# Dense sampling used only for the chamber-volume closure.  These match the
# ``n_pts_*`` defaults of :func:`raosim.chamber_geometry.chamber_contour` so the
# two pipelines integrate the identical polyline and the solved barrel length
# agrees to machine precision rather than to a discretisation tolerance.
_VOL_N_SHOULDER = 40
_VOL_N_CONVERGENT = 80
_VOL_N_UPSTREAM_ARC = 80


def _frustum_volume(x: Array, r: Array) -> Array:
    """Exact revolved volume of a meridional polyline (conical frusta).

    Mirrors :func:`raosim.chamber_geometry.enclosed_volume`:
    ``dV = pi/3 * dx * (r0^2 + r0 r1 + r1^2)``.
    """

    dx = jnp.diff(x)
    r0, r1 = r[:-1], r[1:]
    return (jnp.pi / 3.0) * jnp.sum(dx * (r0 ** 2 + r0 * r1 + r1 ** 2))


def shoulder_radius_factor(mission: MissionSpec) -> Array:
    """``Rs/Rt`` matching :func:`raosim.chamber_geometry.auto_shoulder_factor`.

    The cap is closed-form — the largest fillet that still leaves a
    non-negative straight convergent run between the shoulder and the upstream
    throat arc:

        Rs_max/Rt = (sqrt(CR) - 1 - (Ru/Rt)(1 - cos a)) / (1 - cos a)

    which is independent of ``Rt``.  ``shoulder_fill_fraction`` (0.8) keeps a
    ~20 % straight cone, as the traditional path does.
    """

    alpha = jnp.deg2rad(mission.converging_half_angle_deg)
    one_minus_cos = 1.0 - jnp.cos(alpha)
    cap = (
        jnp.sqrt(mission.contraction_ratio)
        - 1.0
        - mission.throat_ru_factor * one_minus_cos
    ) / one_minus_cos
    return mission.shoulder_fill_fraction * cap


def _upstream_sections(
    Rt: Array, mission: MissionSpec, *, n_shoulder: int, n_convergent: int,
    n_arc: int,
) -> tuple[Array, Array, Array]:
    """Shoulder fillet, straight convergent and upstream throat arc.

    Returns ``(x, r, x_shoulder_start)`` for everything between the end of the
    cylindrical barrel and the throat.  Identical construction to
    :func:`raosim.chamber_geometry.chamber_contour`, in pure ``jnp``.
    """

    alpha = jnp.deg2rad(mission.converging_half_angle_deg)
    Rc = Rt * jnp.sqrt(mission.contraction_ratio)
    Ru = mission.throat_ru_factor * Rt
    Rs = shoulder_radius_factor(mission) * Rt

    x_arc_entry = -Ru * jnp.sin(alpha)
    y_arc_entry = Rt + Ru * (1.0 - jnp.cos(alpha))
    y_shoulder_end = Rc - Rs * (1.0 - jnp.cos(alpha))
    straight_length = (y_shoulder_end - y_arc_entry) / jnp.tan(alpha)
    x_shoulder_end = x_arc_entry - straight_length
    x_shoulder_start = x_shoulder_end - Rs * jnp.sin(alpha)

    # Shoulder fillet: centre (x_shoulder_start, Rc - Rs), swept pi/2 -> pi/2-a.
    t_sh = jnp.pi / 2.0 - alpha * jnp.linspace(0.0, 1.0, n_shoulder)
    x_sh = x_shoulder_start + Rs * jnp.cos(t_sh)
    y_sh = (Rc - Rs) + Rs * jnp.sin(t_sh)

    # Straight convergent cone at half-angle alpha.
    x_cv = x_shoulder_end + jnp.linspace(0.0, 1.0, n_convergent) * (
        x_arc_entry - x_shoulder_end
    )
    y_cv = y_shoulder_end - (x_cv - x_shoulder_end) * jnp.tan(alpha)

    u_ar = jnp.linspace(0.0, 1.0, n_arc)
    t_ar = -(jnp.pi / 2.0 + alpha) + alpha * u_ar
    x_ar = Ru * jnp.cos(t_ar)
    y_ar = Rt + Ru + Ru * jnp.sin(t_ar)

    # Drop the duplicated junction nodes, as ``_concat_sections`` does.
    x = jnp.concatenate([x_sh, x_cv[1:], x_ar[1:]])
    y = jnp.concatenate([y_sh, y_cv[1:], y_ar[1:]])
    return x, y, x_shoulder_start


def chamber_barrel_length(Rt: Array, mission: MissionSpec) -> Array:
    """Cylindrical barrel length that closes ``V_injector->throat = L* A_t``.

    No root solve is needed: the shoulder, convergent and upstream-arc volumes
    do not depend on the barrel length, and the barrel's own revolved volume is
    exactly ``pi Rc^2 Lc`` for a constant radius (every frustum is a cylinder),
    so the closure is linear:

        Lc = (L* A_t - V_shoulder+convergent+arc) / (pi Rc^2)

    The traditional path bisects to the same answer because it integrates the
    same polyline; solving in closed form keeps this differentiable and cheap
    enough to rebuild inside every Newton iterate.

    A negative result means the fixed upstream sections already enclose more
    than ``L* A_t``.  It is returned as-is rather than clamped: clamping would
    silently manufacture chamber volume the design does not have.  The caller
    exposes it as the ``chamber_volume_margin`` feasibility constraint.
    """

    Rc = Rt * jnp.sqrt(mission.contraction_ratio)
    At = jnp.pi * Rt ** 2
    x_up, y_up, _ = _upstream_sections(
        Rt, mission,
        n_shoulder=_VOL_N_SHOULDER,
        n_convergent=_VOL_N_CONVERGENT,
        n_arc=_VOL_N_UPSTREAM_ARC,
    )
    v_fixed = _frustum_volume(x_up, y_up)
    return (mission.l_star * At - v_fixed) / (jnp.pi * Rc ** 2)


def chamber_volume(Rt: Array, mission: MissionSpec) -> Array:
    """Revolved volume from the injector face to the throat plane.

    This is the SP-125 chamber volume (printed p. 88: "the combustion chamber
    volume includes the space between injector face I-I and the nozzle throat
    plane II-II"), computed on the dense construction so it is independent of
    the coarse station grid.  It equals ``L*·A_t`` by construction whenever
    :func:`chamber_barrel_length` is non-negative; reporting it lets the
    snapshot compare against the traditional pipeline's measured volume.
    """

    Rc = Rt * jnp.sqrt(mission.contraction_ratio)
    Lc = chamber_barrel_length(Rt, mission)
    x_up, y_up, x_shoulder_start = _upstream_sections(
        Rt, mission,
        n_shoulder=_VOL_N_SHOULDER,
        n_convergent=_VOL_N_CONVERGENT,
        n_arc=_VOL_N_UPSTREAM_ARC,
    )
    barrel = jnp.pi * Rc ** 2 * Lc
    return barrel + _frustum_volume(x_up, y_up)


def wetted_area(x: Array, r: Array) -> Array:
    """Gas-side wetted area of a meridional polyline.

    Trapezoidal nodal weights on meridional arc length, identical to
    :func:`raosim.regen_profile._nodal_weights_from_segments`, so the area that
    scales the heat load and the area that scales the wall mass are the same
    number.
    """

    seg = jnp.hypot(jnp.diff(x), jnp.diff(r))
    w = jnp.concatenate([
        0.5 * seg[:1],
        0.5 * (seg[:-1] + seg[1:]),
        0.5 * seg[-1:],
    ])
    return jnp.sum(2.0 * jnp.pi * r * w)


def _resample_by_arclength(x: Array, r: Array, n: int) -> tuple[Array, Array]:
    """Resample a meridional polyline at ``n`` equal arc-length stations.

    Equal arc segments give the cooling march equal heat-area weights, and keep
    the coarse station grid from cutting corners on the fillet and throat arcs.
    """

    seg = jnp.hypot(jnp.diff(x), jnp.diff(r))
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    s_new = jnp.linspace(0.0, s[-1], n)
    return jnp.interp(s_new, s, x), jnp.interp(s_new, s, r)


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
    chart_domain_violation: Array  # (4,), positive outside the chart box
    wall_monotonic_margin: Array   # min divergent dr >= 0 for an admissible wall


def build_station_grid(Rt: Array, eps: Array, mission: MissionSpec,
                       topo: GridTopology = GridTopology(), *,
                       gamma: Array | None = None) -> StationGrid:
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
    # Classical bell-length convention: percent of the 15° conical length.
    L_div = (mission.length_pct / 100.0) * (Re - Rt) / jnp.tan(jnp.deg2rad(15.0))

    # --- upstream half: SHARED convention with chamber_geometry ------------- #
    # Barrel length closes the SP-125 chamber volume (injector face to throat
    # plane) at L*·A_t.  See the module docstring: the previous L*/CR barrel
    # ignored the convergent section's share of that volume.
    L_ch = chamber_barrel_length(Rt, mission)
    x_up_dense, r_up_dense, x_shoulder_start = _upstream_sections(
        Rt, mission,
        n_shoulder=_VOL_N_SHOULDER,
        n_convergent=_VOL_N_CONVERGENT,
        n_arc=_VOL_N_UPSTREAM_ARC,
    )

    # --- chamber barrel ----------------------------------------------------- #
    u_ch = jnp.linspace(0.0, 1.0, topo.n_chamber, endpoint=False)
    x_ch = x_shoulder_start - L_ch * (1.0 - u_ch)
    r_ch = jnp.full((topo.n_chamber,), 1.0) * r_c

    # --- shoulder + convergent + upstream throat arc ------------------------ #
    # Resampled from the dense construction by arc length so the coarse station
    # grid does not chord-cut the fillet and throat arcs.  The throat node
    # itself belongs to the divergent block, so it is dropped here.
    x_cv, r_cv = _resample_by_arclength(
        x_up_dense, r_up_dense, topo.n_conv + 1
    )
    x_cv, r_cv = x_cv[:-1], r_cv[:-1]

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
    # The thermo surface's active gamma must drive the cooling-grid Mach field;
    # using the fallback MissionSpec gamma here silently decouples a CEA table.
    gamma = mission.gamma if gamma is None else gamma
    ar_sub = area_ratio[: topo.throat_index]
    ar_sup = area_ratio[topo.throat_index:]
    M_sub = mach_from_area_ratio(ar_sub, gamma, supersonic=False)
    M_sup = mach_from_area_ratio(ar_sup, gamma, supersonic=True)
    mach = jnp.concatenate([M_sub, M_sup])

    dseg = jnp.sqrt(jnp.diff(x) ** 2 + jnp.diff(r) ** 2)
    # The divergent wall must increase monotonically from the throat.  Report
    # rather than repair it: clipping coordinates would hide an invalid nozzle
    # and corrupt gradients.
    wall_monotonic_margin = jnp.min(jnp.diff(r[topo.throat_index:]))
    return StationGrid(
        x=x, r=r, area_ratio=area_ratio, mach=mach, dseg=dseg,
        throat_index=topo.throat_index,
        chart_domain_violation=rao_chart_domain_violation(eps, mission.length_pct),
        wall_monotonic_margin=wall_monotonic_margin,
    )
