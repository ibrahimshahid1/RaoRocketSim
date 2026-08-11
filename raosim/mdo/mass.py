"""mdo/mass.py — differentiable thrust-chamber hardware mass.

This is the JAX mirror of :mod:`raosim.mass_ledger`.  Both integrate the same
solid of revolution over the same wall description; this one does it in pure
``jnp`` on the fixed MDO station grid so ``d(mass)/d(design)`` exists and the
optimizer can trade chamber structure against feed-system mass instead of
minimising only the electric package and calling the result engine mass.

Physical basis
--------------
The equations below are geometric shell-volume calculations (Pappus's centroid
theorem), not a thrust-chamber mass correlation.  NASA SP-125 (Huzel & Huang,
*Design of Liquid Propellant Rocket Engines*, 1967;
``propulsion_texts/19710019929.pdf``, Ch. VIII, “Design of Propellant Tanks,”
§8.1.2.3, source-PDF p. 348 / printed p. 339) uses the same surface-times-
thickness relation for a cylindrical **tank** section:

    W_c = 2 π a l_c t_c ρ                                  (eq. 8-32, p. 339)

with ``a`` the *nominal* (mid-surface) radius.  That tank equation corroborates
the geometry but does not prescribe thrust-chamber thickness or mass.  Applying
the geometric relation per station over the meridional arc gives, for a
regeneratively cooled wall,

    A_liner    = 2 π (r + t_w/2) t_w
    A_land     = π (r_o² − r_i²) · b/(b + w)
    A_closeout = 2 π (r_o + t_j/2) t_j

with ``r_i = r + t_w``, ``r_o = r_i + h``.  The land term is written as an area
*fraction* of the channel annulus rather than ``N·b·h`` because the fraction is
invariant to helical stretch, is unconditionally non-negative, and stays smooth
in the design variables — all three matter inside an SLSQP loop.

What the closeout is and is not
-------------------------------
NASA SP-8087 (*Liquid Rocket Engine Fluid-Cooled Combustion Chambers*, 1973;
``propulsion_texts/19730022965.pdf``) §2.1.3 states the three structural jobs of
chamber reinforcement — hoop support about the combustion chamber, support at
the throat against bending and buckling, and hoop support about the expansion
nozzle against collapse from hoop compression when the nozzle runs overexpanded
at sea level — and quotes the design factors of safety in use: yield 1.0 to
1.32, ultimate 1.3 to 1.8.  **This module does not size the closeout against
those loads.**  It integrates a closeout whose thickness is set by
``MissionSpec.closeout_thickness_ratio``, and the mass it returns is therefore
conditional on that assumption.  Making the jacket thickness a solved structural
variable is tracked as future work; until then the ratio is reported in the
state's provenance so a reader can see the assumption rather than infer it.

Differentiable invalid-domain behavior
--------------------------------------
On physically valid geometry the equations are smooth in ``(Rt, eps, t_wall,
channel_width, channel_height)``.  An infeasible optimizer probe can make a
computed land width negative.  The shared JAX volume kernel uses the absolute
land magnitude only as a nonnegative continuation outside that domain, while
the independent ``land_min`` constraint rejects the point.  This introduces a
kink only at the invalid/valid boundary and prevents negative material from
artificially lowering the objective.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from raosim.mdo.grid import StationGrid
from raosim.mdo.schema import MissionSpec
from raosim.regen_volumes import integrate_regen_volumes_jax

__all__ = [
    "ChamberMassBreakdown",
    "chamber_mass",
    "SP125_SHELL_MASS",
    "SP8087_STRUCTURAL",
]

SP125_SHELL_MASS = (
    "geometric shell volume (Pappus); corroborated, not prescribed for thrust "
    "chambers, by NASA SP-125 eq. 8-32 tank-shell mass"
)
SP8087_STRUCTURAL = "NASA SP-8087 sec. 2.1.3 (chamber reinforcement / FoS)"


class ChamberMassBreakdown(NamedTuple):
    """Differentiable thrust-chamber metal mass, split by load path."""

    liner: Array        # hot-gas wall [kg]
    lands: Array        # channel ribs between coolant passages [kg]
    closeout: Array     # structural jacket over the channels [kg]
    total: Array        # sum of the three [kg]
    land_area_fraction: Array   # (n,) diagnostic: b/(b+w) per station
    wetted_area: Array  # gas-side wetted area [m2], for cross-checks
    closeout_thickness: Array   # (n,) solved jacket thickness [m]
    closeout_thin_shell_margin: Array  # >= 0 keeps t/r within SP-125's 1/15


def _smooth_floor(value: Array, floor: Array, sharpness: Array) -> Array:
    """Conservative differentiable ``max(value, floor)``.

    ``logaddexp(k a, k b)/k`` is never below ``max(a, b)`` and is smooth
    everywhere, so a manufacturing floor that binds on some stations and not
    others does not hand the NLP an active-set kink.  Erring thick is the
    conservative direction for both strength and mass.
    """

    k = sharpness
    return jnp.logaddexp(k * value, k * floor) / k


def closeout_thickness(
    r_outer: Array, coolant_pressure: Array, mission: MissionSpec,
) -> Array:
    """Structural jacket thickness from the SP-125 outer-shell hoop screen.

    NASA SP-125, printed p. 109, on the coaxial-shell chamber: *"the outer
    shell is subjected only to the hoop stress induced by the coolant
    pressure"*.  The thin-shell requirement is therefore

        t_j = FoS * p_coolant * r_outer / sigma_yield

    applied per station, because SP-8087 sec. 2.1.3.1 records that *"the brazed
    jacket can be tapered for optimum strength and weight"* -- a tapered jacket
    is normal practice, not an optimisation nicety.  The result is floored at
    ``closeout_thickness_min`` through a smooth conservative max.

    Setting ``mission.closeout_sizing = "ratio"`` restores the legacy fixed
    ``closeout_thickness_ratio * t_wall`` jacket for back-comparison.
    """

    t_req = (
        mission.closeout_structural_fos
        * coolant_pressure
        * r_outer
        / mission.closeout_sigma_yield
    )
    t_min = jnp.asarray(mission.closeout_thickness_min, dtype=jnp.float64)
    # Sharpness set from the floor so the blend is tight relative to the
    # thickness scale rather than to an arbitrary absolute number.
    return _smooth_floor(t_req, t_min, 40.0 / t_min)


def _nodal_weights(dseg: Array) -> Array:
    """Trapezoidal nodal control lengths from (n-1,) segment lengths.

    Matches :func:`raosim.regen_profile._nodal_weights_from_segments` exactly:
    end nodes take half of their one adjacent segment, interior nodes half of
    each neighbour, so ``sum(weights) == sum(dseg)``.  Summing
    ``hypot(gradient(x), gradient(r))`` instead — as the legacy private helper
    ``raosim.thermal_design._wall_mass`` does — gives each end node a full
    segment and over-counts the meridian by one grid interval.
    """

    interior = 0.5 * (dseg[:-1] + dseg[1:])
    return jnp.concatenate([
        0.5 * dseg[:1],
        interior,
        0.5 * dseg[-1:],
    ])


def chamber_mass(
    grid: StationGrid,
    mission: MissionSpec,
    *,
    t_wall: Array,
    channel_width: Array,
    channel_height: Array,
    coolant_pressure: Array | None = None,
) -> ChamberMassBreakdown:
    """Integrate liner + land + closeout metal over the MDO station grid.

    Parameters
    ----------
    grid
        The station grid already built for the coupled solve, so the mass and
        the cooling march see one geometry.
    mission
        Supplies ``rho_wall``, ``closeout_thickness_ratio``, ``n_channels`` and
        the optional fixed ``land_width``.
    t_wall, channel_width, channel_height
        Design variables, traced.

    Returns
    -------
    ChamberMassBreakdown
        All entries are traced arrays; ``total`` is the differentiable scalar an
        objective or constraint can use.
    """

    t_w = jnp.asarray(t_wall, dtype=jnp.float64)
    w = jnp.asarray(channel_width, dtype=jnp.float64)
    h = jnp.asarray(channel_height, dtype=jnp.float64)
    rho = jnp.asarray(mission.rho_wall, dtype=jnp.float64)
    rho_close = jnp.asarray(
        mission.rho_wall if mission.rho_closeout is None
        else mission.rho_closeout,
        dtype=jnp.float64,
    )

    r = grid.r
    ds = _nodal_weights(grid.dseg)

    r_ch_in = r + t_w
    r_ch_out = r_ch_in + h

    # --- structural jacket -------------------------------------------------- #
    if mission.closeout_sizing == "hoop" and coolant_pressure is not None:
        t_j = closeout_thickness(r_ch_out, coolant_pressure, mission)
    elif mission.closeout_sizing == "hoop":
        raise ValueError(
            "closeout_sizing='hoop' needs the solved coolant pressure "
            "profile; pass coolant_pressure= from the cooling march"
        )
    else:
        t_j = t_w * jnp.asarray(
            mission.closeout_thickness_ratio, dtype=jnp.float64
        ) * jnp.ones_like(r)
    # SP-125 (printed p. 336): the membrane/thin-shell treatment is valid while
    # t/r <= ~1/15.  Reported as a margin, not clamped -- a jacket that needs
    # more than that is telling the designer the alloy or the pressure is
    # wrong, and silently thickening it would hide that.
    thin_shell_margin = mission.closeout_thin_shell_ratio_max - jnp.max(
        t_j / jnp.maximum(r_ch_out, 1.0e-12)
    )
    # Channel mid-radius sets the circumferential pitch, matching
    # RegenWallProfile.uniform's normal-offset construction.
    r_mid = r + t_w + 0.5 * h
    n_ch = jnp.asarray(max(int(mission.n_channels), 1), dtype=jnp.float64)
    pitch = 2.0 * jnp.pi * jnp.maximum(r_mid, 1.0e-9) / n_ch
    if mission.land_width is None:
        b = pitch - w
    else:
        b = jnp.full_like(pitch, float(mission.land_width))
    volumes = integrate_regen_volumes_jax(
        r_inner=r,
        dseg=grid.dseg,
        t_hot=t_w,
        channel_width=w,
        channel_height=h,
        land_width=b,
        t_jacket=t_j,
    )

    liner = rho * volumes.liner
    lands = rho * volumes.lands
    closeout = rho_close * volumes.closeout
    wetted = jnp.sum(2.0 * jnp.pi * r * ds)

    return ChamberMassBreakdown(
        liner=liner,
        lands=lands,
        closeout=closeout,
        total=liner + lands + closeout,
        land_area_fraction=volumes.land_area_fraction,
        wetted_area=wetted,
        closeout_thickness=t_j,
        closeout_thin_shell_margin=thin_shell_margin,
    )
