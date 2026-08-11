"""
raosim.mdo.injector — Phase 5: differentiable pintle-injector block.

Explicit algebra (no implicit state, so no IFT needed — just closed-form jnp
that jacfwd/jacrev flow through), mirroring the audited NumPy oracles in
``raosim.injector`` and ``raosim.movable_pintle``.  Physics (plan §6.3, all
corpus-anchored):

* orifice metering, incompressible (LOX and RP-1 both liquid at injection —
  ``injector._stream_mass_flux`` liquid branch): G = C_d·√(2 ρ Δp),
  A = ṁ/G, v_inj = G/ρ = C_d·√(2 Δp/ρ);
* total momentum ratio TMR = (ṁ_r v_r)/(ṁ_a v_a) with the radial stream = fuel
  through N slots and the axial stream = oxidiser through the annulus (repo
  convention ``PintleGeometrySpec.radial_stream="fuel"``; universally the pintle
  master knob — Son 2015/2017, Hwang 2022 s42405, Sakaki 2015 "TMR usually ≈1");
* spray half-angle from the radial/axial momentum vectors with the deflector
  tilt, θ = atan2(m_r cosδ, m_a + m_r sinδ) (``injector.py`` L3199-3201); at
  δ=0 this is the **leading-order kinematic** θ ≈ arctan(TMR).  *Fidelity note*:
  Escher found the bare arctan a poor fit and proposed θ ∝ TMR^½; Son 2015
  (s11630-015-0753-7) gives an exponential θ(TMR, We) correlation — those are
  the higher-fidelity swaps (like the 4a→4b thermal ladder), deferred;
* blockage factor BF = N·w_slot/(π D_pintle) (Freeberg 2019, Hwang 2022);
* **movable-pintle minimum area as TWO branches** (plan rule 8 / §6.3, Son 2017):
  the tip-opening-limited area A_tip = π[2 r_f L cosθ − L² sinθ cos²θ]
  (``movable_pintle.son_minimum_tip_area``, Son Eq. 1; r_f = D_p/2 − t_post) and
  the fixed center-gap-limited area A_cg = (π/4)(D_cg² − D_pr²)
  (``movable_pintle.center_gap_area``).  We size the opening L that meters the
  required radial area (closed-form inverse of Eq. 1) and expose the
  *consistency inequality* ``transition_margin`` = frac·A_cg − A_tip ≥ 0 — never
  a differentiable ``min()``; which branch governs is an outer-enumeration
  architecture choice;
* injector-drop screen: min(χ_f, χ_o) ≥ 0.2 as a stationless conservative
  design inequality.  The number is the upper endpoint of the 15--20% rule of
  thumb in Huzel & Huang, *Design of Liquid Propellant Rocket Engines*
  (1967), NASA SP-125, sec. 4.2, source-PDF p.137 / printed p.128.  Harrje &
  Reardon (eds.), *Liquid Propellant Rocket Combustion Instability* (1972),
  NASA SP-194, sec. 6.2.3.1, source-PDF pp.293--294 supports the qualitative
  stability coupling but warns that increasing one stream's drop alone can be
  destabilizing.  This row is therefore not a universal chug boundary or a
  substitute for coupled stability analysis.

Parity oracle: tests/test_mdo_injector.py checks these against the NumPy
``movable_pintle`` functions and the ``injector`` orifice/TMR forms to ~1e-9.
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Son 2017 movable-pintle geometry (jnp mirrors; θ is a static float)          #
# --------------------------------------------------------------------------- #
def son_tip_area(opening: Array, r_f: float | Array, tip_angle_deg: float) -> Array:
    """Son et al. 2017 Eq. (1), expanded stable form (θ→0 limit = 2π r_f L).

    Mirror of ``movable_pintle.son_minimum_tip_area`` with r_f = R_post − t_post
    supplied directly."""
    theta = jnp.deg2rad(tip_angle_deg)
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.pi * (2.0 * r_f * opening * c - opening * opening * s * c * c)


def opening_for_tip_area(area: Array, r_f: float | Array,
                         tip_angle_deg: float) -> Array:
    """Invert Son Eq. (1) on its monotone small-opening branch (closed form).

    Mirror of ``movable_pintle.opening_for_tip_area``.  θ is static, so the
    zero-angle branch is a host ``if`` (never traced)."""
    theta = jnp.deg2rad(tip_angle_deg)
    if abs(float(tip_angle_deg)) < 1e-12:
        return area / (2.0 * jnp.pi * r_f)
    s, c = jnp.sin(theta), jnp.cos(theta)
    disc = r_f * r_f - area * s / jnp.pi
    return (r_f - jnp.sqrt(disc)) / (s * c)


def center_gap_area(Dcg: float | Array, Dpr: float | Array) -> Array:
    """Fixed center-gap annulus (π/4)(Dcg² − Dpr²) — the center-gap-limited
    branch.  Mirror of ``movable_pintle.center_gap_area``."""
    return jnp.pi * (Dcg * Dcg - Dpr * Dpr) / 4.0


# --------------------------------------------------------------------------- #
# Injector readouts + constraint margins                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InjectorReadout:
    """Explicit differentiable outputs of the pintle-injector block."""

    dp_fuel: Array            # radial (fuel) injector Δp [Pa]
    dp_ox: Array              # axial (oxidiser) injector Δp [Pa]
    v_fuel: Array             # radial jet velocity [m/s]
    v_ox: Array               # axial jet velocity [m/s]
    area_fuel: Array          # radial metering area [m²]
    area_ox: Array            # axial metering area [m²]
    momentum_ratio: Array     # TMR = (ṁ_r v_r)/(ṁ_a v_a)  (radial/axial)
    spray_half_angle_deg: Array
    slot_width: Array         # per-slot width from area/N/aspect [m]
    blockage_factor: Array    # BF = N w /(π D_pintle)
    # movable-pintle two-branch minimum area (plan rule 8)
    tip_opening: Array        # Son opening L metering the radial area [m]
    area_tip_branch: Array    # A_tip (tip-opening-limited) = required radial area
    area_center_gap: Array    # A_cg (center-gap-limited, fixed)
    transition_margin: Array  # frac·A_cg − A_tip  (≥0 ⇒ tip-controlled)
    branch_consistency: Array # A_cg − A_tip       (≥0 ⇒ tip is the true minimum)
    # stability (chug) screen
    chug_margin_fuel: Array   # χ_f − χ_min   (≥0 feasible)
    chug_margin_ox: Array     # χ_o − χ_min   (≥0 feasible)


def injector_readouts(*, Pc: Array, chi_f: Array, chi_o: Array,
                      D_pintle: Array, mdot_fuel: Array, mdot_ox: Array,
                      mission: MissionSpec) -> InjectorReadout:
    """Closed-form pintle-injector readouts + constraint margins.

    All of ``Pc, chi_f, chi_o, D_pintle, mdot_fuel, mdot_ox`` are differentiable
    inputs; ``mission`` carries the architecture constants.  Pure jnp — safe
    inside jit/jacfwd/jacrev with no host callbacks.
    """
    # --- orifice metering (incompressible) ---------------------------------- #
    dp_f = chi_f * Pc
    dp_o = chi_o * Pc
    G_f = mission.injector_cd_fuel * jnp.sqrt(2.0 * mission.rho_fuel * dp_f)
    G_o = mission.injector_cd_ox * jnp.sqrt(2.0 * mission.rho_ox * dp_o)
    v_f = G_f / mission.rho_fuel
    v_o = G_o / mission.rho_ox
    A_f = mdot_fuel / G_f                     # radial metering area
    A_o = mdot_ox / G_o                       # axial metering area

    # --- momentum ratio + spray angle (radial = fuel, axial = ox) ----------- #
    m_radial = mdot_fuel * v_f
    m_axial = mdot_ox * v_o
    TMR = m_radial / m_axial
    delta = jnp.deg2rad(mission.pintle_deflector_angle_deg)
    radial_comp = m_radial * jnp.cos(delta)
    axial_comp = m_axial + m_radial * jnp.sin(delta)
    spray_half_angle = jnp.rad2deg(jnp.arctan2(radial_comp, axial_comp))

    # --- slot geometry + blockage factor ------------------------------------ #
    N = mission.pintle_slot_count
    AR = mission.pintle_slot_aspect_ratio
    slot_w = jnp.sqrt(A_f / (N * AR))         # A_f = N·w·(AR·w)
    BF = N * slot_w / (jnp.pi * D_pintle)

    # --- movable-pintle two-branch minimum area (Son 2017) ------------------ #
    r_f = 0.5 * D_pintle - mission.pintle_post_thickness
    A_cg = center_gap_area(mission.pintle_center_gap_diameter,
                           mission.pintle_rod_diameter)
    A_tip = A_f                               # tip meters the required radial area
    L_open = opening_for_tip_area(A_tip, r_f, mission.pintle_tip_angle_deg)
    frac = mission.pintle_transition_area_fraction
    transition_margin = frac * A_cg - A_tip   # consistency inequality (≥0)
    branch_consistency = A_cg - A_tip

    # --- chug / feed-decoupling screen -------------------------------------- #
    chi_min = mission.injector_dp_stability_min
    chug_f = chi_f - chi_min
    chug_o = chi_o - chi_min

    return InjectorReadout(
        dp_fuel=dp_f, dp_ox=dp_o, v_fuel=v_f, v_ox=v_o,
        area_fuel=A_f, area_ox=A_o, momentum_ratio=TMR,
        spray_half_angle_deg=spray_half_angle, slot_width=slot_w,
        blockage_factor=BF, tip_opening=L_open, area_tip_branch=A_tip,
        area_center_gap=A_cg, transition_margin=transition_margin,
        branch_consistency=branch_consistency,
        chug_margin_fuel=chug_f, chug_margin_ox=chug_o,
    )
