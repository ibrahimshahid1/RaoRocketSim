"""
raosim.mdo.pintle — the pintle element as an optimizable MDO component.

``mdo/injector.py`` sizes the injector's *flow*: metering areas, momentum
ratio, blockage factor and the Son (2017) two-branch minimum area.  Before this
block the pintle had no hardware in the optimizer: its diameter reached the
objective through no path, so the optimum left it at its starting value, and
the movable-pintle transition row was enforced on a slotted post that has no
moving tip.  This block gives the pintle the same standing as the thrust
chamber: owned design variables, a differentiable hardware mass in the partial
dry-mass objective, and constraints that belong to the architecture actually
selected.

Architecture is an outer-enumeration choice (plan rule 4), selected by
``MissionSpec.pintle_architecture`` and never relaxed into a continuous value:

* ``"fixed_slotted"`` — the radial stream leaves N slots cut through a fixed
  post.  This is the architecture the five-part coaxial CAD exports and the
  first physical article (a water cold-flow pintle) uses.  The blockage-factor
  band applies; the Son transition and minimum-opening rows do not.
* ``"son_movable"`` — the pintle valve of Son et al., *Design Procedure of a
  Movable Pintle Injector for Liquid Rocket Engines*, JPP 33(4), 2017,
  DOI 10.2514/1.B36301: a continuous radial gap metered by the travel of a
  centre rod inside the post.  Its metering opening, transition and throttle
  range apply; the slot blockage factor does not.  The host exporter refuses
  CAD for this architecture, so its optimum is a numerical design only.

Design variables owned here: ``D_pintle`` (post outside diameter, already a
variable), ``t_post`` (post wall) and ``D_cg`` (centre-gap diameter, active only
for ``son_movable``; the NLP pins it for the slotted post).

Physics and its basis:

* **Hardware mass** (direct prismatic screening geometry, the basis the
  thrust-chamber Pappus/shell integration uses).  The post is an annular tube
  of outside diameter D and wall t over ``pintle_body_length_factor * D``; the
  host machined layout uses the same 3 D default body length
  (``raosim.injector_cad.resolve_machined_pintle_layout``, a repository
  heuristic).  The slotted post subtracts N slot prisms w * h * t, exactly the
  "pintle post" row of the host ``raosim.mass_ledger.injector_mass_ledger``
  (its NumPy parity oracle).  The movable valve adds its centre rod,
  pi/4 * D_pr^2 over the same length; no host oracle exists for the rod.  The
  material is the host's injector structural alloy, which the MDO already
  carries as its closeout/jacket material.  Faceplate, injector body, orifice
  plate, manifolds and seals stay outside this block, so the injector total
  remains unavailable.
* **Post pressure screen.**  sigma = p * r_o / t, the thin pressure-vessel
  (Barlow) hoop form ``raosim.interface`` already uses.  Evaluated with the
  OUTSIDE radius it bounds the elastic Lame bore stress
  p (r_o^2 + r_i^2)/(r_o^2 - r_i^2) from above for every wall thickness, so
  the screen stays conservative where a thin-shell t/r limit would not hold.
  The load is the radial stream's full injector-inlet pressure against
  ambient (an unpressurised chamber at start or shutdown), which bounds the
  steady differential chi_f * Pc.  Allowable = closeout yield / closeout
  factor of safety (SP-8087 sec. 2.1.3 yield band, already on MissionSpec).
  Slot ligaments, external pressure from the surrounding oxidizer annulus and
  thermal stress are not screened.
* **Wall fit.**  t <= 0.8 * D/2 mirrors the host layout's own wall cap
  (``injector_cad.resolve_machined_pintle_layout``), keeping a bore for the
  radial stream.
* **Valve opening** (``son_movable``).  Son 2017 reports no uniform liquid
  sheet within 0.1 mm of opening (PDF p. 2) and rejects a throttle schedule
  whose opening falls below 0.1 mm as infeasible because of nonuniform
  injection (PDF p. 10).  The opening at the lowest declared throttle level is
  therefore held above ``pintle_min_opening``.  Throttling follows the host
  ``throttle_map`` convention: the radial pressure-drop fraction is held and
  Pc scales with mass flow (Son 2017 PDF p. 10, "proportional reduction in
  the chamber pressures and mass flow rates"), so the metered radial area
  scales as f**0.5.  This is an algebraic single-point surrogate for the
  throttle requirement, not the multipoint mission of plan Phase 10.
* **Centre-gap fit** (``son_movable``).  The centre gap sits inside the post
  bore (Son 2017 Fig. 2), so D_cg <= D - 2 t.  The rod must stay inside the
  centre gap; the transition row (A_tip <= frac * A_cg, Son PDF pp. 4 and 9)
  already enforces A_cg > 0.
* **Module fit.**  Son 2017 PDF p. 9: "the calculated chamber diameter should
  be larger than the injector module diameter".  The module diameter is the
  outside diameter of the axial annulus around the post,
  sqrt(D^2 + 4 A_axial / pi).

Not-applicable rows return a positive inert value so the fixed-shape
constraint vector is unchanged; ``mdo.constraints`` marks them not applicable.
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.injector import InjectorReadout, opening_for_tip_area
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray

#: Outer-enumeration architecture identities (plan rule 4).
PINTLE_FIXED_SLOTTED = "fixed_slotted"
PINTLE_SON_MOVABLE = "son_movable"
PINTLE_ARCHITECTURES = (PINTLE_FIXED_SLOTTED, PINTLE_SON_MOVABLE)

#: Host wall cap mirrored from ``injector_cad.resolve_machined_pintle_layout``.
POST_WALL_MAX_FRACTION_OF_RADIUS = 0.80

#: Positive value returned by rows that do not apply to the selected
#: architecture.  Finite and positive so the fixed-shape vector never carries a
#: NaN or a violation; the manifest metadata says the row is not applicable.
INERT_MARGIN = 1.0


def is_movable(mission: MissionSpec) -> bool:
    """Static architecture switch (resolved at trace time, never traced)."""

    return mission.pintle_architecture == PINTLE_SON_MOVABLE


def pintle_density(mission: MissionSpec) -> float:
    """Injector structural alloy density, shared with the closeout/jacket."""

    rho = mission.rho_closeout
    return float(mission.rho_wall if rho is None else rho)


def post_allowable_stress(mission: MissionSpec) -> float:
    """Post-factor-of-safety allowable of the injector structural alloy."""

    return float(mission.closeout_sigma_yield / mission.closeout_structural_fos)


@dataclass(frozen=True)
class PintleComponent:
    """Differentiable hardware, structure and valve readouts of the pintle."""

    post_mass: Array            # slotted/plain post tube [kg]
    rod_mass: Array             # movable centre rod [kg]; 0 when there is none
    mass: Array                 # post + rod [kg]
    body_length: Array          # modelled post length [m]
    post_inner_diameter: Array  # bore D - 2 t [m]
    post_hoop_stress: Array     # Barlow outer-radius hoop stress [Pa]
    post_stress_margin: Array   # allowable - stress [Pa]
    wall_fit_margin: Array      # 0.8 D/2 - t [m]
    opening_full: Array         # Son opening at full power [m]
    opening_throttle_min: Array  # Son opening at the lowest throttle level [m]
    min_opening_margin: Array   # opening_throttle_min - L_min [m] (movable)
    center_gap_fit_margin: Array  # D - 2 t - D_cg [m] (movable)
    module_diameter: Array      # axial-annulus outside diameter [m]
    chamber_diameter: Array     # cylindrical chamber diameter [m]
    module_fit_margin: Array    # chamber - module diameter [m]


def pintle_component(*, D_pintle: Array, t_post: Array, D_cg: Array,
                     Pc: Array, chi_f: Array, Rt: Array,
                     injector: InjectorReadout,
                     mission: MissionSpec) -> PintleComponent:
    """Closed-form pintle hardware and valve readouts.

    Pure jnp on ``D_pintle, t_post, D_cg, Pc, chi_f, Rt`` and the injector
    readouts, so jit/jacfwd/jacrev flow through it with no host callbacks.
    """

    D = jnp.asarray(D_pintle, dtype=jnp.float64)
    t = jnp.asarray(t_post, dtype=jnp.float64)
    r_o = 0.5 * D
    r_i = r_o - t
    length = mission.pintle_body_length_factor * D
    rho = pintle_density(mission)

    # --- hardware mass --------------------------------------------------- #
    tube_volume = jnp.pi * (r_o * r_o - r_i * r_i) * length
    if is_movable(mission):
        # Continuous radial gap at the tip: no slots are cut through the post.
        post_volume = tube_volume
        rod_volume = (0.25 * jnp.pi * mission.pintle_rod_diameter ** 2
                      * length)
    else:
        slot_h = mission.pintle_slot_aspect_ratio * injector.slot_width
        slot_volume = (mission.pintle_slot_count * injector.slot_width
                       * slot_h * t)
        post_volume = tube_volume - slot_volume
        rod_volume = jnp.asarray(0.0, dtype=jnp.float64)
    post_mass = rho * post_volume
    rod_mass = rho * rod_volume

    # --- post pressure screen (Barlow on the outside radius) -------------- #
    p_load = Pc * (1.0 + chi_f) - mission.Pa
    hoop = p_load * r_o / t
    stress_margin = post_allowable_stress(mission) - hoop
    wall_fit = POST_WALL_MAX_FRACTION_OF_RADIUS * r_o - t

    # --- Son valve opening over the throttle range ----------------------- #
    # Full power meters the solved radial area; the lowest throttle level
    # meters A * f**0.5 (radial dp fraction held, Pc proportional to mdot).
    opening_full = injector.tip_opening
    f_min = mission.pintle_throttle_min_fraction
    opening_low = opening_for_tip_area(
        injector.area_tip_branch * jnp.sqrt(f_min), r_i,
        mission.pintle_tip_angle_deg,
    )
    inert = jnp.asarray(INERT_MARGIN, dtype=jnp.float64)
    if is_movable(mission):
        min_opening_margin = opening_low - mission.pintle_min_opening
        center_gap_fit = (D - 2.0 * t) - D_cg
    else:
        min_opening_margin = inert
        center_gap_fit = inert

    # --- injector module inside the chamber (Son 2017 PDF p. 9) ---------- #
    module_d = jnp.sqrt(D * D + 4.0 * injector.area_ox / jnp.pi)
    chamber_d = 2.0 * Rt * jnp.sqrt(mission.contraction_ratio)
    module_fit = chamber_d - module_d

    return PintleComponent(
        post_mass=post_mass,
        rod_mass=rod_mass,
        mass=post_mass + rod_mass,
        body_length=length,
        post_inner_diameter=2.0 * r_i,
        post_hoop_stress=hoop,
        post_stress_margin=stress_margin,
        wall_fit_margin=wall_fit,
        opening_full=opening_full,
        opening_throttle_min=opening_low,
        min_opening_margin=min_opening_margin,
        center_gap_fit_margin=center_gap_fit,
        module_diameter=module_d,
        chamber_diameter=chamber_d,
        module_fit_margin=module_fit,
    )


__all__ = [
    "INERT_MARGIN",
    "PINTLE_ARCHITECTURES",
    "PINTLE_FIXED_SLOTTED",
    "PINTLE_SON_MOVABLE",
    "POST_WALL_MAX_FRACTION_OF_RADIUS",
    "PintleComponent",
    "is_movable",
    "pintle_component",
    "pintle_density",
    "post_allowable_stress",
]
