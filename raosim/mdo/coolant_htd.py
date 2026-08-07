"""mdo/coolant_htd.py — supercritical heat-transfer-deterioration screen.

The gap this closes
-------------------
The repository's only coolant-side wall constraint is the RP-1 **coking** limit
(SP-8087, 728 K).  Coking is a hydrocarbon phenomenon, and
``raosim.mdo.propellants`` correctly reports ``coolant_wall_limit_K = None`` for
hydrogen — but nothing replaces it.  For methane and hydrogen the governing
coolant-side failure is not coking, it is **heat-transfer deterioration (HTD)**
at supercritical pressure, and until this module existed a LOX/LCH4 or LOX/LH2
design came back "feasible" when it was really *unmodelled*.

Nasuti & Pizzarelli, *Pseudo-boiling and heat transfer deterioration while
heating supercritical liquid rocket engine propellants*, J. Supercritical
Fluids 168:105066 (2021), ``propulsion_texts/nasuti2021.pdf``, states the
problem exactly:

    "Heating of liquid propellants used as the coolant in rocket engines may
    lead to undesired phenomena such as pseudo-boiling or heat transfer
    deterioration under specific conditions.  This can be an issue for
    propellants characterized at the same time by relatively low critical
    pressure and temperature.  Light hydrocarbons, as for instance methane,
    belong to this family."

The criterion
-------------
Their Eq. (9) gives the onset of deterioration as

.. math::

    \\frac{q_w}{G\\,f_w}\\left(\\frac{\\beta}{c_p}\\right)_b > K ,
    \\qquad K = 0.187

with :math:`q_w` the wall heat flux, :math:`G` the coolant mass flux,
:math:`f_w` the friction factor defined there as
:math:`f_w = 4\\tau_w / (\\tfrac12 \\rho_b u_b^2)` (i.e. the Darcy friction
factor), and :math:`(\\beta/c_p)_b` the isobaric thermal expansion coefficient
over the specific heat, both evaluated at **bulk** conditions.  The threshold
value is quoted as "essentially similar to that proposed in [20] and equal to
that proposed in [21] where the threshold value is also estimated to be
K = 0.187".

The group is dimensionless: ``[W/m2] / [kg/m2/s] = J/kg``, and
``beta/cp`` carries ``[1/K] / [J/(kg K)] = kg/J``.

Physical reading, in the authors' words: deterioration "occurs for large heat
flux q_w and low mass flux G", and "for a given flow in a tube, heat transfer
deterioration occurs in the tube sections where (β/c_p)_b is sufficiently
large" — that group "presents a maximum which nearly occurs" at the Widom
(pseudo-critical) line.  **That peak is the mechanism.**

Why this is not yet a live constraint
-------------------------------------
:math:`(\\beta/c_p)_b` is a real-fluid property that peaks sharply near the
pseudo-critical temperature.  The MDO cooling march currently carries the
coolant as four constants (``MissionSpec.rho_cool``, ``cp_cool``, ``mu_cool``,
``k_cool``) and has no :math:`\\beta` at all.  Evaluating Eq. (9) with constant
properties would return a smooth, flat number that misses precisely the peak
the criterion is detecting — a wrong model, not a coarse one.

So this module implements the criterion honestly and refuses to fake its
inputs: :func:`htd_margin` requires ``beta_over_cp`` to be supplied, and
:func:`htd_availability` reports the screen as unavailable, with the reason,
whenever it cannot be.  Closing it needs a **CoolProp-sampled coolant property
surface** over (T, p) — structurally the same object as the CEA chamber
surfaces in :mod:`raosim.mdo.properties`, and buildable from the same two
sources now in the corpus: Bell et al. 2014 (CoolProp) for the cryogens, and
Huber et al. 2009 (``huber2009.pdf``) for RP-1, which CoolProp does not carry.

Conservatism
------------
The authors note their models "are based on the case of smooth and uniformly
heated tubes which are more prone to deteriorate than the actual rough,
asymmetrically heated channels", and that deterioration "is mitigated in case
of pressure much higher than critical and by wall roughness".  So a margin
computed here errs toward predicting deterioration — the safe direction for a
screen, and worth stating in any report that quotes it.

Buoyancy is neglected, following the paper: "as the typical flow conditions in
rocket engine cooling channels are relevant to relatively high mass flux, of
the order of thousands of kg/s/m2, the resulting turbulent flow can be
considered unaffected by any buoyancy effect."
"""

from __future__ import annotations

from typing import NamedTuple

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

Array = jnp.ndarray

__all__ = [
    "HTD_THRESHOLD",
    "HTDScreen",
    "htd_availability",
    "htd_group",
    "htd_margin",
]

#: Nasuti & Pizzarelli (2021) Eq. (9) threshold K.  Quoted there as equal to
#: the value proposed in their ref. [21].
HTD_THRESHOLD = 0.187

#: Coolants for which supercritical HTD is a governing concern.  The paper
#: singles out fluids with "relatively low critical pressure and temperature —
#: light hydrocarbons, as for instance methane".  Hydrogen is included because
#: it is always supercritical in a regen jacket and has no coking limit to fall
#: back on, so HTD is its only coolant-side wall mechanism.
HTD_RELEVANT_COOLANTS = frozenset({"methane", "lch4", "ch4", "lh2", "hydrogen",
                                   "h2", "propane"})


class HTDScreen(NamedTuple):
    """Heat-transfer-deterioration screen result."""

    group: Array          # the Eq. (9) non-dimensional group, per station
    margin: Array         # K - group; >= 0 means no predicted deterioration
    threshold: float      # K
    available: bool       # False => inputs were not real-fluid; do not report
    reason: str           # why, when unavailable


def htd_group(
    heat_flux: Array,
    mass_flux: Array,
    friction_factor: Array,
    beta_over_cp: Array,
) -> Array:
    """Nasuti & Pizzarelli Eq. (9) group ``q_w / (G f_w) * (beta/cp)_b``.

    Parameters
    ----------
    heat_flux
        Wall heat flux ``q_w`` [W/m2], per station.
    mass_flux
        Coolant mass flux ``G`` [kg/(m2 s)], per station.
    friction_factor
        ``f_w = 4 tau_w / (rho_b u_b^2 / 2)`` — the Darcy friction factor.
    beta_over_cp
        ``(beta/cp)_b`` [kg/J] at bulk conditions: isobaric thermal expansion
        coefficient over specific heat.  This is the real-fluid term; it peaks
        near the pseudo-critical (Widom) temperature and that peak is what the
        criterion detects.

    Returns
    -------
    Array
        Dimensionless; deterioration is predicted where it exceeds
        :data:`HTD_THRESHOLD`.
    """

    q = jnp.asarray(heat_flux, dtype=jnp.float64)
    G = jnp.asarray(mass_flux, dtype=jnp.float64)
    f = jnp.asarray(friction_factor, dtype=jnp.float64)
    bcp = jnp.asarray(beta_over_cp, dtype=jnp.float64)
    # G and f are structurally positive for any flowing, wetted channel, so the
    # floors below cannot activate on a physical design and introduce no
    # design-dependent branch.
    return q / (jnp.maximum(G, 1.0e-12) * jnp.maximum(f, 1.0e-12)) * bcp


def htd_margin(
    heat_flux: Array,
    mass_flux: Array,
    friction_factor: Array,
    beta_over_cp: Array,
    *,
    threshold: float = HTD_THRESHOLD,
) -> Array:
    """``K - group``: ``>= 0`` means no predicted deterioration."""

    return jnp.asarray(threshold, dtype=jnp.float64) - htd_group(
        heat_flux, mass_flux, friction_factor, beta_over_cp)


def htd_availability(coolant_name: str,
                     has_real_fluid_properties: bool) -> tuple[bool, str]:
    """Whether the HTD screen can be evaluated, and why not when it cannot.

    Returns ``(available, reason)``.  ``available`` is ``True`` only when the
    screen is both *relevant* to the coolant and *computable* from real-fluid
    properties.  A screen that is irrelevant reports ``True`` with a reason
    naming the mechanism that governs instead — silence would be
    indistinguishable from an unmodelled risk.
    """

    name = str(coolant_name or "").strip().lower().replace("-", "")
    relevant = name in HTD_RELEVANT_COOLANTS
    if not relevant:
        return True, (
            f"HTD is not a governing mechanism for {coolant_name!r}; the "
            "SP-8087 coking limit is the coolant-side wall constraint"
        )
    if not has_real_fluid_properties:
        return False, (
            f"{coolant_name!r} is HTD-prone (Nasuti & Pizzarelli 2021: light "
            "hydrocarbons and hydrogen), but the cooling march carries the "
            "coolant as constant properties and has no isobaric expansion "
            "coefficient beta. Eq. (9) depends on (beta/cp)_b, which PEAKS at "
            "the pseudo-critical temperature -- a constant-property "
            "evaluation would be flat exactly where the criterion bites. "
            "Needs a CoolProp-sampled coolant property surface over (T, p); "
            "see raosim.mdo.properties for the analogous chamber surfaces"
        )
    return True, ""


def screen(
    coolant_name: str,
    *,
    heat_flux: Array | None = None,
    mass_flux: Array | None = None,
    friction_factor: Array | None = None,
    beta_over_cp: Array | None = None,
    threshold: float = HTD_THRESHOLD,
) -> HTDScreen:
    """Evaluate the screen, or report precisely why it could not be.

    Never returns a fabricated margin: when the real-fluid term is missing the
    group and margin come back as NaN with ``available=False``, so a caller
    cannot mistake an unmodelled risk for a satisfied constraint.
    """

    has_props = beta_over_cp is not None
    available, reason = htd_availability(coolant_name, has_props)

    if not (has_props and heat_flux is not None and mass_flux is not None
            and friction_factor is not None):
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        return HTDScreen(group=nan, margin=nan, threshold=threshold,
                         available=available and not _is_relevant(coolant_name),
                         reason=reason)

    g = htd_group(heat_flux, mass_flux, friction_factor, beta_over_cp)
    return HTDScreen(group=g, margin=jnp.asarray(threshold) - g,
                     threshold=threshold, available=True, reason=reason)


def _is_relevant(coolant_name: str) -> bool:
    name = str(coolant_name or "").strip().lower().replace("-", "")
    return name in HTD_RELEVANT_COOLANTS
