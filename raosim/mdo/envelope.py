"""mdo/envelope.py — differentiable thrust-chamber envelope.

Why this module exists
----------------------
NASA SP-125 §2.1 (printed p. 31, ``propulsion_texts/19710019929.pdf``) lists
**envelope (size)** as one of the nine parameters that *"engine systems design
and development specifications will have to cover... above all"*, and gives the
working definition:

    "In those cases where only approximate values are required for comparison
    or for overall estimates, the term 'envelope' is preferred.  For instance,
    definition of a hypothetical smallest cylinder, cube, or sphere into which
    the engine would fit conveys a good feeling of engine size or bulkiness."

A cylinder is the natural choice for an axisymmetric thrust chamber, so this
module returns the smallest enclosing cylinder of the **thrust-chamber outer
mould line**: a diameter and a length.

SP-125 also records why this is a design driver rather than a nicety:
*"engine size directly affects engine weight... The vehicle structure, which
becomes heavier, especially with upper stages.  Engine size directly affects the
size and thus weight of the aft end and/or interstage structure."*

What this is NOT
----------------
Both returned quantities are **lower bounds on the installed engine envelope**,
and callers must not present them as the engine envelope.

* *Diameter* is built from the same radial stack the mass ledger integrates
  (``r + t_wall + channel_height + t_jacket``), so it is the outside of the
  cooling jacket.  It excludes the **bolted-interface flange**, which is
  resolved host-side in :mod:`raosim.interface` and which
  ``docs/HARDWARE_MASS_LEDGER.md`` records as the *largest* diameter on the
  13 kN baseline — a 285.5 mm flange around a 177 mm chamber.  A flange that
  large governs the envelope outright.
* *Length* runs from the injector face (station 0 of the grid, which is the
  SP-125 chamber-volume datum) to the nozzle exit plane.  It excludes injector
  body depth, the dome, the pump package, valves, lines and the gimbal.

The requirement layer therefore records an envelope requirement as *partially*
screened; see :class:`raosim.requirements.RequirementCoverage`.  The
authoritative envelope comes from the host re-evaluation, which has the flange.

Smoothness
----------
The radial maximum uses the same conservative smooth upper envelope as
:mod:`raosim.mdo.engine` (``logsumexp(k·v)/k``, never below the true maximum).
Erring large is the conservative direction for an envelope constraint, and it
keeps the NLP free of the active-set kink a hard ``max`` over stations would
introduce (plan §0.1; see also gap 12.11 on remaining hard extrema).
"""

from __future__ import annotations

from typing import NamedTuple

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp
import jax.scipy as jsp

from raosim.mdo.grid import StationGrid
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray

#: Sharpness for the radial smooth maximum, in 1/m.  The radial stack varies
#: over O(0.1 m), so 1e3 resolves stations to well under a millimetre while
#: staying far from overflow in float64.
_RADIAL_SHARPNESS = 1.0e3


class ChamberEnvelope(NamedTuple):
    """Smallest enclosing cylinder of the thrust-chamber outer mould line."""

    diameter: Array          # [m] 2 * smooth-max outer radius
    length: Array            # [m] injector face to nozzle exit plane
    r_outer: Array           # (n,) [m] per-station outer mould line
    #: ``True`` label for downstream provenance: neither field includes the
    #: bolted flange, injector body, or feed hardware.
    is_lower_bound: bool


def _smooth_max(values: Array, sharpness: float) -> Array:
    """Conservative differentiable upper envelope (never below ``max``)."""

    v = jnp.asarray(values, dtype=jnp.float64)
    return jsp.special.logsumexp(sharpness * v) / sharpness


def chamber_envelope(
    grid: StationGrid,
    *,
    t_wall: Array,
    channel_height: Array,
    closeout_thickness: Array,
) -> ChamberEnvelope:
    """Enclosing cylinder of the cooled thrust chamber.

    Parameters
    ----------
    grid
        The station grid used by the coupled solve, so envelope, mass and
        cooling all see one geometry.
    t_wall, channel_height
        Design variables, traced.
    closeout_thickness
        Per-station solved jacket thickness from
        :func:`raosim.mdo.mass.chamber_mass` — pass
        ``chamber.closeout_thickness`` rather than recomputing it, so the
        envelope and the mass ledger cannot disagree about the same jacket.

    Returns
    -------
    ChamberEnvelope
        ``diameter`` and ``length`` are differentiable scalars suitable for an
        NLP constraint; both are lower bounds on the installed envelope.
    """

    t_w = jnp.asarray(t_wall, dtype=jnp.float64)
    h = jnp.asarray(channel_height, dtype=jnp.float64)
    t_j = jnp.asarray(closeout_thickness, dtype=jnp.float64)

    # Same radial stack as raosim.mdo.mass.chamber_mass: liner, channel, jacket.
    r_outer = grid.r + t_w + h + t_j
    diameter = 2.0 * _smooth_max(r_outer, _RADIAL_SHARPNESS)

    # grid.x is 0 at the throat, negative through the chamber and convergent,
    # positive through the divergent.  Station 0 is the injector face -- the
    # SP-125 chamber-volume datum (printed p. 88, "injector face to throat
    # plane"), which R0 made the shared convention between both pipelines.
    length = grid.x[-1] - grid.x[0]

    return ChamberEnvelope(
        diameter=diameter,
        length=length,
        r_outer=r_outer,
        is_lower_bound=True,
    )


def envelope_margins(
    envelope: ChamberEnvelope, mission: MissionSpec
) -> tuple[Array, Array]:
    """Signed envelope margins, ``>= 0`` feasible.

    Returns ``(diameter_margin, length_margin)`` in metres.  With the
    :class:`~raosim.mdo.schema.MissionSpec` defaults — deliberately large
    finite sentinels rather than ``inf``, so the Jacobian stays finite — both
    margins are large and the constraints are inert.  A requirement supplies
    real limits through :mod:`raosim.requirements`.
    """

    d_margin = (
        jnp.asarray(mission.envelope_diameter_max, dtype=jnp.float64)
        - envelope.diameter
    )
    l_margin = (
        jnp.asarray(mission.envelope_length_max, dtype=jnp.float64)
        - envelope.length
    )
    return d_margin, l_margin
