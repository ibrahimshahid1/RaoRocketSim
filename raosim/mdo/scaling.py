"""
raosim.mdo.scaling — affine variable/residual scaling (Phase 1).

Gradient-based NLP on raw SI units mixes O(1e6) pressures with O(0.1)
fractions; the standard cure (Martins & Ning 2021, ch. 4 practice) is an
affine map to the unit box and residual scales chosen so a "well-solved"
residual is O(1).  Everything here is closed-form and jit/grad-safe.

``ScaledSpace`` is the single owner of the map:

    z = (x - lo) / (hi - lo)          in [0, 1]
    x = lo + z * (hi - lo)

State/residual scaling for the skeleton solve lives with the assembly
(each residual is nondimensionalized by its natural reference — thrust by
``F_target``, mass flow by the reference mdot, temperature by ``Tc``), which
keeps the §11 gate metric ("normalized constraint violation", "max scaled
residual") meaningful across disciplines.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from raosim.mdo.schema import VariableSpec, bounds_arrays

Array = jnp.ndarray


@dataclass(frozen=True)
class ScaledSpace:
    """Affine [lo, hi] <-> [0, 1] map over an ordered variable space."""

    lo: Array
    hi: Array
    names: tuple[str, ...]

    @classmethod
    def from_specs(cls, space: tuple[VariableSpec, ...]) -> "ScaledSpace":
        lo, hi = bounds_arrays(space)
        return cls(lo=lo, hi=hi, names=tuple(s.name for s in space))

    # -- maps ---------------------------------------------------------------- #
    def to_unit(self, x: Array) -> Array:
        return (jnp.asarray(x) - self.lo) / (self.hi - self.lo)

    def to_physical(self, z: Array) -> Array:
        return self.lo + jnp.asarray(z) * (self.hi - self.lo)

    def clip_unit(self, z: Array) -> Array:
        """Box projection for optimizer safeguarding (NOT for hiding invalid
        physics — plan rule 9: regime limits are constraints, this is only the
        trust-region box)."""
        return jnp.clip(jnp.asarray(z), 0.0, 1.0)

    def contains(self, x: Array, *, atol: float = 0.0) -> Array:
        x = jnp.asarray(x)
        return jnp.all((x >= self.lo - atol) & (x <= self.hi + atol))
