"""Explicit objective identities for the differentiable engine optimizer.

An objective name is a reporting contract, not just a scalar expression.  In
particular, electric feed hardware is not an engine dry mass, and the currently
resolved thrust-chamber-plus-feed subtotal is still not a complete engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "MassObjective",
    "DEFAULT_MASS_OBJECTIVE",
    "coerce_mass_objective",
    "mass_objective_value",
]


class MassObjective(str, Enum):
    """Version-stable identities for implemented differentiable objectives."""

    MIN_ELECTRIC_PACKAGE_MASS = "min_electric_package_mass"
    MIN_DRY_MASS_PARTIAL = "min_dry_mass_partial"


DEFAULT_MASS_OBJECTIVE = MassObjective.MIN_DRY_MASS_PARTIAL


def coerce_mass_objective(value: MassObjective | str) -> MassObjective:
    if isinstance(value, MassObjective):
        return value
    try:
        return MassObjective(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"unknown mass objective {value!r}; expected one of "
            f"{tuple(item.value for item in MassObjective)}"
        ) from exc


def mass_objective_value(result: Any, objective: MassObjective | str):
    """Return the selected traced scalar from an ``EngineResult``.

    The compatibility reads allow this selector to straddle the schema
    migration while keeping the selected meaning explicit at the optimizer
    boundary.
    """

    selected = coerce_mass_objective(objective)
    electric = (
        result.electric_package_objective_mass
        if hasattr(result, "electric_package_objective_mass")
        else result.objective_mass
    )
    if selected is MassObjective.MIN_ELECTRIC_PACKAGE_MASS:
        return electric
    return getattr(
        result,
        "dry_mass_partial_objective_mass",
        electric + result.chamber_mass.total,
    )
