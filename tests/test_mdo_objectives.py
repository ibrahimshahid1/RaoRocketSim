"""Mass-objective names must match the hardware subtotal they select."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raosim.mdo.objectives import (
    DEFAULT_MASS_OBJECTIVE,
    MassObjective,
    coerce_mass_objective,
    mass_objective_value,
)


def _result():
    return SimpleNamespace(
        objective_mass=7.0,
        chamber_mass=SimpleNamespace(total=5.0),
    )


def test_default_min_mass_means_resolved_partial_dry_mass():
    assert DEFAULT_MASS_OBJECTIVE is MassObjective.MIN_DRY_MASS_PARTIAL
    assert mass_objective_value(_result(), DEFAULT_MASS_OBJECTIVE) == 12.0


def test_electric_package_objective_remains_available_by_explicit_name():
    assert mass_objective_value(
        _result(), MassObjective.MIN_ELECTRIC_PACKAGE_MASS
    ) == 7.0


def test_unknown_objective_fails_closed():
    with pytest.raises(ValueError, match="unknown mass objective"):
        coerce_mass_objective("full_engine_mass")
