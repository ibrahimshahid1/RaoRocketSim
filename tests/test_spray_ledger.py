"""Conservation ledgers distinguish parcel closure from carrier closure."""

import numpy as np
import pytest

from raosim.spray.ledger import (
    ConservationLedger,
    close_reservoir_ledger,
    represented_mass,
    represented_momentum,
)


def test_represented_mass_and_momentum():
    density = 1000.0
    d = np.array([1.0e-3, 2.0e-3])
    n = np.array([10.0, 2.0])
    mass = represented_mass(d, n, density)
    assert mass == pytest.approx(density * np.pi / 6.0 * n * d**3)
    velocity = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert represented_momentum(mass, velocity) == pytest.approx(
        [mass[0], 2.0 * mass[1], 0.0]
    )


def test_exact_mass_and_parcel_momentum_closure():
    ledger = close_reservoir_ledger(
        role="water",
        injected_mass=1.0,
        active_mass=0.4,
        vaporized_mass=0.1,
        wall_mass=0.2,
        exit_mass=0.3,
        initial_momentum=(1.0, 0.0, 0.0),
        active_momentum=(0.4, 0.0, 0.0),
        vapor_momentum=(0.1, 0.0, 0.0),
        wall_momentum=(0.2, 0.0, 0.0),
        exit_momentum=(0.5, 0.0, 0.0),
        drag_impulse_on_parcels=(0.2, 0.0, 0.0),
    )
    assert ledger.mass_relative_residual < 1.0e-14
    assert ledger.parcel_momentum_relative_residual < 1.0e-14

    one_way = ConservationLedger(
        {"water": ledger}, mass_tolerance=1.0e-12,
        momentum_tolerance=1.0e-12,
    )
    assert one_way.mass_closed
    assert one_way.parcel_momentum_closed
    assert not one_way.globally_momentum_closed
    data = one_way.to_dict()
    assert data["carrier_momentum_status"] == "one_way_source_demand_unapplied"
    assert data["per_role"]["water"]["carrier_reaction_impulse_demand_n_s"] \
        == pytest.approx([-0.2, 0.0, 0.0])

    two_way = ConservationLedger(
        {"water": ledger}, mass_tolerance=1.0e-12,
        momentum_tolerance=1.0e-12, carrier_coupling="two_way",
    )
    assert two_way.globally_momentum_closed


def test_ledger_detects_missing_mass_and_bad_inputs():
    ledger = close_reservoir_ledger(
        role="lox",
        injected_mass=1.0,
        active_mass=0.9,
        vaporized_mass=0.0,
        wall_mass=0.0,
        exit_mass=0.0,
        initial_momentum=(0.0, 0.0, 0.0),
        active_momentum=(0.0, 0.0, 0.0),
        vapor_momentum=(0.0, 0.0, 0.0),
        wall_momentum=(0.0, 0.0, 0.0),
        exit_momentum=(0.0, 0.0, 0.0),
        drag_impulse_on_parcels=(0.0, 0.0, 0.0),
    )
    assert ledger.mass_relative_residual == pytest.approx(0.1)
    with pytest.raises(ValueError, match="density"):
        represented_mass([1.0], [1.0], 0.0)
