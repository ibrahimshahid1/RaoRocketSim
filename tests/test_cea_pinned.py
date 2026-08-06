"""Contract tests for parity-aligned pinned chamber thermochemistry."""

from __future__ import annotations

import pytest

from raosim.cea import (
    PinnedChamberState,
    THERMO_CEA_FROZEN,
    THERMO_PINNED_CHAMBER,
    resolve_thermochemistry,
)
from raosim.gas_dynamics import characteristic_velocity


def _state() -> PinnedChamberState:
    gamma = 1.23
    temperature = 3450.0
    gas_constant = 361.0
    return PinnedChamberState(
        gamma=gamma,
        Tc=temperature,
        R_gas=gas_constant,
        c_star_ideal=characteristic_velocity(
            gamma, gas_constant, temperature
        ),
        source="manufactured MDO property-surface snapshot",
        surface_fingerprint="0123456789abcdef",
    )


def test_pinned_chamber_state_preserves_the_exact_mdo_convention():
    pinned = _state()
    resolved = resolve_thermochemistry(
        thermo_mode=THERMO_PINNED_CHAMBER,
        propellant_name="LOX/RP-1",
        Pc=3.0e6,
        mixture_ratio=2.27,
        eta_cstar=0.94,
        eta_CF=0.97,
        pinned_chamber_state=pinned,
    )

    propellant = resolved.propellant
    assert resolved.mode == THERMO_PINNED_CHAMBER
    assert resolved.source == pinned.source
    assert propellant.gamma == pytest.approx(pinned.gamma)
    assert propellant.Tc == pytest.approx(pinned.Tc)
    assert propellant.R_gas == pytest.approx(pinned.R_gas)
    assert propellant.c_star == pytest.approx(pinned.c_star_ideal)
    assert propellant.eta_cstar == pytest.approx(0.94)
    assert propellant.eta_CF == pytest.approx(0.97)
    assert resolved.chamber_state["surface_fingerprint"] == (
        pinned.surface_fingerprint
    )
    assert any("common-input parity" in warning for warning in resolved.warnings)


def test_pinned_chamber_state_rejects_an_inconsistent_cstar():
    pinned = _state()
    inconsistent = PinnedChamberState(
        **{
            **pinned.as_dict(),
            "c_star_ideal": pinned.c_star_ideal * 1.01,
        }
    )

    with pytest.raises(ValueError, match="inconsistent"):
        resolve_thermochemistry(
            thermo_mode=THERMO_PINNED_CHAMBER,
            propellant_name="LOX/RP-1",
            Pc=3.0e6,
            mixture_ratio=2.27,
            pinned_chamber_state=inconsistent,
        )


def test_pinned_state_cannot_be_silently_ignored_by_another_mode():
    with pytest.raises(ValueError, match="requires thermo_mode"):
        resolve_thermochemistry(
            thermo_mode=THERMO_CEA_FROZEN,
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            Pc=3.0e6,
            mixture_ratio=2.27,
            pinned_chamber_state=_state(),
        )
