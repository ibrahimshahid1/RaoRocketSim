"""Workflow-independent engine input validation."""

from __future__ import annotations

import math

import pytest

from raosim.input_validation import InputValidationError, validate_engine_inputs


def _valid(**overrides):
    values = dict(
        chamber_pressure=3.0e6,
        expansion_ratio=8.0,
        thrust=13.0e3,
        ambient_pressure=101325.0,
        mixture_ratio=2.27,
        burn_duration=120.0,
    )
    values.update(overrides)
    return validate_engine_inputs(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chamber_pressure", -1.0),
        ("chamber_pressure", math.nan),
        ("expansion_ratio", 0.5),
        ("thrust", 0.0),
        ("mixture_ratio", -2.0),
        ("burn_duration", math.inf),
    ],
)
def test_nonphysical_common_scalars_fail_before_numerics(field, value):
    with pytest.raises(InputValidationError, match=field):
        _valid(**{field: value})


def test_ambient_must_be_below_chamber_pressure():
    with pytest.raises(InputValidationError, match="ambient_pressure"):
        _valid(ambient_pressure=3.0e6)


def test_qualification_duration_cannot_be_shorter_than_flight_duration():
    with pytest.raises(InputValidationError, match="qualification_duration"):
        _valid(flight_duration=120.0, qualification_duration=119.0)


def test_validator_does_not_impose_an_unsourced_pressure_design_box():
    # Search guidance and validated model domains are separate from elementary
    # input validity.  A high pressure is not rejected here merely because it
    # lies outside the historical 13 kN electric-pump search window.
    assert _valid(chamber_pressure=30.0e6, ambient_pressure=0.0) is None


def test_negative_altitude_is_rejected():
    with pytest.raises(InputValidationError, match="altitude"):
        _valid(altitude=-1.0)
