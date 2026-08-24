"""Engineering-notation parsing with a canonical SI store.

The rule under test is that a unit is never guessed. The failure being
prevented is not a parse error but a silent 1000x design error -- the same
class as the pintle STEP metres-as-millimetres defect.
"""

from __future__ import annotations

import math

import pytest

from raosim.units import UnitError, canonical_unit, parse_quantity


@pytest.mark.parametrize(
    "text, dimension, expected",
    [
        ("13 kN", "force", 13.0e3),
        ("13kN", "force", 13.0e3),
        ("30 bar", "pressure", 3.0e6),
        ("2.5 MPa", "pressure", 2.5e6),
        ("7e6 Pa", "pressure", 7.0e6),
        ("0.8 mm", "length", 0.8e-3),
        ("1 in", "length", 0.0254),
        ("2.5 L", "volume", 2.5e-3),
        ("10 in^3", "volume", 10.0 * 0.0254 ** 3),
        ("120 s", "time", 120.0),
        ("2 min", "time", 120.0),
        ("30 krpm", "rotational_speed", 30.0e3),
        ("7.5 MW/m^2", "heat_flux", 7.5e6),
        ("12 W/cm^2", "heat_flux", 12.0e4),
        ("15 deg", "angle", math.pi / 12.0),
        ("600 K", "temperature", 600.0),
        ("-40 degC", "temperature", 233.15),
        ("80 %", "dimensionless", 0.8),
    ],
)
def test_engineering_notation_lands_in_si(text, dimension, expected):
    q = parse_quantity(text, dimension=dimension)
    assert q.value == pytest.approx(expected)
    assert q.unit == canonical_unit(dimension)
    assert q.source == text.strip()


@pytest.mark.parametrize(
    "dimension", ["force", "pressure", "length", "volume", "heat_flux", "time"]
)
def test_a_bare_number_is_rejected_for_a_dimensional_field(dimension):
    """No unit guessing: 0.8 meaning millimetres must not become 0.8 metres."""
    with pytest.raises(UnitError, match="ambiguous"):
        parse_quantity("0.8", dimension=dimension)
    with pytest.raises(UnitError, match="ambiguous"):
        parse_quantity(0.8, dimension=dimension)


def test_a_bare_number_is_accepted_only_where_truly_dimensionless():
    assert parse_quantity(8.0, dimension="dimensionless").value == 8.0
    assert parse_quantity("8", dimension="dimensionless").value == 8.0
    # ...or where the caller opts in explicitly.
    assert parse_quantity(
        3.0e6, dimension="pressure", allow_bare=True
    ).value == pytest.approx(3.0e6)


def test_a_wrong_dimension_names_the_actual_dimension():
    with pytest.raises(UnitError, match="is a force unit"):
        parse_quantity("13 kN", dimension="pressure")
    with pytest.raises(UnitError, match="is a length unit"):
        parse_quantity("5 mm", dimension="time")


def test_unknown_units_and_garbage_are_rejected():
    with pytest.raises(UnitError, match="unknown pressure unit"):
        parse_quantity("30 zorks", dimension="pressure")
    with pytest.raises(UnitError, match="could not parse"):
        parse_quantity("abc", dimension="force")
    with pytest.raises(UnitError, match="unknown dimension"):
        parse_quantity("1 m", dimension="luminosity")


def test_case_sensitive_prefixes_that_actually_matter():
    """``mm`` and ``MPa`` must not collapse into one another."""
    assert parse_quantity("1 mm", dimension="length").value == pytest.approx(1e-3)
    assert parse_quantity("1 m", dimension="length").value == pytest.approx(1.0)
    assert parse_quantity("1 MPa", dimension="pressure").value == pytest.approx(1e6)
    assert parse_quantity("1 kPa", dimension="pressure").value == pytest.approx(1e3)


def test_non_finite_and_sub_absolute_zero_are_rejected():
    with pytest.raises(UnitError):
        parse_quantity(float("nan"), dimension="dimensionless")
    with pytest.raises(UnitError):
        parse_quantity(float("inf"), dimension="dimensionless")
    with pytest.raises(UnitError, match="absolute zero"):
        parse_quantity("-500 degC", dimension="temperature")


def test_booleans_are_not_silently_numeric():
    with pytest.raises(UnitError, match="boolean"):
        parse_quantity(True, dimension="dimensionless")
