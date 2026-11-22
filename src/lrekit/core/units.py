"""Engineering-notation input parsing with a canonical SI store.

Users write ``13 kN``, ``30 bar``, ``0.8 mm``.  The repository stores SI.  The
one rule that makes that safe is: **never guess a unit**.  A bare number where
a unit is expected is rejected, because the failure mode being prevented is not
a parse error -- it is a 1000x design error that runs to completion and
produces a plausible-looking answer.  That is the same class of bug as the
pintle STEP metres-as-millimetres defect and the material label that never
reached the traced core.

Each parse is dimension-checked: asking for a pressure and receiving ``13 kN``
is an error, not a silent reinterpretation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "DIMENSIONS",
    "Quantity",
    "UnitError",
    "canonical_unit",
    "format_quantity",
    "parse_quantity",
]


class UnitError(ValueError):
    """An input could not be parsed, or carried the wrong dimension."""


# Multiplicative unit tables, keyed by dimension.  The first entry of each
# table is the canonical SI unit for that dimension.  Offsets (temperature)
# are handled separately below.
_UNITS: dict[str, dict[str, float]] = {
    "force": {"N": 1.0, "kN": 1.0e3, "MN": 1.0e6, "lbf": 4.4482216152605},
    "pressure": {
        "Pa": 1.0, "kPa": 1.0e3, "MPa": 1.0e6, "GPa": 1.0e9,
        "bar": 1.0e5, "mbar": 1.0e2, "atm": 101325.0,
        "psi": 6894.757293168361, "psia": 6894.757293168361,
        "ksi": 6894757.293168361, "torr": 133.32236842105263,
    },
    "length": {
        "m": 1.0, "km": 1.0e3, "cm": 1.0e-2, "mm": 1.0e-3,
        "um": 1.0e-6, "micron": 1.0e-6, "in": 0.0254, "ft": 0.3048,
    },
    "volume": {
        "m^3": 1.0, "m3": 1.0,
        "L": 1.0e-3, "liter": 1.0e-3, "litre": 1.0e-3,
        "cm^3": 1.0e-6, "cm3": 1.0e-6,
        "mm^3": 1.0e-9, "mm3": 1.0e-9,
        "in^3": 0.0254 ** 3, "in3": 0.0254 ** 3,
        "ft^3": 0.3048 ** 3, "ft3": 0.3048 ** 3,
    },
    "time": {"s": 1.0, "ms": 1.0e-3, "min": 60.0, "h": 3600.0, "hr": 3600.0},
    "mass": {"kg": 1.0, "g": 1.0e-3, "t": 1.0e3, "lb": 0.45359237,
             "lbm": 0.45359237},
    "mass_flow": {"kg/s": 1.0, "g/s": 1.0e-3, "lb/s": 0.45359237},
    "velocity": {"m/s": 1.0, "km/s": 1.0e3, "ft/s": 0.3048, "mm/s": 1.0e-3},
    "angle": {"rad": 1.0, "deg": math.pi / 180.0},
    "rotational_speed": {"rpm": 1.0, "krpm": 1.0e3},
    "power": {"W": 1.0, "kW": 1.0e3, "MW": 1.0e6},
    "heat_flux": {
        "W/m^2": 1.0, "W/m2": 1.0,
        "kW/m^2": 1.0e3, "kW/m2": 1.0e3,
        "MW/m^2": 1.0e6, "MW/m2": 1.0e6,
        "W/cm^2": 1.0e4, "W/cm2": 1.0e4,
    },
    "energy": {"J": 1.0, "kJ": 1.0e3, "MJ": 1.0e6, "Wh": 3600.0,
               "kWh": 3.6e6},
    "temperature": {"K": 1.0},          # offsets handled explicitly
    "dimensionless": {"": 1.0, "-": 1.0, "%": 0.01},
}

#: Dimensions whose canonical unit is the first key of their table.
DIMENSIONS: tuple[str, ...] = tuple(_UNITS)

#: Affine temperature units: (scale, offset) into kelvin.
_TEMPERATURE_AFFINE: dict[str, tuple[float, float]] = {
    "K": (1.0, 0.0),
    "degC": (1.0, 273.15),
    "C": (1.0, 273.15),
    "degF": (5.0 / 9.0, 255.3722222222222),
    "F": (5.0 / 9.0, 255.3722222222222),
    "degR": (5.0 / 9.0, 0.0),
    "R": (5.0 / 9.0, 0.0),
}

# Unit symbols are case-sensitive: an unsupported prefix must not become a
# different magnitude merely because case-folding found a familiar symbol.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_QUANTITY_RE = re.compile(rf"^\s*({_NUMBER})\s*([^\s].*?)?\s*$")


@dataclass(frozen=True)
class Quantity:
    """A parsed value in SI, retaining exactly what the user wrote."""

    value: float
    unit: str
    dimension: str
    source: str

    def __float__(self) -> float:
        return self.value


def canonical_unit(dimension: str) -> str:
    """The SI unit every value of ``dimension`` is stored in."""

    try:
        return next(iter(_UNITS[dimension]))
    except KeyError as exc:
        raise UnitError(
            f"unknown dimension {dimension!r}; expected one of "
            f"{', '.join(DIMENSIONS)}"
        ) from exc


def _resolve_unit(token: str, table: Mapping[str, float]) -> str | None:
    return token if token in table else None


def parse_quantity(
    text: str | float | int,
    *,
    dimension: str,
    allow_bare: bool = False,
    field: str = "value",
) -> Quantity:
    """Parse engineering notation into a canonical SI :class:`Quantity`.

    ``allow_bare`` permits a unitless number and is intended only for genuinely
    dimensionless inputs.  For a dimensional field it must stay False: silently
    assuming SI is how ``0.8`` meant as millimetres becomes 0.8 metres.
    """

    if dimension not in _UNITS:
        raise UnitError(
            f"unknown dimension {dimension!r}; expected one of "
            f"{', '.join(DIMENSIONS)}"
        )
    table = _UNITS[dimension]
    si_unit = canonical_unit(dimension)

    if isinstance(text, bool):
        raise UnitError(f"{field}: expected a quantity, got a boolean")
    if isinstance(text, (int, float)):
        if not (allow_bare or dimension == "dimensionless"):
            raise UnitError(
                f"{field}: a bare number ({text!r}) is ambiguous for a "
                f"{dimension}; write the unit explicitly, e.g. "
                f"'{text} {si_unit}'"
            )
        value = float(text)
        if not math.isfinite(value):
            raise UnitError(f"{field}: value must be finite; got {text!r}")
        return Quantity(value, si_unit, dimension, str(text))

    raw = str(text).strip()
    match = _QUANTITY_RE.match(raw)
    if match is None:
        raise UnitError(f"{field}: could not parse a quantity from {text!r}")
    number, token = match.group(1), (match.group(2) or "").strip()

    magnitude = float(number)
    if not math.isfinite(magnitude):
        raise UnitError(f"{field}: value must be finite; got {text!r}")

    if not token:
        if not (allow_bare or dimension == "dimensionless"):
            raise UnitError(
                f"{field}: {text!r} has no unit, which is ambiguous for a "
                f"{dimension}; write it explicitly, e.g. "
                f"'{number} {si_unit}'"
            )
        return Quantity(magnitude, si_unit, dimension, raw)

    if dimension == "temperature":
        key = _resolve_unit(token.replace("°", "deg"), _TEMPERATURE_AFFINE)
        if key is None:
            raise UnitError(
                f"{field}: {token!r} is not a temperature unit; expected one "
                f"of {', '.join(sorted(_TEMPERATURE_AFFINE))}"
            )
        scale, offset = _TEMPERATURE_AFFINE[key]
        kelvin = magnitude * scale + offset
        if not math.isfinite(kelvin):
            raise UnitError(f"{field}: converted value must be finite")
        if kelvin < 0.0:
            raise UnitError(
                f"{field}: {raw!r} is below absolute zero ({kelvin:.3f} K)"
            )
        return Quantity(kelvin, "K", "temperature", raw)

    key = _resolve_unit(token, table)
    if key is None:
        # A unit that is valid but for a different dimension is a distinct,
        # more useful error than "unrecognized".
        for other, other_table in _UNITS.items():
            if other != dimension and _resolve_unit(token, other_table):
                raise UnitError(
                    f"{field}: {token!r} is a {other} unit, but a {dimension} "
                    f"was expected (canonical {si_unit})"
                )
        raise UnitError(
            f"{field}: unknown {dimension} unit {token!r}; expected one of "
            f"{', '.join(k for k in table if k)}"
        )
    converted = magnitude * table[key]
    if not math.isfinite(converted):
        raise UnitError(f"{field}: converted value must be finite")
    return Quantity(converted, si_unit, dimension, raw)


def format_quantity(value: float, dimension: str) -> str:
    """Render an SI value with its canonical unit, for echoing back."""

    unit = canonical_unit(dimension)
    return f"{value:g}" if not unit else f"{value:g} {unit}"
