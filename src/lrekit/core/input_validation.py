"""Shared host-side validation for engine-analysis inputs.

The traditional, direct-MDO, optimization, and requirement workflows all
consume the same basic physical quantities.  Keeping their elementary domain
checks here prevents workflow dispatch order from changing whether bad input
is rejected cleanly or reaches a logarithm, square root, or division first.

These checks are mathematical/input-contract checks, not literature-derived
design limits.  Model validity domains and engineering recommendations remain
separate constraints with their own provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable

__all__ = [
    "InputValidationError",
    "WorkflowExitCode",
    "validate_engine_inputs",
    "validate_requirement_inputs",
]


class WorkflowExitCode(IntEnum):
    """Stable process verdicts shared by requirement and MDO workflows."""

    MET = 0
    CANDIDATE_VIOLATES = 1
    INVALID_INPUT = 2
    INDETERMINATE = 3
    SOLVER_FAILED = 4


@dataclass(frozen=True)
class InputValidationError(ValueError):
    """One invalid input, retaining its stable field name and rule."""

    field: str
    rule: str
    value: object

    def __str__(self) -> str:
        return f"{self.field} must be {self.rule}; got {self.value!r}"


def _finite(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(name, "a finite number", value) from exc
    if not math.isfinite(parsed):
        raise InputValidationError(name, "a finite number", value)
    return parsed


def _positive(name: str, value: float | int | None) -> float | None:
    parsed = _finite(name, value)
    if parsed is not None and parsed <= 0.0:
        raise InputValidationError(name, "finite and positive", value)
    return parsed


def _nonnegative(name: str, value: float | int | None) -> float | None:
    parsed = _finite(name, value)
    if parsed is not None and parsed < 0.0:
        raise InputValidationError(name, "finite and nonnegative", value)
    return parsed


def _nonnegative_integer(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    try:
        is_integer = not isinstance(value, bool) and int(value) == value
    except (TypeError, ValueError, OverflowError):
        is_integer = False
    if not is_integer or int(value) < 0:
        raise InputValidationError(name, "a nonnegative integer", value)
    return int(value)


def validate_engine_inputs(
    *,
    chamber_pressure: float,
    expansion_ratio: float,
    thrust: float | None = None,
    ambient_pressure: float | None = None,
    ambient_pressure_ratio: float | None = None,
    altitude: float | None = None,
    mixture_ratio: float | None = None,
    burn_duration: float | None = None,
    flight_duration: float | None = None,
    qualification_duration: float | None = None,
    isp_floor: float | None = None,
    envelope_diameter_max: float | None = None,
    envelope_length_max: float | None = None,
    mass_max: float | None = None,
    film_fraction: float | None = None,
    injector_drop_fractions: Iterable[float | None] = (),
    positive_dimensions: Iterable[tuple[str, float | None]] = (),
    restart_count: int | None = None,
    reusable_cycles: int | None = None,
) -> None:
    """Validate physical scalar domains shared by every engine workflow.

    This deliberately does not impose a chamber-pressure design box, a
    propellant-specific O/F band, a recommended burn duration, or any other
    empirical design choice.  Those belong to evidence-backed model domains,
    live constraints, or search-window policy rather than generic input
    validation.
    """

    pc = _positive("chamber_pressure", chamber_pressure)
    eps = _positive("expansion_ratio", expansion_ratio)
    if eps is not None and eps < 1.0:
        raise InputValidationError(
            "expansion_ratio", "finite and at least 1", expansion_ratio
        )

    _positive("thrust", thrust)
    _positive("mixture_ratio", mixture_ratio)
    burn = _positive("burn_duration", burn_duration)
    flight = _positive("flight_duration", flight_duration)
    qual = _positive("qualification_duration", qualification_duration)
    _positive("isp_floor", isp_floor)
    _positive("envelope_diameter_max", envelope_diameter_max)
    _positive("envelope_length_max", envelope_length_max)
    _positive("mass_max", mass_max)

    pa = _nonnegative("ambient_pressure", ambient_pressure)
    if pa is not None and pc is not None and pa >= pc:
        raise InputValidationError(
            "ambient_pressure", "finite, nonnegative, and below chamber_pressure",
            ambient_pressure,
        )
    pa_ratio = _nonnegative("ambient_pressure_ratio", ambient_pressure_ratio)
    if pa_ratio is not None and pa_ratio >= 1.0:
        raise InputValidationError(
            "ambient_pressure_ratio", "finite and in [0, 1)",
            ambient_pressure_ratio,
        )
    _nonnegative("altitude", altitude)

    effective_flight = flight if flight is not None else burn
    if qual is not None and effective_flight is not None and qual < effective_flight:
        raise InputValidationError(
            "qualification_duration",
            "at least the rated flight/burn duration",
            qualification_duration,
        )

    film = _finite("film_fraction", film_fraction)
    if film is not None and not (0.0 <= film < 1.0):
        raise InputValidationError("film_fraction", "finite and in [0, 1)", film)

    for index, value in enumerate(injector_drop_fractions):
        _positive(f"injector_drop_fraction[{index}]", value)
    for name, value in positive_dimensions:
        _positive(name, value)

    for name, value in (
        ("restart_count", restart_count),
        ("reusable_cycles", reusable_cycles),
    ):
        _nonnegative_integer(name, value)


def validate_requirement_inputs(
    *,
    thrust: float,
    flight_duration: float,
    qualification_duration: float | None = None,
    isp_floor: float | None = None,
    mixture_ratio: float | None = None,
    envelope_diameter_max: float | None = None,
    envelope_length_max: float | None = None,
    mass_max: float | None = None,
    throttle_range: tuple[float, float] | None = None,
    restart_count: int | None = None,
    reusable_cycles: int | None = None,
) -> None:
    """Validate requirement-owned scalars before mission resolution.

    Requirements intentionally do not own chamber pressure or expansion ratio,
    so they cannot call :func:`validate_engine_inputs` with real values for
    those design outputs.  This companion entry point reuses the same primitive
    finite/positive/integer rules without inventing placeholder design inputs.
    """

    _positive("thrust", thrust)
    flight = _positive("flight_duration", flight_duration)
    qualification = _positive(
        "qualification_duration", qualification_duration
    )
    _positive("isp_floor", isp_floor)
    _positive("mixture_ratio", mixture_ratio)
    _positive("envelope_diameter_max", envelope_diameter_max)
    _positive("envelope_length_max", envelope_length_max)
    _positive("mass_max", mass_max)

    if (
        qualification is not None
        and flight is not None
        and qualification < flight
    ):
        raise InputValidationError(
            "qualification_duration",
            "a cumulative duration at least the rated flight duration",
            qualification_duration,
        )

    if throttle_range is not None:
        try:
            lo_raw, hi_raw = throttle_range
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "throttle_range", "a finite (lo, hi) pair", throttle_range
            ) from exc
        lo = _positive("throttle_range[0]", lo_raw)
        hi = _positive("throttle_range[1]", hi_raw)
        if lo is not None and hi is not None and lo > hi:
            raise InputValidationError(
                "throttle_range", "finite and satisfy 0 < lo <= hi", throttle_range
            )

    for name, value in (
        ("restart_count", restart_count),
        ("reusable_cycles", reusable_cycles),
    ):
        _nonnegative_integer(name, value)
