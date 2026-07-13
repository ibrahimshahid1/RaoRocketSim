"""Core, dependency-light contracts for deterministic parcel-spray models.

The spray package uses SI units throughout.  These types intentionally contain
no guessed thermophysical properties: every property needed by a selected
physical model must either be supplied here or provided by a separate property
backend.  ``frozen=True`` is reinforced by copying NumPy inputs into read-only
arrays so callers cannot mutate a solved state through an array alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np


class SprayValidationError(ValueError):
    """Raised when a spray-model contract is malformed before integration."""


def _readonly_array(
    value,
    *,
    name: str,
    dtype=float,
    ndim: int | None = None,
) -> np.ndarray:
    """Return an owned, finite, read-only NumPy array."""

    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise SprayValidationError(
            f"{name} must have {ndim} dimensions, got shape {array.shape}"
        )
    if dtype is not bool and not np.all(np.isfinite(array)):
        raise SprayValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SprayValidationError(f"{name} must be finite and > 0")
    return value


def _optional_positive_scalar(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _positive_scalar(name, value)


@dataclass(frozen=True)
class LiquidProperties:
    """Resolved properties of one parcel-forming liquid at its inlet state.

    The first six properties are the minimum needed by primary-breakup and drag
    models.  Thermal/phase-change properties are optional because they are not
    available for every repository propellant.  An evaporation model must
    reject a state whose required optional properties remain ``None`` rather
    than filling them with a generic value.
    """

    name: str
    density: float                    # kg/m^3
    dynamic_viscosity: float          # Pa s
    surface_tension: float            # N/m
    temperature: float                # K
    pressure: float                   # absolute Pa
    specific_heat: float | None = None        # J/(kg K)
    thermal_conductivity: float | None = None # W/(m K)
    latent_heat: float | None = None          # J/kg
    vapor_molar_mass: float | None = None     # kg/mol
    vapor_diffusivity: float | None = None    # m^2/s

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise SprayValidationError("liquid name must be nonblank")
        object.__setattr__(self, "name", name)
        for field_name in (
            "density",
            "dynamic_viscosity",
            "surface_tension",
            "temperature",
            "pressure",
        ):
            object.__setattr__(
                self, field_name, _positive_scalar(field_name, getattr(self, field_name))
            )
        for field_name in (
            "specific_heat",
            "thermal_conductivity",
            "latent_heat",
            "vapor_molar_mass",
            "vapor_diffusivity",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_positive_scalar(field_name, getattr(self, field_name)),
            )


@dataclass(frozen=True)
class CarrierSample:
    """Carrier-gas state sampled at one or more parcel positions.

    ``velocity`` has shape ``(..., 3)``.  Every scalar field is broadcast to the
    same leading shape and stored as an owned read-only array.  Turbulence uses
    the conventional specific kinetic energy ``k`` [m^2/s^2] and dissipation
    rate ``epsilon`` [m^2/s^3].  A laminar sample explicitly uses ``k=epsilon=0``.
    """

    velocity: np.ndarray
    density: np.ndarray | float
    dynamic_viscosity: np.ndarray | float
    temperature: np.ndarray | float
    pressure: np.ndarray | float
    turbulent_kinetic_energy: np.ndarray | float
    turbulent_dissipation_rate: np.ndarray | float

    def __post_init__(self) -> None:
        velocity = _readonly_array(self.velocity, name="velocity", dtype=float)
        if velocity.ndim < 1 or velocity.shape[-1] != 3:
            raise SprayValidationError(
                f"velocity must have shape (..., 3), got {velocity.shape}"
            )
        leading_shape = velocity.shape[:-1]
        object.__setattr__(self, "velocity", velocity)

        scalar_fields = (
            "density",
            "dynamic_viscosity",
            "temperature",
            "pressure",
            "turbulent_kinetic_energy",
            "turbulent_dissipation_rate",
        )
        values: dict[str, np.ndarray] = {}
        for field_name in scalar_fields:
            raw = np.asarray(getattr(self, field_name), dtype=float)
            try:
                broadcast = np.broadcast_to(raw, leading_shape)
            except ValueError as exc:
                raise SprayValidationError(
                    f"{field_name} shape {raw.shape} cannot broadcast to "
                    f"carrier sample shape {leading_shape}"
                ) from exc
            values[field_name] = _readonly_array(
                broadcast, name=field_name, dtype=float
            )
            object.__setattr__(self, field_name, values[field_name])

        for field_name in ("density", "dynamic_viscosity", "temperature", "pressure"):
            if np.any(values[field_name] <= 0.0):
                raise SprayValidationError(f"{field_name} must be > 0 everywhere")
        k = values["turbulent_kinetic_energy"]
        epsilon = values["turbulent_dissipation_rate"]
        if np.any(k < 0.0) or np.any(epsilon < 0.0):
            raise SprayValidationError(
                "turbulent kinetic energy and dissipation rate must be >= 0"
            )
        if np.any((k > 0.0) & (epsilon <= 0.0)):
            raise SprayValidationError(
                "positive turbulent kinetic energy requires positive dissipation rate"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Leading sample shape, excluding the three velocity components."""

        return self.velocity.shape[:-1]

    @property
    def is_turbulent(self) -> np.ndarray:
        return self.turbulent_kinetic_energy > 0.0


@dataclass(frozen=True)
class SpraySolverSpec:
    """Deterministic numerical controls shared by the parcel solver.

    Time step, time horizon, parcel count, and the eddy-lifetime coefficient are
    deliberately required inputs: their appropriate values depend on the case
    and must be convergence-tested.  ``seed`` defaults to zero solely to make an
    otherwise fully specified simulation repeatable.
    """

    time_step: float
    maximum_time: float
    parcels_per_liquid_stream: int
    eddy_lifetime_constant: float
    seed: int = 0
    bit_generator: str = field(init=False, default="PCG64")

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_step", _positive_scalar("time_step", self.time_step))
        object.__setattr__(
            self, "maximum_time", _positive_scalar("maximum_time", self.maximum_time)
        )
        if (
            isinstance(self.parcels_per_liquid_stream, (bool, np.bool_))
            or int(self.parcels_per_liquid_stream) != self.parcels_per_liquid_stream
            or int(self.parcels_per_liquid_stream) < 1
        ):
            raise SprayValidationError("parcels_per_liquid_stream must be an integer >= 1")
        object.__setattr__(
            self, "parcels_per_liquid_stream", int(self.parcels_per_liquid_stream)
        )
        object.__setattr__(
            self,
            "eddy_lifetime_constant",
            _positive_scalar("eddy_lifetime_constant", self.eddy_lifetime_constant),
        )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise SprayValidationError("seed must be a nonnegative integer")
        if int(self.seed) < 0:
            raise SprayValidationError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        # Record the implementation selected by the required default_rng path.
        generator_name = type(np.random.default_rng(self.seed).bit_generator).__name__
        object.__setattr__(self, "bit_generator", generator_name)

    @property
    def maximum_steps(self) -> int:
        return int(math.ceil(self.maximum_time / self.time_step))

    @property
    def rng_metadata(self) -> dict[str, int | str]:
        return {"seed": self.seed, "bit_generator": self.bit_generator}


@dataclass(frozen=True)
class ParcelCloud:
    """Minimal weighted computational-parcel state.

    ``statistical_weight`` is the number of equal physical droplets represented
    by each computational parcel.  Liquid density is intentionally not stored
    here; mass calculations must use the explicitly selected
    :class:`LiquidProperties` for each role.
    """

    position: np.ndarray                 # (n, 3), m
    velocity: np.ndarray                 # (n, 3), m/s
    diameter: np.ndarray                 # (n,), m
    temperature: np.ndarray              # (n,), K
    statistical_weight: np.ndarray       # (n,), physical droplets / parcel
    roles: tuple[str, ...] | Iterable[str]
    active: np.ndarray | None = None      # (n,), bool
    age: np.ndarray | None = None         # (n,), s

    def __post_init__(self) -> None:
        position = _readonly_array(self.position, name="position", dtype=float, ndim=2)
        velocity = _readonly_array(self.velocity, name="velocity", dtype=float, ndim=2)
        if position.shape[1:] != (3,) or velocity.shape != position.shape:
            raise SprayValidationError(
                "position and velocity must both have shape (n, 3)"
            )
        n = position.shape[0]
        if n < 1:
            raise SprayValidationError("a parcel cloud must contain at least one parcel")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)

        if self.active is None:
            active = np.ones(n, dtype=bool)
        else:
            active = np.asarray(self.active)
            if active.dtype.kind != "b":
                raise SprayValidationError("active must contain boolean values")
            active = _readonly_array(active, name="active", dtype=bool, ndim=1)
            if active.shape != (n,):
                raise SprayValidationError(f"active must have shape ({n},)")
        active = np.array(active, dtype=bool, copy=True)
        active.setflags(write=False)
        object.__setattr__(self, "active", active)

        for field_name in ("diameter", "temperature", "statistical_weight"):
            array = _readonly_array(
                getattr(self, field_name), name=field_name, dtype=float, ndim=1
            )
            if array.shape != (n,):
                raise SprayValidationError(f"{field_name} must have shape ({n},)")
            if field_name == "diameter":
                if np.any(array < 0.0) or np.any(array[active] <= 0.0):
                    raise SprayValidationError(
                        "diameter must be >= 0 everywhere and > 0 for active parcels"
                    )
            elif np.any(array <= 0.0):
                raise SprayValidationError(f"{field_name} must be > 0 everywhere")
            object.__setattr__(self, field_name, array)

        roles = tuple(str(role).strip() for role in self.roles)
        if len(roles) != n or any(not role for role in roles):
            raise SprayValidationError(
                f"roles must contain {n} nonblank role labels"
            )
        object.__setattr__(self, "roles", roles)

        if self.age is None:
            age = np.zeros(n, dtype=float)
        else:
            age = _readonly_array(self.age, name="age", dtype=float, ndim=1)
            if age.shape != (n,):
                raise SprayValidationError(f"age must have shape ({n},)")
            if np.any(age < 0.0):
                raise SprayValidationError("age must be >= 0 everywhere")
        age = np.array(age, dtype=float, copy=True)
        age.setflags(write=False)
        object.__setattr__(self, "age", age)

    @property
    def count(self) -> int:
        return int(self.position.shape[0])

    def represented_liquid_mass(self, liquid_density: float) -> np.ndarray:
        """Return represented mass per parcel for an explicitly supplied density."""

        density = _positive_scalar("liquid_density", liquid_density)
        mass = (
            self.statistical_weight
            * density
            * (math.pi / 6.0)
            * self.diameter**3
        )
        mass.setflags(write=False)
        return mass


__all__ = [
    "CarrierSample",
    "LiquidProperties",
    "ParcelCloud",
    "SpraySolverSpec",
    "SprayValidationError",
]
