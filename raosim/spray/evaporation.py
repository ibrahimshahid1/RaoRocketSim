"""Explicit Spalding evaporation primitives for Lagrangian spray parcels.

Radhakrishnan, Lee, and Koo (2021), Eq. (16), give the positive
evaporation-rate magnitude

``m_dot = k_c A_d rho_g ln(1 + B_m)``.

The parcel derivative implemented here is negative.  The mass-transfer
coefficient is closed only through a caller-selected, named Sherwood model;
diffusivity, densities, viscosity, relative speed, and Spalding number are all
required explicitly.  No propellant or thermodynamic defaults are inferred.

For ``k_c = Sh*D/d``, freezing ``Sh`` over a step integrates the corresponding
``d^2`` equation exactly.  Re-evaluating ``Sh`` on subsequent calls gives a
first-order march when Reynolds-dependent Sherwood closures are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence


SherwoodClosureName = Literal["quiescent_sphere", "ranz_marshall_1952"]


@dataclass(frozen=True)
class SherwoodClosure:
    name: SherwoodClosureName
    provenance: str


SHERWOOD_CLOSURES: dict[str, SherwoodClosure] = {
    "quiescent_sphere": SherwoodClosure(
        name="quiescent_sphere",
        provenance="Classical stagnant spherical-film limit, Sh = 2",
    ),
    "ranz_marshall_1952": SherwoodClosure(
        name="ranz_marshall_1952",
        provenance=(
            "Ranz-Marshall forced-convection mass-transfer analogy, "
            "Sh = 2 + 0.6 Re^0.5 Sc^(1/3)"
        ),
    ),
}


@dataclass(frozen=True)
class EvaporationRate:
    mass_derivative_per_droplet: float
    represented_mass_derivative: float
    gas_reynolds: float
    schmidt: float
    sherwood: float
    mass_transfer_coefficient: float
    spalding_mass_number: float
    closure: SherwoodClosureName


@dataclass(frozen=True)
class EvaporationParcelState:
    diameter: float
    multiplicity: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class EvaporationConservation:
    represented_liquid_mass_before: float
    represented_liquid_mass_after: float
    vapor_mass_source_demand: float
    mass_residual: float
    represented_liquid_momentum_before: tuple[float, float, float]
    represented_liquid_momentum_after: tuple[float, float, float]
    carrier_momentum_source_demand: tuple[float, float, float]
    momentum_residual: tuple[float, float, float]
    carrier_coupling_status: str = (
        "one_way_source_demand_only_not_global_carrier_closure"
    )


@dataclass(frozen=True)
class EvaporationStepResult:
    state: EvaporationParcelState
    rate_at_step_start: EvaporationRate
    diameter_squared_loss_rate: float
    fully_evaporated: bool
    conservation: EvaporationConservation


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _vector3(value: Sequence[float]) -> tuple[float, float, float]:
    try:
        result = tuple(float(component) for component in value)
    except TypeError as exc:  # pragma: no cover - defensive error normalization
        raise ValueError("velocity must contain exactly three components") from exc
    if len(result) != 3 or any(not math.isfinite(component) for component in result):
        raise ValueError("velocity must contain exactly three finite components")
    return result  # type: ignore[return-value]


def sherwood_closure(name: SherwoodClosureName | str) -> SherwoodClosure:
    try:
        return SHERWOOD_CLOSURES[str(name)]
    except KeyError as exc:
        choices = ", ".join(sorted(SHERWOOD_CLOSURES))
        raise ValueError(f"unknown Sherwood closure {name!r}; choose {choices}") from exc


def spalding_mass_number(
    *, surface_vapor_mass_fraction: float, bulk_vapor_mass_fraction: float
) -> float:
    """Return ``Bm=(Y_surface-Y_bulk)/(1-Y_surface)`` for evaporation."""

    y_surface = float(surface_vapor_mass_fraction)
    y_bulk = float(bulk_vapor_mass_fraction)
    if not math.isfinite(y_surface) or not 0.0 <= y_surface < 1.0:
        raise ValueError("surface_vapor_mass_fraction must be finite and in [0, 1)")
    if not math.isfinite(y_bulk) or not 0.0 <= y_bulk <= y_surface:
        raise ValueError(
            "bulk_vapor_mass_fraction must be finite and in "
            "[0, surface_vapor_mass_fraction] for evaporation"
        )
    return (y_surface - y_bulk) / (1.0 - y_surface)


def gas_reynolds_number(
    *,
    diameter: float,
    carrier_density: float,
    carrier_dynamic_viscosity: float,
    relative_speed: float,
) -> float:
    d = _finite_positive("diameter", diameter)
    rho_g = _finite_positive("carrier_density", carrier_density)
    mu_g = _finite_positive(
        "carrier_dynamic_viscosity", carrier_dynamic_viscosity
    )
    speed = _finite_nonnegative("relative_speed", relative_speed)
    return rho_g * speed * d / mu_g


def schmidt_number(
    *,
    carrier_density: float,
    carrier_dynamic_viscosity: float,
    mass_diffusivity: float,
) -> float:
    rho_g = _finite_positive("carrier_density", carrier_density)
    mu_g = _finite_positive(
        "carrier_dynamic_viscosity", carrier_dynamic_viscosity
    )
    diffusivity = _finite_positive("mass_diffusivity", mass_diffusivity)
    return mu_g / (rho_g * diffusivity)


def sherwood_number(
    *,
    gas_reynolds: float,
    schmidt: float,
    closure: SherwoodClosureName | str,
) -> float:
    re = _finite_nonnegative("gas_reynolds", gas_reynolds)
    sc = _finite_positive("schmidt", schmidt)
    selected = sherwood_closure(closure)
    if selected.name == "quiescent_sphere":
        return 2.0
    if selected.name == "ranz_marshall_1952":
        return 2.0 + 0.6 * math.sqrt(re) * sc ** (1.0 / 3.0)
    raise AssertionError(f"unhandled Sherwood closure {selected.name}")


def evaporation_rate_2021(
    *,
    diameter: float,
    multiplicity: float,
    carrier_density: float,
    carrier_dynamic_viscosity: float,
    mass_diffusivity: float,
    relative_speed: float,
    spalding_mass_number_value: float,
    closure: SherwoodClosureName | str,
) -> EvaporationRate:
    """Evaluate the negative parcel mass derivative from 2021 Eq. (16)."""

    d = _finite_positive("diameter", diameter)
    count = _finite_positive("multiplicity", multiplicity)
    rho_g = _finite_positive("carrier_density", carrier_density)
    diffusivity = _finite_positive("mass_diffusivity", mass_diffusivity)
    bm = _finite_nonnegative(
        "spalding_mass_number_value", spalding_mass_number_value
    )
    re = gas_reynolds_number(
        diameter=d,
        carrier_density=rho_g,
        carrier_dynamic_viscosity=carrier_dynamic_viscosity,
        relative_speed=relative_speed,
    )
    sc = schmidt_number(
        carrier_density=rho_g,
        carrier_dynamic_viscosity=carrier_dynamic_viscosity,
        mass_diffusivity=diffusivity,
    )
    selected = sherwood_closure(closure)
    sh = sherwood_number(gas_reynolds=re, schmidt=sc, closure=selected.name)
    k_c = sh * diffusivity / d
    area = math.pi * d * d
    derivative = -k_c * area * rho_g * math.log1p(bm)
    return EvaporationRate(
        mass_derivative_per_droplet=derivative,
        represented_mass_derivative=count * derivative,
        gas_reynolds=re,
        schmidt=sc,
        sherwood=sh,
        mass_transfer_coefficient=k_c,
        spalding_mass_number=bm,
        closure=selected.name,
    )


def diameter_squared_evaporation_rate(
    *,
    sherwood: float,
    mass_diffusivity: float,
    carrier_density: float,
    liquid_density: float,
    spalding_mass_number_value: float,
) -> float:
    """Return positive ``-d(d^2)/dt`` implied by Eq. (16)."""

    sh = _finite_positive("sherwood", sherwood)
    diffusivity = _finite_positive("mass_diffusivity", mass_diffusivity)
    rho_g = _finite_positive("carrier_density", carrier_density)
    rho_l = _finite_positive("liquid_density", liquid_density)
    bm = _finite_nonnegative(
        "spalding_mass_number_value", spalding_mass_number_value
    )
    return 4.0 * sh * diffusivity * rho_g / rho_l * math.log1p(bm)


def _represented_mass(diameter: float, multiplicity: float, density: float) -> float:
    return multiplicity * density * math.pi * diameter**3 / 6.0


def advance_evaporation(
    state: EvaporationParcelState,
    *,
    dt: float,
    liquid_density: float,
    carrier_density: float,
    carrier_dynamic_viscosity: float,
    mass_diffusivity: float,
    relative_speed: float,
    spalding_mass_number_value: float,
    closure: SherwoodClosureName | str,
) -> EvaporationStepResult:
    """Advance one evaporation step with a frozen-step Sherwood number.

    The liquid mass lost in the step is recorded as an equal vapor mass source
    demand.  Vapor is assigned the parcel velocity at transfer, so the liquid
    momentum loss and carrier momentum source demand also balance locally.
    This records source demand only; a prescribed one-way carrier field does
    not provide global carrier mass, momentum, or energy closure.
    """

    dt = _finite_positive("dt", dt)
    rho_l = _finite_positive("liquid_density", liquid_density)
    d_old = _finite_positive("diameter", state.diameter)
    count = _finite_positive("multiplicity", state.multiplicity)
    velocity = _vector3(state.velocity)
    rate = evaporation_rate_2021(
        diameter=d_old,
        multiplicity=count,
        carrier_density=carrier_density,
        carrier_dynamic_viscosity=carrier_dynamic_viscosity,
        mass_diffusivity=mass_diffusivity,
        relative_speed=relative_speed,
        spalding_mass_number_value=spalding_mass_number_value,
        closure=closure,
    )
    d2_loss_rate = diameter_squared_evaporation_rate(
        sherwood=rate.sherwood,
        mass_diffusivity=mass_diffusivity,
        carrier_density=carrier_density,
        liquid_density=rho_l,
        spalding_mass_number_value=spalding_mass_number_value,
    )
    d2_new = max(0.0, d_old * d_old - d2_loss_rate * dt)
    d_new = math.sqrt(d2_new)
    mass_before = _represented_mass(d_old, count, rho_l)
    mass_after = _represented_mass(d_new, count, rho_l)
    vapor_source = mass_before - mass_after
    momentum_before = tuple(mass_before * component for component in velocity)
    momentum_after = tuple(mass_after * component for component in velocity)
    carrier_demand = tuple(vapor_source * component for component in velocity)
    momentum_residual = tuple(
        before - after - demand
        for before, after, demand in zip(
            momentum_before, momentum_after, carrier_demand, strict=True
        )
    )
    conservation = EvaporationConservation(
        represented_liquid_mass_before=mass_before,
        represented_liquid_mass_after=mass_after,
        vapor_mass_source_demand=vapor_source,
        mass_residual=mass_before - mass_after - vapor_source,
        represented_liquid_momentum_before=momentum_before,
        represented_liquid_momentum_after=momentum_after,
        carrier_momentum_source_demand=carrier_demand,
        momentum_residual=momentum_residual,
    )
    return EvaporationStepResult(
        state=EvaporationParcelState(
            diameter=d_new,
            multiplicity=count,
            velocity=velocity,
        ),
        rate_at_step_start=rate,
        diameter_squared_loss_rate=d2_loss_rate,
        fully_evaporated=d_new == 0.0,
        conservation=conservation,
    )


__all__ = [
    "EvaporationConservation",
    "EvaporationParcelState",
    "EvaporationRate",
    "EvaporationStepResult",
    "SHERWOOD_CLOSURES",
    "SherwoodClosure",
    "advance_evaporation",
    "diameter_squared_evaporation_rate",
    "evaporation_rate_2021",
    "gas_reynolds_number",
    "schmidt_number",
    "sherwood_closure",
    "sherwood_number",
    "spalding_mass_number",
]
