"""Versioned WAVE/KH-RT breakup primitives for Lagrangian spray parcels.

The Kelvin-Helmholtz equations follow the Reitz WAVE formulation used by
Radhakrishnan et al. (2018, 2021).  Three coefficient sets are deliberately
kept separate:

``reitz_1987``
    The coefficients implemented by the OpenFOAM Foundation ReitzKHRT model:
    0.4 in the Taylor-number factor and 0.865 in the Weber-number factor.

``radhakrishnan_2018``
    The coefficients printed in Radhakrishnan et al. (2018), Eq. (5):
    0.4 and 0.87 respectively.

``radhakrishnan_2021``
    The coefficients printed in Radhakrishnan, Lee, and Koo (2021), Eq. (3):
    0.45 and 0.87 respectively.

The optional Rayleigh-Taylor branch reproduces the equations and timer logic
of the OpenFOAM Foundation ReitzKHRT implementation.  It is disabled by
default because Radhakrishnan et al. use the WAVE/KH branch, not KH-RT, in
their modelling-2 calculation.

This module represents breakup by reducing the diameter of the droplets in a
parcel and increasing parcel multiplicity so represented liquid mass remains
constant.  Velocity is unchanged by the breakup primitive, hence represented
parcel momentum is also an identity.  Any aerodynamic momentum exchange is a
separate one-way carrier source demand and is not a globally closed carrier
momentum balance here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence


WaveVariantName = Literal[
    "reitz_1987", "radhakrishnan_2018", "radhakrishnan_2021"
]


@dataclass(frozen=True)
class WaveCoefficientVariant:
    """Coefficient provenance for the fastest-growing KH wavelength."""

    name: WaveVariantName
    taylor_coefficient: float
    weber_denominator_coefficient: float
    provenance: str


WAVE_COEFFICIENT_VARIANTS: dict[str, WaveCoefficientVariant] = {
    "reitz_1987": WaveCoefficientVariant(
        name="reitz_1987",
        taylor_coefficient=0.4,
        weber_denominator_coefficient=0.865,
        provenance=(
            "Reitz (1987) WAVE form as implemented in the OpenFOAM "
            "Foundation ReitzKHRT model"
        ),
    ),
    "radhakrishnan_2018": WaveCoefficientVariant(
        name="radhakrishnan_2018",
        taylor_coefficient=0.4,
        weber_denominator_coefficient=0.87,
        provenance=(
            "Radhakrishnan et al. (2018), Atomization and Sprays, Eq. (5)"
        ),
    ),
    "radhakrishnan_2021": WaveCoefficientVariant(
        name="radhakrishnan_2021",
        taylor_coefficient=0.45,
        weber_denominator_coefficient=0.87,
        provenance=(
            "Radhakrishnan, Lee, and Koo (2021), Combustion Theory and "
            "Modelling, Eq. (3)"
        ),
    ),
}


@dataclass(frozen=True)
class RayleighTaylorConfig:
    """Optional OpenFOAM-ReitzKHRT Rayleigh-Taylor timer branch."""

    enabled: bool = False
    c_tau: float = 1.0
    c_rt: float = 0.1
    provenance: str = (
        "OpenFOAM Foundation ReitzKHRT.C (v10), fastest-growing RT "
        "frequency, wavelength, timer, and diameter update"
    )


@dataclass(frozen=True)
class WaveBreakupConfig:
    """Explicit WAVE constants and model choices for one breakup march."""

    b0: float
    b1: float
    coefficient_variant: WaveVariantName
    # OpenFOAM's radius-based gas Weber threshold 6 corresponds to a
    # conventional diameter-based Weber number of 12.
    weber_limit: float = 6.0
    rayleigh_taylor: RayleighTaylorConfig = field(
        default_factory=RayleighTaylorConfig
    )


@dataclass(frozen=True)
class WaveMetrics:
    diameter: float
    radius: float
    gas_weber: float
    liquid_weber: float
    liquid_reynolds: float
    ohnesorge: float
    taylor: float
    wavelength: float
    growth_rate: float
    breakup_time: float
    stable_diameter: float
    coefficient_variant: WaveVariantName


@dataclass(frozen=True)
class RayleighTaylorMetrics:
    effective_acceleration: float
    growth_rate: float
    wave_number: float
    wavelength: float
    breakup_time: float
    provenance: str


@dataclass(frozen=True)
class VOFWaveCalibration:
    """WAVE constants reconstructed from VOF sheet output."""

    full_sheet_thickness: float
    half_sheet_thickness: float
    breakup_length: float
    liquid_velocity: float
    wavelength: float
    growth_rate: float
    breakup_time: float
    b0: float
    b1: float
    coefficient_variant: WaveVariantName


@dataclass(frozen=True)
class BreakupParcelState:
    diameter: float
    multiplicity: float
    velocity: tuple[float, float, float]
    # OpenFOAM resets the timer to a very negative value after an RT event.
    # ``-inf`` is the exact, serialization-visible equivalent used here.
    rt_timer: float = 0.0


@dataclass(frozen=True)
class BreakupConservation:
    represented_mass_before: float
    represented_mass_after: float
    mass_residual: float
    represented_momentum_before: tuple[float, float, float]
    represented_momentum_after: tuple[float, float, float]
    momentum_residual: tuple[float, float, float]
    carrier_momentum_source_demand: tuple[float, float, float]
    carrier_coupling_status: str = (
        "one_way_source_demand_only_not_global_carrier_closure"
    )


@dataclass(frozen=True)
class BreakupStepResult:
    state: BreakupParcelState
    event: Literal["none", "kelvin_helmholtz", "rayleigh_taylor"]
    wave: WaveMetrics
    rayleigh_taylor: RayleighTaylorMetrics | None
    conservation: BreakupConservation


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


def wave_coefficient_variant(name: WaveVariantName | str) -> WaveCoefficientVariant:
    try:
        return WAVE_COEFFICIENT_VARIANTS[str(name)]
    except KeyError as exc:
        choices = ", ".join(sorted(WAVE_COEFFICIENT_VARIANTS))
        raise ValueError(f"unknown WAVE coefficient variant {name!r}; choose {choices}") from exc


def gas_weber_number(
    *,
    diameter: float,
    carrier_density: float,
    relative_speed: float,
    surface_tension: float,
) -> float:
    """Return the radius-based gas Weber number used by ReitzKHRT."""

    d = _finite_positive("diameter", diameter)
    rho_g = _finite_positive("carrier_density", carrier_density)
    speed = _finite_nonnegative("relative_speed", relative_speed)
    sigma = _finite_positive("surface_tension", surface_tension)
    return 0.5 * rho_g * speed * speed * d / sigma


def liquid_weber_number(
    *,
    diameter: float,
    liquid_density: float,
    relative_speed: float,
    surface_tension: float,
) -> float:
    """Return the radius-based liquid Weber number used by ReitzKHRT."""

    d = _finite_positive("diameter", diameter)
    rho_l = _finite_positive("liquid_density", liquid_density)
    speed = _finite_nonnegative("relative_speed", relative_speed)
    sigma = _finite_positive("surface_tension", surface_tension)
    return 0.5 * rho_l * speed * speed * d / sigma


def liquid_reynolds_number(
    *,
    diameter: float,
    liquid_density: float,
    liquid_dynamic_viscosity: float,
    relative_speed: float,
) -> float:
    """Return Reitz's radius-based liquid Reynolds number."""

    d = _finite_positive("diameter", diameter)
    rho_l = _finite_positive("liquid_density", liquid_density)
    mu_l = _finite_positive("liquid_dynamic_viscosity", liquid_dynamic_viscosity)
    speed = _finite_nonnegative("relative_speed", relative_speed)
    return rho_l * speed * (0.5 * d) / mu_l


def ohnesorge_number(
    *,
    radius: float,
    liquid_density: float,
    liquid_dynamic_viscosity: float,
    surface_tension: float,
) -> float:
    """Return ``mu/sqrt(rho*r*sigma)``, equal to ``sqrt(We_l)/Re_l``."""

    r = _finite_positive("radius", radius)
    rho_l = _finite_positive("liquid_density", liquid_density)
    mu_l = _finite_positive("liquid_dynamic_viscosity", liquid_dynamic_viscosity)
    sigma = _finite_positive("surface_tension", surface_tension)
    return mu_l / math.sqrt(rho_l * r * sigma)


def taylor_number(*, ohnesorge: float, gas_weber: float) -> float:
    oh = _finite_nonnegative("ohnesorge", ohnesorge)
    we_g = _finite_nonnegative("gas_weber", gas_weber)
    return oh * math.sqrt(we_g)


def kh_wavelength(
    *,
    radius: float,
    gas_weber: float,
    ohnesorge: float,
    taylor: float,
    coefficient_variant: WaveVariantName | str,
) -> float:
    """Fastest-growing Kelvin-Helmholtz wavelength."""

    r = _finite_positive("radius", radius)
    we_g = _finite_nonnegative("gas_weber", gas_weber)
    oh = _finite_nonnegative("ohnesorge", ohnesorge)
    tay = _finite_nonnegative("taylor", taylor)
    variant = wave_coefficient_variant(coefficient_variant)
    numerator = (
        9.02
        * r
        * (1.0 + 0.45 * math.sqrt(oh))
        * (1.0 + variant.taylor_coefficient * tay**0.7)
    )
    denominator = (
        1.0 + variant.weber_denominator_coefficient * we_g**1.67
    ) ** 0.6
    return numerator / denominator


def kh_growth_rate(
    *,
    radius: float,
    gas_weber: float,
    ohnesorge: float,
    taylor: float,
    liquid_density: float,
    surface_tension: float,
) -> float:
    """Frequency of the fastest-growing Kelvin-Helmholtz wave [1/s]."""

    r = _finite_positive("radius", radius)
    we_g = _finite_nonnegative("gas_weber", gas_weber)
    oh = _finite_nonnegative("ohnesorge", ohnesorge)
    tay = _finite_nonnegative("taylor", taylor)
    rho_l = _finite_positive("liquid_density", liquid_density)
    sigma = _finite_positive("surface_tension", surface_tension)
    dimensionless = (0.34 + 0.38 * we_g**1.5) / (
        (1.0 + oh) * (1.0 + 1.4 * tay**0.6)
    )
    return dimensionless * math.sqrt(sigma / (rho_l * r**3))


def kh_breakup_time(
    *, radius: float, wavelength: float, growth_rate: float, b1: float
) -> float:
    r = _finite_positive("radius", radius)
    wavelength = _finite_positive("wavelength", wavelength)
    omega = _finite_positive("growth_rate", growth_rate)
    b1 = _finite_positive("b1", b1)
    return 3.726 * b1 * r / (omega * wavelength)


def kh_stable_diameter(*, wavelength: float, b0: float) -> float:
    wavelength = _finite_positive("wavelength", wavelength)
    b0 = _finite_positive("b0", b0)
    return 2.0 * b0 * wavelength


def compute_wave_metrics(
    *,
    diameter: float,
    liquid_density: float,
    liquid_dynamic_viscosity: float,
    surface_tension: float,
    carrier_density: float,
    relative_speed: float,
    b0: float,
    b1: float,
    coefficient_variant: WaveVariantName | str,
) -> WaveMetrics:
    d = _finite_positive("diameter", diameter)
    r = 0.5 * d
    we_g = gas_weber_number(
        diameter=d,
        carrier_density=carrier_density,
        relative_speed=relative_speed,
        surface_tension=surface_tension,
    )
    we_l = liquid_weber_number(
        diameter=d,
        liquid_density=liquid_density,
        relative_speed=relative_speed,
        surface_tension=surface_tension,
    )
    re_l = liquid_reynolds_number(
        diameter=d,
        liquid_density=liquid_density,
        liquid_dynamic_viscosity=liquid_dynamic_viscosity,
        relative_speed=relative_speed,
    )
    oh = ohnesorge_number(
        radius=r,
        liquid_density=liquid_density,
        liquid_dynamic_viscosity=liquid_dynamic_viscosity,
        surface_tension=surface_tension,
    )
    tay = taylor_number(ohnesorge=oh, gas_weber=we_g)
    variant = wave_coefficient_variant(coefficient_variant)
    wavelength = kh_wavelength(
        radius=r,
        gas_weber=we_g,
        ohnesorge=oh,
        taylor=tay,
        coefficient_variant=variant.name,
    )
    omega = kh_growth_rate(
        radius=r,
        gas_weber=we_g,
        ohnesorge=oh,
        taylor=tay,
        liquid_density=liquid_density,
        surface_tension=surface_tension,
    )
    tau = kh_breakup_time(
        radius=r, wavelength=wavelength, growth_rate=omega, b1=b1
    )
    stable = kh_stable_diameter(wavelength=wavelength, b0=b0)
    return WaveMetrics(
        diameter=d,
        radius=r,
        gas_weber=we_g,
        liquid_weber=we_l,
        liquid_reynolds=re_l,
        ohnesorge=oh,
        taylor=tay,
        wavelength=wavelength,
        growth_rate=omega,
        breakup_time=tau,
        stable_diameter=stable,
        coefficient_variant=variant.name,
    )


def calibrate_wave_constants_from_vof(
    *,
    full_sheet_thickness: float,
    breakup_length: float,
    liquid_velocity: float,
    liquid_density: float,
    liquid_dynamic_viscosity: float,
    surface_tension: float,
    carrier_density: float,
    relative_speed: float,
    coefficient_variant: WaveVariantName | str,
) -> VOFWaveCalibration:
    """Calibrate ``B0`` and ``B1`` from a VOF sheet thickness and length.

    Radhakrishnan et al. define ``h`` in their WAVE equations as half the
    full sheet thickness reported by the VOF tables.  They assume the stable
    child radius equals ``h``, so ``B0=h/lambda``.  The VOF breakup length
    gives ``tau=Lb/Vl``, and ``B1=tau*lambda*omega/(3.726*h)``.
    """

    thickness = _finite_positive("full_sheet_thickness", full_sheet_thickness)
    length = _finite_positive("breakup_length", breakup_length)
    velocity = _finite_positive("liquid_velocity", liquid_velocity)
    h = 0.5 * thickness
    we_g = gas_weber_number(
        diameter=thickness,
        carrier_density=carrier_density,
        relative_speed=relative_speed,
        surface_tension=surface_tension,
    )
    oh = ohnesorge_number(
        radius=h,
        liquid_density=liquid_density,
        liquid_dynamic_viscosity=liquid_dynamic_viscosity,
        surface_tension=surface_tension,
    )
    tay = taylor_number(ohnesorge=oh, gas_weber=we_g)
    variant = wave_coefficient_variant(coefficient_variant)
    wavelength = kh_wavelength(
        radius=h,
        gas_weber=we_g,
        ohnesorge=oh,
        taylor=tay,
        coefficient_variant=variant.name,
    )
    omega = kh_growth_rate(
        radius=h,
        gas_weber=we_g,
        ohnesorge=oh,
        taylor=tay,
        liquid_density=liquid_density,
        surface_tension=surface_tension,
    )
    tau = length / velocity
    b0 = h / wavelength
    b1 = tau * wavelength * omega / (3.726 * h)
    return VOFWaveCalibration(
        full_sheet_thickness=thickness,
        half_sheet_thickness=h,
        breakup_length=length,
        liquid_velocity=velocity,
        wavelength=wavelength,
        growth_rate=omega,
        breakup_time=tau,
        b0=b0,
        b1=b1,
        coefficient_variant=variant.name,
    )


def relax_diameter_backward_euler(
    *, diameter: float, stable_diameter: float, breakup_time: float, dt: float
) -> float:
    """Advance ``dd/dt=(dc-d)/tau`` with semi-implicit backward Euler."""

    d = _finite_positive("diameter", diameter)
    dc = _finite_positive("stable_diameter", stable_diameter)
    tau = _finite_positive("breakup_time", breakup_time)
    dt = _finite_positive("dt", dt)
    fraction = dt / tau
    return (fraction * dc + d) / (1.0 + fraction)


def rayleigh_taylor_metrics(
    *,
    effective_acceleration: float,
    liquid_density: float,
    carrier_density: float,
    surface_tension: float,
    config: RayleighTaylorConfig,
) -> RayleighTaylorMetrics:
    """Return the optional OpenFOAM ReitzKHRT RT scales."""

    if not config.enabled:
        raise ValueError("Rayleigh-Taylor metrics require config.enabled=True")
    acceleration = float(effective_acceleration)
    if not math.isfinite(acceleration):
        raise ValueError("effective_acceleration must be finite")
    rho_l = _finite_positive("liquid_density", liquid_density)
    rho_g = _finite_positive("carrier_density", carrier_density)
    sigma = _finite_positive("surface_tension", surface_tension)
    c_tau = _finite_positive("rayleigh_taylor.c_tau", config.c_tau)
    c_rt = _finite_positive("rayleigh_taylor.c_rt", config.c_rt)
    forcing = abs(acceleration * (rho_l - rho_g))
    if forcing == 0.0:
        return RayleighTaylorMetrics(
            effective_acceleration=acceleration,
            growth_rate=0.0,
            wave_number=0.0,
            wavelength=math.inf,
            breakup_time=math.inf,
            provenance=config.provenance,
        )
    omega = math.sqrt(
        2.0 * forcing**1.5
        / (3.0 * math.sqrt(3.0 * sigma) * (rho_g + rho_l))
    )
    wave_number = math.sqrt(forcing / (3.0 * sigma))
    wavelength = 2.0 * math.pi * c_rt / wave_number
    tau = c_tau / omega
    return RayleighTaylorMetrics(
        effective_acceleration=acceleration,
        growth_rate=omega,
        wave_number=wave_number,
        wavelength=wavelength,
        breakup_time=tau,
        provenance=config.provenance,
    )


def _represented_mass(diameter: float, multiplicity: float, density: float) -> float:
    return multiplicity * density * math.pi * diameter**3 / 6.0


def advance_breakup(
    state: BreakupParcelState,
    *,
    dt: float,
    liquid_density: float,
    liquid_dynamic_viscosity: float,
    surface_tension: float,
    carrier_density: float,
    relative_speed: float,
    config: WaveBreakupConfig,
    effective_acceleration: float | None = None,
) -> BreakupStepResult:
    """Advance one parcel through optional RT then WAVE/KH breakup.

    All material and carrier inputs are explicit.  The RT forcing is the
    signed effective acceleration projected along the trajectory, equivalent
    to OpenFOAM's ``(g + Urel/tMom) & trajectory``.  It is required only when
    the optional RT branch is enabled.
    """

    dt = _finite_positive("dt", dt)
    rho_l = _finite_positive("liquid_density", liquid_density)
    multiplicity = _finite_positive("multiplicity", state.multiplicity)
    velocity = _vector3(state.velocity)
    d_old = _finite_positive("diameter", state.diameter)
    _finite_nonnegative("weber_limit", config.weber_limit)
    metrics = compute_wave_metrics(
        diameter=d_old,
        liquid_density=rho_l,
        liquid_dynamic_viscosity=liquid_dynamic_viscosity,
        surface_tension=surface_tension,
        carrier_density=carrier_density,
        relative_speed=relative_speed,
        b0=config.b0,
        b1=config.b1,
        coefficient_variant=config.coefficient_variant,
    )

    d_new = d_old
    timer = float(state.rt_timer)
    if not (math.isfinite(timer) or timer == -math.inf):
        raise ValueError("rt_timer must be finite or -inf")
    event: Literal["none", "kelvin_helmholtz", "rayleigh_taylor"] = "none"
    rt_metrics: RayleighTaylorMetrics | None = None

    rt_config = config.rayleigh_taylor
    if rt_config.enabled:
        if effective_acceleration is None:
            raise ValueError(
                "effective_acceleration is required when Rayleigh-Taylor is enabled"
            )
        rt_metrics = rayleigh_taylor_metrics(
            effective_acceleration=effective_acceleration,
            liquid_density=rho_l,
            carrier_density=carrier_density,
            surface_tension=surface_tension,
            config=rt_config,
        )
        if timer > 0.0 or rt_metrics.wavelength < d_old:
            timer += dt
        if timer > rt_metrics.breakup_time and rt_metrics.wavelength < d_old:
            n_drops = d_old / rt_metrics.wavelength
            d_new = (d_old**3 / n_drops) ** (1.0 / 3.0)
            timer = -math.inf
            event = "rayleigh_taylor"

    if (
        event == "none"
        and metrics.stable_diameter < d_old
        and metrics.gas_weber > config.weber_limit
    ):
        d_new = relax_diameter_backward_euler(
            diameter=d_old,
            stable_diameter=metrics.stable_diameter,
            breakup_time=metrics.breakup_time,
            dt=dt,
        )
        event = "kelvin_helmholtz"

    # A single parcel continues to represent all child droplets.  Adjusting
    # multiplicity by d^3 exactly preserves represented liquid mass.
    multiplicity_new = multiplicity * (d_old / d_new) ** 3
    mass_before = _represented_mass(d_old, multiplicity, rho_l)
    mass_after = _represented_mass(d_new, multiplicity_new, rho_l)
    momentum_before = tuple(mass_before * component for component in velocity)
    momentum_after = tuple(mass_after * component for component in velocity)
    momentum_residual = tuple(
        before - after
        for before, after in zip(momentum_before, momentum_after, strict=True)
    )
    conservation = BreakupConservation(
        represented_mass_before=mass_before,
        represented_mass_after=mass_after,
        mass_residual=mass_before - mass_after,
        represented_momentum_before=momentum_before,
        represented_momentum_after=momentum_after,
        momentum_residual=momentum_residual,
        carrier_momentum_source_demand=(0.0, 0.0, 0.0),
    )
    return BreakupStepResult(
        state=BreakupParcelState(
            diameter=d_new,
            multiplicity=multiplicity_new,
            velocity=velocity,
            rt_timer=timer,
        ),
        event=event,
        wave=metrics,
        rayleigh_taylor=rt_metrics,
        conservation=conservation,
    )


__all__ = [
    "BreakupConservation",
    "BreakupParcelState",
    "BreakupStepResult",
    "RayleighTaylorConfig",
    "RayleighTaylorMetrics",
    "VOFWaveCalibration",
    "WAVE_COEFFICIENT_VARIANTS",
    "WaveBreakupConfig",
    "WaveCoefficientVariant",
    "WaveMetrics",
    "advance_breakup",
    "calibrate_wave_constants_from_vof",
    "compute_wave_metrics",
    "gas_weber_number",
    "kh_breakup_time",
    "kh_growth_rate",
    "kh_stable_diameter",
    "kh_wavelength",
    "liquid_reynolds_number",
    "liquid_weber_number",
    "ohnesorge_number",
    "rayleigh_taylor_metrics",
    "relax_diameter_backward_euler",
    "taylor_number",
    "wave_coefficient_variant",
]
