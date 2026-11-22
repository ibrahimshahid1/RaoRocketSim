"""One-way spherical-parcel drag using the Schiller--Naumann correlation."""

from __future__ import annotations

import numpy as np

from .types import CarrierSample, SprayValidationError


def _scalar_if_scalar(value: np.ndarray):
    return float(value) if value.shape == () else value


def schiller_naumann_drag_coefficient(reynolds_number):
    """Return spherical-drop drag coefficient for nonnegative particle Re.

    ``Cd = 24/Re * (1 + 0.15 Re**0.687)`` for ``0 < Re <= 1000`` and
    ``Cd = 0.44`` above 1000.  The mathematical Stokes limit is ``inf`` at
    exactly zero Reynolds number; :func:`drag_acceleration` handles zero slip
    separately and returns exactly zero acceleration.
    """

    re = np.asarray(reynolds_number, dtype=float)
    if np.any(~np.isfinite(re)) or np.any(re < 0.0):
        raise SprayValidationError("particle Reynolds number must be finite and >= 0")
    cd = np.empty_like(re)
    zero = re == 0.0
    low = (re > 0.0) & (re <= 1000.0)
    high = re > 1000.0
    cd[zero] = np.inf
    cd[low] = 24.0 / re[low] * (1.0 + 0.15 * re[low] ** 0.687)
    cd[high] = 0.44
    return _scalar_if_scalar(cd)


def particle_reynolds_number(
    carrier_density,
    carrier_dynamic_viscosity,
    diameter,
    relative_speed,
):
    """Return ``Re_p = rho_g |u_g-u_p| d / mu_g`` with strict SI inputs."""

    rho, mu, d, speed = np.broadcast_arrays(
        np.asarray(carrier_density, dtype=float),
        np.asarray(carrier_dynamic_viscosity, dtype=float),
        np.asarray(diameter, dtype=float),
        np.asarray(relative_speed, dtype=float),
    )
    if not all(np.all(np.isfinite(item)) for item in (rho, mu, d, speed)):
        raise SprayValidationError("drag inputs must be finite")
    if np.any(rho <= 0.0) or np.any(mu <= 0.0) or np.any(d <= 0.0):
        raise SprayValidationError(
            "carrier density, viscosity, and parcel diameter must be > 0"
        )
    if np.any(speed < 0.0):
        raise SprayValidationError("relative speed must be >= 0")
    result = rho * speed * d / mu
    return _scalar_if_scalar(result)


def stokes_relaxation_time(liquid_density, diameter, carrier_dynamic_viscosity):
    """Return spherical-particle Stokes relaxation time ``rho_l d^2/(18 mu_g)``."""

    rho_l, d, mu = np.broadcast_arrays(
        np.asarray(liquid_density, dtype=float),
        np.asarray(diameter, dtype=float),
        np.asarray(carrier_dynamic_viscosity, dtype=float),
    )
    if not all(np.all(np.isfinite(item)) for item in (rho_l, d, mu)):
        raise SprayValidationError("Stokes relaxation inputs must be finite")
    if np.any(rho_l <= 0.0) or np.any(d <= 0.0) or np.any(mu <= 0.0):
        raise SprayValidationError(
            "liquid density, parcel diameter, and gas viscosity must be > 0"
        )
    tau = rho_l * d**2 / (18.0 * mu)
    return _scalar_if_scalar(tau)


def drag_acceleration(
    parcel_velocity,
    diameter,
    liquid_density,
    carrier: CarrierSample,
) -> np.ndarray:
    """Return parcel acceleration from one-way spherical aerodynamic drag.

    The acceleration is

    ``a = 3 Cd rho_g |u_rel| u_rel / (4 rho_l d)``.

    Carrier reaction momentum is deliberately outside this one-way microkernel;
    the eventual parcel solver must enter the opposite impulse in its source
    and conservation ledgers.
    """

    velocity = np.asarray(parcel_velocity, dtype=float)
    if velocity.ndim < 1 or velocity.shape[-1] != 3:
        raise SprayValidationError(
            f"parcel_velocity must have shape (..., 3), got {velocity.shape}"
        )
    if not np.all(np.isfinite(velocity)):
        raise SprayValidationError("parcel_velocity must be finite")
    leading = velocity.shape[:-1]
    try:
        carrier_velocity = np.broadcast_to(carrier.velocity, velocity.shape)
        rho_g = np.broadcast_to(carrier.density, leading)
        mu_g = np.broadcast_to(carrier.dynamic_viscosity, leading)
        d = np.broadcast_to(np.asarray(diameter, dtype=float), leading)
        rho_l = np.broadcast_to(np.asarray(liquid_density, dtype=float), leading)
    except ValueError as exc:
        raise SprayValidationError(
            "parcel, liquid, and carrier shapes are not broadcast-compatible"
        ) from exc
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(rho_l)):
        raise SprayValidationError("diameter and liquid density must be finite")
    if np.any(d <= 0.0) or np.any(rho_l <= 0.0):
        raise SprayValidationError("diameter and liquid density must be > 0")

    relative_velocity = carrier_velocity - velocity
    speed = np.linalg.norm(relative_velocity, axis=-1)
    re = np.asarray(particle_reynolds_number(rho_g, mu_g, d, speed))
    cd = np.asarray(schiller_naumann_drag_coefficient(re))
    multiplier = np.zeros(leading, dtype=float)
    moving = speed > 0.0
    multiplier[moving] = (
        0.75
        * cd[moving]
        * rho_g[moving]
        * speed[moving]
        / (rho_l[moving] * d[moving])
    )
    return multiplier[..., None] * relative_velocity


__all__ = [
    "drag_acceleration",
    "particle_reynolds_number",
    "schiller_naumann_drag_coefficient",
    "stokes_relaxation_time",
]
