"""Explicit steady carrier fields for parcel tracking.

The rectilinear field stores cylindrical velocity components on an axisymmetric
``(axial, radial)`` grid and performs bilinear interpolation without hidden
extrapolation.  Cartesian parcel coordinates use column 0 as the engine axis;
columns 1 and 2 span the radial plane.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .types import CarrierSample, SprayValidationError, _readonly_array


OutOfDomainPolicy = Literal["error", "clip"]


def _query_positions(position) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(position, dtype=float)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise SprayValidationError(
            f"carrier query positions must have shape (..., 3), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SprayValidationError("carrier query positions must be finite")
    return array.reshape(-1, 3), array.shape[:-1]


def _validate_time(time: float) -> float:
    time = float(time)
    if not math.isfinite(time) or time < 0.0:
        raise SprayValidationError("carrier sample time must be finite and >= 0")
    return time


@runtime_checkable
class CarrierField(Protocol):
    """Protocol implemented by every carrier field used by the parcel solver."""

    def sample(self, position, time: float = 0.0) -> CarrierSample:
        ...


@dataclass(frozen=True)
class UniformCarrierField:
    """A spatially and temporally uniform, explicitly specified carrier state."""

    velocity: np.ndarray                 # Cartesian (axial, y, z), m/s
    density: float                       # kg/m^3
    dynamic_viscosity: float             # Pa s
    temperature: float                   # K
    pressure: float                      # absolute Pa
    turbulent_kinetic_energy: float = 0.0
    turbulent_dissipation_rate: float = 0.0

    def __post_init__(self) -> None:
        # CarrierSample owns all state validation and immutable copying.
        sample = CarrierSample(
            velocity=self.velocity,
            density=self.density,
            dynamic_viscosity=self.dynamic_viscosity,
            temperature=self.temperature,
            pressure=self.pressure,
            turbulent_kinetic_energy=self.turbulent_kinetic_energy,
            turbulent_dissipation_rate=self.turbulent_dissipation_rate,
        )
        if sample.velocity.shape != (3,):
            raise SprayValidationError("uniform carrier velocity must have shape (3,)")
        for name in (
            "velocity",
            "density",
            "dynamic_viscosity",
            "temperature",
            "pressure",
            "turbulent_kinetic_energy",
            "turbulent_dissipation_rate",
        ):
            value = getattr(sample, name)
            if isinstance(value, np.ndarray) and value.shape == ():
                value = float(value)
            object.__setattr__(self, name, value)

    def sample(self, position, time: float = 0.0) -> CarrierSample:
        _validate_time(time)
        _, shape = _query_positions(position)
        velocity = np.broadcast_to(np.asarray(self.velocity), shape + (3,))
        return CarrierSample(
            velocity=velocity,
            density=np.broadcast_to(self.density, shape),
            dynamic_viscosity=np.broadcast_to(self.dynamic_viscosity, shape),
            temperature=np.broadcast_to(self.temperature, shape),
            pressure=np.broadcast_to(self.pressure, shape),
            turbulent_kinetic_energy=np.broadcast_to(
                self.turbulent_kinetic_energy, shape
            ),
            turbulent_dissipation_rate=np.broadcast_to(
                self.turbulent_dissipation_rate, shape
            ),
        )


def _strict_axis(value, *, name: str, nonnegative: bool = False) -> np.ndarray:
    axis = _readonly_array(value, name=name, dtype=float, ndim=1)
    if axis.size < 2:
        raise SprayValidationError(f"{name} must contain at least two coordinates")
    if np.any(np.diff(axis) <= 0.0):
        raise SprayValidationError(f"{name} must be strictly increasing")
    if nonnegative and np.any(axis < 0.0):
        raise SprayValidationError(f"{name} must be >= 0")
    return axis


@dataclass(frozen=True)
class AxisymmetricRectilinearCarrierField:
    """Steady axisymmetric carrier data on a rectilinear ``(x, r)`` grid.

    ``velocity_cylindrical[..., :]`` stores ``(u_x, u_r, u_theta)``.  All
    remaining fields have shape ``(n_x, n_r)``.  The only supported
    out-of-domain policies are hard error and explicit clipping; numerical
    extrapolation is never performed.
    """

    axial_coordinates: np.ndarray
    radial_coordinates: np.ndarray
    velocity_cylindrical: np.ndarray
    density: np.ndarray
    dynamic_viscosity: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    turbulent_kinetic_energy: np.ndarray
    turbulent_dissipation_rate: np.ndarray
    out_of_domain_policy: OutOfDomainPolicy = "error"

    def __post_init__(self) -> None:
        x = _strict_axis(self.axial_coordinates, name="axial_coordinates")
        r = _strict_axis(
            self.radial_coordinates, name="radial_coordinates", nonnegative=True
        )
        object.__setattr__(self, "axial_coordinates", x)
        object.__setattr__(self, "radial_coordinates", r)
        shape = (x.size, r.size)

        velocity = _readonly_array(
            self.velocity_cylindrical,
            name="velocity_cylindrical",
            dtype=float,
            ndim=3,
        )
        if velocity.shape != shape + (3,):
            raise SprayValidationError(
                f"velocity_cylindrical must have shape {shape + (3,)}, "
                f"got {velocity.shape}"
            )
        object.__setattr__(self, "velocity_cylindrical", velocity)

        scalar_names = (
            "density",
            "dynamic_viscosity",
            "temperature",
            "pressure",
            "turbulent_kinetic_energy",
            "turbulent_dissipation_rate",
        )
        scalars: dict[str, np.ndarray] = {}
        for name in scalar_names:
            value = _readonly_array(getattr(self, name), name=name, dtype=float, ndim=2)
            if value.shape != shape:
                raise SprayValidationError(
                    f"{name} must have shape {shape}, got {value.shape}"
                )
            scalars[name] = value
            object.__setattr__(self, name, value)

        for name in ("density", "dynamic_viscosity", "temperature", "pressure"):
            if np.any(scalars[name] <= 0.0):
                raise SprayValidationError(f"{name} must be > 0 everywhere")
        k = scalars["turbulent_kinetic_energy"]
        epsilon = scalars["turbulent_dissipation_rate"]
        if np.any(k < 0.0) or np.any(epsilon < 0.0):
            raise SprayValidationError(
                "turbulent kinetic energy and dissipation rate must be >= 0"
            )
        if np.any((k > 0.0) & (epsilon <= 0.0)):
            raise SprayValidationError(
                "positive turbulent kinetic energy requires positive dissipation rate"
            )

        policy = str(self.out_of_domain_policy).lower()
        if policy not in {"error", "clip"}:
            raise SprayValidationError(
                "out_of_domain_policy must be 'error' or 'clip'"
            )
        object.__setattr__(self, "out_of_domain_policy", policy)

        # At the symmetry axis radial and azimuthal directions are undefined;
        # their velocities must tend to zero for a single-valued field.
        if r[0] == 0.0 and not np.allclose(
            velocity[:, 0, 1:], 0.0, rtol=0.0, atol=1.0e-14
        ):
            raise SprayValidationError(
                "radial and azimuthal velocity must be zero on the symmetry axis"
            )

    @staticmethod
    def _bilinear(
        values: np.ndarray,
        ix: np.ndarray,
        ir: np.ndarray,
        fx: np.ndarray,
        fr: np.ndarray,
    ) -> np.ndarray:
        v00 = values[ix, ir]
        v10 = values[ix + 1, ir]
        v01 = values[ix, ir + 1]
        v11 = values[ix + 1, ir + 1]
        if values.ndim == 3:
            fx = fx[:, None]
            fr = fr[:, None]
        return (
            (1.0 - fx) * (1.0 - fr) * v00
            + fx * (1.0 - fr) * v10
            + (1.0 - fx) * fr * v01
            + fx * fr * v11
        )

    def sample(self, position, time: float = 0.0) -> CarrierSample:
        _validate_time(time)
        points, leading_shape = _query_positions(position)
        xq = points[:, 0].copy()
        physical_radius = np.hypot(points[:, 1], points[:, 2])
        rq = physical_radius.copy()
        x_axis = self.axial_coordinates
        r_axis = self.radial_coordinates

        outside = (
            (xq < x_axis[0])
            | (xq > x_axis[-1])
            | (rq < r_axis[0])
            | (rq > r_axis[-1])
        )
        if np.any(outside) and self.out_of_domain_policy == "error":
            first = int(np.flatnonzero(outside)[0])
            raise SprayValidationError(
                "carrier query outside tabulated domain at flattened point "
                f"{first}: x={xq[first]:.9g}, r={rq[first]:.9g}; domain "
                f"x=[{x_axis[0]:.9g}, {x_axis[-1]:.9g}], "
                f"r=[{r_axis[0]:.9g}, {r_axis[-1]:.9g}]"
            )
        if self.out_of_domain_policy == "clip":
            xq = np.clip(xq, x_axis[0], x_axis[-1])
            rq = np.clip(rq, r_axis[0], r_axis[-1])

        ix = np.searchsorted(x_axis, xq, side="right") - 1
        ir = np.searchsorted(r_axis, rq, side="right") - 1
        ix = np.clip(ix, 0, x_axis.size - 2)
        ir = np.clip(ir, 0, r_axis.size - 2)
        fx = (xq - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
        fr = (rq - r_axis[ir]) / (r_axis[ir + 1] - r_axis[ir])

        cylindrical = self._bilinear(
            self.velocity_cylindrical, ix, ir, fx, fr
        )
        ux, ur, ut = cylindrical.T
        # Clipping changes only the field lookup radius, never the physical
        # azimuthal basis of the Cartesian query point.
        cos_theta = np.zeros_like(physical_radius)
        sin_theta = np.zeros_like(physical_radius)
        non_axis = physical_radius > 0.0
        cos_theta[non_axis] = points[non_axis, 1] / physical_radius[non_axis]
        sin_theta[non_axis] = points[non_axis, 2] / physical_radius[non_axis]
        cartesian = np.column_stack((
            ux,
            ur * cos_theta - ut * sin_theta,
            ur * sin_theta + ut * cos_theta,
        )).reshape(leading_shape + (3,))

        def scalar(name: str) -> np.ndarray:
            return self._bilinear(
                getattr(self, name), ix, ir, fx, fr
            ).reshape(leading_shape)

        return CarrierSample(
            velocity=cartesian,
            density=scalar("density"),
            dynamic_viscosity=scalar("dynamic_viscosity"),
            temperature=scalar("temperature"),
            pressure=scalar("pressure"),
            turbulent_kinetic_energy=scalar("turbulent_kinetic_energy"),
            turbulent_dissipation_rate=scalar("turbulent_dissipation_rate"),
        )


def carrier_field_fingerprint(carrier: CarrierField) -> str | None:
    """Return a deterministic SHA-256 for built-in prescribed carrier fields.

    Unknown protocol implementations return ``None``.  They remain usable by
    the standalone march, but a cycle handoff must then fail field traceability
    rather than trusting a caller label.
    """

    if isinstance(carrier, UniformCarrierField):
        payload = {
            "type": "uniform",
            "velocity_m_s": np.asarray(carrier.velocity).tolist(),
            "density_kg_m3": float(carrier.density),
            "dynamic_viscosity_pa_s": float(carrier.dynamic_viscosity),
            "temperature_k": float(carrier.temperature),
            "pressure_pa": float(carrier.pressure),
            "turbulent_kinetic_energy_m2_s2": float(
                carrier.turbulent_kinetic_energy
            ),
            "turbulent_dissipation_rate_m2_s3": float(
                carrier.turbulent_dissipation_rate
            ),
        }
    elif isinstance(carrier, AxisymmetricRectilinearCarrierField):
        payload = {
            "type": "axisymmetric_rectilinear",
            "axial_coordinates_m": carrier.axial_coordinates.tolist(),
            "radial_coordinates_m": carrier.radial_coordinates.tolist(),
            "velocity_cylindrical_m_s": carrier.velocity_cylindrical.tolist(),
            "density_kg_m3": carrier.density.tolist(),
            "dynamic_viscosity_pa_s": carrier.dynamic_viscosity.tolist(),
            "temperature_k": carrier.temperature.tolist(),
            "pressure_pa": carrier.pressure.tolist(),
            "turbulent_kinetic_energy_m2_s2": (
                carrier.turbulent_kinetic_energy.tolist()
            ),
            "turbulent_dissipation_rate_m2_s3": (
                carrier.turbulent_dissipation_rate.tolist()
            ),
            "out_of_domain_policy": carrier.out_of_domain_policy,
        }
    else:
        return None
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "AxisymmetricRectilinearCarrierField",
    "CarrierField",
    "OutOfDomainPolicy",
    "UniformCarrierField",
    "carrier_field_fingerprint",
]
