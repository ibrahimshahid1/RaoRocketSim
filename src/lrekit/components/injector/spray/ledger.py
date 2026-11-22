"""Mass and momentum accounting for represented Lagrangian droplets.

The parcel solver currently accepts a prescribed one-way carrier field.  It
therefore closes the *parcel* momentum equation and reports the equal/opposite
carrier impulse as a source demand, but it must not claim globally coupled
momentum conservation until an Eulerian carrier actually consumes that source.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def represented_mass(diameter, multiplicity, liquid_density: float) -> np.ndarray:
    if not math.isfinite(liquid_density) or liquid_density <= 0.0:
        raise ValueError("liquid_density must be finite and > 0")
    d = np.asarray(diameter, dtype=float)
    n = np.asarray(multiplicity, dtype=float)
    if d.shape != n.shape:
        raise ValueError("diameter and multiplicity must have equal shapes")
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("diameter must be finite and >= 0")
    if np.any(~np.isfinite(n)) or np.any(n < 0.0):
        raise ValueError("multiplicity must be finite and >= 0")
    return liquid_density * (np.pi / 6.0) * n * d**3


def represented_momentum(mass, velocity) -> np.ndarray:
    m = np.asarray(mass, dtype=float).reshape(-1)
    u = np.asarray(velocity, dtype=float)
    if u.shape != (m.size, 3):
        raise ValueError("velocity must have shape (n, 3) matching mass")
    if np.any(~np.isfinite(m)) or np.any(m < 0.0) or np.any(~np.isfinite(u)):
        raise ValueError("mass/velocity must be finite and mass >= 0")
    return np.sum(m[:, None] * u, axis=0)


def _vec3(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


@dataclass(frozen=True)
class ReservoirLedger:
    role: str
    injected_mass: float
    active_mass: float
    vaporized_mass: float
    wall_mass: float
    exit_mass: float
    mass_residual: float
    mass_relative_residual: float
    initial_momentum: tuple[float, float, float]
    active_momentum: tuple[float, float, float]
    vapor_momentum: tuple[float, float, float]
    wall_momentum: tuple[float, float, float]
    exit_momentum: tuple[float, float, float]
    drag_impulse_on_parcels: tuple[float, float, float]
    body_force_impulse: tuple[float, float, float]
    parcel_momentum_residual: tuple[float, float, float]
    parcel_momentum_relative_residual: float

    def mass_closed(self, tolerance: float) -> bool:
        return self.mass_relative_residual <= tolerance

    def parcel_momentum_closed(self, tolerance: float) -> bool:
        return self.parcel_momentum_relative_residual <= tolerance

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "injected_mass_kg": self.injected_mass,
            "active_mass_kg": self.active_mass,
            "vaporized_mass_kg": self.vaporized_mass,
            "wall_mass_kg": self.wall_mass,
            "exit_mass_kg": self.exit_mass,
            "mass_residual_kg": self.mass_residual,
            "mass_relative_residual": self.mass_relative_residual,
            "initial_momentum_kg_m_s": list(self.initial_momentum),
            "active_momentum_kg_m_s": list(self.active_momentum),
            "vapor_momentum_kg_m_s": list(self.vapor_momentum),
            "wall_momentum_kg_m_s": list(self.wall_momentum),
            "exit_momentum_kg_m_s": list(self.exit_momentum),
            "drag_impulse_on_parcels_n_s": list(self.drag_impulse_on_parcels),
            "carrier_reaction_impulse_demand_n_s": [
                -item for item in self.drag_impulse_on_parcels
            ],
            "body_force_impulse_n_s": list(self.body_force_impulse),
            "parcel_momentum_residual_kg_m_s": list(
                self.parcel_momentum_residual
            ),
            "parcel_momentum_relative_residual": (
                self.parcel_momentum_relative_residual
            ),
        }


def close_reservoir_ledger(
    *,
    role: str,
    injected_mass: float,
    active_mass: float,
    vaporized_mass: float,
    wall_mass: float,
    exit_mass: float,
    initial_momentum,
    active_momentum,
    vapor_momentum,
    wall_momentum,
    exit_momentum,
    drag_impulse_on_parcels,
    body_force_impulse=(0.0, 0.0, 0.0),
) -> ReservoirLedger:
    masses = np.asarray(
        [injected_mass, active_mass, vaporized_mass, wall_mass, exit_mass],
        dtype=float,
    )
    if np.any(~np.isfinite(masses)) or np.any(masses < 0.0):
        raise ValueError("all reservoir masses must be finite and >= 0")
    initial = _vec3(initial_momentum, "initial_momentum")
    active = _vec3(active_momentum, "active_momentum")
    vapor = _vec3(vapor_momentum, "vapor_momentum")
    wall = _vec3(wall_momentum, "wall_momentum")
    exit_p = _vec3(exit_momentum, "exit_momentum")
    drag = _vec3(drag_impulse_on_parcels, "drag_impulse_on_parcels")
    body = _vec3(body_force_impulse, "body_force_impulse")

    mass_residual = float(
        injected_mass - active_mass - vaporized_mass - wall_mass - exit_mass
    )
    mass_scale = max(float(injected_mass), np.finfo(float).tiny)
    mass_relative = abs(mass_residual) / mass_scale

    final_momentum = active + vapor + wall + exit_p
    momentum_residual = initial + drag + body - final_momentum
    momentum_scale = max(
        float(np.linalg.norm(initial))
        + float(np.linalg.norm(drag))
        + float(np.linalg.norm(body)),
        np.finfo(float).tiny,
    )
    momentum_relative = float(np.linalg.norm(momentum_residual) / momentum_scale)

    return ReservoirLedger(
        role=str(role),
        injected_mass=float(injected_mass),
        active_mass=float(active_mass),
        vaporized_mass=float(vaporized_mass),
        wall_mass=float(wall_mass),
        exit_mass=float(exit_mass),
        mass_residual=mass_residual,
        mass_relative_residual=mass_relative,
        initial_momentum=tuple(float(v) for v in initial),
        active_momentum=tuple(float(v) for v in active),
        vapor_momentum=tuple(float(v) for v in vapor),
        wall_momentum=tuple(float(v) for v in wall),
        exit_momentum=tuple(float(v) for v in exit_p),
        drag_impulse_on_parcels=tuple(float(v) for v in drag),
        body_force_impulse=tuple(float(v) for v in body),
        parcel_momentum_residual=tuple(float(v) for v in momentum_residual),
        parcel_momentum_relative_residual=momentum_relative,
    )


@dataclass(frozen=True)
class ConservationLedger:
    per_role: Mapping[str, ReservoirLedger]
    mass_tolerance: float
    momentum_tolerance: float
    carrier_coupling: str = "one_way_prescribed"
    energy_status: str = "not_evaluated_no_droplet_energy_equation"

    @property
    def mass_closed(self) -> bool:
        return bool(self.per_role) and all(
            item.mass_closed(self.mass_tolerance) for item in self.per_role.values()
        )

    @property
    def parcel_momentum_closed(self) -> bool:
        return bool(self.per_role) and all(
            item.parcel_momentum_closed(self.momentum_tolerance)
            for item in self.per_role.values()
        )

    @property
    def globally_momentum_closed(self) -> bool:
        # A prescribed carrier does not consume the equal/opposite source.
        return self.parcel_momentum_closed and self.carrier_coupling == "two_way"

    def to_dict(self) -> dict:
        return {
            "mass_closed": self.mass_closed,
            "parcel_momentum_closed": self.parcel_momentum_closed,
            "globally_momentum_closed": self.globally_momentum_closed,
            "carrier_coupling": self.carrier_coupling,
            "carrier_momentum_status": (
                "closed_two_way"
                if self.globally_momentum_closed
                else "one_way_source_demand_unapplied"
            ),
            "energy_status": self.energy_status,
            "mass_tolerance": self.mass_tolerance,
            "momentum_tolerance": self.momentum_tolerance,
            "per_role": {
                role: ledger.to_dict() for role, ledger in self.per_role.items()
            },
        }


__all__ = [
    "ConservationLedger",
    "ReservoirLedger",
    "close_reservoir_ledger",
    "represented_mass",
    "represented_momentum",
]
