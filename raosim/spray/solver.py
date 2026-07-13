"""Deterministic one-way Lagrangian parcel march.

This module connects the geometry source, prescribed Eulerian carrier field,
WAVE/KH breakup, turbulent dispersion, Spalding evaporation, trajectory
events, droplet statistics, and conservation ledgers.  It is intentionally a
mid-tier engineering model rather than a CFD solver:

* the carrier is prescribed and does not consume parcel source terms;
* droplet temperature and carrier energy are not solved;
* primary-model and literature-readiness gates remain visible; and
* consequently a successful numerical march is not automatically eligible to
  alter the engine cycle.

All dimensional inputs are SI.  No propellant transport property is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np

from .breakup import (
    BreakupParcelState,
    WaveBreakupConfig,
    advance_breakup,
)
from .carrier import CarrierField, carrier_field_fingerprint
from .dispersion import DiscreteRandomWalk, DispersionState
from .domain import AxisymmetricDomain
from .drag import drag_acceleration
from .evaporation import (
    EvaporationParcelState,
    SherwoodClosureName,
    advance_evaporation,
    sherwood_closure,
)
from .ledger import (
    ConservationLedger,
    close_reservoir_ledger,
    represented_mass,
    represented_momentum,
)
from .primary import PrimaryParcelInitialization
from .statistics import SprayStatistics, summarize_spray
from .types import CarrierSample, ParcelCloud, SpraySolverSpec, SprayValidationError


TerminalReason = Literal["active", "vaporized", "wall", "outlet", "inlet"]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SprayValidationError(f"{name} must be finite and > 0")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise SprayValidationError(f"{name} must be finite and >= 0")
    return value


def _vector3(name: str, value) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise SprayValidationError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class EvaporationModelConfig:
    """Explicit transport closure required by the 2021 Eq. (16) march."""

    mass_diffusivity: float
    spalding_mass_number: float
    sherwood_closure: SherwoodClosureName

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mass_diffusivity",
            _positive("mass_diffusivity", self.mass_diffusivity),
        )
        object.__setattr__(
            self,
            "spalding_mass_number",
            _nonnegative("spalding_mass_number", self.spalding_mass_number),
        )
        selected = sherwood_closure(self.sherwood_closure)
        object.__setattr__(self, "sherwood_closure", selected.name)


@dataclass(frozen=True)
class SprayMarchConfig:
    """Numerical recording and conservation policy for one parcel march."""

    body_acceleration: tuple[float, float, float]
    sampling_planes: tuple[float, ...]
    history_stride: int
    mass_tolerance: float
    momentum_tolerance: float
    strict_conservation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "body_acceleration", _vector3("body_acceleration", self.body_acceleration)
        )
        planes = tuple(float(value) for value in self.sampling_planes)
        if any(not math.isfinite(value) for value in planes):
            raise SprayValidationError("sampling planes must be finite")
        if len(set(planes)) != len(planes):
            raise SprayValidationError("sampling planes must be unique")
        object.__setattr__(self, "sampling_planes", tuple(sorted(planes)))
        if (
            isinstance(self.history_stride, (bool, np.bool_))
            or not isinstance(self.history_stride, (int, np.integer))
            or int(self.history_stride) < 1
        ):
            raise SprayValidationError("history_stride must be an integer >= 1")
        object.__setattr__(self, "history_stride", int(self.history_stride))
        object.__setattr__(
            self, "mass_tolerance", _positive("mass_tolerance", self.mass_tolerance)
        )
        object.__setattr__(
            self,
            "momentum_tolerance",
            _positive("momentum_tolerance", self.momentum_tolerance),
        )
        if not isinstance(self.strict_conservation, (bool, np.bool_)):
            raise SprayValidationError("strict_conservation must be boolean")
        object.__setattr__(self, "strict_conservation", bool(self.strict_conservation))


@dataclass(frozen=True)
class SprayGate:
    name: str
    status: Literal["pass", "warn", "fail", "info"]
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warn", "fail", "info"}:
            raise SprayValidationError(f"invalid spray gate status {self.status!r}")
        if not str(self.name).strip() or not str(self.detail).strip():
            raise SprayValidationError("spray gate name/detail must be nonblank")

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "info"}

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class TrajectoryFrame:
    time: float
    cloud: ParcelCloud
    terminal_reason: tuple[TerminalReason, ...]

    def __post_init__(self) -> None:
        time = float(self.time)
        if not math.isfinite(time) or time < 0.0:
            raise SprayValidationError("trajectory-frame time must be finite and >= 0")
        reasons = tuple(self.terminal_reason)
        if len(reasons) != self.cloud.count:
            raise SprayValidationError("one terminal reason is required per parcel")
        allowed = {"active", "vaporized", "wall", "outlet", "inlet"}
        if any(reason not in allowed for reason in reasons):
            raise SprayValidationError("trajectory frame contains an invalid terminal reason")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "terminal_reason", reasons)


@dataclass(frozen=True)
class SamplingPlaneCloud:
    """First liquid crossing of one axial sampling plane per parcel."""

    axial_position: float
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    diameter: np.ndarray
    statistical_weight: np.ndarray
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        x = float(self.axial_position)
        if not math.isfinite(x):
            raise SprayValidationError("sampling-plane position must be finite")
        object.__setattr__(self, "axial_position", x)
        arrays = {
            "time": np.asarray(self.time, dtype=float),
            "position": np.asarray(self.position, dtype=float),
            "velocity": np.asarray(self.velocity, dtype=float),
            "diameter": np.asarray(self.diameter, dtype=float),
            "statistical_weight": np.asarray(self.statistical_weight, dtype=float),
        }
        n = arrays["time"].size
        if arrays["time"].shape != (n,):
            raise SprayValidationError("sampling time must be one-dimensional")
        for name in ("diameter", "statistical_weight"):
            if arrays[name].shape != (n,):
                raise SprayValidationError(f"sampling {name} must have shape ({n},)")
        for name in ("position", "velocity"):
            if arrays[name].shape != (n, 3):
                raise SprayValidationError(f"sampling {name} must have shape ({n}, 3)")
        if any(np.any(~np.isfinite(value)) for value in arrays.values()):
            raise SprayValidationError("sampling-plane arrays must be finite")
        if np.any(arrays["diameter"] <= 0.0) or np.any(
            arrays["statistical_weight"] <= 0.0
        ):
            raise SprayValidationError("sampled liquid diameter/weight must be > 0")
        roles = tuple(str(role) for role in self.roles)
        if len(roles) != n or any(not role for role in roles):
            raise SprayValidationError("sampling roles must match the sample count")
        for name, value in arrays.items():
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "roles", roles)

    @property
    def count(self) -> int:
        return int(self.time.size)

    def statistics(self, role: str) -> SprayStatistics | None:
        mask = np.asarray([item == role for item in self.roles], dtype=bool)
        if not np.any(mask):
            return None
        return summarize_spray(
            self.diameter[mask], self.statistical_weight[mask], fit_rr=True
        )

    def to_dict(self) -> dict:
        statistics = {
            role: value.to_dict()
            for role in sorted(set(self.roles))
            if (value := self.statistics(role)) is not None
        }
        return {
            "axial_position_m": self.axial_position,
            "sample_count": self.count,
            "statistics_by_role": statistics,
        }


@dataclass(frozen=True)
class SprayMarchResult:
    final_cloud: ParcelCloud
    terminal_reason: tuple[TerminalReason, ...]
    history: tuple[TrajectoryFrame, ...]
    sampling_planes: Mapping[float, SamplingPlaneCloud]
    conservation: ConservationLedger
    eta_vaporization_by_role: Mapping[str, float]
    eta_vaporization: float
    breakup_event_counts: Mapping[str, Mapping[str, int]]
    solver_metadata: Mapping[str, object]
    gates: tuple[SprayGate, ...]

    def __post_init__(self) -> None:
        reasons = tuple(self.terminal_reason)
        if len(reasons) != self.final_cloud.count:
            raise SprayValidationError("one terminal reason is required per final parcel")
        object.__setattr__(self, "terminal_reason", reasons)
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "sampling_planes", MappingProxyType(dict(self.sampling_planes)))
        object.__setattr__(
            self,
            "eta_vaporization_by_role",
            MappingProxyType(dict(self.eta_vaporization_by_role)),
        )
        object.__setattr__(
            self,
            "breakup_event_counts",
            MappingProxyType({
                role: MappingProxyType(dict(counts))
                for role, counts in self.breakup_event_counts.items()
            }),
        )
        object.__setattr__(self, "solver_metadata", MappingProxyType(dict(self.solver_metadata)))
        object.__setattr__(self, "gates", tuple(self.gates))

    @property
    def coupling_eligible(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    @property
    def all_streams_accounted(self) -> bool:
        # This solver receives liquid sources and a field, not carrier species
        # continuity.  The eventual typed cycle handoff must close that gap.
        return False

    def statistics(self, *, role: str, reservoir: TerminalReason) -> SprayStatistics | None:
        mask = np.asarray(
            [
                item_role == role and reason == reservoir and diameter > 0.0
                for item_role, reason, diameter in zip(
                    self.final_cloud.roles,
                    self.terminal_reason,
                    self.final_cloud.diameter,
                    strict=True,
                )
            ],
            dtype=bool,
        )
        if not np.any(mask):
            return None
        return summarize_spray(
            self.final_cloud.diameter[mask],
            self.final_cloud.statistical_weight[mask],
            fit_rr=True,
        )

    def to_dict(self) -> dict:
        roles = sorted(set(self.final_cloud.roles))
        reservoirs = ("active", "wall", "outlet", "inlet")
        terminal_statistics: dict[str, dict] = {}
        for reservoir in reservoirs:
            per_role = {}
            for role in roles:
                value = self.statistics(role=role, reservoir=reservoir)
                if value is not None:
                    per_role[role] = value.to_dict()
            if per_role:
                terminal_statistics[reservoir] = per_role
        return {
            "model": "deterministic_one_way_lagrangian_parcel_march_v1",
            "coupling_eligible": self.coupling_eligible,
            "all_streams_accounted": self.all_streams_accounted,
            "eta_vaporization": self.eta_vaporization,
            "eta_vaporization_by_role": dict(self.eta_vaporization_by_role),
            "terminal_counts": {
                reason: self.terminal_reason.count(reason)
                for reason in sorted(set(self.terminal_reason))
            },
            "terminal_statistics": terminal_statistics,
            "sampling_planes": {
                f"{position:.17g}": sample.to_dict()
                for position, sample in self.sampling_planes.items()
            },
            "breakup_event_counts": {
                role: dict(counts) for role, counts in self.breakup_event_counts.items()
            },
            "conservation": self.conservation.to_dict(),
            "solver_metadata": dict(self.solver_metadata),
            "gates": [gate.to_dict() for gate in self.gates],
        }


class SprayConservationError(RuntimeError):
    """Raised when the strict parcel conservation gate fails."""


def _merge_initializations(
    initializations: Sequence[PrimaryParcelInitialization],
    spec: SpraySolverSpec,
) -> tuple[ParcelCloud, dict[str, PrimaryParcelInitialization]]:
    items = tuple(initializations)
    if not items:
        raise SprayValidationError("at least one liquid primary initialization is required")
    by_role: dict[str, PrimaryParcelInitialization] = {}
    for item in items:
        if not isinstance(item, PrimaryParcelInitialization):
            raise SprayValidationError(
                "initializations must contain PrimaryParcelInitialization values"
            )
        if item.role in by_role:
            raise SprayValidationError(f"duplicate liquid source role {item.role!r}")
        if item.cloud.count != spec.parcels_per_liquid_stream:
            raise SprayValidationError(
                f"role {item.role!r} has {item.cloud.count} parcels but solver spec "
                f"requires {spec.parcels_per_liquid_stream}"
            )
        by_role[item.role] = item
    return ParcelCloud(
        position=np.concatenate([item.cloud.position for item in items]),
        velocity=np.concatenate([item.cloud.velocity for item in items]),
        diameter=np.concatenate([item.cloud.diameter for item in items]),
        temperature=np.concatenate([item.cloud.temperature for item in items]),
        statistical_weight=np.concatenate(
            [item.cloud.statistical_weight for item in items]
        ),
        roles=tuple(role for item in items for role in item.cloud.roles),
    ), by_role


def _effective_carrier(sample: CarrierSample, velocity: np.ndarray) -> CarrierSample:
    return CarrierSample(
        velocity=velocity,
        density=sample.density,
        dynamic_viscosity=sample.dynamic_viscosity,
        temperature=sample.temperature,
        pressure=sample.pressure,
        turbulent_kinetic_energy=sample.turbulent_kinetic_energy,
        turbulent_dissipation_rate=sample.turbulent_dissipation_rate,
    )


def march_parcels(
    initializations: Sequence[PrimaryParcelInitialization],
    *,
    carrier: CarrierField,
    domain: AxisymmetricDomain,
    solver_spec: SpraySolverSpec,
    march_config: SprayMarchConfig,
    breakup_by_role: Mapping[str, WaveBreakupConfig] | None = None,
    evaporation_by_role: Mapping[str, EvaporationModelConfig] | None = None,
) -> SprayMarchResult:
    """March weighted liquid parcels through a prescribed carrier field."""

    if not isinstance(domain, AxisymmetricDomain):
        raise SprayValidationError("domain must be an AxisymmetricDomain")
    if not isinstance(solver_spec, SpraySolverSpec):
        raise SprayValidationError("solver_spec must be a SpraySolverSpec")
    if not isinstance(march_config, SprayMarchConfig):
        raise SprayValidationError("march_config must be a SprayMarchConfig")
    cloud, source_by_role = _merge_initializations(initializations, solver_spec)
    roles = tuple(source_by_role)
    breakup_models = dict(breakup_by_role or {})
    evaporation_models = dict(evaporation_by_role or {})
    unknown = (set(breakup_models) | set(evaporation_models)) - set(roles)
    if unknown:
        raise SprayValidationError(
            "spray models supplied for unknown liquid roles: " + ", ".join(sorted(unknown))
        )
    for plane in march_config.sampling_planes:
        if not domain.axial_start <= plane <= domain.axial_end:
            raise SprayValidationError(
                f"sampling plane {plane:.9g} lies outside the tracking domain"
            )
    if not np.all(domain.contains(cloud.position, include_boundary=True)):
        raise SprayValidationError("all primary parcel positions must lie in the domain")

    n = cloud.count
    position = np.array(cloud.position, copy=True)
    velocity = np.array(cloud.velocity, copy=True)
    diameter = np.array(cloud.diameter, copy=True)
    temperature = np.array(cloud.temperature, copy=True)
    weight = np.array(cloud.statistical_weight, copy=True)
    active = np.array(cloud.active, copy=True)
    age = np.array(cloud.age, copy=True)
    reasons: list[TerminalReason] = ["active"] * n
    rt_timer = np.zeros(n)

    initial_mass = np.zeros(n)
    for role, source in source_by_role.items():
        mask = np.asarray([item == role for item in cloud.roles])
        initial_mass[mask] = represented_mass(
            diameter[mask], weight[mask], source.liquid.density
        )

    mass_accumulator = {
        role: {"vaporized": 0.0, "wall": 0.0, "exit": 0.0}
        for role in roles
    }
    momentum_accumulator = {
        role: {
            "vaporized": np.zeros(3),
            "wall": np.zeros(3),
            "exit": np.zeros(3),
            "drag": np.zeros(3),
            "body": np.zeros(3),
        }
        for role in roles
    }
    breakup_counts = {
        role: {"none": 0, "kelvin_helmholtz": 0, "rayleigh_taylor": 0}
        for role in roles
    }

    dispersion = DiscreteRandomWalk(
        seed=solver_spec.seed,
        eddy_lifetime_constant=solver_spec.eddy_lifetime_constant,
    )
    fluctuation = np.zeros((n, 3))
    eddy_remaining = np.zeros(n)
    body = np.asarray(march_config.body_acceleration)
    history: list[TrajectoryFrame] = [TrajectoryFrame(0.0, cloud, tuple(reasons))]

    plane_seen = np.zeros((n, len(march_config.sampling_planes)), dtype=bool)
    plane_records: dict[float, dict[str, list]] = {
        plane: {
            "time": [], "position": [], "velocity": [], "diameter": [],
            "weight": [], "roles": [],
        }
        for plane in march_config.sampling_planes
    }
    for plane_index, plane in enumerate(march_config.sampling_planes):
        at_plane = np.isclose(position[:, 0], plane, rtol=0.0, atol=1.0e-14)
        for index in np.flatnonzero(at_plane & active):
            record = plane_records[plane]
            record["time"].append(0.0)
            record["position"].append(position[index].copy())
            record["velocity"].append(velocity[index].copy())
            record["diameter"].append(float(diameter[index]))
            record["weight"].append(float(weight[index]))
            record["roles"].append(cloud.roles[index])
            plane_seen[index, plane_index] = True

    time = 0.0
    step = 0
    while time < solver_spec.maximum_time and np.any(active):
        dt = min(solver_spec.time_step, solver_spec.maximum_time - time)
        indices = np.flatnonzero(active)
        sample = carrier.sample(position[indices], time)
        local_dispersion = DispersionState(
            fluctuation[indices], eddy_remaining[indices]
        )
        local_dispersion = dispersion.advance(sample, local_dispersion, dt)
        fluctuation[indices] = local_dispersion.velocity_fluctuation
        eddy_remaining[indices] = local_dispersion.remaining_lifetime
        effective_velocity = dispersion.effective_carrier_velocity(
            sample, local_dispersion
        )
        effective_sample = _effective_carrier(sample, effective_velocity)
        local_density = np.asarray(
            [source_by_role[cloud.roles[index]].liquid.density for index in indices]
        )
        drag = drag_acceleration(
            velocity[indices], diameter[indices], local_density, effective_sample
        )
        total_acceleration = drag + body[None, :]
        predicted_velocity = velocity[indices] + total_acceleration * dt
        # The chord uses average velocity.  Event time is its segment fraction;
        # convergence of this event approximation is controlled by dt.
        predicted_position = position[indices] + 0.5 * (
            velocity[indices] + predicted_velocity
        ) * dt

        for local_index, index in enumerate(indices):
            role = cloud.roles[index]
            source = source_by_role[role]
            liquid = source.liquid
            old_position = position[index].copy()
            old_velocity = velocity[index].copy()
            old_diameter = float(diameter[index])
            old_weight = float(weight[index])
            mass_before = float(initial_mass[index])
            # Recompute current mass after any prior evaporation/breakup.
            mass_before = float(
                represented_mass(
                    np.asarray([old_diameter]),
                    np.asarray([old_weight]),
                    liquid.density,
                )[0]
            )
            crossing = domain.first_crossing(
                old_position, predicted_position[local_index]
            )
            fraction = 1.0 if crossing is None else crossing.fraction
            if crossing is not None and fraction == 0.0:
                # A source placed on a boundary and directed out of the domain
                # terminates immediately.  Do not fabricate an epsilon-sized
                # residence interval merely to satisfy positive-dt microkernels.
                position[index] = np.asarray(crossing.position)
                active[index] = False
                reasons[index] = crossing.kind
                reservoir = "wall" if crossing.kind == "wall" else "exit"
                mass_accumulator[role][reservoir] += mass_before
                momentum_accumulator[role][reservoir] += mass_before * old_velocity
                continue
            dt_effective = fraction * dt
            new_velocity = old_velocity + total_acceleration[local_index] * dt_effective
            if crossing is None:
                new_position = predicted_position[local_index]
            else:
                new_position = np.asarray(crossing.position)

            drag_impulse = mass_before * drag[local_index] * dt_effective
            body_impulse = mass_before * body * dt_effective
            momentum_accumulator[role]["drag"] += drag_impulse
            momentum_accumulator[role]["body"] += body_impulse

            relative_speed = float(
                np.linalg.norm(effective_velocity[local_index] - new_velocity)
            )
            breakup = breakup_models.get(role)
            if breakup is not None:
                trajectory_speed = float(np.linalg.norm(new_velocity))
                effective_rt = 0.0
                if trajectory_speed > 0.0:
                    effective_rt = float(
                        np.dot(total_acceleration[local_index], new_velocity)
                        / trajectory_speed
                    )
                breakup_result = advance_breakup(
                    BreakupParcelState(
                        old_diameter,
                        old_weight,
                        tuple(float(value) for value in new_velocity),
                        rt_timer=float(rt_timer[index]),
                    ),
                    dt=dt_effective,
                    liquid_density=liquid.density,
                    liquid_dynamic_viscosity=liquid.dynamic_viscosity,
                    surface_tension=liquid.surface_tension,
                    carrier_density=float(sample.density[local_index]),
                    relative_speed=relative_speed,
                    config=breakup,
                    effective_acceleration=effective_rt,
                )
                new_diameter = breakup_result.state.diameter
                new_weight = breakup_result.state.multiplicity
                rt_timer[index] = breakup_result.state.rt_timer
                breakup_counts[role][breakup_result.event] += 1
            else:
                new_diameter = old_diameter
                new_weight = old_weight
                breakup_counts[role]["none"] += 1

            evaporation = evaporation_models.get(role)
            vapor_mass = 0.0
            fully_evaporated = False
            if evaporation is not None:
                evaporation_result = advance_evaporation(
                    EvaporationParcelState(
                        new_diameter,
                        new_weight,
                        tuple(float(value) for value in new_velocity),
                    ),
                    dt=dt_effective,
                    liquid_density=liquid.density,
                    carrier_density=float(sample.density[local_index]),
                    carrier_dynamic_viscosity=float(
                        sample.dynamic_viscosity[local_index]
                    ),
                    mass_diffusivity=evaporation.mass_diffusivity,
                    relative_speed=relative_speed,
                    spalding_mass_number_value=evaporation.spalding_mass_number,
                    closure=evaporation.sherwood_closure,
                )
                new_diameter = evaporation_result.state.diameter
                new_weight = evaporation_result.state.multiplicity
                vapor_mass = evaporation_result.conservation.vapor_mass_source_demand
                fully_evaporated = evaporation_result.fully_evaporated
                mass_accumulator[role]["vaporized"] += vapor_mass
                momentum_accumulator[role]["vaporized"] += vapor_mass * new_velocity

            mass_after = float(
                represented_mass(
                    np.asarray([new_diameter]),
                    np.asarray([new_weight]),
                    liquid.density,
                )[0]
            )

            # Record first liquid crossings.  Step interpolation is explicitly
            # numerical (not an analytic breakup history) and converges with dt.
            for plane_index, plane in enumerate(march_config.sampling_planes):
                if plane_seen[index, plane_index] or mass_after <= 0.0:
                    continue
                dx = new_position[0] - old_position[0]
                if dx == 0.0:
                    continue
                plane_fraction = (plane - old_position[0]) / dx
                if 0.0 <= plane_fraction <= 1.0:
                    sampled_diameter = old_diameter + plane_fraction * (
                        new_diameter - old_diameter
                    )
                    sampled_mass = mass_before + plane_fraction * (
                        mass_after - mass_before
                    )
                    if sampled_diameter > 0.0 and sampled_mass > 0.0:
                        sampled_weight = sampled_mass / (
                            liquid.density * math.pi / 6.0 * sampled_diameter**3
                        )
                        record = plane_records[plane]
                        record["time"].append(time + plane_fraction * dt_effective)
                        record["position"].append(
                            old_position + plane_fraction * (new_position - old_position)
                        )
                        record["velocity"].append(
                            old_velocity + plane_fraction * (new_velocity - old_velocity)
                        )
                        record["diameter"].append(sampled_diameter)
                        record["weight"].append(sampled_weight)
                        record["roles"].append(role)
                        plane_seen[index, plane_index] = True

            position[index] = new_position
            velocity[index] = new_velocity
            diameter[index] = new_diameter
            weight[index] = new_weight
            age[index] += dt_effective

            if fully_evaporated:
                active[index] = False
                reasons[index] = "vaporized"
            elif crossing is not None:
                active[index] = False
                reasons[index] = crossing.kind
                reservoir = "wall" if crossing.kind == "wall" else "exit"
                mass_accumulator[role][reservoir] += mass_after
                momentum_accumulator[role][reservoir] += mass_after * new_velocity

        time += dt
        step += 1
        if step % march_config.history_stride == 0 or not np.any(active):
            snapshot = ParcelCloud(
                position=position,
                velocity=velocity,
                diameter=diameter,
                temperature=temperature,
                statistical_weight=weight,
                roles=cloud.roles,
                active=active,
                age=age,
            )
            history.append(TrajectoryFrame(time, snapshot, tuple(reasons)))

    final_cloud = ParcelCloud(
        position=position,
        velocity=velocity,
        diameter=diameter,
        temperature=temperature,
        statistical_weight=weight,
        roles=cloud.roles,
        active=active,
        age=age,
    )
    if history[-1].time != time:
        history.append(TrajectoryFrame(time, final_cloud, tuple(reasons)))

    per_role = {}
    injected_total = 0.0
    vapor_total = 0.0
    eta_by_role = {}
    for role, source in source_by_role.items():
        mask = np.asarray([item == role for item in cloud.roles])
        active_mask = mask & active
        active_mass_values = represented_mass(
            diameter[active_mask], weight[active_mask], source.liquid.density
        )
        active_mass = float(math.fsum(active_mass_values.tolist()))
        active_momentum = represented_momentum(
            active_mass_values, velocity[active_mask]
        )
        injected_mass = source.injected_mass
        vaporized = mass_accumulator[role]["vaporized"]
        per_role[role] = close_reservoir_ledger(
            role=role,
            injected_mass=injected_mass,
            active_mass=active_mass,
            vaporized_mass=vaporized,
            wall_mass=mass_accumulator[role]["wall"],
            exit_mass=mass_accumulator[role]["exit"],
            initial_momentum=source.injected_momentum,
            active_momentum=active_momentum,
            vapor_momentum=momentum_accumulator[role]["vaporized"],
            wall_momentum=momentum_accumulator[role]["wall"],
            exit_momentum=momentum_accumulator[role]["exit"],
            drag_impulse_on_parcels=momentum_accumulator[role]["drag"],
            body_force_impulse=momentum_accumulator[role]["body"],
        )
        eta_by_role[role] = vaporized / injected_mass
        injected_total += injected_mass
        vapor_total += vaporized
    conservation = ConservationLedger(
        per_role=MappingProxyType(per_role),
        mass_tolerance=march_config.mass_tolerance,
        momentum_tolerance=march_config.momentum_tolerance,
    )
    if march_config.strict_conservation and (
        not conservation.mass_closed or not conservation.parcel_momentum_closed
    ):
        raise SprayConservationError(
            "parcel march failed strict conservation: "
            f"mass_closed={conservation.mass_closed}, "
            f"parcel_momentum_closed={conservation.parcel_momentum_closed}"
        )

    samples: dict[float, SamplingPlaneCloud] = {}
    for plane, record in plane_records.items():
        samples[plane] = SamplingPlaneCloud(
            axial_position=plane,
            time=np.asarray(record["time"]),
            position=np.asarray(record["position"]).reshape(-1, 3),
            velocity=np.asarray(record["velocity"]).reshape(-1, 3),
            diameter=np.asarray(record["diameter"]),
            statistical_weight=np.asarray(record["weight"]),
            roles=tuple(record["roles"]),
        )

    primary_ready = all(
        source.primary_path_eligible for source in source_by_role.values()
    )
    gates = (
        SprayGate(
            "primary_geometry_applicability",
            "pass" if primary_ready else "fail",
            "all liquid sources use an applicable primary path"
            if primary_ready else
            "one or more sources are geometric blobs without a validated primary model",
        ),
        SprayGate(
            "parcel_mass_conservation",
            "pass" if conservation.mass_closed else "fail",
            "represented liquid reservoirs close within the declared tolerance",
        ),
        SprayGate(
            "parcel_momentum_conservation",
            "pass" if conservation.parcel_momentum_closed else "fail",
            "parcel momentum and recorded source demands close within tolerance",
        ),
        SprayGate(
            "carrier_stream_continuity",
            "fail",
            "prescribed carrier state has no role/composition/mass-flow continuity contract",
        ),
        SprayGate(
            "two_way_carrier_momentum",
            "fail",
            "equal/opposite carrier impulse is recorded but not applied to the carrier",
        ),
        SprayGate(
            "droplet_and_carrier_energy",
            "fail",
            "droplet temperature and carrier energy equations are not solved",
        ),
        SprayGate(
            "strict_target_benchmark",
            "fail",
            "published fixtures omit the carrier/property/parcel data needed for strict "
            "end-to-end SMD reproduction; 2021 Tables 7/8 are author CFD outputs",
        ),
    )
    domain_payload = {
        "axial_coordinates_m": domain.axial_coordinates.tolist(),
        "wall_radius_m": domain.wall_radius.tolist(),
    }
    domain_fingerprint = hashlib.sha256(
        json.dumps(domain_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    liquid_payload = {
        role: {
            "name": source.liquid.name,
            "density_kg_m3": source.liquid.density,
            "dynamic_viscosity_pa_s": source.liquid.dynamic_viscosity,
            "surface_tension_n_m": source.liquid.surface_tension,
            "temperature_k": source.liquid.temperature,
            "pressure_pa": source.liquid.pressure,
            "specific_heat_j_kg_k": source.liquid.specific_heat,
            "thermal_conductivity_w_m_k": source.liquid.thermal_conductivity,
            "latent_heat_j_kg": source.liquid.latent_heat,
            "vapor_molar_mass_kg_mol": source.liquid.vapor_molar_mass,
            "vapor_diffusivity_m2_s": source.liquid.vapor_diffusivity,
        }
        for role, source in source_by_role.items()
    }
    primary_payload = {
        role: {
            "model_id": source.model.model_id,
            "injection_form": source.model.injection_form,
            "primary_path_eligible": source.primary_path_eligible,
            "mass_flow_rate_kg_s": source.mass_flow_rate,
            "injection_duration_s": source.injection_duration,
            "injected_mass_kg": source.injected_mass,
        }
        for role, source in source_by_role.items()
    }
    breakup_payload = {
        role: {
            "b0": config.b0,
            "b1": config.b1,
            "coefficient_variant": config.coefficient_variant,
            "weber_limit_radius_based": config.weber_limit,
            "rayleigh_taylor": {
                "enabled": config.rayleigh_taylor.enabled,
                "c_tau": config.rayleigh_taylor.c_tau,
                "c_rt": config.rayleigh_taylor.c_rt,
                "provenance": config.rayleigh_taylor.provenance,
            },
        }
        for role, config in breakup_models.items()
    }
    evaporation_payload = {
        role: {
            "mass_diffusivity_m2_s": config.mass_diffusivity,
            "spalding_mass_number": config.spalding_mass_number,
            "sherwood_closure": config.sherwood_closure,
        }
        for role, config in evaporation_models.items()
    }
    carrier_fingerprint = carrier_field_fingerprint(carrier)
    metadata = {
        **solver_spec.rng_metadata,
        "time_step_s": solver_spec.time_step,
        "maximum_time_s": solver_spec.maximum_time,
        "completed_time_s": time,
        "completed_steps": step,
        "parcels_per_liquid_stream": solver_spec.parcels_per_liquid_stream,
        "history_stride": march_config.history_stride,
        "carrier_coupling": "one_way_prescribed",
        "trajectory_event_method": "linear_step_chord_with_fractional_impulse",
        "sampling_method": "first_crossing_step_interpolation_requires_dt_convergence",
        "temperature_model": "fixed_parcel_temperature_no_energy_equation",
        "carrier_field_type": (
            f"{carrier.__class__.__module__}.{carrier.__class__.__qualname__}"
        ),
        "carrier_field_fingerprint_sha256": carrier_fingerprint,
        "carrier_field_traceable": carrier_fingerprint is not None,
        "domain_fingerprint_sha256": domain_fingerprint,
        "domain": domain_payload,
        "liquid_properties_by_role": liquid_payload,
        "primary_sources_by_role": primary_payload,
        "breakup_models_by_role": breakup_payload,
        "evaporation_models_by_role": evaporation_payload,
        "sampling_planes_m": list(march_config.sampling_planes),
    }
    return SprayMarchResult(
        final_cloud=final_cloud,
        terminal_reason=tuple(reasons),
        history=tuple(history),
        sampling_planes=samples,
        conservation=conservation,
        eta_vaporization_by_role=eta_by_role,
        eta_vaporization=vapor_total / injected_total,
        breakup_event_counts=breakup_counts,
        solver_metadata=metadata,
        gates=gates,
    )


__all__ = [
    "EvaporationModelConfig",
    "SamplingPlaneCloud",
    "SprayConservationError",
    "SprayGate",
    "SprayMarchConfig",
    "SprayMarchResult",
    "TerminalReason",
    "TrajectoryFrame",
    "march_parcels",
]
