"""Seeded discrete-random-walk turbulent parcel dispersion.

Each parcel interacts with one isotropic turbulent eddy for
``tau_e = C_L k / epsilon``.  At eddy expiry a new velocity fluctuation is
sampled with component standard deviation ``sqrt(2 k / 3)``.  The model owns a
local generator created by :func:`numpy.random.default_rng`; it never touches
NumPy's process-global random state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import CarrierSample, SprayValidationError, _readonly_array


@dataclass(frozen=True)
class DispersionState:
    """Current eddy fluctuation and remaining interaction time per parcel."""

    velocity_fluctuation: np.ndarray  # (n, 3), m/s
    remaining_lifetime: np.ndarray    # (n,), s

    def __post_init__(self) -> None:
        fluctuation = _readonly_array(
            self.velocity_fluctuation,
            name="velocity_fluctuation",
            dtype=float,
            ndim=2,
        )
        if fluctuation.shape[1:] != (3,):
            raise SprayValidationError(
                "velocity_fluctuation must have shape (n, 3)"
            )
        lifetime = _readonly_array(
            self.remaining_lifetime,
            name="remaining_lifetime",
            dtype=float,
            ndim=1,
        )
        if lifetime.shape != (fluctuation.shape[0],):
            raise SprayValidationError(
                "remaining_lifetime must have one value per parcel"
            )
        if np.any(lifetime < 0.0):
            raise SprayValidationError("remaining_lifetime must be >= 0")
        object.__setattr__(self, "velocity_fluctuation", fluctuation)
        object.__setattr__(self, "remaining_lifetime", lifetime)

    @classmethod
    def quiescent(cls, parcel_count: int) -> "DispersionState":
        if isinstance(parcel_count, (bool, np.bool_)) or int(parcel_count) < 1:
            raise SprayValidationError("parcel_count must be an integer >= 1")
        return cls(
            velocity_fluctuation=np.zeros((int(parcel_count), 3)),
            remaining_lifetime=np.zeros(int(parcel_count)),
        )


class DiscreteRandomWalk:
    """Stateful local-RNG implementation of the isotropic eddy interaction model."""

    def __init__(self, *, seed: int, eddy_lifetime_constant: float):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ) or int(seed) < 0:
            raise SprayValidationError("seed must be a nonnegative integer")
        constant = float(eddy_lifetime_constant)
        if not math.isfinite(constant) or constant <= 0.0:
            raise SprayValidationError(
                "eddy_lifetime_constant must be finite and > 0"
            )
        self.seed = int(seed)
        self.eddy_lifetime_constant = constant
        self._rng = np.random.default_rng(self.seed)

    @property
    def rng_metadata(self) -> dict[str, int | str]:
        return {
            "seed": self.seed,
            "bit_generator": type(self._rng.bit_generator).__name__,
        }

    def advance(
        self,
        carrier: CarrierSample,
        state: DispersionState,
        time_step: float,
    ) -> DispersionState:
        """Advance eddy lifetimes and renew expired turbulent fluctuations.

        A wholly zero-turbulence call returns before accessing the RNG.  This is
        important both physically and reproducibly: inserting laminar steps must
        not alter the later stochastic sequence.
        """

        dt = float(time_step)
        if not math.isfinite(dt) or dt <= 0.0:
            raise SprayValidationError("time_step must be finite and > 0")
        n = state.velocity_fluctuation.shape[0]
        try:
            k = np.broadcast_to(carrier.turbulent_kinetic_energy, (n,))
            epsilon = np.broadcast_to(carrier.turbulent_dissipation_rate, (n,))
        except ValueError as exc:
            raise SprayValidationError(
                "carrier turbulence sample must have one value per parcel"
            ) from exc

        turbulent = k > 0.0
        if not np.any(turbulent):
            # Do not consume random numbers in the exact laminar limit.
            return DispersionState.quiescent(n)
        if np.any(epsilon[turbulent] <= 0.0):
            raise SprayValidationError(
                "positive turbulent kinetic energy requires positive dissipation rate"
            )

        fluctuation = np.array(state.velocity_fluctuation, copy=True)
        remaining = np.maximum(
            np.asarray(state.remaining_lifetime, dtype=float) - dt, 0.0
        )
        fluctuation[~turbulent] = 0.0
        remaining[~turbulent] = 0.0

        expired = turbulent & (remaining <= 0.0)
        indices = np.flatnonzero(expired)
        if indices.size:
            standard_deviation = np.sqrt(2.0 * k[indices] / 3.0)
            draws = self._rng.standard_normal((indices.size, 3))
            fluctuation[indices] = standard_deviation[:, None] * draws
            remaining[indices] = (
                self.eddy_lifetime_constant * k[indices] / epsilon[indices]
            )

        return DispersionState(
            velocity_fluctuation=fluctuation,
            remaining_lifetime=remaining,
        )

    @staticmethod
    def effective_carrier_velocity(
        carrier: CarrierSample,
        state: DispersionState,
    ) -> np.ndarray:
        """Return mean carrier velocity plus the current eddy fluctuation."""

        try:
            mean = np.broadcast_to(
                carrier.velocity, state.velocity_fluctuation.shape
            )
        except ValueError as exc:
            raise SprayValidationError(
                "carrier velocity and dispersion state shapes are incompatible"
            ) from exc
        return mean + state.velocity_fluctuation


__all__ = ["DiscreteRandomWalk", "DispersionState"]
