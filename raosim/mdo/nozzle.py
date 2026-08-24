"""Typed contour-provider seam for the differentiable engine MDO.

The current production provider is deliberately only the fixed-topology
analytic TOP/Bézier construction already implemented in :mod:`raosim.mdo.grid`.
This module does not add another contour model; it gives that construction a
stable provider boundary and an explicit outcome contract so a future provider
can report failure without being silently replaced by the analytic contour.

Provider failures still carry a finite, fixed-shape ``StationGrid`` supplied by
the provider.  That grid is a shape carrier which lets the coupled evaluation
finish and report the candidate.  ``available`` and ``infeasible`` decide
whether the geometry may be accepted; consumers must never infer availability
from finite coordinates or from NaN sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Protocol, runtime_checkable

import raosim.jax  # noqa: F401 -- configure JAX x64 before array creation
import jax.numpy as jnp

from raosim.mdo.grid import GridTopology, StationGrid, build_station_grid
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray

ANALYTIC_TOP_PROVIDER_ID = "rao_top_bezier_fixed_topology"
ANALYTIC_TOP_MODEL_CODE = 1


class ContourOutcomeReason(IntEnum):
    """JAX-safe reason codes returned by a contour provider."""

    AVAILABLE = 0
    PROVIDER_REPORTED_INFEASIBLE = 1
    NONFINITE_GEOMETRY = 2


class ContourOutcome(NamedTuple):
    """Fixed-shape numerical result from one contour-provider evaluation.

    ``grid`` remains present when the provider reports failure so downstream
    arrays keep their static topology.  Such a grid is not an accepted contour:
    the scalar boolean leaves and ``reason_code`` are authoritative.
    """

    grid: StationGrid
    available: Array
    infeasible: Array
    reason_code: Array
    provider_model_code: Array


@runtime_checkable
class ContourProvider(Protocol):
    """Seam for a future fixed-topology contour implementation.

    The variational contour is intentionally not registered here yet.  A
    provider must return a :class:`ContourOutcome`, including its own failure
    shape carrier; the adapter never catches failure and substitutes TOP.
    """

    provider_id: str
    model_code: int

    def evaluate(
        self,
        *,
        Rt: Array,
        eps: Array,
        mission: MissionSpec,
        topo: GridTopology,
        gamma: Array | None,
    ) -> ContourOutcome:
        ...


ContourProviderSelection = ContourProvider | str | None


def _grid_is_finite(grid: StationGrid) -> Array:
    return jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(grid.x)),
        jnp.all(jnp.isfinite(grid.r)),
        jnp.all(jnp.isfinite(grid.area_ratio)),
        jnp.all(jnp.isfinite(grid.mach)),
        jnp.all(jnp.isfinite(grid.dseg)),
        jnp.all(jnp.isfinite(grid.chart_domain_violation)),
        jnp.isfinite(grid.wall_monotonic_margin),
    ]))


@dataclass(frozen=True)
class AnalyticTOPContourProvider:
    """Adapter for the existing analytic TOP fixed-topology path."""

    provider_id: str = ANALYTIC_TOP_PROVIDER_ID
    model_code: int = ANALYTIC_TOP_MODEL_CODE

    def evaluate(
        self,
        *,
        Rt: Array,
        eps: Array,
        mission: MissionSpec,
        topo: GridTopology,
        gamma: Array | None,
    ) -> ContourOutcome:
        # This exact call is the pre-adapter implementation.  Keeping contour
        # construction in grid.py makes the default path bit-identical.
        grid = build_station_grid(Rt, eps, mission, topo, gamma=gamma)
        finite = _grid_is_finite(grid)
        return ContourOutcome(
            grid=grid,
            available=finite,
            infeasible=~finite,
            reason_code=jnp.where(
                finite,
                int(ContourOutcomeReason.AVAILABLE),
                int(ContourOutcomeReason.NONFINITE_GEOMETRY),
            ).astype(jnp.int32),
            provider_model_code=jnp.asarray(self.model_code, dtype=jnp.int32),
        )


DEFAULT_CONTOUR_PROVIDER = AnalyticTOPContourProvider()


def resolve_contour_provider(
    provider: ContourProviderSelection = None,
) -> ContourProvider:
    """Resolve a registered provider without any fallback substitution."""

    if provider is None:
        return DEFAULT_CONTOUR_PROVIDER
    if isinstance(provider, str):
        if provider == ANALYTIC_TOP_PROVIDER_ID:
            return DEFAULT_CONTOUR_PROVIDER
        raise ValueError(
            f"unknown MDO contour provider {provider!r}; the only registered "
            f"provider is {ANALYTIC_TOP_PROVIDER_ID!r}"
        )
    if not isinstance(provider, ContourProvider):
        raise TypeError(
            "contour provider must implement provider_id, model_code, and "
            "evaluate(...) -> ContourOutcome"
        )
    return provider


def evaluate_contour(
    Rt: Array,
    eps: Array,
    mission: MissionSpec,
    topo: GridTopology = GridTopology(),
    *,
    gamma: Array | None = None,
    provider: ContourProviderSelection = None,
) -> ContourOutcome:
    """Evaluate one provider and normalize its scalar outcome leaves.

    Provider exceptions and malformed results propagate.  Treating either as
    a request for the analytic contour would be an unauthorized model switch.
    A provider that cannot construct an admissible contour must return a typed
    infeasible outcome with its own fixed-shape carrier grid.
    """

    selected = resolve_contour_provider(provider)
    outcome = selected.evaluate(
        Rt=Rt, eps=eps, mission=mission, topo=topo, gamma=gamma
    )
    if not isinstance(outcome, ContourOutcome):
        raise TypeError(
            f"contour provider {selected.provider_id!r} returned "
            f"{type(outcome).__name__}, not ContourOutcome"
        )
    return ContourOutcome(
        grid=outcome.grid,
        available=jnp.asarray(outcome.available, dtype=jnp.bool_),
        infeasible=jnp.asarray(outcome.infeasible, dtype=jnp.bool_),
        reason_code=jnp.asarray(outcome.reason_code, dtype=jnp.int32),
        provider_model_code=jnp.asarray(
            outcome.provider_model_code, dtype=jnp.int32
        ),
    )


__all__ = [
    "ANALYTIC_TOP_MODEL_CODE",
    "ANALYTIC_TOP_PROVIDER_ID",
    "AnalyticTOPContourProvider",
    "ContourOutcome",
    "ContourOutcomeReason",
    "ContourProvider",
    "ContourProviderSelection",
    "DEFAULT_CONTOUR_PROVIDER",
    "evaluate_contour",
    "resolve_contour_provider",
]
