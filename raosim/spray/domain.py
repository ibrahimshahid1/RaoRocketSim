"""Axisymmetric parcel-tracking domain and exact segment boundary events.

The chamber/nozzle wall is represented by a piecewise-linear radius
``r_wall(x)``.  Parcel trajectories remain Cartesian with ``x`` as the engine
axis.  Boundary events are found along each straight integration segment so a
parcel that crosses both a wall and the outlet during one numerical step is
assigned to the first physical boundary it reaches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .types import SprayValidationError, _readonly_array


BoundaryKind = Literal["wall", "outlet", "inlet"]


@dataclass(frozen=True)
class BoundaryCrossing:
    """First boundary event along one parcel integration segment."""

    kind: BoundaryKind
    fraction: float
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        fraction = float(self.fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise SprayValidationError("boundary fraction must be in [0, 1]")
        position = tuple(float(value) for value in self.position)
        if len(position) != 3 or any(not math.isfinite(value) for value in position):
            raise SprayValidationError("boundary position must be a finite 3-vector")
        if self.kind not in {"wall", "outlet", "inlet"}:
            raise SprayValidationError(f"unknown boundary kind {self.kind!r}")
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class AxisymmetricDomain:
    """Piecewise-linear axisymmetric chamber/nozzle tracking boundary.

    Parameters are SI.  The first and last axial coordinates are respectively
    the inlet and outlet planes.  Wall radii must be positive; a zero-radius
    axis closure is not a valid liquid-rocket flow domain.
    """

    axial_coordinates: np.ndarray
    wall_radius: np.ndarray

    def __post_init__(self) -> None:
        axial = _readonly_array(
            self.axial_coordinates, name="axial_coordinates", dtype=float, ndim=1
        )
        radius = _readonly_array(
            self.wall_radius, name="wall_radius", dtype=float, ndim=1
        )
        if axial.size < 2 or radius.shape != axial.shape:
            raise SprayValidationError(
                "axial_coordinates and wall_radius must have equal length >= 2"
            )
        if np.any(np.diff(axial) <= 0.0):
            raise SprayValidationError("axial_coordinates must be strictly increasing")
        if np.any(radius <= 0.0):
            raise SprayValidationError("wall_radius must be > 0 everywhere")
        object.__setattr__(self, "axial_coordinates", axial)
        object.__setattr__(self, "wall_radius", radius)

    @classmethod
    def cylinder(
        cls, *, axial_start: float, axial_end: float, radius: float
    ) -> "AxisymmetricDomain":
        """Construct a constant-radius validation or cold-flow domain."""

        values = np.asarray([axial_start, axial_end], dtype=float)
        return cls(values, np.full(2, float(radius)))

    @property
    def axial_start(self) -> float:
        return float(self.axial_coordinates[0])

    @property
    def axial_end(self) -> float:
        return float(self.axial_coordinates[-1])

    def radius_at(self, axial_position):
        """Interpolate wall radius inside the domain; never extrapolate."""

        x = np.asarray(axial_position, dtype=float)
        if np.any(~np.isfinite(x)):
            raise SprayValidationError("axial position must be finite")
        if np.any((x < self.axial_start) | (x > self.axial_end)):
            raise SprayValidationError(
                "wall radius query lies outside the axial domain; extrapolation "
                "is not permitted"
            )
        result = np.interp(x, self.axial_coordinates, self.wall_radius)
        return float(result) if result.ndim == 0 else result

    def contains(self, position, *, include_boundary: bool = True):
        """Return whether Cartesian point(s) are inside the tracking domain."""

        points = np.asarray(position, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise SprayValidationError("position must have shape (..., 3)")
        if np.any(~np.isfinite(points)):
            raise SprayValidationError("position must be finite")
        x = points[..., 0]
        axial = (x >= self.axial_start) & (x <= self.axial_end)
        clipped_x = np.clip(x, self.axial_start, self.axial_end)
        radial = np.hypot(points[..., 1], points[..., 2])
        wall = np.interp(clipped_x, self.axial_coordinates, self.wall_radius)
        result = axial & (radial <= wall if include_boundary else radial < wall)
        return bool(result) if result.ndim == 0 else result

    def first_crossing(self, start, end) -> BoundaryCrossing | None:
        """Return the first wall/inlet/outlet crossing on a straight segment.

        ``start`` must be inside or on the boundary.  Wall roots are bracketed
        at every wall-profile knot crossed by the segment.  Within a wall
        interval, radial position is the norm of an affine vector and hence
        convex; with an inside left endpoint, a positive right endpoint gives
        the unique outward root needed by the march.
        """

        p0 = np.asarray(start, dtype=float)
        p1 = np.asarray(end, dtype=float)
        if p0.shape != (3,) or p1.shape != (3,):
            raise SprayValidationError("segment endpoints must have shape (3,)")
        if np.any(~np.isfinite(p0)) or np.any(~np.isfinite(p1)):
            raise SprayValidationError("segment endpoints must be finite")
        if not self.contains(p0, include_boundary=True):
            raise SprayValidationError("segment start must lie inside the domain")

        delta = p1 - p0
        axial_events: list[tuple[float, BoundaryKind]] = []
        if delta[0] > 0.0 and p1[0] >= self.axial_end:
            axial_events.append(
                ((self.axial_end - p0[0]) / delta[0], "outlet")
            )
        if delta[0] < 0.0 and p1[0] <= self.axial_start:
            axial_events.append(
                ((self.axial_start - p0[0]) / delta[0], "inlet")
            )
        axial_events = [
            (float(s), kind)
            for s, kind in axial_events
            if -1.0e-14 <= s <= 1.0 + 1.0e-14
        ]
        axial_fraction = min((s for s, _ in axial_events), default=1.0)
        search_end = min(1.0, max(0.0, axial_fraction))

        def point(fraction: float) -> np.ndarray:
            return p0 + fraction * delta

        def wall_gap(fraction: float) -> float:
            p = point(fraction)
            wall = float(
                np.interp(p[0], self.axial_coordinates, self.wall_radius)
            )
            return float(math.hypot(p[1], p[2]) - wall)

        fractions = [0.0, search_end]
        if delta[0] != 0.0:
            lower = min(p0[0], point(search_end)[0])
            upper = max(p0[0], point(search_end)[0])
            knots = self.axial_coordinates[
                (self.axial_coordinates > lower) & (self.axial_coordinates < upper)
            ]
            fractions.extend(float((x - p0[0]) / delta[0]) for x in knots)
        fractions = sorted(set(fractions))

        wall_fraction: float | None = None
        gap_left = wall_gap(fractions[0])
        for left, right in zip(fractions[:-1], fractions[1:], strict=True):
            gap_right = wall_gap(right)
            # Starting on a wall and moving tangentially/inward is not a new
            # impact.  Starting on it and moving outward is an immediate event.
            if gap_left >= -1.0e-13 and gap_right > 0.0:
                if left == 0.0 and gap_left <= 0.0:
                    wall_fraction = 0.0
                    break
            if gap_left <= 0.0 and gap_right > 0.0:
                lo, hi = left, right
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if wall_gap(mid) > 0.0:
                        hi = mid
                    else:
                        lo = mid
                wall_fraction = 0.5 * (lo + hi)
                break
            gap_left = gap_right

        candidates: list[tuple[float, BoundaryKind]] = list(axial_events)
        if wall_fraction is not None:
            candidates.append((wall_fraction, "wall"))
        if not candidates:
            return None
        # A wall and outlet can coincide on the lip.  Prefer the wall only if
        # it is genuinely earlier; otherwise the downstream plane owns the tie.
        priority = {"outlet": 0, "inlet": 0, "wall": 1}
        fraction, kind = min(candidates, key=lambda item: (item[0], priority[item[1]]))
        fraction = float(np.clip(fraction, 0.0, 1.0))
        crossing_position = tuple(float(value) for value in point(fraction))
        return BoundaryCrossing(kind, fraction, crossing_position)


__all__ = ["AxisymmetricDomain", "BoundaryCrossing", "BoundaryKind"]
