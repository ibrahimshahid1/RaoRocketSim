"""Shared geometric contract for the chamber/nozzle throat region."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThroatGeometrySpec:
    """Geometry shared by every contour on either side of the throat.

    Radius values are nondimensional ratios to the throat radius.  The
    ``throat_location`` is the axial coordinate of the minimum-radius station.
    """

    upstream_radius_ratio: float = 1.5
    downstream_radius_ratio: float = 0.382
    convergent_half_angle_deg: float = 45.0
    throat_location: float = 0.0

    def validate(self) -> None:
        if self.upstream_radius_ratio <= 0.0:
            raise ValueError("upstream_radius_ratio must be positive")
        if self.downstream_radius_ratio <= 0.0:
            raise ValueError("downstream_radius_ratio must be positive")
        if not 0.0 < self.convergent_half_angle_deg < 90.0:
            raise ValueError("convergent_half_angle_deg must be in (0, 90)")

    def upstream_radius(self, throat_radius: float) -> float:
        self.validate()
        if throat_radius <= 0.0:
            raise ValueError("throat_radius must be positive")
        return self.upstream_radius_ratio * throat_radius

    def downstream_radius(self, throat_radius: float) -> float:
        self.validate()
        if throat_radius <= 0.0:
            raise ValueError("throat_radius must be positive")
        return self.downstream_radius_ratio * throat_radius

    def to_dict(self) -> dict[str, float]:
        return {
            "upstream_radius_ratio": self.upstream_radius_ratio,
            "downstream_radius_ratio": self.downstream_radius_ratio,
            "convergent_half_angle_deg": self.convergent_half_angle_deg,
            "throat_location": self.throat_location,
        }


def resolve_throat_geometry(
    throat_geometry: ThroatGeometrySpec | None,
    *,
    upstream_radius_ratio: float = 1.5,
    downstream_radius_ratio: float = 0.382,
    convergent_half_angle_deg: float = 45.0,
    throat_location: float = 0.0,
) -> ThroatGeometrySpec:
    """Return and validate the authoritative throat specification."""
    spec = throat_geometry or ThroatGeometrySpec(
        upstream_radius_ratio=upstream_radius_ratio,
        downstream_radius_ratio=downstream_radius_ratio,
        convergent_half_angle_deg=convergent_half_angle_deg,
        throat_location=throat_location,
    )
    if not isinstance(spec, ThroatGeometrySpec):
        raise TypeError("throat_geometry must be a ThroatGeometrySpec")
    spec.validate()
    return spec
