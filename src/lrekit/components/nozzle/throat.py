"""Shared geometric contract for the chamber/nozzle throat region."""

from __future__ import annotations

from dataclasses import dataclass


# NASA SP-8120's cited constant-efficiency design range.  Values above 1.5
# can still be useful for parametric work, but must not be labelled SP-8120.
SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS = (0.6, 1.5)
REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS = (0.6, 2.0)


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


def throat_discharge_coefficient_hall(
    upstream_radius_ratio: float,
    gamma: float,
) -> float:
    """Hall leading-order inviscid throat discharge coefficient.

    NASA SP-8120 uses the transonic throat-flow result to show that smaller
    upstream throat radius reduces effective sonic area.  The leading term is
    ``Cd ~= 1 - ((gamma + 1) / 96) * (Rt/Ru)^2``.
    """
    if upstream_radius_ratio <= 0.0:
        raise ValueError("upstream_radius_ratio must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    curvature_term = (gamma + 1.0) / 96.0
    return 1.0 - curvature_term / (upstream_radius_ratio**2)


def upstream_radius_ratio_for_discharge_coefficient(
    cd_target: float,
    gamma: float,
    *,
    min_ratio: float = SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS[0],
    max_ratio: float = SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS[1],
) -> float:
    """Smallest ``Ru/Rt`` satisfying a target inviscid throat ``Cd``.

    The lower and upper defaults are the cited SP-8120 throat-approach radius
    range, 0.6--1.5.  Callers may explicitly pass the repository's broader
    ``REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS`` for diagnostic
    studies, but must preserve that different provenance.  If the lower
    bound already exceeds the requested ``Cd``, the lower bound is returned;
    requesting a ``Cd`` above the upper-bound capability is rejected.
    """
    if not 0.0 < cd_target < 1.0:
        raise ValueError("cd_target must be in (0, 1)")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if min_ratio <= 0.0 or max_ratio <= 0.0 or min_ratio > max_ratio:
        raise ValueError("radius-ratio bounds must satisfy 0 < min <= max")

    cd_min = throat_discharge_coefficient_hall(min_ratio, gamma)
    cd_max = throat_discharge_coefficient_hall(max_ratio, gamma)
    if cd_target <= cd_min:
        return float(min_ratio)
    if cd_target > cd_max:
        range_label = (
            "Hall/SP-8120"
            if (min_ratio, max_ratio) == SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS
            else "Hall configured radius-ratio"
        )
        raise ValueError(
            f"cd_target={cd_target:g} exceeds {range_label} range capability "
            f"{cd_max:g} at Ru/Rt={max_ratio:g} for gamma={gamma:g}"
        )

    curvature_term = (gamma + 1.0) / 96.0
    return float((curvature_term / (1.0 - cd_target)) ** 0.5)
