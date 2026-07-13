import numpy as np
import pytest

from raosim.spray.domain import AxisymmetricDomain
from raosim.spray.types import SprayValidationError


def test_cylinder_contains_and_radius_do_not_extrapolate():
    domain = AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=1.0, radius=0.5
    )
    assert domain.contains([0.25, 0.3, 0.4])
    assert not domain.contains([0.25, 0.31, 0.4])
    assert domain.radius_at([0.0, 0.5, 1.0]).tolist() == [0.5, 0.5, 0.5]
    with pytest.raises(SprayValidationError, match="extrapolation"):
        domain.radius_at(1.01)


def test_first_crossing_finds_curved_radial_wall_intersection():
    domain = AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=2.0, radius=1.0
    )
    crossing = domain.first_crossing(
        np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.8, 0.8])
    )
    assert crossing is not None
    assert crossing.kind == "wall"
    assert crossing.fraction == pytest.approx(1.0 / np.sqrt(1.28))
    assert np.hypot(crossing.position[1], crossing.position[2]) == pytest.approx(1.0)


def test_first_crossing_resolves_wall_before_outlet():
    domain = AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=1.0, radius=0.4
    )
    crossing = domain.first_crossing([0.5, 0.0, 0.0], [1.5, 1.0, 0.0])
    assert crossing is not None
    assert crossing.kind == "wall"
    assert crossing.fraction == pytest.approx(0.4)


def test_first_crossing_resolves_outlet_before_later_wall():
    domain = AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=1.0, radius=0.8
    )
    crossing = domain.first_crossing([0.5, 0.0, 0.0], [1.5, 1.0, 0.0])
    assert crossing is not None
    assert crossing.kind == "outlet"
    assert crossing.fraction == pytest.approx(0.5)


def test_piecewise_wall_knot_is_used_to_bracket_impact():
    domain = AxisymmetricDomain(
        axial_coordinates=np.array([0.0, 0.5, 1.0]),
        wall_radius=np.array([1.0, 0.4, 1.0]),
    )
    crossing = domain.first_crossing([0.0, 0.3, 0.0], [1.0, 0.6, 0.0])
    assert crossing is not None
    assert crossing.kind == "wall"
    assert crossing.fraction < 0.5


def test_invalid_domain_and_outside_segment_start_are_rejected():
    with pytest.raises(SprayValidationError, match="strictly increasing"):
        AxisymmetricDomain([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(SprayValidationError, match="wall_radius"):
        AxisymmetricDomain([0.0, 1.0], [1.0, 0.0])
    domain = AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=1.0, radius=1.0
    )
    with pytest.raises(SprayValidationError, match="start"):
        domain.first_crossing([-0.1, 0.0, 0.0], [0.5, 0.0, 0.0])
