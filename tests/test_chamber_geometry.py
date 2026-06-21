import math

import numpy as np
import pytest

from raosim.chamber_geometry import (
    chamber_contour,
    enclosed_volume,
    full_engine_contour,
    thrust_chamber_geometry_checks,
)
from raosim.design import DesignInput, ThermoSpec, design_nozzle_v2
from raosim.export import _clean_meridian_for_brep
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.throat_geometry import ThroatGeometrySpec


def test_shared_throat_spec_removes_chamber_nozzle_discontinuity():
    spec = ThroatGeometrySpec(
        upstream_radius_ratio=1.0,
        downstream_radius_ratio=0.5,
        convergent_half_angle_deg=30.0,
        throat_location=0.012,
    )
    chamber = chamber_contour(0.020, throat_geometry=spec)
    nozzle = bell_nozzle_contour(0.020, 10.0, throat_geometry=spec)
    full = full_engine_contour(chamber, nozzle)

    assert chamber["throat_geometry"] == nozzle["throat_geometry"]
    assert chamber["x"][-1] == pytest.approx(spec.throat_location, abs=1e-12)
    assert nozzle["x_throat"][0] == pytest.approx(spec.throat_location, abs=1e-12)
    assert full["geometry_checks"]["seam_position_gap"] < 1e-12
    assert full["geometry_checks"]["slope_continuity"]


@pytest.mark.parametrize("L_star", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("contraction_ratio", [2.5, 3.0, 4.0])
def test_sampled_chamber_volume_matches_lstar(L_star, contraction_ratio):
    Rt = 0.020
    chamber = chamber_contour(
        Rt,
        L_star=L_star,
        contraction_ratio=contraction_ratio,
    )
    target = L_star * math.pi * Rt**2
    assert enclosed_volume(chamber["x"], chamber["y"]) == pytest.approx(
        target, rel=1e-10
    )
    assert chamber["Lc"] >= chamber["minimum_cylindrical_length"]


def test_volume_is_exact_polyline_frustum_not_trapezoidal_area():
    x = np.array([0.0, 2.0])
    radius = np.array([1.0, 2.0])
    expected_frustum = math.pi * 2.0 * (1.0 + 2.0 + 4.0) / 3.0
    assert enclosed_volume(x, radius) == pytest.approx(expected_frustum)


def test_step_sampling_preserves_every_chamber_station_and_volume():
    spec = ThroatGeometrySpec()
    chamber = chamber_contour(0.020, throat_geometry=spec)
    nozzle = bell_nozzle_contour(0.020, 10.0, throat_geometry=spec)
    full = full_engine_contour(chamber, nozzle)

    x_cad, r_cad, _ = _clean_meridian_for_brep(
        full["x"],
        full["y"],
        0.001,
        throat_location=spec.throat_location,
    )
    chamber_mask = x_cad <= spec.throat_location + 1e-12
    assert np.count_nonzero(chamber_mask) == len(chamber["x"])
    assert enclosed_volume(
        x_cad[chamber_mask], r_cad[chamber_mask]
    ) == pytest.approx(chamber["V_target"], rel=1e-10)


def test_infeasible_chamber_is_rejected_instead_of_inventing_cylinder():
    with pytest.raises(ValueError, match="infeasible chamber geometry"):
        chamber_contour(
            0.020,
            L_star=0.01,
            contraction_ratio=2.5,
            minimum_cylindrical_length=0.005,
        )


def test_full_contour_geometry_gates_include_wall_offset():
    spec = ThroatGeometrySpec()
    chamber = chamber_contour(0.020, throat_geometry=spec)
    nozzle = bell_nozzle_contour(0.020, 8.0, throat_geometry=spec)
    full = full_engine_contour(chamber, nozzle)
    checks = thrust_chamber_geometry_checks(full, offset_distance=0.001)

    assert checks["axial_coordinates_monotonic"]
    assert checks["seam_watertight"]
    assert checks["measured_volume_within_tolerance"]
    assert checks["offset_checked"]
    assert checks["offset_self_intersection_free"]


def test_v2_workflow_returns_authoritative_injector_to_exit_contour():
    result = design_nozzle_v2(DesignInput(
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
        ),
        Pc=80e5,
        Rt=0.020,
        epsilon=8.0,
        L_star=1.0,
        contraction_ratio=2.5,
        shoulder_radius_factor=0.2,
        minimum_cylindrical_length=0.01,
    ))

    contour = result.contour
    throat_index = int(np.argmin(contour["y"]))
    assert contour["full_thrust_chamber"]
    assert contour["x"][0] == pytest.approx(contour["injector_location"])
    assert contour["x"][throat_index] == pytest.approx(
        contour["throat_location"], abs=1e-12
    )
    assert contour["x"][-1] == pytest.approx(
        contour["throat_location"] + contour["Ln"]
    )
    assert result.report_sections["chamber_geometry"]["geometry_checks"][
        "measured_volume_within_tolerance"
    ]
    assert contour["chamber"]["shoulder_radius_factor"] == pytest.approx(0.2)
    assert contour["chamber"]["minimum_cylindrical_length"] == pytest.approx(
        0.01
    )
