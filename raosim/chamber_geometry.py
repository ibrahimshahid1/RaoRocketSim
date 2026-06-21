"""Combustion-chamber, convergent, and full thrust-chamber geometry."""

from __future__ import annotations

import math

import numpy as np

from raosim.regen_profile import normal_offset_contour
from raosim.throat_geometry import ThroatGeometrySpec, resolve_throat_geometry


def _concat_sections(*sections: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for index, (x, y) in enumerate(sections):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        start = 1 if index and x_parts and np.isclose(x[0], x_parts[-1][-1]) else 0
        x_parts.append(x[start:])
        y_parts.append(y[start:])
    return np.concatenate(x_parts), np.concatenate(y_parts)


def enclosed_volume(x: np.ndarray, radius: np.ndarray) -> float:
    """Exact volume of the solid formed by revolving a polyline.

    Each straight meridional segment revolves into a conical frustum:

    ``dV = pi*dx*(r0**2 + r0*r1 + r1**2)/3``.
    """
    x = np.asarray(x, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if x.shape != radius.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("x and radius must be equal-length one-dimensional arrays")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("volume integration requires strictly increasing x")
    dx = np.diff(x)
    r0 = radius[:-1]
    r1 = radius[1:]
    return float(np.sum(
        (math.pi / 3.0) * dx * (r0**2 + r0 * r1 + r1**2)
    ))


def _bisect_root(function, lower: float, upper: float, *, xtol: float) -> float:
    f_lower = float(function(lower))
    f_upper = float(function(upper))
    if f_lower > 0.0 or f_upper < 0.0:
        raise ValueError("root is not bracketed")
    for _ in range(160):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = float(function(midpoint))
        if abs(upper - lower) <= xtol:
            return midpoint
        if f_midpoint < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def chamber_contour(
    Rt: float,
    L_star: float = 1.0,
    contraction_ratio: float = 2.5,
    convergent_half_angle_deg: float = 45.0,
    n_pts_chamber: int = 50,
    n_pts_convergent: int = 80,
    *,
    throat_geometry: ThroatGeometrySpec | None = None,
    shoulder_radius_factor: float = 0.25,
    n_pts_shoulder: int = 40,
    n_pts_upstream_arc: int = 80,
    minimum_cylindrical_length: float = 1e-6,
) -> dict:
    """Generate injector face → cylinder → shoulder → convergent → throat.

    The cylindrical length is root-solved against the exact conical-frustum
    volume of the sampled polyline, so the revolved CAD meridian encloses
    ``L_star * A_t``. Geometry that cannot contain the target volume while
    retaining ``minimum_cylindrical_length`` is rejected.
    """
    if Rt <= 0.0:
        raise ValueError("Rt must be positive")
    if L_star <= 0.0:
        raise ValueError("L_star must be positive")
    if contraction_ratio <= 1.0:
        raise ValueError("contraction_ratio must be > 1")
    if shoulder_radius_factor <= 0.0:
        raise ValueError("shoulder_radius_factor must be positive")
    if minimum_cylindrical_length <= 0.0:
        raise ValueError("minimum_cylindrical_length must be positive")
    if min(n_pts_chamber, n_pts_convergent, n_pts_shoulder, n_pts_upstream_arc) < 2:
        raise ValueError("each chamber section needs at least two points")

    spec = resolve_throat_geometry(
        throat_geometry,
        convergent_half_angle_deg=convergent_half_angle_deg,
    )
    alpha = math.radians(spec.convergent_half_angle_deg)
    throat_x = float(spec.throat_location)
    Ru = spec.upstream_radius(Rt)
    Rs = shoulder_radius_factor * Rt
    Rc = Rt * math.sqrt(contraction_ratio)
    At = math.pi * Rt**2
    Ac = math.pi * Rc**2
    target_volume = L_star * At

    x_arc_entry = throat_x - Ru * math.sin(alpha)
    y_arc_entry = Rt + Ru * (1.0 - math.cos(alpha))
    y_shoulder_end = Rc - Rs * (1.0 - math.cos(alpha))
    radial_straight_drop = y_shoulder_end - y_arc_entry
    if radial_straight_drop <= 0.0:
        raise ValueError(
            "infeasible chamber geometry: chamber shoulder reaches the upstream "
            "throat arc before a convergent segment can be formed"
        )

    straight_length = radial_straight_drop / math.tan(alpha)
    x_shoulder_end = x_arc_entry - straight_length
    x_shoulder_start = x_shoulder_end - Rs * math.sin(alpha)

    shoulder_t = np.linspace(math.pi / 2.0, math.pi / 2.0 - alpha, n_pts_shoulder)
    shoulder_center_x = x_shoulder_start
    shoulder_center_y = Rc - Rs
    x_shoulder = shoulder_center_x + Rs * np.cos(shoulder_t)
    y_shoulder = shoulder_center_y + Rs * np.sin(shoulder_t)

    x_convergent = np.linspace(x_shoulder_end, x_arc_entry, n_pts_convergent)
    y_convergent = (
        y_shoulder_end
        - (x_convergent - x_shoulder_end) * math.tan(alpha)
    )

    upstream_t = np.linspace(
        -(math.pi / 2.0 + alpha), -math.pi / 2.0, n_pts_upstream_arc
    )
    x_upstream_arc = throat_x + Ru * np.cos(upstream_t)
    y_upstream_arc = Rt + Ru + Ru * np.sin(upstream_t)

    def sampled_contour(cylindrical_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        injector_x = x_shoulder_start - cylindrical_length
        x_cylinder = np.linspace(injector_x, x_shoulder_start, n_pts_chamber)
        y_cylinder = np.full_like(x_cylinder, Rc)
        x, y = _concat_sections(
            (x_cylinder, y_cylinder),
            (x_shoulder, y_shoulder),
            (x_convergent, y_convergent),
            (x_upstream_arc, y_upstream_arc),
        )
        return x, y, x_cylinder, y_cylinder

    minimum_x, minimum_y, _, _ = sampled_contour(minimum_cylindrical_length)
    minimum_volume = enclosed_volume(minimum_x, minimum_y)
    if minimum_volume > target_volume:
        raise ValueError(
            "infeasible chamber geometry: shoulder, convergent, upstream throat "
            f"arc, and minimum cylinder enclose {minimum_volume:.9e} m^3, "
            f"exceeding target L*At={target_volume:.9e} m^3"
        )

    def residual(cylindrical_length: float) -> float:
        x, y, _, _ = sampled_contour(cylindrical_length)
        return enclosed_volume(x, y) - target_volume

    upper = max(2.0 * target_volume / Ac, 2.0 * minimum_cylindrical_length)
    while residual(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e4 * max(L_star, Rt):
            raise RuntimeError("could not bracket the chamber cylindrical length")
    Lc = _bisect_root(
        residual, minimum_cylindrical_length, upper, xtol=1e-13
    )
    x_full, y_full, x_cylinder, y_cylinder = sampled_contour(Lc)
    measured_volume = enclosed_volume(x_full, y_full)
    volume_rel_error = abs(measured_volume - target_volume) / target_volume

    return {
        "x": x_full,
        "y": y_full,
        "x_chamber": x_cylinder,
        "y_chamber": y_cylinder,
        "x_shoulder": x_shoulder,
        "y_shoulder": y_shoulder,
        "x_conv": x_convergent,
        "y_conv": y_convergent,
        "x_upstream_arc": x_upstream_arc,
        "y_upstream_arc": y_upstream_arc,
        "Rc": Rc,
        "Rt": Rt,
        "Ru": Ru,
        "shoulder_radius": Rs,
        "shoulder_radius_factor": shoulder_radius_factor,
        "Lc": Lc,
        "L_conv": straight_length,
        "V_chamber": measured_volume,
        "V_target": target_volume,
        "volume_rel_error": volume_rel_error,
        "L_star": L_star,
        "contraction_ratio": contraction_ratio,
        "minimum_cylindrical_length": minimum_cylindrical_length,
        "throat_geometry": spec.to_dict(),
        "throat_location": throat_x,
        "injector_location": float(x_full[0]),
    }


def _unit_tangent(x: np.ndarray, y: np.ndarray, *, at_start: bool) -> np.ndarray:
    if at_start:
        vector = np.array([x[1] - x[0], y[1] - y[0]], dtype=float)
    else:
        vector = np.array([x[-1] - x[-2], y[-1] - y[-2]], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-30)


def _join_angle_deg(
    left_x: np.ndarray,
    left_y: np.ndarray,
    right_x: np.ndarray,
    right_y: np.ndarray,
) -> float:
    if min(len(left_x), len(left_y), len(right_x), len(right_y)) < 2:
        raise ValueError("slope continuity needs two distinct points per side")
    left = _unit_tangent(left_x, left_y, at_start=False)
    right = _unit_tangent(right_x, right_y, at_start=True)
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _segments_intersect(a, b, c, d, tolerance: float = 1e-12) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (
        ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance))
        and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance))
    )


def _polyline_self_intersects(x: np.ndarray, y: np.ndarray) -> bool:
    if np.all(np.diff(x) > 0.0):
        return False
    points = np.column_stack((x, y))
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


def thrust_chamber_geometry_checks(
    contour: dict,
    *,
    offset_distance: float | None = None,
    volume_rel_tolerance: float = 1e-8,
    seam_position_tolerance: float = 1e-10,
    join_angle_tolerance_deg: float = 1.0,
) -> dict:
    """Evaluate geometry gates for an assembled injector-to-exit contour."""
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    chamber = contour.get("chamber")
    nozzle = contour.get("nozzle")
    if chamber is None or nozzle is None:
        raise ValueError("full thrust-chamber checks require chamber and nozzle sections")

    chamber_x = np.asarray(chamber["x"], dtype=float)
    chamber_y = np.asarray(chamber["y"], dtype=float)
    nozzle_x = np.asarray(nozzle["x_throat"], dtype=float)
    nozzle_y = np.asarray(nozzle["y_throat"], dtype=float)
    bell_x = np.asarray(nozzle["x_bell"], dtype=float)
    bell_y = np.asarray(nozzle["y_bell"], dtype=float)
    downstream_x, downstream_y = _concat_sections(
        (nozzle_x, nozzle_y), (bell_x, bell_y)
    )
    seam_gap = float(math.hypot(
        chamber_x[-1] - nozzle_x[0], chamber_y[-1] - nozzle_y[0]
    ))
    throat_bell_gap = float(math.hypot(
        nozzle_x[-1] - bell_x[0], nozzle_y[-1] - bell_y[0]
    ))
    seam_angle = _join_angle_deg(
        chamber_x, chamber_y, downstream_x, downstream_y
    )

    join_angles = [
        _join_angle_deg(chamber["x_chamber"], chamber["y_chamber"],
                        chamber["x_shoulder"], chamber["y_shoulder"]),
        _join_angle_deg(chamber["x_shoulder"], chamber["y_shoulder"],
                        chamber["x_conv"], chamber["y_conv"]),
        _join_angle_deg(chamber["x_conv"], chamber["y_conv"],
                        chamber["x_upstream_arc"], chamber["y_upstream_arc"]),
        seam_angle,
    ]
    if len(nozzle_x) >= 2 and len(bell_x) >= 2:
        join_angles.append(
            _join_angle_deg(nozzle_x, nozzle_y, bell_x, bell_y)
        )
    maximum_join_angle = float(max(join_angles))

    offset_checked = offset_distance is not None and offset_distance > 0.0
    offset_self_intersects = False
    if offset_checked:
        xo, yo = normal_offset_contour(x, y, float(offset_distance))
        offset_self_intersects = _polyline_self_intersects(xo, yo)

    return {
        "axial_coordinates_monotonic": bool(np.all(np.diff(x) > 0.0)),
        "seam_watertight": seam_gap <= seam_position_tolerance,
        "seam_position_gap": seam_gap,
        "position_continuity": (
            seam_gap <= seam_position_tolerance
            and throat_bell_gap <= seam_position_tolerance
        ),
        "throat_bell_position_gap": throat_bell_gap,
        "slope_continuity": maximum_join_angle <= join_angle_tolerance_deg,
        "maximum_join_angle_deg": maximum_join_angle,
        "measured_volume_within_tolerance": (
            float(chamber["volume_rel_error"]) <= volume_rel_tolerance
        ),
        "measured_volume_rel_error": float(chamber["volume_rel_error"]),
        "positive_minimum_cylindrical_length": (
            float(chamber["Lc"]) >= float(chamber["minimum_cylindrical_length"])
        ),
        "cylindrical_length": float(chamber["Lc"]),
        "offset_checked": offset_checked,
        "offset_self_intersections": bool(offset_self_intersects),
        "offset_self_intersection_free": not offset_self_intersects,
    }


HARD_THRUST_CHAMBER_GEOMETRY_CHECKS = (
    "axial_coordinates_monotonic",
    "seam_watertight",
    "position_continuity",
    "slope_continuity",
    "measured_volume_within_tolerance",
    "positive_minimum_cylindrical_length",
    "offset_self_intersection_free",
)


def failed_thrust_chamber_geometry_checks(checks: dict) -> list[str]:
    """Return failed geometry checks that block CAD and contour export."""
    return [
        name for name in HARD_THRUST_CHAMBER_GEOMETRY_CHECKS
        if not bool(checks.get(name, False))
    ]


def full_engine_contour(chamber: dict, nozzle: dict) -> dict:
    """Assemble one authoritative injector-face → nozzle-exit contour."""
    chamber_spec = chamber.get("throat_geometry")
    nozzle_spec = nozzle.get("throat_geometry")
    if chamber_spec != nozzle_spec:
        raise ValueError("chamber and nozzle must use the same ThroatGeometrySpec")

    nozzle_x = np.asarray(nozzle["x"], dtype=float)
    nozzle_y = np.asarray(nozzle["y"], dtype=float)
    throat_index = int(np.argmin(np.abs(nozzle_y - float(nozzle["Rt"]))))
    seam_gap = math.hypot(
        float(chamber["x"][-1]) - float(nozzle_x[throat_index]),
        float(chamber["y"][-1]) - float(nozzle_y[throat_index]),
    )
    if seam_gap > 1e-10:
        raise ValueError(f"chamber/nozzle throat seam gap is {seam_gap:.6e} m")

    x_full, y_full = _concat_sections(
        (np.asarray(chamber["x"]), np.asarray(chamber["y"])),
        (nozzle_x[throat_index:], nozzle_y[throat_index:]),
    )
    result = dict(nozzle)
    result.update({
        "x": x_full,
        "y": y_full,
        "chamber": chamber,
        "nozzle": nozzle,
        "full_thrust_chamber": True,
        "injector_location": float(x_full[0]),
        "throat_location": float(chamber["throat_location"]),
        "throat_index": int(np.argmin(np.abs(y_full - float(nozzle["Rt"])))),
        "V_chamber": float(chamber["V_chamber"]),
        "V_target": float(chamber["V_target"]),
        "L_star": float(chamber["L_star"]),
        "contraction_ratio": float(chamber["contraction_ratio"]),
        "shoulder_radius_factor": float(chamber["shoulder_radius_factor"]),
        "minimum_cylindrical_length": float(
            chamber["minimum_cylindrical_length"]
        ),
    })
    result["geometry_checks"] = thrust_chamber_geometry_checks(result)
    return result
