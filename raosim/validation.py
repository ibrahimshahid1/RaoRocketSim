"""
validation.py - Design-status labels and engineering gate checks.

These checks are intentionally conservative. Passing them means the generated
nozzle is suitable for downstream design review, not qualified for hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from raosim.gas_dynamics import mach_from_area_ratio


DESIGN_STATUS_BY_METHOD = {
    "bezier": "preliminary_top_geometry",
    "moc": "experimental_moc_geometry",
    "rao": "experimental_variational_geometry",
    "rao_variational_moc": "experimental_rao_variational_moc_bvp",
}

HARDWARE_QUALIFICATION_NOTE = (
    "Not hardware-qualified. Requires independent CFD, thermal/structural "
    "analysis, manufacturing review, inspection, and hot-fire evidence."
)


@dataclass
class DesignGateCheck:
    category: str
    name: str
    passed: bool
    value: Any = None
    limit: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "limit": self.limit,
            "message": self.message,
        }


@dataclass
class DesignGateReport:
    checks: list[DesignGateCheck] = field(default_factory=list)
    hardware_qualified: bool = False
    qualification_note: str = HARDWARE_QUALIFICATION_NOTE

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def warnings(self) -> list[str]:
        return [check.message for check in self.checks if not check.passed]

    def add(self, category: str, name: str, passed: bool, *,
            value: Any = None, limit: Any = None, message: str = "") -> None:
        self.checks.append(DesignGateCheck(
            category=category,
            name=name,
            passed=bool(passed),
            value=value,
            limit=limit,
            message=message,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "hardware_qualified": self.hardware_qualified,
            "qualification_note": self.qualification_note,
            "checks": [check.to_dict() for check in self.checks],
        }


def design_status_for_method(method: str | None) -> str:
    return DESIGN_STATUS_BY_METHOD.get(method or "bezier", "unknown_geometry")


def add_contour_reliability_metadata(
    contour: dict,
    method: str,
    gamma: float,
    *,
    exit_radius_rel_tol: float = 5e-3,
    exit_mach_rel_tol: float = 0.25,
    exit_theta_max_deg: float = 10.0,
) -> dict:
    """Attach status/warnings to a contour dict in-place and return it."""
    contour["design_status"] = design_status_for_method(method)
    contour["hardware_qualified"] = False
    contour["qualification_note"] = HARDWARE_QUALIFICATION_NOTE

    warnings = list(contour.get("warnings", []))

    try:
        expected_re = math.sqrt(float(contour["epsilon"])) * float(contour["Rt"])
        actual_re = float(np.asarray(contour["y"])[-1])
        rel_err = abs(actual_re - expected_re) / max(expected_re, 1e-15)
        contour["exit_radius_rel_error"] = rel_err
        if rel_err > exit_radius_rel_tol:
            warnings.append(
                f"Exit radius differs from sqrt(epsilon)*Rt by "
                f"{rel_err:.3%}; contour is not design-gated."
            )
    except Exception as exc:
        warnings.append(f"Could not verify exit radius: {exc}")

    try:
        ideal_me = mach_from_area_ratio(float(contour["epsilon"]), gamma, supersonic=True)
        contour["ideal_exit_Mach"] = ideal_me
        if "exit_M_mean" in contour:
            moc_me = float(contour["exit_M_mean"])
            rel_err = abs(moc_me - ideal_me) / max(ideal_me, 1e-15)
            contour["exit_M_rel_error"] = rel_err
            if rel_err > exit_mach_rel_tol:
                warnings.append(
                    f"Solver exit Mach mean {moc_me:.3f} differs from ideal "
                    f"area-Mach {ideal_me:.3f} by {rel_err:.1%}; treat solver "
                    "flow metrics as experimental."
                )
    except Exception as exc:
        warnings.append(f"Could not verify exit Mach consistency: {exc}")

    if "exit_theta_max" in contour:
        theta_max = float(contour["exit_theta_max"])
        if theta_max > exit_theta_max_deg:
            warnings.append(
                f"Exit flow angle max {theta_max:.2f} deg exceeds "
                f"{exit_theta_max_deg:.2f} deg design-gate limit."
            )

    if contour.get("optimization_converged") is False:
        warnings.append("Optimizer did not converge; contour is experimental.")

    contour["warnings"] = _dedupe(warnings)
    return contour


def evaluate_design_gates(
    contour: dict,
    Pc: float,
    Pa: float,
    gamma: float,
    *,
    wall_thickness: float | None = None,
    flange_od: float | None = None,
    flange_length: float | None = None,
    exit_radius_rel_tol: float = 5e-3,
    exit_mach_rel_tol: float = 0.25,
) -> DesignGateReport:
    """Evaluate geometry, flow, solver, and CAD readiness checks."""
    report = DesignGateReport()
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    epsilon = float(contour["epsilon"])
    expected_re = math.sqrt(epsilon) * Rt
    actual_re = float(y[-1])
    rel_re = abs(actual_re - expected_re) / max(expected_re, 1e-15)

    report.add(
        "geometry", "exit_radius", rel_re <= exit_radius_rel_tol,
        value=rel_re, limit=exit_radius_rel_tol,
        message="Exit radius must match sqrt(epsilon)*Rt.",
    )
    report.add(
        "geometry", "positive_length", float(contour.get("Ln", 0.0)) > 0.0,
        value=float(contour.get("Ln", 0.0)), limit="> 0",
        message="Nozzle length must be positive.",
    )

    y_bell = np.asarray(contour.get("y_bell", y), dtype=float)
    report.add(
        "geometry", "bell_radius_monotonic",
        bool(np.all(np.diff(y_bell) >= -1e-9)),
        value=float(np.min(np.diff(y_bell))) if len(y_bell) > 1 else None,
        limit=">= -1e-9 m",
        message="Bell radius should not decrease downstream.",
    )
    report.add(
        "geometry", "finite_coordinates",
        bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y))),
        message="Contour coordinates must be finite.",
    )

    geometry_checks = contour.get("geometry_checks")
    if geometry_checks is not None:
        report.add(
            "geometry", "axial_coordinates_monotonic",
            bool(geometry_checks["axial_coordinates_monotonic"]),
            value=geometry_checks["axial_coordinates_monotonic"],
            limit=True,
            message="Full thrust-chamber axial coordinates must increase monotonically.",
        )
        report.add(
            "geometry", "chamber_nozzle_seam",
            bool(geometry_checks["seam_watertight"]),
            value=float(geometry_checks["seam_position_gap"]),
            limit="<= 1e-10 m",
            message="Chamber and nozzle must share one watertight throat station.",
        )
        report.add(
            "geometry", "position_and_slope_continuity",
            bool(
                geometry_checks["position_continuity"]
                and geometry_checks["slope_continuity"]
            ),
            value=float(geometry_checks["maximum_join_angle_deg"]),
            limit="<= 1 deg",
            message="Thrust-chamber section joins must be position- and slope-continuous.",
        )
        report.add(
            "geometry", "chamber_volume",
            bool(geometry_checks["measured_volume_within_tolerance"]),
            value=float(geometry_checks["measured_volume_rel_error"]),
            limit="<= 1e-8 relative",
            message="Integrated chamber volume must match L* times throat area.",
        )
        report.add(
            "geometry", "minimum_cylindrical_length",
            bool(geometry_checks["positive_minimum_cylindrical_length"]),
            value=float(geometry_checks["cylindrical_length"]),
            limit=">= configured positive minimum",
            message="Chamber must retain a positive minimum cylindrical length.",
        )
        report.add(
            "geometry", "offset_self_intersections",
            bool(geometry_checks["offset_self_intersection_free"]),
            value=bool(geometry_checks["offset_self_intersections"]),
            limit=False,
            message="The manufacturing wall offset must not self-intersect.",
        )

    curvature_ok, curvature_max = _curvature_check(x, y, Rt)
    report.add(
        "geometry", "curvature_finite",
        curvature_ok, value=curvature_max, limit=f"< {50.0 / max(Rt, 1e-15):.3g}",
        message="Curvature must be finite and free of severe spikes.",
    )

    try:
        from raosim.wall_pressure import wall_pressure_distribution

        wp = wall_pressure_distribution(contour, Pc, gamma)
        report.add(
            "flow", "wall_pressure_monotonic",
            bool(wp["monotonic"]),
            value=int(len(wp["violation_indices"])), limit=0,
            message="Wall pressure should decrease downstream of the throat.",
        )
    except Exception as exc:
        report.add("flow", "wall_pressure_monotonic", False, message=str(exc))

    try:
        from raosim.separation import check_separation

        sep = check_separation(contour, Pc, Pa, gamma)
        report.add(
            "flow", "separation_margin",
            not bool(sep["separated"]),
            value=float(sep["margin"]), limit=">= 1",
            message="Empirical separation check predicts separated flow.",
        )
    except Exception as exc:
        report.add("flow", "separation_margin", False, message=str(exc))

    if "exit_M_mean" in contour:
        ideal_me = mach_from_area_ratio(epsilon, gamma, supersonic=True)
        rel_me = abs(float(contour["exit_M_mean"]) - ideal_me) / max(ideal_me, 1e-15)
        report.add(
            "solver", "exit_mach_consistency",
            rel_me <= exit_mach_rel_tol,
            value=rel_me, limit=exit_mach_rel_tol,
            message="Solver exit Mach should be consistent with ideal area-Mach.",
        )
    else:
        report.add(
            "solver", "exit_mach_consistency",
            True, value="not traced", limit="informational",
            message="No solver exit Mach trace is available for this method.",
        )

    if "exit_theta_max" in contour:
        report.add(
            "solver", "exit_flow_angle",
            float(contour["exit_theta_max"]) <= 10.0,
            value=float(contour["exit_theta_max"]), limit="<= 10 deg",
            message="Experimental solver exit flow angle is too large.",
        )
    else:
        report.add(
            "solver", "exit_flow_angle",
            True, value="not traced", limit="informational",
            message="No solver exit angle trace is available for this method.",
        )

    report.add(
        "solver", "method_status",
        contour.get("design_status") == DESIGN_STATUS_BY_METHOD["bezier"],
        value=contour.get("design_status"), limit=DESIGN_STATUS_BY_METHOD["bezier"],
        message="Only the Rao/TOP Bezier path is currently treated as the trusted baseline.",
    )

    cad_requested = wall_thickness is not None or flange_od is not None or flange_length is not None
    report.add(
        "cad", "units_meters",
        True, value="m", limit="m",
        message="CAD/export coordinates are expressed in meters.",
    )
    report.add(
        "cad", "wall_thickness",
        (wall_thickness is None and not cad_requested) or (wall_thickness is not None and wall_thickness > 0.0),
        value=wall_thickness, limit="> 0 m when solid CAD is requested",
        message="Solid CAD export requires a positive wall thickness.",
    )
    flange_ok = (
        flange_od is None and flange_length is None
        or (
            flange_od is not None and flange_length is not None
            and flange_od > 2.0 * float(np.max(y))
            and flange_length > 0.0
        )
    )
    report.add(
        "cad", "flange_geometry",
        flange_ok,
        value={"flange_od": flange_od, "flange_length": flange_length},
        limit="flange_od > nozzle OD and flange_length > 0",
        message="Flange dimensions are incomplete or too small.",
    )

    return report


def _curvature_check(x: np.ndarray, y: np.ndarray, Rt: float) -> tuple[bool, float | None]:
    if len(x) < 5:
        return False, None
    ds = np.hypot(np.diff(x), np.diff(y))
    if np.any(ds <= 1e-15):
        return False, None
    s = np.concatenate(([0.0], np.cumsum(ds)))
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    denom = np.power(dx * dx + dy * dy, 1.5)
    valid = denom > 1e-15
    if not np.any(valid):
        return False, None
    kappa = np.abs(dx[valid] * ddy[valid] - dy[valid] * ddx[valid]) / denom[valid]
    if not np.all(np.isfinite(kappa)):
        return False, None
    max_kappa = float(np.max(kappa))
    return max_kappa < (50.0 / max(Rt, 1e-15)), max_kappa


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
